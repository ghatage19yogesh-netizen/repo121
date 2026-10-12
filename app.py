"""
Allocation Planner — single-file Streamlit app (planner + approver login).

Deploy: put this file and requirements.txt in the root of your GitHub repo,
then on share.streamlit.io choose this file as the main file.

Add real users in Streamlit Secrets (generate a block with:
    python app.py adduser <username> "<Full Name>" <planner|approver> <password>
).
"""
from __future__ import annotations
import sys as _sys



# ======================================================================================
# MODEL — sample data + allocation (rules + HiGHS LP)
# ======================================================================================

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog

# --------------------------------------------------------------------------------------
# Sample data
# --------------------------------------------------------------------------------------
WAREHOUSES = ["CFC 1 · East", "CFC 2 · Central", "CFC 3 · West"]
ZONES = ["East", "Central", "West"]
STORAGE_TYPES = ["FLA", "HFG"]

# Base shipping cost per unit (warehouse -> zone). Products scale it by cost_factor.
BASE_COST = pd.DataFrame(
    [[2.0, 3.5, 5.0],
     [3.5, 2.0, 3.5],
     [5.0, 3.5, 2.0]],
    index=WAREHOUSES, columns=ZONES,
)

# Lanes with no cost data in the sample (to show the "missing cost" warning).
MISSING_COST_LANES = [("CFC 3 · West", "East", "305-410-high")]


def sample_products() -> pd.DataFrame:
    rows = [
        # product_type, description, storage, velocity, E, C, W, order_qty, rule, rule_cfcs, allowed_cfcs, cost_factor
        ("101-210-high",   "Men's shirts",        "FLA", "high",   600, 300, 400,  0, "",            "", "", 1.00),
        ("101-215-high",   "Women's tops",        "FLA", "high",   450, 350, 500,  0, "",            "", "", 0.90),
        ("202-330-medium", "Dresses",             "HFG", "medium", 200, 150, 300,  0, "",            "", "CFC 1 · East, CFC 3 · West", 1.20),
        ("202-340-medium", "Blazers",             "HFG", "medium", 120, 180,  90,  0, "",            "", "CFC 1 · East, CFC 2 · Central", 1.30),
        ("305-410-high",   "Coats",               "HFG", "high",   250, 200, 150,  0, "forbidden_in", "CFC 2 · Central", "", 1.50),
        ("410-520-low",    "Silk scarves",        "FLA", "low",     40,  35,  25,  0, "",            "", "", 0.60),
        ("415-525-low",    "Formal gowns",        "HFG", "low",     20,  10,  30,  0, "",            "", "", 1.40),
        ("520-610-high",   "New brand socks",     "FLA", "high",     0,   0,   0, 60, "",            "", "", 0.50),
        ("610-700-high",   "Fine jewellery",      "FLA", "high",    80,  60,  70,  0, "must_be_in",  "CFC 1 · East", "", 0.40),
    ]
    cols = ["product_type", "description", "storage", "velocity", "demand_East", "demand_Central",
            "demand_West", "order_qty", "rule", "rule_cfcs", "allowed_cfcs", "cost_factor"]
    return pd.DataFrame(rows, columns=cols)


def sample_capacity() -> pd.DataFrame:
    return pd.DataFrame(
        {"FLA": [2000, 1500, 1800], "HFG": [700, 450, 500]},
        index=pd.Index(WAREHOUSES, name="Warehouse"),
    )


def sample_limits() -> pd.DataFrame:
    return pd.DataFrame(
        {"min_pct": [25.0, 20.0, 25.0], "max_pct": [50.0, 40.0, 45.0]},
        index=pd.Index(WAREHOUSES, name="Warehouse"),
    )


@dataclass
class Settings:
    overflow_penalty: float = 50.0      # $ per unit over capacity
    time_limit_s: float = 60.0
    slow_tie_break: str = "Lowest cost"  # "Lowest cost" | "Keep last month's" | "Random"
    count_rule_stock_in_shares: bool = True
    missing_cost_fill: str = "Warehouse average"  # "Warehouse average" | "Minimum cost (legacy)"
    random_seed: int | None = None


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _split(s: str) -> list[str]:
    return [x.strip() for x in str(s or "").split(",") if x.strip()]


def demand_by_zone(row) -> dict[str, float]:
    return {z: float(row[f"demand_{z}"]) for z in ZONES}


def build_costs(products: pd.DataFrame, settings: Settings, min_cost: float = 0.01):
    """Return {(w, z, p): cost} and the list of lanes that were filled in."""
    costs, filled = {}, []
    missing = set(MISSING_COST_LANES)
    for _, r in products.iterrows():
        p = r["product_type"]
        for w in WAREHOUSES:
            known = {z: BASE_COST.loc[w, z] * r["cost_factor"] for z in ZONES if (w, z, p) not in missing}
            for z in ZONES:
                if z in known:
                    costs[(w, z, p)] = known[z]
                else:
                    if settings.missing_cost_fill.startswith("Minimum"):
                        costs[(w, z, p)] = min_cost
                    else:
                        costs[(w, z, p)] = float(np.mean(list(known.values())))
                    filled.append({"warehouse": w, "zone": z, "product_type": p,
                                   "filled_cost": round(costs[(w, z, p)], 2)})
    return costs, filled


# --------------------------------------------------------------------------------------
# Validation (pre-run checks)
# --------------------------------------------------------------------------------------
def validate(products: pd.DataFrame, capacity: pd.DataFrame, limits: pd.DataFrame) -> list[dict]:
    checks = []
    bad = limits[limits["min_pct"] > limits["max_pct"]]
    checks.append({"level": "error" if len(bad) else "ok",
                   "text": "Each warehouse min ≤ max" + (f" — fails for {', '.join(bad.index)}" if len(bad) else "")})
    smax, smin = limits["max_pct"].sum(), limits["min_pct"].sum()
    checks.append({"level": "ok" if smax >= 100 else "error",
                   "text": f"Max shares add up to ≥ 100% (now {smax:.0f}%)"})
    checks.append({"level": "ok" if smin <= 100 else "error",
                   "text": f"Min shares add up to ≤ 100% (now {smin:.0f}%)"})
    if (capacity < 0).any().any():
        checks.append({"level": "error", "text": "Capacity cannot be negative"})
    dem_cols = [f"demand_{z}" for z in ZONES]
    if (products[dem_cols] < 0).any().any():
        checks.append({"level": "error", "text": "Demand cannot be negative"})
    checks.append({"level": "warn" if MISSING_COST_LANES else "ok",
                   "text": f"{len(MISSING_COST_LANES)} shipping cost lane(s) missing — will be filled"})
    total = products[dem_cols].sum(axis=1) + products["order_qty"]
    for k in STORAGE_TYPES:
        need = total[products["storage"] == k].sum()
        have = capacity[k].sum()
        if need > have:
            checks.append({"level": "warn",
                           "text": f"{k} demand ({need:,.0f}) is more than total {k} space ({have:,.0f}) — overflow will be penalised"})
    nd = int(((products[dem_cols].sum(axis=1) == 0) & (products["order_qty"] > 0)).sum())
    if nd:
        checks.append({"level": "info", "text": f"{nd} product(s) without demand — split evenly"})
    return checks


# --------------------------------------------------------------------------------------
# Main run
# --------------------------------------------------------------------------------------
def run_allocation(products: pd.DataFrame, capacity: pd.DataFrame, limits: pd.DataFrame,
                   settings: Settings, previous_slow_map: dict | None = None,
                   progress=None) -> dict:
    t0 = time.time()
    say = progress or (lambda msg: None)
    products = products.copy().reset_index(drop=True)
    capacity = capacity.astype(float).copy()
    limits = limits.astype(float).copy()
    previous_slow_map = previous_slow_map or {}
    rng = np.random.default_rng(settings.random_seed)

    say("Loading data and filling missing costs")
    costs, filled_lanes = build_costs(products, settings)

    # ---- 1. classify -----------------------------------------------------------------
    say("Sorting products into groups")
    dem_cols = [f"demand_{z}" for z in ZONES]
    tot = products[dem_cols].sum(axis=1)
    groups = {}
    for i, r in products.iterrows():
        if tot[i] == 0 and r["order_qty"] > 0:
            groups[r["product_type"]] = "No demand"
        elif r["rule"] == "must_be_in":
            groups[r["product_type"]] = "Restricted"
        elif r["velocity"] == "low":
            groups[r["product_type"]] = "Slow seller"
        else:
            groups[r["product_type"]] = "Optimizer"

    # ---- 2. rule-based placements ----------------------------------------------------
    say("Placing special products with business rules")
    manual_rows, slow_map, slow_notes = [], {}, []
    for _, r in products.iterrows():
        p, g = r["product_type"], groups[r["product_type"]]
        forbidden = _split(r["rule_cfcs"]) if r["rule"] == "forbidden_in" else []
        allowed = [w for w in WAREHOUSES if w not in forbidden]
        if g == "No demand":
            share = r["order_qty"] / len(allowed)
            for w in allowed:
                manual_rows.append(dict(warehouse=w, zone="(no demand)", product_type=p, storage=r["storage"],
                                        units=share, unit_cost=np.nan, cost=0.0, method="No demand"))
        elif g == "Restricted":
            w = _split(r["rule_cfcs"])[0]
            for z, d in demand_by_zone(r).items():
                if d > 0:
                    c = costs[(w, z, p)]
                    manual_rows.append(dict(warehouse=w, zone=z, product_type=p, storage=r["storage"],
                                            units=d, unit_cost=c, cost=c * d, method="Restricted"))
        elif g == "Slow seller":
            dz = demand_by_zone(r)
            wcost = {w: sum(costs[(w, z, p)] * d for z, d in dz.items()) for w in allowed}
            best = min(wcost, key=wcost.get)
            ties = [w for w in allowed if abs(wcost[w] - wcost[best]) < 1e-9]
            if settings.slow_tie_break == "Random":
                w = str(rng.choice(allowed))
                note = "Random pick"
            elif settings.slow_tie_break == "Keep last month's" and previous_slow_map.get(p) in allowed:
                w = previous_slow_map[p]
                note = "Kept last month's warehouse"
            else:
                w = ties[0]
                note = "Lowest cost" + (" (tie)" if len(ties) > 1 else "")
            slow_map[p] = w
            prev = previous_slow_map.get(p)
            slow_notes.append(dict(product_type=p, warehouse=w, how=note,
                                   last_month=prev or "—", changed=bool(prev and prev != w)))
            for z, d in dz.items():
                if d > 0:
                    c = costs[(w, z, p)]
                    manual_rows.append(dict(warehouse=w, zone=z, product_type=p, storage=r["storage"],
                                            units=d, unit_cost=c, cost=c * d, method="Slow seller"))
    manual = pd.DataFrame(manual_rows, columns=["warehouse", "zone", "product_type", "storage",
                                                "units", "unit_cost", "cost", "method"])

    # ---- 3. update capacity ----------------------------------------------------------
    say("Updating warehouse space")
    used_manual = manual.groupby(["warehouse", "storage"])["units"].sum() if len(manual) else pd.Series(dtype=float)
    rem_cap = capacity.copy()
    for (w, k), u in used_manual.items():
        rem_cap.loc[w, k] -= u
    manual_overflow = (-rem_cap.clip(upper=0)).copy()
    rem_cap = rem_cap.clip(lower=0)

    # ---- 4. LP -------------------------------------------------------------------------
    say("Building the optimisation model")
    ok = products[products["product_type"].map(groups) == "Optimizer"].reset_index(drop=True)
    W, Z, P, K = len(WAREHOUSES), len(ZONES), len(ok), len(STORAGE_TYPES)
    nx = W * Z * P
    npen = W * K
    xi = lambda i, j, p: (i * Z + j) * P + p
    pi = lambda i, k: nx + i * K + k

    c = np.zeros(nx + npen)
    bounds = [(0, None)] * (nx + npen)
    for p, r in ok.iterrows():
        forbidden = _split(r["rule_cfcs"]) if r["rule"] == "forbidden_in" else []
        allowed_med = _split(r["allowed_cfcs"]) if r["velocity"] == "medium" else []
        for i, w in enumerate(WAREHOUSES):
            blocked = (w in forbidden) or (allowed_med and w not in allowed_med)
            for j, z in enumerate(ZONES):
                c[xi(i, j, p)] = costs[(w, z, r["product_type"])]
                if blocked:
                    bounds[xi(i, j, p)] = (0, 0)
    for i in range(W):
        for k in range(K):
            c[pi(i, k)] = settings.overflow_penalty

    A, b = [], []
    # demand: sum_i x >= d
    for p, r in ok.iterrows():
        for j, z in enumerate(ZONES):
            d = float(r[f"demand_{z}"])
            if d > 0:
                row = np.zeros(nx + npen)
                for i in range(W):
                    row[xi(i, j, p)] = -1
                A.append(row); b.append(-d)
    # capacity (soft): sum x(i, ., p in k) - pen(i,k) <= remaining capacity
    for i, w in enumerate(WAREHOUSES):
        for k, kname in enumerate(STORAGE_TYPES):
            row = np.zeros(nx + npen)
            for p, r in ok.iterrows():
                if r["storage"] == kname:
                    for j in range(Z):
                        row[xi(i, j, p)] = 1
            row[pi(i, k)] = -1
            A.append(row); b.append(rem_cap.loc[w, kname])
    # share limits
    d_lp = float(ok[dem_cols].values.sum())
    manual_by_w = manual.groupby("warehouse")["units"].sum() if len(manual) else pd.Series(dtype=float)
    base = d_lp + (manual["units"].sum() if settings.count_rule_stock_in_shares and len(manual) else 0.0)
    for i, w in enumerate(WAREHOUSES):
        m_i = manual_by_w.get(w, 0.0) if settings.count_rule_stock_in_shares else 0.0
        row = np.zeros(nx + npen)
        for j in range(Z):
            for p in range(P):
                row[xi(i, j, p)] = 1
        A.append(row.copy()); b.append(limits.loc[w, "max_pct"] / 100 * base - m_i)
        A.append(-row); b.append(-(limits.loc[w, "min_pct"] / 100 * base - m_i))

    say("Solving")
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method="highs",
                  options={"time_limit": float(settings.time_limit_s)})
    status_map = {0: "Optimal", 1: "Suboptimal", 2: "Infeasible", 3: "Unbounded", 4: "Numerical error"}
    status = status_map.get(res.status, "Error")
    solved = res.x is not None and status in ("Optimal", "Suboptimal")

    # ---- 5. combine --------------------------------------------------------------------
    say("Saving results")
    opt_rows, pen = [], np.zeros((W, K))
    if solved:
        x = res.x
        for p, r in ok.iterrows():
            for i, w in enumerate(WAREHOUSES):
                for j, z in enumerate(ZONES):
                    u = x[xi(i, j, p)]
                    if u > 1e-6:
                        cc = c[xi(i, j, p)]
                        opt_rows.append(dict(warehouse=w, zone=z, product_type=r["product_type"],
                                             storage=r["storage"], units=u, unit_cost=cc, cost=cc * u,
                                             method="Optimizer"))
        for i in range(W):
            for k in range(K):
                pen[i, k] = x[pi(i, k)]
    detail = pd.concat([manual, pd.DataFrame(opt_rows, columns=manual.columns)], ignore_index=True)
    detail["units"] = detail["units"].round(2)
    detail["cost"] = detail["cost"].round(2)
    detail["unit_cost"] = detail["unit_cost"].round(2)

    # summary by warehouse x storage
    alloc = detail.groupby(["warehouse", "storage"])["units"].sum()
    srows = []
    for i, w in enumerate(WAREHOUSES):
        for k, kname in enumerate(STORAGE_TYPES):
            a = float(alloc.get((w, kname), 0.0))
            cap = float(capacity.loc[w, kname])
            over = max(a - cap, 0.0)
            srows.append(dict(warehouse=w, storage=kname, allocated=round(a, 1), capacity=cap,
                              used_pct=round(100 * a / cap, 1) if cap else 0.0, overflow=round(over, 1)))
    summary = pd.DataFrame(srows)

    tot_units = float(detail["units"].sum())
    by_w = detail.groupby("warehouse")["units"].sum()
    shares = pd.DataFrame([
        dict(warehouse=w, units=round(float(by_w.get(w, 0.0)), 1),
             share_pct=round(100 * float(by_w.get(w, 0.0)) / tot_units, 1) if tot_units else 0.0,
             min_pct=limits.loc[w, "min_pct"], max_pct=limits.loc[w, "max_pct"])
        for w in WAREHOUSES])

    rule_units = float(detail.loc[detail["method"] != "Optimizer", "units"].sum())
    overflow_units = float(summary["overflow"].sum())
    kpis = dict(
        status=status,
        total_cost=round(float(detail["cost"].sum()), 2),
        units=round(tot_units, 1),
        optimizer_units=round(tot_units - rule_units, 1),
        rule_units=round(rule_units, 1),
        overflow_units=round(overflow_units, 1),
        penalty_cost=round(float(pen.sum() * settings.overflow_penalty), 2),
        solve_seconds=round(time.time() - t0, 2),
        message=str(res.message),
    )

    # ---- warnings ----------------------------------------------------------------------
    warns = []
    if not solved:
        warns.append(dict(severity="High", title=f"Solver result: {status}",
                          detail="No valid plan. Check that min/max shares fit the demand and the business rules."))
    for _, s in summary[summary["overflow"] > 0].iterrows():
        warns.append(dict(severity="High", title=f"{s['warehouse']} is over its {s['storage']} space",
                          detail=f"{s['overflow']:,.0f} units above capacity ({s['used_pct']:.0f}% used)."))
    if filled_lanes:
        how = "the warehouse average" if not settings.missing_cost_fill.startswith("Minimum") else "the minimum cost (cheapest lane!)"
        warns.append(dict(severity="High" if settings.missing_cost_fill.startswith("Minimum") else "Medium",
                          title=f"{len(filled_lanes)} shipping cost lane(s) were missing",
                          detail=f"Filled with {how}."))
    changed = [n for n in slow_notes if n["changed"]]
    if changed:
        warns.append(dict(severity="Medium", title=f"{len(changed)} slow seller(s) changed warehouse",
                          detail="Compared with the last approved run."))
    if settings.slow_tie_break == "Random":
        warns.append(dict(severity="Medium", title="Slow sellers were placed at random",
                          detail="Re-running can give a different plan. Use 'Lowest cost' or 'Keep last month's'."))
    n_rule = int((pd.Series(groups) != "Optimizer").sum())
    if n_rule:
        warns.append(dict(severity="Info", title=f"{n_rule} product(s) placed by business rules",
                          detail="Slow sellers, no-demand and restricted products skip the optimizer."))
    for _, s in shares.iterrows():
        if s["units"] and (s["share_pct"] < s["min_pct"] - 0.5 or s["share_pct"] > s["max_pct"] + 0.5):
            warns.append(dict(severity="Medium", title=f"{s['warehouse']} share is outside its target",
                              detail=f"{s['share_pct']:.1f}% vs target {s['min_pct']:.0f}–{s['max_pct']:.0f}%."))

    return dict(
        kpis=kpis,
        detail=detail,
        summary=summary,
        shares=shares,
        warnings=warns,
        filled_lanes=pd.DataFrame(filled_lanes),
        slow_notes=pd.DataFrame(slow_notes),
        slow_map=slow_map,
        groups=groups,
    )


# ======================================================================================
# STORE — shared run history (JSON file)
# ======================================================================================

import json
import os
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("ALLOC_DATA_DIR", Path(__file__).resolve().parent / "data"))
RUNS_FILE = DATA_DIR / "runs.json"
_LOCK = threading.Lock()

DF_KEYS = ["detail", "summary", "shares", "filled_lanes", "slow_notes"]


def _read() -> list[dict]:
    if not RUNS_FILE.exists():
        return []
    try:
        return json.loads(RUNS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _write(runs: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RUNS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(runs, default=str, indent=1))
    tmp.replace(RUNS_FILE)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def serialise_results(res: dict) -> dict:
    out = {k: v for k, v in res.items() if k not in DF_KEYS}
    for k in DF_KEYS:
        df = res.get(k)
        out[k] = [] if df is None else json.loads(df.to_json(orient="records"))
    return out


def deserialise_results(raw: dict) -> dict:
    res = dict(raw)
    for k in DF_KEYS:
        res[k] = pd.DataFrame(raw.get(k, []))
    return res


def save_run(record: dict) -> dict:
    with _LOCK:
        runs = _read()
        runs = [r for r in runs if r["id"] != record["id"]] + [record]
        _write(runs)
    return record


def list_runs() -> list[dict]:
    with _LOCK:
        return sorted(_read(), key=lambda r: r["created_at"], reverse=True)


def get_run(run_id: str) -> dict | None:
    return next((r for r in list_runs() if r["id"] == run_id), None)


def update_run(run_id: str, **changes) -> dict | None:
    with _LOCK:
        runs = _read()
        for r in runs:
            if r["id"] == run_id:
                r.update(changes)
                r.setdefault("history", []).append({"at": now(), **{k: v for k, v in changes.items()
                                                                     if k in ("status", "comment", "by")}})
                _write(runs)
                return r
    return None


def last_approved_slow_map() -> dict:
    for r in list_runs():
        if r.get("status") == "Approved":
            return r.get("results", {}).get("slow_map", {}) or {}
    return {}


# ======================================================================================
# AUTH — login for planner / approver
# ======================================================================================

import hashlib
import hmac
import secrets as pysecrets

ROLES = ("planner", "approver")


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()


def make_user(name: str, role: str, password: str) -> dict:
    salt = pysecrets.token_hex(16)
    return {"name": name, "role": role, "salt": salt, "password_hash": hash_password(password, salt)}


# Demo accounts (password shown on the login page only while these defaults are in use)
_DEMO_SALT_1 = "8f1d2c3b4a5968778695a4b3c2d1e0f1"
_DEMO_SALT_2 = "1a2b3c4d5e6f708192a3b4c5d6e7f809"
DEMO_USERS = {
    "maria": {"name": "Maria Lopez", "role": "planner", "salt": _DEMO_SALT_1,
              "password_hash": hash_password("planner123", _DEMO_SALT_1)},
    "david": {"name": "David Brooks", "role": "approver", "salt": _DEMO_SALT_2,
              "password_hash": hash_password("approver123", _DEMO_SALT_2)},
}
DEMO_PASSWORDS = {"maria": "planner123", "david": "approver123"}


def load_users(st_secrets) -> tuple[dict, bool]:
    """Return (users, using_demo_accounts)."""
    try:
        users = {k: dict(v) for k, v in st_secrets["users"].items()}
        if users:
            return users, False
    except Exception:
        pass
    return DEMO_USERS, True


def check_login(users: dict, username: str, password: str) -> dict | None:
    u = users.get((username or "").strip().lower())
    if not u or u.get("role") not in ROLES:
        # still burn time to avoid a timing hint
        hash_password(password or "", "00" * 16)
        return None
    if hmac.compare_digest(hash_password(password or "", u["salt"]), u["password_hash"]):
        return {"username": username.strip().lower(), "name": u["name"], "role": u["role"]}
    return None




if __name__ == "__main__" and len(_sys.argv) >= 2 and _sys.argv[1] == "adduser":
    import json as _json
    if len(_sys.argv) != 6:
        print('usage: python app.py adduser <username> "<Full Name>" <planner|approver> <password>')
        _sys.exit(1)
    _u, _n, _r, _p = _sys.argv[2:]
    print(f"[users.{_u}]")
    for _k, _v in make_user(_n, _r, _p).items():
        print(f"{_k} = {_json.dumps(_v)}")
    _sys.exit(0)



# ======================================================================================
# APP — Streamlit UI
# ======================================================================================

from dataclasses import asdict
from datetime import date, datetime

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Allocation Planner", page_icon=":material/warehouse:", layout="wide")

ss = st.session_state
PERSIST_KEYS = ["month", "penalty", "time_limit", "tie_break", "cost_fill", "count_rule"]
for _k in PERSIST_KEYS:  # keep widget values when switching pages
    if _k in ss:
        ss[_k] = ss[_k]

STATUS_BADGE = {
    "Draft": ":gray-badge[Draft]",
    "Submitted": ":blue-badge[Submitted for approval]",
    "Approved": ":green-badge[Approved]",
    "Rejected": ":red-badge[Rejected]",
}
NAV = {
    "planner": ["Set up & run", "Plan", "Warnings", "Run history"],
    "approver": ["Approvals", "Plan", "Warnings", "Run history"],
}
ICONS = {"Set up & run": "▶", "Plan": "▦", "Warnings": "!", "Run history": "≡", "Approvals": "✓"}


def goto(page: str, run_id: str | None = None):
    ss["_goto"] = page
    if run_id:
        ss["selected_run"] = run_id
    st.rerun()


def month_options() -> list[str]:
    today = date.today()
    out = []
    for i in range(0, 6):
        m = (today.month - 1 + i) % 12 + 1
        y = today.year + (today.month - 1 + i) // 12
        out.append(date(y, m, 1).strftime("%B %Y"))
    return out


# ======================================================================================
# Login
# ======================================================================================
def login_page():
    users, demo = load_users(st.secrets if _has_secrets() else {})
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        st.write("")
        st.write("")
        st.markdown("## Allocation Planner")
        st.caption("Monthly COM inventory allocation · sign in to continue")
        with st.form("login", border=True):
            username = st.text_input("Username", placeholder="e.g. maria")
            password = st.text_input("Password", type="password")
            ok = st.form_submit_button("Sign in", type="primary", width="stretch")
        if ok:
            user = check_login(users, username, password)
            if user:
                ss["user"] = user
                ss["_goto"] = NAV[user["role"]][0]
                st.rerun()
            else:
                st.error("Username or password is incorrect.")
        st.markdown("**Two roles**")
        c1, c2 = st.columns(2)
        with c1, st.container(border=True):
            st.markdown("**Planner**")
            st.caption("Sets up the run, reviews the plan and warnings, submits it for approval.")
        with c2, st.container(border=True):
            st.markdown("**Approver**")
            st.caption("Reviews submitted plans and approves or sends them back with a comment.")
        if demo:
            with st.expander("Demo accounts"):
                st.markdown(
                    f"| Role | Username | Password |\n|---|---|---|\n"
                    f"| Planner | `maria` | `{DEMO_PASSWORDS['maria']}` |\n"
                    f"| Approver | `david` | `{DEMO_PASSWORDS['david']}` |"
                )
                st.caption("Replace these with real accounts in Streamlit secrets before sharing widely.")


def _has_secrets() -> bool:
    try:
        return len(st.secrets) > 0
    except Exception:
        return False


# ======================================================================================
# Shared pieces
# ======================================================================================
def visible_runs() -> list[dict]:
    runs = list_runs()
    if ss["user"]["role"] == "approver":
        runs = [r for r in runs if r["status"] != "Draft"]
    return runs


def run_label(r: dict) -> str:
    return f"{r['month']} · {r['id']} · {r['status']}"


def pick_run() -> dict | None:
    runs = visible_runs()
    if not runs:
        if ss["user"]["role"] == "planner":
            st.info("No runs yet. Start one from **Set up & run**.")
            if st.button("Go to Set up & run", type="primary"):
                goto("Set up & run")
        else:
            st.info("No plans have been submitted for approval yet.")
        return None
    ids = [r["id"] for r in runs]
    sel = ss.get("selected_run")
    idx = ids.index(sel) if sel in ids else 0
    chosen = st.selectbox("Run", ids, index=idx, format_func=lambda i: run_label(next(r for r in runs if r["id"] == i)),
                          label_visibility="collapsed")
    ss["selected_run"] = chosen
    return get_run(chosen)


def run_header(run: dict, title: str):
    st.markdown(f"## {title} · {run['month']}")
    st.markdown(f"{STATUS_BADGE.get(run['status'], run['status'])} &nbsp; "
                f"<span style='color:#5E5E5A'>{run['id']} · run by {run['created_by']} on {run['created_at']}</span>",
                unsafe_allow_html=True)


# ======================================================================================
# Planner: Set up & run
# ======================================================================================
def page_setup():
    st.markdown("## New allocation run")
    st.caption("Set the month and warehouse limits. Checks run before the solver starts.")

    ss.setdefault("limits_df", sample_limits())
    ss.setdefault("capacity_df", sample_capacity())
    ss.setdefault("products_df", sample_products())

    left, right = st.columns([2, 1], gap="large")
    with left:
        st.selectbox("Target month", month_options(), key="month")

        st.markdown("#### Warehouse share limits")
        st.caption("Min / max share of all units each warehouse should hold.")
        ss["limits_df"] = st.data_editor(
            ss["limits_df"], key="limits_editor", width="stretch",
            column_config={
                "min_pct": st.column_config.NumberColumn("Min %", min_value=0, max_value=100, step=1, format="%d%%"),
                "max_pct": st.column_config.NumberColumn("Max %", min_value=0, max_value=100, step=1, format="%d%%"),
            })

        st.markdown("#### Warehouse space (units)")
        ss["capacity_df"] = st.data_editor(
            ss["capacity_df"], key="capacity_editor", width="stretch",
            column_config={k: st.column_config.NumberColumn(k, min_value=0, step=50, format="%d")
                           for k in STORAGE_TYPES})

        with st.expander("Products & demand"):
            st.caption("Demand per zone for the target month. Rules: *must_be_in* / *forbidden_in*; "
                       "medium-velocity products can only use their allowed warehouses.")
            ss["products_df"] = st.data_editor(
                ss["products_df"], key="products_editor", width="stretch", hide_index=True,
                disabled=["product_type", "description", "storage", "velocity", "rule", "rule_cfcs",
                          "allowed_cfcs", "cost_factor"],
                column_config={
                    "product_type": "Product type", "description": "Description",
                    "storage": "Storage", "velocity": "Velocity",
                    "demand_East": st.column_config.NumberColumn("East", min_value=0, format="%d"),
                    "demand_Central": st.column_config.NumberColumn("Central", min_value=0, format="%d"),
                    "demand_West": st.column_config.NumberColumn("West", min_value=0, format="%d"),
                    "order_qty": st.column_config.NumberColumn("Order qty (no demand)", min_value=0, format="%d"),
                    "rule": "Rule", "rule_cfcs": "Rule warehouses", "allowed_cfcs": "Allowed (medium)",
                    "cost_factor": "Cost factor",
                })

        with st.expander("Solver & rule settings"):
            ss.setdefault("penalty", 50.0)
            ss.setdefault("time_limit", 60)
            ss.setdefault("count_rule", True)
            c1, c2 = st.columns(2)
            c1.number_input("Overflow penalty ($ per unit over capacity)", min_value=0.0, step=5.0, key="penalty")
            c2.number_input("Time limit (seconds)", min_value=5, step=5, key="time_limit")
            c1.selectbox("Slow-seller tie-break", ["Lowest cost", "Keep last month's", "Random"], key="tie_break",
                         help="'Random' is how the current script works; it can change the plan between runs.")
            c2.selectbox("Missing cost fill", ["Warehouse average", "Minimum cost (legacy)"], key="cost_fill",
                         help="The legacy option makes missing lanes the cheapest — not recommended.")
            st.checkbox("Count rule-based stock in warehouse shares", key="count_rule")

        if st.button("Reset to sample data"):
            for k in ["limits_df", "capacity_df", "products_df", "limits_editor", "capacity_editor", "products_editor"]:
                ss.pop(k, None)
            st.rerun()

    with right:
        checks = validate(ss["products_df"], ss["capacity_df"], ss["limits_df"])
        has_error = any(c["level"] == "error" for c in checks)
        with st.container(border=True):
            st.markdown("#### Pre-run checks")
            icon = {"ok": ":green[✓]", "warn": ":orange[!]", "info": ":blue[i]", "error": ":red[✕]"}
            for c in checks:
                st.markdown(f"{icon[c['level']]} &nbsp; {c['text']}")
        run_clicked = st.button("Run allocation", type="primary", width="stretch", disabled=has_error)
        st.caption("Blocked until every ✕ check passes." if has_error
                   else "Runs the rules, then the optimiser. Usually takes a few seconds.")

    if run_clicked:
        settings = Settings(
            overflow_penalty=float(ss.get("penalty", 50.0)),
            time_limit_s=float(ss.get("time_limit", 60)),
            slow_tie_break=ss.get("tie_break", "Lowest cost"),
            missing_cost_fill=ss.get("cost_fill", "Warehouse average"),
            count_rule_stock_in_shares=bool(ss.get("count_rule", True)),
        )
        with st.status("Running allocation…", expanded=True) as status:
            res = run_allocation(ss["products_df"], ss["capacity_df"], ss["limits_df"], settings,
                                 previous_slow_map=last_approved_slow_map(),
                                 progress=lambda m: status.write(m))
            k = res["kpis"]
            status.update(label=f"Finished — {k['status']} in {k['solve_seconds']}s",
                          state="complete" if k["status"] in ("Optimal", "Suboptimal") else "error")
        run_id = "RUN-" + datetime.now().strftime("%y%m%d-%H%M%S")
        save_run({
            "id": run_id,
            "month": ss.get("month", month_options()[0]),
            "created_by": ss["user"]["name"],
            "created_by_user": ss["user"]["username"],
            "created_at": now(),
            "status": "Draft",
            "settings": asdict(settings),
            "inputs": {
                "limits": ss["limits_df"].reset_index().to_dict(orient="records"),
                "capacity": ss["capacity_df"].reset_index().to_dict(orient="records"),
            },
            "results": serialise_results(res),
            "history": [{"at": now(), "status": "Draft", "by": ss["user"]["name"]}],
        })
        goto("Plan", run_id)


# ======================================================================================
# Plan
# ======================================================================================
def page_plan():
    run = pick_run()
    if not run:
        return
    res = deserialise_results(run["results"])
    k = res["kpis"]
    run_header(run, "Allocation plan")

    msg = f"Solver: **{k['status']}** · finished in {k['solve_seconds']}s"
    if k["status"] == "Optimal":
        st.success("Best plan found. " + msg, icon=":material/check_circle:")
    elif k["status"] == "Suboptimal":
        st.warning("Stopped at the time limit — plan is usable but not proven best. " + msg)
    else:
        st.error("No valid plan. " + msg + " — see Warnings.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total shipping cost", f"${k['total_cost']:,.0f}", border=True)
    c2.metric("Units placed", f"{k['units']:,.0f}", border=True)
    opt_pct = 100 * k["optimizer_units"] / k["units"] if k["units"] else 0
    c3.metric("Optimizer vs rules", f"{opt_pct:.0f}% / {100 - opt_pct:.0f}%", border=True)
    c4.metric("Overflow units", f"{k['overflow_units']:,.0f}",
              delta=f"${k['penalty_cost']:,.0f} penalty" if k["overflow_units"] else None,
              delta_color="inverse", border=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### By warehouse and storage type")
        summ = res["summary"].copy()
        if len(summ):
            top = max(100.0, float(summ["used_pct"].max()))
            st.dataframe(summ, hide_index=True, width="stretch", column_config={
                "warehouse": "Warehouse", "storage": "Storage",
                "allocated": st.column_config.NumberColumn("Allocated", format="%d"),
                "capacity": st.column_config.NumberColumn("Space", format="%d"),
                "used_pct": st.column_config.ProgressColumn("Space used", min_value=0, max_value=top, format="%.0f%%"),
                "overflow": st.column_config.NumberColumn("Over by", format="%d"),
            })
        st.markdown("#### Warehouse share vs target")
        sh = res["shares"].copy()
        if len(sh):
            sh["target"] = sh.apply(lambda r: f"{r['min_pct']:.0f}–{r['max_pct']:.0f}%", axis=1)
            sh["in_range"] = sh.apply(lambda r: "✓" if r["min_pct"] - 0.5 <= r["share_pct"] <= r["max_pct"] + 0.5 else "✕",
                                      axis=1)
            st.dataframe(sh[["warehouse", "units", "share_pct", "target", "in_range"]], hide_index=True,
                         width="stretch", column_config={
                             "warehouse": "Warehouse", "units": st.column_config.NumberColumn("Units", format="%d"),
                             "share_pct": st.column_config.NumberColumn("Share", format="%.1f%%"),
                             "target": "Target", "in_range": "In range"})
    with right:
        st.markdown("#### Cost by warehouse ($)")
        det = res["detail"]
        if len(det):
            pv = det.pivot_table(index="warehouse", columns="method", values="cost", aggfunc="sum", fill_value=0)
            palette = {"Optimizer": "#1C1C1A", "Slow seller": "#C26A1B", "Restricted": "#2F5DA8", "No demand": "#A9A69C"}
            cols = [c for c in palette if c in pv.columns and pv[c].sum() > 0]
            st.bar_chart(pv[cols], horizontal=True, height=240, color=[palette[c] for c in cols])
        st.markdown("#### Units by storage type")
        if len(res["summary"]):
            st.bar_chart(res["summary"].pivot_table(index="warehouse", columns="storage", values="allocated"),
                         horizontal=True, height=240, stack=False, color=["#1C1C1A", "#C26A1B"])

    with st.expander("Full allocation detail"):
        det = res["detail"]
        st.dataframe(det, hide_index=True, width="stretch")
        st.download_button("Download CSV", det.to_csv(index=False).encode(), file_name=f"{run['id']}_allocation.csv",
                           mime="text/csv")

    workflow_actions(run)


def workflow_actions(run: dict):
    role = ss["user"]["role"]
    st.divider()
    st.markdown("#### Approval")
    if run.get("comment"):
        st.info(f"**{run.get('decided_by', 'Approver')}:** {run['comment']}")
    if role == "planner":
        if run["status"] in ("Draft", "Rejected"):
            note = st.text_input("Note for the approver (optional)", key=f"note_{run['id']}")
            if st.button("Submit for approval", type="primary"):
                update_run(run["id"], status="Submitted", submitted_at=now(),
                                 submit_note=note, by=ss["user"]["name"], comment=None)
                st.toast("Sent for approval")
                st.rerun()
        elif run["status"] == "Submitted":
            st.caption("Waiting for the approver.")
        else:
            st.caption(f"Approved by {run.get('decided_by')} on {run.get('decided_at')}.")
    else:
        if run.get("submit_note"):
            st.caption(f"Planner’s note: {run['submit_note']}")
        if run["status"] == "Submitted":
            comment = st.text_area("Comment", key=f"comment_{run['id']}",
                                   placeholder="Required when sending back")
            a, r, _ = st.columns([1, 1, 3])
            if a.button("Approve plan", type="primary", width="stretch"):
                update_run(run["id"], status="Approved", decided_by=ss["user"]["name"],
                                 decided_at=now(), comment=comment or None, by=ss["user"]["name"])
                st.toast("Plan approved")
                st.rerun()
            if r.button("Send back", width="stretch"):
                if not comment.strip():
                    st.error("Add a comment so the planner knows what to change.")
                else:
                    update_run(run["id"], status="Rejected", decided_by=ss["user"]["name"],
                                     decided_at=now(), comment=comment, by=ss["user"]["name"])
                    st.toast("Sent back to planner")
                    st.rerun()
        else:
            st.caption(f"Status: {run['status']}.")
    with st.expander("History"):
        hist = pd.DataFrame(run.get("history", []))
        st.dataframe(hist, hide_index=True, width="stretch")


# ======================================================================================
# Warnings
# ======================================================================================
def page_warnings():
    run = pick_run()
    if not run:
        return
    res = deserialise_results(run["results"])
    run_header(run, "Things to check")

    warns = res.get("warnings", [])
    order = {"High": 0, "Medium": 1, "Info": 2}
    if not warns:
        st.success("Nothing to check for this run.")
    for w in sorted(warns, key=lambda w: order.get(w["severity"], 3)):
        badge = {"High": ":red-badge[High]", "Medium": ":orange-badge[Medium]"}.get(w["severity"], ":blue-badge[Info]")
        with st.container(border=True):
            st.markdown(f"{badge} &nbsp; **{w['title']}**")
            st.caption(w["detail"])

    t1, t2, t3 = st.tabs(["Rule-based placements", "Slow sellers", "Filled cost lanes"])
    det = res["detail"]
    with t1:
        rb = det[det["method"] != "Optimizer"] if len(det) else det
        if len(rb):
            g = (rb.groupby(["product_type", "method", "warehouse"], as_index=False)["units"].sum()
                   .rename(columns={"product_type": "Product type", "method": "Reason", "warehouse": "Warehouse",
                                    "units": "Units"}))
            st.dataframe(g, hide_index=True, width="stretch")
        else:
            st.caption("No rule-based placements.")
    with t2:
        sn = res["slow_notes"]
        if len(sn):
            st.dataframe(sn.rename(columns={"product_type": "Product type", "warehouse": "Warehouse", "how": "Chosen by",
                                            "last_month": "Last approved", "changed": "Changed"}),
                         hide_index=True, width="stretch")
        else:
            st.caption("No slow sellers in this run.")
    with t3:
        fl = res["filled_lanes"]
        if len(fl):
            st.dataframe(fl, hide_index=True, width="stretch")
        else:
            st.caption("All cost lanes had data.")


# ======================================================================================
# Approver: queue
# ======================================================================================
def page_approvals():
    st.markdown("## Approvals")
    runs = list_runs()
    pending = [r for r in runs if r["status"] == "Submitted"]
    st.caption(f"{len(pending)} plan(s) waiting for you.")
    if not pending:
        st.info("Nothing to approve right now.")
    for r in pending:
        k = r["results"]["kpis"]
        with st.container(border=True):
            a, b, c, d, e = st.columns([2.2, 1.2, 1.2, 1.2, 1])
            a.markdown(f"**{r['month']}**  \n<span style='color:#5E5E5A'>{r['id']} · {r['created_by']} · "
                       f"submitted {r.get('submitted_at', '')}</span>", unsafe_allow_html=True)
            b.metric("Cost", f"${k['total_cost']:,.0f}")
            c.metric("Units", f"{k['units']:,.0f}")
            d.metric("Warnings", len(r["results"].get("warnings", [])))
            if e.button("Review", key=f"rev_{r['id']}", type="primary", width="stretch"):
                goto("Plan", r["id"])
    decided = [r for r in runs if r["status"] in ("Approved", "Rejected")][:10]
    if decided:
        st.markdown("#### Recently decided")
        st.dataframe(pd.DataFrame([{"Run": r["id"], "Month": r["month"], "Status": r["status"],
                                    "By": r.get("decided_by"), "On": r.get("decided_at"),
                                    "Comment": r.get("comment") or ""} for r in decided]),
                     hide_index=True, width="stretch")


# ======================================================================================
# Run history
# ======================================================================================
def page_history():
    st.markdown("## Run history")
    runs = visible_runs()
    if not runs:
        st.info("No runs yet.")
        return
    df = pd.DataFrame([{
        "Run": r["id"], "Month": r["month"], "Status": r["status"], "Run by": r["created_by"],
        "Created": r["created_at"], "Solver": r["results"]["kpis"]["status"],
        "Cost ($)": r["results"]["kpis"]["total_cost"], "Overflow": r["results"]["kpis"]["overflow_units"],
    } for r in runs])
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={"Cost ($)": st.column_config.NumberColumn(format="$%d")})
    c1, c2 = st.columns([3, 1])
    pick = c1.selectbox("Open a run", [r["id"] for r in runs],
                        format_func=lambda i: run_label(next(r for r in runs if r["id"] == i)))
    c2.write("")
    if c2.button("Open plan", width="stretch"):
        goto("Plan", pick)


# ======================================================================================
# Main
# ======================================================================================
def main():
    if "user" not in ss:
        login_page()
        return
    user = ss["user"]
    pages = NAV[user["role"]]
    if "_goto" in ss:
        target = ss.pop("_goto")
        if target in pages:
            ss["page"] = target
    if ss.get("page") not in pages:
        ss["page"] = pages[0]

    with st.sidebar:
        st.markdown("### Allocation Planner")
        st.markdown(f"**{user['name']}**  \n{user['role'].title()}")
        st.radio("Navigate", pages, key="page", label_visibility="collapsed",
                 format_func=lambda p: f"{ICONS.get(p, '')}  {p}")
        if user["role"] == "approver":
            n = sum(1 for r in list_runs() if r["status"] == "Submitted")
            if n:
                st.caption(f"{n} plan(s) waiting for approval")
        st.divider()
        if st.button("Log out", width="stretch"):
            for k in list(ss.keys()):
                del ss[k]
            st.rerun()

    {"Set up & run": page_setup, "Plan": page_plan, "Warnings": page_warnings,
     "Run history": page_history, "Approvals": page_approvals}[ss["page"]]()


main()
