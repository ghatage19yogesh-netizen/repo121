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
    "yogesh": {"name": "Yogesh", "role": "planner", "salt": _DEMO_SALT_1,
              "password_hash": hash_password("planner123", _DEMO_SALT_1)},
    "kshitij": {"name": "Kshitij", "role": "approver", "salt": _DEMO_SALT_2,
              "password_hash": hash_password("approver123", _DEMO_SALT_2)},
}
DEMO_PASSWORDS = {"yogesh": "planner123", "kshitij": "approver123"}


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
PERSIST_KEYS = ["month", "penalty", "time_limit", "tie_break", "cost_fill", "count_rule",
                "pipe_env", "pipe_month", "pipe_classes", "pipe_use_setup"]
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
    "planner": ["Set up & run", "Pipeline run", "Plan", "Warnings", "Publish to BigQuery", "Run history"],
    "approver": ["Approvals", "Plan", "Warnings", "Publish to BigQuery", "Run history"],
}
ICONS = {"Set up & run": "▶", "Pipeline run": "⚙", "Publish to BigQuery": "⇪", "Plan": "▦", "Warnings": "!", "Run history": "≡", "Approvals": "✓"}


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
            username = st.text_input("Username", placeholder="e.g. yogesh")
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
                    f"| Planner | `yogesh` | `{DEMO_PASSWORDS['yogesh']}` |\n"
                    f"| Approver | `kshitij` | `{DEMO_PASSWORDS['kshitij']}` |"
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
    pub = " :violet-badge[Published to BigQuery]" if run.get("publications") else ""
    st.markdown(f"{STATUS_BADGE.get(run['status'], run['status'])}{pub} &nbsp; "
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
# Pipeline run — model_main.py, step by step (demo)
# ======================================================================================
import io as _io
import pickle as _pickle

# Stands in for config.PIPELINE_CONFIG() in the real repo
PIPELINE_CONFIG = {
    "PIPELINE_NAME": "scs-com-allocation",
    "GCP_PROJECT": "[gcp-project]",
    "BUCKET_NAME": "[allocation-bucket]",
    "PROJECT_INTERIM_FOLDER": "interim",
}
# dept number -> product class (used by split_data_by_class_and_vnd)
PRODUCT_CLASSES = {"101": "Tops", "202": "Dresses & suits", "305": "Outerwear", "410": "Accessories",
                   "415": "Accessories", "520": "Hosiery", "610": "Jewellery"}
GCS_DIR = DATA_DIR / "gcs"  # local folder playing the role of the GCS bucket in this demo

PIPE_STEPS = [
    ("connect", "Connect to BigQuery", "get_gbq_client(env)"),
    ("download", "Download query_parameters.pkl", "s_client.download_file_from_gcs(...)"),
    ("load", "Load pickle", "clf = load_pickle(local_file_path)"),
    ("unpack", "Unpack sets / params / options / constraints", "sets, params, options, constraints = clf[0..3]"),
    ("split", "Split data by class and vendor", "split_data_by_class_and_vnd(...)"),
    ("optimize", "Run optimization", "results = run_optimization(...)"),
    ("save", "Save results_output.pkl", "pickle.dump(results, f)"),
    ("upload", "Upload results to GCS", "s_client.upload_file_to_gcs(...)"),
]


def product_class(product_type: str) -> str:
    return PRODUCT_CLASSES.get(str(product_type).split("-")[0], "Other")


def build_query_parameters(products, capacity, limits, settings):
    """Pipeline steps 1–2 (separate scripts in the real repo): build the 4 objects stored in query_parameters.pkl."""
    dem_cols = [f"demand_{z}" for z in ZONES]
    tot = products[dem_cols].sum(axis=1)
    groups = {"ok": [], "low_vel": [], "no_demand": [], "restricted": []}
    for i, r in products.iterrows():
        if tot[i] == 0 and r["order_qty"] > 0:
            groups["no_demand"].append(r["product_type"])
        elif r["rule"] == "must_be_in":
            groups["restricted"].append(r["product_type"])
        elif r["velocity"] == "low":
            groups["low_vel"].append(r["product_type"])
        else:
            groups["ok"].append(r["product_type"])
    costs, _ = build_costs(products, settings)
    sets = {"cfcs": list(WAREHOUSES), "destinations": list(ZONES), "storages": list(STORAGE_TYPES),
            "products": groups}
    params = {
        "costs": costs,
        "demand": {(z, r["product_type"]): float(r[f"demand_{z}"]) for _, r in products.iterrows() for z in ZONES},
        "supply": dict(zip(products["product_type"], (tot + products["order_qty"]).astype(float))),
        "product_storage": dict(zip(products["product_type"], products["storage"])),
        "product_table": products.copy(),
    }
    options = {"settings": asdict(settings), "storage_sizes": {}}
    constraints = {
        "storage": capacity.copy(),
        "min_alloc": (limits["min_pct"] / 100).to_dict(),
        "max_alloc": (limits["max_pct"] / 100).to_dict(),
        "exceptions": products.loc[products["rule"] != "", ["product_type", "rule", "rule_cfcs"]].reset_index(drop=True),
        "med_velocity": products.loc[products["velocity"] == "medium",
                                     ["product_type", "allowed_cfcs"]].reset_index(drop=True),
    }
    return (sets, params, options, constraints)


def split_data_by_class_and_vnd(sets, params, constraints, classes):
    """Keep only the products whose class is in `classes` (demo version of the repo helper)."""
    keep = {p for p in params["product_table"]["product_type"] if product_class(p) in classes}
    sets = dict(sets, products={g: [p for p in ps if p in keep] for g, ps in sets["products"].items()})
    params = dict(params,
                  costs={k: v for k, v in params["costs"].items() if k[2] in keep},
                  demand={k: v for k, v in params["demand"].items() if k[1] in keep},
                  supply={k: v for k, v in params["supply"].items() if k in keep},
                  product_storage={k: v for k, v in params["product_storage"].items() if k in keep},
                  product_table=params["product_table"][params["product_table"]["product_type"].isin(keep)]
                  .reset_index(drop=True))
    constraints = dict(constraints,
                       exceptions=constraints["exceptions"][constraints["exceptions"]["product_type"].isin(keep)],
                       med_velocity=constraints["med_velocity"][constraints["med_velocity"]["product_type"].isin(keep)])
    return sets, params, constraints


def run_optimization(sets, params, constraints, options):
    """Demo version of src.optim.optimization.run_optimization: rules + LP via run_allocation()."""
    products = params["product_table"]
    capacity = constraints["storage"]
    limits = pd.DataFrame({"min_pct": pd.Series(constraints["min_alloc"]) * 100,
                           "max_pct": pd.Series(constraints["max_alloc"]) * 100}).loc[sets["cfcs"]]
    settings = Settings(**options["settings"])
    return run_allocation(products, capacity, limits, settings, previous_slow_map=last_approved_slow_map())


def _tracker(state: dict) -> str:
    out = []
    for i, (key, label, _) in enumerate(PIPE_STEPS, 1):
        s = state.get(key, "todo")
        badge = {"todo": "gray", "run": "blue", "done": "green", "fail": "red"}[s]
        mark = {"todo": "", "run": "… ", "done": "✓ ", "fail": "✕ "}[s]
        out.append(f":{badge}-badge[{mark}{i} · {label}]")
    return " → ".join(out)


def page_pipeline():
    st.markdown("## Pipeline run · `model_main.py`")
    st.caption("Runs `execute_model()` step by step, the way the production job does: download the inputs, "
               "unpack them, split by class, optimize, save and upload the results. Demo mode: BigQuery and "
               "GCS are simulated and the inputs come from the sample data.")

    with st.expander("Run settings", expanded=True):
        c1, c2, c3 = st.columns([1, 1, 2])
        env = c1.selectbox("Environment", ["dev", "prod"], key="pipe_env",
                           help="The original script hardcodes 'prod'. Here it's a setting.")
        month = c2.selectbox("Target month", month_options(), key="pipe_month")
        all_classes = sorted(set(PRODUCT_CLASSES.values()))
        ss.setdefault("pipe_classes", all_classes)
        ss.setdefault("pipe_use_setup", True)
        classes = c3.multiselect("Product classes (split_data_by_class_and_vnd)", all_classes, key="pipe_classes")
        use_setup = st.checkbox("Use the inputs edited on **Set up & run** (otherwise sample data)",
                                key="pipe_use_setup")
        st.markdown("**Config** (`config.PIPELINE_CONFIG()`)")
        st.code("\n".join(f"{k} = {v!r}" for k, v in PIPELINE_CONFIG.items()), language="python")

    with st.expander("How this maps to model_main.py"):
        st.dataframe(pd.DataFrame([{"#": i, "Step": lbl, "Code in model_main.py": code}
                                   for i, (_, lbl, code) in enumerate(PIPE_STEPS, 1)]),
                     hide_index=True, width="stretch")

    run_clicked = st.button("▶  Run execute_model()", type="primary", disabled=not classes)
    if not classes:
        st.caption("Pick at least one product class.")

    if run_clicked:
        _execute_model_demo(env, month, classes, use_setup)

    last = ss.get("pipeline_last")
    if last:
        _render_pipeline_result(last)


def _execute_model_demo(env, month, classes, use_setup):
    state = {}
    tracker = st.empty()
    log_box = st.empty()
    lines = []

    def log(msg, level="INFO"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        lines.append(f"{ts} - root - {level} - {msg}")
        log_box.code("\n".join(lines[-14:]), language="log")

    def step(key):
        state[key] = "run"
        tracker.markdown(_tracker(state))
        time.sleep(0.35)

    def done(key):
        state[key] = "done"
        tracker.markdown(_tracker(state))

    products = ss.get("products_df") if use_setup and "products_df" in ss else sample_products()
    capacity = ss.get("capacity_df") if use_setup and "capacity_df" in ss else sample_capacity()
    limits = ss.get("limits_df") if use_setup and "limits_df" in ss else sample_limits()
    settings = Settings(
        overflow_penalty=float(ss.get("penalty", 50.0)), time_limit_s=float(ss.get("time_limit", 60)),
        slow_tie_break=ss.get("tie_break", "Lowest cost"), missing_cost_fill=ss.get("cost_fill", "Warehouse average"),
        count_rule_stock_in_shares=bool(ss.get("count_rule", True)))

    folder = GCS_DIR / PIPELINE_CONFIG["PROJECT_INTERIM_FOLDER"]
    folder.mkdir(parents=True, exist_ok=True)
    gs = f"gs://{PIPELINE_CONFIG['BUCKET_NAME']}/{PIPELINE_CONFIG['PROJECT_INTERIM_FOLDER']}"
    art = {}
    try:
        # pipeline steps 1–2 happen in other scripts; in the demo we create their output file first
        (folder / "query_parameters.pkl").write_bytes(
            _pickle.dumps(build_query_parameters(products, capacity, limits, settings)))

        step("connect")
        log(f"Connected to BigQuery ({env}) as pdq, gbq_client [simulated]")
        log("Beginning to run the OPTIMIZATION ...")
        done("connect")

        step("download")
        local_file_path = "query_parameters.pkl"
        raw = (folder / local_file_path).read_bytes()
        log(f"Downloading {gs}/{local_file_path} -> ./{local_file_path} ({len(raw) / 1024:.1f} KB)")
        log("Query parameters file download completed...")
        done("download")

        step("load")
        clf = _pickle.loads(raw)
        log(f"Loaded pickle: tuple of {len(clf)} objects")
        done("load")

        step("unpack")
        sets, params, options, constraints = clf[0], clf[1], clf[2], clf[3]
        log("Parameters of `sets`, `params`, `options`, `constraints` are loaded into memory")
        art["before"] = {g: len(v) for g, v in sets["products"].items()}
        art["sets"], art["params"], art["options"], art["constraints"] = sets, params, options, constraints
        done("unpack")

        step("split")
        log("Determining which data is required at class level")
        sets, params, constraints = split_data_by_class_and_vnd(sets, params, constraints, classes)
        art["after"] = {g: len(v) for g, v in sets["products"].items()}
        art["classes"] = classes
        log(f"Kept {sum(art['after'].values())} of {sum(art['before'].values())} product types "
            f"in classes: {', '.join(classes)}")
        if not len(params["product_table"]):
            raise ValueError("No products left after the class split")
        done("split")

        step("optimize")
        results = run_optimization(sets, params, constraints, options)
        k = results["kpis"]
        log(f"OPTIMIZATION COMPLETED — status {k['status']}, cost ${k['total_cost']:,.0f}, "
            f"{k['units']:,.0f} units, {k['overflow_units']:,.0f} overflow units",
            "INFO" if k["status"] in ("Optimal", "Suboptimal") else "ERROR")
        done("optimize")

        step("save")
        out_name = "results_output.pkl"
        out_bytes = _pickle.dumps(results)
        log(f"Results dumped to ./{out_name} ({len(out_bytes) / 1024:.1f} KB)")
        done("save")

        step("upload")
        (folder / out_name).write_bytes(out_bytes)
        log(f"Uploaded ./{out_name} -> {gs}/{out_name}")
        done("upload")

        run_id = "RUN-" + datetime.now().strftime("%y%m%d-%H%M%S")
        save_run({
            "id": run_id, "month": month, "created_by": ss["user"]["name"],
            "created_by_user": ss["user"]["username"], "created_at": now(), "status": "Draft",
            "source": "Pipeline (model_main.py)", "settings": asdict(settings),
            "inputs": {"classes": classes, "env": env},
            "results": serialise_results(results),
            "history": [{"at": now(), "status": "Draft", "by": ss["user"]["name"]}],
        })
        log(f"Run saved as {run_id} (Draft)")
        ss["pipeline_last"] = {"ok": True, "log": lines, "run_id": run_id, "art": art, "results": results,
                               "query_bytes": raw, "results_bytes": out_bytes, "gs": gs, "env": env}
    except Exception as e:  # show where it failed, like the job logs would
        failed = next((key for key, *_ in PIPE_STEPS if state.get(key) == "run"), None)
        if failed:
            state[failed] = "fail"
        tracker.markdown(_tracker(state))
        log(f"{type(e).__name__}: {e}", "ERROR")
        ss["pipeline_last"] = {"ok": False, "log": lines, "art": art, "error": str(e), "state": state}
    tracker.empty()
    log_box.empty()


def _render_pipeline_result(last):
    if last["ok"]:
        st.markdown(_tracker({k: "done" for k, *_ in PIPE_STEPS}))
        st.success(f"execute_model() finished · run **{last['run_id']}** saved as Draft "
                   f"(environment: {last['env']})")
    else:
        st.markdown(_tracker(last.get("state", {})))
        st.error(f"execute_model() failed: {last['error']}")
    st.markdown("#### Log")
    st.code("\n".join(last["log"]), language="log")
    if not last["ok"]:
        return

    art, res = last["art"], last["results"]
    t1, t2, t3, t4, t5, t6 = st.tabs(["sets", "params", "options", "constraints", "split", "results"])
    with t1:
        st.caption("`sets = clf[0]`: what the problem is about")
        s = art["sets"]
        c = st.columns(4)
        c[0].metric("CFCs", len(s["cfcs"]))
        c[1].metric("Destinations", len(s["destinations"]))
        c[2].metric("Storage types", len(s["storages"]))
        c[3].metric("Product types", sum(len(v) for v in s["products"].values()))
        st.dataframe(pd.DataFrame([{"Group": g, "Count": len(v), "Product types": ", ".join(v)}
                                   for g, v in s["products"].items()]), hide_index=True, width="stretch")
    with t2:
        st.caption("`params = clf[1]`: costs, demand, supply and the product → storage map")
        p = art["params"]
        dem = pd.Series(p["demand"]).unstack(0)[ZONES]
        dem["storage"] = pd.Series(p["product_storage"])
        st.markdown("**Demand by product and zone**")
        st.dataframe(dem, width="stretch")
        st.markdown("**Shipping cost per unit (first product type)**")
        first = next(iter(p["product_storage"]))
        cost = pd.DataFrame({z: {w: p["costs"][(w, z, first)] for w in WAREHOUSES} for z in ZONES})
        st.caption(first)
        st.dataframe(cost.round(2), width="stretch")
    with t3:
        st.caption("`options = clf[2]`: run settings")
        st.json(art["options"])
    with t4:
        st.caption("`constraints = clf[3]`: space, warehouse shares, exceptions, medium-velocity limits")
        cs = art["constraints"]
        a, b = st.columns(2)
        a.markdown("**Storage capacity**")
        a.dataframe(cs["storage"], width="stretch")
        b.markdown("**Min / max share**")
        b.dataframe(pd.DataFrame({"min": cs["min_alloc"], "max": cs["max_alloc"]}), width="stretch")
        a.markdown("**Exceptions**")
        a.dataframe(cs["exceptions"], hide_index=True, width="stretch")
        b.markdown("**Medium velocity (allowed CFCs)**")
        b.dataframe(cs["med_velocity"], hide_index=True, width="stretch")
    with t5:
        st.caption(f"Classes kept: {', '.join(art['classes'])}")
        st.dataframe(pd.DataFrame({"Before split": art["before"], "After split": art["after"]}),
                     width="stretch")
    with t6:
        k = res["kpis"]
        c = st.columns(4)
        c[0].metric("Solver", k["status"], border=True)
        c[1].metric("Shipping cost", f"${k['total_cost']:,.0f}", border=True)
        c[2].metric("Units", f"{k['units']:,.0f}", border=True)
        c[3].metric("Overflow units", f"{k['overflow_units']:,.0f}", border=True)
        st.dataframe(res["summary"], hide_index=True, width="stretch")
        d1, d2, d3 = st.columns(3)
        d1.download_button("Download query_parameters.pkl", last["query_bytes"], "query_parameters.pkl",
                           width="stretch")
        d2.download_button("Download results_output.pkl", last["results_bytes"], "results_output.pkl",
                           width="stretch")
        if d3.button("Open in Plan →", type="primary", width="stretch"):
            goto("Plan", last["run_id"])
        st.caption(f"Files in the (simulated) bucket: {last['gs']}/query_parameters.pkl, "
                   f"{last['gs']}/results_output.pkl")



# ======================================================================================
# Publish to BigQuery — write approved optimizer outputs to BigQuery tables
# Demo mode (default): a local SQLite file stands in for the BigQuery dataset.
# Production mode: add [gcp_service_account] (+ optional [bigquery]) to Streamlit secrets
# and `google-cloud-bigquery`, `pyarrow` to requirements.txt.
# ======================================================================================
import re as _re
import sqlite3 as _sqlite3

BQ_DEMO_DB = DATA_DIR / "bigquery_demo.sqlite"

BQ_TABLES = {
    "allocation_plan": {
        "desc": "One row per warehouse × zone × product: units to stock and ship, cost and how it was decided.",
        "schema": [("run_id", "STRING"), ("target_month", "DATE"), ("warehouse", "STRING"), ("zone", "STRING"),
                   ("product_type", "STRING"), ("storage", "STRING"), ("units", "FLOAT64"),
                   ("unit_cost", "FLOAT64"), ("cost", "FLOAT64"), ("method", "STRING"),
                   ("published_at", "TIMESTAMP")]},
    "capacity_summary": {
        "desc": "Space used per warehouse × storage type, with overflow above capacity.",
        "schema": [("run_id", "STRING"), ("target_month", "DATE"), ("warehouse", "STRING"), ("storage", "STRING"),
                   ("allocated", "FLOAT64"), ("capacity", "FLOAT64"), ("used_pct", "FLOAT64"),
                   ("overflow", "FLOAT64"), ("published_at", "TIMESTAMP")]},
    "warehouse_share": {
        "desc": "Each warehouse's share of all units vs its min / max target.",
        "schema": [("run_id", "STRING"), ("target_month", "DATE"), ("warehouse", "STRING"), ("units", "FLOAT64"),
                   ("share_pct", "FLOAT64"), ("min_pct", "FLOAT64"), ("max_pct", "FLOAT64"),
                   ("published_at", "TIMESTAMP")]},
    "run_log": {
        "desc": "One row per published run: solver status, KPIs, who ran / approved / published it, settings.",
        "schema": [("run_id", "STRING"), ("target_month", "DATE"), ("run_status", "STRING"),
                   ("solver_status", "STRING"), ("total_cost", "FLOAT64"), ("units", "FLOAT64"),
                   ("optimizer_units", "FLOAT64"), ("rule_units", "FLOAT64"), ("overflow_units", "FLOAT64"),
                   ("penalty_cost", "FLOAT64"), ("created_by", "STRING"), ("created_at", "STRING"),
                   ("approved_by", "STRING"), ("approved_at", "STRING"), ("published_by", "STRING"),
                   ("published_at", "TIMESTAMP"), ("settings_json", "STRING"), ("warnings_json", "STRING")]},
}
_ID_PROJECT = _re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
_ID_NAME = _re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def bq_secrets() -> dict:
    try:
        return dict(st.secrets["bigquery"]) if "bigquery" in st.secrets else {}
    except Exception:
        return {}


def bq_live() -> bool:
    try:
        return "gcp_service_account" in st.secrets
    except Exception:
        return False


def target_month_date(month_label: str) -> str:
    try:
        return datetime.strptime(month_label, "%B %Y").date().isoformat()
    except ValueError:
        return date.today().replace(day=1).isoformat()


def build_output_tables(run: dict, published_by: str, published_at: str) -> dict:
    res = deserialise_results(run["results"])
    k = res["kpis"]
    month = target_month_date(run["month"])
    common = {"run_id": run["id"], "target_month": month}

    def frame(df, name):
        df = df.copy()
        for c, v in common.items():
            df[c] = v
        df["published_at"] = published_at
        cols = [c for c, _ in BQ_TABLES[name]["schema"]]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        return df[cols].reset_index(drop=True)

    tables = {
        "allocation_plan": frame(res["detail"], "allocation_plan"),
        "capacity_summary": frame(res["summary"], "capacity_summary"),
        "warehouse_share": frame(res["shares"], "warehouse_share"),
        "run_log": pd.DataFrame([{
            **common, "run_status": run["status"], "solver_status": k["status"],
            "total_cost": k["total_cost"], "units": k["units"], "optimizer_units": k["optimizer_units"],
            "rule_units": k["rule_units"], "overflow_units": k["overflow_units"], "penalty_cost": k["penalty_cost"],
            "created_by": run.get("created_by"), "created_at": run.get("created_at"),
            "approved_by": run.get("decided_by") if run["status"] == "Approved" else None,
            "approved_at": run.get("decided_at") if run["status"] == "Approved" else None,
            "published_by": published_by, "published_at": published_at,
            "settings_json": json.dumps(run.get("settings", {})),
            "warnings_json": json.dumps(run["results"].get("warnings", [])),
        }])[[c for c, _ in BQ_TABLES["run_log"]["schema"]]],
    }
    return tables


def publish_checks(run, tables, project, dataset, table_names, allow_unapproved) -> list[dict]:
    out = []
    approved = run["status"] == "Approved"
    out.append({"level": "ok" if approved else ("warn" if allow_unapproved else "error"),
                "text": "Run is approved" if approved else f"Run is {run['status']}, not approved"})
    ok_ids = bool(_ID_PROJECT.match(project or "")) and bool(_ID_NAME.match(dataset or "")) and \
        all(_ID_NAME.match(t or "") for t in table_names.values())
    out.append({"level": "ok" if ok_ids else "error",
                "text": "Project, dataset and table names are valid" if ok_ids
                else "Check names: project = lower-case letters/digits/hyphens (6–30), dataset/table = letters, digits, _"})
    plan = tables["allocation_plan"]
    k = run["results"]["kpis"]
    match = abs(plan["units"].sum() - k["units"]) < 1
    out.append({"level": "ok" if match else "error",
                "text": f"Plan rows add up to the run total ({plan['units'].sum():,.0f} units)"})
    neg = int((plan["units"] < 0).sum())
    out.append({"level": "ok" if not neg else "error", "text": "No negative units" if not neg else f"{neg} rows with negative units"})
    nulls = int(plan[["warehouse", "product_type", "method"]].isna().sum().sum())
    out.append({"level": "ok" if not nulls else "error",
                "text": "No missing keys (warehouse, product, method)" if not nulls else f"{nulls} missing key values"})
    out.append({"level": "ok" if k["status"] == "Optimal" else "warn", "text": f"Solver status: {k['status']}"})
    if k.get("overflow_units"):
        out.append({"level": "warn", "text": f"{k['overflow_units']:,.0f} overflow units will be published as planned"})
    return out


# ---------- demo backend: SQLite pretending to be a BigQuery dataset -------------------
def _demo_table(dataset, table):
    return f"{dataset}__{table}"


def demo_publish(tables, dataset, table_names, month, mode) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = {}
    with _sqlite3.connect(BQ_DEMO_DB) as con:
        for key, df in tables.items():
            t = _demo_table(dataset, table_names[key])
            exists = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone()
            if exists and mode == "replace":
                con.execute(f'DELETE FROM "{t}" WHERE target_month = ?', (month,))
            df.to_sql(t, con, if_exists="append", index=False)
            rows[key] = len(df)
    return rows


def demo_query(sql: str, project: str, dataset: str) -> pd.DataFrame:
    q = sql.strip().rstrip(";")
    if not _re.match(r"(?is)^\s*(select|with)\b", q) or ";" in q:
        raise ValueError("Only a single SELECT query is allowed here.")
    q = q.replace("`", "")
    q = _re.sub(rf"\b{_re.escape(project)}\.{_re.escape(dataset)}\.(\w+)", rf"{dataset}__\1", q)
    q = _re.sub(rf"\b{_re.escape(dataset)}\.(\w+)", rf"{dataset}__\1", q)
    con = _sqlite3.connect(f"file:{BQ_DEMO_DB}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(q, con)
    finally:
        con.close()


def demo_inventory(dataset) -> pd.DataFrame:
    if not BQ_DEMO_DB.exists():
        return pd.DataFrame()
    out = []
    with _sqlite3.connect(BQ_DEMO_DB) as con:
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
                                           (f"{dataset}__%",))]
        for n in names:
            cnt, months, runs = con.execute(
                f'SELECT COUNT(*), COUNT(DISTINCT target_month), COUNT(DISTINCT run_id) FROM "{n}"').fetchone()
            out.append({"Table": f"{dataset}.{n.split('__', 1)[1]}", "Rows": cnt, "Months": months, "Runs": runs})
    return pd.DataFrame(out)


# ---------- production backend: google-cloud-bigquery -----------------------------------
def bq_client(project: str, location: str):
    from google.cloud import bigquery
    from google.oauth2 import service_account
    info = dict(st.secrets["gcp_service_account"])
    creds = service_account.Credentials.from_service_account_info(info)
    return bigquery.Client(project=project or info.get("project_id"), credentials=creds, location=location)


def bq_publish(client, tables, project, dataset, table_names, month, mode, location) -> dict:
    from google.cloud import bigquery
    client.create_dataset(bigquery.Dataset(f"{project}.{dataset}"), exists_ok=True)
    rows = {}
    for key, df in tables.items():
        table_id = f"{project}.{dataset}.{table_names[key]}"
        schema = [bigquery.SchemaField(n, t) for n, t in BQ_TABLES[key]["schema"]]
        tbl = bigquery.Table(table_id, schema=schema)
        tbl.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.MONTH,
                                                          field="target_month")
        tbl.description = BQ_TABLES[key]["desc"]
        client.create_table(tbl, exists_ok=True)
        if mode == "replace":
            client.query(f"DELETE FROM `{table_id}` WHERE target_month = @m",
                         job_config=bigquery.QueryJobConfig(query_parameters=[
                             bigquery.ScalarQueryParameter("m", "DATE", month)])).result()
        df = df.copy()
        df["target_month"] = pd.to_datetime(df["target_month"]).dt.date
        df["published_at"] = pd.to_datetime(df["published_at"])
        job = client.load_table_from_dataframe(df, table_id, job_config=bigquery.LoadJobConfig(
            schema=schema, write_disposition=bigquery.WriteDisposition.WRITE_APPEND))
        job.result()
        rows[key] = job.output_rows
    return rows


def add_publication(run_id: str, pub: dict):
    with _LOCK:
        runs = _read()
        for r in runs:
            if r["id"] == run_id:
                r.setdefault("publications", []).append(pub)
                r.setdefault("history", []).append({"at": pub["at"], "status": "Published to BigQuery",
                                                    "by": pub["by"]})
        _write(runs)


def page_publish():
    live = bq_live()
    cfg = bq_secrets()
    st.markdown("## Publish to BigQuery")
    st.caption("Write the optimizer outputs of an approved run to BigQuery, so reports and downstream jobs "
               "can read them.")
    if live:
        st.success(f"Connected mode — writing to real BigQuery with the service account in secrets.",
                   icon=":material/cloud_done:")
    else:
        st.info("**Demo mode** — a local BigQuery simulator stands in for the dataset. Nothing leaves this app. "
                "Add a service account to secrets to publish for real.", icon=":material/science:")

    # ---- 1. run --------------------------------------------------------------------------
    st.markdown("#### 1 · Choose the run")
    runs = list_runs()
    allow = st.checkbox("Include runs that are not approved yet (demo only)", value=not live and
                        not any(r["status"] == "Approved" for r in runs), disabled=live, key="pub_allow")
    eligible = [r for r in runs if r["status"] == "Approved" or (allow and not live)]
    if not eligible:
        st.warning("No approved runs yet. Run an allocation, submit it, and have the approver approve it.")
        return
    ids = [r["id"] for r in eligible]
    sel = ss.get("selected_run")
    rid = st.selectbox("Run", ids, index=ids.index(sel) if sel in ids else 0,
                       format_func=lambda i: run_label(next(r for r in eligible if r["id"] == i)))
    run = get_run(rid)
    k = run["results"]["kpis"]
    c = st.columns(4)
    c[0].metric("Month", run["month"])
    c[1].metric("Status", run["status"])
    c[2].metric("Units", f"{k['units']:,.0f}")
    c[3].metric("Shipping cost", f"${k['total_cost']:,.0f}")
    for p in run.get("publications", []):
        st.caption(f"Published before: {p['at']} by {p['by']} → `{p['target']}` ({p['mode']})")

    # ---- 2. destination --------------------------------------------------------------------
    st.markdown("#### 2 · Destination")
    d1, d2, d3, d4 = st.columns([1.3, 1.1, 0.8, 1.4])
    project = d1.text_input("GCP project", value=cfg.get("project", "demo-allocation-project"), key="pub_project")
    dataset = d2.text_input("Dataset", value=cfg.get("dataset", "com_allocation"), key="pub_dataset")
    location = d3.text_input("Location", value=cfg.get("location", "US"), key="pub_location")
    mode_label = d4.radio("Write mode", ["Replace this month's rows", "Append"], key="pub_mode",
                          help="Replace deletes rows for the same target month first, so re-publishing never "
                               "creates duplicates.")
    mode = "replace" if mode_label.startswith("Replace") else "append"
    tbl_cfg = pd.DataFrame([{"publish": True, "output": key, "table": key, "meaning": v["desc"]}
                            for key, v in BQ_TABLES.items()])
    tbl_cfg = st.data_editor(tbl_cfg, hide_index=True, width="stretch", key="pub_tables",
                             disabled=["output", "meaning"],
                             column_config={"publish": st.column_config.CheckboxColumn("Publish", width="small"),
                                            "output": "Optimizer output", "table": "BigQuery table name",
                                            "meaning": st.column_config.TextColumn("What it holds", width="large")})
    chosen = {r["output"]: r["table"] for _, r in tbl_cfg.iterrows() if r["publish"]}

    # ---- 3. preview -----------------------------------------------------------------------
    published_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tables = build_output_tables(run, ss["user"]["name"], published_at)
    st.markdown("#### 3 · Preview")
    if chosen:
        tabs = st.tabs([f"{chosen[k_]} · {len(tables[k_])} rows" for k_ in chosen])
        for tab, key in zip(tabs, chosen):
            with tab:
                st.caption(f"`{project}.{dataset}.{chosen[key]}` — partitioned by month on `target_month`")
                st.dataframe(tables[key].head(200), hide_index=True, width="stretch")
                with st.expander("Schema"):
                    st.dataframe(pd.DataFrame(BQ_TABLES[key]["schema"], columns=["column", "type"]),
                                 hide_index=True, width="stretch")
    else:
        st.caption("Select at least one table.")

    # ---- 4. checks + publish --------------------------------------------------------------
    st.markdown("#### 4 · Checks")
    checks = publish_checks(run, tables, project, dataset, chosen or {"x": "x"}, allow and not live)
    icon = {"ok": ":green[✓]", "warn": ":orange[!]", "error": ":red[✕]"}
    for ch in checks:
        st.markdown(f"{icon[ch['level']]} &nbsp; {ch['text']}")
    blocked = any(ch["level"] == "error" for ch in checks) or not chosen
    month = target_month_date(run["month"])
    if st.button(f"Publish {len(chosen)} table(s) to BigQuery", type="primary", disabled=blocked):
        sub = {k_: tables[k_] for k_ in chosen}
        try:
            with st.spinner("Writing to BigQuery…"):
                if live:
                    rows = bq_publish(bq_client(project, location), sub, project, dataset, chosen, month, mode,
                                      location)
                else:
                    rows = demo_publish(sub, dataset, chosen, month, mode)
            add_publication(run["id"], {"at": published_at, "by": ss["user"]["name"], "mode": mode,
                                        "target": f"{project}.{dataset}", "rows": rows,
                                        "backend": "bigquery" if live else "demo"})
            ss["pub_last"] = {"rows": rows, "target": f"{project}.{dataset}", "names": chosen, "run_id": run["id"]}
            st.rerun()
        except ModuleNotFoundError:
            st.error("google-cloud-bigquery is not installed. Add `google-cloud-bigquery` and `pyarrow` "
                     "to requirements.txt.")
        except Exception as e:
            st.error(f"Publish failed: {type(e).__name__}: {e}")

    last = ss.get("pub_last")
    if last and last["run_id"] == run["id"]:
        st.success("Published to `{}`: ".format(last["target"]) +
                   ", ".join(f"**{last['names'][k_]}** {n:,} rows" for k_, n in last["rows"].items()))

    # ---- 5. what's there now ----------------------------------------------------------------
    st.markdown("#### 5 · Published data")
    if live:
        st.caption(f"Open BigQuery console → `{project}.{dataset}` to see the tables.")
    else:
        inv = demo_inventory(dataset)
        if inv.empty:
            st.caption("Nothing published to this dataset yet.")
        else:
            st.dataframe(inv, hide_index=True, width="stretch")
            plan_tbl = chosen.get("allocation_plan", "allocation_plan")
            default_sql = (f"SELECT warehouse, storage, ROUND(SUM(units)) AS units, ROUND(SUM(cost)) AS cost\n"
                           f"FROM `{project}.{dataset}.{plan_tbl}`\n"
                           f"WHERE target_month = '{month}'\nGROUP BY warehouse, storage\nORDER BY warehouse, storage")
            sql = st.text_area("Query the published tables (BigQuery-style SQL, SELECT only)", value=default_sql,
                               height=150, key=f"pub_sql_{dataset}")
            if st.button("Run query"):
                try:
                    st.dataframe(demo_query(sql, project, dataset), hide_index=True, width="stretch")
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")

    with st.expander("Code used in connected mode"):
        st.code('''from google.cloud import bigquery
client = bigquery.Client(project=PROJECT, credentials=creds, location=LOCATION)
client.create_dataset(f"{PROJECT}.{DATASET}", exists_ok=True)
table = bigquery.Table(table_id, schema=schema)
table.time_partitioning = bigquery.TimePartitioning(type_="MONTH", field="target_month")
client.create_table(table, exists_ok=True)
# replace mode: remove this month's rows first, so re-publishing never duplicates
client.query(f"DELETE FROM `{table_id}` WHERE target_month = @m", job_config=...).result()
client.load_table_from_dataframe(df, table_id,
    job_config=bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND")).result()''',
                language="python")
        st.markdown("Secrets for connected mode:")
        st.code('''[gcp_service_account]   # the service-account JSON key, field by field
type = "service_account"
project_id = "your-project"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
client_email = "allocation-app@your-project.iam.gserviceaccount.com"
token_uri = "https://oauth2.googleapis.com/token"

[bigquery]
project = "your-project"
dataset = "com_allocation"
location = "US"''', language="toml")



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

    {"Set up & run": page_setup, "Pipeline run": page_pipeline, "Plan": page_plan, "Warnings": page_warnings,
     "Run history": page_history, "Approvals": page_approvals,
     "Publish to BigQuery": page_publish}[ss["page"]]()


main()
