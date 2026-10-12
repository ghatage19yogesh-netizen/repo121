import io
import os
import time
import streamlit as st

# pip install google-generativeai
import google.generativeai as genai


st.set_page_config(page_title="Gemini Transcriber", page_icon="🎙️", layout="centered")
st.title("🎙️ Audio → Text with Google Gemini 1.5")

# -----------------------------
# Sidebar: API & settings
# -----------------------------
st.sidebar.header("Gemini Settings")
api_key = st.sidebar.text_input(
    "GOOGLE_API_KEY",
    value=os.getenv("GOOGLE_API_KEY", ""),
    type="password",
    help="You can also set this as an environment variable.",
)

model_name = st.sidebar.selectbox(
    "Model",
    ["gemini-1.5-flash", "gemini-1.5-pro"],
    index=0,
    help="Flash is cheaper/faster; Pro can be more accurate on tricky audio.",
)

use_file_api = st.sidebar.selectbox(
    "Upload method",
    ["Auto (recommended)", "Inline (small files)", "File API (large files)"],
    index=0,
    help=(
        "Inline attaches audio bytes directly in the request. "
        "File API uploads the file to Gemini first and then references it in the prompt. "
        "Auto picks File API if the file is large."
    ),
)

language_hint = st.sidebar.text_input(
    "Language hint (optional, e.g., 'en', 'hi', 'mr')", value=""
)

output_format = st.sidebar.selectbox(
    "Output format",
    ["Plain text (verbatim)", "SRT (timestamps)"],
    index=0,
    help="SRT relies on the model approximating timestamps; treat as best-effort."
)

# -----------------------------
# File uploader
# -----------------------------
st.subheader("1) Upload your audio")
file = st.file_uploader(
    "Choose an audio file (mp3, wav, m4a, mp4, webm, ogg, flac, amr)",
    type=["mp3", "wav", "m4a", "mp4", "webm", "ogg", "flac", "amr"],
)

if file is not None:
    st.audio(file, format=file.type)
else:
    st.info("Upload an audio file to begin.")

# -----------------------------
# Helpers
# -----------------------------
MIME_FALLBACKS = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "m4a": "audio/mp4",
    "mp4": "audio/mp4",
    "webm": "audio/webm",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "amr": "audio/amr",
}

def guess_mime(uploaded):
    if uploaded.type:
        return uploaded.type
    ext = (uploaded.name.rsplit(".", 1)[-1] or "").lower()
    return MIME_FALLBACKS.get(ext, "application/octet-stream")

def make_prompt(output_format: str, language_hint: str) -> str:
    base = [
        "Transcribe the audio as accurately as possible.",
        "Preserve punctuation, numbers, and capitalization.",
        "If the language is clear, use that language.",
    ]
    if language_hint.strip():
        base.append(f"Language hint: {language_hint.strip()}")
    # diarization guidance (best-effort)
    base.append("If speaker changes are obvious, prefix lines like 'Speaker 1:' and 'Speaker 2:' (best-effort).")

    if output_format.startswith("SRT"):
        # Ask for SRT. Gemini will approximate timestamps; it's not guaranteed to be perfect.
        base.append(
            "Return ONLY a valid SRT file with numbered cues, "
            "with reasonable timestamps and short line lengths. "
            "Do not include any extra commentary."
        )
    else:
        base.append("Return ONLY the plain transcript text. Do not add extra commentary.")

    return " ".join(base)

def bytesio_from_uploaded(uploaded):
    # Return a BytesIO positioned at start
    b = io.BytesIO(uploaded.getvalue())
    b.seek(0)
    return b

def configure_genai():
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY. Enter it in the sidebar.")
    genai.configure(api_key=api_key)

def get_model(name: str):
    return genai.GenerativeModel(name)

def use_inline_request(model, uploaded, prompt, mime):
    audio_part = {"mime_type": mime, "data": uploaded.getvalue()}
    resp = model.generate_content([prompt, audio_part], request_options={"timeout": 600})
    return getattr(resp, "text", "") or ""

def upload_via_file_api(uploaded, mime):
    f = bytesio_from_uploaded(uploaded)
    # display_name helps you identify the file in your console
    up = genai.upload_file(f, mime_type=mime, display_name=uploaded.name)
    # Wait until it's processed (ACTIVE)
    while True:
        up = genai.get_file(up.name)
        if up.state.name == "ACTIVE":
            return up
        if up.state.name == "FAILED":
            raise RuntimeError(f"File upload failed: {up.state.name}")
        time.sleep(1)

def use_file_api_request(model, file_ref, prompt):
    resp = model.generate_content([prompt, file_ref], request_options={"timeout": 1200})
    return getattr(resp, "text", "") or ""

def choose_mode(uploaded) -> str:
    if use_file_api == "Inline (small files)":
        return "inline"
    if use_file_api == "File API (large files)":
        return "fileapi"
    # Auto
    size_mb = len(uploaded.getvalue()) / (1024 * 1024)
    return "fileapi" if size_mb > 15 else "inline"  # heuristic threshold

# -----------------------------
# Action
# -----------------------------
st.subheader("2) Transcribe")
go = st.button("Transcribe with Gemini", type="primary", disabled=(file is None))

if go and file is not None:
    try:
        configure_genai()
        model = get_model(model_name)
        prompt = make_prompt(output_format, language_hint)
        mime = guess_mime(file)

        mode = choose_mode(file)
        st.write(f"Using **{model_name}** with **{'File API' if mode=='fileapi' else 'Inline'}** upload.")
        with st.spinner("Transcribing…"):
            if mode == "inline":
                text = use_inline_request(model, file, prompt, mime)
            else:
                ref = upload_via_file_api(file, mime)
                text = use_file_api_request(model, ref, prompt)

        if not text.strip():
            st.warning("No text returned. Try the other upload method or a different model.")
        else:
            if output_format.startswith("SRT"):
                st.success("SRT generated (best-effort).")
                st.code(text, language="srt")
                st.download_button(
                    "Download SRT",
                    data=text.encode("utf-8"),
                    file_name=(file.name.rsplit(".", 1)[0] + ".srt"),
                    mime="application/x-subrip",
                )
            else:
                st.success("Transcript ready!")
                st.text_area("Transcript", text, height=350)
                st.download_button(
                    "Download TXT",
                    data=text.encode("utf-8"),
                    file_name=(file.name.rsplit(".", 1)[0] + ".txt"),
                    mime="text/plain",
                )

    except Exception as e:
        st.error(f"Transcription failed: {e}")
