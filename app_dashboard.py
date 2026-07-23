"""
app_dashboard.py
================
Streamlit dashboard for the AI Resume Screening System.

The user:
  1. pastes their Gemini API key (and optionally a LangSmith key for tracing),
  2. uploads a resume (.pdf / .docx / .txt),
  3. reviews/edits the job description,
  4. clicks "Screen candidate" and sees the fit score + explanation.

It reuses the exact same pipeline as main.py (prompts + chains), so the
dashboard and the CLI produce identical logic.

Run:
    streamlit run app_dashboard.py
"""

import os
import json
import tempfile

import streamlit as st

# Reuse the project's own modules — single source of truth for the logic.
from loaders import read_document
from chains import build_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_model(api_key: str, model_id: str, temperature: float = 0.0):
    """Build the chat model using the key the user typed into the dashboard.

    We set the env var (rather than passing api_key kwargs) because that is the
    most reliable way for langchain-google-genai / OpenAI to pick it up.
    """
    os.environ["GOOGLE_API_KEY"] = api_key      # for google_genai models
    os.environ["OPENAI_API_KEY"] = api_key      # harmless if using OpenAI id
    from langchain.chat_models import init_chat_model
    return init_chat_model(model_id, temperature=temperature)


def uploaded_file_to_text(uploaded_file) -> str:
    """Save the uploaded file to a temp path and extract its text via loaders."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        return read_document(tmp_path)
    finally:
        os.unlink(tmp_path)


def score_color(score: int) -> str:
    """Green / amber / red band for the score display."""
    if score is None:
        return "gray"
    if score >= 80:
        return "green"
    if score >= 50:
        return "orange"
    return "red"


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")
st.title("📄 AI Resume Screening System")
st.caption("Extract → Match → Score → Explain · powered by LangChain + Gemini")

# ---- Sidebar: credentials & settings ----
with st.sidebar:
    st.header("🔑 Settings")
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        help="Get a free key at https://aistudio.google.com/app/apikey (starts with 'AIza').",
    )
    model_id = st.selectbox(
        "Model",
        ["google_genai:gemini-2.5-flash",
         "google_genai:gemini-2.0-flash",
         "google_genai:gemini-1.5-flash"],
        index=0,
        help="If one model 404s, try another from this list.",
    )

    st.divider()
    st.subheader("LangSmith tracing (optional)")
    enable_tracing = st.checkbox("Enable tracing", value=False)
    langsmith_key = st.text_input("LangSmith API key", type="password",
                                  disabled=not enable_tracing)
    if enable_tracing and langsmith_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key
        os.environ["LANGCHAIN_PROJECT"] = "resume-screening-dashboard"
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ---- Main: inputs ----
col_left, col_right = st.columns(2)

# Prefill the job description from data/ if it exists.
default_jd = ""
_jd_path = os.path.join(os.path.dirname(__file__), "data", "job_description.txt")
if os.path.exists(_jd_path):
    default_jd = open(_jd_path, encoding="utf-8").read()

with col_left:
    st.subheader("1. Job description")
    job_description = st.text_area("Paste or edit the job description",
                                   value=default_jd, height=280)

with col_right:
    st.subheader("2. Candidate resume")
    uploaded = st.file_uploader("Upload a resume (.pdf, .docx, or .txt)",
                                type=["pdf", "docx", "txt"])
    pasted = st.text_area("…or paste resume text here", height=150,
                          placeholder="Optional: paste plain resume text instead of uploading.")

run = st.button("🚀 Screen candidate", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
if run:
    # Validate inputs before doing anything.
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
        st.stop()
    if not job_description.strip():
        st.error("Please provide a job description.")
        st.stop()

    # Resolve resume text (uploaded file takes priority, else pasted text).
    try:
        if uploaded is not None:
            resume_text = uploaded_file_to_text(uploaded)
        elif pasted.strip():
            resume_text = pasted
        else:
            st.error("Please upload a resume file or paste resume text.")
            st.stop()
    except Exception as e:
        st.error("Could not read the resume: %s" % e)
        st.stop()

    # Build model + pipeline and run.
    with st.spinner("Screening… (extract → match → score → explain)"):
        try:
            model = build_model(api_key, model_id)
            pipeline = build_pipeline(model)
            result = pipeline.invoke(
                {"resume": resume_text, "job_description": job_description},
                config={"tags": ["dashboard"], "run_name": "dashboard_screen"},
            )
        except Exception as e:
            st.error("Screening failed: %s" % e)
            st.info("If this is an auth error, check the API key. If it's a "
                    "'model not found' error, pick a different model in the sidebar.")
            st.stop()

    # ---- Display results ----
    scoring = result.get("scoring", {})
    matching = result.get("matching", {})
    extraction = result.get("extraction", {})
    explanation = result.get("explanation", {})

    score = scoring.get("score")
    band = scoring.get("band", "")

    st.divider()
    st.subheader("Result")

    m1, m2 = st.columns([1, 3])
    with m1:
        st.metric("Fit Score", "%s / 100" % score, band)
        if isinstance(score, (int, float)):
            st.progress(min(int(score), 100) / 100.0)
    with m2:
        st.markdown("**Verdict:** %s" % explanation.get("summary", "—"))
        st.markdown("**Reasoning:** %s" % explanation.get("reasoning", "—"))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ✅ Key strengths")
        for s in explanation.get("key_strengths", []) or ["—"]:
            st.markdown("- %s" % s)
        st.markdown("#### ✅ Matched required skills")
        for s in matching.get("matched_required_skills", []) or ["—"]:
            st.markdown("- %s" % s)
    with c2:
        st.markdown("#### ⚠️ Key gaps")
        for s in explanation.get("key_gaps", []) or ["—"]:
            st.markdown("- %s" % s)
        st.markdown("#### ❌ Missing required skills")
        for s in matching.get("missing_required_skills", []) or ["—"]:
            st.markdown("- %s" % s)

    # Raw JSON for transparency / debugging.
    with st.expander("🔍 Full pipeline output (raw JSON)"):
        st.markdown("**Extraction**"); st.json(extraction)
        st.markdown("**Matching**");   st.json(matching)
        st.markdown("**Scoring**");    st.json(scoring)
        st.markdown("**Explanation**"); st.json(explanation)

    # Download the result.
    st.download_button(
        "⬇️ Download result as JSON",
        data=json.dumps(result, indent=2),
        file_name="screening_result.json",
        mime="application/json",
    )

    if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
        st.success("Traced to LangSmith project 'resume-screening-dashboard'.")
