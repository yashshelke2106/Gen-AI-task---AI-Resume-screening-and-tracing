# AI Resume Screening System with Tracing

An AI-powered resume screening tool built with **LangChain** and traced with
**LangSmith**. It evaluates candidates against a job description and returns a
**0–100 fit score plus a transparent explanation**.

```
Resume + Job Description  ->  Extract  ->  Match  ->  Score  ->  Explain  ->  Fit Score + Reasoning
```

![Pipeline Architecture](assets/pipeline_architecture.png)

## Features

- **Modular LangChain pipeline** built with **LCEL** (`prompt | model | parser`) and `.invoke()`.
- **4-stage flow:** skill extraction → matching → scoring → explanation.
- **Explainable output:** every score comes with reasoning, strengths, and gaps.
- **LangSmith tracing** (mandatory): every step is logged and visible per run.
- **Anti-hallucination prompts:** the model is forbidden from inventing skills
  not present in the resume.
- **Bonus:** structured JSON output, few-shot calibration in the scoring prompt,
  and LangSmith tags per run.

## Project Structure

```
resume_screening_system/
├── main.py                 # Entry point: screens 3 candidates, 3 LangSmith runs
├── config.py               # Loads .env, builds the shared model, tracing helpers
├── requirements.txt
├── .env.example            # Copy to .env and add your keys
├── prompts/                # One PromptTemplate per stage (input_variables declared)
│   ├── extraction_prompt.py
│   ├── matching_prompt.py
│   ├── scoring_prompt.py
│   └── explanation_prompt.py
├── chains/                 # One LCEL chain per stage + the composed pipeline
│   ├── extraction_chain.py
│   ├── matching_chain.py
│   ├── scoring_chain.py
│   ├── explanation_chain.py
│   └── pipeline.py
├── data/                   # 1 job description + 3 resumes (strong/average/weak)
├── tests/
│   └── test_pipeline_offline.py   # Wiring test with a fake model (no API key)
├── assets/                 # Architecture diagram
└── report/                 # REPORT.md + LangSmith screenshots go here
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure keys + tracing
cp .env.example .env
#    then edit .env and add:
#      OPENAI_API_KEY=...
#      LANGCHAIN_API_KEY=...      (from https://smith.langchain.com)
#      LANGCHAIN_TRACING_V2=true
```

## Providing resumes (input)

Put your files in the `data/` folder. The project auto-discovers them — no code
changes needed:

- **Resumes:** any file whose name starts with `resume` — e.g. `resume_strong.pdf`,
  `resume_average.docx`, `resume_weak.txt`. Supported formats: **.txt, .pdf, .docx**.
- **Job description:** `job_description.txt` (or `.pdf` / `.docx`).

The band (strong/average/weak) is inferred from the filename for labelling only;
it does not influence the score. Drop in as many resumes as you like — each one
becomes its own LangSmith run.

> Scanned/image-only PDFs are not supported (they need OCR). Use text-based PDFs.

## Dashboard (web UI)

A Streamlit dashboard lets you screen resumes interactively — paste your Gemini
API key, upload a resume, and see the score + explanation. No `.env` needed;
the key is entered in the app.

```bash
pip install -r requirements.txt
streamlit run app_dashboard.py
```

This opens a browser tab. In the sidebar, paste your Gemini API key (and,
optionally, a LangSmith key to enable tracing). Then upload a `.pdf`/`.docx`/`.txt`
resume (or paste text), review the job description, and click **Screen candidate**.

## Run


```bash
# Full pipeline over all 3 candidates (produces the 3 LangSmith runs)
python main.py

# Offline wiring test — no API key needed (uses a fake model)
python tests/test_pipeline_offline.py
```

Expected: the strong candidate scores highest, the weak candidate lowest, each
with a written justification. Results are also saved to `results.json`.

## Viewing traces in LangSmith

After running `main.py` with tracing enabled, open
[smith.langchain.com](https://smith.langchain.com) → project
`resume-screening-system`. You will see 3 runs (tagged `strong`, `average`,
`weak`), each expandable into its Extract → Match → Score → Explain steps.
Add screenshots to `report/` (see `report/REPORT.md`).

## Notes

- Built and tested on the **LangChain 1.x** line. The model id in `config.py`
  is provider-agnostic (`openai:gpt-4o-mini` by default) — swap providers in one place.
- `temperature=0` is used for reproducible scoring.
