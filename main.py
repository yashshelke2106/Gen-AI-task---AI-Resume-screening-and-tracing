"""
main.py
=======
Entry point for the AI Resume Screening System.

Runs the LangChain pipeline (Extract -> Match -> Score -> Explain) for every
resume found in the data/ folder, against one job description.

INPUT: drop your resumes into data/ as .txt, .pdf, or .docx files named like
`resume_strong.pdf`, `resume_average.docx`, `resume_weak.txt`, etc. (any name
starting with "resume"). The job description is `job_description.(txt|pdf|docx)`.
Files are auto-discovered and read by loaders.py.

When LANGCHAIN_TRACING_V2=true (see .env), every .invoke() is traced to
LangSmith, with each run tagged by candidate band.

Usage:
    python main.py
"""

import os
import json

import config
from chains import build_pipeline
from loaders import read_document, discover_resumes, find_job_description

# Folder holding the job description + resume files.
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def screen_candidate(pipeline, resume_text: str, job_description: str, candidate: dict) -> dict:
    """Run one candidate through the pipeline with LangSmith tags + metadata."""
    run_config = {
        "run_name": "screen_%s" % candidate["band"],
        "tags": ["resume-screening", candidate["band"]],   # LangSmith tags (bonus)
        "metadata": {"candidate_name": candidate["name"],
                     "candidate_band": candidate["band"],
                     "source_file": os.path.basename(candidate["path"])},
    }
    # Single .invoke() drives the entire Extract->Match->Score->Explain chain.
    return pipeline.invoke(
        {"resume": resume_text, "job_description": job_description},
        config=run_config,
    )


def print_result(candidate: dict, result: dict) -> None:
    """Pretty-print the screening result for one candidate."""
    expl = result["explanation"]
    print("\n" + "=" * 70)
    print("CANDIDATE: %s  (band: %s)" % (candidate["name"], candidate["band"]))
    print("=" * 70)
    print("Fit Score : %s/100  [%s]" % (result["scoring"].get("score"),
                                        result["scoring"].get("band")))
    print("Verdict   : %s" % expl.get("summary"))
    print("Reasoning : %s" % expl.get("reasoning"))
    print("Strengths : %s" % ", ".join(expl.get("key_strengths", []) or ["-"]))
    print("Gaps      : %s" % ", ".join(expl.get("key_gaps", []) or ["-"]))


def main() -> None:
    # Tracing is mandatory for this assignment — warn early if it is off.
    if config.tracing_enabled():
        print("LangSmith tracing ENABLED -> project '%s'\n" % config.LANGSMITH_PROJECT)
    else:
        print("WARNING: LANGCHAIN_TRACING_V2 is not 'true'. LangSmith tracing is OFF.\n"
              "         Set it in your .env to capture the required traces.\n")

    # Read the job description (txt/pdf/docx, auto-detected).
    jd_path = find_job_description(DATA_DIR)
    job_description = read_document(jd_path)
    print("Job description: %s" % os.path.basename(jd_path))

    # Auto-discover every resume file in data/.
    candidates = discover_resumes(DATA_DIR)
    if not candidates:
        raise SystemExit("No resume files found in %s (expected resume*.txt/.pdf/.docx)" % DATA_DIR)
    print("Found %d resume(s): %s\n" % (
        len(candidates), ", ".join(os.path.basename(c["path"]) for c in candidates)))

    # Build the model once and share it across all chains and candidates.
    model = config.get_model(temperature=0.0)
    pipeline = build_pipeline(model)

    all_results = []
    for candidate in candidates:
        resume_text = read_document(candidate["path"])     # txt/pdf/docx -> text
        result = screen_candidate(pipeline, resume_text, job_description, candidate)
        print_result(candidate, result)
        all_results.append({"candidate": candidate["name"],
                            "band": candidate["band"],
                            "source_file": os.path.basename(candidate["path"]),
                            "score": result["scoring"].get("score"),
                            "explanation": result["explanation"]})

    # Save a JSON summary of all runs for the report / submission.
    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    print("\nSaved summary -> results.json")


if __name__ == "__main__":
    main()
