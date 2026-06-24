"""
pipeline.py
===========
Composes the four stage-chains into ONE end-to-end LCEL pipeline:

    Resume -> Extract -> Match -> Score -> Explain

`RunnablePassthrough.assign` runs each sub-chain and merges its output back
into the shared state under a named key, so later steps can read earlier
results. The whole thing is a single Runnable you call with .invoke().
"""

from langchain_core.runnables import RunnablePassthrough
from chains.extraction_chain import build_extraction_chain
from chains.matching_chain import build_matching_chain
from chains.scoring_chain import build_scoring_chain
from chains.explanation_chain import build_explanation_chain


def build_pipeline(model):
    """Build the full resume-screening pipeline for a given chat model.

    Input  (to .invoke):  {"resume": <str>, "job_description": <str>}
    Output (from .invoke): the input plus keys:
        "extraction", "matching", "scoring", "explanation"
    """
    extraction = build_extraction_chain(model)
    matching = build_matching_chain(model)
    scoring = build_scoring_chain(model)
    explanation = build_explanation_chain(model)

    # Each .assign adds one key to the state and passes everything forward.
    pipeline = (
        RunnablePassthrough.assign(extraction=extraction)
        | RunnablePassthrough.assign(matching=matching)
        | RunnablePassthrough.assign(scoring=scoring)
        | RunnablePassthrough.assign(explanation=explanation)
    )

    # A friendly run name makes the trace easy to find in the LangSmith UI.
    return pipeline.with_config(run_name="resume_screening_pipeline")
