"""
Matching chain (Step 2).

Compares the extracted profile to the job description. The extracted profile
(a dict from the previous step) is serialized to a JSON string before being
injected into the prompt, so the LLM sees clean, unambiguous structure.
"""

import json
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from prompts import MATCHING_PROMPT


def build_matching_chain(model):
    """Build the matching chain for a given chat model."""
    shape_input = RunnableLambda(lambda state: {
        "job_description": state["job_description"],
        "extraction": json.dumps(state["extraction"], indent=2),
    })
    return shape_input | MATCHING_PROMPT | model | JsonOutputParser()
