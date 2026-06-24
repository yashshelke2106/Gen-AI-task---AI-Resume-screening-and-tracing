"""
Explanation chain (Step 4).

Produces the human-readable, explainable justification for the score.
"""

import json
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from prompts import EXPLANATION_PROMPT


def build_explanation_chain(model):
    """Build the explanation chain for a given chat model."""
    shape_input = RunnableLambda(lambda state: {
        "scoring": json.dumps(state["scoring"], indent=2),
        "matching": json.dumps(state["matching"], indent=2),
    })
    return shape_input | EXPLANATION_PROMPT | model | JsonOutputParser()
