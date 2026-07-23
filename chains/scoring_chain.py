"""
Scoring chain (Step 3).

Turns the matching analysis into a 0-100 fit score with a band label.
"""

import json
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from prompts import SCORING_PROMPT


def build_scoring_chain(model):
    """Build the scoring chain for a given chat model."""
    shape_input = RunnableLambda(lambda state: {
        "matching": json.dumps(state["matching"], indent=2),
    })
    return shape_input | SCORING_PROMPT | model | JsonOutputParser()
