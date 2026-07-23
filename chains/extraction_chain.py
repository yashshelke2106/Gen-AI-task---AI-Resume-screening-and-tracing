"""
Extraction chain (Step 1).

LCEL: input-shaper | PromptTemplate | model | JSON parser.
Takes the pipeline state and returns a structured profile dict.
"""

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from prompts import EXTRACTION_PROMPT


def build_extraction_chain(model):
    """Build the skill-extraction chain for a given chat model."""
    # Pull only the field this step needs from the shared state.
    shape_input = RunnableLambda(lambda state: {"resume": state["resume"]})
    # LCEL composition with the pipe operator.
    return shape_input | EXTRACTION_PROMPT | model | JsonOutputParser()
