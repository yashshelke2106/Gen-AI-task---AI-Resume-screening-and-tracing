"""Prompt templates for each stage of the resume-screening pipeline."""
from prompts.extraction_prompt import EXTRACTION_PROMPT
from prompts.matching_prompt import MATCHING_PROMPT
from prompts.scoring_prompt import SCORING_PROMPT
from prompts.explanation_prompt import EXPLANATION_PROMPT

__all__ = [
    "EXTRACTION_PROMPT",
    "MATCHING_PROMPT",
    "SCORING_PROMPT",
    "EXPLANATION_PROMPT",
]
