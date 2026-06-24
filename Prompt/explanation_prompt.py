"""
Step 4 - Explanation prompt.

Produces a human-readable, explainable justification for the assigned score,
grounded in the matched/missing skills. This is the "explainability" deliverable.
"""

from langchain_core.prompts import PromptTemplate

_EXPLANATION_TEMPLATE = """You are a recruiter writing a short, transparent justification for a
candidate's fit score, to be read by a hiring manager.

STRICT RULES:
- Explain the score using ONLY the matched and missing skills provided.
- Be specific: name the skills the candidate has and the ones they lack.
- Do NOT introduce any new skills or assumptions.
- 3 to 5 sentences. Clear and professional.

Return JSON with exactly these keys:
{{
  "summary": "<one-sentence verdict>",
  "reasoning": "<3-5 sentences explaining WHY this score was assigned>",
  "key_strengths": [short list],
  "key_gaps": [short list]
}}

FIT SCORE (JSON): {scoring}
MATCHING ANALYSIS (JSON): {matching}
"""

EXPLANATION_PROMPT = PromptTemplate(
    input_variables=["scoring", "matching"],
    template=_EXPLANATION_TEMPLATE,
)
