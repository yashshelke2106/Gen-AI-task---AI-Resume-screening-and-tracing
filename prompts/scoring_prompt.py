"""
Step 3 - Scoring prompt.

Assigns a 0-100 fit score based on the matching analysis. Includes a
FEW-SHOT example (bonus requirement) to anchor the scoring scale and keep
scores consistent and well-calibrated across candidates.
"""

from langchain_core.prompts import PromptTemplate

_SCORING_TEMPLATE = """You are an objective hiring evaluator. Assign a FIT SCORE from 0 to 100
indicating how well the candidate matches the job description, based ONLY on the
matching analysis provided.

SCORING GUIDE:
- 80-100: Strong fit. Has most required skills + relevant experience.
- 50-79 : Partial fit. Some required skills, notable gaps.
- 0-49  : Weak fit. Missing most required skills / wrong domain.

STRICT RULES:
- Base the score strictly on the matching analysis. Do NOT invent strengths.
- Return ONLY valid JSON. No markdown, no commentary.

--- FEW-SHOT EXAMPLES (for calibration) ---
Example A:
Matching: most required skills matched, 5 yrs experience, strong domain fit.
Output: {{"score": 88, "band": "Strong fit"}}

Example B:
Matching: a few required skills matched, missing deep learning and cloud, 2 yrs.
Output: {{"score": 61, "band": "Partial fit"}}

Example C:
Matching: almost no required skills matched, unrelated domain.
Output: {{"score": 18, "band": "Weak fit"}}
--- END EXAMPLES ---

Now score this candidate.

Return JSON with exactly these keys:
{{
  "score": <integer 0-100>,
  "band": "<Strong fit | Partial fit | Weak fit>"
}}

MATCHING ANALYSIS (JSON):
{matching}
"""

SCORING_PROMPT = PromptTemplate(
    input_variables=["matching"],
    template=_SCORING_TEMPLATE,
)
