"""
Step 2 - Matching prompt.

Compares the extracted candidate profile against the job description and
identifies which required skills are matched vs missing. Again constrained
to avoid crediting the candidate with skills they do not have.
"""

from langchain_core.prompts import PromptTemplate

_MATCHING_TEMPLATE = """You are a recruiter comparing a candidate to a job description.

Compare the EXTRACTED CANDIDATE PROFILE against the JOB DESCRIPTION.

STRICT RULES:
- Only count a requirement as "matched" if the candidate profile clearly supports it.
- Do NOT give credit for skills the candidate does not have.
- Separate must-have (required) gaps from nice-to-have (preferred) gaps.
- Return ONLY valid JSON. No markdown, no commentary.

Return JSON with exactly these keys:
{{
  "matched_required_skills": [required skills the candidate clearly has],
  "missing_required_skills": [required skills the candidate is missing],
  "matched_preferred_skills": [preferred/nice-to-have skills the candidate has],
  "experience_match": "<short note: does experience meet the requirement?>"
}}

JOB DESCRIPTION:
{job_description}

EXTRACTED CANDIDATE PROFILE (JSON):
{extraction}
"""

MATCHING_PROMPT = PromptTemplate(
    input_variables=["job_description", "extraction"],
    template=_MATCHING_TEMPLATE,
)
