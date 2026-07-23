"""
Step 1 - Skill Extraction prompt.

Extracts skills, experience, and tools FROM THE RESUME ONLY.
The anti-hallucination rule is explicit: the model must not invent skills
that are not literally supported by the resume text.
"""

from langchain_core.prompts import PromptTemplate

_EXTRACTION_TEMPLATE = """You are an expert technical recruiter and information extractor.

Extract structured information from the candidate's resume below.

STRICT RULES:
- Use ONLY information explicitly present in the resume.
- Do NOT assume, infer, or invent any skill, tool, or experience not stated.
- If a field cannot be found, return an empty list or "not specified".
- Return ONLY valid JSON. No markdown, no commentary.

Return JSON with exactly these keys:
{{
  "skills": [list of technical skills found],
  "tools": [list of tools/frameworks/libraries found],
  "years_of_experience": <number or "not specified">,
  "domains": [list of domains/industries the candidate has worked in]
}}

RESUME:
{resume}
"""

# input_variables is declared explicitly (assignment requirement).
EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["resume"],
    template=_EXTRACTION_TEMPLATE,
)
