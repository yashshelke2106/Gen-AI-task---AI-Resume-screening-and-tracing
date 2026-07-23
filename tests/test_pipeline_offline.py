"""
Offline pipeline-wiring test.
=============================
Proves the LCEL pipeline is wired correctly WITHOUT needing an API key or
network. It injects a deterministic FakeListChatModel that returns canned
JSON for each of the four stages, then asserts the pipeline produces the
expected keys.

This validates structure/plumbing only. The real scoring quality is produced
by a live LLM via main.py.

Run:  python tests/test_pipeline_offline.py
"""

import os
import sys
import json

# Make the project root importable when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from chains import build_pipeline


def make_fake_model():
    """A fake chat model that replies with canned JSON, one per pipeline step.

    The pipeline calls the model 4 times (extract, match, score, explain),
    so we queue 4 responses in order.
    """
    responses = [
        # 1) extraction
        json.dumps({"skills": ["Python", "ML"], "tools": ["scikit-learn"],
                    "years_of_experience": 5, "domains": ["fintech"]}),
        # 2) matching
        json.dumps({"matched_required_skills": ["Python", "ML"],
                    "missing_required_skills": [],
                    "matched_preferred_skills": ["LangChain"],
                    "experience_match": "meets requirement"}),
        # 3) scoring
        json.dumps({"score": 88, "band": "Strong fit"}),
        # 4) explanation
        json.dumps({"summary": "Strong fit.",
                    "reasoning": "Has all required skills and enough experience.",
                    "key_strengths": ["Python", "ML"], "key_gaps": []}),
    ]
    return FakeListChatModel(responses=responses)


def test_pipeline_structure():
    """The pipeline should return all four stage outputs with the right keys."""
    pipeline = build_pipeline(make_fake_model())
    result = pipeline.invoke({
        "resume": "dummy resume text",
        "job_description": "dummy JD text",
    })

    # Structural assertions.
    assert "extraction" in result, "missing extraction"
    assert "matching" in result, "missing matching"
    assert "scoring" in result, "missing scoring"
    assert "explanation" in result, "missing explanation"
    assert result["scoring"]["score"] == 88
    assert result["explanation"]["summary"] == "Strong fit."

    print("PASS: pipeline produced all 4 stages.")
    print("  extraction :", result["extraction"])
    print("  matching   :", result["matching"])
    print("  scoring    :", result["scoring"])
    print("  explanation:", result["explanation"])


if __name__ == "__main__":
    test_pipeline_structure()
    print("\nAll offline tests passed.")
