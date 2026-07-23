"""
config.py
=========
Central configuration. Loads environment variables (including LangSmith
tracing settings) from a .env file and builds the shared chat model.

Keeping model creation in one place means every chain uses the same,
swappable LLM — change the provider here and the whole pipeline follows.
"""

import os
from dotenv import load_dotenv

# Load variables from .env into the environment.
# This is what activates LangSmith tracing: LANGCHAIN_TRACING_V2=true must be
# set BEFORE the LangChain components are imported/invoked.
load_dotenv()

# Provider-agnostic model identifier. FREE Google Gemini by default;
# override via LC_MODEL in .env (e.g. "openai:gpt-4o-mini") without touching chains.
MODEL_ID = os.environ.get("LC_MODEL", "google_genai:gemini-2.5-flash")

# LangSmith project name (used to group traces in the LangSmith UI).
LANGSMITH_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "resume-screening-system")


def get_model(temperature: float = 0.0):
    """Build and return the shared chat model.

    Temperature defaults to 0 for deterministic, reproducible scoring —
    important so the same resume does not get wildly different scores.
    """
    # Imported lazily so that `import config` does not fail in offline tests
    # that never call a real model.
    from langchain.chat_models import init_chat_model
    return init_chat_model(MODEL_ID, temperature=temperature)


def tracing_enabled() -> bool:
    """True when LangSmith tracing is switched on via the environment."""
    return os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
