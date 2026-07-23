"""
check_setup.py
==============
Diagnoses your API-key / model setup before running the full pipeline.

Run:  python check_setup.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("1. Environment check")
print("=" * 60)
gkey = os.environ.get("GOOGLE_API_KEY", "")
lkey = os.environ.get("LANGCHAIN_API_KEY", "")
model = os.environ.get("LC_MODEL", "(not set)")
print("GOOGLE_API_KEY     :", "set (%d chars)" % len(gkey) if gkey else "MISSING")
print("LANGCHAIN_API_KEY  :", "set (%d chars)" % len(lkey) if lkey else "MISSING")
print("LANGCHAIN_TRACING_V2:", os.environ.get("LANGCHAIN_TRACING_V2", "(not set)"))
print("LC_MODEL           :", model)

if not gkey:
    raise SystemExit("\n>> GOOGLE_API_KEY is missing. Add it to your .env and retry.")

print("\n" + "=" * 60)
print("2. Models your key can actually use (generateContent)")
print("=" * 60)
try:
    from google import genai
    client = genai.Client(api_key=gkey)
    usable = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", [])
        if "generateContent" in (actions or []):
            usable.append(m.name)
    for name in usable:
        print("  ", name)
    if not usable:
        print("  (none returned — key may be invalid or restricted)")
except Exception as e:
    print("  Could not list models:", type(e).__name__, "-", str(e)[:300])

print("\n" + "=" * 60)
print("3. Minimal test call with LC_MODEL")
print("=" * 60)
try:
    from langchain.chat_models import init_chat_model
    llm = init_chat_model(model, temperature=0)
    resp = llm.invoke("Say OK.")
    print("  SUCCESS ->", resp.content)
except Exception as e:
    print("  FAILED ->", type(e).__name__)
    print("  Message:", str(e)[:500])
    print("\n  Fix: pick a model from the section-2 list above and set it in .env, e.g.")
    print("       LC_MODEL=google_genai:gemini-2.0-flash")
