import json
import os
from typing import Dict, Any
from openai import OpenAI

# ======================
# Config (Deployment Friendly)
# ======================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY (set in environment variables)")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# ======================
# JSON Extract
# ======================
def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    import re
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {"raw_output": text}

# ======================
# Prompt Builder
# ======================
def build_prompt(intake: Dict[str, Any]) -> str:
    return f"""
You are a travel research agent in a multi-agent travel planning system.

Trip:
Destination: {intake["destination"]}
Days: {intake["trip_duration_days"]}
Budget: {intake["budget_level"]}
Companion: {intake["travel_companion"]}
Activities: {", ".join(intake["activity_preferences"])}

Output STRICT JSON:
{{
  "destination": "...",
  "recommended_areas": [],
  "attractions": [],
  "planning_hints": [],
  "constraints": []
}}
"""

# ======================
# Core Agent
# ======================
def research_agent(intake: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_prompt(intake)

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {"role": "system", "content": "You are a careful travel research assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=1200,
    )

    text = response.choices[0].message.content or ""
    parsed = extract_json(text)

    parsed.setdefault("destination", intake.get("destination", ""))
    parsed.setdefault("recommended_areas", [])
    parsed.setdefault("attractions", [])
    parsed.setdefault("planning_hints", [])
    parsed.setdefault("constraints", [])

    return parsed