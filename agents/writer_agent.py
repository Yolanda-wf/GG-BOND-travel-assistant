import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, Any
from openai import OpenAI

# ======================
# Config (Deployment Ready)
# ======================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# ======================
# Writer Agent
# ======================
def writer_agent(planner_output: Dict[str, Any]) -> str:
    prompt = f"""
You are a travel writer.

Convert the structured itinerary into clean markdown.

Rules:
- Use clear sections
- Each day must be formatted like:

## Day X — Title
- Morning: ...
- Afternoon: ...
- Evening: ...
- Transport: ...
- Meal: ...

- After all days, add summary sections if needed
- Output markdown only

Input:
{json.dumps(planner_output)}
"""

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {"role": "system", "content": "You are a professional travel writer."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=2000,
    )

    return response.choices[0].message.content or ""

# ======================
# Build JSON Output
# ======================
def build_writer_output_json(planner_output: Dict[str, Any], writer_markdown: str) -> Dict[str, Any]:
    return {
        "planner_output": planner_output,
        "writer_markdown": writer_markdown,
        "generated_at": datetime.now(timezone.utc).isoformat()
    }