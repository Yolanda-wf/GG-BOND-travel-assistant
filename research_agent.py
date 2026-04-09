import json
import os
import re
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from .env
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is missing. Put it in your .env file."
    )

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Try to extract a JSON object from model output.
    If parsing fails, return a fallback dict with raw text.
    """
    text = text.strip()

    # First try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Then try to extract the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Fallback
    return {
        "raw_output": text
    }


def build_prompt(destination: str, days: int, preferences: List[str]) -> str:
    return f"""
You are the Research Agent in a multi-agent travel planning system.

Your job:
- Collect and organize travel-relevant information for a trip
- Help the Planner Agent by providing structured research notes
- Do NOT create a final itinerary
- Do NOT write day-by-day scheduling
- Do NOT invent highly specific facts if you are unsure

Trip input:
- Destination: {destination}
- Duration: {days} days
- Preferences: {preferences}

Return ONLY valid JSON with this schema:

{{
  "destination": "...",
  "recommended_areas": [
    {{
      "name": "...",
      "why_relevant": "...",
      "tags": ["..."]
    }}
  ],
  "attractions": [
    {{
      "name": "...",
      "area": "...",
      "category": "...",
      "time_needed": "...",
      "notes": "..."
    }}
  ],
  "planning_hints": [
    "..."
  ],
  "constraints": [
    "..."
  ]
}}

Rules:
- Keep it concise but useful
- Prefer structured bullet-like facts over long prose
- Include only useful items that would help a planner
- If you are uncertain about something, say so in notes
- Output JSON only, no markdown, no extra text
""".strip()


def research_agent(destination: str, days: int, preferences: List[str]) -> Dict[str, Any]:
    prompt = build_prompt(destination, days, preferences)

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a careful travel research assistant."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        temperature=0.4,
        max_tokens=700,
    )

    text = response.choices[0].message.content or ""
    parsed = extract_json_from_text(text)

    # Add a few top-level fields if missing, so downstream code has a stable shape
    if "destination" not in parsed:
        parsed["destination"] = destination
    if "planning_hints" not in parsed:
        parsed["planning_hints"] = []
    if "constraints" not in parsed:
        parsed["constraints"] = []
    if "recommended_areas" not in parsed:
        parsed["recommended_areas"] = []
    if "attractions" not in parsed:
        parsed["attractions"] = []

    return parsed


def main():
    # Simple test input
    destination = "Tokyo"
    days = 3
    preferences = ["museums", "food", "relaxed pace"]

    notes = research_agent(destination, days, preferences)
    print(json.dumps(notes, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()