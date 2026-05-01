import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
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
# JSON Extract
# ======================
def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {"raw_output": text}

# ======================
# Helper functions (保留你原来的逻辑)
# ======================
def parse_date_safe(date_str: str):
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(str(date_str))
    except:
        return None

def add_days(date_str: str, days: int) -> str:
    dt = parse_date_safe(date_str)
    if not dt:
        return ""
    return (dt + timedelta(days=days)).date().isoformat()

def chunk_list(items: List[Any], n: int) -> List[List[Any]]:
    if n <= 0:
        return [items]
    chunks = [[] for _ in range(n)]
    for i, item in enumerate(items):
        chunks[i % n].append(item)
    return chunks

# ======================
# Core Planner Agent
# ======================
def planner_agent(initial: Dict[str, Any], research: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
You are a travel planner.

Input:
{json.dumps(initial)}

Research:
{json.dumps(research)}

Output JSON with:
- itinerary_summary
- days (list)
"""

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {"role": "system", "content": "You are a careful planner."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=2000,
    )

    text = response.choices[0].message.content or ""
    parsed = extract_json(text)

    parsed.setdefault("itinerary_summary", "")
    parsed.setdefault("days", [])
    parsed.setdefault("tips", research.get("planning_hints", []))
    parsed.setdefault("generated_at", datetime.now(timezone.utc).isoformat())

    return parsed

# ======================
# Fallback Planner
# ======================
def build_simple_fallback_planner_output(initial: Dict[str, Any], research: Dict[str, Any]) -> Dict[str, Any]:
    duration = int(initial.get("trip_duration_days", 3))
    travel_date = initial.get("travel_date", "")

    attractions = research.get("attractions", [])

    days = []
    for i in range(duration):
        day_date = add_days(travel_date, i)

        activity = attractions[i % len(attractions)] if attractions else {}

        days.append({
            "day": i + 1,
            "date": day_date,
            "focus_area": activity.get("area", "General"),
            "morning": [activity],
            "afternoon": [],
            "evening": [],
            "meal_notes": "",
            "transport_notes": "",
            "budget_notes": ""
        })

    return {
        **initial,
        "itinerary_summary": "Fallback itinerary",
        "days": days,
        "tips": research.get("planning_hints", []),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }