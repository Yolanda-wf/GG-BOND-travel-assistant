import json
import os
import re
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing. Put it in your .env file.")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Best-effort JSON extraction:
    1) direct parse
    2) first {...} block
    3) fallback raw text container
    """
    text = (text or "").strip()

    if not text:
        return {"raw_output": ""}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to isolate the first JSON object.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return {"raw_output": text}


def clamp_text(value: Any, max_chars: int = 1200) -> str:
    """
    Convert a value to a string and clamp it to a safe length.
    Useful for keeping prompts compact and predictable.
    """
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def compact_research_notes(research_notes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reduce prompt size and remove noise while keeping the important planning context.
    """
    notes = deepcopy(research_notes or {})

    compacted: Dict[str, Any] = {
        "destination": notes.get("destination"),
        "recommended_areas": [],
        "attractions": [],
        "planning_hints": [],
        "constraints": [],
    }

    for area in (notes.get("recommended_areas") or [])[:6]:
        compacted["recommended_areas"].append(
            {
                "name": area.get("name"),
                "why_relevant": clamp_text(area.get("why_relevant", ""), 300),
                "tags": (area.get("tags") or [])[:8],
            }
        )

    for attraction in (notes.get("attractions") or [])[:18]:
        compacted["attractions"].append(
            {
                "name": attraction.get("name"),
                "area": attraction.get("area"),
                "category": attraction.get("category"),
                "time_needed": attraction.get("time_needed"),
                "notes": clamp_text(attraction.get("notes", ""), 250),
            }
        )

    compacted["planning_hints"] = [
        clamp_text(item, 200) for item in (notes.get("planning_hints") or [])[:10]
    ]
    compacted["constraints"] = [
        clamp_text(item, 200) for item in (notes.get("constraints") or [])[:10]
    ]

    return compacted


def build_planner_prompt(
    destination: str,
    days: int,
    preferences: List[str],
    research_notes: Dict[str, Any],
) -> str:
    compact_notes = compact_research_notes(research_notes)

    return f"""
You are the Planner Agent in a multi-agent travel planning system.

Your job:
- Turn research notes into a realistic, high-quality travel itinerary
- Create a day-by-day plan for exactly {days} days
- Prioritize practicality, pacing, and logical geography
- Use only the provided research context as your main source
- Do NOT invent very specific facts unless they are clearly supported by the research notes
- If a hotel, restaurant, or attraction is not present in the research notes, mark it as "needs_verification" instead of fabricating it
- Keep the itinerary coherent and bookable

Important rules:
- Do NOT output markdown
- Do NOT output prose outside JSON
- Do NOT ignore constraints
- Do NOT create more or fewer than {days} day objects
- Use a relaxed but realistic pace
- Balance travel time, meals, and rest
- Include weather-aware advice in each day
- Include estimated costs as rough ranges, not exact prices, unless your research notes are explicit
- If information is uncertain, say so clearly in the relevant notes field

Trip input:
- Destination: {destination}
- Duration: {days} days
- Preferences: {preferences}

Research notes:
{json.dumps(compact_notes, ensure_ascii=False, indent=2)}

Return ONLY valid JSON with this schema:

{{
  "destination": "...",
  "trip_duration_days": {days},
  "travel_style": ["..."],
  "overall_strategy": "...",
  "daily_itinerary": [
    {{
      "day": 1,
      "theme": "...",
      "weather_consideration": "...",
      "morning": [
        {{
          "time": "08:30",
          "plan": "...",
          "location": "...",
          "why_this": "...",
          "verification_status": "supported|needs_verification"
        }}
      ],
      "afternoon": [
        {{
          "time": "13:00",
          "plan": "...",
          "location": "...",
          "why_this": "...",
          "verification_status": "supported|needs_verification"
        }}
      ],
      "evening": [
        {{
          "time": "18:30",
          "plan": "...",
          "location": "...",
          "why_this": "...",
          "verification_status": "supported|needs_verification"
        }}
      ],
      "meals": {{
        "breakfast": {{
          "name": "...",
          "area": "...",
          "verification_status": "supported|needs_verification"
        }},
        "lunch": {{
          "name": "...",
          "area": "...",
          "verification_status": "supported|needs_verification"
        }},
        "dinner": {{
          "name": "...",
          "area": "...",
          "verification_status": "supported|needs_verification"
        }}
      }},
      "lodging_area": "...",
      "estimated_cost": {{
        "lodging": "rough range",
        "food": "rough range",
        "activities": "rough range",
        "local_transport": "rough range",
        "daily_total": "rough range",
        "currency": "..."
      }},
      "backup_plan": "...",
      "notes": "..."
    }}
  ],
  "packing_list": [
    {{
      "item": "...",
      "reason": "...",
      "priority": "essential|recommended|nice_to_have"
    }}
  ],
  "budget_summary": {{
    "currency": "...",
    "lodging_total": "rough range",
    "food_total": "rough range",
    "activities_total": "rough range",
    "transport_total": "rough range",
    "estimated_trip_total": "rough range",
    "confidence": "high|medium|low"
  }},
  "assumptions": [
    "..."
  ],
  "risk_notes": [
    "..."
  ],
  "booking_notes": [
    "..."
  ]
}}

Additional guidance:
- Prefer practical grouping by area to reduce commute time
- Avoid overpacking the schedule
- Include at least one recovery-friendly block if the trip is intense
- Make the plan feel human and realistic
- Keep the output concise but complete
- JSON only, no markdown, no extra text
""".strip()


def validate_planner_output(data: Dict[str, Any], expected_days: int) -> Dict[str, Any]:
    """
    Minimal structural validation with safe fallbacks.
    """
    if not isinstance(data, dict):
        return {}

    data.setdefault("destination", "")
    data.setdefault("trip_duration_days", expected_days)
    data.setdefault("travel_style", [])
    data.setdefault("overall_strategy", "")
    data.setdefault("daily_itinerary", [])
    data.setdefault("packing_list", [])
    data.setdefault("budget_summary", {})
    data.setdefault("assumptions", [])
    data.setdefault("risk_notes", [])
    data.setdefault("booking_notes", [])

    if not isinstance(data["daily_itinerary"], list):
        data["daily_itinerary"] = []

    # Normalize day count
    if len(data["daily_itinerary"]) > expected_days:
        data["daily_itinerary"] = data["daily_itinerary"][:expected_days]

    # Fill missing days with placeholders so downstream code never breaks
    existing_days = {
        item.get("day")
        for item in data["daily_itinerary"]
        if isinstance(item, dict) and isinstance(item.get("day"), int)
    }

    for day_num in range(1, expected_days + 1):
        if day_num not in existing_days:
            data["daily_itinerary"].append(
                {
                    "day": day_num,
                    "theme": "Needs planning",
                    "weather_consideration": "Needs verification",
                    "morning": [],
                    "afternoon": [],
                    "evening": [],
                    "meals": {
                        "breakfast": {
                            "name": "Needs verification",
                            "area": "",
                            "verification_status": "needs_verification",
                        },
                        "lunch": {
                            "name": "Needs verification",
                            "area": "",
                            "verification_status": "needs_verification",
                        },
                        "dinner": {
                            "name": "Needs verification",
                            "area": "",
                            "verification_status": "needs_verification",
                        },
                    },
                    "lodging_area": "",
                    "estimated_cost": {
                        "lodging": "unknown",
                        "food": "unknown",
                        "activities": "unknown",
                        "local_transport": "unknown",
                        "daily_total": "unknown",
                        "currency": "USD",
                    },
                    "backup_plan": "Needs verification",
                    "notes": "Planner output was incomplete; this day was filled with a safe placeholder.",
                }
            )

    data["daily_itinerary"] = sorted(
        data["daily_itinerary"],
        key=lambda x: x.get("day", 999) if isinstance(x, dict) else 999,
    )

    return data


def planner_agent(
    destination: str,
    days: int,
    preferences: List[str],
    research_notes: Dict[str, Any],
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Generate a structured itinerary from research notes.
    """
    if days <= 0:
        raise ValueError("days must be a positive integer")

    if not destination or not destination.strip():
        raise ValueError("destination cannot be empty")

    prompt = build_planner_prompt(destination, days, preferences, research_notes)

    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a meticulous travel planner. "
                            "You must follow the requested JSON schema and avoid fabrication."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=2200,
            )

            text = response.choices[0].message.content or ""
            parsed = extract_json_from_text(text)

            # If the model returned raw output instead of JSON, keep it visible for debugging.
            if "raw_output" in parsed and len(parsed.keys()) == 1:
                parsed = {
                    "destination": destination,
                    "trip_duration_days": days,
                    "travel_style": [],
                    "overall_strategy": "",
                    "daily_itinerary": [],
                    "packing_list": [],
                    "budget_summary": {
                        "currency": "USD",
                        "lodging_total": "unknown",
                        "food_total": "unknown",
                        "activities_total": "unknown",
                        "transport_total": "unknown",
                        "estimated_trip_total": "unknown",
                        "confidence": "low",
                    },
                    "assumptions": [],
                    "risk_notes": [
                        "Planner model did not return valid JSON on this attempt."
                    ],
                    "booking_notes": [],
                    "raw_output": parsed["raw_output"],
                }

            parsed = validate_planner_output(parsed, days)

            # Attach the research notes used so downstream components can trace provenance.
            parsed.setdefault("source_research_notes", compact_research_notes(research_notes))

            return parsed

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1.25 * (attempt + 1))
            else:
                break

    raise RuntimeError(f"planner_agent failed after retries: {last_error}")


def main():
    # Example: plug in the output from your Research Agent here.
    sample_research_notes = {
        "destination": "Tokyo",
        "recommended_areas": [
            {
                "name": "Shibuya",
                "why_relevant": "Great for food, nightlife, and central access.",
                "tags": ["food", "shopping", "nightlife"],
            },
            {
                "name": "Asakusa",
                "why_relevant": "Better for traditional sightseeing and a slower pace.",
                "tags": ["culture", "temples", "history"],
            },
        ],
        "attractions": [
            {
                "name": "Senso-ji",
                "area": "Asakusa",
                "category": "temple",
                "time_needed": "2-3 hours",
                "notes": "Popular early in the day.",
            },
            {
                "name": "Tokyo National Museum",
                "area": "Ueno",
                "category": "museum",
                "time_needed": "2-4 hours",
                "notes": "Good fit for museum-focused trips.",
            },
        ],
        "planning_hints": [
            "Use train-heavy routing.",
            "Mix one busy day with one lighter day.",
        ],
        "constraints": [
            "Avoid overloading the first day after arrival.",
        ],
    }

    itinerary = planner_agent(
        destination="Tokyo",
        days=3,
        preferences=["museums", "food", "relaxed pace"],
        research_notes=sample_research_notes,
    )

    print(json.dumps(itinerary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
