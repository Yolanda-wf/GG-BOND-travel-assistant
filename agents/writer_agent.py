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
WRITER_MODEL = os.getenv(
    "WRITER_MODEL",
    os.getenv("OPENROUTER_MODEL", "openrouter/free"),
)

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing. Put it in your .env file.")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)


def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    if not isinstance(data, dict):
        return default
    return data.get(key, default)


def clamp_text(value: Any, max_chars: int = 500) -> str:
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def compact_planner_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reduce the planner JSON to the essentials the writer needs.
    This keeps prompts smaller and avoids sending noise back to the model.
    """
    plan = deepcopy(plan or {})

    compacted: Dict[str, Any] = {
        "destination": plan.get("destination"),
        "trip_duration_days": plan.get("trip_duration_days"),
        "travel_style": plan.get("travel_style", []),
        "overall_strategy": clamp_text(plan.get("overall_strategy", ""), 900),
        "budget_summary": plan.get("budget_summary", {}),
        "assumptions": [clamp_text(x, 200) for x in (plan.get("assumptions") or [])[:10]],
        "risk_notes": [clamp_text(x, 200) for x in (plan.get("risk_notes") or [])[:10]],
        "booking_notes": [clamp_text(x, 200) for x in (plan.get("booking_notes") or [])[:10]],
        "daily_itinerary": [],
        "packing_list": [],
    }

    for day in (plan.get("daily_itinerary") or [])[:14]:
        if not isinstance(day, dict):
            continue

        compacted["daily_itinerary"].append(
            {
                "day": day.get("day"),
                "theme": day.get("theme"),
                "weather_consideration": clamp_text(day.get("weather_consideration", ""), 250),
                "morning": _compact_schedule_items(day.get("morning", [])),
                "afternoon": _compact_schedule_items(day.get("afternoon", [])),
                "evening": _compact_schedule_items(day.get("evening", [])),
                "meals": _compact_meals(day.get("meals", {})),
                "lodging_area": day.get("lodging_area", ""),
                "estimated_cost": day.get("estimated_cost", {}),
                "backup_plan": clamp_text(day.get("backup_plan", ""), 250),
                "notes": clamp_text(day.get("notes", ""), 250),
            }
        )

    for item in (plan.get("packing_list") or [])[:20]:
        if not isinstance(item, dict):
            continue
        compacted["packing_list"].append(
            {
                "item": item.get("item"),
                "reason": clamp_text(item.get("reason", ""), 180),
                "priority": item.get("priority", "recommended"),
            }
        )

    return compacted


def _compact_schedule_items(items: Any) -> List[Dict[str, Any]]:
    compacted: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return compacted

    for item in items[:6]:
        if not isinstance(item, dict):
            continue
        compacted.append(
            {
                "time": item.get("time", ""),
                "plan": clamp_text(item.get("plan", ""), 220),
                "location": item.get("location", ""),
                "why_this": clamp_text(item.get("why_this", ""), 180),
                "verification_status": item.get("verification_status", "needs_verification"),
            }
        )
    return compacted


def _compact_meals(meals: Any) -> Dict[str, Any]:
    if not isinstance(meals, dict):
        return {}

    result = {}
    for meal_name in ["breakfast", "lunch", "dinner"]:
        meal = meals.get(meal_name, {})
        if isinstance(meal, dict):
            result[meal_name] = {
                "name": meal.get("name", ""),
                "area": meal.get("area", ""),
                "verification_status": meal.get("verification_status", "needs_verification"),
            }
    return result


def build_writer_prompt(plan: Dict[str, Any]) -> str:
    compact_plan = compact_planner_plan(plan)

    return f"""
You are the Writer Agent in a multi-agent travel planning system.

Your job:
- Turn the planner's structured itinerary into a polished, human-readable Markdown travel guide
- Make it feel useful, professional, and pleasant to read
- Keep the facts faithful to the planner output
- Do not invent unsupported hotels, restaurants, attractions, or prices
- If something is marked needs_verification, present it cautiously
- Keep the itinerary realistic and coherent

Hard rules:
- Output Markdown only
- Do not output JSON
- Do not mention internal agent names
- Do not add fake precision
- If the plan is incomplete, say so gracefully and preserve the usable parts
- Use clear headings and concise paragraphs
- Include practical tips, pacing notes, and budget context

Planner input:
{json.dumps(compact_plan, ensure_ascii=False, indent=2)}

Required output structure:
# Trip Overview
## Destination
## Travel Style
## Overall Strategy

# Day-by-Day Itinerary
For each day:
- Use a heading like: ## Day 1 — <theme>
- Include morning / afternoon / evening subsections
- Mention meals
- Mention lodging area
- Mention weather consideration
- Mention backup plan
- Mention approximate daily cost summary

# Packing List
# Budget Summary
# Booking Notes
# Risk Notes
# Assumptions

Writing style:
- Professional, polished, and practical
- Easy to scan
- A little vivid, but not verbose
- Friendly and confident, but honest about uncertainty

Important:
- If a restaurant/hotel/attraction is not verified, say "needs verification" or "to confirm"
- Do not overstate certainty
- Keep the final result directly usable for a traveler

Return Markdown only.
""".strip()


def render_fallback_markdown(plan: Dict[str, Any]) -> str:
    """
    Deterministic fallback renderer.
    Used when the model response is invalid or unavailable.
    """
    destination = safe_get(plan, "destination", "Unknown destination")
    duration = safe_get(plan, "trip_duration_days", len(plan.get("daily_itinerary", [])))
    travel_style = plan.get("travel_style", [])
    overall_strategy = plan.get("overall_strategy", "")
    budget_summary = plan.get("budget_summary", {})
    assumptions = plan.get("assumptions", [])
    risk_notes = plan.get("risk_notes", [])
    booking_notes = plan.get("booking_notes", [])
    packing_list = plan.get("packing_list", [])
    days = plan.get("daily_itinerary", [])

    lines: List[str] = []
    lines.append(f"# Trip Overview")
    lines.append("")
    lines.append(f"**Destination:** {destination}")
    lines.append(f"**Trip Duration:** {duration} days")
    if travel_style:
        lines.append(f"**Travel Style:** {', '.join(map(str, travel_style))}")
    if overall_strategy:
        lines.append("")
        lines.append(overall_strategy)
    lines.append("")

    lines.append("# Day-by-Day Itinerary")
    lines.append("")

    for day in days:
        if not isinstance(day, dict):
            continue

        day_num = day.get("day", "?")
        theme = day.get("theme", "Planned day")
        lines.append(f"## Day {day_num} — {theme}")
        lines.append("")

        weather = day.get("weather_consideration", "")
        if weather:
            lines.append(f"**Weather:** {weather}")
            lines.append("")

        for segment_name in ["morning", "afternoon", "evening"]:
            segment_items = day.get(segment_name, [])
            if not segment_items:
                continue
            lines.append(f"### {segment_name.capitalize()}")
            for item in segment_items:
                if not isinstance(item, dict):
                    continue
                time_label = item.get("time", "")
                plan_text = item.get("plan", "")
                location = item.get("location", "")
                why = item.get("why_this", "")
                status = item.get("verification_status", "needs_verification")
                bullet = f"- **{time_label}** — {plan_text}"
                if location:
                    bullet += f" at {location}"
                if why:
                    bullet += f" ({why})"
                if status != "supported":
                    bullet += " [needs verification]"
                lines.append(bullet)
            lines.append("")

        meals = day.get("meals", {})
        if isinstance(meals, dict) and meals:
            lines.append("### Meals")
            for meal_name in ["breakfast", "lunch", "dinner"]:
                meal = meals.get(meal_name)
                if not isinstance(meal, dict):
                    continue
                name = meal.get("name", "To confirm")
                area = meal.get("area", "")
                status = meal.get("verification_status", "needs_verification")
                meal_line = f"- **{meal_name.capitalize()}:** {name}"
                if area:
                    meal_line += f" ({area})"
                if status != "supported":
                    meal_line += " [needs verification]"
                lines.append(meal_line)
            lines.append("")

        lodging_area = day.get("lodging_area", "")
        if lodging_area:
            lines.append(f"**Lodging area:** {lodging_area}")
        estimated_cost = day.get("estimated_cost", {})
        if isinstance(estimated_cost, dict) and estimated_cost:
            cost_parts = []
            for key in ["lodging", "food", "activities", "local_transport", "daily_total"]:
                if key in estimated_cost:
                    cost_parts.append(f"{key.replace('_', ' ').title()}: {estimated_cost.get(key)}")
            if cost_parts:
                currency = estimated_cost.get("currency", "")
                lines.append(f"**Estimated cost:** {', '.join(cost_parts)}{' (' + currency + ')' if currency else ''}")
        backup = day.get("backup_plan", "")
        if backup:
            lines.append(f"**Backup plan:** {backup}")
        notes = day.get("notes", "")
        if notes:
            lines.append(f"**Notes:** {notes}")
        lines.append("")

    lines.append("# Packing List")
    lines.append("")
    if packing_list:
        for item in packing_list:
            if not isinstance(item, dict):
                continue
            thing = item.get("item", "")
            reason = item.get("reason", "")
            priority = item.get("priority", "recommended")
            lines.append(f"- **{thing}** ({priority}) — {reason}")
    else:
        lines.append("- Packing details not provided.")
    lines.append("")

    lines.append("# Budget Summary")
    lines.append("")
    if isinstance(budget_summary, dict) and budget_summary:
        for key, value in budget_summary.items():
            lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
    else:
        lines.append("- Budget summary not provided.")
    lines.append("")

    if booking_notes:
        lines.append("# Booking Notes")
        lines.append("")
        for item in booking_notes:
            lines.append(f"- {item}")
        lines.append("")

    if risk_notes:
        lines.append("# Risk Notes")
        lines.append("")
        for item in risk_notes:
            lines.append(f"- {item}")
        lines.append("")

    if assumptions:
        lines.append("# Assumptions")
        lines.append("")
        for item in assumptions:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def normalize_markdown(text: str) -> str:
    """
    Light cleanup for model output.
    """
    text = (text or "").strip()

    # Strip accidental code fences if the model wrapped everything
    text = re.sub(r"^\s*```(?:markdown)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text + "\n"


def writer_agent(
    plan: Dict[str, Any],
    max_retries: int = 2,
) -> str:
    """
    Convert a structured planner output into a polished Markdown itinerary.
    """
    if not isinstance(plan, dict):
        raise ValueError("plan must be a dictionary produced by planner_agent")

    prompt = build_writer_prompt(plan)
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=WRITER_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a meticulous travel writer. "
                            "You must produce clean Markdown and avoid fabrication."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2500,
            )

            text = response.choices[0].message.content or ""
            text = normalize_markdown(text)

            # Basic sanity check: the output should look like Markdown, not JSON.
            if text.strip().startswith("{"):
                raise ValueError("Writer model returned JSON instead of Markdown.")

            return text

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1.25 * (attempt + 1))
            else:
                break

    # Deterministic fallback if model fails
    fallback = render_fallback_markdown(plan)
    if fallback.strip():
        return fallback

    raise RuntimeError(f"writer_agent failed after retries: {last_error}")


if __name__ == "__main__":
    # Minimal smoke test
    sample_plan = {
        "destination": "Tokyo",
        "trip_duration_days": 3,
        "travel_style": ["museum-focused", "relaxed pace", "food-oriented"],
        "overall_strategy": "Cluster activities by neighborhood to reduce transit time and keep the trip relaxed.",
        "daily_itinerary": [
            {
                "day": 1,
                "theme": "Traditional Tokyo",
                "weather_consideration": "If it rains, prioritize indoor museum time and nearby covered streets.",
                "morning": [
                    {
                        "time": "09:00",
                        "plan": "Visit Senso-ji",
                        "location": "Asakusa",
                        "why_this": "Iconic cultural landmark with strong local character.",
                        "verification_status": "supported",
                    }
                ],
                "afternoon": [
                    {
                        "time": "13:00",
                        "plan": "Explore Ueno area museums",
                        "location": "Ueno",
                        "why_this": "Good fit for museum interests.",
                        "verification_status": "supported",
                    }
                ],
                "evening": [
                    {
                        "time": "18:30",
                        "plan": "Dinner near your lodging area",
                        "location": "Ueno or Asakusa",
                        "why_this": "Keeps the day low-stress and easy to manage.",
                        "verification_status": "needs_verification",
                    }
                ],
                "meals": {
                    "breakfast": {
                        "name": "Local cafe",
                        "area": "Asakusa",
                        "verification_status": "needs_verification",
                    },
                    "lunch": {
                        "name": "Museum district lunch spot",
                        "area": "Ueno",
                        "verification_status": "needs_verification",
                    },
                    "dinner": {
                        "name": "Neighborhood restaurant",
                        "area": "Ueno",
                        "verification_status": "needs_verification",
                    },
                },
                "lodging_area": "Ueno",
                "estimated_cost": {
                    "lodging": "¥12,000-¥20,000",
                    "food": "¥4,000-¥7,000",
                    "activities": "¥1,000-¥3,000",
                    "local_transport": "¥800-¥1,500",
                    "daily_total": "¥17,800-¥31,500",
                    "currency": "JPY",
                },
                "backup_plan": "Swap outdoor walking for a museum-heavy day if weather turns poor.",
                "notes": "Start early to avoid crowds.",
            }
        ],
        "packing_list": [
            {
                "item": "Comfortable walking shoes",
                "reason": "You will likely walk a lot across neighborhoods.",
                "priority": "essential",
            }
        ],
        "budget_summary": {
            "currency": "JPY",
            "lodging_total": "¥36,000-¥60,000",
            "food_total": "¥12,000-¥21,000",
            "activities_total": "¥3,000-¥9,000",
            "transport_total": "¥2,400-¥4,500",
            "estimated_trip_total": "¥53,400-¥94,500",
            "confidence": "medium",
        },
        "assumptions": ["Assumes moderate comfort level for lodging and meals."],
        "risk_notes": ["Popular attractions may require early arrival."],
        "booking_notes": ["Confirm restaurant availability before peak dining hours."],
    }

    print(writer_agent(sample_plan))
