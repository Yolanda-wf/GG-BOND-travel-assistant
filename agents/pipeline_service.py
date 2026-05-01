import re
from datetime import datetime, timezone

from research_agent import research_agent
from planner_agent import (
    planner_agent,
    build_simple_fallback_planner_output,
    build_full_input_schema,
)
from writer_agent import writer_agent, build_writer_output_json


def run_pipeline_from_initial_data(initial_data: dict) -> dict:
    """
    Run the full travel agent pipeline:
    1. Research Agent
    2. Planner Agent
    3. Writer Agent
    4. Parse writer markdown for frontend display

    This version is more deployment-friendly:
    - If one agent fails, the app will not immediately crash.
    - It uses fallback planner output when needed.
    - It avoids depending on local JSON files in this service layer.
    """

    # 1. Research Agent
    try:
        research_data = research_agent(initial_data)
        if not isinstance(research_data, dict):
            research_data = {}
    except Exception as e:
        print("Research agent failed:", e)
        research_data = {}

    # 2. Planner Agent
    try:
        planner_output = planner_agent(initial_data, research_data)
        if not isinstance(planner_output, dict):
            planner_output = {}
    except Exception as e:
        print("Planner agent failed:", e)
        planner_output = {}

    # 3. Validate planner output
    try:
        expected_days = int(initial_data.get("trip_duration_days", 5))
    except Exception:
        expected_days = 5

    if (
        "days" not in planner_output
        or not isinstance(planner_output.get("days"), list)
        or len(planner_output.get("days", [])) < expected_days
    ):
        planner_output = build_simple_fallback_planner_output(initial_data, research_data)

    # 4. Add flattened input schema
    flat_input = build_full_input_schema(initial_data)
    for key, value in flat_input.items():
        planner_output.setdefault(key, value)

    # 5. Add default fields
    planner_output.setdefault("itinerary_summary", "")
    planner_output.setdefault(
        "research_summary",
        {
            "recommended_areas": research_data.get("recommended_areas", []),
            "attractions": research_data.get("attractions", []),
            "planning_hints": research_data.get("planning_hints", []),
            "constraints": research_data.get("constraints", []),
        },
    )
    planner_output.setdefault("days", [])
    planner_output.setdefault("tips", research_data.get("planning_hints", []))
    planner_output.setdefault("generated_at", datetime.now(timezone.utc).isoformat())

    # 6. Writer Agent
    try:
        writer_markdown = writer_agent(planner_output)
        if not isinstance(writer_markdown, str):
            writer_markdown = str(writer_markdown)
    except Exception as e:
        print("Writer agent failed:", e)
        writer_markdown = (
            "# Travel Plan Generation Failed\n\n"
            "The system was unable to generate the final itinerary. "
            "Please check the API key, model setting, or agent output format, then try again."
        )

    # 7. Build writer output JSON
    try:
        writer_output_json = build_writer_output_json(planner_output, writer_markdown)
    except Exception as e:
        print("Build writer output JSON failed:", e)
        writer_output_json = {
            "planner_output": planner_output,
            "writer_markdown": writer_markdown,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # 8. Parse markdown for frontend cards
    parsed = parse_writer_markdown(writer_markdown)

    return {
        "initial_data": initial_data,
        "research_data": research_data,
        "planner_output": planner_output,
        "writer_markdown": writer_markdown,
        "writer_output_json": writer_output_json,
        "day_cards": parsed.get("day_cards", []),
        "extra_sections": parsed.get("extra_sections", []),
    }


def _clean_inline_markdown(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def _append_field(block: dict, field_name: str, content: str) -> None:
    content = _clean_inline_markdown(content)
    if not content:
        return

    if block[field_name]:
        block[field_name] += " " + content
    else:
        block[field_name] = content


def parse_writer_markdown(writer_markdown: str) -> dict:
    """
    Parse writer markdown into:
    1. day_cards: structured day cards with schedule / transport / meal
    2. extra_sections: additional markdown sections
    """
    if not writer_markdown or not isinstance(writer_markdown, str):
        return {"day_cards": [], "extra_sections": []}

    lines = writer_markdown.splitlines()

    day_cards = []
    current_day = None

    extra_sections = []
    current_extra_section = None

    day_heading_pattern = re.compile(
        r"^(?:#{1,6}\s*)?(?:[-*]\s*)?(?:\*\*)?(Day\s+\d+\s+—\s+.+?)(?:\*\*)?\s*$",
        re.IGNORECASE,
    )

    generic_heading_pattern = re.compile(
        r"^(#{1,6})\s+(.+?)\s*$"
    )

    field_patterns = {
        "transport": re.compile(
            r"^(?:[-*]\s*)?(?:\*\*)?Transport(?:\s+notes?)?(?:\*\*)?\s*:\s*(.*)$",
            re.IGNORECASE,
        ),
        "meal": re.compile(
            r"^(?:[-*]\s*)?(?:\*\*)?Meal(?:\s+notes?)?(?:\*\*)?\s*:\s*(.*)$",
            re.IGNORECASE,
        ),
    }

    for raw_line in lines:
        stripped = raw_line.strip()

        if not stripped:
            continue

        # Detect day headings
        day_match = day_heading_pattern.match(stripped)
        if day_match:
            if current_day:
                day_cards.append(current_day)

            if current_extra_section:
                extra_sections.append(current_extra_section)
                current_extra_section = None

            title = _clean_inline_markdown(day_match.group(1))
            current_day = {
                "title": title,
                "schedule": "",
                "transport": "",
                "meal": "",
            }
            continue

        # Detect generic markdown headings
        heading_match = generic_heading_pattern.match(stripped)
        if heading_match:
            heading_text = _clean_inline_markdown(heading_match.group(2))

            if not re.match(r"^Day\s+\d+\s+—", heading_text, re.IGNORECASE):
                if current_day:
                    day_cards.append(current_day)
                    current_day = None

                if current_extra_section:
                    extra_sections.append(current_extra_section)

                current_extra_section = {
                    "title": heading_text,
                    "content": "",
                }
                continue

        # Extra section content
        if current_extra_section is not None:
            line_text = re.sub(r"^[-*]\s*", "", stripped).strip()
            line_text = _clean_inline_markdown(line_text)

            if line_text:
                if current_extra_section["content"]:
                    current_extra_section["content"] += " " + line_text
                else:
                    current_extra_section["content"] = line_text
            continue

        # Ignore lines outside day sections
        if current_day is None:
            continue

        # Match transport / meal fields
        matched_field = False
        for field_name, pattern in field_patterns.items():
            match = pattern.match(stripped)
            if match:
                _append_field(current_day, field_name, match.group(1))
                matched_field = True
                break

        if matched_field:
            continue

        # Everything else inside a day goes to schedule
        cleaned = re.sub(r"^[-*]\s*", "", stripped).strip()

        # Skip budget notes from card display
        if re.match(
            r"^(?:\*\*)?Budget(?:\s+note|\s+notes)?(?:\*\*)?\s*:",
            cleaned,
            re.IGNORECASE,
        ):
            continue

        _append_field(current_day, "schedule", cleaned)

    if current_day:
        day_cards.append(current_day)

    if current_extra_section:
        extra_sections.append(current_extra_section)

    return {
        "day_cards": day_cards,
        "extra_sections": extra_sections,
    }