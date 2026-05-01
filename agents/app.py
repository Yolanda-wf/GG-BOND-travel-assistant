import os
from flask import Flask, render_template, request, jsonify

from pipeline_service import run_pipeline_from_initial_data

app = Flask(__name__)


def build_initial_data_from_form(form) -> dict:
    destination = form.get("destination", "").strip()
    travel_date = form.get("travel_date", "").strip()

    raw_days = form.get("trip_duration_days", "").strip()
    try:
        trip_duration_days = int(raw_days) if raw_days else 5
    except ValueError:
        trip_duration_days = 5

    budget_level = form.get("budget_level", "").strip() or "medium"
    travel_companion = form.get("travel_companion", "").strip() or "solo"
    activity_preferences = form.getlist("activity_preferences")

    halal = 1 if form.get("halal", "0").strip() == "1" else 0
    vegan = 1 if form.get("vegan", "0").strip() == "1" else 0
    extra_preferences = form.get("extra_preferences", "").strip()

    BUDGET_RANGES = {
        "low": {"min_usd": 0, "max_usd": 1000},
        "medium": {"min_usd": 1000, "max_usd": 2500},
        "high": {"min_usd": 2500, "max_usd": None},
    }

    ACTIVITY_OPTIONS = [
        "city_sightseeing",
        "outdoor_adventures",
        "festivals_nightlife",
        "food_exploration",
        "shopping",
    ]

    ACTIVITY_TO_ID = {
        "city_sightseeing": 1,
        "outdoor_adventures": 2,
        "festivals_nightlife": 3,
        "food_exploration": 4,
        "shopping": 5,
    }

    BUDGET_TO_ID = {"low": 1, "medium": 2, "high": 3}
    COMPANION_TO_ID = {"solo": 1, "couple": 2, "family": 3, "friends": 4}

    if budget_level not in BUDGET_TO_ID:
        budget_level = "medium"

    if travel_companion not in COMPANION_TO_ID:
        travel_companion = "solo"

    activity_binary_vector = [1 if a in activity_preferences else 0 for a in ACTIVITY_OPTIONS]
    activity_ids = [ACTIVITY_TO_ID[a] for a in activity_preferences if a in ACTIVITY_TO_ID]

    return {
        "schema_version": "1.0",
        "destination": destination,
        "travel_date": travel_date,
        "trip_duration_days": trip_duration_days,
        "budget_level": budget_level,
        "budget_level_id": BUDGET_TO_ID[budget_level],
        "budget_range_usd": BUDGET_RANGES[budget_level],
        "travel_companion": travel_companion,
        "travel_companion_id": COMPANION_TO_ID[travel_companion],
        "activity_preferences": activity_preferences,
        "activity_ids": activity_ids,
        "activity_binary_vector": activity_binary_vector,
        "food_preferences": {
            "halal": halal,
            "vegan": vegan,
        },
        "extra_preferences": extra_preferences,
        "extra_preferences_char_count": len(extra_preferences),
        "extra_preferences_word_count": len(extra_preferences.split()) if extra_preferences else 0,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        initial_data = build_initial_data_from_form(request.form)
        result = run_pipeline_from_initial_data(initial_data)

        return jsonify({
            "success": True,
            "writer_markdown": result.get("writer_markdown", ""),
            "day_cards": result.get("day_cards", []),
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
