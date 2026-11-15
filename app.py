from pathlib import Path
import json
import os

from flask import Flask, jsonify, request, send_from_directory

from engine import CatalogueEngine

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT / "static"))


def load_engine():
    with (ROOT / "rules.json").open(encoding="utf-8") as handle:
        rules = json.load(handle)
    with (ROOT / "catalog.json").open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    return CatalogueEngine(catalog, rules)


engine = load_engine()


def _error(message, status):
    return jsonify(error=message), status


def _number(value, name, low, high, default):
    if value is None or value == "":
        return default, None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None, _error(f"{name} must be a number.", 400)
    try:
        number = float(value)
    except ValueError:
        return None, _error(f"{name} must be a number.", 400)
    if number != number or not low <= number <= high:
        return None, _error(f"{name} must be between {low} and {high}.", 400)
    return number, None


def _limit(value, default):
    if value is None or value == "":
        return default, None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None, _error("limit must be a whole number from 1 to 20.", 400)
    try:
        number = int(value)
    except ValueError:
        return None, _error("limit must be a whole number from 1 to 20.", 400)
    if number < 1 or number > 20:
        return None, _error("limit must be a whole number from 1 to 20.", 400)
    return number, None


def _floors_from(source):
    support, error = _number(source.get("min_support"), "min_support", 0, 1, 0)
    if error:
        return None, error
    confidence, error = _number(source.get("min_confidence"), "min_confidence", 0, 1, 0)
    if error:
        return None, error
    lift, error = _number(source.get("min_lift"), "min_lift", 0, 1000, 0)
    if error:
        return None, error
    return {
        "min_support": support,
        "min_confidence": confidence,
        "min_lift": lift,
    }, None


def _codes_from_body(allow_empty):
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, None, None, _error("Expected a JSON object with a codes array.", 400)
    codes = data.get("codes", [])
    if not isinstance(codes, list) or len(codes) > 40:
        return None, None, None, _error("codes must be an array of at most 40 product codes.", 400)
    if not codes and not allow_empty:
        return None, None, None, _error("Add at least one product code.", 400)
    cleaned = []
    for code in codes:
        if not isinstance(code, str) or not code.strip() or len(code.strip()) > 32:
            return None, None, None, _error("Each code must be a non-empty string.", 400)
        cleaned.append(code.strip())
    limit, error = _limit(data.get("limit"), 10)
    if error:
        return None, None, None, error
    floors, error = _floors_from(data)
    if error:
        return None, None, None, error
    return cleaned, limit, floors, None


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/stats")
def stats():
    return jsonify(engine.stats())


@app.route("/api/analytics")
def analytics():
    return jsonify(engine.analytics())


@app.route("/api/catalog")
def get_catalog():
    return jsonify(engine.catalog_public())


@app.route("/api/presets")
def get_presets():
    return jsonify(engine.presets())


@app.route("/api/recommend", methods=["POST"])
def recommend():
    codes, limit, floors, error = _codes_from_body(allow_empty=False)
    if error:
        return error
    return jsonify(engine.recommend(codes, limit=limit, floors=floors))


@app.route("/api/whatif", methods=["POST"])
def what_if():
    codes, limit, _floors, error = _codes_from_body(allow_empty=True)
    if error:
        return error
    return jsonify(engine.what_if(codes, limit=limit))


@app.route("/api/product/<code>")
def product_rules(code):
    cleaned = code.strip()
    if not cleaned or len(cleaned) > 32:
        return _error("Unknown product.", 404)
    floors, error = _floors_from(request.args)
    if error:
        return error
    limit, error = _limit(request.args.get("limit"), 8)
    if error:
        return error
    found = engine.product(cleaned, limit=limit, floors=floors)
    if found is None:
        return _error("Unknown product.", 404)
    return jsonify(found)


@app.route("/api/similar")
def similar():
    query = request.args.get("q", "")
    if len(query) > 500:
        return _error("Query is too long.", 400)
    floors, error = _floors_from(request.args)
    if error:
        return error
    limit, error = _limit(request.args.get("limit"), 12)
    if error:
        return error
    return jsonify(engine.similar(query, limit=limit, floors=floors))


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1",
        port=int(os.environ.get("PORT", "5000")),
    )