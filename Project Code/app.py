"""
OptiCrop — Flask Application

Smart Agricultural Production Optimization Engine.
Provides web-based crop recommendations powered by machine learning.
"""

import os
from flask import Flask, render_template, request, jsonify

from config import config
from ml.predict import CropPredictor, PARAM_RANGES


def create_app(config_name: str = "default") -> Flask:
    """Application factory."""
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # ── Load ML model once at startup ────────────────────────────────────
    model_dir = app.config["MODEL_DIR"]
    try:
        predictor = CropPredictor(model_dir)
        app.config["PREDICTOR"] = predictor
        print("[OK] ML model loaded successfully.")
    except FileNotFoundError as exc:
        print(f"[WARN] {exc}")
        print("   The app will start, but predictions will fail until you train the model.")
        app.config["PREDICTOR"] = None

    # ── Routes ───────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        """Home page with the prediction form."""
        return render_template("index.html", param_ranges=PARAM_RANGES)

    @app.route("/predict", methods=["POST"])
    def predict():
        """Process form submission and display result."""
        predictor = app.config.get("PREDICTOR")
        if predictor is None:
            return render_template("index.html", param_ranges=PARAM_RANGES,
                                   error="Model not loaded. Please train the model first.")

        params = {
            "N": request.form.get("N", "").strip(),
            "P": request.form.get("P", "").strip(),
            "K": request.form.get("K", "").strip(),
            "temperature": request.form.get("temperature", "").strip(),
            "humidity": request.form.get("humidity", "").strip(),
            "ph": request.form.get("ph", "").strip(),
            "rainfall": request.form.get("rainfall", "").strip(),
        }

        # Validate
        errors = predictor.validate_input(params)
        if errors:
            return render_template("index.html", param_ranges=PARAM_RANGES,
                                   errors=errors, form_data=params)

        # Predict
        result = predictor.predict(params)
        return render_template("result.html", result=result)

    @app.route("/about")
    def about():
        """About / methodology page."""
        return render_template("about.html")

    @app.route("/api/predict")
    def api_predict():
        """REST API endpoint — accepts query params, returns JSON."""
        predictor = app.config.get("PREDICTOR")
        if predictor is None:
            return jsonify({"error": "Model not loaded."}), 503

        params = {
            "N": request.args.get("N", "").strip(),
            "P": request.args.get("P", "").strip(),
            "K": request.args.get("K", "").strip(),
            "temperature": request.args.get("temperature", "").strip(),
            "humidity": request.args.get("humidity", "").strip(),
            "ph": request.args.get("ph", "").strip(),
            "rainfall": request.args.get("rainfall", "").strip(),
        }

        errors = predictor.validate_input(params)
        if errors:
            return jsonify({"errors": errors}), 400

        result = predictor.predict(params)
        return jsonify(result)

    # ── Error handlers ───────────────────────────────────────────────────

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template("base.html", error_code=404,
                               error_msg="Page not found."), 404

    @app.errorhandler(500)
    def internal_error(e):
        return render_template("base.html", error_code=500,
                               error_msg="Internal server error."), 500

    return app


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    application = create_app("development")
    application.run(host="127.0.0.1", port=5000, debug=True)
