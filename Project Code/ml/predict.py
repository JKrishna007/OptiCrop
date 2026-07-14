"""
OptiCrop — Prediction Module

Loads the persisted model artifacts and exposes a prediction API used by
the Flask application.
"""

import os
import joblib
import numpy as np


# ── Input parameter constraints (used for validation) ───────────────────────
PARAM_RANGES = {
    "N":           (0, 200,  "Nitrogen (kg/ha)"),
    "P":           (0, 200,  "Phosphorous (kg/ha)"),
    "K":           (0, 300,  "Potassium (kg/ha)"),
    "temperature": (-10, 55, "Temperature (°C)"),
    "humidity":    (0, 100,  "Humidity (%)"),
    "ph":          (0, 14,   "pH Level"),
    "rainfall":    (0, 500,  "Rainfall (mm)"),
}

# ── Extra crop information ──────────────────────────────────────────────────
CROP_INFO = {
    "rice":        {"emoji": "🌾", "season": "Kharif", "tip": "Requires standing water; ideal in clay-loam soils with good water retention."},
    "maize":       {"emoji": "🌽", "season": "Kharif / Rabi", "tip": "Prefers well-drained loamy soil; avoid waterlogging."},
    "chickpea":    {"emoji": "🫘", "season": "Rabi", "tip": "Thrives in cool, dry conditions; minimal irrigation needed."},
    "kidneybeans": {"emoji": "🫘", "season": "Kharif", "tip": "Grows best in well-drained soils with moderate rainfall."},
    "pigeonpeas":  {"emoji": "🌱", "season": "Kharif", "tip": "Drought-tolerant legume; fixes atmospheric nitrogen."},
    "mothbeans":   {"emoji": "🌱", "season": "Kharif", "tip": "Highly drought-resistant; suited for arid regions."},
    "mungbean":    {"emoji": "🌱", "season": "Kharif / Summer", "tip": "Short duration crop; enriches soil nitrogen."},
    "blackgram":   {"emoji": "🌱", "season": "Kharif", "tip": "Grows well in heavy soils; tolerant to shade."},
    "lentil":      {"emoji": "🥣", "season": "Rabi", "tip": "Cool-season crop; well-drained loamy soil is ideal."},
    "pomegranate": {"emoji": "🍎", "season": "Year-round", "tip": "Drought-tolerant fruit; prefers hot, dry summers."},
    "banana":      {"emoji": "🍌", "season": "Year-round", "tip": "Needs rich, well-drained soil with consistent moisture."},
    "mango":       {"emoji": "🥭", "season": "Summer", "tip": "Deep-rooted tree; requires warm climate and good drainage."},
    "grapes":      {"emoji": "🍇", "season": "Winter planting", "tip": "Prefers warm, dry climate with well-drained sandy loam."},
    "watermelon":  {"emoji": "🍉", "season": "Summer", "tip": "Needs warm temperatures and sandy loam soil."},
    "muskmelon":   {"emoji": "🍈", "season": "Summer", "tip": "Requires warm soil and full sunlight; avoid excess water."},
    "apple":       {"emoji": "🍎", "season": "Winter (chilling)", "tip": "Requires cold winters for dormancy; thrives in hilly regions."},
    "orange":      {"emoji": "🍊", "season": "Year-round", "tip": "Prefers subtropical climate with well-drained soil."},
    "papaya":      {"emoji": "🍈", "season": "Year-round", "tip": "Fast-growing tropical fruit; sensitive to frost."},
    "coconut":     {"emoji": "🥥", "season": "Year-round", "tip": "Coastal tropical crop; sandy loam near sea level."},
    "cotton":      {"emoji": "🏵️", "season": "Kharif", "tip": "Needs warm climate with moderate rainfall and black soil."},
    "jute":        {"emoji": "🌿", "season": "Kharif", "tip": "Requires high humidity and warm temperatures; alluvial soil."},
    "coffee":      {"emoji": "☕", "season": "Year-round", "tip": "Shade-grown at higher altitudes; needs well-drained acidic soil."},
}


class CropPredictor:
    """
    Wraps the trained model, scaler, and label encoder to provide a clean
    prediction interface for the Flask app.
    """

    def __init__(self, model_dir: str):
        model_path = os.path.join(model_dir, "model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        encoder_path = os.path.join(model_dir, "label_encoder.pkl")

        for path, name in [(model_path, "model"), (scaler_path, "scaler"), (encoder_path, "label_encoder")]:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{name} artifact not found at {path}.  "
                    "Run `python -m ml.train` first to generate model artifacts."
                )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)

    # ── Validation ──────────────────────────────────────────────────────────
    @staticmethod
    def validate_input(params: dict) -> list[str]:
        """
        Validate a dict of input parameters.

        Returns a list of error messages (empty if valid).
        """
        errors = []
        for key, (lo, hi, label) in PARAM_RANGES.items():
            val = params.get(key)
            if val is None or val == "":
                errors.append(f"{label} is required.")
                continue
            try:
                val = float(val)
            except (ValueError, TypeError):
                errors.append(f"{label} must be a number.")
                continue
            if not (lo <= val <= hi):
                errors.append(f"{label} must be between {lo} and {hi}.")
        return errors

    # ── Prediction ──────────────────────────────────────────────────────────
    def predict(self, params: dict) -> dict:
        """
        Accept a dict with keys N, P, K, temperature, humidity, ph, rainfall.

        Returns
        -------
        dict with keys: crop, confidence, crop_info, input_params
        """
        features = np.array([[
            float(params["N"]),
            float(params["P"]),
            float(params["K"]),
            float(params["temperature"]),
            float(params["humidity"]),
            float(params["ph"]),
            float(params["rainfall"]),
        ]])

        features_scaled = self.scaler.transform(features)

        # Predicted class
        pred_encoded = self.model.predict(features_scaled)[0]
        crop_name = self.label_encoder.inverse_transform([pred_encoded])[0]

        # Confidence (probability of the top class)
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(features_scaled)[0]
            confidence = float(np.max(probas))

        info = CROP_INFO.get(crop_name, {"emoji": "🌱", "season": "—", "tip": "—"})

        return {
            "crop": crop_name,
            "confidence": round(confidence * 100, 1),
            "emoji": info["emoji"],
            "season": info["season"],
            "tip": info["tip"],
            "input_params": {k: float(params[k]) for k in PARAM_RANGES},
        }
