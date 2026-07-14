"""
Generate a synthetic crop recommendation dataset modeled after the well-known
Kaggle Crop Recommendation Dataset.

Each crop has characteristic ranges for N, P, K, temperature, humidity, pH,
and rainfall.  We sample 100 rows per crop from these ranges to produce a
2,200-row CSV file.
"""

import csv
import os
import random

# Reproducibility
random.seed(42)

# ── Crop profiles ────────────────────────────────────────────────────────────
# Format: (N_min, N_max, P_min, P_max, K_min, K_max,
#           temp_min, temp_max, hum_min, hum_max,
#           ph_min, ph_max, rain_min, rain_max)

CROP_PROFILES = {
    "rice":         (60, 100, 35, 65, 35, 55, 20, 28, 80, 95, 5.0, 7.0, 180, 260),
    "maize":        (60, 100, 35, 65, 15, 35, 18, 28, 55, 75, 5.5, 7.5, 60, 110),
    "chickpea":     (15, 45, 55, 85, 75, 85, 15, 22, 14, 20, 6.8, 7.8, 65, 95),
    "kidneybeans":  (15, 35, 55, 80, 15, 30, 15, 22, 18, 24, 5.5, 7.0, 60, 120),
    "pigeonpeas":   (15, 40, 55, 80, 15, 30, 18, 36, 30, 65, 4.5, 7.5, 120, 170),
    "mothbeans":    (15, 35, 40, 65, 15, 30, 24, 32, 40, 65, 3.5, 7.5, 30, 70),
    "mungbean":     (15, 45, 40, 65, 15, 30, 27, 32, 80, 90, 6.0, 7.5, 30, 60),
    "blackgram":    (30, 50, 55, 75, 15, 30, 25, 35, 60, 70, 6.0, 8.0, 60, 80),
    "lentil":       (10, 30, 55, 80, 15, 30, 18, 28, 20, 60, 5.5, 8.0, 35, 55),
    "pomegranate":  (10, 30, 5, 25, 35, 55, 18, 26, 85, 95, 5.5, 7.5, 100, 120),
    "banana":       (80, 120, 70, 90, 45, 55, 25, 32, 75, 85, 5.5, 7.0, 90, 120),
    "mango":        (15, 35, 15, 35, 25, 45, 27, 37, 45, 65, 5.5, 7.5, 90, 110),
    "grapes":       (15, 35, 120, 145, 195, 210, 8, 42, 78, 84, 5.5, 7.0, 60, 80),
    "watermelon":   (80, 110, 5, 20, 45, 55, 24, 28, 80, 90, 6.0, 7.0, 40, 60),
    "muskmelon":    (80, 110, 5, 20, 45, 55, 27, 32, 90, 95, 6.0, 7.0, 20, 40),
    "apple":        (15, 35, 120, 140, 195, 210, 21, 25, 90, 95, 5.5, 6.5, 100, 130),
    "orange":       (15, 30, 5, 15, 5, 15, 10, 35, 90, 95, 6.5, 8.0, 100, 120),
    "papaya":       (35, 65, 45, 70, 45, 60, 25, 42, 90, 95, 6.0, 7.0, 120, 160),
    "coconut":      (15, 30, 5, 15, 25, 40, 25, 30, 90, 95, 5.5, 6.5, 140, 180),
    "cotton":       (100, 140, 40, 60, 15, 25, 23, 27, 75, 85, 6.5, 8.0, 60, 100),
    "jute":         (60, 100, 35, 55, 35, 45, 23, 38, 75, 90, 6.0, 7.5, 150, 200),
    "coffee":       (80, 120, 15, 35, 25, 35, 23, 28, 50, 70, 6.0, 7.0, 140, 180),
}

SAMPLES_PER_CROP = 100


def _rand(lo: float, hi: float) -> float:
    """Return a random float rounded to 2 decimals."""
    return round(random.uniform(lo, hi), 2)


def generate_dataset(output_path: str) -> None:
    """Generate the crop recommendation CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"])

        for crop, profile in CROP_PROFILES.items():
            n_lo, n_hi, p_lo, p_hi, k_lo, k_hi = profile[:6]
            t_lo, t_hi, h_lo, h_hi, ph_lo, ph_hi, r_lo, r_hi = profile[6:]

            for _ in range(SAMPLES_PER_CROP):
                row = [
                    round(_rand(n_lo, n_hi)),   # N  (integer-like)
                    round(_rand(p_lo, p_hi)),   # P
                    round(_rand(k_lo, k_hi)),   # K
                    _rand(t_lo, t_hi),          # temperature
                    _rand(h_lo, h_hi),          # humidity
                    _rand(ph_lo, ph_hi),        # pH
                    _rand(r_lo, r_hi),          # rainfall
                    crop,
                ]
                writer.writerow(row)

    print(f"[OK] Dataset generated -> {output_path}  ({len(CROP_PROFILES) * SAMPLES_PER_CROP} rows)")


if __name__ == "__main__":
    here = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.dirname(here) if os.path.basename(here) == "scripts" else here
    generate_dataset(os.path.join(project_root, "data", "crop_recommendation.csv"))
