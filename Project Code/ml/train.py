"""
OptiCrop — Model Training & Evaluation Module

Trains five ML algorithms, compares their performance, selects the best one,
and persists the model + scaler + label-encoder as .pkl files.
"""

import os
import sys
import time
import joblib
import numpy as np
from tabulate import tabulate

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

# Add project root to path so we can import ml.preprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.preprocess import preprocess_pipeline  # noqa: E402


# ── Algorithm registry ──────────────────────────────────────────────────────
ALGORITHMS = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=2000, solver="lbfgs"),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    "Naive Bayes": GaussianNB(),
}


def train_and_evaluate(data: dict) -> list[dict]:
    """
    Train every algorithm in ALGORITHMS and return a list of result dicts.

    Each dict contains: name, model, accuracy, precision, recall, f1, time_s.
    """
    results = []

    for name, model in ALGORITHMS.items():
        print(f"\n[..] Training {name} ...")
        t0 = time.perf_counter()
        model.fit(data["X_train"], data["y_train"])
        elapsed = time.perf_counter() - t0

        y_pred = model.predict(data["X_test"])
        acc = accuracy_score(data["y_test"], y_pred)
        prec = precision_score(data["y_test"], y_pred, average="weighted", zero_division=0)
        rec = recall_score(data["y_test"], y_pred, average="weighted", zero_division=0)
        f1 = f1_score(data["y_test"], y_pred, average="weighted", zero_division=0)

        results.append({
            "name": name,
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "time_s": elapsed,
        })
        print(f"   [OK] Accuracy: {acc:.4f} | F1: {f1:.4f} | Time: {elapsed:.2f}s")

    return results


def print_comparison(results: list[dict]) -> None:
    """Pretty-print a comparison table of all models."""
    headers = ["Algorithm", "Accuracy", "Precision", "Recall", "F1-Score", "Time (s)"]
    rows = [
        [r["name"], f'{r["accuracy"]:.4f}', f'{r["precision"]:.4f}',
         f'{r["recall"]:.4f}', f'{r["f1"]:.4f}', f'{r["time_s"]:.3f}']
        for r in results
    ]
    print("\n" + "=" * 72)
    print("  MODEL COMPARISON")
    print("=" * 72)
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def select_best(results: list[dict]) -> dict:
    """Return the result dict with the highest accuracy."""
    best = max(results, key=lambda r: r["accuracy"])
    print(f"\n[BEST] Best model: {best['name']} (accuracy={best['accuracy']:.4f})")
    return best


def save_artifacts(model, scaler, label_encoder, model_dir: str) -> None:
    """Persist model, scaler, and label encoder to `model_dir`."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
    print(f"[SAVED] Artifacts saved to {model_dir}/")


def print_classification_report(data: dict, model, label_encoder) -> None:
    """Print the sklearn classification report for the best model."""
    y_pred = model.predict(data["X_test"])
    target_names = label_encoder.classes_
    print("\n[REPORT] Classification Report (Best Model):")
    print(classification_report(data["y_test"], y_pred, target_names=target_names))


# ── CLI entry point ─────────────────────────────────────────────────────────
def main() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_path = os.path.join(project_root, "data", "crop_recommendation.csv")
    model_dir = os.path.join(project_root, "models")

    # If dataset doesn't exist, generate it first
    if not os.path.isfile(dataset_path):
        print("[INFO] Dataset not found -- generating synthetic data ...")
        sys.path.insert(0, project_root)
        from generate_dataset import generate_dataset  # noqa: E402
        generate_dataset(dataset_path)

    print("[INFO] Preprocessing dataset ...")
    data = preprocess_pipeline(dataset_path)
    print(f"   Samples: {len(data['X_train'])} train / {len(data['X_test'])} test")
    print(f"   Crops  : {len(data['label_encoder'].classes_)}")

    results = train_and_evaluate(data)
    print_comparison(results)
    best = select_best(results)

    print_classification_report(data, best["model"], data["label_encoder"])
    save_artifacts(best["model"], data["scaler"], data["label_encoder"], model_dir)

    print("\n[DONE] Training complete!")


if __name__ == "__main__":
    main()
