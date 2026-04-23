"""
Single-exercise prediction module for the baseline multi-label classifier.

Loads a trained baseline model and generates predictions for one JSON exercise file,
displaying per-tag probabilities and final predictions using a fixed 0.5 threshold.
"""

import argparse
import ast
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from algo_classifier.baseline.config import MODEL_DIR, TARGET_TAGS
from algo_classifier.baseline.features import build_feature_matrix, build_text_input


def load_model(model_dir=MODEL_DIR):
    """
    Load all saved artifacts from disk.
    
    Returns:
        Tuple of (clf, vectorizer, scaler)
    """
    model_dir = Path(model_dir)

    with open(model_dir / "classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(model_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return clf, vectorizer, scaler


def predict_single(row: pd.Series, clf, vectorizer, scaler) -> dict:
    """
    Generate predictions for a single exercise using the baseline model.
    
    Args:
        row: Single exercise as pd.Series (one row from DataFrame)
        clf: Trained OneVsRestClassifier with LogisticRegression
        vectorizer: Fitted TF-IDF vectorizer
        scaler: Fitted MaxAbsScaler
    
    Returns:
        Dict with structure:
        {
            "tag_name": {
                "predicted": bool,           # Tag predicted (proba >= 0.5, fixed threshold)
                "probability": float,        # Model confidence (0.0–1.0)
            },
            ...
        }
    """
    # ─── Convert single row to DataFrame (required for build_feature_matrix) ───
    df_single = pd.DataFrame([row])

    # ─── Step 1: Build TF-IDF features ───
    text = build_text_input(row)
    X_tfidf = vectorizer.transform([text])

    # ─── Step 2: Extract meta-features ───
    X_meta, _ = build_feature_matrix(df_single)
    X_meta_scaled = scaler.transform(X_meta)

    # ─── Step 3: Concatenate TF-IDF + meta ───
    X = hstack([X_tfidf, csr_matrix(X_meta_scaled)])

    # ─── Step 4: Generate predictions ───
    # LogisticRegression.predict() returns binary predictions (0 or 1)
    y_pred = clf.predict(X)[0]
    
    # ─── Step 5: Get probability estimates ───
    # LogisticRegression.predict_proba() returns P(class=1) for each tag
    y_proba = clf.predict_proba(X)[0]

    # ─── Step 6: Collect results ───
    # Baseline uses a FIXED threshold of 0.5 for all tags
    # (Optimized model optimizes per-tag thresholds)
    results = {}
    for i, tag in enumerate(TARGET_TAGS):
        results[tag] = {
            "predicted": bool(y_pred[i]),             # Already thresholded at 0.5 by predict()
            "probability": round(float(y_proba[i]), 3),  # Raw probability for display
        }

    return results


def format_output(results: dict, input_path: str) -> None:
    """
    Pretty-print prediction results for human consumption.
    
    Shows two views:
    1. Predicted tags (above 0.5 threshold) with bars
    2. All tags sorted by confidence (descending)
    
    Args:
        results: Dict from predict_single()
        input_path: Path to input JSON (for display)
    """
    print("\n" + "=" * 50)
    print(f"PREDICTION — {Path(input_path).name}")
    print("=" * 50)

    # ─── Section 1: Predicted tags only (proba >= 0.5) ───
    print("\nPredicted tags:")
    predicted = [tag for tag, v in results.items() if v["predicted"]]
    
    if predicted:
        for tag in predicted:
            proba = results[tag]["probability"]
            # Bar chart: 20 segments, height ∝ probability
            bar = "█" * int(proba * 20)
            print(f"  ✅ {tag:<16} {proba:.3f}  {bar}")
    else:
        print("  No tag predicted.")

    # ─── Section 2: All tags sorted by confidence ───
    print("\nAll scores:")
    for tag, v in sorted(results.items(), key=lambda x: -x[1]["probability"]):
        proba = v["probability"]
        # Mark with ✅ if predicted, else blank
        marker = "✅" if v["predicted"] else "  "
        bar = "█" * int(proba * 20)
        print(f"  {marker} {tag:<16} {proba:.3f}  {bar}")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    # ─── Parse command-line arguments ───
    parser = argparse.ArgumentParser(
        description="Predict tags for an algorithmic exercise (baseline model)."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to JSON exercise file")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR, help="Directory containing saved model")
    args = parser.parse_args()

    # ─── Load trained model and artifacts ───
    print("[INFO] Loading model...")
    clf, vectorizer, scaler = load_model(args.model_dir)

    # ─── Load single exercise from JSON ───
    print(f"[INFO] Loading exercise: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        exercise = json.load(f)
    row = pd.Series(exercise)

    # ─── Run prediction and time it ───
    print("[INFO] Running prediction...")
    start = time.time()
    results = predict_single(row, clf, vectorizer, scaler)
    elapsed = time.time() - start
    print(f"[INFO] Prediction time: {elapsed:.3f}s")

    # ─── Display results ───
    format_output(results, args.input)