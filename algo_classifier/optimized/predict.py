import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from algo_classifier.optimized.config import MODEL_DIR, TARGET_TAGS
from algo_classifier.optimized.features import build_feature_matrix, build_text_input


def load_model(model_dir=MODEL_DIR):
    model_dir = Path(model_dir)

    with open(model_dir / "classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(model_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(model_dir / "thresholds.pkl", "rb") as f:
        thresholds = pickle.load(f)

    return clf, vectorizer, scaler, thresholds


def predict_single(row: pd.Series, clf, vectorizer, scaler, thresholds) -> dict:
    df_single = pd.DataFrame([row])

    text = build_text_input(row)
    X_tfidf = vectorizer.transform([text])

    X_meta, _ = build_feature_matrix(df_single)
    X_meta_scaled = scaler.transform(X_meta)

    X = hstack([X_tfidf, csr_matrix(X_meta_scaled)])

    y_proba = clf.predict_proba(X)[0]

    results = {}
    for i, tag in enumerate(TARGET_TAGS):
        proba = float(y_proba[i])
        results[tag] = {
            "predicted": proba >= thresholds[tag],
            "probability": round(proba, 3),
            "threshold": thresholds[tag],
        }

    return results


def format_output(results: dict, input_path: str) -> None:
    print("\n" + "=" * 55)
    print(f"PREDICTION — {Path(input_path).name}")
    print("=" * 55)

    print("\nPredicted tags:")
    predicted = [tag for tag, v in results.items() if v["predicted"]]
    if predicted:
        for tag in predicted:
            proba = results[tag]["probability"]
            thresh = results[tag]["threshold"]
            bar = "█" * int(proba * 20)
            print(f"  ✅ {tag:<16} {proba:.3f} (threshold={thresh:.3f})  {bar}")
    else:
        print("  No tag predicted.")

    print("\nAll scores:")
    for tag, v in sorted(results.items(), key=lambda x: -x[1]["probability"]):
        proba = v["probability"]
        thresh = v["threshold"]
        marker = "✅" if v["predicted"] else "  "
        bar = "█" * int(proba * 20)
        print(f"  {marker} {tag:<16} {proba:.3f} (threshold={thresh:.3f})  {bar}")

    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict tags for an algorithmic exercise (optimized model)."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    args = parser.parse_args()

    print("[INFO] Loading model...")
    clf, vectorizer, scaler, thresholds = load_model(args.model_dir)
    print(f"[INFO] Thresholds loaded: {thresholds}")

    print(f"[INFO] Loading exercise: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        exercise = json.load(f)
    row = pd.Series(exercise)

    print("[INFO] Running prediction...")
    start = time.time()
    results = predict_single(row, clf, vectorizer, scaler, thresholds)
    elapsed = time.time() - start
    print(f"[INFO] Prediction time: {elapsed:.3f}s")

    format_output(results, args.input)
