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
    model_dir = Path(model_dir)

    with open(model_dir / "classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(model_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return clf, vectorizer, scaler


def predict_single(row: pd.Series, clf, vectorizer, scaler) -> dict:
    df_single = pd.DataFrame([row])

    text = build_text_input(row)
    X_tfidf = vectorizer.transform([text])

    X_meta, _ = build_feature_matrix(df_single)
    X_meta_scaled = scaler.transform(X_meta)

    X = hstack([X_tfidf, csr_matrix(X_meta_scaled)])

    y_pred = clf.predict(X)[0]
    y_proba = clf.predict_proba(X)[0]

    results = {}
    for i, tag in enumerate(TARGET_TAGS):
        results[tag] = {
            "predicted": bool(y_pred[i]),
            "probability": round(float(y_proba[i]), 3),
        }

    return results


def format_output(results: dict, input_path: str) -> None:
    print("\n" + "=" * 50)
    print(f"PREDICTION — {Path(input_path).name}")
    print("=" * 50)

    print("\nPredicted tags:")
    predicted = [tag for tag, v in results.items() if v["predicted"]]
    if predicted:
        for tag in predicted:
            proba = results[tag]["probability"]
            bar = "█" * int(proba * 20)
            print(f"  ✅ {tag:<16} {proba:.3f}  {bar}")
    else:
        print("  No tag predicted.")

    print("\nAll scores:")
    for tag, v in sorted(results.items(), key=lambda x: -x[1]["probability"]):
        proba = v["probability"]
        marker = "✅" if v["predicted"] else "  "
        bar = "█" * int(proba * 20)
        print(f"  {marker} {tag:<16} {proba:.3f}  {bar}")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict tags for an algorithmic exercise (baseline)."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    args = parser.parse_args()

    print("[INFO] Loading model...")
    clf, vectorizer, scaler = load_model(args.model_dir)

    print(f"[INFO] Loading exercise: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        exercise = json.load(f)
    row = pd.Series(exercise)

    print("[INFO] Running prediction...")
    start = time.time()
    results = predict_single(row, clf, vectorizer, scaler)
    elapsed = time.time() - start
    print(f"[INFO] Prediction time: {elapsed:.3f}s")

    format_output(results, args.input)
