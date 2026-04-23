"""
Evaluation pipeline for the baseline multi-label classifier.

Loads a trained baseline model and evaluates it on a test set, computing:
- Per-tag metrics (Precision, Recall, F1)
- Global metrics (F1 micro, F1 macro, Hamming Loss)
"""

import argparse
import ast
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

from algo_classifier.baseline.config import MODEL_DIR, TARGET_TAGS
from algo_classifier.baseline.features import build_text_input, build_feature_matrix


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


def build_X(df, vectorizer, scaler):
    """
    Build test features using fitted vectorizer and scaler.
    
    Args:
        df: Test DataFrame
        vectorizer: Fitted TF-IDF vectorizer
        scaler: Fitted MaxAbsScaler
    
    Returns:
        Sparse feature matrix ready for prediction
    """
    # ─── Transform test texts using fitted vectorizer ───
    texts = df.apply(build_text_input, axis=1).tolist()
    X_tfidf = vectorizer.transform(texts)

    # ─── Extract meta-features and scale ───
    X_meta, _ = build_feature_matrix(df)
    X_meta_scaled = scaler.transform(X_meta)

    # ─── Concatenate TF-IDF + scaled meta ───
    return hstack([X_tfidf, csr_matrix(X_meta_scaled)])


def build_y(df):
    """
    Extract binary labels from test DataFrame.
    Creates missing tag columns if needed (for robustness).
    
    Returns:
        Array of shape (n_samples, 8) with binary labels
    """
    tag_cols = [f"tag_{t.replace(' ', '_')}" for t in TARGET_TAGS]
    
    # ─── Ensure all tag columns exist ───
    for tag in TARGET_TAGS:
        col = f"tag_{tag.replace(' ', '_')}"
        if col not in df.columns:
            df[col] = df["tags"].apply(
                lambda tags: int(tag in tags) if isinstance(tags, list) else 0
            )
    
    return df[tag_cols].values


def evaluate(y_true, y_pred):
    """
    Compute and print evaluation metrics.
    
    Per-tag metrics:
    - Precision: Of positive predictions, how many are correct?
    - Recall: Of true positives, how many did we find?
    - F1: Harmonic mean of precision and recall
    - Support: Number of positive examples
    
    Global metrics:
    - F1 micro: Treats all predictions equally (favors common tags)
    - F1 macro: Average F1 across tags (treats each tag equally)
    - Hamming Loss: Fraction of incorrect predictions
    
    Args:
        y_true: Ground truth labels (shape: n_samples, 8)
        y_pred: Predicted labels (shape: n_samples, 8)
    """
    print("\n" + "=" * 65)
    print("MODEL EVALUATION")
    print("=" * 65)

    # ─── Per-tag metrics ───
    print(f"\n{'Tag':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)

    for i, tag in enumerate(TARGET_TAGS):
        p = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        r = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        support = int(y_true[:, i].sum())
        print(f"  {tag:<16} {p:>10.3f} {r:>10.3f} {f1:>10.3f} {support:>10}")

    print("-" * 60)
    
    # ─── Global metrics ───
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    hl = hamming_loss(y_true, y_pred)

    print(f"\n  {'F1 micro (global)':<30} {f1_micro:.3f}")
    print(f"  {'F1 macro (avg per tag)':<30} {f1_macro:.3f}")
    print(f"  {'Hamming Loss':<30} {hl:.4f}")
    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    # ─── Parse command-line arguments ───
    parser = argparse.ArgumentParser(
        description="Evaluate the baseline model on a test dataset."
    )
    parser.add_argument("--test_path", type=str, default="data/test_set.csv")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    args = parser.parse_args()

    # ─── Load trained model and artifacts ───
    print("[INFO] Loading model...")
    clf, vectorizer, scaler = load_model(args.model_dir)

    # ─── Load test set ───
    print(f"[INFO] Loading test dataset: {args.test_path}")
    df_test = pd.read_csv(args.test_path)
    df_test["tags"] = df_test["tags"].apply(ast.literal_eval)

    # ─── Build features ───
    print("[INFO] Building test features...")
    X_test = build_X(df_test, vectorizer, scaler)
    y_true = build_y(df_test)

    # ─── Generate predictions ───
    print("[INFO] Generating predictions...")
    y_pred = clf.predict(X_test)

    # ─── Evaluate ───
    evaluate(y_true, y_pred)