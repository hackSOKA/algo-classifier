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
    model_dir = Path(model_dir)

    with open(model_dir / "classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(model_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return clf, vectorizer, scaler


def build_X(df, vectorizer, scaler):
    texts = df.apply(build_text_input, axis=1).tolist()
    X_tfidf = vectorizer.transform(texts)

    X_meta, _ = build_feature_matrix(df)
    X_meta_scaled = scaler.transform(X_meta)

    return hstack([X_tfidf, csr_matrix(X_meta_scaled)])


def build_y(df):
    tag_cols = [f"tag_{t.replace(' ', '_')}" for t in TARGET_TAGS]
    for tag in TARGET_TAGS:
        col = f"tag_{tag.replace(' ', '_')}"
        if col not in df.columns:
            df[col] = df["tags"].apply(
                lambda tags: int(tag in tags) if isinstance(tags, list) else 0
            )
    return df[tag_cols].values


def evaluate(y_true, y_pred):
    print("\n" + "=" * 65)
    print("ÉVALUATION DU MODÈLE")
    print("=" * 65)

    print(f"\n{'Tag':<16} {'Précision':>10} {'Rappel':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)

    for i, tag in enumerate(TARGET_TAGS):
        p = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        r = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        support = int(y_true[:, i].sum())
        print(f"  {tag:<16} {p:>10.3f} {r:>10.3f} {f1:>10.3f} {support:>10}")

    print("-" * 60)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    hl = hamming_loss(y_true, y_pred)

    print(f"\n  {'F1 micro (global)':<30} {f1_micro:.3f}")
    print(f"  {'F1 macro (moyenne par tag)':<30} {f1_macro:.3f}")
    print(f"  {'Hamming Loss':<30} {hl:.4f}")
    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Évalue le modèle baseline sur un dataset de test."
    )
    parser.add_argument("--test_path", type=str, default="data/test_set.csv")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    args = parser.parse_args()

    print("[INFO] Chargement du modèle...")
    clf, vectorizer, scaler = load_model(args.model_dir)

    print(f"[INFO] Chargement du dataset de test : {args.test_path}")
    df_test = pd.read_csv(args.test_path)
    df_test["tags"] = df_test["tags"].apply(ast.literal_eval)

    print("[INFO] Construction des features...")
    X_test = build_X(df_test, vectorizer, scaler)
    y_true = build_y(df_test)

    print("[INFO] Prédiction...")
    y_pred = clf.predict(X_test)

    evaluate(y_true, y_pred)
