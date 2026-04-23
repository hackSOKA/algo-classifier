import ast
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

from algo_classifier.baseline.config import MODEL_DIR as BASELINE_MODEL_DIR, TARGET_TAGS
from algo_classifier.baseline.features import build_text_input as baseline_text, build_feature_matrix as baseline_matrix
from algo_classifier.optimized.config import MODEL_DIR as OPTIMIZED_MODEL_DIR
from algo_classifier.optimized.features import build_text_input as optimized_text, extract_meta_features, extract_code_features
from algo_classifier.optimized.evaluate import predict_with_thresholds, evaluate, build_y


def load_baseline(model_dir=BASELINE_MODEL_DIR):
    model_dir = Path(model_dir)
    with open(model_dir / "classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(model_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return clf, vectorizer, scaler


def load_optimized(model_dir=OPTIMIZED_MODEL_DIR):
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


def build_X_baseline(df, vectorizer, scaler):
    texts = df.apply(baseline_text, axis=1).tolist()
    X_tfidf = vectorizer.transform(texts)
    X_meta, _ = baseline_matrix(df)
    X_meta_scaled = scaler.transform(X_meta)
    return hstack([X_tfidf, csr_matrix(X_meta_scaled)])


def build_X_optimized(df, vectorizer, scaler, X_emb):
    texts = df.apply(optimized_text, axis=1).tolist()
    X_tfidf = vectorizer.transform(texts)

    rows = []
    for _, row in df.iterrows():
        meta = extract_meta_features(row)
        code = extract_code_features(row.get("source_code", ""))
        rows.append({**meta, **code})
    feature_df = pd.DataFrame(rows).fillna(0)
    X_meta = feature_df.values.astype(np.float32)
    X_meta = np.hstack([X_meta, X_emb])
    X_meta_scaled = scaler.transform(X_meta)
    return hstack([X_tfidf, csr_matrix(X_meta_scaled)])


def print_comparison(metrics_b, metrics_o):
    print("\n" + "=" * 65)
    print("COMPARAISON BASELINE vs OPTIMIZED")
    print("=" * 65)
    print(f"\n{'Métrique':<30} {'Baseline':>10} {'Optimized':>10} {'Delta':>10}")
    print("-" * 65)

    for key in ["f1_micro", "f1_macro", "hamming_loss"]:
        b = metrics_b[key]
        o = metrics_o[key]
        delta = o - b
        sign = "+" if delta > 0 else ""
        arrow = "✅" if (delta < 0 and key == "hamming_loss") or (delta > 0 and key != "hamming_loss") else "❌"
        print(f"  {key:<28} {b:>10.4f} {o:>10.4f} {sign}{delta:>+9.4f} {arrow}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    TEST_PATH = "data/test_set_optimized.csv"

    print(f"[INFO] Chargement du dataset de test : {TEST_PATH}")
    df_test = pd.read_csv(TEST_PATH)
    df_test["tags"] = df_test["tags"].apply(ast.literal_eval)
    y_true = build_y(df_test)

    # --- Baseline ---
    print("\n[INFO] Évaluation BASELINE...")
    clf_b, vec_b, scaler_b = load_baseline()
    X_test_b = build_X_baseline(df_test, vec_b, scaler_b)
    y_pred_b = clf_b.predict(X_test_b)
    metrics_b = evaluate(y_true, y_pred_b, label="BASELINE — TF-IDF + LogisticRegression")

    # --- Optimized ---
    print("\n[INFO] Évaluation OPTIMIZED...")
    clf_o, vec_o, scaler_o, thresholds = load_optimized()
    X_emb_full = np.load("models/optimized/embeddings_cache.npy")
    test_idx = np.load(f"{OPTIMIZED_MODEL_DIR}/test_indices.npy")
    X_emb_test = X_emb_full[test_idx]
    X_test_o = build_X_optimized(df_test, vec_o, scaler_o, X_emb_test)
    y_pred_o = predict_with_thresholds(clf_o, X_test_o, thresholds)
    metrics_o = evaluate(y_true, y_pred_o, label="OPTIMIZED — LightGBM + Embeddings + Seuils")

    # --- Comparaison ---
    print_comparison(metrics_b, metrics_o)
