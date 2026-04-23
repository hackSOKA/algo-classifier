import pickle
import numpy as np
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MaxAbsScaler
from lightgbm import LGBMClassifier

from algo_classifier.optimized.config import DATA_DIR, MODEL_DIR, TARGET_TAGS, LGBM_PARAMS
from algo_classifier.data_loader import load_dataset, add_binary_tag_columns
from algo_classifier.optimized.features import build_text_input, build_feature_matrix


def build_labels(df):
    tag_cols = [f"tag_{t.replace(' ', '_')}" for t in TARGET_TAGS]
    return df[tag_cols].values


def build_feature_pipeline(df_train, df_full, train_idx):
    print("[INFO] Building TF-IDF text...")
    texts_train = df_train.apply(build_text_input, axis=1).tolist()

    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_tfidf = vectorizer.fit_transform(texts_train)
    print(f"[INFO] TF-IDF shape : {X_tfidf.shape}")

    print("[INFO] Building numerical features + embeddings...")
    X_meta_full, feature_names = build_feature_matrix(df_full)
    X_meta = X_meta_full[train_idx]

    scaler = MaxAbsScaler()
    X_meta_scaled = scaler.fit_transform(X_meta)

    X = hstack([X_tfidf, csr_matrix(X_meta_scaled)])
    print(f"[INFO] Final feature matrix: {X.shape}")

    return X, vectorizer, scaler, X_meta_full, feature_names


def build_test_features(df_test, test_idx, vectorizer, scaler, X_meta_full):
    texts_test = df_test.apply(build_text_input, axis=1).tolist()
    X_tfidf = vectorizer.transform(texts_test)

    X_meta = X_meta_full[test_idx]
    X_meta_scaled = scaler.transform(X_meta)

    return hstack([X_tfidf, csr_matrix(X_meta_scaled)])


def optimize_thresholds(clf, X_val, y_val):
    from sklearn.metrics import f1_score

    print("[INFO] Optimizing thresholds per tag...")
    proba = clf.predict_proba(X_val)
    thresholds = {}

    for i, tag in enumerate(TARGET_TAGS):
        best_thresh, best_f1 = 0.5, 0.0
        for thresh in np.linspace(0.1, 0.9, 20):
            preds = (proba[:, i] >= thresh).astype(int)
            f1 = f1_score(y_val[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        thresholds[tag] = round(float(best_thresh), 3)
        print(f"  {tag:<16} threshold={best_thresh:.3f}  F1={best_f1:.3f}")

    return thresholds


def train_model(X, y):
    print("[INFO] Training LightGBM OneVsRest...")
    clf = OneVsRestClassifier(
        LGBMClassifier(**LGBM_PARAMS),
        n_jobs=1,
    )
    clf.fit(X, y)
    print("[INFO] Training complete.")
    return clf


def save_model(clf, vectorizer, scaler, thresholds, train_idx, test_idx, model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(model_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(model_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(model_dir / "thresholds.pkl", "wb") as f:
        pickle.dump(thresholds, f)

    np.save(model_dir / "train_indices.npy", train_idx)
    np.save(model_dir / "test_indices.npy", test_idx)

    print(f"[INFO] Model saved to {model_dir}/")


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


if __name__ == "__main__":
    df = load_dataset()
    df = add_binary_tag_columns(df)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42
    )
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)
    print(f"[INFO] Train: {len(df_train)} | Test: {len(df_test)}")

    df_test.to_csv("data/test_set_optimized.csv", index=False)

    X_train, vectorizer, scaler, X_meta_full, _ = build_feature_pipeline(
        df_train, df, train_idx
    )
    y_train = build_labels(df_train)

    clf = train_model(X_train, y_train)

    X_test = build_test_features(df_test, test_idx, vectorizer, scaler, X_meta_full)
    y_test = build_labels(df_test)
    thresholds = optimize_thresholds(clf, X_test, y_test)

    save_model(clf, vectorizer, scaler, thresholds, train_idx, test_idx)
