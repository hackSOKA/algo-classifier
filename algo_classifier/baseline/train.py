"""
Training pipeline for the baseline multi-label classifier.

Baseline approach: TF-IDF + LogisticRegression (OneVsRest).

Workflow:
1. Load dataset from JSON files
2. Add binary tag columns
3. Split into train/test (80/20, random_state=42)
4. Build TF-IDF vectorizer on training texts
5. Extract meta-features (simple: text lengths, difficulty)
6. Train LogisticRegression OneVsRest
7. Save all artifacts (model, vectorizer, scaler)
"""

import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MaxAbsScaler

from algo_classifier.baseline.config import DATA_DIR, MODEL_DIR, TARGET_TAGS
from algo_classifier.data_loader import load_dataset, add_binary_tag_columns
from algo_classifier.baseline.features import build_text_input, build_feature_matrix


def build_labels(df):
    """
    Extract binary labels from DataFrame.
    Returns shape (n_samples, 8) where each column is one target tag.
    """
    tag_cols = [f"tag_{t.replace(' ', '_')}" for t in TARGET_TAGS]
    return df[tag_cols].values


def build_feature_pipeline(df):
    """
    Build the training feature matrix: TF-IDF + meta-features.
    
    Args:
        df: Training DataFrame (after train_test_split)
    
    Returns:
        Tuple of (X_train, vectorizer, scaler) where X_train is the
        combined sparse matrix ready for LogisticRegression.
    """
    # ─── Step 1: Build TF-IDF on training texts ───
    print("[INFO] Building TF-IDF text vectorizer...")
    texts = df.apply(build_text_input, axis=1).tolist()

    vectorizer = TfidfVectorizer(
        max_features=20_000,      # limit vocabulary size
        ngram_range=(1, 2),       # unigrams + bigrams
        sublinear_tf=True,        # log(1 + TF) — reduces impact of frequent words
        min_df=2,                 # ignore words that appear < 2 times
    )
    X_tfidf = vectorizer.fit_transform(texts)
    print(f"[INFO] TF-IDF shape: {X_tfidf.shape} (sparse matrix)")

    # ─── Step 2: Extract simple meta-features ───
    # Baseline uses only basic features: text lengths, difficulty
    # (no embeddings, no code patterns — those are optimized-only)
    print("[INFO] Extracting numerical meta-features...")
    X_meta, feature_names = build_feature_matrix(df)

    # ─── Step 3: Normalize numerical features ───
    scaler = MaxAbsScaler()
    X_meta_scaled = scaler.fit_transform(X_meta)

    # ─── Step 4: Concatenate TF-IDF + scaled meta ───
    X = hstack([X_tfidf, csr_matrix(X_meta_scaled)])
    print(f"[INFO] Final feature matrix: {X.shape} (sparse)")

    return X, vectorizer, scaler


def train_model(X, y):
    """
    Train LogisticRegression in a OneVsRest wrapper.
    
    OneVsRest: For each of 8 tags, train a binary classifier (tag vs. not tag).
    This converts multi-label problem into 8 independent binary problems.
    
    LogisticRegression hyperparameters:
    - max_iter=1000 : sufficient iterations for convergence
    - class_weight="balanced" : handle class imbalance (rare tags get higher weight)
    - C=1.0 : inverse regularization strength (higher = less regularization)
    - solver="lbfgs" : numerical optimization method (good for small datasets)
    
    Args:
        X: Training feature matrix (sparse or dense)
        y: Training labels (shape: n_samples, 8)
    
    Returns:
        Trained OneVsRestClassifier
    """
    print("[INFO] Training LogisticRegression OneVsRest...")
    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,              # iterations for convergence
            class_weight="balanced",    # handle class imbalance
            C=1.0,                      # inverse regularization strength
            solver="lbfgs",             # numerical solver
        ),
        n_jobs=-1,                      # use all CPU cores (parallel binary classifiers)
    )
    clf.fit(X, y)
    print("[INFO] Training complete.")
    return clf


def save_model(clf, vectorizer, scaler, model_dir=MODEL_DIR):
    """
    Save all training artifacts to disk.
    
    Saved files:
    - classifier.pkl: Trained OneVsRestClassifier
    - vectorizer.pkl: Fitted TF-IDF vectorizer
    - scaler.pkl: Fitted MaxAbsScaler
    
    Note: Baseline doesn't save decision thresholds because it uses a fixed 0.5.
    (Optimized model saves per-tag thresholds in thresholds.pkl)
    
    Args:
        clf, vectorizer, scaler: Model artifacts
        model_dir: Directory to save to
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ─── Save sklearn objects as pickle ───
    with open(model_dir / "classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(model_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(model_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"[INFO] Model saved to {model_dir}/")


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


if __name__ == "__main__":
    # ─── Load full dataset ───
    df = load_dataset()
    df = add_binary_tag_columns(df)

    # ─── Reproducible train/test split (80/20) ───
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"[INFO] Train: {len(df_train)} samples | Test: {len(df_test)} samples")

    # ─── Save test set for reproducible evaluation ───
    df_test.to_csv("data/test_set.csv", index=False)
    print("[INFO] Test dataset saved → data/test_set.csv")

    # ─── Build training features ───
    X_train, vectorizer, scaler = build_feature_pipeline(df_train)
    y_train = build_labels(df_train)

    # ─── Train classifier ───
    clf = train_model(X_train, y_train)

    # ─── Save all artifacts ───
    save_model(clf, vectorizer, scaler)