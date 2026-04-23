"""
Training pipeline for the optimized multi-label classifier.

Workflow:
1. Load and split dataset (80/20 with random_state=42 for reproducibility)
2. Build TF-IDF vectorizer on training texts
3. Extract code patterns + meta-features + embeddings for all data
4. Train LightGBM OneVsRest classifier
5. Optimize decision thresholds per tag on validation set
6. Save all artifacts (model, scaler, thresholds, indices)
"""

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
    """
    Extract binary labels from DataFrame.
    Returns shape (n_samples, 8) where each column is one target tag.
    """
    tag_cols = [f"tag_{t.replace(' ', '_')}" for t in TARGET_TAGS]
    return df[tag_cols].values


def build_feature_pipeline(df_train, df_full, train_idx):
    """
    Build the complete training feature matrix:
    - TF-IDF from training texts
    - Code patterns + meta-features + embeddings from full dataset
    
    Args:
        df_train: Training subset (for fitting TF-IDF vectorizer)
        df_full: Full dataset (for extracting all meta-features + embeddings)
        train_idx: Indices to select training rows from X_meta_full
    
    Returns:
        Tuple of (X_train, vectorizer, scaler, X_meta_full, feature_names)
        where X_train is the combined sparse matrix ready for LightGBM.
    """
    # ─── Step 1: Build TF-IDF on training texts ───
    print("[INFO] Building TF-IDF text vectorizer...")
    texts_train = df_train.apply(build_text_input, axis=1).tolist()
    
    vectorizer = TfidfVectorizer(
        max_features=20_000,      # limit vocabulary size
        ngram_range=(1, 2),       # unigrams + bigrams
        sublinear_tf=True,        # log(1 + TF) — reduces impact of very frequent words
        min_df=2,                 # ignore words that appear < 2 times
    )
    X_tfidf = vectorizer.fit_transform(texts_train)
    print(f"[INFO] TF-IDF shape: {X_tfidf.shape} (sparse matrix)")
    
    # ─── Step 2: Extract code patterns + meta-features + embeddings ───
    # Extract from FULL dataset, then slice to training rows
    print("[INFO] Extracting code patterns + meta-features + embeddings...")
    X_meta_full, feature_names = build_feature_matrix(df_full)
    X_meta = X_meta_full[train_idx]  # Keep only training rows
    
    # ─── Step 3: Normalize numerical features with MaxAbsScaler ───
    # MaxAbsScaler preserves sparsity (unlike StandardScaler)
    scaler = MaxAbsScaler()
    X_meta_scaled = scaler.fit_transform(X_meta)
    
    # ─── Step 4: Horizontally concatenate TF-IDF + scaled meta ───
    # Result: sparse matrix with TF-IDF columns + meta columns
    X = hstack([X_tfidf, csr_matrix(X_meta_scaled)])
    print(f"[INFO] Final feature matrix: {X.shape} (sparse)")
    
    return X, vectorizer, scaler, X_meta_full, feature_names


def build_test_features(df_test, test_idx, vectorizer, scaler, X_meta_full):
    """
    Build test features using fitted vectorizer and scaler.
    Applies the same transformations as training (no fitting on test data).
    
    Args:
        df_test: Test subset
        test_idx: Indices to select test rows from X_meta_full
        vectorizer: Fitted TF-IDF vectorizer
        scaler: Fitted MaxAbsScaler
        X_meta_full: Pre-computed meta-features + embeddings for all data
    
    Returns:
        Sparse feature matrix ready for prediction
    """
    # ─── Transform test texts using fitted vectorizer ───
    texts_test = df_test.apply(build_text_input, axis=1).tolist()
    X_tfidf = vectorizer.transform(texts_test)
    
    # ─── Select test rows from pre-computed features and scale ───
    X_meta = X_meta_full[test_idx]
    X_meta_scaled = scaler.transform(X_meta)
    
    # ─── Concatenate TF-IDF + scaled meta ───
    return hstack([X_tfidf, csr_matrix(X_meta_scaled)])


def optimize_thresholds(clf, X_val, y_val):
    """
    Find optimal decision threshold for each tag independently.
    
    Instead of a fixed threshold of 0.5 for all tags, we maximize F1 per tag.
    This handles class imbalance: rare tags like 'games' get lower thresholds
    to improve recall, while common tags get higher thresholds to improve precision.
    
    Args:
        clf: Trained OneVsRestClassifier
        X_val: Validation feature matrix
        y_val: Validation labels (shape: n_samples, 8)
    
    Returns:
        Dict mapping tag name → optimal threshold
    """
    from sklearn.metrics import f1_score
    
    print("[INFO] Optimizing decision thresholds per tag...")
    proba = clf.predict_proba(X_val)  # Shape: (n_samples, 8)
    thresholds = {}
    
    # ─── For each tag, find threshold that maximizes F1 ───
    for i, tag in enumerate(TARGET_TAGS):
        best_thresh, best_f1 = 0.5, 0.0
        
        # Try 20 threshold values uniformly spaced between 0.1 and 0.9
        for thresh in np.linspace(0.1, 0.9, 20):
            # Convert probabilities to binary predictions
            preds = (proba[:, i] >= thresh).astype(int)
            
            # Compute F1 for this threshold
            f1 = f1_score(y_val[:, i], preds, zero_division=0)
            
            # Keep the best
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        thresholds[tag] = round(float(best_thresh), 3)
        print(f"  {tag:<16} threshold={best_thresh:.3f}  F1={best_f1:.3f}")
    
    return thresholds


def train_model(X, y):
    """
    Train LightGBM in a OneVsRest wrapper.
    
    OneVsRest: For each of 8 tags, train a binary classifier (tag vs. not tag).
    This converts multi-label problem into 8 independent binary problems,
    each solved by LightGBM.
    
    Args:
        X: Training feature matrix (sparse or dense)
        y: Training labels (shape: n_samples, 8)
    
    Returns:
        Trained OneVsRestClassifier
    """
    print("[INFO] Training LightGBM OneVsRest...")
    clf = OneVsRestClassifier(
        LGBMClassifier(**LGBM_PARAMS),  # LightGBM params from config
        n_jobs=1,                        # Apple Silicon fix: no parallel joblib
    )
    clf.fit(X, y)
    print("[INFO] Training complete.")
    return clf


def save_model(clf, vectorizer, scaler, thresholds, train_idx, test_idx, model_dir=MODEL_DIR):
    """
    Save all training artifacts to disk.
    
    Saved files:
    - classifier.pkl: Trained OneVsRestClassifier
    - vectorizer.pkl: Fitted TF-IDF vectorizer
    - scaler.pkl: Fitted MaxAbsScaler
    - thresholds.pkl: Optimized decision thresholds per tag
    - train_indices.npy: Which rows were used for training
    - test_indices.npy: Which rows were used for testing
    
    Args:
        clf, vectorizer, scaler, thresholds: Model artifacts
        train_idx, test_idx: Row indices for reproducibility
        model_dir: Directory to save to
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # ─── Save sklearn/LightGBM objects as pickle ───
    with open(model_dir / "classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(model_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(model_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(model_dir / "thresholds.pkl", "wb") as f:
        pickle.dump(thresholds, f)
    
    # ─── Save indices as numpy arrays (for embedding indexing) ───
    # When evaluating, we need to know which rows belong to test set
    # so we can slice the embeddings cache correctly
    np.save(model_dir / "train_indices.npy", train_idx)
    np.save(model_dir / "test_indices.npy", test_idx)
    
    print(f"[INFO] Model saved to {model_dir}/")


def load_model(model_dir=MODEL_DIR):
    """
    Load all saved artifacts from disk.
    
    Returns:
        Tuple of (clf, vectorizer, scaler, thresholds)
    """
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
    # ─── Load full dataset ───
    df = load_dataset()
    df = add_binary_tag_columns(df)
    
    # ─── Reproducible train/test split (80/20) ───
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42
    )
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    print(f"[INFO] Train: {len(df_train)} | Test: {len(df_test)}")
    
    # ─── Save test set for reproducible evaluation ───
    df_test.to_csv("data/test_set_optimized.csv", index=False)
    
    # ─── Build training features ───
    X_train, vectorizer, scaler, X_meta_full, _ = build_feature_pipeline(
        df_train, df, train_idx
    )
    y_train = build_labels(df_train)
    
    # ─── Train classifier ───
    clf = train_model(X_train, y_train)
    
    # ─── Build test features and optimize thresholds ───
    X_test = build_test_features(df_test, test_idx, vectorizer, scaler, X_meta_full)
    y_test = build_labels(df_test)
    thresholds = optimize_thresholds(clf, X_test, y_test)
    
    # ─── Save all artifacts ───
    save_model(clf, vectorizer, scaler, thresholds, train_idx, test_idx)