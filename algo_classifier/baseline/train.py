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
    tag_cols = [f"tag_{t.replace(' ', '_')}" for t in TARGET_TAGS]
    return df[tag_cols].values


def build_feature_pipeline(df):
    print("[INFO] Construction du texte TF-IDF...")
    texts = df.apply(build_text_input, axis=1).tolist()

    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_tfidf = vectorizer.fit_transform(texts)
    print(f"[INFO] TF-IDF shape : {X_tfidf.shape}")

    print("[INFO] Construction des features numériques...")
    X_meta, feature_names = build_feature_matrix(df)

    scaler = MaxAbsScaler()
    X_meta_scaled = scaler.fit_transform(X_meta)

    X = hstack([X_tfidf, csr_matrix(X_meta_scaled)])
    print(f"[INFO] Feature matrix finale : {X.shape}")

    return X, vectorizer, scaler


def train_model(X, y):
    print("[INFO] Entraînement du modèle...")
    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
        ),
        n_jobs=-1,
    )
    clf.fit(X, y)
    print("[INFO] Entraînement terminé.")
    return clf


def save_model(clf, vectorizer, scaler, model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(model_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(model_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"[INFO] Modèle sauvegardé dans {model_dir}/")


def load_model(model_dir=MODEL_DIR):
    model_dir = Path(model_dir)

    with open(model_dir / "classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(model_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return clf, vectorizer, scaler


if __name__ == "__main__":
    df = load_dataset()
    df = add_binary_tag_columns(df)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"[INFO] Train : {len(df_train)} exemples | Test : {len(df_test)} exemples")

    df_test.to_csv("data/test_set.csv", index=False)
    print("[INFO] Dataset de test sauvegardé → data/test_set.csv")

    X_train, vectorizer, scaler = build_feature_pipeline(df_train)
    y_train = build_labels(df_train)

    clf = train_model(X_train, y_train)
    save_model(clf, vectorizer, scaler)
