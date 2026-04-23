# optimized/config.py

DATA_DIR = "./data"
MODEL_DIR = "./models/optimized"

TARGET_TAGS = [
    "math",
    "graphs",
    "strings",
    "number theory",
    "trees",
    "geometry",
    "games",
    "probabilities",
]

DEFAULT_THRESHOLDS = {tag: 0.5 for tag in TARGET_TAGS}

LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
