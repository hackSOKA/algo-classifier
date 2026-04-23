# Paths, label set, and hyper-parameters for the optimized model.

DATA_DIR = "./data"
MODEL_DIR = "./models/optimized"

# The 8 algorithm categories the classifier predicts.
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

# Fallback thresholds (0.5) — replaced by per-tag optimized values after training.
DEFAULT_THRESHOLDS = {tag: 0.5 for tag in TARGET_TAGS}

# LightGBM hyper-parameters tuned for multi-label imbalanced classification.
LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,       # small LR with 300 trees avoids overfitting
    "num_leaves": 31,
    "min_child_samples": 10,
    "class_weight": "balanced",  # compensates for rare tags (e.g. probabilities)
    "random_state": 42,
    "n_jobs": 1,                 # 1 to avoid conflicts with OneVsRestClassifier parallelism
    "verbose": -1,
}

# Lightweight sentence-transformer (~22M params) — good quality/speed trade-off.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
