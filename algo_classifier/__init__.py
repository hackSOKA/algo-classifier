"""
algo-classifier: multi-label tag predictor for Codeforces exercises.

Predicts up to 8 algorithm categories (math, graphs, strings, …) from an
exercise's description and source code.

Sub-packages:
    baseline   — TF-IDF + LogisticRegression         (F1 macro ≈ 0.621)
    optimized  — LightGBM + sentence embeddings
                 + per-tag decision thresholds        (F1 macro ≈ 0.657)
"""

__version__ = "0.1.0"
