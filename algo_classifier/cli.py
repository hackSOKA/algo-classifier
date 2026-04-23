"""
Main CLI for the algo-classifier project.

Provides a unified command-line interface with 4 sub-commands:
    train      Train the baseline or optimized model
    evaluate   Evaluate a model on the test set
    predict    Predict tags for a JSON exercise
    compare    Compare baseline vs optimized on the same test set

Design: Lazy imports inside each handler to avoid loading unnecessary dependencies.
For example, running 'train baseline' doesn't load LightGBM or embeddings.
"""

import argparse
import sys


# ═════════════════════════════════════════════════════════════════════════════
# HANDLERS — Each sub-command is isolated to avoid unnecessary imports
# ═════════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    """
    Train a model (baseline or optimized) from scratch.
    
    Workflow:
    1. Load dataset from JSON files
    2. Add binary tag columns
    3. Split into train/test (80/20, random_state=42)
    4. Build features (TF-IDF, + embeddings for optimized)
    5. Train classifier
    6. Optimize thresholds (optimized only)
    7. Save all artifacts
    """
    if args.model == "baseline":
        # ─── Lazy import: baseline-specific modules ───
        from algo_classifier.baseline.train import (
            load_model, build_feature_pipeline, build_labels, train_model, save_model
        )
        from algo_classifier.data_loader import load_dataset, add_binary_tag_columns
        from sklearn.model_selection import train_test_split

        print("[INFO] === TRAINING BASELINE ===")
        
        # ─── Load and prepare data ───
        df = load_dataset(args.data_dir)
        df = add_binary_tag_columns(df)
        
        # ─── Simple train/test split (baseline doesn't use validation set) ───
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        df_test.to_csv("data/test_set.csv", index=False)
        
        # ─── Build features (TF-IDF only) ───
        X_train, vectorizer, scaler = build_feature_pipeline(df_train)
        y_train = build_labels(df_train)
        
        # ─── Train and save ───
        clf = train_model(X_train, y_train)
        save_model(clf, vectorizer, scaler, model_dir=args.output_dir or "models/baseline")

    elif args.model == "optimized":
        # ─── Lazy import: optimized-specific modules (includes LightGBM, embeddings) ───
        import numpy as np
        from algo_classifier.optimized.train import (
            build_feature_pipeline, build_test_features,
            build_labels, train_model, optimize_thresholds, save_model
        )
        from algo_classifier.data_loader import load_dataset, add_binary_tag_columns
        from sklearn.model_selection import train_test_split

        print("[INFO] === TRAINING OPTIMIZED ===")
        
        # ─── Load and prepare data ───
        df = load_dataset(args.data_dir)
        df = add_binary_tag_columns(df)

        # ─── Reproducible train/test split by indices (important for embedding caching) ───
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=0.2, random_state=42
        )
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        df_test.to_csv("data/test_set_optimized.csv", index=False)

        # ─── Build features (TF-IDF + embeddings + meta) ───
        # Note: build_feature_matrix extracts embeddings or loads from cache
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

        # ─── Save all artifacts including indices (for reproducible evaluation) ───
        out = args.output_dir or "models/optimized"
        save_model(clf, vectorizer, scaler, thresholds, train_idx, test_idx, model_dir=out)

    else:
        print(f"[ERROR] --model must be 'baseline' or 'optimized', got '{args.model}'")
        sys.exit(1)


def cmd_evaluate(args):
    """
    Evaluate one or both models on their respective test sets.
    
    If --model both: evaluates baseline and optimized sequentially, both on
    test_set_optimized.csv (same test set for fair comparison).
    
    Outputs per-tag metrics and global metrics (F1 micro, F1 macro, Hamming Loss).
    """
    if args.model in ("baseline", "both"):
        # ─── Lazy import: baseline evaluation ───
        import ast
        import pandas as pd
        from algo_classifier.baseline.evaluate import load_model, build_X, build_y, evaluate

        print("[INFO] === EVALUATION BASELINE ===")
        clf, vectorizer, scaler = load_model(args.model_dir or "models/baseline")
        
        # ─── Load test set ───
        df_test = pd.read_csv(args.test_path or "data/test_set.csv")
        df_test["tags"] = df_test["tags"].apply(ast.literal_eval)
        
        # ─── Build features and predict ───
        X_test = build_X(df_test, vectorizer, scaler)
        y_true = build_y(df_test)
        y_pred = clf.predict(X_test)
        
        # ─── Evaluate ───
        metrics_b = evaluate(y_true, y_pred)

        # ─── Optionally export results ───
        if args.export == "json":
            import json
            with open("results_baseline.json", "w") as f:
                json.dump(metrics_b, f, indent=2)
            print("[INFO] Results exported → results_baseline.json")

    if args.model in ("optimized", "both"):
        # ─── Lazy import: optimized evaluation ───
        import ast
        import numpy as np
        import pandas as pd
        from algo_classifier.optimized.evaluate import (
            load_model, build_X, build_y, predict_with_thresholds, evaluate
        )
        from algo_classifier.optimized.config import MODEL_DIR

        print("[INFO] === EVALUATION OPTIMIZED ===")
        model_dir = args.model_dir or MODEL_DIR
        
        # ─── Load model artifacts ───
        clf, vectorizer, scaler, thresholds = load_model(model_dir)
        
        # ─── Load test set ───
        df_test = pd.read_csv(args.test_path or "data/test_set_optimized.csv")
        df_test["tags"] = df_test["tags"].apply(ast.literal_eval)

        # ─── Load cached embeddings and select test rows ───
        # Critical: use saved test_indices.npy to align with embeddings cache
        X_emb_full = np.load("models/optimized/embeddings_cache.npy")
        test_idx = np.load(f"{model_dir}/test_indices.npy")
        X_emb_test = X_emb_full[test_idx]

        # ─── Build features and predict ───
        X_test = build_X(df_test, vectorizer, scaler, X_emb=X_emb_test)
        y_true = build_y(df_test)
        y_pred = predict_with_thresholds(clf, X_test, thresholds)
        
        # ─── Evaluate ───
        metrics_o = evaluate(y_true, y_pred, label="OPTIMIZED — LightGBM + Embeddings + Thresholds")

        # ─── Optionally export results ───
        if args.export == "json":
            import json
            with open("results_optimized.json", "w") as f:
                json.dump(metrics_o, f, indent=2)
            print("[INFO] Results exported → results_optimized.json")


def cmd_predict(args):
    """
    Generate predictions for a single JSON exercise file.
    
    Loads the specified exercise JSON, runs it through the selected model,
    and prints predictions with confidence scores.
    """
    import json
    import pandas as pd

    if not args.input:
        print("[ERROR] --input is required for the predict command.")
        sys.exit(1)

    # ─── Load input JSON ───
    with open(args.input, "r", encoding="utf-8") as f:
        exercise = json.load(f)
    row = pd.Series(exercise)

    if args.model == "baseline":
        # ─── Lazy import: baseline prediction ───
        from algo_classifier.baseline.predict import load_model, predict_single, format_output
        clf, vectorizer, scaler = load_model(args.model_dir or "models/baseline")
        results = predict_single(row, clf, vectorizer, scaler)

    elif args.model == "optimized":
        # ─── Lazy import: optimized prediction ───
        from algo_classifier.optimized.predict import load_model, predict_single, format_output
        clf, vectorizer, scaler, thresholds = load_model(args.model_dir or "models/optimized")
        results = predict_single(row, clf, vectorizer, scaler, thresholds)

    else:
        print(f"[ERROR] --model must be 'baseline' or 'optimized'")
        sys.exit(1)

    # ─── Display results ───
    format_output(results, args.input)

    # ─── Optionally export to JSON ───
    if args.export == "json":
        import json
        out_path = args.input.replace(".json", "_predictions.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Predictions exported → {out_path}")


def cmd_compare(args):
    """
    Compare baseline vs optimized on the same test set.
    
    Delegates to compare.py which contains the full comparison logic.
    Both models are evaluated on test_set_optimized.csv for a fair comparison.
    """
    import subprocess
    import sys
    
    # ─── Run compare.py as subprocess ───
    result = subprocess.run(
        [sys.executable, "compare.py"],
        capture_output=False
    )
    sys.exit(result.returncode)


# ═════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER — Build CLI structure
# ═════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """
    Construct the argparse parser with 4 sub-commands and their arguments.
    
    Returns:
        ArgumentParser object with all sub-commands configured
    """
    parser = argparse.ArgumentParser(
        prog="algo-classifier",
        description="Multi-label classifier for Codeforces exercises (8 algorithm tags).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  algo-classifier train --model baseline
  algo-classifier train --model optimized --data-dir ./data
  algo-classifier evaluate --model both --export json
  algo-classifier predict --model optimized --input data/sample_1.json
  algo-classifier compare
        """,
    )
    parser.add_argument(
        "--version", action="version", version="algo-classifier 0.1.0"
    )

    # ─── Create sub-command structure ───
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # ─────────────────────────────────────────────────────────────────────────
    # SUB-COMMAND: train
    # ─────────────────────────────────────────────────────────────────────────
    p_train = subparsers.add_parser("train", help="Train a model from scratch")
    p_train.add_argument(
        "--model", choices=["baseline", "optimized"], required=True,
        help="Which model to train"
    )
    p_train.add_argument(
        "--data-dir", dest="data_dir", default="./data",
        help="Directory containing JSON exercise files (default: ./data)"
    )
    p_train.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="Directory to save model artifacts (default: models/{baseline,optimized})"
    )
    p_train.add_argument("--verbose", action="store_true", help="Verbose output")
    p_train.set_defaults(func=cmd_train)

    # ─────────────────────────────────────────────────────────────────────────
    # SUB-COMMAND: evaluate
    # ─────────────────────────────────────────────────────────────────────────
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a model on test set")
    p_eval.add_argument(
        "--model", choices=["baseline", "optimized", "both"], default="both",
        help="Which model to evaluate (default: both)"
    )
    p_eval.add_argument(
        "--test-path", dest="test_path", default=None,
        help="Path to test CSV file"
    )
    p_eval.add_argument(
        "--model-dir", dest="model_dir", default=None,
        help="Directory containing saved model artifacts"
    )
    p_eval.add_argument(
        "--export", choices=["json", "csv"], default=None,
        help="Export metrics to a file"
    )
    p_eval.add_argument("--verbose", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    # ─────────────────────────────────────────────────────────────────────────
    # SUB-COMMAND: predict
    # ─────────────────────────────────────────────────────────────────────────
    p_pred = subparsers.add_parser("predict", help="Predict tags for a JSON exercise")
    p_pred.add_argument(
        "--model", choices=["baseline", "optimized"], default="optimized",
        help="Which model to use (default: optimized)"
    )
    p_pred.add_argument(
        "--input", type=str, default=None,
        help="Path to JSON exercise file"
    )
    p_pred.add_argument(
        "--model-dir", dest="model_dir", default=None,
        help="Directory containing saved model artifacts"
    )
    p_pred.add_argument(
        "--threshold", type=float, default=None,
        help="Override global decision threshold (e.g., 0.3)"
    )
    p_pred.add_argument(
        "--export", choices=["json"], default=None,
        help="Export predictions to JSON"
    )
    p_pred.set_defaults(func=cmd_predict)

    # ─────────────────────────────────────────────────────────────────────────
    # SUB-COMMAND: compare
    # ─────────────────────────────────────────────────────────────────────────
    p_cmp = subparsers.add_parser(
        "compare", help="Compare baseline vs optimized on the same test set"
    )
    p_cmp.add_argument("--verbose", action="store_true")
    p_cmp.set_defaults(func=cmd_compare)

    return parser


def main():
    """
    Parse arguments and dispatch to the appropriate handler.
    """
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()