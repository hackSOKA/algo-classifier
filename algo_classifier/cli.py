"""
CLI principal du projet algo-classifier.

Sous-commandes disponibles :
    train      Entraîne le modèle baseline ou optimized
    evaluate   Évalue un modèle sur le test set
    predict    Prédit les tags d'un exercice JSON
    compare    Compare baseline vs optimized sur le même test set
"""

import argparse
import sys


# ---------------------------------------------------------------------------
# Handlers — chaque sous-commande est isolée pour éviter les imports inutiles
# ---------------------------------------------------------------------------

def cmd_train(args):
    if args.model == "baseline":
        from algo_classifier.baseline.train import (
            load_model, build_feature_pipeline, build_labels, train_model, save_model
        )
        from algo_classifier.data_loader import load_dataset, add_binary_tag_columns
        from sklearn.model_selection import train_test_split

        print("[INFO] === ENTRAÎNEMENT BASELINE ===")
        df = load_dataset(args.data_dir)
        df = add_binary_tag_columns(df)
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        df_test.to_csv("data/test_set.csv", index=False)

        X_train, vectorizer, scaler = build_feature_pipeline(df_train)
        y_train = build_labels(df_train)
        clf = train_model(X_train, y_train)
        save_model(clf, vectorizer, scaler, model_dir=args.output_dir or "models/baseline")

    elif args.model == "optimized":
        import numpy as np
        from algo_classifier.optimized.train import (
            build_feature_pipeline, build_test_features,
            build_labels, train_model, optimize_thresholds, save_model
        )
        from algo_classifier.data_loader import load_dataset, add_binary_tag_columns
        from sklearn.model_selection import train_test_split

        print("[INFO] === ENTRAÎNEMENT OPTIMIZED ===")
        df = load_dataset(args.data_dir)
        df = add_binary_tag_columns(df)

        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=0.2, random_state=42
        )
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test  = df.iloc[test_idx].reset_index(drop=True)
        df_test.to_csv("data/test_set_optimized.csv", index=False)

        X_train, vectorizer, scaler, X_meta_full, _ = build_feature_pipeline(
            df_train, df, train_idx
        )
        y_train = build_labels(df_train)
        clf = train_model(X_train, y_train)

        X_test = build_test_features(df_test, test_idx, vectorizer, scaler, X_meta_full)
        y_test = build_labels(df_test)
        thresholds = optimize_thresholds(clf, X_test, y_test)

        out = args.output_dir or "models/optimized"
        save_model(clf, vectorizer, scaler, thresholds, train_idx, test_idx, model_dir=out)

    else:
        print(f"[ERROR] --model doit être 'baseline' ou 'optimized', pas '{args.model}'")
        sys.exit(1)


def cmd_evaluate(args):
    if args.model in ("baseline", "both"):
        import ast
        import pandas as pd
        from algo_classifier.baseline.evaluate import load_model, build_X, build_y, evaluate

        print("[INFO] === ÉVALUATION BASELINE ===")
        clf, vectorizer, scaler = load_model(args.model_dir or "models/baseline")
        df_test = pd.read_csv(args.test_path or "data/test_set.csv")
        df_test["tags"] = df_test["tags"].apply(ast.literal_eval)
        X_test = build_X(df_test, vectorizer, scaler)
        y_true = build_y(df_test)
        y_pred = clf.predict(X_test)
        metrics_b = evaluate(y_true, y_pred)

        if args.export == "json":
            import json
            with open("results_baseline.json", "w") as f:
                json.dump(metrics_b, f, indent=2)
            print("[INFO] Résultats exportés → results_baseline.json")

    if args.model in ("optimized", "both"):
        import ast
        import numpy as np
        import pandas as pd
        from algo_classifier.optimized.evaluate import (
            load_model, build_X, build_y, predict_with_thresholds, evaluate
        )
        from algo_classifier.optimized.config import MODEL_DIR

        print("[INFO] === ÉVALUATION OPTIMIZED ===")
        model_dir = args.model_dir or MODEL_DIR
        clf, vectorizer, scaler, thresholds = load_model(model_dir)
        df_test = pd.read_csv(args.test_path or "data/test_set_optimized.csv")
        df_test["tags"] = df_test["tags"].apply(ast.literal_eval)

        X_emb_full = np.load("models/optimized/embeddings_cache.npy")
        test_idx = np.load(f"{model_dir}/test_indices.npy")
        X_emb_test = X_emb_full[test_idx]

        X_test = build_X(df_test, vectorizer, scaler, X_emb=X_emb_test)
        y_true = build_y(df_test)
        y_pred = predict_with_thresholds(clf, X_test, thresholds)
        metrics_o = evaluate(y_true, y_pred, label="OPTIMIZED — LightGBM + Embeddings + Seuils")

        if args.export == "json":
            import json
            with open("results_optimized.json", "w") as f:
                json.dump(metrics_o, f, indent=2)
            print("[INFO] Résultats exportés → results_optimized.json")


def cmd_predict(args):
    import json
    import pandas as pd

    if not args.input:
        print("[ERROR] --input requis pour la commande predict.")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        exercise = json.load(f)
    row = pd.Series(exercise)

    if args.model == "baseline":
        from algo_classifier.baseline.predict import load_model, predict_single, format_output
        clf, vectorizer, scaler = load_model(args.model_dir or "models/baseline")
        results = predict_single(row, clf, vectorizer, scaler)

    elif args.model == "optimized":
        from algo_classifier.optimized.predict import load_model, predict_single, format_output
        clf, vectorizer, scaler, thresholds = load_model(args.model_dir or "models/optimized")
        results = predict_single(row, clf, vectorizer, scaler, thresholds)

    else:
        print(f"[ERROR] --model doit être 'baseline' ou 'optimized'")
        sys.exit(1)

    format_output(results, args.input)

    if args.export == "json":
        import json
        out_path = args.input.replace(".json", "_predictions.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Prédictions exportées → {out_path}")


def cmd_compare(args):
    import subprocess
    import sys
    # Délègue à compare.py qui contient déjà toute la logique
    result = subprocess.run(
        [sys.executable, "compare.py"],
        capture_output=False
    )
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Parser principal
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="algo-classifier",
        description="Classificateur multi-label d'exercices Codeforces (8 tags algo).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
exemples :
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

    subparsers = parser.add_subparsers(dest="command", metavar="<commande>")
    subparsers.required = True

    # --- train ---
    p_train = subparsers.add_parser("train", help="Entraîne un modèle")
    p_train.add_argument(
        "--model", choices=["baseline", "optimized"], required=True,
        help="Modèle à entraîner"
    )
    p_train.add_argument(
        "--data-dir", dest="data_dir", default="./data",
        help="Dossier contenant les JSON (défaut: ./data)"
    )
    p_train.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="Dossier de sauvegarde du modèle (défaut selon le modèle)"
    )
    p_train.add_argument("--verbose", action="store_true", help="Affichage détaillé")
    p_train.set_defaults(func=cmd_train)

    # --- evaluate ---
    p_eval = subparsers.add_parser("evaluate", help="Évalue un modèle sur le test set")
    p_eval.add_argument(
        "--model", choices=["baseline", "optimized", "both"], default="both",
        help="Modèle à évaluer (défaut: both)"
    )
    p_eval.add_argument(
        "--test-path", dest="test_path", default=None,
        help="Chemin vers le CSV de test"
    )
    p_eval.add_argument(
        "--model-dir", dest="model_dir", default=None,
        help="Dossier du modèle sauvegardé"
    )
    p_eval.add_argument(
        "--export", choices=["json", "csv"], default=None,
        help="Exporter les métriques dans un fichier"
    )
    p_eval.add_argument("--verbose", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    # --- predict ---
    p_pred = subparsers.add_parser("predict", help="Prédit les tags d'un exercice JSON")
    p_pred.add_argument(
        "--model", choices=["baseline", "optimized"], default="optimized",
        help="Modèle à utiliser (défaut: optimized)"
    )
    p_pred.add_argument(
        "--input", type=str, default=None,
        help="Chemin vers le fichier JSON de l'exercice"
    )
    p_pred.add_argument(
        "--model-dir", dest="model_dir", default=None,
        help="Dossier du modèle sauvegardé"
    )
    p_pred.add_argument(
        "--threshold", type=float, default=None,
        help="Override global du seuil de décision (ex: 0.3)"
    )
    p_pred.add_argument(
        "--export", choices=["json"], default=None,
        help="Exporter les prédictions en JSON"
    )
    p_pred.set_defaults(func=cmd_predict)

    # --- compare ---
    p_cmp = subparsers.add_parser(
        "compare", help="Compare baseline vs optimized sur le même test set"
    )
    p_cmp.add_argument("--verbose", action="store_true")
    p_cmp.set_defaults(func=cmd_compare)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
