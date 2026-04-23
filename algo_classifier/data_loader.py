import json
import os
from pathlib import Path
from collections import Counter
from typing import Optional
import pandas as pd

from algo_classifier.baseline.config import DATA_DIR, TARGET_TAGS


def load_dataset(data_dir: str = DATA_DIR, limit: Optional[int] = None) -> pd.DataFrame:
    data_dir = Path(data_dir)
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"Aucun fichier JSON trouvé dans : {data_dir}")
    if limit:
        json_files = json_files[:limit]
    records = []
    errors = []
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
                record["_filename"] = path.name
                records.append(record)
        except (json.JSONDecodeError, OSError) as e:
            errors.append((path.name, str(e)))
    if errors:
        print(f"[WARNING] {len(errors)} fichiers ignorés.")
    df = pd.DataFrame(records)
    print(f"[INFO] {len(df)} exemples chargés depuis {data_dir}")
    return df


def add_binary_tag_columns(df: pd.DataFrame) -> pd.DataFrame:
    for tag in TARGET_TAGS:
        col = f"tag_{tag.replace(' ', '_')}"
        df[col] = df["tags"].apply(
            lambda tags: int(tag in tags) if isinstance(tags, list) else 0
        )
    return df


def explore(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("EXPLORATION DU DATASET")
    print("=" * 60)
    print(f"\n{'Dimensions':.<40} {df.shape[0]} exemples × {df.shape[1]} colonnes")
    print("\n--- Valeurs manquantes (features clés) ---")
    key_cols = [
        "prob_desc_description",
        "prob_desc_input_spec",
        "prob_desc_output_spec",
        "prob_desc_notes",
        "source_code",
        "difficulty",
        "tags",
    ]
    for col in key_cols:
        if col in df.columns:
            n_null = df[col].isna().sum()
            pct = 100 * n_null / len(df)
            print(f"  {col:<40} {n_null:>5} nulls ({pct:.1f}%)")
    print("\n--- Distribution de la difficulté ---")
    if "difficulty" in df.columns:
        stats = df["difficulty"].describe()
        print(f"  min={stats['min']:.0f}  mean={stats['mean']:.0f}  max={stats['max']:.0f}")
    print("\n--- Distribution des 8 tags cibles ---")
    tag_cols = [f"tag_{t.replace(' ', '_')}" for t in TARGET_TAGS]
    available = [c for c in tag_cols if c in df.columns]
    if available:
        tag_counts = df[available].sum().sort_values(ascending=False)
        for col, count in tag_counts.items():
            tag_name = col.replace("tag_", "").replace("_", " ")
            bar = "█" * int(40 * count / len(df))
            pct = 100 * count / len(df)
            print(f"  {tag_name:<16} {count:>5} ({pct:5.1f}%)  {bar}")
    print("\n--- Top 20 tags globaux ---")
    if "tags" in df.columns:
        all_tags = [tag for tags in df["tags"].dropna() for tag in tags]
        for tag, count in Counter(all_tags).most_common(20):
            marker = " ← CIBLE" if tag in TARGET_TAGS else ""
            print(f"  {tag:<30} {count:>5}{marker}")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    df = load_dataset()
    df = add_binary_tag_columns(df)
    explore(df)
