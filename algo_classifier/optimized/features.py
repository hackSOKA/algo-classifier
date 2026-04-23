import re
import numpy as np
import pandas as pd
from pathlib import Path

from algo_classifier.optimized.config import TARGET_TAGS, EMBEDDING_MODEL

EMBEDDINGS_CACHE = Path("models/optimized/embeddings_cache.npy")

CODE_PATTERNS = {
    "import_math":        r"\bimport math\b",
    "import_fractions":   r"\bimport fractions\b",
    "import_sympy":       r"\bimport sympy\b",
    "use_gcd":            r"\bgcd\b",
    "use_mod":            r"\bmod\b",
    "use_prime":          r"\bprime\b",
    "use_factorial":      r"\bfactorial\b",
    "use_sieve":          r"\bsieve\b",
    "use_euler":          r"\beuler\b",
    "use_totient":        r"\btotient\b",
    "use_modpow":         r"\bpow\s*\(.*,.*,",
    "use_divisor":        r"\bdivisor\b",
    "use_lcm":            r"\blcm\b",
    "use_coprime":        r"\bcoprime\b",
    "use_chinese":        r"\bchinese\b",
    "use_fermat":         r"\bfermat\b",
    "import_collections": r"\bimport collections\b",
    "use_deque":          r"\bdeque\b",
    "use_bfs":            r"\bbfs\b",
    "use_dfs":            r"\bdfs\b",
    "use_graph":          r"\bgraph\b",
    "use_adjacency":      r"\badj\b",
    "use_visited":        r"\bvisited\b",
    "use_tree":           r"\btree\b",
    "use_node":           r"\bnode\b",
    "import_cmath":       r"\bimport cmath\b",
    "use_sqrt":           r"\bsqrt\b",
    "use_atan":           r"\batan\b",
    "use_cos":            r"\bcos\b",
    "use_sin":            r"\bsin\b",
    "use_pi":             r"\bpi\b",
    "use_cross":          r"\bcross\b",
    "use_dot":            r"\bdot\b",
    "use_split":          r"\bsplit\b",
    "use_join":           r"\bjoin\b",
    "use_replace":        r"\breplace\b",
    "import_re":          r"\bimport re\b",
    "use_regex":          r"\bre\.(findall|match|search|sub)\b",
    "use_random":         r"\bimport random\b",
    "use_probability":    r"\bprobabilit\b",
    "use_expected":       r"\bexpected\b",
    "use_grundy":         r"\bgrundy\b",
    "use_nim":            r"\bnim\b",
    "use_game":           r"\bgame\b",
    "use_win":            r"\bwin\b",
    "use_lose":           r"\blose\b",
    "import_heapq":       r"\bimport heapq\b",
    "import_bisect":      r"\bimport bisect\b",
    "use_heap":           r"\bheap\b",
    "use_stack":          r"\bstack\b",
    "use_dp":             r"\bdp\[",
}


def extract_code_features(source_code: str) -> dict:
    if not isinstance(source_code, str):
        return {k: 0 for k in CODE_PATTERNS}
    code_lower = source_code.lower()
    return {
        key: int(bool(re.search(pattern, code_lower)))
        for key, pattern in CODE_PATTERNS.items()
    }


def safe_str(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val)


def extract_meta_features(row: pd.Series) -> dict:
    difficulty = row.get("difficulty", 1668)
    if difficulty is None or (isinstance(difficulty, float) and np.isnan(difficulty)) or difficulty < 0:
        difficulty = 1668

    time_limit = 0.0
    raw_time = row.get("prob_desc_time_limit", "")
    if isinstance(raw_time, str):
        match = re.search(r"[\d.]+", raw_time)
        if match:
            time_limit = float(match.group())

    desc_len = len(safe_str(row.get("prob_desc_description")))
    code_len = len(safe_str(row.get("source_code")))
    input_spec_len = len(safe_str(row.get("prob_desc_input_spec")))

    return {
        "difficulty": difficulty,
        "time_limit": time_limit,
        "desc_len": desc_len,
        "code_len": code_len,
        "input_spec_len": input_spec_len,
    }


def build_text_input(row: pd.Series) -> str:
    fields = [
        safe_str(row.get("prob_desc_description")),
        safe_str(row.get("prob_desc_input_spec")),
        safe_str(row.get("prob_desc_output_spec")),
        safe_str(row.get("prob_desc_notes")),
        safe_str(row.get("source_code")),
    ]
    return " ".join(fields)


def extract_embeddings(texts: list[str] = None) -> np.ndarray:
    if EMBEDDINGS_CACHE.exists():
        print("[INFO] Embeddings chargés depuis le cache.")
        return np.load(EMBEDDINGS_CACHE)

    print("[INFO] Génération des embeddings (première fois, ~10-20 min CPU)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    X_emb = model.encode(texts, show_progress_bar=True, batch_size=64)

    EMBEDDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_CACHE, X_emb)
    print(f"[INFO] Embeddings sauvegardés → {EMBEDDINGS_CACHE}")

    return X_emb


def build_feature_matrix(df: pd.DataFrame, indices: np.ndarray = None) -> tuple[np.ndarray, list[str]]:
    rows = []
    for _, row in df.iterrows():
        meta = extract_meta_features(row)
        code = extract_code_features(row.get("source_code", ""))
        rows.append({**meta, **code})

    feature_df = pd.DataFrame(rows).fillna(0)
    feature_names = list(feature_df.columns)
    X_meta = feature_df.values.astype(np.float32)

    texts = df.apply(build_text_input, axis=1).tolist()
    X_emb_full = extract_embeddings(texts if not EMBEDDINGS_CACHE.exists() else None)

    if indices is not None:
        X_emb = X_emb_full[indices]
    else:
        X_emb = X_emb_full[:len(df)]

    feature_names = feature_names + [f"emb_{i}" for i in range(X_emb.shape[1])]
    X_meta = np.hstack([X_meta, X_emb])

    return X_meta, feature_names


if __name__ == "__main__":
    from algo_classifier.data_loader import load_dataset, add_binary_tag_columns

    df = load_dataset()
    df = add_binary_tag_columns(df)

    row = df.iloc[0]
    print("=== Text input (extrait) ===")
    print(build_text_input(row)[:300])

    print("\n=== Meta features ===")
    print(extract_meta_features(row))

    print("\n=== Code features (non nuls) ===")
    code_feats = extract_code_features(row.get("source_code", ""))
    active = {k: v for k, v in code_feats.items() if v == 1}
    print(active)

    print("\n=== Feature matrix ===")
    X, names = build_feature_matrix(df)
    print(f"Shape : {X.shape}")
    print(f"Features : {names[:10]} ... ({len(names)} total)")
