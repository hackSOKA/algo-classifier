# Feature engineering for the optimized model.
# Extends the baseline by adding sentence-transformer embeddings
# (cached to disk to avoid recomputing on every run).

import re
import numpy as np
import pandas as pd
from pathlib import Path

from algo_classifier.optimized.config import TARGET_TAGS, EMBEDDING_MODEL

# Cache file for sentence embeddings — loaded/saved here to avoid recomputation
EMBEDDINGS_CACHE = Path("models/optimized/embeddings_cache.npy")

# ─────────────────────────────────────────────────────────────────────────────
# REGEX PATTERNS — Binary signals for 9 algorithm families
# ─────────────────────────────────────────────────────────────────────────────
# Each pattern detects a specific concept (e.g., "gcd" for number theory).
# These capture what TF-IDF alone cannot: structural patterns in code.
CODE_PATTERNS = {
    # Number theory patterns — modular arithmetic, primes, factorials
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
    "use_modpow":         r"\bpow\s*\(.*,.*,",  # pow(base, exp, mod) — modular exponentiation
    "use_divisor":        r"\bdivisor\b",
    "use_lcm":            r"\blcm\b",
    "use_coprime":        r"\bcoprime\b",
    "use_chinese":        r"\bchinese\b",
    "use_fermat":         r"\bfermat\b",
    
    # Graph algorithms — BFS, DFS, trees, adjacency lists
    "import_collections": r"\bimport collections\b",
    "use_deque":          r"\bdeque\b",
    "use_bfs":            r"\bbfs\b",
    "use_dfs":            r"\bdfs\b",
    "use_graph":          r"\bgraph\b",
    "use_adjacency":      r"\badj\b",
    "use_visited":        r"\bvisited\b",
    "use_tree":           r"\btree\b",
    "use_node":           r"\bnode\b",
    
    # Geometry — trigonometry, coordinates, vectors
    "import_cmath":       r"\bimport cmath\b",
    "use_sqrt":           r"\bsqrt\b",
    "use_atan":           r"\batan\b",
    "use_cos":            r"\bcos\b",
    "use_sin":            r"\bsin\b",
    "use_pi":             r"\bpi\b",
    "use_cross":          r"\bcross\b",
    "use_dot":            r"\bdot\b",
    
    # String manipulation — regex, split, join
    "use_split":          r"\bsplit\b",
    "use_join":           r"\bjoin\b",
    "use_replace":        r"\breplace\b",
    "import_re":          r"\bimport re\b",
    "use_regex":          r"\bre\.(findall|match|search|sub)\b",
    
    # Probability & game theory — nim, grundy numbers, expected value
    "use_random":         r"\bimport random\b",
    "use_probability":    r"\bprobabilit\b",
    "use_expected":       r"\bexpected\b",
    "use_grundy":         r"\bgrundy\b",
    "use_nim":            r"\bnim\b",
    "use_game":           r"\bgame\b",
    "use_win":            r"\bwin\b",
    "use_lose":           r"\blose\b",
    
    # Data structures — heaps, stacks, bisect
    "import_heapq":       r"\bimport heapq\b",
    "import_bisect":      r"\bimport bisect\b",
    "use_heap":           r"\bheap\b",
    "use_stack":          r"\bstack\b",
    
    # Dynamic programming — dp array access
    "use_dp":             r"\bdp\[",
}


def extract_code_features(source_code: str) -> dict:
    """
    Return binary features for source_code: 1 if pattern found, 0 otherwise.
    These binary signals are strong predictors of algorithm tags.
    """
    # Handle missing/invalid source code gracefully
    if not isinstance(source_code, str):
        return {k: 0 for k in CODE_PATTERNS}
    
    # Case-insensitive pattern matching
    code_lower = source_code.lower()
    return {
        key: int(bool(re.search(pattern, code_lower)))
        for key, pattern in CODE_PATTERNS.items()
    }


def safe_str(val) -> str:
    """
    Convert value to string safely: returns '' for None or NaN.
    Prevents errors when text fields contain missing values.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val)


def extract_meta_features(row: pd.Series) -> dict:
    """
    Extract numerical meta-features from an exercise.
    
    Features:
    - difficulty: problem rating (1200–4000), correlates with complexity
    - time_limit: CPU time allowed (seconds), extracted from text
    - desc_len, code_len, input_spec_len: text lengths, proxy for complexity
    
    Missing/invalid values are replaced with dataset defaults.
    """
    # ─── Difficulty (problem rating) ───
    difficulty = row.get("difficulty", 1668)
    if difficulty is None or (isinstance(difficulty, float) and np.isnan(difficulty)) or difficulty < 0:
        difficulty = 1668  # replace missing/invalid with dataset median
    
    # ─── Time limit (extract from text like "2 seconds") ───
    time_limit = 0.0
    raw_time = row.get("prob_desc_time_limit", "")
    if isinstance(raw_time, str):
        match = re.search(r"[\d.]+", raw_time)
        if match:
            time_limit = float(match.group())
    
    # ─── Text field lengths (proxy for problem complexity) ───
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
    """
    Concatenate all text fields into a single string for TF-IDF vectorization
    or sentence-transformer encoding.
    
    Concatenation order: description → input spec → output spec → notes → source code
    This preserves semantic flow for embeddings.
    """
    fields = [
        safe_str(row.get("prob_desc_description")),
        safe_str(row.get("prob_desc_input_spec")),
        safe_str(row.get("prob_desc_output_spec")),
        safe_str(row.get("prob_desc_notes")),
        safe_str(row.get("source_code")),
    ]
    return " ".join(fields)


def extract_embeddings(texts: list[str] = None) -> np.ndarray:
    """
    Compute or load sentence-transformer embeddings for all exercises.
    
    Uses a .npy cache to avoid recomputing (~10-20 min on CPU) on every run.
    On first run: encodes texts with 'all-MiniLM-L6-v2' (384 dims) and saves.
    On subsequent runs: loads from cache instantly.
    
    Args:
        texts: List of text strings. Only used on first run (cache miss).
    
    Returns:
        np.ndarray: Shape (n_samples, 384) — one embedding per exercise.
    """
    # ─── Try to load from cache first (fast) ───
    if EMBEDDINGS_CACHE.exists():
        print("[INFO] Embeddings loaded from cache.")
        return np.load(EMBEDDINGS_CACHE)
    
    # ─── First run: encode texts using sentence-transformers ───
    print("[INFO] Generating embeddings (first run, ~10-20 min CPU)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    X_emb = model.encode(texts, show_progress_bar=True, batch_size=64)
    
    # ─── Save to disk for future runs ───
    EMBEDDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_CACHE, X_emb)
    print(f"[INFO] Embeddings saved → {EMBEDDINGS_CACHE}")
    
    return X_emb


def build_feature_matrix(df: pd.DataFrame, indices: np.ndarray = None) -> tuple[np.ndarray, list[str]]:
    """
    Build the complete feature matrix combining three sources:
    1. Code patterns (binary regex matches)
    2. Meta-features (difficulty, text lengths)
    3. Sentence embeddings (semantic representation)
    
    Args:
        df: Input DataFrame with exercise data.
        indices: If provided, use these indices to slice cached embeddings.
                 Used when df is a train/test subset of the full dataset.
    
    Returns:
        Tuple of (X_features, feature_names) where X_features is the combined matrix.
    """
    # ─── Extract code patterns + meta features for each exercise ───
    rows = []
    for _, row in df.iterrows():
        meta = extract_meta_features(row)
        code = extract_code_features(row.get("source_code", ""))
        rows.append({**meta, **code})
    
    # ─── Convert to numpy array and get feature names ───
    feature_df = pd.DataFrame(rows).fillna(0)
    feature_names = list(feature_df.columns)
    X_meta = feature_df.values.astype(np.float32)
    
    # ─── Generate or load sentence embeddings ───
    texts = df.apply(build_text_input, axis=1).tolist()
    X_emb_full = extract_embeddings(texts if not EMBEDDINGS_CACHE.exists() else None)
    
    # ─── Select embedding rows corresponding to this subset ───
    # When training: indices = train_idx → select embeddings for train set
    # When testing: indices = test_idx → select embeddings for test set
    if indices is not None:
        X_emb = X_emb_full[indices]
    else:
        X_emb = X_emb_full[:len(df)]
    
    # ─── Horizontally stack all features: patterns + meta + embeddings ───
    feature_names = feature_names + [f"emb_{i}" for i in range(X_emb.shape[1])]
    X_meta = np.hstack([X_meta, X_emb])
    
    return X_meta, feature_names


if __name__ == "__main__":
    from algo_classifier.data_loader import load_dataset, add_binary_tag_columns
    
    df = load_dataset()
    df = add_binary_tag_columns(df)
    
    row = df.iloc[0]
    print("=== Text input (first 300 chars) ===")
    print(build_text_input(row)[:300])
    
    print("\n=== Meta features ===")
    print(extract_meta_features(row))
    
    print("\n=== Code features (active patterns) ===")
    code_feats = extract_code_features(row.get("source_code", ""))
    active = {k: v for k, v in code_feats.items() if v == 1}
    print(active)
    
    print("\n=== Feature matrix ===")
    X, names = build_feature_matrix(df)
    print(f"Shape : {X.shape}")
    print(f"Features : {names[:10]} ... ({len(names)} total)")