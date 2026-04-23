# Feature engineering for the baseline model.
# Produces two feature groups:
#   - code features : binary flags for algorithmic patterns in source_code
#   - meta features : numerical fields (difficulty, text lengths, time limit)

import re
import numpy as np
import pandas as pd
from typing import Optional
from algo_classifier.baseline.config import TARGET_TAGS

# ------------------------------------------------------------------ #
# Algorithmic patterns detectable in source_code                     #
# ------------------------------------------------------------------ #
# Each key maps to a regex; a match → feature value 1, else 0.
CODE_PATTERNS = {
    # Math / Number theory
    "import_math":       r"\bimport math\b",
    "import_fractions":  r"\bimport fractions\b",
    "import_sympy":      r"\bimport sympy\b",
    "use_gcd":           r"\bgcd\b",
    "use_mod":           r"\bmod\b",
    "use_prime":         r"\bprime\b",
    "use_factorial":     r"\bfactorial\b",

    # Graphs / Trees
    "import_collections": r"\bimport collections\b",
    "use_deque":          r"\bdeque\b",
    "use_bfs":            r"\bbfs\b",
    "use_dfs":            r"\bdfs\b",
    "use_graph":          r"\bgraph\b",
    "use_adjacency":      r"\badj\b",
    "use_visited":        r"\bvisited\b",
    "use_tree":           r"\btree\b",
    "use_node":           r"\bnode\b",

    # Geometry
    "import_cmath":      r"\bimport cmath\b",
    "use_sqrt":          r"\bsqrt\b",
    "use_atan":          r"\batan\b",
    "use_cos":           r"\bcos\b",
    "use_sin":           r"\bsin\b",
    "use_pi":            r"\bpi\b",
    "use_cross":         r"\bcross\b",
    "use_dot":           r"\bdot\b",

    # Strings
    "use_split":         r"\bsplit\b",
    "use_join":          r"\bjoin\b",
    "use_replace":       r"\breplace\b",
    "import_re":         r"\bimport re\b",
    "use_regex":         r"\bre\.(findall|match|search|sub)\b",

    # Probabilities
    "use_random":        r"\bimport random\b",
    "use_probability":   r"\bprobabilit\b",
    "use_expected":      r"\bexpected\b",

    # Games
    "use_grundy":        r"\bgrundy\b",
    "use_nim":           r"\bnim\b",
    "use_game":          r"\bgame\b",
    "use_win":           r"\bwin\b",
    "use_lose":          r"\blose\b",

    # Data structures
    "import_heapq":      r"\bimport heapq\b",
    "import_bisect":     r"\bimport bisect\b",
    "use_heap":          r"\bheap\b",
    "use_stack":         r"\bstack\b",
    "use_dp":            r"\bdp\[",
}


def extract_code_features(source_code: str) -> dict:
    """Extract binary features from source_code. Each pattern returns 1 if found, 0 otherwise."""
    if not isinstance(source_code, str):
        return {k: 0 for k in CODE_PATTERNS}

    code_lower = source_code.lower()
    return {
        key: int(bool(re.search(pattern, code_lower)))
        for key, pattern in CODE_PATTERNS.items()
    }

def safe_str(val) -> str:
    """Return str(val), or '' for None/NaN — avoids errors when concatenating text fields."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val)

def extract_meta_features(row: pd.Series) -> dict:
    """Extract numerical/meta features from a sample."""

    # Difficulty: replace -1 and NaN with the median (1668)
    difficulty = row.get("difficulty", 1668)
    if difficulty is None or (isinstance(difficulty, float) and np.isnan(difficulty)) or difficulty < 0:
        difficulty = 1668

    # Time limit: extract the number (e.g. "2 seconds" → 2.0)
    time_limit = 0.0
    raw_time = row.get("prob_desc_time_limit", "")
    if isinstance(raw_time, str):
        match = re.search(r"[\d.]+", raw_time)
        if match:
            time_limit = float(match.group())

    # Text field lengths — proxy for problem complexity
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
    """Concatenate all text fields into a single string for TF-IDF. Null fields are replaced with an empty string."""
    fields = [
        safe_str(row.get("prob_desc_description")),
        safe_str(row.get("prob_desc_input_spec")),
        safe_str(row.get("prob_desc_output_spec")),
        safe_str(row.get("prob_desc_notes")),
        safe_str(row.get("source_code")),
    ]
    return " ".join(fields)


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Build the numerical feature matrix (code patterns + meta).
    Returns:
        - X_meta: np.ndarray of shape (n_samples, n_features)
        - feature_names: list of column names
    """
    rows = []
    for _, row in df.iterrows():
        meta = extract_meta_features(row)
        code = extract_code_features(row.get("source_code", ""))
        rows.append({**meta, **code})

    feature_df = pd.DataFrame(rows)

    # Safety check: replace any remaining NaN values with 0
    feature_df = feature_df.fillna(0)

    feature_names = list(feature_df.columns)
    return feature_df.values.astype(np.float32), feature_names


if __name__ == "__main__":
    from algo_classifier.data_loader import load_dataset, add_binary_tag_columns

    df = load_dataset()
    df = add_binary_tag_columns(df)

    # Test on a sample
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
    print(f"Features : {names}")
