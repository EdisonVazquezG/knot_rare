# src/consensus_hardness/preprocessing.py

from __future__ import annotations

import pandas as pd


ID_COLS = ["knot_id", "knot_id_clean", "knot_id_base"]

META_CANDIDATES = [
    "number_of_crossings",
    "table_number",
    "is_alternating",
    "signature",
    "minimum_exponent",
    "maximum_exponent",
    "s_invariant",
]


def clean_knot_id(x) -> str | None:
    if pd.isna(x):
        return None
    return str(x).strip()


def base_knot_id(x, mirror_symbol: str = "!") -> str | None:
    if pd.isna(x):
        return None
    return str(x).strip().replace(mirror_symbol, "")


def add_knot_ids(
    df: pd.DataFrame,
    id_col: str = "knot_id",
    mirror_symbol: str = "!",
) -> pd.DataFrame:
    if id_col not in df.columns:
        raise KeyError(f"Expected id column '{id_col}' not found.")

    out = df.copy()
    out["knot_id_clean"] = out[id_col].map(clean_knot_id)
    out["knot_id_base"] = out[id_col].map(
        lambda x: base_knot_id(x, mirror_symbol=mirror_symbol)
    )
    return out


def filter_crossings(
    df: pd.DataFrame,
    min_crossings: int = 3,
    max_crossings: int = 15,
    crossing_col: str = "number_of_crossings",
) -> pd.DataFrame:
    """Keep the declared crossing-number analysis universe.

    The lower bound is deliberate: it removes the unknot/identity (``00_1``)
    before alignment, scaling, PCA, splitting, or any learned model is run.
    """
    out = df.copy()
    if crossing_col in out.columns:
        crossing = pd.to_numeric(out[crossing_col], errors="coerce")
        out = out[crossing.between(min_crossings, max_crossings)].copy()
    return out


def excluded_crossing_rows(
    df: pd.DataFrame,
    min_crossings: int = 3,
    max_crossings: int = 15,
    crossing_col: str = "number_of_crossings",
) -> pd.DataFrame:
    """Return rows excluded by the crossing-number universe definition."""

    if crossing_col not in df.columns:
        return df.iloc[0:0].copy()

    crossing = pd.to_numeric(df[crossing_col], errors="coerce")
    return df.loc[~crossing.between(min_crossings, max_crossings)].copy()


def canonicalize_mirrors_by_signature(
    df: pd.DataFrame,
    id_col: str = "knot_id",
    base_col: str = "knot_id_base",
    signature_col: str = "signature",
    mirror_symbol: str = "!",
) -> pd.DataFrame:
    """
    Keep one representative per base knot ID.

    Priority:
    1. Prefer representative with non-negative signature.
    2. If tied, prefer non-mirror ID, i.e. without mirror_symbol.
    """

    required = {id_col, base_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df.copy()
    out["_is_nonmirror"] = ~out[id_col].astype(str).str.contains(
        mirror_symbol,
        regex=False,
    )

    if signature_col in out.columns:
        out["_signature_numeric"] = pd.to_numeric(
            out[signature_col],
            errors="coerce",
        )
        out["_has_nonnegative_signature"] = out["_signature_numeric"] >= 0

        sort_cols = [
            base_col,
            "_has_nonnegative_signature",
            "_is_nonmirror",
        ]
        ascending = [True, False, False]

        drop_cols = [
            "_signature_numeric",
            "_has_nonnegative_signature",
            "_is_nonmirror",
        ]
    else:
        sort_cols = [base_col, "_is_nonmirror"]
        ascending = [True, False]
        drop_cols = ["_is_nonmirror"]

    out = (
        out.sort_values(sort_cols, ascending=ascending)
        .drop_duplicates(base_col, keep="first")
        .drop(columns=drop_cols)
        .copy()
    )

    return out


def audit_mirror_canonicalization(
    aligned_tables: dict[str, pd.DataFrame],
    clean_id_col: str = "knot_id_clean",
    base_col: str = "knot_id_base",
    mirror_symbol: str = "!",
) -> pd.DataFrame:
    rows = []

    for name, df in aligned_tables.items():
        rows.append(
            {
                "invariant": name,
                "n_rows": len(df),
                "n_unique_base_ids": df[base_col].nunique(),
                "n_selected_mirror_representatives": (
                    df[clean_id_col]
                    .astype(str)
                    .str.contains(mirror_symbol, regex=False)
                    .sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_signature_distribution(
    meta: pd.DataFrame,
    signature_col: str = "signature",
) -> pd.DataFrame:
    if signature_col not in meta.columns:
        raise KeyError(f"Column '{signature_col}' not found in metadata.")

    out = (
        meta[signature_col]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    out.columns = [signature_col, "count"]
    out["percentage"] = 100 * out["count"] / out["count"].sum()

    return out


def signature_bin(sig) -> str:
    sig = int(sig)

    if sig <= 4:
        return "0-4"
    if sig == 6:
        return "6"
    if sig == 8:
        return "8"
    if sig == 10:
        return "10"
    return "12+"


def add_signature_bin(
    meta: pd.DataFrame,
    signature_col: str = "signature",
    output_col: str = "signature_bin",
) -> pd.DataFrame:
    out = meta.copy()
    out[output_col] = out[signature_col].apply(signature_bin)
    return out
