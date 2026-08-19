# src/consensus_hardness/concordance.py

from __future__ import annotations

import pandas as pd


def add_concordance_columns(
    meta: pd.DataFrame,
    signature_col: str = "signature",
    s_col: str | None = None,
    alternating_col: str = "is_alternating",
) -> pd.DataFrame:
    """
    Add standard concordance-related columns:
    - abs_signature
    - abs_s_invariant
    - sigma_minus_s
    - sigma_eq_s
    - is_alt_bool
    """

    if s_col is None:
        s_col = "s_invariant_qc" if "s_invariant_qc" in meta.columns else "s_invariant"
    required = [signature_col, s_col, alternating_col]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = meta.copy()

    out[signature_col] = pd.to_numeric(out[signature_col], errors="coerce")
    out[s_col] = pd.to_numeric(out[s_col], errors="coerce")

    out["abs_signature"] = out[signature_col].abs()
    out["abs_s_invariant"] = out[s_col].abs()
    out["sigma_minus_s"] = out[signature_col] - out[s_col]
    out["sigma_eq_s"] = out[signature_col] == out[s_col]
    out["is_alt_bool"] = out[alternating_col].astype(bool)

    return out


def sigma_s_joint_counts(
    meta: pd.DataFrame,
    signature_col: str = "signature",
    s_col: str | None = None,
) -> pd.DataFrame:
    """
    Return counts of (signature, s_invariant, sigma_minus_s).
    """

    if s_col is None:
        s_col = "s_invariant_qc" if "s_invariant_qc" in meta.columns else "s_invariant"
    required = [signature_col, s_col, "sigma_minus_s"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise KeyError(
            f"Missing columns {missing}. Run add_concordance_columns first."
        )

    out = (
        meta[[signature_col, s_col, "sigma_minus_s"]]
        .value_counts()
        .sort_index()
        .reset_index(name="count")
    )

    return out


def summarize_sigma_s_agreement(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize sigma=s agreement overall and by alternating status.
    """

    required = ["sigma_eq_s", "is_alt_bool"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise KeyError(
            f"Missing columns {missing}. Run add_concordance_columns first."
        )

    rows = []

    groups = {
        "all": meta,
        "alternating": meta[meta["is_alt_bool"]],
        "non_alternating": meta[~meta["is_alt_bool"]],
    }

    for name, df in groups.items():
        n = len(df)
        n_equal = int(df["sigma_eq_s"].sum())
        n_not_equal = int((~df["sigma_eq_s"]).sum())

        rows.append(
            {
                "group": name,
                "n": n,
                "sigma_eq_s": n_equal,
                "sigma_not_eq_s": n_not_equal,
                "sigma_eq_s_prop": n_equal / n if n > 0 else float("nan"),
                "sigma_not_eq_s_prop": n_not_equal / n if n > 0 else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def find_alternating_sigma_s_mismatches(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Return alternating knots where sigma != s.
    """

    required = [
        "knot_id_base",
        "knot_id",
        "number_of_crossings",
        "is_alternating",
        "signature",
        "s_invariant",
        "sigma_minus_s",
        "sigma_eq_s",
        "is_alt_bool",
    ]

    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise KeyError(
            f"Missing columns {missing}. Run add_concordance_columns first."
        )

    cols = [
        "knot_id_base",
        "knot_id",
        "number_of_crossings",
        "is_alternating",
        "signature",
        "s_invariant",
        "sigma_minus_s",
    ]

    return meta[
        (meta["is_alt_bool"]) & (~meta["sigma_eq_s"])
    ][cols].copy()


def inspect_knot_by_base_id(
    meta: pd.DataFrame,
    knot_base_id: str,
    mirror_symbol: str = "!",
) -> pd.DataFrame:
    """
    Inspect a knot by base ID, ignoring mirror symbol in knot_id.
    """

    required = ["knot_id_base", "knot_id"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    mask = (
        (meta["knot_id_base"] == knot_base_id)
        | (
            meta["knot_id"]
            .astype(str)
            .str.replace(mirror_symbol, "", regex=False)
            == knot_base_id
        )
    )

    preferred_cols = [
        "knot_id_base",
        "knot_id",
        "number_of_crossings",
        "is_alternating",
        "signature",
        "s_invariant",
        "sigma_minus_s",
        "abs_signature",
        "abs_s_invariant",
        "sigma_eq_s",
    ]

    cols = [c for c in preferred_cols if c in meta.columns]

    return meta.loc[mask, cols].copy()


def add_s_invariant_qc_column(
    meta: pd.DataFrame,
    signature_col: str = "signature",
    s_col: str = "s_invariant",
    output_col: str = "s_invariant_qc",
    original_col: str = "s_invariant_original",
    flag_col: str = "s_invariant_was_corrected",
) -> pd.DataFrame:
    """
    Create a QC version of the Rasmussen s-invariant.

    Correction rule:
    If a knot is alternating, sigma != s, and |sigma| == |s|,
    then replace s_qc by sigma.

    This preserves the original s-invariant in original_col.
    """

    required = [
        signature_col,
        s_col,
        "is_alt_bool",
    ]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise KeyError(
            f"Missing columns {missing}. Run add_concordance_columns first."
        )

    out = meta.copy()

    out[original_col] = out[s_col]
    out[output_col] = out[s_col]

    mask_correct_sign_flip = (
        out["is_alt_bool"]
        & (out[signature_col] != out[s_col])
        & (out[signature_col].abs() == out[s_col].abs())
    )

    out.loc[mask_correct_sign_flip, output_col] = out.loc[
        mask_correct_sign_flip,
        signature_col,
    ]

    out[flag_col] = mask_correct_sign_flip

    return out


def corrected_s_invariant_table(
    meta: pd.DataFrame,
    flag_col: str = "s_invariant_was_corrected",
) -> pd.DataFrame:
    """
    Return rows where the QC s-invariant was corrected.
    """

    if flag_col not in meta.columns:
        raise KeyError(
            f"Column '{flag_col}' not found. Run add_s_invariant_qc_column first."
        )

    cols = [
        "knot_id_base",
        "knot_id",
        "signature",
        "s_invariant_original",
        "s_invariant_qc",
        "s_invariant_was_corrected",
    ]

    cols = [c for c in cols if c in meta.columns]

    return meta.loc[meta[flag_col], cols].copy()


def build_concordance_qc_report(meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Convenience wrapper for all sigma/s QC summaries.
    """

    meta_qc = meta.copy()
    if "s_invariant_qc" not in meta_qc.columns:
        meta_qc = add_concordance_columns(meta_qc, s_col="s_invariant")
        meta_qc = add_s_invariant_qc_column(meta_qc)
    meta_qc = add_concordance_columns(meta_qc, s_col="s_invariant_qc")

    return {
        "meta": meta_qc,
        "agreement_summary": summarize_sigma_s_agreement(meta_qc),
        "joint_counts": sigma_s_joint_counts(meta_qc),
        "alternating_mismatches": find_alternating_sigma_s_mismatches(meta_qc),
        "corrected_s_rows": corrected_s_invariant_table(meta_qc),
    }


def add_sigma_minus_s(
    df: pd.DataFrame,
    signature_col: str = "signature",
    s_col: str | None = None,
    output_col: str = "sigma_minus_s",
) -> pd.DataFrame:
    """
    Add sigma - s column to any dataframe.
    """

    if s_col is None:
        s_col = "s_invariant_qc" if "s_invariant_qc" in df.columns else "s_invariant"
    out = df.copy()
    out[output_col] = out[signature_col] - out[s_col]
    return out


def summarize_sigma_minus_s_by_alternating(
    df: pd.DataFrame,
    sigma_minus_s_col: str = "sigma_minus_s",
    alternating_col: str = "is_alternating",
) -> dict[str, pd.DataFrame]:
    """
    Summarize sigma-s globally and stratified by alternating status.
    """

    if sigma_minus_s_col not in df.columns:
        df = add_sigma_minus_s(df)

    overall = (
        df[sigma_minus_s_col]
        .value_counts()
        .sort_index()
        .reset_index(name="count")
        .rename(columns={"index": sigma_minus_s_col})
    )

    by_alt = (
        df.groupby(alternating_col)[sigma_minus_s_col]
        .value_counts()
        .sort_index()
        .reset_index(name="count")
    )

    return {
        "overall": overall,
        "by_alternating": by_alt,
    }


def summarize_non_alternating_concordance(
    df: pd.DataFrame,
    alternating_col: str = "is_alternating",
    s_col: str | None = None,
) -> dict[str, pd.DataFrame | int]:
    """
    Summaries of signature, s-invariant and sigma-s in the non-alternating subset.
    """

    if s_col is None:
        s_col = "s_invariant_qc" if "s_invariant_qc" in df.columns else "s_invariant"
    if "sigma_minus_s" not in df.columns:
        df = add_sigma_minus_s(df)

    non_alt = df[df[alternating_col] == 0].copy()

    return {
        "n_non_alternating": len(non_alt),
        "signature_distribution": (
            non_alt["signature"]
            .value_counts()
            .sort_index()
            .reset_index(name="count")
            .rename(columns={"index": "signature"})
        ),
        "s_invariant_distribution": (
            non_alt[s_col]
            .value_counts()
            .sort_index()
            .reset_index(name="count")
            .rename(columns={"index": s_col})
        ),
        "sigma_minus_s_distribution": (
            non_alt["sigma_minus_s"]
            .value_counts()
            .sort_index()
            .reset_index(name="count")
            .rename(columns={"index": "sigma_minus_s"})
        ),
    }
