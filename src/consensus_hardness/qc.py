from __future__ import annotations

import pandas as pd


def add_s_invariant_qc_column(
    meta: pd.DataFrame,
    signature_col: str = "signature",
    s_col: str = "s_invariant",
    alternating_col: str = "is_alternating",
    output_col: str = "s_invariant_qc",
    original_col: str = "s_invariant_original",
    flag_col: str = "s_invariant_was_corrected",
) -> pd.DataFrame:
    """Preserve source ``s`` and correct the known alternating sign flip.

    A row is corrected only when it is alternating, ``sigma != s``, and
    ``abs(sigma) == abs(s)``.  This is conservative and auditable.
    """

    required = [signature_col, s_col, alternating_col]
    missing = [col for col in required if col not in meta.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = meta.copy()
    sigma = pd.to_numeric(out[signature_col], errors="coerce")
    s_value = pd.to_numeric(out[s_col], errors="coerce")
    alternating = out[alternating_col].astype(bool)

    out[original_col] = s_value
    out[output_col] = s_value

    correction = alternating & sigma.ne(s_value) & sigma.abs().eq(s_value.abs())
    out.loc[correction, output_col] = sigma.loc[correction]
    out[flag_col] = correction
    return out


def corrected_s_invariant_table(
    meta: pd.DataFrame,
    flag_col: str = "s_invariant_was_corrected",
) -> pd.DataFrame:
    if flag_col not in meta.columns:
        raise KeyError(f"Missing QC flag column: {flag_col}")

    cols = [
        "knot_id_base",
        "knot_id",
        "signature",
        "s_invariant_original",
        "s_invariant_qc",
        flag_col,
    ]
    return meta.loc[meta[flag_col], [c for c in cols if c in meta.columns]].copy()


def audit_analysis_universe(
    meta: pd.DataFrame,
    expected_n: int | None = 313_230,
    min_crossings: int = 3,
    max_crossings: int = 15,
    id_col: str = "knot_id_base",
    identity_id: str = "00_1",
    expected_s_qc_corrections: int | None = 1,
) -> dict:
    """Validate the corrected paper universe and return a compact audit."""

    required = {id_col, "number_of_crossings"}
    missing = required - set(meta.columns)
    if missing:
        raise KeyError(f"Missing universe columns: {sorted(missing)}")

    ids = meta[id_col].astype(str)
    crossings = pd.to_numeric(meta["number_of_crossings"], errors="coerce")
    duplicate_ids = int(ids.duplicated().sum())
    identity_rows = int(ids.eq(identity_id).sum())
    qc_count = (
        int(meta.get("s_invariant_was_corrected", pd.Series(False, index=meta.index)).sum())
    )

    if expected_n is not None and len(meta) != expected_n:
        raise AssertionError(f"Expected N={expected_n}, found N={len(meta)}")
    if duplicate_ids:
        raise AssertionError(f"Found {duplicate_ids} duplicated {id_col} values")
    if identity_rows:
        raise AssertionError(f"Identity {identity_id} remains in analysis universe")
    if not crossings.between(min_crossings, max_crossings).all():
        raise AssertionError("Crossing numbers fall outside the configured universe")
    if expected_s_qc_corrections is not None and qc_count != expected_s_qc_corrections:
        raise AssertionError(
            f"Expected {expected_s_qc_corrections} s-invariant QC corrections, found {qc_count}"
        )

    return {
        "n_knots": int(len(meta)),
        "n_unique_ids": int(ids.nunique()),
        "crossing_min": int(crossings.min()),
        "crossing_max": int(crossings.max()),
        "identity_rows": identity_rows,
        "duplicate_ids": duplicate_ids,
        "s_qc_corrections": qc_count,
    }
