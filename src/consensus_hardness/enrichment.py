# src/consensus_hardness/enrichment.py

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


def hypergeom_log_enrichment(
    N: int,
    K: int,
    n: int,
    x: int,
) -> tuple[float, float]:
    """
    Hypergeometric log survival probability and -log10(p).

    This is more stable than using p-values directly for extremely
    enriched tails.
    """

    logp = hypergeom.logsf(x - 1, N, K, n)
    neg_log10_p = -logp / np.log(10)

    return float(logp), float(neg_log10_p)


def enrichment_for_binary_property(
    selected_df: pd.DataFrame,
    background_df: pd.DataFrame,
    mask_func,
    property_name: str,
) -> dict:
    """
    Compute enrichment of a binary property in selected_df relative to background_df.
    """

    N = len(background_df)
    K = int(mask_func(background_df).sum())

    n = len(selected_df)
    x = int(mask_func(selected_df).sum()) if n > 0 else 0

    bg_prop = K / N if N > 0 else np.nan
    selected_prop = x / n if n > 0 else np.nan
    enrichment = selected_prop / bg_prop if bg_prop and bg_prop > 0 else np.nan

    p_value = hypergeom.sf(x - 1, N, K, n) if n > 0 else np.nan

    if n > 0:
        logp, neg_log10_p = hypergeom_log_enrichment(N, K, n, x)
    else:
        logp, neg_log10_p = np.nan, np.nan

    return {
        "property": property_name,
        "background_count": int(K),
        "background_total": int(N),
        "background_prop": bg_prop,
        "selected_count": int(x),
        "selected_total": int(n),
        "selected_prop": selected_prop,
        "enrichment": enrichment,
        "hypergeom_p_value": p_value,
        "hypergeom_log_p_value": logp,
        "neg_log10_p": neg_log10_p,
    }


def standard_knot_enrichment_table(
    selected_df: pd.DataFrame,
    background_df: pd.DataFrame,
    signature_col: str = "signature",
    s_col: str = "s_invariant_qc",
    crossing_col: str = "number_of_crossings",
    alternating_col: str = "is_alternating",
) -> pd.DataFrame:
    """
    Standard enrichment table for knot consensus hard sets.
    """

    if s_col not in background_df.columns and "s_invariant" in background_df.columns:
        s_col = "s_invariant"

    rows = []

    if signature_col in background_df.columns:
        for threshold in [8, 10, 12]:
            rows.append(
                enrichment_for_binary_property(
                    selected_df,
                    background_df,
                    lambda df, t=threshold: df[signature_col] >= t,
                    f"signature >= {threshold}",
                )
            )

    if s_col in background_df.columns:
        for threshold in [8, 10, 12]:
            rows.append(
                enrichment_for_binary_property(
                    selected_df,
                    background_df,
                    lambda df, t=threshold: df[s_col] >= t,
                    f"s >= {threshold}",
                )
            )

    if crossing_col in background_df.columns:
        rows.append(
            enrichment_for_binary_property(
                selected_df,
                background_df,
                lambda df: df[crossing_col] == 15,
                "crossing number = 15",
            )
        )

    if alternating_col in background_df.columns:
        rows.append(
            enrichment_for_binary_property(
                selected_df,
                background_df,
                lambda df: df[alternating_col].astype(bool),
                "alternating",
            )
        )

    return pd.DataFrame(rows)


def threshold_enrichment_table(
    selected_df: pd.DataFrame,
    background_df: pd.DataFrame,
    targets: dict[str, list[int]],
) -> pd.DataFrame:
    """
    Generic threshold enrichment table.

    Example:
    targets = {
        "signature": [8, 10, 12],
        "s_invariant": [8, 10, 12],
    }
    """

    rows = []

    for col, thresholds in targets.items():
        if col not in background_df.columns:
            continue

        for threshold in thresholds:
            rows.append(
                enrichment_for_binary_property(
                    selected_df,
                    background_df,
                    lambda df, c=col, t=threshold: df[c] >= t,
                    f"{col} >= {threshold}",
                )
            )

    return pd.DataFrame(rows)
