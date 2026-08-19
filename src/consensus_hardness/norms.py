# src/consensus_hardness/norms.py

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.preprocessing import StandardScaler


def safe_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "_")
    )


def compute_standardized_norms(
    meta: pd.DataFrame,
    X_dict: dict[str, np.ndarray],
    fit_indices: np.ndarray | list[int] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Compute standardized L2, squared L2, and log squared norm
    for each representation.
    """

    out = meta.copy()
    norm_results = {}

    for name, X in X_dict.items():
        print("Computing standardized norms:", name)

        scaler = StandardScaler()
        if fit_indices is None:
            scaler.fit(X)
        else:
            scaler.fit(X[np.asarray(fit_indices, dtype=int)])
        Xs = scaler.transform(X).astype(np.float32)

        l2_norm = np.linalg.norm(Xs, axis=1)
        sq_norm = np.sum(Xs**2, axis=1)
        log_sq_norm = np.log1p(sq_norm)

        key = safe_name(name)

        out[f"{key}_l2_norm"] = l2_norm
        out[f"{key}_sq_norm"] = sq_norm
        out[f"{key}_log_sq_norm"] = log_sq_norm

        norm_results[name] = {
            "scaler": scaler,
            "l2_norm": l2_norm,
            "sq_norm": sq_norm,
            "log_sq_norm": log_sq_norm,
        }

    sq_norm_cols = [f"{safe_name(name)}_sq_norm" for name in X_dict.keys()]
    log_sq_norm_cols = [f"{safe_name(name)}_log_sq_norm" for name in X_dict.keys()]

    out["mean_sq_norm"] = out[sq_norm_cols].mean(axis=1)
    out["max_sq_norm"] = out[sq_norm_cols].max(axis=1)
    out["mean_log_sq_norm"] = out[log_sq_norm_cols].mean(axis=1)
    out["max_log_sq_norm"] = out[log_sq_norm_cols].max(axis=1)

    return out, norm_results


def add_consensus_indicator(
    meta: pd.DataFrame,
    consensus: set[int] | set[str],
    output_col: str,
    id_col: str = "knot_id_base",
) -> pd.DataFrame:
    out = meta.copy()
    out[output_col] = False
    if consensus:
        first = next(iter(consensus))
        if isinstance(first, (str, np.str_)):
            out.loc[out[id_col].astype(str).isin(consensus), output_col] = True
        else:
            out.iloc[list(consensus), out.columns.get_loc(output_col)] = True
    return out


def compare_selected_vs_background(
    df: pd.DataFrame,
    selected_col: str,
    value_cols: list[str],
) -> pd.DataFrame:
    """
    Compare selected vs non-selected background for numeric columns.
    """

    rows = []

    selected = df[df[selected_col]].copy()
    background = df[~df[selected_col]].copy()

    for col in value_cols:
        x = selected[col].dropna().values
        y = background[col].dropna().values

        stat, p = mannwhitneyu(x, y, alternative="two-sided")

        pooled_sd = np.sqrt((np.var(x) + np.var(y)) / 2)
        cohen_d = (np.mean(x) - np.mean(y)) / pooled_sd if pooled_sd > 0 else np.nan

        rows.append(
            {
                "feature": col,
                "selected_mean": np.mean(x),
                "background_mean": np.mean(y),
                "selected_median": np.median(x),
                "background_median": np.median(y),
                "selected_q90": np.quantile(x, 0.90),
                "background_q90": np.quantile(y, 0.90),
                "mannwhitney_p": p,
                "cohen_d": cohen_d,
            }
        )

    return pd.DataFrame(rows)


def add_pca_errors_to_metadata(
    meta: pd.DataFrame,
    pca_results: dict,
    X_dict: dict[str, np.ndarray],
    alpha: float = 0.99,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Add invariant-specific PCA SSE/NRE errors and aggregate mean/max errors.
    """

    out = meta.copy()

    for name in X_dict.keys():
        key = safe_name(name)
        out[f"{key}_pca_sse_{str(alpha).replace('.', '')}"] = pca_results[name][alpha]["sse"]
        out[f"{key}_pca_nre_{str(alpha).replace('.', '')}"] = pca_results[name][alpha]["nre"]

    suffix = str(alpha).replace(".", "")

    pca_sse_cols = [
        f"{safe_name(name)}_pca_sse_{suffix}"
        for name in X_dict.keys()
    ]

    pca_nre_cols = [
        f"{safe_name(name)}_pca_nre_{suffix}"
        for name in X_dict.keys()
    ]

    out[f"mean_pca_sse_{suffix}"] = out[pca_sse_cols].mean(axis=1)
    out[f"max_pca_sse_{suffix}"] = out[pca_sse_cols].max(axis=1)
    out[f"mean_pca_nre_{suffix}"] = out[pca_nre_cols].mean(axis=1)
    out[f"max_pca_nre_{suffix}"] = out[pca_nre_cols].max(axis=1)

    return out, pca_sse_cols, pca_nre_cols


def norm_error_correlations(
    meta_norm: pd.DataFrame,
    pca_results: dict,
    X_dict: dict[str, np.ndarray],
    alpha: float = 0.99,
    s_col: str = "s_invariant_qc",
) -> pd.DataFrame:
    """
    Spearman correlations between norm, signature, s, crossing and PCA errors.
    """

    if s_col not in meta_norm.columns:
        s_col = "s_invariant"

    rows = []

    for name in X_dict.keys():
        key = safe_name(name)

        sse = pca_results[name][alpha]["sse"]
        nre = pca_results[name][alpha]["nre"]
        sq_norm = meta_norm[f"{key}_sq_norm"].values
        log_sq_norm = meta_norm[f"{key}_log_sq_norm"].values

        rows.append(
            {
                "invariant": name,
                "spearman_sqnorm_sse": spearmanr(sq_norm, sse).statistic,
                "spearman_logsqnorm_sse": spearmanr(log_sq_norm, sse).statistic,
                "spearman_signature_sse": spearmanr(meta_norm["signature"], sse).statistic,
                "spearman_s_invariant_sse": spearmanr(meta_norm[s_col], sse).statistic,
                "spearman_crossing_sse": spearmanr(meta_norm["number_of_crossings"], sse).statistic,
                "spearman_sqnorm_nre": spearmanr(sq_norm, nre).statistic,
                "spearman_signature_nre": spearmanr(meta_norm["signature"], nre).statistic,
                "spearman_s_invariant_nre": spearmanr(meta_norm[s_col], nre).statistic,
            }
        )

    return pd.DataFrame(rows)
