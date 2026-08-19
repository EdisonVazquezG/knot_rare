from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from .hardsets import (
    build_hard_sets_from_fixed_results,
    consensus_dataframe,
    consensus_from_hard_sets,
)


def summarize_selected_set(
    name: str,
    selected: set[int] | set[str],
    meta: pd.DataFrame,
    s_col: str = "s_invariant_qc",
) -> dict:
    """Common outcome contract for all hard-set sensitivity analyses."""

    if s_col not in meta.columns:
        s_col = "s_invariant"
    df = consensus_dataframe(selected, meta)
    nonalt = df.loc[df["is_alternating"].eq(0)]
    n = len(df)

    return {
        "analysis": name,
        "n": n,
        "median_sigma": df["signature"].median() if n else np.nan,
        "mean_sigma": df["signature"].mean() if n else np.nan,
        "median_s": df[s_col].median() if n else np.nan,
        "mean_s": df[s_col].mean() if n else np.nan,
        "sigma_ge_8_n": int(df["signature"].ge(8).sum()),
        "sigma_ge_8_prop": float(df["signature"].ge(8).mean()) if n else np.nan,
        "sigma_ge_10_n": int(df["signature"].ge(10).sum()),
        "sigma_ge_10_prop": float(df["signature"].ge(10).mean()) if n else np.nan,
        "s_ge_10_n": int(df[s_col].ge(10).sum()),
        "s_ge_10_prop": float(df[s_col].ge(10).mean()) if n else np.nan,
        "s_ge_12_n": int(df[s_col].ge(12).sum()),
        "s_ge_12_prop": float(df[s_col].ge(12).mean()) if n else np.nan,
        "crossing15_prop": float(df["number_of_crossings"].eq(15).mean()) if n else np.nan,
        "alternating_prop": float(df["is_alternating"].mean()) if n else np.nan,
        "nonalternating_n": int(len(nonalt)),
        "nonalt_s_gt_sigma_n": int(nonalt[s_col].gt(nonalt["signature"]).sum()),
        "nonalt_s_gt_sigma_prop": (
            float(nonalt[s_col].gt(nonalt["signature"]).mean())
            if len(nonalt)
            else np.nan
        ),
    }


def leave_one_representation_out(
    hard_sets: dict[str, set[int]],
    meta: pd.DataFrame,
    s_col: str = "s_invariant_qc",
) -> tuple[dict[str, set[int]], pd.DataFrame]:
    sets = {"All": consensus_from_hard_sets(hard_sets)}
    for removed in hard_sets:
        subset = {name: values for name, values in hard_sets.items() if name != removed}
        sets[f"Without {removed}"] = consensus_from_hard_sets(subset)
    summary = pd.DataFrame(
        [summarize_selected_set(name, values, meta, s_col=s_col) for name, values in sets.items()]
    )
    return sets, summary


def tail_mass_sensitivity(
    fixed_results: dict,
    meta: pd.DataFrame,
    tail_masses: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05),
    include: tuple[str, ...] | None = None,
    score_name: str = "sse",
    s_col: str = "s_invariant_qc",
) -> tuple[dict[float, set[int]], pd.DataFrame]:
    sets = {}
    rows = []
    stable_ids = meta["knot_id_base"] if "knot_id_base" in meta.columns else None
    for tail_mass in tail_masses:
        hard_sets = build_hard_sets_from_fixed_results(
            fixed_results,
            score_name=score_name,
            tail_mass=tail_mass,
            stable_ids=stable_ids,
        )
        if include is not None:
            hard_sets = {name: hard_sets[name] for name in include}
        selected = consensus_from_hard_sets(hard_sets)
        sets[tail_mass] = selected
        row = summarize_selected_set(
            f"tail={100 * tail_mass:.1f}%", selected, meta, s_col=s_col
        )
        row.update({"tail_mass": tail_mass, "tail_percent": 100 * tail_mass})
        rows.append(row)
    return sets, pd.DataFrame(rows)


def at_least_k_consensus(
    hard_sets: dict[str, set[int]],
    names: list[str] | tuple[str, ...],
    k: int,
    n_objects: int,
) -> set[int]:
    counts = np.zeros(n_objects, dtype=np.uint8)
    for name in names:
        counts[np.fromiter(hard_sets[name], dtype=int)] += 1
    return set(map(int, np.flatnonzero(counts >= k)))


def conditional_hardness_by_norm(
    sse: np.ndarray,
    norm_sq: np.ndarray,
    n_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Return within-norm-bin SSE percentiles and integer norm-bin codes."""

    work = pd.DataFrame(
        {"sse": np.asarray(sse, dtype=float), "log_norm": np.log1p(norm_sq)}
    )
    if not np.isfinite(work.to_numpy()).all():
        raise ValueError("SSE and squared norms must be finite.")
    work["norm_bin"] = pd.qcut(
        work["log_norm"], q=n_bins, labels=False, duplicates="drop"
    ).astype(np.int32)
    work["conditional_percentile"] = work.groupby(
        "norm_bin", sort=False
    )["sse"].rank(method="average", pct=True)
    return (
        work["conditional_percentile"].to_numpy(),
        work["norm_bin"].to_numpy(),
    )


def crossfitted_norm_residual(
    sse: np.ndarray,
    norm_sq: np.ndarray,
    n_splits: int = 5,
    n_knots: int = 8,
    degree: int = 3,
    alpha: float = 1.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Out-of-fold residual of log(1+SSE) given log(1+norm squared)."""

    y = np.log1p(np.asarray(sse, dtype=float))
    X = np.log1p(np.asarray(norm_sq, dtype=float)).reshape(-1, 1)
    if not np.isfinite(y).all() or not np.isfinite(X).all():
        raise ValueError("SSE and squared norms must be finite.")

    prediction = np.full(len(y), np.nan)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(X):
        model = make_pipeline(
            SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False),
            StandardScaler(),
            Ridge(alpha=alpha),
        )
        model.fit(X[train_idx], y[train_idx])
        prediction[test_idx] = model.predict(X[test_idx])

    return {
        "residual": y - prediction,
        "prediction": prediction,
        "log_sse": y,
        "log_norm": X[:, 0],
    }


def norm_adjusted_scores(
    fixed_results: dict,
    norm_meta: pd.DataFrame,
    norm_col_map: dict[str, str],
    method: str = "conditional_percentile",
    n_bins: int = 100,
    seed: int = 42,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Build fixed-result compatible scores after adjusting for vector norm."""

    adjusted = {}
    diagnostics = []
    for name, result in fixed_results.items():
        sse = np.asarray(result["sse"])
        norm_sq = norm_meta[norm_col_map[name]].to_numpy()
        rho_before = spearmanr(norm_sq, sse).statistic
        if method == "conditional_percentile":
            score, bins = conditional_hardness_by_norm(sse, norm_sq, n_bins=n_bins)
            adjusted[name] = {"sse": score, "norm_bin": bins}
        elif method == "crossfitted_residual":
            result_adjusted = crossfitted_norm_residual(sse, norm_sq, seed=seed)
            score = result_adjusted["residual"]
            adjusted[name] = {"sse": score, **result_adjusted}
        else:
            raise ValueError(f"Unknown norm-adjustment method: {method}")
        diagnostics.append(
            {
                "invariant": name,
                "method": method,
                "spearman_norm_sse": rho_before,
                "spearman_norm_adjusted": spearmanr(norm_sq, score).statistic,
            }
        )
    return adjusted, pd.DataFrame(diagnostics)
