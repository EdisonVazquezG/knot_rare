from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DEFAULT_CK_FAMILIES = {
    "All 5": {
        "names": ("Alexander", "Jones", "HOMFLY-PT", "Theta", "Khovanov"),
        "k": 3,
    },
    "No Khovanov": {
        "names": ("Alexander", "Jones", "HOMFLY-PT", "Theta"),
        "k": 3,
    },
    "Polynomial only": {
        "names": ("Alexander", "Jones", "HOMFLY-PT"),
        "k": 2,
    },
}


def kth_largest_score(arrays: list[np.ndarray] | tuple[np.ndarray, ...], k: int) -> np.ndarray:
    """Row-wise kth largest score across a declared family of views."""

    matrix = np.column_stack([np.asarray(array, dtype=float) for array in arrays])
    if not 1 <= k <= matrix.shape[1]:
        raise ValueError("k must lie between 1 and the number of views.")
    return np.partition(matrix, matrix.shape[1] - k, axis=1)[:, -k]


def build_ck_scores(
    scores_by_view: dict[str, np.ndarray],
    families: dict[str, dict] | None = None,
) -> dict[str, np.ndarray]:
    families = families or DEFAULT_CK_FAMILIES
    output = {}
    for family, spec in families.items():
        missing = set(spec["names"]) - set(scores_by_view)
        if missing:
            raise KeyError(f"{family} is missing views: {sorted(missing)}")
        output[family] = kth_largest_score(
            [scores_by_view[name] for name in spec["names"]], int(spec["k"])
        )
    return output


def exact_top_fraction(
    score: np.ndarray,
    fraction: float = 0.01,
    eligible: np.ndarray | None = None,
    stable_ids: np.ndarray | pd.Series | None = None,
) -> tuple[np.ndarray, float]:
    """Select exactly ceil(fraction * eligible N), with deterministic ties."""

    score = np.asarray(score, dtype=float)
    if eligible is None:
        eligible = np.isfinite(score)
    else:
        eligible = np.asarray(eligible, dtype=bool) & np.isfinite(score)
    positions = np.flatnonzero(eligible)
    k = int(np.ceil(fraction * len(positions)))
    selected = np.zeros(len(score), dtype=bool)
    if k == 0:
        return selected, np.nan
    if stable_ids is None:
        order = np.argsort(-score[positions], kind="mergesort")
    else:
        ids = np.asarray(stable_ids).astype(str)[positions]
        order = np.lexsort((ids, -score[positions]))
    chosen = positions[order[:k]]
    selected[chosen] = True
    return selected, float(score[chosen[-1]])


def add_delta(
    meta: pd.DataFrame,
    s_col: str = "s_invariant_qc",
    signature_col: str = "signature",
    output_col: str = "delta_s_minus_sigma",
) -> pd.DataFrame:
    if s_col not in meta.columns:
        s_col = "s_invariant"
    out = meta.copy()
    out[output_col] = out[s_col] - out[signature_col]
    return out


def exact_stratum_expected_mean(
    outcome: np.ndarray,
    selected: np.ndarray,
    eligible: np.ndarray,
    strata: pd.DataFrame,
) -> float:
    """Expected selected mean under exact stratum-count matching."""

    frame = strata.copy()
    frame["outcome"] = outcome
    frame["selected"] = selected
    frame["eligible"] = eligible
    frame = frame.loc[frame["eligible"]].copy()
    grouped = frame.groupby(list(strata.columns), dropna=False)
    weights = grouped["selected"].sum()
    means = grouped["outcome"].mean()
    total = weights.sum()
    return float(np.average(means, weights=weights)) if total else np.nan


def continuous_association_summary(
    ck_scores: dict[str, np.ndarray],
    meta: pd.DataFrame,
    top_fraction: float = 0.01,
    s_col: str = "s_invariant_qc",
    nonalternating_only: bool = True,
    matching_cols: tuple[str, ...] = ("number_of_crossings", "signature"),
) -> pd.DataFrame:
    """Predeclared C_k association with Delta=s-sigma."""

    work = add_delta(meta, s_col=s_col)
    delta = work["delta_s_minus_sigma"].to_numpy(dtype=float)
    eligible = np.isfinite(delta)
    if nonalternating_only:
        eligible &= work["is_alternating"].eq(0).to_numpy()
    stable_ids = work.get("knot_id_base")
    rows = []
    for family, score in ck_scores.items():
        score = np.asarray(score, dtype=float)
        valid = eligible & np.isfinite(score)
        selected, threshold = exact_top_fraction(
            score, fraction=top_fraction, eligible=valid, stable_ids=stable_ids
        )
        observed = delta[selected]
        expected = exact_stratum_expected_mean(
            delta, selected, valid, work.loc[:, list(matching_cols)]
        )
        rows.append(
            {
                "family": family,
                "n_valid": int(valid.sum()),
                "n_selected": int(selected.sum()),
                "top_fraction": top_fraction,
                "threshold": threshold,
                "spearman_ck_delta": float(spearmanr(score[valid], delta[valid]).statistic),
                "selected_mean_delta": float(observed.mean()) if len(observed) else np.nan,
                "selected_median_delta": float(np.median(observed)) if len(observed) else np.nan,
                "matched_expected_mean_delta": expected,
                "matched_mean_delta_effect": (
                    float(observed.mean() - expected) if len(observed) else np.nan
                ),
                "delta_ge_2_prop": float((observed >= 2).mean()) if len(observed) else np.nan,
                "delta_ge_4_prop": float((observed >= 4).mean()) if len(observed) else np.nan,
            }
        )
    return pd.DataFrame(rows)
