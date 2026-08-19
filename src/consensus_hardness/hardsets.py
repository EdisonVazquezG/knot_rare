# src/consensus_hardness/hardsets.py

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def top_tail_indices(
    scores: np.ndarray,
    tail_mass: float = 0.01,
    stable_ids: np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    """
    Return indices of top tail_mass fraction by score.
    """

    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = int(np.ceil(tail_mass * n))

    if k <= 0:
        raise ValueError("tail_mass too small; selected tail size is zero.")

    if not 0 < tail_mass <= 1:
        raise ValueError("tail_mass must lie in (0, 1].")
    if not np.isfinite(scores).all():
        raise ValueError("Hardness scores contain NaN or infinite values.")

    if stable_ids is None:
        # Stable sorting makes tied-score behavior reproducible for a fixed
        # aligned manifest.
        order = np.argsort(-scores, kind="mergesort")
    else:
        ids = np.asarray(stable_ids).astype(str)
        if len(ids) != n:
            raise ValueError("stable_ids and scores must have equal length.")
        order = np.lexsort((ids, -scores))

    return order[:k]


def indices_to_ids(
    indices: set[int] | list[int] | np.ndarray,
    meta: pd.DataFrame,
    id_col: str = "knot_id_base",
) -> set[str]:
    """Convert positional hard-set indices into durable knot identifiers."""

    if id_col not in meta.columns:
        raise KeyError(f"Missing stable ID column: {id_col}")
    pos = np.asarray(sorted(indices), dtype=int)
    return set(meta.iloc[pos][id_col].astype(str))


def ids_to_indices(
    ids: set[str] | list[str] | np.ndarray,
    meta: pd.DataFrame,
    id_col: str = "knot_id_base",
) -> set[int]:
    """Resolve durable knot identifiers against the current aligned manifest."""

    if id_col not in meta.columns:
        raise KeyError(f"Missing stable ID column: {id_col}")
    if meta[id_col].astype(str).duplicated().any():
        raise ValueError(f"Column {id_col} must be unique.")

    lookup = pd.Series(np.arange(len(meta)), index=meta[id_col].astype(str))
    requested = pd.Index([str(value) for value in ids])
    missing = requested.difference(lookup.index)
    if len(missing):
        raise KeyError(f"Unknown knot IDs: {missing[:5].tolist()}")
    return set(map(int, lookup.loc[requested].to_numpy()))


def hard_sets_to_id_sets(
    hard_sets: dict[str, set[int]],
    meta: pd.DataFrame,
    id_col: str = "knot_id_base",
) -> dict[str, set[str]]:
    return {
        name: indices_to_ids(indices, meta, id_col=id_col)
        for name, indices in hard_sets.items()
    }


def build_hard_sets_from_results(
    reconstruction_results: dict,
    alpha: float = 0.99,
    tail_mass: float = 0.01,
    score_name: str = "sse",
    stable_ids: np.ndarray | pd.Series | None = None,
) -> dict[str, set[int]]:
    """
    Build top-tail hard sets from PCA or AE reconstruction result dictionaries.

    Expected structure:
    reconstruction_results[invariant][alpha][score_name]
    """

    hard_sets = {}

    for name in reconstruction_results.keys():
        scores = reconstruction_results[name][alpha][score_name]
        idx = top_tail_indices(scores, tail_mass=tail_mass, stable_ids=stable_ids)
        hard_sets[name] = set(map(int, idx))

    return hard_sets


def consensus_from_hard_sets(hard_sets: dict[str, set[int]]) -> set[int]:
    """
    Return intersection of all invariant-specific hard sets.
    """

    if not hard_sets:
        raise ValueError("No hard sets provided.")

    return set.intersection(*hard_sets.values())


def consensus_dataframe(
    consensus: set[int] | set[str],
    meta: pd.DataFrame,
    id_col: str = "knot_id_base",
) -> pd.DataFrame:
    """
    Return metadata rows corresponding to a consensus set.
    """

    if not consensus:
        return meta.iloc[0:0].copy()
    first = next(iter(consensus))
    if isinstance(first, (str, np.str_)):
        positions = ids_to_indices(consensus, meta, id_col=id_col)
        return meta.iloc[sorted(positions)].copy()
    return meta.iloc[sorted(consensus)].copy()


def summarize_consensus_set(
    consensus: set[int],
    meta: pd.DataFrame,
    score_name: str | None = None,
    evr_threshold: float | None = None,
    tail_mass: float = 0.01,
    s_col: str = "s_invariant_qc",
) -> dict:
    """
    Standard summary of a consensus hard set.
    """

    consensus_df = consensus_dataframe(consensus, meta)

    row = {
        "score": score_name,
        "evr_threshold": evr_threshold,
        "tail_mass": tail_mass,
        "tail_size_each_invariant": int(np.ceil(tail_mass * len(meta))),
        "consensus_size": len(consensus),
        "mean_signature": (
            consensus_df["signature"].mean() if len(consensus_df) else np.nan
        ),
        "median_signature": (
            consensus_df["signature"].median() if len(consensus_df) else np.nan
        ),
        "max_signature": (
            consensus_df["signature"].max() if len(consensus_df) else np.nan
        ),
        "prop_alternating": (
            consensus_df["is_alternating"].mean() if len(consensus_df) else np.nan
        ),
        "prop_crossing_15": (
            (consensus_df["number_of_crossings"] == 15).mean()
            if len(consensus_df)
            else np.nan
        ),
        "sigma_8_or_more": (
            int((consensus_df["signature"] >= 8).sum()) if len(consensus_df) else 0
        ),
        "sigma_10_or_more": (
            int((consensus_df["signature"] >= 10).sum()) if len(consensus_df) else 0
        ),
        "sigma_12_or_more": (
            int((consensus_df["signature"] >= 12).sum()) if len(consensus_df) else 0
        ),
    }

    if s_col not in meta.columns and "s_invariant" in meta.columns:
        s_col = "s_invariant"

    if s_col in meta.columns:
        row["s_8_or_more"] = (
            int((consensus_df[s_col] >= 8).sum())
            if len(consensus_df)
            else 0
        )
        row["s_10_or_more"] = (
            int((consensus_df[s_col] >= 10).sum())
            if len(consensus_df)
            else 0
        )
        row["s_12_or_more"] = (
            int((consensus_df[s_col] >= 12).sum())
            if len(consensus_df)
            else 0
        )

    return row


def consensus_summary_table(
    reconstruction_results: dict,
    meta: pd.DataFrame,
    alphas: list[float] | tuple[float, ...] = (0.94, 0.99, 0.999),
    scores: list[str] | tuple[str, ...] = ("sse", "nre"),
    tail_mass: float = 0.01,
) -> pd.DataFrame:
    """
    Build consensus summaries across scores and EVR thresholds.
    """

    rows = []

    for score_name in scores:
        for alpha in alphas:
            hard_sets = build_hard_sets_from_results(
                reconstruction_results,
                alpha=alpha,
                tail_mass=tail_mass,
                score_name=score_name,
            )
            consensus = consensus_from_hard_sets(hard_sets)

            rows.append(
                summarize_consensus_set(
                    consensus,
                    meta,
                    score_name=score_name,
                    evr_threshold=alpha,
                    tail_mass=tail_mass,
                )
            )

    return pd.DataFrame(rows)


def pairwise_overlaps(hard_sets: dict[str, set[int]]) -> pd.DataFrame:
    """
    Pairwise overlaps and Jaccard index between invariant-specific hard tails.
    """

    rows = []

    for a, b in combinations(hard_sets.keys(), 2):
        A = hard_sets[a]
        B = hard_sets[b]

        inter = A & B
        union = A | B

        rows.append(
            {
                "invariant_a": a,
                "invariant_b": b,
                "size_a": len(A),
                "size_b": len(B),
                "intersection": len(inter),
                "jaccard": len(inter) / len(union) if len(union) > 0 else np.nan,
                "overlap_over_tail": (
                    len(inter) / min(len(A), len(B))
                    if min(len(A), len(B)) > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def membership_count_table(
    hard_sets: dict[str, set[int]],
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Count in how many invariant-specific tails each object appears.
    """

    n = len(meta)
    counts = np.zeros(n, dtype=int)

    for s in hard_sets.values():
        counts[list(s)] += 1

    membership_df = pd.DataFrame(
        {
            "membership_count": counts,
        }
    )

    summary = (
        membership_df["membership_count"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    summary.columns = ["n_invariant_tails", "n_knots"]
    summary["percentage"] = 100 * summary["n_knots"] / n

    return membership_df, summary


def characterize_persistent_hard(
    hard_sets: dict[str, set[int]],
    meta: pd.DataFrame,
    min_membership: int = 4,
) -> tuple[pd.DataFrame, dict]:
    """
    Characterize knots appearing in at least min_membership invariant tails.
    """

    membership_df, _ = membership_count_table(hard_sets, meta)

    df = meta.copy()
    df["n_hard_invariant_tails"] = membership_df["membership_count"].values

    hard = df[df["n_hard_invariant_tails"] >= min_membership].copy()

    summary = {
        "min_membership": min_membership,
        "n_knots": len(hard),
        "mean_signature": hard["signature"].mean() if len(hard) else np.nan,
        "median_signature": hard["signature"].median() if len(hard) else np.nan,
        "max_signature": hard["signature"].max() if len(hard) else np.nan,
        "prop_alternating": hard["is_alternating"].mean() if len(hard) else np.nan,
        "prop_crossing_15": (
            (hard["number_of_crossings"] == 15).mean()
            if len(hard)
            else np.nan
        ),
        "sigma_10_or_more": (
            int((hard["signature"] >= 10).sum()) if len(hard) else 0
        ),
        "sigma_12_or_more": (
            int((hard["signature"] >= 12).sum()) if len(hard) else 0
        ),
    }

    return hard, summary



def build_hard_sets_from_fixed_results(
    fixed_results: dict,
    score_name: str = "sse",
    tail_mass: float = 0.01,
    stable_ids: np.ndarray | pd.Series | None = None,
) -> dict[str, set[int]]:
    """
    Build hard sets from fixed-k reconstruction results.

    Expected structure:
    fixed_results[invariant][score_name]
    """

    hard_sets = {}

    for name in fixed_results.keys():
        if score_name not in fixed_results[name]:
            raise KeyError(f"Score '{score_name}' not found for '{name}'.")

        scores = fixed_results[name][score_name]
        idx = top_tail_indices(scores, tail_mass=tail_mass, stable_ids=stable_ids)
        hard_sets[name] = set(map(int, idx))

    return hard_sets


def consensus_summary_from_fixed_results(
    fixed_results: dict,
    meta: pd.DataFrame,
    score_name: str = "sse",
    tail_mass: float = 0.01,
) -> tuple[set[int], pd.DataFrame, dict]:
    """
    Build hard sets, consensus set, consensus dataframe and summary
    from fixed-k reconstruction results.
    """

    hard_sets = build_hard_sets_from_fixed_results(
        fixed_results,
        score_name=score_name,
        tail_mass=tail_mass,
    )

    consensus = consensus_from_hard_sets(hard_sets)
    consensus_df = consensus_dataframe(consensus, meta)

    summary = summarize_consensus_set(
        consensus,
        meta,
        score_name=score_name,
        evr_threshold=None,
        tail_mass=tail_mass,
    )

    return consensus, consensus_df, summary


def compare_consensus_sets(
    set_a: set[int],
    set_b: set[int],
    name_a: str = "A",
    name_b: str = "B",
) -> dict:
    """
    Compare two consensus sets.
    """

    inter = set_a & set_b
    union = set_a | set_b

    return {
        "set_a": name_a,
        "set_b": name_b,
        "size_a": len(set_a),
        "size_b": len(set_b),
        "overlap": len(inter),
        "jaccard": len(inter) / len(union) if len(union) else np.nan,
        "fraction_a_in_b": len(inter) / len(set_a) if len(set_a) else np.nan,
        "fraction_b_in_a": len(inter) / len(set_b) if len(set_b) else np.nan,
    }


def distribution_table(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Value-count table for a column.
    """

    return (
        df[column]
        .value_counts()
        .sort_index()
        .reset_index(name="count")
        .rename(columns={"index": column})
    )


def captured_threshold_table(
    selected_df: pd.DataFrame,
    background_df: pd.DataFrame,
    column: str,
    thresholds: list[int],
) -> pd.DataFrame:
    """
    For each threshold, compute how many background objects satisfying
    column >= threshold are captured by selected_df.
    """

    rows = []

    for threshold in thresholds:
        total = int((background_df[column] >= threshold).sum())
        captured = int((selected_df[column] >= threshold).sum())

        rows.append(
            {
                "column": column,
                "threshold": threshold,
                "background_count": total,
                "captured_count": captured,
                "capture_rate": captured / total if total > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)
