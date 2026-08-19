# src/consensus_hardness/null_models.py

from __future__ import annotations

import numpy as np
import pandas as pd


def random_tail_null(
    n: int,
    tail_size: int,
    n_sets: int = 5,
    n_reps: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Independent random-tail null model.

    Samples n_sets random tails of size tail_size from n objects
    and records their intersection size.
    """

    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)
    intersections = []

    for _ in range(n_reps):
        sets = [
            set(rng.choice(all_idx, size=tail_size, replace=False))
            for _ in range(n_sets)
        ]
        intersections.append(len(set.intersection(*sets)))

    return np.asarray(intersections)


def stratified_random_tail(
    meta: pd.DataFrame,
    reference_indices: set[int] | list[int],
    strata_cols: list[str],
    rng: np.random.Generator,
) -> set[int]:
    """
    Sample a random tail with the same stratum counts as reference_indices.
    """

    sampled = []

    ref = meta.iloc[list(reference_indices)].copy()
    full = meta.copy()

    stratum_counts = (
        ref.groupby(strata_cols, dropna=False)
        .size()
        .reset_index(name="n")
    )

    for _, row in stratum_counts.iterrows():
        mask = np.ones(len(full), dtype=bool)

        for col in strata_cols:
            if pd.isna(row[col]):
                mask &= full[col].isna().to_numpy()
            else:
                mask &= full[col].to_numpy() == row[col]

        candidates = np.where(mask)[0]
        n_sample = int(row["n"])

        if len(candidates) < n_sample:
            raise ValueError(
                f"Not enough candidates for stratum: "
                f"{ {col: row[col] for col in strata_cols} }"
            )

        sampled.extend(
            rng.choice(candidates, size=n_sample, replace=False)
        )

    return set(map(int, sampled))


def stratified_intersection_null(
    meta: pd.DataFrame,
    observed_hard_sets: dict[str, set[int]],
    strata_cols: list[str],
    n_reps: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Stratified null model.

    For each repetition, generate one random hard tail per invariant,
    preserving the stratum distribution of that invariant's observed tail.
    Then compute the five-way intersection size.
    """

    rng = np.random.default_rng(seed)
    intersections = []

    for _ in range(n_reps):
        random_sets = []

        for _, indices in observed_hard_sets.items():
            random_set = stratified_random_tail(
                meta=meta,
                reference_indices=indices,
                strata_cols=strata_cols,
                rng=rng,
            )
            random_sets.append(random_set)

        intersections.append(len(set.intersection(*random_sets)))

    return np.asarray(intersections)


def summarize_null_distribution(
    null_values: np.ndarray,
    observed: int,
    label: str,
) -> dict:
    """
    Summarize a null distribution of intersection sizes.
    """

    null_values = np.asarray(null_values)

    return {
        "null_model": label,
        "n_reps": len(null_values),
        "null_mean": float(np.mean(null_values)),
        "null_median": float(np.median(null_values)),
        "null_max": int(np.max(null_values)),
        "observed": int(observed),
        "n_ge_observed": int(np.sum(null_values >= observed)),
        # Plus-one correction: a finite Monte Carlo run never reports p=0.
        "empirical_p": float(
            (1 + np.sum(null_values >= observed)) / (len(null_values) + 1)
        ),
        "p50": float(np.percentile(null_values, 50)),
        "p90": float(np.percentile(null_values, 90)),
        "p95": float(np.percentile(null_values, 95)),
        "p99": float(np.percentile(null_values, 99)),
        "p100": float(np.percentile(null_values, 100)),
    }


def run_standard_intersection_nulls(
    meta: pd.DataFrame,
    observed_hard_sets: dict[str, set[int]],
    observed_consensus_size: int,
    tail_mass: float = 0.01,
    n_reps: int = 1000,
    seed: int = 42,
    signature_strata_col: str = "signature_bin",
    include_crossing_alternating: bool = True,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Run the nested intersection null models used in the paper:
    1. independent random tails
    2. crossing-number matched tails
    3. crossing + alternating matched tails
    4. crossing + alternating + signature matched tails
    """

    n = len(meta)
    tail_size = int(np.ceil(tail_mass * n))
    n_sets = len(observed_hard_sets)

    nulls = {}

    nulls["independent_random_tails"] = random_tail_null(
        n=n,
        tail_size=tail_size,
        n_sets=n_sets,
        n_reps=n_reps,
        seed=seed,
    )

    nulls["crossing_matched_tails"] = stratified_intersection_null(
        meta=meta,
        observed_hard_sets=observed_hard_sets,
        strata_cols=["number_of_crossings"],
        n_reps=n_reps,
        seed=seed + 1,
    )

    if include_crossing_alternating:
        nulls["crossing_alt_matched_tails"] = stratified_intersection_null(
            meta=meta,
            observed_hard_sets=observed_hard_sets,
            strata_cols=["number_of_crossings", "is_alternating"],
            n_reps=n_reps,
            seed=seed + 2,
        )

    nulls["crossing_alt_signature_bin_matched_tails"] = stratified_intersection_null(
        meta=meta,
        observed_hard_sets=observed_hard_sets,
        strata_cols=["number_of_crossings", "is_alternating", signature_strata_col],
        n_reps=n_reps,
        seed=seed + 3,
    )

    summary_rows = [
        summarize_null_distribution(
            nulls["independent_random_tails"],
            observed=observed_consensus_size,
            label="Independent random tails",
        ),
        summarize_null_distribution(
            nulls["crossing_matched_tails"],
            observed=observed_consensus_size,
            label="Crossing-matched tails",
        ),
    ]

    if include_crossing_alternating:
        summary_rows.append(
            summarize_null_distribution(
                nulls["crossing_alt_matched_tails"],
                observed=observed_consensus_size,
                label="Crossing + alternating matched tails",
            )
        )

    summary_rows.append(
        summarize_null_distribution(
            nulls["crossing_alt_signature_bin_matched_tails"],
            observed=observed_consensus_size,
            label=f"Crossing + alternating + {signature_strata_col} matched tails",
        )
    )

    return pd.DataFrame(summary_rows), nulls
