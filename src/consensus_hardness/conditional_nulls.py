from __future__ import annotations

import numpy as np
import pandas as pd


def hard_sets_to_masks(hard_sets: dict[str, set[int]], n: int) -> dict[str, np.ndarray]:
    masks = {}
    for name, indices in hard_sets.items():
        mask = np.zeros(n, dtype=bool)
        if indices:
            mask[np.fromiter(indices, dtype=int)] = True
        masks[name] = mask
    return masks


def build_joint_strata_codes(
    norm_bins_by_invariant: dict[str, np.ndarray],
    meta: pd.DataFrame,
    extra_cols: tuple[str, ...] = (
        "number_of_crossings",
        "is_alternating",
        "signature",
    ),
) -> dict[str, np.ndarray]:
    codes = {}
    for name, norm_bins in norm_bins_by_invariant.items():
        frame = pd.DataFrame({"norm_bin": norm_bins})
        for col in extra_cols:
            frame[col] = meta[col].astype(str).to_numpy()
        code, _ = pd.factorize(pd.MultiIndex.from_frame(frame), sort=False)
        codes[name] = code.astype(np.int32)
    return codes


def prepare_stratified_sampler(
    strata_codes: np.ndarray,
    hard_mask: np.ndarray,
) -> dict:
    """Preserve the exact number of hard observations in every stratum."""

    frame = pd.DataFrame(
        {"position": np.arange(len(hard_mask)), "stratum": strata_codes, "hard": hard_mask}
    )
    random_groups = []
    fixed_groups = []
    for _, group in frame.groupby("stratum", sort=False):
        indices = group["position"].to_numpy(dtype=int)
        k = int(group["hard"].sum())
        if k == 0:
            continue
        if k == len(indices):
            fixed_groups.append(indices)
        else:
            random_groups.append((indices, k))
    return {
        "random_groups": random_groups,
        "fixed_idx": (
            np.concatenate(fixed_groups) if fixed_groups else np.empty(0, dtype=int)
        ),
    }


def sample_stratified_hard_mask(
    sampler: dict,
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[sampler["fixed_idx"]] = True
    for indices, k in sampler["random_groups"]:
        mask[rng.choice(indices, size=k, replace=False)] = True
    return mask


def default_metrics_from_mask(
    mask: np.ndarray,
    meta: pd.DataFrame,
    s_col: str = "s_invariant_qc",
) -> dict:
    if s_col not in meta.columns:
        s_col = "s_invariant"
    selected = meta.iloc[np.flatnonzero(mask)]
    nonalt = selected.loc[selected["is_alternating"].eq(0)]
    n = len(selected)
    return {
        "n": n,
        "median_sigma": selected["signature"].median() if n else np.nan,
        "median_s": selected[s_col].median() if n else np.nan,
        "sigma_ge_10_prop": selected["signature"].ge(10).mean() if n else np.nan,
        "s_ge_10_prop": selected[s_col].ge(10).mean() if n else np.nan,
        "s_ge_12_prop": selected[s_col].ge(12).mean() if n else np.nan,
        "nonalt_s_gt_sigma_prop": (
            nonalt[s_col].gt(nonalt["signature"]).mean() if len(nonalt) else np.nan
        ),
        "mean_s_minus_sigma": (
            (nonalt[s_col] - nonalt["signature"]).mean() if len(nonalt) else np.nan
        ),
        "delta_ge_4_prop": (
            (nonalt[s_col] - nonalt["signature"]).ge(4).mean()
            if len(nonalt)
            else np.nan
        ),
    }


def run_stratified_membership_null(
    observed_hard_masks: dict[str, np.ndarray],
    strata_codes_by_invariant: dict[str, np.ndarray],
    families: dict[str, dict],
    meta: pd.DataFrame,
    n_reps: int = 1_000,
    seed: int = 42,
    s_col: str = "s_invariant_qc",
) -> pd.DataFrame:
    """Permutation null for at-least-k membership statistics."""

    n = len(meta)
    samplers = {
        name: prepare_stratified_sampler(strata_codes_by_invariant[name], mask)
        for name, mask in observed_hard_masks.items()
    }
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(n_reps):
        sampled = {
            name: sample_stratified_hard_mask(sampler, rng, n)
            for name, sampler in samplers.items()
        }
        for family, spec in families.items():
            count = np.zeros(n, dtype=np.uint8)
            for name in spec["names"]:
                count += sampled[name]
            for k in spec["thresholds"]:
                row = {
                    "replicate": replicate,
                    "family": family,
                    "k": int(k),
                    "m": int(len(spec["names"])),
                }
                row.update(default_metrics_from_mask(count >= k, meta, s_col=s_col))
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_null_against_observed(
    observed: pd.DataFrame,
    null: pd.DataFrame,
    metrics: tuple[str, ...] = (
        "n",
        "median_s",
        "sigma_ge_10_prop",
        "s_ge_10_prop",
        "s_ge_12_prop",
        "nonalt_s_gt_sigma_prop",
        "mean_s_minus_sigma",
        "delta_ge_4_prop",
    ),
) -> pd.DataFrame:
    rows = []
    for _, row in observed.iterrows():
        subset = null.loc[null["family"].eq(row["family"]) & null["k"].eq(row["k"])]
        for metric in metrics:
            if metric not in row or metric not in subset:
                continue
            values = subset[metric].dropna().to_numpy(dtype=float)
            observed_value = float(row[metric])
            if not len(values) or not np.isfinite(observed_value):
                continue
            exceedances = int(np.sum(values >= observed_value))
            rows.append(
                {
                    "family": row["family"],
                    "k": int(row["k"]),
                    "m": int(row["m"]),
                    "metric": metric,
                    "observed": observed_value,
                    "null_mean": float(values.mean()),
                    "null_sd": float(values.std(ddof=1)),
                    "null_q95": float(np.quantile(values, 0.95)),
                    "null_q99": float(np.quantile(values, 0.99)),
                    "n_ge_observed": exceedances,
                    "empirical_p": (1 + exceedances) / (len(values) + 1),
                }
            )
    return pd.DataFrame(rows)
