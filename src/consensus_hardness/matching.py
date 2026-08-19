# src/consensus_hardness/matching.py

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu, wilcoxon


def standardized_mean_difference(x, y) -> float:
    """
    Standardized mean difference between two numeric samples.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    pooled = np.sqrt((np.var(x) + np.var(y)) / 2)

    if pooled == 0:
        return 0.0

    return float((np.mean(x) - np.mean(y)) / pooled)


def add_zscore_columns(
    df: pd.DataFrame,
    cols: list[str],
    suffix: str = "_z",
) -> pd.DataFrame:
    """
    Add z-scored versions of selected columns.
    """

    out = df.copy()

    for col in cols:
        mu = out[col].mean()
        sd = out[col].std()

        if sd == 0 or pd.isna(sd):
            out[col + suffix] = 0.0
        else:
            out[col + suffix] = (out[col] - mu) / sd

    return out


def add_quantile_bin(
    df: pd.DataFrame,
    source_col: str,
    output_col: str,
    q: int = 10,
    subset_index=None,
) -> pd.DataFrame:
    """
    Add quantile bins for source_col.

    If subset_index is provided, bins are computed only within that subset.
    Useful for held-out test-set norm bins.
    """

    out = df.copy()
    out[output_col] = np.nan

    if subset_index is None:
        values = out[source_col]
        target_index = out.index
    else:
        values = out.loc[subset_index, source_col]
        target_index = subset_index

    out.loc[target_index, output_col] = pd.qcut(
        values,
        q=q,
        labels=False,
        duplicates="drop",
    )

    out[output_col] = out[output_col].astype("float")

    return out


def sample_exact_matched_controls(
    df: pd.DataFrame,
    selected_col: str,
    strata_cols: list[str],
    ratio: int = 5,
    seed: int = 42,
    allow_replace: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample exact matched controls from non-selected objects.

    Matching is exact on strata_cols.
    """

    rng = np.random.default_rng(seed)

    selected = df[df[selected_col]].copy()
    pool = df[~df[selected_col]].copy()

    sampled_records = []
    unmatched_rows = []
    used_controls = set()

    for selected_idx, row in selected.iterrows():
        mask = np.ones(len(pool), dtype=bool)

        for col in strata_cols:
            val = row[col]
            if pd.isna(val):
                mask &= pool[col].isna().values
            else:
                mask &= pool[col].values == val

        candidates = pool.index.to_numpy()[mask]
        if not allow_replace:
            candidates = np.asarray([i for i in candidates if i not in used_controls])

        if len(candidates) == 0:
            unmatched_info = {col: row[col] for col in strata_cols}
            unmatched_info["selected_index"] = selected_idx
            unmatched_info["reason"] = "no exact-stratum candidates"
            unmatched_rows.append(unmatched_info)
            continue

        replace = allow_replace and len(candidates) < ratio
        n_sample = ratio if replace else min(ratio, len(candidates))

        sampled = rng.choice(
            candidates,
            size=n_sample,
            replace=replace,
        )

        for control_idx in sampled.tolist():
            used_controls.add(control_idx)
            sampled_records.append(
                {
                    "selected_index": selected_idx,
                    "control_index": control_idx,
                    "match_group_id": str(selected_idx),
                }
            )

    if sampled_records:
        pairs = pd.DataFrame(sampled_records)
        controls = df.loc[pairs["control_index"].to_numpy()].copy()
        controls["match_group_id"] = pairs["match_group_id"].to_numpy()
        controls["matched_selected_index"] = pairs["selected_index"].to_numpy()
    else:
        controls = df.iloc[0:0].copy()
    unmatched = pd.DataFrame(unmatched_rows)

    return controls, unmatched


def compare_two_groups(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    value_cols: list[str],
    label_a: str = "selected",
    label_b: str = "control",
) -> pd.DataFrame:
    """
    Compare numeric features between two groups.
    """

    rows = []

    for col in value_cols:
        a = df_a[col].dropna().values
        b = df_b[col].dropna().values

        if len(a) == 0 or len(b) == 0:
            continue

        stat, p = mannwhitneyu(a, b, alternative="two-sided")

        rows.append(
            {
                "feature": col,
                f"{label_a}_mean": np.mean(a),
                f"{label_b}_mean": np.mean(b),
                f"{label_a}_median": np.median(a),
                f"{label_b}_median": np.median(b),
                f"{label_a}_q90": np.quantile(a, 0.90),
                f"{label_b}_q90": np.quantile(b, 0.90),
                "mannwhitney_p": p,
                "smd": standardized_mean_difference(a, b),
            }
        )

    return pd.DataFrame(rows)


def nearest_norm_matched_controls(
    df: pd.DataFrame,
    selected_col: str,
    exact_cols: list[str],
    norm_cols: list[str],
    ratio: int = 5,
    caliper: float | None = 0.25,
    replace: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Exact match on exact_cols, then nearest-neighbor match on norm_cols.

    Steps:
    1. Z-score norm_cols.
    2. For each selected object, restrict controls to same exact stratum.
    3. Choose nearest controls in standardized norm space.
    4. Optionally enforce a caliper.

    caliper is Euclidean distance in standardized norm space.
    If caliper=None, the nearest controls are always taken.
    """

    work = df.copy()
    z_cols = [c + "_z" for c in norm_cols]
    work = add_zscore_columns(work, norm_cols, suffix="_z")

    selected = work[work[selected_col]].copy()
    pool = work[~work[selected_col]].copy()

    matched_rows = []
    unmatched_rows = []
    used_controls = set()

    for selected_idx, selected_row in selected.iterrows():
        mask = np.ones(len(pool), dtype=bool)

        for col in exact_cols:
            if pd.isna(selected_row[col]):
                mask &= pool[col].isna().to_numpy()
            else:
                mask &= pool[col].to_numpy() == selected_row[col]

        candidates = pool.loc[mask].copy()

        if len(candidates) == 0:
            unmatched_rows.append(
                {
                    "selected_index": selected_idx,
                    "reason": "no exact-stratum candidates",
                    **{col: selected_row[col] for col in exact_cols},
                }
            )
            continue

        if not replace:
            candidates = candidates[~candidates.index.isin(used_controls)]

        if len(candidates) == 0:
            unmatched_rows.append(
                {
                    "selected_index": selected_idx,
                    "reason": "all candidates already used",
                    **{col: selected_row[col] for col in exact_cols},
                }
            )
            continue

        selected_vec = selected_row[z_cols].values.astype(float)
        candidate_mat = candidates[z_cols].values.astype(float)

        distances = np.sqrt(np.sum((candidate_mat - selected_vec) ** 2, axis=1))

        candidates = candidates.copy()
        candidates["match_distance"] = distances

        if caliper is not None:
            candidates = candidates[candidates["match_distance"] <= caliper].copy()

        if len(candidates) == 0:
            unmatched_rows.append(
                {
                    "selected_index": selected_idx,
                    "reason": "no candidates within caliper",
                    "caliper": caliper,
                    **{col: selected_row[col] for col in exact_cols},
                }
            )
            continue

        candidates = candidates.assign(_control_index=candidates.index.astype(str))
        candidates = candidates.sort_values(
            ["match_distance", "_control_index"], kind="mergesort"
        )

        n_take = min(ratio, len(candidates))

        # deterministic nearest controls
        chosen = candidates.head(n_take).copy()

        for control_idx, control_row in chosen.iterrows():
            used_controls.add(control_idx)

            matched_rows.append(
                {
                    "selected_index": selected_idx,
                    "control_index": control_idx,
                    "match_group_id": str(selected_idx),
                    "match_distance": control_row["match_distance"],
                    **{col: selected_row[col] for col in exact_cols},
                }
            )

    matched_pairs = pd.DataFrame(matched_rows)
    unmatched = pd.DataFrame(unmatched_rows)

    if len(matched_pairs) == 0:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            matched_pairs,
            unmatched,
        )

    control_indices = matched_pairs["control_index"].values
    selected_indices = matched_pairs["selected_index"].unique()

    matched_selected = df.loc[selected_indices].copy()
    matched_controls = df.loc[control_indices].copy()
    matched_selected["match_group_id"] = matched_selected.index.astype(str)
    matched_controls["match_group_id"] = matched_pairs["match_group_id"].to_numpy()
    matched_controls["match_distance"] = matched_pairs["match_distance"].to_numpy()

    return matched_selected, matched_controls, matched_pairs, unmatched


def balance_table(
    selected_df: pd.DataFrame,
    control_df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """
    Balance diagnostics for selected vs matched controls.
    """

    rows = []

    for col in cols:
        x = selected_df[col].dropna().values
        y = control_df[col].dropna().values

        if len(x) == 0 or len(y) == 0:
            continue

        rows.append(
            {
                "feature": col,
                "selected_mean": np.mean(x),
                "control_mean": np.mean(y),
                "selected_median": np.median(x),
                "control_median": np.median(y),
                "selected_q25": np.quantile(x, 0.25),
                "control_q25": np.quantile(y, 0.25),
                "selected_q75": np.quantile(x, 0.75),
                "control_q75": np.quantile(y, 0.75),
                "smd": standardized_mean_difference(x, y),
            }
        )

    return pd.DataFrame(rows)


def compare_matched_outcomes(
    selected_df: pd.DataFrame,
    control_df: pd.DataFrame,
    outcome_cols: list[str],
) -> pd.DataFrame:
    """
    Compare outcomes between selected and norm-nearest matched controls.
    """

    rows = []

    for col in outcome_cols:
        x = selected_df[col].dropna().values
        y = control_df[col].dropna().values

        if len(x) == 0 or len(y) == 0:
            continue

        stat, p = mannwhitneyu(x, y, alternative="two-sided")

        rows.append(
            {
                "feature": col,
                "selected_mean": np.mean(x),
                "control_mean": np.mean(y),
                "selected_median": np.median(x),
                "control_median": np.median(y),
                "selected_q90": np.quantile(x, 0.90),
                "control_q90": np.quantile(y, 0.90),
                "mannwhitney_p": p,
                "smd": standardized_mean_difference(x, y),
            }
        )

    return pd.DataFrame(rows)


def run_caliper_sensitivity(
    df: pd.DataFrame,
    selected_col: str,
    exact_cols: list[str],
    norm_cols: list[str],
    outcome_cols: list[str],
    ratio: int = 5,
    calipers: tuple[float | None, ...] = (0.10, 0.15, 0.20, 0.25, 0.35, 0.50, None),
    replace: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Run norm-nearest matching over several calipers.
    """

    rows = []
    results = {}

    for caliper in calipers:
        selected_m, controls_m, pairs, unmatched = nearest_norm_matched_controls(
            df=df,
            selected_col=selected_col,
            exact_cols=exact_cols,
            norm_cols=norm_cols,
            ratio=ratio,
            caliper=caliper,
            replace=replace,
            seed=seed,
        )

        if len(selected_m):
            bal = balance_table(selected_m, controls_m, norm_cols)
            out = compare_matched_outcomes_grouped(df, pairs, outcome_cols)
            max_abs_norm_smd = bal["smd"].abs().max() if len(bal) else np.nan
        else:
            bal = pd.DataFrame()
            out = pd.DataFrame()
            max_abs_norm_smd = np.nan

        rows.append(
            {
                "caliper": "None" if caliper is None else caliper,
                "n_selected_matched": len(selected_m),
                "n_controls": len(controls_m),
                "n_pairs": len(pairs),
                "n_unmatched": len(unmatched),
                "match_coverage": (
                    len(selected_m) / int(df[selected_col].sum())
                    if int(df[selected_col].sum())
                    else np.nan
                ),
                "n_unique_controls": (
                    int(pairs["control_index"].nunique()) if len(pairs) else 0
                ),
                "max_abs_norm_smd": max_abs_norm_smd,
                "mean_match_distance": pairs["match_distance"].mean() if len(pairs) else np.nan,
                "median_match_distance": pairs["match_distance"].median() if len(pairs) else np.nan,
            }
        )

        results[caliper] = {
            "selected": selected_m,
            "controls": controls_m,
            "pairs": pairs,
            "unmatched": unmatched,
            "balance": bal,
            "outcomes": out,
        }

    return pd.DataFrame(rows), results


def summarize_outcome_across_seeds(
    outcome_df: pd.DataFrame,
    feature_patterns: list[str],
    feature_col: str = "feature",
    seed_col: str = "seed",
) -> pd.DataFrame:
    """
    Summarize matched-outcome tables across AE seeds.
    """

    rows = []

    for pattern in feature_patterns:
        sub = outcome_df[outcome_df[feature_col].str.contains(pattern, regex=False)].copy()

        if len(sub) == 0:
            continue

        rows.append(
            {
                "feature_pattern": pattern,
                "selected_mean_mean": sub["selected_mean"].mean(),
                "control_mean_mean": sub["control_mean"].mean(),
                "selected_median_mean": sub["selected_median"].mean(),
                "control_median_mean": sub["control_median"].mean(),
                "smd_mean": sub["smd"].mean(),
                "smd_sd": sub["smd"].std(),
                "p_median": sub["mannwhitney_p"].median(),
                "n_seeds": sub[seed_col].nunique() if seed_col in sub.columns else np.nan,
            }
        )

    return pd.DataFrame(rows)


def compare_matched_outcomes_grouped(
    df: pd.DataFrame,
    matched_pairs: pd.DataFrame,
    outcome_cols: list[str],
) -> pd.DataFrame:
    """Use one control mean per selected knot and paired inference.

    Replacement matching creates repeated control rows.  Collapsing controls
    within each match group prevents those repeats from being treated as
    independent observations.
    """

    required = {"selected_index", "control_index", "match_group_id"}
    missing = required - set(matched_pairs.columns)
    if missing:
        raise KeyError(f"Missing matched-pair columns: {sorted(missing)}")

    rows = []
    for col in outcome_cols:
        group_rows = []
        for selected_idx, pairs in matched_pairs.groupby("selected_index", sort=False):
            selected_value = pd.to_numeric(
                pd.Series([df.loc[selected_idx, col]]), errors="coerce"
            ).iloc[0]
            control_values = pd.to_numeric(
                df.loc[pairs["control_index"].to_numpy(), col], errors="coerce"
            )
            control_mean = control_values.mean()
            if pd.notna(selected_value) and pd.notna(control_mean):
                group_rows.append((float(selected_value), float(control_mean)))

        if not group_rows:
            continue

        values = np.asarray(group_rows, dtype=float)
        selected_values = values[:, 0]
        control_means = values[:, 1]
        differences = selected_values - control_means
        p_value = (
            1.0
            if np.allclose(differences, 0)
            else float(wilcoxon(differences, alternative="two-sided").pvalue)
        )

        rows.append(
            {
                "feature": col,
                "n_match_groups": int(len(values)),
                "selected_mean": float(selected_values.mean()),
                "matched_control_mean": float(control_means.mean()),
                "mean_paired_difference": float(differences.mean()),
                "median_paired_difference": float(np.median(differences)),
                "paired_wilcoxon_p": p_value,
                "smd": standardized_mean_difference(selected_values, control_means),
            }
        )

    return pd.DataFrame(rows)
