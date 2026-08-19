# %% [markdown]
# Stage 18A — Mirror-invariant concordance-gap sensitivity
# Paste this cell AFTER the frozen corrected_run_20260819 notebook.
# It reuses: meta, CONFIG, ch, norm_bins_100,
# observed_conditional_masks, conditional_membership_sets, INVARIANTS,
# NO_KHOVANOV, POLYNOMIAL_ONLY, and OUTPUT_DIR.

# %%
from pathlib import Path
import json
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


# ------------------------------------------------------------
# 0. Preconditions and output directory
# ------------------------------------------------------------
REQUIRED_OBJECTS = (
    "meta",
    "CONFIG",
    "ch",
    "norm_bins_100",
    "observed_conditional_masks",
    "conditional_membership_sets",
    "INVARIANTS",
    "NO_KHOVANOV",
    "POLYNOMIAL_ONLY",
    "OUTPUT_DIR",
)

missing_objects = [
    name for name in REQUIRED_OBJECTS
    if name not in globals()
]

if missing_objects:
    raise RuntimeError(
        "Run this cell after the frozen analysis notebook. "
        f"Missing objects: {missing_objects}"
    )

MIRROR_DIR = Path(OUTPUT_DIR) / "18A_mirror_invariant_gap"
MIRROR_DIR.mkdir(parents=True, exist_ok=True)

S_COL = CONFIG.s_col
N_REPS = 5_000

required_columns = {
    CONFIG.universe.id_col,
    "number_of_crossings",
    "is_alternating",
    "signature",
    S_COL,
}

missing_columns = sorted(required_columns - set(meta.columns))
if missing_columns:
    raise KeyError(f"Missing metadata columns: {missing_columns}")

if meta[list(required_columns)].isna().any().any():
    raise ValueError(
        "Mirror analysis requires complete IDs, crossing, alternation, "
        "signature and QC Rasmussen invariant."
    )

if set(norm_bins_100) != set(INVARIANTS):
    raise AssertionError(
        "norm_bins_100 must contain one bin vector per invariant."
    )

if set(observed_conditional_masks) != set(INVARIANTS):
    raise AssertionError(
        "observed_conditional_masks must contain one hard mask per invariant."
    )

for invariant in INVARIANTS:
    if len(norm_bins_100[invariant]) != len(meta):
        raise AssertionError(f"Wrong norm-bin length for {invariant}.")
    if len(observed_conditional_masks[invariant]) != len(meta):
        raise AssertionError(f"Wrong hard-mask length for {invariant}.")


# ------------------------------------------------------------
# 1. Define signed and mirror-invariant endpoints
# ------------------------------------------------------------
meta_mirror = meta.copy()
meta_mirror["delta_signed"] = (
    meta_mirror[S_COL].astype(float)
    - meta_mirror["signature"].astype(float)
)
meta_mirror["delta_abs"] = meta_mirror["delta_signed"].abs()
meta_mirror["signature_abs"] = (
    meta_mirror["signature"].astype(float).abs()
)

# Existing null code computes (s_col - signature).  This proxy makes that
# internal difference exactly |s - sigma| without changing the null sampler.
ABS_PROXY_COL = "s_abs_gap_proxy"
meta_mirror[ABS_PROXY_COL] = (
    meta_mirror["signature"].astype(float)
    + meta_mirror["delta_abs"]
)

if not np.allclose(
    meta_mirror[ABS_PROXY_COL] - meta_mirror["signature"],
    meta_mirror["delta_abs"],
):
    raise AssertionError("Absolute-gap proxy was constructed incorrectly.")


# ------------------------------------------------------------
# 2. Primary families: the same three focused 3-of-M analyses
# ------------------------------------------------------------
MIRROR_FAMILIES = {
    "All 5": {
        "names": tuple(INVARIANTS),
        "thresholds": (3,),
    },
    "No Khovanov": {
        "names": tuple(NO_KHOVANOV),
        "thresholds": (3,),
    },
    "Polynomial only": {
        "names": tuple(POLYNOMIAL_ONLY),
        "thresholds": (3,),
    },
}

ABS_INTERNAL_METRICS = (
    "nonalt_s_gt_sigma_prop",
    "mean_s_minus_sigma",
    "delta_ge_4_prop",
)

METRIC_LABELS = {
    "nonalt_s_gt_sigma_prop": "abs_delta_positive_prop",
    "mean_s_minus_sigma": "mean_abs_delta",
    "delta_ge_4_prop": "abs_delta_ge_4_prop",
}


def get_selected_set(family, k=3):
    key = (family, k)
    if key not in conditional_membership_sets:
        raise KeyError(
            f"conditional_membership_sets does not contain {key}."
        )
    return set(map(int, conditional_membership_sets[key]))


def observed_absolute_gap_table():
    rows = []

    for family, specification in MIRROR_FAMILIES.items():
        k = int(specification["thresholds"][0])
        m = len(specification["names"])
        selected = get_selected_set(family, k)

        selected_df = meta_mirror.iloc[sorted(selected)]
        nonalt = selected_df.loc[
            selected_df["is_alternating"].eq(0)
        ]
        delta_abs = nonalt["delta_abs"].to_numpy(float)

        if len(delta_abs) == 0:
            raise AssertionError(
                f"Observed {family} selection has no nonalternating knots."
            )

        rows.append({
            "family": family,
            "k": k,
            "m": m,
            "n": len(selected),
            "nonalternating_n": len(nonalt),
            "nonalt_s_gt_sigma_prop": float(
                np.mean(delta_abs > 0)
            ),
            "mean_s_minus_sigma": float(np.mean(delta_abs)),
            "delta_ge_4_prop": float(np.mean(delta_abs >= 4)),
        })

    return pd.DataFrame(rows)


observed_abs = observed_absolute_gap_table()

expected_sizes = {
    "All 5": (413, 338),
    "No Khovanov": (220, 167),
    "Polynomial only": (35, 8),
}

for row in observed_abs.itertuples(index=False):
    expected = expected_sizes[row.family]
    actual = (int(row.n), int(row.nonalternating_n))
    if actual != expected:
        raise AssertionError(
            f"Frozen-set mismatch for {row.family}: "
            f"expected {expected}, found {actual}."
        )

display(
    observed_abs.rename(columns=METRIC_LABELS)
)


# ------------------------------------------------------------
# 3. Two nulls
#    A. Original convention: own-view norm bin + crossing + alt + sigma
#    B. Mirror-invariant: own-view norm bin + crossing + alt + |sigma|
# ------------------------------------------------------------
joint_strata_signed_sigma = ch.build_joint_strata_codes(
    norm_bins_by_invariant={
        invariant: np.asarray(norm_bins_100[invariant], dtype=np.int32)
        for invariant in INVARIANTS
    },
    meta=meta_mirror,
    extra_cols=(
        "number_of_crossings",
        "is_alternating",
        "signature",
    ),
)

joint_strata_abs_sigma = ch.build_joint_strata_codes(
    norm_bins_by_invariant={
        invariant: np.asarray(norm_bins_100[invariant], dtype=np.int32)
        for invariant in INVARIANTS
    },
    meta=meta_mirror,
    extra_cols=(
        "number_of_crossings",
        "is_alternating",
        "signature_abs",
    ),
)

null_design_audit = pd.DataFrame([
    {
        "null_model": "signed_exact_sigma",
        "per_view_norm_bin": True,
        "crossing_exact": True,
        "alternation_exact": True,
        "signature_stratum": "signed exact sigma",
        "mirror_invariant_strata": False,
        "n_reps": N_REPS,
    },
    {
        "null_model": "absolute_exact_abs_sigma",
        "per_view_norm_bin": True,
        "crossing_exact": True,
        "alternation_exact": True,
        "signature_stratum": "exact |sigma|",
        "mirror_invariant_strata": True,
        "n_reps": N_REPS,
    },
])

display(null_design_audit)


def run_or_load_null(label, strata, seed):
    path = MIRROR_DIR / f"{label}_null_{N_REPS}_raw.parquet"

    if path.exists():
        print(f"Loading existing null: {path.name}")
        return pd.read_parquet(path)

    print(f"Running {label}: {N_REPS:,} repetitions")

    result = ch.run_stratified_membership_null(
        observed_hard_masks=observed_conditional_masks,
        strata_codes_by_invariant=strata,
        families=MIRROR_FAMILIES,
        meta=meta_mirror,
        n_reps=N_REPS,
        seed=seed,
        s_col=ABS_PROXY_COL,
    )

    result.to_parquet(path, index=False)
    result.to_csv(path.with_suffix(".csv"), index=False)
    return result


null_abs_signed_sigma = run_or_load_null(
    label="abs_gap_signed_exact_sigma",
    strata=joint_strata_signed_sigma,
    seed=20260820,
)

null_abs_abs_sigma = run_or_load_null(
    label="abs_gap_exact_abs_sigma",
    strata=joint_strata_abs_sigma,
    seed=20260821,
)


# ------------------------------------------------------------
# 4. Summaries, valid counts, Monte Carlo floor, BH/BY/Holm
# ------------------------------------------------------------
def summarize_one_null(raw_null, label):
    summary = ch.summarize_null_against_observed(
        observed=observed_abs,
        null=raw_null,
        metrics=ABS_INTERNAL_METRICS,
    )
    summary.insert(0, "null_model", label)

    count_rows = []
    for (family, k), group in raw_null.groupby(
        ["family", "k"], sort=False
    ):
        for metric in ABS_INTERNAL_METRICS:
            values = group[metric]
            count_rows.append({
                "family": family,
                "k": int(k),
                "metric": metric,
                "n_valid_null": int(values.notna().sum()),
                "n_missing_null": int(values.isna().sum()),
            })

    counts = pd.DataFrame(count_rows)
    summary = summary.merge(
        counts,
        on=["family", "k", "metric"],
        how="left",
    )

    # Recompute the finite-Monte-Carlo p-value explicitly using only
    # valid replicates. This makes the simulation floor transparent.
    summary["mc_corrected_p"] = (
        summary["n_ge_observed"] + 1
    ) / (
        summary["n_valid_null"] + 1
    )
    summary["mc_p_floor"] = 1 / (
        summary["n_valid_null"] + 1
    )
    summary["zero_exceedances"] = (
        summary["n_ge_observed"].eq(0)
    )

    return summary


summary_signed_sigma = summarize_one_null(
    null_abs_signed_sigma,
    "own-view norm + crossing + alt + exact sigma",
)

summary_abs_sigma = summarize_one_null(
    null_abs_abs_sigma,
    "own-view norm + crossing + alt + exact |sigma|",
)

mirror_null_summary = pd.concat(
    [summary_signed_sigma, summary_abs_sigma],
    ignore_index=True,
)

mirror_null_summary["metric_internal"] = (
    mirror_null_summary["metric"]
)
mirror_null_summary["metric"] = (
    mirror_null_summary["metric"].map(METRIC_LABELS)
)

for null_label, index in mirror_null_summary.groupby(
    "null_model"
).groups.items():
    pvals = mirror_null_summary.loc[index, "mc_corrected_p"]

    mirror_null_summary.loc[index, "q_bh"] = multipletests(
        pvals,
        method="fdr_bh",
    )[1]
    mirror_null_summary.loc[index, "q_by"] = multipletests(
        pvals,
        method="fdr_by",
    )[1]
    mirror_null_summary.loc[index, "p_holm"] = multipletests(
        pvals,
        method="holm",
    )[1]

mirror_null_summary["bh_significant"] = (
    mirror_null_summary["q_bh"] < 0.05
)
mirror_null_summary["by_significant"] = (
    mirror_null_summary["q_by"] < 0.05
)
mirror_null_summary["holm_significant"] = (
    mirror_null_summary["p_holm"] < 0.05
)

output_columns = [
    "null_model",
    "family",
    "k",
    "m",
    "metric",
    "observed",
    "null_mean",
    "null_q95",
    "null_q99",
    "n_ge_observed",
    "n_valid_null",
    "n_missing_null",
    "mc_corrected_p",
    "mc_p_floor",
    "zero_exceedances",
    "q_bh",
    "q_by",
    "p_holm",
    "bh_significant",
    "by_significant",
    "holm_significant",
]

display(
    mirror_null_summary[output_columns].sort_values(
        ["null_model", "family", "metric"]
    )
)


# ------------------------------------------------------------
# 5. Explain missing polynomial-only nulls
# ------------------------------------------------------------
missing_reason_rows = []

for label, raw in {
    "signed_exact_sigma": null_abs_signed_sigma,
    "exact_abs_sigma": null_abs_abs_sigma,
}.items():
    for (family, k), group in raw.groupby(
        ["family", "k"], sort=False
    ):
        # Gap metrics are undefined precisely when the randomized selected
        # set has no nonalternating knots. The null table's gap metrics expose
        # those rows as NaN even if an explicit n_nonalt column is unavailable.
        all_gap_missing = group[
            list(ABS_INTERNAL_METRICS)
        ].isna().all(axis=1)

        missing_reason_rows.append({
            "null_model": label,
            "family": family,
            "k": int(k),
            "n_reps": len(group),
            "n_reps_without_nonalternating_selected": int(
                all_gap_missing.sum()
            ),
            "valid_gap_reps": int((~all_gap_missing).sum()),
        })

missing_null_audit = pd.DataFrame(missing_reason_rows)
display(missing_null_audit)


# ------------------------------------------------------------
# 6. Orientation-randomization diagnostic on the fixed observed sets
#    This diagnoses sign-convention dependence. It is NOT a full mirror
#    rerun because the coefficient representations are held fixed.
# ------------------------------------------------------------
def orientation_randomization_diagnostic(n_reps=5_000, seed=20260822):
    rng = np.random.default_rng(seed)
    rows = []

    for family in MIRROR_FAMILIES:
        selected = get_selected_set(family, 3)
        selected_df = meta_mirror.iloc[sorted(selected)]
        delta = selected_df.loc[
            selected_df["is_alternating"].eq(0),
            "delta_signed",
        ].to_numpy(float)

        observed = {
            "signed_positive_prop": float(np.mean(delta > 0)),
            "mean_signed_delta": float(np.mean(delta)),
            "signed_delta_ge_4_prop": float(np.mean(delta >= 4)),
        }

        random_values = {
            key: np.empty(n_reps, dtype=float)
            for key in observed
        }

        for rep in range(n_reps):
            randomized = delta * rng.choice(
                (-1.0, 1.0),
                size=len(delta),
                replace=True,
            )
            random_values["signed_positive_prop"][rep] = np.mean(
                randomized > 0
            )
            random_values["mean_signed_delta"][rep] = np.mean(
                randomized
            )
            random_values["signed_delta_ge_4_prop"][rep] = np.mean(
                randomized >= 4
            )

        for metric, observed_value in observed.items():
            values = random_values[metric]
            rows.append({
                "family": family,
                "metric": metric,
                "observed": observed_value,
                "random_orientation_mean": float(np.mean(values)),
                "random_orientation_q95": float(
                    np.quantile(values, 0.95)
                ),
                "random_orientation_q99": float(
                    np.quantile(values, 0.99)
                ),
                "n_ge_observed": int(np.sum(values >= observed_value)),
                "mc_corrected_p": float(
                    (1 + np.sum(values >= observed_value))
                    / (n_reps + 1)
                ),
                "interpretation": (
                    "sign-convention diagnostic only; "
                    "representations were not mirrored"
                ),
            })

    return pd.DataFrame(rows)


orientation_diagnostic = orientation_randomization_diagnostic()
display(orientation_diagnostic)


# ------------------------------------------------------------
# 7. Save all paper-facing outputs and a machine-readable decision file
# ------------------------------------------------------------
observed_abs.rename(columns=METRIC_LABELS).to_csv(
    MIRROR_DIR / "observed_mirror_invariant_gap_metrics.csv",
    index=False,
)

mirror_null_summary[output_columns].to_csv(
    MIRROR_DIR / "mirror_invariant_gap_null_summary.csv",
    index=False,
)

null_design_audit.to_csv(
    MIRROR_DIR / "mirror_null_design_audit.csv",
    index=False,
)

missing_null_audit.to_csv(
    MIRROR_DIR / "missing_null_replicates_audit.csv",
    index=False,
)

orientation_diagnostic.to_csv(
    MIRROR_DIR / "orientation_randomization_diagnostic.csv",
    index=False,
)

primary_abs_sigma = mirror_null_summary.loc[
    mirror_null_summary["null_model"].str.contains(
        "exact \\|sigma\\|",
        regex=True,
    )
].copy()

decision = {
    "stage": "18A_mirror_invariant_gap",
    "primary_endpoint_family": [
        "P(|Delta| > 0)",
        "mean |Delta|",
        "P(|Delta| >= 4)",
    ],
    "primary_null": (
        "own-view norm bin + crossing number + alternation + exact |sigma|"
    ),
    "n_reps": N_REPS,
    "all_primary_by_significant": bool(
        primary_abs_sigma["by_significant"].all()
    ),
    "all_primary_holm_significant": bool(
        primary_abs_sigma["holm_significant"].all()
    ),
    "warning": (
        "Orientation randomization with fixed representations is only a "
        "sign-convention diagnostic. A full mirror robustness analysis must "
        "transform the invariant representations and rerun the scoring pipeline."
    ),
}

with open(
    MIRROR_DIR / "mirror_analysis_decision.json",
    "w",
    encoding="utf-8",
) as stream:
    json.dump(decision, stream, indent=2)

print("\n" + "=" * 80)
print("STAGE 18A COMPLETE")
print("=" * 80)
print(json.dumps(decision, indent=2))
print(f"\nSaved to:\n{MIRROR_DIR}")

