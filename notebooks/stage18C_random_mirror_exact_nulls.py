# %% [markdown]
# Stage 18C — Exact stratified nulls under randomized mirror representatives
#
# Run after the frozen notebook, Stage 18A and Stage 18B in the SAME runtime.
# The script loads the five Stage-18B random-orientation checkpoints and tests
# the mirror-invariant endpoint |s-sigma|.  It includes the essential
# no-Khovanov family that was not part of the first Stage-18B summary.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests


# ------------------------------------------------------------------
# 0. Preconditions and frozen specification
# ------------------------------------------------------------------
REQUIRED = (
    "meta",
    "INVARIANTS",
    "CONFIG",
    "OUTPUT_DIR",
    "ch",
    "conditional_hard_sets_100",
    "norm_bins_100",
)
missing = [name for name in REQUIRED if name not in globals()]
if missing:
    raise RuntimeError(
        "Run the frozen notebook and Stages 18A/18B first. "
        f"Missing objects: {missing}"
    )

OUT = Path(OUTPUT_DIR) / "18C_random_mirror_exact_nulls"
OUT.mkdir(parents=True, exist_ok=True)

STAGE18B = Path(OUTPUT_DIR) / "18B_mirror_representation_robustness"
if not STAGE18B.exists():
    raise FileNotFoundError(STAGE18B)

N = len(meta)
N_REPS = 5_000
RANDOM_SEEDS = (20260830, 20260831, 20260832, 20260833, 20260834)

ALL5 = tuple(INVARIANTS)
NO_KHOVANOV = tuple(x for x in INVARIANTS if x != "Khovanov")
NO_THETA = tuple(x for x in INVARIANTS if x != "Theta")
POLYNOMIAL = ("Alexander", "Jones", "HOMFLY-PT")

FAMILIES = {
    "All 5": {"names": ALL5, "thresholds": (3,)},
    "No Khovanov": {"names": NO_KHOVANOV, "thresholds": (3,)},
    "No Theta": {"names": NO_THETA, "thresholds": (3,)},
    "Polynomial only": {"names": POLYNOMIAL, "thresholds": (3,)},
}

ABS_PROXY_COL = "mirror_abs_gap_proxy"
meta_abs = meta.copy()
meta_abs["signature_abs"] = meta_abs["signature"].abs()
meta_abs["delta_abs"] = (
    meta_abs[CONFIG.s_col] - meta_abs["signature"]
).abs()
# The existing null code computes s_col-signature.  This proxy makes that
# internal difference equal |s-sigma| without changing the tested sampler.
meta_abs[ABS_PROXY_COL] = meta_abs["signature"] + meta_abs["delta_abs"]

METRICS = (
    "nonalt_s_gt_sigma_prop",
    "mean_s_minus_sigma",
    "delta_ge_4_prop",
)
METRIC_LABELS = {
    "nonalt_s_gt_sigma_prop": "abs_delta_positive_prop",
    "mean_s_minus_sigma": "mean_abs_delta",
    "delta_ge_4_prop": "abs_delta_ge_4_prop",
}


# ------------------------------------------------------------------
# 1. Helpers
# ------------------------------------------------------------------
def checkpoint_path(label):
    return STAGE18B / f"orientation_{label}_checkpoint.npz"


def load_checkpoint(label):
    path = checkpoint_path(label)
    if not path.exists():
        raise FileNotFoundError(path)

    hard_masks = {}
    norm_bins = {}
    with np.load(path, allow_pickle=False) as payload:
        for invariant in INVARIANTS:
            key = ch.safe_name(invariant)
            hard_masks[invariant] = np.asarray(
                payload[f"cond_{key}"], dtype=bool
            )
            norm_bins[invariant] = np.asarray(
                payload[f"normbin_{key}"], dtype=np.int32
            )
    return hard_masks, norm_bins


def baseline_masks_and_bins():
    masks = ch.hard_sets_to_masks(conditional_hard_sets_100, N)
    bins = {
        name: np.asarray(norm_bins_100[name], dtype=np.int32)
        for name in INVARIANTS
    }
    return masks, bins


def family_mask(hard_masks, names, k=3):
    count = np.zeros(N, dtype=np.uint8)
    for name in names:
        count += hard_masks[name]
    return count >= k


def observed_table(hard_masks):
    rows = []
    nonalt_all = meta_abs["is_alternating"].to_numpy() == 0
    delta = meta_abs["delta_abs"].to_numpy(float)

    for family, spec in FAMILIES.items():
        k = int(spec["thresholds"][0])
        selected_mask = family_mask(hard_masks, spec["names"], k)
        nonalt_mask = selected_mask & nonalt_all
        values = delta[nonalt_mask]
        rows.append({
            "family": family,
            "k": k,
            "m": len(spec["names"]),
            "n": int(selected_mask.sum()),
            "nonalternating_n": int(nonalt_mask.sum()),
            "nonalt_s_gt_sigma_prop": (
                float(np.mean(values > 0)) if len(values) else np.nan
            ),
            "mean_s_minus_sigma": (
                float(np.mean(values)) if len(values) else np.nan
            ),
            "delta_ge_4_prop": (
                float(np.mean(values >= 4)) if len(values) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def run_one(label, hard_masks, norm_bins, null_seed):
    print("\n" + "=" * 76)
    print("Exact mirror null:", label)
    print("=" * 76)

    observed = observed_table(hard_masks)
    observed.insert(0, "run", label)

    strata = ch.build_joint_strata_codes(
        norm_bins_by_invariant=norm_bins,
        meta=meta_abs,
        extra_cols=(
            "number_of_crossings",
            "is_alternating",
            "signature_abs",
        ),
    )

    raw_path = OUT / f"{label}_exact_abs_sigma_null_{N_REPS}.parquet"
    if raw_path.exists():
        print("Loading checkpoint:", raw_path.name)
        raw = pd.read_parquet(raw_path)
    else:
        raw = ch.run_stratified_membership_null(
            observed_hard_masks=hard_masks,
            strata_codes_by_invariant=strata,
            families=FAMILIES,
            meta=meta_abs,
            n_reps=N_REPS,
            seed=null_seed,
            s_col=ABS_PROXY_COL,
        )
        raw.to_parquet(raw_path, index=False)
        raw.to_csv(raw_path.with_suffix(".csv"), index=False)

    summary = ch.summarize_null_against_observed(
        observed=observed,
        null=raw,
        metrics=METRICS,
    )
    summary.insert(0, "run", label)
    summary["n_valid_null"] = N_REPS
    summary["mc_corrected_p"] = summary["empirical_p"]
    summary["metric_internal"] = summary["metric"]
    summary["metric"] = summary["metric"].map(METRIC_LABELS)

    observed.to_csv(OUT / f"{label}_observed.csv", index=False)
    summary.to_csv(OUT / f"{label}_null_summary.csv", index=False)
    display(observed.rename(columns=METRIC_LABELS))
    display(summary)
    return observed, summary


# ------------------------------------------------------------------
# 2. Numerical positive-control discrepancy audit
# ------------------------------------------------------------------
baseline_masks, baseline_bins = baseline_masks_and_bins()
global_masks, global_bins = load_checkpoint("global_all_mirrored")

audit_rows = []
for invariant in INVARIANTS:
    changed = np.flatnonzero(baseline_masks[invariant] != global_masks[invariant])
    for idx in changed:
        audit_rows.append({
            "invariant": invariant,
            "row_index": int(idx),
            "knot_id": meta.iloc[idx][CONFIG.universe.id_col],
            "baseline_hard": bool(baseline_masks[invariant][idx]),
            "global_mirror_hard": bool(global_masks[invariant][idx]),
            "baseline_norm_bin": int(baseline_bins[invariant][idx]),
            "global_norm_bin": int(global_bins[invariant][idx]),
        })

global_positive_control_audit = pd.DataFrame(audit_rows)
display(global_positive_control_audit)
global_positive_control_audit.to_csv(
    OUT / "global_mirror_boundary_discrepancy_audit.csv", index=False
)


# ------------------------------------------------------------------
# 3. Baseline plus five randomized orientations
# ------------------------------------------------------------------
all_observed = []
all_summaries = []

# The baseline is inexpensive to summarize.  Its raw null from Stage 18A is
# not reused because this run adds the no-Khovanov and no-Theta families.
obs, summ = run_one(
    "baseline_canonical",
    baseline_masks,
    baseline_bins,
    null_seed=20260900,
)
all_observed.append(obs)
all_summaries.append(summ)

for offset, seed in enumerate(RANDOM_SEEDS, start=1):
    label = f"random_seed_{seed}"
    masks, bins = load_checkpoint(label)
    obs, summ = run_one(
        label,
        masks,
        bins,
        null_seed=20260900 + offset,
    )
    all_observed.append(obs)
    all_summaries.append(summ)

mirror_random_observed = pd.concat(all_observed, ignore_index=True)
mirror_random_null_summary = pd.concat(all_summaries, ignore_index=True)

# Correction is reported transparently.  Holm and BY are robust choices for
# these dependent, nested tests; BH is retained for continuity with the paper.
for method, output_col in (
    ("fdr_bh", "q_bh"),
    ("fdr_by", "q_by"),
    ("holm", "p_holm"),
):
    mirror_random_null_summary[output_col] = multipletests(
        mirror_random_null_summary["mc_corrected_p"].to_numpy(float),
        method=method,
    )[1]

mirror_random_observed.to_csv(
    OUT / "mirror_random_observed_all_runs.csv", index=False
)
mirror_random_null_summary.to_csv(
    OUT / "mirror_random_exact_null_summary_all_runs.csv", index=False
)


# ------------------------------------------------------------------
# 4. Compact decision table across the five random seeds
# ------------------------------------------------------------------
random_only = mirror_random_null_summary.loc[
    mirror_random_null_summary["run"].str.startswith("random_seed_")
].copy()

mirror_random_decision = (
    random_only
    .groupby(["family", "metric"], as_index=False)
    .agg(
        observed_min=("observed", "min"),
        observed_mean=("observed", "mean"),
        observed_max=("observed", "max"),
        null_mean_min=("null_mean", "min"),
        null_mean_max=("null_mean", "max"),
        empirical_p_min=("mc_corrected_p", "min"),
        empirical_p_max=("mc_corrected_p", "max"),
        q_by_max=("q_by", "max"),
        p_holm_max=("p_holm", "max"),
        n_random_seeds=("run", "nunique"),
    )
)
mirror_random_decision["all_five_raw_p_lt_0_05"] = (
    mirror_random_decision["empirical_p_max"] < 0.05
)
mirror_random_decision["all_five_by_q_lt_0_05"] = (
    mirror_random_decision["q_by_max"] < 0.05
)
mirror_random_decision["all_five_holm_lt_0_05"] = (
    mirror_random_decision["p_holm_max"] < 0.05
)

display(mirror_random_decision)
mirror_random_decision.to_csv(
    OUT / "mirror_random_decision_table.csv", index=False
)

print("\nSaved Stage 18C to:")
print(OUT)

