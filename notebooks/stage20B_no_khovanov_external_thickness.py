# %% [markdown]
# Stage 20B — Does hardness selected without Khovanov predict Khovanov thickness?
#
# Selection uses only Alexander, Jones, HOMFLY-PT and Theta (>=3/4).
# Khovanov support and q-2t diagonal thickness are held out as outcomes.
# Exact stratified membership nulls preserve, separately for every selection
# view, its own norm bin, crossing number, alternation status and exact |sigma|.
# The baseline representative and the five randomized mirror representatives
# are all tested, with per-run checkpoints.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from consensus_hardness.conditional_nulls import (
    prepare_stratified_sampler,
    sample_stratified_hard_mask,
)


# ------------------------------------------------------------------
# 0. Preconditions and frozen design
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
        "Run the frozen notebook and Stages 18B/20 first. Missing: "
        + str(missing)
    )

OUT = Path(OUTPUT_DIR) / "20B_no_khovanov_external_thickness"
OUT.mkdir(parents=True, exist_ok=True)

STAGE18B = Path(OUTPUT_DIR) / "18B_mirror_representation_robustness"
STAGE20 = Path(OUTPUT_DIR) / "20_mathematical_phenotype"

N = len(meta)
N_REPS = 5_000
NO_KHOVANOV = tuple(x for x in INVARIANTS if x != "Khovanov")
if len(NO_KHOVANOV) != 4:
    raise AssertionError(NO_KHOVANOV)

RANDOM_SEEDS = (20260830, 20260831, 20260832, 20260833, 20260834)
RUN_LABELS = ("baseline_canonical",) + tuple(
    f"random_seed_{seed}" for seed in RANDOM_SEEDS
)


# ------------------------------------------------------------------
# 1. Load the Stage-20 outcome atlas if it is not still in memory
# ------------------------------------------------------------------
if "phenotype" not in globals():
    phenotype_path = STAGE20 / "complete_mathematical_phenotype_atlas.parquet"
    if not phenotype_path.exists():
        raise FileNotFoundError(
            "Run stage20_mathematical_phenotype_tests.py first: "
            f"{phenotype_path}"
        )
    phenotype = pd.read_parquet(phenotype_path)

KH_SUPPORT_COL = "khovanov_support_size"
KH_DIAGONAL_COL = "khovanov_q_minus_2t_diagonal_count"
for column in (KH_SUPPORT_COL, KH_DIAGONAL_COL):
    if column not in phenotype:
        raise KeyError(column)

kh_support = phenotype[KH_SUPPORT_COL].to_numpy(float)
kh_diagonal = phenotype[KH_DIAGONAL_COL].to_numpy(float)


# ------------------------------------------------------------------
# 2. Load per-view conditional masks and norm bins
# ------------------------------------------------------------------
def baseline_payload():
    masks = ch.hard_sets_to_masks(conditional_hard_sets_100, N)
    bins = {
        name: np.asarray(norm_bins_100[name], dtype=np.int32)
        for name in INVARIANTS
    }
    return masks, bins


def randomized_payload(label):
    path = STAGE18B / f"orientation_{label}_checkpoint.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    masks = {}
    bins = {}
    with np.load(path, allow_pickle=False) as payload:
        for name in INVARIANTS:
            key = ch.safe_name(name)
            masks[name] = np.asarray(payload[f"cond_{key}"], dtype=bool)
            bins[name] = np.asarray(payload[f"normbin_{key}"], dtype=np.int32)
    return masks, bins


def load_run(label):
    if label == "baseline_canonical":
        return baseline_payload()
    return randomized_payload(label)


# ------------------------------------------------------------------
# 3. Outcomes and one-sided upper-tail null
# ------------------------------------------------------------------
METRICS = (
    "kh_diagonal_mean",
    "kh_diagonal_median",
    "kh_diagonal_ge_3_prop",
    "kh_diagonal_ge_4_prop",
    "kh_support_mean",
    "kh_support_median",
)
PRIMARY_METRICS = (
    "kh_diagonal_mean",
    "kh_diagonal_ge_3_prop",
    "kh_support_mean",
)


def outcome_metrics(selected_mask):
    idx = np.flatnonzero(selected_mask)
    if len(idx) == 0:
        return {"n": 0, **{metric: np.nan for metric in METRICS}}
    diagonal = kh_diagonal[idx]
    support = kh_support[idx]
    return {
        "n": len(idx),
        "kh_diagonal_mean": float(np.mean(diagonal)),
        "kh_diagonal_median": float(np.median(diagonal)),
        "kh_diagonal_ge_3_prop": float(np.mean(diagonal >= 3)),
        "kh_diagonal_ge_4_prop": float(np.mean(diagonal >= 4)),
        "kh_support_mean": float(np.mean(support)),
        "kh_support_median": float(np.median(support)),
    }


def no_khovanov_consensus(hard_masks):
    count = np.zeros(N, dtype=np.uint8)
    for name in NO_KHOVANOV:
        count += hard_masks[name]
    return count >= 3


def summarize_against_null(label, observed, null):
    rows = []
    for metric in METRICS:
        values = null[metric].dropna().to_numpy(float)
        value = float(observed[metric])
        exceedances = int(np.sum(values >= value))
        rows.append({
            "run": label,
            "metric": metric,
            "endpoint_role": (
                "primary" if metric in PRIMARY_METRICS else "secondary"
            ),
            "observed": value,
            "null_mean": float(np.mean(values)),
            "null_sd": float(np.std(values, ddof=1)),
            "null_q95": float(np.quantile(values, 0.95)),
            "null_q99": float(np.quantile(values, 0.99)),
            "n_ge_observed": exceedances,
            "n_valid_null": len(values),
            "empirical_p": (1 + exceedances) / (1 + len(values)),
        })
    return pd.DataFrame(rows)


def run_one(label, null_seed):
    print("\n" + "=" * 78)
    print("No-Khovanov -> external Khovanov thickness:", label)
    print("=" * 78)

    hard_masks, norm_bins = load_run(label)
    selected = no_khovanov_consensus(hard_masks)
    observed = outcome_metrics(selected)
    observed_row = {"run": label, **observed}

    strata = ch.build_joint_strata_codes(
        norm_bins_by_invariant={
            name: norm_bins[name] for name in NO_KHOVANOV
        },
        meta=meta.assign(signature_abs=meta["signature"].abs()),
        extra_cols=(
            "number_of_crossings",
            "is_alternating",
            "signature_abs",
        ),
    )
    samplers = {
        name: prepare_stratified_sampler(strata[name], hard_masks[name])
        for name in NO_KHOVANOV
    }

    raw_path = OUT / f"{label}_external_khovanov_null_{N_REPS}.parquet"
    if raw_path.exists():
        print("Loading checkpoint:", raw_path.name)
        null = pd.read_parquet(raw_path)
    else:
        rng = np.random.default_rng(null_seed)
        rows = []
        for replicate in range(N_REPS):
            count = np.zeros(N, dtype=np.uint8)
            for name in NO_KHOVANOV:
                count += sample_stratified_hard_mask(
                    samplers[name], rng, N
                )
            row = {"replicate": replicate}
            row.update(outcome_metrics(count >= 3))
            rows.append(row)
            if (replicate + 1) % 500 == 0:
                print(f"{label}: {replicate + 1:,} / {N_REPS:,}")
        null = pd.DataFrame(rows)
        null.to_parquet(raw_path, index=False)
        null.to_csv(raw_path.with_suffix(".csv"), index=False)

    summary = summarize_against_null(label, observed, null)
    pd.DataFrame([observed_row]).to_csv(
        OUT / f"{label}_observed.csv", index=False
    )
    summary.to_csv(OUT / f"{label}_summary.csv", index=False)
    display(pd.DataFrame([observed_row]))
    display(summary)
    return observed_row, summary


# ------------------------------------------------------------------
# 4. Baseline and five mirror-randomized selections
# ------------------------------------------------------------------
observed_rows = []
summary_frames = []
for offset, label in enumerate(RUN_LABELS):
    observed, summary = run_one(label, null_seed=20261000 + offset)
    observed_rows.append(observed)
    summary_frames.append(summary)

no_khovanov_external_observed = pd.DataFrame(observed_rows)
no_khovanov_external_null_summary = pd.concat(
    summary_frames, ignore_index=True
)

# Report both a correction across the three predeclared primary outcomes and
# a conservative correction across every displayed outcome/run combination.
no_khovanov_external_null_summary["q_by_all"] = multipletests(
    no_khovanov_external_null_summary["empirical_p"].to_numpy(float),
    method="fdr_by",
)[1]
no_khovanov_external_null_summary["p_holm_all"] = multipletests(
    no_khovanov_external_null_summary["empirical_p"].to_numpy(float),
    method="holm",
)[1]

primary = no_khovanov_external_null_summary["endpoint_role"].eq("primary")
no_khovanov_external_null_summary["q_by_primary"] = np.nan
no_khovanov_external_null_summary["p_holm_primary"] = np.nan
no_khovanov_external_null_summary.loc[primary, "q_by_primary"] = (
    multipletests(
        no_khovanov_external_null_summary.loc[
            primary, "empirical_p"
        ].to_numpy(float),
        method="fdr_by",
    )[1]
)
no_khovanov_external_null_summary.loc[primary, "p_holm_primary"] = (
    multipletests(
        no_khovanov_external_null_summary.loc[
            primary, "empirical_p"
        ].to_numpy(float),
        method="holm",
    )[1]
)


# ------------------------------------------------------------------
# 5. Decision table across randomized representatives
# ------------------------------------------------------------------
random_summary = no_khovanov_external_null_summary.loc[
    no_khovanov_external_null_summary["run"].str.startswith("random_seed_")
].copy()

no_khovanov_external_decision = (
    random_summary
    .groupby(["metric", "endpoint_role"], as_index=False)
    .agg(
        observed_min=("observed", "min"),
        observed_mean=("observed", "mean"),
        observed_max=("observed", "max"),
        null_mean_min=("null_mean", "min"),
        null_mean_max=("null_mean", "max"),
        empirical_p_min=("empirical_p", "min"),
        empirical_p_max=("empirical_p", "max"),
        q_by_primary_max=("q_by_primary", "max"),
        p_holm_primary_max=("p_holm_primary", "max"),
        q_by_all_max=("q_by_all", "max"),
        p_holm_all_max=("p_holm_all", "max"),
        n_random_seeds=("run", "nunique"),
    )
)
no_khovanov_external_decision["all_five_raw_p_lt_0_05"] = (
    no_khovanov_external_decision["empirical_p_max"] < 0.05
)
no_khovanov_external_decision["all_five_primary_holm_lt_0_05"] = (
    no_khovanov_external_decision["p_holm_primary_max"] < 0.05
)

display(no_khovanov_external_observed)
display(no_khovanov_external_null_summary)
display(no_khovanov_external_decision)

no_khovanov_external_observed.to_csv(
    OUT / "no_khovanov_external_observed_all_runs.csv", index=False
)
no_khovanov_external_null_summary.to_csv(
    OUT / "no_khovanov_external_null_summary.csv", index=False
)
no_khovanov_external_decision.to_csv(
    OUT / "no_khovanov_external_decision.csv", index=False
)

print("\nSaved Stage 20B to:")
print(OUT)

