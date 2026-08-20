# %% [markdown]
# Stage 27 — Coarse Khovanov-norm-bin sensitivity for the out-of-view null
#
# The Stage-22 out-of-view null preserves a 100-bin Khovanov norm together
# with each selection view's own norm bin, crossing number, alternation and
# exact |signature|.  This can create very small permutation cells.  Here the
# observed no-Khovanov selection is kept frozen while adjacent Khovanov norm
# bins are collapsed.  The null is rerun at each resolution and reports both
# inferential results and the exact cell-size/mobility diagnostics requested
# in review.
#
# Run after Stage 22:
#   %run /content/drive/MyDrive/consensus_hardness_refactored/notebooks/\
#       stage27_khovanov_norm_bin_null_sensitivity.py

# %%
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

#from consensus_hardness.bootstrap import coarsen_ordered_bins
from collections.abc import Callable




def percentile_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a two-sided percentile interval after dropping non-finite draws."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, (alpha, 1.0 - alpha))
    return float(low), float(high)


def bootstrap_rows(
    n_rows: int,
    statistic: Callable[[np.ndarray], float],
    *,
    n_reps: int = 2_000,
    seed: int = 42,
) -> np.ndarray:
    """Bootstrap a row-index statistic by sampling rows with replacement."""

    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    if n_reps < 1:
        raise ValueError("n_reps must be positive")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_reps, dtype=float)
    for replicate in range(n_reps):
        idx = rng.integers(0, n_rows, size=n_rows)
        draws[replicate] = statistic(idx)
    return draws


def summarize_bootstrap(
    observed: float,
    draws: np.ndarray,
    *,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Summarize bootstrap draws with a percentile CI and bootstrap SE."""

    draws = np.asarray(draws, dtype=float)
    valid = draws[np.isfinite(draws)]
    low, high = percentile_interval(valid, confidence=confidence)
    return {
        "observed": float(observed),
        "bootstrap_mean": float(np.mean(valid)) if len(valid) else np.nan,
        "bootstrap_se": (
            float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan
        ),
        "ci_level": float(confidence),
        "ci_low": low,
        "ci_high": high,
        "n_valid_bootstrap": int(len(valid)),
    }


def multinomial_jaccard_bootstrap(
    overlap: int,
    only_a: int,
    only_b: int,
    *,
    n_reps: int = 2_000,
    seed: int = 42,
) -> np.ndarray:
    """Bootstrap Jaccard from the three categories inside the observed union."""

    counts = np.asarray([overlap, only_a, only_b], dtype=int)
    if np.any(counts < 0):
        raise ValueError("Jaccard category counts must be non-negative")
    union = int(counts.sum())
    if union == 0:
        return np.full(n_reps, np.nan)
    rng = np.random.default_rng(seed)
    sampled = rng.multinomial(union, counts / union, size=n_reps)
    return sampled[:, 0] / sampled.sum(axis=1)


def coarsen_ordered_bins(
    bin_codes: np.ndarray,
    target_bins: int,
) -> np.ndarray:
    """Collapse adjacent ordered bin codes into at most ``target_bins`` bins.

    The original conditional scores use ordered, approximately equal-frequency
    quantile bins.  Collapsing adjacent codes therefore gives a deterministic
    coarser quantile partition without needing to reload the raw norm values.
    """

    codes = np.asarray(bin_codes)
    if codes.ndim != 1:
        raise ValueError("bin_codes must be one-dimensional")
    if target_bins < 1:
        raise ValueError("target_bins must be positive")
    unique = np.unique(codes)
    if not len(unique):
        return np.empty(0, dtype=np.int32)
    rank = np.searchsorted(unique, codes)
    n_out = min(int(target_bins), len(unique))
    collapsed = np.floor(rank * n_out / len(unique)).astype(np.int32)
    return np.minimum(collapsed, n_out - 1)

# ---------------------------------------------------------------------------
# 0. Frozen design
# ---------------------------------------------------------------------------
DEFAULT_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants/"
    "processed_consensus_hardness/corrected_run_20260819"
)
ROOT = Path(globals().get("OUTPUT_DIR", DEFAULT_ROOT))
if not ROOT.exists():
    raise FileNotFoundError(f"Frozen run not found: {ROOT}")

OUT = ROOT / "27_khovanov_norm_bin_null_sensitivity"
FIG = OUT / "figures"
for directory in (OUT, FIG):
    directory.mkdir(parents=True, exist_ok=True)

N_REPS = int(os.environ.get("STAGE27_NULL_REPS", "5000"))
SEED = int(os.environ.get("STAGE27_NULL_SEED", "20261227"))
TARGET_BINS = tuple(
    int(value)
    for value in os.environ.get("STAGE27_KH_NORM_BINS", "100,50,25,20,10,5,1").split(",")
)
if not TARGET_BINS or any(value < 1 for value in TARGET_BINS):
    raise ValueError("STAGE27_KH_NORM_BINS must contain positive integers")
TARGET_BINS = tuple(dict.fromkeys(TARGET_BINS))

INVARIANTS = ("Alexander", "Jones", "HOMFLY-PT", "Theta", "Khovanov")
NO_KHOVANOV = tuple(name for name in INVARIANTS if name != "Khovanov")
ID_COL = "knot_id_base"


def safe_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "_")
    )


def find_one(filename: str) -> Path:
    found = sorted(ROOT.rglob(filename))
    if not found:
        raise FileNotFoundError(f"Could not find {filename} below {ROOT}")
    return found[0]


# ---------------------------------------------------------------------------
# 1. Align frozen selection, Khovanov outcomes and norm-bin codes
# ---------------------------------------------------------------------------
atlas = pd.read_parquet(find_one("final_hard_regime_atlas.parquet")).reset_index(drop=True)
phenotype = pd.read_parquet(
    find_one("complete_mathematical_phenotype_atlas.parquet")
).reset_index(drop=True)
for frame_name, frame in (("atlas", atlas), ("phenotype", phenotype)):
    if ID_COL not in frame:
        raise KeyError(f"{frame_name} lacks {ID_COL}")
    frame[ID_COL] = frame[ID_COL].astype(str)
    if frame[ID_COL].duplicated().any():
        raise AssertionError(f"Duplicated IDs in {frame_name}")

KH_DIAGONAL_COL = next(
    name
    for name in (
        "khovanov_q_minus_2t_diagonal_count",
        "kh_diagonal_count",
        "khovanov_diagonal_count",
    )
    if name in phenotype
)
KH_SUPPORT_COL = next(
    name
    for name in ("khovanov_support_size", "kh_support_size")
    if name in phenotype
)

data = atlas.merge(
    phenotype[[ID_COL, KH_DIAGONAL_COL, KH_SUPPORT_COL]],
    on=ID_COL,
    how="left",
    validate="one_to_one",
)
if data[[KH_DIAGONAL_COL, KH_SUPPORT_COL]].isna().any().any():
    raise RuntimeError("Missing Khovanov outcomes after atlas alignment")
N = len(data)
kh_diagonal = data[KH_DIAGONAL_COL].to_numpy(float)
kh_support = data[KH_SUPPORT_COL].to_numpy(float)

scores_path = find_one("conditional_100bins_scores.npz")
with np.load(scores_path, allow_pickle=False) as payload:
    norm_bins = {
        name: np.asarray(payload[f"{safe_name(name)}_norm_bin"], dtype=np.int32)
        for name in INVARIANTS
    }
for name, values in norm_bins.items():
    if len(values) != N:
        raise AssertionError(f"Norm-bin length mismatch for {name}")

hard_saved = pd.read_csv(find_one("conditional_100bins_hard_sets.csv"), dtype={ID_COL: str})
if not {"invariant", ID_COL}.issubset(hard_saved.columns):
    raise KeyError(f"Unexpected hard-set schema: {hard_saved.columns.tolist()}")
id_to_pos = pd.Series(np.arange(N, dtype=np.int64), index=data[ID_COL]).to_dict()
hard_masks: dict[str, np.ndarray] = {}
for name in INVARIANTS:
    ids = hard_saved.loc[hard_saved["invariant"].eq(name), ID_COL].astype(str)
    unknown = sorted(set(ids) - set(id_to_pos))
    if unknown:
        raise KeyError(f"Unknown IDs in {name}, e.g. {unknown[:5]}")
    mask = np.zeros(N, dtype=bool)
    mask[[id_to_pos[knot_id] for knot_id in ids]] = True
    hard_masks[name] = mask


def consensus_mask() -> np.ndarray:
    count = np.zeros(N, dtype=np.uint8)
    for name in NO_KHOVANOV:
        count += hard_masks[name]
    return count >= 3


observed_mask = consensus_mask()


def thickness_metrics(mask: np.ndarray) -> dict[str, float | int]:
    diagonal = kh_diagonal[mask]
    support = kh_support[mask]
    return {
        "n": int(mask.sum()),
        "kh_diagonal_mean": float(np.mean(diagonal)),
        "kh_diagonal_ge_3_prop": float(np.mean(diagonal >= 3)),
        "kh_diagonal_ge_4_prop": float(np.mean(diagonal >= 4)),
        "kh_support_mean": float(np.mean(support)),
    }


OBSERVED = thickness_metrics(observed_mask)
METRICS = (
    "kh_diagonal_mean",
    "kh_diagonal_ge_3_prop",
    "kh_diagonal_ge_4_prop",
    "kh_support_mean",
)


# ---------------------------------------------------------------------------
# 2. Exact per-view strata with a controllable Khovanov norm resolution
# ---------------------------------------------------------------------------
def factorized_strata(view: str, kh_norm_bins: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "view_norm_bin_100": norm_bins[view],
            "crossings": data["number_of_crossings"].astype(str).to_numpy(),
            "alternating": data["is_alternating"].astype(str).to_numpy(),
            "abs_signature": data["signature"].abs().astype(str).to_numpy(),
            "khovanov_norm_bin": kh_norm_bins,
        }
    )
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(frame), sort=False)
    return codes.astype(np.int32)


def prepare_sampler(codes: np.ndarray, selected: np.ndarray) -> tuple[dict, dict]:
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(sorted_codes))]
    stops = np.r_[starts[1:], len(order)]
    random_groups: list[tuple[np.ndarray, int]] = []
    fixed_groups: list[np.ndarray] = []
    sizes = []
    selected_counts = []
    for start, stop in zip(starts, stops):
        idx = order[start:stop]
        k = int(selected[idx].sum())
        sizes.append(len(idx))
        selected_counts.append(k)
        if k == 0:
            continue
        if k == len(idx):
            fixed_groups.append(idx)
        else:
            random_groups.append((idx, k))
    fixed_idx = (
        np.concatenate(fixed_groups)
        if fixed_groups
        else np.empty(0, dtype=np.int64)
    )
    sizes_arr = np.asarray(sizes, dtype=int)
    selected_counts_arr = np.asarray(selected_counts, dtype=int)
    diagnostic = {
        "n_strata": len(sizes_arr),
        "stratum_size_min": int(sizes_arr.min()),
        "stratum_size_q25": float(np.quantile(sizes_arr, 0.25)),
        "stratum_size_median": float(np.median(sizes_arr)),
        "stratum_size_q75": float(np.quantile(sizes_arr, 0.75)),
        "stratum_size_max": int(sizes_arr.max()),
        "singleton_strata_prop": float(np.mean(sizes_arr == 1)),
        "small_lt_10_strata_prop": float(np.mean(sizes_arr < 10)),
        "selected_total": int(selected.sum()),
        "selected_fixed": int(len(fixed_idx)),
        "selected_movable": int(sum(k for _, k in random_groups)),
        "n_random_groups": len(random_groups),
        "n_fixed_selected_groups": len(fixed_groups),
        "n_strata_with_selected": int(np.sum(selected_counts_arr > 0)),
    }
    return {"random_groups": random_groups, "fixed_idx": fixed_idx}, diagnostic


def sample_mask(sampler: dict, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(N, dtype=bool)
    mask[sampler["fixed_idx"]] = True
    for indices, k in sampler["random_groups"]:
        mask[rng.choice(indices, size=k, replace=False)] = True
    return mask


def summarize_null(target_bins: int, null: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        values = null[metric].dropna().to_numpy(float)
        observed = float(OBSERVED[metric])
        exceedances = int(np.sum(values >= observed))
        null_sd = float(np.std(values, ddof=1))
        rows.append(
            {
                "khovanov_norm_target_bins": target_bins,
                "metric": metric,
                "n_selected": int(OBSERVED["n"]),
                "observed": observed,
                "null_mean": float(np.mean(values)),
                "observed_minus_null_mean": observed - float(np.mean(values)),
                "null_sd": null_sd,
                "standardized_excess": (
                    (observed - float(np.mean(values))) / null_sd
                    if null_sd > 0
                    else np.nan
                ),
                "null_q95": float(np.quantile(values, 0.95)),
                "null_q99": float(np.quantile(values, 0.99)),
                "n_ge_observed": exceedances,
                "n_valid_null": len(values),
                "empirical_p": (1 + exceedances) / (1 + len(values)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Run the sensitivity grid
# ---------------------------------------------------------------------------
summary_parts = []
diagnostic_parts = []
source_kh_bins = norm_bins["Khovanov"]

for offset, target_bins in enumerate(TARGET_BINS):
    coarse_kh_bins = coarsen_ordered_bins(source_kh_bins, target_bins)
    effective_bins = int(np.unique(coarse_kh_bins).size)
    samplers = {}
    for view in NO_KHOVANOV:
        codes = factorized_strata(view, coarse_kh_bins)
        sampler, diagnostic = prepare_sampler(codes, hard_masks[view])
        samplers[view] = sampler
        diagnostic_parts.append(
            pd.DataFrame(
                [
                    {
                        "khovanov_norm_target_bins": target_bins,
                        "khovanov_norm_effective_bins": effective_bins,
                        "invariant": view,
                        **diagnostic,
                    }
                ]
            )
        )

    raw_path = OUT / f"no_khovanov_kh_norm_{target_bins}bins_null_{N_REPS}.parquet"
    stage22_path = (
        ROOT
        / "22_gap_thickness_dependence"
        / f"no_khovanov_external_with_kh_norm_null_{N_REPS}.parquet"
    )
    if raw_path.exists():
        print("Loading checkpoint:", raw_path.name)
        null = pd.read_parquet(raw_path)
    elif target_bins >= np.unique(source_kh_bins).size and stage22_path.exists():
        print("Reusing Stage-22 100-bin checkpoint:", stage22_path.name)
        null = pd.read_parquet(stage22_path)
    else:
        rng = np.random.default_rng(SEED + offset)
        rows = []
        for replicate in range(N_REPS):
            count = np.zeros(N, dtype=np.uint8)
            for view in NO_KHOVANOV:
                count += sample_mask(samplers[view], rng)
            row = {"replicate": replicate}
            row.update(thickness_metrics(count >= 3))
            rows.append(row)
            if (replicate + 1) % 500 == 0:
                print(
                    f"Khovanov norm bins={target_bins}: "
                    f"{replicate + 1:,}/{N_REPS:,}"
                )
        null = pd.DataFrame(rows)
        null.to_parquet(raw_path, index=False)
        null.to_csv(raw_path.with_suffix(".csv"), index=False)
    summary_parts.append(summarize_null(target_bins, null))


summary = pd.concat(summary_parts, ignore_index=True)
diagnostics = pd.concat(diagnostic_parts, ignore_index=True)

summary["p_holm_grid"] = multipletests(
    summary["empirical_p"].to_numpy(float), method="holm"
)[1]
summary["q_by_grid"] = multipletests(
    summary["empirical_p"].to_numpy(float), method="fdr_by"
)[1]
summary["exceeds_null_q99"] = summary["observed"] > summary["null_q99"]

diagnostic_summary = (
    diagnostics.groupby("khovanov_norm_target_bins", as_index=False)
    .agg(
        effective_bins=("khovanov_norm_effective_bins", "max"),
        median_cell_size_min=("stratum_size_median", "min"),
        median_cell_size_max=("stratum_size_median", "max"),
        q25_cell_size_min=("stratum_size_q25", "min"),
        singleton_prop_max=("singleton_strata_prop", "max"),
        small_lt_10_prop_max=("small_lt_10_strata_prop", "max"),
        selected_fixed_max=("selected_fixed", "max"),
        selected_movable_min=("selected_movable", "min"),
        random_groups_min=("n_random_groups", "min"),
    )
)
diagnostic_summary["median_cells_at_least_10_all_views"] = (
    diagnostic_summary["median_cell_size_min"] >= 10
)

decision = summary.merge(
    diagnostic_summary,
    on="khovanov_norm_target_bins",
    how="left",
    validate="many_to_one",
)
decision["survives_holm_and_q99"] = (
    decision["p_holm_grid"].lt(0.05) & decision["exceeds_null_q99"]
)

eligible = diagnostic_summary.loc[
    diagnostic_summary["median_cells_at_least_10_all_views"]
].sort_values("khovanov_norm_target_bins", ascending=False)
recommended_bins = int(eligible.iloc[0]["khovanov_norm_target_bins"]) if len(eligible) else None
decision["recommended_finest_grid_with_median_cells_ge_10"] = (
    decision["khovanov_norm_target_bins"].eq(recommended_bins)
    if recommended_bins is not None
    else False
)

summary.to_csv(OUT / "khovanov_norm_bin_null_sensitivity_summary.csv", index=False)
diagnostics.to_csv(OUT / "khovanov_norm_bin_strata_by_view.csv", index=False)
diagnostic_summary.to_csv(OUT / "khovanov_norm_bin_strata_summary.csv", index=False)
decision.to_csv(OUT / "khovanov_norm_bin_sensitivity_decision.csv", index=False)


# ---------------------------------------------------------------------------
# 4. Compact supplementary figure
# ---------------------------------------------------------------------------
mean_rows = decision.loc[decision["metric"].eq("kh_diagonal_mean")].copy()
order = list(TARGET_BINS)
mean_rows["order"] = mean_rows["khovanov_norm_target_bins"].map(
    {value: i for i, value in enumerate(order)}
)
mean_rows = mean_rows.sort_values("order")
x = np.arange(len(mean_rows))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
axes[0].plot(x, mean_rows["null_mean"], marker="o", color="#4C78A8", label="Null mean")
axes[0].fill_between(
    x,
    mean_rows["null_q95"].to_numpy(float),
    mean_rows["null_q99"].to_numpy(float),
    color="#4C78A8",
    alpha=0.20,
    label="Null 95th--99th percentiles",
)
axes[0].axhline(
    float(OBSERVED["kh_diagonal_mean"]),
    color="#1B9E77",
    linestyle="--",
    label="Observed",
)
axes[0].set_xticks(x, mean_rows["khovanov_norm_target_bins"].astype(str))
axes[0].set_xlabel("Khovanov norm bins in null")
axes[0].set_ylabel("Mean Khovanov diagonal count")
axes[0].set_title("A  Out-of-view effect across null resolutions", loc="left", fontweight="bold")
axes[0].legend(frameon=False, fontsize=8)

diag_plot = diagnostic_summary.set_index("khovanov_norm_target_bins").reindex(order)
axes[1].plot(
    x,
    diag_plot["median_cell_size_min"],
    marker="o",
    color="#D95F02",
    label="Minimum median cell size",
)
axes[1].axhline(10, color="0.4", linestyle="--", label="Reviewer target = 10")
axes[1].set_xticks(x, [str(value) for value in order])
axes[1].set_xlabel("Khovanov norm bins in null")
axes[1].set_ylabel("Knots per exact permutation cell")
axes[1].set_title("B  Coarsening restores permutation mobility", loc="left", fontweight="bold")
axes[1].legend(frameon=False, fontsize=8)

fig.suptitle("Khovanov-norm-bin sensitivity of the held-out thickness null", fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "khovanov_norm_bin_null_sensitivity.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "khovanov_norm_bin_null_sensitivity.pdf", bbox_inches="tight")
plt.show()

print("\nCell-size diagnostics:")
print(diagnostic_summary.to_string(index=False))
print("\nMean-diagonal sensitivity:")
print(
    mean_rows[
        [
            "khovanov_norm_target_bins",
            "observed",
            "null_mean",
            "observed_minus_null_mean",
            "null_sd",
            "standardized_excess",
            "empirical_p",
            "p_holm_grid",
            "median_cell_size_min",
            "singleton_prop_max",
            "selected_fixed_max",
        ]
    ].to_string(index=False)
)
if recommended_bins is None:
    print("\nWARNING: no tested grid achieved median cell size >=10 in every view.")
else:
    print(
        "\nFinest tested grid with median cell size >=10 in every view: "
        f"{recommended_bins} Khovanov norm bins"
    )
print(f"\nSaved Stage 27 to: {OUT}")
