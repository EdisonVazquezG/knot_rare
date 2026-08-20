# %% [markdown]
# Stage 26 — Knot-level bootstrap confidence intervals
#
# This stage adds uncertainty intervals to the paper's principal descriptive
# effects without refitting the frozen anomaly models.  A bootstrap row is one
# canonical knot: all invariant views and mathematical outcomes travel
# together.  The intervals quantify empirical stability under reweightings of
# the observed finite atlas; they do not turn the complete knot census into a
# random sample from an unspecified super-population.
#
# Run in Colab after Stages 19, 20, 22, 23, 23B and 25:
#   %run /content/drive/MyDrive/consensus_hardness_refactored/notebooks/\
#       stage26_bootstrap_confidence_intervals.py

# %%
from __future__ import annotations
import os
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
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
# 0. Frozen paths and predeclared bootstrap design
# ---------------------------------------------------------------------------
DEFAULT_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants/"
    "processed_consensus_hardness/corrected_run_20260819"
)
ROOT = Path(globals().get("OUTPUT_DIR", DEFAULT_ROOT))
if not ROOT.exists():
    raise FileNotFoundError(f"Frozen run not found: {ROOT}")

OUT = ROOT / "26_bootstrap_confidence_intervals"
OUT.mkdir(parents=True, exist_ok=True)

N_REPS = int(os.environ.get("STAGE26_BOOTSTRAP_REPS", "5000"))
SEED = int(os.environ.get("STAGE26_BOOTSTRAP_SEED", "20261226"))
CONFIDENCE = float(os.environ.get("STAGE26_CI_LEVEL", "0.95"))
ID_COL = "knot_id_base"
INVARIANTS = ("Alexander", "Jones", "HOMFLY-PT", "Theta", "Khovanov")
NO_KHOVANOV = tuple(name for name in INVARIANTS if name != "Khovanov")


def find_one(filename: str, required: bool = True) -> Path | None:
    found = sorted(ROOT.rglob(filename))
    if found:
        return found[0]
    if required:
        raise FileNotFoundError(f"Could not find {filename} below {ROOT}")
    return None


def as_bool(series: pd.Series) -> pd.Series:
    """Parse bool-like checkpoint columns without treating 'False' as true."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.fillna(False).astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    unexpected = sorted(set(normalized) - allowed)
    if unexpected:
        raise ValueError(f"Unexpected boolean values: {unexpected[:5]}")
    return normalized.isin({"true", "1"})


# ---------------------------------------------------------------------------
# 1. One aligned row-level atlas
# ---------------------------------------------------------------------------
atlas = pd.read_parquet(find_one("final_hard_regime_atlas.parquet"))
phenotype = pd.read_parquet(find_one("complete_mathematical_phenotype_atlas.parquet"))
for frame_name, frame in (("atlas", atlas), ("phenotype", phenotype)):
    if ID_COL not in frame:
        raise KeyError(f"{frame_name} lacks {ID_COL}")
    frame[ID_COL] = frame[ID_COL].astype(str)
    if frame[ID_COL].duplicated().any():
        raise AssertionError(f"Duplicated knot IDs in {frame_name}")

S_COL = next(
    (name for name in ("s_invariant_qc", "s_invariant", "s") if name in atlas),
    None,
)
KH_DIAGONAL_COL = next(
    (
        name
        for name in (
            "khovanov_q_minus_2t_diagonal_count",
            "kh_diagonal_count",
            "khovanov_diagonal_count",
        )
        if name in phenotype
    ),
    None,
)
KH_SUPPORT_COL = next(
    (name for name in ("khovanov_support_size", "kh_support_size") if name in phenotype),
    None,
)
if S_COL is None or KH_DIAGONAL_COL is None or KH_SUPPORT_COL is None:
    raise KeyError("Required concordance or Khovanov phenotype columns are missing")

outcomes = phenotype[[ID_COL, KH_DIAGONAL_COL, KH_SUPPORT_COL]].copy()
data = atlas.merge(outcomes, on=ID_COL, how="left", validate="one_to_one")
required_cols = (
    "number_of_crossings",
    "is_alternating",
    "signature",
    S_COL,
    KH_DIAGONAL_COL,
    KH_SUPPORT_COL,
)
if data[list(required_cols)].isna().any().any():
    raise RuntimeError("Missing values in the aligned bootstrap outcome table")
data = data.set_index(ID_COL, drop=False)


METRIC_ORDER = (
    "abs_gap_positive_prop",
    "mean_abs_gap",
    "abs_gap_ge_4_prop",
    "kh_diagonal_mean",
    "kh_diagonal_ge_3_prop",
    "kh_diagonal_ge_4_prop",
    "kh_support_mean",
    "crossing_15_prop",
    "alternating_prop",
)


def population_metrics(frame: pd.DataFrame, idx: np.ndarray) -> dict[str, float]:
    signature = frame["signature"].to_numpy(float)[idx]
    s_value = frame[S_COL].to_numpy(float)[idx]
    nonalternating = frame["is_alternating"].to_numpy(int)[idx] == 0
    gap = np.abs(s_value - signature)
    gap = gap[nonalternating]
    diagonal = frame[KH_DIAGONAL_COL].to_numpy(float)[idx]
    support = frame[KH_SUPPORT_COL].to_numpy(float)[idx]
    crossings = frame["number_of_crossings"].to_numpy(int)[idx]
    alternating = frame["is_alternating"].to_numpy(int)[idx]
    return {
        "abs_gap_positive_prop": float(np.mean(gap > 0)) if len(gap) else np.nan,
        "mean_abs_gap": float(np.mean(gap)) if len(gap) else np.nan,
        "abs_gap_ge_4_prop": float(np.mean(gap >= 4)) if len(gap) else np.nan,
        "kh_diagonal_mean": float(np.mean(diagonal)),
        "kh_diagonal_ge_3_prop": float(np.mean(diagonal >= 3)),
        "kh_diagonal_ge_4_prop": float(np.mean(diagonal >= 4)),
        "kh_support_mean": float(np.mean(support)),
        "crossing_15_prop": float(np.mean(crossings == 15)),
        "alternating_prop": float(np.mean(alternating == 1)),
    }


def bootstrap_population(
    frame: pd.DataFrame,
    labels: dict[str, object],
    seed: int,
) -> list[dict[str, object]]:
    """Return one CI row per metric for a fixed selected knot population."""

    frame = frame.reset_index(drop=True)
    n = len(frame)
    if n == 0:
        raise ValueError(f"Empty selected population: {labels}")
    observed = population_metrics(frame, np.arange(n, dtype=np.int64))
    draws = {metric: np.empty(N_REPS, dtype=float) for metric in METRIC_ORDER}
    rng = np.random.default_rng(seed)
    for replicate in range(N_REPS):
        idx = rng.integers(0, n, size=n)
        values = population_metrics(frame, idx)
        for metric in METRIC_ORDER:
            draws[metric][replicate] = values[metric]

    rows = []
    for metric in METRIC_ORDER:
        rows.append(
            {
                **labels,
                "metric": metric,
                "n_selected": n,
                "bootstrap_unit": "canonical knot",
                "inference_target": "finite-atlas empirical stability",
                **summarize_bootstrap(
                    observed[metric],
                    draws[metric],
                    confidence=CONFIDENCE,
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 2. Score-selected populations used in Figures 1 and 2
# ---------------------------------------------------------------------------
population_rows: list[dict[str, object]] = []
selection_sources: list[tuple[str, pd.DataFrame, tuple[str, ...]]] = []

stage23_ids = pd.read_csv(find_one("score_selected_test_ids.csv"), dtype={ID_COL: str})
selection_sources.append(("stage23_standard_scores", stage23_ids, ("family", "method")))

stage23b_path = find_one("size_matched_selected_ids.csv", required=False)
if stage23b_path is not None:
    stage23b_ids = pd.read_csv(stage23b_path, dtype={ID_COL: str})
    selection_sources.append(("stage23B_equal_size_scores", stage23b_ids, ("family", "method")))

seed_offset = 0
for source, selected_ids, group_cols in selection_sources:
    unknown = sorted(set(selected_ids[ID_COL]) - set(data.index))
    if unknown:
        raise KeyError(f"Unknown selected IDs in {source}, e.g. {unknown[:5]}")
    for keys, group in selected_ids.groupby(list(group_cols), sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ids = pd.Index(group[ID_COL].astype(str).unique())
        labels = {"source": source, **dict(zip(group_cols, keys))}
        population_rows.extend(
            bootstrap_population(data.loc[ids], labels, SEED + seed_offset)
        )
        seed_offset += 1


# ---------------------------------------------------------------------------
# 3. Full-atlas no-Khovanov conditional population used in Figure 6
# ---------------------------------------------------------------------------
hard_path = find_one("conditional_100bins_hard_sets.csv")
hard = pd.read_csv(hard_path, dtype={ID_COL: str})
if not {"invariant", ID_COL}.issubset(hard.columns):
    raise KeyError(f"Unexpected hard-set schema: {hard.columns.tolist()}")
no_kh_counts = (
    hard.loc[hard["invariant"].isin(NO_KHOVANOV)]
    .drop_duplicates(["invariant", ID_COL])
    .groupby(ID_COL)["invariant"]
    .nunique()
)
no_kh_ids = pd.Index(no_kh_counts.index[no_kh_counts >= 3])
population_rows.extend(
    bootstrap_population(
        data.loc[no_kh_ids],
        {
            "source": "stage22_no_khovanov_full_atlas",
            "family": "No Khovanov >=3/4",
            "method": "conditional_percentile_100",
        },
        SEED + seed_offset,
    )
)
seed_offset += 1


# ---------------------------------------------------------------------------
# 4. Held-out conditional populations used in Figure 5, when available
# ---------------------------------------------------------------------------
heldout_path = find_one("conditional_heldout_members.csv", required=False)
if heldout_path is not None:
    heldout = pd.read_csv(heldout_path, dtype={ID_COL: str})
    if ID_COL not in heldout:
        raise KeyError(f"Held-out member table lacks {ID_COL}")
    member_cols = [
        col
        for col in heldout
        if col.endswith(("_pca_conditional", "_ae_majority", "_frozen_test"))
    ]
    for col in member_cols:
        mask = as_bool(heldout[col])
        ids = pd.Index(heldout.loc[mask, ID_COL].astype(str).unique())
        population_rows.extend(
            bootstrap_population(
                data.loc[ids],
                {"source": "stage19_heldout", "member_set": col},
                SEED + seed_offset,
            )
        )
        seed_offset += 1


population_cis = pd.DataFrame(population_rows)
population_cis.to_csv(OUT / "bootstrap_population_cis.csv", index=False)


# ---------------------------------------------------------------------------
# 5. Pairwise Jaccard intervals for standard and equal-size score sets
# ---------------------------------------------------------------------------
jaccard_rows = []
for source, selected_ids, group_cols in selection_sources:
    if "method" not in group_cols:
        continue
    family_cols = tuple(col for col in group_cols if col != "method")
    grouped = (
        selected_ids.groupby(list(family_cols), sort=False)
        if family_cols
        else [((), selected_ids)]
    )
    for family_keys, family_group in grouped:
        family_keys = family_keys if isinstance(family_keys, tuple) else (family_keys,)
        family_labels = dict(zip(family_cols, family_keys))
        methods = list(dict.fromkeys(family_group["method"].astype(str)))
        id_sets = {
            method: set(family_group.loc[family_group["method"].eq(method), ID_COL].astype(str))
            for method in methods
        }
        for pair_offset, (method_a, method_b) in enumerate(combinations(methods, 2)):
            a, b = id_sets[method_a], id_sets[method_b]
            overlap = len(a & b)
            only_a = len(a - b)
            only_b = len(b - a)
            union = overlap + only_a + only_b
            observed = overlap / union if union else np.nan
            draws = multinomial_jaccard_bootstrap(
                overlap,
                only_a,
                only_b,
                n_reps=N_REPS,
                seed=SEED + 10_000 + pair_offset + len(jaccard_rows),
            )
            jaccard_rows.append(
                {
                    "source": source,
                    **family_labels,
                    "method_a": method_a,
                    "method_b": method_b,
                    "size_a": len(a),
                    "size_b": len(b),
                    "overlap": overlap,
                    "union": union,
                    "metric": "jaccard",
                    "bootstrap_unit": "knot category within observed union",
                    **summarize_bootstrap(observed, draws, confidence=CONFIDENCE),
                }
            )

jaccard_cis = pd.DataFrame(jaccard_rows)
jaccard_cis.to_csv(OUT / "bootstrap_jaccard_cis.csv", index=False)


# ---------------------------------------------------------------------------
# 6. Paired bootstrap for matched geometric effects in Figure 7
# ---------------------------------------------------------------------------
paired_path = find_one("score_geometry_pairs_with_outcomes.csv", required=False)
paired_rows = []
if paired_path is not None:
    paired = pd.read_csv(paired_path)
    geometry_specs = {
        "hyperbolic_volume_mean_difference": (
            "selected_volume",
            "control_volume",
            "continuous",
        ),
        "tetrahedra_mean_difference": (
            "selected_num_tetrahedra",
            "control_num_tetrahedra",
            "continuous",
        ),
        "nontrivial_symmetry_risk_difference": (
            "selected_nontrivial_symmetry",
            "control_nontrivial_symmetry",
            "binary",
        ),
    }
    for method_offset, (method, group) in enumerate(paired.groupby("method", sort=False)):
        for metric_offset, (metric, (x_col, y_col, outcome_type)) in enumerate(
            geometry_specs.items()
        ):
            x = group[x_col].to_numpy(float)
            y = group[y_col].to_numpy(float)
            valid = np.isfinite(x) & np.isfinite(y)
            difference = x[valid] - y[valid]
            if not len(difference):
                continue
            observed = float(np.mean(difference))
            rng = np.random.default_rng(SEED + 20_000 + 100 * method_offset + metric_offset)
            draws = np.empty(N_REPS, dtype=float)
            for replicate in range(N_REPS):
                idx = rng.integers(0, len(difference), size=len(difference))
                draws[replicate] = float(np.mean(difference[idx]))
            paired_rows.append(
                {
                    "source": "stage25_matched_geometry",
                    "method": method,
                    "metric": metric,
                    "outcome_type": outcome_type,
                    "n_pairs": len(difference),
                    "bootstrap_unit": "matched selected-control pair",
                    **summarize_bootstrap(observed, draws, confidence=CONFIDENCE),
                }
            )

paired_cis = pd.DataFrame(paired_rows)
paired_cis.to_csv(OUT / "bootstrap_paired_geometry_cis.csv", index=False)


# ---------------------------------------------------------------------------
# 7. Compact critical table and run manifest
# ---------------------------------------------------------------------------
critical = population_cis.loc[
    population_cis["source"].eq("stage22_no_khovanov_full_atlas")
    & population_cis["metric"].isin(
        (
            "kh_diagonal_mean",
            "kh_diagonal_ge_3_prop",
            "kh_diagonal_ge_4_prop",
            "kh_support_mean",
        )
    )
].copy()
critical.to_csv(OUT / "bootstrap_no_khovanov_critical_cis.csv", index=False)

manifest = pd.DataFrame(
    [
        {
            "n_bootstrap_reps": N_REPS,
            "seed": SEED,
            "ci_level": CONFIDENCE,
            "ci_method": "percentile",
            "bootstrap_unit": "canonical knot; matched pair for geometry",
            "models_refit": False,
            "claim_scope": "empirical stability within the finite knot atlas",
        }
    ]
)
manifest.to_csv(OUT / "bootstrap_manifest.csv", index=False)

print("Critical no-Khovanov bootstrap intervals:")
print(
    critical[
        ["metric", "n_selected", "observed", "bootstrap_se", "ci_low", "ci_high"]
    ].to_string(index=False)
)
print(f"\nSaved Stage 26 to: {OUT}")
