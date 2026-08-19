# %% [markdown]
# Stage 22 — Is the mirror-invariant concordance gap explained by
# Khovanov thickness?
#
# This stage is standalone: it reads only frozen artifacts from disk.
# It performs four audits:
#   1. cross-tabulate G=|s-sigma| against Khovanov q-2t thickness;
#   2. list every G>0 knot supported on at most two diagonals (audit, not an
#      automatic data-error assertion);
#   3. rerun the all-five conditional exact null after adding Khovanov
#      thickness to every view-specific stratum;
#   4. rerun the no-Khovanov external-thickness null while additionally
#      preserving the Khovanov norm bin.
#
# Run in Colab with:
#   %run /content/drive/MyDrive/consensus_hardness_refactored/notebooks/stage22_gap_thickness_dependence.py

# %%
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# 0. Frozen paths and settings
# ---------------------------------------------------------------------------
DEFAULT_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants/"
    "processed_consensus_hardness/corrected_run_20260819"
)
ROOT = Path(globals().get("OUTPUT_DIR", DEFAULT_ROOT))
if not ROOT.exists():
    raise FileNotFoundError(
        f"Frozen run not found: {ROOT}. Set OUTPUT_DIR before %run."
    )

OUT = ROOT / "22_gap_thickness_dependence"
OUT.mkdir(parents=True, exist_ok=True)

N_REPS = int(os.environ.get("STAGE22_N_REPS", "5000"))
SEED = int(os.environ.get("STAGE22_SEED", "20261122"))
INVARIANTS = ("Alexander", "Jones", "HOMFLY-PT", "Theta", "Khovanov")
NO_KHOVANOV = tuple(x for x in INVARIANTS if x != "Khovanov")
ID_COL = "knot_id_base"


def safe_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "_")
    )


def first_existing(candidates: list[Path], recursive_name: str | None = None) -> Path:
    for path in candidates:
        if path.exists():
            return path
    if recursive_name is not None:
        found = sorted(ROOT.rglob(recursive_name))
        if found:
            return found[0]
    raise FileNotFoundError(
        "Missing frozen artifact. Tried:\n" + "\n".join(map(str, candidates))
    )


ATLAS_PATH = first_existing(
    [
        ROOT / "17_final_paper_outputs" / "final_hard_regime_atlas.parquet",
        ROOT / "13_hard_regime_candidates" / "complete_hard_regime_atlas.parquet",
        ROOT / "12_hard_regime_atlas" / "hard_regime_atlas.parquet",
    ],
    "final_hard_regime_atlas.parquet",
)
PHENOTYPE_PATH = first_existing(
    [ROOT / "20_mathematical_phenotype" / "complete_mathematical_phenotype_atlas.parquet"],
    "complete_mathematical_phenotype_atlas.parquet",
)
SCORES_PATH = first_existing([], "conditional_100bins_scores.npz")
HARD_SETS_PATH = first_existing([], "conditional_100bins_hard_sets.csv")

print("Atlas:", ATLAS_PATH)
print("Phenotype:", PHENOTYPE_PATH)
print("Conditional scores:", SCORES_PATH)
print("Conditional hard sets:", HARD_SETS_PATH)


# ---------------------------------------------------------------------------
# 1. Load and align frozen data
# ---------------------------------------------------------------------------
atlas = pd.read_parquet(ATLAS_PATH).reset_index(drop=True)
phenotype = pd.read_parquet(PHENOTYPE_PATH).reset_index(drop=True)

for frame_name, frame in (("atlas", atlas), ("phenotype", phenotype)):
    if ID_COL not in frame:
        raise KeyError(f"{frame_name} lacks {ID_COL}")
    if frame[ID_COL].duplicated().any():
        raise AssertionError(f"Duplicated IDs in {frame_name}")
    frame[ID_COL] = frame[ID_COL].astype(str)

S_COL = next(
    (c for c in ("s_invariant_qc", "s_invariant", "s") if c in atlas),
    None,
)
if S_COL is None:
    raise KeyError("Rasmussen invariant column not found in atlas")

KH_DIAGONAL_COL = next(
    (
        c
        for c in (
            "khovanov_q_minus_2t_diagonal_count",
            "kh_diagonal_count",
            "khovanov_diagonal_count",
        )
        if c in phenotype
    ),
    None,
)
KH_SUPPORT_COL = next(
    (
        c
        for c in (
            "khovanov_support_size",
            "kh_support_size",
        )
        if c in phenotype
    ),
    None,
)
if KH_DIAGONAL_COL is None or KH_SUPPORT_COL is None:
    raise KeyError("Khovanov thickness/support columns not found in phenotype atlas")

needed = phenotype[[ID_COL, KH_DIAGONAL_COL, KH_SUPPORT_COL]].copy()
data = atlas.merge(needed, on=ID_COL, how="left", validate="one_to_one")
if len(data) != len(atlas):
    raise AssertionError("Atlas/phenotype merge changed row count")
if data[[KH_DIAGONAL_COL, KH_SUPPORT_COL]].isna().any().any():
    bad = data.loc[data[KH_DIAGONAL_COL].isna(), ID_COL].head().tolist()
    raise RuntimeError(f"Missing Khovanov outcomes after merge, e.g. {bad}")

N = len(data)
if N != 313_230:
    print(f"Warning: expected 313,230 rows, found {N:,}")

signature = data["signature"].to_numpy(float)
s_value = data[S_COL].to_numpy(float)
gap = np.abs(s_value - signature)
signed_gap = s_value - signature
nonalternating = data["is_alternating"].to_numpy(int) == 0
kh_diagonal = data[KH_DIAGONAL_COL].to_numpy(float)
kh_support = data[KH_SUPPORT_COL].to_numpy(float)
kh_coarse = np.where(kh_diagonal <= 2, 2, np.where(kh_diagonal == 3, 3, 4)).astype(np.int8)


with np.load(SCORES_PATH, allow_pickle=False) as payload:
    norm_bins = {
        name: np.asarray(payload[f"{safe_name(name)}_norm_bin"], dtype=np.int32)
        for name in INVARIANTS
    }

for name, values in norm_bins.items():
    if len(values) != N:
        raise AssertionError(f"Norm-bin length mismatch for {name}: {len(values)} != {N}")

hard_saved = pd.read_csv(HARD_SETS_PATH)
if not {"invariant", ID_COL}.issubset(hard_saved.columns):
    raise KeyError(f"Unexpected hard-set schema: {hard_saved.columns.tolist()}")
id_to_pos = pd.Series(np.arange(N, dtype=np.int64), index=data[ID_COL]).to_dict()
hard_masks: dict[str, np.ndarray] = {}
for name in INVARIANTS:
    ids = hard_saved.loc[hard_saved["invariant"].eq(name), ID_COL].astype(str)
    unknown = sorted(set(ids) - set(id_to_pos))
    if unknown:
        raise KeyError(f"Unknown IDs in {name} hard set, e.g. {unknown[:5]}")
    mask = np.zeros(N, dtype=bool)
    mask[[id_to_pos[x] for x in ids]] = True
    hard_masks[name] = mask
    if mask.sum() != 3_133:
        raise AssertionError(f"Expected 3,133 selected in {name}; found {mask.sum()}")


def consensus_mask(names: tuple[str, ...], k: int) -> np.ndarray:
    count = np.zeros(N, dtype=np.uint8)
    for name in names:
        count += hard_masks[name]
    return count >= k


all5_selected = consensus_mask(INVARIANTS, 3)
no_kh_selected = consensus_mask(NO_KHOVANOV, 3)
print("All-five >=3/5:", int(all5_selected.sum()))
print("No-Khovanov >=3/4:", int(no_kh_selected.sum()))


# ---------------------------------------------------------------------------
# 2. Pointwise consistency/cross-tab audit
# ---------------------------------------------------------------------------
def thickness_label(values: np.ndarray) -> pd.Categorical:
    label = np.where(values <= 2, "2", np.where(values == 3, "3", ">=4"))
    return pd.Categorical(label, categories=["2", "3", ">=4"], ordered=True)


audit_rows = []
regime_masks = {
    "full_universe": np.ones(N, dtype=bool),
    "all5_conditional": all5_selected,
    "no_khovanov_conditional": no_kh_selected,
}
if "hard_regime" in data:
    for value in (
        "conditional_hard_only",
        "amplitude_hard_only",
        "shared_raw_and_conditional",
    ):
        regime_masks[value] = data["hard_regime"].astype(str).eq(value).to_numpy()

for regime, base_mask in regime_masks.items():
    for population, pop_mask in (
        ("all", base_mask),
        ("nonalternating", base_mask & nonalternating),
    ):
        for thickness in ("2", "3", ">=4"):
            thick_mask = np.asarray(thickness_label(kh_diagonal) == thickness)
            mask = pop_mask & thick_mask
            audit_rows.append(
                {
                    "regime": regime,
                    "population": population,
                    "kh_diagonal_group": thickness,
                    "n": int(mask.sum()),
                    "gap_positive_n": int((mask & (gap > 0)).sum()),
                    "gap_positive_prop": float(np.mean(gap[mask] > 0)) if mask.any() else np.nan,
                    "mean_abs_gap": float(np.mean(gap[mask])) if mask.any() else np.nan,
                    "gap_ge_4_prop": float(np.mean(gap[mask] >= 4)) if mask.any() else np.nan,
                }
            )

gap_thickness_crosstab = pd.DataFrame(audit_rows)
gap_thickness_crosstab.to_csv(OUT / "gap_by_khovanov_thickness_crosstab.csv", index=False)

thin_discrepancy_mask = (gap > 0) & (kh_diagonal <= 2)
thin_gap_audit_cols = [
    c
    for c in (
        ID_COL,
        "number_of_crossings",
        "is_alternating",
        "signature",
        S_COL,
        KH_DIAGONAL_COL,
        KH_SUPPORT_COL,
        "conditional_membership",
        "hard_regime",
    )
    if c in data
]
thin_gap_audit = data.loc[thin_discrepancy_mask, thin_gap_audit_cols].copy()
thin_gap_audit["signed_gap"] = signed_gap[thin_discrepancy_mask]
thin_gap_audit["abs_gap"] = gap[thin_discrepancy_mask]
thin_gap_audit["all5_conditional"] = all5_selected[thin_discrepancy_mask]
thin_gap_audit["no_khovanov_conditional"] = no_kh_selected[thin_discrepancy_mask]
thin_gap_audit.to_csv(OUT / "gap_positive_two_diagonal_knot_audit.csv", index=False)

print("G>0 with <=2 stored q-2t diagonals:", len(thin_gap_audit))
print(
    "Selected by all-five conditional:",
    int(thin_gap_audit["all5_conditional"].sum()),
)


# ---------------------------------------------------------------------------
# 3. Exact-stratum utilities and diagnostics
# ---------------------------------------------------------------------------
def factorized_strata(view: str, thickness_mode: str, include_kh_norm: bool = False) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "view_norm_bin": norm_bins[view],
            "crossings": data["number_of_crossings"].astype(str).to_numpy(),
            "alternating": data["is_alternating"].astype(str).to_numpy(),
            "abs_signature": data["signature"].abs().astype(str).to_numpy(),
        }
    )
    if thickness_mode == "exact":
        frame["kh_thickness"] = kh_diagonal.astype(np.int16)
    elif thickness_mode == "coarse":
        frame["kh_thickness"] = kh_coarse
    elif thickness_mode != "none":
        raise ValueError(thickness_mode)
    if include_kh_norm:
        frame["khovanov_norm_bin"] = norm_bins["Khovanov"]
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
    fixed_idx = np.concatenate(fixed_groups) if fixed_groups else np.empty(0, dtype=np.int64)
    sizes_arr = np.asarray(sizes)
    selected_counts_arr = np.asarray(selected_counts)
    diagnostic = {
        "n_strata": len(sizes_arr),
        "stratum_size_min": int(sizes_arr.min()),
        "stratum_size_q25": float(np.quantile(sizes_arr, 0.25)),
        "stratum_size_median": float(np.median(sizes_arr)),
        "stratum_size_q75": float(np.quantile(sizes_arr, 0.75)),
        "stratum_size_max": int(sizes_arr.max()),
        "singleton_strata_prop": float(np.mean(sizes_arr == 1)),
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


def gap_metrics(mask: np.ndarray) -> dict:
    values = gap[mask & nonalternating]
    return {
        "n": int(mask.sum()),
        "nonalternating_n": int(len(values)),
        "abs_gap_positive_prop": float(np.mean(values > 0)) if len(values) else np.nan,
        "mean_abs_gap": float(np.mean(values)) if len(values) else np.nan,
        "abs_gap_ge_4_prop": float(np.mean(values >= 4)) if len(values) else np.nan,
    }


def thickness_metrics(mask: np.ndarray) -> dict:
    diagonal = kh_diagonal[mask]
    support = kh_support[mask]
    return {
        "n": int(mask.sum()),
        "kh_diagonal_mean": float(np.mean(diagonal)) if len(diagonal) else np.nan,
        "kh_diagonal_ge_3_prop": float(np.mean(diagonal >= 3)) if len(diagonal) else np.nan,
        "kh_diagonal_ge_4_prop": float(np.mean(diagonal >= 4)) if len(diagonal) else np.nan,
        "kh_support_mean": float(np.mean(support)) if len(support) else np.nan,
    }


def summarize_null(observed: dict, null: pd.DataFrame, metrics: tuple[str, ...], **labels) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        values = null[metric].dropna().to_numpy(float)
        value = float(observed[metric])
        exceed = int(np.sum(values >= value))
        rows.append(
            {
                **labels,
                "metric": metric,
                "observed": value,
                "null_mean": float(values.mean()),
                "null_sd": float(values.std(ddof=1)),
                "null_q95": float(np.quantile(values, 0.95)),
                "null_q99": float(np.quantile(values, 0.99)),
                "n_ge_observed": exceed,
                "n_valid_null": len(values),
                "empirical_p": (1 + exceed) / (1 + len(values)),
            }
        )
    return pd.DataFrame(rows)


def run_membership_null(
    label: str,
    names: tuple[str, ...],
    k: int,
    thickness_mode: str,
    include_kh_norm: bool,
    metric_function,
    metric_names: tuple[str, ...],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed_mask = consensus_mask(names, k)
    observed = metric_function(observed_mask)

    samplers = {}
    diag_rows = []
    for name in names:
        codes = factorized_strata(name, thickness_mode, include_kh_norm)
        sampler, diag = prepare_sampler(codes, hard_masks[name])
        samplers[name] = sampler
        diag_rows.append(
            {
                "analysis": label,
                "invariant": name,
                "thickness_mode": thickness_mode,
                "include_khovanov_norm_bin": include_kh_norm,
                **diag,
            }
        )

    raw_path = OUT / f"{label}_null_{N_REPS}.parquet"
    if raw_path.exists():
        print("Loading checkpoint:", raw_path.name)
        raw = pd.read_parquet(raw_path)
    else:
        rng = np.random.default_rng(seed)
        rows = []
        for replicate in range(N_REPS):
            count = np.zeros(N, dtype=np.uint8)
            for name in names:
                count += sample_mask(samplers[name], rng)
            row = {"replicate": replicate}
            row.update(metric_function(count >= k))
            rows.append(row)
            if (replicate + 1) % 500 == 0:
                print(f"{label}: {replicate + 1:,}/{N_REPS:,}")
        raw = pd.DataFrame(rows)
        raw.to_parquet(raw_path, index=False)
        raw.to_csv(raw_path.with_suffix(".csv"), index=False)

    summary = summarize_null(
        observed,
        raw,
        metric_names,
        analysis=label,
        thickness_mode=thickness_mode,
        include_khovanov_norm_bin=include_kh_norm,
        n_selected=int(observed_mask.sum()),
    )
    return pd.DataFrame([{"analysis": label, **observed}]), summary, pd.DataFrame(diag_rows)


# ---------------------------------------------------------------------------
# 4. Decisive G null: canonical bridge, exact thickness, coarse sensitivity
# ---------------------------------------------------------------------------
G_METRICS = ("abs_gap_positive_prop", "mean_abs_gap", "abs_gap_ge_4_prop")
gap_specs = (
    ("all5_gap_no_thickness", "none"),
    ("all5_gap_exact_thickness", "exact"),
    ("all5_gap_coarse_thickness", "coarse"),
)

observed_parts = []
summary_parts = []
diagnostic_parts = []
for offset, (label, mode) in enumerate(gap_specs):
    obs, summ, diag = run_membership_null(
        label=label,
        names=INVARIANTS,
        k=3,
        thickness_mode=mode,
        include_kh_norm=False,
        metric_function=gap_metrics,
        metric_names=G_METRICS,
        seed=SEED + offset,
    )
    observed_parts.append(obs)
    summary_parts.append(summ)
    diagnostic_parts.append(diag)

gap_null_observed = pd.concat(observed_parts, ignore_index=True)
gap_null_summary = pd.concat(summary_parts, ignore_index=True)
gap_stratum_diagnostics = pd.concat(diagnostic_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 5. No-Khovanov external-thickness sensitivity with Khovanov norm controlled
# ---------------------------------------------------------------------------
KH_METRICS = (
    "kh_diagonal_mean",
    "kh_diagonal_ge_3_prop",
    "kh_diagonal_ge_4_prop",
    "kh_support_mean",
)
obs_kh, summ_kh, diag_kh = run_membership_null(
    label="no_khovanov_external_with_kh_norm",
    names=NO_KHOVANOV,
    k=3,
    thickness_mode="none",
    include_kh_norm=True,
    metric_function=thickness_metrics,
    metric_names=KH_METRICS,
    seed=SEED + 100,
)


# ---------------------------------------------------------------------------
# 6. Multiplicity, decisions and frozen output
# ---------------------------------------------------------------------------
all_summaries = pd.concat([gap_null_summary, summ_kh], ignore_index=True)
for method, output_col in (
    ("fdr_by", "q_by"),
    ("holm", "p_holm"),
):
    all_summaries[output_col] = multipletests(
        all_summaries["empirical_p"].to_numpy(float), method=method
    )[1]

gap_null_summary = all_summaries.loc[
    all_summaries["analysis"].str.startswith("all5_gap")
].copy()
no_kh_kh_norm_summary = all_summaries.loc[
    all_summaries["analysis"].eq("no_khovanov_external_with_kh_norm")
].copy()

decision_rows = []
for analysis, group in gap_null_summary.groupby("analysis", sort=False):
    decision_rows.append(
        {
            "analysis": analysis,
            "gap_positive_observed": float(
                group.loc[group["metric"].eq("abs_gap_positive_prop"), "observed"].iloc[0]
            ),
            "gap_positive_null_mean": float(
                group.loc[group["metric"].eq("abs_gap_positive_prop"), "null_mean"].iloc[0]
            ),
            "max_empirical_p": float(group["empirical_p"].max()),
            "max_q_by": float(group["q_by"].max()),
            "max_p_holm": float(group["p_holm"].max()),
            "all_metrics_holm_0_05": bool((group["p_holm"] < 0.05).all()),
        }
    )
gap_thickness_decision = pd.DataFrame(decision_rows)

gap_null_observed.to_csv(OUT / "gap_null_observed.csv", index=False)
gap_null_summary.to_csv(OUT / "gap_thickness_conditioned_null_summary.csv", index=False)
gap_stratum_diagnostics.to_csv(OUT / "gap_null_stratum_diagnostics.csv", index=False)
obs_kh.to_csv(OUT / "no_khovanov_external_with_kh_norm_observed.csv", index=False)
no_kh_kh_norm_summary.to_csv(
    OUT / "no_khovanov_external_with_kh_norm_summary.csv", index=False
)
diag_kh.to_csv(OUT / "no_khovanov_external_with_kh_norm_strata.csv", index=False)
gap_thickness_decision.to_csv(OUT / "gap_thickness_decision.csv", index=False)

print("\nGap/thickness cross-tab:")
print(
    gap_thickness_crosstab.loc[
        gap_thickness_crosstab["regime"].isin(["full_universe", "all5_conditional"])
        & gap_thickness_crosstab["population"].eq("nonalternating")
    ].to_string(index=False)
)
print("\nThickness-conditioned G null:")
print(gap_null_summary.to_string(index=False))
print("\nNo-Khovanov external outcome with Khovanov norm controlled:")
print(no_kh_kh_norm_summary.to_string(index=False))
print("\nDecision table:")
print(gap_thickness_decision.to_string(index=False))
print("\nSaved Stage 22 to:", OUT)

