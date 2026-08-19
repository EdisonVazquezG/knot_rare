# %% [markdown]
# Stage 23B — Size-matched multiview score sensitivity
#
# Reuses Stage-23 score checkpoints; no PCA or anomaly model is refit.
# Per-view scores are converted to empirical test percentiles, combined by the
# kth-largest percentile (C_k), and truncated to the same final population
# size for every method. This isolates population composition from the very
# different consensus sizes produced by a fixed >=3/m vote.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


# ---------------------------------------------------------------------------
# 0. Frozen paths and design
# ---------------------------------------------------------------------------
DEFAULT_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants/"
    "processed_consensus_hardness/corrected_run_20260819"
)
ROOT = Path(globals().get("OUTPUT_DIR", DEFAULT_ROOT))
STAGE23 = ROOT / "23_anomaly_score_baselines"
CHECKPOINTS = STAGE23 / "checkpoints"
OUT = ROOT / "23B_size_matched_score_sensitivity"
FIG = OUT / "figures"
for directory in (OUT, FIG):
    directory.mkdir(parents=True, exist_ok=True)

if not CHECKPOINTS.exists():
    raise FileNotFoundError(
        f"Stage-23 checkpoints not found: {CHECKPOINTS}. Run Stage 23 first."
    )

INVARIANTS = ("Alexander", "Jones", "HOMFLY-PT", "Theta", "Khovanov")
NO_KHOVANOV = tuple(x for x in INVARIANTS if x != "Khovanov")
METHODS = (
    "raw_sse",
    "relative_nre",
    "residual_mahalanobis",
    "residual_isolation_forest",
    "conditional_percentile_100",
)
METHOD_LABELS = {
    "raw_sse": "Raw SSE",
    "relative_nre": "Relative error",
    "residual_mahalanobis": "Residual Mahalanobis",
    "residual_isolation_forest": "Residual Isolation Forest",
    "conditional_percentile_100": "Conditional percentile",
}
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
# 1. Load test IDs, metadata and external phenotype
# ---------------------------------------------------------------------------
split_path = find_one("heldout_ae_seed_0.npz")
with np.load(split_path, allow_pickle=False) as payload:
    test_idx = np.asarray(payload["test_idx"], dtype=np.int64)

atlas_path = find_one("final_hard_regime_atlas.parquet")
atlas = pd.read_parquet(atlas_path).reset_index(drop=True)
atlas[ID_COL] = atlas[ID_COL].astype(str)
test_meta = atlas.iloc[test_idx].reset_index(drop=True)
test_ids = test_meta[ID_COL].to_numpy(str)
N_TEST = len(test_meta)

S_COL = next(
    c for c in ("s_invariant_qc", "s_invariant", "s") if c in test_meta
)
signature = test_meta["signature"].to_numpy(float)
s_value = test_meta[S_COL].to_numpy(float)
gap = np.abs(s_value - signature)
nonalternating = test_meta["is_alternating"].to_numpy(int) == 0
crossings = test_meta["number_of_crossings"].to_numpy(int)
alternating = test_meta["is_alternating"].to_numpy(int)

phenotype = pd.read_parquet(find_one("complete_mathematical_phenotype_atlas.parquet"))
phenotype[ID_COL] = phenotype[ID_COL].astype(str)
phenotype = phenotype.set_index(ID_COL).reindex(test_ids)
KH_DIAGONAL_COL = next(
    c
    for c in (
        "khovanov_q_minus_2t_diagonal_count",
        "kh_diagonal_count",
        "khovanov_diagonal_count",
    )
    if c in phenotype
)
KH_SUPPORT_COL = next(
    c for c in ("khovanov_support_size", "kh_support_size") if c in phenotype
)
if phenotype[[KH_DIAGONAL_COL, KH_SUPPORT_COL]].isna().any().any():
    raise RuntimeError("Missing Khovanov phenotype rows after test-ID alignment")
kh_diagonal = phenotype[KH_DIAGONAL_COL].to_numpy(float)
kh_support = phenotype[KH_SUPPORT_COL].to_numpy(float)


# ---------------------------------------------------------------------------
# 2. Load Stage-23 scores and transform every view to empirical percentiles
# ---------------------------------------------------------------------------
score_by_view = {}
log_norm_by_view = {}
for invariant in INVARIANTS:
    path = CHECKPOINTS / f"{safe_name(invariant)}_test_scores.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        score_by_view[invariant] = {
            method: np.asarray(payload[f"test_{method}"], dtype=float)
            for method in METHODS
        }
        log_norm_by_view[invariant] = np.asarray(
            payload["test_log_norm"], dtype=float
        )
    for method in METHODS:
        if len(score_by_view[invariant][method]) != N_TEST:
            raise AssertionError(f"Score length mismatch: {invariant}, {method}")


def empirical_percentile(values: np.ndarray) -> np.ndarray:
    """Average-rank empirical percentile in (0,1], monotone in the score."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite Stage-23 score")
    return rankdata(values, method="average") / len(values)


view_percentiles: dict[str, dict[str, np.ndarray]] = {
    method: {
        invariant: empirical_percentile(score_by_view[invariant][method])
        for invariant in INVARIANTS
    }
    for method in METHODS
}


def ck_score(method: str, names: tuple[str, ...], k: int) -> np.ndarray:
    matrix = np.column_stack([view_percentiles[method][name] for name in names])
    # kth-largest across m views. For >=3/5 and >=3/4, k=3.
    return np.sort(matrix, axis=1)[:, -k]


all5_ck = {method: ck_score(method, INVARIANTS, 3) for method in METHODS}
no_kh_ck = {method: ck_score(method, NO_KHOVANOV, 3) for method in METHODS}


def stable_top_n(score: np.ndarray, n: int) -> np.ndarray:
    order = np.lexsort((test_ids, np.asarray(score, dtype=float)))
    mask = np.zeros(N_TEST, dtype=bool)
    mask[order[-int(n):]] = True
    return mask


# Stage 23 produced 60 conditional all-five knots and 31 conditional
# no-Khovanov knots. Read these values instead of silently hard-coding them.
stage23_summary_path = STAGE23 / "score_family_external_phenotypes.csv"
stage23_summary = pd.read_csv(stage23_summary_path)


def reference_size(family: str) -> int:
    row = stage23_summary.loc[
        stage23_summary["method"].eq("conditional_percentile_100")
        & stage23_summary["family"].eq(family)
    ]
    if len(row) != 1:
        raise RuntimeError(f"Could not identify conditional reference size: {family}")
    return int(row["n"].iloc[0])


N_ALL5 = reference_size("All 5 >=3/5")
N_NO_KH = reference_size("No Khovanov >=3/4")
SIZE_MULTIPLIERS = (0.5, 1.0, 2.0)
print(f"Size-matched references: all-five n={N_ALL5}; no-Khovanov n={N_NO_KH}")


# ---------------------------------------------------------------------------
# 3. Evaluate size-matched scientific populations
# ---------------------------------------------------------------------------
def phenotype_metrics(mask: np.ndarray) -> dict:
    nonalt_values = gap[mask & nonalternating]
    diagonal = kh_diagonal[mask]
    support = kh_support[mask]
    return {
        "n": int(mask.sum()),
        "nonalternating_n": int(len(nonalt_values)),
        "abs_gap_positive_prop": (
            float(np.mean(nonalt_values > 0)) if len(nonalt_values) else np.nan
        ),
        "mean_abs_gap": (
            float(np.mean(nonalt_values)) if len(nonalt_values) else np.nan
        ),
        "abs_gap_ge_4_prop": (
            float(np.mean(nonalt_values >= 4)) if len(nonalt_values) else np.nan
        ),
        "kh_diagonal_mean": float(np.mean(diagonal)) if len(diagonal) else np.nan,
        "kh_diagonal_ge_3_prop": (
            float(np.mean(diagonal >= 3)) if len(diagonal) else np.nan
        ),
        "kh_diagonal_ge_4_prop": (
            float(np.mean(diagonal >= 4)) if len(diagonal) else np.nan
        ),
        "kh_support_mean": float(np.mean(support)) if len(support) else np.nan,
        "crossing_15_prop": float(np.mean(crossings[mask] == 15)),
        "alternating_prop": float(np.mean(alternating[mask] == 1)),
    }


family_specs = {
    "All 5 C3": (all5_ck, N_ALL5),
    "No Khovanov C3": (no_kh_ck, N_NO_KH),
}
summary_rows = []
selected_rows = []
selection_masks = {}

for family, (scores, reference_n) in family_specs.items():
    for multiplier in SIZE_MULTIPLIERS:
        target_n = max(1, int(round(reference_n * multiplier)))
        for method in METHODS:
            mask = stable_top_n(scores[method], target_n)
            selection_masks[(family, multiplier, method)] = mask
            summary_rows.append(
                {
                    "family": family,
                    "size_multiplier": multiplier,
                    "target_n": target_n,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    **phenotype_metrics(mask),
                }
            )
            if multiplier == 1.0:
                for knot_id in test_ids[mask]:
                    selected_rows.append(
                        {
                            "family": family,
                            "method": method,
                            ID_COL: knot_id,
                        }
                    )

size_matched_summary = pd.DataFrame(summary_rows)
size_matched_selected_ids = pd.DataFrame(selected_rows)


# ---------------------------------------------------------------------------
# 4. Overlap and residual aggregate-score/norm association
# ---------------------------------------------------------------------------
def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = a | b
    return float(np.sum(a & b) / np.sum(union)) if union.any() else np.nan


overlap_rows = []
for family in family_specs:
    for method_a in METHODS:
        for method_b in METHODS:
            a = selection_masks[(family, 1.0, method_a)]
            b = selection_masks[(family, 1.0, method_b)]
            overlap_rows.append(
                {
                    "family": family,
                    "method_a": method_a,
                    "method_b": method_b,
                    "overlap": int(np.sum(a & b)),
                    "jaccard": jaccard(a, b),
                }
            )
size_matched_overlap = pd.DataFrame(overlap_rows)

mean_all5_norm = np.mean(
    np.column_stack([log_norm_by_view[name] for name in INVARIANTS]), axis=1
)
max_all5_norm = np.max(
    np.column_stack([log_norm_by_view[name] for name in INVARIANTS]), axis=1
)
mean_no_kh_norm = np.mean(
    np.column_stack([log_norm_by_view[name] for name in NO_KHOVANOV]), axis=1
)
max_no_kh_norm = np.max(
    np.column_stack([log_norm_by_view[name] for name in NO_KHOVANOV]), axis=1
)

aggregate_rows = []
for family, scores, mean_norm, max_norm in (
    ("All 5 C3", all5_ck, mean_all5_norm, max_all5_norm),
    ("No Khovanov C3", no_kh_ck, mean_no_kh_norm, max_no_kh_norm),
):
    for method in METHODS:
        aggregate_rows.append(
            {
                "family": family,
                "method": method,
                "spearman_ck_mean_log_norm": float(
                    spearmanr(scores[method], mean_norm).statistic
                ),
                "spearman_ck_max_log_norm": float(
                    spearmanr(scores[method], max_norm).statistic
                ),
            }
        )
aggregate_norm_diagnostics = pd.DataFrame(aggregate_rows)


# ---------------------------------------------------------------------------
# 5. Save and visualize the primary equal-size comparison
# ---------------------------------------------------------------------------
size_matched_summary.to_csv(OUT / "size_matched_score_phenotypes.csv", index=False)
size_matched_selected_ids.to_csv(OUT / "size_matched_selected_ids.csv", index=False)
size_matched_overlap.to_csv(OUT / "size_matched_score_jaccard.csv", index=False)
aggregate_norm_diagnostics.to_csv(
    OUT / "size_matched_aggregate_norm_diagnostics.csv", index=False
)

primary = size_matched_summary.loc[
    size_matched_summary["size_multiplier"].eq(1.0)
].copy()
short_labels = ["Raw", "Relative", "Mahalanobis", "IF", "Conditional"]
colors = ["#D95F02", "#4C78A8", "#7570B3", "#9C755F", "#1B9E77"]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
for ax, family, title in (
    (axes[0], "All 5 C3", f"All five views, equal n={N_ALL5}"),
    (axes[1], "No Khovanov C3", f"Khovanov held out, equal n={N_NO_KH}"),
):
    subset = primary.loc[primary["family"].eq(family)].set_index("method").reindex(METHODS)
    x = np.arange(len(METHODS))
    width = 0.36
    ax.bar(
        x - width / 2,
        subset["abs_gap_positive_prop"],
        width,
        color=colors,
        label=r"$P(|s-\sigma|>0)$",
    )
    ax.bar(
        x + width / 2,
        subset["kh_diagonal_ge_3_prop"],
        width,
        color=colors,
        alpha=0.45,
        hatch="//",
        label=r"$P(\delta\mathrm{-width}\geq3)$",
    )
    ax.set_xticks(x, short_labels, rotation=32, ha="right")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("Proportion in size-matched selected set")
axes[1].legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle(
    "Scientific phenotypes remain score-dependent after matching final set size",
    fontsize=14,
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(FIG / "figure_size_matched_score_comparison.png", dpi=350, bbox_inches="tight")
fig.savefig(FIG / "figure_size_matched_score_comparison.pdf", bbox_inches="tight")
plt.close(fig)

print("\nPrimary size-matched comparison:")
print(
    primary[
        [
            "family",
            "method",
            "n",
            "nonalternating_n",
            "abs_gap_positive_prop",
            "mean_abs_gap",
            "kh_diagonal_mean",
            "kh_diagonal_ge_3_prop",
            "kh_diagonal_ge_4_prop",
            "crossing_15_prop",
            "alternating_prop",
        ]
    ].to_string(index=False)
)
print("\nAggregate score--norm diagnostics:")
print(aggregate_norm_diagnostics.to_string(index=False))
print("\nSaved Stage 23B to:", OUT)

