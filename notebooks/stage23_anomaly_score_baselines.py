# %% [markdown]
# Stage 23 — MLST anomaly-score baselines
#
# Scientific question: does the anomaly score change the population that is
# discovered, even when the representation, PCA fit, per-view tail mass and
# multiview voting rule are held fixed?
#
# Main score baselines:
#   * raw PCA reconstruction SSE;
#   * relative reconstruction error SSE / ||x||^2;
#   * shrinkage-style Mahalanobis distance in a learned residual sketch;
#   * Isolation Forest in the same residual sketch;
#   * the proposed percentile of SSE conditional on representation norm.
#
# All PCA/scaler/residual models are fit on the frozen training split. Norm-bin
# calibration uses validation only. Every method selects exactly the upper 1%
# of the test score in each view, and every primary comparison uses >=3/5.
# The no-Khovanov >=3/4 family is evaluated separately against held-out
# Khovanov thickness.
#
# This script is restart-safe and writes one checkpoint per representation.
# If X_dict/meta are absent, it rebuilds the aligned matrices from the frozen
# source files.  Expect this stage to be compute- and memory-intensive.

# %%
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 0. Paths, package and frozen settings
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(
    globals().get(
        "PROJECT_DIR",
        "/content/drive/MyDrive/consensus_hardness_refactored",
    )
)
if (PROJECT_DIR / "src").exists():
    sys.path.insert(0, str(PROJECT_DIR / "src"))

import consensus_hardness as ch

DATA_DIR = Path(
    globals().get(
        "DATA_DIR",
        "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants",
    )
)
DEFAULT_ROOT = (
    DATA_DIR / "processed_consensus_hardness" / "corrected_run_20260819"
)
ROOT = Path(globals().get("OUTPUT_DIR", DEFAULT_ROOT))
OUT = ROOT / "23_anomaly_score_baselines"
CHECKPOINTS = OUT / "checkpoints"
FIGURES = OUT / "figures"
for directory in (OUT, CHECKPOINTS, FIGURES):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG = globals().get("CONFIG", ch.canonical_run_config())
INVARIANTS = ("Alexander", "Jones", "HOMFLY-PT", "Theta", "Khovanov")
NO_KHOVANOV = tuple(x for x in INVARIANTS if x != "Khovanov")
ID_COL = CONFIG.universe.id_col
S_COL = CONFIG.s_col
TAIL_MASS = float(CONFIG.pca.tail_mass)
NORM_BIN_COUNTS = (50, 100, 200)
MAIN_NORM_BINS = 100
RESIDUAL_SKETCH_DIM = int(os.environ.get("STAGE23_RESIDUAL_DIM", "64"))
RESIDUAL_FIT_SIZE = int(os.environ.get("STAGE23_RESIDUAL_FIT_SIZE", "40000"))
IFOREST_TREES = int(os.environ.get("STAGE23_IFOREST_TREES", "300"))
BATCH_SIZE = int(os.environ.get("STAGE23_BATCH_SIZE", "8192"))
SEED = int(os.environ.get("STAGE23_SEED", "20261123"))

FIXED_K = dict(CONFIG.pca.primary_k)

FILE_MAP = {
    "alex": "Alexander_upto17.csv",
    "homfly": "HomflyPt_upto15_MIRRORS.csv",
    "jones": "Jones_upto17_MIRRORS.csv",
    "theta": "theta_upto15.csv",
    "kh": "even_KH_upto17.pkl",
}
REPRESENTATION_SPECS = {
    "Alexander": {"source": "alex", "feature_prefixes": ["A"]},
    "Jones": {"source": "jones", "feature_prefixes": ["J"]},
    "HOMFLY-PT": {"source": "homfly", "feature_prefixes": ["a"]},
    "Theta": {"source": "theta", "feature_prefixes": ["T"]},
    "Khovanov": {"source": "kh", "feature_prefixes": ["F_"]},
}


def safe_name(name: str) -> str:
    return ch.safe_name(name)


def find_one(filename: str) -> Path:
    found = sorted(ROOT.rglob(filename))
    if not found:
        raise FileNotFoundError(f"Could not find {filename} below {ROOT}")
    return found[0]


# ---------------------------------------------------------------------------
# 1. Load/rebuild aligned universe and frozen split
# ---------------------------------------------------------------------------
if "meta" in globals() and "X_dict" in globals():
    meta = globals()["meta"].reset_index(drop=True).copy()
    X_dict = globals()["X_dict"]
    print("Using meta and X_dict from current runtime.")
else:
    print("Rebuilding frozen aligned matrices from source files...")
    aligned = ch.build_aligned_dataset(
        DATA_DIR,
        FILE_MAP,
        REPRESENTATION_SPECS,
        min_crossings=CONFIG.universe.min_crossings,
        max_crossings=CONFIG.universe.max_crossings,
        expected_n=CONFIG.universe.expected_n,
        expected_s_qc_corrections=CONFIG.universe.expected_s_qc_corrections,
        preferred_metadata_sources=["alex", "jones", "homfly", "theta", "kh"],
        output_dir=ROOT / "00_alignment",
    )
    meta = aligned["meta"].reset_index(drop=True)
    X_dict = aligned["X_dict"]

if tuple(X_dict) != INVARIANTS:
    X_dict = {name: X_dict[name] for name in INVARIANTS}

split_path = find_one("heldout_ae_seed_0.npz")
with np.load(split_path, allow_pickle=False) as split_payload:
    train_idx = np.asarray(split_payload["train_idx"], dtype=np.int64)
    val_idx = np.asarray(split_payload["val_idx"], dtype=np.int64)
    test_idx = np.asarray(split_payload["test_idx"], dtype=np.int64)

if len(set(train_idx) & set(val_idx)) or len(set(train_idx) & set(test_idx)) or len(set(val_idx) & set(test_idx)):
    raise AssertionError("Frozen train/validation/test split overlaps")
if len(train_idx) + len(val_idx) + len(test_idx) != len(meta):
    raise AssertionError("Frozen split does not cover the aligned universe")

test_ids = meta.iloc[test_idx][ID_COL].astype(str).to_numpy()
N_TEST = len(test_idx)
TAIL_N = int(np.ceil(TAIL_MASS * N_TEST))
print(
    f"Frozen split: train={len(train_idx):,}, val={len(val_idx):,}, "
    f"test={N_TEST:,}; per-view test tail={TAIL_N:,}"
)


# External phenotype atlas
phenotype_path = find_one("complete_mathematical_phenotype_atlas.parquet")
phenotype = pd.read_parquet(phenotype_path)
phenotype[ID_COL] = phenotype[ID_COL].astype(str)
phenotype = phenotype.set_index(ID_COL).reindex(test_ids)
if phenotype.index.has_duplicates:
    raise AssertionError("Phenotype atlas did not align to test IDs")

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
    missing_ids = phenotype.index[
        phenotype[[KH_DIAGONAL_COL, KH_SUPPORT_COL]].isna().any(axis=1)
    ][:10].tolist()
    raise RuntimeError(
        "Phenotype atlas is missing Khovanov outcomes for test IDs, e.g. "
        f"{missing_ids}"
    )

if S_COL not in meta:
    S_COL = next(c for c in ("s_invariant_qc", "s_invariant", "s") if c in meta)

test_signature = meta.iloc[test_idx]["signature"].to_numpy(float)
test_s = meta.iloc[test_idx][S_COL].to_numpy(float)
test_gap = np.abs(test_s - test_signature)
test_nonalt = meta.iloc[test_idx]["is_alternating"].to_numpy(int) == 0
test_crossings = meta.iloc[test_idx]["number_of_crossings"].to_numpy(int)
test_alternating = meta.iloc[test_idx]["is_alternating"].to_numpy(int)
test_kh_diag = phenotype[KH_DIAGONAL_COL].to_numpy(float)
test_kh_support = phenotype[KH_SUPPORT_COL].to_numpy(float)


# ---------------------------------------------------------------------------
# 2. Calibration/scoring helpers
# ---------------------------------------------------------------------------
def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(
        np.asarray(values, dtype=float), np.linspace(0.0, 1.0, n_bins + 1)
    )
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.array([-np.inf, np.inf], dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges[1:-1], values, side="right").astype(np.int32)


def conditional_percentile(
    val_score: np.ndarray,
    val_norm: np.ndarray,
    test_score: np.ndarray,
    test_norm: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    edges = quantile_edges(val_norm, n_bins)
    val_bin = assign_bins(val_norm, edges)
    test_bin = assign_bins(test_norm, edges)
    populated = np.unique(val_bin)
    out = np.empty(len(test_score), dtype=float)
    for b in np.unique(test_bin):
        ref = np.sort(val_score[val_bin == b])
        if len(ref) == 0:
            nearest = int(populated[np.argmin(np.abs(populated - b))])
            ref = np.sort(val_score[val_bin == nearest])
        where = test_bin == b
        out[where] = np.searchsorted(ref, test_score[where], side="right") / (
            len(ref) + 1.0
        )
    return out, test_bin


def stable_top_tail(scores: np.ndarray, n_select: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite anomaly score")
    # Stable ID is a deterministic secondary key for boundary ties.
    order = np.lexsort((test_ids, scores))
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[-n_select:]] = True
    return mask


def residual_batch(X, indices, scaler, pca, residual_pca=None, batch_size=BATCH_SIZE):
    n = len(indices)
    sse = np.empty(n, dtype=np.float64)
    norm_sq = np.empty(n, dtype=np.float64)
    sketch = None
    if residual_pca is not None:
        sketch = np.empty((n, residual_pca.n_components_), dtype=np.float32)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        block = np.asarray(X[indices[start:stop]], dtype=np.float32)
        z = scaler.transform(block)
        retained = pca.transform(z)
        residual = z - pca.inverse_transform(retained)
        sse[start:stop] = np.sum(residual.astype(np.float64) ** 2, axis=1)
        norm_sq[start:stop] = np.sum(z.astype(np.float64) ** 2, axis=1)
        if residual_pca is not None:
            sketch[start:stop] = residual_pca.transform(residual).astype(np.float32)
    return sse, norm_sq, sketch


def fit_one_view(name: str, X: np.ndarray) -> dict[str, np.ndarray]:
    score_path = CHECKPOINTS / f"{safe_name(name)}_test_scores.npz"
    model_path = CHECKPOINTS / f"{safe_name(name)}_models.joblib"
    if score_path.exists():
        print(f"[{name}] loading score checkpoint")
        with np.load(score_path, allow_pickle=False) as payload:
            return {key: np.asarray(payload[key]) for key in payload.files}

    print(f"[{name}] fitting train-only PCA k={FIXED_K[name]}, shape={X.shape}")
    scaler = StandardScaler(copy=True)
    X_train = np.asarray(X[train_idx], dtype=np.float32)
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    pca = PCA(
        n_components=int(FIXED_K[name]),
        svd_solver="randomized",
        random_state=SEED,
    )
    pca.fit(X_train_scaled)
    del X_train_scaled
    gc.collect()

    rng = np.random.default_rng(SEED + list(INVARIANTS).index(name))
    sample_n = min(RESIDUAL_FIT_SIZE, len(train_idx))
    fit_idx = np.sort(rng.choice(train_idx, size=sample_n, replace=False))
    # Recompute the sampled residual matrix once; it is the only large
    # residual matrix retained in memory.
    sampled_residual = np.empty((sample_n, X.shape[1]), dtype=np.float32)
    for start in range(0, sample_n, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, sample_n)
        z = scaler.transform(np.asarray(X[fit_idx[start:stop]], dtype=np.float32))
        sampled_residual[start:stop] = z - pca.inverse_transform(pca.transform(z))

    residual_dim = min(
        RESIDUAL_SKETCH_DIM,
        max(1, X.shape[1] - int(FIXED_K[name])),
        sample_n - 1,
    )
    residual_pca = PCA(
        n_components=residual_dim,
        svd_solver="randomized",
        random_state=SEED + 1,
    )
    fit_sketch = residual_pca.fit_transform(sampled_residual).astype(np.float32)
    del sampled_residual, X_train
    gc.collect()

    iforest = IsolationForest(
        n_estimators=IFOREST_TREES,
        max_samples=min(4096, sample_n),
        contamination="auto",
        random_state=SEED + 2,
        n_jobs=-1,
    )
    iforest.fit(fit_sketch)

    val_sse, val_norm_sq, val_sketch = residual_batch(
        X, val_idx, scaler, pca, residual_pca=residual_pca
    )
    test_sse, test_norm_sq, test_sketch = residual_batch(
        X, test_idx, scaler, pca, residual_pca=residual_pca
    )

    variance = np.maximum(residual_pca.explained_variance_.astype(float), 1e-10)
    val_mahal = np.sum(val_sketch.astype(float) ** 2 / variance, axis=1)
    test_mahal = np.sum(test_sketch.astype(float) ** 2 / variance, axis=1)
    val_iforest = -iforest.score_samples(val_sketch)
    test_iforest = -iforest.score_samples(test_sketch)

    result = {
        "val_raw_sse": val_sse,
        "test_raw_sse": test_sse,
        "val_relative_nre": val_sse / np.maximum(val_norm_sq, 1e-12),
        "test_relative_nre": test_sse / np.maximum(test_norm_sq, 1e-12),
        "val_residual_mahalanobis": val_mahal,
        "test_residual_mahalanobis": test_mahal,
        "val_residual_isolation_forest": val_iforest,
        "test_residual_isolation_forest": test_iforest,
        "val_log_norm": np.log1p(val_norm_sq),
        "test_log_norm": np.log1p(test_norm_sq),
        "residual_sketch_dimension": np.asarray([residual_dim], dtype=np.int32),
        "pca_k": np.asarray([FIXED_K[name]], dtype=np.int32),
    }
    for n_bins in NORM_BIN_COUNTS:
        score, bins = conditional_percentile(
            val_sse,
            np.log1p(val_norm_sq),
            test_sse,
            np.log1p(test_norm_sq),
            n_bins=n_bins,
        )
        result[f"test_conditional_percentile_{n_bins}"] = score
        result[f"test_norm_bin_{n_bins}"] = bins

    np.savez_compressed(score_path, **result)
    joblib.dump(
        {
            "scaler": scaler,
            "pca": pca,
            "residual_pca": residual_pca,
            "iforest": iforest,
            "fit_indices": fit_idx,
        },
        model_path,
    )
    print(f"[{name}] saved {score_path.name}")
    return result


# ---------------------------------------------------------------------------
# 3. Fit/load every view and construct equal-tail hard masks
# ---------------------------------------------------------------------------
score_by_view: dict[str, dict[str, np.ndarray]] = {}
for name in INVARIANTS:
    score_by_view[name] = fit_one_view(name, X_dict[name])
    gc.collect()

MAIN_METHODS = (
    "raw_sse",
    "relative_nre",
    "residual_mahalanobis",
    "residual_isolation_forest",
    "conditional_percentile_100",
)
DISPLAY_NAMES = {
    "raw_sse": "Raw SSE",
    "relative_nre": "Relative error",
    "residual_mahalanobis": "Residual Mahalanobis",
    "residual_isolation_forest": "Residual Isolation Forest",
    "conditional_percentile_100": "Conditional percentile",
}

hard_masks: dict[str, dict[str, np.ndarray]] = {method: {} for method in MAIN_METHODS}
diagnostic_rows = []
for method in MAIN_METHODS:
    key = f"test_{method}"
    for name in INVARIANTS:
        score = np.asarray(score_by_view[name][key], dtype=float)
        hard_masks[method][name] = stable_top_tail(score, TAIL_N)
        diagnostic_rows.append(
            {
                "method": method,
                "invariant": name,
                "n_selected": int(hard_masks[method][name].sum()),
                "spearman_score_log_norm": float(
                    spearmanr(score, score_by_view[name]["test_log_norm"]).statistic
                ),
                "score_median": float(np.median(score)),
                "score_q99": float(np.quantile(score, 0.99)),
            }
        )

score_norm_diagnostics = pd.DataFrame(diagnostic_rows)


def family_mask(method: str, names: tuple[str, ...], k: int) -> np.ndarray:
    count = np.zeros(N_TEST, dtype=np.uint8)
    for name in names:
        count += hard_masks[method][name]
    return count >= k


family_masks: dict[tuple[str, str], np.ndarray] = {}
for method in MAIN_METHODS:
    family_masks[(method, "All 5 >=3/5")] = family_mask(method, INVARIANTS, 3)
    family_masks[(method, "No Khovanov >=3/4")] = family_mask(method, NO_KHOVANOV, 3)
    if method == "raw_sse":
        family_masks[(method, "Original raw strict 5/5")] = family_mask(method, INVARIANTS, 5)


# ---------------------------------------------------------------------------
# 4. External scientific phenotypes and overlap
# ---------------------------------------------------------------------------
def phenotype_metrics(mask: np.ndarray) -> dict:
    nonalt_values = test_gap[mask & test_nonalt]
    diagonal = test_kh_diag[mask]
    support = test_kh_support[mask]
    return {
        "n": int(mask.sum()),
        "nonalternating_n": int(len(nonalt_values)),
        "abs_gap_positive_prop": float(np.mean(nonalt_values > 0)) if len(nonalt_values) else np.nan,
        "mean_abs_gap": float(np.mean(nonalt_values)) if len(nonalt_values) else np.nan,
        "abs_gap_ge_4_prop": float(np.mean(nonalt_values >= 4)) if len(nonalt_values) else np.nan,
        "kh_diagonal_mean": float(np.mean(diagonal)) if len(diagonal) else np.nan,
        "kh_diagonal_ge_3_prop": float(np.mean(diagonal >= 3)) if len(diagonal) else np.nan,
        "kh_support_mean": float(np.mean(support)) if len(support) else np.nan,
        "crossing_15_prop": float(np.mean(test_crossings[mask] == 15)) if mask.any() else np.nan,
        "alternating_prop": float(np.mean(test_alternating[mask] == 1)) if mask.any() else np.nan,
    }


family_rows = []
selected_id_rows = []
for (method, family), mask in family_masks.items():
    family_rows.append(
        {
            "method": method,
            "method_label": DISPLAY_NAMES.get(method, method),
            "family": family,
            **phenotype_metrics(mask),
        }
    )
    for knot_id in test_ids[mask]:
        selected_id_rows.append(
            {"method": method, "family": family, ID_COL: knot_id}
        )

score_family_summary = pd.DataFrame(family_rows)
score_selected_ids = pd.DataFrame(selected_id_rows)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = a | b
    return float(np.sum(a & b) / np.sum(union)) if union.any() else np.nan


overlap_rows = []
for family in ("All 5 >=3/5", "No Khovanov >=3/4"):
    for method_a in MAIN_METHODS:
        for method_b in MAIN_METHODS:
            a = family_masks[(method_a, family)]
            b = family_masks[(method_b, family)]
            overlap_rows.append(
                {
                    "family": family,
                    "method_a": method_a,
                    "method_b": method_b,
                    "size_a": int(a.sum()),
                    "size_b": int(b.sum()),
                    "overlap": int(np.sum(a & b)),
                    "jaccard": jaccard(a, b),
                }
            )
score_overlap = pd.DataFrame(overlap_rows)


# ---------------------------------------------------------------------------
# 5. Conditional bin-count sensitivity (50/100/200)
# ---------------------------------------------------------------------------
bin_masks: dict[int, dict[str, np.ndarray]] = {}
for n_bins in NORM_BIN_COUNTS:
    bin_masks[n_bins] = {}
    for name in INVARIANTS:
        score = score_by_view[name][f"test_conditional_percentile_{n_bins}"]
        bin_masks[n_bins][name] = stable_top_tail(score, TAIL_N)

bin_rows = []
bin_family_masks = {}
for n_bins in NORM_BIN_COUNTS:
    count = np.zeros(N_TEST, dtype=np.uint8)
    for name in INVARIANTS:
        count += bin_masks[n_bins][name]
    mask = count >= 3
    bin_family_masks[n_bins] = mask
    bin_rows.append({"n_bins": n_bins, **phenotype_metrics(mask)})

for a in NORM_BIN_COUNTS:
    for b in NORM_BIN_COUNTS:
        bin_rows.append(
            {
                "n_bins": a,
                "comparison_bins": b,
                "jaccard": jaccard(bin_family_masks[a], bin_family_masks[b]),
            }
        )
bin_sensitivity = pd.DataFrame(bin_rows)


# ---------------------------------------------------------------------------
# 6. Save tables and make an MLST-oriented baseline figure
# ---------------------------------------------------------------------------
score_norm_diagnostics.to_csv(OUT / "score_norm_correlation_by_view.csv", index=False)
score_family_summary.to_csv(OUT / "score_family_external_phenotypes.csv", index=False)
score_overlap.to_csv(OUT / "score_family_jaccard.csv", index=False)
score_selected_ids.to_csv(OUT / "score_selected_test_ids.csv", index=False)
bin_sensitivity.to_csv(OUT / "conditional_bin_sensitivity.csv", index=False)

colors = ["#D95F02", "#4C78A8", "#7570B3", "#9C755F", "#1B9E77"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# A: score--norm association
pivot = score_norm_diagnostics.pivot(
    index="method", columns="invariant", values="spearman_score_log_norm"
).reindex(index=MAIN_METHODS, columns=INVARIANTS)
im = axes[0].imshow(pivot.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
axes[0].set_xticks(range(len(INVARIANTS)), INVARIANTS, rotation=35, ha="right")
axes[0].set_yticks(range(len(MAIN_METHODS)), [DISPLAY_NAMES[x] for x in MAIN_METHODS])
axes[0].set_title("A  Residual score--norm association", loc="left", fontweight="bold")
for i in range(len(MAIN_METHODS)):
    for j in range(len(INVARIANTS)):
        axes[0].text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03, label="Spearman rho")

# B: all-five consensus Jaccard
jac = score_overlap.loc[score_overlap["family"].eq("All 5 >=3/5")].pivot(
    index="method_a", columns="method_b", values="jaccard"
).reindex(index=MAIN_METHODS, columns=MAIN_METHODS)
im2 = axes[1].imshow(jac.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
short = ["Raw", "Relative", "Mahalanobis", "IF", "Conditional"]
axes[1].set_xticks(range(len(short)), short, rotation=35, ha="right")
axes[1].set_yticks(range(len(short)), short)
axes[1].set_title("B  Selected-population overlap", loc="left", fontweight="bold")
for i in range(len(short)):
    for j in range(len(short)):
        axes[1].text(j, i, f"{jac.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.03, label="Jaccard")

# C: externally checkable phenotypes
all5 = score_family_summary.loc[
    score_family_summary["family"].eq("All 5 >=3/5")
].set_index("method").reindex(MAIN_METHODS)
x = np.arange(len(MAIN_METHODS))
width = 0.36
axes[2].bar(
    x - width / 2,
    all5["abs_gap_positive_prop"],
    width,
    color=colors,
    alpha=0.9,
    label=r"$P(|s-\sigma|>0)$",
)
axes[2].bar(
    x + width / 2,
    all5["kh_diagonal_ge_3_prop"],
    width,
    color=colors,
    alpha=0.45,
    hatch="//",
    label=r"$P(\delta\mathrm{-width}\geq3)$",
)
axes[2].set_xticks(x, short, rotation=35, ha="right")
axes[2].set_ylim(0, 1.05)
axes[2].set_ylabel("Proportion among selected test knots")
axes[2].set_title("C  Scientific phenotype depends on score", loc="left", fontweight="bold")
axes[2].legend(frameon=False, fontsize=8)

fig.suptitle(
    "Standard anomaly scores select different scientific populations",
    fontweight="bold",
    fontsize=15,
    y=1.03,
)
fig.tight_layout()
fig.savefig(FIGURES / "figure_score_baseline_comparison.png", dpi=350, bbox_inches="tight")
fig.savefig(FIGURES / "figure_score_baseline_comparison.pdf", bbox_inches="tight")
plt.close(fig)

manifest = {
    "train_n": len(train_idx),
    "validation_n": len(val_idx),
    "test_n": len(test_idx),
    "test_tail_per_view": TAIL_N,
    "pca_k": FIXED_K,
    "residual_sketch_dimension_requested": RESIDUAL_SKETCH_DIM,
    "residual_fit_size": RESIDUAL_FIT_SIZE,
    "isolation_forest_trees": IFOREST_TREES,
    "main_methods": MAIN_METHODS,
    "primary_vote_rule": ">=3/5",
    "external_vote_rule": ">=3/4 without Khovanov",
    "note": (
        "The Mahalanobis score is evaluated in the leading train-fitted "
        "orthogonal residual sketch, not in the retained PCA coordinates."
    ),
}
(OUT / "stage23_design.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("\nScore--norm diagnostics:")
print(score_norm_diagnostics.to_string(index=False))
print("\nExternal phenotype summary:")
print(score_family_summary.to_string(index=False))
print("\nSaved Stage 23 to:", OUT)
