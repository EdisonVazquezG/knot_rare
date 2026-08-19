# %% [markdown]
# Stage 19 — Held-out validation of norm-conditioned hardness
#
# This is the missing validation of the paper's PRIMARY conditional regime.
# It reuses the already-fitted held-out PCA and the five already-trained AEs.
# No network is retrained.  Validation scores calibrate norm-conditional
# percentiles; the frozen test split is evaluated once at percentile >= 0.99.

# %%
from pathlib import Path
import joblib

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# 0. Preconditions
# ------------------------------------------------------------------
REQUIRED = (
    "meta",
    "X_dict",
    "INVARIANTS",
    "CONFIG",
    "OUTPUT_DIR",
    "ch",
    "train_idx",
    "val_idx",
    "test_idx",
    "conditional_hard_sets_100",
)
missing = [name for name in REQUIRED if name not in globals()]
if missing:
    raise RuntimeError(
        "Run the frozen notebook first. Missing objects: " + str(missing)
    )

if "ae_holdout_k99_runs" not in globals():
    raise RuntimeError(
        "ae_holdout_k99_runs is absent. Run/load the five completed held-out "
        "AE score blocks; the models do not need to be trained again."
    )

OUT = Path(OUTPUT_DIR) / "19_conditional_heldout_validation"
OUT.mkdir(parents=True, exist_ok=True)

N = len(meta)
PERCENTILE_THRESHOLD = 0.99
NORM_BINS = 100
AE_SEEDS = tuple(sorted(ae_holdout_k99_runs))

ALL5 = tuple(INVARIANTS)
NO_KHOVANOV = tuple(x for x in INVARIANTS if x != "Khovanov")
FAMILY_SPECS = {
    "All 5 >=3": (ALL5, 3),
    "No Khovanov >=3/4": (NO_KHOVANOV, 3),
}

val_idx_arr = np.asarray(val_idx, dtype=np.int64)
test_idx_arr = np.asarray(test_idx, dtype=np.int64)
test_universe = set(map(int, test_idx_arr))


# ------------------------------------------------------------------
# 1. Locate the already-fitted held-out PCA result dictionary
# ------------------------------------------------------------------
def _is_pca_holdout_dict(obj):
    if not isinstance(obj, dict) or set(obj) != set(INVARIANTS):
        return False
    return all(
        isinstance(obj[name], dict)
        and {"scaler", "pca", "test_sse"}.issubset(obj[name])
        for name in INVARIANTS
    )


pca_candidates = [
    (name, obj)
    for name, obj in list(globals().items())
    if _is_pca_holdout_dict(obj)
]
if not pca_candidates:
    raise RuntimeError(
        "Could not locate the held-out PCA result dictionary. Re-run the "
        "original held-out PCA block (it does not train AEs)."
    )

pca_candidates.sort(
    key=lambda pair: (
        "holdout" not in pair[0].lower(),
        "pca" not in pair[0].lower(),
        pair[0],
    )
)
PCA_RESULT_NAME, pca_heldout = pca_candidates[0]
print("Using held-out PCA object:", PCA_RESULT_NAME)


# ------------------------------------------------------------------
# 2. Target-free calibration utilities
# ------------------------------------------------------------------
def _scaled_log_sq_norm(X, scaler):
    Z = scaler.transform(X)
    return np.log1p(np.sum(np.asarray(Z, dtype=np.float64) ** 2, axis=1))


def _pca_sse(X, scaler, pca):
    Z = scaler.transform(X)
    Zhat = pca.inverse_transform(pca.transform(Z))
    return np.sum((Z - Zhat) ** 2, axis=1)


def _quantile_edges(values, n_bins):
    edges = np.quantile(
        np.asarray(values, dtype=float),
        np.linspace(0.0, 1.0, n_bins + 1),
    )
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf], dtype=float)
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    return edges


def _assign_bins(values, edges):
    return np.searchsorted(edges[1:-1], values, side="right").astype(np.int32)


def calibrated_conditional_percentiles(
    calibration_score,
    calibration_norm,
    evaluation_score,
    evaluation_norm,
    n_bins=100,
):
    """Empirical score percentile within validation-defined norm bins."""
    calibration_score = np.asarray(calibration_score, dtype=float)
    evaluation_score = np.asarray(evaluation_score, dtype=float)
    edges = _quantile_edges(calibration_norm, n_bins=n_bins)
    cal_bin = _assign_bins(calibration_norm, edges)
    eval_bin = _assign_bins(evaluation_norm, edges)

    percentile = np.empty(len(evaluation_score), dtype=float)
    populated = np.unique(cal_bin)

    for b in np.unique(eval_bin):
        target = int(b)
        reference = calibration_score[cal_bin == target]
        if len(reference) == 0:
            nearest = int(populated[np.argmin(np.abs(populated - target))])
            reference = calibration_score[cal_bin == nearest]
        reference = np.sort(reference)
        where = eval_bin == target
        # Add-one denominator avoids assigning a finite observation p=1.
        percentile[where] = (
            np.searchsorted(reference, evaluation_score[where], side="right")
            / (len(reference) + 1.0)
        )

    return percentile, eval_bin, edges


def sets_to_family_consensus(view_sets):
    result = {}
    for family, (names, k) in FAMILY_SPECS.items():
        count = np.zeros(N, dtype=np.uint8)
        for name in names:
            mask = np.zeros(N, dtype=bool)
            if view_sets[name]:
                mask[np.fromiter(view_sets[name], dtype=np.int64)] = True
            count += mask
        result[family] = set(map(int, np.flatnonzero(count >= k)))
    return result


def compare_sets(name_a, set_a, name_b, set_b):
    overlap = set_a & set_b
    union = set_a | set_b
    return {
        "set_a": name_a,
        "set_b": name_b,
        "size_a": len(set_a),
        "size_b": len(set_b),
        "overlap": len(overlap),
        "jaccard": len(overlap) / len(union) if union else np.nan,
        "fraction_a_in_b": len(overlap) / len(set_a) if set_a else np.nan,
        "fraction_b_in_a": len(overlap) / len(set_b) if set_b else np.nan,
    }


delta_abs = np.abs(
    meta[CONFIG.s_col].to_numpy(float)
    - meta["signature"].to_numpy(float)
)
nonalt = meta["is_alternating"].to_numpy() == 0


def phenotype(name, selected):
    idx = np.asarray(sorted(selected), dtype=np.int64)
    values = delta_abs[idx[nonalt[idx]]]
    return {
        "analysis": name,
        "n": len(idx),
        "nonalternating_n": len(values),
        "abs_delta_positive_prop": (
            float(np.mean(values > 0)) if len(values) else np.nan
        ),
        "mean_abs_delta": float(np.mean(values)) if len(values) else np.nan,
        "abs_delta_ge_4_prop": (
            float(np.mean(values >= 4)) if len(values) else np.nan
        ),
    }


# ------------------------------------------------------------------
# 3. Held-out conditional PCA
# ------------------------------------------------------------------
pca_view_sets = {}
pca_view_rows = []

for invariant in INVARIANTS:
    result = pca_heldout[invariant]
    scaler = result["scaler"]
    pca = result["pca"]

    cal_sse = _pca_sse(X_dict[invariant][val_idx_arr], scaler, pca)
    test_sse = np.asarray(result["test_sse"], dtype=float)
    if len(test_sse) != len(test_idx_arr):
        raise AssertionError(f"PCA test length mismatch for {invariant}")

    cal_norm = _scaled_log_sq_norm(X_dict[invariant][val_idx_arr], scaler)
    test_norm = _scaled_log_sq_norm(X_dict[invariant][test_idx_arr], scaler)
    percentile, test_bins, edges = calibrated_conditional_percentiles(
        cal_sse,
        cal_norm,
        test_sse,
        test_norm,
        n_bins=NORM_BINS,
    )

    selected_local = percentile >= PERCENTILE_THRESHOLD
    selected = set(map(int, test_idx_arr[selected_local]))
    pca_view_sets[invariant] = selected
    pca_view_rows.append({
        "method": "heldout_PCA",
        "invariant": invariant,
        "n_validation": len(val_idx_arr),
        "n_test": len(test_idx_arr),
        "n_selected": len(selected),
        "selected_prop": len(selected) / len(test_idx_arr),
        "percentile_threshold": PERCENTILE_THRESHOLD,
        "n_effective_norm_bins": len(np.unique(test_bins)),
    })

pca_family_sets = sets_to_family_consensus(pca_view_sets)


# ------------------------------------------------------------------
# 4. Held-out conditional AEs, using saved validation/test scores
# ------------------------------------------------------------------
if "AE_SCALER_DIR" not in globals():
    raise RuntimeError(
        "AE_SCALER_DIR is missing. Re-run the lightweight AE-loading block "
        "from the frozen notebook; training is not required."
    )

ae_seed_view_sets = {}
ae_seed_family_sets = {}
ae_view_rows = []

for seed in AE_SEEDS:
    scaler_path = Path(AE_SCALER_DIR) / f"heldout_ae_scalers_seed_{seed}.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(scaler_path)
    scalers = joblib.load(scaler_path)
    view_sets = {}

    for invariant in INVARIANTS:
        result = ae_holdout_k99_runs[seed][invariant]
        if not np.array_equal(np.asarray(result["val_idx"]), val_idx_arr):
            raise AssertionError(f"AE val-index mismatch: {seed}, {invariant}")
        if not np.array_equal(np.asarray(result["test_idx"]), test_idx_arr):
            raise AssertionError(f"AE test-index mismatch: {seed}, {invariant}")

        scaler = scalers[invariant]
        cal_sse = np.asarray(result["val_sse"], dtype=float)
        test_sse = np.asarray(result["test_sse"], dtype=float)
        cal_norm = _scaled_log_sq_norm(X_dict[invariant][val_idx_arr], scaler)
        test_norm = _scaled_log_sq_norm(X_dict[invariant][test_idx_arr], scaler)

        percentile, test_bins, edges = calibrated_conditional_percentiles(
            cal_sse,
            cal_norm,
            test_sse,
            test_norm,
            n_bins=NORM_BINS,
        )
        selected = set(map(
            int,
            test_idx_arr[percentile >= PERCENTILE_THRESHOLD],
        ))
        view_sets[invariant] = selected
        ae_view_rows.append({
            "method": f"heldout_AE_seed_{seed}",
            "seed": seed,
            "invariant": invariant,
            "n_validation": len(val_idx_arr),
            "n_test": len(test_idx_arr),
            "n_selected": len(selected),
            "selected_prop": len(selected) / len(test_idx_arr),
            "percentile_threshold": PERCENTILE_THRESHOLD,
            "n_effective_norm_bins": len(np.unique(test_bins)),
        })

    ae_seed_view_sets[seed] = view_sets
    ae_seed_family_sets[seed] = sets_to_family_consensus(view_sets)


# ------------------------------------------------------------------
# 5. Majority across AE seeds and comparison with frozen conditional set
# ------------------------------------------------------------------
frozen_view_masks = ch.hard_sets_to_masks(conditional_hard_sets_100, N)
frozen_family_sets = sets_to_family_consensus({
    name: set(map(int, np.flatnonzero(frozen_view_masks[name])))
    for name in INVARIANTS
})
frozen_test_sets = {
    family: selected & test_universe
    for family, selected in frozen_family_sets.items()
}

ae_majority_sets = {}
ae_strict_sets = {}
for family in FAMILY_SPECS:
    frequency = np.zeros(N, dtype=np.uint8)
    for seed in AE_SEEDS:
        selected = ae_seed_family_sets[seed][family]
        if selected:
            frequency[np.fromiter(selected, dtype=np.int64)] += 1
    ae_majority_sets[family] = set(map(int, np.flatnonzero(frequency >= 3)))
    ae_strict_sets[family] = set(map(int, np.flatnonzero(frequency == 5)))

overlap_rows = []
phenotype_rows = []
family_summary_rows = []

for family in FAMILY_SPECS:
    frozen_set = frozen_test_sets[family]
    pca_set = pca_family_sets[family]
    ae_majority = ae_majority_sets[family]
    ae_strict = ae_strict_sets[family]

    family_summary_rows.extend([
        {"family": family, "method": "frozen_full_fit_restricted_to_test", "n": len(frozen_set)},
        {"family": family, "method": "heldout_PCA_conditional", "n": len(pca_set)},
        {"family": family, "method": "heldout_AE_majority_3of5", "n": len(ae_majority)},
        {"family": family, "method": "heldout_AE_strict_5of5", "n": len(ae_strict)},
    ])

    overlap_rows.extend([
        {"family": family, **compare_sets("frozen_test", frozen_set, "heldout_PCA", pca_set)},
        {"family": family, **compare_sets("frozen_test", frozen_set, "AE_majority", ae_majority)},
        {"family": family, **compare_sets("heldout_PCA", pca_set, "AE_majority", ae_majority)},
    ])

    phenotype_rows.extend([
        {"family": family, **phenotype("frozen_test", frozen_set)},
        {"family": family, **phenotype("heldout_PCA", pca_set)},
        {"family": family, **phenotype("AE_majority", ae_majority)},
        {"family": family, **phenotype("AE_strict", ae_strict)},
    ])

    for seed in AE_SEEDS:
        seed_set = ae_seed_family_sets[seed][family]
        overlap_rows.append({
            "family": family,
            **compare_sets("heldout_PCA", pca_set, f"AE_seed_{seed}", seed_set),
        })

conditional_heldout_view_summary = pd.DataFrame(pca_view_rows + ae_view_rows)
conditional_heldout_family_summary = pd.DataFrame(family_summary_rows)
conditional_heldout_overlap = pd.DataFrame(overlap_rows)
conditional_heldout_phenotype = pd.DataFrame(phenotype_rows)

display(conditional_heldout_view_summary)
display(conditional_heldout_family_summary)
display(conditional_heldout_overlap)
display(conditional_heldout_phenotype)

conditional_heldout_view_summary.to_csv(
    OUT / "conditional_heldout_view_summary.csv", index=False
)
conditional_heldout_family_summary.to_csv(
    OUT / "conditional_heldout_family_summary.csv", index=False
)
conditional_heldout_overlap.to_csv(
    OUT / "conditional_heldout_overlap.csv", index=False
)
conditional_heldout_phenotype.to_csv(
    OUT / "conditional_heldout_phenotype.csv", index=False
)

# Reproducible member table.
member_frame = meta.loc[test_idx_arr, [
    CONFIG.universe.id_col,
    "number_of_crossings",
    "is_alternating",
    "signature",
    CONFIG.s_col,
]].copy()
member_frame["delta_abs"] = (
    member_frame[CONFIG.s_col] - member_frame["signature"]
).abs()
for family in FAMILY_SPECS:
    key = ch.safe_name(family)
    member_frame[f"{key}_pca_conditional"] = member_frame.index.isin(
        pca_family_sets[family]
    )
    member_frame[f"{key}_ae_majority"] = member_frame.index.isin(
        ae_majority_sets[family]
    )
    member_frame[f"{key}_frozen_test"] = member_frame.index.isin(
        frozen_test_sets[family]
    )
member_frame.to_csv(OUT / "conditional_heldout_members.csv", index=True)

print("\nSaved Stage 19 to:")
print(OUT)

