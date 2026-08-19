# %% [markdown]
# Stage 18B — Representation-level mirror robustness
#
# Run after the frozen corrected_run_20260819 notebook and the Stage 18B
# schema/closure audits.  The analysis uses the frozen PCA dimensions and
# tail/binnning specification.  It runs:
#   1. a global mirror transformation (positive control), and
#   2. five independent per-knot mirror choices shared across all views.
#
# Alexander and Theta are reused because they are unchanged under the audited
# transformations.  Jones, HOMFLY-PT and Khovanov are recomputed.  Khovanov is
# embedded in the 398-coordinate mirror-closed union.

# %%
from pathlib import Path
import gc
import json
import re

import numpy as np
import pandas as pd

try:
    _pca_errors_for_k = ch.pca_reconstruction_errors_for_k
except AttributeError:
    from consensus_hardness.pca import (
        pca_reconstruction_errors_for_k as _pca_errors_for_k,
    )


# ------------------------------------------------------------
# 0. Preconditions and frozen design
# ------------------------------------------------------------
REQUIRED_OBJECTS = (
    "meta",
    "meta_norm",
    "X_dict",
    "feature_cols_dict",
    "primary",
    "conditional_hard_sets_100",
    "conditional_results_100",
    "conditional_membership_sets",
    "INVARIANTS",
    "CONFIG",
    "OUTPUT_DIR",
    "ch",
    "target_free_canonicalization_audit",
    "homfly_mirror_validation",
    "theta_inversion_audit",
    "kh_union_audit",
)

missing_objects = [
    name for name in REQUIRED_OBJECTS
    if name not in globals()
]
if missing_objects:
    raise RuntimeError(
        "Run this script after the frozen notebook and the preceding "
        f"mirror audits. Missing objects: {missing_objects}"
    )

if not target_free_canonicalization_audit[
    "exact_feature_matrix_match"
].all():
    raise AssertionError("Target-free canonicalization audit did not pass.")

if float(
    homfly_mirror_validation["exact_row_match_prop"].iloc[0]
) != 1.0:
    raise AssertionError("Exact HOMFLY mirror validation did not pass.")

if float(theta_inversion_audit["equal_prop"].iloc[0]) != 1.0:
    raise AssertionError(
        "Theta is not invariant under the audited simultaneous inversion."
    )

if int(kh_union_audit["mirror_closed_dimension"].iloc[0]) != 398:
    raise AssertionError("Unexpected Khovanov mirror-closed dimension.")

MIRROR_B_DIR = (
    Path(OUTPUT_DIR)
    / "18B_mirror_representation_robustness"
)
MIRROR_B_DIR.mkdir(parents=True, exist_ok=True)

N = len(meta)
IDS = meta[CONFIG.universe.id_col].astype(str).to_numpy()
TAIL_MASS = float(CONFIG.pca.tail_mass)
NORM_BINS = 100
RANDOM_SEEDS = (20260830, 20260831, 20260832, 20260833, 20260834)

FIXED_K = {
    name: int(primary["fixed_results"][name]["k"])
    for name in INVARIANTS
}

if set(FIXED_K) != set(INVARIANTS):
    raise AssertionError("Frozen PCA dimensions are incomplete.")


# ------------------------------------------------------------
# 1. Coordinate transformations
# ------------------------------------------------------------
def _parse_homfly(column):
    match = re.fullmatch(r"a(-?\d+)_z(-?\d+)", str(column))
    if match is None:
        raise ValueError(f"Cannot parse HOMFLY coordinate: {column}")
    return tuple(map(int, match.groups()))


def _parse_khovanov(column):
    match = re.fullmatch(r"F_q(-?\d+)_t(-?\d+)", str(column))
    if match is None:
        raise ValueError(f"Cannot parse Khovanov coordinate: {column}")
    return tuple(map(int, match.groups()))


jones_cols = list(feature_cols_dict["Jones"])
jones_exponents = [int(str(col)[1:]) for col in jones_cols]
jones_index = {degree: i for i, degree in enumerate(jones_exponents)}
JONES_PERM = np.array(
    [jones_index[-degree] for degree in jones_exponents],
    dtype=np.int64,
)

homfly_cols = list(feature_cols_dict["HOMFLY-PT"])
homfly_coords = [_parse_homfly(col) for col in homfly_cols]
homfly_index = {coord: i for i, coord in enumerate(homfly_coords)}
HOMFLY_PERM = np.array(
    [homfly_index[(-a, z)] for a, z in homfly_coords],
    dtype=np.int64,
)

kh_cols = list(feature_cols_dict["Khovanov"])
kh_coords = [_parse_khovanov(col) for col in kh_cols]
kh_original_set = set(kh_coords)
kh_reflected_set = {(-q, -t) for q, t in kh_coords}
kh_missing = sorted(kh_reflected_set - kh_original_set)
KH_UNION_COORDS = kh_coords + kh_missing
kh_union_index = {coord: i for i, coord in enumerate(KH_UNION_COORDS)}
KH_ORIGINAL_POSITIONS = np.array(
    [kh_union_index[(q, t)] for q, t in kh_coords],
    dtype=np.int64,
)
KH_MIRROR_POSITIONS = np.array(
    [kh_union_index[(-q, -t)] for q, t in kh_coords],
    dtype=np.int64,
)

assert len(KH_UNION_COORDS) == 398
assert len(np.unique(KH_ORIGINAL_POSITIONS)) == len(kh_coords)
assert len(np.unique(KH_MIRROR_POSITIONS)) == len(kh_coords)


# ------------------------------------------------------------
# 2. Memory-conscious matrix and norm helpers
# ------------------------------------------------------------
def _apply_closed_permutation(X, mirror_mask, permutation, chunk=5000):
    """Apply a coordinate permutation only to mirrored rows."""
    X = np.asarray(X, dtype=np.float32)
    out = X.copy()
    selected = np.flatnonzero(mirror_mask)

    for start in range(0, len(selected), chunk):
        idx = selected[start:start + chunk]
        out[idx] = X[idx][:, permutation]

    return out


def _build_khovanov_union(X, mirror_mask, chunk=5000):
    """Embed original/mirrored Khovanov rows in the 398-D union."""
    X = np.asarray(X, dtype=np.float32)
    out = np.zeros((len(X), len(KH_UNION_COORDS)), dtype=np.float32)
    out[:, KH_ORIGINAL_POSITIONS] = X

    selected = np.flatnonzero(mirror_mask)
    for start in range(0, len(selected), chunk):
        idx = selected[start:start + chunk]
        out[idx] = 0.0
        out[np.ix_(idx, KH_MIRROR_POSITIONS)] = X[idx]

    return out


def _standardized_log_sq_norm(X, scaler, chunk=5000):
    """Compute log(1 + standardized squared norm) without retaining Xs."""
    out = np.empty(len(X), dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)

    for start in range(0, len(X), chunk):
        stop = min(start + chunk, len(X))
        block = np.asarray(X[start:stop], dtype=np.float64)
        block = (block - mean) / scale
        out[start:stop] = np.log1p(np.sum(block * block, axis=1))

    return out


def _work_matrix(name, mirror_mask):
    if name == "Jones":
        return _apply_closed_permutation(
            X_dict[name], mirror_mask, JONES_PERM
        )
    if name == "HOMFLY-PT":
        return _apply_closed_permutation(
            X_dict[name], mirror_mask, HOMFLY_PERM
        )
    if name == "Khovanov":
        return _build_khovanov_union(X_dict[name], mirror_mask)
    raise KeyError(name)


# ------------------------------------------------------------
# 3. Baseline comparison sets
# ------------------------------------------------------------
ALL5 = tuple(INVARIANTS)
NO_THETA = tuple(name for name in INVARIANTS if name != "Theta")
POLYNOMIAL = ("Alexander", "Jones", "HOMFLY-PT")


def _at_least(hard_sets, names, k):
    return ch.at_least_k_consensus(
        hard_sets=hard_sets,
        names=names,
        k=k,
        n_objects=N,
    )


BASELINE_SETS = {
    "raw_all5_strict": set(map(int, primary["consensus"])),
    "conditional_all5_ge3": set(map(
        int, conditional_membership_sets[("All 5", 3)]
    )),
    "conditional_no_theta_ge3": _at_least(
        conditional_hard_sets_100, NO_THETA, 3
    ),
    "conditional_polynomial_all3": set(map(
        int, conditional_membership_sets[("Polynomial only", 3)]
    )),
}

assert len(BASELINE_SETS["raw_all5_strict"]) == 292
assert len(BASELINE_SETS["conditional_all5_ge3"]) == 413
assert len(BASELINE_SETS["conditional_polynomial_all3"]) == 35


def _jaccard_details(reference, candidate):
    intersection = reference & candidate
    union = reference | candidate
    return {
        "baseline_size": len(reference),
        "selected_size": len(candidate),
        "overlap": len(intersection),
        "jaccard": len(intersection) / len(union) if union else np.nan,
        "baseline_recovered_prop": (
            len(intersection) / len(reference) if reference else np.nan
        ),
        "selected_in_baseline_prop": (
            len(intersection) / len(candidate) if candidate else np.nan
        ),
    }


delta_abs = np.abs(
    meta[CONFIG.s_col].to_numpy(float)
    - meta["signature"].to_numpy(float)
)
is_nonalt = meta["is_alternating"].to_numpy() == 0


def _gap_metrics(selected):
    idx = np.array(sorted(selected), dtype=np.int64)
    nonalt_idx = idx[is_nonalt[idx]]
    values = delta_abs[nonalt_idx]

    if len(values) == 0:
        return {
            "nonalternating_n": 0,
            "abs_delta_positive_prop": np.nan,
            "mean_abs_delta": np.nan,
            "abs_delta_ge_4_prop": np.nan,
        }

    return {
        "nonalternating_n": len(values),
        "abs_delta_positive_prop": float(np.mean(values > 0)),
        "mean_abs_delta": float(np.mean(values)),
        "abs_delta_ge_4_prop": float(np.mean(values >= 4)),
    }


# ------------------------------------------------------------
# 4. Compute/load one orientation realization
# ------------------------------------------------------------
CHANGED_VIEWS = ("Jones", "HOMFLY-PT", "Khovanov")
UNCHANGED_VIEWS = ("Alexander", "Theta")


def _checkpoint_path(label):
    return MIRROR_B_DIR / f"orientation_{label}_checkpoint.npz"


def _sets_from_checkpoint(payload):
    raw_hard = {}
    conditional_hard = {}
    norm_bins = {}

    for name in INVARIANTS:
        key = ch.safe_name(name)
        raw_hard[name] = set(np.flatnonzero(payload[f"raw_{key}"]))
        conditional_hard[name] = set(np.flatnonzero(payload[f"cond_{key}"]))
        norm_bins[name] = np.asarray(payload[f"normbin_{key}"], dtype=np.int32)

    return raw_hard, conditional_hard, norm_bins


def _compute_or_load(label, mirror_mask):
    checkpoint = _checkpoint_path(label)

    if checkpoint.exists():
        print(f"Loading checkpoint: {checkpoint.name}")
        with np.load(checkpoint, allow_pickle=False) as payload:
            saved_mask = np.asarray(payload["mirror_mask"], dtype=bool)
            if not np.array_equal(saved_mask, mirror_mask):
                raise AssertionError(f"Mirror-mask mismatch for {label}.")
            return _sets_from_checkpoint(payload)

    fixed_results = {}
    norm_frame = pd.DataFrame(index=np.arange(N))
    norm_col_map = {}

    # Alexander and Theta are exactly unchanged under the audited action.
    for name in UNCHANGED_VIEWS:
        fixed_results[name] = {
            "sse": np.asarray(primary["fixed_results"][name]["sse"])
        }
        norm_col = f"{ch.safe_name(name)}_log_sq_norm"
        norm_frame[norm_col] = meta_norm[norm_col].to_numpy()
        norm_col_map[name] = norm_col

    # Refit the three mirror-sensitive views at the frozen PCA dimensions.
    for name in CHANGED_VIEWS:
        print(
            f"[{label}] {name}: building matrix; "
            f"mirrored rows={int(mirror_mask.sum()):,}/{N:,}"
        )
        X_work = _work_matrix(name, mirror_mask)

        print(
            f"[{label}] {name}: PCA k={FIXED_K[name]}, "
            f"shape={X_work.shape}"
        )
        result = _pca_errors_for_k(X_work, k=FIXED_K[name])
        fixed_results[name] = {"sse": np.asarray(result["sse"])}

        norm_col = f"{ch.safe_name(name)}_log_sq_norm"
        norm_frame[norm_col] = _standardized_log_sq_norm(
            X_work, result["scaler"]
        )
        norm_col_map[name] = norm_col

        del result, X_work
        gc.collect()

    # Restore canonical view order.
    fixed_results = {name: fixed_results[name] for name in INVARIANTS}
    norm_col_map = {name: norm_col_map[name] for name in INVARIANTS}

    raw_hard = ch.build_hard_sets_from_fixed_results(
        fixed_results=fixed_results,
        score_name="sse",
        tail_mass=TAIL_MASS,
        stable_ids=meta[CONFIG.universe.id_col],
    )

    adjusted, diagnostics = ch.norm_adjusted_scores(
        fixed_results=fixed_results,
        norm_meta=norm_frame,
        norm_col_map=norm_col_map,
        method="conditional_percentile",
        n_bins=NORM_BINS,
        seed=CONFIG.random_seed,
    )

    conditional_hard = ch.build_hard_sets_from_fixed_results(
        fixed_results=adjusted,
        score_name="sse",
        tail_mass=TAIL_MASS,
        stable_ids=meta[CONFIG.universe.id_col],
    )

    norm_bins = {
        name: np.asarray(adjusted[name]["norm_bin"], dtype=np.int32)
        for name in INVARIANTS
    }

    payload = {"mirror_mask": mirror_mask.astype(bool)}
    for name in INVARIANTS:
        key = ch.safe_name(name)
        raw_mask = np.zeros(N, dtype=bool)
        cond_mask = np.zeros(N, dtype=bool)
        raw_mask[list(raw_hard[name])] = True
        cond_mask[list(conditional_hard[name])] = True
        payload[f"raw_{key}"] = raw_mask
        payload[f"cond_{key}"] = cond_mask
        payload[f"normbin_{key}"] = norm_bins[name].astype(np.int16)

    np.savez_compressed(checkpoint, **payload)
    diagnostics.to_csv(
        MIRROR_B_DIR / f"orientation_{label}_norm_diagnostics.csv",
        index=False,
    )
    print(f"Saved checkpoint: {checkpoint.name}")

    return raw_hard, conditional_hard, norm_bins


# ------------------------------------------------------------
# 5. Global mirror plus five randomized mirror choices
# ------------------------------------------------------------
run_specs = [
    ("global_all_mirrored", None, np.ones(N, dtype=bool))
]

for seed in RANDOM_SEEDS:
    rng = np.random.default_rng(seed)
    run_specs.append((
        f"random_seed_{seed}",
        seed,
        rng.random(N) < 0.5,
    ))

summary_rows = []
per_view_rows = []
selected_id_rows = []
run_selected_sets = {}

for label, seed, mirror_mask in run_specs:
    print("\n" + "=" * 72)
    print("Orientation run:", label)
    print("Mirror fraction:", float(mirror_mask.mean()))

    raw_hard, conditional_hard, norm_bins = _compute_or_load(
        label, mirror_mask
    )

    analysis_sets = {
        "raw_all5_strict": ch.consensus_from_hard_sets(raw_hard),
        "conditional_all5_ge3": _at_least(
            conditional_hard, ALL5, 3
        ),
        "conditional_no_theta_ge3": _at_least(
            conditional_hard, NO_THETA, 3
        ),
        "conditional_polynomial_all3": _at_least(
            conditional_hard, POLYNOMIAL, 3
        ),
    }

    run_selected_sets[label] = analysis_sets

    for analysis_name, selected in analysis_sets.items():
        row = {
            "run": label,
            "seed": seed,
            "mirror_fraction": float(mirror_mask.mean()),
            "analysis": analysis_name,
            **_jaccard_details(BASELINE_SETS[analysis_name], selected),
            **_gap_metrics(selected),
        }
        summary_rows.append(row)

        for idx in sorted(selected):
            selected_id_rows.append({
                "run": label,
                "seed": seed,
                "analysis": analysis_name,
                "knot_id_base": IDS[idx],
            })

    for name in INVARIANTS:
        for score_type, current, baseline in (
            ("raw", raw_hard[name], primary["hard_sets"][name]),
            (
                "conditional",
                conditional_hard[name],
                conditional_hard_sets_100[name],
            ),
        ):
            per_view_rows.append({
                "run": label,
                "seed": seed,
                "mirror_fraction": float(mirror_mask.mean()),
                "invariant": name,
                "score_type": score_type,
                **_jaccard_details(set(baseline), set(current)),
            })

mirror_selection_summary = pd.DataFrame(summary_rows)
mirror_per_view_stability = pd.DataFrame(per_view_rows)
mirror_selected_ids = pd.DataFrame(selected_id_rows)


# ------------------------------------------------------------
# 6. Pairwise stability across randomized choices
# ------------------------------------------------------------
pairwise_rows = []
random_labels = [f"random_seed_{seed}" for seed in RANDOM_SEEDS]

for analysis_name in BASELINE_SETS:
    for i, label_a in enumerate(random_labels):
        for label_b in random_labels[i + 1:]:
            details = _jaccard_details(
                run_selected_sets[label_a][analysis_name],
                run_selected_sets[label_b][analysis_name],
            )
            pairwise_rows.append({
                "analysis": analysis_name,
                "run_a": label_a,
                "run_b": label_b,
                **details,
            })

mirror_random_pairwise = pd.DataFrame(pairwise_rows)


# ------------------------------------------------------------
# 7. Save and display
# ------------------------------------------------------------
mirror_selection_summary.to_csv(
    MIRROR_B_DIR / "mirror_selection_stability_summary.csv",
    index=False,
)
mirror_per_view_stability.to_csv(
    MIRROR_B_DIR / "mirror_per_view_stability.csv",
    index=False,
)
mirror_selected_ids.to_csv(
    MIRROR_B_DIR / "mirror_selected_knot_ids.csv",
    index=False,
)
mirror_random_pairwise.to_csv(
    MIRROR_B_DIR / "mirror_random_seed_pairwise_jaccard.csv",
    index=False,
)

display(
    mirror_selection_summary.sort_values(["analysis", "run"])
)

display(
    mirror_per_view_stability.sort_values(
        ["score_type", "invariant", "run"]
    )
)

random_summary = (
    mirror_selection_summary[
        mirror_selection_summary["run"].str.startswith("random_seed_")
    ]
    .groupby("analysis", as_index=False)
    .agg(
        selected_size_mean=("selected_size", "mean"),
        selected_size_min=("selected_size", "min"),
        selected_size_max=("selected_size", "max"),
        jaccard_mean=("jaccard", "mean"),
        jaccard_min=("jaccard", "min"),
        baseline_recovered_mean=("baseline_recovered_prop", "mean"),
        abs_delta_positive_prop_mean=(
            "abs_delta_positive_prop", "mean"
        ),
        mean_abs_delta_mean=("mean_abs_delta", "mean"),
        abs_delta_ge_4_prop_mean=("abs_delta_ge_4_prop", "mean"),
    )
)

display(random_summary)

random_summary.to_csv(
    MIRROR_B_DIR / "mirror_random_seed_aggregate.csv",
    index=False,
)

decision = {
    "stage": "18B_mirror_representation_robustness",
    "n_random_orientation_seeds": len(RANDOM_SEEDS),
    "frozen_pca_dimensions": FIXED_K,
    "khovanov_mirror_closed_dimension": len(KH_UNION_COORDS),
    "theta_action_used": (
        "simultaneous exponent inversion; feature vectors audited as "
        "exactly invariant under this action"
    ),
    "global_mirror_all5_jaccard": float(
        mirror_selection_summary.loc[
            (mirror_selection_summary["run"] == "global_all_mirrored")
            & (
                mirror_selection_summary["analysis"]
                == "conditional_all5_ge3"
            ),
            "jaccard",
        ].iloc[0]
    ),
    "interpretation": (
        "The global mirror is a positive control. Random per-knot mirror "
        "choices assess whether hard-regime membership and the absolute "
        "Rasmussen-signature gap depend materially on catalogue chirality."
    ),
}

with open(
    MIRROR_B_DIR / "mirror_selection_decision.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(decision, handle, indent=2)

print("\nSaved Stage 18B outputs to:")
print(MIRROR_B_DIR)

