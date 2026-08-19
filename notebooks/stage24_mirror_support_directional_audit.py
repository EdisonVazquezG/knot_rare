# %% [markdown]
# Stage 24 — Mirror support bridge and directional-gap audit
#
# This standalone stage closes three presentation gaps:
#   1. it makes the canonical 373 -> 398 Khovanov support bridge explicit;
#   2. it adds canonical rows beside the randomized-mirror rows used in Table 3;
#   3. it reports the secondary signed diagnostic P(s-sigma>0 | |s-sigma|>0)
#      under the canonical representative and all five randomized mirror choices.
#
# Important: the canonical 398 representation is the 373-vector embedded by
# 25 identically zero reflected coordinates. This zero padding preserves every
# standardized norm, PCA reconstruction error, norm bin and hard-set decision.
# It therefore isolates support expansion from orientation randomization.

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 0. Paths and frozen constants
# ---------------------------------------------------------------------------
DEFAULT_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants/"
    "processed_consensus_hardness/corrected_run_20260819"
)
ROOT = Path(globals().get("OUTPUT_DIR", DEFAULT_ROOT))
if not ROOT.exists():
    raise FileNotFoundError(f"Frozen run not found: {ROOT}")

OUT = ROOT / "24_mirror_support_directional_audit"
OUT.mkdir(parents=True, exist_ok=True)
STAGE18B = ROOT / "18B_mirror_representation_robustness"
STAGE18C = ROOT / "18C_random_mirror_exact_nulls"

INVARIANTS = ("Alexander", "Jones", "HOMFLY-PT", "Theta", "Khovanov")
ID_COL = "knot_id_base"
RANDOM_SEEDS = (20260830, 20260831, 20260832, 20260833, 20260834)


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
    if recursive_name:
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
COND_HARD_PATH = first_existing([], "conditional_100bins_hard_sets.csv")
COND_SCORE_PATH = first_existing([], "conditional_100bins_scores.npz")
RAW_HARD_PATH = first_existing([], "hard_sets_by_stable_id.csv")

atlas = pd.read_parquet(ATLAS_PATH).reset_index(drop=True)
atlas[ID_COL] = atlas[ID_COL].astype(str)
if atlas[ID_COL].duplicated().any():
    raise AssertionError("Duplicated atlas IDs")
N = len(atlas)

S_COL = next(
    c for c in ("s_invariant_qc", "s_invariant", "s") if c in atlas
)
signature = atlas["signature"].to_numpy(float)
s_value = atlas[S_COL].to_numpy(float)
canonical_delta = s_value - signature
canonical_gap = np.abs(canonical_delta)
nonalternating = atlas["is_alternating"].to_numpy(int) == 0

id_to_pos = pd.Series(np.arange(N, dtype=np.int64), index=atlas[ID_COL]).to_dict()


def load_id_masks(path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    if not {"invariant", ID_COL}.issubset(frame):
        raise KeyError(f"Unexpected hard-set schema in {path}: {frame.columns.tolist()}")
    result = {}
    for name in INVARIANTS:
        ids = frame.loc[frame["invariant"].eq(name), ID_COL].astype(str)
        mask = np.zeros(N, dtype=bool)
        mask[[id_to_pos[x] for x in ids]] = True
        result[name] = mask
    return result


baseline_cond = load_id_masks(COND_HARD_PATH)
baseline_raw = load_id_masks(RAW_HARD_PATH)
with np.load(COND_SCORE_PATH, allow_pickle=False) as payload:
    baseline_bins = {
        name: np.asarray(payload[f"{safe_name(name)}_norm_bin"], dtype=np.int32)
        for name in INVARIANTS
    }


def at_least(masks: dict[str, np.ndarray], k: int) -> np.ndarray:
    count = np.zeros(N, dtype=np.uint8)
    for name in INVARIANTS:
        count += masks[name]
    return count >= k


def strict_all(masks: dict[str, np.ndarray]) -> np.ndarray:
    out = np.ones(N, dtype=bool)
    for name in INVARIANTS:
        out &= masks[name]
    return out


baseline_condensus = at_least(baseline_cond, 3)
baseline_raw_strict = strict_all(baseline_raw)
if baseline_condensus.sum() != 413:
    raise AssertionError(f"Expected canonical conditional n=413; found {baseline_condensus.sum()}")
if baseline_raw_strict.sum() != 292:
    raise AssertionError(f"Expected canonical raw n=292; found {baseline_raw_strict.sum()}")


# ---------------------------------------------------------------------------
# 1. Canonical 373 -> zero-padded 398 support bridge
# ---------------------------------------------------------------------------
# There is no model refit here: the 25 new columns are zero for every
# canonical row. StandardScaler maps them to zero; padding the PCA loading
# matrix by the same 25 zeros produces exactly the original scores.
support_audit_rows = []
for name in INVARIANTS:
    if name == "Khovanov":
        original_dim, padded_dim, padded_zero = 373, 398, 25
        reason = "25 reflected coordinates are identically zero in the canonical embedding"
    else:
        # No support change was made for these views.
        original_dim = padded_dim = {
            "Alexander": 17,
            "Jones": 51,
            "HOMFLY-PT": 152,
            "Theta": 841,
        }[name]
        padded_zero = 0
        reason = "representation unchanged"
    support_audit_rows.append(
        {
            "invariant": name,
            "canonical_dimension": original_dim,
            "mirror_closed_dimension": padded_dim,
            "zero_padded_coordinates": padded_zero,
            "canonical_norms_preserved": True,
            "canonical_reconstruction_scores_preserved": True,
            "canonical_norm_bins_preserved": True,
            "canonical_hard_mask_preserved": True,
            "reason": reason,
        }
    )

canonical_support_audit = pd.DataFrame(support_audit_rows)
canonical_support_audit.to_csv(OUT / "canonical_373_to_398_support_audit.csv", index=False)

# Save an explicit lightweight canonical-398 checkpoint. It is deliberately
# identical to the canonical 373 checkpoint because zero padding changes no
# score. This gives downstream table code a paired baseline with the same
# declared support as the randomized runs.
canonical_398_checkpoint = OUT / "orientation_canonical_398_checkpoint.npz"
checkpoint_payload = {"mirror_mask": np.zeros(N, dtype=bool)}
for name in INVARIANTS:
    key = safe_name(name)
    checkpoint_payload[f"raw_{key}"] = baseline_raw[name]
    checkpoint_payload[f"cond_{key}"] = baseline_cond[name]
    checkpoint_payload[f"normbin_{key}"] = baseline_bins[name].astype(np.int16)
np.savez_compressed(canonical_398_checkpoint, **checkpoint_payload)


# ---------------------------------------------------------------------------
# 2. Load mirror runs and compare selections on equal declared support
# ---------------------------------------------------------------------------
def load_checkpoint(label: str) -> tuple[np.ndarray, dict, dict]:
    path = STAGE18B / f"orientation_{label}_checkpoint.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        mirror_mask = np.asarray(payload["mirror_mask"], dtype=bool)
        raw = {
            name: np.asarray(payload[f"raw_{safe_name(name)}"], dtype=bool)
            for name in INVARIANTS
        }
        cond = {
            name: np.asarray(payload[f"cond_{safe_name(name)}"], dtype=bool)
            for name in INVARIANTS
        }
    return mirror_mask, raw, cond


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = a | b
    return float(np.sum(a & b) / np.sum(union)) if union.any() else np.nan


run_payloads = {
    "baseline_canonical_373": (
        np.zeros(N, dtype=bool),
        baseline_raw,
        baseline_cond,
    ),
    "baseline_canonical_398": (
        np.zeros(N, dtype=bool),
        baseline_raw,
        baseline_cond,
    ),
}
run_payloads["global_all_mirrored_398"] = load_checkpoint("global_all_mirrored")
for seed in RANDOM_SEEDS:
    run_payloads[f"random_seed_{seed}_398"] = load_checkpoint(f"random_seed_{seed}")

selection_rows = []
directional_rows = []
for run, (mirror_mask, raw_masks, cond_masks) in run_payloads.items():
    conditional = at_least(cond_masks, 3)
    raw_strict = strict_all(raw_masks)
    regimes = {
        "conditional_all5_ge3": conditional,
        "conditional_only": conditional & ~raw_strict,
        "shared_raw_and_conditional": conditional & raw_strict,
        "background": ~(conditional | raw_strict),
    }
    declared_support = 373 if run.endswith("373") else 398
    selection_rows.append(
        {
            "run": run,
            "declared_khovanov_dimension": declared_support,
            "mirror_fraction": float(mirror_mask.mean()),
            "conditional_n": int(conditional.sum()),
            "raw_strict_n": int(raw_strict.sum()),
            "overlap_with_canonical_conditional": int(np.sum(conditional & baseline_condensus)),
            "jaccard_with_canonical_conditional": jaccard(conditional, baseline_condensus),
            "canonical_conditional_recovered_prop": float(
                np.sum(conditional & baseline_condensus) / baseline_condensus.sum()
            ),
        }
    )

    # Under mirroring, both s and sigma change sign, hence delta changes sign
    # while |delta| is invariant.
    oriented_delta = canonical_delta * np.where(mirror_mask, -1.0, 1.0)
    oriented_gap = np.abs(oriented_delta)
    for regime, mask in regimes.items():
        target = mask & nonalternating
        nonzero = target & (oriented_gap > 0)
        directional_rows.append(
            {
                "run": run,
                "regime": regime,
                "n": int(mask.sum()),
                "nonalternating_n": int(target.sum()),
                "nonzero_gap_n": int(nonzero.sum()),
                "delta_positive_prop_among_nonalternating": (
                    float(np.mean(oriented_delta[target] > 0)) if target.any() else np.nan
                ),
                "abs_gap_positive_prop_among_nonalternating": (
                    float(np.mean(oriented_gap[target] > 0)) if target.any() else np.nan
                ),
                "positive_given_nonzero_gap": (
                    float(np.mean(oriented_delta[nonzero] > 0)) if nonzero.any() else np.nan
                ),
                "negative_given_nonzero_gap": (
                    float(np.mean(oriented_delta[nonzero] < 0)) if nonzero.any() else np.nan
                ),
                "mean_signed_gap": (
                    float(np.mean(oriented_delta[target])) if target.any() else np.nan
                ),
                "mean_abs_gap": (
                    float(np.mean(oriented_gap[target])) if target.any() else np.nan
                ),
            }
        )

mirror_support_selection = pd.DataFrame(selection_rows)
directional_gap_audit = pd.DataFrame(directional_rows)
mirror_support_selection.to_csv(OUT / "mirror_support_selection_bridge.csv", index=False)
directional_gap_audit.to_csv(OUT / "mirror_directional_gap_audit.csv", index=False)


# ---------------------------------------------------------------------------
# 3. Canonical rows for the mirror-null paper table
# ---------------------------------------------------------------------------
summary_path = STAGE18C / "mirror_random_exact_null_summary_all_runs.csv"
observed_path = STAGE18C / "mirror_random_observed_all_runs.csv"
if not summary_path.exists() or not observed_path.exists():
    raise FileNotFoundError(
        "Stage 18C outputs are required for canonical Table-3 rows. "
        "Run stage18C_random_mirror_exact_nulls.py first."
    )

mirror_null_summary = pd.read_csv(summary_path)
mirror_observed = pd.read_csv(observed_path)
baseline_summary = mirror_null_summary.loc[
    mirror_null_summary["run"].eq("baseline_canonical")
].copy()
if baseline_summary.empty:
    raise RuntimeError("Stage 18C summary has no baseline_canonical rows")

baseline_398_summary = baseline_summary.copy()
baseline_398_summary["run"] = "baseline_canonical_398"
baseline_398_summary["support_note"] = (
    "373 canonical Khovanov coordinates plus 25 identically zero reflected coordinates"
)
baseline_summary["run"] = "baseline_canonical_373"
baseline_summary["support_note"] = "original 373-coordinate canonical Khovanov support"

random_summary = mirror_null_summary.loc[
    mirror_null_summary["run"].astype(str).str.startswith("random_seed_")
].copy()
random_summary["support_note"] = "398-coordinate mirror-closed Khovanov support"

table3_with_canonical_rows = pd.concat(
    [baseline_summary, baseline_398_summary, random_summary], ignore_index=True
)
table3_with_canonical_rows.to_csv(
    OUT / "table3_mirror_nulls_with_canonical_support_rows.csv", index=False
)


# ---------------------------------------------------------------------------
# 4. Explicit representation-action note for Methods/Discussion
# ---------------------------------------------------------------------------
view_action = pd.DataFrame(
    [
        {
            "invariant": "Alexander",
            "mirror_sensitive_in_stored_representation": False,
            "action": "unchanged",
            "interpretation": "mirror-invariant polynomial coefficients; fixed vote",
        },
        {
            "invariant": "Jones",
            "mirror_sensitive_in_stored_representation": True,
            "action": "exponent reversal J_q -> J_-q",
            "interpretation": "recomputed under each orientation",
        },
        {
            "invariant": "HOMFLY-PT",
            "mirror_sensitive_in_stored_representation": True,
            "action": "a-exponent reversal (a,z) -> (-a,z)",
            "interpretation": "recomputed under each orientation",
        },
        {
            "invariant": "Theta",
            "mirror_sensitive_in_stored_representation": False,
            "action": "simultaneous inversion audited as vector-invariant",
            "interpretation": "fixed vote in this stored representation",
        },
        {
            "invariant": "Khovanov",
            "mirror_sensitive_in_stored_representation": True,
            "action": "(q,t) -> (-q,-t) in 398-coordinate closed support",
            "interpretation": "recomputed under each orientation",
        },
    ]
)
view_action.to_csv(OUT / "mirror_action_by_representation.csv", index=False)

methods_note = r"""
\paragraph{Mirror action and support control.}
Alexander coefficients and the stored $\Theta$ vectors are invariant under the
audited mirror action and therefore contribute two fixed votes to the five-view
consensus. Jones, HOMFLY--PT and Khovanov are mirror-sensitive and were
recomputed. The canonical 373-coordinate Khovanov vector was also embedded in
the 398-coordinate mirror-closed support by adding 25 identically zero
coordinates. This embedding preserved norms, reconstruction scores, norm bins
and memberships exactly; consequently, comparisons between the canonical-398
baseline and randomized-398 runs isolate orientation choice rather than a
change of coordinate support.
""".strip()

results_note = r"""
\paragraph{Directional diagnostic.}
The primary mirror-robust endpoint is $G=|s-\sigma|$. As a secondary diagnostic
we report $\Pr(s-\sigma>0\mid G>0)$ for the canonical representative and for
each randomized mirror assignment. Because $s-\sigma$ changes sign under
mirroring, this directional quantity is interpreted as an orientation
sensitivity diagnostic, not as an invariant claim of positivity or
quasipositivity.
""".strip()

(OUT / "latex_mirror_support_methods_note.tex").write_text(methods_note + "\n", encoding="utf-8")
(OUT / "latex_directional_diagnostic_note.tex").write_text(results_note + "\n", encoding="utf-8")

print("\nSupport bridge:")
print(canonical_support_audit.to_string(index=False))
print("\nSelection bridge:")
print(mirror_support_selection.to_string(index=False))
print("\nDirectional diagnostic (conditional only/background):")
print(
    directional_gap_audit.loc[
        directional_gap_audit["regime"].isin(["conditional_only", "background"])
    ].to_string(index=False)
)
print("\nSaved Stage 24 to:", OUT)

