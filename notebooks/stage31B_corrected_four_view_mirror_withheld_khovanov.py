# %% [markdown]
# Stage 31B — Corrected four-view mirror robustness with Khovanov fully withheld
#
# Revision question:
#   Is the no-Khovanov 3-of-4 selection partly an artifact of the particular
#   representative chosen from each mirror pair?
#
# Prerequisite:
#   Stage 31A must have empirically established the Theta mirror action for the
#   stored encoding on checked small knots.  This stage HARD-STOPS otherwise.
#
# Design:
#   * Mirror ONLY the four selection views: Alexander, Jones, HOMFLY-PT, Theta.
#   * Leave Khovanov completely untouched: it is not mirrored and is never used
#     to construct a selection score.
#   * Use archived opposite-representative rows for Jones/HOMFLY-PT when present.
#   * Use Alexander unchanged under mirroring.
#   * Use the Stage-31A-gated Theta action (currently allowed only when the gate
#     says coefficient_sign_flip).
#   * Refit the full-atlas scaler/PCA and 100-bin conditional percentile after
#     every representative assignment, then select the upper 1% per view and
#     form the 3-of-4 consensus.
#   * Evaluate the resulting selections against the ORIGINAL stored F_-part
#     Khovanov diagonal count/support.  These are descriptive robustness results;
#     this stage does not attach a new permutation p-value.
#
# Fresh-session usage in Colab:
#   %run /content/drive/MyDrive/consensus_hardness_refactored/notebooks/\
#       stage31B_corrected_four_view_mirror_withheld_khovanov.py
#
# Environment overrides:
#   KNOT_PROJECT_DIR
#   KNOT_DATA_DIR
#   KNOT_OUTPUT_DIR
#   STAGE31B_RANDOM_SEEDS="20260830,20260831,20260832,20260833,20260834"
#   STAGE31B_INCLUDE_GLOBAL="1"
#   STAGE31B_PCA_SEED="20261123"
#   STAGE31B_BATCH_SIZE="8192"

# %%
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 0. Fresh-session setup
# ---------------------------------------------------------------------------
DEFAULT_PROJECT_DIR = Path("/content/drive/MyDrive/consensus_hardness_refactored")
DEFAULT_DATA_DIR = Path("/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants")
DEFAULT_ROOT = DEFAULT_DATA_DIR / "processed_consensus_hardness" / "corrected_run_20260819"

PROJECT_DIR = Path(os.environ.get("KNOT_PROJECT_DIR", str(DEFAULT_PROJECT_DIR)))
DATA_DIR = Path(os.environ.get("KNOT_DATA_DIR", str(DEFAULT_DATA_DIR)))
ROOT = Path(os.environ.get("KNOT_OUTPUT_DIR", str(DEFAULT_ROOT)))
OUT = ROOT / "31B_corrected_four_view_mirror_withheld_khovanov"
CHECKPOINTS = OUT / "checkpoints"


def maybe_mount_drive() -> None:
    if Path("/content").exists() and not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive  # type: ignore
            print("Google Drive is not mounted; requesting mount...")
            drive.mount("/content/drive")
        except Exception as exc:
            print("WARNING: could not mount Google Drive automatically:", exc)


maybe_mount_drive()
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
if not ROOT.exists():
    raise FileNotFoundError(f"Frozen run not found: {ROOT}")
for d in (OUT, CHECKPOINTS):
    d.mkdir(parents=True, exist_ok=True)
if (PROJECT_DIR / "src").exists() and str(PROJECT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "src"))

try:
    from consensus_hardness.io import load_table
    from consensus_hardness.preprocessing import (
        add_knot_ids,
        filter_crossings,
        canonicalize_mirrors_by_signature,
    )
    from consensus_hardness.representations import split_metadata_and_features
except Exception as exc:
    raise ImportError(
        "Could not import consensus_hardness from the project. "
        f"Expected project at {PROJECT_DIR}."
    ) from exc

ID_COL = "knot_id_base"
MIRROR_SYMBOL = "!"
NO_KHOVANOV = ("Alexander", "Jones", "HOMFLY-PT", "Theta")
VIEW_SPECS = {
    "Alexander": ("Alexander_upto17.csv", ["A"]),
    "Jones": ("Jones_upto17_MIRRORS.csv", ["J"]),
    "HOMFLY-PT": ("HomflyPt_upto15_MIRRORS.csv", ["a"]),
    "Theta": ("theta_upto15.csv", ["T"]),
}
FIXED_K = {"Alexander": 4, "Jones": 10, "HOMFLY-PT": 32, "Theta": 10}
TAIL_MASS = 0.01
N_BINS = 100
PCA_SEED = int(os.environ.get("STAGE31B_PCA_SEED", "20261123"))
BATCH_SIZE = int(os.environ.get("STAGE31B_BATCH_SIZE", "8192"))
INCLUDE_GLOBAL = os.environ.get("STAGE31B_INCLUDE_GLOBAL", "1") not in {"0", "false", "False"}
RANDOM_SEEDS = tuple(
    int(x.strip())
    for x in os.environ.get(
        "STAGE31B_RANDOM_SEEDS",
        "20260830,20260831,20260832,20260833,20260834",
    ).split(",")
    if x.strip()
)


def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "_")


def find_one(filename: str) -> Path:
    hits = sorted(ROOT.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename} below {ROOT}")
    return hits[0]


# ---------------------------------------------------------------------------
# 1. Gate from Stage 31A
# ---------------------------------------------------------------------------
GATE_PATH = ROOT / "31A_theta_mirror_action_audit" / "theta_mirror_action_gate.json"
if not GATE_PATH.exists():
    raise FileNotFoundError(
        "Stage 31A gate is missing. Run stage31A_theta_mirror_action_audit.py first:\n"
        f"  {GATE_PATH}"
    )
with open(GATE_PATH) as fh:
    gate = json.load(fh)

if gate.get("status") != "EMPIRICALLY_VERIFIED_ON_TESTED_SMALL_KNOTS":
    raise RuntimeError(
        "Stage 31A has not empirically verified the Theta mirror action. "
        f"Gate status: {gate.get('status')}"
    )
if gate.get("theta_mirror_action_for_stage31B") != "coefficient_sign_flip":
    raise RuntimeError(
        "This stage only implements a Theta coefficient sign flip after Stage 31A "
        "supports that action. The gate requested: "
        f"{gate.get('theta_mirror_action_for_stage31B')}"
    )
print("Stage 31A gate: PASS")
print("Theta mirror action used here: coefficient sign flip")
print("No general theorem is inferred from this empirical gate.")

# ---------------------------------------------------------------------------
# 2. Frozen atlas and FIXED Khovanov outcomes
# ---------------------------------------------------------------------------
atlas = pd.read_parquet(find_one("final_hard_regime_atlas.parquet")).reset_index(drop=True)
phenotype = pd.read_parquet(find_one("complete_mathematical_phenotype_atlas.parquet")).reset_index(drop=True)
for label, frame in (("atlas", atlas), ("phenotype", phenotype)):
    if ID_COL not in frame:
        raise KeyError(f"{label} lacks {ID_COL}")
    frame[ID_COL] = frame[ID_COL].astype(str)
    if frame[ID_COL].duplicated().any():
        raise AssertionError(f"Duplicated IDs in {label}")

if not atlas[ID_COL].equals(phenotype[ID_COL]):
    phenotype = atlas[[ID_COL]].merge(phenotype, on=ID_COL, how="left", validate="one_to_one")

N = len(atlas)
FROZEN_IDS = atlas[ID_COL].astype(str).to_numpy()
TAIL_N = int(np.ceil(TAIL_MASS * N))
if N != 313230:
    print(f"WARNING: frozen atlas N={N:,}; historical run used 313,230.")

KH_DIAG_COL = next(
    c for c in (
        "khovanov_q_minus_2t_diagonal_count",
        "kh_diagonal_count",
        "khovanov_diagonal_count",
    ) if c in phenotype
)
KH_SUPPORT_COL = next(
    c for c in ("khovanov_support_size", "kh_support_size", "khovanov_f_support_size")
    if c in phenotype
)
KH_DIAG = phenotype[KH_DIAG_COL].to_numpy(float)
KH_SUPPORT = phenotype[KH_SUPPORT_COL].to_numpy(float)
if not np.isfinite(KH_DIAG).all() or not np.isfinite(KH_SUPPORT).all():
    raise RuntimeError("Non-finite Khovanov endpoint after frozen alignment")

# The external target is intentionally fixed before any mirror assignment.
pd.DataFrame({
    "design_item": [
        "Khovanov used in selection",
        "Khovanov representation mirrored",
        "Khovanov endpoint changed by assignment",
        "Theta mirror action gated by Stage31A",
    ],
    "value": [False, False, False, True],
}).to_csv(OUT / "khovanov_fully_withheld_design_audit.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Frozen canonical conditional masks (reference only)
# ---------------------------------------------------------------------------
hard_saved = pd.read_csv(find_one("conditional_100bins_hard_sets.csv"), dtype={ID_COL: str})
id_to_pos = pd.Series(np.arange(N, dtype=np.int64), index=FROZEN_IDS).to_dict()
canonical_masks: dict[str, np.ndarray] = {}
for name in NO_KHOVANOV:
    ids = hard_saved.loc[hard_saved["invariant"].eq(name), ID_COL].astype(str)
    mask = np.zeros(N, dtype=bool)
    unknown = sorted(set(ids) - set(id_to_pos))
    if unknown:
        raise KeyError(f"Unknown canonical {name} hard-set IDs, e.g. {unknown[:5]}")
    mask[[id_to_pos[x] for x in ids]] = True
    canonical_masks[name] = mask
    if int(mask.sum()) != TAIL_N:
        raise RuntimeError(
            f"Canonical {name} tail has {int(mask.sum())}; expected {TAIL_N}."
        )


def consensus_from_masks(masks: dict[str, np.ndarray]) -> np.ndarray:
    votes = np.zeros(N, dtype=np.uint8)
    for name in NO_KHOVANOV:
        votes += masks[name].astype(np.uint8)
    return votes >= 3


canonical_consensus = consensus_from_masks(canonical_masks)
if int(canonical_consensus.sum()) != 220:
    raise RuntimeError(
        f"Frozen canonical no-Khovanov consensus has n={int(canonical_consensus.sum())}, expected 220."
    )


def endpoint_summary(mask: np.ndarray) -> dict[str, float | int]:
    d = KH_DIAG[mask]
    s = KH_SUPPORT[mask]
    return {
        "selected_n": int(mask.sum()),
        "kh_diagonal_mean": float(np.mean(d)),
        "kh_diagonal_ge_3_prop": float(np.mean(d >= 3)),
        "kh_diagonal_ge_4_prop": float(np.mean(d >= 4)),
        "kh_support_mean": float(np.mean(s)),
    }


canonical_endpoint = endpoint_summary(canonical_consensus)
print("Frozen canonical no-Khovanov selection:", canonical_endpoint)
if abs(float(canonical_endpoint["kh_diagonal_mean"]) - 2.9727) > 0.02:
    raise RuntimeError(
        "Canonical endpoint does not reproduce the expected frozen mean near 2.973. "
        "Stop before interpreting mirror robustness."
    )

# ---------------------------------------------------------------------------
# 4. Mirror assignments
# ---------------------------------------------------------------------------
assignment_masks: dict[str, np.ndarray] = {"baseline_canonical": np.zeros(N, dtype=bool)}
if INCLUDE_GLOBAL:
    assignment_masks["global_all_mirrored"] = np.ones(N, dtype=bool)
for seed in RANDOM_SEEDS:
    rng = np.random.default_rng(seed)
    assignment_masks[f"random_seed_{seed}"] = rng.random(N) < 0.5

np.savez_compressed(
    OUT / "mirror_assignments.npz",
    **{label: mask for label, mask in assignment_masks.items()},
)

# Start result dictionaries with baseline reference; per-view mirrored masks are
# filled one representation at a time to keep memory use bounded.
run_view_masks: dict[str, dict[str, np.ndarray]] = {
    label: {} for label in assignment_masks
}
for name in NO_KHOVANOV:
    run_view_masks["baseline_canonical"][name] = canonical_masks[name].copy()

# ---------------------------------------------------------------------------
# 5. Score helpers matching the canonical full-atlas idea
# ---------------------------------------------------------------------------
def full_atlas_conditional(
    X: np.ndarray,
    *,
    k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-atlas scaler/PCA + 100-bin within-norm average-rank percentile."""
    X = np.asarray(X, dtype=np.float32)
    scaler = StandardScaler(copy=True)
    Z = scaler.fit_transform(X).astype(np.float32)

    pca = PCA(n_components=int(k), svd_solver="randomized", random_state=seed)
    pca.fit(Z)

    n = len(Z)
    sse = np.empty(n, dtype=np.float64)
    norm_sq = np.empty(n, dtype=np.float64)
    for start in range(0, n, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, n)
        z = Z[start:stop]
        retained = pca.transform(z)
        residual = z - pca.inverse_transform(retained)
        sse[start:stop] = np.sum(residual.astype(np.float64) ** 2, axis=1)
        norm_sq[start:stop] = np.sum(z.astype(np.float64) ** 2, axis=1)

    lognorm = np.log1p(norm_sq)
    edges = np.quantile(lognorm, np.linspace(0.0, 1.0, N_BINS + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf], dtype=float)
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    bins = np.searchsorted(edges[1:-1], lognorm, side="right").astype(np.int16)

    cond = np.empty(n, dtype=np.float64)
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        # Average-rank percentile, matching the frozen full-atlas convention.
        cond[idx] = rankdata(sse[idx], method="average") / len(idx)

    del Z
    gc.collect()
    return cond, bins


def stable_top_tail(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if len(scores) != N or not np.isfinite(scores).all():
        raise ValueError("Invalid full-atlas score")
    order = np.lexsort((FROZEN_IDS, scores))
    mask = np.zeros(N, dtype=bool)
    mask[order[-TAIL_N:]] = True
    return mask


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = a | b
    return float(np.sum(a & b) / np.sum(u)) if u.any() else np.nan

# ---------------------------------------------------------------------------
# 6. Load one representation at a time and rerun every mirror assignment
# ---------------------------------------------------------------------------
view_audit_rows = []
reproduction_rows = []

for view_index, name in enumerate(NO_KHOVANOV):
    filename, prefixes = VIEW_SPECS[name]
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)

    print("\n" + "=" * 78)
    print(f"VIEW: {name} | loading {path.name}")
    print("=" * 78)

    raw = load_table(path)
    cleaned = add_knot_ids(raw, id_col="knot_id", mirror_symbol=MIRROR_SYMBOL)
    cleaned = filter_crossings(cleaned, min_crossings=3, max_crossings=15)
    canonical = canonicalize_mirrors_by_signature(
        cleaned,
        id_col="knot_id",
        base_col=ID_COL,
        signature_col="signature",
        mirror_symbol=MIRROR_SYMBOL,
    )

    # Reorder canonical rows to the exact frozen atlas order.
    canon_index = canonical.set_index(ID_COL, drop=False)
    missing = [x for x in FROZEN_IDS if x not in canon_index.index]
    if missing:
        raise RuntimeError(f"{name}: {len(missing)} frozen IDs missing from source, e.g. {missing[:5]}")
    canon_rows = canon_index.loc[FROZEN_IDS].reset_index(drop=True)
    _, ids_check, Xcanon, feat_cols = split_metadata_and_features(
        canon_rows,
        feature_prefixes=prefixes,
        id_col=ID_COL,
    )
    if not np.array_equal(ids_check.astype(str), FROZEN_IDS):
        raise RuntimeError(f"{name}: canonical feature alignment failed")

    # Construct the actual opposite representative in the stored coordinate
    # convention. Jones/HOMFLY use the archived mirror row. Alexander is kept
    # fixed. Theta uses the empirically gated sign action.
    opposite_missing = 0
    if name in ("Jones", "HOMFLY-PT"):
        # Vectorized opposite-representative lookup.  The previous implementation
        # used a Python loop over 313k knots; this version builds the alternative
        # table once and reindexes it to the frozen IDs.
        canonical_id_by_base = pd.Series(
            canon_rows["knot_id_clean"].astype(str).to_numpy(),
            index=canon_rows[ID_COL].astype(str).to_numpy(),
        )
        current_id = cleaned[ID_COL].astype(str).map(canonical_id_by_base)
        alt_rows = cleaned.loc[
            current_id.notna()
            & cleaned["knot_id_clean"].astype(str).ne(current_id.astype(str))
        ].copy()
        alt_rows = alt_rows.drop_duplicates(ID_COL, keep="first").set_index(ID_COL, drop=False)

        has_alt = pd.Index(FROZEN_IDS).isin(alt_rows.index)
        opposite_missing = int((~has_alt).sum())
        mirror_rows = canon_rows.copy()
        if has_alt.any():
            replacement = alt_rows.loc[FROZEN_IDS[has_alt], feat_cols]
            mirror_rows.loc[has_alt, feat_cols] = replacement.to_numpy()
        Xmirror = (
            mirror_rows[feat_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        del canonical_id_by_base, current_id, alt_rows, mirror_rows
    elif name == "Theta":
        # Avoid storing a second 313k x 841 matrix.  The sign flip is applied
        # in-place to the mirrored rows of each temporary Xrun below.
        Xmirror = None
    elif name == "Alexander":
        Xmirror = None
    else:  # defensive
        raise ValueError(name)

    view_audit_rows.append({
        "view": name,
        "input_dim": int(Xcanon.shape[1]),
        "n_frozen_knots": N,
        "distinct_archived_opposite_missing": int(opposite_missing),
        "mirror_action": (
            "archived_opposite_row" if name in ("Jones", "HOMFLY-PT")
            else "coefficient_sign_flip_stage31A_gate" if name == "Theta"
            else "unchanged"
        ),
    })

    # The raw source DataFrames are no longer needed.  Free them before the PCA
    # fits; this is important for the 841-dimensional Theta view in Colab.
    del raw, cleaned, canonical, canon_rows
    gc.collect()

    # Optional implementation audit: recompute canonical tail and require very
    # high agreement with the frozen canonical hard mask before mirror results
    # from this implementation are accepted.
    canonical_ckpt = CHECKPOINTS / f"{safe_name(name)}__baseline_recomputed.npz"
    if canonical_ckpt.exists():
        with np.load(canonical_ckpt, allow_pickle=False) as p:
            recomputed_canonical = np.asarray(p["hard_mask"], dtype=bool)
    else:
        score0, bins0 = full_atlas_conditional(
            Xcanon,
            k=FIXED_K[name],
            seed=PCA_SEED + view_index,
        )
        recomputed_canonical = stable_top_tail(score0)
        np.savez_compressed(
            canonical_ckpt,
            hard_mask=recomputed_canonical,
            norm_bin=bins0,
        )
        del score0, bins0
        gc.collect()

    jac = jaccard(recomputed_canonical, canonical_masks[name])
    reproduction_rows.append({
        "view": name,
        "frozen_tail_n": int(canonical_masks[name].sum()),
        "recomputed_tail_n": int(recomputed_canonical.sum()),
        "jaccard": jac,
        "overlap": int(np.sum(recomputed_canonical & canonical_masks[name])),
    })
    print(f"Canonical-tail reproduction Jaccard for {name}: {jac:.6f}")
    if jac < 0.98:
        raise RuntimeError(
            f"{name}: canonical full-atlas implementation reproduces frozen tail only at "
            f"Jaccard={jac:.4f}. Stop and reconcile the PCA/conditional implementation "
            "before interpreting mirror results."
        )

    # Mirror assignments. Alexander is invariant, so it reuses the frozen mask
    # exactly and avoids unnecessary PCA refits.
    for label, mirror_mask in assignment_masks.items():
        if label == "baseline_canonical":
            continue
        if name == "Alexander":
            run_view_masks[label][name] = canonical_masks[name].copy()
            continue

        ckpt = CHECKPOINTS / f"{safe_name(name)}__{label}.npz"
        if ckpt.exists():
            print(f"[{name} / {label}] loading checkpoint")
            with np.load(ckpt, allow_pickle=False) as p:
                hard = np.asarray(p["hard_mask"], dtype=bool)
        else:
            print(f"[{name} / {label}] fitting mirror-assignment PCA")
            Xrun = Xcanon.copy()
            if name == "Theta":
                Xrun[mirror_mask] *= -1.0
            else:
                Xrun[mirror_mask] = Xmirror[mirror_mask]
            score, bins = full_atlas_conditional(
                Xrun,
                k=FIXED_K[name],
                seed=PCA_SEED + view_index,
            )
            hard = stable_top_tail(score)
            np.savez_compressed(
                ckpt,
                hard_mask=hard,
                norm_bin=bins,
                mirror_mask=mirror_mask,
            )
            del Xrun, score, bins
            gc.collect()
        if int(hard.sum()) != TAIL_N:
            raise RuntimeError(f"{name}/{label}: tail n={int(hard.sum())}, expected {TAIL_N}")
        run_view_masks[label][name] = hard

    del Xcanon, Xmirror
    gc.collect()

pd.DataFrame(view_audit_rows).to_csv(OUT / "view_mirror_action_audit.csv", index=False)
pd.DataFrame(reproduction_rows).to_csv(OUT / "canonical_tail_reproduction_audit.csv", index=False)

# ---------------------------------------------------------------------------
# 7. Consensus and fixed-Khovanov phenotype summaries
# ---------------------------------------------------------------------------
summary_rows = []
selected_rows = []
for label, mirror_mask in assignment_masks.items():
    masks = run_view_masks[label]
    missing_views = [v for v in NO_KHOVANOV if v not in masks]
    if missing_views:
        raise RuntimeError(f"{label}: missing selection masks {missing_views}")
    selected = consensus_from_masks(masks)
    overlap = int(np.sum(selected & canonical_consensus))
    row = {
        "assignment": label,
        "mirror_fraction": float(np.mean(mirror_mask)),
        **endpoint_summary(selected),
        "overlap_with_canonical": overlap,
        "jaccard_with_canonical": jaccard(selected, canonical_consensus),
        "canonical_recovered_prop": overlap / int(canonical_consensus.sum()),
        "delta_mean_diagonal_vs_canonical": (
            float(np.mean(KH_DIAG[selected])) - float(canonical_endpoint["kh_diagonal_mean"])
        ),
    }
    summary_rows.append(row)
    for knot_id in FROZEN_IDS[selected]:
        selected_rows.append({"assignment": label, ID_COL: knot_id})

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "four_view_mirror_withheld_khovanov_summary.csv", index=False)
pd.DataFrame(selected_rows).to_csv(OUT / "four_view_mirror_selected_ids.csv", index=False)

random_summary = summary.loc[summary["assignment"].str.startswith("random_seed_")].copy()
if len(random_summary):
    aggregate = pd.DataFrame([{
        "n_random_assignments": int(len(random_summary)),
        "selected_n_min": int(random_summary["selected_n"].min()),
        "selected_n_max": int(random_summary["selected_n"].max()),
        "jaccard_min": float(random_summary["jaccard_with_canonical"].min()),
        "jaccard_max": float(random_summary["jaccard_with_canonical"].max()),
        "recovery_min": float(random_summary["canonical_recovered_prop"].min()),
        "recovery_max": float(random_summary["canonical_recovered_prop"].max()),
        "kh_diagonal_mean_min": float(random_summary["kh_diagonal_mean"].min()),
        "kh_diagonal_mean_max": float(random_summary["kh_diagonal_mean"].max()),
        "kh_ge3_min": float(random_summary["kh_diagonal_ge_3_prop"].min()),
        "kh_ge3_max": float(random_summary["kh_diagonal_ge_3_prop"].max()),
        "kh_ge4_min": float(random_summary["kh_diagonal_ge_4_prop"].min()),
        "kh_ge4_max": float(random_summary["kh_diagonal_ge_4_prop"].max()),
    }])
    aggregate.to_csv(OUT / "random_mirror_population_stability_summary.csv", index=False)
else:
    aggregate = pd.DataFrame()

print("\n" + "=" * 78)
print("STAGE 31B COMPLETE")
print("=" * 78)
print("\nCanonical-tail reproduction audit:")
print(pd.DataFrame(reproduction_rows).to_string(index=False))
print("\nMirror assignment summaries (Khovanov fixed/withheld):")
print(summary.to_string(index=False))
if len(aggregate):
    print("\nRandom-assignment range summary:")
    print(aggregate.to_string(index=False))
print("\nInterpretation rule: membership may change; ask whether the FIXED stored-Kh population phenotype remains stable.")
print("No mirror-robustness p-value is claimed by this stage.")
print("\nSaved Stage 31B to:", OUT)
