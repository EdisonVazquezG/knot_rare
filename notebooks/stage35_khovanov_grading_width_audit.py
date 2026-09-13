# %% [markdown]
# Stage 35 — Khovanov F_/torsion grading and stored-width convention audit
#
# Purpose
# -------
# Stage 34 verified the counts of selected knots with operational W_Kh >= 4,
# where W_Kh is the number of occupied q-2t diagonals in the stored F_ part.
#
# This stage checks the archived Khovanov encoding itself:
#
#   1. identify F_, T2_, and T4_ coordinate families;
#   2. parse their two grading indices from the coordinate names;
#   3. infer which stored index order reproduces the frozen q-2t diagonal count;
#   4. recompute F_-only diagonal count for all 313,230 knots;
#   5. compute the union width from F_ + T2_ + T4_;
#   6. verify W_F <= W_full_stored row-by-row;
#   7. quantify the known F-only vs full-stored differences;
#   8. audit all knots used by Stage 34 at the W_F >= 4 threshold.
#
# Scope
# -----
# This is a COMPUTATIONAL ENCODING audit.  It verifies what the archived
# coordinates numerically encode and how the operational width is calculated.
# The manuscript should still cite the mathematical/source documentation that
# identifies F_ as the free/rational Khovanov ranks and T2_/T4_ as torsion
# coordinate families.
#
# Fresh-session usage
# -------------------
# %run "/content/drive/MyDrive/consensus_hardness_refactored/notebooks/stage35_khovanov_grading_width_audit.py"

from __future__ import annotations

import gc
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PROJECT_DIR = Path("/content/drive/MyDrive/consensus_hardness_refactored")
DEFAULT_DATA_DIR = Path("/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants")
DEFAULT_ROOT = DEFAULT_DATA_DIR / "processed_consensus_hardness" / "corrected_run_20260819"

PROJECT_DIR = Path(os.environ.get("KNOT_PROJECT_DIR", str(DEFAULT_PROJECT_DIR)))
DATA_DIR = Path(os.environ.get("KNOT_DATA_DIR", str(DEFAULT_DATA_DIR)))
ROOT = Path(os.environ.get("KNOT_OUTPUT_DIR", str(DEFAULT_ROOT)))
OUT = ROOT / "35_khovanov_grading_width_audit"
ID_COL = "knot_id_base"
BATCH = int(os.environ.get("STAGE35_BATCH_ROWS", "4096"))


def maybe_mount_drive():
    if Path("/content").exists() and not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive  # type: ignore
            print("Google Drive is not mounted; requesting mount...")
            drive.mount("/content/drive")
        except Exception as exc:
            print("WARNING: could not mount Drive automatically:", exc)


maybe_mount_drive()
if not ROOT.exists() or not DATA_DIR.exists():
    raise FileNotFoundError("Mount Drive or set KNOT_DATA_DIR/KNOT_OUTPUT_DIR.")
OUT.mkdir(parents=True, exist_ok=True)

if (PROJECT_DIR / "src").exists() and str(PROJECT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "src"))

try:
    from consensus_hardness.io import load_table
    from consensus_hardness.preprocessing import (
        add_knot_ids,
        filter_crossings,
        canonicalize_mirrors_by_signature,
    )
except Exception as exc:
    raise ImportError("Could not import consensus_hardness helpers") from exc


def find_one(filename: str) -> Path:
    hits = sorted(ROOT.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename} below {ROOT}")
    return hits[0]


# ---------------------------------------------------------------------------
# 1. Frozen atlas and frozen operational endpoint
# ---------------------------------------------------------------------------
atlas = pd.read_parquet(find_one("final_hard_regime_atlas.parquet")).reset_index(drop=True)
phenotype = pd.read_parquet(find_one("complete_mathematical_phenotype_atlas.parquet")).reset_index(drop=True)
atlas[ID_COL] = atlas[ID_COL].astype(str)
phenotype[ID_COL] = phenotype[ID_COL].astype(str)
if not atlas[ID_COL].equals(phenotype[ID_COL]):
    phenotype = atlas[[ID_COL]].merge(phenotype, on=ID_COL, how="left", validate="one_to_one")

ids = atlas[ID_COL].astype(str).to_numpy()
N = len(ids)
frozen_diag_col = next(c for c in (
    "khovanov_q_minus_2t_diagonal_count",
    "kh_diagonal_count",
    "khovanov_diagonal_count",
) if c in phenotype.columns)
frozen_diag = phenotype[frozen_diag_col].to_numpy(float)

# ---------------------------------------------------------------------------
# 2. Load and align archived Khovanov table
# ---------------------------------------------------------------------------
kh_path = DATA_DIR / "even_KH_upto17.pkl"
if not kh_path.exists():
    raise FileNotFoundError(kh_path)

print("Loading archived Khovanov table:", kh_path)
raw = load_table(kh_path)
clean = add_knot_ids(raw, id_col="knot_id", mirror_symbol="!")
clean = filter_crossings(clean, min_crossings=3, max_crossings=15)
canonical = canonicalize_mirrors_by_signature(
    clean,
    id_col="knot_id",
    base_col=ID_COL,
    signature_col="signature",
    mirror_symbol="!",
)
idx = canonical.set_index(ID_COL, drop=False)
missing = [x for x in ids if x not in idx.index]
if missing:
    raise RuntimeError(f"Archived Khovanov table misses frozen IDs, e.g. {missing[:5]}")
kh = idx.loc[ids].reset_index(drop=True)
if not np.array_equal(kh[ID_COL].astype(str).to_numpy(), ids):
    raise RuntimeError("Khovanov alignment failure")

F_COLS = [c for c in kh.columns if str(c).startswith("F_")]
T2_COLS = [c for c in kh.columns if str(c).startswith("T2_")]
T4_COLS = [c for c in kh.columns if str(c).startswith("T4_")]
print(f"Coordinate counts: F={len(F_COLS)}, T2={len(T2_COLS)}, T4={len(T4_COLS)}")
if not F_COLS:
    raise RuntimeError("No F_ coordinates found")

pd.DataFrame({
    "family": (["F"]*len(F_COLS) + ["T2"]*len(T2_COLS) + ["T4"]*len(T4_COLS)),
    "coordinate": list(map(str, F_COLS + T2_COLS + T4_COLS)),
}).to_csv(OUT / "khovanov_coordinate_inventory.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Parse the two grading indices from coordinate names
# ---------------------------------------------------------------------------
def parse_two_indices(col: str, prefix: str):
    s = str(col)
    tail = s[len(prefix):] if s.startswith(prefix) else s

    # Prefer explicit q/t labels when available.
    mq = re.search(r"(?:^|_)q(?:=)?(-?\d+)", tail, flags=re.I)
    mt = re.search(r"(?:^|_)t(?:=)?(-?\d+)", tail, flags=re.I)
    if mq and mt:
        return int(mq.group(1)), int(mt.group(1)), "explicit_q_t"

    nums = re.findall(r"-?\d+", tail)
    if len(nums) < 2:
        raise ValueError(f"Could not parse two grading indices from coordinate {col!r}")
    return int(nums[-2]), int(nums[-1]), "last_two_integers"


def parsed_family(cols, prefix):
    rows = []
    for c in cols:
        a, b, method = parse_two_indices(str(c), prefix)
        rows.append({"coordinate": str(c), "index1": a, "index2": b, "parse_method": method})
    return pd.DataFrame(rows)


parsed_F = parsed_family(F_COLS, "F_")
parsed_T2 = parsed_family(T2_COLS, "T2_") if T2_COLS else pd.DataFrame()
parsed_T4 = parsed_family(T4_COLS, "T4_") if T4_COLS else pd.DataFrame()
parsed_F.assign(family="F").to_csv(OUT / "parsed_F_grading_coordinates.csv", index=False)
if len(parsed_T2):
    parsed_T2.assign(family="T2").to_csv(OUT / "parsed_T2_grading_coordinates.csv", index=False)
if len(parsed_T4):
    parsed_T4.assign(family="T4").to_csv(OUT / "parsed_T4_grading_coordinates.csv", index=False)

# Candidate index order: either index1=q,index2=t or swapped.
candidate_diagonals = {
    "index1_minus_2_index2": parsed_F["index1"].to_numpy(int) - 2*parsed_F["index2"].to_numpy(int),
    "index2_minus_2_index1": parsed_F["index2"].to_numpy(int) - 2*parsed_F["index1"].to_numpy(int),
}


def occupied_widths(frame: pd.DataFrame, cols: list[str], diag_values: np.ndarray):
    out = np.empty(len(frame), dtype=np.int16)
    for start in range(0, len(frame), BATCH):
        stop = min(start + BATCH, len(frame))
        arr = frame.iloc[start:stop][cols].to_numpy()
        occ = arr != 0
        for r in range(stop - start):
            out[start + r] = len(np.unique(diag_values[occ[r]]))
        if stop % (BATCH*20) == 0 or stop == len(frame):
            print(f"Width audit: {stop:,}/{len(frame):,}")
    return out


candidate_rows = []
candidate_counts = {}
for label, diag_values in candidate_diagonals.items():
    print("Testing grading candidate:", label)
    counts = occupied_widths(kh, F_COLS, diag_values)
    candidate_counts[label] = counts
    exact = int(np.sum(counts == frozen_diag))
    mae = float(np.mean(np.abs(counts.astype(float) - frozen_diag)))
    candidate_rows.append({
        "grading_candidate": label,
        "exact_agreement_n": exact,
        "exact_agreement_prop": exact / N,
        "mean_abs_difference_vs_frozen_endpoint": mae,
        "max_abs_difference_vs_frozen_endpoint": float(np.max(np.abs(counts - frozen_diag))),
    })

candidate_summary = pd.DataFrame(candidate_rows).sort_values(
    ["exact_agreement_prop", "mean_abs_difference_vs_frozen_endpoint"],
    ascending=[False, True],
).reset_index(drop=True)
candidate_summary.to_csv(OUT / "grading_candidate_reproduction.csv", index=False)
best_label = str(candidate_summary.iloc[0]["grading_candidate"])
W_F = candidate_counts[best_label]

if candidate_summary.iloc[0]["exact_agreement_prop"] < 0.999:
    raise RuntimeError(
        "Neither parsed grading order reproduces the frozen F_-width endpoint closely enough. "
        "Inspect grading_candidate_reproduction.csv before using Stage 34 topological language."
    )

# Map chosen candidate to q/t index roles.
if best_label == "index1_minus_2_index2":
    q_is, t_is = "index1", "index2"
else:
    q_is, t_is = "index2", "index1"


def diag_for_parsed(parsed: pd.DataFrame):
    if parsed.empty:
        return np.empty(0, dtype=int)
    if best_label == "index1_minus_2_index2":
        return parsed["index1"].to_numpy(int) - 2*parsed["index2"].to_numpy(int)
    return parsed["index2"].to_numpy(int) - 2*parsed["index1"].to_numpy(int)


# ---------------------------------------------------------------------------
# 4. Full stored support: F + T2 + T4
# ---------------------------------------------------------------------------
all_cols = F_COLS + T2_COLS + T4_COLS
all_diag = np.concatenate([
    diag_for_parsed(parsed_F),
    diag_for_parsed(parsed_T2),
    diag_for_parsed(parsed_T4),
])

if len(all_cols) == len(F_COLS):
    W_FULL = W_F.copy()
else:
    print("Computing full stored F+T2+T4 occupied-diagonal count...")
    W_FULL = occupied_widths(kh, all_cols, all_diag)

diff = W_FULL.astype(int) - W_F.astype(int)
if np.any(diff < 0):
    raise RuntimeError("Found W_full_stored < W_F, which should be impossible for a coordinate union.")

summary = {
    "n_knots": int(N),
    "n_F_coordinates": int(len(F_COLS)),
    "n_T2_coordinates": int(len(T2_COLS)),
    "n_T4_coordinates": int(len(T4_COLS)),
    "chosen_grading_formula": best_label,
    "interpreted_q_index": q_is,
    "interpreted_t_index": t_is,
    "F_width_matches_frozen_endpoint_n": int(np.sum(W_F == frozen_diag)),
    "F_width_matches_frozen_endpoint_prop": float(np.mean(W_F == frozen_diag)),
    "F_vs_full_exact_agreement_n": int(np.sum(W_F == W_FULL)),
    "F_vs_full_exact_agreement_prop": float(np.mean(W_F == W_FULL)),
    "F_vs_full_max_difference": int(np.max(diff)),
    "F_vs_full_mean_difference": float(np.mean(diff)),
    "n_full_width_larger_than_F": int(np.sum(diff > 0)),
    "W_F_le_W_full_for_all": bool(np.all(W_F <= W_FULL)),
}
(OUT / "khovanov_width_encoding_summary.json").write_text(json.dumps(summary, indent=2))
pd.DataFrame([summary]).to_csv(OUT / "khovanov_width_encoding_summary.csv", index=False)

disagree_idx = np.flatnonzero(diff != 0)
disagreements = pd.DataFrame({
    ID_COL: ids[disagree_idx],
    "W_F": W_F[disagree_idx],
    "W_full_stored": W_FULL[disagree_idx],
    "full_minus_F": diff[disagree_idx],
})
disagreements.to_csv(OUT / "F_vs_full_width_disagreements.csv", index=False)

# ---------------------------------------------------------------------------
# 5. Audit the exact Stage-34 threshold knots
# ---------------------------------------------------------------------------
stage34_path = ROOT / "34_turaev_width_consequence_audit" / "WKh_ge4_selected_knot_lists.csv"
if not stage34_path.exists():
    raise FileNotFoundError(
        f"{stage34_path}\nRun Stage 34 before Stage 35."
    )
selected34 = pd.read_csv(stage34_path, dtype={ID_COL: str})
lookup = pd.DataFrame({
    ID_COL: ids,
    "W_F_recomputed": W_F.astype(int),
    "W_full_stored": W_FULL.astype(int),
}).set_index(ID_COL)
audit34 = selected34.merge(
    lookup,
    left_on=ID_COL,
    right_index=True,
    how="left",
    validate="many_to_one",
)
audit34["F_threshold_ge4_recomputed"] = audit34["W_F_recomputed"] >= 4
audit34["full_stored_threshold_ge4"] = audit34["W_full_stored"] >= 4
audit34["F_threshold_preserved_in_full_stored"] = (
    audit34["F_threshold_ge4_recomputed"] & audit34["full_stored_threshold_ge4"]
)
audit34.to_csv(OUT / "stage34_threshold_knot_encoding_audit.csv", index=False)

if audit34["W_F_recomputed"].isna().any():
    raise RuntimeError("Could not align all Stage-34 threshold knots.")
if not audit34["F_threshold_ge4_recomputed"].all():
    raise RuntimeError("At least one Stage-34 threshold knot failed W_F>=4 on recomputation.")
if not audit34["F_threshold_preserved_in_full_stored"].all():
    raise RuntimeError("At least one W_F>=4 knot did not remain >=4 in full stored support.")

# ---------------------------------------------------------------------------
# 6. Basic coordinate-value audit
# ---------------------------------------------------------------------------
value_rows = []
for family, cols in (("F", F_COLS), ("T2", T2_COLS), ("T4", T4_COLS)):
    if not cols:
        continue
    min_val = np.inf
    max_val = -np.inf
    noninteger = 0
    negative = 0
    nonzero = 0
    for start in range(0, len(kh), BATCH):
        stop = min(start+BATCH, len(kh))
        arr = kh.iloc[start:stop][cols].to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            min_val = min(min_val, float(finite.min()))
            max_val = max(max_val, float(finite.max()))
            noninteger += int(np.sum(np.abs(finite - np.round(finite)) > 1e-10))
            negative += int(np.sum(finite < 0))
            nonzero += int(np.sum(finite != 0))
    value_rows.append({
        "family": family,
        "min_value": min_val,
        "max_value": max_val,
        "negative_entry_count": negative,
        "noninteger_entry_count": noninteger,
        "nonzero_entry_count": nonzero,
    })
value_audit = pd.DataFrame(value_rows)
value_audit.to_csv(OUT / "khovanov_coordinate_value_audit.csv", index=False)

gate = {
    "status": (
        "COMPUTATIONAL_ENCODING_VERIFIED"
        if (
            summary["F_width_matches_frozen_endpoint_prop"] >= 0.999
            and summary["W_F_le_W_full_for_all"]
            and bool(audit34["F_threshold_preserved_in_full_stored"].all())
        )
        else "FAILED"
    ),
    "what_is_verified": [
        "the parsed F_ grading convention reproduces the frozen operational q-2t endpoint",
        "adding stored T2_/T4_ coordinates never decreases occupied-diagonal count",
        "every Stage-34 W_F>=4 threshold knot remains >=4 in the full stored coordinate union",
    ],
    "what_still_requires_source_citation": (
        "The mathematical interpretation of F_ as free/rational Khovanov ranks "
        "and T2_/T4_ as torsion families should be tied to the dataset/source documentation."
    ),
}
(OUT / "stage35_gate.json").write_text(json.dumps(gate, indent=2))

print("\n" + "="*88)
print("STAGE 35 COMPLETE")
print("="*88)
print("\nGrading candidate reproduction:")
print(candidate_summary.to_string(index=False))
print("\nEncoding summary:")
print(pd.DataFrame([summary]).to_string(index=False))
print("\nCoordinate-value audit:")
print(value_audit.to_string(index=False))
print(f"\nF/full disagreements saved: {len(disagreements)}")
if len(disagreements):
    print(disagreements.head(30).to_string(index=False))
print("\nStage-34 threshold audit:")
print(
    audit34.groupby("selection", as_index=False)
    .agg(
        n=(ID_COL, "size"),
        min_W_F=("W_F_recomputed", "min"),
        min_W_full=("W_full_stored", "min"),
        all_preserved=("F_threshold_preserved_in_full_stored", "all"),
    ).to_string(index=False)
)
print("\nGate:", gate["status"])
print(gate["what_still_requires_source_citation"])
print("Saved Stage 35 to:", OUT)

del raw, clean, canonical, idx, kh
gc.collect()
