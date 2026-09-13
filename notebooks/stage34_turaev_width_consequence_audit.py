# %% [markdown]
# Stage 34 — W_Kh >= 4 consequence audit and knot lists
#
# Revision question:
#   Give the stored-width threshold a concrete topological meaning and verify the
#   exact knots behind the reported counts.
#
# Mathematical implication to be USED ONLY AFTER the stored grading convention
# is confirmed:
#
#       w_Kh(K) - 2 <= g_T(K) <= d_alt(K).
#
# The paper's operational W_Kh counts occupied diagonals in the stored F_-part.
# If those F_ coordinates are the free/rational Khovanov ranks in the stated
# grading, then W_Kh <= w_Kh. Hence
#
#       W_Kh >= 4  =>  g_T >= 2 and d_alt >= 2,
#
# so the knot cannot be almost alternating.
#
# This stage does NOT claim enrichment of actual Turaev genus. It only verifies
# which selected knots satisfy the sufficient stored-width threshold.
#
# Fresh-session usage:
#   %run /content/drive/MyDrive/consensus_hardness_refactored/notebooks/\
#       stage34_turaev_width_consequence_audit.py

# %%
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

DEFAULT_DATA_DIR = Path("/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants")
DEFAULT_ROOT = DEFAULT_DATA_DIR / "processed_consensus_hardness" / "corrected_run_20260819"
DATA_DIR = Path(os.environ.get("KNOT_DATA_DIR", str(DEFAULT_DATA_DIR)))
ROOT = Path(os.environ.get("KNOT_OUTPUT_DIR", str(DEFAULT_ROOT)))
OUT = ROOT / "34_turaev_width_consequence_audit"


def maybe_mount_drive() -> None:
    if Path("/content").exists() and not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive  # type: ignore
            print("Google Drive is not mounted; requesting mount...")
            drive.mount("/content/drive")
        except Exception as exc:
            print("WARNING: could not mount Drive automatically:", exc)


maybe_mount_drive()
if not ROOT.exists():
    raise FileNotFoundError(f"Frozen run not found: {ROOT}")
OUT.mkdir(parents=True, exist_ok=True)

ID_COL = "knot_id_base"
NO_KHOVANOV = ("Alexander", "Jones", "HOMFLY-PT", "Theta")
N_SELECT = 31
EXPECTED_COUNTS = {
    "canonical_noKh_n220": 48,
    "heldout_conditional_n31": 9,
    "heldout_relative_n31": 5,
}


def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "_")


def find_one(filename: str) -> Path:
    hits = sorted(ROOT.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename} below {ROOT}")
    return hits[0]


# ---------------------------------------------------------------------------
# 1. Frozen atlas / phenotype
# ---------------------------------------------------------------------------
atlas = pd.read_parquet(find_one("final_hard_regime_atlas.parquet")).reset_index(drop=True)
phenotype = pd.read_parquet(find_one("complete_mathematical_phenotype_atlas.parquet")).reset_index(drop=True)
atlas[ID_COL] = atlas[ID_COL].astype(str)
phenotype[ID_COL] = phenotype[ID_COL].astype(str)
if not atlas[ID_COL].equals(phenotype[ID_COL]):
    phenotype = atlas[[ID_COL]].merge(phenotype, on=ID_COL, how="left", validate="one_to_one")

N = len(atlas)
ids = atlas[ID_COL].astype(str).to_numpy()
kh_diag_col = next(c for c in (
    "khovanov_q_minus_2t_diagonal_count", "kh_diagonal_count", "khovanov_diagonal_count"
) if c in phenotype.columns)
kh_support_col = next(c for c in (
    "khovanov_support_size", "kh_support_size", "khovanov_f_support_size"
) if c in phenotype.columns)
kh_diag = phenotype[kh_diag_col].to_numpy(float)
kh_support = phenotype[kh_support_col].to_numpy(float)
if not np.isfinite(kh_diag).all():
    raise RuntimeError("Non-finite stored Khovanov diagonal count")

# ---------------------------------------------------------------------------
# 2. Canonical full-atlas no-Khovanov 3-of-4 selection
# ---------------------------------------------------------------------------
hard_saved = pd.read_csv(find_one("conditional_100bins_hard_sets.csv"), dtype={ID_COL: str})
id_to_pos = pd.Series(np.arange(N, dtype=np.int64), index=ids).to_dict()
votes = np.zeros(N, dtype=np.uint8)
for name in NO_KHOVANOV:
    view_ids = hard_saved.loc[hard_saved["invariant"].eq(name), ID_COL].astype(str)
    m = np.zeros(N, dtype=bool)
    m[[id_to_pos[x] for x in view_ids]] = True
    votes += m.astype(np.uint8)
canonical = votes >= 3
if int(canonical.sum()) != 220:
    raise RuntimeError(f"Canonical no-Kh selection n={int(canonical.sum())}, expected 220")

# ---------------------------------------------------------------------------
# 3. Frozen held-out n=31 conditional and relative selections
#    Methods correction: per-view TEST empirical percentiles BEFORE aggregation.
# ---------------------------------------------------------------------------
with np.load(find_one("heldout_ae_seed_0.npz"), allow_pickle=False) as p:
    test_idx = np.asarray(p["test_idx"], dtype=np.int64)
test_ids = ids[test_idx]
N_TEST = len(test_idx)

per_method: dict[str, dict[str, np.ndarray]] = {
    "conditional": {},
    "relative": {},
}
for name in NO_KHOVANOV:
    score_path = ROOT / "23_anomaly_score_baselines" / "checkpoints" / f"{safe_name(name)}_test_scores.npz"
    if not score_path.exists():
        candidates = sorted(ROOT.rglob(f"{safe_name(name)}_test_scores.npz"))
        if not candidates:
            raise FileNotFoundError(score_path)
        score_path = candidates[0]
    with np.load(score_path, allow_pickle=False) as p:
        cond = np.asarray(p["test_conditional_percentile_100"], dtype=float)
        rel = np.asarray(p["test_relative_nre"], dtype=float)
    if len(cond) != N_TEST or len(rel) != N_TEST:
        raise RuntimeError(f"{name}: Stage-23 score length mismatch")
    per_method["conditional"][name] = rankdata(cond, method="average") / N_TEST
    per_method["relative"][name] = rankdata(rel, method="average") / N_TEST


def heldout_mask(method: str) -> np.ndarray:
    mat = np.column_stack([per_method[method][v] for v in NO_KHOVANOV])
    c3 = np.sort(mat, axis=1)[:, 1]  # third-largest of four
    order = np.lexsort((test_ids, c3))
    m = np.zeros(N_TEST, dtype=bool)
    m[order[-N_SELECT:]] = True
    return m


held_cond = heldout_mask("conditional")
held_rel = heldout_mask("relative")

# ---------------------------------------------------------------------------
# 4. Verify counts and write exact knot lists
# ---------------------------------------------------------------------------
sets = {
    "canonical_noKh_n220": (ids, canonical, kh_diag, kh_support),
    "heldout_conditional_n31": (test_ids, held_cond, kh_diag[test_idx], kh_support[test_idx]),
    "heldout_relative_n31": (test_ids, held_rel, kh_diag[test_idx], kh_support[test_idx]),
}

summary_rows = []
all_threshold_rows = []
for label, (set_ids, mask, diag, support) in sets.items():
    sel_diag = diag[mask]
    threshold = mask & (diag >= 4)
    n_threshold = int(np.sum(threshold))
    expected = EXPECTED_COUNTS[label]
    summary_rows.append({
        "selection": label,
        "selected_n": int(mask.sum()),
        "mean_stored_F_diagonal_count": float(np.mean(sel_diag)),
        "n_WKh_ge_4": n_threshold,
        "prop_WKh_ge_4": float(np.mean(sel_diag >= 4)),
        "expected_n_WKh_ge_4_from_revision_letter": expected,
        "count_matches_expected": bool(n_threshold == expected),
    })
    for knot_id, w, sup in zip(set_ids[threshold], diag[threshold], support[threshold]):
        all_threshold_rows.append({
            "selection": label,
            ID_COL: str(knot_id),
            "stored_F_diagonal_count_WKh": float(w),
            "stored_F_support_size": float(sup),
            "certified_Turaev_genus_lower_bound_if_convention_verified": 2,
            "certified_dealternating_number_lower_bound_if_convention_verified": 2,
            "almost_alternating_excluded_if_convention_verified": True,
        })

summary = pd.DataFrame(summary_rows)
threshold_knots = pd.DataFrame(all_threshold_rows)
summary.to_csv(OUT / "WKh_ge4_selected_count_audit.csv", index=False)
threshold_knots.to_csv(OUT / "WKh_ge4_selected_knot_lists.csv", index=False)

if not summary["count_matches_expected"].all():
    print("\nCOUNT MISMATCH — do not copy the expected counts into the manuscript.")
    print(summary.to_string(index=False))
    raise RuntimeError(
        "At least one WKh>=4 count does not match the revision-letter count. "
        "Reconcile the held-out aggregation/tie convention before manuscript edits."
    )

# ---------------------------------------------------------------------------
# 5. Record the logical scope of the topological consequence
# ---------------------------------------------------------------------------
logic = {
    "established_inequality_to_cite": "w_Kh(K) - 2 <= g_T(K) <= d_alt(K)",
    "operational_quantity": "number of occupied q-2t diagonals in stored F_ coordinates",
    "required_encoding_fact": (
        "the stored F_ coordinates represent free/rational Khovanov ranks in the stated grading, "
        "so the operational occupied-diagonal count is no larger than full homological width"
    ),
    "consequence_after_encoding_fact_is_verified": (
        "W_Kh >= 4 implies g_T >= 2 and d_alt >= 2; therefore the knot is not almost alternating"
    ),
    "not_claimed": (
        "This threshold does not by itself prove enrichment in actual Turaev genus or dealternating number, "
        "because knots below the threshold can also have g_T >= 2."
    ),
}
(OUT / "turaev_consequence_logic.json").write_text(json.dumps(logic, indent=2))

print("\n" + "="*78)
print("STAGE 34 COMPLETE")
print("="*78)
print("\nVerified WKh >= 4 counts:")
print(summary.to_string(index=False))
print("\nFirst threshold knots:")
print(threshold_knots.head(30).to_string(index=False))
print("\nAll three counts reproduce 48/220, 9/31, and 5/31.")
print("Topological implication remains explicitly conditional on verification of the stored F_ grading convention.")
print("Saved Stage 34 to:", OUT)
