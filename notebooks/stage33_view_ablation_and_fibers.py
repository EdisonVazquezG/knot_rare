# %% [markdown]
# Stage 33 — Held-out view ablation and (HOMFLY-PT, Theta) fiber audit
#
# Revision questions:
#   1. Do the derived/redundant views help beyond HOMFLY-PT and Theta themselves?
#   2. Are fixed-size comparisons aggregating comparable scales correctly?
#   3. What happens on fibers of identical (HOMFLY-PT, Theta) input tuples?
#
# Design:
#   * Use the frozen Stage-23 held-out TEST population.
#   * Convert every per-view TEST score to an empirical TEST percentile before
#     aggregation, as requested in the methods correction.
#   * Compare, at the same final n=31:
#       - HOMFLY-PT alone
#       - Theta alone
#       - HOMFLY-PT + Theta (2-of-2, aggregate = min percentile)
#       - Alexander + Jones + HOMFLY-PT + Theta (3-of-4 aggregate)
#   * Khovanov is excluded from every selection and used only as an outcome.
#   * Inspect exact held-out HOMFLY-PT and Theta coefficient fibers and report
#     boundary splits caused only by deterministic ID tie-breaking.
#   * Search for checked examples sharing HOMFLY-PT but differing in Theta and
#     stored Khovanov diagonal count.
#
# This stage is DESCRIPTIVE. It does not claim a formal between-method test.
#
# Fresh-session usage:
#   %run /content/drive/MyDrive/consensus_hardness_refactored/notebooks/\
#       stage33_view_ablation_and_fibers.py

# %%
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# ---------------------------------------------------------------------------
# 0. Fresh-session setup
# ---------------------------------------------------------------------------
DEFAULT_PROJECT_DIR = Path("/content/drive/MyDrive/consensus_hardness_refactored")
DEFAULT_DATA_DIR = Path("/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants")
DEFAULT_ROOT = DEFAULT_DATA_DIR / "processed_consensus_hardness" / "corrected_run_20260819"
PROJECT_DIR = Path(os.environ.get("KNOT_PROJECT_DIR", str(DEFAULT_PROJECT_DIR)))
DATA_DIR = Path(os.environ.get("KNOT_DATA_DIR", str(DEFAULT_DATA_DIR)))
ROOT = Path(os.environ.get("KNOT_OUTPUT_DIR", str(DEFAULT_ROOT)))
OUT = ROOT / "33_view_ablation_and_fibers"


def maybe_mount_drive() -> None:
    if Path("/content").exists() and not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive  # type: ignore
            print("Google Drive is not mounted; requesting mount...")
            drive.mount("/content/drive")
        except Exception as exc:
            print("WARNING: could not mount Drive automatically:", exc)


maybe_mount_drive()
if not ROOT.exists() or not DATA_DIR.exists():
    raise FileNotFoundError("Mount Drive or set KNOT_DATA_DIR/KNOT_OUTPUT_DIR")
OUT.mkdir(parents=True, exist_ok=True)
if (PROJECT_DIR / "src").exists() and str(PROJECT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "src"))

try:
    from consensus_hardness.preprocessing import (
        add_knot_ids,
        canonicalize_mirrors_by_signature,
    )
except Exception as exc:
    raise ImportError(f"Could not import consensus_hardness from {PROJECT_DIR}") from exc

ID_COL = "knot_id_base"
NO_KHOVANOV = ("Alexander", "Jones", "HOMFLY-PT", "Theta")
N_SELECT = int(os.environ.get("STAGE33_N_SELECT", "31"))
BOOT_REPS = int(os.environ.get("STAGE33_BOOT_REPS", "5000"))
SEED = int(os.environ.get("STAGE33_SEED", "202609033"))
CHUNK_ROWS = int(os.environ.get("STAGE33_CSV_CHUNK_ROWS", "5000"))


def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "_")


def find_one(filename: str) -> Path:
    hits = sorted(ROOT.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename} below {ROOT}")
    return hits[0]


# ---------------------------------------------------------------------------
# 1. Frozen held-out population and external outcomes
# ---------------------------------------------------------------------------
atlas = pd.read_parquet(find_one("final_hard_regime_atlas.parquet")).reset_index(drop=True)
atlas[ID_COL] = atlas[ID_COL].astype(str)
with np.load(find_one("heldout_ae_seed_0.npz"), allow_pickle=False) as p:
    test_idx = np.asarray(p["test_idx"], dtype=np.int64)

test_meta = atlas.iloc[test_idx].reset_index(drop=True).copy()
test_ids = test_meta[ID_COL].astype(str).to_numpy()
N_TEST = len(test_ids)
if N_TEST != 46985:
    print(f"WARNING: test n={N_TEST:,}; historical Stage 23 used 46,985.")

phenotype = pd.read_parquet(find_one("complete_mathematical_phenotype_atlas.parquet"))
phenotype[ID_COL] = phenotype[ID_COL].astype(str)
phenotype = phenotype.set_index(ID_COL).reindex(test_ids)
KH_DIAG_COL = next(c for c in (
    "khovanov_q_minus_2t_diagonal_count", "kh_diagonal_count", "khovanov_diagonal_count"
) if c in phenotype.columns)
KH_SUPPORT_COL = next(c for c in (
    "khovanov_support_size", "kh_support_size", "khovanov_f_support_size"
) if c in phenotype.columns)
kh_diag = phenotype[KH_DIAG_COL].to_numpy(float)
kh_support = phenotype[KH_SUPPORT_COL].to_numpy(float)
if not np.isfinite(kh_diag).all() or not np.isfinite(kh_support).all():
    raise RuntimeError("Missing/non-finite held-out Khovanov outcomes")

# ---------------------------------------------------------------------------
# 2. Load Stage-23 conditional scores and convert each view to TEST percentiles
# ---------------------------------------------------------------------------
score_percentiles: dict[str, np.ndarray] = {}
for name in NO_KHOVANOV:
    score_path = ROOT / "23_anomaly_score_baselines" / "checkpoints" / f"{safe_name(name)}_test_scores.npz"
    if not score_path.exists():
        # Fall back to recursive search for old output layouts.
        candidates = sorted(ROOT.rglob(f"{safe_name(name)}_test_scores.npz"))
        if not candidates:
            raise FileNotFoundError(score_path)
        score_path = candidates[0]
    with np.load(score_path, allow_pickle=False) as p:
        if "test_conditional_percentile_100" not in p.files:
            raise KeyError(f"{score_path} missing test_conditional_percentile_100")
        score = np.asarray(p["test_conditional_percentile_100"], dtype=float)
    if len(score) != N_TEST or not np.isfinite(score).all():
        raise RuntimeError(f"{name}: invalid Stage-23 test score")
    # Explicit methods correction: put every view on an empirical TEST percentile
    # scale before aggregating across views.
    score_percentiles[name] = rankdata(score, method="average") / N_TEST


def top_n_from_aggregate(aggregate: np.ndarray, n_select: int = N_SELECT) -> np.ndarray:
    aggregate = np.asarray(aggregate, dtype=float)
    order = np.lexsort((test_ids, aggregate))
    mask = np.zeros(N_TEST, dtype=bool)
    mask[order[-n_select:]] = True
    return mask


# Same final cardinality for every ablation.
aggregates = {
    "HOMFLY_PT_alone": score_percentiles["HOMFLY-PT"],
    "Theta_alone": score_percentiles["Theta"],
    "HOMFLY_PT_plus_Theta_2of2": np.minimum(
        score_percentiles["HOMFLY-PT"], score_percentiles["Theta"]
    ),
}
four_matrix = np.column_stack([score_percentiles[v] for v in NO_KHOVANOV])
# Third-largest of four = second-smallest after ascending sort.
aggregates["four_view_3of4"] = np.sort(four_matrix, axis=1)[:, 1]
selections = {name: top_n_from_aggregate(a) for name, a in aggregates.items()}

# ---------------------------------------------------------------------------
# 3. Descriptive endpoint summary + bootstrap intervals
# ---------------------------------------------------------------------------
def metrics(mask: np.ndarray) -> dict[str, float | int]:
    d = kh_diag[mask]
    s = kh_support[mask]
    return {
        "n": int(mask.sum()),
        "kh_diagonal_mean": float(np.mean(d)),
        "kh_diagonal_ge_3_prop": float(np.mean(d >= 3)),
        "kh_diagonal_ge_4_prop": float(np.mean(d >= 4)),
        "kh_support_mean": float(np.mean(s)),
        "alternating_prop": float(np.mean(test_meta.loc[mask, "is_alternating"].to_numpy(int) == 1)),
        "crossing_15_prop": float(np.mean(test_meta.loc[mask, "number_of_crossings"].to_numpy(int) == 15)),
    }


def bootstrap_selected(mask: np.ndarray, reps: int, seed: int):
    idx = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    vals = np.empty((reps, 4), dtype=float)
    for b in range(reps):
        sample = rng.choice(idx, size=len(idx), replace=True)
        d = kh_diag[sample]
        s = kh_support[sample]
        vals[b] = [np.mean(d), np.mean(d >= 3), np.mean(d >= 4), np.mean(s)]
    return vals


summary_rows = []
selected_rows = []
for i, (name, mask) in enumerate(selections.items()):
    m = metrics(mask)
    boot = bootstrap_selected(mask, BOOT_REPS, SEED + i)
    row = {"selection": name, **m}
    for j, endpoint in enumerate((
        "kh_diagonal_mean", "kh_diagonal_ge_3_prop", "kh_diagonal_ge_4_prop", "kh_support_mean"
    )):
        row[f"{endpoint}_ci_low"] = float(np.quantile(boot[:, j], 0.025))
        row[f"{endpoint}_ci_high"] = float(np.quantile(boot[:, j], 0.975))
    summary_rows.append(row)
    for knot_id in test_ids[mask]:
        selected_rows.append({"selection": name, ID_COL: knot_id})

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "heldout_view_ablation_n31_summary.csv", index=False)
pd.DataFrame(selected_rows).to_csv(OUT / "heldout_view_ablation_n31_selected_ids.csv", index=False)

# Pairwise Jaccard.
overlap_rows = []
names = list(selections)
for i, a_name in enumerate(names):
    for b_name in names[i+1:]:
        a, b = selections[a_name], selections[b_name]
        union = a | b
        overlap_rows.append({
            "selection_a": a_name,
            "selection_b": b_name,
            "overlap": int(np.sum(a & b)),
            "union": int(np.sum(union)),
            "jaccard": float(np.sum(a & b) / np.sum(union)),
        })
overlap_df = pd.DataFrame(overlap_rows)
overlap_df.to_csv(OUT / "heldout_view_ablation_pairwise_jaccard.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Read only TEST rows from the huge HOMFLY/Theta source CSVs in chunks
# ---------------------------------------------------------------------------
def read_test_rows_chunked(path: Path, wanted_ids: np.ndarray) -> pd.DataFrame:
    wanted = set(map(str, wanted_ids))
    parts = []
    for chunk_no, chunk in enumerate(pd.read_csv(path, chunksize=CHUNK_ROWS)):
        if "knot_id" not in chunk.columns:
            raise KeyError(f"{path.name} lacks knot_id")
        base = chunk["knot_id"].astype(str).str.strip().str.replace("!", "", regex=False)
        keep = base.isin(wanted)
        if keep.any():
            parts.append(chunk.loc[keep].copy())
        if (chunk_no + 1) % 20 == 0:
            print(f"{path.name}: scanned {(chunk_no+1)*CHUNK_ROWS:,} rows")
    if not parts:
        raise RuntimeError(f"No held-out rows found in {path}")
    out = pd.concat(parts, ignore_index=True)
    out = add_knot_ids(out, id_col="knot_id", mirror_symbol="!")
    out = canonicalize_mirrors_by_signature(
        out,
        id_col="knot_id",
        base_col=ID_COL,
        signature_col="signature",
        mirror_symbol="!",
    )
    out = out.set_index(ID_COL, drop=False).loc[wanted_ids].reset_index(drop=True)
    if not np.array_equal(out[ID_COL].astype(str).to_numpy(), wanted_ids):
        raise RuntimeError(f"{path.name}: test-row alignment failed")
    return out


print("\nReading held-out HOMFLY-PT coefficient rows for fiber audit...")
homfly_rows = read_test_rows_chunked(DATA_DIR / "HomflyPt_upto15_MIRRORS.csv", test_ids)
print("Reading held-out Theta coefficient rows for fiber audit...")
theta_rows = read_test_rows_chunked(DATA_DIR / "theta_upto15.csv", test_ids)

meta_like = {
    "knot_id", "knot_id_clean", "knot_id_base", "number_of_crossings", "table_number",
    "is_alternating", "signature", "minimum_exponent", "maximum_exponent", "s_invariant",
}
h_cols = [c for c in homfly_rows.columns if c not in meta_like and str(c).startswith("a")]
t_cols = [c for c in theta_rows.columns if c not in meta_like and str(c).startswith("T")]
if not h_cols or not t_cols:
    raise RuntimeError("Could not identify HOMFLY/Theta coefficient columns")

# pandas' row hash is only a candidate-fiber key; every multirow candidate fiber
# is verified below by exact coefficient-array equality before being reported.
h_hash = pd.util.hash_pandas_object(homfly_rows[h_cols], index=False).to_numpy(np.uint64)
t_hash = pd.util.hash_pandas_object(theta_rows[t_cols], index=False).to_numpy(np.uint64)
pair_hash = pd.util.hash_pandas_object(
    pd.DataFrame({"h": h_hash.astype(str), "t": t_hash.astype(str)}), index=False
).to_numpy(np.uint64)

fiber = pd.DataFrame({
    ID_COL: test_ids,
    "homfly_hash": h_hash,
    "theta_hash": t_hash,
    "pair_hash": pair_hash,
    "kh_diagonal": kh_diag,
})
for name, mask in selections.items():
    fiber[f"selected__{name}"] = mask

# Verify all repeated hash groups exactly to protect against hash collisions.
def exact_group_ok(indices: np.ndarray, frame: pd.DataFrame, cols: list[str]) -> bool:
    arr = frame.iloc[indices][cols].to_numpy()
    return bool(np.all(arr == arr[0]))

verified_h_groups = []
for key, idx in fiber.groupby("homfly_hash").indices.items():
    idx = np.asarray(idx, dtype=int)
    if len(idx) > 1 and exact_group_ok(idx, homfly_rows, h_cols):
        verified_h_groups.append(idx)
verified_pair_groups = []
for key, idx in fiber.groupby("pair_hash").indices.items():
    idx = np.asarray(idx, dtype=int)
    if len(idx) > 1 and exact_group_ok(idx, homfly_rows, h_cols) and exact_group_ok(idx, theta_rows, t_cols):
        verified_pair_groups.append(idx)

fiber_summary_rows = []
for family, groups in (("HOMFLY_PT", verified_h_groups), ("HOMFLY_PT_plus_Theta", verified_pair_groups)):
    member_count = int(sum(len(g) for g in groups))
    row = {
        "fiber_family": family,
        "n_nontrivial_fibers": int(len(groups)),
        "n_knots_in_nontrivial_fibers": member_count,
        "max_fiber_size": int(max([len(g) for g in groups], default=1)),
    }
    for sel_name, mask in selections.items():
        split = 0
        for g in groups:
            vals = mask[g]
            if vals.any() and (~vals).any():
                split += 1
        row[f"split_fibers__{sel_name}"] = split
    fiber_summary_rows.append(row)

fiber_summary = pd.DataFrame(fiber_summary_rows)
fiber_summary.to_csv(OUT / "exact_input_fiber_summary.csv", index=False)

# Save exact (P,theta) fibers that are split by the four-view fixed-n boundary.
split_rows = []
for g in verified_pair_groups:
    vals = selections["four_view_3of4"][g]
    if vals.any() and (~vals).any():
        for i in g:
            split_rows.append({
                ID_COL: test_ids[i],
                "pair_hash": int(pair_hash[i]),
                "selected_four_view_3of4": bool(selections["four_view_3of4"][i]),
                "four_view_aggregate": float(aggregates["four_view_3of4"][i]),
                "kh_diagonal": float(kh_diag[i]),
            })
pd.DataFrame(split_rows).to_csv(OUT / "pair_fibers_split_by_four_view_boundary.csv", index=False)

# ---------------------------------------------------------------------------
# 5. Checked examples: same HOMFLY-PT, different Theta and different Kh width
# ---------------------------------------------------------------------------
example_rows = []
for g in verified_h_groups:
    # Require at least two genuinely different exact Theta vectors.
    theta_keys = t_hash[g]
    if len(np.unique(theta_keys)) < 2 or len(np.unique(kh_diag[g])) < 2:
        continue
    # Pick the widest separation in stored Khovanov diagonal count.
    lo = g[np.argmin(kh_diag[g])]
    hi = g[np.argmax(kh_diag[g])]
    if t_hash[lo] == t_hash[hi]:
        # Find a different-theta pair if min/max happened to share theta.
        found = None
        for i in g:
            for j in g:
                if t_hash[i] != t_hash[j] and kh_diag[i] != kh_diag[j]:
                    found = (i, j)
                    break
            if found:
                break
        if found is None:
            continue
        lo, hi = found
    example_rows.append({
        "knot_a": test_ids[lo],
        "knot_b": test_ids[hi],
        "same_HOMFLY_PT_exact": True,
        "theta_differs": bool(t_hash[lo] != t_hash[hi]),
        "kh_diagonal_a": float(kh_diag[lo]),
        "kh_diagonal_b": float(kh_diag[hi]),
        "kh_diagonal_abs_difference": float(abs(kh_diag[hi] - kh_diag[lo])),
        "four_view_selected_a": bool(selections["four_view_3of4"][lo]),
        "four_view_selected_b": bool(selections["four_view_3of4"][hi]),
        "pair_selected_a": bool(selections["HOMFLY_PT_plus_Theta_2of2"][lo]),
        "pair_selected_b": bool(selections["HOMFLY_PT_plus_Theta_2of2"][hi]),
    })

examples = pd.DataFrame(example_rows)
if len(examples):
    examples = examples.sort_values("kh_diagonal_abs_difference", ascending=False).head(100)
examples.to_csv(OUT / "same_homfly_different_theta_khovanov_examples.csv", index=False)

# Save fiber identifiers without the huge coefficient matrices.
fiber.to_parquet(OUT / "heldout_homfly_theta_fiber_index.parquet", index=False)

del homfly_rows, theta_rows
gc.collect()

print("\n" + "="*78)
print("STAGE 33 COMPLETE")
print("="*78)
print("\nEqual-cardinality n=31 ablation:")
print(summary.to_string(index=False))
print("\nPairwise Jaccard:")
print(overlap_df.to_string(index=False))
print("\nExact fiber summary:")
print(fiber_summary.to_string(index=False))
print(f"\nChecked same-HOMFLY / different-Theta+Kh example pairs saved: {len(examples)}")
if len(examples):
    print(examples.head(10).to_string(index=False))
print("\nInterpret all score ordering descriptively unless a direct between-score test is added.")
print("Saved Stage 33 to:", OUT)
