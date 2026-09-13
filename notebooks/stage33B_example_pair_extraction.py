# %% [markdown]
# Stage 33B — Select and document a concrete mathematical example pair
#
# Purpose
# -------
# Stage 33 found exact HOMFLY-PT fibers containing knots with different Theta
# representations and different stored Khovanov widths.  This stage ranks those
# checked candidate pairs and produces one paper-ready example with:
#
#   * identical HOMFLY-PT representation
#   * different Theta representation
#   * per-view conditional percentiles
#   * four-view aggregate percentile
#   * fixed-n selection membership
#   * stored Khovanov width/support
#   * nonzero HOMFLY-PT and Theta coefficients for the two knots
#
# Preference order
# ----------------
#   1. four-view selection splits the pair, if such a checked example exists
#   2. larger difference in stored Khovanov diagonal count
#   3. larger difference in Theta score percentile
#   4. larger difference in four-view aggregate score
#
# Fresh-session usage
# -------------------
# %run "/content/drive/MyDrive/consensus_hardness_refactored/notebooks/stage33B_example_pair_extraction.py"

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

DEFAULT_PROJECT_DIR = Path("/content/drive/MyDrive/consensus_hardness_refactored")
DEFAULT_DATA_DIR = Path("/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants")
DEFAULT_ROOT = DEFAULT_DATA_DIR / "processed_consensus_hardness" / "corrected_run_20260819"

PROJECT_DIR = Path(os.environ.get("KNOT_PROJECT_DIR", str(DEFAULT_PROJECT_DIR)))
DATA_DIR = Path(os.environ.get("KNOT_DATA_DIR", str(DEFAULT_DATA_DIR)))
ROOT = Path(os.environ.get("KNOT_OUTPUT_DIR", str(DEFAULT_ROOT)))
STAGE33 = ROOT / "33_view_ablation_and_fibers"
OUT = ROOT / "33B_example_pair_extraction"
ID_COL = "knot_id_base"
NO_KHOVANOV = ("Alexander", "Jones", "HOMFLY-PT", "Theta")
N_SELECT = 31
CHUNK_ROWS = int(os.environ.get("STAGE33B_CHUNK_ROWS", "30000"))


def maybe_mount_drive():
    if Path("/content").exists() and not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive  # type: ignore
            print("Google Drive is not mounted; requesting mount...")
            drive.mount("/content/drive")
        except Exception as exc:
            print("WARNING: could not mount Drive automatically:", exc)


maybe_mount_drive()
if not ROOT.exists() or not STAGE33.exists():
    raise FileNotFoundError("Run Stage 33 first and ensure the frozen root is mounted.")
OUT.mkdir(parents=True, exist_ok=True)

if (PROJECT_DIR / "src").exists() and str(PROJECT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "src"))

try:
    from consensus_hardness.preprocessing import add_knot_ids, canonicalize_mirrors_by_signature
except Exception as exc:
    raise ImportError("Could not import consensus_hardness preprocessing helpers") from exc


def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "_")


def find_one(filename: str) -> Path:
    hits = sorted(ROOT.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename} below {ROOT}")
    return hits[0]


examples_path = STAGE33 / "same_homfly_different_theta_khovanov_examples.csv"
if not examples_path.exists():
    raise FileNotFoundError(examples_path)
examples = pd.read_csv(examples_path, dtype={"knot_a": str, "knot_b": str})
if examples.empty:
    raise RuntimeError("Stage 33 found no checked candidate pairs.")

atlas = pd.read_parquet(find_one("final_hard_regime_atlas.parquet")).reset_index(drop=True)
atlas[ID_COL] = atlas[ID_COL].astype(str)
with np.load(find_one("heldout_ae_seed_0.npz"), allow_pickle=False) as p:
    test_idx = np.asarray(p["test_idx"], dtype=np.int64)
test_meta = atlas.iloc[test_idx].reset_index(drop=True).copy()
test_ids = test_meta[ID_COL].astype(str).to_numpy()
N_TEST = len(test_ids)
id_to_test = pd.Series(np.arange(N_TEST, dtype=np.int64), index=test_ids).to_dict()

phenotype = pd.read_parquet(find_one("complete_mathematical_phenotype_atlas.parquet"))
phenotype[ID_COL] = phenotype[ID_COL].astype(str)
phenotype = phenotype.set_index(ID_COL).reindex(test_ids)
kh_diag_col = next(c for c in (
    "khovanov_q_minus_2t_diagonal_count", "kh_diagonal_count", "khovanov_diagonal_count"
) if c in phenotype.columns)
kh_support_col = next(c for c in (
    "khovanov_support_size", "kh_support_size", "khovanov_f_support_size"
) if c in phenotype.columns)
kh_diag = phenotype[kh_diag_col].to_numpy(float)
kh_support = phenotype[kh_support_col].to_numpy(float)

# Reproduce the exact Stage-33 percentile convention.
score_percentiles = {}
for name in NO_KHOVANOV:
    score_path = ROOT / "23_anomaly_score_baselines" / "checkpoints" / f"{safe_name(name)}_test_scores.npz"
    if not score_path.exists():
        hits = sorted(ROOT.rglob(f"{safe_name(name)}_test_scores.npz"))
        if not hits:
            raise FileNotFoundError(score_path)
        score_path = hits[0]
    with np.load(score_path, allow_pickle=False) as p:
        score = np.asarray(p["test_conditional_percentile_100"], dtype=float)
    if len(score) != N_TEST:
        raise RuntimeError(f"{name}: held-out score length mismatch")
    score_percentiles[name] = rankdata(score, method="average") / N_TEST

four_matrix = np.column_stack([score_percentiles[v] for v in NO_KHOVANOV])
four_aggregate = np.sort(four_matrix, axis=1)[:, 1]
pair_aggregate = np.minimum(
    score_percentiles["HOMFLY-PT"], score_percentiles["Theta"]
)


def top_n_mask(values, n=N_SELECT):
    order = np.lexsort((test_ids, np.asarray(values, float)))
    m = np.zeros(N_TEST, dtype=bool)
    m[order[-n:]] = True
    return m


four_selected = top_n_mask(four_aggregate)
pair_selected = top_n_mask(pair_aggregate)

rows = []
for _, r in examples.iterrows():
    a, b = str(r["knot_a"]), str(r["knot_b"])
    if a not in id_to_test or b not in id_to_test:
        continue
    ia, ib = id_to_test[a], id_to_test[b]
    rec = {
        "knot_a": a,
        "knot_b": b,
        "same_HOMFLY_PT_exact": bool(r.get("same_HOMFLY_PT_exact", True)),
        "kh_diagonal_a": float(kh_diag[ia]),
        "kh_diagonal_b": float(kh_diag[ib]),
        "kh_diagonal_abs_difference": float(abs(kh_diag[ia] - kh_diag[ib])),
        "kh_support_a": float(kh_support[ia]),
        "kh_support_b": float(kh_support[ib]),
        "theta_percentile_a": float(score_percentiles["Theta"][ia]),
        "theta_percentile_b": float(score_percentiles["Theta"][ib]),
        "theta_percentile_abs_difference": float(abs(
            score_percentiles["Theta"][ia] - score_percentiles["Theta"][ib]
        )),
        "four_view_aggregate_a": float(four_aggregate[ia]),
        "four_view_aggregate_b": float(four_aggregate[ib]),
        "four_view_aggregate_abs_difference": float(abs(four_aggregate[ia] - four_aggregate[ib])),
        "four_view_selected_a": bool(four_selected[ia]),
        "four_view_selected_b": bool(four_selected[ib]),
        "four_view_split_pair": bool(four_selected[ia] != four_selected[ib]),
        "pair_aggregate_a": float(pair_aggregate[ia]),
        "pair_aggregate_b": float(pair_aggregate[ib]),
        "pair_selected_a": bool(pair_selected[ia]),
        "pair_selected_b": bool(pair_selected[ib]),
        "alternating_a": int(test_meta.loc[ia, "is_alternating"]),
        "alternating_b": int(test_meta.loc[ib, "is_alternating"]),
        "crossing_a": int(test_meta.loc[ia, "number_of_crossings"]),
        "crossing_b": int(test_meta.loc[ib, "number_of_crossings"]),
    }
    for view in NO_KHOVANOV:
        rec[f"{safe_name(view)}_percentile_a"] = float(score_percentiles[view][ia])
        rec[f"{safe_name(view)}_percentile_b"] = float(score_percentiles[view][ib])
    rows.append(rec)

ranked = pd.DataFrame(rows)
if ranked.empty:
    raise RuntimeError("No candidate pair could be aligned to the held-out test set.")

ranked = ranked.sort_values(
    [
        "four_view_split_pair",
        "kh_diagonal_abs_difference",
        "theta_percentile_abs_difference",
        "four_view_aggregate_abs_difference",
    ],
    ascending=[False, False, False, False],
).reset_index(drop=True)
ranked.insert(0, "example_rank", np.arange(1, len(ranked) + 1))
ranked.to_csv(OUT / "ranked_checked_example_pairs.csv", index=False)

best = ranked.iloc[0].copy()
best.to_frame().T.to_csv(OUT / "best_example_pair.csv", index=False)
best_ids = [str(best["knot_a"]), str(best["knot_b"])]


def read_canonical_rows(path: Path, wanted_ids: list[str]) -> pd.DataFrame:
    wanted = set(wanted_ids)
    parts = []
    for i, chunk in enumerate(pd.read_csv(path, chunksize=CHUNK_ROWS)):
        base = chunk["knot_id"].astype(str).str.strip().str.replace("!", "", regex=False)
        keep = base.isin(wanted)
        if keep.any():
            parts.append(chunk.loc[keep].copy())
        if (i + 1) % 20 == 0:
            print(f"{path.name}: scanned {(i+1)*CHUNK_ROWS:,} rows")
    if not parts:
        raise RuntimeError(f"No candidate rows found in {path}")
    df = pd.concat(parts, ignore_index=True)
    df = add_knot_ids(df, id_col="knot_id", mirror_symbol="!")
    df = canonicalize_mirrors_by_signature(
        df,
        id_col="knot_id",
        base_col=ID_COL,
        signature_col="signature",
        mirror_symbol="!",
    )
    return df.set_index(ID_COL, drop=False).loc[wanted_ids].reset_index(drop=True)


print("\nExtracting source coefficients for the best example...")
hom = read_canonical_rows(DATA_DIR / "HomflyPt_upto15_MIRRORS.csv", best_ids)
theta = read_canonical_rows(DATA_DIR / "theta_upto15.csv", best_ids)

meta_like = {
    "knot_id", "knot_id_clean", "knot_id_base", "number_of_crossings",
    "table_number", "is_alternating", "signature", "minimum_exponent",
    "maximum_exponent", "s_invariant",
}
h_cols = [c for c in hom.columns if c not in meta_like and str(c).startswith("a")]
t_cols = [c for c in theta.columns if c not in meta_like and str(c).startswith("T")]

# Verify HOMFLY equality one more time from exact coefficients.
h_arr = hom[h_cols].to_numpy()
if not np.array_equal(h_arr[0], h_arr[1]):
    raise RuntimeError("Best pair is not exactly equal in HOMFLY coefficients after source extraction.")

coef_rows = []
for family, frame, cols in (
    ("HOMFLY_PT", hom, h_cols),
    ("Theta", theta, t_cols),
):
    for row_i, knot in enumerate(best_ids):
        values = frame.iloc[row_i][cols]
        nonzero = values[values != 0]
        for col, value in nonzero.items():
            coef_rows.append({
                "family": family,
                ID_COL: knot,
                "coordinate": str(col),
                "coefficient": value,
            })

coef_df = pd.DataFrame(coef_rows)
coef_df.to_csv(OUT / "best_example_pair_nonzero_coefficients.csv", index=False)

# Compact wide table for the manuscript notebook.
paper_cols = [
    "knot_a", "knot_b",
    "kh_diagonal_a", "kh_diagonal_b",
    "kh_support_a", "kh_support_b",
    "Alexander_percentile_a", "Alexander_percentile_b",
    "Jones_percentile_a", "Jones_percentile_b",
    "HOMFLY_PT_percentile_a", "HOMFLY_PT_percentile_b",
    "Theta_percentile_a", "Theta_percentile_b",
    "four_view_aggregate_a", "four_view_aggregate_b",
    "four_view_selected_a", "four_view_selected_b",
]
best_table = best.to_frame().T
best_table[paper_cols].to_csv(OUT / "best_example_pair_paper_table.csv", index=False)

summary_text = f"""Best checked example pair
=========================
Knot A: {best['knot_a']}
Knot B: {best['knot_b']}

Exact HOMFLY-PT equality: True
Stored Khovanov diagonal counts: {best['kh_diagonal_a']} vs {best['kh_diagonal_b']}
Stored Khovanov support sizes: {best['kh_support_a']} vs {best['kh_support_b']}
Theta percentiles: {best['theta_percentile_a']:.6f} vs {best['theta_percentile_b']:.6f}
Four-view aggregate: {best['four_view_aggregate_a']:.6f} vs {best['four_view_aggregate_b']:.6f}
Four-view selected: {bool(best['four_view_selected_a'])} vs {bool(best['four_view_selected_b'])}
Four-view splits pair: {bool(best['four_view_split_pair'])}

Selection of this example is descriptive.  Its purpose is to show concretely
that an exact HOMFLY-PT fiber can contain knots separated by Theta and by the
stored Khovanov endpoint.
"""
(OUT / "best_example_pair_summary.txt").write_text(summary_text)

del hom, theta
gc.collect()

print("\n" + "="*88)
print("STAGE 33B COMPLETE")
print("="*88)
print("\nTop 10 ranked checked examples:")
print(ranked.head(10).to_string(index=False))
print("\nChosen example:")
print(best.to_string())
print("\nSaved source coefficients and paper table to:", OUT)
