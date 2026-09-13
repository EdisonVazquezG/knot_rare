# %% [markdown]
# Stage 29b — Final dependence-preserving fixed-cardinality null
#
# Purpose
# -------
# Replace the legacy per-view independent permutation null for paper-facing
# inference.
#
# Null hypothesis (consensus-level):
#   Conditional on the exact structural strata and coarse joint amplitude
#   strata below, the observed multiview consensus label is exchangeable with
#   respect to the external mathematical endpoint.
#
# Why this preserves cross-view dependence:
#   Permuting the complete observed vote vector as one atomic object within a
#   common stratum preserves every joint vote pattern and therefore all
#   cross-view dependence. For an endpoint that depends only on m-of-M
#   consensus membership, the induced recipient set is exactly a simple random
#   sample of the observed number of consensus-positive vote vectors in that
#   stratum. We implement that mathematically equivalent, memory-efficient
#   sampler.
#
# Primary amplitude resolution:
#   2 bins per relevant representation. This is the finest nontrivial common
#   grid from Stage 29 for which median exact cell size is >=10 in BOTH paper
#   families, with high selected-set mobility.
#
# Families:
#   A) all-five conditional consensus, exact Khovanov diagonal-count adjustment
#      endpoints: P(G>0), mean G, P(G>=4)
#   B) no-Khovanov conditional consensus, Khovanov external outcome
#      endpoints: mean diagonal count, P(diag>=3), P(diag>=4), mean support
#
# Holm correction is performed within each endpoint family.
#
# Standalone: reads frozen artifacts below KNOT_OUTPUT_DIR.

from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants/"
    "processed_consensus_hardness/corrected_run_20260819"
)
ROOT = Path(os.environ.get("KNOT_OUTPUT_DIR", str(DEFAULT_ROOT)))
if not ROOT.exists():
    raise FileNotFoundError(ROOT)

OUT = ROOT / "29b_final_joint_null"
OUT.mkdir(parents=True, exist_ok=True)

B = int(os.environ.get("STAGE29B_REPS", "5000"))
SEED = int(os.environ.get("STAGE29B_SEED", "202608291"))
JOINT_BINS = int(os.environ.get("STAGE29B_JOINT_BINS", "2"))
ID_COL = "knot_id_base"
INVARIANTS = ("Alexander", "Jones", "HOMFLY-PT", "Theta", "Khovanov")
NO_KHOVANOV = tuple(x for x in INVARIANTS if x != "Khovanov")

def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "_")

def find_one(filename: str) -> Path:
    found = sorted(ROOT.rglob(filename))
    if not found:
        raise FileNotFoundError(f"Could not find {filename} below {ROOT}")
    return found[0]

atlas = pd.read_parquet(find_one("final_hard_regime_atlas.parquet")).reset_index(drop=True)
phenotype = pd.read_parquet(find_one("complete_mathematical_phenotype_atlas.parquet")).reset_index(drop=True)
atlas[ID_COL] = atlas[ID_COL].astype(str)
phenotype[ID_COL] = phenotype[ID_COL].astype(str)
if not atlas[ID_COL].equals(phenotype[ID_COL]):
    phenotype = atlas[[ID_COL]].merge(phenotype, on=ID_COL, how="left", validate="one_to_one")

N = len(atlas)
S_COL = next(c for c in ("s_invariant_qc", "s_invariant", "s") if c in atlas.columns)
G = np.abs(atlas[S_COL].to_numpy(float) - atlas["signature"].to_numpy(float))
NONALT = atlas["is_alternating"].to_numpy(int) == 0

KH_DIAG_COL = next(
    c for c in (
        "khovanov_q_minus_2t_diagonal_count",
        "kh_diagonal_count",
        "khovanov_diagonal_count",
    )
    if c in phenotype.columns
)
KH_SUPPORT_COL = next(
    c for c in ("khovanov_support_size", "kh_support_size")
    if c in phenotype.columns
)
KH_DIAG = phenotype[KH_DIAG_COL].to_numpy(float)
KH_SUPPORT = phenotype[KH_SUPPORT_COL].to_numpy(float)

with np.load(find_one("conditional_100bins_scores.npz"), allow_pickle=False) as payload:
    norm_bins = {
        name: np.asarray(payload[f"{safe_name(name)}_norm_bin"], dtype=np.int32)
        for name in INVARIANTS
    }

hard_saved = pd.read_csv(find_one("conditional_100bins_hard_sets.csv"), dtype={ID_COL: str})
id_to_pos = pd.Series(np.arange(N, dtype=np.int64), index=atlas[ID_COL]).to_dict()
hard_masks = {}
for name in INVARIANTS:
    ids = hard_saved.loc[hard_saved["invariant"].eq(name), ID_COL].astype(str)
    mask = np.zeros(N, dtype=bool)
    mask[[id_to_pos[x] for x in ids]] = True
    hard_masks[name] = mask

vote_matrix = np.column_stack([hard_masks[n].astype(np.uint8) for n in INVARIANTS])
all5_mask = vote_matrix.sum(axis=1) >= 3
nokh_mask = vote_matrix[:, :4].sum(axis=1) >= 3  # first four are non-Kh

def coarsen_bins(codes: np.ndarray, target_bins: int) -> np.ndarray:
    codes = np.asarray(codes)
    unique = np.unique(codes)
    rank = np.searchsorted(unique, codes)
    n_out = min(target_bins, len(unique))
    out = np.floor(rank * n_out / len(unique)).astype(np.int32)
    return np.minimum(out, n_out - 1)

def factorize(frame: pd.DataFrame) -> np.ndarray:
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(frame.astype(str)), sort=False)
    return codes.astype(np.int32)

def prepare_sampler(codes: np.ndarray, selected: np.ndarray):
    order = np.argsort(codes, kind="stable")
    c = codes[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(c))]
    stops = np.r_[starts[1:], len(order)]
    random_groups, fixed_groups, sizes = [], [], []
    for a, b in zip(starts, stops):
        idx = order[a:b]
        k = int(selected[idx].sum())
        sizes.append(len(idx))
        if k == 0:
            continue
        if k == len(idx):
            fixed_groups.append(idx)
        else:
            random_groups.append((idx, k))
    fixed_idx = np.concatenate(fixed_groups) if fixed_groups else np.empty(0, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=int)
    return {
        "random_groups": random_groups,
        "fixed_idx": fixed_idx,
    }, {
        "n_strata": int(len(sizes)),
        "cell_min": int(sizes.min()),
        "cell_q25": float(np.quantile(sizes, .25)),
        "cell_median": float(np.median(sizes)),
        "cell_q75": float(np.quantile(sizes, .75)),
        "singleton_prop": float(np.mean(sizes == 1)),
        "lt10_prop": float(np.mean(sizes < 10)),
        "selected_total": int(selected.sum()),
        "selected_fixed": int(len(fixed_idx)),
        "selected_movable": int(sum(k for _, k in random_groups)),
        "selected_movable_prop": float(
            sum(k for _, k in random_groups) / max(1, int(selected.sum()))
        ),
        "random_groups": int(len(random_groups)),
    }

def sample_mask(sampler, rng):
    mask = np.zeros(N, dtype=bool)
    mask[sampler["fixed_idx"]] = True
    for idx, k in sampler["random_groups"]:
        mask[rng.choice(idx, size=k, replace=False)] = True
    return mask

def gap_metrics(mask):
    target = mask & NONALT
    x = G[target]
    return {
        "P_G_gt_0": float(np.mean(x > 0)),
        "mean_G": float(np.mean(x)),
        "P_G_ge_4": float(np.mean(x >= 4)),
    }

def kh_metrics(mask):
    d = KH_DIAG[mask]
    s = KH_SUPPORT[mask]
    return {
        "mean_diag": float(np.mean(d)),
        "P_diag_ge_3": float(np.mean(d >= 3)),
        "P_diag_ge_4": float(np.mean(d >= 4)),
        "mean_support": float(np.mean(s)),
    }

def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    out = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running = max(running, candidate)
        out[idx] = min(1.0, running)
    return out

# ---------------------------------------------------------------------------
# Common exact/coarse strata
# ---------------------------------------------------------------------------
struct = pd.DataFrame({
    "crossing": atlas["number_of_crossings"].to_numpy(),
    "alternating": atlas["is_alternating"].to_numpy(),
    "abs_sigma": atlas["signature"].abs().to_numpy(),
})

# A) all-five G after exact Khovanov thickness
gap_frame = struct.copy()
for name in INVARIANTS:
    gap_frame[f"{safe_name(name)}_norm{JOINT_BINS}"] = coarsen_bins(
        norm_bins[name], JOINT_BINS
    )
gap_frame["kh_diag_exact"] = KH_DIAG.astype(int)
gap_codes = factorize(gap_frame)
gap_sampler, gap_diag = prepare_sampler(gap_codes, all5_mask)

# B) no-Khovanov external Khovanov phenotype
kh_frame = struct.copy()
for name in NO_KHOVANOV:
    kh_frame[f"{safe_name(name)}_norm{JOINT_BINS}"] = coarsen_bins(
        norm_bins[name], JOINT_BINS
    )
kh_frame[f"Khovanov_norm{JOINT_BINS}"] = coarsen_bins(
    norm_bins["Khovanov"], JOINT_BINS
)
kh_codes = factorize(kh_frame)
kh_sampler, kh_diag = prepare_sampler(kh_codes, nokh_mask)

diagnostics = pd.DataFrame([
    {"analysis": "gap_exact_thickness_all5", "joint_bins": JOINT_BINS, **gap_diag},
    {"analysis": "noKh_external_Khovanov", "joint_bins": JOINT_BINS, **kh_diag},
])
diagnostics.to_csv(OUT / "primary_joint_strata_diagnostics.csv", index=False)

# ---------------------------------------------------------------------------
# Monte Carlo randomization
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
gap_obs = gap_metrics(all5_mask)
kh_obs = kh_metrics(nokh_mask)

gap_rows, kh_rows = [], []
for rep in range(B):
    gap_rows.append({"replicate": rep, **gap_metrics(sample_mask(gap_sampler, rng))})
    kh_rows.append({"replicate": rep, **kh_metrics(sample_mask(kh_sampler, rng))})

gap_null = pd.DataFrame(gap_rows)
kh_null = pd.DataFrame(kh_rows)
gap_null.to_parquet(OUT / "gap_exact_thickness_joint_null.parquet", index=False)
kh_null.to_parquet(OUT / "noKh_external_Khovanov_joint_null.parquet", index=False)

def summarize_family(analysis, observed, null_df):
    rows = []
    for metric, obs in observed.items():
        vals = null_df[metric].to_numpy(float)
        p = (1 + int(np.sum(vals >= obs))) / (B + 1)
        rows.append({
            "analysis": analysis,
            "metric": metric,
            "observed": obs,
            "null_mean": float(vals.mean()),
            "null_sd": float(vals.std(ddof=1)),
            "null_q025": float(np.quantile(vals, .025)),
            "null_q975": float(np.quantile(vals, .975)),
            "empirical_p_upper": float(p),
            "B": B,
        })
    frame = pd.DataFrame(rows)
    frame["Holm_p_within_family"] = holm_adjust(frame["empirical_p_upper"].to_numpy())
    return frame

gap_table = summarize_family("gap_exact_thickness_all5", gap_obs, gap_null)
kh_table = summarize_family("noKh_external_Khovanov", kh_obs, kh_null)
paper_table = pd.concat([gap_table, kh_table], ignore_index=True)
paper_table.to_csv(OUT / "paper_facing_joint_null_results.csv", index=False)

# ---------------------------------------------------------------------------
# Audit of actual vote-pattern preservation implied by the null
# ---------------------------------------------------------------------------
pattern = pd.Series(
    ["".join(map(str, row.tolist())) for row in vote_matrix],
    name="vote_pattern",
)
pattern_table = (
    pd.DataFrame({"vote_pattern": pattern})
    .value_counts()
    .rename("n")
    .reset_index()
    .sort_values(["n", "vote_pattern"], ascending=[False, True])
)
pattern_table.to_csv(OUT / "observed_joint_vote_patterns.csv", index=False)

# ---------------------------------------------------------------------------
# Text summary for manuscript update
# ---------------------------------------------------------------------------
gap_sig = gap_table.set_index("metric")
kh_sig = kh_table.set_index("metric")

summary_lines = [
    "Stage 29b final joint-null summary",
    "==================================",
    f"Monte Carlo repetitions: {B}",
    f"Joint amplitude bins per relevant view: {JOINT_BINS}",
    "",
    "Cell diagnostics:",
    diagnostics.to_string(index=False),
    "",
    "Paper-facing endpoint table:",
    paper_table.to_string(index=False),
    "",
    "Interpretation:",
    (
        "- The null preserves the observed m-of-M consensus cardinality exactly "
        "within every common stratum and, equivalently, preserves the complete "
        "cross-view vote vector under joint permutation."
    ),
    (
        "- Fine joint grids (20/10 bins) from Stage 29 were not adopted because "
        "they lock a large fraction of selected observations. Two bins is the "
        "finest nontrivial common grid satisfying the median-cell >=10 criterion "
        "in both endpoint families."
    ),
]
(OUT / "stage29b_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

print("\nPrimary joint-strata diagnostics:")
print(diagnostics.to_string(index=False))
print("\nPaper-facing corrected null results:")
print(paper_table.to_string(index=False))
print("\nSaved Stage 29b to:", OUT)
