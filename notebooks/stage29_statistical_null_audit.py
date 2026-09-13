# %% [markdown]
# Stage 29 — Statistical audit of the factorized multiview permutation null
#
# Reviewer concern:
#   The current null samples each view-specific hard mask independently inside
#   its own exact strata and then recomputes the m-of-M consensus. This breaks
#   observed cross-view vote dependence and does not fix consensus cardinality.
#
# This standalone audit does FOUR things:
#   A. quantifies observed cross-view vote dependence;
#   B. measures how much the legacy null moves consensus size;
#   C. compares paper endpoints with a fixed-cardinality joint-consensus
#      permutation sensitivity (equivalent, for consensus-level endpoints, to
#      permuting the joint vote vector and then thresholding);
#   D. performs a Type-I simulation under a covariate-driven synthetic null.
#
# The script is an AUDIT. Do not replace paper p-values automatically. The
# decision table tells you whether the old null is empirically calibrated and
# whether the fixed-cardinality sensitivity agrees qualitatively.

from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

DEFAULT_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants/"
    "processed_consensus_hardness/corrected_run_20260819"
)
ROOT = Path(globals().get("OUTPUT_DIR", DEFAULT_ROOT))
if not ROOT.exists():
    raise FileNotFoundError(ROOT)
OUT = ROOT / "29_statistical_null_audit"
OUT.mkdir(parents=True, exist_ok=True)

N_REPS = int(os.environ.get("STAGE29_NULL_REPS", "2000"))
TYPE1_INNER = int(os.environ.get("STAGE29_TYPE1_INNER", "399"))
TYPE1_OUTER = int(os.environ.get("STAGE29_TYPE1_OUTER", "200"))
SEED = int(os.environ.get("STAGE29_SEED", "20260829"))
JOINT_BIN_GRID = tuple(
    int(x) for x in os.environ.get("STAGE29_JOINT_BINS", "20,10,5,2,1").split(",")
)

INVARIANTS = ("Alexander", "Jones", "HOMFLY-PT", "Theta", "Khovanov")
NO_KHOVANOV = tuple(x for x in INVARIANTS if x != "Khovanov")
ID_COL = "knot_id_base"

def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "_")

def find_one(filename: str) -> Path:
    found = sorted(ROOT.rglob(filename))
    if not found:
        raise FileNotFoundError(f"Could not find {filename} below {ROOT}")
    return found[0]

atlas = pd.read_parquet(find_one("final_hard_regime_atlas.parquet")).reset_index(drop=True)
phenotype = pd.read_parquet(find_one("complete_mathematical_phenotype_atlas.parquet")).reset_index(drop=True)
for frame in (atlas, phenotype):
    frame[ID_COL] = frame[ID_COL].astype(str)
if not atlas[ID_COL].equals(phenotype[ID_COL]):
    phenotype = atlas[[ID_COL]].merge(phenotype, on=ID_COL, how="left", validate="one_to_one")

N = len(atlas)
S_COL = next(c for c in ("s_invariant_qc", "s_invariant", "s") if c in atlas)
G = np.abs(atlas[S_COL].to_numpy(float) - atlas["signature"].to_numpy(float))
NONALT = atlas["is_alternating"].to_numpy(int) == 0
KH_DIAG_COL = next(
    c for c in ("khovanov_q_minus_2t_diagonal_count", "kh_diagonal_count", "khovanov_diagonal_count")
    if c in phenotype
)
KH_SUPPORT_COL = next(c for c in ("khovanov_support_size", "kh_support_size") if c in phenotype)
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

def family_consensus(names, k=3):
    count = np.zeros(N, dtype=np.uint8)
    for name in names:
        count += hard_masks[name]
    return count >= k, count

ALL5_MASK, ALL5_VOTES = family_consensus(INVARIANTS, 3)
NOKH_MASK, NOKH_VOTES = family_consensus(NO_KHOVANOV, 3)

# ---------------------------------------------------------------------------
# A. Vote-dependence diagnostics
# ---------------------------------------------------------------------------
vote_matrix = np.column_stack([hard_masks[n].astype(int) for n in INVARIANTS])
corr = np.corrcoef(vote_matrix, rowvar=False)
pd.DataFrame(corr, index=INVARIANTS, columns=INVARIANTS).to_csv(
    OUT / "observed_vote_phi_correlations.csv"
)
for label, votes in (("all5", ALL5_VOTES), ("no_khovanov", NOKH_VOTES)):
    pd.Series(votes).value_counts().sort_index().rename_axis("vote_count").reset_index(
        name="n_knots"
    ).to_csv(OUT / f"{label}_observed_vote_count_distribution.csv", index=False)

# ---------------------------------------------------------------------------
# Sampler utilities
# ---------------------------------------------------------------------------
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
    random_groups, fixed_groups = [], []
    sizes, selected_counts = [], []
    for a, b in zip(starts, stops):
        idx = order[a:b]
        k = int(selected[idx].sum())
        sizes.append(len(idx))
        selected_counts.append(k)
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
        "n_strata": len(sizes),
        "cell_min": int(sizes.min()),
        "cell_q25": float(np.quantile(sizes, .25)),
        "cell_median": float(np.median(sizes)),
        "cell_q75": float(np.quantile(sizes, .75)),
        "singleton_prop": float(np.mean(sizes == 1)),
        "lt10_prop": float(np.mean(sizes < 10)),
        "selected_total": int(selected.sum()),
        "selected_fixed": int(len(fixed_idx)),
        "selected_movable": int(sum(k for _, k in random_groups)),
        "random_groups": len(random_groups),
    }

def sample_mask(sampler, rng):
    mask = np.zeros(N, dtype=bool)
    mask[sampler["fixed_idx"]] = True
    for idx, k in sampler["random_groups"]:
        mask[rng.choice(idx, size=k, replace=False)] = True
    return mask

def structural_frame():
    return pd.DataFrame({
        "crossing": atlas["number_of_crossings"].to_numpy(),
        "alternating": atlas["is_alternating"].to_numpy(),
        "abs_sigma": atlas["signature"].abs().to_numpy(),
    })

def legacy_view_codes(view, extra=None):
    frame = structural_frame()
    frame.insert(0, "view_norm_bin", norm_bins[view])
    if extra:
        for key, values in extra.items():
            frame[key] = values
    return factorize(frame)

def joint_codes(names, target_bins, extra=None):
    frame = structural_frame()
    for name in names:
        frame[f"{safe_name(name)}_norm"] = coarsen_bins(norm_bins[name], target_bins)
    if extra:
        for key, values in extra.items():
            frame[key] = values
    return factorize(frame)

def gap_metrics(mask):
    target = mask & NONALT
    values = G[target]
    return {
        "n": int(mask.sum()),
        "nonalt_n": int(target.sum()),
        "P_G_gt_0": float(np.mean(values > 0)) if len(values) else np.nan,
        "mean_G": float(np.mean(values)) if len(values) else np.nan,
        "P_G_ge_4": float(np.mean(values >= 4)) if len(values) else np.nan,
    }

def kh_metrics(mask):
    d = KH_DIAG[mask]
    s = KH_SUPPORT[mask]
    return {
        "n": int(mask.sum()),
        "mean_diag": float(np.mean(d)),
        "P_diag_ge_3": float(np.mean(d >= 3)),
        "P_diag_ge_4": float(np.mean(d >= 4)),
        "mean_support": float(np.mean(s)),
    }

# ---------------------------------------------------------------------------
# B. Legacy null: quantify consensus-size variation
# ---------------------------------------------------------------------------
legacy_specs = {
    "gap_exact_thickness_all5": {
        "names": INVARIANTS,
        "k": 3,
        "observed": ALL5_MASK,
        "extra": {"exact_kh_diag": KH_DIAG.astype(int)},
        "metrics": gap_metrics,
    },
    "noKh_external_kh_norm_100": {
        "names": NO_KHOVANOV,
        "k": 3,
        "observed": NOKH_MASK,
        "extra": {"kh_norm_100": norm_bins["Khovanov"]},
        "metrics": kh_metrics,
    },
}
legacy_summary_rows = []
legacy_null_tables = {}
rng = np.random.default_rng(SEED)

for label, spec in legacy_specs.items():
    samplers = {}
    for view in spec["names"]:
        codes = legacy_view_codes(view, extra=spec["extra"])
        samplers[view], _ = prepare_sampler(codes, hard_masks[view])
    rows = []
    for rep in range(N_REPS):
        count = np.zeros(N, dtype=np.uint8)
        for view in spec["names"]:
            count += sample_mask(samplers[view], rng)
        mask = count >= spec["k"]
        rows.append({"replicate": rep, **spec["metrics"](mask)})
    null = pd.DataFrame(rows)
    legacy_null_tables[label] = null
    null.to_parquet(OUT / f"{label}_legacy_null.parquet", index=False)
    nobs = int(spec["observed"].sum())
    legacy_summary_rows.append({
        "analysis": label,
        "observed_consensus_n": nobs,
        "legacy_null_n_mean": float(null["n"].mean()),
        "legacy_null_n_sd": float(null["n"].std(ddof=1)),
        "legacy_null_n_q025": float(null["n"].quantile(.025)),
        "legacy_null_n_q975": float(null["n"].quantile(.975)),
        "observed_n_percentile_in_legacy": float(np.mean(null["n"] <= nobs)),
    })

legacy_size_summary = pd.DataFrame(legacy_summary_rows)
legacy_size_summary.to_csv(OUT / "legacy_consensus_size_audit.csv", index=False)

# ---------------------------------------------------------------------------
# C. Fixed-cardinality joint-consensus sensitivity
#
# At the level of consensus endpoints, shuffling the complete vote vector
# within a common stratum and thresholding is equivalent to shuffling the
# resulting observed consensus indicator within that stratum. This preserves
# the number of m-of-M selected objects exactly in every stratum and globally.
# ---------------------------------------------------------------------------
joint_rows, joint_diag_rows = [], []
for label, spec in legacy_specs.items():
    observed_metrics = spec["metrics"](spec["observed"])
    for target_bins in JOINT_BIN_GRID:
        extra = dict(spec["extra"])
        # For the external-Kh analysis, coarsen the extra Kh norm in lockstep.
        if "kh_norm_100" in extra:
            extra["kh_norm"] = coarsen_bins(norm_bins["Khovanov"], target_bins)
            del extra["kh_norm_100"]
        codes = joint_codes(spec["names"], target_bins, extra=extra)
        sampler, diag = prepare_sampler(codes, spec["observed"])
        joint_diag_rows.append({
            "analysis": label,
            "joint_norm_bins_per_view": target_bins,
            **diag,
        })
        rng_joint = np.random.default_rng(SEED + 1000 + target_bins)
        rows = []
        for rep in range(N_REPS):
            rows.append(spec["metrics"](sample_mask(sampler, rng_joint)))
        null = pd.DataFrame(rows)
        for metric, observed_value in observed_metrics.items():
            if metric.endswith("_n") or metric == "n":
                continue
            vals = null[metric].dropna().to_numpy(float)
            if not len(vals):
                continue
            p = (1 + np.sum(vals >= observed_value)) / (1 + len(vals))
            joint_rows.append({
                "analysis": label,
                "joint_norm_bins_per_view": target_bins,
                "metric": metric,
                "observed": observed_value,
                "null_mean": float(vals.mean()),
                "null_sd": float(vals.std(ddof=1)),
                "empirical_p_upper": float(p),
                "consensus_n_fixed": int(spec["observed"].sum()),
            })

joint_results = pd.DataFrame(joint_rows)
joint_diags = pd.DataFrame(joint_diag_rows)
joint_results.to_csv(OUT / "joint_fixed_cardinality_null_results.csv", index=False)
joint_diags.to_csv(OUT / "joint_fixed_cardinality_strata_diagnostics.csv", index=False)

# ---------------------------------------------------------------------------
# D. Type-I simulation for the legacy null
#
# Synthetic outcome = joint-stratum effect + independent noise.
# By construction it has NO association with consensus after conditioning on
# the joint stratum. We compare rejection rates for:
#   - legacy independent-per-view mask sampling
#   - fixed-cardinality joint-consensus sampling
# ---------------------------------------------------------------------------
TYPE1_BINS = int(os.environ.get("STAGE29_TYPE1_JOINT_BINS", "2"))
type1_codes = joint_codes(INVARIANTS, TYPE1_BINS)
n_codes = int(type1_codes.max()) + 1
joint_sampler, joint_diag = prepare_sampler(type1_codes, ALL5_MASK)

# Precompute null selected-index lists once.
legacy_samplers = {
    v: prepare_sampler(legacy_view_codes(v), hard_masks[v])[0]
    for v in INVARIANTS
}
rng_pre = np.random.default_rng(SEED + 2222)
legacy_idx = []
joint_idx = []
for rep in range(TYPE1_INNER):
    count = np.zeros(N, dtype=np.uint8)
    for v in INVARIANTS:
        count += sample_mask(legacy_samplers[v], rng_pre)
    legacy_idx.append(np.flatnonzero(count >= 3))
    joint_idx.append(np.flatnonzero(sample_mask(joint_sampler, rng_pre)))

obs_idx = np.flatnonzero(ALL5_MASK)
rng_outer = np.random.default_rng(SEED + 3333)
legacy_reject = 0
joint_reject = 0
p_rows = []

for outer in range(TYPE1_OUTER):
    stratum_effect = rng_outer.normal(0, 1, size=n_codes)
    y = stratum_effect[type1_codes] + rng_outer.normal(0, 1, size=N)
    obs = float(np.mean(y[obs_idx]))
    legacy_stats = np.asarray([np.mean(y[idx]) for idx in legacy_idx])
    joint_stats = np.asarray([np.mean(y[idx]) for idx in joint_idx])
    p_legacy = float((1 + np.sum(legacy_stats >= obs)) / (1 + len(legacy_stats)))
    p_joint = float((1 + np.sum(joint_stats >= obs)) / (1 + len(joint_stats)))
    legacy_reject += p_legacy <= .05
    joint_reject += p_joint <= .05
    p_rows.append({"outer": outer, "p_legacy": p_legacy, "p_joint": p_joint})

type1 = pd.DataFrame(p_rows)
type1.to_csv(OUT / "type1_simulation_pvalues.csv", index=False)
type1_summary = pd.DataFrame([{
    "outer_reps": TYPE1_OUTER,
    "inner_permutations": TYPE1_INNER,
    "joint_bins_used": TYPE1_BINS,
    "legacy_rejection_rate_alpha_0_05": legacy_reject / TYPE1_OUTER,
    "joint_rejection_rate_alpha_0_05": joint_reject / TYPE1_OUTER,
    "legacy_pass_0_025_to_0_075": 0.025 <= legacy_reject / TYPE1_OUTER <= 0.075,
    "joint_pass_0_025_to_0_075": 0.025 <= joint_reject / TYPE1_OUTER <= 0.075,
}])
type1_summary.to_csv(OUT / "type1_simulation_summary.csv", index=False)

# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------
max_abs_vote_corr = float(np.max(np.abs(corr - np.eye(len(INVARIANTS)))))
legacy_type1 = float(type1_summary.iloc[0]["legacy_rejection_rate_alpha_0_05"])
joint_type1 = float(type1_summary.iloc[0]["joint_rejection_rate_alpha_0_05"])
decision = pd.DataFrame([{
    "max_abs_pairwise_vote_phi": max_abs_vote_corr,
    "cross_view_dependence_present": bool(max_abs_vote_corr > 0.05),
    "legacy_type1_rate": legacy_type1,
    "joint_type1_rate": joint_type1,
    "legacy_type1_pass": bool(0.025 <= legacy_type1 <= 0.075),
    "joint_type1_pass": bool(0.025 <= joint_type1 <= 0.075),
    "interpretation": (
        "Legacy null is empirically calibrated in this synthetic audit; retain it only if "
        "joint fixed-cardinality endpoint sensitivities agree qualitatively."
        if 0.025 <= legacy_type1 <= 0.075
        else
        "Legacy null shows Type-I miscalibration in this audit. Replace formal endpoint "
        "inference with a dependence-preserving/fixed-cardinality design."
    ),
}])
decision.to_csv(OUT / "statistical_audit_decision.csv", index=False)

print("\nLegacy consensus-size audit:")
print(legacy_size_summary.to_string(index=False))
print("\nJoint-strata diagnostics:")
print(joint_diags.to_string(index=False))
print("\nType-I simulation:")
print(type1_summary.to_string(index=False))
print("\nDecision:")
print(decision.to_string(index=False))
print("\nSaved Stage 29 to:", OUT)
