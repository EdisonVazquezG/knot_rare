# %% [markdown]
# Stage 29c — Cross-fitted continuous-amplitude residualized sensitivity
#
# Reviewer question:
#   Does the association survive fine amplitude adjustment without creating
#   nearly frozen exact strata?
#
# Design:
#   1. Rebuild the aligned five-view matrices using the frozen universe.
#   2. Standardize each representation on the full aligned universe, matching
#      the canonical full-fit representation convention used for the atlas.
#   3. Compute continuous log squared norms for all five views.
#   4. Cross-fit a flexible outcome regression on continuous log norms plus the
#      structural covariates used in the null.
#   5. Test whether the observed consensus has positive mean cross-fitted
#      residual using a fixed-cardinality permutation of the CONSENSUS LABEL
#      within exact structural strata (not amplitude bins).
#
# This is a MODEL-ASSISTED SENSITIVITY, not the paper's primary exact
# randomization test. It deliberately separates:
#   (a) dependence-preserving fixed-cardinality randomization, from
#   (b) fine/continuous amplitude adjustment.
#
# Endpoints:
#   - No-Khovanov: mean stored F_-part diagonal count (primary withheld-view)
#   - All-five nonalternating: P(G>0) (primary residual concordance endpoint)
#   - All-five nonalternating: mean G (secondary supporting endpoint)
#
# The same consensus membership is used as in the frozen canonical analysis.

from __future__ import annotations
from pathlib import Path
import os
import re
import hashlib
import numpy as np
import pandas as pd

import consensus_hardness as ch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

DEFAULT_DATA_DIR = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants"
)
DEFAULT_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants/"
    "processed_consensus_hardness/corrected_run_20260819"
)

DATA_DIR = Path(os.environ.get("KNOT_DATA_DIR", str(DEFAULT_DATA_DIR)))
ROOT = Path(os.environ.get("KNOT_OUTPUT_DIR", str(DEFAULT_ROOT)))
OUT = ROOT / "29c_continuous_amplitude_residualized"
OUT.mkdir(parents=True, exist_ok=True)

B = int(os.environ.get("STAGE29C_REPS", "5000"))
N_FOLDS = int(os.environ.get("STAGE29C_FOLDS", "5"))
SEED = int(os.environ.get("STAGE29C_SEED", "202608292"))
ID_COL = "knot_id_base"
INVARIANTS = ("Alexander", "Jones", "HOMFLY-PT", "Theta", "Khovanov")
NO_KHOVANOV = ("Alexander", "Jones", "HOMFLY-PT", "Theta")


def deterministic_offset(label: str, modulus: int = 10000) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % modulus

def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "_")

def find_one(filename: str) -> Path:
    hits = sorted(ROOT.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename} below {ROOT}")
    return hits[0]

# ---------------------------------------------------------------------
# Frozen atlas / canonical consensus
# ---------------------------------------------------------------------
atlas = pd.read_parquet(find_one("final_hard_regime_atlas.parquet")).reset_index(drop=True)
phenotype = pd.read_parquet(find_one("complete_mathematical_phenotype_atlas.parquet")).reset_index(drop=True)
atlas[ID_COL] = atlas[ID_COL].astype(str)
phenotype[ID_COL] = phenotype[ID_COL].astype(str)
if not atlas[ID_COL].equals(phenotype[ID_COL]):
    phenotype = atlas[[ID_COL]].merge(phenotype, on=ID_COL, how="left", validate="one_to_one")

N = len(atlas)
S_COL = next(c for c in ("s_invariant_qc", "s_invariant", "s") if c in atlas.columns)
G = np.abs(atlas[S_COL].to_numpy(float) - atlas["signature"].to_numpy(float))

KH_DIAG_COL = next(
    c for c in (
        "khovanov_q_minus_2t_diagonal_count",
        "kh_diagonal_count",
        "khovanov_diagonal_count",
    )
    if c in phenotype.columns
)
KH_DIAG = phenotype[KH_DIAG_COL].to_numpy(float)

hard_saved = pd.read_csv(find_one("conditional_100bins_hard_sets.csv"), dtype={ID_COL: str})
id_to_pos = pd.Series(np.arange(N, dtype=np.int64), index=atlas[ID_COL]).to_dict()
hard_masks = {}
for name in INVARIANTS:
    ids = hard_saved.loc[hard_saved["invariant"].eq(name), ID_COL].astype(str)
    m = np.zeros(N, dtype=bool)
    m[[id_to_pos[x] for x in ids]] = True
    hard_masks[name] = m

votes = np.column_stack([hard_masks[name].astype(np.uint8) for name in INVARIANTS])
all5 = votes.sum(axis=1) >= 3
nokh = votes[:, :4].sum(axis=1) >= 3

# ---------------------------------------------------------------------
# Rebuild aligned matrices and continuous norms.
# ---------------------------------------------------------------------
config = ch.canonical_run_config()
FILE_MAP = {
    "alex": "Alexander_upto17.csv",
    "homfly": "HomflyPt_upto15_MIRRORS.csv",
    "jones": "Jones_upto17_MIRRORS.csv",
    "theta": "theta_upto15.csv",
    "kh": "even_KH_upto17.pkl",
}
REPRESENTATION_SPECS = {
    "Alexander": {"source": "alex", "feature_prefixes": ["A"]},
    "Jones": {"source": "jones", "feature_prefixes": ["J"]},
    "HOMFLY-PT": {"source": "homfly", "feature_prefixes": ["a"]},
    "Theta": {"source": "theta", "feature_prefixes": ["T"]},
    "Khovanov": {"source": "kh", "feature_prefixes": ["F_"]},
}

aligned = ch.build_aligned_dataset(
    base_dir=DATA_DIR,
    file_map=FILE_MAP,
    representation_specs=REPRESENTATION_SPECS,
    min_crossings=config.universe.min_crossings,
    max_crossings=config.universe.max_crossings,
    output_dir=None,
    preferred_metadata_sources=["alex", "jones", "homfly", "theta", "kh"],
    expected_n=config.universe.expected_n,
    expected_s_qc_corrections=config.universe.expected_s_qc_corrections,
)
meta2 = aligned["meta"].copy()
meta2[ID_COL] = meta2[ID_COL].astype(str)

# Reorder rebuilt matrices to frozen atlas order if necessary.
pos2 = pd.Series(np.arange(len(meta2), dtype=np.int64), index=meta2[ID_COL])
take = pos2.loc[atlas[ID_COL]].to_numpy()
if len(np.unique(take)) != N:
    raise RuntimeError("Could not uniquely align rebuilt matrices to frozen atlas.")

continuous_norms = {}
norm_audit_rows = []
with np.load(find_one("conditional_100bins_scores.npz"), allow_pickle=False) as payload:
    saved_bins = {
        name: np.asarray(payload[f"{safe_name(name)}_norm_bin"], dtype=np.int32)
        for name in INVARIANTS
    }

for name in INVARIANTS:
    X = np.asarray(aligned["X_dict"][name])[take]
    Z = StandardScaler().fit_transform(X)
    lognorm = np.log1p(np.sum(Z.astype(np.float64) ** 2, axis=1))
    continuous_norms[name] = lognorm

    # The saved 100-bin index should be nearly perfectly monotone in the
    # reconstructed continuous norm. This is an audit, not an equality check.
    rho = pd.Series(lognorm).corr(pd.Series(saved_bins[name]), method="spearman")
    norm_audit_rows.append({
        "view": name,
        "continuous_norm_vs_saved_100bin_spearman": float(rho),
        "lognorm_min": float(np.min(lognorm)),
        "lognorm_median": float(np.median(lognorm)),
        "lognorm_max": float(np.max(lognorm)),
    })

norm_audit = pd.DataFrame(norm_audit_rows)
norm_audit.to_csv(OUT / "continuous_norm_reconstruction_audit.csv", index=False)

if (norm_audit["continuous_norm_vs_saved_100bin_spearman"] < 0.995).any():
    raise RuntimeError(
        "Reconstructed continuous norms are not sufficiently aligned with the "
        "saved 100-bin norm indices. Inspect continuous_norm_reconstruction_audit.csv."
    )

# ---------------------------------------------------------------------
# Cross-fitted flexible residualization.
# ---------------------------------------------------------------------
norm_feature_names = [f"lognorm_{safe_name(v)}" for v in INVARIANTS]
Xnorm = np.column_stack([continuous_norms[v] for v in INVARIANTS])

def cross_fitted_residuals(X, y, *, seed):
    """5-fold cross-fitted flexible regression residuals."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    pred = np.full(len(y), np.nan, dtype=float)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr, te) in enumerate(kf.split(X)):
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.06,
            max_iter=220,
            max_leaf_nodes=31,
            min_samples_leaf=80,
            l2_regularization=1.0,
            random_state=seed + fold,
        )
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])

    if not np.isfinite(pred).all():
        raise RuntimeError("Cross-fitted predictions contain missing values.")
    return y - pred, pred

def factorize(frame):
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(frame.astype(str)), sort=False)
    return codes.astype(np.int32)

def prepare_sampler(codes, selected):
    order = np.argsort(codes, kind="stable")
    c = codes[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(c))]
    stops = np.r_[starts[1:], len(order)]
    random_groups, fixed_groups = [], []
    for a, b in zip(starts, stops):
        idx = order[a:b]
        k = int(selected[idx].sum())
        if k == 0:
            continue
        if k == len(idx):
            fixed_groups.append(idx)
        else:
            random_groups.append((idx, k))
    fixed = np.concatenate(fixed_groups) if fixed_groups else np.empty(0, dtype=int)
    return {"random_groups": random_groups, "fixed": fixed}

def sample_mask(n, sampler, rng):
    out = np.zeros(n, dtype=bool)
    out[sampler["fixed"]] = True
    for idx, k in sampler["random_groups"]:
        out[rng.choice(idx, size=k, replace=False)] = True
    return out

def residualized_test(label, y, selected, structure_df, extra_numeric_features=None):
    # Outcome model gets continuous norms plus structural covariates.
    # Structural covariates are also retained exactly by the permutation.
    structural_numeric = structure_df.astype(float).to_numpy()
    features = [Xnorm]
    if structural_numeric.shape[1]:
        features.append(structural_numeric)
    if extra_numeric_features is not None:
        features.append(np.asarray(extra_numeric_features, dtype=float))
    X = np.column_stack(features)

    resid, pred = cross_fitted_residuals(X, y, seed=SEED + deterministic_offset(label))

    codes = factorize(structure_df)
    sampler = prepare_sampler(codes, selected)

    obs = float(np.mean(resid[selected]))
    rng = np.random.default_rng(SEED + deterministic_offset(label + "_perm"))
    null = np.empty(B, dtype=float)
    for b in range(B):
        m = sample_mask(len(selected), sampler, rng)
        null[b] = float(np.mean(resid[m]))

    p = float((1 + np.sum(null >= obs)) / (B + 1))
    out = {
        "analysis": label,
        "n": len(y),
        "selected_n": int(selected.sum()),
        "observed_mean_raw": float(np.mean(y[selected])),
        "observed_mean_crossfit_residual": obs,
        "null_mean_residual": float(np.mean(null)),
        "null_sd_residual": float(np.std(null, ddof=1)),
        "null_q025_residual": float(np.quantile(null, .025)),
        "null_q975_residual": float(np.quantile(null, .975)),
        "empirical_p_upper": p,
        "crossfit_rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "B": B,
    }
    return out, pd.DataFrame({"null_stat": null})

# A) No-Khovanov mean diagonal count
structure_nokh = pd.DataFrame({
    "crossing": atlas["number_of_crossings"].to_numpy(int),
    "alternating": atlas["is_alternating"].to_numpy(int),
    "abs_sigma": atlas["signature"].abs().to_numpy(int),
})
res_nokh, null_nokh = residualized_test(
    "noKh_mean_diag_continuous_norm",
    KH_DIAG,
    nokh,
    structure_nokh,
)

# B/C) G endpoints among nonalternating only, exact diagonal count retained.
idx_nonalt = np.flatnonzero(atlas["is_alternating"].to_numpy(int) == 0)
Xnorm_full = Xnorm
Xnorm = Xnorm_full[idx_nonalt]  # temporarily subset global used by helper
structure_g = pd.DataFrame({
    "crossing": atlas.iloc[idx_nonalt]["number_of_crossings"].to_numpy(int),
    "abs_sigma": atlas.iloc[idx_nonalt]["signature"].abs().to_numpy(int),
    "kh_diag_exact": KH_DIAG[idx_nonalt].astype(int),
})
sel_g = all5[idx_nonalt]
G_nonalt = G[idx_nonalt]

res_g_inc, null_g_inc = residualized_test(
    "all5_P_G_gt0_continuous_norm",
    (G_nonalt > 0).astype(float),
    sel_g,
    structure_g,
)
res_g_mean, null_g_mean = residualized_test(
    "all5_mean_G_continuous_norm",
    G_nonalt,
    sel_g,
    structure_g,
)
Xnorm = Xnorm_full

results = pd.DataFrame([res_nokh, res_g_inc, res_g_mean])
results.to_csv(OUT / "continuous_amplitude_residualized_results.csv", index=False)
null_nokh.to_parquet(OUT / "null_noKh_mean_diag.parquet", index=False)
null_g_inc.to_parquet(OUT / "null_all5_P_G_gt0.parquet", index=False)
null_g_mean.to_parquet(OUT / "null_all5_mean_G.parquet", index=False)

print("\nContinuous-norm reconstruction audit:")
print(norm_audit.to_string(index=False))
print("\nCross-fitted continuous-amplitude residualized sensitivity:")
print(results.to_string(index=False))
print("\nSaved Stage 29c to:", OUT)
