# %% [markdown]
# Stage 31 — Crossing-number extrapolation: train <=13, calibrate 14, test 15
#
# Reviewer-requested out-of-range sensitivity.
#
# Design:
#   train: knots with <=13 crossings
#   calibration: 14-crossing knots
#   test: 15-crossing knots
#
# All scalers and PCA subspaces are fitted ONLY on train.
# k99 is selected from TRAIN only (no 15-crossing leakage).
# 14-crossing scores calibrate norm-conditioned percentiles.
# The 0.99 percentile threshold is frozen and applied once to 15 crossings.
#
# Run in the SAME runtime as paper_run.ipynb because X_dict and
# feature_cols_dict are required.

from __future__ import annotations
from pathlib import Path
import os, re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

REQUIRED = ("meta", "X_dict", "feature_cols_dict", "CONFIG", "OUTPUT_DIR")
missing = [name for name in REQUIRED if name not in globals()]
if missing:
    raise RuntimeError("Run paper_run.ipynb first. Missing: " + str(missing))

OUT = Path(OUTPUT_DIR) / "31_crossing_number_extrapolation"
OUT.mkdir(parents=True, exist_ok=True)

INVARIANTS = ("Alexander", "Jones", "HOMFLY-PT", "Theta", "Khovanov")
NO_KHOVANOV = tuple(x for x in INVARIANTS if x != "Khovanov")
ID_COL = CONFIG.universe.id_col
S_COL = CONFIG.s_col if CONFIG.s_col in meta else "s_invariant"
NORM_BINS = int(os.environ.get("STAGE31_NORM_BINS", "100"))
PCT_THRESHOLD = float(os.environ.get("STAGE31_PERCENTILE_THRESHOLD", "0.99"))
NULL_REPS = int(os.environ.get("STAGE31_NULL_REPS", "2000"))
SEED = int(os.environ.get("STAGE31_SEED", "20260831"))

cross = meta["number_of_crossings"].to_numpy(int)
train_idx = np.flatnonzero(cross <= 13)
cal_idx = np.flatnonzero(cross == 14)
test_idx = np.flatnonzero(cross == 15)

split_summary = pd.DataFrame([
    {"split": "train_le13", "n": len(train_idx)},
    {"split": "calibration_14", "n": len(cal_idx)},
    {"split": "test_15", "n": len(test_idx)},
])
split_summary.to_csv(OUT / "crossing_split_counts.csv", index=False)

def quantile_edges(values, n_bins):
    edges = np.quantile(np.asarray(values, float), np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges

def assign_bins(values, edges):
    return np.searchsorted(edges[1:-1], values, side="right").astype(np.int32)

def calibrated_percentiles(cal_score, cal_norm, test_score, test_norm, n_bins):
    edges = quantile_edges(cal_norm, n_bins)
    cb = assign_bins(cal_norm, edges)
    tb = assign_bins(test_norm, edges)
    out = np.empty(len(test_score), float)
    populated = np.unique(cb)
    for b in np.unique(tb):
        ref = np.sort(cal_score[cb == b])
        if len(ref) == 0:
            nearest = populated[np.argmin(np.abs(populated - b))]
            ref = np.sort(cal_score[cb == nearest])
        where = tb == b
        out[where] = np.searchsorted(ref, test_score[where], side="right") / (len(ref) + 1.0)
    return out, tb, edges

view_results = {}
view_rows = []
test_view_masks = {}
test_norm_bins = {}
test_log_norm = {}

for name in INVARIANTS:
    X = np.asarray(X_dict[name])
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[train_idx]).astype(np.float32)
    Xcal = scaler.transform(X[cal_idx]).astype(np.float32)
    Xte = scaler.transform(X[test_idx]).astype(np.float32)

    # TRAIN-ONLY k99.
    pca = PCA(n_components=0.99, svd_solver="full")
    pca.fit(Xtr)
    k99 = int(pca.n_components_)

    def sse(Z):
        Zhat = pca.inverse_transform(pca.transform(Z))
        return np.sum((Z - Zhat) ** 2, axis=1)

    cal_sse = sse(Xcal)
    test_sse = sse(Xte)
    cal_norm = np.log1p(np.sum(Xcal.astype(np.float64) ** 2, axis=1))
    test_norm = np.log1p(np.sum(Xte.astype(np.float64) ** 2, axis=1))

    pct, bins, edges = calibrated_percentiles(
        cal_sse, cal_norm, test_sse, test_norm, NORM_BINS
    )
    selected_local = pct >= PCT_THRESHOLD
    test_view_masks[name] = selected_local
    test_norm_bins[name] = bins
    test_log_norm[name] = test_norm

    view_results[name] = {
        "scaler": scaler,
        "pca": pca,
        "k99_train": k99,
        "cal_sse": cal_sse,
        "test_sse": test_sse,
        "test_percentile": pct,
        "test_norm_bin": bins,
        "test_log_norm": test_norm,
    }
    view_rows.append({
        "invariant": name,
        "input_dim": X.shape[1],
        "train_k99": k99,
        "train_evr": float(pca.explained_variance_ratio_.sum()),
        "n_train": len(train_idx),
        "n_calibration": len(cal_idx),
        "n_test": len(test_idx),
        "n_test_selected": int(selected_local.sum()),
        "test_selected_prop": float(selected_local.mean()),
        "spearman_test_percentile_vs_log_norm": float(
            spearmanr(pct, test_norm).statistic
        ),
    })

view_summary = pd.DataFrame(view_rows)
view_summary.to_csv(OUT / "crossing_split_view_summary.csv", index=False)

def family_mask(names, k=3):
    count = np.zeros(len(test_idx), dtype=np.uint8)
    for name in names:
        count += test_view_masks[name]
    return count >= k

all5 = family_mask(INVARIANTS, 3)
nokh = family_mask(NO_KHOVANOV, 3)

# ---------------------------------------------------------------------------
# Test-only external outcomes
# ---------------------------------------------------------------------------
s = meta.iloc[test_idx][S_COL].to_numpy(float)
sigma = meta.iloc[test_idx]["signature"].to_numpy(float)
G_diff = np.abs(s - sigma)
G_absabs = np.abs(np.abs(s) - np.abs(sigma))
nonalt = meta.iloc[test_idx]["is_alternating"].to_numpy(int) == 0

# Khovanov F_ support and q-2t diagonal count on test only.
kh_cols = [str(c) for c in feature_cols_dict["Khovanov"]]
coord_re = re.compile(r"F_q(-?\d+)_t(-?\d+)$")
coords = []
for col in kh_cols:
    m = coord_re.fullmatch(col)
    if m is None:
        raise ValueError(f"Unparsed Khovanov coordinate: {col}")
    coords.append(tuple(map(int, m.groups())))
delta = np.asarray([q - 2*t for q, t in coords], dtype=np.int32)
groups = [np.flatnonzero(delta == d) for d in np.unique(delta)]
Xkh_test = np.asarray(X_dict["Khovanov"])[test_idx]
active = np.abs(Xkh_test) > 1e-12
kh_support = active.sum(axis=1)
kh_diag = np.zeros(len(test_idx), dtype=np.int16)
for g in groups:
    kh_diag += active[:, g].any(axis=1)

def summary(mask, label):
    target = mask & nonalt
    gv = G_diff[target]
    ga = G_absabs[target]
    d = kh_diag[mask]
    return {
        "family": label,
        "n": int(mask.sum()),
        "nonalternating_n": int(target.sum()),
        "alternating_prop": float(np.mean(meta.iloc[test_idx[mask]]["is_alternating"])) if mask.any() else np.nan,
        "P_Gdiff_gt0": float(np.mean(gv > 0)) if len(gv) else np.nan,
        "mean_Gdiff": float(np.mean(gv)) if len(gv) else np.nan,
        "P_Gabsabs_gt0": float(np.mean(ga > 0)) if len(ga) else np.nan,
        "mean_Gabsabs": float(np.mean(ga)) if len(ga) else np.nan,
        "mean_kh_diag": float(np.mean(d)) if len(d) else np.nan,
        "P_kh_diag_ge3": float(np.mean(d >= 3)) if len(d) else np.nan,
        "P_kh_diag_ge4": float(np.mean(d >= 4)) if len(d) else np.nan,
        "mean_kh_support": float(np.mean(kh_support[mask])) if mask.any() else np.nan,
    }

family_summary = pd.DataFrame([
    summary(all5, "all5_ge3"),
    summary(nokh, "no_khovanov_ge3of4"),
])
family_summary.to_csv(OUT / "crossing_split_family_summary.csv", index=False)

# Save members.
members = meta.iloc[test_idx][[
    ID_COL, "number_of_crossings", "is_alternating", "signature", S_COL
]].copy()
members["all5_ge3"] = all5
members["no_khovanov_ge3of4"] = nokh
members["G_abs_s_minus_sigma"] = G_diff
members["G_abs_abs_s_minus_abs_sigma"] = G_absabs
members["khovanov_F_support_size"] = kh_support
members["khovanov_F_q_minus_2t_diagonal_count"] = kh_diag
members.to_csv(OUT / "crossing15_selected_members.csv", index=False)

# ---------------------------------------------------------------------------
# Fixed-selection, coarse joint-amplitude null on the 15-crossing test set.
#
# This is a ROBUSTNESS null, not a replacement for the primary paper null.
# It fixes consensus cardinality and avoids the cross-view-vote issue.
# We use 2 bins per relevant norm dimension to retain nontrivial amplitude
# control without making the joint cells unusably sparse.
# ---------------------------------------------------------------------------
def coarsen(codes, target=2):
    codes = np.asarray(codes)
    u = np.unique(codes)
    r = np.searchsorted(u, codes)
    out = np.floor(r * min(target, len(u)) / len(u)).astype(int)
    return np.minimum(out, min(target, len(u)) - 1)

def factorize(frame):
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(frame.astype(str)), sort=False)
    return codes.astype(np.int32)

def prepare(codes, selected):
    order = np.argsort(codes, kind="stable")
    c = codes[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(c))]
    stops = np.r_[starts[1:], len(order)]
    rg, fg, sizes = [], [], []
    for a,b in zip(starts,stops):
        idx = order[a:b]
        k = int(selected[idx].sum())
        sizes.append(len(idx))
        if k == 0:
            continue
        if k == len(idx):
            fg.append(idx)
        else:
            rg.append((idx,k))
    fixed = np.concatenate(fg) if fg else np.empty(0, dtype=int)
    sizes = np.asarray(sizes)
    return {"random_groups": rg, "fixed_idx": fixed}, {
        "n_strata": len(sizes),
        "median_cell": float(np.median(sizes)),
        "q25_cell": float(np.quantile(sizes,.25)),
        "singleton_prop": float(np.mean(sizes==1)),
        "selected_fixed": len(fixed),
        "selected_movable": int(sum(k for _,k in rg)),
    }

def sample(sampler, rng):
    m = np.zeros(len(test_idx), dtype=bool)
    m[sampler["fixed_idx"]] = True
    for idx,k in sampler["random_groups"]:
        m[rng.choice(idx, size=k, replace=False)] = True
    return m

base = pd.DataFrame({
    "alternating": meta.iloc[test_idx]["is_alternating"].to_numpy(),
    "abs_sigma": np.abs(sigma),
})
for name in INVARIANTS:
    base[f"{name}_norm2"] = coarsen(test_norm_bins[name], 2)

# Gap beyond thickness: exact F-diagonal count.
gap_strata = base.copy()
gap_strata["kh_diag_exact"] = kh_diag
gap_codes = factorize(gap_strata)
gap_sampler, gap_diag = prepare(gap_codes, all5)

# External Khovanov thickness: include coarse Khovanov norm.
nokh_strata = base[[c for c in base.columns if c not in ("Khovanov_norm2",)]].copy()
nokh_strata["khovanov_norm2"] = coarsen(test_norm_bins["Khovanov"], 2)
nokh_codes = factorize(nokh_strata)
nokh_sampler, nokh_diag = prepare(nokh_codes, nokh)

rng = np.random.default_rng(SEED)
gap_rows, kh_rows = [], []
for rep in range(NULL_REPS):
    mg = sample(gap_sampler, rng)
    mn = sample(nokh_sampler, rng)
    gap_rows.append(summary(mg, "gap_null"))
    kh_rows.append(summary(mn, "kh_null"))

gap_null = pd.DataFrame(gap_rows)
kh_null = pd.DataFrame(kh_rows)
gap_null.to_parquet(OUT / "crossing15_gap_fixed_selection_null.parquet", index=False)
kh_null.to_parquet(OUT / "crossing15_noKh_fixed_selection_null.parquet", index=False)

def upper_p(vals, obs):
    vals = np.asarray(vals, float)
    return float((1 + np.sum(vals >= obs)) / (1 + len(vals)))

obs_all = summary(all5, "all5")
obs_nokh = summary(nokh, "nokh")
null_summary = pd.DataFrame([
    {
        "analysis": "all5_G_incidence_with_exact_thickness",
        "metric": "P_Gdiff_gt0",
        "observed": obs_all["P_Gdiff_gt0"],
        "null_mean": float(gap_null["P_Gdiff_gt0"].mean()),
        "empirical_p_upper": upper_p(gap_null["P_Gdiff_gt0"], obs_all["P_Gdiff_gt0"]),
        **{f"strata_{k}": v for k,v in gap_diag.items()},
    },
    {
        "analysis": "all5_mean_G_with_exact_thickness",
        "metric": "mean_Gdiff",
        "observed": obs_all["mean_Gdiff"],
        "null_mean": float(gap_null["mean_Gdiff"].mean()),
        "empirical_p_upper": upper_p(gap_null["mean_Gdiff"], obs_all["mean_Gdiff"]),
        **{f"strata_{k}": v for k,v in gap_diag.items()},
    },
    {
        "analysis": "noKh_external_mean_diagonal",
        "metric": "mean_kh_diag",
        "observed": obs_nokh["mean_kh_diag"],
        "null_mean": float(kh_null["mean_kh_diag"].mean()),
        "empirical_p_upper": upper_p(kh_null["mean_kh_diag"], obs_nokh["mean_kh_diag"]),
        **{f"strata_{k}": v for k,v in nokh_diag.items()},
    },
    {
        "analysis": "noKh_external_P_diag_ge3",
        "metric": "P_kh_diag_ge3",
        "observed": obs_nokh["P_kh_diag_ge3"],
        "null_mean": float(kh_null["P_kh_diag_ge3"].mean()),
        "empirical_p_upper": upper_p(kh_null["P_kh_diag_ge3"], obs_nokh["P_kh_diag_ge3"]),
        **{f"strata_{k}": v for k,v in nokh_diag.items()},
    },
])
null_summary.to_csv(OUT / "crossing15_fixed_selection_null_summary.csv", index=False)

print("\nCrossing split:")
print(split_summary.to_string(index=False))
print("\nTrain-derived PCA / 15-crossing selection:")
print(view_summary.to_string(index=False))
print("\n15-crossing family phenotypes:")
print(family_summary.to_string(index=False))
print("\n15-crossing fixed-selection null sensitivity:")
print(null_summary.to_string(index=False))
print("\nSaved Stage 31 to:", OUT)
