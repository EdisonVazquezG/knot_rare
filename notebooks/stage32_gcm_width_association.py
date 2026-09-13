# %% [markdown]
# Stage 32 — Generalized-covariance-style test for withheld Khovanov width
#
# Revision question:
#   Does the no-Khovanov selection remain associated with stored Khovanov width
#   after adjusting continuously for representation amplitudes and structural
#   variables WITHOUT permuting residuals across different norm values?
#
# Motivation:
#   Cross-fitting alone does not make regression residuals exchangeable.  This
#   stage therefore replaces the earlier residual-permutation p-value with a
#   generalized-covariance-style statistic:
#
#       psi_i = (Y_i - m_hat(Z_i)) (H_i - p_hat(Z_i)),
#
#   where Y is stored F_-part Khovanov diagonal count, H is the frozen 3-of-4
#   no-Khovanov selection indicator, and Z contains all five continuous log
#   squared norms plus crossing number, alternation, and |signature|.
#
#   Both nuisance functions are cross-fitted.  Inference is based on the
#   studentized mean of psi and a multiplier (wild) bootstrap of the centered
#   influence values; selected labels are NOT permuted among residuals.
#
# IMPORTANT:
#   This is a candidate calibrated sensitivity analysis, not an automatic proof
#   that every GCM assumption holds in this atlas.  The stage therefore includes
#   null calibration stress tests with uneven selection probabilities,
#   heteroskedastic outcomes, and tail-concentrated regression difficulty.
#
# Fresh-session usage:
#   %run /content/drive/MyDrive/consensus_hardness_refactored/notebooks/\
#       stage32_gcm_width_association.py
#
# Environment overrides:
#   KNOT_PROJECT_DIR, KNOT_DATA_DIR, KNOT_OUTPUT_DIR
#   STAGE32_FOLDS=5
#   STAGE32_BOOT_REPS=5000
#   STAGE32_SEED=202609032
#   STAGE32_ORACLE_SIM_REPS=500
#   STAGE32_EST_SIM_REPS=30
#   STAGE32_SIM_N=12000

# %%
from __future__ import annotations

import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
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
OUT = ROOT / "32_gcm_width_association"


def maybe_mount_drive() -> None:
    if Path("/content").exists() and not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive  # type: ignore
            print("Google Drive is not mounted; requesting mount...")
            drive.mount("/content/drive")
        except Exception as exc:
            print("WARNING: could not mount Google Drive automatically:", exc)


maybe_mount_drive()
if not DATA_DIR.exists() or not ROOT.exists():
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
    from consensus_hardness.representations import split_metadata_and_features
except Exception as exc:
    raise ImportError(f"Could not import consensus_hardness from {PROJECT_DIR}") from exc

ID_COL = "knot_id_base"
INVARIANTS = ("Alexander", "Jones", "HOMFLY-PT", "Theta", "Khovanov")
NO_KHOVANOV = ("Alexander", "Jones", "HOMFLY-PT", "Theta")
VIEW_SPECS = {
    "Alexander": ("Alexander_upto17.csv", ["A"]),
    "Jones": ("Jones_upto17_MIRRORS.csv", ["J"]),
    "HOMFLY-PT": ("HomflyPt_upto15_MIRRORS.csv", ["a"]),
    "Theta": ("theta_upto15.csv", ["T"]),
    "Khovanov": ("even_KH_upto17.pkl", ["F_"]),
}
N_FOLDS = int(os.environ.get("STAGE32_FOLDS", "5"))
BOOT_REPS = int(os.environ.get("STAGE32_BOOT_REPS", "5000"))
SEED = int(os.environ.get("STAGE32_SEED", "202609032"))
ORACLE_SIM_REPS = int(os.environ.get("STAGE32_ORACLE_SIM_REPS", "500"))
EST_SIM_REPS = int(os.environ.get("STAGE32_EST_SIM_REPS", "30"))
SIM_N = int(os.environ.get("STAGE32_SIM_N", "12000"))


def find_one(filename: str) -> Path:
    hits = sorted(ROOT.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename} below {ROOT}")
    return hits[0]


# ---------------------------------------------------------------------------
# 1. Frozen atlas, selection, and outcome
# ---------------------------------------------------------------------------
atlas = pd.read_parquet(find_one("final_hard_regime_atlas.parquet")).reset_index(drop=True)
phenotype = pd.read_parquet(find_one("complete_mathematical_phenotype_atlas.parquet")).reset_index(drop=True)
for label, frame in (("atlas", atlas), ("phenotype", phenotype)):
    if ID_COL not in frame:
        raise KeyError(f"{label} lacks {ID_COL}")
    frame[ID_COL] = frame[ID_COL].astype(str)
if not atlas[ID_COL].equals(phenotype[ID_COL]):
    phenotype = atlas[[ID_COL]].merge(phenotype, on=ID_COL, how="left", validate="one_to_one")
N = len(atlas)
frozen_ids = atlas[ID_COL].astype(str).to_numpy()

kh_diag_col = next(
    c for c in (
        "khovanov_q_minus_2t_diagonal_count",
        "kh_diagonal_count",
        "khovanov_diagonal_count",
    ) if c in phenotype
)
Y = phenotype[kh_diag_col].to_numpy(float)
if not np.isfinite(Y).all():
    raise RuntimeError("Non-finite Khovanov width endpoint")

hard_saved = pd.read_csv(find_one("conditional_100bins_hard_sets.csv"), dtype={ID_COL: str})
id_to_pos = pd.Series(np.arange(N, dtype=np.int64), index=frozen_ids).to_dict()
votes = np.zeros(N, dtype=np.uint8)
for name in NO_KHOVANOV:
    ids = hard_saved.loc[hard_saved["invariant"].eq(name), ID_COL].astype(str)
    m = np.zeros(N, dtype=bool)
    m[[id_to_pos[x] for x in ids]] = True
    votes += m.astype(np.uint8)
H = (votes >= 3).astype(np.int8)
if int(H.sum()) != 220:
    raise RuntimeError(f"Frozen no-Khovanov selection n={int(H.sum())}, expected 220")
print(f"Frozen selection: {int(H.sum())}/{N} ({H.mean():.6g})")
print(f"Observed selected mean stored diagonal count: {Y[H == 1].mean():.6f}")

# ---------------------------------------------------------------------------
# 2. Build/reuse continuous log-norm adjustment design one view at a time
# ---------------------------------------------------------------------------
DESIGN_PATH = OUT / "continuous_adjustment_design.npz"
if DESIGN_PATH.exists():
    print("Loading continuous-norm design checkpoint:", DESIGN_PATH)
    with np.load(DESIGN_PATH, allow_pickle=False) as p:
        Xnorm = np.asarray(p["Xnorm"], dtype=np.float64)
        ids_ckpt = np.asarray(p["ids"]).astype(str)
    if not np.array_equal(ids_ckpt, frozen_ids):
        raise RuntimeError("Continuous-norm checkpoint IDs do not match frozen atlas")
else:
    norm_cols = []
    norm_audit = []
    for name in INVARIANTS:
        filename, prefixes = VIEW_SPECS[name]
        print(f"Reconstructing continuous standardized norm: {name}")
        raw = load_table(DATA_DIR / filename)
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
        missing = [x for x in frozen_ids if x not in idx.index]
        if missing:
            raise RuntimeError(f"{name}: missing frozen IDs, e.g. {missing[:5]}")
        rows = idx.loc[frozen_ids].reset_index(drop=True)
        _, ids, X, feat_cols = split_metadata_and_features(rows, prefixes, id_col=ID_COL)
        if not np.array_equal(ids.astype(str), frozen_ids):
            raise RuntimeError(f"{name}: representation alignment failure")

        scaler = StandardScaler(copy=True)
        scaler.fit(X)
        lognorm = np.empty(N, dtype=np.float64)
        batch = 8192
        for start in range(0, N, batch):
            stop = min(start + batch, N)
            z = scaler.transform(X[start:stop]).astype(np.float64)
            lognorm[start:stop] = np.log1p(np.sum(z * z, axis=1))
        norm_cols.append(lognorm)
        norm_audit.append({
            "view": name,
            "input_dim": int(X.shape[1]),
            "lognorm_min": float(lognorm.min()),
            "lognorm_median": float(np.median(lognorm)),
            "lognorm_max": float(lognorm.max()),
        })
        del raw, clean, canonical, rows, X, scaler, lognorm
        gc.collect()

    Xnorm = np.column_stack(norm_cols).astype(np.float64)
    np.savez_compressed(DESIGN_PATH, Xnorm=Xnorm, ids=frozen_ids)
    pd.DataFrame(norm_audit).to_csv(OUT / "continuous_norm_reconstruction_audit.csv", index=False)

if Xnorm.shape != (N, 5) or not np.isfinite(Xnorm).all():
    raise RuntimeError(f"Unexpected continuous norm design shape {Xnorm.shape}")

Z = np.column_stack([
    Xnorm,
    atlas["number_of_crossings"].to_numpy(float),
    atlas["is_alternating"].to_numpy(float),
    atlas["signature"].abs().to_numpy(float),
])
feature_names = [
    "lognorm_Alexander", "lognorm_Jones", "lognorm_HOMFLY_PT",
    "lognorm_Theta", "lognorm_Khovanov", "crossing", "alternating", "abs_signature",
]
pd.DataFrame({"feature": feature_names}).to_csv(OUT / "adjustment_variables.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Cross-fitted nuisance regressions
# ---------------------------------------------------------------------------
def make_y_model(seed: int, *, fast: bool = False):
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=120 if fast else 260,
        max_leaf_nodes=15 if fast else 31,
        min_samples_leaf=60 if fast else 100,
        l2_regularization=1.0,
        random_state=seed,
    )


def make_h_model(seed: int, *, fast: bool = False):
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=140 if fast else 300,
        max_leaf_nodes=15 if fast else 31,
        min_samples_leaf=40 if fast else 80,
        l2_regularization=1.0,
        random_state=seed,
    )


def crossfit_nuisances(Z, y, h, *, n_folds: int, seed: int, fast: bool = False):
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    h = np.asarray(h, dtype=int)
    my = np.full(len(y), np.nan, dtype=float)
    ph = np.full(len(y), np.nan, dtype=float)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(splitter.split(Z, h)):
        ym = make_y_model(seed + 100 * fold, fast=fast)
        hm = make_h_model(seed + 100 * fold + 1, fast=fast)
        ym.fit(Z[tr], y[tr])
        hm.fit(Z[tr], h[tr])
        my[te] = ym.predict(Z[te])
        ph[te] = hm.predict_proba(Z[te])[:, 1]
    if not np.isfinite(my).all() or not np.isfinite(ph).all():
        raise RuntimeError("Cross-fitted nuisance predictions contain non-finite values")
    ph = np.clip(ph, 1e-8, 1 - 1e-8)
    return my, ph


def gcm_from_predictions(y, h, my, ph):
    ry = np.asarray(y, dtype=float) - np.asarray(my, dtype=float)
    rh = np.asarray(h, dtype=float) - np.asarray(ph, dtype=float)
    psi = ry * rh
    sd = float(np.std(psi, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError("Degenerate GCM influence variance")
    stat = float(np.sqrt(len(psi)) * np.mean(psi) / sd)
    p_normal = float(norm.sf(stat))
    return psi, stat, p_normal, ry, rh


PRED_PATH = OUT / "crossfit_nuisance_predictions.npz"
if PRED_PATH.exists():
    print("Loading cross-fit prediction checkpoint:", PRED_PATH)
    with np.load(PRED_PATH, allow_pickle=False) as p:
        mY = np.asarray(p["mY"], dtype=float)
        pH = np.asarray(p["pH"], dtype=float)
else:
    print("Fitting cross-fitted E[Y|Z] and E[H|Z]...")
    mY, pH = crossfit_nuisances(Z, Y, H, n_folds=N_FOLDS, seed=SEED)
    np.savez_compressed(PRED_PATH, mY=mY, pH=pH)

psi, stat, p_normal, rY, rH = gcm_from_predictions(Y, H, mY, pH)

# ---------------------------------------------------------------------------
# 4. Multiplier bootstrap: no residual-label permutation
# ---------------------------------------------------------------------------
def multiplier_pvalue(psi: np.ndarray, observed_stat: float, reps: int, seed: int):
    centered = np.asarray(psi, dtype=float) - float(np.mean(psi))
    sd = float(np.std(psi, ddof=1))
    n = len(psi)
    rng = np.random.default_rng(seed)
    null_stats = np.empty(reps, dtype=np.float64)
    block_reps = 32
    pos = 0
    while pos < reps:
        b = min(block_reps, reps - pos)
        # Rademacher multipliers preserve observation-specific heteroskedasticity
        # in the first-order influence values while imposing a zero-mean null.
        xi = rng.integers(0, 2, size=(b, n), dtype=np.int8)
        xi = (2 * xi - 1).astype(np.float64)
        null_stats[pos:pos+b] = (xi @ centered) / (np.sqrt(n) * sd)
        pos += b
        if pos % 320 == 0 or pos == reps:
            print(f"Multiplier bootstrap: {pos:,}/{reps:,}")
    p = float((1 + np.sum(null_stats >= observed_stat)) / (reps + 1))
    return p, null_stats


NULL_PATH = OUT / f"gcm_multiplier_null_{BOOT_REPS}.npz"
if NULL_PATH.exists():
    with np.load(NULL_PATH, allow_pickle=False) as p:
        null_stats = np.asarray(p["null_stat"], dtype=float)
    p_mult = float((1 + np.sum(null_stats >= stat)) / (len(null_stats) + 1))
else:
    p_mult, null_stats = multiplier_pvalue(psi, stat, BOOT_REPS, SEED + 77)
    np.savez_compressed(NULL_PATH, null_stat=null_stats)

main_result = pd.DataFrame([{
    "analysis": "noKh_mean_stored_diagonal_GCM",
    "n": N,
    "selected_n": int(H.sum()),
    "selected_mean_raw": float(Y[H == 1].mean()),
    "background_mean_raw": float(Y[H == 0].mean()),
    "mean_product_residual": float(np.mean(psi)),
    "sd_product_residual": float(np.std(psi, ddof=1)),
    "studentized_gcm_stat": stat,
    "one_sided_normal_p": p_normal,
    "one_sided_multiplier_p": p_mult,
    "multiplier_reps": int(len(null_stats)),
    "outcome_crossfit_rmse": float(np.sqrt(np.mean(rY ** 2))),
    "selection_brier": float(np.mean((H - pH) ** 2)),
    "mean_predicted_selection_probability": float(np.mean(pH)),
    "max_predicted_selection_probability": float(np.max(pH)),
}])
main_result.to_csv(OUT / "gcm_width_result.csv", index=False)

# ---------------------------------------------------------------------------
# 5. Calibration stress tests under conditional independence
# ---------------------------------------------------------------------------
def wilson_interval(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return np.nan, np.nan
    z = float(norm.ppf(1 - alpha / 2))
    phat = k / n
    den = 1 + z*z/n
    center = (phat + z*z/(2*n)) / den
    half = z * math.sqrt(phat*(1-phat)/n + z*z/(4*n*n)) / den
    return center - half, center + half


def simulate_null_data(n: int, rng: np.random.Generator, scenario: str):
    z = rng.normal(size=(n, 4))
    tail = np.maximum(z[:, 0] - 1.0, 0.0)
    # Strongly uneven but nondegenerate selection probability.
    lin_h = -3.2 + 1.4*z[:,0] - 0.7*z[:,1] + 0.5*z[:,2]*z[:,2]
    p = expit(lin_h)
    h = rng.binomial(1, p, size=n)
    if h.sum() < 10:
        # Extremely unlikely at default n; fail explicitly rather than hide it.
        raise RuntimeError("Synthetic selection produced too few positives")
    m = 2.2 + 0.35*z[:,0] - 0.25*z[:,1] + 0.15*z[:,2]**2
    if scenario == "uneven_propensity":
        sigma = np.ones(n)
    elif scenario == "heteroskedastic":
        sigma = 0.55 + 0.65*expit(1.5*z[:,0]) + 0.25*np.abs(z[:,1])
    elif scenario == "tail_difficulty":
        # Mean remains a function of Z only; H and Y are conditionally independent.
        # The sharp tail term intentionally challenges a smooth nuisance learner.
        m = m + 0.9*tail**2
        sigma = 0.7 + 0.5*expit(z[:,0])
    else:
        raise ValueError(scenario)
    y = m + sigma*rng.normal(size=n)
    return z, y, h, m, p


def oracle_gcm_p(y, h, m, p):
    psi0 = (y - m) * (h - p)
    sd = np.std(psi0, ddof=1)
    t = np.sqrt(len(y))*np.mean(psi0)/sd
    return float(norm.sf(t))


SCENARIOS = ("uneven_propensity", "heteroskedastic", "tail_difficulty")
calibration_rows = []
for s_i, scenario in enumerate(SCENARIOS):
    rng = np.random.default_rng(SEED + 1000 + s_i)
    pvals = []
    for rep in range(ORACLE_SIM_REPS):
        z, y, h, m, p = simulate_null_data(SIM_N, rng, scenario)
        pvals.append(oracle_gcm_p(y, h, m, p))
    pvals = np.asarray(pvals)
    rejected = int(np.sum(pvals < 0.05))
    lo, hi = wilson_interval(rejected, len(pvals))
    calibration_rows.append({
        "calibration": "oracle_nuisance",
        "scenario": scenario,
        "reps": int(len(pvals)),
        "n_per_rep": SIM_N,
        "alpha": 0.05,
        "rejections": rejected,
        "type1_rate": rejected / len(pvals),
        "wilson_low": lo,
        "wilson_high": hi,
    })

oracle_calibration = pd.DataFrame(calibration_rows)
oracle_calibration.to_csv(OUT / "gcm_calibration_oracle.csv", index=False)

# Estimated-nuisance stress calibration is deliberately smaller because it fits
# two cross-fitted boosting models per replicate. Increase STAGE32_EST_SIM_REPS
# before final submission if the runtime is acceptable.
est_rows = []
for s_i, scenario in enumerate(SCENARIOS):
    rng = np.random.default_rng(SEED + 2000 + s_i)
    pvals = []
    for rep in range(EST_SIM_REPS):
        z, y, h, _, _ = simulate_null_data(SIM_N, rng, scenario)
        my, ph = crossfit_nuisances(
            z, y, h,
            n_folds=min(3, N_FOLDS),
            seed=SEED + 3000 + 100*s_i + rep,
            fast=True,
        )
        _, t, pval, _, _ = gcm_from_predictions(y, h, my, ph)
        pvals.append(pval)
        print(f"Estimated calibration {scenario}: {rep+1}/{EST_SIM_REPS}")
    pvals = np.asarray(pvals)
    rejected = int(np.sum(pvals < 0.05))
    lo, hi = wilson_interval(rejected, len(pvals))
    est_rows.append({
        "calibration": "estimated_nuisance",
        "scenario": scenario,
        "reps": int(len(pvals)),
        "n_per_rep": SIM_N,
        "alpha": 0.05,
        "rejections": rejected,
        "type1_rate": rejected / len(pvals),
        "wilson_low": lo,
        "wilson_high": hi,
    })

estimated_calibration = pd.DataFrame(est_rows)
estimated_calibration.to_csv(OUT / "gcm_calibration_estimated.csv", index=False)

assumptions = {
    "null_question": "conditional association of stored Khovanov diagonal count Y and frozen no-Khovanov selection H given Z",
    "adjustment_variables": feature_names,
    "inference": "studentized generalized covariance with cross-fitted nuisance estimates; one-sided multiplier bootstrap reported",
    "not_used": "no permutation of selected labels among regression residuals",
    "important_limit": (
        "Validity still depends on nuisance-estimation and sampling assumptions. "
        "Calibration simulations are targeted stress tests, not a proof of calibration for every dependence structure in the atlas."
    ),
}
(OUT / "gcm_assumptions.json").write_text(json.dumps(assumptions, indent=2))

print("\n" + "="*78)
print("STAGE 32 COMPLETE")
print("="*78)
print("\nMain GCM-style result:")
print(main_result.to_string(index=False))
print("\nOracle null calibration:")
print(oracle_calibration.to_string(index=False))
print("\nEstimated-nuisance stress calibration:")
print(estimated_calibration.to_string(index=False))
print("\nDo not promote this test to primary inference until the calibration rows and assumptions are reviewed.")
print("Saved Stage 32 to:", OUT)
