# %% [markdown]
# Stage 32B — Direct calibration of the multiplier-GCM test
#
# Purpose
# -------
# Stage 32 found a positive generalized-covariance association between the
# frozen no-Khovanov selection H and stored Khovanov width Y after continuous
# adjustment.  Its paper-facing p-value used a Rademacher multiplier bootstrap.
#
# This stage calibrates THAT SAME multiplier-bootstrap test under conditional
# independence, rather than calibrating only the asymptotic normal statistic.
#
# Stress dimensions
# -----------------
#   * uneven selection probabilities
#   * heteroskedastic outcomes
#   * tail-concentrated nuisance difficulty
#   * combined heteroskedastic + tail difficulty
#
# Prevalence regimes
# ------------------
#   * moderate: 2%
#   * atlas_like: 220 / 313230 ~= 0.000702
#
# Both oracle-nuisance and estimated-nuisance versions are evaluated.
# Estimated-nuisance fits use the same model family as Stage 32, with a faster
# configuration for repeated simulation.
#
# IMPORTANT
# ---------
# This is a calibration study.  Do not interpret a single Monte-Carlo rejection
# rate mechanically.  Inspect Wilson intervals, rare-selection counts, and the
# direction/size of any inflation.
#
# Fresh-session usage
# -------------------
# %run "/content/drive/MyDrive/consensus_hardness_refactored/notebooks/stage32B_multiplier_gcm_calibration.py"
#
# Defaults are substantial but resumable.  Every scenario/regime/mode writes a
# checkpoint CSV after each replicate.
#
# Environment overrides
# ---------------------
# STAGE32B_ORACLE_REPS=100
# STAGE32B_EST_REPS=30
# STAGE32B_MULT_REPS=399
# STAGE32B_FOLDS=3
# STAGE32B_N_MODERATE=12000
# STAGE32B_N_RARE=60000
# STAGE32B_SEED=202609132
# STAGE32B_ALPHA=0.05

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

DEFAULT_DATA_DIR = Path("/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants")
DEFAULT_ROOT = DEFAULT_DATA_DIR / "processed_consensus_hardness" / "corrected_run_20260819"
DATA_DIR = Path(os.environ.get("KNOT_DATA_DIR", str(DEFAULT_DATA_DIR)))
ROOT = Path(os.environ.get("KNOT_OUTPUT_DIR", str(DEFAULT_ROOT)))
OUT = ROOT / "32B_multiplier_gcm_calibration"

ORACLE_REPS = int(os.environ.get("STAGE32B_ORACLE_REPS", "100"))
EST_REPS = int(os.environ.get("STAGE32B_EST_REPS", "30"))
MULT_REPS = int(os.environ.get("STAGE32B_MULT_REPS", "399"))
N_FOLDS = int(os.environ.get("STAGE32B_FOLDS", "3"))
N_MODERATE = int(os.environ.get("STAGE32B_N_MODERATE", "12000"))
N_RARE = int(os.environ.get("STAGE32B_N_RARE", "60000"))
SEED = int(os.environ.get("STAGE32B_SEED", "202609132"))
ALPHA = float(os.environ.get("STAGE32B_ALPHA", "0.05"))

ATLAS_PREVALENCE = 220 / 313230
PREVALENCE = {
    "moderate": 0.02,
    "atlas_like": ATLAS_PREVALENCE,
}
N_BY_REGIME = {
    "moderate": N_MODERATE,
    "atlas_like": N_RARE,
}
SCENARIOS = (
    "uneven_propensity",
    "heteroskedastic",
    "tail_difficulty",
    "combined",
)


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


def wilson_interval(k: int, n: int, alpha: float = 0.05):
    if n <= 0:
        return np.nan, np.nan
    z = float(norm.ppf(1 - alpha / 2))
    phat = k / n
    den = 1 + z*z/n
    center = (phat + z*z/(2*n)) / den
    half = z * math.sqrt(phat*(1-phat)/n + z*z/(4*n*n)) / den
    return center - half, center + half


def solve_intercept(signal: np.ndarray, target: float) -> float:
    """Find b such that mean(expit(b + signal)) ~= target."""
    lo, hi = -40.0, 20.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        val = float(np.mean(expit(mid + signal)))
        if val > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def simulate_null_data(
    n: int,
    rng: np.random.Generator,
    scenario: str,
    target_prevalence: float,
):
    """
    Generate H ⟂ Y | Z exactly.

    H and Y share Z-dependence, but their random noises are independent.
    """
    z = rng.normal(size=(n, 4))
    tail = np.maximum(z[:, 0] - 1.0, 0.0)

    h_signal = 1.35*z[:, 0] - 0.7*z[:, 1] + 0.45*z[:, 2]**2 - 0.2*z[:, 3]
    intercept = solve_intercept(h_signal, target_prevalence)
    p = expit(intercept + h_signal)
    h = rng.binomial(1, p, size=n).astype(np.int8)

    m = 2.2 + 0.35*z[:, 0] - 0.25*z[:, 1] + 0.15*z[:, 2]**2
    sigma = np.ones(n, dtype=float)

    if scenario == "uneven_propensity":
        pass
    elif scenario == "heteroskedastic":
        sigma = 0.50 + 0.70*expit(1.6*z[:, 0]) + 0.25*np.abs(z[:, 1])
    elif scenario == "tail_difficulty":
        m = m + 0.95*tail**2 + 0.25*np.sin(3.0*z[:, 1])*(z[:, 0] > 1.0)
        sigma = 0.70 + 0.45*expit(z[:, 0])
    elif scenario == "combined":
        m = m + 0.95*tail**2 + 0.25*np.sin(3.0*z[:, 1])*(z[:, 0] > 1.0)
        sigma = 0.45 + 0.85*expit(1.8*z[:, 0]) + 0.30*np.abs(z[:, 1])
    else:
        raise ValueError(scenario)

    y = m + sigma*rng.normal(size=n)
    return z, y, h, m, p


def make_y_model(seed: int):
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=120,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=1.0,
        random_state=seed,
    )


def make_h_model(seed: int):
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=140,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=seed,
    )


def crossfit_nuisances(z, y, h, *, seed: int):
    h = np.asarray(h, dtype=int)
    positives = int(h.sum())
    negatives = int(len(h) - positives)
    folds = min(N_FOLDS, positives, negatives)
    if folds < 2:
        raise RuntimeError(
            f"Too few observations in one class for cross-fitting: positives={positives}"
        )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    my = np.full(len(y), np.nan, dtype=float)
    ph = np.full(len(y), np.nan, dtype=float)
    for fold, (tr, te) in enumerate(splitter.split(z, h)):
        ym = make_y_model(seed + 100*fold)
        hm = make_h_model(seed + 100*fold + 1)
        ym.fit(z[tr], y[tr])
        hm.fit(z[tr], h[tr])
        my[te] = ym.predict(z[te])
        ph[te] = hm.predict_proba(z[te])[:, 1]
    if not np.isfinite(my).all() or not np.isfinite(ph).all():
        raise RuntimeError("Non-finite nuisance predictions")
    return my, np.clip(ph, 1e-10, 1 - 1e-10)


def gcm_stat(y, h, my, ph):
    psi = (np.asarray(y, float) - np.asarray(my, float)) * (
        np.asarray(h, float) - np.asarray(ph, float)
    )
    sd = float(np.std(psi, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError("Degenerate GCM variance")
    stat = float(np.sqrt(len(psi)) * np.mean(psi) / sd)
    return psi, stat, float(norm.sf(stat))


def multiplier_pvalue(psi, observed_stat, *, reps: int, seed: int):
    centered = np.asarray(psi, dtype=float) - float(np.mean(psi))
    sd = float(np.std(psi, ddof=1))
    n = len(centered)
    rng = np.random.default_rng(seed)
    ge = 0
    block = 32
    for start in range(0, reps, block):
        b = min(block, reps - start)
        xi = rng.integers(0, 2, size=(b, n), dtype=np.int8)
        xi = (2*xi - 1).astype(np.float64)
        stats = (xi @ centered) / (np.sqrt(n) * sd)
        ge += int(np.sum(stats >= observed_stat))
    return float((1 + ge) / (reps + 1))


def checkpoint_path(mode: str, regime: str, scenario: str) -> Path:
    return OUT / f"replicates__{mode}__{regime}__{scenario}.csv"


def run_combo(mode: str, regime: str, scenario: str, reps: int):
    path = checkpoint_path(mode, regime, scenario)
    if path.exists():
        old = pd.read_csv(path)
    else:
        old = pd.DataFrame()

    done = set(old["replicate"].astype(int)) if len(old) else set()
    rows = old.to_dict("records") if len(old) else []

    n = N_BY_REGIME[regime]
    target = PREVALENCE[regime]

    for rep in range(reps):
        if rep in done:
            continue

        # Deterministic independent seed per combination/replicate.
        combo_code = (
            1000000 * (0 if mode == "oracle" else 1)
            + 100000 * list(PREVALENCE).index(regime)
            + 10000 * SCENARIOS.index(scenario)
            + rep
        )
        rng = np.random.default_rng(SEED + combo_code)

        # Rare selection can occasionally generate too few positives; regenerate.
        for retry in range(20):
            z, y, h, m, p = simulate_null_data(n, rng, scenario, target)
            if h.sum() >= max(6, N_FOLDS):
                break
        else:
            raise RuntimeError(
                f"Could not generate enough selected observations for {regime}/{scenario}"
            )

        if mode == "oracle":
            my, ph = m, p
        elif mode == "estimated":
            my, ph = crossfit_nuisances(
                z, y, h,
                seed=SEED + 5000000 + combo_code,
            )
        else:
            raise ValueError(mode)

        psi, stat, p_normal = gcm_stat(y, h, my, ph)
        p_mult = multiplier_pvalue(
            psi,
            stat,
            reps=MULT_REPS,
            seed=SEED + 9000000 + combo_code,
        )

        rows.append({
            "mode": mode,
            "prevalence_regime": regime,
            "scenario": scenario,
            "replicate": rep,
            "n": n,
            "target_prevalence": target,
            "selected_n": int(h.sum()),
            "realized_prevalence": float(h.mean()),
            "gcm_stat": stat,
            "normal_p": p_normal,
            "multiplier_p": p_mult,
            "multiplier_reps": MULT_REPS,
        })
        pd.DataFrame(rows).sort_values("replicate").to_csv(path, index=False)

        print(
            f"{mode:9s} | {regime:10s} | {scenario:18s} | "
            f"{rep+1}/{reps} | selected={int(h.sum())} | p_mult={p_mult:.4g}"
        )

    return pd.DataFrame(rows)


all_parts = []
for mode, reps in (("oracle", ORACLE_REPS), ("estimated", EST_REPS)):
    for regime in PREVALENCE:
        for scenario in SCENARIOS:
            all_parts.append(run_combo(mode, regime, scenario, reps))

replicates = pd.concat(all_parts, ignore_index=True)
replicates.to_csv(OUT / "all_multiplier_calibration_replicates.csv", index=False)

summary_rows = []
for keys, g in replicates.groupby(
    ["mode", "prevalence_regime", "scenario"], sort=False
):
    mode, regime, scenario = keys
    for test_name, pcol in (
        ("multiplier", "multiplier_p"),
        ("normal", "normal_p"),
    ):
        vals = g[pcol].to_numpy(float)
        k = int(np.sum(vals < ALPHA))
        lo, hi = wilson_interval(k, len(vals))
        summary_rows.append({
            "mode": mode,
            "prevalence_regime": regime,
            "scenario": scenario,
            "test": test_name,
            "reps": int(len(vals)),
            "n_per_rep": int(g["n"].iloc[0]),
            "target_prevalence": float(g["target_prevalence"].iloc[0]),
            "median_selected_n": float(g["selected_n"].median()),
            "min_selected_n": int(g["selected_n"].min()),
            "max_selected_n": int(g["selected_n"].max()),
            "alpha": ALPHA,
            "rejections": k,
            "type1_rate": k / len(vals),
            "wilson_low": lo,
            "wilson_high": hi,
            "nominal_0_05_inside_wilson": bool(lo <= ALPHA <= hi),
            "clear_inflation_flag": bool(lo > ALPHA),
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "multiplier_gcm_calibration_summary.csv", index=False)

# Compact decision table only for the actual multiplier test.
decision = summary.loc[summary["test"].eq("multiplier")].copy()
decision["status"] = np.select(
    [
        decision["clear_inflation_flag"],
        decision["nominal_0_05_inside_wilson"],
    ],
    [
        "CLEAR_INFLATION",
        "COMPATIBLE_WITH_0.05_AT_CURRENT_MONTE_CARLO_PRECISION",
    ],
    default="DEVIATES_BUT_CI_DOES_NOT_EXCLUDE_0.05",
)
decision.to_csv(OUT / "multiplier_gcm_calibration_decision.csv", index=False)

print("\n" + "="*88)
print("STAGE 32B COMPLETE")
print("="*88)
print("\nMultiplier-GCM calibration:")
print(decision.to_string(index=False))
print(
    "\nInterpretation rule: promote the Stage-32 multiplier p-value only if the "
    "estimated-nuisance rows, especially atlas_like heteroskedastic/tail/combined, "
    "show no clear Type-I inflation. Wide intervals mean more simulations are needed."
)
print("Saved Stage 32B to:", OUT)
