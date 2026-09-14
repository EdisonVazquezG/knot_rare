# %% [markdown]
# Stage 38 — Sensitivity of the baselines to the auxiliary residual compression
#
# Revision request (Ernesto, "Bring the methods into agreement with the
# computation"):
#
#   "In the public implementation, a further PCA projection compresses the
#    residuals to at most 64 coordinates, fitted from a default sample of
#    40,000 training observations.  This can discard unusual residual
#    directions, so it is part of the method being compared. ... A small
#    sensitivity check should establish whether this additional compression
#    materially changes the comparison."
#
# Manuscript section 3.3 already promises this check in the Supplementary
# Information.  This stage supplies it.
#
# TWO IMPLEMENTATION HAZARDS, both handled below:
#
#   1. Stage 23 hardcodes  OUT = ROOT / "23_anomaly_score_baselines"  and puts
#      its per-representation score checkpoints underneath it.  Simply setting
#      STAGE23_RESIDUAL_DIM and rerunning would (a) reload the frozen 64-dim
#      checkpoints and silently produce an identical sweep, and (b) overwrite
#      the frozen Stage 23 outputs.  We therefore rewrite that single line of
#      the Stage 23 source so OUT points into a sweep-specific directory, while
#      leaving ROOT untouched because Stage 23's find_one() resolves frozen
#      artifacts against it.
#
#   2. Reruns of this stage would hit their own caches.  Each sweep directory is
#      deleted before use unless STAGE38_KEEP_CACHE=1.
#
# Only residual Mahalanobis and residual isolation forest use the auxiliary
# projection.  Raw SSE, relative error and the conditional percentile are
# invariant to it and serve as controls: if any of them moves, the sweep is not
# isolating the compression and the stage aborts.
#
# Usage (same runtime as paper_run.ipynb):
#   %run .../stage38_residual_compression_sensitivity.py

from __future__ import annotations

from pathlib import Path
import itertools
import json
import os
import shutil

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants/"
    "processed_consensus_hardness/corrected_run_20260819"
)
ROOT_DIR = Path(
    os.environ.get("KNOT_OUTPUT_DIR", str(globals().get("OUTPUT_DIR", DEFAULT_ROOT)))
)
# NOTE: Stage 23 assigns its own `OUT` into globals() when exec'd, so this
# stage must not use that name for its own directory.
SWEEP_ROOT = ROOT_DIR / "38_residual_compression_sensitivity"
SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

STAGE23 = Path(__file__).resolve().parent / "stage23_anomaly_score_baselines.py"
if not STAGE23.exists():
    raise FileNotFoundError(f"Stage 23 not found next to this file: {STAGE23}")

FROZEN_DIM = 64
FULL_RANK_SENTINEL = 100000  # Stage 23 clamps with min(), so this means full rank
SWEEP = [s.strip() for s in os.environ.get(
    "STAGE38_SWEEP", "32,64,128,none"
).split(",") if s.strip()]
KEEP_CACHE = os.environ.get("STAGE38_KEEP_CACHE", "0") == "1"

AFFECTED = ("residual_mahalanobis", "residual_isolation_forest")
CONTROLS = ("raw_sse", "relative_nre", "conditional_percentile_100")

# The line we must redirect, verbatim from the Stage 23 source.
OUT_LINE = 'OUT = ROOT / "23_anomaly_score_baselines"'

source = STAGE23.read_text()
if OUT_LINE not in source:
    raise RuntimeError(
        "Could not find the Stage 23 output-directory line to redirect:\n"
        f"  {OUT_LINE}\n"
        "Stage 38 refuses to run rather than overwrite the frozen Stage 23 "
        "outputs or silently reuse their checkpoints."
    )

print("=" * 78)
print("STAGE 38 — auxiliary residual-compression sensitivity")
print("=" * 78)
print("Sweep:", SWEEP, " frozen default:", FROZEN_DIM)

families, diagnostics, selections, actual_dims = [], [], {}, {}

for setting in SWEEP:
    label = str(setting)
    dim = FULL_RANK_SENTINEL if label == "none" else int(label)

    run_dir = SWEEP_ROOT / f"residual_dim_{label}"
    if run_dir.exists() and not KEEP_CACHE:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'-' * 70}\n--- residual_dim = {label}  ->  {run_dir}\n{'-' * 70}")

    patched = source.replace(
        OUT_LINE, f'OUT = Path(r"{run_dir}")  # redirected by Stage 38'
    )
    os.environ["STAGE23_RESIDUAL_DIM"] = str(dim)

    # Stage 23 reads meta / X_dict / feature_cols_dict / CONFIG / OUTPUT_DIR from
    # the calling namespace, so exec into globals() rather than importing.
    exec(compile(patched, f"{STAGE23}::dim={label}", "exec"), globals())

    for name in ("score_family_summary", "score_norm_diagnostics", "score_selected_ids"):
        if name not in globals():
            raise RuntimeError(f"Stage 23 did not expose `{name}`.")

    fam = globals()["score_family_summary"].copy()
    fam.insert(0, "residual_dim", label)
    families.append(fam)

    dia = globals()["score_norm_diagnostics"].copy()
    dia.insert(0, "residual_dim", label)
    diagnostics.append(dia)

    sel = globals()["score_selected_ids"].copy()
    id_col = next(c for c in sel.columns if c not in ("method", "family"))
    selections[label] = {
        (str(m), str(f)): set(g[id_col].astype(str))
        for (m, f), g in sel.groupby(["method", "family"])
    }

    # Confirm the projection actually changed: Stage 23 records the clamped
    # dimension it used inside each per-view checkpoint.
    dims = {}
    for npz in sorted((run_dir / "checkpoints").glob("*_test_scores.npz")):
        with np.load(npz, allow_pickle=True) as z:
            if "residual_sketch_dimension" in z:
                dims[npz.stem.replace("_test_scores", "")] = int(
                    np.asarray(z["residual_sketch_dimension"]).ravel()[0]
                )
    actual_dims[label] = dims
    print(f"[stage38] realised residual dimensions: {dims}")

os.environ.pop("STAGE23_RESIDUAL_DIM", None)

sweep_fam = pd.concat(families, ignore_index=True)
sweep_dia = pd.concat(diagnostics, ignore_index=True)
sweep_fam.to_csv(SWEEP_ROOT / "residual_compression_family_summary.csv", index=False)
sweep_dia.to_csv(SWEEP_ROOT / "residual_compression_norm_diagnostics.csv", index=False)
pd.DataFrame(actual_dims).to_csv(SWEEP_ROOT / "residual_compression_realised_dims.csv")

# ---------------------------------------------------------------------------
# Guard: the sweep must actually have changed something.
# ---------------------------------------------------------------------------
distinct = {tuple(sorted(d.items())) for d in actual_dims.values()}
if len(distinct) < 2:
    raise RuntimeError(
        "Every sweep setting realised the same residual dimension, so nothing "
        "was varied. Check that Stage 23 recomputed rather than loading "
        f"checkpoints. Realised: {actual_dims}"
    )

# ---------------------------------------------------------------------------
# Deviation from the frozen 64-dimensional default
# ---------------------------------------------------------------------------
frozen = str(FROZEN_DIM)
if frozen not in [str(s) for s in SWEEP]:
    raise RuntimeError(f"The sweep must include the frozen default {FROZEN_DIM}.")

metric_cols = [
    c for c in sweep_fam.columns
    if sweep_fam[c].dtype.kind in "fi" and c != "residual_dim"
]
base = sweep_fam[sweep_fam["residual_dim"] == frozen].set_index(["method", "family"])

rows = []
for label in [str(s) for s in SWEEP]:
    if label == frozen:
        continue
    cur = sweep_fam[sweep_fam["residual_dim"] == label].set_index(["method", "family"])
    for key in cur.index:
        if key not in base.index:
            continue
        for m in metric_cols:
            rows.append(
                {
                    "residual_dim": label,
                    "method": key[0],
                    "family": key[1],
                    "affected_by_compression": key[0] in AFFECTED,
                    "metric": m,
                    "value": float(cur.loc[key, m]),
                    f"value_at_dim{FROZEN_DIM}": float(base.loc[key, m]),
                    "difference": float(cur.loc[key, m] - base.loc[key, m]),
                }
            )

deviation = pd.DataFrame(rows)
deviation.to_csv(SWEEP_ROOT / "residual_compression_deviation_from_frozen.csv", index=False)

ctrl = deviation[deviation["method"].isin(CONTROLS)]
max_ctrl = float(ctrl["difference"].abs().max()) if len(ctrl) else 0.0
if max_ctrl > 1e-9:
    worst = ctrl.loc[ctrl["difference"].abs().idxmax()]
    raise RuntimeError(
        "A score that does not use the auxiliary projection changed by "
        f"{max_ctrl:g} ({worst['method']}, {worst['family']}, {worst['metric']}). "
        "The sweep is not isolating the compression."
    )

aff = deviation[deviation["affected_by_compression"]]
max_aff = float(aff["difference"].abs().max()) if len(aff) else float("nan")

# ---------------------------------------------------------------------------
# Membership stability of the two affected selections
# ---------------------------------------------------------------------------
jac_rows = []
keys = sorted({k for d in selections.values() for k in d if k[0] in AFFECTED})
for key in keys:
    for a, b in itertools.combinations([str(s) for s in SWEEP], 2):
        sa = selections[a].get(key, set())
        sb = selections[b].get(key, set())
        if not sa and not sb:
            continue
        jac_rows.append(
            {
                "method": key[0],
                "family": key[1],
                "dim_a": a,
                "dim_b": b,
                "n_a": len(sa),
                "n_b": len(sb),
                "n_shared": len(sa & sb),
                "jaccard": len(sa & sb) / max(len(sa | sb), 1),
            }
        )

jac = pd.DataFrame(jac_rows)
if len(jac):
    jac.to_csv(SWEEP_ROOT / "residual_compression_membership_jaccard.csv", index=False)

gate = {
    "stage": 38,
    "frozen_residual_dim": FROZEN_DIM,
    "frozen_residual_fit_size": 40000,
    "isolation_forest_trees": 300,
    "isolation_forest_max_samples": "min(4096, n)",
    "residual_pca_variance_floor": 1e-10,
    "swept_settings": [str(s) for s in SWEEP],
    "realised_dimensions": actual_dims,
    "max_abs_change_unaffected_scores": max_ctrl,
    "max_abs_change_affected_scores": max_aff,
    "min_affected_jaccard": float(jac["jaccard"].min()) if len(jac) else None,
    "median_affected_jaccard": float(jac["jaccard"].median()) if len(jac) else None,
}
(SWEEP_ROOT / "stage38_gate.json").write_text(json.dumps(gate, indent=2, default=str))

print("\n" + "=" * 78)
print("Key external phenotypes by residual dimension:")
keep = [c for c in ("kh_diagonal_mean", "kh_diagonal_ge_3_prop", "alternating_prop", "n")
        if c in sweep_fam.columns]
print(
    sweep_fam[sweep_fam["method"].isin(AFFECTED)]
    .pivot_table(index=["method", "family"], columns="residual_dim", values=keep)
    .to_string()
)
if len(jac):
    print("\nMembership Jaccard across dimensions (affected scores):")
    print(jac.to_string(index=False))
print(f"\nMax |change| for scores using the projection: {max_aff:.4g}")
print(f"Max |change| for control scores (must be 0):  {max_ctrl:.3g}")
print("\nSaved Stage 38 to:", SWEEP_ROOT)