# %% [markdown]
# Stage 39 — Per-representation matching balance
#
# Revision request (Ernesto):
#   "Report balance in the individual representation norms as well as their
#    mean and maximum."
#
# Manuscript section 3.5 promises exactly this in the Supplementary Information.
#
# WHY THIS IS A SEPARATE STAGE
#   Stage 25 uses a single list, NORM_COLS = ["mean_log_sq_norm",
#   "max_log_sq_norm"], for BOTH the matching distance and the balance report.
#   Adding the five per-representation norms to that list would change which
#   controls are selected, and therefore every paper-facing geometry result
#   (volume, tetrahedra, Holm-corrected p-values). The reviewer asked for a
#   different balance REPORT, not a different matching rule.
#
#   This stage therefore reads the frozen pairs written by Stage 25 and
#   recomputes balance on the five individual norms without refitting anything.
#   Stage 25 outputs are opened read-only and never rewritten.
#
# Usage (same runtime as the other stages):
#   %run .../stage39_per_norm_matching_balance.py

from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants/"
    "processed_consensus_hardness/corrected_run_20260819"
)
ROOT = Path(
    os.environ.get("KNOT_OUTPUT_DIR", str(globals().get("OUTPUT_DIR", DEFAULT_ROOT)))
)
OUT = ROOT / "39_per_norm_matching_balance"
OUT.mkdir(parents=True, exist_ok=True)

ID_COL = "knot_id_base"

# Matching columns (Stage 25) versus the columns we now report balance on.
MATCHED_ON = ["mean_log_sq_norm", "max_log_sq_norm"]
INDIVIDUAL = [
    "Alexander_log_sq_norm",
    "Jones_log_sq_norm",
    "HOMFLY_PT_log_sq_norm",
    "Theta_log_sq_norm",
    "Khovanov_log_sq_norm",
]

METHOD_LABELS = {
    "raw_sse": "Raw SSE",
    "relative_nre": "Relative error",
    "residual_mahalanobis": "Residual Mahalanobis",
    "residual_isolation_forest": "Residual isolation forest",
    "conditional_percentile_100": "Conditional percentile",
}

print("=" * 78)
print("STAGE 39 — per-representation matching balance")
print("=" * 78)


def find_one(filename: str) -> Path:
    hits = sorted(ROOT.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"{filename} not found under {ROOT}")
    return hits[0]


pairs_path = find_one("score_geometry_pairs.csv")
print("Frozen matched pairs:", pairs_path)
pairs = pd.read_csv(pairs_path, dtype={ID_COL: str})

atlas = pd.read_parquet(find_one("final_hard_regime_atlas.parquet")).reset_index(
    drop=True
)
phenotype = pd.read_parquet(
    find_one("complete_mathematical_phenotype_atlas.parquet")
).reset_index(drop=True)
atlas[ID_COL] = atlas[ID_COL].astype(str)
phenotype[ID_COL] = phenotype[ID_COL].astype(str)

# Assemble a single norm lookup from whichever table carries each column.
norm_frames = [phenotype, atlas]
lookup = atlas[[ID_COL]].copy()
missing_cols = []
for col in MATCHED_ON + INDIVIDUAL:
    src = next((f for f in norm_frames if col in f.columns), None)
    if src is None:
        missing_cols.append(col)
        continue
    lookup = lookup.merge(src[[ID_COL, col]], on=ID_COL, how="left", validate="1:1")

if missing_cols:
    available = sorted(
        {c for f in norm_frames for c in f.columns if "norm" in str(c).lower()}
    )
    raise RuntimeError(
        f"Norm columns not found: {missing_cols}\nAvailable: {available}"
    )

# Identify the selected/control id columns as written by Stage 25.
sel_col = next(
    c for c in (ID_COL, "selected_id", "selected_knot_id") if c in pairs.columns
)
ctl_col = next(
    c for c in ("control_id", "control_knot_id") if c in pairs.columns
)
print(f"Pair columns: selected={sel_col!r}, control={ctl_col!r}, n={len(pairs):,}")

merged = (
    pairs[["method", sel_col, ctl_col]]
    .merge(lookup, left_on=sel_col, right_on=ID_COL, how="left")
    .merge(
        lookup,
        left_on=ctl_col,
        right_on=ID_COL,
        how="left",
        suffixes=("_sel", "_ctl"),
    )
)


def paired_smd(diff: np.ndarray) -> float:
    """Paired SMD: mean difference over the SD of the within-pair differences."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return float("nan")
    sd = float(diff.std(ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        return float("nan")
    return float(diff.mean() / sd)


def conventional_smd(a: np.ndarray, b: np.ndarray) -> float:
    """Conventional SMD: mean difference over the pooled between-group SD.

    This is the quantity the Supplementary Information reports as balance.
    It is not interchangeable with the paired SMD: matching shrinks the
    within-pair spread, so the paired version is systematically the larger of
    the two and the two must never be compared with each other.
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 2:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if not np.isfinite(pooled) or pooled == 0.0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


rows = []
for method, g in merged.groupby("method"):
    for col in MATCHED_ON + INDIVIDUAL:
        d = g[f"{col}_sel"].to_numpy(float) - g[f"{col}_ctl"].to_numpy(float)
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(str(method), str(method)),
                "norm": col,
                "role": "matched on" if col in MATCHED_ON else "reported only",
                "n_pairs": int(np.sum(np.isfinite(d))),
                "mean_selected": float(np.nanmean(g[f"{col}_sel"].to_numpy(float))),
                "mean_control": float(np.nanmean(g[f"{col}_ctl"].to_numpy(float))),
                "mean_difference": float(np.nanmean(d)),
                "conventional_smd": conventional_smd(
                    g[f"{col}_sel"].to_numpy(float), g[f"{col}_ctl"].to_numpy(float)
                ),
                "paired_smd": paired_smd(d),
            }
        )

balance = pd.DataFrame(rows)
balance.to_csv(OUT / "per_norm_matching_balance.csv", index=False)

wide = {}
for metric in ("conventional_smd", "paired_smd"):
    w = balance.pivot_table(index="norm", columns="method_label", values=metric)
    wide[metric] = w.reindex(MATCHED_ON + INDIVIDUAL)
    wide[metric].to_csv(OUT / f"per_norm_balance_{metric}.csv")

# The reviewer's request concerns the reported balance metric, so the headline
# comparison uses the conventional SMD on both column groups.
worst = (
    balance[balance["role"] == "reported only"]
    .assign(abs_smd=lambda d: d["conventional_smd"].abs())
    .sort_values("abs_smd", ascending=False)
)

max_matched = float(
    balance.loc[balance["role"] == "matched on", "conventional_smd"].abs().max()
)
max_individual = float(worst["abs_smd"].max())
max_matched_paired = float(
    balance.loc[balance["role"] == "matched on", "paired_smd"].abs().max()
)
max_individual_paired = float(
    balance.loc[balance["role"] == "reported only", "paired_smd"].abs().max()
)

gate = {
    "stage": 39,
    "source_pairs": str(pairs_path),
    "matched_on": MATCHED_ON,
    "reported_only": INDIVIDUAL,
    "matching_unchanged": True,
    "balance_metric": "conventional SMD (matches the Supplementary Information)",
    "max_abs_conventional_smd_matched_columns": max_matched,
    "max_abs_conventional_smd_individual_norms": max_individual,
    "max_abs_paired_smd_matched_columns": max_matched_paired,
    "max_abs_paired_smd_individual_norms": max_individual_paired,
    "worst_individual": {
        "method": str(worst.iloc[0]["method_label"]),
        "norm": str(worst.iloc[0]["norm"]),
        "conventional_smd": float(worst.iloc[0]["conventional_smd"]),
    },
}
(OUT / "stage39_gate.json").write_text(json.dumps(gate, indent=2))

print("\nCONVENTIONAL SMD (the metric reported as balance in the SI):")
print(wide["conventional_smd"].round(4).to_string())
print("\nPAIRED SMD (shown for completeness; not comparable to the above):")
print(wide["paired_smd"].round(4).to_string())
print(f"\nConventional  max |SMD| matched columns : {max_matched:.4f}")
print(f"Conventional  max |SMD| individual norms: {max_individual:.4f}")
print(f"Paired        max |SMD| matched columns : {max_matched_paired:.4f}")
print(f"Paired        max |SMD| individual norms: {max_individual_paired:.4f}")
print(
    "\nWorst individual (conventional): "
    f"{gate['worst_individual']['method']} / {gate['worst_individual']['norm']} "
    f"= {gate['worst_individual']['conventional_smd']:+.4f}"
)
print("\nSaved Stage 39 to:", OUT)
