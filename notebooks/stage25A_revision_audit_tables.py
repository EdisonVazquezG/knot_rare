"""Stage 25A: paper-ready diagnostics from already-frozen revision stages.

No models or permutations are rerun.  The script exposes the exact-stratum
cell-size diagnostics requested in review and copies the final bin-count
sensitivity table into one small revision directory.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants/"
    "processed_consensus_hardness/corrected_run_20260819"
)
OUT = ROOT / "25A_revision_audit_tables"
OUT.mkdir(parents=True, exist_ok=True)

GAP_DIAG = ROOT / "22_gap_thickness_dependence/gap_null_stratum_diagnostics.csv"
NO_KH_DIAG = (
    ROOT
    / "22_gap_thickness_dependence/no_khovanov_external_with_kh_norm_strata.csv"
)

for path in (GAP_DIAG, NO_KH_DIAG):
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen Stage 22 output: {path}")

diag = pd.concat(
    [pd.read_csv(GAP_DIAG), pd.read_csv(NO_KH_DIAG)], ignore_index=True
)

keep = [
    "analysis",
    "invariant",
    "thickness_mode",
    "include_khovanov_norm_bin",
    "n_strata",
    "stratum_size_min",
    "stratum_size_q25",
    "stratum_size_median",
    "stratum_size_q75",
    "stratum_size_max",
    "singleton_strata_prop",
    "selected_total",
    "selected_fixed",
    "selected_movable",
    "n_random_groups",
    "n_fixed_selected_groups",
    "n_strata_with_selected",
]
missing = [c for c in keep if c not in diag]
if missing:
    raise KeyError(f"Missing Stage 22 diagnostic columns: {missing}")
detail = diag[keep].copy()
detail.to_csv(OUT / "permutation_stratum_sizes_by_view.csv", index=False)

# Compact ranges for the main text/supplement.  Cell sizes refer to the exact
# per-view strata used by the membership samplers, not to a Cartesian product
# of all five norm bins.
summary_rows = []
for analysis, group in detail.groupby("analysis", sort=False):
    summary_rows.append(
        {
            "analysis": analysis,
            "n_views": group["invariant"].nunique(),
            "n_strata_min": int(group["n_strata"].min()),
            "n_strata_max": int(group["n_strata"].max()),
            "median_cell_size_min": float(group["stratum_size_median"].min()),
            "median_cell_size_max": float(group["stratum_size_median"].max()),
            "q75_cell_size_min": float(group["stratum_size_q75"].min()),
            "q75_cell_size_max": float(group["stratum_size_q75"].max()),
            "maximum_cell_size_min": int(group["stratum_size_max"].min()),
            "maximum_cell_size_max": int(group["stratum_size_max"].max()),
            "singleton_prop_min": float(group["singleton_strata_prop"].min()),
            "singleton_prop_max": float(group["singleton_strata_prop"].max()),
            "selected_movable_min": int(group["selected_movable"].min()),
            "selected_movable_max": int(group["selected_movable"].max()),
            "selected_fixed_min": int(group["selected_fixed"].min()),
            "selected_fixed_max": int(group["selected_fixed"].max()),
            "random_groups_min": int(group["n_random_groups"].min()),
            "random_groups_max": int(group["n_random_groups"].max()),
        }
    )
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "permutation_stratum_size_summary.csv", index=False)

print("Exact permutation-stratum detail:")
print(detail.to_string(index=False))
print("\nCompact ranges for reporting:")
print(summary.to_string(index=False))
print(f"\nSaved Stage 25A to: {OUT}")
