# %% [markdown]
# Stage 28 — Definition/provenance audits requested in review
#
# Run in the SAME runtime as paper_run.ipynb (or any runtime containing
# meta, X_dict, feature_cols_dict, CONFIG, OUTPUT_DIR).
#
# Closes/diagnoses:
#   P4  Theta is theta-only vs the full pair (Delta, theta)
#   P6  sign convention for s and sigma; whether |s-sigma| == ||s|-|sigma||
#   P8  exactly which stored Khovanov coordinates enter the analysis
#   P9  313,230 vs 313,231 and unknot exclusion
#   P12 representation dimensions
#
# This stage DOES NOT change any scientific result. It writes an explicit
# decision table and the exact rows that would force a re-analysis.

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import json

# ---------------------------------------------------------------------------
# Bootstrap: Stage 28 can now run in a fresh Colab runtime.
#
# Edit these two paths if your Drive layout differs, or define environment
# variables KNOT_DATA_DIR / KNOT_OUTPUT_DIR before running this file.
# ---------------------------------------------------------------------------
import os
import consensus_hardness as ch

DEFAULT_DATA_DIR = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants"
)
DEFAULT_OUTPUT_DIR = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants/"
    "processed_consensus_hardness/corrected_run_20260819"
)

if "CONFIG" not in globals():
    CONFIG = ch.canonical_run_config()

if "OUTPUT_DIR" not in globals():
    OUTPUT_DIR = Path(os.environ.get("KNOT_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))

REQUIRED_IN_MEMORY = ("meta", "X_dict", "feature_cols_dict")
missing = [name for name in REQUIRED_IN_MEMORY if name not in globals()]

if missing:
    DATA_DIR = Path(os.environ.get("KNOT_DATA_DIR", str(DEFAULT_DATA_DIR)))
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw knot-data directory not found: {DATA_DIR}\n"
            "Set it before running, e.g.\n"
            "  import os\n"
            "  os.environ['KNOT_DATA_DIR']='/your/path/to/Invariants'\n"
        )

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

    print("Stage 28: rebuilding the aligned in-memory dataset only.")
    print("This does NOT rerun PCA, nulls, autoencoders, or paper results.")
    print("DATA_DIR   =", DATA_DIR)
    print("OUTPUT_DIR =", OUTPUT_DIR)

    aligned = ch.build_aligned_dataset(
        base_dir=DATA_DIR,
        file_map=FILE_MAP,
        representation_specs=REPRESENTATION_SPECS,
        min_crossings=CONFIG.universe.min_crossings,
        max_crossings=CONFIG.universe.max_crossings,
        output_dir=None,  # audit-only bootstrap: do not overwrite frozen artifacts
        preferred_metadata_sources=["alex", "jones", "homfly", "theta", "kh"],
        expected_n=CONFIG.universe.expected_n,
        expected_s_qc_corrections=CONFIG.universe.expected_s_qc_corrections,
    )
    meta = aligned["meta"]
    X_dict = aligned["X_dict"]
    feature_cols_dict = aligned["feature_cols_dict"]
    aligned_tables = aligned["aligned_tables"]

OUT = Path(OUTPUT_DIR) / "28_definition_provenance_audit"
OUT.mkdir(parents=True, exist_ok=True)

ID_COL = CONFIG.universe.id_col
S_COL = CONFIG.s_col if CONFIG.s_col in meta.columns else "s_invariant"
INVARIANTS = ("Alexander", "Jones", "HOMFLY-PT", "Theta", "Khovanov")

# ---------------------------------------------------------------------------
# P4 — Theta
# ---------------------------------------------------------------------------
theta_cols = [str(c) for c in feature_cols_dict["Theta"]]
theta_a_cols = [c for c in theta_cols if c.startswith("A")]
theta_t_cols = [c for c in theta_cols if c.startswith("T")]

theta_audit = {
    "n_theta_features": len(theta_cols),
    "n_T_prefixed": len(theta_t_cols),
    "n_A_prefixed": len(theta_a_cols),
    "theta_view_contains_alexander_columns": bool(theta_a_cols),
    "decision": (
        "theta_component_only"
        if len(theta_t_cols) == len(theta_cols) and not theta_a_cols
        else "REVIEW_REQUIRED"
    ),
}
pd.DataFrame([theta_audit]).to_csv(OUT / "p4_theta_audit.csv", index=False)

# ---------------------------------------------------------------------------
# P6 — sign convention
# ---------------------------------------------------------------------------
s = pd.to_numeric(meta[S_COL], errors="coerce").to_numpy(float)
sigma = pd.to_numeric(meta["signature"], errors="coerce").to_numpy(float)
finite = np.isfinite(s) & np.isfinite(sigma)
both_nonzero = finite & (s != 0) & (sigma != 0)
same_sign = np.sign(s) == np.sign(sigma)

g_difference = np.abs(s - sigma)
g_absolute_values = np.abs(np.abs(s) - np.abs(sigma))
g_equal = np.isclose(g_difference, g_absolute_values, rtol=0.0, atol=1e-12)

sign_mismatch = both_nonzero & ~same_sign
g_mismatch = finite & ~g_equal
alternating = meta["is_alternating"].to_numpy(int) == 1

sign_summary = pd.DataFrame(
    [
        {
            "n_total": len(meta),
            "n_finite_s_sigma": int(finite.sum()),
            "n_both_nonzero": int(both_nonzero.sum()),
            "n_same_sign_when_nonzero": int((both_nonzero & same_sign).sum()),
            "n_sign_mismatch_when_nonzero": int(sign_mismatch.sum()),
            "n_G_definitions_differ": int(g_mismatch.sum()),
            "all_nonzero_same_sign": bool(sign_mismatch.sum() == 0),
            "G_abs_difference_equals_abs_abs_everywhere": bool(g_mismatch.sum() == 0),
            "alternating_n": int(alternating.sum()),
            "alternating_s_equals_sigma": int(
                (alternating & finite & np.isclose(s, sigma, atol=1e-12)).sum()
            ),
            "alternating_s_equals_minus_sigma": int(
                (alternating & finite & np.isclose(s, -sigma, atol=1e-12)).sum()
            ),
            "alternating_abs_s_equals_abs_sigma": int(
                (
                    alternating
                    & finite
                    & np.isclose(np.abs(s), np.abs(sigma), atol=1e-12)
                ).sum()
            ),
        }
    ]
)
sign_summary.to_csv(OUT / "p6_sign_convention_summary.csv", index=False)

cols = [ID_COL, "number_of_crossings", "is_alternating", "signature", S_COL]
mismatch_table = meta.loc[sign_mismatch | g_mismatch, cols].copy()
mismatch_table["G_abs_s_minus_sigma"] = g_difference[sign_mismatch | g_mismatch]
mismatch_table["G_abs_abs_s_minus_abs_sigma"] = g_absolute_values[
    sign_mismatch | g_mismatch
]
mismatch_table.to_csv(OUT / "p6_sign_or_G_mismatches.csv", index=False)

if sign_mismatch.sum() == 0 and g_mismatch.sum() == 0:
    p6_decision = r"""PASS.

The archived catalogue records $s$ and $\sigma$ under a common sign convention
on this aligned universe. We verified that
$\operatorname{sign}(s_{\mathrm{QC}})=\operatorname{sign}(\sigma)$ whenever both
are nonzero, and consequently
$|s_{\mathrm{QC}}-\sigma|=\bigl||s_{\mathrm{QC}}|-|\sigma|\bigr|$ for every
aligned knot.

No gap endpoint needs to be recomputed.
"""
else:
    p6_decision = f"""REANALYSIS REQUIRED.

Found {int(sign_mismatch.sum())} nonzero sign mismatches and
{int(g_mismatch.sum())} rows on which |s-sigma| differs from ||s|-|sigma||.

Do NOT insert the proposed sign-convention paragraph.
Redefine the paper endpoint as G=||s_QC|-|sigma|| and rerun every G-based
descriptive/null/bootstrap result.
"""
(OUT / "p6_decision.txt").write_text(p6_decision, encoding="utf-8")

# ---------------------------------------------------------------------------
# P8 — Khovanov representation actually loaded
# ---------------------------------------------------------------------------
kh_cols = [str(c) for c in feature_cols_dict["Khovanov"]]
f_cols = [c for c in kh_cols if c.startswith("F_")]
coord_re = re.compile(r"F_q(-?\d+)_t(-?\d+)$")
unparsed = [c for c in kh_cols if coord_re.fullmatch(c) is None]

# Inspect the source table too when paper_run retained it.
source_coordinate_families = {}
if "aligned_tables" in globals() and isinstance(aligned_tables, dict):
    kh_source = None
    for key in ("kh", "Khovanov", "khovanov"):
        if key in aligned_tables:
            kh_source = aligned_tables[key]
            break
    if kh_source is not None:
        # Coordinate-like columns: capture text before q..._t...
        fam_re = re.compile(r"^(.+?)q-?\d+_t-?\d+$")
        for col in map(str, kh_source.columns):
            m = fam_re.match(col)
            if m:
                source_coordinate_families[m.group(1)] = (
                    source_coordinate_families.get(m.group(1), 0) + 1
                )

kh_audit = pd.DataFrame(
    [
        {
            "n_loaded_khovanov_features": len(kh_cols),
            "n_loaded_F_features": len(f_cols),
            "all_loaded_features_are_F_coordinates": len(f_cols) == len(kh_cols),
            "n_unparsed_loaded_coordinates": len(unparsed),
            "operational_consequence": (
                "Current support/diagonal counts use only F_ coordinates; "
                "torsion-only coordinate families cannot affect the reported count."
            ),
            "source_coordinate_families_if_available": json.dumps(
                source_coordinate_families, sort_keys=True
            ),
        }
    ]
)
kh_audit.to_csv(OUT / "p8_khovanov_coordinate_audit.csv", index=False)
pd.DataFrame({"loaded_khovanov_feature": kh_cols}).to_csv(
    OUT / "p8_loaded_khovanov_features.csv", index=False
)
if unparsed:
    pd.DataFrame({"unparsed": unparsed}).to_csv(
        OUT / "p8_unparsed_khovanov_features.csv", index=False
    )

# ---------------------------------------------------------------------------
# P9 / P12 — universe and dimensions
# ---------------------------------------------------------------------------
crossing_counts = (
    meta["number_of_crossings"]
    .value_counts()
    .sort_index()
    .rename_axis("crossings")
    .reset_index(name="n")
)
crossing_counts.to_csv(OUT / "p9_crossing_counts.csv", index=False)

dimension_rows = []
for name in INVARIANTS:
    dimension_rows.append(
        {
            "representation": name,
            "input_dimension": int(np.asarray(X_dict[name]).shape[1]),
            "feature_columns": int(len(feature_cols_dict[name])),
        }
    )
dimensions = pd.DataFrame(dimension_rows)
dimensions.to_csv(OUT / "p12_representation_dimensions.csv", index=False)

universe = pd.DataFrame(
    [
        {
            "n_aligned": len(meta),
            "expected_n": int(CONFIG.universe.expected_n),
            "contains_unknot_id_00_1": bool(
                meta[ID_COL].astype(str).eq("00_1").any()
            ),
            "min_crossings": int(meta["number_of_crossings"].min()),
            "max_crossings": int(meta["number_of_crossings"].max()),
        }
    ]
)
universe.to_csv(OUT / "p9_universe_audit.csv", index=False)

# ---------------------------------------------------------------------------
# Compact decision table
# ---------------------------------------------------------------------------
decision = pd.DataFrame(
    [
        {
            "item": "P4 Theta",
            "status": "PASS" if theta_audit["decision"] == "theta_component_only" else "CHECK",
            "result": (
                "Theta matrix contains only T-prefixed theta coordinates and no "
                "Alexander A-prefixed coordinates."
                if theta_audit["decision"] == "theta_component_only"
                else "Theta feature schema is not theta-only."
            ),
        },
        {
            "item": "P6 sign convention",
            "status": "PASS" if sign_mismatch.sum() == 0 and g_mismatch.sum() == 0 else "RERUN",
            "result": f"sign mismatches={int(sign_mismatch.sum())}; G-definition mismatches={int(g_mismatch.sum())}",
        },
        {
            "item": "P8 Khovanov",
            "status": "PASS_SCHEMA" if len(f_cols) == len(kh_cols) and not unparsed else "CHECK",
            "result": (
                f"{len(f_cols)}/{len(kh_cols)} loaded coordinates are F_q*_t*. "
                "Current diagonal count therefore uses F_ support only."
            ),
        },
        {
            "item": "P9 universe",
            "status": "PASS" if len(meta) == CONFIG.universe.expected_n and not meta[ID_COL].astype(str).eq("00_1").any() else "CHECK",
            "result": f"N={len(meta)}; unknot present={meta[ID_COL].astype(str).eq('00_1').any()}",
        },
        {
            "item": "P12 dimensions",
            "status": "PASS",
            "result": ", ".join(
                f"{r.representation}={r.input_dimension}"
                for r in dimensions.itertuples()
            ),
        },
    ]
)
decision.to_csv(OUT / "review_definition_decision_table.csv", index=False)

print("\nDefinition/provenance audit:")
print(decision.to_string(index=False))
print("\nP6:")
print(sign_summary.to_string(index=False))
print("\nP8 source coordinate families (if aligned_tables was available):")
print(source_coordinate_families)
print("\nSaved Stage 28 to:", OUT)