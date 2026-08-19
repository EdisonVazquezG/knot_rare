"""Stage 25: score-specific matched geometry on the held-out test split.

This stage answers the MLST baseline question: after fixing the selected-set
size, do standard anomaly scores identify populations with different geometric
correlates?  For each of the five All-five C3 score rules (n=60), it constructs
one unique control per selected knot, exactly matching crossing number,
alternation status, and signature, and nearest-matching the mean and maximum
log squared representation norms with caliper 0.50.

Inputs are frozen Stage 17/23B artifacts.  SnapPy results are checkpointed and
the stage is safe to resume after a Colab disconnect.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest, wilcoxon
from statsmodels.stats.multitest import multipletests


ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants/"
    "processed_consensus_hardness/corrected_run_20260819"
)
OUT = ROOT / "25_score_matched_geometry"
OUT.mkdir(parents=True, exist_ok=True)

ATLAS_PATH = ROOT / "17_final_paper_outputs/final_hard_regime_atlas.parquet"
SELECTED_PATH = (
    ROOT
    / "23B_size_matched_score_sensitivity/size_matched_selected_ids.csv"
)
CALIPER = 0.50
FAMILY = "All 5 C3"
ID_COL = "knot_id_base"
EXACT_COLS = ["number_of_crossings", "is_alternating", "signature"]
NORM_COLS = ["mean_log_sq_norm", "max_log_sq_norm"]
METHODS = [
    "raw_sse",
    "relative_nre",
    "residual_mahalanobis",
    "residual_isolation_forest",
    "conditional_percentile_100",
]
METHOD_LABELS = {
    "raw_sse": "Raw SSE",
    "relative_nre": "Relative error",
    "residual_mahalanobis": "Residual Mahalanobis",
    "residual_isolation_forest": "Residual isolation forest",
    "conditional_percentile_100": "Conditional percentile",
}


def find_test_indices() -> np.ndarray:
    """Recover the frozen held-out indices from an existing checkpoint."""
    candidates = sorted(ROOT.rglob("heldout_ae_seed_0.npz"))
    candidates += sorted(ROOT.rglob("*_test_scores.npz"))
    for path in candidates:
        try:
            with np.load(path, allow_pickle=True) as z:
                for key in ("test_idx", "indices", "row_indices"):
                    if key in z.files:
                        arr = np.asarray(z[key], dtype=int)
                        if len(arr) == 46985:
                            print(f"Frozen test indices: {path}")
                            return arr
        except Exception:
            continue
    raise FileNotFoundError(
        "No encuentro un checkpoint con test_idx. Run Stage 23/23B first."
    )


def standardized_norms(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    zcols = []
    for col in NORM_COLS:
        sd = float(out[col].std(ddof=0))
        if not np.isfinite(sd) or sd == 0:
            raise RuntimeError(f"Invalid standard deviation for {col}: {sd}")
        zcol = f"{col}_z"
        out[zcol] = (out[col] - float(out[col].mean())) / sd
        zcols.append(zcol)
    return out, zcols


def unique_exact_caliper_match(
    selected: pd.DataFrame,
    controls: pd.DataFrame,
    zcols: list[str],
    caliper: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optimal 1:1 unique matching within every exact stratum.

    Dummy columns allow selected knots without common support to remain
    unmatched.  A valid match always costs less than a dummy assignment.
    """
    pairs: list[dict] = []
    unmatched: list[dict] = []

    selected = selected.sort_values(ID_COL, kind="mergesort")
    controls = controls.sort_values(ID_COL, kind="mergesort")
    for stratum, sel_g in selected.groupby(EXACT_COLS, dropna=False, sort=True):
        if not isinstance(stratum, tuple):
            stratum = (stratum,)
        mask = np.ones(len(controls), dtype=bool)
        for col, value in zip(EXACT_COLS, stratum):
            if pd.isna(value):
                mask &= controls[col].isna().to_numpy()
            else:
                mask &= controls[col].eq(value).to_numpy()
        ctl_g = controls.loc[mask]
        if ctl_g.empty:
            for _, row in sel_g.iterrows():
                unmatched.append({ID_COL: row[ID_COL], "reason": "empty exact stratum"})
            continue

        a = sel_g[zcols].to_numpy(float)
        b = ctl_g[zcols].to_numpy(float)
        distances = np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2))
        n_sel, n_ctl = distances.shape
        cost = np.full((n_sel, n_ctl + n_sel), 1.0e6, dtype=float)
        valid = distances <= caliper
        cost[:, :n_ctl][valid] = distances[valid]
        for i in range(n_sel):
            cost[i, n_ctl + i] = caliper + 1.0e-6

        rows, cols = linear_sum_assignment(cost)
        sel_index = list(sel_g.index)
        ctl_index = list(ctl_g.index)
        assigned = {}
        for i, j in zip(rows, cols):
            assigned[i] = j
            if j < n_ctl and distances[i, j] <= caliper:
                srow = sel_g.loc[sel_index[i]]
                crow = ctl_g.loc[ctl_index[j]]
                pairs.append(
                    {
                        "selected_id": srow[ID_COL],
                        "control_id": crow[ID_COL],
                        "match_distance": float(distances[i, j]),
                        **{col: srow[col] for col in EXACT_COLS},
                    }
                )
            else:
                srow = sel_g.loc[sel_index[i]]
                unmatched.append(
                    {ID_COL: srow[ID_COL], "reason": "no unique control within caliper"}
                )
    return pd.DataFrame(pairs), pd.DataFrame(unmatched)


def snappy_name_candidates(knot_id: str) -> list[str]:
    raw = str(knot_id).strip()
    compact = raw.replace("_", "")
    candidates: list[str] = []
    m = re.fullmatch(r"0*(\d+)([an])(\d+)", compact, flags=re.I)
    if m:
        crossing, kind, rank = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        candidates.extend([f"K{crossing}{kind}{rank}", f"{crossing}{kind}{rank}"])
    m = re.fullmatch(r"0*(\d+)_?0*(\d+)", raw)
    if m:
        crossing, rank = int(m.group(1)), int(m.group(2))
        candidates.extend([f"{crossing}_{rank}", f"K{crossing}_{rank}"])
    candidates.extend([raw, compact, f"K{compact}"])
    return list(dict.fromkeys(candidates))


def compute_snappy_row(knot_id: str) -> dict:
    import snappy

    try:
        import snappy_15_knots  # noqa: F401  (registers the 15-crossing census)
    except Exception:
        pass

    last_error = None
    for name in snappy_name_candidates(knot_id):
        try:
            manifold = snappy.Manifold(name)
            solution_type = str(manifold.solution_type())
            positive = "positively oriented" in solution_type.lower()
            volume_raw = float(manifold.volume())
            volume = volume_raw if positive and np.isfinite(volume_raw) else np.nan
            tetrahedra = float(manifold.num_tetrahedra())
            try:
                symmetry_order = float(manifold.symmetry_group().order())
            except Exception:
                symmetry_order = np.nan
            return {
                ID_COL: knot_id,
                "snappy_name": name,
                "snappy_loaded": True,
                "solution_type": solution_type,
                "numerical_positive": positive,
                "volume": volume,
                "num_tetrahedra": tetrahedra,
                "symmetry_order": symmetry_order,
                "nontrivial_symmetry": (
                    float(symmetry_order > 1) if np.isfinite(symmetry_order) else np.nan
                ),
                "error": np.nan,
            }
        except Exception as exc:
            last_error = repr(exc)
    return {
        ID_COL: knot_id,
        "snappy_name": np.nan,
        "snappy_loaded": False,
        "solution_type": np.nan,
        "numerical_positive": np.nan,
        "volume": np.nan,
        "num_tetrahedra": np.nan,
        "symmetry_order": np.nan,
        "nontrivial_symmetry": np.nan,
        "error": last_error,
    }


def paired_smd(diff: np.ndarray) -> float:
    sd = float(np.std(diff, ddof=1))
    return float(np.mean(diff) / sd) if np.isfinite(sd) and sd > 0 else np.nan


def continuous_result(method: str, outcome: str, x: np.ndarray, y: np.ndarray) -> dict:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    diff = x - y
    if len(diff) == 0:
        p = np.nan
    elif np.allclose(diff, 0):
        p = 1.0
    else:
        p = float(wilcoxon(diff, alternative="two-sided").pvalue)
    return {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "outcome": outcome,
        "outcome_type": "continuous",
        "n_valid_pairs": len(diff),
        "selected_mean": float(np.mean(x)) if len(x) else np.nan,
        "control_mean": float(np.mean(y)) if len(y) else np.nan,
        "mean_paired_difference": float(np.mean(diff)) if len(diff) else np.nan,
        "median_paired_difference": float(np.median(diff)) if len(diff) else np.nan,
        "paired_smd": paired_smd(diff),
        "selected_prop": np.nan,
        "control_prop": np.nan,
        "risk_difference": np.nan,
        "discordant_selected_yes": np.nan,
        "discordant_control_yes": np.nan,
        "p_value": p,
        "test": "paired Wilcoxon",
    }


def binary_result(method: str, outcome: str, x: np.ndarray, y: np.ndarray) -> dict:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid].astype(int), y[valid].astype(int)
    b = int(np.sum((x == 1) & (y == 0)))
    c = int(np.sum((x == 0) & (y == 1)))
    p = float(binomtest(b, b + c, p=0.5).pvalue) if b + c else 1.0
    return {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "outcome": outcome,
        "outcome_type": "binary",
        "n_valid_pairs": len(x),
        "selected_mean": np.nan,
        "control_mean": np.nan,
        "mean_paired_difference": np.nan,
        "median_paired_difference": np.nan,
        "paired_smd": np.nan,
        "selected_prop": float(np.mean(x)) if len(x) else np.nan,
        "control_prop": float(np.mean(y)) if len(y) else np.nan,
        "risk_difference": float(np.mean(x - y)) if len(x) else np.nan,
        "discordant_selected_yes": b,
        "discordant_control_yes": c,
        "p_value": p,
        "test": "exact paired McNemar",
    }


# ---------------------------------------------------------------------------
# 1. Frozen test population and equal-size score selections
# ---------------------------------------------------------------------------
assert ATLAS_PATH.exists(), ATLAS_PATH
assert SELECTED_PATH.exists(), SELECTED_PATH
atlas = pd.read_parquet(ATLAS_PATH).reset_index(drop=True)
test_idx = find_test_indices()
test = atlas.iloc[test_idx].copy().reset_index(drop=True)
required = [ID_COL, *EXACT_COLS, *NORM_COLS]
missing = [col for col in required if col not in test.columns]
if missing:
    raise KeyError(f"Missing atlas columns: {missing}")
test[ID_COL] = test[ID_COL].astype(str)
test, ZCOLS = standardized_norms(test)

selected_ids = pd.read_csv(SELECTED_PATH, dtype={ID_COL: str})
selected_ids = selected_ids.loc[selected_ids["family"].eq(FAMILY)].copy()
counts = selected_ids.groupby("method")[ID_COL].nunique().reindex(METHODS)
print("Size-matched selected counts:\n", counts.to_string())
if counts.isna().any() or counts.nunique() != 1:
    raise RuntimeError("The five primary score sets must have the same size.")

# Common control universe prevents one score's controls from being another
# score's discoveries and makes the five comparisons directly comparable.
union_selected = set(selected_ids[ID_COL])
control_pool = test.loc[~test[ID_COL].isin(union_selected)].copy()

all_pairs = []
all_unmatched = []
design_rows = []
for method in METHODS:
    ids = set(selected_ids.loc[selected_ids["method"].eq(method), ID_COL])
    selected = test.loc[test[ID_COL].isin(ids)].copy()
    if len(selected) != len(ids):
        missing_ids = sorted(ids - set(selected[ID_COL]))[:10]
        raise RuntimeError(f"Selected IDs absent from test atlas: {missing_ids}")
    pairs, unmatched = unique_exact_caliper_match(selected, control_pool, ZCOLS, CALIPER)
    pairs.insert(0, "method", method)
    unmatched.insert(0, "method", method)
    all_pairs.append(pairs)
    all_unmatched.append(unmatched)

    if len(pairs):
        paired = pairs.merge(
            test[[ID_COL, *NORM_COLS]], left_on="selected_id", right_on=ID_COL
        ).rename(columns={c: f"selected_{c}" for c in NORM_COLS})
        paired = paired.merge(
            test[[ID_COL, *NORM_COLS]], left_on="control_id", right_on=ID_COL,
            suffixes=("", "_control"),
        ).rename(columns={c: f"control_{c}" for c in NORM_COLS})
        smds = []
        for col in NORM_COLS:
            d = paired[f"selected_{col}"].to_numpy() - paired[f"control_{col}"].to_numpy()
            smds.append(paired_smd(d))
        max_abs_smd = float(np.nanmax(np.abs(smds)))
    else:
        max_abs_smd = np.nan
    design_rows.append(
        {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "caliper": CALIPER,
            "n_selected_total": len(selected),
            "n_selected_matched": len(pairs),
            "coverage": len(pairs) / len(selected),
            "mean_match_distance": pairs["match_distance"].mean() if len(pairs) else np.nan,
            "median_match_distance": pairs["match_distance"].median() if len(pairs) else np.nan,
            "max_abs_paired_norm_smd": max_abs_smd,
        }
    )

pairs = pd.concat(all_pairs, ignore_index=True)
unmatched = pd.concat(all_unmatched, ignore_index=True)
design = pd.DataFrame(design_rows)
pairs.to_csv(OUT / "score_geometry_pairs.csv", index=False)
unmatched.to_csv(OUT / "score_geometry_unmatched.csv", index=False)
design.to_csv(OUT / "score_matched_design_summary.csv", index=False)
print("\nMatched designs:\n", design.to_string(index=False))


# ---------------------------------------------------------------------------
# 2. SnapPy checkpoint for every knot used by the matched comparisons
# ---------------------------------------------------------------------------
needed_ids = sorted(set(pairs["selected_id"]) | set(pairs["control_id"]))
checkpoint = OUT / "score_geometry_snappy_checkpoint.csv"
if checkpoint.exists():
    geometry = pd.read_csv(checkpoint, dtype={ID_COL: str})
else:
    geometry = pd.DataFrame()
done = set(geometry[ID_COL]) if ID_COL in geometry else set()
remaining = [k for k in needed_ids if k not in done]
print(f"\nUnique SnapPy knots: {len(needed_ids)}; remaining: {len(remaining)}")

new_rows = []
for i, knot_id in enumerate(remaining, start=1):
    new_rows.append(compute_snappy_row(knot_id))
    if i % 25 == 0 or i == len(remaining):
        block = pd.DataFrame(new_rows)
        geometry = pd.concat([geometry, block], ignore_index=True)
        geometry = geometry.drop_duplicates(ID_COL, keep="last")
        geometry.to_csv(checkpoint, index=False)
        new_rows = []
        print(f"SnapPy checkpoint: {i}/{len(remaining)}")

if geometry.empty or not set(needed_ids).issubset(set(geometry[ID_COL])):
    raise RuntimeError("Incomplete SnapPy checkpoint.")
geometry.to_csv(OUT / "score_geometry_lookup.csv", index=False)
print("SnapPy load status:\n", geometry["snappy_loaded"].value_counts(dropna=False))


# ---------------------------------------------------------------------------
# 3. Paired outcomes and multiplicity correction across 15 comparisons
# ---------------------------------------------------------------------------
paired_geometry = pairs.merge(
    geometry.add_prefix("selected_"), left_on="selected_id", right_on=f"selected_{ID_COL}", how="left"
).merge(
    geometry.add_prefix("control_"), left_on="control_id", right_on=f"control_{ID_COL}", how="left"
)
paired_geometry.to_csv(OUT / "score_geometry_pairs_with_outcomes.csv", index=False)

result_rows = []
for method in METHODS:
    sub = paired_geometry.loc[paired_geometry["method"].eq(method)]
    result_rows.append(
        continuous_result(
            method,
            "Hyperbolic volume",
            sub["selected_volume"].to_numpy(float),
            sub["control_volume"].to_numpy(float),
        )
    )
    result_rows.append(
        continuous_result(
            method,
            "Census triangulation tetrahedra",
            sub["selected_num_tetrahedra"].to_numpy(float),
            sub["control_num_tetrahedra"].to_numpy(float),
        )
    )
    result_rows.append(
        binary_result(
            method,
            "Nontrivial symmetry",
            sub["selected_nontrivial_symmetry"].to_numpy(float),
            sub["control_nontrivial_symmetry"].to_numpy(float),
        )
    )

results = pd.DataFrame(result_rows)
valid_p = results["p_value"].notna()
for correction, col in (("fdr_by", "q_by"), ("holm", "p_holm")):
    results[col] = np.nan
    results.loc[valid_p, col] = multipletests(
        results.loc[valid_p, "p_value"], method=correction
    )[1]
results["holm_significant"] = results["p_holm"].lt(0.05)
results.to_csv(OUT / "score_geometry_results.csv", index=False)
print("\nScore-specific matched geometry:\n", results.to_string(index=False))


# ---------------------------------------------------------------------------
# 4. Paper-ready effect-size figure
# ---------------------------------------------------------------------------
colors = ["#D95F02", "#4C78A8", "#7570B3", "#9C755F", "#1B9E77"]
labels = ["Raw", "Relative", "Mahalanobis", "IF", "Conditional"]
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))

continuous = results.loc[results["outcome_type"].eq("continuous")]
y = np.arange(len(METHODS))
offsets = {"Hyperbolic volume": -0.12, "Census triangulation tetrahedra": 0.12}
markers = {"Hyperbolic volume": "o", "Census triangulation tetrahedra": "s"}
for outcome in offsets:
    sub = continuous.loc[continuous["outcome"].eq(outcome)].set_index("method").reindex(METHODS)
    axes[0].scatter(
        sub["paired_smd"], y + offsets[outcome], s=72, marker=markers[outcome],
        facecolors=colors, edgecolors="black", linewidths=0.5, label=outcome,
    )
axes[0].axvline(0, color="0.35", ls="--", lw=1)
axes[0].set_yticks(y, labels)
axes[0].invert_yaxis()
axes[0].set_xlabel("Standardized paired difference")
axes[0].set_title("A  Continuous geometric outcomes", loc="left", fontweight="bold")
axes[0].legend(frameon=False, fontsize=9)

binary = results.loc[results["outcome"].eq("Nontrivial symmetry")].set_index("method").reindex(METHODS)
axes[1].scatter(
    binary["risk_difference"], y, s=82, facecolors=colors,
    edgecolors="black", linewidths=0.5,
)
axes[1].axvline(0, color="0.35", ls="--", lw=1)
axes[1].set_yticks(y, labels)
axes[1].invert_yaxis()
axes[1].set_xlabel("Paired risk difference")
axes[1].set_title("B  Nontrivial symmetry", loc="left", fontweight="bold")
for i, method in enumerate(METHODS):
    row = binary.loc[method]
    axes[1].annotate(
        f"Holm p={row['p_holm']:.3g}",
        (row["risk_difference"], i), xytext=(6, 0), textcoords="offset points",
        va="center", fontsize=8, color="0.25",
    )

fig.suptitle(
    "Matched geometric effects depend on the anomaly score",
    fontsize=16, fontweight="bold", y=1.02,
)
fig.text(
    0.5, -0.02,
    "Equal-size All-five C3 sets; unique controls exact on crossings, alternation, and signature; norm caliper 0.50.",
    ha="center", fontsize=9, color="0.3",
)
fig.tight_layout()
fig.savefig(OUT / "figure_score_matched_geometry.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "figure_score_matched_geometry.pdf", bbox_inches="tight")
plt.close(fig)

print(f"\nSaved Stage 25 to: {OUT}")
