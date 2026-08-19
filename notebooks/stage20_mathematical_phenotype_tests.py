# %% [markdown]
# Stage 20 — Mathematical phenotype / competing-explanation tests
#
# Tests whether the hard regimes are primarily recovering familiar structure:
# Alexander monicity/support breadth, Khovanov diagonal thickness, and the
# KnotInfo positivity/fibered annotations where available.  Primary inference
# reuses the already-frozen unique-control matching design.

# %%
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from statsmodels.stats.multitest import multipletests


# ------------------------------------------------------------------
# 0. Preconditions
# ------------------------------------------------------------------
REQUIRED = (
    "meta",
    "X_dict",
    "feature_cols_dict",
    "primary",
    "conditional_membership_sets",
    "geometry_unique_pairs",
    "CONFIG",
    "OUTPUT_DIR",
    "ch",
)
missing = [name for name in REQUIRED if name not in globals()]
if missing:
    raise RuntimeError(
        "Run the frozen notebook first. Missing objects: " + str(missing)
    )

OUT = Path(OUTPUT_DIR) / "20_mathematical_phenotype"
OUT.mkdir(parents=True, exist_ok=True)

N = len(meta)
ID_COL = CONFIG.universe.id_col
TOL = 1e-12


# ------------------------------------------------------------------
# 1. Frozen regime labels
# ------------------------------------------------------------------
raw_set = set(map(int, primary["consensus"]))
conditional_set = set(map(
    int,
    conditional_membership_sets[("All 5", 3)],
))

raw_mask = np.zeros(N, dtype=bool)
conditional_mask = np.zeros(N, dtype=bool)
raw_mask[list(raw_set)] = True
conditional_mask[list(conditional_set)] = True

regime = np.full(N, "neither", dtype=object)
regime[raw_mask & ~conditional_mask] = "amplitude_hard_only"
regime[~raw_mask & conditional_mask] = "conditional_hard_only"
regime[raw_mask & conditional_mask] = "shared_raw_and_conditional"

phenotype = meta[[
    ID_COL,
    "number_of_crossings",
    "is_alternating",
    "signature",
    CONFIG.s_col,
]].copy()
phenotype["hard_regime"] = regime
phenotype["delta_abs"] = (
    phenotype[CONFIG.s_col] - phenotype["signature"]
).abs()


# ------------------------------------------------------------------
# 2. Full-coverage Alexander proxies
# ------------------------------------------------------------------
alex = np.asarray(X_dict["Alexander"])
alex_active = np.abs(alex) > TOL
alex_support_size = alex_active.sum(axis=1).astype(np.int16)

has_support = alex_support_size > 0
first = np.argmax(alex_active, axis=1)
last = alex.shape[1] - 1 - np.argmax(alex_active[:, ::-1], axis=1)
alex_breadth = np.where(has_support, last - first, 0).astype(np.int16)

row_index = np.arange(N)
first_coeff = alex[row_index, first]
last_coeff = alex[row_index, last]
alex_monic = (
    has_support
    & np.isclose(np.abs(first_coeff), 1.0)
    & np.isclose(np.abs(last_coeff), 1.0)
)

phenotype["alexander_support_size"] = alex_support_size
phenotype["alexander_support_breadth"] = alex_breadth
phenotype["alexander_monic_proxy"] = alex_monic
phenotype["alexander_breadth_minus_abs_signature"] = (
    alex_breadth - phenotype["signature"].abs().to_numpy()
)


# ------------------------------------------------------------------
# 3. Full-coverage Khovanov support/diagonal proxies, computed in chunks
# ------------------------------------------------------------------
kh_cols = list(feature_cols_dict["Khovanov"])


def parse_kh_coordinate(column):
    match = re.fullmatch(r"F_q(-?\d+)_t(-?\d+)", str(column))
    if match is None:
        raise ValueError(f"Cannot parse Khovanov coordinate: {column}")
    return tuple(map(int, match.groups()))


kh_coords = [parse_kh_coordinate(col) for col in kh_cols]
kh_delta = np.asarray([q - 2 * t for q, t in kh_coords], dtype=np.int32)
unique_delta = np.unique(kh_delta)
delta_groups = [np.flatnonzero(kh_delta == d) for d in unique_delta]

kh_support_size = np.empty(N, dtype=np.int16)
kh_diagonal_count = np.empty(N, dtype=np.int16)
KH_CHUNK = 20_000

for start in range(0, N, KH_CHUNK):
    stop = min(start + KH_CHUNK, N)
    active = np.abs(np.asarray(X_dict["Khovanov"])[start:stop]) > TOL
    kh_support_size[start:stop] = active.sum(axis=1).astype(np.int16)
    diagonal_count = np.zeros(stop - start, dtype=np.int16)
    for indices in delta_groups:
        diagonal_count += active[:, indices].any(axis=1)
    kh_diagonal_count[start:stop] = diagonal_count
    print(f"Khovanov phenotype: {stop:,} / {N:,}")

phenotype["khovanov_support_size"] = kh_support_size
phenotype["khovanov_q_minus_2t_diagonal_count"] = kh_diagonal_count


# ------------------------------------------------------------------
# 4. Descriptive regime table
# ------------------------------------------------------------------
def q25(series):
    return series.quantile(0.25)


def q75(series):
    return series.quantile(0.75)


mathematical_phenotype_by_regime = (
    phenotype
    .groupby("hard_regime", as_index=False)
    .agg(
        n=(ID_COL, "size"),
        alexander_monic_prop=("alexander_monic_proxy", "mean"),
        alexander_breadth_median=("alexander_support_breadth", "median"),
        alexander_breadth_q25=("alexander_support_breadth", q25),
        alexander_breadth_q75=("alexander_support_breadth", q75),
        alexander_slack_median=(
            "alexander_breadth_minus_abs_signature", "median"
        ),
        kh_support_median=("khovanov_support_size", "median"),
        kh_diagonal_count_median=(
            "khovanov_q_minus_2t_diagonal_count", "median"
        ),
        kh_diagonal_count_mean=(
            "khovanov_q_minus_2t_diagonal_count", "mean"
        ),
    )
)
display(mathematical_phenotype_by_regime)


# ------------------------------------------------------------------
# 5. Paired inference using the frozen unique-control designs
# ------------------------------------------------------------------
CONTINUOUS = (
    ("Alexander support breadth", "alexander_support_breadth"),
    (
        "Alexander breadth minus absolute signature",
        "alexander_breadth_minus_abs_signature",
    ),
    ("Khovanov support size", "khovanov_support_size"),
    (
        "Khovanov q-2t diagonal count",
        "khovanov_q_minus_2t_diagonal_count",
    ),
)
BINARY = (
    ("Alexander monic proxy", "alexander_monic_proxy"),
)


def paired_continuous(regime_name, pairs, label, column):
    selected = phenotype.loc[pairs["selected_index"], column].to_numpy(float)
    control = phenotype.loc[pairs["control_index"], column].to_numpy(float)
    difference = selected - control
    nonzero = difference[~np.isclose(difference, 0.0)]
    p_value = (
        float(wilcoxon(nonzero, alternative="two-sided").pvalue)
        if len(nonzero) else 1.0
    )
    pooled_sd = np.sqrt((np.var(selected, ddof=1) + np.var(control, ddof=1)) / 2)
    smd = float(np.mean(difference) / pooled_sd) if pooled_sd > 0 else 0.0
    return {
        "regime": regime_name,
        "outcome": label,
        "outcome_type": "continuous",
        "n_valid_pairs": len(difference),
        "selected_mean": float(np.mean(selected)),
        "control_mean": float(np.mean(control)),
        "mean_paired_difference": float(np.mean(difference)),
        "median_paired_difference": float(np.median(difference)),
        "paired_smd": smd,
        "p_value": p_value,
        "test": "paired Wilcoxon",
    }


def paired_binary(regime_name, pairs, label, column, frame=phenotype):
    selected = frame.loc[pairs["selected_index"], column].astype(bool).to_numpy()
    control = frame.loc[pairs["control_index"], column].astype(bool).to_numpy()
    selected_yes_control_no = int(np.sum(selected & ~control))
    selected_no_control_yes = int(np.sum(~selected & control))
    discordant = selected_yes_control_no + selected_no_control_yes
    p_value = (
        float(binomtest(
            selected_yes_control_no,
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue)
        if discordant else 1.0
    )
    return {
        "regime": regime_name,
        "outcome": label,
        "outcome_type": "binary",
        "n_valid_pairs": len(selected),
        "selected_prop": float(np.mean(selected)),
        "control_prop": float(np.mean(control)),
        "risk_difference": float(np.mean(selected) - np.mean(control)),
        "selected_yes_control_no": selected_yes_control_no,
        "selected_no_control_yes": selected_no_control_yes,
        "p_value": p_value,
        "test": "exact paired McNemar",
    }


paired_rows = []
for regime_name in ("amplitude_only", "conditional_only"):
    if regime_name not in geometry_unique_pairs:
        raise KeyError(f"Missing frozen unique pairs for {regime_name}")
    pairs = geometry_unique_pairs[regime_name].copy()
    for label, column in CONTINUOUS:
        paired_rows.append(paired_continuous(regime_name, pairs, label, column))
    for label, column in BINARY:
        paired_rows.append(paired_binary(regime_name, pairs, label, column))

matched_mathematical_phenotype = pd.DataFrame(paired_rows)


# ------------------------------------------------------------------
# 6. KnotInfo falsification subset: fibered / positivity / adequacy
# ------------------------------------------------------------------
KNOTINFO_PROPERTIES = (
    "fibered",
    "positive_braid",
    "positive",
    "strongly_quasipositive",
    "quasipositive",
    "adequate",
    "quasi_alternating",
    "almost_alternating",
    "l_space",
)


def parse_yes_no(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"y", "yes", "true", "1", "t"}:
        return 1.0
    if text in {"n", "no", "false", "0", "f"}:
        return 0.0
    return np.nan


knotinfo_rows = []
if "atlas_knotinfo" in globals():
    available_props = [
        col for col in KNOTINFO_PROPERTIES if col in atlas_knotinfo.columns
    ]
    annotation = (
        atlas_knotinfo[[ID_COL] + available_props]
        .drop_duplicates(subset=[ID_COL])
        .set_index(ID_COL)
    )
    for prop in available_props:
        annotation[prop] = annotation[prop].map(parse_yes_no)

    id_by_index = phenotype[ID_COL]
    for regime_name in ("amplitude_only", "conditional_only"):
        pairs = geometry_unique_pairs[regime_name].copy()
        pairs["selected_id"] = pairs["selected_index"].map(id_by_index)
        pairs["control_id"] = pairs["control_index"].map(id_by_index)

        for prop in available_props:
            selected_value = pairs["selected_id"].map(annotation[prop])
            control_value = pairs["control_id"].map(annotation[prop])
            valid = selected_value.notna() & control_value.notna()
            n_valid = int(valid.sum())
            if n_valid == 0:
                knotinfo_rows.append({
                    "regime": regime_name,
                    "property": prop,
                    "n_total_pairs": len(pairs),
                    "n_valid_pairs": 0,
                    "coverage": 0.0,
                    "selected_prop": np.nan,
                    "control_prop": np.nan,
                    "risk_difference": np.nan,
                    "p_value": np.nan,
                    "test": "insufficient coverage",
                })
                continue

            selected = selected_value[valid].astype(bool).to_numpy()
            control = control_value[valid].astype(bool).to_numpy()
            a = int(np.sum(selected & ~control))
            b = int(np.sum(~selected & control))
            discordant = a + b
            p_value = (
                float(binomtest(a, discordant, 0.5).pvalue)
                if discordant else 1.0
            )
            knotinfo_rows.append({
                "regime": regime_name,
                "property": prop,
                "n_total_pairs": len(pairs),
                "n_valid_pairs": n_valid,
                "coverage": n_valid / len(pairs),
                "selected_prop": float(np.mean(selected)),
                "control_prop": float(np.mean(control)),
                "risk_difference": float(np.mean(selected) - np.mean(control)),
                "selected_yes_control_no": a,
                "selected_no_control_yes": b,
                "p_value": p_value,
                "test": "exact paired McNemar",
            })
else:
    print("atlas_knotinfo not present; skipping KnotInfo subset tests.")

knotinfo_phenotype_tests = pd.DataFrame(knotinfo_rows)


# ------------------------------------------------------------------
# 7. Multiple-testing labels and decisions
# ------------------------------------------------------------------
def add_multiplicity(frame):
    frame = frame.copy()
    valid = frame["p_value"].notna()
    frame["q_bh"] = np.nan
    frame["p_holm"] = np.nan
    if valid.any():
        p = frame.loc[valid, "p_value"].to_numpy(float)
        frame.loc[valid, "q_bh"] = multipletests(p, method="fdr_bh")[1]
        frame.loc[valid, "p_holm"] = multipletests(p, method="holm")[1]
    frame["holm_significant"] = frame["p_holm"] < 0.05
    return frame


matched_mathematical_phenotype = add_multiplicity(
    matched_mathematical_phenotype
)
if len(knotinfo_phenotype_tests):
    knotinfo_phenotype_tests = add_multiplicity(knotinfo_phenotype_tests)

display(matched_mathematical_phenotype)
display(knotinfo_phenotype_tests)


# ------------------------------------------------------------------
# 8. Save
# ------------------------------------------------------------------
phenotype.to_parquet(OUT / "complete_mathematical_phenotype_atlas.parquet")
mathematical_phenotype_by_regime.to_csv(
    OUT / "mathematical_phenotype_by_regime.csv", index=False
)
matched_mathematical_phenotype.to_csv(
    OUT / "matched_mathematical_phenotype_tests.csv", index=False
)
knotinfo_phenotype_tests.to_csv(
    OUT / "matched_knotinfo_phenotype_tests.csv", index=False
)

print("\nSaved Stage 20 to:")
print(OUT)

