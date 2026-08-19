# %% [markdown]
# Stage 21 — Final MLST figures, source data and LaTeX tables
#
# Run after Stages 18C, 19 and 20B have written their frozen files.  A fresh
# runtime is fine: the script can locate OUTPUT_DIR and load the final atlas
# from disk.  It does not refit a model or rerun a null.

# %%
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 0. Standalone paths and final-atlas loader
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants/"
    "processed_consensus_hardness/corrected_run_20260819"
)
ROOT = Path(globals().get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
if not ROOT.exists():
    raise FileNotFoundError(
        "Could not locate the frozen run directory. Set OUTPUT_DIR before "
        f"running this script. Tried: {ROOT}"
    )


def load_final_atlas():
    if "atlas" in globals() and isinstance(globals()["atlas"], pd.DataFrame):
        print("Using atlas already present in the notebook runtime.")
        return globals()["atlas"].copy()

    preferred = [
        ROOT / "17_final_paper_outputs" / "final_hard_regime_atlas.parquet",
        ROOT / "13_hard_regime_candidates" / "complete_hard_regime_atlas.parquet",
        ROOT / "12_hard_regime_atlas" / "hard_regime_atlas.parquet",
    ]
    for path in preferred:
        if path.exists():
            print("Loading final atlas:", path)
            return pd.read_parquet(path)

    names = {
        "final_hard_regime_atlas.parquet",
        "complete_hard_regime_atlas.parquet",
        "hard_regime_atlas.parquet",
    }
    discovered = [p for p in ROOT.rglob("*.parquet") if p.name in names]
    if discovered:
        discovered.sort(key=lambda p: ("final_" not in p.name, len(p.parts)))
        print("Loading discovered atlas:", discovered[0])
        return pd.read_parquet(discovered[0])

    raise FileNotFoundError(
        "No saved hard-regime atlas was found below OUTPUT_DIR. Expected one "
        "of final_hard_regime_atlas.parquet, complete_hard_regime_atlas.parquet "
        "or hard_regime_atlas.parquet."
    )


atlas = load_final_atlas()
S_COL = next(
    (name for name in ("s_invariant_qc", "s_invariant", "s") if name in atlas),
    None,
)
if S_COL is None:
    raise KeyError("Could not identify the Rasmussen-invariant column in atlas.")

OUT = ROOT / "21_final_mlst_artifacts"
FIG = OUT / "figures"
TAB = OUT / "tables"
SRC = OUT / "source_data"
for directory in (OUT, FIG, TAB, SRC):
    directory.mkdir(parents=True, exist_ok=True)

STAGE18C = ROOT / "18C_random_mirror_exact_nulls"
STAGE19 = ROOT / "19_conditional_heldout_validation"
STAGE20B = ROOT / "20B_no_khovanov_external_thickness"


def require(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen result: {path}\n"
            "Run the corresponding stage before Stage 21."
        )
    return path


mirror_summary = pd.read_csv(require(
    STAGE18C / "mirror_random_exact_null_summary_all_runs.csv"
))
mirror_decision = pd.read_csv(require(
    STAGE18C / "mirror_random_decision_table.csv"
))
heldout_family = pd.read_csv(require(
    STAGE19 / "conditional_heldout_family_summary.csv"
))
heldout_overlap = pd.read_csv(require(
    STAGE19 / "conditional_heldout_overlap.csv"
))
heldout_phenotype = pd.read_csv(require(
    STAGE19 / "conditional_heldout_phenotype.csv"
))
external_summary = pd.read_csv(require(
    STAGE20B / "no_khovanov_external_null_summary.csv"
))
external_decision = pd.read_csv(require(
    STAGE20B / "no_khovanov_external_decision.csv"
))


# ---------------------------------------------------------------------------
# 1. Shared visual style and helpers
# ---------------------------------------------------------------------------
COLORS = {
    "background": "#B8BEC7",
    "amplitude": "#D95F02",
    "conditional": "#1B9E77",
    "shared": "#7570B3",
    "universal": "#CC0066",
    "no_kh": "#4C78A8",
    "null": "#9E9E9E",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def panel_title(ax, letter, title):
    ax.set_title(rf"$\bf{{{letter}}}$  {title}", loc="left", pad=8)


def save_figure(fig, stem):
    fig.savefig(FIG / f"{stem}.png", dpi=350, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def pct(value, digits=1):
    return f"{100.0 * float(value):.{digits}f}"


def fnum(value, digits=3):
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def pvalue(value):
    if pd.isna(value):
        return "--"
    value = float(value)
    if np.isclose(value, 1 / 5001, rtol=1e-3):
        return r"$1/5001$"
    if value < 0.001:
        return rf"${value:.2e}$"
    if value < 0.1:
        return f"{value:.3g}"
    return f"{value:.3f}"


def latex_table(path, caption, label, columns, rows, notes=None):
    """Write a compact, self-contained LaTeX table fragment."""
    align = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
    ])
    if notes:
        lines.append(r"\begin{minipage}{0.98\textwidth}\footnotesize")
        lines.append(notes)
        lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 2. Figure 1 and Table 2 — final hard-regime overview using G=|s-sigma|
# ---------------------------------------------------------------------------
frame = atlas.copy()
required_cols = {
    "hard_regime", "number_of_crossings", "is_alternating", "signature",
    S_COL, "mean_log_sq_norm",
}
absent = sorted(required_cols - set(frame.columns))
if absent:
    raise KeyError(f"The final atlas is missing columns: {absent}")

frame["G_abs"] = (
    frame[S_COL].astype(float) - frame["signature"].astype(float)
).abs()

REGIMES = [
    ("neither", "Background", COLORS["background"]),
    ("amplitude_hard_only", "Amplitude\nonly", COLORS["amplitude"]),
    ("conditional_hard_only", "Conditional\nonly", COLORS["conditional"]),
    ("shared_raw_and_conditional", "Shared", COLORS["shared"]),
]

overview_rows = []
for key, label, _ in REGIMES:
    sub = frame.loc[frame["hard_regime"].eq(key)]
    nonalt = sub.loc[sub["is_alternating"].eq(0)]
    overview_rows.append({
        "hard_regime": key,
        "label": label.replace("\n", " "),
        "n": len(sub),
        "crossing_15_prop": np.mean(sub["number_of_crossings"].eq(15)),
        "alternating_prop": np.mean(sub["is_alternating"].eq(1)),
        "signature_median": np.median(sub["signature"]),
        "s_median": np.median(sub[S_COL]),
        "mean_log_norm_median": np.median(sub["mean_log_sq_norm"]),
        "nonalternating_n": len(nonalt),
        "G_positive_prop": np.mean(nonalt["G_abs"] > 0) if len(nonalt) else np.nan,
        "G_ge_4_prop": np.mean(nonalt["G_abs"] >= 4) if len(nonalt) else np.nan,
        "G_mean": np.mean(nonalt["G_abs"]) if len(nonalt) else np.nan,
    })
overview = pd.DataFrame(overview_rows)
overview.to_csv(SRC / "figure1_and_table2_hard_regime_summary.csv", index=False)

fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.3))
fig.suptitle(
    "Absolute and norm-conditioned multiview hardness identify distinct knot regimes",
    y=0.99,
)

# A — disjoint counts
ax = axes[0, 0]
disjoint = overview.loc[overview["hard_regime"].ne("neither")]
bars = ax.bar(
    np.arange(3), disjoint["n"],
    color=[COLORS["amplitude"], COLORS["conditional"], COLORS["shared"]],
)
ax.set_xticks(np.arange(3), ["Amplitude only", "Conditional only", "Shared"])
ax.set_ylabel("Number of knots")
for bar, value in zip(bars, disjoint["n"]):
    ax.text(bar.get_x() + bar.get_width()/2, value + 8, str(int(value)),
            ha="center", fontweight="bold")
ax.text(0.02, 0.94, "Raw total = 292\nConditional total = 413\nOverlap = 46",
        transform=ax.transAxes, va="top")
panel_title(ax, "A", "Two partially distinct hard regimes")

# B — amplitude distributions
ax = axes[0, 1]
box_data = [
    frame.loc[frame["hard_regime"].eq(key), "mean_log_sq_norm"].dropna()
    for key, _, _ in REGIMES
]
bp = ax.boxplot(box_data, patch_artist=True, widths=0.62, showfliers=False)
for patch, (_, _, color) in zip(bp["boxes"], REGIMES):
    patch.set_facecolor(color)
    patch.set_alpha(0.78)
ax.set_xticks(range(1, 5), [x[1] for x in REGIMES])
ax.set_ylabel("Mean log squared norm")
panel_title(ax, "B", "Raw hardness is strongly amplitude-associated")

# C — structural composition
ax = axes[1, 0]
x = np.arange(4)
width = 0.36
ax.bar(x - width/2, overview["alternating_prop"], width,
       label="Alternating", color="#4C78A8")
ax.bar(x + width/2, overview["crossing_15_prop"], width,
       label="15 crossings", color="#F2B134")
ax.set_xticks(x, [r[1] for r in REGIMES])
ax.set_ylim(0, 1.08)
ax.set_ylabel("Proportion")
ax.legend(frameon=False, loc="upper left")
panel_title(ax, "C", "Structural composition differs sharply")

# D — mirror-invariant discrepancy
ax = axes[1, 1]
ax.bar(x - width/2, overview["G_positive_prop"], width,
       label=r"$G>0$", color="#2A9D8F")
ax.bar(x + width/2, overview["G_ge_4_prop"], width,
       label=r"$G\geq4$", color="#8E5EA2")
ax.set_xticks(x, [r[1] for r in REGIMES])
ax.set_ylim(0, 1.12)
ax.set_ylabel("Proportion among nonalternating knots")
ax.legend(frameon=False, loc="upper left")
for i, n in enumerate(overview["nonalternating_n"]):
    ax.text(i, max(overview.loc[i, "G_positive_prop"], 0.04) + 0.04,
            f"n={int(n):,}", ha="center", fontsize=8, color="#444444")
panel_title(ax, "D", "Absolute concordance gaps concentrate in hard regimes")

fig.tight_layout(rect=(0, 0, 1, 0.965))
save_figure(fig, "figure1_two_hardness_regimes")

table2_rows = []
for row in overview.itertuples(index=False):
    table2_rows.append([
        row.label,
        f"{int(row.n):,}",
        pct(row.crossing_15_prop),
        pct(row.alternating_prop),
        fnum(row.signature_median, 1),
        fnum(row.s_median, 1),
        fnum(row.mean_log_norm_median, 2),
        f"{int(row.nonalternating_n):,}",
        pct(row.G_positive_prop),
        pct(row.G_ge_4_prop),
        fnum(row.G_mean, 2),
    ])
latex_table(
    TAB / "table2_hard_regime_summary_paper.tex",
    r"Composition of the disjoint hard regimes. Discrepancy summaries use $G=|s-\sigma|$ among nonalternating knots.",
    "tab:regime-summary",
    ["Regime", "$n$", "15 cr. (\\%)", "Alt. (\\%)", "Med. $\\sigma$",
     "Med. $s$", "Med. log norm", "Nonalt. $n$", "$G>0$ (\\%)",
     "$G\\geq4$ (\\%)", "Mean $G$"],
    table2_rows,
)


# ---------------------------------------------------------------------------
# 3. Figure 2 and Table 3 — randomized mirrors and exact nulls
# ---------------------------------------------------------------------------
random_mirror = mirror_summary.loc[
    mirror_summary["run"].astype(str).str.startswith("random_seed_")
    & mirror_summary["family"].isin(["All 5", "No Khovanov"])
].copy()
random_mirror.to_csv(SRC / "figure2_mirror_randomized_nulls.csv", index=False)

metric_order = [
    ("abs_delta_positive_prop", r"Proportion with $G>0$"),
    ("mean_abs_delta", r"Mean $G$"),
    ("abs_delta_ge_4_prop", r"Proportion with $G\geq4$"),
]
families = ["All 5", "No Khovanov"]
family_colors = {"All 5": COLORS["conditional"], "No Khovanov": COLORS["no_kh"]}

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7))
fig.suptitle(
    "Randomized mirror representatives preserve the conditional concordance phenotype",
    y=1.02,
)
for ax, (metric, ylabel), letter in zip(axes, metric_order, "ABC"):
    data = random_mirror.loc[random_mirror["metric"].eq(metric)].copy()
    for j, family in enumerate(families):
        fam = data.loc[data["family"].eq(family)].sort_values("run")
        jitter = np.linspace(-0.12, 0.12, len(fam))
        xvals = j + jitter
        ax.vlines(xvals, fam["null_q95"], fam["null_q99"],
                  color=COLORS["null"], lw=5, alpha=0.9,
                  label="Null 95th--99th percentiles" if j == 0 else None)
        ax.scatter(xvals, fam["null_mean"], s=42, facecolors="white",
                   edgecolors="#333333", zorder=3,
                   label="Null mean" if j == 0 else None)
        ax.scatter(xvals, fam["observed"], s=58, marker="D",
                   color=family_colors[family], edgecolors="white", linewidths=0.6,
                   zorder=4, label="Observed" if j == 0 else None)
    ax.set_xticks(range(2), ["All five\nviews", "Without\nKhovanov"])
    ax.set_ylabel(ylabel)
    panel_title(ax, letter, ylabel)
axes[0].legend(frameon=False, fontsize=8, loc="best")
fig.text(
    0.5, -0.015,
    "Five randomized representatives; exact nulls preserve per-view norm bin, crossings, alternation and exact |signature|.",
    ha="center", fontsize=9, color="#555555",
)
fig.tight_layout()
save_figure(fig, "figure2_mirror_randomized_nulls")

decision_main = mirror_decision.loc[
    mirror_decision["family"].isin(families)
    & mirror_decision["metric"].isin([x[0] for x in metric_order])
].copy()
decision_main.to_csv(SRC / "table3_mirror_randomized_nulls.csv", index=False)
metric_labels = {
    "abs_delta_positive_prop": r"$P(G>0)$",
    "mean_abs_delta": r"Mean $G$",
    "abs_delta_ge_4_prop": r"$P(G\geq4)$",
}
table3_rows = []
for family in families:
    for metric, _ in metric_order:
        row = decision_main.loc[
            decision_main["family"].eq(family)
            & decision_main["metric"].eq(metric)
        ].iloc[0]
        table3_rows.append([
            family,
            metric_labels[metric],
            f"{fnum(row.observed_min)}--{fnum(row.observed_max)}",
            f"{fnum(row.null_mean_min)}--{fnum(row.null_mean_max)}",
            pvalue(row.empirical_p_max),
            pvalue(row.q_by_max),
            pvalue(row.p_holm_max),
        ])
latex_table(
    TAB / "table3_mirror_randomized_nulls_paper.tex",
    r"Mirror-randomized exact-null results across five representative assignments. Ranges are minima and maxima across assignments; adjusted values are worst-case maxima.",
    "tab:mirror-nulls",
    ["View family", "Endpoint", "Observed range", "Null-mean range",
     "Max. $p_{\\rm emp}$", "Max. BY $q$", "Max. Holm $p$"],
    table3_rows,
    r"The Monte Carlo floor is $1/5001$. Nulls preserve each view's norm bin, crossing number, alternation status and exact $|\sigma|$.",
)


# ---------------------------------------------------------------------------
# 4. Figure 3 and Table 4 — conditional held-out validation
# ---------------------------------------------------------------------------
heldout_family.to_csv(SRC / "figure3_heldout_set_sizes.csv", index=False)
heldout_overlap.to_csv(SRC / "figure3_heldout_overlaps.csv", index=False)
heldout_phenotype.to_csv(SRC / "figure3_heldout_phenotype.csv", index=False)

method_labels = {
    "frozen_full_fit_restricted_to_test": "Frozen test",
    "heldout_PCA_conditional": "Held-out PCA",
    "heldout_AE_majority_3of5": "AE majority",
    "heldout_AE_strict_5of5": "AE strict",
    "frozen_test": "Frozen test",
    "heldout_PCA": "Held-out PCA",
    "AE_majority": "AE majority",
    "AE_strict": "AE strict",
}
family_short = {"All 5 >=3": "All five", "No Khovanov >=3/4": "No Khovanov"}
method_order = [
    "frozen_full_fit_restricted_to_test", "heldout_PCA_conditional",
    "heldout_AE_majority_3of5", "heldout_AE_strict_5of5",
]
phen_order = ["frozen_test", "heldout_PCA", "AE_majority", "AE_strict"]

fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
fig.suptitle(
    "Held-out linear and nonlinear reconstruction support conditional hardness",
    y=0.99,
)

# A — sizes
ax = axes[0, 0]
x = np.arange(4)
width = 0.36
for offset, family in zip((-width/2, width/2), family_short):
    d = heldout_family.loc[heldout_family["family"].eq(family)].set_index("method")
    vals = [d.loc[m, "n"] for m in method_order]
    ax.bar(x + offset, vals, width, label=family_short[family],
           color=COLORS["conditional"] if family.startswith("All") else COLORS["no_kh"],
           alpha=0.9)
ax.set_xticks(x, [method_labels[m] for m in method_order], rotation=18, ha="right")
ax.set_ylabel("Number of test knots")
ax.legend(frameon=False)
panel_title(ax, "A", "Conditional set sizes")

# B — frozen/PCA recovery and Jaccard
ax = axes[0, 1]
comparisons = heldout_overlap.loc[
    heldout_overlap["set_a"].eq("frozen_test")
    & heldout_overlap["set_b"].eq("heldout_PCA")
].set_index("family")
x2 = np.arange(2)
precision = [comparisons.loc[f, "fraction_b_in_a"] for f in family_short]
jaccard = [comparisons.loc[f, "jaccard"] for f in family_short]
ax.bar(x2 - width/2, precision, width, label="PCA recovered by frozen", color="#59A14F")
ax.bar(x2 + width/2, jaccard, width, label="Jaccard", color="#9C755F")
ax.set_xticks(x2, [family_short[f] for f in family_short])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Set agreement")
ax.legend(frameon=False)
panel_title(ax, "B", "Held-out PCA recovers a high-precision core")

# C/D — phenotype
phen_colors = [COLORS["background"], COLORS["conditional"], "#76A5C5", COLORS["shared"]]
for ax, column, ylabel, letter, title in [
    (axes[1, 0], "abs_delta_positive_prop", r"$P(G>0)$ among nonalternating", "C", "Nonzero absolute gap"),
    (axes[1, 1], "mean_abs_delta", r"Mean $G$ among nonalternating", "D", "Mean absolute gap"),
]:
    for offset, family in zip((-width/2, width/2), family_short):
        d = heldout_phenotype.loc[heldout_phenotype["family"].eq(family)].set_index("analysis")
        vals = [d.loc[m, column] for m in phen_order]
        ax.bar(x + offset, vals, width,
               color=COLORS["conditional"] if family.startswith("All") else COLORS["no_kh"],
               alpha=0.9, label=family_short[family])
    ax.set_xticks(x, [method_labels[m] for m in phen_order], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    panel_title(ax, letter, title)
axes[1, 0].set_ylim(0, 1.08)
axes[1, 1].legend(frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save_figure(fig, "figure3_conditional_heldout_validation")

table4_rows = []
for family in family_short:
    d = heldout_phenotype.loc[heldout_phenotype["family"].eq(family)].set_index("analysis")
    for method in phen_order:
        row = d.loc[method]
        table4_rows.append([
            family_short[family], method_labels[method], f"{int(row['n'])}",
            f"{int(row['nonalternating_n'])}", fnum(row["abs_delta_positive_prop"]),
            fnum(row["mean_abs_delta"]), fnum(row["abs_delta_ge_4_prop"]),
        ])
latex_table(
    TAB / "table4_conditional_heldout_validation_paper.tex",
    r"Held-out norm-conditioned validation. Phenotypes use $G=|s-\sigma|$ among nonalternating selected knots.",
    "tab:heldout-conditional",
    ["View family", "Selection", "$n$", "Nonalt. $n$", "$P(G>0)$",
     "Mean $G$", r"$P(G\geq4)$"],
    table4_rows,
    r"PCA and autoencoder percentiles were calibrated on validation scores within validation-defined norm bins before one-time test evaluation.",
)


# ---------------------------------------------------------------------------
# 5. Figure 4 and Table 5 — Khovanov thickness held outside selection
# ---------------------------------------------------------------------------
random_external = external_summary.loc[
    external_summary["run"].astype(str).str.startswith("random_seed_")
].copy()
random_external.to_csv(SRC / "figure4_external_khovanov_thickness.csv", index=False)

external_metrics = [
    ("kh_diagonal_ge_3_prop", r"Proportion with $\geq3$ diagonals"),
    ("kh_diagonal_mean", "Mean supported diagonals"),
    ("kh_support_mean", "Mean Khovanov support size"),
]
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7))
fig.suptitle(
    "Four-view conditional hardness predicts Khovanov thickness out of view",
    y=1.02,
)
for ax, (metric, ylabel), letter in zip(axes, external_metrics, "ABC"):
    d = random_external.loc[random_external["metric"].eq(metric)].sort_values("run")
    x = np.arange(len(d))
    ax.vlines(x, d["null_q95"], d["null_q99"], color=COLORS["null"], lw=7,
              label="Null 95th--99th percentiles")
    ax.scatter(x, d["null_mean"], s=48, facecolors="white", edgecolors="#333333",
               zorder=3, label="Null mean")
    ax.scatter(x, d["observed"], s=68, marker="D", color=COLORS["conditional"],
               edgecolors="white", linewidths=0.7, zorder=4, label="Observed")
    ax.set_xticks(x, [f"R{i+1}" for i in range(len(d))])
    ax.set_xlabel("Random mirror assignment")
    ax.set_ylabel(ylabel)
    panel_title(ax, letter, ylabel)
axes[0].legend(frameon=False, fontsize=8)
fig.text(
    0.5, -0.015,
    "Selection uses Alexander, Jones, HOMFLY--PT and Theta only; Khovanov enters solely as the external outcome.",
    ha="center", fontsize=9, color="#555555",
)
fig.tight_layout()
save_figure(fig, "figure4_external_khovanov_thickness")

external_decision.to_csv(SRC / "table5_external_khovanov_thickness.csv", index=False)
external_labels = {
    "kh_diagonal_ge_3_prop": r"$P(\#\mathrm{diag}\geq3)$",
    "kh_diagonal_mean": r"Mean $\#\mathrm{diag}$",
    "kh_support_mean": "Mean support size",
    "kh_diagonal_ge_4_prop": r"$P(\#\mathrm{diag}\geq4)$",
}
external_order = [
    "kh_diagonal_ge_3_prop", "kh_diagonal_mean", "kh_support_mean",
    "kh_diagonal_ge_4_prop",
]
table5_rows = []
for metric in external_order:
    row = external_decision.loc[external_decision["metric"].eq(metric)].iloc[0]
    table5_rows.append([
        external_labels[metric], str(row.endpoint_role).capitalize(),
        f"{fnum(row.observed_min)}--{fnum(row.observed_max)}",
        f"{fnum(row.null_mean_min)}--{fnum(row.null_mean_max)}",
        pvalue(row.empirical_p_max),
        pvalue(row.q_by_primary_max) if row.endpoint_role == "primary" else "--",
        pvalue(row.p_holm_primary_max) if row.endpoint_role == "primary" else "--",
    ])
latex_table(
    TAB / "table5_external_khovanov_thickness_paper.tex",
    r"Khovanov thickness after selection without Khovanov. Ranges and worst-case adjusted values summarize five randomized representative assignments.",
    "tab:external-khovanov",
    ["External endpoint", "Role", "Observed range", "Null-mean range",
     "Max. $p_{\\rm emp}$", "Max. primary BY $q$", "Max. primary Holm $p$"],
    table5_rows,
    r"The predeclared primary family comprises the three endpoints labelled primary. The four-diagonal endpoint is shown as a secondary distributional summary.",
)


# ---------------------------------------------------------------------------
# 6. Preserve the already-final geometry figure/table as Figure/Table 5/6
# ---------------------------------------------------------------------------
geometry_figures = [
    p for p in ROOT.rglob("figure4_geometric_characterization.*")
    if OUT not in p.parents and p.suffix.lower() in {".png", ".pdf"}
]
for source in geometry_figures:
    shutil.copy2(source, FIG / f"figure5_geometric_characterization{source.suffix.lower()}")

geometry_tables = [
    p for p in ROOT.rglob("table4_matched_geometric_inference_paper.tex")
    if OUT not in p.parents
]
if geometry_tables:
    shutil.copy2(
        geometry_tables[0],
        TAB / "table6_matched_geometric_inference_paper.tex",
    )
else:
    print(
        "WARNING: the existing geometry table was not found under OUTPUT_DIR. "
        "Copy table4_matched_geometric_inference_paper.tex to "
        "table6_matched_geometric_inference_paper.tex manually."
    )

design_tables = [
    p for p in ROOT.rglob("table1_representation_pca_design_paper.tex")
    if OUT not in p.parents
]
if design_tables:
    shutil.copy2(
        design_tables[0],
        TAB / "table1_representation_pca_design_paper.tex",
    )


# ---------------------------------------------------------------------------
# 7. Compact inventory for reproducibility
# ---------------------------------------------------------------------------
inventory = []
for kind, directory in (("figure", FIG), ("table", TAB), ("source_data", SRC)):
    for path in sorted(directory.glob("*")):
        inventory.append({
            "kind": kind,
            "file": path.name,
            "bytes": path.stat().st_size,
        })
inventory = pd.DataFrame(inventory)
inventory.to_csv(OUT / "artifact_inventory.csv", index=False)

display(inventory)
print("\nFinal MLST artifacts saved to:")
print(OUT)
print("\nCopy manuscript_v0_4.tex into this folder and compile with figures/ and tables/.")
