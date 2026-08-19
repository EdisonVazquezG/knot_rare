# src/consensus_hardness/scaling.py

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .norms import compute_standardized_norms


def build_scaling_metadata(
    meta: pd.DataFrame,
    X_dict: dict[str, np.ndarray] | None = None,
    signature_col: str = "signature",
    s_col: str = "s_invariant_qc",
    amplitude_col: str = "A_multi",
    mean_log_norm_col: str = "mean_log_sq_norm",
    pca_sse_col: str | None = "mean_pca_sse_099",
    ae_sse_col: str | None = "mean_ae_sse_99",
) -> pd.DataFrame:
    """
    Build metadata used for empirical concordance-amplitude trend analysis.

    Defines:
    - A_multi = mean_log_sq_norm
    - sigma_abs = |signature|
    - s_abs = |s_invariant|
    - optional log reconstruction errors if PCA/AE columns exist
    """

    out = meta.copy()
    if s_col not in out.columns and "s_invariant" in out.columns:
        s_col = "s_invariant"

    if mean_log_norm_col not in out.columns:
        if X_dict is None:
            raise ValueError(
                f"Column '{mean_log_norm_col}' not found. "
                "Pass X_dict to compute standardized norms."
            )

        out, _ = compute_standardized_norms(out, X_dict)

    out[amplitude_col] = out[mean_log_norm_col]
    out["sigma_abs"] = out[signature_col].abs()
    out["s_abs"] = out[s_col].abs()

    if pca_sse_col is not None and pca_sse_col in out.columns:
        out[f"log_{pca_sse_col}"] = np.log1p(out[pca_sse_col])

    if ae_sse_col is not None and ae_sse_col in out.columns:
        out[f"log_{ae_sse_col}"] = np.log1p(out[ae_sse_col])

    return out


def scaling_summary_by_group(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> pd.DataFrame:
    """
    Summary of value_col grouped by group_col.
    """

    out = (
        df.groupby(group_col)[value_col]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )

    out["sem"] = out["std"] / np.sqrt(out["count"])

    return out


def weighted_r2(
    y: np.ndarray,
    yhat: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Weighted R^2.
    """

    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    weights = np.asarray(weights, dtype=float)

    ybar = np.average(y, weights=weights)

    ss_res = np.sum(weights * (y - yhat) ** 2)
    ss_tot = np.sum(weights * (y - ybar) ** 2)

    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def fit_scaling_law_group_medians(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    min_count_per_group: int = 20,
) -> tuple[dict, pd.DataFrame, LinearRegression]:
    """
    Fit a weighted linear trend to group medians.

    The fit uses only groups with count >= min_count_per_group.
    Weights are group counts.
    """

    summary = scaling_summary_by_group(df, group_col, value_col)
    fit_df = summary[summary["count"] >= min_count_per_group].copy()

    if len(fit_df) < 2:
        raise ValueError(
            f"Not enough groups to fit trend for {value_col} by {group_col}."
        )

    X = fit_df[[group_col]].values.astype(float)
    y = fit_df["median"].values.astype(float)
    weights = fit_df["count"].values.astype(float)

    model = LinearRegression()
    model.fit(X, y, sample_weight=weights)

    yhat = model.predict(X)
    r2 = weighted_r2(y, yhat, weights)

    rho_individual, p_individual = spearmanr(
        df[group_col].values,
        df[value_col].values,
    )

    result = {
        "group_col": group_col,
        "value_col": value_col,
        "min_count_per_group": min_count_per_group,
        "n_groups_total": len(summary),
        "n_groups_used": len(fit_df),
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "weighted_r2_group_medians": float(r2),
        "spearman_rho_individual": float(rho_individual),
        "spearman_p_individual": float(p_individual),
    }

    return result, fit_df, model


def fit_scaling_laws(
    df: pd.DataFrame,
    group_cols: tuple[str, ...] = ("s_abs", "sigma_abs"),
    value_cols: tuple[str, ...] = ("A_multi",),
    min_count_per_group: int = 20,
) -> pd.DataFrame:
    """
    Fit scaling/trend models for several group/value pairs.
    """

    rows = []

    for group_col in group_cols:
        for value_col in value_cols:
            if group_col not in df.columns or value_col not in df.columns:
                continue

            result, _, _ = fit_scaling_law_group_medians(
                df=df,
                group_col=group_col,
                value_col=value_col,
                min_count_per_group=min_count_per_group,
            )

            rows.append(result)

    return pd.DataFrame(rows)


def plot_scaling_law(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    min_count_per_group: int = 20,
    title: str | None = None,
    ylabel: str | None = None,
    save_path=None,
) -> dict:
    """
    Plot group medians and weighted linear fit.
    """

    result, fit_df, model = fit_scaling_law_group_medians(
        df=df,
        group_col=group_col,
        value_col=value_col,
        min_count_per_group=min_count_per_group,
    )

    summary = scaling_summary_by_group(df, group_col, value_col)

    plt.figure(figsize=(7, 5))

    plt.errorbar(
        summary[group_col],
        summary["median"],
        yerr=summary["sem"],
        marker="o",
        linestyle="none",
        capsize=3,
        label="Group median ± SEM",
    )

    x_fit = np.linspace(
        fit_df[group_col].min(),
        fit_df[group_col].max(),
        100,
    )

    y_fit = model.predict(x_fit.reshape(-1, 1))

    plt.plot(
        x_fit,
        y_fit,
        linewidth=2,
        label=(
            f"Weighted fit: slope={result['slope']:.3f}, "
            f"$R^2$={result['weighted_r2_group_medians']:.3f}"
        ),
    )

    for _, row in summary.iterrows():
        plt.text(
            row[group_col],
            row["median"],
            f"n={int(row['count'])}",
            fontsize=8,
            ha="center",
            va="bottom",
        )

    plt.xlabel(group_col)
    plt.ylabel(ylabel if ylabel else value_col)
    plt.title(title if title else f"{value_col} vs {group_col}")
    plt.grid(alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return result


def fit_controlled_regression(
    df: pd.DataFrame,
    target_col: str,
    predictors: list[str],
    alpha: float = 1.0,
) -> tuple[pd.DataFrame, dict, Pipeline]:
    """
    Ridge regression with standardized predictors.
    """

    data = df[[target_col] + predictors].dropna().copy()

    X = data[predictors].astype(float).values
    y = data[target_col].astype(float).values

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )

    pipe.fit(X, y)
    yhat = pipe.predict(X)

    coefs = pipe.named_steps["ridge"].coef_

    coef_df = pd.DataFrame(
        {
            "predictor": predictors,
            "standardized_coef": coefs,
        }
    )

    metrics = {
        "target": target_col,
        "n": len(data),
        "r2_in_sample": float(r2_score(y, yhat)),
    }

    return coef_df, metrics, pipe


def controlled_regression_by_subset(
    df: pd.DataFrame,
    target_col: str,
    predictors: list[str],
    subset_specs: dict[str, pd.Series],
    alpha: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit controlled regression over several subsets.

    Example:
    subset_specs = {
        "all": pd.Series(True, index=df.index),
        "alternating": df["is_alternating"] == 1,
        "non_alternating": df["is_alternating"] == 0,
    }
    """

    coef_rows = []
    metric_rows = []

    for subset_name, mask in subset_specs.items():
        subset_df = df.loc[mask].copy()

        coef_df, metrics, _ = fit_controlled_regression(
            subset_df,
            target_col=target_col,
            predictors=predictors,
            alpha=alpha,
        )

        coef_df["subset"] = subset_name
        metrics["subset"] = subset_name

        coef_rows.append(coef_df)
        metric_rows.append(metrics)

    return (
        pd.concat(coef_rows, ignore_index=True),
        pd.DataFrame(metric_rows),
    )


def standard_scaling_analysis(
    meta_scaling: pd.DataFrame,
    min_count_per_group: int = 20,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Run the standard empirical concordance-amplitude trend analysis.

    Returns:
    - scaling_results_df
    - summaries dictionary
    """

    value_cols = ["A_multi"]

    if "log_mean_pca_sse_099" in meta_scaling.columns:
        value_cols.append("log_mean_pca_sse_099")

    if "log_mean_ae_sse_99" in meta_scaling.columns:
        value_cols.append("log_mean_ae_sse_99")

    scaling_results_df = fit_scaling_laws(
        meta_scaling,
        group_cols=["s_abs", "sigma_abs"],
        value_cols=value_cols,
        min_count_per_group=min_count_per_group,
    )

    summaries = {
        "amp_by_s": scaling_summary_by_group(
            meta_scaling,
            group_col="s_abs",
            value_col="A_multi",
        ),
        "amp_by_sigma": scaling_summary_by_group(
            meta_scaling,
            group_col="sigma_abs",
            value_col="A_multi",
        ),
    }

    return scaling_results_df, summaries
