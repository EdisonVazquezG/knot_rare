# src/consensus_hardness/pca.py

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def k_for_threshold(cum_evr: np.ndarray, alpha: float) -> int:
    """
    Smallest k such that cumulative explained variance ratio >= alpha.
    """
    return int(np.searchsorted(cum_evr, alpha) + 1)


def elbow_max_distance(cum_evr: np.ndarray) -> tuple[int, float, float]:
    """
    Elbow detector based on maximum distance to the straight line
    joining the first and last point of the cumulative EVR curve.
    """

    y = np.asarray(cum_evr)
    x = np.arange(1, len(y) + 1)

    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    points = np.column_stack([x_norm, y_norm])

    start = points[0]
    end = points[-1]
    line = end - start

    distances = np.abs(np.cross(line, points - start)) / (
        np.linalg.norm(line) + 1e-12
    )

    k_elbow = int(np.argmax(distances) + 1)

    return (
        k_elbow,
        float(y[k_elbow - 1]),
        float(distances[k_elbow - 1]),
    )


def plateau_k(
    cum_evr: np.ndarray,
    marginal_threshold: float = 1e-3,
    consecutive: int = 3,
) -> tuple[int | None, float | None, float | None]:
    """
    First k after which the marginal EVR contribution remains below
    marginal_threshold for `consecutive` components.
    """

    cum_evr = np.asarray(cum_evr)
    marginal = np.diff(np.concatenate([[0.0], cum_evr]))

    for i in range(len(marginal) - consecutive + 1):
        window = marginal[i : i + consecutive]
        if np.all(window < marginal_threshold):
            return int(i + 1), float(cum_evr[i]), float(marginal[i])

    return None, None, None


def pca_reconstruction_errors_for_k(
    X: np.ndarray,
    k: int,
    eps: float = 1e-8,
    svd_solver: str = "full",
) -> dict:
    """
    Standardize X, fit PCA with fixed k, and compute SSE, MSE and NRE.
    """

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=k, svd_solver=svd_solver)
    Z = pca.fit_transform(Xs)
    Xhat = pca.inverse_transform(Z)

    residual = Xs - Xhat

    sse = np.sum(residual**2, axis=1)
    mse = sse / X.shape[1]
    nre = sse / (np.sum(Xs**2, axis=1) + eps)

    return {
        "k": int(k),
        "evr": float(np.sum(pca.explained_variance_ratio_)),
        "sse": sse,
        "mse": mse,
        "nre": nre,
        "compression_ratio": float(X.shape[1] / k),
        "scaler": scaler,
        "pca": pca,
    }


def pca_reconstruction_errors_by_evr(
    X: np.ndarray,
    evr_thresholds: tuple[float, ...] = (0.94, 0.99, 0.999),
    eps: float = 1e-8,
    svd_solver: str = "full",
) -> tuple[dict, np.ndarray]:
    """
    Fit full PCA to estimate cumulative EVR, then compute reconstruction
    errors at each requested EVR threshold.
    """

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca_full = PCA(svd_solver=svd_solver)
    pca_full.fit(Xs)

    cum_evr = np.cumsum(pca_full.explained_variance_ratio_)

    results = {}

    for alpha in evr_thresholds:
        k = k_for_threshold(cum_evr, alpha)

        pca = PCA(n_components=k, svd_solver=svd_solver)
        Z = pca.fit_transform(Xs)
        Xhat = pca.inverse_transform(Z)

        residual = Xs - Xhat

        sse = np.sum(residual**2, axis=1)
        mse = sse / X.shape[1]
        nre = sse / (np.sum(Xs**2, axis=1) + eps)

        results[alpha] = {
            "k": int(k),
            "evr": float(cum_evr[k - 1]),
            "sse": sse,
            "mse": mse,
            "nre": nre,
            "compression_ratio": float(X.shape[1] / k),
            "scaler": scaler,
            "pca": pca,
        }

    return results, cum_evr


def run_pca_by_evr_for_representations(
    X_dict: dict[str, np.ndarray],
    evr_thresholds: tuple[float, ...] = (0.94, 0.99, 0.999),
) -> tuple[dict, dict, pd.DataFrame]:
    """
    Run PCA reconstruction analysis for every representation in X_dict.
    """

    pca_results = {}
    evr_curves = {}
    rows = []

    for name, X in X_dict.items():
        print(f"Running PCA: {name}")

        res, cum_evr = pca_reconstruction_errors_by_evr(
            X,
            evr_thresholds=evr_thresholds,
        )

        pca_results[name] = res
        evr_curves[name] = cum_evr

        for alpha, out in res.items():
            rows.append(
                {
                    "invariant": name,
                    "input_dim": X.shape[1],
                    "evr_threshold": alpha,
                    "k": out["k"],
                    "actual_evr": out["evr"],
                    "compression_ratio": out["compression_ratio"],
                }
            )

    evr_table = pd.DataFrame(rows)

    return pca_results, evr_curves, evr_table


def summarize_pca_effective_dimension(
    evr_curves: dict[str, np.ndarray],
    thresholds: tuple[float, ...] = (0.94, 0.99, 0.999),
) -> pd.DataFrame:
    """
    Build PCA effective dimensionality summary.
    """

    rows = []

    for name, cum_evr in evr_curves.items():
        input_dim = len(cum_evr)

        k_elbow, evr_elbow, elbow_distance = elbow_max_distance(cum_evr)

        k_plateau_001, evr_plateau_001, marginal_001 = plateau_k(
            cum_evr,
            marginal_threshold=1e-3,
            consecutive=3,
        )

        k_plateau_0001, evr_plateau_0001, marginal_0001 = plateau_k(
            cum_evr,
            marginal_threshold=1e-4,
            consecutive=3,
        )

        row = {
            "invariant": name,
            "input_dim": input_dim,
            "k_elbow": k_elbow,
            "evr_at_elbow": evr_elbow,
            "elbow_distance": elbow_distance,
            "compression_at_elbow": input_dim / k_elbow,
            "k_plateau_marginal_lt_0.001": k_plateau_001,
            "evr_plateau_0.001": evr_plateau_001,
            "marginal_plateau_0.001": marginal_001,
            "k_plateau_marginal_lt_0.0001": k_plateau_0001,
            "evr_plateau_0.0001": evr_plateau_0001,
            "marginal_plateau_0.0001": marginal_0001,
        }

        for alpha in thresholds:
            key = str(alpha).replace(".", "")
            row[f"k_{key}"] = k_for_threshold(cum_evr, alpha)
            row[f"evr_{key}"] = float(cum_evr[row[f"k_{key}"] - 1])
            row[f"compression_at_{key}"] = input_dim / row[f"k_{key}"]

        rows.append(row)

    return pd.DataFrame(rows)


def make_pca_summary_main(elbow_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Format the main PCA table used in the manuscript.
    """

    out = elbow_summary.copy()

    required_cols = [
        "invariant",
        "input_dim",
        "k_elbow",
        "evr_at_elbow",
        "compression_at_elbow",
        "k_099",
        "compression_at_099",
        "k_0999",
    ]

    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")

    out = out[required_cols].copy()

    out = out.rename(
        columns={
            "k_099": "k_99",
            "compression_at_099": "compression_at_99",
            "k_0999": "k_999",
        }
    )

    out["evr_at_elbow"] = out["evr_at_elbow"].round(4)
    out["compression_at_elbow"] = out["compression_at_elbow"].round(2)
    out["compression_at_99"] = out["compression_at_99"].round(2)

    return out


def run_pca_fixed_k_for_representations(
    X_dict: dict[str, np.ndarray],
    fixed_k_dict: dict[str, int],
) -> dict:
    """
    Run standardized PCA reconstruction with a fixed number of components
    per representation.

    Example:
    fixed_k_dict = {
        "Alexander": 4,
        "Jones": 10,
        "HOMFLY-PT": 32,
        "Theta": 10,
        "Khovanov": 77,
    }
    """

    results = {}

    for name, X in X_dict.items():
        if name not in fixed_k_dict:
            raise KeyError(f"Missing fixed k for representation '{name}'.")

        k = fixed_k_dict[name]

        results[name] = pca_reconstruction_errors_for_k(
            X,
            k=k,
        )

        print(
            name,
            "k:", k,
            "EVR:", results[name]["evr"],
            "compression:", results[name]["compression_ratio"],
        )

    return results


def fixed_k_dict_from_evr_table(
    evr_table: pd.DataFrame,
    evr_threshold: float,
    invariant_col: str = "invariant",
    threshold_col: str = "evr_threshold",
    k_col: str = "k",
) -> dict[str, int]:
    """
    Extract fixed-k dictionary from an EVR table.
    """

    df = evr_table[evr_table[threshold_col] == evr_threshold].copy()

    if df.empty:
        raise ValueError(f"No rows found for EVR threshold {evr_threshold}.")

    return {
        row[invariant_col]: int(row[k_col])
        for _, row in df.iterrows()
    }