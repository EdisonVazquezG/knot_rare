from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def concatenate_standardized_views(
    X_dict: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, StandardScaler]]:
    """Concatenate independently standardized views; memory intensive."""

    blocks = []
    scalers = {}
    for name, X in X_dict.items():
        scaler = StandardScaler()
        blocks.append(scaler.fit_transform(X).astype(np.float32))
        scalers[name] = scaler
    return np.concatenate(blocks, axis=1), scalers


def global_anomaly_baselines(
    X_concat: np.ndarray,
    pca_dim: int = 64,
    lof_dim: int = 16,
    seed: int = 42,
    run_lof: bool = False,
) -> tuple[dict[str, np.ndarray], dict]:
    """Target-free concatenated-view anomaly baselines; larger is harder."""

    pca = PCA(n_components=min(pca_dim, X_concat.shape[1]), random_state=seed)
    Z = pca.fit_transform(X_concat).astype(np.float32)
    covariance = LedoitWolf().fit(Z)
    isolation = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    ).fit(Z)
    scores = {
        "Mahalanobis::concat_PCA": covariance.mahalanobis(Z),
        "IsolationForest::concat_PCA": -isolation.score_samples(Z),
    }
    models = {"pca": pca, "ledoit_wolf": covariance, "isolation_forest": isolation}
    if run_lof:
        Z_lof = Z[:, : min(lof_dim, Z.shape[1])]
        lof = LocalOutlierFactor(n_neighbors=35, n_jobs=-1)
        lof.fit_predict(Z_lof)
        scores[f"LOF::concat_PCA{Z_lof.shape[1]}"] = -lof.negative_outlier_factor_
        models["lof"] = lof
    return scores, models


def external_metadata_coverage(
    meta: pd.DataFrame,
    external: pd.DataFrame,
    meta_id_col: str = "knot_id_base",
    external_id_col: str = "knot_id_base",
) -> tuple[pd.DataFrame, dict]:
    """Left join external metadata and report coverage without imputation."""

    if external[external_id_col].astype(str).duplicated().any():
        raise ValueError("External metadata IDs must be unique before joining.")
    merged = meta.merge(
        external,
        how="left",
        left_on=meta_id_col,
        right_on=external_id_col,
        suffixes=("", "_external"),
        validate="one_to_one",
        indicator="external_merge_status",
    )
    matched = merged["external_merge_status"].eq("both")
    return merged, {
        "n_analysis": int(len(meta)),
        "n_external_rows": int(len(external)),
        "n_matched": int(matched.sum()),
        "coverage": float(matched.mean()),
    }
