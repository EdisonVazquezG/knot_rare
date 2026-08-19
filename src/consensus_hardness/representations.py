# src/consensus_hardness/representations.py

from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocessing import ID_COLS, META_CANDIDATES


def validate_alignment(
    aligned_tables: dict[str, pd.DataFrame],
    id_col: str = "knot_id_base",
) -> list[str]:
    if not aligned_tables:
        raise ValueError("No aligned tables provided.")

    names = list(aligned_tables.keys())
    reference_name = names[0]
    reference_ids = aligned_tables[reference_name][id_col].astype(str).tolist()

    for name in names[1:]:
        ids = aligned_tables[name][id_col].astype(str).tolist()
        if ids != reference_ids:
            raise ValueError(
                f"Table '{name}' is not aligned with '{reference_name}'."
            )

    return reference_ids


def collect_metadata(
    aligned_tables: dict[str, pd.DataFrame],
    preferred_sources: list[str] | None = None,
    id_col: str = "knot_id_base",
    metadata_columns: list[str] | None = None,
) -> pd.DataFrame:
    if preferred_sources is None:
        preferred_sources = list(aligned_tables.keys())

    if metadata_columns is None:
        metadata_columns = [
            "knot_id",
            "knot_id_clean",
            "number_of_crossings",
            "table_number",
            "is_alternating",
            "signature",
            "s_invariant",
        ]

    validate_alignment(aligned_tables, id_col=id_col)

    first_source = preferred_sources[0]
    meta = pd.DataFrame(
        {
            id_col: aligned_tables[first_source][id_col].values
        }
    )

    for col in metadata_columns:
        for source in preferred_sources:
            if source in aligned_tables and col in aligned_tables[source].columns:
                meta[col] = aligned_tables[source][col].values
                break

    return meta


def split_metadata_and_features(
    df: pd.DataFrame,
    feature_prefixes: list[str],
    id_col: str = "knot_id_base",
    extra_meta_candidates: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    if id_col not in df.columns:
        raise KeyError(f"ID column '{id_col}' not found.")

    ids = df[id_col].astype(str).values

    meta_candidates = ID_COLS + META_CANDIDATES
    if extra_meta_candidates:
        meta_candidates += extra_meta_candidates

    meta_cols = [c for c in meta_candidates if c in df.columns]
    metadata = df[meta_cols].copy()

    exclude_cols = set(meta_cols)

    feat_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and any(str(c).startswith(pref) for pref in feature_prefixes)
    ]

    if len(feat_cols) == 0:
        raise ValueError(
            f"No feature columns found for prefixes: {feature_prefixes}"
        )

    X = (
        df[feat_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    return metadata, ids, X, feat_cols


def build_representation_dict(
    aligned_tables: dict[str, pd.DataFrame],
    representation_specs: dict[str, dict],
    id_col: str = "knot_id_base",
) -> dict:
    """
    representation_specs example:

    {
        "Alexander": {"source": "alex", "feature_prefixes": ["A"]},
        "Jones": {"source": "jones", "feature_prefixes": ["J"]},
    }
    """

    X_dict = {}
    feature_cols_dict = {}
    ids_dict = {}
    metadata_dict = {}

    for public_name, spec in representation_specs.items():
        source = spec["source"]
        prefixes = spec["feature_prefixes"]

        if source not in aligned_tables:
            raise KeyError(f"Source table '{source}' not found.")

        metadata, ids, X, feat_cols = split_metadata_and_features(
            aligned_tables[source],
            feature_prefixes=prefixes,
            id_col=id_col,
        )

        X_dict[public_name] = X
        feature_cols_dict[public_name] = feat_cols
        ids_dict[public_name] = ids
        metadata_dict[public_name] = metadata

    reference_name = next(iter(ids_dict))
    reference_ids = ids_dict[reference_name]

    for name, ids in ids_dict.items():
        if not np.array_equal(reference_ids, ids):
            raise ValueError(
                f"Representation '{name}' is not aligned with '{reference_name}'."
            )

    return {
        "X_dict": X_dict,
        "feature_cols_dict": feature_cols_dict,
        "ids_dict": ids_dict,
        "metadata_dict": metadata_dict,
    }


def summarize_representations(X_dict: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "invariant": list(X_dict.keys()),
            "n_knots": [X.shape[0] for X in X_dict.values()],
            "input_dim": [X.shape[1] for X in X_dict.values()],
        }
    )


def build_manifest(
    meta: pd.DataFrame,
    aligned_tables: dict[str, pd.DataFrame],
    id_cols: tuple[str, str] = ("knot_id_clean", "knot_id"),
) -> pd.DataFrame:
    manifest = meta.copy()

    clean_col, original_col = id_cols

    for name, df in aligned_tables.items():
        if clean_col in df.columns:
            manifest[f"{name}_knot_id_clean"] = df[clean_col].values
        if original_col in df.columns:
            manifest[f"{name}_knot_id_original"] = df[original_col].values

    return manifest