# src/consensus_hardness/io.py

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from .preprocessing import (
    add_knot_ids,
    filter_crossings,
    canonicalize_mirrors_by_signature,
    audit_mirror_canonicalization,
    summarize_signature_distribution,
    add_signature_bin,
)
from .qc import add_s_invariant_qc_column, audit_analysis_universe
from .representations import (
    collect_metadata,
    build_representation_dict,
    summarize_representations,
    build_manifest,
    validate_alignment,
)


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as fh:
            return pickle.load(fh)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def load_raw_tables(
    base_dir: str | Path,
    file_map: dict[str, str],
) -> dict[str, pd.DataFrame]:
    base_dir = Path(base_dir)
    out = {}

    for name, rel_path in file_map.items():
        out[name] = load_table(base_dir / rel_path)

    return out


def clean_filter_canonicalize_tables(
    raw_tables: dict[str, pd.DataFrame],
    min_crossings: int = 3,
    max_crossings: int = 15,
    id_col: str = "knot_id",
    mirror_symbol: str = "!",
    canonicalize_mirrors: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    cleaned = {}
    canonical = {}

    for name, df in raw_tables.items():
        tmp = add_knot_ids(
            df,
            id_col=id_col,
            mirror_symbol=mirror_symbol,
        )
        tmp = filter_crossings(
            tmp,
            min_crossings=min_crossings,
            max_crossings=max_crossings,
        )
        cleaned[name] = tmp

        if canonicalize_mirrors:
            tmp = canonicalize_mirrors_by_signature(
                tmp,
                id_col=id_col,
                base_col="knot_id_base",
                signature_col="signature",
                mirror_symbol=mirror_symbol,
            )

        canonical[name] = tmp

    return cleaned, canonical


def common_universe(
    tables: dict[str, pd.DataFrame],
    id_col: str = "knot_id_base",
) -> set[str]:
    if not tables:
        raise ValueError("No tables provided.")

    return set.intersection(*[set(df[id_col]) for df in tables.values()])


def align_tables_to_common_universe(
    tables: dict[str, pd.DataFrame],
    ids_common: set[str],
    id_col: str = "knot_id_base",
) -> dict[str, pd.DataFrame]:
    aligned = {}

    for name, df in tables.items():
        tmp = df[df[id_col].isin(ids_common)].copy()
        tmp = tmp.sort_values(id_col).reset_index(drop=True)
        aligned[name] = tmp

    validate_alignment(aligned, id_col=id_col)

    return aligned


def build_aligned_dataset(
    base_dir: str | Path,
    file_map: dict[str, str],
    representation_specs: dict[str, dict],
    min_crossings: int = 3,
    max_crossings: int = 15,
    output_dir: str | Path | None = None,
    id_col: str = "knot_id",
    base_id_col: str = "knot_id_base",
    mirror_symbol: str = "!",
    preferred_metadata_sources: list[str] | None = None,
    apply_s_qc: bool = True,
    expected_n: int | None = None,
    expected_s_qc_corrections: int | None = None,
) -> dict:
    raw_tables = load_raw_tables(base_dir, file_map)

    cleaned_tables, canonical_tables = clean_filter_canonicalize_tables(
        raw_tables,
        min_crossings=min_crossings,
        max_crossings=max_crossings,
        id_col=id_col,
        mirror_symbol=mirror_symbol,
        canonicalize_mirrors=True,
    )

    ids_common = common_universe(
        canonical_tables,
        id_col=base_id_col,
    )

    aligned_tables = align_tables_to_common_universe(
        canonical_tables,
        ids_common=ids_common,
        id_col=base_id_col,
    )

    if preferred_metadata_sources is None:
        preferred_metadata_sources = list(aligned_tables.keys())

    meta = collect_metadata(
        aligned_tables,
        preferred_sources=preferred_metadata_sources,
        id_col=base_id_col,
    )

    if apply_s_qc and {"signature", "s_invariant", "is_alternating"}.issubset(meta.columns):
        meta = add_s_invariant_qc_column(meta)

    if "signature" in meta.columns:
        meta = add_signature_bin(meta)

    universe_audit = audit_analysis_universe(
        meta,
        expected_n=expected_n,
        min_crossings=min_crossings,
        max_crossings=max_crossings,
        expected_s_qc_corrections=expected_s_qc_corrections,
    )

    excluded_rows = {}
    crossing_exclusion_rows = []
    for name, raw in raw_tables.items():
        with_ids = add_knot_ids(raw, id_col=id_col, mirror_symbol=mirror_symbol)
        crossing = pd.to_numeric(with_ids["number_of_crossings"], errors="coerce")
        # Persist the scientifically special lower-bound exclusions (the
        # identity), not millions of rows above the upper crossing cutoff.
        excluded_rows[name] = with_ids.loc[crossing < min_crossings].copy()
        crossing_exclusion_rows.append(
            {
                "source": name,
                "below_min": int((crossing < min_crossings).sum()),
                "above_max": int((crossing > max_crossings).sum()),
                "non_numeric": int(crossing.isna().sum()),
                "excluded_total": int(
                    ((crossing < min_crossings) | (crossing > max_crossings) | crossing.isna()).sum()
                ),
            }
        )
    crossing_exclusion_summary = pd.DataFrame(crossing_exclusion_rows)

    rep_result = build_representation_dict(
        aligned_tables,
        representation_specs=representation_specs,
        id_col=base_id_col,
    )

    X_dict = rep_result["X_dict"]
    feature_cols_dict = rep_result["feature_cols_dict"]

    representation_summary = summarize_representations(X_dict)
    mirror_audit = audit_mirror_canonicalization(aligned_tables)
    manifest = build_manifest(meta, aligned_tables)

    signature_summary = None
    if "signature" in meta.columns:
        signature_summary = summarize_signature_distribution(meta)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest.to_csv(output_dir / "aligned_canonical_manifest.csv", index=False)
        mirror_audit.to_csv(output_dir / "mirror_canonicalization_audit.csv", index=False)
        representation_summary.to_csv(output_dir / "representation_summary.csv", index=False)

        if signature_summary is not None:
            signature_summary.to_csv(output_dir / "signature_distribution.csv", index=False)

        pd.DataFrame([universe_audit]).to_csv(
            output_dir / "analysis_universe_audit.csv", index=False
        )
        excluded_manifest = pd.concat(
            [df.assign(source=name) for name, df in excluded_rows.items()],
            ignore_index=True,
        )
        excluded_manifest.to_csv(output_dir / "excluded_crossing_rows.csv", index=False)
        crossing_exclusion_summary.to_csv(
            output_dir / "crossing_exclusion_summary.csv", index=False
        )

    return {
        "raw_tables": raw_tables,
        "cleaned_tables": cleaned_tables,
        "canonical_tables": canonical_tables,
        "aligned_tables": aligned_tables,
        "ids_common": ids_common,
        "meta": meta,
        "X_dict": X_dict,
        "feature_cols_dict": feature_cols_dict,
        "representation_summary": representation_summary,
        "mirror_audit": mirror_audit,
        "manifest": manifest,
        "signature_summary": signature_summary,
        "universe_audit": universe_audit,
        "excluded_rows": excluded_rows,
        "crossing_exclusion_summary": crossing_exclusion_summary,
    }
