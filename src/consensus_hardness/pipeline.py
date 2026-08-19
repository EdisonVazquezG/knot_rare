from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import save_hard_sets_by_id, stage_directory, write_run_manifest
from .config import RunConfig, canonical_run_config
from .hardsets import (
    build_hard_sets_from_fixed_results,
    consensus_dataframe,
    consensus_from_hard_sets,
    membership_count_table,
    pairwise_overlaps,
    summarize_consensus_set,
)
from .null_models import run_standard_intersection_nulls
from .norms import safe_name
from .pca import run_pca_fixed_k_for_representations
from .qc import audit_analysis_universe, corrected_s_invariant_table


def run_primary_pca_analysis(
    meta: pd.DataFrame,
    X_dict: dict,
    output_dir: str | Path,
    config: RunConfig | None = None,
    run_nulls: bool = True,
    assert_expected: bool = True,
) -> dict:
    """Run the corrected, frozen primary PCA-SSE analysis.

    Exploratory norm adjustment, leave-one-out views, autoencoders, and
    held-out analyses intentionally live outside this primary entry point.
    """

    config = config or canonical_run_config()
    output_dir = Path(output_dir)
    qc_dir = stage_directory(output_dir, 1, "universe_qc")
    pca_dir = stage_directory(output_dir, 2, "primary_pca")
    null_dir = stage_directory(output_dir, 3, "intersection_nulls")

    audit = audit_analysis_universe(
        meta,
        expected_n=config.universe.expected_n,
        min_crossings=config.universe.min_crossings,
        max_crossings=config.universe.max_crossings,
        id_col=config.universe.id_col,
        identity_id=config.universe.identity_id,
        expected_s_qc_corrections=config.universe.expected_s_qc_corrections,
    )
    pd.DataFrame([audit]).to_csv(qc_dir / "analysis_universe_audit.csv", index=False)
    corrected_s_invariant_table(meta).to_csv(qc_dir / "s_invariant_corrections.csv", index=False)

    fixed_results = run_pca_fixed_k_for_representations(X_dict, config.pca.primary_k)
    score_payload = {}
    pca_rows = []
    for name, result in fixed_results.items():
        key = safe_name(name)
        for score_name in ("sse", "mse", "nre"):
            score_payload[f"{key}__{score_name}"] = result[score_name]
        pca_rows.append(
            {
                "invariant": name,
                "input_dim": int(X_dict[name].shape[1]),
                "k": int(result["k"]),
                "actual_evr": float(result["evr"]),
                "compression_ratio": float(result["compression_ratio"]),
            }
        )
    np.savez_compressed(pca_dir / "primary_pca_scores.npz", **score_payload)
    pd.DataFrame(pca_rows).to_csv(pca_dir / "primary_pca_dimensions.csv", index=False)
    hard_sets = build_hard_sets_from_fixed_results(
        fixed_results,
        score_name="sse",
        tail_mass=config.pca.tail_mass,
        stable_ids=meta[config.universe.id_col],
    )
    consensus = consensus_from_hard_sets(hard_sets)
    tail_sizes = {name: len(values) for name, values in hard_sets.items()}
    if assert_expected:
        if set(tail_sizes.values()) != {config.pca.expected_tail_size}:
            raise AssertionError(
                f"Expected tail size {config.pca.expected_tail_size}, found {tail_sizes}"
            )
        if len(consensus) != config.pca.expected_primary_consensus_size:
            raise AssertionError(
                "Primary consensus drift: expected "
                f"{config.pca.expected_primary_consensus_size}, found {len(consensus)}"
            )
    consensus_df = consensus_dataframe(consensus, meta)
    summary = summarize_consensus_set(
        consensus,
        meta,
        score_name="sse",
        evr_threshold=config.pca.primary_evr,
        tail_mass=config.pca.tail_mass,
        s_col=config.s_col,
    )
    _, membership_summary = membership_count_table(hard_sets, meta)

    pd.DataFrame([summary]).to_csv(pca_dir / "primary_consensus_summary.csv", index=False)
    consensus_df.to_csv(pca_dir / "primary_consensus_members.csv", index=False)
    pairwise_overlaps(hard_sets).to_csv(pca_dir / "pairwise_tail_overlaps.csv", index=False)
    membership_summary.to_csv(pca_dir / "membership_summary.csv", index=False)
    save_hard_sets_by_id(
        hard_sets, meta, pca_dir / "hard_sets_by_stable_id.csv", id_col=config.universe.id_col
    )

    null_summary = None
    null_values = None
    if run_nulls:
        null_summary, null_values = run_standard_intersection_nulls(
            meta=meta,
            observed_hard_sets=hard_sets,
            observed_consensus_size=len(consensus),
            tail_mass=config.pca.tail_mass,
            n_reps=config.null_reps,
            seed=config.random_seed,
        )
        null_summary.to_csv(null_dir / "intersection_null_summary.csv", index=False)
        np.savez_compressed(null_dir / "intersection_null_values.npz", **null_values)

    write_run_manifest(
        output_dir,
        config=config.to_dict(),
        extra={"universe_audit": audit, "primary_consensus_size": len(consensus)},
    )
    return {
        "fixed_results": fixed_results,
        "hard_sets": hard_sets,
        "consensus": consensus,
        "consensus_df": consensus_df,
        "summary": summary,
        "null_summary": null_summary,
        "null_values": null_values,
        "audit": audit,
    }
