"""Public API for the consensus-hardness analysis package.

Recommended notebook usage is ``import consensus_hardness as ch``.  A curated
``from consensus_hardness import *`` also works; TensorFlow-dependent symbols
are exported only when TensorFlow is installed.
"""

from .artifacts import save_hard_sets_by_id, stage_directory, write_run_manifest
from .conditional_nulls import (
    build_joint_strata_codes,
    hard_sets_to_masks,
    run_stratified_membership_null,
    summarize_null_against_observed,
)
from .config import PCAConfig, RunConfig, UniverseConfig, canonical_run_config
from .continuous import (
    add_delta,
    build_ck_scores,
    continuous_association_summary,
    exact_top_fraction,
    kth_largest_score,
)
from .concordance import (
    add_concordance_columns,
    add_sigma_minus_s,
    build_concordance_qc_report,
    find_alternating_sigma_s_mismatches,
    inspect_knot_by_base_id,
    sigma_s_joint_counts,
    summarize_non_alternating_concordance,
    summarize_sigma_minus_s_by_alternating,
    summarize_sigma_s_agreement,
)
from .enrichment import standard_knot_enrichment_table
from .dev import reload_package
from .hardsets import (
    build_hard_sets_from_fixed_results,
    build_hard_sets_from_results,
    consensus_dataframe,
    consensus_from_hard_sets,
    consensus_summary_from_fixed_results,
    consensus_summary_table,
    compare_consensus_sets,
    characterize_persistent_hard,
    captured_threshold_table,
    distribution_table,
    hard_sets_to_id_sets,
    ids_to_indices,
    indices_to_ids,
    membership_count_table,
    pairwise_overlaps,
    top_tail_indices,
)
from .io import build_aligned_dataset, load_raw_tables, load_table
from .matching import (
    compare_two_groups,
    compare_matched_outcomes_grouped,
    nearest_norm_matched_controls,
    run_caliper_sensitivity,
    sample_exact_matched_controls,
)
from .models import logistic_model_comparison
from .norms import (
    add_consensus_indicator,
    add_pca_errors_to_metadata,
    compare_selected_vs_background,
    compute_standardized_norms,
    norm_error_correlations,
    safe_name,
)
from .null_models import run_standard_intersection_nulls
from .pca import (
    run_pca_by_evr_for_representations,
    run_pca_fixed_k_for_representations,
    fixed_k_dict_from_evr_table,
    make_pca_summary_main,
    summarize_pca_effective_dimension,
)
from .pipeline import run_primary_pca_analysis
from .qc import (
    add_s_invariant_qc_column,
    audit_analysis_universe,
    corrected_s_invariant_table,
)
from .robustness import (
    at_least_k_consensus,
    conditional_hardness_by_norm,
    crossfitted_norm_residual,
    leave_one_representation_out,
    norm_adjusted_scores,
    summarize_selected_set,
    tail_mass_sensitivity,
)

__all__ = [name for name in globals() if not name.startswith("_")]

# Optional neural-network API.  Core PCA/null workflows remain importable on
# machines without TensorFlow.
try:
    from .autoencoders import (
        add_autoencoder_errors_to_meta,
        summarize_autoencoder_reconstruction,
        train_autoencoders_for_representations,
    )
    from .heldout import (
        add_holdout_errors_to_meta,
        pairwise_jaccard_matrix,
        make_train_val_test_indices,
        run_holdout_pca_for_representations,
        summarize_heldout_ae_runs,
        summarize_holdout_reconstruction,
        summarize_heldout_run,
        train_heldout_autoencoders,
    )

    __all__ += [
        "add_autoencoder_errors_to_meta",
        "summarize_autoencoder_reconstruction",
        "train_autoencoders_for_representations",
        "add_holdout_errors_to_meta",
        "pairwise_jaccard_matrix",
        "make_train_val_test_indices",
        "run_holdout_pca_for_representations",
        "summarize_heldout_ae_runs",
        "summarize_holdout_reconstruction",
        "summarize_heldout_run",
        "train_heldout_autoencoders",
    ]
except (ImportError, ModuleNotFoundError):
    pass
