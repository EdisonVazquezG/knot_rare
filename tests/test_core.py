import numpy as np
import pandas as pd

from consensus_hardness.hardsets import top_tail_indices
from consensus_hardness.null_models import summarize_null_distribution
from consensus_hardness.null_models import run_standard_intersection_nulls
from consensus_hardness.matching import run_caliper_sensitivity
from consensus_hardness.continuous import build_ck_scores, continuous_association_summary
from consensus_hardness.preprocessing import filter_crossings
from consensus_hardness.qc import add_s_invariant_qc_column, audit_analysis_universe


def test_identity_is_removed_by_crossing_filter():
    data = pd.DataFrame(
        {"knot_id": ["00_1", "03_1", "16_1"], "number_of_crossings": [0, 3, 16]}
    )
    result = filter_crossings(data, min_crossings=3, max_crossings=15)
    assert result["knot_id"].tolist() == ["03_1"]


def test_s_qc_preserves_raw_and_corrects_only_sign_flip():
    meta = pd.DataFrame(
        {
            "signature": [2, 2, 4],
            "s_invariant": [-2, 0, 4],
            "is_alternating": [1, 1, 0],
        }
    )
    result = add_s_invariant_qc_column(meta)
    assert result["s_invariant_original"].tolist() == [-2, 0, 4]
    assert result["s_invariant_qc"].tolist() == [2, 0, 4]
    assert result["s_invariant_was_corrected"].tolist() == [True, False, False]


def test_universe_audit_rejects_identity_and_accepts_corrected_small_universe():
    meta = pd.DataFrame(
        {
            "knot_id_base": ["03_1", "04_1"],
            "number_of_crossings": [3, 4],
            "s_invariant_was_corrected": [True, False],
        }
    )
    audit = audit_analysis_universe(
        meta, expected_n=2, expected_s_qc_corrections=1
    )
    assert audit["identity_rows"] == 0


def test_top_tail_is_deterministic_under_ties():
    scores = np.array([1.0, 2.0, 2.0, 0.0])
    ids = np.array(["d", "b", "a", "c"])
    selected = top_tail_indices(scores, tail_mass=0.5, stable_ids=ids)
    assert selected.tolist() == [2, 1]


def test_monte_carlo_p_value_uses_plus_one_correction():
    summary = summarize_null_distribution(np.zeros(100), observed=1, label="test")
    assert summary["n_ge_observed"] == 0
    assert summary["empirical_p"] == 1 / 101


def test_standard_null_suite_has_four_nested_models():
    meta = pd.DataFrame(
        {
            "number_of_crossings": [3, 3, 4, 4, 5, 5, 6, 6],
            "is_alternating": [0, 1] * 4,
            "signature_bin": ["0-4"] * 8,
        }
    )
    hard_sets = {
        "A": {0, 2},
        "B": {0, 3},
        "C": {0, 4},
    }
    summary, values = run_standard_intersection_nulls(
        meta, hard_sets, observed_consensus_size=1, tail_mass=0.25, n_reps=5
    )
    assert len(summary) == 4
    assert "crossing_alt_matched_tails" in values
    assert summary["empirical_p"].gt(0).all()


def test_caliper_matching_keeps_match_groups():
    df = pd.DataFrame(
        {
            "selected": [True, False, False, False],
            "crossing": [15, 15, 15, 15],
            "norm": [0.0, 0.01, 0.02, 1.0],
            "outcome": [2.0, 1.0, 1.5, 0.0],
        }
    )
    summary, results = run_caliper_sensitivity(
        df,
        selected_col="selected",
        exact_cols=["crossing"],
        norm_cols=["norm"],
        outcome_cols=["outcome"],
        ratio=2,
        calipers=(None,),
    )
    assert summary.loc[0, "n_selected_matched"] == 1
    assert results[None]["pairs"]["match_group_id"].nunique() == 1
    assert results[None]["outcomes"].loc[0, "n_match_groups"] == 1


def test_continuous_ck_contract():
    scores = {
        "A": np.array([0.1, 0.8, 0.2, 0.9]),
        "B": np.array([0.2, 0.7, 0.3, 0.8]),
        "C": np.array([0.3, 0.6, 0.4, 0.7]),
    }
    ck = build_ck_scores(scores, {"poly": {"names": ("A", "B", "C"), "k": 2}})
    meta = pd.DataFrame(
        {
            "knot_id_base": ["a", "b", "c", "d"],
            "signature": [0, 2, 2, 4],
            "s_invariant_qc": [0, 4, 2, 8],
            "is_alternating": [0, 0, 0, 0],
            "number_of_crossings": [10, 10, 11, 11],
        }
    )
    summary = continuous_association_summary(ck, meta, top_fraction=0.5)
    assert summary.loc[0, "n_selected"] == 2
    assert summary.loc[0, "family"] == "poly"
