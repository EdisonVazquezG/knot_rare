from __future__ import annotations

import argparse
from pathlib import Path

import consensus_hardness as ch


FILE_MAP = {
    "alex": "Alexander_upto17.csv",
    "homfly": "HomflyPt_upto15_MIRRORS.csv",
    "jones": "Jones_upto17_MIRRORS.csv",
    "theta": "theta_upto15.csv",
    "kh": "even_KH_upto17.pkl",
}

REPRESENTATION_SPECS = {
    "Alexander": {"source": "alex", "feature_prefixes": ["A"]},
    "Jones": {"source": "jones", "feature_prefixes": ["J"]},
    "HOMFLY-PT": {"source": "homfly", "feature_prefixes": ["a"]},
    "Theta": {"source": "theta", "feature_prefixes": ["T"]},
    "Khovanov": {"source": "kh", "feature_prefixes": ["F_"]},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the corrected primary knot analysis")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-nulls", action="store_true")
    args = parser.parse_args()

    config = ch.canonical_run_config()
    aligned = ch.build_aligned_dataset(
        base_dir=args.data_dir,
        file_map=FILE_MAP,
        representation_specs=REPRESENTATION_SPECS,
        min_crossings=config.universe.min_crossings,
        max_crossings=config.universe.max_crossings,
        output_dir=args.output_dir / "00_alignment",
        preferred_metadata_sources=["alex", "jones", "homfly", "theta", "kh"],
        expected_n=config.universe.expected_n,
        expected_s_qc_corrections=config.universe.expected_s_qc_corrections,
    )
    result = ch.run_primary_pca_analysis(
        aligned["meta"],
        aligned["X_dict"],
        args.output_dir,
        config=config,
        run_nulls=not args.skip_nulls,
    )
    print(result["summary"])


if __name__ == "__main__":
    main()
