# Consensus hardness — corrected reproducible run

This refactor freezes the prime-knot universe at crossings 3–15. The unknot
`00_1` is removed during preprocessing, before alignment, scaling, PCA, data
splits, or neural models. The raw Rasmussen value is preserved in
`s_invariant_original`; analyses use the audited `s_invariant_qc` column.

## Install and import

```bash
pip install -e .
```

```python
import consensus_hardness as ch
```

The package root exports the curated public API, so notebooks no longer need
one import statement per module. TensorFlow-dependent functions are exported
when TensorFlow is installed.

## Run order

| Stage | Role | Default manuscript status |
|---|---|---|
| `00_alignment` | Load, canonicalize mirrors, exclude identity, align five views | Required |
| `01_universe_qc` | Assert N=313,230 and audit the `10_004` s correction | Required |
| `02_primary_pca` | Fixed EVR-99 dimensions, PCA-SSE 1% tails, 5-way consensus | Primary |
| `03_intersection_nulls` | Four nested intersection null models | Primary |
| norm / matching | Norm diagnostics and exact+nearest matched controls | Robustness |
| autoencoders / heldout | Nonlinear and held-out reconstruction | Robustness |
| LOO / tail / conditional | Leave-one-view-out, tail mass, norm-adjusted hardness | Sensitivity |
| continuous C_k / external ML | Harmonization, anomaly models, KnotInfo | Exploratory |

Run the primary analysis from a shell:

```bash
python scripts/run_primary.py --data-dir /path/to/Invariants --output-dir results/run_001
```

Every run writes stable knot IDs, an analysis-universe audit, stage-specific
tables, and a JSON manifest. Integer row positions are retained only as a
convenience and are never the sole saved identifier.

The reusable C_k core is in `consensus_hardness.continuous`. Memory-intensive
concatenated anomaly baselines and external-metadata coverage helpers are kept
in `consensus_hardness.exploratory`; they are not part of the primary run.

## Important configurations that must not be conflated

- EVR-99 primary k: `4, 10, 32, 10, 77`.
- EVR-99.9 sensitivity k: `5, 13, 45, 16, 115`.
- Fixed-compression check: `5, 10, 20, 10, 50`.
- Full-data AE is descriptive; held-out AE is the leakage-resistant check.
- Target-free train/validation/test splitting is now the default. The old
  `signature_bin`-stratified split must be requested explicitly as sensitivity.
