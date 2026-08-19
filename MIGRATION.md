# Migration from the exploratory notebooks

## Imports

Replace the repeated module imports and reload calls with:

```python
import consensus_hardness as ch
```

Use `ch.function_name(...)`. If bare function names are essential, the curated
`from consensus_hardness import *` works. During package development only:

```python
ch = ch.reload_package()
```

Restart the kernel before a frozen manuscript run.

## Canonical names

| Ambiguous exploratory name | Canonical name |
|---|---|
| `fixed_results_k99` / `fixed_k_results_99` | `pca_primary_results` |
| `hard_sets_1pct` / `hard_sets_sse_099` | `pca_primary_hard_sets` |
| `consensus_all5` / `consensus_99` | `pca_primary_consensus` |
| `fixed_results_k999` | `pca_evr999_results` |
| k=`5,10,20,10,50` | `pca_fixed_compression_results` |
| `s_invariant` in outcomes | `s_invariant_qc` |

Do not reuse the same variable for the EVR-99, EVR-99.9, and fixed-compression
analyses. They answer different sensitivity questions.

## Retired assumptions

- Any bootstrap or assertion expecting 313,231 rows is stale: it includes the
  identity `00_1`.
- Do not remove the identity after PCA or after splitting. It is excluded at
  ingestion through `min_crossings=3`.
- Do not save hard sets only as row numbers. Save `knot_id_base` as well.
- Monte Carlo p-values are `(exceedances + 1) / (repetitions + 1)`, never zero.
- Replacement-matched controls are analyzed by match group; duplicated control
  rows are not treated as independent observations.
- The default held-out split is target-free. `stratify_col='signature_bin'` is
  retained only to reproduce the earlier sensitivity run.

## Analysis hierarchy

1. Primary: corrected universe, fixed EVR-99 PCA, SSE 1% tails, five-way
   intersection, enrichment, and four nested nulls.
2. Robustness: EVR-99.9, fixed compression, NRE/MSE, persistent membership,
   leave-one-representation-out, tail sensitivity, norm matching, held-out PCA,
   and multi-seed held-out AE.
3. Norm-adjusted sensitivity: conditional percentiles, cross-fitted residuals,
   bin sensitivity, and stratified permutation nulls.
4. Exploratory: continuous C_k harmonization, isolation/LOF/one-class SVM, and
   external KnotInfo checks.
