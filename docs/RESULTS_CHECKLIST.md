# Manuscript results checklist after removing the identity

This table distinguishes values already rerun on the corrected universe from
values that must not be copied from the old frozen notebook.

| Result | Corrected status | Manuscript action |
|---|---:|---|
| Analysis universe | 313,230 | Replace every 313,231 census statement. |
| Crossing range | 3–15 | State explicitly that `00_1` is excluded at ingestion. |
| Signature 0 count | 80,015 | Replace 80,016; other signature counts are unchanged. |
| Raw alternating sigma/s mismatches | 1 (`10_004`) | Describe as source-data QC, not a mathematical exception. |
| QC alternating sigma/s mismatches | 0 | Use `s_invariant_qc` downstream. |
| Overall sigma=s | 291,144 after QC | The raw-source count 291,143 is an audit only. |
| Overall sigma!=s | 22,086 after QC | The raw-source count 22,087 is an audit only. |
| PCA EVR-99 dimensions | 4, 10, 32, 10, 77 | Corrected and frozen primary configuration. |
| 1% tail size | 3,133 per view | Corrected. |
| PCA-SSE five-way consensus | 292 | Corrected primary result. |
| Median signature in consensus | 10 | Corrected. |
| Signature >=10 in consensus | 231/292 | Corrected. |
| Signature >=12 in consensus | 20/292 | Corrected. |
| Crossing 15 in consensus | 283/292 | Corrected. |
| EVR-99.9 consensus | 85 | Sensitivity; overlap with primary is 56. |
| Full matched intersection null | mean 204.056, max 220 | Observed 292; report plus-one Monte Carlo p, not p=0. |
| Norm/matched-control p-values | rerun required | Replacement controls now use match-group inference. |
| Full-data AE outputs | descriptive, rerun required | Removing identity and target-free defaults may alter training. |
| Held-out AE/PCA outputs | rerun required | Freeze both target-free primary split and old stratified sensitivity separately. |
| Continuous C_k / referee controls | rerun required | Old bootstrap asserts N=313,231 and is stale. |

## Values that remain qualitatively unchanged

The corrected PCA result still identifies a very small five-view consensus
strongly enriched for large signature, large Rasmussen s, 15 crossings, and
alternating knots. The identity removal changes the census and tiny background
percentages; it does not change the reported primary consensus size of 292.

## Values that should not be mixed

- EVR-99 (`4,10,32,10,77`) is the primary PCA configuration.
- EVR-99.9 (`5,13,45,16,115`) is a sensitivity analysis.
- Fixed compression (`5,10,20,10,50`) is a separate control.
- Raw SSE, NRE, norm-conditioned percentiles, cross-fitted residuals, and C_k
  are distinct scores and require distinct names and output directories.
