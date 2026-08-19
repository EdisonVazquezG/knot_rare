# Validation report

Validation performed in the refactor workspace:

- Python byte-compilation passed for package, scripts, and tests.
- Package-root import passed without TensorFlow installed; the optional AE API
  stays isolated.
- Configuration and notebook JSON both parse successfully.
- Eight direct synthetic tests passed:
  - identity removal at the crossing filter;
  - conservative s-invariant sign correction;
  - corrected-universe audit;
  - deterministic top-tail ties using stable IDs;
  - plus-one Monte Carlo p-values;
  - all four nested intersection nulls;
  - match-group preservation in caliper matching;
  - continuous C_k output contract.
- A synthetic five-representation end-to-end alignment passed, including
  exclusion of `00_1`, synchronized feature arrays, one s correction, and the
  requested expected-N assertion.

The raw 313,230-row invariant files were not included in the uploaded source
archive, so the expensive PCA/AE/permutation analyses were not rerun here.
`docs/RESULTS_CHECKLIST.md` separates already corrected reported values from
outputs that must be regenerated with the new runner.
