# Nonuniform Lambda-distribution verification

The material is grouped by matrix group, with one executable marimo notebook
and one proof PDF per independent family.

- `cyclic_groups`: arbitrary measures and convolution walks on diagonal
  representations of `C3`, `C4`, and `C5`.
- `symmetric_groups`: permutation matrices for `S3`, `S4`, and `S5`, including
  random transpositions, random 3-cycles, adjacent transpositions, riffle
  shuffles, Ewens measures, cycle weights, and an `S3` Fourier decomposition.
- `compact_u1_heat`: heat-kernel interpolation on `U(1)` in dimensions 1, 2,
  and 3.

Compiled reports are under
`output/pdf/lambda_distributions/proofs/nonuniform_lambda_distributions/`, grouped by the same
three families. The editable LaTeX sources remain beside their corresponding
notebooks and verification modules.

Run all automated checks from the repository root:

```bash
.venv/bin/python -m pytest -q \
  lambda_distributions/proofs/nonuniform_lambda_distributions/test_verification.py
```

Launch a notebook, for example:

```bash
.venv/bin/marimo edit \
  marimo_notebooks/nonuniform_symmetric.py
```
