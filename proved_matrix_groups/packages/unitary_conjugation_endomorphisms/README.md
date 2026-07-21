# Unitary conjugation on endomorphisms

This folder verifies the sigma-MGF formulas for the conjugation action

\[
U(n)\curvearrowright W=\operatorname{End}(\mathbb C^n),
\qquad A\mapsto gAg^{-1}.
\]

Contents:

- `unitary_conjugation_core.py`: matrix construction, direct common-kernel
  computation, Littlewood--Richardson enumeration, stable orbit count, and
  necklace Euler-product coefficient extraction.
- `marimo_notebooks/unitary_conjugation.py`
  (from the repository root): interactive marimo notebook.
- `test_verification.py`: automated regression tests.

Run the checks and notebook from the repository root with

```bash
python3 proved_matrix_groups/packages/unitary_conjugation_endomorphisms/unitary_conjugation_core.py
python3 -m pytest proved_matrix_groups/packages/unitary_conjugation_endomorphisms/test_verification.py
marimo edit marimo_notebooks/unitary_conjugation.py
```

The proof source and compiled PDF live in
`output/pdf/for_this_guy/unitary_conjugation_endomorphisms/`.
