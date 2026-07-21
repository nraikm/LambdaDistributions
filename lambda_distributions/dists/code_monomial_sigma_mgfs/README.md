# Code-monomial sigma-MGFs

This standalone package proves and verifies the cyclic-alphabet formula for

\[
G(C,P)=D_C\rtimes P\leq U(N),\qquad
C\leq(\mathbb Z/r\mathbb Z)^N.
\]

The verification is organized by group family: pure diagonal, binary
semidirect, ternary semidirect, and a composite-alphabet semidirect example.
It constructs dense matrices and compares the direct generalized Molien
average with weighted-cycle, root-filtered dual-code, and exact orbit-count
formulas.

Run the exact suite from the repository root:

```bash
.venv/bin/python -m lambda_distributions.dists.code_monomial_sigma_mgfs.verification
```

Run or edit the marimo notebook:

```bash
.venv/bin/marimo edit marimo_notebooks/code_monomial.py
```

Run the package tests:

```bash
.venv/bin/pytest -q lambda_distributions/dists/code_monomial_sigma_mgfs/test_verification.py
```

The standalone proof note is
`code_monomial_sigma_mgfs_proof_and_verification.pdf`.  The combined proof
anthology is intentionally not modified by this package.
