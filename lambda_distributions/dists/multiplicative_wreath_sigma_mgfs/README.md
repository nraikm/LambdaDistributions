# Multiplicative wreath-product sigma-MGF verification

This package audits the exact and asymptotic formulas for wreath class powers,
product actions, tensor-induced modules, fixed-tail bipartitions, diagonal
coset actions, twisted regular actions, and graded Fock modules.

Run the exact suite:

```bash
.venv/bin/python -m lambda_distributions.dists.multiplicative_wreath_sigma_mgfs.verification
```

Run the marimo laboratory:

```bash
.venv/bin/marimo run marimo_notebooks/multiplicative_wreath.py
```

The matrix representations are exact.  Permutation matrices are stored by
their nonzero column positions, and monomial matrices additionally store their
signs.  This avoids dense zero storage without changing the represented group.
