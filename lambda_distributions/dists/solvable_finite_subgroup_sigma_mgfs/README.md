# Solvable and finite-subgroup sigma-MGF verification

This package verifies the representation-dependent formulas for finite
lamplighter and metacyclic groups, affine and Frobenius actions, triangular
and flag actions, finite defining subgroups of `SU(3)`, `SU(4)`, `Sp(4)`, and
`SO(4)`, and symplectic reflection wreath products.

Run the verification suite:

```bash
.venv/bin/python -m lambda_distributions.dists.solvable_finite_subgroup_sigma_mgfs.verification
```

Run the marimo notebook:

```bash
.venv/bin/marimo run marimo_notebooks/solvable_finite.py
```

The standalone proof document is
`solvable_finite_subgroup_sigma_mgfs_proof_and_verification.tex` with its
compiled PDF beside it. The combined proof anthology is intentionally not
modified.
