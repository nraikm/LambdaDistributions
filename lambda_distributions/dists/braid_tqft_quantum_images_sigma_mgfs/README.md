# Braid, TQFT, and quantum-image sigma-MGFs

This package verifies the finite-image and compact-closure formulas using
explicitly constructed matrix groups. Group families are kept in separate
sections because a braid/TQFT label alone does not specify a representation.

Run the test suite:

```bash
.venv/bin/python -m pytest -q \
  lambda_distributions/dists/braid_tqft_quantum_images_sigma_mgfs/test_verification.py
```

Run the marimo notebook:

```bash
.venv/bin/marimo run \
  marimo_notebooks/braid_tqft.py
```

The standalone proof note is
`braid_tqft_quantum_images_sigma_mgfs_proof_and_verification.tex`, with the
compiled PDF beside it. Neither combined proof PDF was edited.
