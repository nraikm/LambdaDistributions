# Finite braid-image sigma-MGF checks

This folder tests the finite-image formulas in the supplied braided-fusion-category note at the matrix-group level.  Each substantially different image has its own notebook, verification module, LaTeX source, and PDF.

- `cyclic_scalar/`: scalar cyclic quotients of `B_3`, dimensions 1 through 4.
- `s3_permutation/`: the 3D permutation representation of `B_3 -> S_3`.
- `ising_clifford/`: the 2D Ising/Jones image and the 4D projective adjoint representation.

The shared `core.py` compares an explicit Reynolds-projector rank with the elementwise power-trace Molien coefficient and the conjugacy-class version.  Run all automated checks with:

```bash
.venv/bin/pytest -q lambda_distributions/proofs/finite_braid_images/test_verification.py
```

Open a notebook, for example, with:

```bash
.venv/bin/marimo edit marimo_notebooks/braid_ising_clifford.py
```

