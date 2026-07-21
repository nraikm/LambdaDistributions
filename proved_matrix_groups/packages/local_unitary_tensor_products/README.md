# Local-unitary tensor-product family

This folder keeps the proof and verification for

\[
U(d_1)\times\cdots\times U(d_k)\curvearrowright
\mathbb C^{d_1}\otimes\cdots\otimes\mathbb C^{d_k}
\]

together. The ordinary one-sided sigma-MGF and the balanced two-sided
sigma-MGF are treated separately.

- `marimo_notebooks/local_unitary_tensor_products.py`: interactive marimo experiment.
- `verification.py`: exact combinatorics and direct Haar-matrix checks.
- `test_verification.py`: deterministic regression tests.
- `local_unitary_balanced_sigma_mgf_proof_and_verification.tex`: proof source.
- `local_unitary_balanced_sigma_mgf_proof_and_verification.pdf`: compiled note.

Run the notebook with:

```sh
.venv/bin/marimo edit marimo_notebooks/local_unitary_tensor_products.py
```

Run the verification with:

```sh
.venv/bin/python -m for_this_guy.local_unitary_tensor_products.verification
.venv/bin/python -m pytest -q for_this_guy/local_unitary_tensor_products/test_verification.py
```

