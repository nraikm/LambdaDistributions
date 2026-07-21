# Finite Lie-type permutation actions

This folder is the executable companion to the supplied finite Lie-type
σ-MGF formulas.

- `marimo_notebooks/finite_linear_groups.py`: marimo notebook for vector, projective, and
  complete-flag actions of small `GL`, `SL`, `PGL`, and `PSL` groups.
- `marimo_notebooks/finite_polar_groups.py`: marimo notebook for isotropic/singular point
  actions of `Sp_4(2)`, `O_4^+(2)`, and `U_4(2)`.
- `verification_core.py`: exact finite-field group construction, induced
  permutations, direct tuple-of-multisets orbit counts, cycle formulas, Möbius
  reconstruction, and numerical determinant checks.
- `linear/` and `polar/`: separate LaTeX proof and verification documents.

Run the notebooks with:

```sh
marimo run marimo_notebooks/finite_linear_groups.py
marimo run marimo_notebooks/finite_polar_groups.py
```

Run the exact automated sweep with:

```sh
pytest -q lambda_distributions/proofs/finite_lie_type_permutation_actions/test_verification.py
```
