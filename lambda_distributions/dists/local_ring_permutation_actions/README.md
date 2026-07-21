# Finite local-ring permutation representations

This standalone package verifies the sigma-MGF formulas for finite matrix
groups over `Z / p^a Z` acting through permutation modules.

- `verification.py` constructs the groups and sets, computes direct orbits,
  and checks Smith-kernel, homogeneous-space, Möbius-cycle, Grassmann-pair,
  formed-space, and adjoint-congruence formulas.
- `../notebooks/local_ring_actions.py` is the marimo laboratory, divided by group/action family.
- `test_verification.py` is the exact automated test suite.
- `local_ring_permutation_actions_proof_and_verification.tex` and `.pdf` are
  the standalone proof note, ready for later inclusion in the anthology.

Run:

```sh
marimo run marimo_notebooks/local_ring_actions.py
pytest -q lambda_distributions/dists/local_ring_permutation_actions/test_verification.py
```
