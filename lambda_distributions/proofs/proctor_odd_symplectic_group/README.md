# Proctor odd symplectic group

This folder keeps the proof and verification artifacts for Proctor's odd
symplectic group separate from unrelated matrix groups.

The full complex group is noncompact, so there is no normalized Haar
probability to test. The code instead verifies the well-defined replacements:

- exact maximal-compact formulas for `Sp(1) x U(1)` and the fixed-radical
  `Sp(1)` subgroup;
- the vanishing certificate for positive-degree covariant invariants of the
  full algebraic group;
- finite odd symplectic groups over prime fields in their complex permutation
  representations on projective points.

Run the exact suite from the repository root:

```bash
.venv/bin/python lambda_distributions/proofs/proctor_odd_symplectic_group/verification.py
```

Open the interactive marimo notebook:

```bash
.venv/bin/marimo edit marimo_notebooks/proctor_odd_symplectic.py
```

The compiled exposition is `proctor_odd_symplectic_proof_and_verification.pdf`.
