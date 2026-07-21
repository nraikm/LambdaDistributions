# Finite-group sigma-MGF verification

This folder tests formulas (7.1)--(7.39) from the supplied note in explicit,
small matrix groups.  The materials are separated by group/representation
family:

- `regular_representations/`: regular representations of `C_n`, `D_n`, `S_n`,
  and `A_n`;
- `symmetric_group_representations/`: sign, standard, derived, and
  `Ind_{C_3}^{S_3}(omega)` representations;
- `permutation_actions/`: the actions of `S_n` and `A_n` on `k`-subsets.
- `hyperoctahedral_cube/`: the affine action of
  `B_n = C_2^n rtimes S_n` on the `2^n` vertices of the binary cube, with
  exhaustive matrix, orbit, fixed-point, and signed-cycle checks for
  `1 <= n <= 5`.

Each folder contains a marimo notebook.  The corresponding LaTeX source and
compiled PDF are under `output/pdf/lambda_distributions/proofs/finite_group_sigma_mgf/` in the
same family layout.

Run the regression suite from the repository root:

```sh
python -m pytest -q lambda_distributions/proofs/finite_group_sigma_mgf/test_verification.py
```

Run a notebook, for example:

```sh
marimo run marimo_notebooks/regular_representations.py
```

The direct coefficient route builds the Reynolds projector on
`tensor_b Sym^{tau_b}(V)`.  Independent routes use element orders, cycle
types, fixed-point/Mobius inversion, or coset monodromy.  All interactive
controls are deliberately capped because symmetric-power dimensions grow
quickly.
