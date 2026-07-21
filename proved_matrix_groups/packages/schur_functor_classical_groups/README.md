# Schur-functor representations of the compact classical groups

The material is separated by group:

- `unitary/`: the exact holomorphic obstruction for `U(n)` and the nontrivial
  joint law with the conjugate;
- `orthogonal/`: the `O(n)` plethystic pushforward and even-degree invariant;
- `symplectic/`: the compact `USp(2n)` pushforward and alternating-form
  invariant. (This is the compact group denoted `Sp(2n)` in the source note.)

Each folder contains a marimo notebook, a command-line verification module,
and a compiled proof-and-verification PDF. Shared Young-symmetrizer matrix
construction and Jacobi--Trudi formula code is in `common.py`.

Run the deterministic suites from the repository root:

```bash
.venv/bin/python -m for_this_guy.schur_functor_classical_groups.unitary.verification
.venv/bin/python -m for_this_guy.schur_functor_classical_groups.orthogonal.verification
.venv/bin/python -m for_this_guy.schur_functor_classical_groups.symplectic.verification
```

Open a notebook, for example, with:

```bash
.venv/bin/marimo edit marimo_notebooks/schur_unitary.py
```

