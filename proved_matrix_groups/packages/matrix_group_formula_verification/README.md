# Matrix-group formula verification

This folder contains seven executable marimo notebooks and their independent
calculation modules. They test finite truncations of the generalized Molien
coefficient

\[
\dim\left(\bigotimes_i \operatorname{Sym}^{\tau_i}V\right)^G.
\]

## Notebooks

- `marimo_notebooks/classical_compact.py`: defining representations of SU(n), SO(n),
  and O(n), using Lie-generator nullities versus Kostka formulas.
- `marimo_notebooks/tori.py`: weighted U(1)^r representations, using alias-free finite
  diagonal grids versus zero-weight lattice counts.
- `marimo_notebooks/h3_reflection.py`: A4, S4, and A5 rotation groups plus the full H3
  reflection group, using full matrix averages versus spectral classes.
- `marimo_notebooks/restricted_monomial.py`: D(A) semidirect H and G(r,p,n), using
  matrix averages versus dual-code cycle and multiset counters.
- `marimo_notebooks/representation_variants.py`: exterior and regular representations
  of small finite groups.
- `marimo_notebooks/a5_representations.py`: permutation,
  deleted-permutation, and rotation representations of A5.
- `marimo_notebooks/binary_polyhedral.py`: the natural SU(2)
  representations of the binary tetrahedral, octahedral, and icosahedral
  groups.

Run a notebook from the repository root, for example:

```sh
marimo edit marimo_notebooks/classical_compact.py
```

The compiled proof documents and their LaTeX sources are in
`output/pdf/for_this_guy/matrix_group_formula_verification/`.

## Verified result

The suites execute 928 coefficient comparisons. All pass. The A5 rotation
case is intentionally repeated in the polyhedral and representation-centered
notebooks (883 case/representation/tau combinations are distinct). Additional finite
Reynolds projectors have numerical idempotence errors below 9e-16.

The proof documents explicitly correct two details in the supplied proposal:

1. orthogonal Schur shapes require `length(lambda) <= n` in dimension n;
2. a three-dimensional rotation factor is
   `(1-t)^(-1) (1-2*cos(theta)*t+t^2)^(-1)`.
