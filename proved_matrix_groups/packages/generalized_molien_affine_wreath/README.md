# Generalized Molien verification: affine and wreath-product families

This companion keeps the two unrelated group families in separate folders.

## Finite affine groups

- Marimo notebook: `marimo_notebooks/affine_groups.py`
- Exact verification engine: `affine_groups/verification.py`
- Proof source and compiled PDF:
  `../../output/pdf/for_this_guy/generalized_molien_affine_wreath/affine_groups/`

The code constructs `AGL_n(q)` for prime `q`, constructs its permutation
matrices on `F_q^n`, enumerates orbits of tuples of multisets literally, and
compares those counts with the cycle-index coefficient formula.  It also
checks the fixed-point moment formula and the advertised stable range.

## Wreath products

- Marimo notebook: `marimo_notebooks/affine_wreath_products.py`
- Exact verification engine: `wreath_products/verification.py`
- Proof source and compiled PDF:
  `../../output/pdf/for_this_guy/generalized_molien_affine_wreath/wreath_products/`

The code constructs block-monomial matrices for `H wr S_n`, averages the
generalized Molien coefficients directly, compares them with the exact
cycle-product recurrence, and checks the stable series in degrees at most
`n`.  Exact integer models of the sign representation of `C2` and the
two-dimensional rational representation of `C3` are included.

## Run

From the repository root:

```sh
.venv/bin/python for_this_guy/generalized_molien_affine_wreath/affine_groups/verification.py
.venv/bin/python for_this_guy/generalized_molien_affine_wreath/wreath_products/verification.py
.venv/bin/marimo edit marimo_notebooks/affine_groups.py
.venv/bin/marimo edit marimo_notebooks/affine_wreath_products.py
```
