# Sporadic matrix-group verification

This folder separates the two 24-dimensional representations in the source
note.

- `m24_permutation/` constructs the ATLAS 24-point generators, compares
  direct orbit counts with the complete permutation class sum, and tests the
  first post-5-transitive degree.
- `co0_leech/` proves the full Conway frame-shape/parity statements and tests
  the determinant method on the explicit signed-coordinate subgroup at
  dimension 24. It does not mislabel subgroup even-degree coefficients as
  full-Co0 coefficients.

Each group has a marimo notebook, a plain Python verifier, and a compiled
proof-and-verification PDF.
