# Primitive-action sigma-MGF verification

This package verifies the almost-simple, diagonal, product-action,
twisted-wreath, and holomorph formulas from the accompanying standalone proof
note.

Run the exact suite:

```bash
.venv/bin/python -m lambda_distributions.dists.primitive_action_sigma_mgfs.verification
```

Run the marimo laboratory:

```bash
.venv/bin/marimo run marimo_notebooks/primitive_actions.py
```

The implementation stores a permutation matrix by its unique nonzero entry in
each column.  This is exact and permits the degree-3600 cases to be constructed
without allocating dense 3600-by-3600 arrays.
