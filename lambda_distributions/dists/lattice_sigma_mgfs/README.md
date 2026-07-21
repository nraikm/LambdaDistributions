# Lattice automorphism sigma-MGFs

This folder contains a standalone proof and exact verification of the
frame-shape sigma-MGF for positive-definite lattices.  The cases are grouped
as type-A root lattices, signed-permutation root/cubic lattices, explicit
low-symmetry lattices, repeated orthogonal sums, and tensor products.

Run the exact suite with

```bash
python3 -m lambda_distributions.dists.lattice_sigma_mgfs.verification
```

Open the interactive audit with

```bash
marimo edit marimo_notebooks/lattices.py
```

The standalone LaTeX/PDF is intentionally independent of
`proved_matrix_groups/all_new_matrix_groups_proofs.pdf`; the combined PDF is
not modified by this package.
