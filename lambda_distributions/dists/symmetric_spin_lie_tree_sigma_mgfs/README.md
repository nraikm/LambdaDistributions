# Symmetric, spin-cover, Lie, and rooted-tree sigma-MGFs

This standalone package verifies the formulas supplied for ordinary
symmetric-group modules, a genuine Schur-cover representation, Foulkes and
multilinear free-Lie modules, iterated local wreath products, and arbitrary
finite rooted trees.

Run the noninteractive audit from the workspace root:

```bash
.venv/bin/python -m lambda_distributions.dists.symmetric_spin_lie_tree_sigma_mgfs.verification
```

Open the interactive laboratory with:

```bash
.venv/bin/marimo edit marimo_notebooks/symmetric_spin_lie_tree.py
```

The proof document is
`symmetric_spin_lie_tree_sigma_mgfs_proof_and_verification.tex`; its sibling
PDF is the compiled standalone artifact.  Neither file is included in or
written into the combined proof PDF.
