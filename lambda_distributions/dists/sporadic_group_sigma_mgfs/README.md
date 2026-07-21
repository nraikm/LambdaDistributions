# Sporadic-group sigma-MGFs: M11

This standalone package verifies the complete sigma-MGF for two related
complex representations of the Mathieu sporadic simple group `M11`:

- the natural 11-point permutation representation;
- its 10-dimensional deleted permutation constituent.

Run the exact verification suite from the repository root:

```bash
.venv/bin/python -m lambda_distributions.dists.sporadic_group_sigma_mgfs.verification
```

Open the interactive notebook with:

```bash
.venv/bin/marimo edit marimo_notebooks/sporadic.py
```

The standalone proof is `sporadic_group_sigma_mgfs_proof_and_verification.tex`
and its compiled PDF.  No combined anthology file is modified.

