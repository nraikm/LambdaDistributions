# Non-Haar Fourier-weighted sigma-MGFs

This standalone package verifies the non-Haar master formula in three
separate settings:

- central and noncentral random walks on the natural and deleted matrix
  representations of `S3`;
- heat-kernel measure on `U(1)` acting as `diag(z,z^-1)`;
- Ewens measures on natural permutation matrices for `S3` through `S6`.

Run the exact tests with:

```bash
.venv/bin/python -m pytest -q \
  lambda_distributions/dists/non_haar_fourier_sigma_mgfs/test_verification.py
```

Open the marimo laboratory with:

```bash
.venv/bin/marimo edit \
  marimo_notebooks/non_haar_fourier.py
```

Build the standalone proof document with:

```bash
cd lambda_distributions/dists/non_haar_fourier_sigma_mgfs
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  non_haar_fourier_sigma_mgfs_proof_and_verification.tex
```

The repository's combined proof PDF is intentionally not edited.
