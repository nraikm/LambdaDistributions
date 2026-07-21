# Central extensions and projective representations

- `verification.py` constructs the matrix groups and runs independent formula checks.
- `../notebooks/central_extensions.py` is the marimo laboratory, organized by group family.
- `central_extensions_projective_representations_proof_and_verification.tex` is the standalone proof note.
- `central_extensions_projective_representations_proof_and_verification.pdf` is its compiled and visually checked PDF.

Run the checks with:

```bash
.venv/bin/python -m pytest lambda_distributions/dists/central_extensions_projective_representations/test_verification.py
.venv/bin/marimo run marimo_notebooks/central_extensions.py
```

The combined proof anthology and its PDF are intentionally not modified.
