# Finite-group exact formula verification

The material is separated by group family:

- `dicyclic/`: dicyclic/generalized quaternion representations, including
  single irreducibles, direct sums, one-dimensional characters, and the regular
  representation.
- `symmetric/`: the exact finite-`n` formula for the permutation representation
  of `S_n`.
- `alternating/`: the exact finite-`n` formula for `A_n`, including the
  distinct-label orbit-splitting correction.

Each subfolder contains an interactive marimo notebook, a LaTeX source, and its
compiled proof/verification PDF.  The notebooks construct the matrix groups and
compare a direct Reynolds-projector rank with independent character,
cycle/weight, and orbit-count formulas.

From the repository root, run a notebook with (for example):

```bash
.venv/bin/marimo run marimo_notebooks/dicyclic_exact.py
```

Run the complete small-case regression suite with:

```bash
.venv/bin/pytest -q lambda_distributions/proofs/finite_group_exact_formulas/test_verification_core.py
```

The common implementation is in `verification_core.py`; target tensor spaces
are capped to prevent accidental large allocations.
