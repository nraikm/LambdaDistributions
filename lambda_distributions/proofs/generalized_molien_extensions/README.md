# Generalized Molien extensions

This folder tests the three unrelated representation families from the supplied
statement.  They are deliberately separated by group family:

- `code_monomial/`: diagonal code groups semidirect coordinate automorphisms;
- `delta_su3/`: the three-dimensional groups Delta(3n^2) and Delta(6n^2);
- `psl2_projective_line/`: PSL(2,q) on the complex permutation module of the
  projective line.

`marimo_notebooks/generalized_molien_extensions.py` is a Marimo dashboard that runs all three suites.  Each family
also has its own verification module and LaTeX/PDF note, so assumptions and
computational methods do not bleed between genuinely different groups.

Run everything from the repository root:

```bash
.venv/bin/python -m lambda_distributions.proofs.generalized_molien_extensions.run_all
.venv/bin/python -m pytest lambda_distributions/proofs/generalized_molien_extensions/test_verification.py -q
.venv/bin/marimo run marimo_notebooks/generalized_molien_extensions.py
```

