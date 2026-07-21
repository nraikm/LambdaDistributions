# Lambda distributions

This project is a small laboratory for checking formulas for
Lambda-distributions of finite matrix groups.  The reusable mathematics lives
in `lambda_distributions/`; runnable experiments live in `notebooks/` and are
Marimo applications.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the layer boundaries, normalization
conventions, and the recommended process for adding a theorem.

Completed proof-and-verification packages have a consolidated, non-disruptive
entry point at [`proved_matrix_groups/`](proved_matrix_groups/). Its
[`RESULTS_SUMMARY.md`](proved_matrix_groups/RESULTS_SUMMARY.md) records the
verified outcomes and the latest completed-suite regression run.

## Organization

```text
lambda_distributions/
  core.py           partitions and z_lambda
  groups.py         finite matrix-group representations
  distribution.py   p_lambda, h_lambda, averages, and sigma-MGF coefficients
  formulas.py       proposed closed formulas
  sampling.py       valid random unitary/symplectic matrix experiments
  verification.py   formula-versus-enumeration reports
marimo_notebooks/lambda_lab.py      inspect one group and one partition interactively
  marimo_notebooks/formula_checks.py  check a formula over every partition up to a degree
  marimo_notebooks/cyclic_monomial_walkthrough.py
                     derive and test the C_n monomial-basis theorem
marimo_notebooks/cyclic.py, marimo_notebooks/dihedral.py, marimo_notebooks/dicyclic.py, marimo_notebooks/pauli.py
                     focused Marimo apps for individual group families
verify_*.py          batch and combinatorial formula-checking apps
sigma-mgfs/          group catalog and sigma-MGF explorer
tests/
  test_lambda_distributions.py
archive/
  dated, recoverable cleanup snapshots and inventories
```

Generated caches, temporary renders, stale build products, and superseded
copies are moved into dated folders under `archive/` rather than mixed with
the active source tree.

The package contains no prompts, printing, or notebook state.  A formula is
just a function from a partition to a number, so a new conjecture can be tested
without modifying the averaging engine.

The original top-level scripts have been converted into focused Marimo clients
of the package.  They no longer duplicate group classes or depend on Sage
globals and terminal `input()` calls.  New formulas belong in `formulas.py`, not
inside a notebook cell.

Included constructors cover cyclic characters, cyclic permutation and real
two-dimensional representations, dihedral and dicyclic groups, Pauli groups,
alternating groups, and generalized symmetric groups ``C_l wreath S_n``.
`sampling.py` contains the separate numerical utilities for random unitary and
real symplectic matrices.

## Run

All repository marimo apps are cataloged under
[`marimo_notebooks/`](marimo_notebooks/README.md); proof backends and pytest
files remain beside their family sources. Each Marimo file contains PEP 723
dependency metadata. From a fresh checkout,
run a notebook in an isolated environment with only `uv` installed globally:

```bash
uvx marimo run --sandbox marimo_notebooks/cyclic_monomial_walkthrough.py
```

Open it for editing the same way:

```bash
uvx marimo edit --sandbox marimo_notebooks/cyclic_monomial_walkthrough.py
```

The inline metadata points to this repository's package as an editable local
dependency. Changes under `lambda_distributions/` therefore appear immediately
without installing the package globally. The repository must remain alongside
the notebook because that dependency is a relative path.

For repeated development across all notebooks, create the shared project
environment once:

```bash
uv sync
uv run marimo run marimo_notebooks/lambda_lab.py
```

The focused applications can be run the same way, for example:

```bash
uvx marimo run --sandbox marimo_notebooks/cyclic.py
uvx marimo run --sandbox marimo_notebooks/alternating.py
uvx marimo run --sandbox marimo_notebooks/sigma_mgfs.py
```

Use `uvx marimo edit --sandbox` instead of `uvx marimo run --sandbox` to
change an experiment live.

## Minimal library example

```python
from lambda_distributions import check_formula, cyclic_character
from lambda_distributions.formulas import cyclic_character_moment

n, k = 8, 3
group = cyclic_character(n, k)
report = check_formula(
    group,
    lambda partition: cyclic_character_moment(partition, n, k),
    max_degree=10,
)
assert report.passed
```

To add a group, return a `FiniteMatrixGroup` from a constructor in `groups.py`.
To test a new formula, write it in `formulas.py` (or directly in a notebook)
and pass it to `check_formula`.  Power-sum formulas use `basis="power_sum"`;
formulas expressed in complete homogeneous symmetric functions use
`basis="homogeneous"`.
