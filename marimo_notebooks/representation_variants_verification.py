# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.8", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import sys

    import marimo as mo

    folder = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/matrix_group_formula_verification/representation_variants/representation_variants_verification.py").parent
    shared = folder.parent
    repository_root = folder.parents[2]
    for path in (repository_root, shared, folder):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from verification import run_suite

    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # One master formula, different representations

    Two independent tests are grouped here:

    1. (\Lambda^2) of the three-dimensional reflection representation of (S_4):
       explicit compound matrices versus pairwise products of the original eigenvalues.
    2. The six-dimensional regular representation of (S_3): explicit left-regular
       permutation matrices versus the element-order formula with counts (1,3,2).
    """)
    return


@app.cell
def _(mo):
    maximum_degree = mo.ui.slider(0, 7, value=5, show_value=True, label="maximum degree")
    run = mo.ui.run_button(label="Construct representations and verify")
    mo.hstack([maximum_degree, run], justify="start", gap=1)
    return maximum_degree, run


@app.cell
def _(maximum_degree, mo, run, run_suite):
    mo.stop(not run.value, mo.md("Press the button to run both representation checks."))
    suite = run_suite(maximum_degree.value)
    return suite


@app.cell
def _(mo, suite):
    panels = []
    for name, result in suite.items():
        panels.extend(
            [
                mo.md(f"## {name}"),
                mo.callout(
                    "All coefficients agree." if result["passed"] else "A discrepancy was found.",
                    kind="success" if result["passed"] else "danger",
                ),
                mo.md(
                    f"Order {result['order']}, representation dimension {result['dimension']}. "
                    f"Projector check: `{result['projector']}`"
                ),
                mo.ui.table(list(result["rows"]), pagination=True),
            ]
        )
    mo.vstack(panels)
    return


if __name__ == "__main__":
    app.run()
