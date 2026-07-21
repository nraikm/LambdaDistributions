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

    repository = (Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/matrix_group_formula_verification/a5_representations/a5_representations_verification.py").parents[3]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from for_this_guy.matrix_group_formula_verification.a5_representations.verification import (
        run_suite,
    )

    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Three representations associated with \(A_5\)

    This notebook constructs all 60 matrices in the five-dimensional
    permutation representation, its four-dimensional deleted summand, and the
    three-dimensional icosahedral rotation representation. Direct group
    averages are compared with the proposed class-spectrum formulas.
    """)
    return


@app.cell
def _(mo):
    maximum_degree = mo.ui.slider(0, 9, value=7, show_value=True, label="maximum degree")
    run = mo.ui.run_button(label="Construct A5 representations and verify")
    mo.hstack([maximum_degree, run], justify="start", gap=1)
    return maximum_degree, run


@app.cell
def _(maximum_degree, mo, run, run_suite):
    mo.stop(not run.value, mo.md("Press the button to run all three checks."))
    suite = run_suite(maximum_degree.value)
    return suite


@app.cell
def _(mo, suite):
    panels = []
    for result in suite:
        panels.extend(
            [
                mo.md(f"## {result['representation']}"),
                mo.callout(
                    "All coefficients agree." if result["passed"] else "A discrepancy was found.",
                    kind="success" if result["passed"] else "danger",
                ),
                mo.md(
                    f"Order {result['group order']}; dimension {result['dimension']}; "
                    f"selected coefficients `{result['selected coefficients']}`."
                ),
                mo.md(f"**Symmetric-square Reynolds projector:** `{result['projector Sym^2']}`"),
                mo.ui.table(list(result["rows"]), pagination=True),
            ]
        )
    mo.vstack(panels)
    return


if __name__ == "__main__":
    app.run()
