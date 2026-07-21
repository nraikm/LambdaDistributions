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

    repository = (Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/matrix_group_formula_verification/binary_polyhedral/binary_polyhedral_verification.py").parents[3]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from for_this_guy.matrix_group_formula_verification.binary_polyhedral.verification import (
        run_suite,
    )

    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Binary polyhedral groups in their natural \(SU(2)\) representation

    The notebook reconstructs \(2T\), \(2O\), and \(2I\) by lifting explicit
    rotation matrices to unit quaternions and then to \(2\times2\) complex
    matrices. It compares direct averages with the proposed paired-lift
    functions \(B_\theta\), checks closure, and verifies the odd-total-degree
    cancellation forced by the central matrix \(-I\).
    """)
    return


@app.cell
def _(mo):
    maximum_degree = mo.ui.slider(0, 10, value=8, show_value=True, label="maximum degree")
    run = mo.ui.run_button(label="Build binary groups and verify")
    mo.hstack([maximum_degree, run], justify="start", gap=1)
    return maximum_degree, run


@app.cell
def _(maximum_degree, mo, run, run_suite):
    mo.stop(not run.value, mo.md("Press the button to construct all three binary groups."))
    suite = run_suite(maximum_degree.value)
    return suite


@app.cell
def _(mo, suite):
    panels = []
    for result in suite:
        panels.extend(
            [
                mo.md(f"## {result['group']}"),
                mo.callout(
                    "Class formula and parity filter pass."
                    if result["passed"]
                    else "A discrepancy was found.",
                    kind="success" if result["passed"] else "danger",
                ),
                mo.md(f"**Diagnostics:** `{result['diagnostics']}`"),
                mo.md(f"**Reynolds projector on V tensor V:** `{result['projector V tensor V']}`"),
                mo.ui.table(list(result["rows"]), pagination=True),
            ]
        )
    mo.vstack(panels)
    return


if __name__ == "__main__":
    app.run()
