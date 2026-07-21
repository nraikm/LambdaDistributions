# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.8", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    source_directory = (
        Path(__file__).resolve().parents[1]
        / "lambda_distributions/proofs/finite_braid_images/s3_permutation"
    )
    if str(source_directory) not in sys.path:
        sys.path.insert(0, str(source_directory))
    from verification import run_sweep

    return mo, run_sweep


@app.cell
def _(mo):
    mo.md(r"""
    # The $B_3\twoheadrightarrow S_3$ permutation representation

    Adjacent braid generators become the transposition permutation matrices
    $(12)$ and $(23)$ on $\mathbb C^3$.  This is a non-scalar finite image with
    three conjugacy classes.  Every row compares an explicit Reynolds rank
    against both the six-element Molien average and its three-class reduction.
    """)
    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Construct S3 and verify")
    run
    return (run,)


@app.cell
def _(mo, run, run_sweep):
    mo.stop(not run.value, mo.md("Press the button to close the matrix group."))
    rows = run_sweep()
    mo.vstack([
        mo.callout("Braid relation, closure order 6, projector ranks, and both formulas agree.", kind="success"),
        mo.ui.table(rows, pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()
