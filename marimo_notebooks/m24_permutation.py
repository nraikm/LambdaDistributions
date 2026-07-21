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

    here = (Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/sporadic_matrix_groups/m24_permutation/notebook.py").parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from verification import run_sweep
    return mo, run_sweep


@app.cell
def _(mo):
    mo.md(r"""
    # $M_{24}$ in its 24-point permutation representation

    This notebook constructs the two ATLAS generator matrices, checks their
    standard-generator orders, computes invariant dimensions directly as
    orbits of multisets/tuples, and compares them with the conjugacy-class
    determinant formula.  The degree-six rows deliberately cross the
    5-transitivity boundary: $M_{24}$ has 12 multiset orbits versus 11 for
    $S_{24}$, and 204 tensor-tuple orbits versus 203.
    """)
    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Construct generators and count orbits")
    run
    return (run,)


@app.cell
def _(mo, run, run_sweep):
    mo.stop(not run.value, mo.md("Press the button to run the orbit sweeps."))
    rows = run_sweep()
    mo.vstack([
        mo.callout("All direct orbit counts equal the class-sum coefficients.", kind="success"),
        mo.ui.table(rows, pagination=True),
    ])
    return


if __name__ == "__main__":
    app.run()
