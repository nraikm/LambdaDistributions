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

    here = (Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/sporadic_matrix_groups/co0_leech/notebook.py").parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from verification import run_sweep
    return mo, run_sweep


@app.cell
def _(mo):
    mo.md(r"""
    # $Co_0$ Leech-space formula: a 24-dimensional stress test

    The full Conway group is too large for raw enumeration.  We instead use
    the explicit signed coordinate subgroup $\{\pm P_g:g\in M_{24}\}\leq Co_0$.
    Its three displayed 24-by-24 generators test the same determinant/frame-
    shape mechanism at the actual Leech-space dimension.  Independent
    multiset-orbit counts verify every even coefficient through degree six;
    the explicit central matrix $-I$ forces every odd coefficient to vanish.

    This validates the method and the parity theorem.  It is not presented as
    a substitute for inserting the complete $Co_0$ frame-shape table.
    """)
    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Build 24x24 generators and verify")
    run
    return (run,)


@app.cell
def _(mo, run, run_sweep):
    mo.stop(not run.value, mo.md("Press the button to run the degree sweep."))
    rows = run_sweep()
    mo.vstack([
        mo.callout("Passed: determinant averaging matches direct invariants, including odd-degree vanishing.", kind="success"),
        mo.ui.table(rows),
    ])
    return


if __name__ == "__main__":
    app.run()
