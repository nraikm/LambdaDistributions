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

    here = (Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/quantum_matrix_groups/extraspecial_groups/notebook.py").parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from verification import run_sweep
    return mo, run_sweep


@app.cell
def _(mo):
    mo.md(r"""
    # Extraspecial-group spectral checks

    The odd-prime section compares the two nonisomorphic groups of order
    \(3^3\): the Heisenberg group of exponent 3 and
    \(C_9\rtimes C_3\) of exponent 9.  The 2-group section constructs the
    faithful two-dimensional representations of \(D_8\) and \(Q_8\), testing
    the plus/minus quadratic-form sectors separately.
    """)
    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Run extraspecial checks")
    run
    return run


@app.cell
def _(mo, run, run_sweep):
    mo.stop(not run.value, mo.md("Press the button to construct and average the matrices."))
    rows = run_sweep()
    passed = all(row["passed"] for row in rows)
    mo.vstack([
        mo.callout(
            f"{'All' if passed else 'Not all'} {len(rows)} checks passed.",
            kind="success" if passed else "danger",
        ),
        mo.ui.table(rows, pagination=True),
    ])
    return


if __name__ == "__main__":
    app.run()

