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

    here = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/quantum_matrix_groups/clifford_groups/notebook.py").parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from verification import run_sweep
    return mo, run_sweep


@app.cell
def _(mo):
    mo.md(r"""
    # Clifford groups: balanced sigma-MGF verification

    The one-sided sigma-MGF depends on scalar phases and therefore does not
    encode unitary-design order.  This notebook closes the one- and two-qubit
    projective Clifford groups (orders 24 and 11,520), computes
    $|G|^{-1}\sum_g|\mathrm{Tr}(g)|^{2k}$ directly, and compares it with the
    Haar $U(2^n)$ value.  It also retains a finite-lift scalar-rule check.

    Only the phase-neutral sector $|\alpha|=|\beta|$ descends to a projective
    group.  This is slightly sharper than calling the entire two-sided series
    projectively well-defined.
    """)
    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Close the matrix groups and verify")
    run
    return (run,)


@app.cell
def _(mo, run, run_sweep):
    mo.stop(not run.value, mo.md("Press the button to run the exact finite sweeps."))
    rows = run_sweep()
    mo.vstack(
        [
            mo.callout(
                "Passed: Clifford agrees with Haar for k <= 3 and first differs at k = 4.",
                kind="success",
            ),
            mo.ui.table(rows, pagination=True),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
