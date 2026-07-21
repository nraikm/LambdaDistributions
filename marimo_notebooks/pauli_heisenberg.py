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

    here = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/quantum_matrix_groups/pauli_heisenberg/notebook.py").parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from verification import run_sweep
    return mo, run_sweep


@app.cell
def _(mo):
    mo.md(r"""
    # Qudit Pauli and finite-Heisenberg verification

    This notebook constructs the Weyl matrices themselves and evaluates
    \(\frac1{|G|}\sum_g\prod_i h_{\tau_i}(\operatorname{spec}g)\).
    It tests \(q=3,5,9\), with the \(q=9\) case implemented in
    \(\mathbb F_3[u]/(u^2+1)\).  The abstract \(H_3(\mathbb F_9)\) sweep
    deliberately includes the threefold kernel of the central character.
    """)
    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Construct matrices and verify all cases")
    run
    return run


@app.cell
def _(mo, run, run_sweep):
    mo.stop(not run.value, mo.md("Press the button to run the exact representative sweep."))
    rows = run_sweep()
    passed = all(row["passed"] for row in rows)
    mo.vstack([
        mo.callout(
            f"{'All' if passed else 'Not all'} {len(rows)} direct coefficient checks passed.",
            kind="success" if passed else "danger",
        ),
        mo.ui.table(rows, pagination=True),
    ])
    return


if __name__ == "__main__":
    app.run()

