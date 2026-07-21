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

    here = (Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/quantum_matrix_groups/scalar_extended_pauli/notebook.py").parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from verification import run_sweep
    return mo, run_sweep


@app.cell
def _(mo):
    mo.md(r"""
    # Scalar-extended Pauli verification

    Construct \(\mu_4P_{2,n}\) for \(n=1,2\) and \(\mu_6P_{3,1}\), then
    compare the explicit generalized Molien coefficient with the proposed
    central/noncentral formula.  These cases check the stronger scalar rule
    \(s\mid |\tau|\), including mixed partitions.

    The qubit calculation also diagnoses the sign typo in the supplied
    “qubit recovery” paragraph: the noncentral coefficient is positive.
    """)
    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Run scalar-extended checks")
    run
    return run


@app.cell
def _(mo, run, run_sweep):
    mo.stop(not run.value, mo.md("Press the button to enumerate the finite matrix groups."))
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

