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
        / "lambda_distributions/proofs/finite_braid_images/ising_clifford"
    )
    if str(source_directory) not in sys.path:
        sys.path.insert(0, str(source_directory))
    from verification import run_sweep

    return mo, run_sweep


@app.cell
def _(mo):
    mo.md(r"""
    # Ising/Jones $B_3$ image and projective adjoint

    The two braid matrices close to a 192-element finite lift in $U(2)$.
    Their full twist is $e^{-\pi i/4}I$, giving the order-eight scalar
    selection rule.  The notebook also removes lift ambiguity by constructing
    the 24-element conjugation image on $\mathrm{End}(\mathbb C^2)$ and tests
    the projectively intrinsic formula there.
    """)
    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Close both images and verify")
    run
    return (run,)


@app.cell
def _(mo, run, run_sweep):
    mo.stop(not run.value, mo.md("Press the button to run the finite closures and projector checks."))
    rows = run_sweep()
    mo.vstack([
        mo.callout(
            "The 192-element lift and 24-element adjoint image pass every independent check.",
            kind="success",
        ),
        mo.ui.table(rows, pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()
