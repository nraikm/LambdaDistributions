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

    root = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/finite_braid_images/cyclic_scalar/notebook.py").parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from core import braid_diagnostics, cyclic_scalar_braid_image, verify_case
    from verification import run_sweep
    return braid_diagnostics, cyclic_scalar_braid_image, mo, run_sweep, verify_case


@app.cell
def _(mo):
    mo.md(r"""
    # Scalar cyclic finite braid images

    Send both generators of $B_3$ to $\zeta_m I_D$.  The notebook closes the
    matrix group, checks unitarity and the braid relation, forms the Reynolds
    projector on $\bigotimes_a\mathrm{Sym}^{\tau_a}(\mathbb C^D)$, and compares
    its rank with the elementwise and conjugacy-class Molien formulas.
    """)
    return


@app.cell
def _(mo):
    order = mo.ui.slider(2, 8, value=4, label="scalar order m", show_value=True)
    dimension = mo.ui.slider(1, 4, value=3, label="dimension D", show_value=True)
    tau_text = mo.ui.text(value="2, 2", label="tau")
    run = mo.ui.run_button(label="Construct and verify")
    mo.hstack([order, dimension, tau_text, run], justify="start", gap=1)
    return dimension, order, run, tau_text


@app.cell
def _(braid_diagnostics, cyclic_scalar_braid_image, dimension, mo, order, run, tau_text, verify_case):
    mo.stop(not run.value, mo.md("Choose parameters and press **Construct and verify**."))
    tau = tuple(int(item.strip()) for item in tau_text.value.split(",") if item.strip())
    group, generators = cyclic_scalar_braid_image(order.value, dimension.value)
    diagnostics = braid_diagnostics(generators)
    row = verify_case(group, tau, f"C_{order.value} scalar B_3 image")
    expected = row["target_dimension"] if sum(tau) % order.value == 0 else 0
    passed = diagnostics["passed"] and row["passed"] and row["projector_rank"] == expected
    row.update({"selection_prediction": expected, **diagnostics})
    mo.vstack([
        mo.callout(
            "All three coefficient routes and the scalar selection rule agree."
            if passed else "A check failed; inspect the diagnostics.",
            kind="success" if passed else "danger",
        ),
        mo.ui.table([row]),
    ])
    return


@app.cell
def _(mo, run_sweep):
    mo.vstack([mo.md("## Representative dimensions 1-4"), mo.ui.table(run_sweep(), pagination=False)])
    return


if __name__ == "__main__":
    app.run()

