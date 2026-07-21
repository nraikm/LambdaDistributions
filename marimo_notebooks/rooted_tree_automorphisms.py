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

    here = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/limiting_lambda_claims/rooted_tree_automorphisms/notebook.py").parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from verification import run_sweep
    return mo, run_sweep


@app.cell
def _(mo):
    mo.md(r"""
    # Rooted-tree automorphisms: wreath/plethysm laboratory

    The complete (b)-ary height-(h) group is constructed recursively as
    (G_h=G_{h-1}\wr S_b) on its (b^h) leaves. Every permutation matrix is
    materialized. For six mixed symmetric-power coefficients, a direct matrix
    average is compared with

    \[
    [u^b]\exp\!\left(\sum_{\ell\ge1}
      \frac{u^\ell}{\ell}\mathcal M_{h-1}(\mathbf t^\ell)\right).
    \]

    This verifies the exact finite-tree law. It does not assert the open
    general limiting Lambda-distribution for random rooted trees.
    """)
    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Construct tree groups and verify plethysm")
    run
    return (run,)


@app.cell
def _(mo, run, run_sweep):
    mo.stop(not run.value, mo.md("Press the button to run the exact wreath-product checks."))
    result = run_sweep()
    summaries = [
        {key: value for key, value in case.items() if key != "rows"}
        for case in result["cases"]
    ]
    mo.vstack([
        mo.callout(
            f"PASS: {result['total matrices']} matrices and {len(result['rows'])} exact coefficients.",
            kind="success",
        ),
        mo.ui.table(summaries),
        mo.ui.table(result["rows"], pagination=True, page_size=12),
    ])
    return


if __name__ == "__main__":
    app.run()
