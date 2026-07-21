import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import sys

    root = str((Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/nonuniform_lambda_distributions/cyclic_groups/notebook.py").parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)
    from lambda_distributions.proofs.nonuniform_lambda_distributions.cyclic_groups.verification import run_suite
    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Nonuniform $\Lambda$-distributions on cyclic matrix groups

    This notebook constructs $C_3,C_4,C_5$ as explicit diagonal unitary
    matrix groups in dimensions $1,2,3$. It compares direct matrix-character
    averages with weight-space Fourier sums for an arbitrary measure and for
    convolution walks. It also checks the unexpanded determinant target at a
    stable pair of variables.
    """)
    degree = mo.ui.slider(2, 7, value=5, label="maximum total degree")
    degree
    return (degree,)


@app.cell
def _(degree, run_suite):
    suite = run_suite(maximum_degree=degree.value)
    return (suite,)


@app.cell
def _(mo, suite):
    status = "PASS" if suite["passed"] else "FAIL"
    mo.vstack(
        [
            mo.md(
                f"## {status}: {suite['checks']} comparisons; "
                f"maximum error `{suite['maximum_error']:.3e}`"
            ),
            mo.md("### Direct determinant checks"),
            mo.ui.table(suite["sigma_rows"]),
            mo.md("### Coefficient and convolution checks"),
            mo.ui.table(suite["rows"], pagination=True, page_size=25),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
