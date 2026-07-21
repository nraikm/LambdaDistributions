import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import sys

    root = str((Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/nonuniform_lambda_distributions/symmetric_groups/notebook.py").parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)
    from lambda_distributions.proofs.nonuniform_lambda_distributions.symmetric_groups.verification import run_suite
    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Nonuniform measures on symmetric permutation groups

    Explicit permutation matrices for $S_3,S_4,S_5$ test lazy random
    transpositions, random $3$-cycles, adjacent transpositions, inverse riffle
    shuffles, Ewens measure, and general cycle weights. The $S_3$ section also
    performs a full irreducible Fourier decomposition and exposes the scalar
    central operator versus the matrix-valued adjacent-transposition operator.
    """)
    degree = mo.ui.slider(2, 6, value=4, label="maximum total degree")
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
            mo.md("### Central versus noncentral Fourier operators"),
            mo.ui.table(suite["operator_rows"]),
            mo.md("### $S_3$ Fourier-operator checks"),
            mo.ui.table(suite["fourier_rows"], pagination=True, page_size=20),
            mo.md("### $S_n$ matrix-versus-cycle checks"),
            mo.ui.table(suite["cycle_rows"], pagination=True, page_size=25),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
