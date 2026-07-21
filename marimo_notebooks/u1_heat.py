import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import sys

    root = str((Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/nonuniform_lambda_distributions/compact_u1_heat/notebook.py").parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)
    from for_this_guy.nonuniform_lambda_distributions.compact_u1_heat.verification import run_suite
    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Heat-kernel $\Lambda$-distributions on $U(1)$

    This notebook builds diagonal $U(1)$ representations with one, two, and
    three weights. Direct trapezoidal integration against the Fourier heat
    kernel is compared with the Casimir-filter formula
    $\sum_q m_q e^{-q^2t/2}$. The endpoint $t=0$ is evaluated directly at the
    identity and large time exhibits Haar projection onto weight zero.
    """)
    degree = mo.ui.slider(2, 7, value=5, label="maximum total degree")
    grid = mo.ui.dropdown([256, 512, 1024], value=512, label="quadrature grid")
    mo.hstack([degree, grid], justify="start")
    return degree, grid


@app.cell
def _(degree, grid, run_suite):
    suite = run_suite(maximum_degree=degree.value, grid_size=grid.value)
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
            mo.md(
                f"Grid: {suite['grid_size']} angles; heat Fourier cutoff: "
                f"{suite['cutoff']}."
            ),
            mo.ui.table(suite["rows"], pagination=True, page_size=25),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
