import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import sys
    repo_root = str((Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/matrix_group_formula_verification/restricted_monomial/notebook.py").parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from lambda_distributions.proofs.matrix_group_formula_verification.restricted_monomial.verification import (
        g_r_p_n_case,
        run_suite,
    )
    from lambda_distributions.proofs.matrix_group_formula_verification.common import projector_check
    return g_r_p_n_case, mo, projector_check, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Restricted monomial matrix groups

    This notebook tests the novel dual-code cycle counter for
    $D(A)\rtimes H$ and the specialized coefficient formula for
    $G(r,p,n)$. Every allowed diagonal codeword and permutation is turned
    into a matrix before the generalized Molien coefficient is averaged.
    """)
    return


@app.cell
def _(g_r_p_n_case, projector_check, run_suite):
    suite = run_suite()
    projector = projector_check(g_r_p_n_case(2, 2, 3).matrices, (2, 1))
    return projector, suite


@app.cell
def _(mo, projector, suite):
    rows = [{"case": result["case"], **row} for result in suite for row in result["rows"]]
    passed = all(result["passed"] for result in suite)
    mo.vstack(
        [
            mo.md(f"## {'PASS' if passed else 'FAIL'}: {len(rows)} checks"),
            mo.md("### Independent Reynolds-projector diagnostic for G(2,2,3)"),
            mo.json(projector),
            mo.ui.table(rows, pagination=True, page_size=20),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
