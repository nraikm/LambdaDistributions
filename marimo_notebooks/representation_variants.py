import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import sys
    repo_root = str((Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/matrix_group_formula_verification/representation_variants/notebook.py").parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from lambda_distributions.proofs.matrix_group_formula_verification.representation_variants.verification import run_suite
    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Alternate representations of finite matrix groups

    This checks two claims from the general framework: functorial weight
    substitution using $\bigwedge^2$ of the standard $S_4$ reflection
    representation, and the element-order formula for the regular
    representation of $S_3$. Explicit matrices are enumerated in both cases.
    """)
    return


@app.cell
def _(run_suite):
    suite = run_suite()
    return (suite,)


@app.cell
def _(mo, suite):
    rows = [{"representation": name, **row} for name, result in suite.items() for row in result["rows"]]
    passed = all(result["passed"] for result in suite.values())
    projectors = {name: result["projector"] for name, result in suite.items()}
    mo.vstack(
        [
            mo.md(f"## {'PASS' if passed else 'FAIL'}: {len(rows)} checks"),
            mo.md("### Reynolds-projector diagnostics"),
            mo.json(projectors),
            mo.ui.table(rows, pagination=True, page_size=20),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
