import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import sys
    repo_root = str((Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/matrix_group_formula_verification/tori/notebook.py").parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from lambda_distributions.proofs.matrix_group_formula_verification.tori.verification import run_suite
    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Compact tori: matrix average versus zero-weight arrays

    The notebook constructs diagonal root-of-unity matrix groups. If the
    grid modulus is larger than twice the largest possible coordinate
    weight in the tested truncation, congruence to zero cannot alias a
    nonzero integer weight. The finite matrix average is therefore exactly
    the continuous-torus constant term in this degree range.
    """)
    return


@app.cell
def _(run_suite):
    suite = run_suite()
    return (suite,)


@app.cell
def _(mo, suite):
    mo.vstack(
        [
            mo.md(f"## {'PASS' if suite['passed'] else 'FAIL'}: {suite['checks']} coefficient checks"),
            mo.ui.table(list(suite["rows"]), pagination=True, page_size=20),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
