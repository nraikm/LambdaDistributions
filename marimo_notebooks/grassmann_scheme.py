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
    root = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/association_scheme_permutation_representations/grassmann/notebook.py").parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from lambda_distributions.proofs.association_scheme_permutation_representations.grassmann.verification import run_suite
    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Grassmann permutation-representation laboratory

    Prime-field matrices in \(GL_N(q)\) and all RREF \(k\)-subspaces are
    explicitly enumerated.  Invariant subspaces of every matrix power supply
    (21), Möbius inversion supplies (22), and the resulting cycle formula is
    checked against actual permutation matrices and direct configuration orbits.
    """)
    return


@app.cell
def _(run_suite):
    result = run_suite()
    return (result,)


@app.cell
def _(mo, result):
    _, _, error = result["numerical determinant"]
    mo.vstack([
        mo.callout(f"{'PASS' if result['passed'] else 'FAIL'}: {len(result['rows'])} exact coefficient comparisons.", kind="success" if result["passed"] else "danger"),
        mo.ui.table(list(result["cases"]), pagination=False),
        mo.md(f"**All invariant-subspace reconstructions:** `{result['fixed-point reconstruction']}`  \n**Determinant/cycle error:** `{error:.2e}`"),
        mo.ui.table(list(result["rows"]), pagination=True, page_size=18),
    ])
    return


if __name__ == "__main__":
    app.run()

