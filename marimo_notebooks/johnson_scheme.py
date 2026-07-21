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
    root = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/association_scheme_permutation_representations/johnson/notebook.py").parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from lambda_distributions.proofs.association_scheme_permutation_representations.johnson.verification import run_suite
    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Johnson permutation-representation laboratory

    This notebook constructs the full induced permutation matrices on
    \(k\)-subsets.  It includes the complementation coset for \(J(4,2)\),
    reconstructs every vertex-cycle count from the fixed-subset formulas
    (9) and (14), and compares matrix traces, cycle factors, and explicit
    configuration orbits through total degree three.
    """)
    return


@app.cell
def _(run_suite):
    result = run_suite()
    return (result,)


@app.cell
def _(mo, result):
    direct, cycle, error = result["numerical determinant"]
    mo.vstack([
        mo.callout(f"{'PASS' if result['passed'] else 'FAIL'}: {len(result['rows'])} exact coefficient comparisons.", kind="success" if result["passed"] else "danger"),
        mo.md(f"**Cases:** `{result['cases']}`  \n**All fixed-point reconstructions:** `{result['fixed-point reconstruction']}`  \n**Determinant/cycle error:** `{error:.2e}`"),
        mo.ui.table(list(result["rows"]), pagination=True, page_size=18),
    ])
    return


if __name__ == "__main__":
    app.run()

