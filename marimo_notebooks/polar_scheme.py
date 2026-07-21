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
    root = (Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/association_scheme_permutation_representations/polar/notebook.py").parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from for_this_guy.association_scheme_permutation_representations.polar.verification import run_suite
    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Symplectic polar-space laboratory

    This notebook constructs all 720 matrices of \(Sp_4(2)\), its action on
    15 isotropic points, and its dual-polar action on the 15 maximal totally
    isotropic 2-spaces.  Matrix invariance, Möbius-recovered cycles, Molien
    coefficients, and direct configuration orbits are checked independently.
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
        mo.md(f"**All invariant-polar-subspace reconstructions:** `{result['fixed-point reconstruction']}`  \n**Determinant/cycle error:** `{error:.2e}`"),
        mo.ui.table(list(result["rows"]), pagination=True, page_size=18),
    ])
    return


if __name__ == "__main__":
    app.run()

