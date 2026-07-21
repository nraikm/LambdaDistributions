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
    root = (Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/association_scheme_permutation_representations/hamming/notebook.py").parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from for_this_guy.association_scheme_permutation_representations.hamming.verification import run_suite
    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Hamming wreath-product laboratory

    The full group \(S_q^D\rtimes S_D\) is constructed, including independent
    alphabet relabelings.  For every element, coordinate-cycle label products
    give the fixed-word formula (16); Möbius inversion then recovers the exact
    cycle structure on \(Q^D\).  Matrix, cycle, and orbit calculations are
    compared through total degree three.
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
        mo.md(f"**(q,D) cases:** `{result['cases']}`  \n**All fixed-point reconstructions:** `{result['fixed-point reconstruction']}`  \n**Determinant/cycle error:** `{error:.2e}`"),
        mo.ui.table(list(result["rows"]), pagination=True, page_size=18),
    ])
    return


if __name__ == "__main__":
    app.run()

