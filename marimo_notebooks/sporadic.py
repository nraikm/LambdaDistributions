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

    root = (Path(__file__).resolve().parents[1] / "new_dists/notebooks/sporadic.py").parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from lambda_distributions.dists.sporadic_group_sigma_mgfs.verification import run_suite

    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Sporadic-group $\sigma$-MGF laboratory: $M_{11}$

    A sporadic group alone does not determine a $\sigma$-MGF; a representation
    must be chosen.  This notebook tests two related choices for $M_{11}$:

    - the natural $11$-dimensional permutation representation; and
    - its $10$-dimensional deleted permutation constituent.

    The direct route generates all $7{,}920$ group elements from the ATLAS
    standard generators and constructs every matrix.  It is compared with an
    independently entered class-size/cycle-shape formula, the equivalent
    power-character formula, and generator orbits on multiset bases.
    """)
    return


@app.cell
def _(mo):
    degree = mo.ui.slider(1, 6, value=6, label="Maximum total degree")
    run = mo.ui.run_button(label="Construct both matrix groups and verify")
    mo.vstack([degree, run])
    return degree, run


@app.cell
def _(degree, mo, run, run_suite):
    mo.stop(
        not run.value,
        mo.callout("Press the button to run the exhaustive checks.", kind="info"),
    )
    result = run_suite(degree.value)
    return (result,)


@app.cell
def _(mo, result):
    representation = result["representation"]
    mo.vstack([
        mo.callout(
            f"{'PASS' if result['passed'] else 'FAIL'}: generated "
            f"{representation['group order']:,} elements in dimensions 11 and 10; "
            f"all {result['exact coefficient checks']} coefficient comparisons agree.",
            kind="success" if result["passed"] else "danger",
        ),
        mo.ui.table([representation], pagination=False),
    ])
    return


@app.cell
def _(mo, result):
    selected = {
        (1,), (2,), (1, 1), (3,), (2, 1), (4,),
        (3, 1), (2, 2), (1, 1, 1, 1), (6,),
    }
    rows = [
        {**row, "tau": str(row["tau"])}
        for row in result["coefficients"]
        if row["tau"] in selected
    ]
    mo.vstack([
        mo.md(r"""
        ## Exact coefficient comparison

        For each partition $\tau$, the direct value averages products of
        complete symmetric characters computed from traces of the explicit
        matrices.  The other columns use only class sizes, cycle shapes, and
        characters of powers.
        """),
        mo.ui.table(rows, pagination=True),
    ])
    return


@app.cell
def _(mo, result):
    orbit_rows = [{**row, "tau": str(row["tau"])} for row in result["orbits"]]
    mo.vstack([
        mo.md(r"""
        ## Independent orbit check for the permutation representation

        The multiset-basis states are traversed using only the two generators,
        without enumerating conjugacy classes or evaluating determinants.
        """),
        mo.ui.table(orbit_rows, pagination=False),
    ])
    return


@app.cell
def _(mo, result):
    mo.vstack([
        mo.md(r"""
        ## Direct evaluation of the full multivariate function

        At $(t_1,t_2,t_3)=(0.037,-0.051,0.083)$, every matrix determinant is
        evaluated directly and averaged.  The class formula is evaluated
        separately from the eight aggregated spectra.
        """),
        mo.ui.table(list(result["numerical"]), pagination=False),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Scope

    The proof is universal for any finite complex representation once its
    character and power maps are fixed.  The computation is intentionally
    concrete: it validates the representation choice, the $M_{11}$ class
    transcription, the deletion of the fixed line, and coefficient extraction.
    There is no intrinsic asymptotic over the 26 sporadic simple groups because
    they do not form an infinite family.
    """)
    return


if __name__ == "__main__":
    app.run()

