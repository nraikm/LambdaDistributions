# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.8"]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    root = (Path(__file__).resolve().parents[1] / "new_dists/notebooks/primitive_actions.py").parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from new_dists.primitive_action_sigma_mgfs.verification import run_suite

    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Primitive-action permutation $\sigma$-MGF laboratory

    For every finite action $G\curvearrowright X$, the notebook constructs the
    permutation representation on $\mathbb C[X]$.  The zero-one matrices are
    stored exactly by their nonzero column positions.  It compares three
    independent quantities:

    1. direct fixed points of powers of each represented matrix;
    2. the structural formula for the relevant group type, followed by
       Möbius recovery of cycle counts; and
    3. direct orbits on products of multisets, which equal the target
       $\sigma$-MGF coefficients.

    $A_5$ supplies the simple-group cases.  Exhaustive $S_3$ examples are used
    for product and twisted wreath actions so that every group element and
    power can be checked quickly.
    """)
    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Construct matrix groups and run exact checks")
    mo.vstack([
        mo.callout(
            "The full suite checks representations of dimensions 5, 9, 27, "
            "36, 60, and 3600, with dense matrices avoided at the largest size.",
            kind="info",
        ),
        run,
    ])
    return (run,)


@app.cell
def _(mo, run, run_suite):
    mo.stop(
        not run.value,
        mo.callout("Press the button to run the exhaustive suite.", kind="info"),
    )
    result = run_suite()
    return (result,)


@app.cell
def _(mo, result):
    mo.callout(
        f"{'PASS' if result['passed'] else 'FAIL'}: "
        f"{result['exact checks']:,} exact matrix, fixed-point, cycle, and orbit checks.",
        kind="success" if result["passed"] else "danger",
    )
    return


@app.cell
def _(mo, result):
    almost = result["almost simple"]
    mo.vstack([
        mo.md(r"""
        ## 1. Almost-simple action: $A_5\curvearrowright A_5/A_4$

        All 60 matrices and powers $1\le r\le6$ are checked against the
        centralizer/class-intersection formula.  The table compares Molien
        coefficients with literal configuration orbits.
        """),
        mo.ui.table(list(almost["rows"]), pagination=False),
        mo.md(
            f"Class-power checks: **{almost['class-power checks']}**; "
            f"representation law: **{almost['representation law']}**."
        ),
    ])
    return


@app.cell
def _(mo, result):
    diagonal = result["diagonal and holomorph simple"]
    diagonal_compound = result["compound diagonal"]
    mo.vstack([
        mo.md(r"""
        ## 2. Simple and compound diagonal actions

        The $k=2$ and $k=3$ $A_5$ actions have dimensions 60 and 3600.
        Fixed points are computed directly from the represented permutations,
        while pair coefficients are direct stabilizer-orbit counts.  The
        compound case uses two diagonal blocks and tests both the base action
        and the added block swap.
        """),
        mo.ui.table(list(diagonal["rows"]), pagination=False),
        mo.ui.table(list(diagonal_compound["rows"]), pagination=False),
        mo.md(
            f"Centralizer sizes in $A_5$: `{diagonal['centralizer sizes']}`."
        ),
    ])
    return


@app.cell
def _(mo, result):
    cases = result["product action"]["cases"]
    rows = []
    for case in cases:
        for row in case["rows"]:
            rows.append({"k": case["k"], "degree": case["degree"], **row})
    mo.vstack([
        mo.md(r"""
        ## 3. Product action: $S_3\wr S_k\curvearrowright\{0,1,2\}^k$

        For $k=2,3$, every label tuple, top permutation, and power through six
        is compared with the marked cycle-product formula.  Ordered-tuple
        orbits and Burnside averages independently verify the exact moment
        theorem.
        """),
        mo.ui.table(rows, pagination=False),
    ])
    return


@app.cell
def _(mo, result):
    twisted = result["twisted wreath"]
    mo.vstack([
        mo.md(r"""
        ## 4. Twisted-wreath affine action

        On the regular socle $N=S_3^2$, all 72 affine matrices are tested
        against the nonabelian Lang-map formula.  Conditioning on each top
        permutation also verifies the exact Bernoulli-spike law.
        """),
        mo.ui.table(list(twisted["rows"]), pagination=False),
        mo.md(
            f"Direct $P\\backslash N$ orbits: **{twisted['pair direct']}**; "
            f"formula: **{twisted['pair formula']}**."
        ),
    ])
    return


@app.cell
def _(mo, result):
    simple = result["diagonal and holomorph simple"]
    holomorph_compound = result["holomorph compound"]
    mo.vstack([
        mo.md(r"""
        ## 5. Holomorph-simple and holomorph-compound actions

        The $A_5\times A_5$ left-right representation is checked on all 3600
        group elements.  The compound benchmark constructs all 2592 matrices
        on $S_3^2$ and checks the product-action formula and pair coefficient.
        """),
        mo.ui.table([simple["rows"][-1]], pagination=False),
        mo.ui.table([{
            "degree": holomorph_compound["degree"],
            "group order": holomorph_compound["group order"],
            "direct pair orbits": holomorph_compound["direct pair orbits"],
            "Burnside": holomorph_compound["Burnside pair coefficient"],
            "formula": holomorph_compound["formula"],
        }], pagination=False),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation

    The computations confirm the finite formulas, but the moment identities
    also show why the growing families usually have no finite coefficientwise
    $\sigma$-MGF limit.  They produce multiplicative rare events: fixed points
    converge to zero in probability while higher moments grow polynomially or
    exponentially.  This is distinct from an ordinary compound-Poisson limit.
    """)
    return


if __name__ == "__main__":
    app.run()
