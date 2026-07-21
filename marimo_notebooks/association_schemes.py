# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo>=0.23.8", "numpy>=1.24"]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    root = (Path(__file__).resolve().parents[1] / "new_dists/notebooks/association_schemes.py").parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from lambda_distributions.dists.association_scheme_sigma_mgfs.verification import run_suite

    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Association-scheme permutation $\sigma$-MGF laboratory

    For a finite action $G\curvearrowright\Omega$, every represented matrix is
    the permutation matrix on $\mathbb C[\Omega]$.  This notebook checks

    \[
    F_r(g)=|\operatorname{Fix}_\Omega(g^r)|,
    \qquad
    c_d(g)=\frac1d\sum_{r\mid d}\mu(d/r)F_r(g),
    \]

    and compares the direct matrix-trace average, the cycle-product formula,
    and literal orbits on products of multisets.  Results are grouped by the
    construction that supplies $F_r$: homogeneous spaces, wreath products,
    affine translation actions, and graph products.
    """)
    return


@app.cell
def _(run_suite):
    result = run_suite()
    return (result,)


@app.cell
def _(mo, result):
    mo.callout(
        f"{'PASS' if result['passed'] else 'FAIL'}: "
        f"{result['coefficient comparisons']} exact coefficient comparisons.",
        kind="success" if result["passed"] else "danger",
    )
    return


@app.cell
def _(mo, result):
    homogeneous = result["homogeneous"]
    _rows = []
    for case in homogeneous.values():
        _rows.extend(case["rows"])
    mo.vstack([
        mo.md(r"""
        ## 1. Homogeneous, parabolic, and building actions

        The constructed examples are Johnson $S_v$ actions, Grassmann
        $GL_n(q)$ actions, $Sp_4(2)$ on polar points and Lagrangians, and
        $GL_3(2)$ on the 21 complete flags of its type-$A_2$ building.  The
        flag case independently checks
        \[
        |\operatorname{Fix}_{G/H}(x)|
        =\frac{|C_G(x)|\,|x^G\cap H|}{|H|}.
        \]
        """),
        mo.ui.table(_rows, pagination=True),
    ])
    return


@app.cell
def _(mo, result):
    _rows = list(result["wreath"]["Hamming"]["rows"])
    mo.vstack([
        mo.md(r"""
        ## 2. Product and wreath actions

        Exhaustive $S_q\wr S_d$ checks for $(q,d)=(2,2),(2,3),(3,2)$ compare
        direct fixed words with the marked-cycle monodromy formula.  The same
        calculation is reused below for the product factors of a Doob graph.
        """),
        mo.ui.table(_rows, pagination=True),
    ])
    return


@app.cell
def _(mo, result):
    affine = result["affine"]
    summary = [
        {
            "action": case["name"],
            "|A|": case["|A|"],
            "|H image|": case["|H image|"],
            "|A semidirect H image|": case["|A semidirect H image|"],
            "fixed formula": case["fixed-point formula"],
            "pass": case["passed"],
        }
        for case in affine.values()
    ]
    _rows = [row for case in affine.values() for row in case["rows"]]
    mo.vstack([
        mo.md(r"""
        ## 3. Translation schemes and affine-polar actions

        The suite explicitly constructs bilinear, alternating, Hermitian, and
        quadratic form spaces, the affine polar action $\mathbb F_2^4\rtimes
        O_4^+(2)$, and the folded and halved 4-cubes.  For every affine group
        element and every relevant power it solves
        \[
        (I-h^r)x=(I+h+\cdots+h^{r-1})b
        \]
        and compares the kernel/image answer with direct fixed points.
        """),
        mo.ui.table(summary, pagination=False),
        mo.ui.table(_rows, pagination=True),
    ])
    return


@app.cell
def _(mo, result):
    graphs = result["graphs"]
    _rows = [row for name in ("Shrikhande", "rook", "Doob") for row in graphs[name]["rows"]]
    mo.vstack([
        mo.md(r"""
        ## 4. Graph automorphisms and Doob products

        The affine automorphism group of the Shrikhande Cayley graph and the
        wreath-product automorphism group of the $4\times4$ rook graph are
        constructed directly.  Their ordered-pair coefficients are 4 and 3,
        despite identical strongly regular parameters.  For $D(1,1)$, the
        direct value is $4\cdot2=8$, and every tested power satisfies the
        product fixed-point formula.
        """),
        mo.ui.table(_rows, pagination=True),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What the computation proves - and what it does not

    The universal identities are proved symbolically in the companion PDF.
    These finite calculations are an independent audit of specialization and
    implementation.  The asymptotic statements are likewise proof statements:
    bounded-support families stabilize, while growing-rank families with an
    unbounded ordered-pair coefficient cannot converge coefficientwise.
    """)
    return


if __name__ == "__main__":
    app.run()
