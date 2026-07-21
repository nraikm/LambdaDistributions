# /// script
# requires-python = ">=3.10"
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

    root = (Path(__file__).resolve().parents[1] / "new_dists/notebooks/lattices.py").parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from lambda_distributions.dists.lattice_sigma_mgfs.verification import run_suite

    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Lattice automorphism $\sigma$-MGF laboratory

    For a positive-definite integral lattice $L$ and
    $V_L=L\otimes_{\mathbb Z}\mathbb C$, this notebook tests

    \[
      \mathcal S_L(\mathbf t)=\frac1{|\operatorname{Aut}(L)|}
      \sum_g\prod_i\det(I-t_i g)^{-1}
      =\sum_\tau\dim\!\left(
      \bigotimes_j\operatorname{Sym}^{\tau_j}V_L\right)^Gm_\tau.
    \]

    Every finite case is built from concrete integral matrices.  The audit
    deliberately keeps three coefficient routes separate:

    1. matrix-power traces and Newton recurrence;
    2. frame shapes recovered by Möbius inversion;
    3. the proposed family cycle formula.

    For $\tau=(1),(2),(1,1),(3)$ it additionally constructs the represented
    symmetric-power matrices and solves the simultaneous fixed equations over
    $\mathbb Q$.  Use the slider to rerun all partitions through a chosen total
    degree; degree four is the documented verification target.
    """)
    return


@app.cell
def _(mo):
    degree = mo.ui.slider(2, 4, value=4, step=1, label="maximum total degree")
    degree
    return (degree,)


@app.cell
def _(degree, run_suite):
    result = run_suite(degree.value)
    return (result,)


@app.cell
def _(mo, result):
    mo.callout(
        f"{'PASS' if result['passed'] else 'FAIL'}: {result['case count']} matrix cases; "
        f"{result['coefficient comparisons']} coefficient-route comparisons, "
        f"{result['fixed-space comparisons']} fixed-space comparisons, and "
        f"{result['frame comparisons']} elementwise frame checks.",
        kind="success" if result["passed"] else "danger",
    )
    return


@app.cell
def _(mo, result):
    summaries = [
        {
            "family": case["section"],
            "matrix image": case["case"],
            "dimension": case["dimension"],
            "|matrix image|": case["group order"],
            "lattice/structure check": case["lattice preservation"],
            "frame check": case["frame pass"],
            "pass": case["passed"],
        }
        for case in result["cases"]
    ]
    mo.vstack([
        mo.md(r"""
        ## Overview

        The tensor actions are not faithful: $(-I,-I)$ is in the kernel, so
        their displayed matrix-image orders are $72$ and $144$, rather than
        the abstract local and wreath orders $144$ and $288$.  Averaging is
        unchanged because every image element has the same number of lifts.
        """),
        mo.ui.table(summaries, pagination=False),
    ])
    return


@app.cell
def _(mo, result):
    sections = (
        (
            "Root lattices A",
            r"""## 1. Type $A$ root lattices

            The matrices are $\pm P_\sigma$ on the basis
            $e_1-e_n,\ldots,e_{n-1}-e_n$ for $n=3,4,5$.  The formula route
            groups permutations by cycle partition and uses
            \[
            \operatorname{Tr}((\varepsilon\sigma)^r\mid V)
            =\varepsilon^r\left(\sum_{m\mid r}m c_m(\sigma)-1\right).
            \]""",
        ),
        (
            "Signed permutation B/C/D",
            r"""## 2. Signed-permutation lattices

            All signed permutation matrices in dimensions $2,3,4$ are
            enumerated.  The independent formula assigns one sign product
            $\delta_C\in\{\pm1\}$ to each permutation cycle $C$ and uses
            \[
            \operatorname{Tr}(g^r)=
            \sum_{|C|\mid r}|C|\,\delta_C^{r/|C|}.
            \]""",
        ),
        (
            "Small symmetry",
            r"""## 3. Lattices with only scalar symmetry

            Three explicit positive-definite Gram matrices are searched with
            a rigorous Gershgorin coordinate bound.  The complete integral
            automorphism enumeration returns exactly $\{\pm I\}$ in ranks
            $2,3,4$, and the coefficients match the parity-projected binomial
            formula.""",
        ),
        (
            "Repeated orthogonal sums",
            r"""## 4. Repeated orthogonal sums

            The order-$288$ image $\operatorname{Aut}(A_2)\wr S_2$ is built
            as $4\times4$ block-monomial matrices on $A_2\perp A_2$.  Its
            coefficients are compared with the marked cycle-product formula,
            the $k=2$ specialization of the compound-Poisson generating law.""",
        ),
        (
            "Tensor products",
            r"""## 5. Tensor products

            Both $\operatorname{Aut}(A_2)^2$ and
            $\operatorname{Aut}(A_2)\wr S_2$ are built on
            $V_{A_2}\otimes V_{A_2}$.  Direct Kronecker matrices are compared
            with the product-of-traces formula and, for the wreath action,
            \[
            \operatorname{Tr}(x^r)=\prod_C
            \chi(a_C^{r/d_C(r)})^{d_C(r)},\qquad
            d_C(r)=\gcd(|C|,r).
            \]""",
        ),
    )
    views = []
    for section, description in sections:
        cases = [case for case in result["cases"] if case["section"] == section]
        coefficient_rows = [
            {"matrix image": case["case"], **row}
            for case in cases
            for row in case["rows"]
        ]
        fixed_rows = [
            {"matrix image": case["case"], **row}
            for case in cases
            for row in case["fixed-space rows"]
        ]
        views.append(
            mo.vstack([
                mo.md(description),
                mo.md("**Coefficient comparison**"),
                mo.ui.table(coefficient_rows, pagination=True),
                mo.md("**Independent fixed-space comparison**"),
                mo.ui.table(fixed_rows, pagination=True),
            ])
        )
    mo.vstack(views)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Scope of the finite audit

    The generalized Molien and frame-shape identities are proved for every
    finite lattice automorphism group in the companion PDF.  The notebook
    checks tractable representatives and the structural formulas for growing
    families.  It does not pretend to enumerate the full automorphism groups
    of the Niemeier lattices, the Coxeter--Todd lattice, or isolated extremal
    lattices; those require complete external class-size and frame-shape data.
    Their exact answer remains the finite class sum proved in the document.
    """)
    return


if __name__ == "__main__":
    app.run()
