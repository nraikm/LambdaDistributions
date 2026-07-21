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

    root = (Path(__file__).resolve().parents[1] / "new_dists/notebooks/central_extensions.py").parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from lambda_distributions.dists.central_extensions_projective_representations.verification import run_suite

    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Central extensions and projective representations

    This laboratory constructs the matrices, computes the target quantities
    directly, and compares them with independent formulas.  Results are kept
    separate by group family:

    1. odd extraspecial Heisenberg groups (E_n(p));
    2. the (+) and (-) extraspecial groups (D_8) and (Q_8);
    3. the binary octahedral basic-spin representation (2.S_4\to U(2));
    4. its scalar lift (mu_4,2.S_4) and projective quotient;
    5. finite Weil projective representations of
       (mathrm{SL}(2,3)) and (mathrm{SL}(2,5)).

    For a partition (	au), the matrix route averages
    
    \[
      \prod_j \operatorname{Tr}(\operatorname{Sym}^{\tau_j}g).
    \]

    It is compared with central root filters and the closed extraspecial
    coefficient formulas.  Balanced trace moments are compared with finite
    symplectic orbit counts and with the Haar values in the Clifford
    (3)-design range.
    """)
    return


@app.cell
def _(run_suite):
    result = run_suite(max_degree=6)
    assert result["passed"]
    return (result,)


@app.cell
def _(mo, result):
    mo.callout(
        f"PASS: {result['groups']} constructed matrix groups; "
        f"{result['coefficient comparisons']} coefficient checks; "
        f"{result['determinant comparisons']} determinant-average checks; "
        f"{result['moment comparisons']} balanced checks; maximum error "
        f"{result['maximum error']:.3e}.",
        kind="success",
    )
    return


@app.cell
def _(mo, result):
    mo.vstack([
        mo.md(r"""
        ## 1. Matrix-group audit

        Every listed matrix is checked for distinctness and unitarity.  Exact
        multiplication closure is checked for groups of order at most (125);
        (E_2(3)) uses the exact Heisenberg parametrization
        ((c,a,b)\in\mathbf F_3\times\mathbf F_3^2\times\mathbf F_3^2),
        whose multiplication law proves closure without a quadratic matrix
        scan.  Weil groups are closed projectively, as appropriate for their
        phase-ambiguous representation.
        """),
        mo.ui.table(list(result["group rows"]), pagination=False),
    ])
    return


@app.cell
def _(mo, result):
    rows = {}
    for _det_row in result["determinant rows"]:
        rows.setdefault(_det_row["family"], []).append(_det_row)
    mo.vstack([
        mo.md(r"""
        ## 2. Full generalized Molien integrand averages

        At ((t_1,t_2)=(0.07,0.11)), the dense calculation averages
        (prod_i\det(I-t_i g)^{-1}) over every group element.  The comparator
        uses only the spectral-sector formulas (29) and (38).
        """),
        mo.ui.tabs({family: mo.ui.table(values, pagination=False) for family, values in rows.items()}),
    ])
    return


@app.cell
def _(mo, result):
    selected_taus = {"()", "(1,)", "(2,)", "(1, 1)", "(3,)", "(2, 1)", "(4,)", "(3, 1)", "(6,)"}
    families = {}
    for _coefficient_row in result["coefficient rows"]:
        if _coefficient_row["tau"] in selected_taus:
            families.setdefault(_coefficient_row["family"], []).append(_coefficient_row)
    mo.vstack([
        mo.md(r"""
        ## 3. Representative exact coefficients, grouped by family

        Odd total degree vanishes when the central involution acts by (-I).
        For the (mu_4) scalar lift, only total degree divisible by four
        survives.  Extraspecial groups exhibit the additional coordinatewise
        divisibility sector predicted by their noncentral spectra.
        """),
        mo.ui.tabs({family: mo.ui.table(values, pagination=False) for family, values in families.items()}),
    ])
    return


@app.cell
def _(mo, result):
    mo.vstack([
        mo.md("## 4. Complete coefficient audit through total degree six"),
        mo.ui.table(list(result["coefficient rows"]), pagination=True),
    ])
    return


@app.cell
def _(mo, result):
    mo.vstack([
        mo.md(r"""
        ## 5. Projective and balanced quantities

        The binary octahedral matrices are multiplied individually by
        unrelated deterministic phases.  All charge-zero coefficients remain
        unchanged.  Its first three frame potentials agree with Haar (U(2)),
        while degree four is recorded as the first possible separating degree.

        For each finite Weil representation, matrices are generated without
        using the orbit formula.  The independent comparator enumerates
        (mathrm{SL}(2,q)\backslash(\mathbf F_q^2)^k).
        """),
        mo.ui.table(list(result["moment rows"]), pagination=False),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Scope and verification boundary

    The companion proof is uniform for arbitrary finite or compact central
    extensions.  The explicit calculations are representative audits, not a
    substitute for character tables of every sporadic cover.  In particular,
    (2.G,3.G,6.G) cases are computed from their ATLAS character values,
    class sizes, and power maps when those external data are supplied.  No
    sporadic character table is fabricated here.  Continuous (U(1)) scalar
    saturation is proved by Fourier orthogonality; finite exact quadrature
    would only test a bounded degree truncation.
    """)
    return


if __name__ == "__main__":
    app.run()
