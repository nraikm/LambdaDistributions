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

    root = (Path(__file__).resolve().parents[1] / "new_dists/notebooks/code_monomial.py").parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from new_dists.code_monomial_sigma_mgfs.verification import run_suite

    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Code-monomial group \(\sigma\)-MGF laboratory

    For an additive code \(C\leq(\mathbb Z/r\mathbb Z)^N\) and a coordinate
    permutation group \(P\leq\operatorname{Aut}_{\rm perm}(C)\), this
    notebook constructs every matrix
    \[
      D(c)P_\pi\in G(C,P)\leq U(N)
    \]
    and tests the generalized Molien series by three routes:

    1. dense complex matrix determinants and traces of actual matrix powers;
    2. weighted coordinate-cycle factors;
    3. the root-filtered dual-code formula and exact orbits of exponent arrays.

    The cases are grouped below as pure diagonal, binary semidirect, ternary
    semidirect, and composite cyclic-alphabet groups.
    """)
    return


@app.cell
def _(run_suite):
    result = run_suite(max_degree=5)
    assert result["passed"]
    return (result,)


@app.cell
def _(mo, result):
    mo.callout(
        f"PASS: {result['groups']} explicitly constructed groups; "
        f"{result['coefficient comparisons']} exact coefficient comparisons; "
        f"maximum determinant error {result['maximum determinant error']:.3e}.",
        kind="success",
    )
    return


@app.cell
def _(mo, result):
    summary = [
        {
            "family": case["family"],
            "group": case["name"],
            "r": case["modulus"],
            "N": case["dimension"],
            "|C|": case["code order"],
            "|C^perp|": case["dual order"],
            "|P|": case["permutation order"],
            "|G|": case["group order"],
            "unitary error": case["matrix audit"]["maximum unitarity error"],
            "pass": case["passed"],
        }
        for case in result["cases"]
    ]
    mo.vstack([
        mo.md(r"""
        ## 1. Constructed matrix groups

        The matrix audit checks the expected order, distinctness, and
        unitarity.  The \(r=4\) case confirms that the theorem is genuinely a
        \(\mathbb Z/r\mathbb Z\) statement and does not require \(r\) prime.
        """),
        mo.ui.table(summary, pagination=False),
    ])
    return


@app.cell
def _(mo, result):
    mo.vstack([
        mo.md(r"""
        ## 2. Full series at a convergent test point

        Each row evaluates
        \(\mathcal S_{C,P}(0.07,0.11)\).  The first column computes the
        target directly from dense matrices; the second uses the cycle sums
        of codewords; the third uses \(C^\perp\) and the functions
        \(H_a^{[\ell]}\).
        """),
        mo.ui.table(list(result["determinant rows"]), pagination=False),
    ])
    return


@app.cell
def _(mo, result):
    families = {}
    for case in result["cases"]:
        selected = [
            row
            for row in case["coefficient rows"]
            if row["tau"] in {"()", "(1,)", "(2,)", "(1, 1)", "(3,)", "(2, 1)", "(4,)"}
        ]
        families.setdefault(case["family"], []).extend(selected)
    tabs = {
        family: mo.ui.table(rows, pagination=False)
        for family, rows in families.items()
    }
    mo.vstack([
        mo.md(r"""
        ## 3. Exact coefficient checks, grouped by group family

        The matrix route averages
        \(\prod_j\operatorname{Tr}(\operatorname{Sym}^{\tau_j}g)\).
        The comparator enumerates exponent arrays, imposes the dual-code
        congruence, and counts \(P\)-orbits.  Representative coefficients are
        shown here; the complete 95-row audit appears in the next table.
        """),
        mo.ui.tabs(tabs),
    ])
    return


@app.cell
def _(mo, result):
    mo.vstack([
        mo.md("## 4. Complete coefficient audit through total degree five"),
        mo.ui.table(list(result["rows"]), pagination=True),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Verification boundary

    The finite computations audit the implementations and specializations;
    the companion PDF proves the formulas for every cyclic modulus, dimension,
    additive code, and coordinate-permutation automorphism group.  The scalar
    formula does not by itself cover noncyclic additive alphabets or
    semilinear/coordinate-scaling automorphisms; those require the appropriate
    character-module representation.
    """)
    return


if __name__ == "__main__":
    app.run()
