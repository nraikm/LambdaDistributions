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

    root = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/finite_lie_type_permutation_actions/linear_groups_verification.py").parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from lambda_distributions.proofs.finite_lie_type_permutation_actions.verification_core import linear_suite

    return linear_suite, mo


@app.cell
def _(mo):
    mo.md(r"""
    # Linear and projective finite-group actions

    This notebook constructs (GL_n(q)) and (SL_n(q)) as finite-field
    matrices, induces their actions on nonzero vectors, projective points, and
    complete flags, and checks the proposed σ-MGF formula in three independent
    ways:

    1. direct orbit enumeration on tuples of multisets;
    2. coefficient extraction from actual permutation cycle counts;
    3. cycle-count reconstruction from ​(operatorname{Fix}(g^r)) by Möbius inversion.

    A numerical determinant average also checks
    ​(det(I-tP_g)^{-1}=prod_d(1-t^d)^{-c_d(g)}).
    """)
    return


@app.cell
def _(linear_suite):
    suite = linear_suite()
    return (suite,)


@app.cell
def _(mo, suite):
    direct, cycle, error = suite["numerical determinant"]
    status = "PASS" if suite["passed"] and suite["cycle reconstruction"] and error < 1e-10 else "FAIL"
    mo.vstack([
        mo.callout(
            f"{status}: all {len(suite['rows'])} coefficient checks agree.",
            kind="success" if status == "PASS" else "danger",
        ),
        mo.md(
            f"**Matrix groups:** `{suite['group orders']}`  \n"
            f"**All PSL₃(2) cycle reconstructions:** `{suite['cycle reconstruction']}`  \n"
            f"**Numerical Molien comparison:** direct={direct:.12f}, cycle={cycle:.12f}, error={error:.2e}"
        ),
        mo.ui.table(list(suite["rows"]), pagination=True, page_size=20),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What the boundary cases show

    - (PSL_3(2)) on seven projective points gives the stable degree-three
      coefficients (4,5,6).
    - (PSL_2(2)) has only (3) orbits on 3-multisets, so (n\geq|τ|)
      is a real stable-range condition.
    - (SL_2(3)), τ=((1,1)), gives (4), while (GL_2(3)) gives (3).
      Thus the stated (SL/GL) agreement needs the strict range (n>|τ|).
    - Complete flags give (2) and (6) pair orbits in ranks (2) and (3),
      exactly (|W|=n!), demonstrating non-stability.
    """)
    return


if __name__ == "__main__":
    app.run()

