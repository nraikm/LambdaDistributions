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

    root = (Path(__file__).resolve().parents[1] / "new_dists/notebooks/local_ring_actions.py").parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from new_dists.local_ring_permutation_actions.verification import run_suite

    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Finite local-ring matrix-group laboratory

    The complex representation tested here is the permutation module
    \(\mathbf C[X]\), not a nonexistent canonical complex realization of
    \(GL_n(\mathbb Z/p^a)\subset M_n(\mathbb Z/p^a)\).

    Every finite matrix group and finite set below is constructed explicitly.
    The notebook compares direct orbits of tuples of multisets with the
    fixed-point/Möbius/cycle formula, and checks the Smith-kernel expressions
    element by element and power by power.
    """)
    return


@app.cell
def _(run_suite):
    result = run_suite()
    return (result,)


@app.cell
def _(mo, result):
    mo.vstack([
        mo.callout(
            f"{'PASS' if result['passed'] else 'FAIL'}: all exact local-ring and formed-space checks.",
            kind="success" if result["passed"] else "danger",
        ),
        mo.ui.table(list(result["rows"]), pagination=False),
    ])
    return


@app.cell
def _(mo, result):
    local_rows = []
    for case in result["local cases"]:
        local_rows.append({key: value for key, value in case.items() if key not in {"rows"}})
    mo.vstack([
        mo.md(r"""
        ## 1. \(GL_2(\mathbb Z/p^2)\): primitive vectors and projective points

        The \(p=2\) and \(p=3\) cases verify the set cardinalities, every
        Smith fixed-point count, every projective homogeneous-space fixed-point
        count, Möbius cycle reconstruction, and the pair coefficients
        \(p^a\) and \(a+1\).
        """),
        mo.ui.table(local_rows, pagination=False),
    ])
    return


@app.cell
def _(mo, result):
    grassmann = result["grassmann"]
    mo.vstack([
        mo.md(r"""
        ## 2. Free rank-two summands

        All free rank-two summands of \((\mathbb Z/4)^4\) are built as graph
        lifts of two-planes in \(\mathbf F_2^4\).  A generating set of the
        stabilizer of the standard summand acts directly on this 560-element
        set.  Its orbits are compared with the six possible Smith types.
        """),
        mo.ui.table([grassmann], pagination=False),
    ])
    return


@app.cell
def _(mo, result):
    formed = result["formed field"]
    mo.vstack([
        mo.md(r"""
        ## 3. Symplectic and orthogonal representatives

        Locally, \(Sp_2(\mathbb Z/4)=SL_2(\mathbb Z/4)\) is tested on its six
        projective isotropic lines.  Independently, the existing exact
        finite-field constructors test \(Sp_4(2)\) and \(O_4^+(2)\) on their
        isotropic or singular points.
        """),
        mo.ui.table([result["symplectic local"]], pagination=False),
        mo.ui.table(list(formed["rows"]), pagination=False),
    ])
    return


@app.cell
def _(mo, result):
    mo.vstack([
        mo.md(r"""
        ## 4. Congruence adjoint quotient

        \(GL_2(\mathbb Z/4)\) acts through reduction on the 16 matrices in
        \(M_2(\mathbf F_2)\).  Direct conjugation orbits agree with the
        permutation-cycle formula, while every fixed-point value agrees with
        the kernel of \(\operatorname{Ad}(\bar g^d)-I\).
        """),
        mo.ui.table(list(result["adjoint"]["rows"]), pagination=False),
        mo.md(f"Adjoint kernel check: `{result['adjoint']['adjoint-kernel fixed points']}`"),
    ])
    return


if __name__ == "__main__":
    app.run()
