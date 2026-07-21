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

    root = (Path(__file__).resolve().parents[1] / "new_dists/notebooks/symmetric_spin_lie_tree.py").parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from new_dists.symmetric_spin_lie_tree_sigma_mgfs.verification import run_suite

    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Symmetric, spin-cover, Foulkes, free-Lie, and rooted-tree $\sigma$-MGFs

    This notebook audits the formulas in five group-separated laboratories.
    In every case the represented group is actually constructed.  Ordinary
    Specht and free-Lie modules use explicit integral matrices; the Schur
    cover uses Clifford matrices; Foulkes and tree actions use zero-one
    permutation matrices stored by their nonzero column positions.

    The target coefficient is
    \[
      [m_\tau]\operatorname{SMG}_{G,V}
      =\dim\left(\bigotimes_j\operatorname{Sym}^{\tau_j}V\right)^G.
    \]
    Generator fixed spaces or literal configuration orbits provide the
    direct route.  Full Molien averages, Frobenius characteristics, Witt
    characters, and wreath cycle indices provide the proposed-formula route.
    """)
    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Construct all matrix groups and run the audit")
    mo.vstack([
        mo.callout(
            "The recommended exhaustive suite reaches matrix dimensions 15 "
            "for Foulkes actions, 6 for non-permutation modules, and 9 leaves "
            "for the largest fully enumerated tree groups.",
            kind="info",
        ),
        run,
    ])
    return (run,)


@app.cell
def _(mo, run, run_suite):
    mo.stop(not run.value, mo.callout("Press the button to run the verification.", kind="info"))
    report = run_suite()
    return (report,)


@app.cell
def _(mo, report):
    mo.callout(
        f"{'PASS' if report['passed'] else 'FAIL'}: "
        f"{report['exact checks']:,} character, formula, matrix, and orbit checks.",
        kind="success" if report["passed"] else "danger",
    )
    return


@app.cell
def _(mo, report):
    _section = report["symmetric groups"]
    _summaries = [
        {key: value for key, value in case.items() if key not in {"rows", "passed"}}
        for case in _section["cases"]
    ]
    mo.vstack([
        mo.md(r"""
        ## I. Symmetric groups: hooks and two-row Specht modules

        The hook matrices are exterior powers of the integral standard
        representation.  The two-row matrices are the vertex-sum-zero
        subspace of the two-subset permutation module.  Traces of four powers
        of every matrix are compared with the hook and subset character
        polynomials; invariant dimensions are then checked independently.
        """),
        mo.ui.table(_summaries, pagination=False),
        mo.ui.table(_section["rows"], pagination=True, page_size=12),
        mo.md(f"Character-power comparisons: **{_section['character-power checks']:,}**."),
    ])
    return


@app.cell
def _(mo, report):
    _section = report["Schur cover"]
    mo.vstack([
        mo.md(r"""
        ## II. Schur double cover

        Adjacent transpositions lift to
        $u_i=(\gamma_i-\gamma_{i+1})/\sqrt2$ in the Clifford algebra.
        They generate the order-$48$ Pin preimage of $S_4$; the central
        element $-I$ acts genuinely.  Consequently every odd-total-degree
        coefficient vanishes.  Common fixed spaces are compared with the
        generalized Molien average, including the fourth tensor moment.
        """),
        mo.ui.table([{
            "group": _section["group"],
            "order": _section["group order"],
            "dimension": _section["dimension"],
            "Clifford residual": f"{_section['Clifford residual']:.1e}",
            "odd-degree vanishing": _section["odd-degree vanishing"],
        }], pagination=False),
        mo.ui.table(_section["rows"], pagination=False),
    ])
    return


@app.cell
def _(mo, report):
    _section = report["Foulkes"]
    _summaries = [
        {
            "module": case["module"],
            "ambient group": case["ambient S_n"],
            "matrix dimension": case["dimension"],
            "pair/contingency coefficient": case["contingency formula"],
        }
        for case in _section["cases"]
    ]
    mo.vstack([
        mo.md(r"""
        ## III. Foulkes permutation modules

        The program lists every equal-block set partition and constructs the
        full $S_{ab}$ action.  Independently it expands $h_b[h_a]$ in the
        power-sum basis.  All class characters, five Molien coefficients, and
        the contingency-table description of the pair coefficient are tested.
        """),
        mo.ui.table(_summaries, pagination=False),
        mo.ui.table(_section["rows"], pagination=False),
        mo.md(f"Frobenius-character comparisons: **{_section['character checks']:,}**."),
    ])
    return


@app.cell
def _(mo, report):
    _section = report["free Lie"]
    _summaries = [
        {
            "module": case["module"],
            "order": case["group order"],
            "dimension": case["dimension"],
            "Witt norm": case["second moment formula"],
        }
        for case in _section["cases"]
    ]
    mo.vstack([
        mo.md(r"""
        ## IV. Multilinear free-Lie modules

        A concrete $(n-1)!$-element bracket basis is expanded in associative
        words, then relabeled by every permutation to obtain the matrices.
        The resulting traces are compared with
        $L_n=n^{-1}\sum_{d\mid n}\mu(d)p_d^{n/d}$.
        The direct pair invariant also matches the closed norm formula.
        """),
        mo.ui.table(_summaries, pagination=False),
        mo.ui.table(_section["rows"], pagination=False),
    ])
    return


@app.cell
def _(mo, report):
    _section = report["rooted trees"]
    _summaries = [
        {key: value for key, value in case.items() if key not in {"rows", "passed"}}
        for case in _section["iterated cases"]
    ]
    mo.vstack([
        mo.md(r"""
        ## V. Iterated rooted-tree groups

        The local actions $C_2,C_3$, and $S_3$ are iterated on leaves.  Every
        group element is enumerated.  Six coefficients are checked against
        the wreath recursion.  Direct fixed-point second moments verify
        $1+h(r_L-1)$, and pair orbits verify the universal lower bound $h+1$.
        """),
        mo.ui.table(_summaries, pagination=True, page_size=10),
        mo.ui.table(_section["rows"], pagination=True, page_size=14),
    ])
    return


@app.cell
def _(mo, report):
    _section = report["rooted trees"]["arbitrary"]
    _summaries = [
        {key: value for key, value in case.items() if key not in {"rows", "passed"}}
        for case in _section["cases"]
    ]
    mo.vstack([
        mo.md(r"""
        ## VI. Arbitrary finite rooted trees

        Three non-regular trees combine repeated and nonisomorphic root
        branches.  Their full automorphism groups are constructed recursively,
        and the direct leaf-action coefficients match the product of symmetric
        cycle indices for each branch type.
        """),
        mo.ui.table(_summaries, pagination=False),
        mo.ui.table(_section["rows"], pagination=False),
        mo.callout(
            "The finite formulas are proved and verified.  No universal "
            "height-infinity law is inferred for arbitrary random trees; "
            "the raw level-transitive leaf series cannot converge because "
            "its pair coefficient is at least h+1.",
            kind="warn",
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
