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

    root = (Path(__file__).resolve().parents[1] / "new_dists/notebooks/multiplicative_wreath.py").parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from new_dists.multiplicative_wreath_sigma_mgfs.verification import run_suite

    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Multiplicative wreath-product $\sigma$-MGF laboratory

    Let $G_n=H\wr S_n$.  This notebook tests the organizing distinction
    
    \[
      \text{direct sums are additive over cycles},\qquad
      \text{product and tensor-induced actions are multiplicative over cycles}.
    \]

    Every finite example is an explicit matrix group.  A permutation matrix is
    stored by the row containing the unique nonzero entry in each column; a
    signed monomial matrix also stores that entry's sign.  Target invariant
    dimensions are computed by literal orbit traversal (including sign
    consistency), independently of the proposed character and class formulas.
    """)
    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Construct the matrix groups and run all exact checks")
    mo.vstack([
        mo.callout(
            "The exhaustive suite uses C2 wr S_n, S3 wr S_n, diagonal S3^n, "
            "and the twisted regular action S3^2 semidirect S2.  Dimensions "
            "range from 2 to 36.",
            kind="info",
        ),
        run,
    ])
    return (run,)


@app.cell
def _(mo, run, run_suite):
    mo.stop(not run.value, mo.callout("Press the button to run the audit.", kind="info"))
    report = run_suite()
    return (report,)


@app.cell
def _(mo, report):
    mo.callout(
        f"{'PASS' if report['passed'] else 'FAIL'}: "
        f"{report['exact checks']:,} exact group-law, power, matrix, orbit, and formula checks.",
        kind="success" if report["passed"] else "danger",
    )
    return


@app.cell
def _(mo, report):
    _section = report["wreath class powers"]
    mo.vstack([
        mo.md(r"""
        ## I. Wreath conjugacy types and powers

        For $C_2\wr S_3$ and $S_3\wr S_2$, the program forms every
        semidirect-product element.  It verifies that an $\ell$-cycle with
        cycle product $c$ splits under the $r$th power into
        $d=\gcd(\ell,r)$ cycles of length $\ell/d$ and product class
        $[c^{r/d}]$.  It also counts every wreath type and verifies its exact
        probability $1/Z_{\boldsymbol\mu}$.
        """),
        mo.ui.table(list(_section["cases"]), pagination=False),
    ])
    return


@app.cell
def _(mo, report):
    _section = report["product actions"]
    _rows = []
    for _case in _section["cases"]:
        for _row in _case["rows"]:
            _rows.append({"group": f"S3 wr S{_case['k']}", "matrix dimension": _case["degree"], **_row})
    mo.vstack([
        mo.md(r"""
        ## II. Product actions on $\Omega^n$

        Here $S_3$ acts naturally on $\Omega=\{0,1,2\}$.  All matrices and
        their first six powers are checked against the multiplicative
        fixed-point formula.  Direct orbits on $(\Omega^n)^s$ and Burnside
        averages then agree with
        \[
          [m_{(1^s)}]\operatorname{SMG}
          =\binom{n+R_s-1}{R_s-1},\qquad
          R_s=\#(S_3\backslash\Omega^s).
        \]
        """),
        mo.ui.table(_rows, pagination=False),
    ])
    return


@app.cell
def _(mo, report):
    _section = report["tensor-induced"]
    _summaries = [
        {key: value for key, value in case.items() if key != "rows"}
        for case in _section["cases"]
    ]
    _rows = [
        {"group": case["group"], **row}
        for case in _section["cases"]
        for row in case["rows"]
    ]
    mo.vstack([
        mo.md(r"""
        ## III. Tensor-induced signed matrix groups

        Take $H=C_2$ and $V=\mathbf1\oplus\mathrm{sgn}$.  The module
        $T_n(V)=V^{\otimes n}$ has dimension $2^n$.  The code constructs every
        signed monomial matrix in $C_2\wr S_n$, compares traces of six powers
        with the cycle-product formula, and computes invariants by signed-orbit
        propagation.  For tensor moments,
        \[
          a_s=\dim(V^{\otimes s})^{C_2}=2^{s-1},\qquad
          [m_{(1^s)}]=\binom{n+2^{s-1}-1}{2^{s-1}-1}.
        \]
        """),
        mo.ui.table(_summaries, pagination=False),
        mo.ui.table(_rows, pagination=False),
    ])
    return


@app.cell
def _(mo, report):
    _section = report["fixed-tail bipartitions"]
    _rows = [
        {"group": case["group"], **row}
        for case in _section["cases"]
        for row in case["rows"]
    ]
    mo.vstack([
        mo.md(r"""
        ## IV. A genuine fixed-tail bipartition family

        The natural signed representation of $C_2\wr S_n$ is the irreducible
        bipartition family $((n-1),(1))$.  Its character polynomial is
        \[
          Q=X_{1,+}-X_{1,-}.
        \]
        Matrix traces of powers are compared with $Q(\boldsymbol\mu^{[r]})$,
        and six invariant coefficients are checked by two independent routes.
        The limiting character is $Z_+-Z_-$ for independent
        $Z_\pm\sim\operatorname{Poisson}(1/2)$.
        """),
        mo.ui.table(_rows, pagination=True, page_size=10),
    ])
    return


@app.cell
def _(mo, report):
    _diagonal = report["diagonal cosets"]
    _twisted = report["twisted regular"]
    _diagonal_rows = [
        {"group": case["group"], "matrix dimension": case["dimension"], **row}
        for case in _diagonal["cases"]
        for row in case["rows"]
    ]
    mo.vstack([
        mo.md(r"""
        ## V. Diagonal cosets and twisted regular actions

        These are separate matrix groups, not tensor modules.  The diagonal
        laboratory constructs $S_3^n$ on $S_3^n/\Delta S_3$ and checks the
        common-conjugacy-class formula.  The twisted laboratory constructs the
        affine action $S_3^2\rtimes S_2\curvearrowright S_3^2$ and checks the
        nonabelian Lang-map criterion and its Bernoulli-spike law.
        """),
        mo.ui.table(_diagonal_rows, pagination=False),
        mo.ui.table(list(_twisted["rows"]), pagination=False),
        mo.md(
            f"Twisted direct pair orbits: **{_twisted['pair direct']}**; "
            f"multiset formula: **{_twisted['pair formula']}**."
        ),
    ])
    return


@app.cell
def _(mo, report):
    _fock = report["Fock"]
    _varying = report["varying base"]
    mo.vstack([
        mo.md(r"""
        ## VI. Graded Fock identities and varying bases

        For $C_2$ on $\mathbf1\oplus\mathrm{sgn}$, explicit symmetric-monomial
        and exterior bases verify the determinant characters in every energy
        through six.  The $q$-grading is retained: setting $q=1$ in the
        bosonic space would discard the finiteness that makes the statement
        meaningful.

        The final table specializes the two-parameter laws to regular cyclic
        bases, where $R_s=a_s=|H|^{s-1}$.
        """),
        mo.ui.table(list(_fock["rows"]), pagination=False),
        mo.ui.table(list(_varying["rows"]), pagination=False),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Conclusion

    The finite tests support the exact formulas and expose the asymptotic
    obstruction.  Whenever $R_s\ge2$ or $a_s\ge2$, the coefficient
    $[m_{(1^s)}]$ grows like a positive power of $n$.  Thus multiplicative
    wreath representations generally have rare-event moment divergence; the
    compound-Poisson behavior belongs to the additive block representation.
    """)
    return


if __name__ == "__main__":
    app.run()
