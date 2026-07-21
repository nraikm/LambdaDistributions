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

    root = (Path(__file__).resolve().parents[1] / "new_dists/notebooks/solvable_finite.py").parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from lambda_distributions.dists.solvable_finite_subgroup_sigma_mgfs.verification import run_suite

    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Solvable and finite-subgroup $sigma$-MGF laboratory

    The group name does not determine a $sigma$-MGF: every panel states and
    constructs the representation being tested.  The laboratory compares
    explicit represented matrices with a structurally independent formula.

    - monomial groups use weighted-cycle determinants;
    - permutation groups use direct fixed points, powers, and literal orbits;
    - finite defining representations use class spectra and power traces; and
    - symplectic wreath products use the exponential cycle recurrence.

    Permutation matrices are stored exactly by the row containing the unique
    nonzero entry in each column.  Complex defining matrices are checked at
    high precision for the representation relations and Molien values.
    """)
    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Construct the groups and run all checks")
    mo.vstack([
        mo.callout(
            "Representative dimensions range from 2 to 27. The suite also "
            "constructs the 168-element PSL(2,7) subgroup of SU(3), all "
            "1,344 affine matrices of AGL(3,2), and the 64-element spin cover.",
            kind="info",
        ),
        run,
    ])
    return (run,)


@app.cell
def _(mo, run, run_suite):
    mo.stop(not run.value, mo.callout("Press the button to run the exact suite.", kind="info"))
    result = run_suite()
    return (result,)


@app.cell
def _(mo, result):
    mo.callout(
        f"{'PASS' if result['passed'] else 'FAIL'}: "
        f"{result['fixed-power checks']:,} fixed-power checks and "
        f"{result['coefficient comparisons']} coefficient comparisons.",
        kind="success" if result["passed"] else "danger",
    )
    return


@app.cell
def _(mo, result):
    monomial_rows = [row for case in result["lamplighter monomial"]["cases"] for row in case["rows"]]
    affine_rows = list(result["lamplighter affine"]["rows"])
    mo.vstack([
        mo.md(r"""
        ## 1. Finite lamplighter groups

        The natural representation of
        $L_{r,n}=mu_r^n\rtimes C_n$ is built as
        $\operatorname{diag}(z_1,\ldots,z_n)P^s$.  Its direct matrix average is
        compared with $n^{-1}\sum_{\ell\mid n}\varphi(\ell)A_{r,\ell}^{n/\ell}$.
        Separately, the affine action on $C_r^n$ checks every fixed power and
        the necklace formula for the pair coefficient.
        """),
        mo.ui.table(monomial_rows, pagination=True),
        mo.ui.table(affine_rows, pagination=False),
    ])
    return


@app.cell
def _(mo, result):
    _metacyclic_rows = [row for case in result["metacyclic"]["cases"] for row in case["rows"]]
    mo.vstack([
        mo.md(r"""
        ## 2. Split metacyclic groups

        The dihedral case $M(5,2,4)$ and the non-dihedral case $M(7,3,2)$
        test the induced monomial matrices.  The comparison spectra are
        reconstructed solely from permutation cycles and their label products
        $\alpha_C(x,y)$.
        """),
        mo.ui.table(_metacyclic_rows, pagination=True),
    ])
    return


@app.cell
def _(mo, result):
    moment_rows = []
    for case in result["affine"]["cases"]:
        moment_rows.extend({"action": case["action"], **row} for row in case["moments"])
    mo.vstack([
        mo.md(r"""
        ## 3. General affine and Frobenius actions

        For $AGL(d,q)$ the notebook verifies the image/ker fixed-point test for
        every represented element and power.  Literal $GL_d(q)$-orbits on
        difference tuples are compared with Gaussian-binomial sums.  The
        Frobenius panels independently compare complete cycle inventories and
        the pair formula $1+(|K|-1)/|H|$.
        """),
        mo.ui.table(moment_rows, pagination=False),
        mo.ui.table(list(result["Frobenius"]["rows"]), pagination=False),
    ])
    return


@app.cell
def _(mo, result):
    mo.vstack([
        mo.md(r"""
        ## 4. Unitriangular, Borel, and flag actions

        $UT_3(2)$ and $B_2(3)$ are tested on nonzero vectors and projective
        points.  The $B_3(2)$ flag action constructs all 21 flags and compares
        direct fixed points with the centralizer/class-intersection formula in
        $GL_3(2)$; its six one-point orbits equal $|S_3|$.
        """),
        mo.ui.table(list(result["triangular and parabolic"]["rows"]), pagination=False),
    ])
    return


@app.cell
def _(mo, result):
    su3_rows = [row for case in result["finite SU3"]["cases"] for row in case["rows"]]
    mo.vstack([
        mo.md(r"""
        ## 5. Finite subgroups of $SU(3)$

        Explicit generators close to the 60 rotation matrices of $A_5$ and
        the 168 defining matrices of $PSL_2(7)$.  Their direct Molien averages
        are compared with the displayed class spectra.  The determinant-one
        monomial group $\Delta(12)$ separately tests the type-C weighted-cycle
        theorem.
        """),
        mo.ui.table(su3_rows, pagination=True),
    ])
    return


@app.cell
def _(mo, result):
    _rank4_rows = [row for case in result["finite SU4 Sp4 SO4"]["cases"] for row in case["rows"]]
    mo.vstack([
        mo.md(r"""
        ## 6. Finite subgroups of $SU(4)$, $Sp(4)$, and $SO(4)$

        The panels construct the standard 4-dimensional $A_5$ module,
        $Q_8\wr S_2$ on two symplectic planes, and the tensor representation
        of $Q_8\times Q_8$.  The last representation has kernel
        $\{(I,I),(-I,-I)\}$, so averaging over the 64-element spin cover agrees
        with averaging over its 32 represented $SO(4)$ matrices.
        """),
        mo.ui.table(_rank4_rows, pagination=True),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Asymptotic interpretation

    The finite checks support three genuinely different behaviors.  The
    lamplighter and fixed-complement Frobenius families have growing low-degree
    moments and no raw coefficientwise limit.  Dense torsion in a fixed
    diagonal torus stabilizes each bounded degree.  For $K\wr S_n$ on
    $E^{\oplus n}$, the wreath recurrence stabilizes coefficientwise to
    $\operatorname{Exp}_{\sigma}(\mathcal S_K-1)$.
    """)
    return


if __name__ == "__main__":
    app.run()
