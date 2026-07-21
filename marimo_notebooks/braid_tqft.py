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

    root = (Path(__file__).resolve().parents[1] / "new_dists/notebooks/braid_tqft.py").parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from new_dists.braid_tqft_quantum_images_sigma_mgfs.verification import run_suite

    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Finite braid images and compact-closure sigma-MGFs

    This laboratory tests the finite-image/class-power theorem and the Haar
    replacement for infinite images on concrete, separately identified groups.

    - The Ising braid generators give a finite (192)-element lift in (U(2)).
    - The qutrit Heisenberg group tests the monomial cycle-phase formula.
    - Fourier and chirp generators give the rank-one (q=3) Weil lift.
    - The projective qubit Clifford group tests the balanced design series.
    - Regular and affine phase-space actions test the SIC/Weyl-Heisenberg formulas.
    - Seeded Haar matrices diagnose the (U), (SU), (O), and (USp) closures.

    Every finite group is exhaustively closed and averaged. Haar rows are
    statistical checks with standard errors; the companion PDF supplies the proofs.
    """)
    return


@app.cell
def _(mo):
    haar_samples = mo.ui.slider(
        1000, 5000, value=3000, step=500, label="Haar samples per diagnostic"
    )
    run = mo.ui.run_button(label="Construct all represented groups and verify")
    mo.vstack([haar_samples, run])
    return haar_samples, run


@app.cell
def _(haar_samples, mo, run, run_suite):
    mo.stop(
        not run.value,
        mo.callout("Press the button to run the exhaustive suite.", kind="info"),
    )
    result = run_suite(haar_samples.value)
    return (result,)


@app.cell
def _(mo, result):
    mo.callout(
        (
            f"{'PASS' if result['passed'] else 'FAIL'}: "
            f"{result['finite coefficient comparisons']} finite comparisons, "
            f"{result['fixed-power checks']} affine fixed-power checks, and "
            f"{result['haar diagnostics']} compact-closure diagnostics."
        ),
        kind="success" if result["passed"] else "danger",
    )
    return


@app.cell
def _(mo, result):
    _section = result["ising"]
    summary = [{
        "lift order": _section["lift order"],
        "projective order": _section["projective order"],
        "scalar center": _section["scalar center order"],
        "braid residual": f"{_section['braid residual']:.2e}",
        "MGF error": f"{_section['numeric error']:.2e}",
    }]
    mo.vstack([
        mo.md(r"""
        ## Ising braid image

        The displayed matrices satisfy the braid relation directly. Their finite
        lift contains (mu_8 I), so all one-sided coefficients outside total
        degree divisible by eight vanish.
        """),
        mo.ui.table(summary, pagination=False),
        mo.ui.table(_section["coefficient rows"], pagination=False),
    ])
    return


@app.cell
def _(mo, result):
    _section = result["heisenberg"]
    mo.vstack([
        mo.md(r"""
        ## Qutrit Heisenberg image

        All (27) Schrödinger matrices are constructed. Dense determinants are
        compared elementwise with products over the permutation cycles and their
        accumulated phases.
        """),
        mo.callout(f"Dense/cycle-phase MGF error: {_section['numeric error']:.2e}"),
        mo.ui.table(_section["coefficient rows"], pagination=False),
    ])
    return


@app.cell
def _(mo, result):
    _section = result["weil"]
    magnitude_rows = [
        {
            "A": row["A"],
            "trace magnitude squared": round(row["|Tr omega(A)|^2"], 10),
            "fixed-space size": int(row["|ker(A-I)|"]),
            "status": "PASS" if row["passed"] else "FAIL",
        }
        for row in _section["magnitude rows"]
    ]
    mo.vstack([
        mo.md(r"""
        ## Rank-one qutrit Weil lift

        Projective words are tracked simultaneously in (SL_2(mathbf F_3)).
        This tests the phase-insensitive identity
        (lvertchi_omega(A)
vert^2=lvertker(A-I)
vert) for every element,
        then tests the phase-sensitive class-power formula on the (96)-matrix lift.
        """),
        mo.ui.table(_section["coefficient rows"], pagination=False),
        mo.ui.table(magnitude_rows, pagination=True, page_size=8),
    ])
    return


@app.cell
def _(mo, result):
    mo.vstack([
        mo.md(r"""
        ## Balanced Clifford design test

        The phase-neutral frame coefficient agrees with Haar through degree three
        and first differs at degree four.
        """),
        mo.ui.table(result["design"]["rows"], pagination=False),
    ])
    return


@app.cell
def _(mo, result):
    _section = result["phase space"]
    mo.vstack([
        mo.md(r"""
        ## Weyl-Heisenberg regular and affine actions

        The regular action on (p^2) points tests the prime SIC formula for
        (p=2,3,5). The affine groups
        (mathbf F_p^2
times SL_2(mathbf F_p)) test the fixed-point formula
        for six powers of every element and the predicted pair coefficient (2).
        """),
        mo.ui.table(_section["regular rows"], pagination=True, page_size=8),
        mo.ui.table(_section["affine rows"], pagination=False),
    ])
    return


@app.cell
def _(mo, result):
    rows = [
        {
            "group": row["group"],
            "tau": row["tau"],
            "estimate": f"{row['direct Haar estimate'].real:.4f}"
            + (f" {row['direct Haar estimate'].imag:+.4f}i"),
            "target": row["formula target"],
            "SE": f"{row['standard error']:.4f}",
            "status": "PASS" if row["passed"] else "FAIL",
        }
        for row in result["compact closures"]["rows"]
    ]
    mo.vstack([
        mo.md(r"""
        ## Compact closures

        These seeded experiments directly average symmetric-power characters.
        They illustrate the exact scalar obstruction for (U(d)), the determinant
        threshold for (SU(d)), and the stable orthogonal/symplectic coefficients.
        """),
        mo.ui.table(rows, pagination=True, page_size=8),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Scope and interpretation

    A label such as “Jones”, “TQFT”, or “Hadamard-generated” does not determine a
    matrix group. The formula becomes testable only after the represented image or
    its compact closure is specified. The Ising and rank-one Weil examples are
    finite representatives; they do not assert finiteness for generic braid or
    mapping-class-group sectors. Likewise, Haar sampling corroborates but does not
    prove a compact-group identity; the proof is by Reynolds projection and
    classical invariant theory in the companion document.
    """)
    return


if __name__ == "__main__":
    app.run()
