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

    root = (Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/finite_lie_type_permutation_actions/polar_groups_verification.py").parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from for_this_guy.finite_lie_type_permutation_actions.verification_core import polar_suite

    return mo, polar_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Polar-space permutation representations

    We construct (Sp_4(2)), (O_4^+(2)), and (U_4(2)) as matrices preserving
    explicit alternating, quadratic, and Hermitian forms. Their induced
    permutation matrices act on isotropic or singular projective points.

    The pair coefficients are computed both by direct orbit enumeration and by
    the cycle-product formula. The full unitary check enumerates 77,760 matrices
    and normally takes about 20--30 seconds.
    """)
    return


@app.cell
def _(mo):
    include_unitary = mo.ui.checkbox(value=True, label="Include the full U₄(2) enumeration")
    run = mo.ui.run_button(label="Construct groups and verify")
    mo.hstack([include_unitary, run], justify="start", gap=1)
    return include_unitary, run


@app.cell
def _(include_unitary, mo, polar_suite, run):
    mo.stop(not run.value, mo.md("Press **Construct groups and verify** to run the exact sweep."))
    suite = polar_suite(include_unitary=include_unitary.value)
    return (suite,)


@app.cell
def _(mo, suite):
    direct, cycle, error = suite["numerical determinant"]
    status = "PASS" if suite["passed"] and suite["cycle reconstruction"] and error < 1e-10 else "FAIL"
    mo.vstack([
        mo.callout(
            f"{status}: all {len(suite['rows'])} polar coefficient checks agree.",
            kind="success" if status == "PASS" else "danger",
        ),
        mo.md(
            f"**Constructed matrix-group orders:** `{suite['group orders']}`  \n"
            f"**Cycle reconstructions:** `{suite['cycle reconstruction']}`  \n"
            f"**Numerical Molien comparison:** direct={direct:.12f}, cycle={cycle:.12f}, error={error:.2e}"
        ),
        mo.ui.table(list(suite["rows"]), pagination=False),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Geometric interpretation

    In each sufficiently large polar action there are exactly three pair types:
    the same point, distinct perpendicular points, and distinct nonperpendicular
    points. Hence both ([m_{(2)}]) and ([m_{(1,1)}]) equal (3). The exact
    small-rank enumerations above realize all three types.

    The Weil-representation paragraph is not a permutation action and does not
    specify a unique representation. A numerical test additionally requires a
    choice of (q,n), additive character, linearization, and full representation
    versus constituent. Exceptional (G/P) tests similarly require a concrete
    group and parabolic.
    """)
    return


if __name__ == "__main__":
    app.run()

