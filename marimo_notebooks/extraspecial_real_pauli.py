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

    here = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/barnes_wall_lattice_automorphism_groups/extraspecial_real_pauli/notebook.py").parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from verification import run_sweep
    return mo, run_sweep


@app.cell
def _(mo):
    mo.md(r"""
    # Barnes--Wall family I: extraspecial real Pauli groups

    This notebook constructs
    \(E_m=\{\pm X^aZ^b:a,b\in\mathbf F_2^m\}\) as actual
    \(2^m\times2^m\) matrices. For every listed partition \(\tau\), it
    averages \(\prod_i h_{\tau_i}(\operatorname{spec}g)\) over every group
    element and independently evaluates the four spectral sectors in (11.5).

    The ranks \(m=1,2,3\) exercise dimensions \(2,4,8\), group orders
    \(8,32,128\), odd-degree vanishing, mixed symmetric powers, and both
    noncentral square classes. Every constructed matrix is also checked to
    be real and orthogonal in the claimed representation.
    """)
    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Construct E_m and check formula (11.5)")
    run
    return (run,)


@app.cell
def _(mo, run, run_sweep):
    mo.stop(not run.value, mo.md("Press the button to run the complete finite averages."))
    rows = run_sweep()
    passed = all(row["passed"] for row in rows)
    mo.vstack([
        mo.callout(
            f"{'All' if passed else 'Not all'} {len(rows)} coefficient checks passed.",
            kind="success" if passed else "danger",
        ),
        mo.ui.table(rows, pagination=True),
    ])
    return


if __name__ == "__main__":
    app.run()
