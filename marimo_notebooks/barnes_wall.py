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

    here = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/barnes_wall_lattice_automorphism_groups/barnes_wall/notebook.py").parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from verification import run_sweep
    return mo, run_sweep


@app.cell
def _(mo):
    mo.md(r"""
    # Barnes--Wall family III: lattice automorphism groups

    In the small concrete models \(BW_2=\mathbf Z^2\) and \(BW_4=D_4\),
    the verifier closes groups of orders \(8\) and \(1152\). It then
    independently filters the real Clifford group for matrices preserving
    the displayed lattice basis; the filtered set must equal the constructed
    Barnes--Wall group exactly.

    For each partition, the direct \(B_m\)-average is compared with the sum
    of the untwisted Clifford average and the \(\varepsilon_m\)-twisted
    average in (11.19). Rank \(m=3\) is intentionally excluded because the
    index-two assertion is exceptional there. The constructed Clifford and
    lattice groups are also checked matrix-by-matrix for real orthogonality.
    """)
    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Construct Aut(BW_2), Aut(BW_4) and check (11.19)")
    run
    return (run,)


@app.cell
def _(mo, run, run_sweep):
    mo.stop(not run.value, mo.md("Press the button to verify lattice preservation and twisted averages."))
    rows = run_sweep()
    passed = all(row["passed"] for row in rows)
    mo.vstack([
        mo.callout(
            f"{'All' if passed else 'Not all'} {len(rows)} Barnes--Wall checks passed.",
            kind="success" if passed else "danger",
        ),
        mo.ui.table(rows, pagination=True),
    ])
    return


if __name__ == "__main__":
    app.run()
