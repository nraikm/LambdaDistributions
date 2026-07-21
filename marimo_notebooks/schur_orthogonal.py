# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "marimo>=0.23.8"]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import sys
    import marimo as mo
    repo = str((Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/schur_functor_classical_groups/orthogonal/notebook.py").parents[4])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from for_this_guy.schur_functor_classical_groups.orthogonal.verification import run_suite
    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Schur functors of Haar $O(n)$

    Direct Young-symmetrizer matrices are compared with the plethystic
    trace formula and with the complete-character sigma-MGF integrands.
    The Haar diagnostics check the stable means for $S_{(2)}$, $S_{(1,1)}$,
    and the odd-degree parity obstruction caused by $-I\in O(n)$.
    """)
    run = mo.ui.run_button(label="Construct O(n) samples and verify")
    run
    return run


@app.cell
def _(mo, run, run_suite):
    mo.stop(not run.value, mo.callout("Press the button to run the deterministic suite.", kind="info"))
    report = run_suite()
    return report


@app.cell
def _(mo, report):
    mo.vstack([
        mo.callout(
            f"{'PASS' if report['passed'] else 'FAIL'}: {report['exact checks']} exact matrix/formula checks and {report['moment checks']} Haar checks. Maximum exact error {report['maximum exact error']:.3e}.",
            kind="success" if report["passed"] else "danger",
        ),
        mo.md("## Exact finite-dimensional identities"),
        mo.ui.table(list(report["exact rows"]), pagination=True, page_size=15),
        mo.md("## Orthogonal invariant/parity diagnostics"),
        mo.ui.table(list(report["moment rows"]), pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()

