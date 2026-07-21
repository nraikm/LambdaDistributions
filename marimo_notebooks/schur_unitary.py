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
    repo = str((Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/schur_functor_classical_groups/unitary/notebook.py").parents[4])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from lambda_distributions.proofs.schur_functor_classical_groups.unitary.verification import run_suite
    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Schur functors of Haar $U(n)$

    The exact route constructs $S_\lambda(U)$ inside $(\mathbb C^n)^{\otimes |\lambda|}$
    with a Young symmetrizer. The independent formula route uses
    $\operatorname{Tr}(S_\lambda(U)^r)=s_\lambda[X(U^r)]$ and Jacobi--Trudi.
    It also compares complete-character products $\prod_i h_{\tau_i}$, the
    integrands of the sigma-MGF coefficients. The Haar rows test the crucial
    correction: positive purely holomorphic moments vanish; mixed moments with
    the conjugate are the nontrivial unitary object.
    """)
    run = mo.ui.run_button(label="Construct U(n) samples and verify")
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
        mo.md("## Central-character Haar diagnostics"),
        mo.ui.table(list(report["moment rows"]), pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()

