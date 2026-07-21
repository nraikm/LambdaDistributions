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
    import numpy as np
    sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/compact_connected_lie_groups/f4/f4_verification.py").parents[1]))
    from verification_core import exceptional_representations, verify_expected, verify_monomial_expected
    return exceptional_representations, mo, np, verify_expected, verify_monomial_expected


@app.cell
def _(mo):
    mo.md(r"""
    # $F_4$ on the trace-zero Albert module $V_{26}$

    The 24 short-root weights and two zero weights produce an explicit $26$
    by $26$ torus matrix. Exact Weyl integration tests
    $1+h_2+h_3+O_{\ge4}$.
    """)
    return


@app.cell
def _(exceptional_representations, np, verify_expected, verify_monomial_expected):
    rep = exceptional_representations()["f4_26"]
    matrix = rep.torus_matrix((0.12, -0.19, 0.31, -0.27))
    ts = (0.03, 0.07)
    direct = np.prod([1 / np.linalg.det(np.eye(rep.dimension) - t * matrix) for t in ts])
    spectral = np.prod([1 / (1 - t * value) for t in ts for value in np.diag(matrix)])
    schur_checks = verify_expected(rep, {(): 1, (2,): 1, (3,): 1}, 3)
    monomial_checks = verify_monomial_expected(
        rep,
        {(): 1, (2,): 1, (1, 1): 1, (3,): 1, (2, 1): 1, (1, 1, 1): 1},
        3,
    )
    summary = {"dimension": rep.dimension, "roots": len(rep.root_system), "zero weights": 2,
               "Weyl order": rep.weyl_order,
               "unitarity residual": float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(rep.dimension))),
               "integrand residual": float(abs(direct - spectral)),
               "all coefficient checks pass": all(row["pass"] for row in schur_checks + monomial_checks)}
    return monomial_checks, schur_checks, summary


@app.cell
def _(mo, monomial_checks, schur_checks, summary):
    mo.vstack([mo.ui.table([summary]), mo.md("## Requested monomial coefficients"), mo.ui.table(monomial_checks), mo.md("## Equivalent Schur coefficients"), mo.ui.table(schur_checks)])
    return


if __name__ == "__main__":
    app.run()
