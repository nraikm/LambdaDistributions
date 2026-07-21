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
    sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/compact_connected_lie_groups/e7/e7_verification.py").parents[1]))
    from verification_core import exceptional_representations, verify_expected, verify_monomial_expected
    return exceptional_representations, mo, np, verify_expected, verify_monomial_expected


@app.cell
def _(mo):
    mo.md(r"""
    # Simply connected $E_7$ on the minuscule $56$

    The executable exact check is deliberately limited to total degree two:
    it detects the unique alternating symplectic form. The companion PDF gives
    the tensor-contraction verification of the degree-four terms
    $h_4+s_{(2,2)}+e_4$.
    """)
    return


@app.cell
def _(exceptional_representations, np, verify_expected, verify_monomial_expected):
    rep = exceptional_representations()["e7_56"]
    matrix = rep.torus_matrix((0.06, -0.09, 0.13, -0.17, 0.21, -0.25, 0.28))
    ts = (0.02, 0.04)
    direct = np.prod([1 / np.linalg.det(np.eye(rep.dimension) - t * matrix) for t in ts])
    spectral = np.prod([1 / (1 - t * value) for t in ts for value in np.diag(matrix)])
    schur_checks = verify_expected(rep, {(): 1, (1, 1): 1}, 2)
    monomial_checks = verify_monomial_expected(rep, {(): 1, (1, 1): 1}, 2)
    summary = {"dimension": rep.dimension, "weights in minuscule orbit": len(set(rep.weights)),
               "roots": len(rep.root_system), "Weyl order": rep.weyl_order,
               "unitarity residual": float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(rep.dimension))),
               "integrand residual": float(abs(direct - spectral)),
               "all quadratic checks pass": all(row["pass"] for row in schur_checks + monomial_checks)}
    return monomial_checks, schur_checks, summary


@app.cell
def _(mo, monomial_checks, schur_checks, summary):
    mo.vstack([mo.ui.table([summary]), mo.md("## Requested monomial coefficients"), mo.ui.table(monomial_checks), mo.md("## Equivalent Schur coefficients"), mo.ui.table(schur_checks)])
    return


if __name__ == "__main__":
    app.run()
