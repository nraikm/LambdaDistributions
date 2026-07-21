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
    sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/compact_connected_lie_groups/su_adjoint/su_adjoint_verification.py").parents[1]))
    from verification_core import su_adjoint, verify_expected
    return mo, np, su_adjoint, verify_expected


@app.cell
def _(mo):
    mo.md(r"""
    # Adjoint $SU(n)$: small representative dimensions

    We test $n=2,3,4$. Every case has $h_2$ and $e_3$; the symmetric cubic
    $h_3$ occurs exactly for $n\ge3$.
    """)
    return


@app.cell
def _(np, su_adjoint, verify_expected):
    rows = []
    for n in (2, 3, 4):
        rep = su_adjoint(n)
        angles = tuple(0.17 * (i + 1) * (-1) ** i for i in range(rep.rank))
        matrix = rep.torus_matrix(angles)
        t = 0.03
        direct = 1 / np.linalg.det(np.eye(rep.dimension) - t * matrix)
        spectral = np.prod([1 / (1 - t * value) for value in np.diag(matrix)])
        expected = {(): 1, (2,): 1, (1, 1, 1): 1}
        if n >= 3:
            expected[(3,)] = 1
        checks = verify_expected(rep, expected, 3)
        rows.append({"n": n, "dimension": rep.dimension, "roots": len(rep.root_system),
                     "integrand residual": float(abs(direct - spectral)),
                     "all degree <= 3 checks pass": all(row["pass"] for row in checks)})
    return rows


@app.cell
def _(mo, rows):
    mo.ui.table(rows)
    return


if __name__ == "__main__":
    app.run()
