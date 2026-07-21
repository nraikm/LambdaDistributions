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

    sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/compact_connected_lie_groups/spin7/spin7_verification.py").parents[1]))
    from verification_core import (
        exceptional_representations,
        verify_expected,
        verify_monomial_expected,
    )

    return exceptional_representations, mo, np, verify_expected, verify_monomial_expected


@app.cell
def _(mo):
    mo.md(r"""
    # $\operatorname{Spin}(7)$ on its complexified real spin module $S_8$

    This notebook constructs an explicit maximal-torus matrix from the eight
    spin weights, evaluates the determinant integrand directly, and performs
    exact Weyl constant-term checks through total degree four.  The degree-four
    computation detects the Cayley form independently of the tensor-square
    decomposition used in the proof.
    """)
    return


@app.cell
def _(exceptional_representations, np, verify_expected, verify_monomial_expected):
    rep = exceptional_representations()["spin7_8"]
    angles = (0.17, -0.29, 0.41)
    matrix = rep.torus_matrix(angles)
    ts = (0.04, 0.07, 0.09)

    determinant_value = np.prod(
        [1 / np.linalg.det(np.eye(rep.dimension) - t * matrix) for t in ts]
    )
    eigenvalue_value = np.prod(
        [1 / (1 - t * eigenvalue) for t in ts for eigenvalue in np.diag(matrix)]
    )

    schur_expected = {
        (): 1,
        (2,): 1,
        (4,): 1,
        (2, 2): 1,
        (1, 1, 1, 1): 1,
    }
    monomial_expected = {
        (): 1,
        (2,): 1,
        (1, 1): 1,
        (4,): 1,
        (3, 1): 1,
        (2, 2): 2,
        (2, 1, 1): 2,
        (1, 1, 1, 1): 4,
    }
    schur_checks = verify_expected(rep, schur_expected, 4)
    monomial_checks = verify_monomial_expected(rep, monomial_expected, 4)

    summary = {
        "representation dimension": rep.dimension,
        "root count": len(rep.root_system),
        "spin-weight count": len(set(rep.weights)),
        "Weyl-group order": rep.weyl_order,
        "unitarity residual": float(
            np.linalg.norm(matrix.conj().T @ matrix - np.eye(rep.dimension))
        ),
        "determinant/product residual": float(abs(determinant_value - eigenvalue_value)),
        "all exact checks pass": all(
            row["pass"] for row in schur_checks + monomial_checks
        ),
    }
    return matrix, monomial_checks, schur_checks, summary


@app.cell
def _(matrix, mo, monomial_checks, schur_checks, summary):
    mo.vstack(
        [
            mo.md("## Constructed matrix and direct integrand check"),
            mo.ui.table([summary]),
            mo.md("The sampled $8\\times8$ torus matrix is diagonal:"),
            mo.ui.table(
                [
                    {
                        "index": index,
                        "diagonal entry": f"{value.real:+.10f}{value.imag:+.10f}i",
                    }
                    for index, value in enumerate(matrix.diagonal(), start=1)
                ]
            ),
            mo.md("## Requested monomial-basis coefficients $[m_\\tau]$"),
            mo.ui.table(monomial_checks),
            mo.md("## Equivalent Schur-basis invariant multiplicities"),
            mo.ui.table(schur_checks),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
