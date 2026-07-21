# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.8", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import sys

    import marimo as mo
    import numpy as np

    folder = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/matrix_group_formula_verification/restricted_monomial/restricted_monomial_verification.py").parent
    shared = folder.parent
    repository_root = folder.parents[2]
    for path in (repository_root, shared, folder):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from common import projector_check
    from verification import representative_cases, verify_case

    return mo, np, projector_check, representative_cases, verify_case


@app.cell
def _(mo):
    mo.md(r"""
    # Restricted monomial groups: direct matrix checks

    For (G=D(A)\rtimes H), the direct route constructs every monomial matrix and
    averages the character of (\bigotimes_j\operatorname{Sym}^{\tau_j}\mathbb C^n).
    The independent route uses cycle lengths and the dual-code condition.  For
    (G(r,p,n)), it instead extracts the coefficient of the exact multiset formula.
    """)
    return


@app.cell
def _(mo, representative_cases):
    available = {case.name: index for index, (case, _) in enumerate(representative_cases())}
    selected = mo.ui.dropdown(options=available, value=next(iter(available)), label="group")
    maximum_degree = mo.ui.slider(0, 8, value=6, show_value=True, label="maximum degree")
    run = mo.ui.run_button(label="Construct matrices and verify")
    mo.hstack([selected, maximum_degree, run], justify="start", gap=1)
    return maximum_degree, run, selected


@app.cell
def _(maximum_degree, mo, representative_cases, run, selected, verify_case):
    mo.stop(not run.value, mo.md("Choose a case and run the verification."))
    case, formula_name = representative_cases()[selected.value]
    rows = verify_case(case, maximum_degree.value)
    return case, formula_name, rows


@app.cell
def _(case, formula_name, mo, np, projector_check, rows):
    passed = all(row["pass"] for row in rows)
    tau = (2, 1) if case.matrices[0].shape[0] <= 3 else (2,)
    projector = projector_check(case.matrices, tau)
    sample = next(
        (matrix for matrix in case.matrices if not np.allclose(matrix, np.eye(matrix.shape[0]))),
        case.matrices[0],
    )
    mo.vstack(
        [
            mo.callout(
                f"{'All checks passed' if passed else 'A discrepancy was found'} for {case.name}. "
                f"Constructed {len(case.matrices)} matrices; formula route: {formula_name}.",
                kind="success" if passed else "danger",
            ),
            mo.md(f"**Sample matrix**\n```text\n{np.array2string(sample, precision=3, suppress_small=True)}\n```"),
            mo.md(f"**Reynolds projector check for tau={tau}:** `{projector}`"),
            mo.ui.table(list(rows), pagination=True),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
