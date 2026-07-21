# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.23.8",
#   "numpy>=2.0",
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import sys

    import marimo as mo
    import numpy as np

    source_directory = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/exact_finite_n_molien/g_r_p_n/g_r_p_n_verification.py").parent
    if str(source_directory) not in sys.path:
        sys.path.insert(0, str(source_directory))
    from verification import verify_case

    return mo, np, verify_case


@app.cell
def _(mo):
    mo.md(r"""
    # Exact finite-$n$ check for $G(r,p,n)$

    This notebook constructs every monomial matrix in $G(r,p,n)$ and computes
    $$
    \dim\!\left(\bigotimes_j \operatorname{Sym}^{\tau_j}\mathbb C^n\right)^{G(r,p,n)}
    =\frac{1}{|G|}\sum_{g\in G}\prod_j h_{\tau_j}(\operatorname{eig}g).
    $$
    It independently extracts the same coefficient from
    $$
    [u^n]\sum_{q=0}^{p-1}\operatorname{Exp}_\sigma
      \left(u\sum_{j\ge0}h_{qr/p+jr}\right).
    $$
    The table checks every partition through the selected total degree.
    """)
    return


@app.cell
def _(mo):
    case = mo.ui.dropdown(
        options={
            "B2/C2 = G(2,1,2)": (2, 1, 2),
            "D4 = G(2,2,4)": (2, 2, 4),
            "G(3,1,2)": (3, 1, 2),
            "G(4,2,2)": (4, 2, 2),
            "G(4,4,2)": (4, 4, 2),
        },
        value="B2/C2 = G(2,1,2)",
        label="matrix group",
    )
    maximum_degree = mo.ui.slider(0, 8, value=6, label="maximum total degree", show_value=True)
    run_check = mo.ui.run_button(label="Construct matrices and verify")
    mo.hstack([case, maximum_degree, run_check], justify="start", gap=1)
    return case, maximum_degree, run_check


@app.cell
def _(case, maximum_degree, mo, run_check, verify_case):
    mo.stop(not run_check.value, mo.md("Choose a case and press **Construct matrices and verify**."))
    selected_r, selected_p, selected_n = case.value
    check_result = verify_case(
        selected_r, selected_p, selected_n, max_degree=maximum_degree.value
    )
    return check_result


@app.cell
def _(check_result, mo, np):
    result_kind = "success" if check_result.passed else "danger"
    result_text = "All coefficients agree." if check_result.passed else "A discrepancy was found."
    matrix_text = np.array2string(
        check_result.sample_matrix,
        precision=3,
        suppress_small=True,
    )
    mo.vstack(
        [
            mo.callout(
                f"{result_text} Constructed {check_result.group_order} matrices in "
                f"G({check_result.r},{check_result.p},{check_result.n}).",
                kind=result_kind,
            ),
            mo.md(f"**One constructed nonidentity matrix**\n```text\n{matrix_text}\n```"),
            mo.ui.table(list(check_result.rows), pagination=True),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
