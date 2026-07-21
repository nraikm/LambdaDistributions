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

    source_directory = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/exact_finite_n_molien/wreath_products/wreath_product_verification.py").parent
    if str(source_directory) not in sys.path:
        sys.path.insert(0, str(source_directory))
    from verification import BASE_GROUPS, verify_case

    return BASE_GROUPS, mo, np, verify_case


@app.cell
def _(mo):
    mo.md(r"""
    # Exact finite-$n$ check for $H\wr S_n$

    The direct side constructs all block-monomial matrices of $H\wr S_n$ on
    $V^{\oplus n}$ and averages tensor-symmetric-power characters.  The
    independent prediction uses the recurrence obtained from
    $$
    \sum_{n\ge0}u^n\mathcal M_{H\wr S_n}(\mathbf t)
      =\exp\!\left(\sum_{\ell\ge1}\frac{u^\ell}{\ell}
      \mathcal M_H(t_1^\ell,t_2^\ell,\ldots)\right).
    $$
    Scalar cyclic, reducible reflection, and irreducible nonabelian base
    representations are included.
    """)
    return


@app.cell
def _(mo):
    base_choice = mo.ui.dropdown(
        options=["C2 scalar", "C3 scalar", "C2 reflection", "S3 standard"],
        value="S3 standard",
        label="base matrix group H",
    )
    block_count = mo.ui.slider(1, 3, value=2, label="number of blocks n", show_value=True)
    maximum_degree = mo.ui.slider(0, 7, value=5, label="maximum total degree", show_value=True)
    run_check = mo.ui.run_button(label="Construct wreath product and verify")
    mo.hstack([base_choice, block_count, maximum_degree, run_check], justify="start", gap=1)
    return base_choice, block_count, maximum_degree, run_check


@app.cell
def _(BASE_GROUPS, base_choice, block_count, maximum_degree, mo, run_check, verify_case):
    mo.stop(not run_check.value, mo.md("Choose a case and press **Construct wreath product and verify**."))
    selected_base = BASE_GROUPS[base_choice.value]()
    check_result = verify_case(
        selected_base, block_count.value, max_degree=maximum_degree.value
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
                f"{result_text} Constructed {check_result.group_order} matrices for "
                f"{check_result.base_name} wreath S_{check_result.n}; ambient dimension "
                f"{check_result.block_dimension * check_result.n}.",
                kind=result_kind,
            ),
            mo.md(f"**One constructed nonidentity matrix**\n```text\n{matrix_text}\n```"),
            mo.ui.table(list(check_result.rows), pagination=True),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
