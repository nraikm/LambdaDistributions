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

    root = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/finite_group_sigma_mgf/regular_representations/notebook.py").parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from lambda_distributions.proofs.finite_group_sigma_mgf.verification_core import (
        regular_family_matrices,
        regular_order_counts,
        representative_suites,
        verify_regular_case,
    )

    return (
        mo,
        regular_family_matrices,
        regular_order_counts,
        representative_suites,
        verify_regular_case,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Regular representations: exact sigma-MGF checks

    Choose (C_n), (D_n) (of order (2n)), (S_n), or (A_n).  The
    notebook constructs every left-regular permutation matrix.  For the chosen
    partition (	au), it compares the rank of the explicit Reynolds projector
    on (igotimes_joperatorname{Sym}^{	au_j}R_G) with the element-order
    formula.  It also evaluates the full determinant average at two safe test
    values and compares it with the closed rational expression.
    """)
    return


@app.cell
def _(mo):
    family = mo.ui.dropdown(
        options={"cyclic": "cyclic", "dihedral": "dihedral", "symmetric": "symmetric", "alternating": "alternating"},
        value="cyclic",
        label="group family",
    )
    parameter = mo.ui.slider(2, 6, value=5, label="n", show_value=True)
    tau_text = mo.ui.text(value="2, 1", label="tau")
    run = mo.ui.run_button(label="Construct matrices and verify")
    mo.hstack([family, parameter, tau_text, run], justify="start", gap=1)
    return family, parameter, run, tau_text


@app.cell
def _(family, mo, parameter, regular_family_matrices, regular_order_counts, run, tau_text, verify_regular_case):
    mo.stop(not run.value, mo.md("Set the parameters and press **Construct matrices and verify**."))
    selected_parameter = parameter.value
    if family.value == "symmetric":
        selected_parameter = min(selected_parameter, 4)
    if family.value == "alternating":
        selected_parameter = min(max(selected_parameter, 3), 4)
    selected_tau = tuple(int(item.strip()) for item in tau_text.value.split(",") if item.strip())
    result = verify_regular_case(family.value, selected_parameter, selected_tau)
    group_order, order_counts = regular_order_counts(family.value, selected_parameter)
    sample_matrix = regular_family_matrices(family.value, selected_parameter)[1]
    return group_order, order_counts, result, sample_matrix, selected_parameter, selected_tau


@app.cell
def _(family, group_order, mo, order_counts, result, sample_matrix, selected_parameter):
    mo.vstack([
        mo.callout(
            "Projector, character average, order formula, and determinant value agree."
            if result.passed else "A route disagrees; inspect the diagnostics.",
            kind="success" if result.passed else "danger",
        ),
        mo.ui.table([result.as_dict()], pagination=False),
        mo.md(
            f"**Selected group:** {family.value}({selected_parameter}); "
            f"**order:** {group_order}; **element-order counts:** `{dict(sorted(order_counts.items()))}`"
        ),
        mo.md("A nonidentity left-regular matrix (rows and columns are group elements):"),
        mo.ui.table(sample_matrix.real.astype(int).tolist(), pagination=False),
    ])
    return


@app.cell
def _(mo, representative_suites):
    mo.vstack([
        mo.md("## Fixed representative sweep"),
        mo.ui.table(representative_suites()["regular"], pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()

