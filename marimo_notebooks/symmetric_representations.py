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

    root = (Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/finite_group_sigma_mgf/symmetric_group_representations/notebook.py").parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from for_this_guy.finite_group_sigma_mgf.verification_core import (
        derived_character_errors,
        induced_s3_from_c3,
        induced_verification_row,
        representative_suites,
        sign_matrices,
        standard_matrices,
        verify_sign_case,
        verify_standard_case,
    )

    return (
        derived_character_errors,
        induced_s3_from_c3,
        induced_verification_row,
        mo,
        representative_suites,
        sign_matrices,
        standard_matrices,
        verify_sign_case,
        verify_standard_case,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # (S_n): sign, standard, derived, and induced representations

    For the sign and standard representations, the notebook constructs every
    matrix, forms the Reynolds projector, and compares its rank with the
    proposed closed formula.  The determinant average is checked independently
    using cycle types.  The induced example constructs
    (operatorname{Ind}_{C_3}^{S_3}omega) from coset representatives and
    checks the decorated coset-cycle determinant formula.
    """)
    return


@app.cell
def _(mo):
    representation = mo.ui.dropdown(
        options={"standard": "standard", "sign": "sign", "Ind(C3 character) to S3": "induced"},
        value="standard",
        label="representation",
    )
    n = mo.ui.slider(3, 6, value=4, label="n", show_value=True)
    tau_text = mo.ui.text(value="2, 1", label="tau")
    run = mo.ui.run_button(label="Construct matrices and verify")
    mo.hstack([representation, n, tau_text, run], justify="start", gap=1)
    return n, representation, run, tau_text


@app.cell
def _(
    induced_s3_from_c3,
    induced_verification_row,
    mo,
    n,
    representation,
    run,
    sign_matrices,
    standard_matrices,
    tau_text,
    verify_sign_case,
    verify_standard_case,
):
    mo.stop(not run.value, mo.md("Set the parameters and press **Construct matrices and verify**."))
    selected_tau = tuple(int(item.strip()) for item in tau_text.value.split(",") if item.strip())
    if representation.value == "induced":
        selected_n = 3
        result = induced_verification_row(selected_tau)
        matrices = induced_s3_from_c3()[0]
    elif representation.value == "sign":
        selected_n = n.value
        result = verify_sign_case(selected_n, selected_tau)
        matrices = sign_matrices(selected_n)
    else:
        selected_n = n.value
        result = verify_standard_case(selected_n, selected_tau)
        matrices = standard_matrices(selected_n)
    sample_matrix = matrices[1]
    return result, sample_matrix, selected_n, selected_tau


@app.cell
def _(mo, representation, result, sample_matrix, selected_n):
    mo.vstack([
        mo.callout(
            "All independent routes agree." if result.passed else "A route disagrees; inspect the diagnostics.",
            kind="success" if result.passed else "danger",
        ),
        mo.ui.table([result.as_dict()], pagination=False),
        mo.md(f"**Selected:** {representation.value}, n={selected_n}.  One nonidentity matrix:"),
        mo.ui.table(sample_matrix.round(8).tolist(), pagination=False),
    ])
    return


@app.cell
def _(derived_character_errors, mo):
    errors = derived_character_errors(3)
    mo.vstack([
        mo.md("## Symmetric, exterior, and tensor construction checks"),
        mo.ui.table([{"identity": key, "maximum error": value} for key, value in errors.items()], pagination=False),
    ])
    return


@app.cell
def _(mo, representative_suites):
    mo.vstack([
        mo.md("## Fixed representative sweep"),
        mo.ui.table(representative_suites()["symmetric"], pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()

