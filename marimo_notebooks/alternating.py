# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "lambda-distributions",
#   "marimo>=0.23.8",
# ]
#
# [tool.uv.sources]
# lambda-distributions = { path = "..", editable = true }
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from lambda_distributions import alternating_group, check_formula, sigma_mgf_coefficients
    from lambda_distributions.formulas import alternating_permutation_moment
    from lambda_distributions.notebook_support import coefficient_records, verification_records

    return alternating_group, alternating_permutation_moment, check_formula, coefficient_records, mo, sigma_mgf_coefficients, verification_records


@app.cell
def _(mo):
    mo.md(r"""
    # Alternating-group verification

    Compare explicit permutation matrices with a cycle-type formula for
    $\mathbb{E}[p_\lambda]$ on the permutation representation of $A_n$.
    """)
    return


@app.cell
def _(mo):
    n = mo.ui.slider(1, 8, value=4, label="n", show_value=True)
    max_degree = mo.ui.slider(0, 10, value=7, label="maximum degree", show_value=True)
    run = mo.ui.run_button(label="Verify")
    mo.hstack([n, max_degree, run], justify="start", gap=1)
    return max_degree, n, run


@app.cell
def _(
    alternating_group,
    alternating_permutation_moment,
    check_formula,
    max_degree,
    mo,
    n,
    run,
    sigma_mgf_coefficients,
):
    mo.stop(not run.value, mo.md("Press **Verify** to enumerate the group."))
    group = alternating_group(n.value)
    report = check_formula(
        group,
        lambda partition: alternating_permutation_moment(partition, n.value),
        max_degree=max_degree.value,
    )
    coefficients = sigma_mgf_coefficients(group, max_degree.value)
    return coefficients, group, report


@app.cell
def _(coefficient_records, coefficients, group, mo, report, verification_records):
    mo.vstack(
        [
            mo.md(f"## {group.name}\n\nOrder **{group.order}**, dimension **{group.dimension}**."),
            mo.callout(
                f"{'All checks passed' if report.passed else 'A check failed'}; maximum error {report.max_error:.3g}.",
                kind="success" if report.passed else "danger",
            ),
            mo.ui.tabs(
                {
                    "Formula checks": mo.ui.table(verification_records(report), pagination=True),
                    "Sigma-MGF coefficients": mo.ui.table(coefficient_records(coefficients), pagination=True),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
