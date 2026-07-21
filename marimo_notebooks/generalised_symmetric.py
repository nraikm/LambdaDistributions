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

    from lambda_distributions import check_formula, generalized_symmetric_group, sigma_mgf_coefficients
    from lambda_distributions.formulas import generalized_symmetric_moment
    from lambda_distributions.notebook_support import coefficient_records, verification_records

    return check_formula, coefficient_records, generalized_symmetric_group, generalized_symmetric_moment, mo, sigma_mgf_coefficients, verification_records


@app.cell
def _(mo):
    mo.md(r"""
    # Generalized symmetric-group verification

    Compare explicit monomial matrices for $C_\ell\wr S_n$ with the independent
    colored-cycle formula for $\mathbb{E}[p_\lambda]$.
    """)
    return


@app.cell
def _(mo):
    n = mo.ui.slider(1, 5, value=3, label="n", show_value=True)
    level = mo.ui.slider(1, 4, value=2, label="level ℓ", show_value=True)
    max_degree = mo.ui.slider(0, 9, value=6, label="maximum degree", show_value=True)
    run = mo.ui.run_button(label="Verify")
    mo.hstack([n, level, max_degree, run], justify="start", gap=1)
    return level, max_degree, n, run


@app.cell
def _(
    check_formula,
    generalized_symmetric_group,
    generalized_symmetric_moment,
    level,
    max_degree,
    mo,
    n,
    run,
    sigma_mgf_coefficients,
):
    mo.stop(not run.value, mo.md("Press **Verify** to enumerate the group."))
    estimated_order = level.value**n.value
    for factor in range(2, n.value + 1):
        estimated_order *= factor
    mo.stop(
        estimated_order > 150_000,
        mo.callout(
            f"This selection has {estimated_order:,} elements. Choose smaller n or ℓ for an interactive check.",
            kind="warn",
        ),
    )

    group = generalized_symmetric_group(n.value, level.value)
    report = check_formula(
        group,
        lambda partition: generalized_symmetric_moment(partition, n.value, level.value),
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
