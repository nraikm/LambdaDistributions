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

    from lambda_distributions import check_formula, cyclic_character, power_sum_moment, sigma_mgf_coefficients
    from lambda_distributions.formulas import cyclic_character_moment
    from lambda_distributions.notebook_support import (
        coefficient_records,
        display_number,
        parse_partition,
        verification_records,
    )

    return (
        check_formula,
        coefficient_records,
        cyclic_character,
        cyclic_character_moment,
        display_number,
        mo,
        parse_partition,
        power_sum_moment,
        sigma_mgf_coefficients,
        verification_records,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Cyclic character

    For the character $m \mapsto e^{2\pi i k m/n}$, this notebook compares
    direct group averaging with
    $\mathbb{E}[p_\lambda]=\mathbf{1}_{n\mid k|\lambda|}$.

    For the related monomial-basis theorem with coefficient $n$, open
    `notebooks/cyclic_monomial_walkthrough.py`.
    """)
    return


@app.cell
def _(mo):
    n = mo.ui.slider(1, 30, value=8, label="n", show_value=True)
    k = mo.ui.slider(1, 30, value=3, label="k", show_value=True)
    partition_text = mo.ui.text(value="4, 2", label="partition λ")
    max_degree = mo.ui.slider(0, 14, value=8, label="check through degree", show_value=True)
    run = mo.ui.run_button(label="Evaluate")
    mo.hstack([n, k, partition_text, max_degree, run], justify="start", gap=1)
    return k, max_degree, n, partition_text, run


@app.cell
def _(
    check_formula,
    cyclic_character,
    cyclic_character_moment,
    k,
    max_degree,
    mo,
    n,
    parse_partition,
    partition_text,
    power_sum_moment,
    run,
    sigma_mgf_coefficients,
):
    mo.stop(not run.value, mo.md("Choose parameters, then press **Evaluate**."))
    try:
        partition = parse_partition(partition_text.value)
    except ValueError as error:
        mo.stop(True, mo.callout(str(error), kind="danger"))

    group = cyclic_character(n.value, k.value)
    explicit = power_sum_moment(group, partition)
    predicted = cyclic_character_moment(partition, n.value, k.value)
    report = check_formula(
        group,
        lambda candidate: cyclic_character_moment(candidate, n.value, k.value),
        max_degree=max_degree.value,
    )
    coefficients = sigma_mgf_coefficients(group, max_degree.value)
    return coefficients, explicit, group, partition, predicted, report


@app.cell
def _(
    coefficient_records,
    coefficients,
    display_number,
    explicit,
    group,
    mo,
    partition,
    predicted,
    report,
    verification_records,
):
    summary = mo.md(
        fr"""
        ## Result for {group.name}

        $\lambda={partition}$, direct average **{display_number(explicit)}**,
        formula **{display_number(predicted)}**.  Maximum batch-check error:
        **{report.max_error:.3g}**.
        """
    )
    status = mo.callout(
        "All formula checks passed." if report.passed else "A formula check failed.",
        kind="success" if report.passed else "danger",
    )
    mo.vstack(
        [
            summary,
            status,
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
