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

    from lambda_distributions import average_homogeneous, check_formula, dicyclic_group
    from lambda_distributions.formulas import dicyclic_homogeneous
    from lambda_distributions.notebook_support import display_number, parse_partition, verification_records

    return average_homogeneous, check_formula, dicyclic_group, dicyclic_homogeneous, display_number, mo, parse_partition, verification_records


@app.cell
def _(mo):
    mo.md(r"""
    # Dicyclic group

    Check the complete-homogeneous formula for the faithful two-dimensional
    representation of $\mathrm{Dic}_n$.  The group average and formula are kept
    independent by the framework.
    """)
    return


@app.cell
def _(mo):
    n = mo.ui.slider(2, 20, value=4, label="n", show_value=True)
    partition_text = mo.ui.text(value="4, 2", label="partition λ")
    max_degree = mo.ui.slider(0, 12, value=7, label="check through degree", show_value=True)
    run = mo.ui.run_button(label="Evaluate")
    mo.hstack([n, partition_text, max_degree, run], justify="start", gap=1)
    return max_degree, n, partition_text, run


@app.cell
def _(
    average_homogeneous,
    check_formula,
    dicyclic_group,
    dicyclic_homogeneous,
    max_degree,
    mo,
    n,
    parse_partition,
    partition_text,
    run,
):
    mo.stop(not run.value, mo.md("Choose parameters, then press **Evaluate**."))
    try:
        partition = parse_partition(partition_text.value)
    except ValueError as error:
        mo.stop(True, mo.callout(str(error), kind="danger"))

    group = dicyclic_group(n.value)
    explicit = average_homogeneous(group, partition)
    predicted = dicyclic_homogeneous(partition, n.value)
    report = check_formula(
        group,
        lambda candidate: dicyclic_homogeneous(candidate, n.value),
        max_degree=max_degree.value,
        basis="homogeneous",
    )
    return explicit, group, partition, predicted, report


@app.cell
def _(display_number, explicit, group, mo, partition, predicted, report, verification_records):
    mo.vstack(
        [
            mo.md(
                f"""
                ## {group.name}

                For $h_{{{partition}}}$: direct average **{display_number(explicit)}**;
                formula **{display_number(predicted)}**.
                """
            ),
            mo.callout(
                f"{'All checks passed' if report.passed else 'A check failed'}; maximum error {report.max_error:.3g}.",
                kind="success" if report.passed else "danger",
            ),
            mo.ui.table(verification_records(report), pagination=True),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
