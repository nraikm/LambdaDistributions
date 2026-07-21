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
    import numpy as np
    import marimo as mo

    from lambda_distributions import check_formula, dihedral_group, power_sum_moment
    from lambda_distributions.formulas import dihedral_moment
    from lambda_distributions.notebook_support import display_number, parse_partition, verification_records

    return check_formula, dihedral_group, dihedral_moment, display_number, mo, np, parse_partition, power_sum_moment, verification_records


@app.cell
def _(mo):
    mo.md(r"""
    # Dihedral group

    Inspect the standard two-dimensional representation of $D_n$ and verify its
    power-sum moment formula over all partitions through a selected degree.
    """)
    return


@app.cell
def _(mo):
    n = mo.ui.slider(2, 30, value=6, label="n", show_value=True)
    k = mo.ui.slider(1, 30, value=1, label="k", show_value=True)
    partition_text = mo.ui.text(value="4, 2, 2", label="partition λ")
    max_degree = mo.ui.slider(0, 12, value=7, label="check through degree", show_value=True)
    run = mo.ui.run_button(label="Evaluate")
    mo.hstack([n, k, partition_text, max_degree, run], justify="start", gap=1)
    return k, max_degree, n, partition_text, run


@app.cell
def _(
    check_formula,
    dihedral_group,
    dihedral_moment,
    k,
    max_degree,
    mo,
    n,
    parse_partition,
    partition_text,
    power_sum_moment,
    run,
):
    mo.stop(not run.value, mo.md("Choose parameters, then press **Evaluate**."))
    try:
        partition = parse_partition(partition_text.value)
    except ValueError as error:
        mo.stop(True, mo.callout(str(error), kind="danger"))

    group = dihedral_group(n.value, k.value)
    explicit = power_sum_moment(group, partition)
    predicted = dihedral_moment(partition, n.value, k.value)
    report = check_formula(
        group,
        lambda candidate: dihedral_moment(candidate, n.value, k.value),
        max_degree=max_degree.value,
    )
    rotation = group.elements[1]
    reflection = group.elements[n.value]
    return explicit, group, partition, predicted, reflection, report, rotation


@app.cell
def _(
    display_number,
    explicit,
    group,
    mo,
    np,
    partition,
    predicted,
    reflection,
    report,
    rotation,
    verification_records,
):
    matrices = f"""```text
rotation r:
{np.array2string(rotation, precision=4, suppress_small=True)}

reflection f:
{np.array2string(reflection, precision=4, suppress_small=True)}
```"""
    mo.vstack(
        [
            mo.md(
                fr"""
                ## {group.name}

                For $\lambda={partition}$: direct average **{display_number(explicit)}**;
                formula **{display_number(predicted)}**.
                """
            ),
            mo.callout(
                f"{'All checks passed' if report.passed else 'A check failed'}; maximum error {report.max_error:.3g}.",
                kind="success" if report.passed else "danger",
            ),
            mo.ui.tabs(
                {
                    "Formula checks": mo.ui.table(verification_records(report), pagination=True),
                    "Representative matrices": mo.md(matrices),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
