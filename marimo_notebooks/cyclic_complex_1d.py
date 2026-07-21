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

    from lambda_distributions import check_formula, cyclic_character
    from lambda_distributions.formulas import cyclic_character_moment

    return check_formula, cyclic_character, cyclic_character_moment, mo


@app.cell
def _(mo):
    mo.md(r"""
    # Batch verification: one-dimensional cyclic characters

    Check $\mathbb{E}[p_\lambda]=\mathbf{1}_{n\mid k|\lambda|}$ for every
    $1\leq n\leq N$, every character $1\leq k\leq n$, and every partition up
    to the chosen degree.
    """)
    return


@app.cell
def _(mo):
    max_n = mo.ui.slider(1, 24, value=10, label="maximum n", show_value=True)
    max_degree = mo.ui.slider(0, 12, value=8, label="maximum degree", show_value=True)
    run = mo.ui.run_button(label="Run batch check")
    mo.hstack([max_n, max_degree, run], justify="start", gap=1)
    return max_degree, max_n, run


@app.cell
def _(check_formula, cyclic_character, cyclic_character_moment, max_degree, max_n, mo, run):
    mo.stop(not run.value, mo.md("Press **Run batch check** to start."))
    reports = []
    for order in range(1, max_n.value + 1):
        for character in range(1, order + 1):
            report = check_formula(
                cyclic_character(order, character),
                lambda partition, order=order, character=character: cyclic_character_moment(
                    partition, order, character
                ),
                max_degree=max_degree.value,
            )
            reports.append(
                {
                    "n": order,
                    "k": character,
                    "checks": len(report.checks),
                    "max error": f"{report.max_error:.3g}",
                    "passed": report.passed,
                }
            )
    all_passed = all(row["passed"] for row in reports)
    return all_passed, reports


@app.cell
def _(all_passed, mo, reports):
    mo.vstack(
        [
            mo.callout(
                "Every cyclic-character check passed." if all_passed else "At least one check failed.",
                kind="success" if all_passed else "danger",
            ),
            mo.ui.table(reports, pagination=True),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
