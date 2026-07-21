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

    from lambda_distributions import (
        alternating_group,
        check_formula,
        cyclic_character,
        cyclic_real_2d,
        dicyclic_group,
        dihedral_group,
        generalized_symmetric_group,
        pauli_group,
    )
    from lambda_distributions.formulas import (
        alternating_permutation_moment,
        cyclic_character_moment,
        cyclic_real_2d_moment,
        dicyclic_homogeneous,
        dihedral_moment,
        generalized_symmetric_moment,
        pauli_homogeneous,
    )
    from lambda_distributions.notebook_support import verification_records

    return (
        check_formula,
        cyclic_character,
        cyclic_character_moment,
        cyclic_real_2d,
        cyclic_real_2d_moment,
        dicyclic_group,
        dicyclic_homogeneous,
        dihedral_group,
        dihedral_moment,
        generalized_symmetric_group,
        generalized_symmetric_moment,
        mo,
        pauli_group,
        pauli_homogeneous,
        verification_records,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Formula checks

    This notebook compares a closed formula with direct finite-group averaging
    for every partition up to the selected degree.  Green means the formula and
    enumeration agree to the selected numerical tolerance.
    """)
    return


@app.cell
def _(mo):
    formula_name = mo.ui.dropdown(
        options=[
            "Cyclic character / power sums",
            "Cyclic real 2D / power sums",
            "Dihedral / power sums",
            "Alternating / power sums",
            "Generalized symmetric / power sums",
            "Dicyclic / homogeneous",
            "Pauli / homogeneous",
        ],
        value="Cyclic character / power sums",
        label="formula",
    )
    n = mo.ui.slider(2, 10, value=5, label="n", show_value=True)
    k = mo.ui.slider(1, 10, value=1, label="k", show_value=True)
    level = mo.ui.slider(1, 4, value=2, label="level ℓ", show_value=True)
    max_degree = mo.ui.slider(0, 10, value=6, label="maximum degree", show_value=True)
    tolerance_exponent = mo.ui.slider(6, 13, value=9, label="-log10(tolerance)", show_value=True)
    run = mo.ui.run_button(label="Check formula")
    mo.hstack([formula_name, n, k, level, max_degree, tolerance_exponent, run], justify="start", gap=1)
    return formula_name, k, level, max_degree, n, run, tolerance_exponent


@app.cell
def _(
    check_formula,
    cyclic_character,
    cyclic_character_moment,
    cyclic_real_2d,
    cyclic_real_2d_moment,
    dicyclic_group,
    dicyclic_homogeneous,
    dihedral_group,
    dihedral_moment,
    formula_name,
    generalized_symmetric_group,
    generalized_symmetric_moment,
    k,
    level,
    max_degree,
    mo,
    n,
    pauli_group,
    pauli_homogeneous,
    run,
    tolerance_exponent,
):
    mo.stop(not run.value, mo.md("Choose a formula, then press **Check formula**."))

    selected = formula_name.value
    order = n.value
    character = k.value
    if selected == "Cyclic character / power sums":
        group = cyclic_character(order, character)
        formula = lambda partition: cyclic_character_moment(partition, order, character)
        basis = "power_sum"
    elif selected == "Cyclic real 2D / power sums":
        group = cyclic_real_2d(order, character)
        formula = lambda partition: cyclic_real_2d_moment(partition, order, character)
        basis = "power_sum"
    elif selected == "Dihedral / power sums":
        group = dihedral_group(order, character)
        formula = lambda partition: dihedral_moment(partition, order, character)
        basis = "power_sum"
    elif selected == "Alternating / power sums":
        alternating_degree = min(8, order)
        group = alternating_group(alternating_degree)
        formula = lambda partition: alternating_permutation_moment(partition, alternating_degree)
        basis = "power_sum"
    elif selected == "Generalized symmetric / power sums":
        generalized_degree = min(4, order)
        generalized_level = min(3, level.value)
        group = generalized_symmetric_group(generalized_degree, generalized_level)
        formula = lambda partition: generalized_symmetric_moment(
            partition, generalized_degree, generalized_level
        )
        basis = "power_sum"
    elif selected == "Dicyclic / homogeneous":
        group = dicyclic_group(order)
        formula = lambda partition: dicyclic_homogeneous(partition, order)
        basis = "homogeneous"
    else:
        pauli_qubits = min(3, order)
        group = pauli_group(pauli_qubits)
        formula = lambda partition: pauli_homogeneous(partition, pauli_qubits)
        basis = "homogeneous"

    report = check_formula(
        group,
        formula,
        max_degree=max_degree.value,
        basis=basis,
        tolerance=10 ** (-tolerance_exponent.value),
    )
    return report


@app.cell
def _(mo, report, verification_records):
    status = "All checks passed" if report.passed else "At least one check failed"
    kind = "success" if report.passed else "danger"
    mo.vstack(
        [
            mo.callout(
                f"{status} for {report.group_name}. Maximum error: {report.max_error:.3g}",
                kind=kind,
            ),
            mo.ui.table(verification_records(report), pagination=True),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
