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
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from lambda_distributions import (
        alternating_group,
        average_homogeneous,
        cyclic_character,
        cyclic_permutation_group,
        cyclic_real_2d,
        dicyclic_group,
        dihedral_group,
        generalized_symmetric_group,
        lambda_distribution,
        normalize_partition,
        pauli_group,
        power_sum_moment,
        sigma_mgf_coefficients,
    )

    return (
        alternating_group,
        average_homogeneous,
        cyclic_character,
        cyclic_permutation_group,
        cyclic_real_2d,
        dicyclic_group,
        dihedral_group,
        generalized_symmetric_group,
        lambda_distribution,
        mo,
        normalize_partition,
        pauli_group,
        power_sum_moment,
        sigma_mgf_coefficients,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Lambda-distribution laboratory

    Choose a finite matrix-group representation and a partition.  The notebook
    evaluates both the power-sum observable $p_\lambda$ and the complete
    homogeneous observable $h_\lambda$ by averaging over the represented group.
    """)
    return


@app.cell
def _(mo):
    family = mo.ui.dropdown(
        options=[
            "Cyclic character",
            "Cyclic permutation",
            "Cyclic real 2D",
            "Dihedral",
            "Dicyclic",
            "Pauli",
            "Alternating",
            "Generalized symmetric",
        ],
        value="Cyclic character",
        label="group",
    )
    n = mo.ui.slider(1, 8, value=5, label="n", show_value=True)
    k = mo.ui.slider(1, 8, value=1, label="k", show_value=True)
    level = mo.ui.slider(1, 4, value=2, label="level", show_value=True)
    partition_text = mo.ui.text(value="3, 1", label="partition λ")
    max_degree = mo.ui.slider(0, 8, value=4, label="sigma-MGF degree", show_value=True)
    run = mo.ui.run_button(label="Evaluate")

    controls = mo.vstack(
        [
            mo.hstack([family, n, k, level], justify="start", gap=1),
            mo.hstack([partition_text, max_degree, run], justify="start", gap=1),
        ]
    )
    controls
    return family, k, level, max_degree, n, partition_text, run


@app.cell
def _(
    alternating_group,
    cyclic_character,
    cyclic_permutation_group,
    cyclic_real_2d,
    dicyclic_group,
    dihedral_group,
    family,
    generalized_symmetric_group,
    k,
    level,
    mo,
    n,
    normalize_partition,
    pauli_group,
    partition_text,
    run,
):
    mo.stop(not run.value, mo.md("Select parameters, then press **Evaluate**."))

    try:
        partition = normalize_partition(
            int(piece.strip())
            for piece in partition_text.value.split(",")
            if piece.strip()
        )
    except ValueError as error:
        mo.stop(True, mo.callout(str(error), kind="danger"))

    family_name = family.value
    if family_name == "Cyclic character":
        group = cyclic_character(n.value, k.value)
    elif family_name == "Cyclic permutation":
        group = cyclic_permutation_group(n.value)
    elif family_name == "Cyclic real 2D":
        group = cyclic_real_2d(n.value, k.value)
    elif family_name == "Dihedral":
        group = dihedral_group(max(2, n.value), k.value)
    elif family_name == "Dicyclic":
        group = dicyclic_group(max(2, n.value))
    elif family_name == "Pauli":
        group = pauli_group(min(3, n.value))
    elif family_name == "Alternating":
        group = alternating_group(min(6, n.value))
    else:
        group = generalized_symmetric_group(min(4, n.value), min(3, level.value))

    return group, partition


@app.cell
def _(
    average_homogeneous,
    group,
    lambda_distribution,
    max_degree,
    mo,
    partition,
    power_sum_moment,
    sigma_mgf_coefficients,
):
    p_moment = power_sum_moment(group, partition)
    h_moment = average_homogeneous(group, partition)
    distribution_value = lambda_distribution(group, {partition: 1})
    coefficients = sigma_mgf_coefficients(group, max_degree.value)
    nonzero_coefficients = [
        {"partition": str(key), "coefficient": value}
        for key, value in coefficients.items()
        if abs(value) > 1e-10
    ]

    mo.vstack(
        [
            mo.md(
                fr"""
                ## {group.name}

                - abstract group elements averaged: **{group.order}**
                - representation dimension: **{group.dimension}**
                - partition: $\lambda={partition}$
                - $\mathbb{{E}}[p_\lambda(X)] = {p_moment}$
                - $\mathbb{{E}}[h_\lambda(X)] = {h_moment}$
                - $\mu_X(p_\lambda) = {distribution_value}$
                """
            ),
            mo.md(f"### Nonzero sigma-MGF coefficients through degree {max_degree.value}"),
            mo.ui.table(nonzero_coefficients, pagination=True),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
