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
    import numpy as np

    from lambda_distributions import (
        alternating_group,
        cyclic_character,
        cyclic_permutation_group,
        cyclic_real_2d,
        dicyclic_group,
        dihedral_group,
        generalized_symmetric_group,
        pauli_group,
    )

    return alternating_group, cyclic_character, cyclic_permutation_group, cyclic_real_2d, dicyclic_group, dihedral_group, generalized_symmetric_group, mo, np, pauli_group


@app.cell
def _(mo):
    mo.md(r"""
    # Matrix-group catalog

    This replaces the original collection of unrelated Sage experiments with a
    single view of every representation available in the framework.  Repeated
    matrices are intentionally retained when a representation has a kernel,
    because expectations are averages over the abstract group.
    """)
    return


@app.cell
def _(mo):
    n = mo.ui.slider(1, 6, value=3, label="base parameter n", show_value=True)
    k = mo.ui.slider(1, 6, value=1, label="character k", show_value=True)
    level = mo.ui.slider(1, 3, value=2, label="wreath level ℓ", show_value=True)
    build = mo.ui.run_button(label="Build catalog")
    mo.hstack([n, k, level, build], justify="start", gap=1)
    return build, k, level, n


@app.cell
def _(
    alternating_group,
    build,
    cyclic_character,
    cyclic_permutation_group,
    cyclic_real_2d,
    dicyclic_group,
    dihedral_group,
    generalized_symmetric_group,
    k,
    level,
    mo,
    n,
    pauli_group,
):
    mo.stop(not build.value, mo.md("Press **Build catalog** to construct the examples."))
    order = n.value
    groups = [
        cyclic_character(order, k.value),
        cyclic_permutation_group(order),
        cyclic_real_2d(order, k.value),
        dihedral_group(max(2, order), k.value),
        dicyclic_group(max(2, order)),
        pauli_group(min(3, order)),
        alternating_group(min(6, order)),
        generalized_symmetric_group(min(4, order), level.value),
    ]
    catalog = {group.name: group for group in groups}
    rows = [
        {"representation": group.name, "group order": group.order, "dimension": group.dimension}
        for group in groups
    ]
    return catalog, rows


@app.cell
def _(catalog, mo):
    selected_name = mo.ui.dropdown(
        options=list(catalog),
        value=next(iter(catalog)),
        label="inspect representation",
    )
    selected_name
    return (selected_name,)


@app.cell
def _(catalog, mo, np, rows, selected_name):
    selected_group = catalog[selected_name.value]
    representative = selected_group.elements[min(1, selected_group.order - 1)]
    matrix_text = np.array2string(representative, precision=4, suppress_small=True)
    mo.vstack(
        [
            mo.ui.table(rows, pagination=False),
            mo.md(
                f"""
                ## {selected_group.name}

                Representative non-identity element:

                ```text
                {matrix_text}
                ```
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
