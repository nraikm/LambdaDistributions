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
        is_symplectic,
        is_unitary,
        log_sigma_mgf,
        pauli_group,
        random_symplectic,
        random_unitary,
        sigma_mgf,
        sigma_mgf_coefficients,
    )
    from lambda_distributions.notebook_support import coefficient_records, display_number

    return alternating_group, coefficient_records, cyclic_character, cyclic_permutation_group, cyclic_real_2d, dicyclic_group, dihedral_group, display_number, generalized_symmetric_group, is_symplectic, is_unitary, log_sigma_mgf, mo, np, pauli_group, random_symplectic, random_unitary, sigma_mgf, sigma_mgf_coefficients


@app.cell
def _(mo):
    mo.md(r"""
    # Sigma-MGF explorer

    For finite groups, inspect the expected sigma-MGF coefficients
    $\mathbb{E}[p_\lambda]/z_\lambda$.  The sampling modes retain the useful
    random-unitary and real-symplectic experiments from the original notebook,
    now with validity checks and corrected matrix construction.
    """)
    return


@app.cell
def _(mo):
    mode = mo.ui.dropdown(
        options=["Finite group expectation", "Random unitary", "Random symplectic"],
        value="Finite group expectation",
        label="mode",
    )
    family = mo.ui.dropdown(
        options=["Cyclic character", "Cyclic permutation", "Cyclic real 2D", "Dihedral", "Dicyclic", "Pauli", "Alternating", "Generalized symmetric"],
        value="Cyclic character",
        label="finite group",
    )
    n = mo.ui.slider(1, 7, value=3, label="n", show_value=True)
    k = mo.ui.slider(1, 7, value=1, label="k", show_value=True)
    level = mo.ui.slider(1, 3, value=2, label="level ℓ", show_value=True)
    max_degree = mo.ui.slider(0, 10, value=6, label="degree / max power", show_value=True)
    t = mo.ui.slider(0.05, 0.9, step=0.05, value=0.25, label="t", show_value=True)
    run = mo.ui.run_button(label="Evaluate")
    mo.vstack(
        [
            mo.hstack([mode, family, n, k, level], justify="start", gap=1),
            mo.hstack([max_degree, t, run], justify="start", gap=1),
        ]
    )
    return family, k, level, max_degree, mode, n, run, t


@app.cell
def _(
    alternating_group,
    coefficient_records,
    cyclic_character,
    cyclic_permutation_group,
    cyclic_real_2d,
    dicyclic_group,
    dihedral_group,
    display_number,
    family,
    generalized_symmetric_group,
    is_symplectic,
    is_unitary,
    k,
    level,
    log_sigma_mgf,
    max_degree,
    mo,
    mode,
    n,
    np,
    pauli_group,
    random_symplectic,
    random_unitary,
    run,
    sigma_mgf,
    sigma_mgf_coefficients,
    t,
):
    mo.stop(not run.value, mo.md("Choose an experiment, then press **Evaluate**."))

    if mode.value == "Finite group expectation":
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
            group = generalized_symmetric_group(min(4, n.value), level.value)
        coefficients = sigma_mgf_coefficients(group, max_degree.value)
        output = mo.vstack(
            [
                mo.md(f"## {group.name}\n\nOrder **{group.order}**, dimension **{group.dimension}**."),
                mo.ui.table(coefficient_records(coefficients), pagination=True),
            ]
        )
    else:
        if mode.value == "Random unitary":
            matrix = random_unitary(n.value)
            valid = is_unitary(matrix)
            group_label = "unitary"
        else:
            matrix = random_symplectic(n.value)
            valid = is_symplectic(matrix)
            group_label = "symplectic"
        powers = max(1, max_degree.value)
        log_value = log_sigma_mgf(matrix, t.value, powers)
        value = sigma_mgf(matrix, t.value, powers)
        matrix_text = np.array2string(matrix, precision=4, suppress_small=True)
        output = mo.vstack(
            [
                mo.callout(
                    f"The sampled matrix passes the {group_label} identity check: {valid}.",
                    kind="success" if valid else "danger",
                ),
                mo.md(
                    fr"""
                    $\log\Sigma({t.value})$ = **{display_number(log_value)}**  
                    $\Sigma({t.value})$ = **{display_number(value)}**

                    ```text
                    {matrix_text}
                    ```
                    """
                ),
            ]
        )
    output
    return


if __name__ == "__main__":
    app.run()
