# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.8", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    root = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/finite_group_exact_formulas/dicyclic/dicyclic_verification.py").parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from verification_core import (
        character_average,
        dicyclic_relation_error,
        dicyclic_representation,
        dicyclic_single_formula,
        dicyclic_spectral_formula,
        representative_sweeps,
        reynolds_check,
    )

    return (
        character_average,
        dicyclic_relation_error,
        dicyclic_representation,
        dicyclic_single_formula,
        dicyclic_spectral_formula,
        mo,
        representative_sweeps,
        reynolds_check,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Dicyclic / generalized quaternion verification

    This notebook constructs the matrices of
    \(\mathrm{Dic}_m=\langle a,x:a^{2m}=1,\ x^2=a^m,\ xax^{-1}=a^{-1}\rangle\).
    For a selected \(\rho_k\) and partition \(\tau\), it independently compares:

    1. the rank of the explicit Reynolds projector on
       \(\bigotimes_b\mathrm{Sym}^{\tau_b}V\);
    2. the direct character average;
    3. the proposed cyclic-weight plus coset coefficient formula; and
    4. coefficient extraction from the displayed spectral Molien expression.
    """)
    return


@app.cell
def _(mo):
    m = mo.ui.slider(2, 8, value=4, label="m", show_value=True)
    k = mo.ui.slider(1, 7, value=1, label="k (clipped to m-1)", show_value=True)
    tau_text = mo.ui.text(value="2, 2", label="tau")
    run = mo.ui.run_button(label="Construct and verify")
    mo.hstack([m, k, tau_text, run], justify="start", gap=1)
    return k, m, run, tau_text


@app.cell
def _(
    character_average,
    dicyclic_relation_error,
    dicyclic_representation,
    dicyclic_single_formula,
    dicyclic_spectral_formula,
    k,
    m,
    mo,
    reynolds_check,
    run,
    tau_text,
):
    mo.stop(not run.value, mo.md("Choose parameters and press **Construct and verify**."))
    tau = tuple(int(item.strip()) for item in tau_text.value.split(",") if item.strip())
    selected_k = min(k.value, m.value - 1)
    matrices = dicyclic_representation(m.value, {selected_k: 1})
    direct = reynolds_check(matrices, tau)
    character = character_average(matrices, tau)
    coefficient = dicyclic_single_formula(m.value, selected_k, tau)
    spectral = dicyclic_spectral_formula(m.value, {selected_k: 1}, (), tau)
    relation_error = dicyclic_relation_error(m.value, {selected_k: 1})
    passed = (
        direct["projector_rank"] == coefficient
        and abs(character - coefficient) < 1e-8
        and abs(spectral - coefficient) < 1e-8
        and relation_error < 1e-8
    )
    return character, coefficient, direct, passed, relation_error, selected_k, spectral, tau


@app.cell
def _(character, coefficient, direct, m, mo, passed, relation_error, selected_k, spectral, tau):
    mo.vstack([
        mo.callout(
            "All four routes agree." if passed else "A route disagrees; inspect the values below.",
            kind="success" if passed else "danger",
        ),
        mo.ui.table([{
            "representation": f"Dic_{m.value}, rho_{selected_k}",
            "tau": str(tau),
            "target dimension": direct["target_dimension"],
            "projector rank": direct["projector_rank"],
            "character average": round(character.real, 10),
            "coefficient formula": coefficient,
            "spectral formula": round(spectral.real, 10),
            "relation error": relation_error,
            "projector error": direct["idempotence_error"],
        }]),
    ])
    return


@app.cell
def _(mo, representative_sweeps):
    rows = representative_sweeps()["dicyclic"]
    mo.vstack([
        mo.md("## Built-in representative sweep"),
        mo.ui.table(rows, pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()
