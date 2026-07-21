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

    root = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/finite_group_exact_formulas/symmetric/symmetric_verification.py").parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from verification_core import (
        character_average,
        permutation_cycle_formula,
        permutation_matrices,
        representative_sweeps,
        reynolds_check,
        vector_partition_counts,
    )

    return (
        character_average,
        mo,
        permutation_cycle_formula,
        permutation_matrices,
        representative_sweeps,
        reynolds_check,
        vector_partition_counts,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Exact finite-\(n\) verification for \(S_n\)

    The permutation matrices of every element of \(S_n\) are constructed.
    The rank of the Reynolds projector on
    \(\bigotimes_b\mathrm{Sym}^{\tau_b}(\mathbb C^n)\) is compared with the
    cycle-type Molien formula and the vector-partition orbit count.
    """)
    return


@app.cell
def _(mo):
    n = mo.ui.slider(2, 6, value=4, label="n", show_value=True)
    tau_text = mo.ui.text(value="2, 1", label="tau")
    run = mo.ui.run_button(label="Construct and verify")
    mo.hstack([n, tau_text, run], justify="start", gap=1)
    return n, run, tau_text


@app.cell
def _(
    character_average,
    n,
    mo,
    permutation_cycle_formula,
    permutation_matrices,
    reynolds_check,
    run,
    tau_text,
    vector_partition_counts,
):
    mo.stop(not run.value, mo.md("Choose parameters and press **Construct and verify**."))
    tau = tuple(int(item.strip()) for item in tau_text.value.split(",") if item.strip())
    matrices = permutation_matrices(n.value)
    direct = reynolds_check(matrices, tau)
    character = character_average(matrices, tau)
    cycle = permutation_cycle_formula(n.value, tau)
    orbit, _ = vector_partition_counts(tau, n.value)
    passed = direct["projector_rank"] == cycle == orbit and abs(character - cycle) < 1e-8
    return character, cycle, direct, orbit, passed, tau


@app.cell
def _(character, cycle, direct, n, mo, orbit, passed, tau):
    mo.vstack([
        mo.callout(
            "Projector, character, cycle-type, and orbit routes agree."
            if passed else "A route disagrees; inspect the values below.",
            kind="success" if passed else "danger",
        ),
        mo.ui.table([{
            "group": f"S_{n.value}",
            "tau": str(tau),
            "target dimension": direct["target_dimension"],
            "projector rank": direct["projector_rank"],
            "character average": round(character.real, 10),
            "cycle-type formula": cycle,
            "orbit formula": orbit,
            "projector error": direct["idempotence_error"],
        }]),
    ])
    return


@app.cell
def _(mo, representative_sweeps):
    mo.vstack([
        mo.md("## Built-in representative sweep"),
        mo.ui.table(representative_sweeps()["symmetric"], pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()
