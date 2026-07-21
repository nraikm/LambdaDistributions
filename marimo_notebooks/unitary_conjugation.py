# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo>=0.23.8",
#   "numpy>=2.0",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np

    notebook_directory = (Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/unitary_conjugation_endomorphisms/unitary_conjugation_verification.py").parent
    if str(notebook_directory) not in sys.path:
        sys.path.insert(0, str(notebook_directory))

    from unitary_conjugation_core import (
        conjugation_matrix,
        direct_invariant_dimension,
        finite_rank_lr_coefficient,
        matrix_group_diagnostics,
        necklace_euler_coefficient,
        run_representative_checks,
        stable_orbit_count,
        target_dimension,
        unitary_generator_matrices,
    )

    return (
        conjugation_matrix,
        direct_invariant_dimension,
        finite_rank_lr_coefficient,
        matrix_group_diagnostics,
        mo,
        necklace_euler_coefficient,
        np,
        run_representative_checks,
        stable_orbit_count,
        target_dimension,
        unitary_generator_matrices,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # (U(n)) conjugation on (operatorname{End}(mathbb C^n))

    This notebook tests the finite-rank and stable formulas for
    (W=operatorname{End}(mathbb C^n)), with (gcdot A=gAg^{-1}).
    If (g) is unitary, the conjugation matrix in column-major vectorization is

    \[
      \rho_W(g)=\overline g\otimes g,
      \qquad
      \operatorname{Tr}(\rho_W(g)^k)=|\operatorname{Tr}(g^k)|^2.
    \]

    For \(\tau=(\tau_1,\ldots,\tau_s)\), the target coefficient is

    \[
      \dim\left(\bigotimes_a\operatorname{Sym}^{\tau_a}W\right)^{U(n)}.
    \]

    The checks below use four genuinely independent descriptions.

    1. **Direct matrix target:** construct the derived \(\mathfrak u(n)\)-action
       on the full tensor product and compute its common kernel.
    2. **Finite-rank formula:** enumerate partitions and
       Littlewood--Richardson tableaux in equation (6).
    3. **Stable contractions:** count \(H_\tau\)-conjugacy orbits in \(S_d\).
    4. **Stable Euler product:** extract the same coefficient from the product
       over colored necklaces in equation (9).

    Since (U(n)) is connected, the common kernel of the derived action is
    exactly the (U(n))-fixed space; no random sampling or numerical
    integration is used.
    """)
    return


@app.cell
def _(mo):
    n_control = mo.ui.slider(1, 4, value=3, step=1, label="rank n")
    tau_control = mo.ui.text(value="2,1", label="tau")
    mo.hstack([n_control, tau_control], justify="start")
    return n_control, tau_control


@app.cell
def _(mo, n_control, target_dimension, tau_control):
    try:
        selected_tau = tuple(
            int(piece.strip()) for piece in tau_control.value.split(",") if piece.strip()
        )
        if not selected_tau or any(part < 1 for part in selected_tau):
            raise ValueError("tau must contain positive integers")
        selected_n = int(n_control.value)
        selected_dimension = target_dimension(selected_n, selected_tau)
        selection_message = mo.md(
            f"Selected target: $n={selected_n}$, $\\tau={selected_tau}$, "
            f"full matrix dimension **{selected_dimension}**."
        )
    except ValueError as error:
        selected_n, selected_tau, selected_dimension = 1, (1,), 1
        selection_message = mo.callout(str(error), kind="danger")
    selection_message
    return selected_dimension, selected_n, selected_tau


@app.cell
def _(
    direct_invariant_dimension,
    finite_rank_lr_coefficient,
    mo,
    necklace_euler_coefficient,
    selected_dimension,
    selected_n,
    selected_tau,
    stable_orbit_count,
):
    if selected_dimension <= 800:
        direct_result = direct_invariant_dimension(
            selected_n, selected_tau, dimension_cap=800
        )
        finite_result = finite_rank_lr_coefficient(selected_n, selected_tau)
        interactive_row = {
            "n": selected_n,
            "tau": str(selected_tau),
            "full target dimension": selected_dimension,
            "direct common-kernel dimension": direct_result["invariant_dimension"],
            "finite-rank LR formula": finite_result,
            "finite match": direct_result["invariant_dimension"] == finite_result,
        }
        if selected_n >= sum(selected_tau):
            interactive_row["stable orbit count"] = stable_orbit_count(selected_tau)
            interactive_row["necklace Euler coefficient"] = necklace_euler_coefficient(
                selected_tau
            )
        interactive_output = mo.ui.table([interactive_row], selection=None)
    else:
        interactive_output = mo.callout(
            f"The selected full target dimension is {selected_dimension}; "
            "raise the notebook's safety cap deliberately if you want this calculation.",
            kind="warn",
        )
    interactive_output
    return


@app.cell
def _(mo, run_representative_checks):
    verification_rows, group_rows = run_representative_checks()
    batch_table_rows = [
        {
            "n": row["n"],
            "tau": str(row["tau"]),
            "target dim": row["target_dimension"],
            "direct": row["direct"],
            "finite LR": row["finite_lr"],
            "orbits": "-" if row["stable_orbits"] is None else row["stable_orbits"],
            "necklaces": "-" if row["necklace_euler"] is None else row["necklace_euler"],
            "pass": row["finite_match"] and row["stable_match"] in (None, True),
        }
        for row in verification_rows
    ]
    mo.vstack(
        [
            mo.md("## Representative finite-rank and stable checks"),
            mo.ui.table(batch_table_rows, selection=None, pagination=True, page_size=18),
        ]
    )
    return group_rows, verification_rows


@app.cell
def _(group_rows, mo):
    diagnostic_rows = [
        {
            "n": row["n"],
            "generators": row["group_generators"],
            "Ad dimension": row["ad_dimension"],
            "unitarity error": f'{row["unitarity_error"]:.2e}',
            "homomorphism error": f'{row["homomorphism_error"]:.2e}',
            "trace identity error": f'{row["trace_identity_error"]:.2e}',
        }
        for row in group_rows
    ]
    mo.vstack(
        [
            mo.md("## Constructed matrix-group diagnostics"),
            mo.ui.table(diagnostic_rows, selection=None),
        ]
    )
    return


@app.cell
def _(
    conjugation_matrix,
    matrix_group_diagnostics,
    mo,
    np,
    selected_n,
    unitary_generator_matrices,
):
    selected_generators = unitary_generator_matrices(selected_n)
    sample_g = selected_generators[0]
    sample_ad = conjugation_matrix(sample_g)
    sample_diagnostics = matrix_group_diagnostics(selected_n)
    with np.printoptions(precision=3, suppress=True):
        matrix_display = mo.md(
            "## One explicit generator and its conjugation matrix\n\n"
            f"For the selected rank, one unitary generator is\n\n"
            f"```text\n{sample_g}\n```\n\n"
            f"and its {sample_ad.shape[0]} by {sample_ad.shape[1]} matrix "
            f"$\\overline g\\otimes g$ is\n\n"
            f"```text\n{sample_ad}\n```\n\n"
            f"Maximum trace-identity error through $k=4$: "
            f"`{sample_diagnostics['trace_identity_error']:.3e}`."
        )
    matrix_display
    return


@app.cell
def _(mo, verification_rows):
    all_passed = all(row["finite_match"] for row in verification_rows) and all(
        row["stable_match"] in (None, True) for row in verification_rows
    )
    mo.callout(
        f"All {len(verification_rows)} representative comparisons passed."
        if all_passed
        else "At least one comparison failed; inspect the table above.",
        kind="success" if all_passed else "danger",
    )
    return


if __name__ == "__main__":
    app.run()

