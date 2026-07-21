# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.23.8",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    from dataclasses import asdict
    from pathlib import Path
    import sys

    import marimo as mo

    source_directory = (Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/generalized_molien_affine_wreath/affine_groups/notebook.py").parent
    if str(source_directory) not in sys.path:
        sys.path.insert(0, str(source_directory))
    from verification import affine_action, permutation_matrix, verify_fixed_point_moment, verify_orbits

    return (
        affine_action,
        asdict,
        mo,
        permutation_matrix,
        verify_fixed_point_moment,
        verify_orbits,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Finite affine groups: direct orbit test of the generalized Molien formula

    For (G=AGL_n(q)) acting on (S=\mathbb F_q^n), this notebook compares

    \[
    \#\left(G\backslash\prod_j \operatorname{Mult}_{\tau_j}(S)\right)
    \quad\text{with}\quad
    \frac1{|G|}\sum_{g\in G}\prod_j
    [z^{\tau_j}]\prod_{C\in\operatorname{Cyc}(g)}(1-z^{|C|})^{-1}.
    \]

    The left side is obtained by literally constructing every multiset tuple,
    moving it by every affine permutation, and counting canonical orbit
    representatives.  The right side is the coefficient of the proposed
    cycle-index/Molien formula.  The implementation also constructs the
    (q^n\times q^n) permutation matrices.
    """)
    return


@app.cell
def _(mo):
    group_case = mo.ui.dropdown(
        options={
            "AGL_1(2), degree 2": (1, 2),
            "AGL_2(2), degree 4": (2, 2),
            "AGL_3(2), degree 8": (3, 2),
            "AGL_1(3), degree 3": (1, 3),
            "AGL_2(3), degree 9": (2, 3),
            "AGL_1(5), degree 5": (1, 5),
        },
        value="AGL_2(2), degree 4",
        label="affine action",
    )
    tau_text = mo.ui.text(value="2,1", label="partition tau")
    moment_power = mo.ui.slider(1, 5, value=3, label="fixed-point moment", show_value=True)
    run = mo.ui.run_button(label="Construct matrices and verify")
    mo.hstack([group_case, tau_text, moment_power, run], justify="start", gap=1)
    return group_case, moment_power, run, tau_text


@app.cell
def _(
    affine_action,
    group_case,
    mo,
    moment_power,
    permutation_matrix,
    run,
    tau_text,
    verify_fixed_point_moment,
    verify_orbits,
):
    mo.stop(not run.value, mo.md("Choose a representative case and press **Construct matrices and verify**."))
    try:
        tau = tuple(int(part.strip()) for part in tau_text.value.split(",") if part.strip())
        if not tau or any(part <= 0 for part in tau) or sum(tau) > 4:
            raise ValueError
    except ValueError:
        mo.stop(True, mo.callout("Enter positive comma-separated parts of total size at most 4.", kind="danger"))

    n, q = group_case.value
    action = affine_action(n, q)
    orbit_result = verify_orbits(n, q, tau)
    moment_result = verify_fixed_point_moment(n, q, moment_power.value)
    sample = permutation_matrix(action.permutations[-1])
    sample_text = "\n".join(" ".join(str(entry) for entry in row) for row in sample)
    return action, moment_result, orbit_result, sample_text, tau


@app.cell
def _(asdict, mo, moment_result, orbit_result, sample_text):
    passed = orbit_result.passed and moment_result.passed
    mo.vstack(
        [
            mo.callout(
                "Both independent comparisons passed." if passed else "A discrepancy was found.",
                kind="success" if passed else "danger",
            ),
            mo.ui.table([asdict(orbit_result)]),
            mo.ui.table([asdict(moment_result)]),
            mo.md(f"**One explicitly constructed permutation matrix**\n```text\n{sample_text}\n```"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
