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

    root = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/finite_group_sigma_mgf/permutation_actions/notebook.py").parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from lambda_distributions.proofs.finite_group_sigma_mgf.verification_core import (
        permutation_elements,
        representative_suites,
        subset_cycle_counts_formula,
        subset_matrices,
        verify_subset_case,
    )

    return (
        mo,
        permutation_elements,
        representative_suites,
        subset_cycle_counts_formula,
        subset_matrices,
        verify_subset_case,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # (S_n) and (A_n) acting on (k)-subsets

    The notebook constructs the induced permutation matrices on
    (inom{[n]}k).  It compares the Reynolds-projector rank with the proposed
    cycle formula.  The latter obtains fixed (k)-subsets as the coefficient of
    (prod_j(1+z^j)^{X_j(\sigma^r)}), then recovers induced cycle counts by
    Möbius inversion.  A separate determinant evaluation checks the untruncated
    rational function at safe numerical values.
    """)
    return


@app.cell
def _(mo):
    group = mo.ui.dropdown(options={"S_n": "symmetric", "A_n": "alternating"}, value="symmetric", label="group")
    n = mo.ui.slider(3, 6, value=5, label="n", show_value=True)
    k = mo.ui.slider(1, 3, value=2, label="k", show_value=True)
    tau_text = mo.ui.text(value="2, 1", label="tau")
    run = mo.ui.run_button(label="Construct matrices and verify")
    mo.hstack([group, n, k, tau_text, run], justify="start", gap=1)
    return group, k, n, run, tau_text


@app.cell
def _(
    group,
    k,
    mo,
    n,
    permutation_elements,
    run,
    subset_cycle_counts_formula,
    subset_matrices,
    tau_text,
    verify_subset_case,
):
    mo.stop(not run.value, mo.md("Set the parameters and press **Construct matrices and verify**."))
    selected_n = n.value
    selected_k = min(k.value, selected_n - 1)
    selected_tau = tuple(int(item.strip()) for item in tau_text.value.split(",") if item.strip())
    alternating = group.value == "alternating"
    result = verify_subset_case(selected_n, selected_k, selected_tau, alternating)
    matrices = subset_matrices(selected_n, selected_k, alternating)
    group_elements = permutation_elements(selected_n, alternating)
    sample_cycles = subset_cycle_counts_formula(group_elements[1], selected_k)
    sample_matrix = matrices[1]
    return result, sample_cycles, sample_matrix, selected_k, selected_n, selected_tau


@app.cell
def _(group, mo, result, sample_cycles, sample_matrix, selected_k, selected_n):
    mo.vstack([
        mo.callout(
            "Projector, character, fixed-point/Möbius, and determinant routes agree."
            if result.passed else "A route disagrees; inspect the diagnostics.",
            kind="success" if result.passed else "danger",
        ),
        mo.ui.table([result.as_dict()], pagination=False),
        mo.md(
            f"**Selected:** {group.value}, n={selected_n}, k={selected_k}. "
            f"Induced cycle counts for one group element: `{sample_cycles}`"
        ),
        mo.md("The corresponding subset-permutation matrix:"),
        mo.ui.table(sample_matrix.real.astype(int).tolist(), pagination=False),
    ])
    return


@app.cell
def _(mo, representative_suites):
    mo.vstack([
        mo.md("## Fixed representative sweep"),
        mo.ui.table(representative_suites()["subsets"], pagination=False),
        mo.callout(
            r"For n >= 2k, both [m_(2)] and [m_(1,1)] equal k+1; the regression suite checks k=1,2,3.",
            kind="info",
        ),
    ])
    return


if __name__ == "__main__":
    app.run()

