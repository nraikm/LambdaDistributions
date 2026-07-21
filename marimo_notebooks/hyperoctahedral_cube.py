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
    from math import factorial
    from pathlib import Path

    import marimo as mo

    root = (Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/finite_group_sigma_mgf/hyperoctahedral_cube/notebook.py").parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from for_this_guy.finite_group_sigma_mgf.hyperoctahedral_cube.verification import (
        action_cycle_counts,
        action_permutation,
        group_elements,
        permutation_matrix,
        run_suite,
        signed_cycle_type,
    )

    return (
        action_cycle_counts,
        action_permutation,
        factorial,
        group_elements,
        mo,
        permutation_matrix,
        run_suite,
        signed_cycle_type,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # The hyperoctahedral group on cube vertices

    For (B_n=C_2^n\rtimes S_n) and (X_n=\mathbb F_2^n), this laboratory
    constructs every (2^n\times2^n) permutation matrix of
    
    \[
    (b,\pi)\cdot x=b+\pi x.
    \]

    It checks the proposed formulas in three genuinely separate ways:

    1. literal matrix traces and determinant averages;
    2. direct orbit traversal on the monomial bases of
       (\bigotimes_j\operatorname{Sym}^{\tau_j}(\mathbb C[X_n]));
    3. the signed-coordinate-cycle formula followed by Möbius inversion.

    The exhaustive suite uses (1\le n\le5), reaching a group of order
    (2^5 5!=3840) represented by (32\times32) matrices.
    """)
    return


@app.cell
def _(run_suite):
    suite = run_suite()
    return (suite,)


@app.cell
def _(mo, suite):
    summary_rows = [
        {
            "n": result["n"],
            "|B_n|": result["group order"],
            "matrix size": f"{result['matrix dimension']} x {result['matrix dimension']}",
            "Pr(F_n > 0)": result["positive probability"],
            "fixed law": result["conditional fixed law"],
            "cycle formula": result["cycle reconstruction"],
            "max numeric error": f"{result['numeric error']:.2e}",
            "passed": result["passed"],
        }
        for result in suite["results"]
    ]
    mo.vstack(
        [
            mo.callout(
                f"{'PASS' if suite['passed'] else 'FAIL'}: all exhaustive matrix, orbit, and signed-type checks.",
                kind="success" if suite["passed"] else "danger",
            ),
            mo.ui.table(summary_rows, pagination=False),
        ]
    )
    return


@app.cell
def _(mo):
    selected_n = mo.ui.slider(1, 5, value=3, label="cube dimension n", show_value=True)
    selected_n
    return (selected_n,)


@app.cell
def _(factorial, mo, selected_n):
    selected_index = mo.ui.slider(
        0,
        2 ** selected_n.value * factorial(selected_n.value) - 1,
        value=0,
        label="group-element index",
        show_value=True,
    )
    selected_index
    return (selected_index,)


@app.cell
def _(
    action_cycle_counts,
    action_permutation,
    group_elements,
    mo,
    permutation_matrix,
    selected_index,
    selected_n,
    signed_cycle_type,
):
    n = selected_n.value
    b, pi = group_elements(n)[selected_index.value]
    action = action_permutation(n, b, pi)
    matrix = permutation_matrix(action)
    positive, negative = signed_cycle_type(b, pi)
    nonzero_entries = [{"column x": source, "row g.x": target} for source, target in enumerate(action)]
    block_size = min(12, len(matrix))
    dense_rows = [
        {"row": row, **{str(column): int(matrix[row, column]) for column in range(block_size)}}
        for row in range(block_size)
    ]
    mo.vstack(
        [
            mo.md(
                rf"""
                ## One literal representation matrix

                (b={b}), (\pi={pi}).  The signed cycle counts are
                (a_\ell={positive[1:]}) and (b_\ell={negative[1:]}).
                The induced vertex-cycle counts are `{dict(action_cycle_counts(action))}`.

                The table gives the unique (1) in each column of the
                (2^{n}\times2^{n}) matrix (P_g); its dense top-left block is shown below.
                """
            ),
            mo.hstack(
                [
                    mo.ui.table(nonzero_entries, pagination=True, page_size=10),
                    mo.ui.table(dense_rows, pagination=True, page_size=12),
                ],
                widths=[1, 2],
                gap=1,
            ),
        ]
    )
    return


@app.cell
def _(mo, selected_n, suite):
    result = suite["results"][selected_n.value - 1]
    mo.vstack(
        [
            mo.md(r"""
            ## Exact coefficient comparison

            For each partition (\tau), the displayed number is
            
            \[
            [m_\tau]\,\mathcal H_n
            =\dim\left(\bigotimes_j\operatorname{Sym}^{\tau_j}(V_n)\right)^{B_n}.
            \]
            """),
            mo.ui.table(list(result["coefficient rows"]), pagination=False),
            mo.md("## Fixed-point moments"),
            mo.ui.table(list(result["moment rows"]), pagination=False),
            mo.md(
                f"At `(t1,t2)=(0.037,0.061)`, the full determinant average is "
                f"`{result['numeric matrix']:.15f}`, the vertex-cycle product is "
                f"`{result['numeric cycle']:.15f}`, and the signed-type sum is "
                f"`{result['numeric signed type']:.15f}`."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
