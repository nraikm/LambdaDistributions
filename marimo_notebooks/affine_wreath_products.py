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
    from pathlib import Path
    import sys

    import marimo as mo

    source_directory = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/generalized_molien_affine_wreath/wreath_products/notebook.py").parent
    if str(source_directory) not in sys.path:
        sys.path.insert(0, str(source_directory))
    from verification import BASE_GROUPS, verify_case, verify_cycle_product_lemma, wreath_group

    return BASE_GROUPS, mo, verify_case, verify_cycle_product_lemma, wreath_group


@app.cell
def _(mo):
    mo.md(r"""
    # (H\wr S_n): exact matrices, cycle products, and the stable limit

    The direct side constructs all block-monomial matrices on (V^{\oplus n})
    and averages

    \[
      \prod_j\chi_{\operatorname{Sym}^{\tau_j}V^{\oplus n}}(g).
    \]

    The independent finite formula is extracted from

    \[
    \sum_{n\ge0}\mathcal W_{H,n}(\mathbf t)u^n
      =\exp\!\left(\sum_{r\ge1}\frac{u^r}{r}
      \mathcal S_H(t_1^r,t_2^r,\ldots)\right),
    \]

    and a second recurrence extracts the proposed stable series
    (\operatorname{Exp}_\sigma(\mathcal S_H-1)) directly.  Exact rational
    arithmetic is used throughout; no eigenvalue rounding is involved.
    """)
    return


@app.cell
def _(mo):
    base_choice = mo.ui.dropdown(
        options=["C2 sign", "C3 rational 2D"],
        value="C2 sign",
        label="base representation H -> GL(V)",
    )
    block_count = mo.ui.slider(1, 4, value=3, label="blocks n", show_value=True)
    maximum_degree = mo.ui.slider(1, 6, value=5, label="maximum degree", show_value=True)
    run = mo.ui.run_button(label="Construct wreath group and verify")
    mo.hstack([base_choice, block_count, maximum_degree, run], justify="start", gap=1)
    return base_choice, block_count, maximum_degree, run


@app.cell
def _(
    BASE_GROUPS,
    base_choice,
    block_count,
    maximum_degree,
    mo,
    run,
    verify_case,
    verify_cycle_product_lemma,
    wreath_group,
):
    mo.stop(not run.value, mo.md("Choose a representation and press **Construct wreath group and verify**."))
    base = BASE_GROUPS[base_choice.value]()
    rows = verify_case(base, block_count.value, maximum_degree.value)
    lemma_passed = verify_cycle_product_lemma(base)
    matrices = wreath_group(base, block_count.value)
    sample = matrices[-1]
    sample_text = "\n".join(" ".join(f"{entry:2d}" for entry in row) for row in sample)
    table_rows = [
        {
            "tau": row.tau,
            "degree": sum(row.tau),
            "direct matrix average": row.direct_matrix_average,
            "finite cycle formula": row.cycle_product_formula,
            "stable formula (degree <= n)": row.stable_formula,
            "pass": row.passed,
        }
        for row in rows
    ]
    return base, lemma_passed, matrices, rows, sample_text, table_rows


@app.cell
def _(base, lemma_passed, matrices, mo, rows, sample_text, table_rows):
    passed = lemma_passed and all(row.passed for row in rows)
    mo.vstack(
        [
            mo.callout(
                f"{'All checks passed' if passed else 'A discrepancy was found'} for {base.name}. "
                f"Constructed {len(matrices)} exact matrices; cycle-product lemma: {lemma_passed}.",
                kind="success" if passed else "danger",
            ),
            mo.ui.table(table_rows, pagination=True),
            mo.md(f"**One block-monomial matrix**\n```text\n{sample_text}\n```"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
