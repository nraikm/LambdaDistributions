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

    repository = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/generalized_molien_extensions/notebook.py").parents[2]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))

    from lambda_distributions.proofs.generalized_molien_extensions.code_monomial.verification import (
        run_suite as run_code,
    )
    from lambda_distributions.proofs.generalized_molien_extensions.delta_su3.verification import (
        run_suite as run_delta,
    )
    from lambda_distributions.proofs.generalized_molien_extensions.psl2_projective_line.verification import (
        run_suite as run_psl2,
    )

    return mo, run_code, run_delta, run_psl2


@app.cell
def _(mo):
    mo.md(r"""
    # Generalized Molien formula laboratory

    This notebook constructs each representation, averages
    \(\prod_j\det(I-t_jg)^{-1}\) or its exact coefficients directly, and
    compares the result with the proposed group-specific formula.

    The three panels are intentionally independent: code orthogonality,
    monomial \(SU(3)\) cosets, and projective-line cycle types use different
    mathematical inputs and different verification routes.
    """)
    return


@app.cell
def _(mo):
    code_degree = mo.ui.slider(2, 6, value=5, label="code maximum degree", show_value=True)
    delta_degree = mo.ui.slider(2, 7, value=6, label="Delta maximum degree", show_value=True)
    psl_degree = mo.ui.slider(2, 4, value=4, label="PSL2 maximum degree", show_value=True)
    run = mo.ui.run_button(label="Construct groups and verify formulas")
    mo.hstack([code_degree, delta_degree, psl_degree, run], justify="start", gap=1)
    return code_degree, delta_degree, psl_degree, run


@app.cell
def _(code_degree, delta_degree, mo, psl_degree, run, run_code, run_delta, run_psl2):
    mo.stop(not run.value, mo.callout("Choose bounds, then run the verification suites.", kind="info"))
    code = run_code(max_degree=code_degree.value)
    delta = run_delta(max_degree=delta_degree.value)
    psl2 = run_psl2(max_degree=psl_degree.value)
    return code, delta, psl2


@app.cell
def _(code, mo):
    mo.vstack(
        [
            mo.md("## Code-monomial groups"),
            mo.callout(
                f"{'PASS' if code['passed'] else 'FAIL'}: {len(code['rows'])} coefficient comparisons.",
                kind="success" if code["passed"] else "danger",
            ),
            mo.ui.table(code["numerical"], pagination=False),
            mo.ui.table(code["rows"], pagination=True, page_size=15),
        ]
    )
    return


@app.cell
def _(delta, mo):
    mo.vstack(
        [
            mo.md(r"## \(\Delta(3n^2)\) and \(\Delta(6n^2)\)"),
            mo.callout(
                f"{'PASS' if delta['passed'] else 'FAIL'}: {len(delta['rows'])} coefficient comparisons.",
                kind="success" if delta["passed"] else "danger",
            ),
            mo.ui.table(delta["numerical"], pagination=True, page_size=8),
            mo.ui.table(delta["rows"], pagination=True, page_size=15),
        ]
    )
    return


@app.cell
def _(mo, psl2):
    mo.vstack(
        [
            mo.md(r"## \(PSL_2(q)\) on \(\mathbf P^1(\mathbf F_q)\)"),
            mo.callout(
                f"{'PASS' if psl2['passed'] else 'FAIL'}: {len(psl2['rows'])} orbit/coefficient comparisons.",
                kind="success" if psl2["passed"] else "danger",
            ),
            mo.ui.table(psl2["cycle types"], pagination=False),
            mo.ui.table(psl2["numerical"], pagination=False),
            mo.ui.table(psl2["rows"], pagination=True, page_size=15),
        ]
    )
    return


if __name__ == "__main__":
    app.run()

