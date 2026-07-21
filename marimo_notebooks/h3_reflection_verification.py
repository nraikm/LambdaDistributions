# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.8", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import sys

    import marimo as mo

    folder = (Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/matrix_group_formula_verification/h3_reflection/h3_reflection_verification.py").parent
    shared = folder.parent
    repository_root = folder.parents[2]
    for path in (repository_root, shared, folder):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from verification import h3_group, run_suite, spectral_classes

    return h3_group, mo, run_suite, spectral_classes


@app.cell
def _(mo):
    mo.md(r"""
    # The (H_3) reflection group

    This notebook reconstructs the 120 matrices from the symmetries of an
    icosahedron.  It compares direct matrix averages with the five rotation
    spectra (and their negatives) in the proposed class formula, and separately
    checks the ordinary Molien denominator ((1-t^2)(1-t^6)(1-t^{10})).
    """)
    return


@app.cell
def _(mo):
    maximum_degree = mo.ui.slider(0, 12, value=8, show_value=True, label="maximum degree")
    run = mo.ui.run_button(label="Build H3 and verify")
    mo.hstack([maximum_degree, run], justify="start", gap=1)
    return maximum_degree, run


@app.cell
def _(maximum_degree, mo, run, run_suite):
    mo.stop(not run.value, mo.md("Press **Build H3 and verify** to run the checks."))
    result = run_suite(maximum_degree.value)
    return result


@app.cell
def _(mo, result):
    mo.vstack(
        [
            mo.callout(
                "All H3 checks passed." if result["passed"] else "An H3 check failed.",
                kind="success" if result["passed"] else "danger",
            ),
            mo.md(f"**Matrix-group diagnostics:** `{result['diagnostics']}`"),
            mo.md(f"**Reynolds projector:** `{result['projector']}`"),
            mo.md("## Multivariate coefficients"),
            mo.ui.table(list(result["coefficient checks"]), pagination=True),
            mo.md("## One-variable Molien check"),
            mo.ui.table(list(result["one-variable checks"]), pagination=True),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
