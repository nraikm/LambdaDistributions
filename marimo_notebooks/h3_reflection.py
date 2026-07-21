import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import sys
    repo_root = str((Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/matrix_group_formula_verification/h3_reflection/notebook.py").parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from for_this_guy.matrix_group_formula_verification.h3_reflection.verification import (
        run_polyhedral_suite,
        run_suite,
    )
    return mo, run_polyhedral_suite, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Polyhedral rotation groups and the H3 reflection representation

    The tetrahedral, octahedral, and icosahedral groups are constructed as
    explicit real $3\times3$ matrices. Complete homogeneous characters are
    averaged over every matrix and compared with the angle-class formulas.
    The full order-120 $H_3$ reflection group is checked separately against
    its ten spectral classes and its one-variable invariant degrees 2, 6, 10.
    """)
    return


@app.cell
def _(run_polyhedral_suite, run_suite):
    rotations = run_polyhedral_suite()
    h3 = run_suite(10)
    return h3, rotations


@app.cell
def _(h3, mo, rotations):
    rotation_rows = [
        {"group": result["group"], **row}
        for result in rotations
        for row in result["rows"]
    ]
    passed = h3["passed"] and all(result["passed"] for result in rotations)
    mo.vstack(
        [
            mo.md(f"## {'PASS' if passed else 'FAIL'}"),
            mo.md("### Rotation-group class formulas"),
            mo.ui.table(rotation_rows, pagination=True, page_size=20),
            mo.md("### H3 diagnostics"),
            mo.json(h3["diagnostics"]),
            mo.md("### H3 multivariate checks"),
            mo.ui.table(list(h3["coefficient checks"]), pagination=True, page_size=20),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
