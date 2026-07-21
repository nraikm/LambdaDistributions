import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import sys
    repo_root = str((Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/matrix_group_formula_verification/classical_compact/notebook.py").parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from for_this_guy.matrix_group_formula_verification.classical_compact.verification import run_suite
    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Defining representations of SU(n), SO(n), and O(n)

    For each displayed partition $\tau$, the left side is computed as the
    common kernel of the explicit derived matrix generators on
    $W_\tau=\bigotimes_i\operatorname{Sym}^{\tau_i}(\mathbb C^n)$.
    The right side is computed independently by enumerating semistandard
    tableaux (Kostka numbers). A reflection is added for $O(n)$.
    """)
    return


@app.cell
def _(run_suite):
    suite = run_suite()
    suite
    return (suite,)


@app.cell
def _(mo, suite):
    status = "PASS" if suite["passed"] else "FAIL"
    mo.vstack(
        [
            mo.md(f"## {status}: {suite['checks']} coefficient checks"),
            mo.md(f"Largest directly tested target space: **{suite['largest target']} dimensions**."),
            mo.ui.table(list(suite["rows"]), pagination=True, page_size=20),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
