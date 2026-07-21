import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import sys

    import marimo as mo

    repo_root = str((Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/power_character_bounded_flag_stability/notebook.py").parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from proved_matrix_groups.packages.power_character_bounded_flag_stability.verification import (
        run_suite,
    )

    return mo, run_suite


@app.cell
def _(mo):
    mo.md(r"""
    # Power characters, parabolic cycles, conjugation, and bounded flags

    This notebook constructs each finite group or generating set explicitly.
    The panels are separated by representation type so that a passing
    permutation check cannot mask a failing linear-representation check.

    - **Linear/Steinberg example:** the integral two-dimensional standard
      representation of $S_3\cong GL_2(2)$.
    - **Parabolic example:** $GL_3(2)$ on $G/P=\mathbf P^2(\mathbf F_2)$.
    - **Conjugation example:** $S_3$ acting on itself.
    - **Stable bounded-support examples:** projective points through rank four
      and $2$-planes in ranks four and five.
    """)
    return


@app.cell
def _(run_suite):
    suite = run_suite()
    return (suite,)


@app.cell
def _(mo, suite):
    mo.vstack(
        [
            mo.md(f"## {'PASS' if suite['passed'] else 'FAIL'}: universal power-character formula"),
            mo.ui.table(suite["power-character"]),
            mo.md("## Parabolic quotient $GL_3(2)/P$"),
            mo.json(suite["parabolic diagnostics"]),
            mo.ui.table(suite["parabolic"]),
            mo.md("## Conjugation action"),
            mo.json(suite["conjugation diagnostics"]),
            mo.ui.table(suite["conjugation"]),
            mo.md("## Fixed-$q$, growing-rank bounded-support checks"),
            mo.ui.table(suite["stability"]),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
