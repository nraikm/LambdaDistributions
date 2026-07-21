# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.8"]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import sys

    import marimo as mo

    folder = (Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/proctor_odd_symplectic_group/notebook.py").parent
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    from verification import DEFAULT_CASES, FiniteCase, run_all

    return DEFAULT_CASES, FiniteCase, mo, run_all


@app.cell
def _(mo):
    mo.md(r"""
    # Proctor's odd symplectic group: proof-driven verification

    Let $V=W\oplus L$, $\dim W=2n$, and let the alternating form be
    nondegenerate on $W$ with radical $L$. Every stabilizer matrix is

    \[
    g=\begin{pmatrix}A&0\\ \ell&a\end{pmatrix},\qquad
    A\in Sp_{2n},\quad \ell\in W^*,\quad a\ne0.
    \]

    The full complex group is noncompact and nonreductive, so a normalized
    Haar expectation is **undefined**. This notebook never substitutes a
    finite truncation for that nonexistent probability. It tests three valid
    alternatives:

    1. exact maximal-compact constant terms in the $n=1$ case;
    2. the algebraic invariant series, certified to be $1$;
    3. finite odd symplectic groups over $\mathbb F_q$, using their
       genuine complex permutation representations on projective points.

    For the finite version, two independent computations are compared:

    \[
    \frac1{|G|}\sum_g\prod_j h_{\tau_j}(P_g)
    \quad\text{and}\quad
    \frac1{|G|}\sum_g\prod_j[t^{\tau_j}]
      \prod_{C\in\mathrm{Cycles}(g)}(1-t^{|C|})^{-1}.
    \]

    The first route constructs each permutation matrix implicitly and uses
    traces of its powers plus Newton's recurrence. The second uses only cycle
    lengths. Selected values are checked a third way by explicit orbit
    enumeration on products of multiset spaces.
    """)
    return


@app.cell
def _(mo):
    preset = mo.ui.dropdown(
        options={
            "Full representative suite (recommended)": "full",
            "Quick: n=1, q=2": "quick",
            "Custom prime-field case": "custom",
        },
        value="full",
        label="verification suite",
    )
    n = mo.ui.slider(1, 2, value=1, show_value=True, label="custom n")
    q = mo.ui.dropdown(options=[2, 3, 5], value=3, label="custom prime q")
    maximum_degree = mo.ui.slider(1, 5, value=3, show_value=True, label="custom maximum degree")
    run = mo.ui.run_button(label="Construct groups and verify")
    mo.vstack([preset, mo.hstack([n, q, maximum_degree], gap=1), run])
    return maximum_degree, n, preset, q, run


@app.cell
def _(DEFAULT_CASES, FiniteCase, maximum_degree, mo, n, preset, q, run, run_all):
    mo.stop(not run.value, mo.callout("Choose a suite and press **Construct groups and verify**.", kind="info"))
    mo.stop(
        preset.value == "custom" and n.value == 2 and q.value != 2,
        mo.callout(
            "The exhaustive all-matrix constructor supports n=2 only at q=2. "
            "Choose q=2, or use n=1 for q=3 or q=5.",
            kind="warn",
        ),
    )
    if preset.value == "full":
        cases = DEFAULT_CASES
    elif preset.value == "quick":
        cases = (FiniteCase(1, 2, 4, ((1,), (2,), (1, 1))),)
    else:
        cases = (FiniteCase(n.value, q.value, maximum_degree.value),)
    report = run_all(cases)
    return cases, report


@app.cell
def _(mo, report):
    summary_rows = [
        {
            "n": result["n"],
            "q": result["q"],
            "dimension": result["dimension"],
            "|G| direct": result["order"],
            "|G| formula": result["expected order"],
            "|P(E)|": result["projective points"],
            "[m_(1)]": str(result["m_(1)"]),
            "coefficients": len(result["coefficient rows"]),
            "status": "PASS" if result["passed"] else "FAIL",
        }
        for result in report["finite results"]
    ]
    mo.vstack(
        [
            mo.callout(
                "All requested checks passed." if report["passed"] else "At least one check failed.",
                kind="success" if report["passed"] else "danger",
            ),
            mo.md("## Finite odd symplectic matrix groups"),
            mo.ui.table(summary_rows, pagination=False),
        ]
    )
    return


@app.cell
def _(mo, report):
    compact_selected = [
        row
        for row in report["compact rows"]
        if row["partition"] in ((), (1,), (2,), (1, 1), (3, 1), (2, 2))
    ]
    mo.vstack(
        [
            mo.md(r"""
            ## Compact substitutes

            For (K=Sp(1)\times U(1)) on (W\oplus\chi), exact (U(1))
            constant-term extraction removes every positive radical weight,
            leaving the (Sp(1))-series on (W). For the subgroup fixing the
            radical vector, the extra eigenvalue is (1), so each factor gains
            ((1-t_i)^{-1}).
            """),
            mo.ui.table(compact_selected, pagination=False),
            mo.md("## Full-group invariant-series certificate"),
            mo.ui.table(list(report["invariant rows"]), pagination=False),
        ]
    )
    return


@app.cell
def _(mo, report):
    coefficient_rows = []
    orbit_rows = []
    for result in report["finite results"]:
        for row in result["coefficient rows"]:
            coefficient_rows.append(
                {
                    "n,q": f"{result['n']},{result['q']}",
                    "partition": row["partition"],
                    "matrix average": str(row["matrix average"]),
                    "cycle formula": str(row["cycle formula"]),
                    "status": "PASS" if row["passed"] else "FAIL",
                }
            )
        for row in result["orbit rows"]:
            orbit_rows.append(
                {
                    "n,q": f"{result['n']},{result['q']}",
                    "partition": row["partition"],
                    "explicit orbit count": row["explicit orbits"],
                    "matrix average": str(row["matrix average"]),
                    "status": "PASS" if row["passed"] else "FAIL",
                }
            )
    blocks = [
        mo.md("## All direct-matrix versus cycle-formula coefficients"),
        mo.ui.table(coefficient_rows, pagination=True, page_size=20),
    ]
    if orbit_rows:
        blocks += [
            mo.md("## Independent orbit enumeration"),
            mo.ui.table(orbit_rows, pagination=False),
        ]
    mo.vstack(blocks)
    return


if __name__ == "__main__":
    app.run()
