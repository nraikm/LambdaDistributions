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

    root = (Path(__file__).resolve().parents[1] / "new_dists/notebooks/spin_pin.py").parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from new_dists.verification import run_all

    return mo, run_all


@app.cell
def _(mo):
    mo.md(r"""
    # Compact Spin and Pin sigma-MGFs

    This notebook tests the proposed Molien--Weyl formulas by two independent
    routes.

    1. **Exact Weyl route.**  It constructs the full spin or half-spin weight
       multiset, builds the characters
       $h_{\tau_1}\cdots h_{\tau_\ell}$, and applies the signed
       $B_n$ or $D_n$ Weyl alternant.  Every displayed coefficient is an
       integer; there is no quadrature or rounding.
    2. **Direct matrix route.**  It constructs Hermitian Clifford gamma
       matrices, exponentiates the bivectors $\frac12\gamma_a\gamma_b$ to
       obtain represented Spin elements, and diagonalizes the tensor-square
       quadratic Casimir.  For Pin, it uses $u_+=\gamma_1$ and
       $u_-=i\gamma_1$ and checks the two odd components explicitly.

    The decisive coefficients are
    \[
    \dim(\Delta_n^{\otimes4})^{\operatorname{Spin}(2n+1)}=n+1
    \]
    and
    \[
    \dim((\Delta_n^+)^{\otimes4})^{\operatorname{Spin}(2n)}=
    \begin{cases}n/2+1,&n\text{ even},\\(n-1)/2,&n\text{ odd}.
    \end{cases}
    \]
    Their linear growth rules out a finite coefficientwise limit for the
    unnormalized spinor sigma-MGFs.
    """)
    return


@app.cell
def _(mo):
    maximum_rank = mo.ui.slider(
        start=2,
        stop=5,
        value=5,
        step=1,
        show_value=True,
        label="maximum rank for exact Weyl checks",
    )
    run = mo.ui.run_button(label="Construct representations and verify")
    mo.vstack(
        [
            mo.callout(
                "Rank five is the recommended suite: it includes both parity "
                "and bilinear-type regimes while remaining an exact, quick calculation.",
                kind="info",
            ),
            mo.hstack([maximum_rank, run], justify="start", gap=2),
        ]
    )
    return maximum_rank, run


@app.cell
def _(maximum_rank, mo, run, run_all):
    mo.stop(
        not run.value,
        mo.callout(
            "Choose a maximum rank and press **Construct representations and verify**.",
            kind="info",
        ),
    )
    report = run_all(maximum_rank.value)
    return (report,)


@app.cell
def _(mo, report):
    mo.vstack(
        [
            mo.callout(
                f"All {report['checks']} checks passed."
                if report["passed"]
                else "At least one check failed; inspect the tables below.",
                kind="success" if report["passed"] else "danger",
            ),
            mo.md(
                f"Exact coefficients were evaluated through rank "
                f"$n={report['maximum rank']}$."
            ),
        ]
    )
    return


@app.cell
def _(mo, report):
    spin_rows = [
        {
            "group": row["group"],
            "module dimension": row["dimension"],
            "tau": str(row["tau"]),
            "exact Weyl": row["Weyl coefficient"],
            "formula": row["formula"],
            "status": "PASS" if row["passed"] else "FAIL",
        }
        for row in report["spin rows"]
    ]
    mo.vstack(
        [
            mo.md(r"""
            ## Odd Spin groups: $\operatorname{Spin}(2n+1)$ on $\Delta_n$

            The exact route uses all $2^n$ weights
            $\frac12(\pm e_1\pm\cdots\pm e_n)$.  The central element kills
            odd total degree, the invariant bilinear form gives the quadratic
            rows, and the fourth multilinear coefficient grows as $n+1$.
            """),
            mo.ui.table(spin_rows, pagination=True, page_size=15),
        ]
    )
    return


@app.cell
def _(mo, report):
    half_rows = [
        {
            "group": row["group"],
            "module dimension": row["dimension"],
            "tau": str(row["tau"]),
            "exact Weyl": row["Weyl coefficient"],
            "formula": row["formula"],
            "status": "PASS" if row["passed"] else "FAIL",
        }
        for row in report["half-spin rows"]
    ]
    balanced_rows = [
        {
            "group": row["group"],
            "exact Weyl": row["exact Weyl coefficient"],
            "formula": row["formula"],
            "status": "PASS" if row["passed"] else "FAIL",
        }
        for row in report["balanced rows"]
    ]
    mo.vstack(
        [
            mo.md(r"""
            ## Even Spin groups: half-spin modules

            The $D_n$ calculation retains one parity class of the spin weights.
            It detects self-duality only for even $n$, the symmetric/alternating
            period modulo four, and the one-sided fourth-moment alternation.
            The second table checks the two-sided coefficient
            $\dim((\Delta^+\otimes\Delta^{+*})^{\otimes2})^G$.
            """),
            mo.ui.table(half_rows, pagination=True, page_size=15),
            mo.md("### Balanced half-spin fourth coefficient"),
            mo.ui.table(balanced_rows, pagination=False),
        ]
    )
    return


@app.cell
def _(mo, report):
    casimir_rows = [
        {
            "group": row["group"],
            "spin dimension": row["spin dimension"],
            "Clifford residual": f"{row['Clifford residual']:.1e}",
            "Casimir blocks": row["Casimir blocks"],
            "formula blocks": row["formula blocks"],
            "observed block dimensions": str(row["block dimensions"]),
            "status": "PASS" if row["passed"] else "FAIL",
        }
        for row in report["casimir rows"]
    ]
    mo.vstack(
        [
            mo.md(r"""
            ## Independent Clifford-matrix and Casimir check

            The matrices satisfy
            $\gamma_a\gamma_b+\gamma_b\gamma_a=2\delta_{ab}I$.
            On $\Delta_n\otimes\Delta_n$, the distinct Casimir eigenspaces
            have dimensions $\binom{2n+1}{0},\ldots,\binom{2n+1}{n}$,
            recovering the exterior-power decomposition without using the
            Weyl coefficient code.
            """),
            mo.ui.table(casimir_rows, pagination=False),
        ]
    )
    return


@app.cell
def _(mo, report):
    pin_rows = [
        {
            "group": row["group"],
            "pinor dimension": row["pinor dimension"],
            "u^2 residual": f"{row['u^2 residual']:.1e}",
            "chirality swap": f"{row['chirality-swap residual']:.1e}",
            "odd trace max": f"{row['max odd-power trace']:.1e}",
            "unitarity": f"{row['unitarity residual']:.1e}",
            "status": "PASS" if row["passed"] else "FAIL",
        }
        for row in report["pin rows"]
    ]
    mo.vstack(
        [
            mo.md(r"""
            ## Disconnected Pin groups

            For $N=2,4,6$, the odd representative anticommutes with chirality,
            so it exchanges the two half-spin summands and every odd power has
            trace zero.  The two Clifford conventions remain distinct:
            $u_+^2=I$ while $u_-^2=-I$.
            """),
            mo.ui.table(pin_rows, pagination=False),
        ]
    )
    return


@app.cell
def _(mo, report):
    derived_rows = [
        {
            "group": row["group"],
            "check": row["identity"],
            "residual": (
                row["residual"]
                if isinstance(row["residual"], int)
                else f"{row['residual']:.2e}"
            ),
            "status": "PASS" if row["passed"] else "FAIL",
        }
        for row in report["derived rows"]
    ]
    mo.vstack(
        [
            mo.md(r"""
            ## Mixed spinor--vector and adjoint representations

            Explicit Kronecker matrices check
            $\chi_{\Delta\otimes V}=\chi_\Delta\chi_V$.
            Explicit adjoint torus matrices check
            \[
            \operatorname{Tr}(\operatorname{Ad}(g)^r)
            =\tfrac12\bigl(\operatorname{Tr}(V(g)^r)^2
            -\operatorname{Tr}(V(g)^{2r})\bigr).
            \]
            Exact Weyl rows independently recover the metric and structure
            tensor in low degree.
            """),
            mo.ui.table(derived_rows, pagination=True, page_size=15),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
