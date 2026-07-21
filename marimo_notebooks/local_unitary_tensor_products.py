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

    folder = (Path(__file__).resolve().parents[1] / "proved_matrix_groups/packages/local_unitary_tensor_products/notebook.py").parent
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    from verification import DEFAULT_CASES, exact_tables, run_all

    return DEFAULT_CASES, exact_tables, mo, run_all


@app.cell
def _(mo):
    mo.md(r"""
    # Local-unitary tensor products: balanced sigma-MGF verification

    This notebook tests

    \[
    G=U(d_1)\times\cdots\times U(d_k),\qquad
    \rho(g_1,\ldots,g_k)=g_1\otimes\cdots\otimes g_k.
    \]

    The ordinary one-sided sigma-MGF is $1$: the central element
    $(zI_{d_1},I,\ldots,I)$ acts on every positive total degree $n$ by
    $z^n$. The informative object is the balanced series

    \[
    \mathcal B_{G,V}=\sum_{\alpha,\beta}
    \dim\!\left(\operatorname{Sym}^{\alpha}V\otimes
    \operatorname{Sym}^{\beta}V^*\right)^G m_\alpha m_\beta.
    \]

    Four independent checks are made:

    1. explicit Kronecker-product matrices are unitary and satisfy
       $\operatorname{Tr}(\rho(g)^r)=\prod_a\operatorname{Tr}(g_a^r)$;
    2. explicit central matrices eliminate the one-sided positive degrees;
    3. direct Haar-matrix averages agree with balanced formulas;
    4. exact partition and hook-length computations test stable and
       dimension-boundary coefficients without Monte Carlo.
    """)
    return


@app.cell
def _(mo):
    samples = mo.ui.slider(
        1000,
        20000,
        step=1000,
        value=6000,
        show_value=True,
        label="Haar samples per representation",
    )
    run = mo.ui.run_button(label="Construct matrices and verify")
    mo.vstack(
        [
            mo.callout(
                "The default 6,000 samples are deterministic and normally finish in a few seconds.",
                kind="info",
            ),
            mo.hstack([samples, run], gap=2),
        ]
    )
    return run, samples


@app.cell
def _(mo, run, run_all, samples):
    mo.stop(
        not run.value,
        mo.callout("Press **Construct matrices and verify** to run the experiment.", kind="info"),
    )
    report = run_all(samples=samples.value)
    return (report,)


@app.cell
def _(mo, report):
    construction_rows = [
        {
            "representation": row["label"],
            "local dimensions": str(row["dimensions"]),
            "matrix dimension": row["matrix dimension"],
            "unitarity error": f"{row['unitarity error']:.2e}",
            "trace identity error": f"{row['trace-factorization error']:.2e}",
            "status": "PASS" if row["passed"] else "FAIL",
        }
        for row in report["construction"]
    ]
    center_rows = [
        {
            "local dimensions": str(row["dimensions"]),
            "positive degree": row["degree"],
            "matrix scalar error": f"{row['rho scalar error']:.2e}",
            "|z^degree - 1|": f"{row['distance from 1']:.3f}",
            "status": "PASS" if row["passed"] else "FAIL",
        }
        for row in report["center"]
    ]
    mo.vstack(
        [
            mo.callout(
                "All checks passed." if report["passed"] else "At least one check failed.",
                kind="success" if report["passed"] else "danger",
            ),
            mo.md("## Constructed matrix representations"),
            mo.ui.table(construction_rows, pagination=False),
            mo.md("## One-sided scalar-center witnesses"),
            mo.ui.table(center_rows, pagination=False),
        ]
    )
    return


@app.cell
def _(mo, report):
    numerical_rows = []
    for result in report["numerical"]:
        for row in result["rows"]:
            estimate = row["estimate"]
            numerical_rows.append(
                {
                    "representation": result["label"],
                    "sector": row["sector"],
                    "index": str(row["index"]),
                    "direct Haar estimate": (
                        f"{estimate.real:.4f}{estimate.imag:+.4f}i"
                    ),
                    "formula": row["expected"],
                    "standard error": f"{row['standard error']:.3g}",
                    "status": "PASS" if row["passed"] else "FAIL",
                }
            )
    mo.vstack(
        [
            mo.md(r"""
            ## Direct target quantities versus formulas

            - **ordinary**: $\mathbb E\prod_j h_{\tau_j}(\rho(g))=0$;
            - **power**: $\mathbb E[p_\lambda(\rho)\overline{p_\mu(\rho)}]
              =\delta_{\lambda\mu}z_\lambda^k$ in the stable range;
            - **symmetric**: $\mathbb E|h_m(\rho)|^2$;
            - **multilinear**: $\mathbb E|\operatorname{Tr}\rho|^{2r}$.

            The matrices are drawn from Haar measure by complex Gaussian QR.
            A row passes when its error is within six empirical standard errors
            plus a small fixed numerical margin.
            """),
            mo.ui.table(numerical_rows, pagination=True, page_size=20),
        ]
    )
    return


@app.cell
def _(mo, report):
    tables = report["exact tables"]
    stable_rows = [
        {
            "number of local factors k": row["k"],
            "degree m": row["degree"],
            "stable [m_(m)m_(m)]": row["coefficient"],
        }
        for row in tables["stable symmetric"]
    ]
    bipartite_rows = [
        {
            "dimensions": str(row["dimensions"]),
            "degree m": row["degree"],
            "exact coefficient": row["coefficient"],
        }
        for row in tables["bipartite"]
    ]
    multilinear_rows = [
        {
            "dimensions": str(row["dimensions"]),
            "tensor power r": row["degree"],
            "exact coefficient": row["exact"],
            "(r!)^k": row["stable formula"],
            "stable?": row["in stable range"],
        }
        for row in tables["multilinear"]
    ]
    mo.vstack(
        [
            mo.md("## Exact combinatorial verification"),
            mo.md(r"Stable symmetric sector: $\sum_{\lambda\vdash m}z_\lambda^{k-2}$"),
            mo.ui.table(stable_rows, pagination=False),
            mo.md(r"Bipartite all-dimensional sector: $p_{\leq\min(d_1,d_2)}(m)$"),
            mo.ui.table(bipartite_rows, pagination=False),
            mo.md(r"Multilinear sector: $\prod_a\sum_{\ell(\lambda)\leq d_a}(f^\lambda)^2$"),
            mo.ui.table(multilinear_rows, pagination=False),
        ]
    )
    return


if __name__ == "__main__":
    app.run()

