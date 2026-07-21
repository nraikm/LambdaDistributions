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

    root = (Path(__file__).resolve().parents[1] / "new_dists/notebooks/non_haar_fourier.py").parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from new_dists.non_haar_fourier_sigma_mgfs.verification import run_all

    return mo, run_all


@app.cell
def _(mo):
    mo.md(r"""
    # Non-Haar Fourier-weighted sigma-MGFs

    For (W_\tau=\bigotimes_j\operatorname{Sym}^{\tau_j}V), the
    coefficient under a random-walk law is

    \[
    [m_\tau]\mathcal S_{\nu^{*k},V}
      =\sum_{\pi\in\widehat G}m_\pi(W_\tau)
        \operatorname{Tr}\!\left(\widehat\nu(\pi)^k\right).
    \]

    This laboratory keeps the three mathematically different settings in
    separate sections.  It constructs every finite matrix explicitly and
    compares the literal target average with an independently implemented
    formula route.
    """)
    return


@app.cell
def _(mo, run_all):
    run = mo.ui.run_button(label="Construct groups and run all checks")
    mo.vstack(
        [
            mo.callout(
                "The exact suite enumerates S3 and S_n through n=6, then performs a 256-node U(1) heat quadrature.",
                kind="info",
            ),
            run,
        ]
    )
    return (run,)


@app.cell
def _(mo, run, run_all):
    mo.stop(not run.value, mo.md("Press **Construct groups and run all checks** to begin."))
    report = run_all()
    return (report,)


@app.cell
def _(mo, report):
    finite = report["finite walks"]
    transform = finite["noncentral transform"]
    finite_summary = {
        "group": "S3",
        "order": finite["group order"],
        "representations": "natural (3D), deleted (2D)",
        "exact comparisons": finite["comparisons"],
        "noncentral standard transform": str(transform),
        "is scalar?": finite["noncentral transform is scalar"],
        "status": "PASS" if finite["passed"] else "FAIL",
    }
    mo.vstack(
        [
            mo.md(r"""
            ## 1. Finite matrix group: central and noncentral walks on (S_3)

            All six (3\times3) permutation matrices and their (2\times2)
            deleted restrictions are built.  Convolution is computed element
            by element.  A separate route decomposes each (W_\tau) by exact
            character inner products and evaluates all three Fourier blocks.
            The central law is also checked using scalar character ratios.

            The noncentral step law is
            (\frac12\delta_{(23)}+\frac12\delta_{(123)}).  Its standard
            Fourier block is visibly not scalar, so replacing it by a
            character ratio would be invalid.
            """),
            mo.ui.table([finite_summary], pagination=False),
            mo.ui.table(list(finite["rows"]), pagination=True, page_size=15),
        ]
    )
    return


@app.cell
def _(mo, report):
    heat = report["compact heat"]
    heat_summary = {
        "group": "U(1)",
        "representation": "diag(z, z^-1)",
        "dimension": heat["representation dimension"],
        "comparisons": heat["comparisons"],
        "maximum error": f"{heat['maximum error']:.3e}",
        "endpoint checks": heat["endpoint checks"],
        "status": "PASS" if heat["passed"] else "FAIL",
    }
    mo.vstack(
        [
            mo.md(r"""
            ## 2. Compact matrix group: the (U(1)) heat kernel

            At each quadrature node the code constructs
            (\operatorname{diag}(e^{i\phi},e^{-i\phi})), computes symmetric
            characters from matrix-power traces, multiplies by the periodic
            heat density, and integrates.  The comparison route decomposes
            (W_\tau) into integer weights and sums
            (e^{-tq^2/2}) with multiplicity.  Times (0.2,0.7,2.0) and all
            partitions through degree five are tested, together with the
            identity and Haar endpoints.
            """),
            mo.ui.table([heat_summary], pagination=False),
            mo.ui.table(list(heat["rows"]), pagination=True, page_size=12),
        ]
    )
    return


@app.cell
def _(mo, report):
    ewens = report["ewens"]
    ewens_summary = {
        "groups": "S3, S4, S5, S6",
        "matrix dimensions": str(ewens["dimensions"]),
        "theta": str(ewens["theta values"]),
        "coefficient checks": ewens["coefficient comparisons"],
        "full determinant checks": ewens["full determinant comparisons"],
        "status": "PASS" if ewens["passed"] else "FAIL",
    }
    mo.vstack(
        [
            mo.md(r"""
            ## 3. Ewens laws on symmetric matrix groups

            The direct route enumerates every permutation matrix and weights
            it by (\theta^{K(\sigma)}/\theta^{(n)}).  The formula route
            forgets the elements and aggregates by cycle shape.  Exact
            rational coefficients through degree four are compared for
            (n=3,4,5,6) and (\theta=1/2,1,2).  Full two-variable
            determinant averages provide a non-truncated check.
            """),
            mo.ui.table([ewens_summary], pagination=False),
            mo.ui.table(list(ewens["rows"]), pagination=True, page_size=15),
        ]
    )
    return


@app.cell
def _(mo, report):
    mo.callout(
        "All proof-driven checks passed." if report["passed"] else "At least one check failed.",
        kind="success" if report["passed"] else "danger",
    )
    return


if __name__ == "__main__":
    app.run()
