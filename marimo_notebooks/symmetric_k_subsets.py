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

    here = (Path(__file__).resolve().parents[1] / "lambda_distributions/proofs/limiting_lambda_claims/symmetric_k_subsets/notebook.py").parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from verification import run_sweep
    return mo, run_sweep


@app.cell
def _(mo):
    mo.md(r"""
    # \(S_n\) on \(k\)-subsets: matrix and limit laboratory

    For every permutation, this notebook constructs the full
    \(\binom nk\)-dimensional permutation matrix. It compares
    \(\operatorname{Tr}(\rho(\sigma)^r)\), for \(1\le r\le4\), with

    \[
    [u^k]\prod_{j\le kr}
    (1+u^{j/\gcd(j,r)})^{\gcd(j,r)A_j(\sigma)}.
    \]

    It also compares six joint power-trace averages and checks the exact
    mean/variance fingerprint \(1,k\) in the stable range \(n\ge2k\).
    The Poisson-cycle limit and its proof are documented in the companion PDF.
    """)
    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Construct S_n matrices and run exhaustive sweep")
    run
    return (run,)


@app.cell
def _(mo, run, run_sweep):
    mo.stop(not run.value, mo.md("Press the button to run all matrix comparisons."))
    result = run_sweep()
    summaries = [
        {key: value for key, value in case.items() if key != "coefficient rows"}
        for case in result["cases"]
    ]
    mo.vstack([
        mo.callout(
            f"PASS: {result['total matrices']} matrices and "
            f"{result['total trace checks']} trace/formula comparisons.",
            kind="success",
        ),
        mo.md("## Representative dimensions and moment checks"),
        mo.ui.table(summaries),
        mo.md("## Joint power-trace numerator averages"),
        mo.ui.table(result["coefficient rows"], pagination=True, page_size=12),
    ])
    return


if __name__ == "__main__":
    app.run()
