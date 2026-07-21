# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "lambda-distributions",
#   "marimo>=0.23.8",
# ]
#
# [tool.uv.sources]
# lambda-distributions = { path = "..", editable = true }
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from lambda_distributions import (
        check_formula,
        cyclic_character,
        scalar_sigma_monomial_coefficient,
        scalar_sigma_monomial_coefficients,
    )
    from lambda_distributions.formulas import (
        cyclic_character_moment,
        cyclic_sigma_monomial_coefficient,
    )
    from lambda_distributions.notebook_support import (
        display_number,
        parse_partition,
        verification_records,
    )

    return (
        check_formula,
        cyclic_character,
        cyclic_character_moment,
        cyclic_sigma_monomial_coefficient,
        display_number,
        mo,
        parse_partition,
        scalar_sigma_monomial_coefficient,
        scalar_sigma_monomial_coefficients,
        verification_records,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Walkthrough: test the stated cyclic sigma-MGF formula

    Consider the embedding

    \[
    C_n=\mathbb Z/n\mathbb Z \hookrightarrow \mathrm{GL}_1(\mathbb C),
    \qquad j\longmapsto \omega^j,\qquad \omega=e^{2\pi i/n}.
    \]

    We use the standard uniform expectation

    \[
    \mathbb E[f(X)]=\frac1n\sum_{j=0}^{n-1}f(\omega^j).
    \]

    The candidate formula is preserved exactly as stated:

    \[
    \mathbb E[\operatorname{Exp}_\sigma(Xh_1)]
    \stackrel{?}{=}
    \sum_\tau
    \begin{cases}
    n,&n\mid |\tau|,\\
    0,&n\nmid |\tau|
    \end{cases}m_\tau.
    \]

    The purpose of this notebook is to test that expression, not repair it in
    advance. For $n>1$, the verification should reject it.

    ## Architecture used by the test

    1. `groups.py` constructs all scalar matrices in $C_n$.
    2. `distribution.py` computes the normalized expectation directly.
    3. `formulas.py` retains the submitted $n$-valued candidate.
    4. `verification.py` compares these independent paths over partitions.
    5. This notebook explains the discrepancy and displays the counterexamples.
    """)
    return


@app.cell
def _(mo):
    n = mo.ui.slider(2, 20, value=5, label="group order n", show_value=True)
    partition_text = mo.ui.text(value="3, 2", label="test partition τ")
    max_degree = mo.ui.slider(0, 16, value=10, label="verify through degree", show_value=True)
    run = mo.ui.run_button(label="Test the stated formula")
    mo.hstack([n, partition_text, max_degree, run], justify="start", gap=1)
    return max_degree, n, partition_text, run


@app.cell
def _(
    check_formula,
    cyclic_character,
    cyclic_character_moment,
    cyclic_sigma_monomial_coefficient,
    display_number,
    max_degree,
    mo,
    n,
    parse_partition,
    partition_text,
    run,
    scalar_sigma_monomial_coefficient,
    scalar_sigma_monomial_coefficients,
):
    mo.stop(not run.value, mo.md("Choose parameters, then press **Test the stated formula**."))
    try:
        partition = parse_partition(partition_text.value)
    except ValueError as error:
        mo.stop(True, mo.callout(str(error), kind="danger"))

    group = cyclic_character(n.value)
    degree = sum(partition)
    explicit = scalar_sigma_monomial_coefficient(group, partition)
    predicted = cyclic_sigma_monomial_coefficient(partition, n.value)
    correct_value = cyclic_character_moment(partition, n.value)
    coefficients = scalar_sigma_monomial_coefficients(group, max_degree.value)

    stated_report = check_formula(
        group,
        lambda candidate: cyclic_sigma_monomial_coefficient(candidate, n.value),
        max_degree=max_degree.value,
        basis="scalar_sigma_monomial",
    )
    normalized_report = check_formula(
        group,
        lambda candidate: cyclic_character_moment(candidate, n.value),
        max_degree=max_degree.value,
        basis="scalar_sigma_monomial",
    )

    root_rows = []
    for index, matrix in enumerate(group.elements):
        root = complex(matrix[0, 0])
        root_rows.append(
            {
                "j": index,
                "ω^j": display_number(root),
                f"(ω^j)^{degree}": display_number(root**degree),
            }
        )

    degree_rows = []
    for selected_degree in range(max_degree.value + 1):
        representative = () if selected_degree == 0 else (selected_degree,)
        enumerated = coefficients[representative]
        stated = cyclic_sigma_monomial_coefficient(representative, n.value)
        normalized = cyclic_character_moment(representative, n.value)
        degree_rows.append(
            {
                "degree d": selected_degree,
                "n divides d": selected_degree % n.value == 0,
                "normalized expectation": display_number(enumerated),
                "stated n-formula": display_number(stated),
                "correct normalized value": display_number(normalized),
                "stated formula passes": abs(enumerated - stated) <= 1e-9,
            }
        )
    return (
        correct_value,
        degree,
        degree_rows,
        explicit,
        group,
        normalized_report,
        partition,
        predicted,
        root_rows,
        stated_report,
    )


@app.cell
def _(correct_value, degree, display_number, explicit, group, mo, partition, predicted):
    selected_agrees = abs(explicit - predicted) <= 1e-9
    selected_message = (
        "This particular partition does not expose the error because both sides vanish. "
        "The batch check below still tests divisible degrees."
        if selected_agrees
        else "Counterexample detected: the normalized expectation and stated formula disagree."
    )
    mo.vstack(
        [
            mo.md(
                fr"""
                ## Step 1: compute the coefficient from the definition

                Since
                \[
                \operatorname{{Exp}}_\sigma(xh_1)
                =\sum_{{d\geq0}}x^d h_d
                =\sum_\tau x^{{|\tau|}}m_\tau,
                \]
                the coefficient of $m_{{{partition}}}$ in the expectation is
                \[
                \frac1{{{group.order}}}\sum_{{j=0}}^{{{group.order - 1}}}
                (\omega^j)^{{{degree}}}.
                \]

                The sum in the numerator is ${group.order}$ when
                ${group.order}\mid {degree}$ and zero otherwise. Dividing by
                the group order makes the surviving expectation coefficient
                **1**, not **${group.order}**.
                """
            ),
            mo.callout(
                f"For τ={partition}: direct expectation = {display_number(explicit)}, "
                f"stated formula = {display_number(predicted)}, and the normalized "
                f"divisibility value = {display_number(correct_value)}.",
                kind="info",
            ),
            mo.callout(
                selected_message,
                kind="info" if selected_agrees else "success",
            ),
        ]
    )
    return


@app.cell
def _(
    degree,
    degree_rows,
    mo,
    normalized_report,
    root_rows,
    stated_report,
    verification_records,
):
    candidate_rejected = not stated_report.passed
    test_message = (
        "The stated n-formula was rejected, as expected. Red rows identify its counterexamples."
        if candidate_rejected
        else "No counterexample was found in the selected range. Increase the checked degree."
    )
    mo.vstack(
        [
            mo.md(
                fr"""
                ## Step 2: inspect the root-of-unity sum

                The third column lists every summand for $|\tau|={degree}$.
                """
            ),
            mo.ui.table(root_rows, pagination=False),
            mo.md("## Step 3: ask the verifier to judge the formula as written"),
            mo.ui.tabs(
                {
                    "Summary by degree": mo.ui.table(degree_rows, pagination=True),
                    "Stated n-formula": mo.ui.table(
                        verification_records(stated_report), pagination=True
                    ),
                    "Normalized reference": mo.ui.table(
                        verification_records(normalized_report), pagination=True
                    ),
                }
            ),
            mo.callout(
                test_message,
                kind="success" if candidate_rejected else "warn",
            ),
            mo.callout(
                f"The independent normalized reference "
                f"{'passes' if normalized_report.passed else 'fails'}; maximum error "
                f"{normalized_report.max_error:.3g}.",
                kind="success" if normalized_report.passed else "danger",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why the factor (n) appears at all

    The value (n) is useful as the **unnormalized root-of-unity sum**
    (sum_jomega^{jd}). Such sums occur as numerators in character
    orthogonality and Reynolds-operator computations. But once the expression
    is called an expectation, the factor (1/|C_n|=1/n) must be included.

    The app therefore keeps raw sums out of its expectation API. A new proposed
    formula goes in `formulas.py`, while `distribution.py` remains the normalized
    source of numerical evidence.
    """)
    return


if __name__ == "__main__":
    app.run()
