# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.23.8",
#   "numpy>=2.0",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    from functools import lru_cache
    from itertools import product
    from math import comb, pi

    import marimo as mo
    import numpy as np

    return comb, lru_cache, mo, np, pi, product


@app.cell
def _(lru_cache, np, pi, product):
    @lru_cache(maxsize=None)
    def weak_compositions(total: int, length: int) -> tuple[tuple[int, ...], ...]:
        """All length-part weak compositions of total."""

        if total < 0 or length < 1:
            raise ValueError("total must be nonnegative and length must be positive")
        if length == 1:
            return ((total,),)
        return tuple(
            (first,) + rest
            for first in range(total + 1)
            for rest in weak_compositions(total - first, length - 1)
        )

    def normalize_inputs(partition, weights, modulus):
        tau = tuple(int(part) for part in partition)
        a = tuple(int(weight) % int(modulus) for weight in weights)
        if modulus < 1:
            raise ValueError("the group order m must be positive")
        if any(part < 0 for part in tau):
            raise ValueError("partition parts must be nonnegative")
        if not a:
            raise ValueError("provide at least one cyclic weight")
        return tau, a, int(modulus)

    def direct_weight_count(partition, weights, modulus):
        """Count the arrays q_(i,j) in the boxed congruence formula."""

        tau, a, m = normalize_inputs(partition, weights, modulus)
        block_rows = [weak_compositions(degree, len(a)) for degree in tau]
        return sum(
            sum(a_j * q_ij for row in rows for a_j, q_ij in zip(a, row)) % m == 0
            for rows in product(*block_rows)
        )

    def block_residue_counts(degree, weights, modulus):
        """Count degree-r monomials in each residue class."""

        counts = [0] * modulus
        for row in weak_compositions(degree, len(weights)):
            residue = sum(weight * exponent for weight, exponent in zip(weights, row)) % modulus
            counts[residue] += 1
        return tuple(counts)

    def residue_convolution_count(partition, weights, modulus):
        """Convolve block weight distributions using exact integer arithmetic."""

        tau, a, m = normalize_inputs(partition, weights, modulus)
        totals = [0] * m
        totals[0] = 1
        for degree in tau:
            block = block_residue_counts(degree, a, m)
            totals = [
                sum(totals[left] * block[(residue - left) % m] for left in range(m))
                for residue in range(m)
            ]
        return totals[0]

    def complete_homogeneous_value(degree, eigenvalues):
        """Coefficient of z^degree in product_j (1-lambda_j z)^(-1)."""

        coefficients = np.zeros(degree + 1, dtype=complex)
        coefficients[0] = 1
        for eigenvalue in eigenvalues:
            updated = np.zeros(degree + 1, dtype=complex)
            for old_degree in range(degree + 1):
                for copies in range(degree - old_degree + 1):
                    updated[old_degree + copies] += coefficients[old_degree] * eigenvalue**copies
            coefficients = updated
        return coefficients[degree]

    def molien_average(partition, weights, modulus):
        """Compute (1/m) sum_k product_i h_{tau_i}(omega^(k a_1),...)."""

        tau, a, m = normalize_inputs(partition, weights, modulus)
        omega = np.exp(2j * pi / m)
        total = 0j
        for k in range(m):
            eigenvalues = tuple(omega ** (k * weight) for weight in a)
            total += np.prod(
                [complete_homogeneous_value(degree, eigenvalues) for degree in tau],
                dtype=complex,
            )
        return total / m

    def parse_integer_list(text, *, allow_empty=False):
        stripped = text.strip()
        if not stripped:
            if allow_empty:
                return ()
            raise ValueError("enter a comma-separated list of integers")
        return tuple(int(item.strip()) for item in stripped.split(","))

    def integer_partitions(total, largest=None):
        if total == 0:
            yield ()
            return
        upper = total if largest is None else min(total, largest)
        for first in range(upper, 0, -1):
            for rest in integer_partitions(total - first, first):
                yield (first,) + rest

    def verification_suite(max_order=8, max_total_degree=7):
        """Check arbitrary weights and cyclic-shift representations."""

        cases = []
        for m in range(2, max_order + 1):
            weight_sets = {
                (1,),
                tuple(range(m)),
                tuple((j * j + 1) % m for j in range(min(m + 2, 6))),
                tuple(0 for _ in range(min(m, 4))),
            }
            partitions = tuple(
                partition
                for degree in range(max_total_degree + 1)
                for partition in integer_partitions(degree)
                if len(partition) <= 3
            )
            for weights in sorted(weight_sets):
                for partition in partitions:
                    exact = direct_weight_count(partition, weights, m)
                    convolution = residue_convolution_count(partition, weights, m)
                    molien = molien_average(partition, weights, m)
                    nearest = round(molien.real)
                    error = abs(molien - exact)
                    cases.append(
                        {
                            "m": m,
                            "d": len(weights),
                            "weights": str(weights),
                            "tau": str(partition),
                            "direct": exact,
                            "convolution": convolution,
                            "Molien rounded": nearest,
                            "Molien error": error,
                            "passed": exact == convolution == nearest and error < 1e-8,
                        }
                    )
        return cases

    return (
        direct_weight_count,
        molien_average,
        parse_integer_list,
        residue_convolution_count,
        verification_suite,
        weak_compositions,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Cyclic-weight distribution: proof-by-computation

    Let the generator of $C_m$ act diagonally on $V=\mathbb C^d$ with
    eigenvalues $\omega^{a_1},\ldots,\omega^{a_d}$. For
    $\tau=(\tau_1,\ldots,\tau_s)$, the proposed formula is

    \[
    \dim\!\left(\bigotimes_{i=1}^s\operatorname{Sym}^{\tau_i}(V)\right)^{C_m}
    =\#\left\{(q_{i,j}):
    \sum_jq_{i,j}=\tau_i,\quad
    \sum_{i,j}a_jq_{i,j}\equiv0\pmod m\right\}.
    \]

    This notebook tests that identity three ways:

    1. literal enumeration of every array $(q_{i,j})$;
    2. exact cyclic convolution of the residue counts of the symmetric-power blocks;
    3. numerical evaluation of the root-of-unity (refined Molien) average.

    The first two paths use exact integers. The third is independent complex
    arithmetic and should differ from the integer answer only by floating-point noise.
    """)
    return


@app.cell
def _(mo):
    group_order = mo.ui.slider(2, 14, value=5, label="group order m", show_value=True)
    weights_text = mo.ui.text(value="0, 1, 2, 3, 4", label="weights a_j")
    partition_text = mo.ui.text(value="3, 2", label="partition tau")
    mo.hstack([group_order, weights_text, partition_text], justify="start", gap=1)
    return group_order, partition_text, weights_text


@app.cell
def _(
    direct_weight_count,
    group_order,
    mo,
    molien_average,
    parse_integer_list,
    partition_text,
    residue_convolution_count,
    weights_text,
):
    try:
        selected_tau = parse_integer_list(partition_text.value, allow_empty=True)
        selected_weights = parse_integer_list(weights_text.value)
        selected_direct = direct_weight_count(
            selected_tau, selected_weights, group_order.value
        )
        selected_convolution = residue_convolution_count(
            selected_tau, selected_weights, group_order.value
        )
        selected_molien = molien_average(
            selected_tau, selected_weights, group_order.value
        )
        selected_error = abs(selected_molien - selected_direct)
        selected_passed = (
            selected_direct == selected_convolution
            and round(selected_molien.real) == selected_direct
            and selected_error < 1e-8
        )
        selected_message = mo.callout(
            (
                f"direct count = {selected_direct}; exact convolution = "
                f"{selected_convolution}; Molien average = "
                f"{selected_molien.real:.12g}{selected_molien.imag:+.2g}i; "
                f"error = {selected_error:.3g}."
            ),
            kind="success" if selected_passed else "danger",
        )
    except ValueError as selected_exception:
        selected_tau = ()
        selected_weights = ()
        selected_direct = None
        selected_convolution = None
        selected_molien = None
        selected_error = None
        selected_passed = False
        selected_message = mo.callout(str(selected_exception), kind="danger")

    mo.vstack([mo.md("## Selected example"), selected_message])
    return selected_passed, selected_tau, selected_weights


@app.cell
def _(comb, group_order, mo, selected_tau, selected_weights, weak_compositions):
    if selected_weights:
        block_sizes = [
            {
                "block i": index + 1,
                "degree tau_i": degree,
                "number of monomials": comb(degree + len(selected_weights) - 1, degree),
                "enumerated rows": len(weak_compositions(degree, len(selected_weights))),
            }
            for index, degree in enumerate(selected_tau)
        ]
        detail = mo.vstack(
            [
                mo.md(
                    fr"""
                    The selected representation has $m={group_order.value}$ and
                    $d={len(selected_weights)}$. Its weights modulo $m$ are
                    ${tuple(weight % group_order.value for weight in selected_weights)}$.
                    """
                ),
                mo.ui.table(block_sizes, pagination=False),
            ]
        )
    else:
        detail = mo.md("")
    detail
    return


@app.cell
def _(mo, verification_suite):
    suite_rows = verification_suite(max_order=8, max_total_degree=7)
    failed_rows = [row for row in suite_rows if not row["passed"]]
    maximum_molien_error = max(row["Molien error"] for row in suite_rows)
    suite_summary = mo.callout(
        (
            f"{len(suite_rows):,} cases checked for 2 <= m <= 8, total degree <= 7, "
            f"and at most three tensor blocks. Failures: {len(failed_rows)}. "
            f"Maximum Molien error: {maximum_molien_error:.3g}."
        ),
        kind="success" if not failed_rows else "danger",
    )
    mo.vstack(
        [
            mo.md("## Batch verification over reasonable dimensions"),
            suite_summary,
            mo.ui.tabs(
                {
                    "Failures": mo.ui.table(failed_rows, pagination=True),
                    "All cases": mo.ui.table(suite_rows, pagination=True),
                }
            ),
        ]
    )
    return failed_rows, maximum_molien_error, suite_rows


@app.cell
def _(mo):
    mo.md(r"""
    ## What the computation verifies

    For a fixed group element $R^k$, the eigenvalue of the monomial indexed by
    $(q_{i,j})$ is
    $\omega^{k\sum_{i,j}a_jq_{i,j}}$. Averaging this eigenvalue over $k$ is
    $1$ exactly when the exponent is $0$ modulo $m$, and $0$ otherwise. This
    is character orthogonality, so the direct count and Molien average are the
    same quantity.

    The cyclic permutation representation is obtained by choosing $d=m$ and
    weights $(0,1,\ldots,m-1)$. The suite includes that case at every tested
    order. It also includes faithful one-dimensional characters, repeated
    zero weights, and nonuniform repeated-weight representations. Thus the
    verification checks the dependence on the representation, not merely on
    the abstract group.

    The infinite refined Molien product is interpreted coefficientwise:

    \[
    \frac1m\sum_{k=0}^{m-1}\prod_{r\ge1}\prod_{j=1}^{d}
       \frac{1}{1-\omega^{ka_j}t_r}.
    \]

    Its coefficient of $m_\tau$ is the invariant count above. No convergence
    claim for an analytic infinite product is needed.
    """)
    return


if __name__ == "__main__":
    app.run()
