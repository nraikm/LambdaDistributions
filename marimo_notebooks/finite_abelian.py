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
    from math import comb, pi, prod

    import marimo as mo
    import numpy as np

    return comb, lru_cache, mo, np, pi, prod, product


@app.cell
def _(lru_cache, np, pi, prod, product):
    @lru_cache(maxsize=None)
    def weak_compositions(total: int, length: int) -> tuple[tuple[int, ...], ...]:
        """Return all length-part weak compositions of total."""

        if total < 0 or length < 1:
            raise ValueError("total must be nonnegative and length must be positive")
        if length == 1:
            return ((total,),)
        return tuple(
            (first,) + rest
            for first in range(total + 1)
            for rest in weak_compositions(total - first, length - 1)
        )

    def normalize_data(moduli, weight_rows, partition):
        normalized_moduli = tuple(int(value) for value in moduli)
        if not normalized_moduli or any(value < 1 for value in normalized_moduli):
            raise ValueError("moduli must be a nonempty list of positive integers")
        integer_weights = tuple(
            tuple(int(value) for value in row) for row in weight_rows
        )
        if len(integer_weights) != len(normalized_moduli):
            raise ValueError("the weight matrix must have one row per modulus")
        normalized_weights = tuple(
            tuple(int(value) % normalized_moduli[row_index] for value in row)
            for row_index, row in enumerate(integer_weights)
        )
        normalized_partition = tuple(int(value) for value in partition)
        if not normalized_weights or not normalized_weights[0]:
            raise ValueError("the representation dimension must be positive")
        width = len(normalized_weights[0])
        if any(len(row) != width for row in normalized_weights):
            raise ValueError("all rows of the weight matrix must have equal length")
        if any(value < 0 for value in normalized_partition):
            raise ValueError("partition entries must be nonnegative")
        return normalized_moduli, normalized_weights, normalized_partition

    def construct_matrix_group(moduli, weight_rows):
        """Construct rho(A) as diagonal matrices, retaining kernel repetitions."""

        normalized_moduli, normalized_weights, _ = normalize_data(
            moduli, weight_rows, ()
        )
        dimension = len(normalized_weights[0])
        matrices = []
        labels = []
        for group_element in product(
            *(range(modulus) for modulus in normalized_moduli)
        ):
            diagonal = []
            for column in range(dimension):
                phase = sum(
                    group_element[row] * normalized_weights[row][column]
                    / normalized_moduli[row]
                    for row in range(len(normalized_moduli))
                )
                diagonal.append(np.exp(2j * pi * phase))
            labels.append(tuple(group_element))
            matrices.append(np.diag(diagonal).astype(complex))
        assert len(matrices) == prod(normalized_moduli)
        return tuple(labels), tuple(matrices)

    def represented_group_error(moduli, labels, matrices):
        """Maximum identity, unitarity, and homomorphism residual."""

        dimension = matrices[0].shape[0]
        identity = np.eye(dimension, dtype=complex)
        by_label = dict(zip(labels, matrices))
        residuals = [np.linalg.norm(by_label[(0,) * len(moduli)] - identity)]
        residuals.extend(
            np.linalg.norm(matrix.conj().T @ matrix - identity)
            for matrix in matrices
        )
        for left_label, left_matrix in zip(labels, matrices):
            for right_label, right_matrix in zip(labels, matrices):
                product_label = tuple(
                    (left + right) % modulus
                    for left, right, modulus in zip(
                        left_label, right_label, moduli
                    )
                )
                residuals.append(
                    np.linalg.norm(
                        left_matrix @ right_matrix - by_label[product_label]
                    )
                )
        return float(max(residuals, default=0.0))

    def symmetric_power_matrix(matrix, degree):
        """Matrix of Sym^degree(matrix) in the monomial basis."""

        eigenvalues = np.diag(matrix)
        diagonal = [
            np.prod(eigenvalues ** np.asarray(exponents), dtype=complex)
            for exponents in weak_compositions(int(degree), matrix.shape[0])
        ]
        return np.diag(diagonal).astype(complex)

    def target_matrix(matrix, partition):
        """Explicit matrix on tensor_b Sym^(tau_b)(V)."""

        result = np.ones((1, 1), dtype=complex)
        for degree in partition:
            result = np.kron(result, symmetric_power_matrix(matrix, degree))
        return result

    def projector_verification(moduli, weight_rows, partition, dimension_cap=1000):
        """Average explicit target matrices and obtain the invariant dimension."""

        normalized_moduli, normalized_weights, normalized_partition = normalize_data(
            moduli, weight_rows, partition
        )
        ambient_dimension = prod(
            len(weak_compositions(degree, len(normalized_weights[0])))
            for degree in normalized_partition
        )
        if ambient_dimension > dimension_cap:
            raise ValueError(
                f"target dimension {ambient_dimension} exceeds the safety cap "
                f"{dimension_cap}; lower tau or the representation dimension"
            )
        labels, matrices = construct_matrix_group(
            normalized_moduli, normalized_weights
        )
        projector = sum(
            (target_matrix(matrix, normalized_partition) for matrix in matrices),
            start=np.zeros((ambient_dimension, ambient_dimension), dtype=complex),
        ) / len(matrices)
        singular_values = np.linalg.svd(projector, compute_uv=False)
        rank = int(np.count_nonzero(singular_values > 1e-8))
        trace = np.trace(projector)
        idempotence_error = float(np.linalg.norm(projector @ projector - projector))
        return {
            "group order": len(matrices),
            "representation dimension": len(normalized_weights[0]),
            "target dimension": ambient_dimension,
            "projector rank": rank,
            "projector trace": trace,
            "idempotence error": idempotence_error,
            "group law error": represented_group_error(
                normalized_moduli, labels, matrices
            ),
        }

    def zero_weight_formula(moduli, weight_rows, partition):
        """Exact count in the simultaneous-congruence formula."""

        normalized_moduli, normalized_weights, normalized_partition = normalize_data(
            moduli, weight_rows, partition
        )
        dimension = len(normalized_weights[0])
        block_rows = [
            weak_compositions(degree, dimension)
            for degree in normalized_partition
        ]
        count = 0
        for selected_rows in product(*block_rows):
            column_totals = tuple(map(sum, zip(*selected_rows))) if selected_rows else (0,) * dimension
            if all(
                sum(weight * exponent for weight, exponent in zip(weight_row, column_totals))
                % modulus
                == 0
                for modulus, weight_row in zip(normalized_moduli, normalized_weights)
            ):
                count += 1
        return count

    def molien_trace_average(moduli, weight_rows, partition):
        """Average products of symmetric-power traces over explicit matrices."""

        normalized_moduli, normalized_weights, normalized_partition = normalize_data(
            moduli, weight_rows, partition
        )
        _, matrices = construct_matrix_group(normalized_moduli, normalized_weights)
        values = []
        for matrix in matrices:
            block_traces = [
                np.trace(symmetric_power_matrix(matrix, degree))
                for degree in normalized_partition
            ]
            values.append(np.prod(block_traces, dtype=complex))
        return sum(values) / len(values)

    def parse_integer_list(text):
        stripped = text.strip()
        return () if not stripped else tuple(int(item.strip()) for item in stripped.split(","))

    def parse_weight_matrix(text):
        rows = tuple(
            parse_integer_list(row.strip())
            for row in text.strip().split(";")
            if row.strip()
        )
        if not rows:
            raise ValueError("enter at least one semicolon-separated weight row")
        return rows

    return (
        construct_matrix_group,
        molien_trace_average,
        normalize_data,
        parse_integer_list,
        parse_weight_matrix,
        projector_verification,
        weak_compositions,
        zero_weight_formula,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Finite abelian zero-weight formula

    Let
    \(A=C_{m_1}\times\cdots\times C_{m_r}\) and let the columns of
    \(W\in\mathbb Z^{r\times d}\) be the weights of a diagonal representation
    \(V=\bigoplus_{j=1}^d\mathbb C_{w_j}\).  For
    \(\tau=(\tau_1,\ldots,\tau_s)\), this notebook tests

    \[
    \dim\!\left(\bigotimes_{b=1}^s\operatorname{Sym}^{\tau_b}V\right)^A
    =\#\left\{(q_{bj})\geq0:
    \sum_jq_{bj}=\tau_b,\quad
    W\sum_bq_b\equiv0\pmod{(m_1,\ldots,m_r)}\right\}.
    \]

    Three routes are compared:

    1. construct every diagonal matrix \(\rho(a)\), construct its full matrix on
       \(\bigotimes_b\operatorname{Sym}^{\tau_b}V\), and find the rank of the
       Reynolds projector \(|A|^{-1}\sum_a\rho_\tau(a)\);
    2. count the integer arrays satisfying the simultaneous congruences exactly;
    3. evaluate the generalized Molien/character average numerically.

    The image matrices are deliberately retained with multiplicity when the
    representation has a kernel, because the average is over the abstract group.
    """)
    return


@app.cell
def _(mo):
    moduli_text = mo.ui.text(value="3, 4", label="moduli m_l")
    weights_text = mo.ui.text(value="1, 0, 2; 0, 1, 3", label="weight rows (use ; between rows)")
    partition_text = mo.ui.text(value="3, 2", label="partition tau")
    mo.hstack([moduli_text, weights_text, partition_text], justify="start", gap=1)
    return moduli_text, partition_text, weights_text


@app.cell
def _(
    mo,
    moduli_text,
    molien_trace_average,
    parse_integer_list,
    parse_weight_matrix,
    partition_text,
    projector_verification,
    weights_text,
    zero_weight_formula,
):
    try:
        selected_moduli = parse_integer_list(moduli_text.value)
        selected_weights = parse_weight_matrix(weights_text.value)
        selected_partition = parse_integer_list(partition_text.value)
        selected_direct = projector_verification(
            selected_moduli, selected_weights, selected_partition
        )
        selected_formula = zero_weight_formula(
            selected_moduli, selected_weights, selected_partition
        )
        selected_molien = molien_trace_average(
            selected_moduli, selected_weights, selected_partition
        )
        selected_error = abs(selected_molien - selected_formula)
        selected_passed = (
            selected_direct["projector rank"] == selected_formula
            and abs(selected_direct["projector trace"] - selected_formula) < 1e-8
            and selected_direct["idempotence error"] < 1e-8
            and selected_direct["group law error"] < 1e-8
            and selected_error < 1e-8
        )
        selected_summary = mo.callout(
            (
                f"projector rank = {selected_direct['projector rank']}; "
                f"zero-weight count = {selected_formula}; "
                f"Molien average = {selected_molien.real:.12g}"
                f"{selected_molien.imag:+.2g}i; "
                f"target dimension = {selected_direct['target dimension']}; "
                f"max numerical discrepancy = "
                f"{max(selected_error, selected_direct['idempotence error'], selected_direct['group law error']):.3g}."
            ),
            kind="success" if selected_passed else "danger",
        )
    except (ValueError, OverflowError) as selected_exception:
        selected_moduli = ()
        selected_weights = ()
        selected_partition = ()
        selected_direct = None
        selected_formula = None
        selected_molien = None
        selected_error = None
        selected_passed = False
        selected_summary = mo.callout(str(selected_exception), kind="danger")

    mo.vstack([mo.md("## Interactive check"), selected_summary])
    return


@app.cell
def _():
    representative_cases = (
        {
            "name": "C2: trivial plus sign",
            "moduli": (2,),
            "weights": ((0, 1),),
            "partitions": ((), (1,), (2,), (1, 1), (3, 2)),
        },
        {
            "name": "C4: nonfaithful 3D",
            "moduli": (4,),
            "weights": ((0, 2, 2),),
            "partitions": ((1,), (2,), (3,), (2, 1)),
        },
        {
            "name": "C2 x C2: all characters",
            "moduli": (2, 2),
            "weights": ((0, 1, 0, 1), (0, 0, 1, 1)),
            "partitions": ((1,), (2,), (1, 1), (2, 1)),
        },
        {
            "name": "C3 x C4: mixed 3D",
            "moduli": (3, 4),
            "weights": ((1, 0, 2), (0, 1, 3)),
            "partitions": ((1,), (2,), (3,), (2, 1), (3, 2)),
        },
        {
            "name": "C2 x C3: repeated weights",
            "moduli": (2, 3),
            "weights": ((0, 1, 1, 0), (0, 1, 2, 2)),
            "partitions": ((1,), (2,), (1, 1), (2, 2)),
        },
    )
    return (representative_cases,)


@app.cell
def _(
    molien_trace_average,
    projector_verification,
    representative_cases,
    zero_weight_formula,
):
    suite_rows = []
    for case in representative_cases:
        for case_partition in case["partitions"]:
            case_direct = projector_verification(
                case["moduli"], case["weights"], case_partition
            )
            case_formula = zero_weight_formula(
                case["moduli"], case["weights"], case_partition
            )
            case_molien = molien_trace_average(
                case["moduli"], case["weights"], case_partition
            )
            case_error = max(
                abs(case_direct["projector trace"] - case_formula),
                abs(case_molien - case_formula),
                case_direct["idempotence error"],
                case_direct["group law error"],
            )
            suite_rows.append(
                {
                    "case": case["name"],
                    "tau": str(case_partition),
                    "target dim": case_direct["target dimension"],
                    "projector rank": case_direct["projector rank"],
                    "formula": case_formula,
                    "Molien rounded": round(case_molien.real),
                    "error": case_error,
                    "passed": case_direct["projector rank"] == case_formula
                    and case_error < 1e-8,
                }
            )
    suite_passed = all(row["passed"] for row in suite_rows)
    return suite_passed, suite_rows


@app.cell
def _(mo, suite_passed, suite_rows):
    suite_status = mo.callout(
        f"{'All' if suite_passed else 'Not all'} {len(suite_rows)} representative checks passed.",
        kind="success" if suite_passed else "danger",
    )
    mo.vstack(
        [
            mo.md("## Automated representative suite"),
            suite_status,
            mo.ui.table(suite_rows, pagination=True, page_size=10),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Practical size guidance

    If \(d=\dim V\), the target matrix dimension is
    \[
    N=\prod_b \binom{\tau_b+d-1}{\tau_b}.
    \]
    The full projector method stores an \(N\times N\) complex matrix, so this
    notebook caps \(N\) at 1000.  The default has \(d=3\),
    \(\tau=(3,2)\), and \(N=10\cdot6=60\).  For larger parameters, the exact
    congruence count or residue-class convolution is preferable to forming the
    projector explicitly.
    """)
    return


if __name__ == "__main__":
    app.run()
