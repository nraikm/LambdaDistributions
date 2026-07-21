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
    from math import prod

    import marimo as mo
    import numpy as np

    return lru_cache, mo, np, prod, product


@app.cell
def _(lru_cache, np, prod, product):
    TOLERANCE = 1.0e-8

    @lru_cache(maxsize=None)
    def weak_compositions(total: int, length: int) -> tuple[tuple[int, ...], ...]:
        if total < 0 or length < 1:
            raise ValueError("total must be nonnegative and length must be positive")
        if length == 1:
            return ((total,),)
        return tuple(
            (first,) + rest
            for first in range(total + 1)
            for rest in weak_compositions(total - first, length - 1)
        )

    def character_value(moduli, weight, element):
        phase = sum(
            (int(w) % modulus) * coordinate / modulus
            for modulus, w, coordinate in zip(moduli, weight, element, strict=True)
        )
        return np.exp(2j * np.pi * phase)

    def validate_spec(spec):
        moduli = tuple(int(value) for value in spec["moduli"])
        if not moduli or any(value < 2 for value in moduli):
            raise ValueError("moduli must be a nonempty tuple of integers at least 2")
        for weight, epsilon, multiplicity in spec["one_d"]:
            if len(weight) != len(moduli):
                raise ValueError("each character needs one weight per cyclic factor")
            if epsilon not in (-1, 1) or int(multiplicity) < 1:
                raise ValueError("one-dimensional data require epsilon = +/-1 and positive multiplicity")
            if any((2 * int(w)) % modulus for modulus, w in zip(moduli, weight, strict=True)):
                raise ValueError(f"one-dimensional weight {weight} is not self-inverse")
        for weight, multiplicity in spec["two_d"]:
            if len(weight) != len(moduli) or int(multiplicity) < 1:
                raise ValueError("two-dimensional data require a full weight and positive multiplicity")
            if not any((2 * int(w)) % modulus for modulus, w in zip(moduli, weight, strict=True)):
                raise ValueError(f"two-dimensional weight {weight} is self-inverse")
        dimension = sum(mult for _, _, mult in spec["one_d"]) + 2 * sum(
            mult for _, mult in spec["two_d"]
        )
        if dimension < 1:
            raise ValueError("the representation must have positive dimension")
        return moduli, dimension

    def block_diagonal(blocks):
        dimension = sum(block.shape[0] for block in blocks)
        result = np.zeros((dimension, dimension), dtype=complex)
        offset = 0
        for block in blocks:
            width = block.shape[0]
            result[offset : offset + width, offset : offset + width] = block
            offset += width
        return result

    def matrix_for_element(spec, element, reflected=False):
        moduli, _ = validate_spec(spec)
        blocks = []
        for weight, epsilon, multiplicity in spec["one_d"]:
            scalar = character_value(moduli, weight, element)
            if reflected:
                scalar *= epsilon
            blocks.extend(np.array([[scalar]], dtype=complex) for _ in range(multiplicity))
        for weight, multiplicity in spec["two_d"]:
            scalar = character_value(moduli, weight, element)
            if reflected:
                block = np.array([[0, scalar], [scalar.conjugate(), 0]], dtype=complex)
            else:
                block = np.diag([scalar, scalar.conjugate()]).astype(complex)
            blocks.extend(block.copy() for _ in range(multiplicity))
        return block_diagonal(blocks)

    def construct_matrix_group(spec):
        moduli, _ = validate_spec(spec)
        elements_a = tuple(product(*(range(modulus) for modulus in moduli)))
        rotations = tuple(matrix_for_element(spec, element) for element in elements_a)
        reflections = tuple(
            matrix_for_element(spec, element, reflected=True) for element in elements_a
        )
        assert len(rotations) + len(reflections) == 2 * prod(moduli)
        return elements_a, rotations, reflections

    def multiply_polynomial_by_linear(polynomial, coefficients, dimension):
        result = {}
        for exponent, value in polynomial.items():
            for row, coefficient in enumerate(coefficients):
                if abs(coefficient) < 1.0e-14:
                    continue
                updated = list(exponent)
                updated[row] += 1
                key = tuple(updated)
                result[key] = result.get(key, 0j) + value * coefficient
        return result

    def symmetric_power_matrix(matrix, degree):
        """Construct Sym^degree(matrix) by expanding images of monomials."""

        dimension = matrix.shape[0]
        basis = weak_compositions(int(degree), dimension)
        row_for = {exponent: index for index, exponent in enumerate(basis)}
        result = np.zeros((len(basis), len(basis)), dtype=complex)
        zero = (0,) * dimension
        for column, source_exponent in enumerate(basis):
            polynomial = {zero: 1.0 + 0j}
            for source_index, power in enumerate(source_exponent):
                for _ in range(power):
                    polynomial = multiply_polynomial_by_linear(
                        polynomial, matrix[:, source_index], dimension
                    )
            for target_exponent, coefficient in polynomial.items():
                result[row_for[target_exponent], column] = coefficient
        return result

    def target_matrix(matrix, partition):
        result = np.ones((1, 1), dtype=complex)
        for degree in partition:
            result = np.kron(result, symmetric_power_matrix(matrix, int(degree)))
        return result

    def complete_homogeneous(eigenvalues, degree):
        coefficients = np.zeros(int(degree) + 1, dtype=complex)
        coefficients[0] = 1
        for eigenvalue in eigenvalues:
            updated = np.zeros_like(coefficients)
            for total in range(int(degree) + 1):
                updated[total] = sum(
                    coefficients[total - power] * eigenvalue**power
                    for power in range(total + 1)
                )
            coefficients = updated
        return coefficients[int(degree)]

    def trace_product_from_eigenvalues(eigenvalues, partition):
        return np.prod(
            [complete_homogeneous(eigenvalues, degree) for degree in partition],
            dtype=complex,
        )

    def master_formula(spec, partition):
        """Coefficient predicted by the rotation/reflection master formula."""

        moduli, _ = validate_spec(spec)
        elements_a = tuple(product(*(range(modulus) for modulus in moduli)))
        rotation_values = []
        reflection_values = []
        for element in elements_a:
            rotation_eigenvalues = []
            reflection_eigenvalues = []
            for weight, epsilon, multiplicity in spec["one_d"]:
                scalar = character_value(moduli, weight, element)
                rotation_eigenvalues.extend([scalar] * multiplicity)
                reflection_eigenvalues.extend([epsilon * scalar] * multiplicity)
            for weight, multiplicity in spec["two_d"]:
                scalar = character_value(moduli, weight, element)
                rotation_eigenvalues.extend([scalar, scalar.conjugate()] * multiplicity)
                reflection_eigenvalues.extend([1, -1] * multiplicity)
            rotation_values.append(
                trace_product_from_eigenvalues(rotation_eigenvalues, partition)
            )
            reflection_values.append(
                trace_product_from_eigenvalues(reflection_eigenvalues, partition)
            )
        rotation_average = sum(rotation_values) / len(elements_a)
        reflection_average = sum(reflection_values) / len(elements_a)
        return {
            "rotation average": rotation_average,
            "reflection average": reflection_average,
            "total": (rotation_average + reflection_average) / 2,
        }

    def direct_projector_check(spec, partition, dimension_cap=500):
        """Construct the full Reynolds projector on tensor products of Sym powers."""

        _, dimension = validate_spec(spec)
        target_dimension = prod(
            len(weak_compositions(int(degree), dimension)) for degree in partition
        )
        if target_dimension > dimension_cap:
            raise ValueError(
                f"target dimension {target_dimension} exceeds safety cap {dimension_cap}"
            )
        _, rotations, reflections = construct_matrix_group(spec)
        matrices = rotations + reflections
        projector = sum(
            (target_matrix(matrix, partition) for matrix in matrices),
            start=np.zeros((target_dimension, target_dimension), dtype=complex),
        ) / len(matrices)
        singular_values = np.linalg.svd(projector, compute_uv=False)
        return {
            "group order": len(matrices),
            "representation dimension": dimension,
            "target dimension": target_dimension,
            "rank": int(np.count_nonzero(singular_values > TOLERANCE)),
            "trace": np.trace(projector),
            "idempotence error": float(np.linalg.norm(projector @ projector - projector)),
        }

    def relation_error(spec):
        moduli, dimension = validate_spec(spec)
        identity_element = (0,) * len(moduli)
        reflection = matrix_for_element(spec, identity_element, reflected=True)
        identity = np.eye(dimension, dtype=complex)
        errors = [np.linalg.norm(reflection @ reflection - identity)]
        for factor, modulus in enumerate(moduli):
            generator = [0] * len(moduli)
            generator[factor] = 1
            inverse = [0] * len(moduli)
            inverse[factor] = modulus - 1
            rotation = matrix_for_element(spec, tuple(generator))
            inverse_rotation = matrix_for_element(spec, tuple(inverse))
            errors.append(
                np.linalg.norm(reflection @ rotation @ reflection - inverse_rotation)
            )
        return float(max(errors))

    def parse_partition(text):
        stripped = text.strip()
        if not stripped:
            return ()
        partition = tuple(int(item.strip()) for item in stripped.split(","))
        if any(value < 1 for value in partition):
            raise ValueError("partition parts must be positive")
        return partition

    return (
        TOLERANCE,
        construct_matrix_group,
        direct_projector_check,
        master_formula,
        matrix_for_element,
        parse_partition,
        relation_error,
        validate_spec,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Generalized-dihedral master theorem: direct matrix verification

    Let \(G=\operatorname{Dih}(A)=A\rtimes\langle s\rangle\), with
    \(sas=a^{-1}\).  This notebook constructs the full matrices of a complex
    representation

    \[
    V=\bigoplus_{\eta^2=1,\,\epsilon=\pm1}
      L_{\eta,\epsilon}^{\oplus c_{\eta,\epsilon}}
      \oplus
      \bigoplus_{\chi\ne\chi^{-1}}
      \rho_\chi^{\oplus m_\chi},
    \]

    and tests the coefficient of \(m_\tau\) in the proposed refined Molien
    series.  The two calculations are intentionally independent:

    1. **Direct matrix route:** construct every \(\rho(g)\), expand its action on
       \(W_\tau=\bigotimes_b\operatorname{Sym}^{\tau_b}V\), average the full
       matrices, and compute the Reynolds projector's rank.
    2. **Master-formula route:** split \(G=A\sqcup As\), use the stated
       rotation and reflection eigenvalue multisets, and extract the complete
       homogeneous coefficient in each \(t_b\).

    Character weights for \(A=C_{n_1}\times\cdots\times C_{n_r}\) are stored as
    residue vectors.  Abstract group elements are retained with multiplicity
    even when the representation has a kernel.
    """)
    return


@app.cell
def _():
    representative_cases = (
        {
            "name": "D5 standard 2D irrep",
            "moduli": (5,),
            "one_d": (),
            "two_d": (((1,), 1),),
            "partitions": ((), (1,), (2,), (3,), (2, 1), (2, 2)),
        },
        {
            "name": "D6 mixed even-order representation",
            "moduli": (6,),
            "one_d": (((0,), 1, 1), ((3,), -1, 1)),
            "two_d": (((1,), 1),),
            "partitions": ((), (1,), (2,), (3,), (1, 1), (2, 1), (2, 2)),
        },
        {
            "name": "Dih(C2 x C2), self-inverse characters only",
            "moduli": (2, 2),
            "one_d": (
                ((0, 0), 1, 1),
                ((1, 0), -1, 1),
                ((0, 1), 1, 1),
                ((1, 1), -1, 2),
            ),
            "two_d": (),
            "partitions": ((), (1,), (2,), (3,), (1, 1), (2, 1)),
        },
        {
            "name": "Dih(C3 x C4), mixed 1D and 2D blocks",
            "moduli": (3, 4),
            "one_d": (((0, 0), -1, 1), ((0, 2), 1, 1)),
            "two_d": (((1, 1), 1),),
            "partitions": ((), (1,), (2,), (3,), (1, 1), (2, 1), (2, 2)),
        },
        {
            "name": "D5 with a multiplicity-two 2D block",
            "moduli": (5,),
            "one_d": (),
            "two_d": (((1,), 2),),
            "partitions": ((), (1,), (2,), (3,), (1, 1), (2, 1), (2, 2)),
        },
    )
    return (representative_cases,)


@app.cell
def _(
    TOLERANCE,
    direct_projector_check,
    master_formula,
    relation_error,
    representative_cases,
):
    suite_rows = []
    for case in representative_cases:
        case_relation_error = relation_error(case)
        for case_partition in case["partitions"]:
            direct = direct_projector_check(case, case_partition)
            predicted = master_formula(case, case_partition)
            integer_prediction = int(round(predicted["total"].real))
            discrepancy = max(
                abs(direct["trace"] - predicted["total"]),
                abs(predicted["total"] - integer_prediction),
                direct["idempotence error"],
                case_relation_error,
            )
            passed = (
                direct["rank"] == integer_prediction
                and abs(predicted["total"].imag) < TOLERANCE
                and discrepancy < TOLERANCE
            )
            suite_rows.append(
                {
                    "case": case["name"],
                    "tau": str(case_partition),
                    "|G|": direct["group order"],
                    "dim V": direct["representation dimension"],
                    "dim W_tau": direct["target dimension"],
                    "projector rank": direct["rank"],
                    "formula": integer_prediction,
                    "max error": discrepancy,
                    "passed": passed,
                }
            )
    suite_passed = all(row["passed"] for row in suite_rows)
    suite_max_error = max(row["max error"] for row in suite_rows)
    suite_max_target_dimension = max(row["dim W_tau"] for row in suite_rows)
    return suite_max_error, suite_max_target_dimension, suite_passed, suite_rows


@app.cell
def _(mo, suite_max_error, suite_max_target_dimension, suite_passed, suite_rows):
    suite_status = mo.callout(
        (
            f"{'All' if suite_passed else 'Not all'} {len(suite_rows)} checks passed; "
            f"largest target dimension {suite_max_target_dimension}; "
            f"maximum numerical error {suite_max_error:.3g}."
        ),
        kind="success" if suite_passed else "danger",
    )
    mo.vstack(
        [
            mo.md("## Automated representative suite"),
            suite_status,
            mo.ui.table(suite_rows, pagination=True, page_size=12),
        ]
    )
    return


@app.cell
def _(mo, representative_cases):
    case_names = [case["name"] for case in representative_cases]
    selected_name = mo.ui.dropdown(
        options=case_names, value=case_names[0], label="representation"
    )
    selected_partition_text = mo.ui.text(value="2, 1", label="partition tau")
    mo.hstack([selected_name, selected_partition_text], justify="start", gap=1)
    return selected_name, selected_partition_text


@app.cell
def _(
    TOLERANCE,
    construct_matrix_group,
    direct_projector_check,
    master_formula,
    mo,
    np,
    parse_partition,
    relation_error,
    representative_cases,
    selected_name,
    selected_partition_text,
):
    try:
        selected_case = next(
            case for case in representative_cases if case["name"] == selected_name.value
        )
        selected_partition = parse_partition(selected_partition_text.value)
        selected_direct = direct_projector_check(selected_case, selected_partition)
        selected_prediction = master_formula(selected_case, selected_partition)
        selected_integer = int(round(selected_prediction["total"].real))
        selected_relation_error = relation_error(selected_case)
        selected_error = max(
            abs(selected_direct["trace"] - selected_prediction["total"]),
            abs(selected_prediction["total"] - selected_integer),
            selected_direct["idempotence error"],
            selected_relation_error,
        )
        selected_passed = (
            selected_direct["rank"] == selected_integer
            and selected_error < TOLERANCE
        )
        _, selected_rotations, selected_reflections = construct_matrix_group(selected_case)
        selected_matrix_text = (
            "rho(a) for the first nonidentity a:\n"
            + np.array2string(
                selected_rotations[min(1, len(selected_rotations) - 1)],
                precision=3,
                suppress_small=True,
            )
            + "\n\nrho(s):\n"
            + np.array2string(selected_reflections[0], precision=3, suppress_small=True)
        )
        selected_result = mo.vstack(
            [
                mo.callout(
                    (
                        f"projector rank = {selected_direct['rank']}; formula = "
                        f"{selected_prediction['total'].real:.12g}"
                        f"{selected_prediction['total'].imag:+.2g}i; rotation half-average "
                        f"= {selected_prediction['rotation average'].real / 2:.12g}; "
                        f"reflection half-average = "
                        f"{selected_prediction['reflection average'].real / 2:.12g}; "
                        f"maximum error = {selected_error:.3g}."
                    ),
                    kind="success" if selected_passed else "danger",
                ),
                mo.ui.tabs(
                    {
                        "Dimensions": mo.ui.table([selected_direct]),
                        "Representative matrices": mo.md(
                            "    " + selected_matrix_text.replace("\n", "\n    ")
                        ),
                    }
                ),
            ]
        )
    except (ValueError, StopIteration) as selected_exception:
        selected_result = mo.callout(str(selected_exception), kind="danger")

    mo.vstack([mo.md("## Interactive case check"), selected_result])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What this catches

    The relation check verifies \(s^2=1\) and \(s\rho(a)s=\rho(a^{-1})\) on
    generators.  The matrix route detects errors in the representation,
    normalization, tensor-product action, and invariant dimension.  The
    formula route separately detects mistakes in the \(A/As\) split, the
    multiplicities \(c_{\eta,\epsilon}\) and \(m_\chi\), and the fact that every
    two-dimensional reflection block contributes eigenvalues \(+1,-1\),
    independent of \(a\).

    For \(d=\dim V\), the direct target has dimension

    \[
      \dim W_\tau=\prod_b {\,\tau_b+d-1\choose\tau_b}.
    \]

    The full projector therefore grows quadratically in this number.  The
    notebook uses a safety cap of 500 and keeps the automated examples at or
    below 100; larger degrees should use the master formula without explicitly
    materializing the Reynolds projector.
    """)
    return


if __name__ == "__main__":
    app.run()
