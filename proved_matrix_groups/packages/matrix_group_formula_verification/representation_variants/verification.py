"""Checks for exterior-power and regular representations of small groups."""

from __future__ import annotations

from itertools import combinations, permutations
from math import comb, prod

import numpy as np

from for_this_guy.matrix_group_formula_verification.common import (
    coefficient_rows,
    complete_characters,
    projector_check,
)


def permutation_matrix(permutation: tuple[int, ...]) -> np.ndarray:
    matrix = np.zeros((len(permutation), len(permutation)), dtype=complex)
    for column, row in enumerate(permutation):
        matrix[row, column] = 1
    return matrix


def standard_symmetric_group(n: int) -> tuple[np.ndarray, ...]:
    """The (n-1)-dimensional sum-zero reflection representation of S_n."""

    spanning = np.column_stack(
        [np.eye(n)[:, index] - np.eye(n)[:, n - 1] for index in range(n - 1)]
    )
    basis, _ = np.linalg.qr(spanning)
    return tuple(
        basis.conj().T @ permutation_matrix(permutation) @ basis
        for permutation in permutations(range(n))
    )


def exterior_power_matrix(matrix: np.ndarray, degree: int) -> np.ndarray:
    indices = tuple(combinations(range(matrix.shape[0]), degree))
    result = np.zeros((len(indices), len(indices)), dtype=complex)
    for column, source in enumerate(indices):
        for row, target in enumerate(indices):
            result[row, column] = np.linalg.det(matrix[np.ix_(target, source)])
    return result


def exterior_spectral_formula(
    original_matrices: tuple[np.ndarray, ...], exterior_degree: int, tau: tuple[int, ...]
) -> int:
    maximum_degree = max(tau, default=0)
    values = []
    for matrix in original_matrices:
        eigenvalues = np.linalg.eigvals(matrix)
        exterior_eigenvalues = tuple(
            prod(eigenvalues[index] for index in subset)
            for subset in combinations(range(len(eigenvalues)), exterior_degree)
        )
        diagonal = np.diag(exterior_eigenvalues)
        h = complete_characters(diagonal, maximum_degree)
        values.append(prod(h[degree] for degree in tau))
    average = np.mean(values)
    rounded = int(round(float(average.real)))
    if abs(average.imag) > 5e-7 or abs(average.real - rounded) > 5e-7:
        raise ArithmeticError(f"nonintegral exterior spectral result {average}")
    return rounded


def _compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(left[right[index]] for index in range(len(left)))


def regular_symmetric_group(n: int) -> tuple[np.ndarray, ...]:
    labels = tuple(permutations(range(n)))
    position = {label: index for index, label in enumerate(labels)}
    matrices = []
    for left in labels:
        matrix = np.zeros((len(labels), len(labels)), dtype=complex)
        for column, right in enumerate(labels):
            matrix[position[_compose(left, right)], column] = 1
        matrices.append(matrix)
    return tuple(matrices)


def regular_s3_formula(tau: tuple[int, ...]) -> int:
    """Equation (25), using the S3 order counts N_1=1, N_2=3, N_3=2."""

    total = 0
    for order, multiplicity in ((1, 1), (2, 3), (3, 2)):
        cycle_count = 6 // order
        coefficient = 1
        for degree in tau:
            if degree % order:
                coefficient = 0
                break
            quotient = degree // order
            coefficient *= comb(cycle_count + quotient - 1, quotient)
        total += multiplicity * coefficient
    quotient, remainder = divmod(total, 6)
    if remainder:
        raise ArithmeticError("regular formula did not give an integer")
    return quotient


def run_suite(maximum_degree: int = 5):
    standard = standard_symmetric_group(4)
    exterior = tuple(exterior_power_matrix(matrix, 2) for matrix in standard)
    exterior_rows = coefficient_rows(
        exterior,
        lambda tau: exterior_spectral_formula(standard, 2, tau),
        maximum_degree,
    )

    regular = regular_symmetric_group(3)
    regular_rows = coefficient_rows(regular, regular_s3_formula, maximum_degree)
    return {
        "exterior S4": {
            "order": len(exterior),
            "dimension": exterior[0].shape[0],
            "rows": exterior_rows,
            "projector": projector_check(exterior, (2,)),
            "passed": all(row["pass"] for row in exterior_rows),
        },
        "regular S3": {
            "order": len(regular),
            "dimension": regular[0].shape[0],
            "rows": regular_rows,
            "projector": projector_check(regular, (2,)),
            "passed": all(row["pass"] for row in regular_rows),
        },
    }


if __name__ == "__main__":
    suite = run_suite()
    for name, result in suite.items():
        print(
            f"{'PASS' if result['passed'] else 'FAIL'} {name}: "
            f"order={result['order']}, dimension={result['dimension']}, "
            f"coefficients={len(result['rows'])}"
        )
        print("  projector:", result["projector"])
    raise SystemExit(0 if all(result["passed"] for result in suite.values()) else 1)
