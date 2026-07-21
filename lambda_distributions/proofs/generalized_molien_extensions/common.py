"""Small exact/numerical tools shared by the three verification suites."""

from __future__ import annotations

from collections import Counter
from itertools import permutations
from math import comb, prod

import numpy as np


def integer_partitions(total: int, maximum: int | None = None):
    """Yield integer partitions as decreasing tuples."""

    if total == 0:
        yield ()
        return
    maximum = total if maximum is None else min(total, maximum)
    for first in range(maximum, 0, -1):
        for tail in integer_partitions(total - first, first):
            yield (first,) + tail


def partitions_through(max_degree: int):
    return tuple(
        partition
        for degree in range(max_degree + 1)
        for partition in integer_partitions(degree)
    )


def weak_compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in weak_compositions(total - first, parts - 1):
            yield (first,) + tail


def permutation_matrix(permutation: tuple[int, ...], dtype=complex):
    matrix = np.zeros((len(permutation), len(permutation)), dtype=dtype)
    for column, row in enumerate(permutation):
        matrix[row, column] = 1
    return matrix


def symmetric_power_character(matrix: np.ndarray, degree: int) -> complex:
    """Character of Sym^degree from the eigenvalue generating function."""

    coefficients = np.zeros(degree + 1, dtype=complex)
    coefficients[0] = 1
    for eigenvalue in np.linalg.eigvals(matrix):
        old = coefficients.copy()
        for target in range(1, degree + 1):
            coefficients[target] = old[target] + eigenvalue * coefficients[target - 1]
    return complex(coefficients[degree])


def matrix_average_coefficient(matrices, tau: tuple[int, ...]) -> int:
    value = sum(
        prod(symmetric_power_character(matrix, degree) for degree in tau)
        for matrix in matrices
    ) / len(matrices)
    rounded = int(round(value.real))
    if abs(value - rounded) > 2e-7:
        raise AssertionError(f"coefficient {tau} did not round to an integer: {value}")
    return rounded


def numerical_molien_average(matrices, t_values=(0.07, 0.11)) -> float:
    dimension = matrices[0].shape[0]
    identity = np.eye(dimension, dtype=complex)
    return float(
        (
            sum(
                prod(1 / np.linalg.det(identity - t * matrix) for t in t_values)
                for matrix in matrices
            )
            / len(matrices)
        ).real
    )


def cycle_counts(permutation: tuple[int, ...]) -> Counter[int]:
    seen: set[int] = set()
    answer: Counter[int] = Counter()
    for start in range(len(permutation)):
        if start in seen:
            continue
        current = start
        length = 0
        while current not in seen:
            seen.add(current)
            current = permutation[current]
            length += 1
        answer[length] += 1
    return answer


def complete_from_cycles(counts: Counter[int], degree: int) -> int:
    """[t^degree] product_d (1-t^d)^(-counts[d])."""

    coefficients = [0] * (degree + 1)
    coefficients[0] = 1
    for length, count in counts.items():
        updated = [0] * (degree + 1)
        for old_degree, old_value in enumerate(coefficients):
            for copies in range((degree - old_degree) // length + 1):
                updated[old_degree + copies * length] += old_value * comb(
                    count + copies - 1, copies
                )
        coefficients = updated
    return coefficients[degree]


def all_permutations(n: int):
    return tuple(permutations(range(n)))


def apply_permutation_to_block(block: tuple[int, ...], permutation: tuple[int, ...]):
    return tuple(sorted(permutation[index] for index in block))

