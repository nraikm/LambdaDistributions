"""Small, transparent helpers shared by the grouped verification notebooks."""

from __future__ import annotations

from collections import Counter
from math import prod
from typing import Iterable

import numpy as np


def partitions_up_to(maximum_degree: int) -> tuple[tuple[int, ...], ...]:
    """Integer partitions through ``maximum_degree``, including the empty one."""

    def partitions(total: int, ceiling: int) -> Iterable[tuple[int, ...]]:
        if total == 0:
            yield ()
            return
        for first in range(min(total, ceiling), 0, -1):
            for tail in partitions(total - first, first):
                yield (first, *tail)

    return tuple(
        tau
        for total in range(maximum_degree + 1)
        for tau in partitions(total, total)
    )


def complete_characters(matrix: np.ndarray, maximum_degree: int) -> np.ndarray:
    """Return h_0,...,h_D from the matrix traces via Newton's recurrence."""

    h = np.zeros(maximum_degree + 1, dtype=complex)
    h[0] = 1.0
    powers = np.eye(matrix.shape[0], dtype=complex)
    traces = np.zeros(maximum_degree + 1, dtype=complex)
    for degree in range(1, maximum_degree + 1):
        powers = powers @ matrix
        traces[degree] = np.trace(powers)
        h[degree] = sum(traces[k] * h[degree - k] for k in range(1, degree + 1)) / degree
    return h


def tensor_symmetric_character(matrix: np.ndarray, tau: tuple[int, ...]) -> complex:
    h = complete_characters(matrix, max(tau, default=0))
    return complex(prod(h[degree] for degree in tau))


def direct_matrix_coefficient(
    matrices: tuple[np.ndarray, ...], probabilities: np.ndarray, tau: tuple[int, ...]
) -> complex:
    """Directly average the character of tensor products of symmetric powers."""

    return complex(
        sum(
            probability * tensor_symmetric_character(matrix, tau)
            for matrix, probability in zip(matrices, probabilities, strict=True)
        )
    )


def symmetric_weight_multiplicities(
    weights: tuple[int, ...], degree: int, modulus: int | None = None
) -> Counter[int]:
    """Weight multiplicities in Sym^degree of a diagonal representation."""

    by_degree: list[Counter[int]] = [Counter() for _ in range(degree + 1)]
    by_degree[0][0] = 1
    for weight in weights:
        updated = [Counter(level) for level in by_degree]
        for old_degree in range(degree + 1):
            for old_weight, multiplicity in by_degree[old_degree].items():
                for copies in range(1, degree - old_degree + 1):
                    new_weight = old_weight + copies * weight
                    if modulus is not None:
                        new_weight %= modulus
                    updated[old_degree + copies][new_weight] += multiplicity
        by_degree = updated
    return by_degree[degree]


def tensor_weight_multiplicities(
    weights: tuple[int, ...], tau: tuple[int, ...], modulus: int | None = None
) -> Counter[int]:
    result: Counter[int] = Counter({0: 1})
    for degree in tau:
        factor = symmetric_weight_multiplicities(weights, degree, modulus)
        updated: Counter[int] = Counter()
        for left, left_multiplicity in result.items():
            for right, right_multiplicity in factor.items():
                weight = left + right
                if modulus is not None:
                    weight %= modulus
                updated[weight] += left_multiplicity * right_multiplicity
        result = updated
    return result


def permutation_matrix(permutation: tuple[int, ...]) -> np.ndarray:
    matrix = np.zeros((len(permutation), len(permutation)), dtype=complex)
    for column, row in enumerate(permutation):
        matrix[row, column] = 1.0
    return matrix


def cycle_counts(permutation: tuple[int, ...]) -> Counter[int]:
    seen: set[int] = set()
    counts: Counter[int] = Counter()
    for start in range(len(permutation)):
        if start in seen:
            continue
        length = 0
        cursor = start
        while cursor not in seen:
            seen.add(cursor)
            cursor = permutation[cursor]
            length += 1
        counts[length] += 1
    return counts


def complete_from_cycle_counts(counts: Counter[int], maximum_degree: int) -> tuple[int, ...]:
    """Coefficients of product_d (1-t^d)^(-C_d), by integer DP."""

    coefficients = [0] * (maximum_degree + 1)
    coefficients[0] = 1
    for length, cycle_count in counts.items():
        for _ in range(cycle_count):
            updated = coefficients.copy()
            for old_degree, value in enumerate(coefficients):
                if value == 0:
                    continue
                for copies in range(1, (maximum_degree - old_degree) // length + 1):
                    updated[old_degree + copies * length] += value
            coefficients = updated
    return tuple(coefficients)


def cycle_formula_coefficient(permutation: tuple[int, ...], tau: tuple[int, ...]) -> int:
    h = complete_from_cycle_counts(cycle_counts(permutation), max(tau, default=0))
    return prod(h[degree] for degree in tau)


def clean(value: complex, tolerance: float = 1e-10) -> float | complex:
    if abs(value.imag) <= tolerance:
        return float(value.real)
    return value

