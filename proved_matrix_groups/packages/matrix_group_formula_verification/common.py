"""Small, transparent linear-algebra tools for the matrix-group notebooks.

The routines deliberately favor auditability over speed.  They construct
symmetric powers inside tensor powers, form Reynolds projectors, and evaluate
complete homogeneous characters from explicit matrices.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import product
from math import comb, prod, sqrt

import numpy as np


@lru_cache(maxsize=None)
def kostka_number(shape: tuple[int, ...], content: tuple[int, ...]) -> int:
    """Count semistandard tableaux of ``shape`` and ``content`` directly.

    This intentionally small backtracking implementation is independent of
    the matrix nullity calculation used in the continuous-group notebooks.
    """

    shape = tuple(part for part in shape if part)
    if sum(shape) != sum(content):
        return 0
    cells = tuple((row, column) for row, width in enumerate(shape) for column in range(width))
    tableau: dict[tuple[int, int], int] = {}
    remaining = list(content)

    def fill(index: int) -> int:
        if index == len(cells):
            return 1
        row, column = cells[index]
        lower = 0
        if column:
            lower = tableau[(row, column - 1)]
        if row and column < shape[row - 1]:
            lower = max(lower, tableau[(row - 1, column)] + 1)
        answer = 0
        for value in range(lower, len(content)):
            if remaining[value]:
                remaining[value] -= 1
                tableau[(row, column)] = value
                answer += fill(index + 1)
                remaining[value] += 1
        tableau.pop((row, column), None)
        return answer

    return fill(0)


def integer_partitions(total: int, largest: int | None = None):
    if total == 0:
        yield ()
        return
    largest = min(total, total if largest is None else largest)
    for first in range(largest, 0, -1):
        for tail in integer_partitions(total - first, first):
            yield (first, *tail)


def partitions_up_to(maximum_degree: int):
    for total in range(maximum_degree + 1):
        yield from integer_partitions(total)


def complete_characters(matrix: np.ndarray, maximum_degree: int) -> np.ndarray:
    """Return h_d(eigenvalues(matrix)) for 0 <= d <= maximum_degree."""

    coefficients = np.zeros(maximum_degree + 1, dtype=complex)
    coefficients[0] = 1
    for eigenvalue in np.linalg.eigvals(matrix):
        for degree in range(1, maximum_degree + 1):
            coefficients[degree] += eigenvalue * coefficients[degree - 1]
    return coefficients


def direct_dimension(matrices: tuple[np.ndarray, ...], tau: tuple[int, ...]) -> int:
    maximum_degree = max(tau, default=0)
    values = []
    for matrix in matrices:
        h = complete_characters(matrix, maximum_degree)
        values.append(prod(h[degree] for degree in tau))
    average = np.mean(values)
    rounded = int(round(float(average.real)))
    if abs(average.imag) > 3e-7 or abs(average.real - rounded) > 3e-7:
        raise ArithmeticError(f"nonintegral average {average} for tau={tau}")
    return rounded


@lru_cache(maxsize=None)
def weak_compositions(total: int, length: int) -> tuple[tuple[int, ...], ...]:
    if length == 1:
        return ((total,),)
    return tuple(
        (first, *tail)
        for first in range(total + 1)
        for tail in weak_compositions(total - first, length - 1)
    )


@lru_cache(maxsize=None)
def symmetric_isometry(dimension: int, degree: int) -> np.ndarray:
    """Embed the normalized occupancy basis of Sym^degree(V) in V^tensor degree."""

    if degree == 0:
        return np.ones((1, 1), dtype=complex)
    compositions = weak_compositions(degree, dimension)
    column = {composition: index for index, composition in enumerate(compositions)}
    words = tuple(product(range(dimension), repeat=degree))
    occupancies = [tuple(word.count(i) for i in range(dimension)) for word in words]
    multiplicities = Counter(occupancies)
    result = np.zeros((dimension**degree, len(compositions)), dtype=complex)
    for row, occupancy in enumerate(occupancies):
        result[row, column[occupancy]] = 1 / sqrt(multiplicities[occupancy])
    return result


def symmetric_power_matrix(matrix: np.ndarray, degree: int) -> np.ndarray:
    if degree == 0:
        return np.ones((1, 1), dtype=complex)
    tensor_power = np.ones((1, 1), dtype=complex)
    for _ in range(degree):
        tensor_power = np.kron(tensor_power, matrix)
    isometry = symmetric_isometry(matrix.shape[0], degree)
    return isometry.conj().T @ tensor_power @ isometry


def target_matrix(matrix: np.ndarray, tau: tuple[int, ...]) -> np.ndarray:
    result = np.ones((1, 1), dtype=complex)
    for degree in tau:
        result = np.kron(result, symmetric_power_matrix(matrix, degree))
    return result


def symmetric_lie_matrix(generator: np.ndarray, degree: int) -> np.ndarray:
    """Derived action of a Lie-algebra matrix on normalized occupancy states."""

    dimension = generator.shape[0]
    states = weak_compositions(degree, dimension)
    position = {state: index for index, state in enumerate(states)}
    result = np.zeros((len(states), len(states)), dtype=complex)
    for column, state in enumerate(states):
        for source in range(dimension):
            if not state[source]:
                continue
            for target in range(dimension):
                coefficient = generator[target, source]
                if not coefficient:
                    continue
                if target == source:
                    amplitude = state[source]
                    new_state = state
                else:
                    amplitude = sqrt(state[source] * (state[target] + 1))
                    updated = list(state)
                    updated[source] -= 1
                    updated[target] += 1
                    new_state = tuple(updated)
                result[position[new_state], column] += coefficient * amplitude
    return result


def target_lie_matrix(generator: np.ndarray, tau: tuple[int, ...]) -> np.ndarray:
    """Derived action on the tensor product of symmetric powers W_tau."""

    factor_dimensions = tuple(len(weak_compositions(degree, generator.shape[0])) for degree in tau)
    total_dimension = prod(factor_dimensions, start=1)
    result = np.zeros((total_dimension, total_dimension), dtype=complex)
    for active, degree in enumerate(tau):
        term = np.ones((1, 1), dtype=complex)
        for index, factor_dimension in enumerate(factor_dimensions):
            factor = (
                symmetric_lie_matrix(generator, degree)
                if index == active
                else np.eye(factor_dimension, dtype=complex)
            )
            term = np.kron(term, factor)
        result += term
    return result


def common_kernel_dimension(
    generators: tuple[np.ndarray, ...],
    tau: tuple[int, ...],
    discrete_constraints: tuple[np.ndarray, ...] = (),
    tolerance: float = 1e-8,
) -> int:
    """Dimension fixed by Lie generators and optional group elements."""

    blocks = [target_lie_matrix(generator, tau) for generator in generators]
    if blocks:
        target_dimension = blocks[0].shape[1]
    elif discrete_constraints:
        target_dimension = target_matrix(discrete_constraints[0], tau).shape[0]
    else:
        raise ValueError("at least one Lie generator or discrete constraint is required")
    identity = np.eye(target_dimension, dtype=complex)
    blocks.extend(target_matrix(matrix, tau) - identity for matrix in discrete_constraints)
    if not blocks:
        return target_dimension
    singular_values = np.linalg.svd(np.vstack(blocks), compute_uv=False)
    rank = int(np.count_nonzero(singular_values > tolerance))
    return target_dimension - rank


def projector_check(
    matrices: tuple[np.ndarray, ...], tau: tuple[int, ...], dimension_cap: int = 3000
) -> dict[str, float | int]:
    representation_dimension = matrices[0].shape[0]
    target_dimension = prod(
        comb(representation_dimension + degree - 1, degree) for degree in tau
    )
    if target_dimension > dimension_cap:
        raise ValueError(f"target dimension {target_dimension} exceeds cap {dimension_cap}")
    projector = sum(
        (target_matrix(matrix, tau) for matrix in matrices),
        start=np.zeros((target_dimension, target_dimension), dtype=complex),
    ) / len(matrices)
    singular_values = np.linalg.svd(projector, compute_uv=False)
    return {
        "group order": len(matrices),
        "representation dimension": representation_dimension,
        "target dimension": target_dimension,
        "projector rank": int(np.count_nonzero(singular_values > 1e-7)),
        "projector trace": float(np.trace(projector).real),
        "idempotence error": float(np.linalg.norm(projector @ projector - projector)),
    }


def coefficient_rows(matrices, formula, maximum_degree: int):
    rows = []
    for tau in partitions_up_to(maximum_degree):
        observed = direct_dimension(matrices, tau)
        predicted = formula(tau)
        rows.append(
            {
                "tau": str(tau),
                "degree": sum(tau),
                "direct matrix average": observed,
                "independent formula": predicted,
                "pass": observed == predicted,
            }
        )
    return tuple(rows)
