"""Evaluation of symmetric functions and Lambda-distributions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import prod

import numpy as np
from numpy.typing import ArrayLike

from .core import Partition, normalize_partition, partitions_up_to, z_partition
from .groups import FiniteMatrixGroup


def power_sum_value(matrix: ArrayLike, partition: Iterable[int]) -> complex:
    """Evaluate ``p_partition`` on the eigenvalues of ``matrix`` using traces. Note p_tao(g)=prod r in tao [Tr(g^r)]"""

    matrix_array = np.asarray(matrix, dtype=complex)
    normalized = normalize_partition(partition)
    return complex(prod(np.trace(np.linalg.matrix_power(matrix_array, degree)) for degree in normalized))


def homogeneous_value(matrix: ArrayLike, partition: Iterable[int]) -> complex:
    """Evaluate ``h_partition`` on a matrix spectrum via Newton's recurrence."""

    square = np.asarray(matrix, dtype=complex)
    normalized = normalize_partition(partition)
    if not normalized:
        return 1 + 0j

    maximum = max(normalized)
    power_sums = [0j] + [
        complex(np.trace(np.linalg.matrix_power(square, degree)))
        for degree in range(1, maximum + 1)
    ]
    complete = [1 + 0j]
    for degree in range(1, maximum + 1):
        complete.append(
            sum(power_sums[index] * complete[degree - index] for index in range(1, degree + 1))
            / degree
        )
    return complex(prod(complete[degree] for degree in normalized))


def _average(group: FiniteMatrixGroup, evaluator) -> complex:
    return complex(sum(evaluator(matrix) for matrix in group.elements) / group.order)


def power_sum_moment(group: FiniteMatrixGroup, partition: Iterable[int]) -> complex:
    """Compute ``E[p_partition(X)]`` for a uniform random group element ``X``."""

    normalized = normalize_partition(partition)
    return _average(group, lambda matrix: power_sum_value(matrix, normalized))


def average_homogeneous(group: FiniteMatrixGroup, partition: Iterable[int]) -> complex:
    """Compute ``E[h_partition(X)]`` for a uniform random group element ``X``."""

    normalized = normalize_partition(partition)
    return _average(group, lambda matrix: homogeneous_value(matrix, normalized))


def lambda_distribution(
    group: FiniteMatrixGroup,
    power_sum_expansion: Mapping[Iterable[int], complex],
) -> complex:
    """Evaluate the Lambda-distribution on a power-sum expansion.

    ``power_sum_expansion`` maps a partition ``lambda`` to the coefficient of
    ``p_lambda``.  Linearity then gives ``mu_X(f)``.
    """

    return complex(
        sum(
            coefficient * power_sum_moment(group, partition)
            for partition, coefficient in power_sum_expansion.items()
        )
    )


def sigma_mgf_coefficients(
    group: FiniteMatrixGroup,
    max_degree: int,
) -> dict[Partition, complex]:
    """Return coefficients of the truncated sigma-MGF in the power-sum basis.

    The coefficient of ``p_lambda`` is ``E[p_lambda(X)] / z_lambda``.
    """

    return {
        partition: power_sum_moment(group, partition) / z_partition(partition)
        for partition in partitions_up_to(max_degree)
    }


def scalar_sigma_monomial_coefficient(
    group: FiniteMatrixGroup,
    partition: Iterable[int],
) -> complex:
    """Compute one monomial-basis coefficient for a scalar representation.

    If a group element acts by the scalar ``x``, then

    ``Exp_sigma(x h_1) = sum_d x^d h_d = sum_tau x^|tau| m_tau``.

    Thus the coefficient of ``m_tau`` is the uniform group expectation of
    ``x^|tau|``.
    """

    if group.dimension != 1:
        raise ValueError("monomial scalar sigma-MGF coefficients require a 1D representation")
    degree = sum(normalize_partition(partition))
    total = sum(complex(matrix[0, 0]) ** degree for matrix in group.elements)
    return complex(total / group.order)


def scalar_sigma_monomial_coefficients(
    group: FiniteMatrixGroup,
    max_degree: int,
) -> dict[Partition, complex]:
    """Return scalar sigma-MGF coefficients in the monomial basis."""

    return {
        partition: scalar_sigma_monomial_coefficient(
            group,
            partition,
        )
        for partition in partitions_up_to(max_degree)
    }
