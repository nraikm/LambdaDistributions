"""Closed formulas extracted from the original experiments."""

from __future__ import annotations

from collections import Counter
from itertools import permutations, product
from math import comb, factorial, prod

import numpy as np

from .core import integer_partitions, normalize_partition, z_partition


def cyclic_character_moment(partition, n: int, k: int = 1) -> int:
    """Formula for ``E[p_lambda]`` in a one-dimensional C_n character."""

    partition = normalize_partition(partition)
    return int((k * sum(partition)) % n == 0)


def cyclic_sigma_monomial_coefficient(
    partition,
    n: int,
    k: int = 1,
) -> int:
    """The candidate cyclic formula as originally stated.

    It predicts ``n`` when ``n`` divides ``k*|tau|`` and zero otherwise.
    For uniform expectation the surviving coefficient is actually ``1``;
    this function intentionally preserves the stated candidate so verification
    can reject it.
    """

    if n < 1:
        raise ValueError("n must be positive")
    return n if (k * sum(normalize_partition(partition))) % n == 0 else 0


def cyclic_real_2d_moment(partition, n: int, k: int = 1) -> int:
    """Formula for the two-dimensional real C_n representation."""

    partition = normalize_partition(partition)
    return sum(
        (k * sum(sign * part for sign, part in zip(signs, partition, strict=True))) % n == 0
        for signs in product((-1, 1), repeat=len(partition))
    )


def dihedral_moment(partition, n: int, k: int = 1) -> float:
    """Formula for ``E[p_lambda]`` in the standard D_n representation."""

    partition = normalize_partition(partition)
    rotation_count = cyclic_real_2d_moment(partition, n, k)
    reflection_term = 2 ** (len(partition) - 1) if partition and all(part % 2 == 0 for part in partition) else 0
    if not partition:
        reflection_term = 0.5
    return rotation_count / 2 + reflection_term


def dicyclic_homogeneous(partition, n: int) -> complex:
    """Formula for ``E[h_lambda]`` in the two-dimensional Dic_n representation."""

    partition = normalize_partition(partition)
    cyclic_sum = 0j
    for exponent in range(2 * n):
        theta = np.pi * exponent / n
        if exponent == 0:
            factors = [degree + 1 for degree in partition]
        elif exponent == n:
            factors = [(degree + 1) * (-1) ** degree for degree in partition]
        else:
            factors = [np.sin((degree + 1) * theta) / np.sin(theta) for degree in partition]
        cyclic_sum += prod(factors)

    coset_term = (-1) ** (sum(partition) // 2) if all(degree % 2 == 0 for degree in partition) else 0
    return (cyclic_sum + 2 * n * coset_term) / (4 * n)


def pauli_homogeneous(partition, n: int) -> float:
    """Molien formula for ``E[h_lambda]`` in the n-qubit Pauli group."""

    partition = normalize_partition(partition)
    if sum(partition) % 4 != 0:
        return 0.0
    dimension = 2**n
    identity_term = prod(comb(dimension + degree - 1, degree) for degree in partition)
    if all(degree % 2 == 0 for degree in partition):
        balanced_term = prod(
            comb(dimension // 2 + degree // 2 - 1, degree // 2)
            for degree in partition
        )
        return float(balanced_term + (identity_term - balanced_term) / 4**n)
    return float(identity_term / 4**n)


def alternating_permutation_moment(partition, n: int) -> float:
    """Cycle-type formula for ``E[p_lambda]`` on the permutation representation of A_n."""

    if n < 1:
        raise ValueError("n must be positive")
    partition = normalize_partition(partition)
    group_order = 1 if n < 2 else factorial(n) // 2
    weighted_sum = 0
    for cycle_type in integer_partitions(n):
        if (n - len(cycle_type)) % 2:
            continue
        multiplicities = Counter(cycle_type)
        class_size = factorial(n) // z_partition(cycle_type)
        value = prod(
            sum(length * count for length, count in multiplicities.items() if degree % length == 0)
            for degree in partition
        )
        weighted_sum += class_size * value
    return weighted_sum / group_order


def _permutation_cycles(permutation: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    seen: set[int] = set()
    cycles = []
    for start in range(len(permutation)):
        if start in seen:
            continue
        cycle = []
        current = start
        while current not in seen:
            seen.add(current)
            cycle.append(current)
            current = permutation[current]
        cycles.append(tuple(cycle))
    return tuple(cycles)


def generalized_symmetric_moment(partition, n: int, level: int) -> complex:
    """Combinatorial formula for the monomial representation of ``C_level wreath S_n``.

    This evaluates traces from colored permutation cycles, independently of the
    explicit matrices used by :func:`generalized_symmetric_group`.
    """

    if n < 1 or level < 1:
        raise ValueError("n and level must be positive")
    partition = normalize_partition(partition)
    roots = np.exp(2j * np.pi * np.arange(level) / level)
    total = 0j
    for permutation in permutations(range(n)):
        cycles = _permutation_cycles(permutation)
        for colors in product(range(level), repeat=n):
            traces = []
            for degree in partition:
                trace = 0j
                for cycle in cycles:
                    length = len(cycle)
                    if degree % length == 0:
                        cycle_phase = prod(roots[colors[index]] for index in cycle)
                        trace += length * cycle_phase ** (degree // length)
                traces.append(trace)
            total += prod(traces)
    return total / (factorial(n) * level**n)
