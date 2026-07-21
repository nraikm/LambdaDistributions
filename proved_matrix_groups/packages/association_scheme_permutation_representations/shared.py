"""Independent exact routes for finite permutation-representation checks.

The three coefficient calculations deliberately use different objects:

* actual permutation matrices and Newton's trace recurrence;
* cycle-factor coefficient extraction;
* explicit orbits on tuples of multisets.

This makes the verification stronger than comparing two rewrites of the same
cycle-index program.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import combinations_with_replacement, product
from math import comb, prod

import numpy as np


TAUS = ((1,), (2,), (1, 1), (3,), (2, 1), (1, 1, 1))


def compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    """Return left after right."""

    return tuple(left[right[i]] for i in range(len(right)))


def permutation_power(permutation: tuple[int, ...], exponent: int) -> tuple[int, ...]:
    answer = tuple(range(len(permutation)))
    base = permutation
    while exponent:
        if exponent & 1:
            answer = compose(base, answer)
        base = compose(base, base)
        exponent //= 2
    return answer


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


def divisors(number: int) -> tuple[int, ...]:
    return tuple(value for value in range(1, number + 1) if number % value == 0)


def mobius(number: int) -> int:
    primes = 0
    candidate = 2
    remaining = number
    while candidate * candidate <= remaining:
        if remaining % candidate == 0:
            remaining //= candidate
            primes += 1
            if remaining % candidate == 0:
                return 0
            while remaining % candidate == 0:
                remaining //= candidate
        candidate += 1
    if remaining > 1:
        primes += 1
    return -1 if primes % 2 else 1


def cycles_from_fixed_counts(fixed_count, maximum: int) -> Counter[int]:
    result: Counter[int] = Counter()
    for length in range(1, maximum + 1):
        numerator = sum(
            mobius(length // divisor) * fixed_count(divisor)
            for divisor in divisors(length)
        )
        assert numerator % length == 0
        count = numerator // length
        assert count >= 0
        if count:
            result[length] = count
    return result


def complete_from_cycles(counts: Counter[int], degree: int) -> int:
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


def cycle_formula_coefficient(permutations, tau: tuple[int, ...]) -> int:
    numerator = sum(
        prod(complete_from_cycles(cycle_counts(p), degree) for degree in tau)
        for p in permutations
    )
    assert numerator % len(permutations) == 0
    return numerator // len(permutations)


def permutation_matrix(permutation: tuple[int, ...]) -> np.ndarray:
    matrix = np.zeros((len(permutation), len(permutation)), dtype=np.int64)
    matrix[permutation, np.arange(len(permutation))] = 1
    return matrix


def matrix_symmetric_characters(permutation: tuple[int, ...], max_degree: int) -> tuple[int, ...]:
    """Compute h_d(P) from actual matrix powers and Newton's recurrence."""

    matrix = permutation_matrix(permutation)
    power = np.eye(len(permutation), dtype=np.int64)
    traces = [0]
    for _ in range(max_degree):
        power = power @ matrix
        traces.append(int(np.trace(power)))
    complete = [1]
    for degree in range(1, max_degree + 1):
        numerator = sum(traces[r] * complete[degree - r] for r in range(1, degree + 1))
        assert numerator % degree == 0
        complete.append(numerator // degree)
    return tuple(complete)


def matrix_average_coefficients(permutations, taus=TAUS) -> dict[tuple[int, ...], int]:
    maximum = max(max(tau) for tau in taus)
    characters = [matrix_symmetric_characters(p, maximum) for p in permutations]
    answer = {}
    for tau in taus:
        numerator = sum(prod(character[d] for d in tau) for character in characters)
        assert numerator % len(permutations) == 0
        answer[tau] = numerator // len(permutations)
    return answer


@lru_cache(maxsize=None)
def multisets(size: int, degree: int):
    return tuple(combinations_with_replacement(range(size), degree))


def direct_orbit_count(permutations, tau: tuple[int, ...]) -> int:
    configurations = set(product(*(multisets(len(permutations[0]), d) for d in tau)))
    orbit_count = 0
    while configurations:
        representative = next(iter(configurations))
        orbit = {
            tuple(tuple(sorted(p[x] for x in block)) for block in representative)
            for p in permutations
        }
        configurations.difference_update(orbit)
        orbit_count += 1
    return orbit_count


def numerical_determinant_check(permutations, t_values=(0.037, 0.061)):
    matrix_average = 0.0
    cycle_average = 0.0
    dimension = len(permutations[0])
    identity = np.eye(dimension)
    for permutation in permutations:
        matrix = permutation_matrix(permutation).astype(float)
        matrix_average += prod(
            1.0 / np.linalg.det(identity - t * matrix) for t in t_values
        )
        counts = cycle_counts(permutation)
        cycle_average += prod(
            (1.0 - t**length) ** (-count)
            for t in t_values
            for length, count in counts.items()
        )
    matrix_average /= len(permutations)
    cycle_average /= len(permutations)
    return matrix_average, cycle_average, abs(matrix_average - cycle_average)


def verify_action(name: str, permutations, *, pair_orbits: int | None = None, taus=TAUS):
    permutations = tuple(dict.fromkeys(permutations))
    matrix_values = matrix_average_coefficients(permutations, taus)
    rows = []
    for tau in taus:
        matrix_value = matrix_values[tau]
        cycle_value = cycle_formula_coefficient(permutations, tau)
        orbit_value = direct_orbit_count(permutations, tau)
        low_degree_prediction = (
            1 if tau == (1,) else pair_orbits if tau in {(2,), (1, 1)} else None
        )
        rows.append(
            {
                "action": name,
                "|G image|": len(permutations),
                "|X|": len(permutations[0]),
                "tau": str(tau),
                "matrix average": matrix_value,
                "cycle formula": cycle_value,
                "direct orbits": orbit_value,
                "low-degree prediction": low_degree_prediction,
                "pass": matrix_value == cycle_value == orbit_value
                and (low_degree_prediction is None or matrix_value == low_degree_prediction),
            }
        )
    return rows


def fixed_formula_reconstructs(permutation, predicted_fixed_count) -> bool:
    actual = cycle_counts(permutation)
    maximum = max(actual)
    reconstructed = cycles_from_fixed_counts(predicted_fixed_count, maximum)
    return reconstructed == actual

