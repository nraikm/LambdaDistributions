from __future__ import annotations

from collections import deque
import hashlib
from itertools import combinations, combinations_with_replacement, product
import json
from math import comb, gcd, lcm

import numpy as np


Permutation = tuple[int, ...]


def permutation_from_cycles(size: int, cycles: tuple[tuple[int, ...], ...]) -> Permutation:
    result = list(range(size))
    for cycle in cycles:
        zero = tuple(value - 1 for value in cycle)
        for source, target in zip(zero, zero[1:] + zero[:1]):
            result[source] = target
    return tuple(result)


def compose(left: Permutation, right: Permutation) -> Permutation:
    """Return left after right."""
    return tuple(left[right[index]] for index in range(len(left)))


def inverse(permutation: Permutation) -> Permutation:
    result = [0] * len(permutation)
    for source, target in enumerate(permutation):
        result[target] = source
    return tuple(result)


def permutation_order(permutation: Permutation) -> int:
    seen: set[int] = set()
    answer = 1
    for start in range(len(permutation)):
        if start in seen:
            continue
        length = 0
        point = start
        while point not in seen:
            seen.add(point)
            length += 1
            point = permutation[point]
        answer = lcm(answer, length)
    return answer


def permutation_matrix(permutation: Permutation) -> np.ndarray:
    matrix = np.zeros((len(permutation), len(permutation)), dtype=np.int8)
    for source, target in enumerate(permutation):
        matrix[target, source] = 1
    return matrix


def m24_generators() -> tuple[Permutation, Permutation]:
    """ATLAS standard generators in the natural 24-point action."""
    a = permutation_from_cycles(
        24,
        (
            (1, 4), (2, 7), (3, 17), (5, 13), (6, 9), (8, 15),
            (10, 19), (11, 18), (12, 21), (14, 16), (20, 24), (22, 23),
        ),
    )
    b = permutation_from_cycles(
        24,
        ((1, 4, 6), (2, 21, 14), (3, 9, 15), (5, 18, 10), (13, 17, 16), (19, 24, 23)),
    )
    return a, b


def word(permutations: dict[str, Permutation], letters: str) -> Permutation:
    result = tuple(range(len(next(iter(permutations.values())))))
    for letter in letters:
        result = compose(permutations[letter], result)
    return result


def apply_to_state(permutation: Permutation, state: tuple[int, ...], sort: bool) -> tuple[int, ...]:
    image = tuple(permutation[value] for value in state)
    return tuple(sorted(image)) if sort else image


def orbit_count(states, generators: tuple[Permutation, ...], sort: bool) -> int:
    remaining = set(states)
    count = 0
    moves = (*generators, *(inverse(generator) for generator in generators))
    while remaining:
        count += 1
        seed = remaining.pop()
        queue = deque([seed])
        while queue:
            current = queue.popleft()
            for generator in moves:
                candidate = apply_to_state(generator, current, sort)
                if candidate in remaining:
                    remaining.remove(candidate)
                    queue.append(candidate)
    return count


def multiset_orbit_count(degree: int) -> int:
    return orbit_count(combinations_with_replacement(range(24), degree), m24_generators(), sort=True)


def tuple_orbit_count(degree: int) -> int:
    return orbit_count(product(range(24), repeat=degree), m24_generators(), sort=False)


def subset_orbit_count(degree: int) -> int:
    return orbit_count(combinations(range(24), degree), m24_generators(), sort=True)


# Conjugacy classes with the same cycle shape are aggregated.  The five AB
# rows each combine two power-conjugate classes, which is sufficient for the
# permutation Molien calculation.
M24_CYCLE_DATA: tuple[tuple[str, int, dict[int, int]], ...] = (
    ("1A", 1, {1: 24}),
    ("2A", 11_385, {1: 8, 2: 8}),
    ("2B", 31_878, {2: 12}),
    ("3A", 226_688, {1: 6, 3: 6}),
    ("3B", 485_760, {3: 8}),
    ("4A", 637_560, {2: 4, 4: 4}),
    ("4B", 1_912_680, {1: 4, 2: 2, 4: 4}),
    ("4C", 2_550_240, {4: 6}),
    ("5A", 4_080_384, {1: 4, 5: 4}),
    ("6A", 10_200_960, {1: 2, 2: 2, 3: 2, 6: 2}),
    ("6B", 10_200_960, {6: 4}),
    ("7AB", 11_658_240, {1: 3, 7: 3}),
    ("8A", 15_301_440, {1: 2, 2: 1, 4: 1, 8: 2}),
    ("10A", 12_241_152, {2: 2, 10: 2}),
    ("11A", 22_256_640, {1: 2, 11: 2}),
    ("12A", 20_401_920, {2: 1, 4: 1, 6: 1, 12: 1}),
    ("12B", 20_401_920, {12: 2}),
    ("14AB", 34_974_720, {1: 1, 2: 1, 7: 1, 14: 1}),
    ("15AB", 32_643_072, {1: 1, 3: 1, 5: 1, 15: 1}),
    ("21AB", 23_316_480, {3: 1, 21: 1}),
    ("23AB", 21_288_960, {1: 1, 23: 1}),
)
M24_ORDER = 244_823_040


def m24_cycle_data_sha256() -> str:
    """Checksum a canonical serialization of the aggregated ATLAS input."""

    payload = json.dumps(
        [(label, size, sorted(shape.items())) for label, size, shape in M24_CYCLE_DATA],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def determinant_inverse_coeff(cycle_shape: dict[int, int], degree: int, sign: int = 1) -> int:
    """Coefficient of prod_r(1-(sign*t)^r)^(-c_r)."""
    coefficients = [0] * (degree + 1)
    coefficients[0] = 1
    for cycle_length, multiplicity in cycle_shape.items():
        updated = [0] * (degree + 1)
        for old_degree, old_value in enumerate(coefficients):
            for copies in range((degree - old_degree) // cycle_length + 1):
                added = copies * cycle_length
                scalar = sign ** added
                updated[old_degree + added] += old_value * comb(multiplicity + copies - 1, copies) * scalar
        coefficients = updated
    return coefficients[degree]


def m24_symmetric_coefficient(degree: int) -> int:
    numerator = sum(
        class_size * determinant_inverse_coeff(shape, degree)
        for _, class_size, shape in M24_CYCLE_DATA
    )
    assert numerator % M24_ORDER == 0
    return numerator // M24_ORDER


def m24_tensor_coefficient(degree: int) -> int:
    numerator = sum(
        class_size * shape.get(1, 0) ** degree
        for _, class_size, shape in M24_CYCLE_DATA
    )
    assert numerator % M24_ORDER == 0
    return numerator // M24_ORDER


def signed_m24_symmetric_coefficient(degree: int) -> int:
    """Coefficient for the 24-dimensional subgroup {+/- P_g}."""
    numerator = sum(
        class_size
        * (
            determinant_inverse_coeff(shape, degree, sign=1)
            + determinant_inverse_coeff(shape, degree, sign=-1)
        )
        for _, class_size, shape in M24_CYCLE_DATA
    )
    denominator = 2 * M24_ORDER
    assert numerator % denominator == 0
    return numerator // denominator


def partition_number(total: int) -> int:
    values = [0] * (total + 1)
    values[0] = 1
    for part in range(1, total + 1):
        for value in range(part, total + 1):
            values[value] += values[value - part]
    return values[total]


def bell_number(total: int) -> int:
    triangle = [[1]]
    for row in range(1, total + 1):
        current = [triangle[-1][-1]]
        for column in range(1, row + 1):
            current.append(current[-1] + triangle[-1][column - 1])
        triangle.append(current)
    return triangle[total][0]
