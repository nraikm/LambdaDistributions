"""Exact verification for B_n = C_2^n rtimes S_n acting on F_2^n.

The module deliberately keeps three computations separate:

* literal permutation matrices for the full group;
* direct orbit enumeration on products of multisets of cube vertices;
* the proposed signed-cycle-type formula.

The public ``run_suite`` function is used by both pytest and the Marimo app.
"""

from __future__ import annotations

from collections import Counter, deque
from fractions import Fraction
from functools import lru_cache
from itertools import combinations_with_replacement, permutations, product
from math import comb, factorial, gcd, lcm

import numpy as np


Element = tuple[tuple[int, ...], tuple[int, ...]]
SignedType = tuple[tuple[int, ...], tuple[int, ...]]


@lru_cache(maxsize=None)
def cube_vertices(n: int) -> tuple[tuple[int, ...], ...]:
    return tuple(product((0, 1), repeat=n))


@lru_cache(maxsize=None)
def group_elements(n: int) -> tuple[Element, ...]:
    """All pairs (b, pi), with (pi x)_i = x_{pi(i)}."""

    return tuple(
        (tuple(b), tuple(pi))
        for b in product((0, 1), repeat=n)
        for pi in permutations(range(n))
    )


@lru_cache(maxsize=None)
def action_permutation(n: int, b: tuple[int, ...], pi: tuple[int, ...]) -> tuple[int, ...]:
    vertices = cube_vertices(n)
    position = {vertex: index for index, vertex in enumerate(vertices)}
    return tuple(
        position[tuple(b[i] ^ vertex[pi[i]] for i in range(n))]
        for vertex in vertices
    )


def permutation_matrix(action: tuple[int, ...]) -> np.ndarray:
    """Matrix P satisfying P e_x = e_{g x}."""

    matrix = np.zeros((len(action), len(action)), dtype=np.int64)
    matrix[action, np.arange(len(action))] = 1
    return matrix


def permutation_power(action: tuple[int, ...], exponent: int) -> tuple[int, ...]:
    result = tuple(range(len(action)))
    base = action
    while exponent:
        if exponent & 1:
            result = tuple(base[result[i]] for i in range(len(action)))
        base = tuple(base[base[i]] for i in range(len(action)))
        exponent //= 2
    return result


def action_cycle_counts(action: tuple[int, ...]) -> Counter[int]:
    counts: Counter[int] = Counter()
    seen: set[int] = set()
    for start in range(len(action)):
        if start in seen:
            continue
        current = start
        length = 0
        while current not in seen:
            seen.add(current)
            current = action[current]
            length += 1
        counts[length] += 1
    return counts


def coordinate_cycles(pi: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    cycles = []
    seen: set[int] = set()
    for start in range(len(pi)):
        if start in seen:
            continue
        cycle = []
        current = start
        while current not in seen:
            seen.add(current)
            cycle.append(current)
            current = pi[current]
        cycles.append(tuple(cycle))
    return tuple(cycles)


def signed_cycle_type(b: tuple[int, ...], pi: tuple[int, ...]) -> SignedType:
    positive = [0] * (len(pi) + 1)
    negative = [0] * (len(pi) + 1)
    for cycle in coordinate_cycles(pi):
        parity = sum(b[i] for i in cycle) % 2
        (negative if parity else positive)[len(cycle)] += 1
    return tuple(positive), tuple(negative)


def fixed_vertices_formula(b: tuple[int, ...], pi: tuple[int, ...]) -> int:
    cycles = coordinate_cycles(pi)
    if any(sum(b[i] for i in cycle) % 2 for cycle in cycles):
        return 0
    return 2 ** len(cycles)


def fixed_power_from_type(signed_type: SignedType, exponent: int) -> int:
    positive, negative = signed_type
    dimension = 0
    for length in range(1, len(positive)):
        multiplicity = positive[length] + negative[length]
        if not multiplicity:
            continue
        divisor = gcd(length, exponent)
        if negative[length] and (exponent // divisor) % 2:
            return 0
        dimension += multiplicity * divisor
    return 2**dimension


def mobius(n: int) -> int:
    factors = 0
    remaining = n
    prime = 2
    while prime * prime <= remaining:
        if remaining % prime == 0:
            remaining //= prime
            factors += 1
            if remaining % prime == 0:
                return 0
            while remaining % prime == 0:
                remaining //= prime
        prime += 1
    if remaining > 1:
        factors += 1
    return -1 if factors % 2 else 1


def type_order(signed_type: SignedType) -> int:
    positive, negative = signed_type
    order = 1
    for length in range(1, len(positive)):
        if positive[length]:
            order = lcm(order, length)
        if negative[length]:
            order = lcm(order, 2 * length)
    return order


def cycle_counts_from_type(signed_type: SignedType) -> Counter[int]:
    counts: Counter[int] = Counter()
    for cycle_length in range(1, type_order(signed_type) + 1):
        numerator = sum(
            mobius(cycle_length // divisor)
            * fixed_power_from_type(signed_type, divisor)
            for divisor in range(1, cycle_length + 1)
            if cycle_length % divisor == 0
        )
        value, remainder = divmod(numerator, cycle_length)
        if remainder:
            raise ArithmeticError("Möbius inversion did not produce an integer")
        if value:
            counts[cycle_length] = value
    return counts


def unsigned_stirling_first(n: int, k: int) -> int:
    table = [[0] * (n + 1) for _ in range(n + 1)]
    table[0][0] = 1
    for size in range(1, n + 1):
        for cycles in range(1, size + 1):
            table[size][cycles] = (
                table[size - 1][cycles - 1]
                + (size - 1) * table[size - 1][cycles]
            )
    return table[n][k]


def proposed_fixed_distribution_counts(n: int) -> Counter[int]:
    order = (2**n) * factorial(n)
    counts: Counter[int] = Counter()
    for cycles in range(1, n + 1):
        counts[2**cycles] = (2 ** (n - cycles)) * unsigned_stirling_first(n, cycles)
    counts[0] = order - sum(counts.values())
    return counts


def _signed_types_recursive(n: int, length: int, remaining: int, a: list[int], b: list[int]):
    if length > n:
        if remaining == 0:
            yield tuple(a), tuple(b)
        return
    maximum = remaining // length
    for positive_count in range(maximum + 1):
        for negative_count in range(maximum - positive_count + 1):
            used = length * (positive_count + negative_count)
            a[length] = positive_count
            b[length] = negative_count
            yield from _signed_types_recursive(n, length + 1, remaining - used, a, b)
    a[length] = 0
    b[length] = 0


@lru_cache(maxsize=None)
def signed_types(n: int) -> tuple[SignedType, ...]:
    a = [0] * (n + 1)
    b = [0] * (n + 1)
    return tuple(_signed_types_recursive(n, 1, n, a, b))


def type_probability(signed_type: SignedType) -> Fraction:
    positive, negative = signed_type
    probability = Fraction(1)
    for length in range(1, len(positive)):
        probability /= (
            (2 * length) ** (positive[length] + negative[length])
            * factorial(positive[length])
            * factorial(negative[length])
        )
    return probability


def complete_character(degree: int, fixed_powers: dict[int, int]) -> int:
    values = [1]
    for current_degree in range(1, degree + 1):
        numerator = sum(
            fixed_powers[power] * values[current_degree - power]
            for power in range(1, current_degree + 1)
        )
        value, remainder = divmod(numerator, current_degree)
        if remainder:
            raise ArithmeticError("Newton recurrence did not produce an integer")
        values.append(value)
    return values[degree]


def character_product_from_type(signed_type: SignedType, partition: tuple[int, ...]) -> int:
    max_degree = max(partition, default=0)
    fixed_powers = {
        exponent: fixed_power_from_type(signed_type, exponent)
        for exponent in range(1, max_degree + 1)
    }
    return int(np.prod([complete_character(degree, fixed_powers) for degree in partition]))


def signed_type_coefficient(n: int, partition: tuple[int, ...]) -> Fraction:
    return sum(
        (
            type_probability(signed_type)
            * character_product_from_type(signed_type, partition)
            for signed_type in signed_types(n)
        ),
        start=Fraction(0),
    )


def matrix_coefficient(n: int, partition: tuple[int, ...]) -> Fraction:
    """Direct average using traces of literal 2^n by 2^n matrices."""

    total = 0
    max_degree = max(partition, default=0)
    for b, pi in group_elements(n):
        action = action_permutation(n, b, pi)
        matrix = permutation_matrix(action)
        power = np.eye(len(action), dtype=np.int64)
        fixed_powers = {}
        for exponent in range(1, max_degree + 1):
            power = power @ matrix
            fixed_powers[exponent] = int(np.trace(power))
        total += int(np.prod([complete_character(degree, fixed_powers) for degree in partition]))
    return Fraction(total, len(group_elements(n)))


@lru_cache(maxsize=None)
def generator_actions(n: int) -> tuple[tuple[int, ...], ...]:
    identity = tuple(range(n))
    generators = []
    for coordinate in range(n):
        flip = tuple(int(i == coordinate) for i in range(n))
        generators.append(action_permutation(n, flip, identity))
    for coordinate in range(n - 1):
        swap = list(identity)
        swap[coordinate], swap[coordinate + 1] = swap[coordinate + 1], swap[coordinate]
        generators.append(action_permutation(n, (0,) * n, tuple(swap)))
    return tuple(generators)


def _basis_states(vertex_count: int, partition: tuple[int, ...]):
    blocks = [tuple(combinations_with_replacement(range(vertex_count), degree)) for degree in partition]
    return tuple(product(*blocks))


def _move_state(state, action):
    return tuple(tuple(sorted(action[vertex] for vertex in block)) for block in state)


def direct_orbit_coefficient(n: int, partition: tuple[int, ...]) -> int:
    """Count orbits on the monomial basis of tensor products of symmetric powers."""

    remaining = set(_basis_states(2**n, partition))
    orbits = 0
    generators = generator_actions(n)
    while remaining:
        seed = remaining.pop()
        queue = deque([seed])
        orbits += 1
        while queue:
            state = queue.popleft()
            for action in generators:
                moved = _move_state(state, action)
                if moved in remaining:
                    remaining.remove(moved)
                    queue.append(moved)
    return orbits


def numeric_sigma_mgf(n: int, t_values=(0.037, 0.061)) -> tuple[float, float, float]:
    """Full (not truncated) sigma-MGF by matrices, cycles, and signed types."""

    identity = np.eye(2**n)
    matrix_total = 0.0
    cycle_total = 0.0
    for b, pi in group_elements(n):
        action = action_permutation(n, b, pi)
        matrix = permutation_matrix(action)
        matrix_term = 1.0
        cycle_term = 1.0
        counts = action_cycle_counts(action)
        for value in t_values:
            matrix_term /= float(np.linalg.det(identity - value * matrix))
            for length, multiplicity in counts.items():
                cycle_term *= (1.0 - value**length) ** (-multiplicity)
        matrix_total += matrix_term
        cycle_total += cycle_term
    order = len(group_elements(n))

    type_total = 0.0
    for signed_type in signed_types(n):
        term = 1.0
        counts = cycle_counts_from_type(signed_type)
        for value in t_values:
            for length, multiplicity in counts.items():
                term *= (1.0 - value**length) ** (-multiplicity)
        type_total += float(type_probability(signed_type)) * term
    return matrix_total / order, cycle_total / order, type_total


def verify_dimension(n: int, partitions: tuple[tuple[int, ...], ...]):
    elements = group_elements(n)
    actions = tuple(action_permutation(n, b, pi) for b, pi in elements)
    fixed_counts = Counter(sum(action[index] == index for index in range(len(action))) for action in actions)

    matrix_valid = all(
        np.array_equal(matrix.sum(axis=0), np.ones(2**n, dtype=np.int64))
        and np.array_equal(matrix.sum(axis=1), np.ones(2**n, dtype=np.int64))
        for matrix in (permutation_matrix(action) for action in actions)
    )
    faithful = len(set(actions)) == len(elements)
    fixed_formula = all(
        sum(action[index] == index for index in range(len(action)))
        == fixed_vertices_formula(b, pi)
        for (b, pi), action in zip(elements, actions)
    )

    conditional = True
    for pi in permutations(range(n)):
        cycle_number = len(coordinate_cycles(tuple(pi)))
        values = Counter(
            fixed_vertices_formula(tuple(b), tuple(pi))
            for b in product((0, 1), repeat=n)
        )
        conditional &= values[2**cycle_number] == 2 ** (n - cycle_number)
        conditional &= values[0] == 2**n - 2 ** (n - cycle_number)

    distribution_formula = fixed_counts == proposed_fixed_distribution_counts(n)
    moment_rows = []
    for moment in range(1, 5):
        direct = Fraction(sum(value**moment * count for value, count in fixed_counts.items()), len(elements))
        proposed = Fraction(comb(n + 2 ** (moment - 1) - 1, n))
        moment_rows.append(
            {"n": n, "moment": moment, "direct": int(direct), "formula": int(proposed), "passed": direct == proposed}
        )

    cycle_reconstruction = all(
        action_cycle_counts(action) == cycle_counts_from_type(signed_cycle_type(b, pi))
        for (b, pi), action in zip(elements, actions)
    )
    type_mass = sum((type_probability(kind) for kind in signed_types(n)), start=Fraction(0))

    coefficient_rows = []
    for partition in partitions:
        matrix_value = matrix_coefficient(n, partition)
        orbit_value = direct_orbit_coefficient(n, partition)
        formula_value = signed_type_coefficient(n, partition)
        passed = matrix_value == orbit_value == formula_value and matrix_value.denominator == 1
        coefficient_rows.append(
            {
                "n": n,
                "partition": str(partition),
                "matrix": int(matrix_value),
                "orbits": orbit_value,
                "signed-type formula": int(formula_value),
                "passed": passed,
            }
        )

    matrix_numeric, cycle_numeric, type_numeric = numeric_sigma_mgf(n)
    numeric_error = max(abs(matrix_numeric - cycle_numeric), abs(matrix_numeric - type_numeric))
    positive_probability = Fraction(sum(count for value, count in fixed_counts.items() if value), len(elements))
    proposed_positive_probability = Fraction(comb(2 * n, n), 4**n)

    passed = all(
        (
            matrix_valid,
            faithful,
            fixed_formula,
            conditional,
            distribution_formula,
            cycle_reconstruction,
            type_mass == 1,
            positive_probability == proposed_positive_probability,
            all(row["passed"] for row in moment_rows),
            all(row["passed"] for row in coefficient_rows),
            numeric_error < 1e-11,
        )
    )
    return {
        "n": n,
        "group order": len(elements),
        "matrix dimension": 2**n,
        "faithful permutation matrices": faithful and matrix_valid,
        "conditional fixed law": conditional,
        "fixed distribution": distribution_formula,
        "cycle reconstruction": cycle_reconstruction,
        "signed-type probability mass": str(type_mass),
        "positive probability": str(positive_probability),
        "positive probability formula": str(proposed_positive_probability),
        "moment rows": tuple(moment_rows),
        "coefficient rows": tuple(coefficient_rows),
        "numeric matrix": matrix_numeric,
        "numeric cycle": cycle_numeric,
        "numeric signed type": type_numeric,
        "numeric error": numeric_error,
        "passed": passed,
    }


DEFAULT_PARTITIONS = ((1,), (2,), (1, 1), (3,), (2, 1), (1, 1, 1))


def run_suite(dimensions=(1, 2, 3, 4, 5)):
    dimension_results = tuple(verify_dimension(n, DEFAULT_PARTITIONS) for n in dimensions)
    return {
        "family": "B_n = C_2^n rtimes S_n on F_2^n",
        "dimensions": tuple(dimensions),
        "partitions": DEFAULT_PARTITIONS,
        "results": dimension_results,
        "moment rows": tuple(row for result in dimension_results for row in result["moment rows"]),
        "coefficient rows": tuple(row for result in dimension_results for row in result["coefficient rows"]),
        "passed": all(result["passed"] for result in dimension_results),
    }


if __name__ == "__main__":
    suite = run_suite()
    comparisons = len(suite["moment rows"]) + len(suite["coefficient rows"])
    print(f"{'PASS' if suite['passed'] else 'FAIL'}: {comparisons} exact comparisons")
