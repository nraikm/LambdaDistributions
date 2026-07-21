"""Prime-field construction of PSL(2,q) on the projective line."""

from __future__ import annotations

from collections import Counter
from itertools import combinations_with_replacement, product
from math import gcd, prod

import numpy as np

from ..common import (
    apply_permutation_to_block,
    complete_from_cycles,
    cycle_counts,
    numerical_molien_average,
    partitions_through,
    permutation_matrix,
)


def sl2_matrices(q: int):
    return tuple(
        (a, b, c, d)
        for a, b, c, d in product(range(q), repeat=4)
        if (a * d - b * c) % q == 1
    )


def projective_permutation(matrix: tuple[int, int, int, int], q: int):
    """Act on F_q union {infinity}; infinity is encoded by q."""

    a, b, c, d = matrix
    images = []
    for point in range(q + 1):
        if point == q:
            image = q if c == 0 else a * pow(c, -1, q) % q
        else:
            denominator = (c * point + d) % q
            image = q if denominator == 0 else (a * point + b) * pow(denominator, -1, q) % q
        images.append(image)
    return tuple(images)


def psl2_permutations(q: int):
    return tuple(sorted({projective_permutation(matrix, q) for matrix in sl2_matrices(q)}))


def divisors(value: int):
    return tuple(candidate for candidate in range(1, value + 1) if value % candidate == 0)


def euler_phi(value: int) -> int:
    return sum(gcd(candidate, value) == 1 for candidate in range(1, value + 1))


def type_histogram_formula(q: int):
    """Cycle-type multiplicities in formula (3.3)."""

    d = gcd(2, q - 1)
    l_minus = (q - 1) // d
    l_plus = (q + 1) // d
    histogram: Counter[tuple[tuple[int, int], ...]] = Counter()

    def add(counts, multiplicity):
        histogram[tuple(sorted((length, count) for length, count in counts.items() if count))] += multiplicity

    add({1: q + 1}, 1)
    add({1: 1, q: q // q}, q * q - 1)  # one fixed point and q/p cycles; here p=q
    for order in divisors(l_minus):
        if order > 1:
            add({1: 2, order: (q - 1) // order}, q * (q + 1) * euler_phi(order) // 2)
    for order in divisors(l_plus):
        if order > 1:
            add({order: (q + 1) // order}, q * (q - 1) * euler_phi(order) // 2)
    return histogram


def actual_type_histogram(permutations):
    return Counter(
        tuple(sorted(cycle_counts(permutation).items())) for permutation in permutations
    )


def coefficient_from_histogram(histogram, tau: tuple[int, ...]) -> int:
    numerator = 0
    order = sum(histogram.values())
    for cycle_type, multiplicity in histogram.items():
        counts = Counter(dict(cycle_type))
        numerator += multiplicity * prod(complete_from_cycles(counts, degree) for degree in tau)
    assert numerator % order == 0
    return numerator // order


def direct_orbit_count(permutations, tau: tuple[int, ...]) -> int:
    number_of_points = len(permutations[0])
    block_spaces = [tuple(combinations_with_replacement(range(number_of_points), d)) for d in tau]
    remaining = set(product(*block_spaces) if block_spaces else [()])
    orbit_count = 0
    while remaining:
        representative = next(iter(remaining))
        orbit = {
            tuple(apply_permutation_to_block(block, permutation) for block in representative)
            for permutation in permutations
        }
        remaining.difference_update(orbit)
        orbit_count += 1
    return orbit_count


def claimed_low_degree(q: int):
    d = gcd(2, q - 1)
    return {
        (1,): 1,
        (2,): 2,
        (1, 1): 2,
        (1, 1, 1): 4 + d,
        (1, 1, 1, 1): 8 + d * (q + 4),
    }


def run_suite(q_values=(2, 3, 5, 7), max_degree: int = 4):
    rows = []
    type_rows = []
    numerical_rows = []
    for q in q_values:
        permutations = psl2_permutations(q)
        d = gcd(2, q - 1)
        expected_order = q * (q * q - 1) // d
        assert len(permutations) == expected_order
        actual_types = actual_type_histogram(permutations)
        proposed_types = type_histogram_formula(q)
        type_rows.append(
            {
                "q": q,
                "dimension": q + 1,
                "group order": expected_order,
                "number of cycle types": len(actual_types),
                "type multiplicities agree": actual_types == proposed_types,
            }
        )
        for tau in partitions_through(max_degree):
            orbit_count = direct_orbit_count(permutations, tau)
            actual_cycle = coefficient_from_histogram(actual_types, tau)
            proposed = coefficient_from_histogram(proposed_types, tau)
            low_claim = claimed_low_degree(q).get(tau)
            rows.append(
                {
                    "q": q,
                    "dimension": q + 1,
                    "tau": str(tau),
                    "direct orbit count": orbit_count,
                    "actual cycle average": actual_cycle,
                    "formula (3.3)": proposed,
                    "stated low-degree value": low_claim if low_claim is not None else "-",
                    "pass": orbit_count == actual_cycle == proposed and (low_claim is None or proposed == low_claim),
                }
            )
        matrices = tuple(permutation_matrix(permutation) for permutation in permutations)
        determinant_average = numerical_molien_average(matrices)
        cycle_average = 0.0
        for cycle_type, multiplicity in proposed_types.items():
            value = prod(
                (1 - t**length) ** (-count)
                for t in (0.07, 0.11)
                for length, count in cycle_type
            )
            cycle_average += multiplicity * value
        cycle_average /= expected_order
        numerical_rows.append(
            {
                "q": q,
                "direct permutation-matrix determinant average": determinant_average,
                "formula (3.3) numerical value": cycle_average,
                "absolute error": abs(determinant_average - cycle_average),
            }
        )
    passed = (
        all(row["pass"] for row in rows)
        and all(row["type multiplicities agree"] for row in type_rows)
        and all(row["absolute error"] < 1e-10 for row in numerical_rows)
    )
    return {
        "passed": passed,
        "rows": rows,
        "cycle types": type_rows,
        "numerical": numerical_rows,
    }


if __name__ == "__main__":
    suite = run_suite()
    print(f"PSL2 projective line: {'PASS' if suite['passed'] else 'FAIL'} ({len(suite['rows'])} coefficients)")
    for row in suite["cycle types"]:
        print(row)

