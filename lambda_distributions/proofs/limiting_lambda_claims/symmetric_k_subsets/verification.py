"""Exhaustive matrix checks for S_n acting on k-subsets.

The implementation deliberately constructs every induced permutation matrix.
It compares its power traces with the cycle-product formula and compares
small sigma-MGF power-sum numerators with the same formula.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import combinations, permutations
from math import comb, factorial, gcd

import numpy as np


LAMBDA_CASES = ((1,), (2,), (3,), (1, 1), (2, 1), (1, 1, 1))


def compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    """Return left after right."""
    return tuple(left[right[index]] for index in range(len(left)))


def power(permutation: tuple[int, ...], exponent: int) -> tuple[int, ...]:
    result = tuple(range(len(permutation)))
    factor = permutation
    while exponent:
        if exponent & 1:
            result = compose(factor, result)
        factor = compose(factor, factor)
        exponent //= 2
    return result


def cycle_counts(permutation: tuple[int, ...]) -> Counter[int]:
    seen: set[int] = set()
    counts: Counter[int] = Counter()
    for start in range(len(permutation)):
        if start in seen:
            continue
        length = 0
        point = start
        while point not in seen:
            seen.add(point)
            point = permutation[point]
            length += 1
        counts[length] += 1
    return counts


def induced_permutation(
    permutation: tuple[int, ...], subsets: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    position = {subset: index for index, subset in enumerate(subsets)}
    return tuple(
        position[tuple(sorted(permutation[point] for point in subset))]
        for subset in subsets
    )


def permutation_matrix(action: tuple[int, ...]) -> np.ndarray:
    matrix = np.zeros((len(action), len(action)), dtype=np.int8)
    matrix[list(action), np.arange(len(action))] = 1
    return matrix


def formula_fixed_count(permutation: tuple[int, ...], k: int, exponent: int) -> int:
    """Coefficient in formula (5) of the supplied claim."""
    coefficients = [0] * (k + 1)
    coefficients[0] = 1
    for length, multiplicity in cycle_counts(permutation).items():
        if length > k * exponent:
            continue
        divisor = gcd(length, exponent)
        part = length // divisor
        for _ in range(divisor * multiplicity):
            for degree in range(k, part - 1, -1):
                coefficients[degree] += coefficients[degree - part]
    return coefficients[k]


def direct_power_trace(matrix: np.ndarray, exponent: int) -> int:
    return int(np.trace(np.linalg.matrix_power(matrix.astype(np.int64), exponent)))


def construct_matrix_group(n: int, k: int):
    subsets = tuple(combinations(range(n), k))
    records = []
    keys = set()
    for sigma in permutations(range(n)):
        sigma = tuple(sigma)
        action = induced_permutation(sigma, subsets)
        matrix = permutation_matrix(action)
        records.append((sigma, matrix))
        keys.add(matrix.tobytes())
    if len(records) != factorial(n) or len(keys) != factorial(n):
        raise AssertionError("the induced matrices did not form a faithful S_n copy")
    return subsets, records


def verify_case(n: int, k: int) -> dict[str, object]:
    subsets, records = construct_matrix_group(n, k)
    trace_checks = 0
    lambda_direct = {partition: 0 for partition in LAMBDA_CASES}
    lambda_formula = {partition: 0 for partition in LAMBDA_CASES}
    fixed_values: list[int] = []

    for sigma, matrix in records:
        direct_by_exponent = {}
        formula_by_exponent = {}
        for exponent in range(1, 5):
            direct = direct_power_trace(matrix, exponent)
            proposed = formula_fixed_count(sigma, k, exponent)
            if direct != proposed:
                raise AssertionError((n, k, sigma, exponent, direct, proposed))
            direct_by_exponent[exponent] = direct
            formula_by_exponent[exponent] = proposed
            trace_checks += 1
        fixed_values.append(direct_by_exponent[1])
        for partition in LAMBDA_CASES:
            direct_product = 1
            formula_product = 1
            for part in partition:
                direct_product *= direct_by_exponent[part]
                formula_product *= formula_by_exponent[part]
            lambda_direct[partition] += direct_product
            lambda_formula[partition] += formula_product

    order = factorial(n)
    coefficient_rows = []
    for partition in LAMBDA_CASES:
        direct = Fraction(lambda_direct[partition], order)
        proposed = Fraction(lambda_formula[partition], order)
        coefficient_rows.append(
            {
                "lambda": str(partition),
                "direct numerator average": str(direct),
                "formula average": str(proposed),
                "passed": direct == proposed,
            }
        )

    mean = Fraction(sum(fixed_values), order)
    second = Fraction(sum(value * value for value in fixed_values), order)
    variance = second - mean * mean
    expected_variance = min(k, n - k)
    passed = (
        mean == 1
        and variance == expected_variance
        and all(row["passed"] for row in coefficient_rows)
    )
    return {
        "group": f"S_{n} on {k}-subsets",
        "n": n,
        "k": k,
        "dimension": comb(n, k),
        "order": order,
        "matrices": len(records),
        "power-trace checks": trace_checks,
        "mean trace": str(mean),
        "trace variance": str(variance),
        "expected variance": expected_variance,
        "coefficient rows": coefficient_rows,
        "passed": passed,
    }


def run_sweep() -> dict[str, object]:
    cases = [verify_case(n, k) for n, k in ((4, 2), (5, 2), (6, 2), (6, 3))]
    rows = [row for case in cases for row in case["coefficient rows"]]
    result = {
        "cases": cases,
        "coefficient rows": rows,
        "total matrices": sum(int(case["matrices"]) for case in cases),
        "total trace checks": sum(int(case["power-trace checks"]) for case in cases),
        "passed": all(case["passed"] for case in cases),
    }
    if not result["passed"]:
        raise AssertionError("at least one symmetric-group check failed")
    return result


if __name__ == "__main__":
    result = run_sweep()
    for case in result["cases"]:
        print(
            f"{case['group']}: PASS; dim={case['dimension']}, "
            f"|G|={case['order']}, Var={case['trace variance']}"
        )
    print(
        f"PASS: {result['total matrices']} matrices and "
        f"{result['total trace checks']} power traces"
    )
