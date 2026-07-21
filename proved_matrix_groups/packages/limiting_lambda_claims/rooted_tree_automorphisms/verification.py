"""Exact matrix and plethystic checks for regular rooted trees."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import product, permutations
from math import factorial

import numpy as np


TAUS = ((1,), (2,), (3,), (1, 1), (2, 1), (2, 2))


def cycle_counts(permutation: tuple[int, ...]) -> Counter[int]:
    counts: Counter[int] = Counter()
    seen: set[int] = set()
    for start in range(len(permutation)):
        if start in seen:
            continue
        point = start
        length = 0
        while point not in seen:
            seen.add(point)
            point = permutation[point]
            length += 1
        counts[length] += 1
    return counts


def permutation_matrix(action: tuple[int, ...]) -> np.ndarray:
    matrix = np.zeros((len(action), len(action)), dtype=np.int8)
    matrix[list(action), np.arange(len(action))] = 1
    return matrix


def wreath_group(
    base_group: tuple[tuple[int, ...], ...], branches: int
) -> tuple[tuple[int, ...], ...]:
    """Construct H wr S_branches on branches copies of the base set."""
    base_size = len(base_group[0])
    elements = []
    for top in permutations(range(branches)):
        for internal in product(base_group, repeat=branches):
            action = [0] * (branches * base_size)
            for block in range(branches):
                for point in range(base_size):
                    source = block * base_size + point
                    target = top[block] * base_size + internal[block][point]
                    action[source] = target
            elements.append(tuple(action))
    result = tuple(elements)
    if len(set(result)) != len(result):
        raise AssertionError("wreath construction produced duplicate permutations")
    expected = factorial(branches) * len(base_group) ** branches
    if len(result) != expected:
        raise AssertionError("wreath group order is incorrect")
    return result


def regular_tree_group(branches: int, height: int) -> tuple[tuple[int, ...], ...]:
    group: tuple[tuple[int, ...], ...] = ((0,),)
    for _ in range(height):
        group = wreath_group(group, branches)
    return group


def complete_homogeneous_from_matrix(matrix: np.ndarray, maximum: int) -> list[Fraction]:
    """Use Newton's identity d h_d = sum_{r<=d} p_r h_{d-r}."""
    traces = [0]
    integer_matrix = matrix.astype(np.int64)
    for exponent in range(1, maximum + 1):
        traces.append(int(np.trace(np.linalg.matrix_power(integer_matrix, exponent))))
    h = [Fraction(1)]
    for degree in range(1, maximum + 1):
        h.append(
            sum(Fraction(traces[r]) * h[degree - r] for r in range(1, degree + 1))
            / degree
        )
    return h


def direct_matrix_coefficient(
    matrices: tuple[np.ndarray, ...], tau: tuple[int, ...]
) -> Fraction:
    maximum = max(tau, default=0)
    total = Fraction(0)
    for matrix in matrices:
        h = complete_homogeneous_from_matrix(matrix, maximum)
        value = Fraction(1)
        for degree in tau:
            value *= h[degree]
        total += value
    return total / len(matrices)


def h_from_cycles(action: tuple[int, ...], maximum: int) -> list[int]:
    coefficients = [0] * (maximum + 1)
    coefficients[0] = 1
    for length, multiplicity in cycle_counts(action).items():
        for _ in range(multiplicity):
            for degree in range(length, maximum + 1):
                coefficients[degree] += coefficients[degree - length]
    return coefficients


def molien_polynomial(
    group: tuple[tuple[int, ...], ...], bounds: tuple[int, ...]
) -> dict[tuple[int, ...], Fraction]:
    result: dict[tuple[int, ...], Fraction] = {}
    indices = tuple(product(*(range(bound + 1) for bound in bounds)))
    for action in group:
        h = h_from_cycles(action, max(bounds, default=0))
        for alpha in indices:
            value = 1
            for degree in alpha:
                value *= h[degree]
            result[alpha] = result.get(alpha, Fraction(0)) + value
    return {alpha: value / len(group) for alpha, value in result.items()}


def multiply_polynomials(
    left: dict[tuple[int, ...], Fraction],
    right: dict[tuple[int, ...], Fraction],
    bounds: tuple[int, ...],
) -> dict[tuple[int, ...], Fraction]:
    result: dict[tuple[int, ...], Fraction] = {}
    for alpha, left_value in left.items():
        for beta, right_value in right.items():
            exponent = tuple(a + b for a, b in zip(alpha, beta, strict=True))
            if all(value <= bound for value, bound in zip(exponent, bounds, strict=True)):
                result[exponent] = result.get(exponent, Fraction(0)) + left_value * right_value
    return result


def substitute_power(
    polynomial: dict[tuple[int, ...], Fraction],
    exponent: int,
    bounds: tuple[int, ...],
) -> dict[tuple[int, ...], Fraction]:
    return {
        tuple(exponent * value for value in alpha): coefficient
        for alpha, coefficient in polynomial.items()
        if all(
            exponent * value <= bound
            for value, bound in zip(alpha, bounds, strict=True)
        )
    }


def plethystic_coefficient(
    base_group: tuple[tuple[int, ...], ...], branches: int, tau: tuple[int, ...]
) -> Fraction:
    """Coefficient of [u^branches] exp(sum u^ell M_H(t^ell)/ell)."""
    base_series = molien_polynomial(base_group, tau)
    zero = (0,) * len(tau)
    total = Fraction(0)
    for top in permutations(range(branches)):
        product_series = {zero: Fraction(1)}
        for length, multiplicity in cycle_counts(tuple(top)).items():
            factor = substitute_power(base_series, length, tau)
            for _ in range(multiplicity):
                product_series = multiply_polynomials(product_series, factor, tau)
        total += product_series.get(tau, Fraction(0))
    return total / factorial(branches)


def verify_case(branches: int, height: int) -> dict[str, object]:
    base = regular_tree_group(branches, height - 1)
    group = wreath_group(base, branches)
    matrices = tuple(permutation_matrix(action) for action in group)
    if len({matrix.tobytes() for matrix in matrices}) != len(group):
        raise AssertionError("matrix representation was not faithful")

    rows = []
    for tau in TAUS:
        direct = direct_matrix_coefficient(matrices, tau)
        proposed = plethystic_coefficient(base, branches, tau)
        rows.append(
            {
                "tree": f"{branches}-ary height {height}",
                "tau": str(tau),
                "direct matrix average": str(direct),
                "plethystic formula": str(proposed),
                "passed": direct == proposed,
            }
        )
    result = {
        "tree": f"complete {branches}-ary height-{height}",
        "branches": branches,
        "height": height,
        "leaves/dimension": branches**height,
        "base order": len(base),
        "group order": len(group),
        "matrices": len(matrices),
        "comparisons": len(rows),
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
    }
    if not result["passed"]:
        raise AssertionError(f"failed {result['tree']}")
    return result


def run_sweep() -> dict[str, object]:
    cases = [
        verify_case(2, 1),
        verify_case(2, 2),
        verify_case(2, 3),
        verify_case(3, 1),
        verify_case(3, 2),
    ]
    return {
        "cases": cases,
        "rows": [row for case in cases for row in case["rows"]],
        "total matrices": sum(int(case["matrices"]) for case in cases),
        "passed": all(case["passed"] for case in cases),
    }


if __name__ == "__main__":
    result = run_sweep()
    for case in result["cases"]:
        print(
            f"{case['tree']}: PASS; leaves={case['leaves/dimension']}, "
            f"|G|={case['group order']}"
        )
    print(f"PASS: {result['total matrices']} matrices, {len(result['rows'])} coefficients")
