"""Explicit 3 by 3 matrix checks for the finite SU(3) Delta families."""

from __future__ import annotations

from itertools import permutations, product
from math import prod

import numpy as np

from ..common import (
    matrix_average_coefficient,
    numerical_molien_average,
    partitions_through,
    permutation_matrix,
    weak_compositions,
)


S3 = tuple(permutations(range(3)))
IDENTITY = (0, 1, 2)
THREE_CYCLES = ((1, 2, 0), (2, 0, 1))


def sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def diagonal(n: int, a: int, b: int):
    root = np.exp(2j * np.pi / n)
    return np.diag([root**a, root**b, root ** ((-a - b) % n)])


def delta_matrices(n: int, six: bool):
    permutations_to_use = S3 if six else (IDENTITY,) + THREE_CYCLES
    answer = []
    for a, b in product(range(n), repeat=2):
        diag = diagonal(n, a, b)
        for permutation in permutations_to_use:
            lift = sign(permutation) if six else 1
            answer.append(diag @ (lift * permutation_matrix(permutation)))
    return tuple(answer)


def q_count(n: int, tau: tuple[int, ...]) -> int:
    rows = [tuple(weak_compositions(degree, 3)) for degree in tau]
    count = 0
    for array in product(*rows) if rows else [()]:
        totals = [sum(row[column] for row in array) for column in range(3)]
        if totals[0] % n == totals[1] % n == totals[2] % n:
            count += 1
    return count


def r_count(n: int, tau: tuple[int, ...]) -> int:
    choices = []
    for degree in tau:
        choices.append(tuple((degree - 2 * y, y) for y in range(degree // 2 + 1)))
    total = 0
    for selected in product(*choices) if choices else [()]:
        xs = sum(x for x, _ in selected)
        ys = sum(y for _, y in selected)
        if xs % n == ys % n:
            total += (-1) ** xs
    return total


def delta3_formula(n: int, tau: tuple[int, ...]) -> int:
    indicator = int(all(degree % 3 == 0 for degree in tau))
    numerator = q_count(n, tau) + 2 * indicator
    assert numerator % 3 == 0
    return numerator // 3


def delta6_formula(n: int, tau: tuple[int, ...]) -> int:
    indicator = int(all(degree % 3 == 0 for degree in tau))
    numerator = q_count(n, tau) + 2 * indicator + 3 * r_count(n, tau)
    assert numerator % 6 == 0
    return numerator // 6


def formula_numerical_average(n: int, six: bool, t_values=(0.07, 0.11)) -> float:
    root = np.exp(2j * np.pi / n)
    diagonal_average = sum(
        prod(
            1
            / ((1 - root**a * t) * (1 - root**b * t) * (1 - root ** (-a - b) * t))
            for t in t_values
        )
        for a, b in product(range(n), repeat=2)
    ) / n**2
    three_cycle = prod(1 / (1 - t**3) for t in t_values)
    if not six:
        return float((diagonal_average / 3 + 2 * three_cycle / 3).real)
    transposition = sum(
        prod(1 / ((1 - root**a * t**2) * (1 + root ** (-a) * t)) for t in t_values)
        for a in range(n)
    ) / n
    return float((diagonal_average / 6 + three_cycle / 3 + transposition / 2).real)


def run_suite(max_degree: int = 6, n_values=(2, 3, 4, 5)):
    rows = []
    numerical_rows = []
    for n in n_values:
        for six in (False, True):
            family = f"Delta({'6' if six else '3'}*{n}^2)"
            group = delta_matrices(n, six)
            expected_order = (6 if six else 3) * n * n
            assert len(group) == expected_order
            assert max(abs(np.linalg.det(matrix) - 1) for matrix in group) < 2e-12
            formula = delta6_formula if six else delta3_formula
            for tau in partitions_through(max_degree):
                direct = matrix_average_coefficient(group, tau)
                proposed = formula(n, tau)
                rows.append(
                    {
                        "family": family,
                        "n": n,
                        "dimension": 3,
                        "group order": expected_order,
                        "tau": str(tau),
                        "matrix average": direct,
                        "proposed formula": proposed,
                        "pass": direct == proposed,
                    }
                )
            direct_value = numerical_molien_average(group)
            formula_value = formula_numerical_average(n, six)
            numerical_rows.append(
                {
                    "family": family,
                    "direct determinant average": direct_value,
                    "coset formula": formula_value,
                    "absolute error": abs(direct_value - formula_value),
                }
            )
    passed = all(row["pass"] for row in rows) and all(
        row["absolute error"] < 1e-10 for row in numerical_rows
    )
    return {"passed": passed, "rows": rows, "numerical": numerical_rows}


if __name__ == "__main__":
    suite = run_suite()
    print(f"Delta SU(3): {'PASS' if suite['passed'] else 'FAIL'} ({len(suite['rows'])} coefficients)")
    for row in suite["numerical"]:
        print(row)

