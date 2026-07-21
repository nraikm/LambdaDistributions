"""Exact finite-grid checks of the compact-torus zero-weight formula."""

from __future__ import annotations

from itertools import product

import numpy as np

from for_this_guy.matrix_group_formula_verification.common import (
    direct_dimension,
    partitions_up_to,
    weak_compositions,
)


def torus_grid_matrices(weights: tuple[tuple[int, ...], ...], modulus: int):
    rank = len(weights[0])
    root = np.exp(2j * np.pi / modulus)
    matrices = []
    for angles in product(range(modulus), repeat=rank):
        eigenvalues = [
            root ** sum(angle * weight for angle, weight in zip(angles, vector))
            for vector in weights
        ]
        matrices.append(np.diag(eigenvalues))
    return tuple(matrices)


def zero_weight_count(weights: tuple[tuple[int, ...], ...], tau: tuple[int, ...]) -> int:
    rank = len(weights[0])
    answer = 0
    choices = tuple(tuple(weak_compositions(degree, len(weights))) for degree in tau)
    for rows in product(*choices) if choices else ((),):
        total_weight = [0] * rank
        for row in rows:
            for multiplicity, weight in zip(row, weights):
                for coordinate in range(rank):
                    total_weight[coordinate] += multiplicity * weight[coordinate]
        answer += all(value == 0 for value in total_weight)
    return answer


CASES = (
    ("U(1), weights (-1,1)", ((-1,), (1,)), 5),
    ("U(1), weights (-2,1,3)", ((-2,), (1,), (3,)), 4),
    ("U(1)^2, A2 weights", ((1, 0), (0, 1), (-1, -1)), 4),
)


def verify_case(name: str, weights: tuple[tuple[int, ...], ...], maximum_degree: int):
    bound = maximum_degree * max(abs(coordinate) for weight in weights for coordinate in weight)
    modulus = 2 * bound + 1
    matrices = torus_grid_matrices(weights, modulus)
    rows = []
    for tau in partitions_up_to(maximum_degree):
        direct = direct_dimension(matrices, tau)
        predicted = zero_weight_count(weights, tau)
        rows.append(
            {
                "torus representation": name,
                "tau": str(tau),
                "degree": sum(tau),
                "grid modulus": modulus,
                "grid group order": len(matrices),
                "matrix average": direct,
                "zero-weight arrays": predicted,
                "pass": direct == predicted,
            }
        )
    return tuple(rows)


def run_suite():
    rows = tuple(row for case in CASES for row in verify_case(*case))
    return {
        "rows": rows,
        "checks": len(rows),
        "passed": all(row["pass"] for row in rows),
    }


if __name__ == "__main__":
    result = run_suite()
    print(f"{'PASS' if result['passed'] else 'FAIL'} tori: {result['checks']} checks")
    raise SystemExit(0 if result["passed"] else 1)
