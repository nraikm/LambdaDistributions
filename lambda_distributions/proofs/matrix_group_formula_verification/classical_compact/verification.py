"""Direct Lie-generator checks of the SU(n), SO(n), and O(n) formulas."""

from __future__ import annotations

from math import comb, prod

import numpy as np

from lambda_distributions.proofs.matrix_group_formula_verification.common import (
    common_kernel_dimension,
    integer_partitions,
    kostka_number,
)


def sl_generators(n: int) -> tuple[np.ndarray, ...]:
    """A basis of sl_n(C), the complexified Lie algebra of SU(n)."""

    answer = []
    for row in range(n):
        for column in range(n):
            if row != column:
                matrix = np.zeros((n, n), dtype=complex)
                matrix[row, column] = 1
                answer.append(matrix)
    for index in range(n - 1):
        matrix = np.zeros((n, n), dtype=complex)
        matrix[index, index] = 1
        matrix[index + 1, index + 1] = -1
        answer.append(matrix)
    return tuple(answer)


def so_generators(n: int) -> tuple[np.ndarray, ...]:
    answer = []
    for row in range(n):
        for column in range(row + 1, n):
            matrix = np.zeros((n, n), dtype=complex)
            matrix[row, column] = 1
            matrix[column, row] = -1
            answer.append(matrix)
    return tuple(answer)


def reflection(n: int) -> np.ndarray:
    matrix = np.eye(n, dtype=complex)
    matrix[0, 0] = -1
    return matrix


def su_formula(n: int, tau: tuple[int, ...]) -> int:
    total = sum(tau)
    if total % n:
        return 0
    return kostka_number((total // n,) * n, tau)


def orthogonal_formula(n: int, tau: tuple[int, ...], special: bool) -> int:
    """Kostka sum, explicitly restricted to partitions of length <= n."""

    answer = 0
    for shape in integer_partitions(sum(tau)):
        if len(shape) > n:
            continue
        is_even = all(part % 2 == 0 for part in shape)
        is_orientation_twist = len(shape) == n and all(part % 2 == 1 for part in shape)
        if is_even or (special and is_orientation_twist):
            answer += kostka_number(shape, tau)
    return answer


def target_dimension(n: int, tau: tuple[int, ...]) -> int:
    return prod((comb(n + degree - 1, degree) for degree in tau), start=1)


CASES = (
    ("SU(2)", "SU", 2, ((), (1,), (2,), (1, 1), (2, 1), (3, 1), (2, 2))),
    ("SU(3)", "SU", 3, ((), (1,), (2,), (1, 1), (3,), (2, 1), (1, 1, 1), (2, 2))),
    ("SO(2)", "SO", 2, ((), (1,), (2,), (1, 1), (3,), (2, 1), (1, 1, 1, 1))),
    ("O(2)", "O", 2, ((), (1,), (2,), (1, 1), (3,), (2, 1), (1, 1, 1, 1))),
    ("SO(3)", "SO", 3, ((), (1,), (2,), (1, 1), (3,), (2, 1), (1, 1, 1), (2, 2))),
    ("O(3)", "O", 3, ((), (1,), (2,), (1, 1), (3,), (2, 1), (1, 1, 1), (2, 2))),
    ("SO(4)", "SO", 4, ((), (1,), (2,), (1, 1), (1, 1, 1), (1, 1, 1, 1))),
    ("O(4)", "O", 4, ((), (1,), (2,), (1, 1), (1, 1, 1), (1, 1, 1, 1))),
)


def verify_case(name: str, kind: str, n: int, taus: tuple[tuple[int, ...], ...]):
    generators = sl_generators(n) if kind == "SU" else so_generators(n)
    discrete = (reflection(n),) if kind == "O" else ()
    rows = []
    for tau in taus:
        direct = common_kernel_dimension(generators, tau, discrete)
        predicted = su_formula(n, tau) if kind == "SU" else orthogonal_formula(n, tau, kind == "SO")
        rows.append(
            {
                "group": name,
                "tau": str(tau),
                "degree": sum(tau),
                "dim W_tau": target_dimension(n, tau),
                "generator nullity": direct,
                "Kostka formula": predicted,
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
        "largest target": max(row["dim W_tau"] for row in rows),
    }


if __name__ == "__main__":
    result = run_suite()
    print(f"{'PASS' if result['passed'] else 'FAIL'} classical compact: {result['checks']} checks")
    print(f"largest target dimension: {result['largest target']}")
    raise SystemExit(0 if result["passed"] else 1)
