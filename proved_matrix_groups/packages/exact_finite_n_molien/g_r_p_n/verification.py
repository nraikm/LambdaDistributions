"""Exact finite-n checks for the monomial groups G(r,p,n).

The two sides compared here are deliberately independent:

* ``direct_dimensions`` constructs every monomial matrix and averages the
  characters of tensor products of symmetric powers;
* ``formula_dimension`` extracts the coefficient from the multiset formula
  by bounded integer dynamic programming.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from math import factorial

import numpy as np


@dataclass(frozen=True)
class CaseResult:
    r: int
    p: int
    n: int
    max_degree: int
    group_order: int
    rows: tuple[dict[str, object], ...]
    sample_matrix: np.ndarray

    @property
    def passed(self) -> bool:
        return all(bool(row["pass"]) for row in self.rows)


def integer_partitions(total: int, largest: int | None = None):
    """Yield partitions in weakly decreasing order."""
    if total == 0:
        yield ()
        return
    if largest is None or largest > total:
        largest = total
    for first in range(largest, 0, -1):
        for tail in integer_partitions(total - first, first):
            yield (first, *tail)


def partitions_up_to(max_degree: int):
    for total in range(max_degree + 1):
        yield from integer_partitions(total)


def monomial_group(r: int, p: int, n: int) -> tuple[np.ndarray, ...]:
    """Construct G(r,p,n) as concrete n by n complex matrices."""
    if r < 1 or p < 1 or n < 1 or r % p:
        raise ValueError("Require r,n >= 1 and p a positive divisor of r")
    root = np.exp(2j * np.pi / r)
    matrices: list[np.ndarray] = []
    for permutation in permutations(range(n)):
        for exponents in product(range(r), repeat=n):
            if sum(exponents) % p:
                continue
            matrix = np.zeros((n, n), dtype=complex)
            for column, row in enumerate(permutation):
                matrix[row, column] = root ** exponents[column]
            matrices.append(matrix)
    expected_order = (r**n) * factorial(n) // p
    if len(matrices) != expected_order:
        raise AssertionError((len(matrices), expected_order))
    return tuple(matrices)


def _complete_characters(matrix: np.ndarray, max_degree: int) -> np.ndarray:
    """Return h_d(eigenvalues(matrix)) for 0 <= d <= max_degree."""
    coefficients = np.zeros(max_degree + 1, dtype=complex)
    coefficients[0] = 1.0
    for eigenvalue in np.linalg.eigvals(matrix):
        for degree in range(1, max_degree + 1):
            coefficients[degree] += eigenvalue * coefficients[degree - 1]
    return coefficients


def direct_dimensions(
    matrices: tuple[np.ndarray, ...], max_degree: int
) -> dict[tuple[int, ...], int]:
    """Average product characters for every partition through max_degree."""
    character_table = np.stack(
        [_complete_characters(matrix, max_degree) for matrix in matrices]
    )
    answer: dict[tuple[int, ...], int] = {}
    for tau in partitions_up_to(max_degree):
        values = np.ones(len(matrices), dtype=complex)
        for degree in tau:
            values *= character_table[:, degree]
        average = np.mean(values)
        rounded = int(round(float(average.real)))
        if abs(average.imag) > 2e-8 or abs(average.real - rounded) > 2e-8:
            raise ArithmeticError(f"nonintegral numerical average {average} for {tau}")
        answer[tau] = rounded
    return answer


def _bounded_label_vectors(tau: tuple[int, ...]):
    if not tau:
        yield ()
        return
    yield from product(*(range(part + 1) for part in tau))


def formula_dimension(r: int, p: int, n: int, tau: tuple[int, ...]) -> int:
    """Extract [u^n t^tau] sum_q Exp_sigma(u H_q^(r,p))."""
    zero = (0,) * len(tau)
    total_answer = 0
    for q in range(p):
        residue = q * r // p
        labels = [
            label
            for label in _bounded_label_vectors(tau)
            if sum(label) % r == residue
        ]
        # Each label type contributes (1-u*t^label)^(-1).  Processing label
        # types once and choosing a multiplicity counts multisets, i.e. S_n-orbits.
        states: dict[tuple[int, tuple[int, ...]], int] = {(0, zero): 1}
        for label in labels:
            updated: dict[tuple[int, tuple[int, ...]], int] = {}
            for (used, degrees), count in states.items():
                for copies in range(n - used + 1):
                    candidate = tuple(
                        degrees[index] + copies * label[index]
                        for index in range(len(tau))
                    )
                    if all(candidate[index] <= tau[index] for index in range(len(tau))):
                        key = (used + copies, candidate)
                        updated[key] = updated.get(key, 0) + count
            states = updated
        total_answer += states.get((n, tau), 0)
    return total_answer


def verify_case(r: int, p: int, n: int, max_degree: int = 6) -> CaseResult:
    matrices = monomial_group(r, p, n)
    direct = direct_dimensions(matrices, max_degree)
    rows = []
    for tau in partitions_up_to(max_degree):
        predicted = formula_dimension(r, p, n, tau)
        observed = direct[tau]
        rows.append(
            {
                "partition tau": "()" if not tau else str(tau),
                "total degree": sum(tau),
                "direct matrix average": observed,
                "coefficient formula": predicted,
                "pass": observed == predicted,
            }
        )
    sample = next(
        (matrix for matrix in matrices if not np.allclose(matrix, np.eye(n))),
        matrices[0],
    )
    return CaseResult(r, p, n, max_degree, len(matrices), tuple(rows), sample)


REPRESENTATIVE_CASES = (
    (2, 1, 2),  # type B_2/C_2
    (2, 2, 4),  # type D_4; exercises the odd branch at degree four
    (3, 1, 2),  # C_3 wreath S_2
    (4, 2, 2),  # nontrivial determinant constraint
    (4, 4, 2),  # all four common-residue branches
)


def representative_suite(max_degree: int = 6) -> tuple[CaseResult, ...]:
    return tuple(verify_case(r, p, n, max_degree) for r, p, n in REPRESENTATIVE_CASES)


if __name__ == "__main__":
    suite = representative_suite()
    for result in suite:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status} G({result.r},{result.p},{result.n}), "
            f"order={result.group_order}, checks={len(result.rows)}"
        )
    raise SystemExit(0 if all(result.passed for result in suite) else 1)
