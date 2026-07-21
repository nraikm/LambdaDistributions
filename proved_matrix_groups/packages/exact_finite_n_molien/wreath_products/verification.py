"""Direct matrix and cycle-index checks for H wreath S_n."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import permutations, product
from math import factorial

import numpy as np


@dataclass(frozen=True)
class BaseGroup:
    name: str
    matrices: tuple[np.ndarray, ...]

    @property
    def dimension(self) -> int:
        return int(self.matrices[0].shape[0])


@dataclass(frozen=True)
class CaseResult:
    base_name: str
    base_order: int
    block_dimension: int
    n: int
    group_order: int
    max_degree: int
    rows: tuple[dict[str, object], ...]
    sample_matrix: np.ndarray

    @property
    def passed(self) -> bool:
        return all(bool(row["pass"]) for row in self.rows)


def integer_partitions(total: int, largest: int | None = None):
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


def cyclic_scalar(order: int) -> BaseGroup:
    root = np.exp(2j * np.pi / order)
    matrices = tuple(np.array([[root**power]], dtype=complex) for power in range(order))
    return BaseGroup(f"C{order} on C", matrices)


def reflection_c2() -> BaseGroup:
    identity = np.eye(2, dtype=complex)
    reflection = np.diag([1.0, -1.0]).astype(complex)
    return BaseGroup("C2 reflection representation on C^2", (identity, reflection))


def standard_s3() -> BaseGroup:
    matrices: list[np.ndarray] = []
    reflection = np.diag([1.0, -1.0])
    for k in range(3):
        angle = 2 * np.pi * k / 3
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=complex,
        )
        matrices.extend((rotation, rotation @ reflection))
    return BaseGroup("S3 standard representation on C^2", tuple(matrices))


BASE_GROUPS = {
    "C2 scalar": lambda: cyclic_scalar(2),
    "C3 scalar": lambda: cyclic_scalar(3),
    "C2 reflection": reflection_c2,
    "S3 standard": standard_s3,
}


def wreath_group(base: BaseGroup, n: int) -> tuple[np.ndarray, ...]:
    """Construct H wreath S_n on the direct sum of n copies of V."""
    block_size = base.dimension
    matrices: list[np.ndarray] = []
    for permutation in permutations(range(n)):
        for block_indices in product(range(len(base.matrices)), repeat=n):
            matrix = np.zeros((n * block_size, n * block_size), dtype=complex)
            for column_block, row_block in enumerate(permutation):
                row = slice(row_block * block_size, (row_block + 1) * block_size)
                column = slice(column_block * block_size, (column_block + 1) * block_size)
                matrix[row, column] = base.matrices[block_indices[column_block]]
            matrices.append(matrix)
    expected = len(base.matrices) ** n * factorial(n)
    if len(matrices) != expected:
        raise AssertionError((len(matrices), expected))
    return tuple(matrices)


def _complete_characters(matrix: np.ndarray, max_degree: int) -> np.ndarray:
    coefficients = np.zeros(max_degree + 1, dtype=complex)
    coefficients[0] = 1.0
    for eigenvalue in np.linalg.eigvals(matrix):
        for degree in range(1, max_degree + 1):
            coefficients[degree] += eigenvalue * coefficients[degree - 1]
    return coefficients


def direct_dimensions(
    matrices: tuple[np.ndarray, ...], max_degree: int
) -> dict[tuple[int, ...], int]:
    table = np.stack([_complete_characters(matrix, max_degree) for matrix in matrices])
    answer: dict[tuple[int, ...], int] = {}
    for tau in partitions_up_to(max_degree):
        values = np.ones(len(matrices), dtype=complex)
        for degree in tau:
            values *= table[:, degree]
        average = np.mean(values)
        rounded = int(round(float(average.real)))
        if abs(average.imag) > 3e-8 or abs(average.real - rounded) > 3e-8:
            raise ArithmeticError(f"nonintegral numerical average {average} for {tau}")
        answer[tau] = rounded
    return answer


def _base_dimension(base: BaseGroup, degrees: tuple[int, ...]) -> int:
    if not degrees:
        return 1
    max_degree = max(degrees)
    values = []
    for matrix in base.matrices:
        characters = _complete_characters(matrix, max_degree)
        value = 1.0 + 0.0j
        for degree in degrees:
            value *= characters[degree]
        values.append(value)
    average = sum(values) / len(values)
    rounded = int(round(float(average.real)))
    if abs(average.imag) > 3e-8 or abs(average.real - rounded) > 3e-8:
        raise ArithmeticError(f"nonintegral base average {average} for {degrees}")
    return rounded


def _multiindices(bound: tuple[int, ...]):
    if not bound:
        yield ()
        return
    yield from product(*(range(entry + 1) for entry in bound))


def _convolve(
    left: dict[tuple[int, ...], Fraction],
    right: dict[tuple[int, ...], Fraction],
    bound: tuple[int, ...],
) -> dict[tuple[int, ...], Fraction]:
    result: dict[tuple[int, ...], Fraction] = {}
    for left_degree, left_value in left.items():
        for right_degree, right_value in right.items():
            degree = tuple(a + b for a, b in zip(left_degree, right_degree))
            if all(degree[index] <= bound[index] for index in range(len(bound))):
                result[degree] = result.get(degree, Fraction(0)) + left_value * right_value
    return result


def formula_dimension(base: BaseGroup, n: int, tau: tuple[int, ...]) -> int:
    """Use n F_n = sum_{ell=1}^n A_ell F_{n-ell} coefficientwise."""
    zero = (0,) * len(tau)

    @lru_cache(maxsize=None)
    def base_coefficient(degrees: tuple[int, ...]) -> int:
        return _base_dimension(base, degrees)

    a_polynomials: dict[int, dict[tuple[int, ...], Fraction]] = {}
    for ell in range(1, n + 1):
        polynomial: dict[tuple[int, ...], Fraction] = {}
        for degree in _multiindices(tau):
            if all(entry % ell == 0 for entry in degree):
                reduced = tuple(entry // ell for entry in degree)
                polynomial[degree] = Fraction(base_coefficient(reduced))
        a_polynomials[ell] = polynomial

    coefficients: list[dict[tuple[int, ...], Fraction]] = [{zero: Fraction(1)}]
    for block_count in range(1, n + 1):
        accumulated: dict[tuple[int, ...], Fraction] = {}
        for ell in range(1, block_count + 1):
            term = _convolve(
                a_polynomials[ell], coefficients[block_count - ell], tau
            )
            for degree, value in term.items():
                accumulated[degree] = accumulated.get(degree, Fraction(0)) + value
        coefficients.append(
            {degree: value / block_count for degree, value in accumulated.items()}
        )
    answer = coefficients[n].get(tau, Fraction(0))
    if answer.denominator != 1:
        raise ArithmeticError(f"cycle-index coefficient did not cancel: {answer}")
    return answer.numerator


def verify_case(base: BaseGroup, n: int, max_degree: int = 5) -> CaseResult:
    matrices = wreath_group(base, n)
    direct = direct_dimensions(matrices, max_degree)
    rows = []
    for tau in partitions_up_to(max_degree):
        predicted = formula_dimension(base, n, tau)
        observed = direct[tau]
        rows.append(
            {
                "partition tau": "()" if not tau else str(tau),
                "total degree": sum(tau),
                "direct matrix average": observed,
                "cycle-index formula": predicted,
                "pass": observed == predicted,
            }
        )
    dimension = n * base.dimension
    sample = next(
        (matrix for matrix in matrices if not np.allclose(matrix, np.eye(dimension))),
        matrices[0],
    )
    return CaseResult(
        base.name,
        len(base.matrices),
        base.dimension,
        n,
        len(matrices),
        max_degree,
        tuple(rows),
        sample,
    )


REPRESENTATIVE_CASES = (
    ("C2 scalar", 3),
    ("C3 scalar", 2),
    ("C2 reflection", 2),
    ("S3 standard", 2),
)


def representative_suite(max_degree: int = 5) -> tuple[CaseResult, ...]:
    return tuple(
        verify_case(BASE_GROUPS[name](), n, max_degree)
        for name, n in REPRESENTATIVE_CASES
    )


if __name__ == "__main__":
    suite = representative_suite()
    for result in suite:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status} {result.base_name} wreath S_{result.n}, "
            f"order={result.group_order}, checks={len(result.rows)}"
        )
    raise SystemExit(0 if all(result.passed for result in suite) else 1)
