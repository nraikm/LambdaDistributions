"""Exact matrix checks for H wreath S_n and its stable sigma-MGF."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import permutations, product
from math import factorial, gcd


Matrix = tuple[tuple[int, ...], ...]
Tau = tuple[int, ...]


@dataclass(frozen=True)
class BaseGroup:
    name: str
    matrices: tuple[Matrix, ...]

    @property
    def dimension(self) -> int:
        return len(self.matrices[0])


@dataclass(frozen=True)
class WreathCheck:
    base_name: str
    base_order: int
    base_dimension: int
    n: int
    group_order: int
    tau: Tau
    direct_matrix_average: int
    cycle_product_formula: int
    stable_formula: int | None

    @property
    def finite_passed(self) -> bool:
        return self.direct_matrix_average == self.cycle_product_formula

    @property
    def stable_passed(self) -> bool:
        return self.stable_formula is None or self.direct_matrix_average == self.stable_formula

    @property
    def passed(self) -> bool:
        return self.finite_passed and self.stable_passed


def identity(size: int) -> Matrix:
    return tuple(
        tuple(1 if row == column else 0 for column in range(size))
        for row in range(size)
    )


def matmul(left: Matrix, right: Matrix) -> Matrix:
    size = len(left)
    return tuple(
        tuple(
            sum(left[row][middle] * right[middle][column] for middle in range(size))
            for column in range(size)
        )
        for row in range(size)
    )


def trace(matrix: Matrix) -> int:
    return sum(matrix[index][index] for index in range(len(matrix)))


def complete_characters(matrix: Matrix, max_degree: int) -> tuple[Fraction, ...]:
    """Characters h_d from Newton's identity d h_d=sum p_k h_(d-k)."""
    powers = []
    current = identity(len(matrix))
    for _ in range(max_degree):
        current = matmul(current, matrix)
        powers.append(trace(current))
    complete = [Fraction(1)]
    for degree in range(1, max_degree + 1):
        complete.append(
            sum(Fraction(powers[k - 1]) * complete[degree - k] for k in range(1, degree + 1))
            / degree
        )
    return tuple(complete)


def determinant_polynomial(matrix: Matrix) -> tuple[Fraction, ...]:
    """Coefficients of det(I-tM), computed from traces rather than eigenvalues."""
    size = len(matrix)
    powers = []
    current = identity(size)
    for _ in range(size):
        current = matmul(current, matrix)
        powers.append(trace(current))
    elementary = [Fraction(1)]
    for degree in range(1, size + 1):
        elementary.append(
            sum(
                Fraction((-1) ** (k - 1) * powers[k - 1]) * elementary[degree - k]
                for k in range(1, degree + 1)
            )
            / degree
        )
    return tuple(Fraction((-1) ** degree) * elementary[degree] for degree in range(size + 1))


def c2_sign() -> BaseGroup:
    return BaseGroup("C2 sign representation on Q", (((1,),), ((-1,),)))


def c3_rational() -> BaseGroup:
    rotation: Matrix = ((0, -1), (1, -1))
    return BaseGroup(
        "C3 rational 2D representation", (identity(2), rotation, matmul(rotation, rotation))
    )


BASE_GROUPS = {"C2 sign": c2_sign, "C3 rational 2D": c3_rational}


def block_monomial_matrix(
    base: BaseGroup, permutation: tuple[int, ...], labels: tuple[int, ...]
) -> Matrix:
    block_size = base.dimension
    n = len(permutation)
    size = n * block_size
    entries = [[0] * size for _ in range(size)]
    for column_block, row_block in enumerate(permutation):
        block = base.matrices[labels[column_block]]
        for row in range(block_size):
            for column in range(block_size):
                entries[row_block * block_size + row][column_block * block_size + column] = block[row][column]
    return tuple(tuple(row) for row in entries)


def wreath_group(base: BaseGroup, n: int) -> tuple[Matrix, ...]:
    matrices = tuple(
        block_monomial_matrix(base, permutation, labels)
        for permutation in permutations(range(n))
        for labels in product(range(len(base.matrices)), repeat=n)
    )
    expected = len(base.matrices) ** n * factorial(n)
    if len(matrices) != expected or len(set(matrices)) != expected:
        raise AssertionError("the wreath representation should be faithful in these cases")
    return matrices


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


@lru_cache(maxsize=None)
def _characters(matrix: Matrix, max_degree: int) -> tuple[Fraction, ...]:
    return complete_characters(matrix, max_degree)


def direct_dimension(matrices: tuple[Matrix, ...], tau: Tau) -> int:
    max_degree = max(tau, default=0)
    total = Fraction(0)
    for matrix in matrices:
        characters = _characters(matrix, max_degree)
        value = Fraction(1)
        for degree in tau:
            value *= characters[degree]
        total += value
    average = total / len(matrices)
    if average.denominator != 1:
        raise ArithmeticError(f"nonintegral invariant dimension {average}")
    return average.numerator


@lru_cache(maxsize=None)
def base_sigma_coefficient(base: BaseGroup, degrees: Tau) -> Fraction:
    max_degree = max(degrees, default=0)
    total = Fraction(0)
    for matrix in base.matrices:
        characters = _characters(matrix, max_degree)
        value = Fraction(1)
        for degree in degrees:
            value *= characters[degree]
        total += value
    return total / len(base.matrices)


def _multiindices(bound: Tau):
    if not bound:
        yield ()
        return
    yield from product(*(range(entry + 1) for entry in bound))


def _convolve(
    left: dict[Tau, Fraction], right: dict[Tau, Fraction], bound: Tau
) -> dict[Tau, Fraction]:
    result: dict[Tau, Fraction] = {}
    for alpha, left_value in left.items():
        for beta, right_value in right.items():
            degree = tuple(a + b for a, b in zip(alpha, beta))
            if all(value <= limit for value, limit in zip(degree, bound)):
                result[degree] = result.get(degree, Fraction(0)) + left_value * right_value
    return result


def formula_dimension(base: BaseGroup, n: int, tau: Tau) -> int:
    """Coefficient recurrence implied by the exact u-generating function."""
    zero = (0,) * len(tau)
    adams: dict[int, dict[Tau, Fraction]] = {}
    for cycle_length in range(1, n + 1):
        polynomial = {}
        for degree in _multiindices(tau):
            if all(entry % cycle_length == 0 for entry in degree):
                reduced = tuple(entry // cycle_length for entry in degree)
                polynomial[degree] = base_sigma_coefficient(base, reduced)
        adams[cycle_length] = polynomial

    coefficients: list[dict[Tau, Fraction]] = [{zero: Fraction(1)}]
    for block_count in range(1, n + 1):
        accumulated: dict[Tau, Fraction] = {}
        for cycle_length in range(1, block_count + 1):
            term = _convolve(
                adams[cycle_length], coefficients[block_count - cycle_length], tau
            )
            for degree, value in term.items():
                accumulated[degree] = accumulated.get(degree, Fraction(0)) + value
        coefficients.append(
            {degree: value / block_count for degree, value in accumulated.items()}
        )
    answer = coefficients[n].get(tau, Fraction(0))
    if answer.denominator != 1:
        raise ArithmeticError(f"finite formula did not cancel: {answer}")
    return answer.numerator


def _common_divisors(degree: Tau) -> tuple[int, ...]:
    common = 0
    for entry in degree:
        common = gcd(common, entry)
    return tuple(divisor for divisor in range(1, common + 1) if common % divisor == 0)


def stable_dimension(base: BaseGroup, tau: Tau) -> int:
    """Coefficient in Exp_sigma(S_H-1), extracted without a finite-n proxy."""
    zero = (0,) * len(tau)
    coefficients: dict[Tau, Fraction] = {zero: Fraction(1)}
    log_coefficients: dict[Tau, Fraction] = {}
    degrees = sorted(_multiindices(tau), key=lambda alpha: (sum(alpha), alpha))
    for degree in degrees:
        if degree == zero:
            continue
        log_coefficients[degree] = sum(
            base_sigma_coefficient(base, tuple(entry // divisor for entry in degree))
            / divisor
            for divisor in _common_divisors(degree)
        )
        weighted = Fraction(0)
        for beta, log_value in log_coefficients.items():
            if all(b <= a for a, b in zip(degree, beta)):
                remainder = tuple(a - b for a, b in zip(degree, beta))
                weighted += sum(beta) * log_value * coefficients.get(remainder, Fraction(0))
        coefficients[degree] = weighted / sum(degree)
    answer = coefficients.get(tau, Fraction(0))
    if answer.denominator != 1:
        raise ArithmeticError(f"stable coefficient did not cancel: {answer}")
    return answer.numerator


def verify_cycle_product_lemma(base: BaseGroup, max_cycle_length: int = 3) -> bool:
    """Check det(I-tg_C)=det(I-t^r h_C) for all small labelings."""
    for cycle_length in range(1, max_cycle_length + 1):
        permutation = tuple((index + 1) % cycle_length for index in range(cycle_length))
        for labels in product(range(len(base.matrices)), repeat=cycle_length):
            cycle_matrix = block_monomial_matrix(base, permutation, labels)
            product_matrix = identity(base.dimension)
            for index in reversed(range(cycle_length)):
                product_matrix = matmul(product_matrix, base.matrices[labels[index]])
            small_polynomial = determinant_polynomial(product_matrix)
            predicted = [Fraction(0)] * (cycle_length * base.dimension + 1)
            for degree, value in enumerate(small_polynomial):
                predicted[degree * cycle_length] = value
            if determinant_polynomial(cycle_matrix) != tuple(predicted):
                return False
    return True


def verify_case(base: BaseGroup, n: int, max_degree: int = 5) -> tuple[WreathCheck, ...]:
    matrices = wreath_group(base, n)
    rows = []
    for tau in partitions_up_to(max_degree):
        direct = direct_dimension(matrices, tau)
        finite = formula_dimension(base, n, tau)
        stable = stable_dimension(base, tau) if sum(tau) <= n else None
        rows.append(
            WreathCheck(
                base.name,
                len(base.matrices),
                base.dimension,
                n,
                len(matrices),
                tau,
                direct,
                finite,
                stable,
            )
        )
    return tuple(rows)


REPRESENTATIVE_CASES = (("C2 sign", 4, 5), ("C3 rational 2D", 3, 5))


def representative_suite() -> tuple[tuple[WreathCheck, ...], dict[str, bool]]:
    lemma_checks = {
        name: verify_cycle_product_lemma(factory()) for name, factory in BASE_GROUPS.items()
    }
    rows = tuple(
        row
        for name, n, max_degree in REPRESENTATIVE_CASES
        for row in verify_case(BASE_GROUPS[name](), n, max_degree)
    )
    return rows, lemma_checks


if __name__ == "__main__":
    rows, lemma_checks = representative_suite()
    for name, passed in lemma_checks.items():
        print(f"{'PASS' if passed else 'FAIL'} cycle-product determinant lemma: {name}")
    for name, n, _ in REPRESENTATIVE_CASES:
        selected = [row for row in rows if row.base_name == BASE_GROUPS[name]().name and row.n == n]
        print(
            f"{'PASS' if all(row.passed for row in selected) else 'FAIL'} "
            f"{name} wr S_{n}: order={selected[0].group_order}, "
            f"finite checks={len(selected)}, stable checks="
            f"{sum(row.stable_formula is not None for row in selected)}"
        )
    passed = all(lemma_checks.values()) and all(row.passed for row in rows)
    raise SystemExit(0 if passed else 1)
