"""Exact checks for the generalized Molien formula of AGL_n(q).

The implementation deliberately uses only prime fields.  That keeps the
finite-field layer transparent while covering representative dimensions and
field sizes.  Three paths are kept separate:

1. construct every affine permutation and its permutation matrix;
2. enumerate orbits of tuples of multisets literally;
3. extract the proposed cycle-index coefficient by exact integer DP.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement, product
from math import prod


Matrix = tuple[tuple[int, ...], ...]
Permutation = tuple[int, ...]
Counts = tuple[int, ...]
Tau = tuple[int, ...]


@dataclass(frozen=True)
class AffineAction:
    q: int
    n: int
    points: tuple[tuple[int, ...], ...]
    permutations: tuple[Permutation, ...]

    @property
    def order(self) -> int:
        return len(self.permutations)

    @property
    def degree(self) -> int:
        return len(self.points)


@dataclass(frozen=True)
class OrbitCheck:
    q: int
    n: int
    tau: Tau
    group_order: int
    matrix_dimension: int
    configuration_count: int
    direct_orbits: int
    cycle_formula: int

    @property
    def passed(self) -> bool:
        return self.direct_orbits == self.cycle_formula


@dataclass(frozen=True)
class MomentCheck:
    q: int
    n: int
    power: int
    direct: Fraction
    gaussian_formula: int

    @property
    def passed(self) -> bool:
        return self.direct == self.gaussian_formula


def _rank_mod(matrix: Matrix, q: int) -> int:
    work = [list(row) for row in matrix]
    rows = len(work)
    columns = len(work[0]) if rows else 0
    pivot_row = 0
    for column in range(columns):
        pivot = next(
            (row for row in range(pivot_row, rows) if work[row][column] % q),
            None,
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        inverse = pow(work[pivot_row][column] % q, -1, q)
        work[pivot_row] = [(value * inverse) % q for value in work[pivot_row]]
        for row in range(rows):
            if row == pivot_row:
                continue
            scale = work[row][column] % q
            if scale:
                work[row] = [
                    (left - scale * right) % q
                    for left, right in zip(work[row], work[pivot_row])
                ]
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row


def invertible_matrices(n: int, q: int) -> tuple[Matrix, ...]:
    """Enumerate GL_n(q), with q prime."""
    matrices = []
    for entries in product(range(q), repeat=n * n):
        matrix = tuple(
            tuple(entries[row * n + column] for column in range(n))
            for row in range(n)
        )
        if _rank_mod(matrix, q) == n:
            matrices.append(matrix)
    return tuple(matrices)


def affine_action(n: int, q: int) -> AffineAction:
    points = tuple(product(range(q), repeat=n))
    point_index = {point: index for index, point in enumerate(points)}
    permutations = []
    for matrix in invertible_matrices(n, q):
        for translation in points:
            image = []
            for point in points:
                mapped = tuple(
                    (sum(matrix[row][column] * point[column] for column in range(n))
                     + translation[row])
                    % q
                    for row in range(n)
                )
                image.append(point_index[mapped])
            permutations.append(tuple(image))
    if len(set(permutations)) != len(permutations):
        raise AssertionError("the affine action should be faithful")
    return AffineAction(q, n, points, tuple(permutations))


def permutation_matrix(permutation: Permutation) -> Matrix:
    size = len(permutation)
    return tuple(
        tuple(1 if row == permutation[column] else 0 for column in range(size))
        for row in range(size)
    )


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    size = len(left)
    return tuple(
        tuple(
            sum(left[row][middle] * right[middle][column] for middle in range(size))
            for column in range(size)
        )
        for row in range(size)
    )


def _determinant_polynomial(matrix: Matrix) -> tuple[Fraction, ...]:
    """Coefficients of det(I-tM), recovered exactly from matrix traces."""
    size = len(matrix)
    identity_matrix = tuple(
        tuple(1 if row == column else 0 for column in range(size))
        for row in range(size)
    )
    current = identity_matrix
    traces = []
    for _ in range(size):
        current = _matmul(current, matrix)
        traces.append(sum(current[index][index] for index in range(size)))
    elementary = [Fraction(1)]
    for degree in range(1, size + 1):
        elementary.append(
            sum(
                Fraction((-1) ** (power - 1) * traces[power - 1])
                * elementary[degree - power]
                for power in range(1, degree + 1)
            )
            / degree
        )
    return tuple(Fraction((-1) ** degree) * elementary[degree] for degree in range(size + 1))


def _multiply_polynomials(left: list[int], right: list[int]) -> list[int]:
    answer = [0] * (len(left) + len(right) - 1)
    for left_degree, left_value in enumerate(left):
        for right_degree, right_value in enumerate(right):
            answer[left_degree + right_degree] += left_value * right_value
    return answer


def verify_matrix_cycle_identity(action: AffineAction) -> bool:
    for permutation in action.permutations:
        expected = [1]
        for length in cycle_lengths(permutation):
            factor = [0] * (length + 1)
            factor[0], factor[-1] = 1, -1
            expected = _multiply_polynomials(expected, factor)
        if _determinant_polynomial(permutation_matrix(permutation)) != tuple(expected):
            return False
    return True


def cycle_lengths(permutation: Permutation) -> tuple[int, ...]:
    seen = [False] * len(permutation)
    lengths = []
    for start in range(len(permutation)):
        if seen[start]:
            continue
        current = start
        length = 0
        while not seen[current]:
            seen[current] = True
            length += 1
            current = permutation[current]
        lengths.append(length)
    return tuple(sorted(lengths))


def fixed_multisets(permutation: Permutation, size: int) -> int:
    """Coefficient of z^size in product_C (1-z^|C|)^(-1)."""
    coefficients = [0] * (size + 1)
    coefficients[0] = 1
    for length in cycle_lengths(permutation):
        updated = [0] * (size + 1)
        for degree, value in enumerate(coefficients):
            if not value:
                continue
            for copies in range((size - degree) // length + 1):
                updated[degree + copies * length] += value
        coefficients = updated
    return coefficients[size]


def multiset_counts(point_count: int, size: int) -> tuple[Counts, ...]:
    answer = []
    for selection in combinations_with_replacement(range(point_count), size):
        counts = [0] * point_count
        for index in selection:
            counts[index] += 1
        answer.append(tuple(counts))
    return tuple(answer)


def _move_counts(counts: Counts, permutation: Permutation) -> Counts:
    moved = [0] * len(counts)
    for source, count in enumerate(counts):
        moved[permutation[source]] = count
    return tuple(moved)


def direct_orbit_count(action: AffineAction, tau: Tau) -> tuple[int, int]:
    factors = tuple(multiset_counts(action.degree, size) for size in tau)
    configurations = tuple(product(*factors))
    representatives = set()
    for configuration in configurations:
        representative = min(
            tuple(_move_counts(counts, permutation) for counts in configuration)
            for permutation in action.permutations
        )
        representatives.add(representative)
    return len(representatives), len(configurations)


def cycle_formula_count(action: AffineAction, tau: Tau) -> int:
    total = sum(
        prod(fixed_multisets(permutation, size) for size in tau)
        for permutation in action.permutations
    )
    answer = Fraction(total, action.order)
    if answer.denominator != 1:
        raise ArithmeticError(f"Burnside average did not cancel: {answer}")
    return answer.numerator


def verify_orbits(n: int, q: int, tau: Tau) -> OrbitCheck:
    action = affine_action(n, q)
    direct, configuration_count = direct_orbit_count(action, tau)
    predicted = cycle_formula_count(action, tau)
    sample_matrix = permutation_matrix(action.permutations[-1])
    if len(sample_matrix) != q**n:
        raise AssertionError("wrong permutation-matrix dimension")
    return OrbitCheck(
        q, n, tau, action.order, action.degree, configuration_count, direct, predicted
    )


def gaussian_binomial(m: int, r: int, q: int) -> int:
    if r < 0 or r > m:
        return 0
    r = min(r, m - r)
    numerator = prod(q ** (m - index) - 1 for index in range(r))
    denominator = prod(q ** (r - index) - 1 for index in range(r))
    return numerator // denominator


def _matmul_mod(left: Matrix, right: Matrix, q: int) -> Matrix:
    size = len(left)
    return tuple(
        tuple(
            sum(left[row][middle] * right[middle][column] for middle in range(size)) % q
            for column in range(size)
        )
        for row in range(size)
    )


def _linear_fixed_count(matrix: Matrix, translation: tuple[int, ...], power: int, q: int) -> int:
    """Equation (5): solve (I-A^d)x=(I+...+A^(d-1))b."""
    n = len(matrix)
    identity_matrix = tuple(
        tuple(1 if row == column else 0 for column in range(n)) for row in range(n)
    )
    matrix_power = identity_matrix
    translation_sum = [0] * n
    for _ in range(power):
        for row in range(n):
            translation_sum[row] = (
                translation_sum[row]
                + sum(matrix_power[row][column] * translation[column] for column in range(n))
            ) % q
        matrix_power = _matmul_mod(matrix_power, matrix, q)
    system = tuple(
        tuple((identity_matrix[row][column] - matrix_power[row][column]) % q for column in range(n))
        for row in range(n)
    )
    rank = _rank_mod(system, q)
    augmented = tuple(system[row] + (translation_sum[row],) for row in range(n))
    return q ** (n - rank) if _rank_mod(augmented, q) == rank else 0


def _permutation_power_fixed(permutation: Permutation, power: int) -> int:
    return sum(length for length in cycle_lengths(permutation) if power % length == 0)


def _mobius(value: int) -> int:
    prime_count = 0
    remaining = value
    factor = 2
    while factor * factor <= remaining:
        if remaining % factor == 0:
            remaining //= factor
            prime_count += 1
            if remaining % factor == 0:
                return 0
            while remaining % factor == 0:
                remaining //= factor
        factor += 1
    if remaining > 1:
        prime_count += 1
    return -1 if prime_count % 2 else 1


def verify_linear_fixed_point_and_mobius(n: int, q: int) -> bool:
    action = affine_action(n, q)
    element_index = 0
    for matrix in invertible_matrices(n, q):
        for translation in action.points:
            permutation = action.permutations[element_index]
            element_index += 1
            lengths = cycle_lengths(permutation)
            for power in range(1, action.degree + 1):
                observed = _permutation_power_fixed(permutation, power)
                if observed != _linear_fixed_count(matrix, translation, power, q):
                    return False
                divisors = [divisor for divisor in range(1, power + 1) if power % divisor == 0]
                recovered = sum(
                    _mobius(power // divisor)
                    * _linear_fixed_count(matrix, translation, divisor, q)
                    for divisor in divisors
                )
                actual_cycles = lengths.count(power)
                if recovered != power * actual_cycles:
                    return False
    return True


def verify_fixed_point_moment(n: int, q: int, power: int) -> MomentCheck:
    action = affine_action(n, q)
    direct = Fraction(
        sum(
            sum(index == image for index, image in enumerate(permutation)) ** power
            for permutation in action.permutations
        ),
        action.order,
    )
    formula = sum(
        gaussian_binomial(power - 1, rank, q)
        for rank in range(min(n, power - 1) + 1)
    )
    return MomentCheck(q, n, power, direct, formula)


ORBIT_CASES: tuple[tuple[int, int, Tau], ...] = (
    (1, 2, (3,)),
    (1, 2, (2, 1)),
    (2, 2, (3,)),
    (2, 2, (2, 1)),
    (2, 2, (1, 1, 1)),
    (3, 2, (1, 1, 1)),
    (1, 3, (3,)),
    (2, 3, (2, 1)),
    (1, 5, (1, 1, 1)),
)

MOMENT_CASES = (
    (1, 2, 3),
    (2, 2, 3),
    (3, 2, 4),
    (1, 3, 3),
    (2, 3, 3),
    (1, 5, 3),
)


def representative_suite() -> tuple[
    tuple[OrbitCheck, ...], tuple[MomentCheck, ...], dict[str, bool]
]:
    actions: dict[tuple[int, int], AffineAction] = {}

    def get_action(n: int, q: int) -> AffineAction:
        key = (n, q)
        if key not in actions:
            actions[key] = affine_action(n, q)
        return actions[key]

    orbit_rows = []
    for n, q, tau in ORBIT_CASES:
        action = get_action(n, q)
        direct, configuration_count = direct_orbit_count(action, tau)
        orbit_rows.append(
            OrbitCheck(
                q,
                n,
                tau,
                action.order,
                action.degree,
                configuration_count,
                direct,
                cycle_formula_count(action, tau),
            )
        )

    moment_rows = []
    for n, q, power in MOMENT_CASES:
        action = get_action(n, q)
        direct = Fraction(
            sum(
                sum(index == image for index, image in enumerate(permutation)) ** power
                for permutation in action.permutations
            ),
            action.order,
        )
        formula = sum(
            gaussian_binomial(power - 1, rank, q)
            for rank in range(min(n, power - 1) + 1)
        )
        moment_rows.append(MomentCheck(q, n, power, direct, formula))
    structural_checks = {
        "permutation-matrix determinant/cycle identity": all(
            verify_matrix_cycle_identity(action) for action in actions.values()
        ),
        "linear fixed-point formula and Mobius inversion for AGL_2(2)":
            verify_linear_fixed_point_and_mobius(2, 2),
        "linear fixed-point formula and Mobius inversion for AGL_2(3)":
            verify_linear_fixed_point_and_mobius(2, 3),
    }
    return tuple(orbit_rows), tuple(moment_rows), structural_checks


if __name__ == "__main__":
    orbit_rows, moment_rows, structural_checks = representative_suite()
    for row in orbit_rows:
        print(
            f"{'PASS' if row.passed else 'FAIL'} AGL_{row.n}({row.q}) "
            f"tau={row.tau}: orbit={row.direct_orbits}, formula={row.cycle_formula}, "
            f"|G|={row.group_order}, configurations={row.configuration_count}"
        )
    for row in moment_rows:
        print(
            f"{'PASS' if row.passed else 'FAIL'} AGL_{row.n}({row.q}) "
            f"E[F^{row.power}]={row.direct}, Gaussian={row.gaussian_formula}"
        )
    for name, passed in structural_checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    passed = (
        all(row.passed for row in (*orbit_rows, *moment_rows))
        and all(structural_checks.values())
    )
    raise SystemExit(0 if passed else 1)
