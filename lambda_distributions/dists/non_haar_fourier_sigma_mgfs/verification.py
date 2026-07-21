"""Exact and numerical checks for non-Haar Fourier-weighted sigma-MGFs.

The finite-group routes use exact ``Fraction`` arithmetic.  The compact
U(1) route compares a literal matrix-integrand quadrature against its heat
Fourier series.  The Ewens route independently compares element enumeration
with cycle-type aggregation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from itertools import permutations
from math import cos, exp, factorial, pi, prod
from typing import Callable, Iterable, Sequence


Permutation = tuple[int, ...]
Matrix = tuple[tuple[Fraction, ...], ...]
Partition = tuple[int, ...]


def identity_permutation(n: int) -> Permutation:
    return tuple(range(n))


def compose(left: Permutation, right: Permutation) -> Permutation:
    """Return left after right, so P(left*right)=P(left)P(right)."""

    return tuple(left[right[index]] for index in range(len(left)))


def inverse(permutation: Permutation) -> Permutation:
    answer = [0] * len(permutation)
    for source, target in enumerate(permutation):
        answer[target] = source
    return tuple(answer)


def cycle_lengths(permutation: Permutation) -> tuple[int, ...]:
    seen: set[int] = set()
    lengths: list[int] = []
    for start in range(len(permutation)):
        if start in seen:
            continue
        current = start
        length = 0
        while current not in seen:
            seen.add(current)
            current = permutation[current]
            length += 1
        lengths.append(length)
    return tuple(sorted(lengths))


def sign(permutation: Permutation) -> int:
    return -1 if (len(permutation) - len(cycle_lengths(permutation))) % 2 else 1


def permutation_matrix(permutation: Permutation) -> Matrix:
    n = len(permutation)
    rows = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for source, target in enumerate(permutation):
        rows[target][source] = Fraction(1)
    return tuple(tuple(row) for row in rows)


def deleted_permutation_matrix(permutation: Permutation) -> Matrix:
    """Restriction to sum-zero vectors in basis e_i-e_n."""

    n = len(permutation)
    rows = [[Fraction(0) for _ in range(n - 1)] for _ in range(n - 1)]
    anchor = permutation[-1]
    for source in range(n - 1):
        target = permutation[source]
        if target < n - 1:
            rows[target][source] += 1
        if anchor < n - 1:
            rows[anchor][source] -= 1
    return tuple(tuple(row) for row in rows)


def scalar_matrix(value: Fraction) -> Matrix:
    return ((value,),)


def matrix_identity(n: int) -> Matrix:
    return tuple(
        tuple(Fraction(int(i == j)) for j in range(n)) for i in range(n)
    )


def matrix_add(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(left[i][j] + right[i][j] for j in range(len(left)))
        for i in range(len(left))
    )


def matrix_scale(value: Fraction, matrix: Matrix) -> Matrix:
    return tuple(tuple(value * entry for entry in row) for row in matrix)


def matrix_multiply(left: Matrix, right: Matrix) -> Matrix:
    n = len(left)
    return tuple(
        tuple(
            sum((left[i][k] * right[k][j] for k in range(n)), Fraction(0))
            for j in range(n)
        )
        for i in range(n)
    )


def matrix_power(matrix: Matrix, exponent: int) -> Matrix:
    answer = matrix_identity(len(matrix))
    base = matrix
    while exponent:
        if exponent & 1:
            answer = matrix_multiply(answer, base)
        base = matrix_multiply(base, base)
        exponent //= 2
    return answer


def trace(matrix: Matrix) -> Fraction:
    return sum((matrix[i][i] for i in range(len(matrix))), Fraction(0))


def complete_characters(matrix: Matrix, maximum_degree: int) -> tuple[Fraction, ...]:
    """Characters of Sym^d from matrix power traces via Newton recurrence."""

    traces = [Fraction(0)]
    current = matrix_identity(len(matrix))
    for _degree in range(1, maximum_degree + 1):
        current = matrix_multiply(current, matrix)
        traces.append(trace(current))
    complete = [Fraction(1)]
    for degree in range(1, maximum_degree + 1):
        numerator = sum(
            (traces[r] * complete[degree - r] for r in range(1, degree + 1)),
            Fraction(0),
        )
        complete.append(numerator / degree)
    return tuple(complete)


def partitions_through(maximum_degree: int) -> tuple[Partition, ...]:
    def of(total: int, cap: int) -> Iterable[Partition]:
        if total == 0:
            yield ()
            return
        for first in range(min(total, cap), 0, -1):
            for rest in of(total - first, first):
                yield (first, *rest)

    return tuple(
        partition
        for total in range(1, maximum_degree + 1)
        for partition in of(total, total)
    )


def representation_matrix(permutation: Permutation, representation: str) -> Matrix:
    if representation == "natural":
        return permutation_matrix(permutation)
    if representation == "deleted":
        return deleted_permutation_matrix(permutation)
    raise ValueError(representation)


def tensor_symmetric_character(
    permutation: Permutation, representation: str, partition: Partition
) -> Fraction:
    matrix = representation_matrix(permutation, representation)
    complete = complete_characters(matrix, max(partition, default=0))
    return prod((complete[degree] for degree in partition), start=Fraction(1))


def convolve(
    left: dict[Permutation, Fraction], right: dict[Permutation, Fraction]
) -> dict[Permutation, Fraction]:
    answer: defaultdict[Permutation, Fraction] = defaultdict(Fraction)
    for left_element, left_weight in left.items():
        for right_element, right_weight in right.items():
            answer[compose(left_element, right_element)] += left_weight * right_weight
    return dict(answer)


def convolution_power(
    step: dict[Permutation, Fraction], exponent: int
) -> dict[Permutation, Fraction]:
    n = len(next(iter(step)))
    answer = {identity_permutation(n): Fraction(1)}
    base = step
    while exponent:
        if exponent & 1:
            answer = convolve(answer, base)
        base = convolve(base, base)
        exponent //= 2
    return answer


def s3_elements() -> tuple[Permutation, ...]:
    return tuple(permutations(range(3)))


def s3_irreducible_matrix(permutation: Permutation, irrep: str) -> Matrix:
    if irrep == "trivial":
        return scalar_matrix(Fraction(1))
    if irrep == "sign":
        return scalar_matrix(Fraction(sign(permutation)))
    if irrep == "standard":
        return deleted_permutation_matrix(permutation)
    raise ValueError(irrep)


S3_IRREPS = ("trivial", "sign", "standard")
S3_DIMENSIONS = {"trivial": 1, "sign": 1, "standard": 2}


def s3_multiplicity(representation: str, partition: Partition, irrep: str) -> int:
    total = Fraction(0)
    for element in s3_elements():
        chi_w = tensor_symmetric_character(element, representation, partition)
        chi_pi_inverse = trace(s3_irreducible_matrix(inverse(element), irrep))
        total += chi_w * chi_pi_inverse
    answer = total / 6
    assert answer.denominator == 1 and answer >= 0
    return answer.numerator


def fourier_transform(
    step: dict[Permutation, Fraction], irrep: str
) -> Matrix:
    dimension = S3_DIMENSIONS[irrep]
    answer = tuple(tuple(Fraction(0) for _ in range(dimension)) for _ in range(dimension))
    for element, weight in step.items():
        answer = matrix_add(
            answer, matrix_scale(weight, s3_irreducible_matrix(element, irrep))
        )
    return answer


def direct_walk_coefficient(
    step: dict[Permutation, Fraction],
    exponent: int,
    representation: str,
    partition: Partition,
) -> Fraction:
    law = convolution_power(step, exponent)
    return sum(
        (
            weight * tensor_symmetric_character(element, representation, partition)
            for element, weight in law.items()
        ),
        Fraction(0),
    )


def full_fourier_coefficient(
    step: dict[Permutation, Fraction],
    exponent: int,
    representation: str,
    partition: Partition,
) -> Fraction:
    return sum(
        (
            s3_multiplicity(representation, partition, irrep)
            * trace(matrix_power(fourier_transform(step, irrep), exponent))
            for irrep in S3_IRREPS
        ),
        Fraction(0),
    )


def central_fourier_coefficient(
    step: dict[Permutation, Fraction],
    exponent: int,
    representation: str,
    partition: Partition,
) -> Fraction:
    answer = Fraction(0)
    for irrep in S3_IRREPS:
        dimension = S3_DIMENSIONS[irrep]
        transform = fourier_transform(step, irrep)
        theta = trace(transform) / dimension
        assert transform == matrix_scale(theta, matrix_identity(dimension))
        answer += (
            s3_multiplicity(representation, partition, irrep)
            * dimension
            * theta**exponent
        )
    return answer


def s3_walk_suite(maximum_degree: int = 4) -> dict[str, object]:
    elements = s3_elements()
    identity = identity_permutation(3)
    transpositions = tuple(g for g in elements if cycle_lengths(g) == (1, 2))
    three_cycles = tuple(g for g in elements if cycle_lengths(g) == (3,))
    central_step = {identity: Fraction(1, 4), **{g: Fraction(1, 4) for g in transpositions}}
    noncentral_step = {transpositions[0]: Fraction(1, 2), three_cycles[0]: Fraction(1, 2)}
    partitions = partitions_through(maximum_degree)
    rows: list[dict[str, object]] = []
    passed = True
    for walk_name, step in (("central lazy transposition", central_step), ("noncentral two-generator", noncentral_step)):
        for representation in ("natural", "deleted"):
            for exponent in (0, 1, 2, 3, 5):
                for partition in partitions:
                    direct = direct_walk_coefficient(step, exponent, representation, partition)
                    fourier = full_fourier_coefficient(step, exponent, representation, partition)
                    central = (
                        central_fourier_coefficient(step, exponent, representation, partition)
                        if walk_name.startswith("central")
                        else None
                    )
                    row_passed = direct == fourier and (central is None or direct == central)
                    passed &= row_passed
                    if partition in ((1,), (2,), (1, 1), (2, 1), (2, 2)) and exponent in (1, 3, 5):
                        rows.append(
                            {
                                "walk": walk_name,
                                "representation": representation,
                                "k": exponent,
                                "partition": partition,
                                "direct": direct,
                                "full Fourier": fourier,
                                "central scalar": central,
                                "passed": row_passed,
                            }
                        )
    noncentral_standard = fourier_transform(noncentral_step, "standard")
    scalar = matrix_scale(trace(noncentral_standard) / 2, matrix_identity(2))
    return {
        "group order": len(elements),
        "matrix dimensions": (3, 2),
        "partitions": len(partitions),
        "comparisons": 2 * 2 * 5 * len(partitions),
        "noncentral transform": noncentral_standard,
        "noncentral transform is scalar": noncentral_standard == scalar,
        "rows": tuple(rows),
        "passed": passed and noncentral_standard != scalar,
    }


def convolve_weight_polynomials(left: Counter[int], right: Counter[int]) -> Counter[int]:
    answer: Counter[int] = Counter()
    for a, multiplicity_a in left.items():
        for b, multiplicity_b in right.items():
            answer[a + b] += multiplicity_a * multiplicity_b
    return answer


def u1_tensor_weights(partition: Partition) -> Counter[int]:
    """Weights in tensor_j Sym^(tau_j)(z + z^-1)."""

    answer: Counter[int] = Counter({0: 1})
    for degree in partition:
        symmetric_weights = Counter({degree - 2 * index: 1 for index in range(degree + 1)})
        answer = convolve_weight_polynomials(answer, symmetric_weights)
    return answer


def u1_heat_formula(partition: Partition, time: float) -> float:
    return sum(
        multiplicity * exp(-time * weight * weight / 2)
        for weight, multiplicity in u1_tensor_weights(partition).items()
    )


def complex_complete_characters(theta: float, maximum_degree: int) -> tuple[complex, ...]:
    """Directly use the 2x2 matrix diag(e^(i theta),e^(-i theta))."""

    traces = [0j]
    for exponent in range(1, maximum_degree + 1):
        traces.append(2 * cos(exponent * theta))
    complete = [1 + 0j]
    for degree in range(1, maximum_degree + 1):
        complete.append(
            sum(traces[r] * complete[degree - r] for r in range(1, degree + 1))
            / degree
        )
    return tuple(complete)


def u1_heat_direct(
    partition: Partition, time: float, grid_size: int = 256, kernel_cutoff: int = 48
) -> float:
    total = 0j
    for index in range(grid_size):
        theta = 2 * pi * index / grid_size
        heat_density = 1 + 2 * sum(
            exp(-time * frequency * frequency / 2) * cos(frequency * theta)
            for frequency in range(1, kernel_cutoff + 1)
        )
        complete = complex_complete_characters(theta, max(partition, default=0))
        integrand = prod((complete[degree] for degree in partition), start=1 + 0j)
        total += heat_density * integrand
    return (total / grid_size).real


def u1_heat_suite(maximum_degree: int = 5) -> dict[str, object]:
    partitions = partitions_through(maximum_degree)
    rows: list[dict[str, object]] = []
    maximum_error = 0.0
    endpoint_passed = True
    for time in (0.2, 0.7, 2.0):
        for partition in partitions:
            direct = u1_heat_direct(partition, time)
            formula = u1_heat_formula(partition, time)
            error = abs(direct - formula)
            maximum_error = max(maximum_error, error)
            if partition in ((1,), (2,), (1, 1), (3,), (2, 1), (2, 2), (5,)):
                rows.append(
                    {
                        "time": time,
                        "partition": partition,
                        "quadrature": direct,
                        "weight formula": formula,
                        "absolute error": error,
                        "passed": error < 1e-11,
                    }
                )
    for partition in partitions:
        weights = u1_tensor_weights(partition)
        identity_value = prod((degree + 1 for degree in partition), start=1)
        haar_value = weights[0]
        endpoint_passed &= sum(weights.values()) == identity_value
        endpoint_passed &= abs(u1_heat_formula(partition, 100.0) - haar_value) < 1e-12
    return {
        "representation dimension": 2,
        "weights": (-1, 1),
        "partitions": len(partitions),
        "comparisons": 3 * len(partitions),
        "maximum error": maximum_error,
        "endpoint checks": endpoint_passed,
        "rows": tuple(rows),
        "passed": maximum_error < 1e-11 and endpoint_passed,
    }


def rising_factorial(theta: Fraction, n: int) -> Fraction:
    return prod((theta + index for index in range(n)), start=Fraction(1))


def cycle_complete_characters(lengths: Sequence[int], maximum_degree: int) -> tuple[int, ...]:
    coefficients = [0] * (maximum_degree + 1)
    coefficients[0] = 1
    for length in lengths:
        for degree in range(length, maximum_degree + 1):
            coefficients[degree] += coefficients[degree - length]
    return tuple(coefficients)


def ewens_element_coefficient(n: int, theta: Fraction, partition: Partition) -> Fraction:
    denominator = rising_factorial(theta, n)
    answer = Fraction(0)
    for element in permutations(range(n)):
        lengths = cycle_lengths(element)
        weight = theta ** len(lengths) / denominator
        matrix = permutation_matrix(element)
        complete = complete_characters(matrix, max(partition, default=0))
        answer += weight * prod((complete[d] for d in partition), start=Fraction(1))
    return answer


def ewens_cycle_coefficient(n: int, theta: Fraction, partition: Partition) -> Fraction:
    denominator = rising_factorial(theta, n)
    shapes: Counter[tuple[int, ...]] = Counter(cycle_lengths(g) for g in permutations(range(n)))
    answer = Fraction(0)
    for lengths, class_size in shapes.items():
        complete = cycle_complete_characters(lengths, max(partition, default=0))
        answer += (
            class_size
            * theta ** len(lengths)
            * prod((complete[d] for d in partition), start=1)
            / denominator
        )
    return answer


def determinant_from_cycle_lengths(lengths: Sequence[int], variables: Sequence[Fraction]) -> Fraction:
    return prod(
        (
            Fraction(1, 1 - variable**length)
            for variable in variables
            for length in lengths
        ),
        start=Fraction(1),
    )


def determinant(matrix: Matrix) -> Fraction:
    """Exact determinant by Gaussian elimination over the rationals."""

    work = [list(row) for row in matrix]
    answer = Fraction(1)
    for column in range(len(work)):
        pivot_row = next(
            (row for row in range(column, len(work)) if work[row][column]), None
        )
        if pivot_row is None:
            return Fraction(0)
        if pivot_row != column:
            work[column], work[pivot_row] = work[pivot_row], work[column]
            answer *= -1
        pivot = work[column][column]
        answer *= pivot
        for row in range(column + 1, len(work)):
            factor = work[row][column] / pivot
            for index in range(column + 1, len(work)):
                work[row][index] -= factor * work[column][index]
    return answer


def determinant_integrand_from_matrix(
    matrix: Matrix, variables: Sequence[Fraction]
) -> Fraction:
    dimension = len(matrix)
    answer = Fraction(1)
    for variable in variables:
        shifted = tuple(
            tuple(
                Fraction(int(row == column)) - variable * matrix[row][column]
                for column in range(dimension)
            )
            for row in range(dimension)
        )
        answer /= determinant(shifted)
    return answer


def ewens_full_element_average(n: int, theta: Fraction, variables: Sequence[Fraction]) -> Fraction:
    denominator = rising_factorial(theta, n)
    return sum(
        (
            theta ** len(cycle_lengths(element))
            * determinant_integrand_from_matrix(permutation_matrix(element), variables)
            / denominator
            for element in permutations(range(n))
        ),
        Fraction(0),
    )


def ewens_full_cycle_average(n: int, theta: Fraction, variables: Sequence[Fraction]) -> Fraction:
    denominator = rising_factorial(theta, n)
    shapes: Counter[tuple[int, ...]] = Counter(cycle_lengths(g) for g in permutations(range(n)))
    return sum(
        (
            class_size
            * theta ** len(lengths)
            * determinant_from_cycle_lengths(lengths, variables)
            / denominator
            for lengths, class_size in shapes.items()
        ),
        Fraction(0),
    )


def ewens_suite(maximum_degree: int = 4) -> dict[str, object]:
    partitions = partitions_through(maximum_degree)
    rows: list[dict[str, object]] = []
    passed = True
    comparisons = 0
    for n in (3, 4, 5, 6):
        for theta in (Fraction(1, 2), Fraction(1), Fraction(2)):
            for partition in partitions:
                element = ewens_element_coefficient(n, theta, partition)
                cycle = ewens_cycle_coefficient(n, theta, partition)
                row_passed = element == cycle
                passed &= row_passed
                comparisons += 1
                if partition in ((1,), (2,), (1, 1), (2, 1), (2, 2)) and n in (3, 6):
                    rows.append(
                        {
                            "n": n,
                            "theta": theta,
                            "partition": partition,
                            "matrix enumeration": element,
                            "cycle formula": cycle,
                            "passed": row_passed,
                        }
                    )
            variables = (Fraction(1, 10), Fraction(-1, 20))
            full_element = ewens_full_element_average(n, theta, variables)
            full_cycle = ewens_full_cycle_average(n, theta, variables)
            passed &= full_element == full_cycle
    return {
        "dimensions": (3, 4, 5, 6),
        "theta values": (Fraction(1, 2), Fraction(1), Fraction(2)),
        "partitions": len(partitions),
        "coefficient comparisons": comparisons,
        "full determinant comparisons": 12,
        "rows": tuple(rows),
        "passed": passed,
    }


def run_all() -> dict[str, object]:
    finite = s3_walk_suite()
    heat = u1_heat_suite()
    ewens = ewens_suite()
    return {
        "finite walks": finite,
        "compact heat": heat,
        "ewens": ewens,
        "passed": bool(finite["passed"] and heat["passed"] and ewens["passed"]),
    }


if __name__ == "__main__":
    report = run_all()
    print(f"overall: {'PASS' if report['passed'] else 'FAIL'}")
    for name in ("finite walks", "compact heat", "ewens"):
        print(name, report[name])
