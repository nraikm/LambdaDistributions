"""Exact matrix checks for lattice-automorphism sigma-MGF formulas.

The verifier keeps three routes separate:

* concrete integral matrices and Newton's trace recurrence;
* frame shapes recovered by Moebius inversion and determinant factorization;
* family formulas using permutation cycles, signed cycles, or wreath cycles.

For a small set of low-degree coefficients it also constructs the actual
tensor products of symmetric-power matrices and computes their common fixed
space over the rationals.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations, product
from math import comb, factorial, gcd, prod


Matrix = tuple[tuple[int, ...], ...]
Tau = tuple[int, ...]
MAX_DEGREE = 4
FIXED_SPACE_TAUS: tuple[Tau, ...] = ((1,), (2,), (1, 1), (3,))


def identity(dimension: int) -> Matrix:
    return tuple(tuple(int(i == j) for j in range(dimension)) for i in range(dimension))


def scalar_matrix(scale: int, dimension: int) -> Matrix:
    return tuple(
        tuple(scale * int(i == j) for j in range(dimension))
        for i in range(dimension)
    )


def transpose(matrix: Matrix) -> Matrix:
    return tuple(tuple(matrix[j][i] for j in range(len(matrix))) for i in range(len(matrix)))


def matmul(left: Matrix, right: Matrix) -> Matrix:
    size = len(left)
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(size)) for j in range(size))
        for i in range(size)
    )


def matrix_power(matrix: Matrix, exponent: int) -> Matrix:
    answer = identity(len(matrix))
    base = matrix
    while exponent:
        if exponent & 1:
            answer = matmul(answer, base)
        base = matmul(base, base)
        exponent //= 2
    return answer


def matrix_trace(matrix: Matrix) -> int:
    return sum(matrix[i][i] for i in range(len(matrix)))


def matrix_order(matrix: Matrix, limit: int = 1000) -> int:
    current = identity(len(matrix))
    for exponent in range(1, limit + 1):
        current = matmul(current, matrix)
        if current == identity(len(matrix)):
            return exponent
    raise ArithmeticError("matrix order exceeded verification limit")


def kron(left: Matrix, right: Matrix) -> Matrix:
    rows = []
    for left_row in left:
        for right_row in right:
            rows.append(
                tuple(a * b for a in left_row for b in right_row)
            )
    return tuple(rows)


def kron_all(matrices: tuple[Matrix, ...]) -> Matrix:
    answer: Matrix = ((1,),)
    for matrix in matrices:
        answer = kron(answer, matrix)
    return answer


def block_action(labels: tuple[Matrix, ...], permutation: tuple[int, ...]) -> Matrix:
    """Map source block j to target block permutation[j] using labels[j]."""

    block_size = len(labels[0])
    blocks = len(labels)
    dimension = block_size * blocks
    answer = [[0] * dimension for _ in range(dimension)]
    for source, target in enumerate(permutation):
        label = labels[source]
        for row in range(block_size):
            for column in range(block_size):
                answer[target * block_size + row][source * block_size + column] = label[row][column]
    return tuple(tuple(row) for row in answer)


def tensor_factor_permutation(
    factor_dimension: int, permutation: tuple[int, ...]
) -> Matrix:
    factors = len(permutation)
    words = tuple(product(range(factor_dimension), repeat=factors))
    location = {word: index for index, word in enumerate(words)}
    answer = [[0] * len(words) for _ in words]
    for column, word in enumerate(words):
        image = [0] * factors
        for source, target in enumerate(permutation):
            image[target] = word[source]
        answer[location[tuple(image)]][column] = 1
    return tuple(tuple(row) for row in answer)


def tensor_wreath_action(
    labels: tuple[Matrix, ...], permutation: tuple[int, ...]
) -> Matrix:
    local = kron_all(labels)
    top = tensor_factor_permutation(len(labels[0]), permutation)
    return matmul(top, local)


def permutation_matrix_on_standard(permutation: tuple[int, ...]) -> Matrix:
    """S_n on the A_(n-1) basis e_0-e_(n-1),...,e_(n-2)-e_(n-1)."""

    n = len(permutation)
    d = n - 1
    answer = [[0] * d for _ in range(d)]
    for column in range(d):
        positive = permutation[column]
        negative = permutation[n - 1]
        if positive != n - 1:
            answer[positive][column] += 1
        if negative != n - 1:
            answer[negative][column] -= 1
    return tuple(tuple(row) for row in answer)


def a_lattice_group(n: int) -> tuple[Matrix, ...]:
    matrices = []
    for permutation in permutations(range(n)):
        base = permutation_matrix_on_standard(permutation)
        matrices.extend((base, tuple(tuple(-x for x in row) for row in base)))
    return tuple(dict.fromkeys(matrices))


def signed_permutation_group(n: int) -> tuple[Matrix, ...]:
    matrices = []
    for permutation in permutations(range(n)):
        for signs in product((-1, 1), repeat=n):
            answer = [[0] * n for _ in range(n)]
            for column, target in enumerate(permutation):
                answer[target][column] = signs[column]
            matrices.append(tuple(tuple(row) for row in answer))
    return tuple(matrices)


def gram_preserved(matrix: Matrix, gram: Matrix) -> bool:
    return matmul(matmul(transpose(matrix), gram), matrix) == gram


def inner(gram: Matrix, left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(
        left[i] * gram[i][j] * right[j]
        for i in range(len(gram))
        for j in range(len(gram))
    )


def enumerate_lattice_automorphisms(gram: Matrix) -> tuple[Matrix, ...]:
    """Enumerate Aut(G) using a rigorous Gershgorin coordinate bound."""

    d = len(gram)
    lower_bound = min(
        gram[i][i] - sum(abs(gram[i][j]) for j in range(d) if j != i)
        for i in range(d)
    )
    if lower_bound <= 0:
        raise ValueError("Gram matrix must be strictly diagonally dominant")
    maximum_norm = max(gram[i][i] for i in range(d))
    coordinate_bound = 0
    while (coordinate_bound + 1) ** 2 * lower_bound <= maximum_norm:
        coordinate_bound += 1
    vectors = tuple(product(range(-coordinate_bound, coordinate_bound + 1), repeat=d))
    by_norm = {
        norm: tuple(vector for vector in vectors if inner(gram, vector, vector) == norm)
        for norm in set(gram[i][i] for i in range(d))
    }
    answers: list[Matrix] = []

    def extend(columns: tuple[tuple[int, ...], ...]) -> None:
        column = len(columns)
        if column == d:
            matrix = tuple(tuple(columns[j][i] for j in range(d)) for i in range(d))
            if gram_preserved(matrix, gram):
                answers.append(matrix)
            return
        for vector in by_norm[gram[column][column]]:
            if all(inner(gram, columns[j], vector) == gram[j][column] for j in range(column)):
                extend((*columns, vector))

    extend(())
    return tuple(dict.fromkeys(answers))


SMALL_SYMMETRY_GRAMS: dict[int, Matrix] = {
    2: ((6, -1), (-1, 8)),
    3: ((6, 1, 2), (1, 7, -1), (2, -1, 10)),
    4: ((6, 1, 1, 2), (1, 7, 1, 2), (1, 1, 11, 1), (2, 2, 1, 13)),
}


def integer_partitions(total: int, largest: int | None = None):
    if total == 0:
        yield ()
        return
    largest = total if largest is None else min(largest, total)
    for first in range(largest, 0, -1):
        for tail in integer_partitions(total - first, first):
            yield (first, *tail)


def partitions_up_to(max_degree: int):
    for degree in range(max_degree + 1):
        yield from integer_partitions(degree)


def permutation_sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def polynomial_multiply(left: list[int], right: list[int]) -> list[int]:
    answer = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            answer[i + j] += a * b
    while len(answer) > 1 and answer[-1] == 0:
        answer.pop()
    return answer


def determinant_polynomial(matrix: Matrix) -> tuple[int, ...]:
    """Exact coefficients of det(I-tA), in ascending degree."""

    d = len(matrix)
    answer = [0]
    for permutation in permutations(range(d)):
        term = [1]
        for row, column in enumerate(permutation):
            factor = [int(row == column), -matrix[row][column]]
            term = polynomial_multiply(term, factor)
        sign = permutation_sign(permutation)
        if len(answer) < len(term):
            answer.extend([0] * (len(term) - len(answer)))
        for degree, coefficient in enumerate(term):
            answer[degree] += sign * coefficient
    while len(answer) > 1 and answer[-1] == 0:
        answer.pop()
    return tuple(answer)


def moebius(n: int) -> int:
    primes = 0
    candidate = 2
    remaining = n
    while candidate * candidate <= remaining:
        if remaining % candidate == 0:
            remaining //= candidate
            primes += 1
            if remaining % candidate == 0:
                return 0
            while remaining % candidate == 0:
                remaining //= candidate
        candidate += 1
    if remaining > 1:
        primes += 1
    return -1 if primes % 2 else 1


def divisors(n: int) -> tuple[int, ...]:
    return tuple(candidate for candidate in range(1, n + 1) if n % candidate == 0)


def frame_shape(matrix: Matrix) -> dict[int, int]:
    order = matrix_order(matrix)
    traces = {r: matrix_trace(matrix_power(matrix, r)) for r in range(1, order + 1)}
    answer = {}
    for m in range(1, order + 1):
        numerator = sum(moebius(m // divisor) * traces[divisor] for divisor in divisors(m))
        if numerator % m:
            raise ArithmeticError("nonintegral frame exponent")
        exponent = numerator // m
        if exponent:
            answer[m] = exponent
    return answer


def polynomial_power(base: list[int], exponent: int) -> list[int]:
    answer = [1]
    for _ in range(exponent):
        answer = polynomial_multiply(answer, base)
    return answer


def frame_determinant_identity(matrix: Matrix, frame: dict[int, int]) -> bool:
    numerator = [1]
    denominator = [1]
    for m, exponent in frame.items():
        factor = [1] + [0] * (m - 1) + [-1]
        if exponent > 0:
            numerator = polynomial_multiply(numerator, polynomial_power(factor, exponent))
        else:
            denominator = polynomial_multiply(denominator, polynomial_power(factor, -exponent))
    determinant = list(determinant_polynomial(matrix))
    return numerator == polynomial_multiply(determinant, denominator)


def complete_from_traces(traces: dict[int, int], max_degree: int) -> tuple[int, ...]:
    complete = [1]
    for degree in range(1, max_degree + 1):
        numerator = sum(traces[r] * complete[degree - r] for r in range(1, degree + 1))
        if numerator % degree:
            raise ArithmeticError("Newton recurrence did not remain integral")
        complete.append(numerator // degree)
    return tuple(complete)


def complete_from_frame(frame: dict[int, int], max_degree: int) -> tuple[int, ...]:
    series = [1] + [0] * max_degree
    for m, a_m in frame.items():
        exponent = -a_m
        factor = [0] * (max_degree + 1)
        if exponent >= 0:
            for k in range(min(exponent, max_degree // m) + 1):
                factor[k * m] = (-1) ** k * comb(exponent, k)
        else:
            positive = -exponent
            for k in range(max_degree // m + 1):
                factor[k * m] = comb(positive + k - 1, k)
        updated = [0] * (max_degree + 1)
        for i, left in enumerate(series):
            for j, right in enumerate(factor[: max_degree + 1 - i]):
                updated[i + j] += left * right
        series = updated
    return tuple(series)


def matrix_complete_characters(matrix: Matrix, max_degree: int) -> tuple[int, ...]:
    traces = {r: matrix_trace(matrix_power(matrix, r)) for r in range(1, max_degree + 1)}
    return complete_from_traces(traces, max_degree)


def direct_coefficient(matrices: tuple[Matrix, ...], tau: Tau, max_degree: int) -> int:
    numerator = 0
    for matrix in matrices:
        complete = matrix_complete_characters(matrix, max_degree)
        numerator += prod(complete[degree] for degree in tau)
    if numerator % len(matrices):
        raise ArithmeticError("direct Reynolds trace was not integral")
    return numerator // len(matrices)


def cycle_z(parts: tuple[int, ...]) -> int:
    counts = Counter(parts)
    return prod((length ** multiplicity) * factorial(multiplicity) for length, multiplicity in counts.items())


def coefficient_from_trace_vector(traces: dict[int, int], tau: Tau, max_degree: int) -> int:
    complete = complete_from_traces(traces, max_degree)
    return prod(complete[degree] for degree in tau)


def a_cycle_formula(n: int, tau: Tau, max_degree: int) -> Fraction:
    answer = Fraction(0)
    for parts in integer_partitions(n):
        counts = Counter(parts)
        for epsilon in (-1, 1):
            traces = {
                r: epsilon**r * (sum(m * counts[m] for m in counts if r % m == 0) - 1)
                for r in range(1, max_degree + 1)
            }
            answer += Fraction(coefficient_from_trace_vector(traces, tau, max_degree), 2 * cycle_z(parts))
    return answer


def signed_cycle_formula(n: int, tau: Tau, max_degree: int) -> Fraction:
    answer = Fraction(0)
    for parts in integer_partitions(n):
        for cycle_signs in product((-1, 1), repeat=len(parts)):
            traces = {
                r: sum(
                    length * sign ** (r // length)
                    for length, sign in zip(parts, cycle_signs, strict=True)
                    if r % length == 0
                )
                for r in range(1, max_degree + 1)
            }
            answer += Fraction(
                coefficient_from_trace_vector(traces, tau, max_degree),
                cycle_z(parts) * 2 ** len(parts),
            )
    return answer


def scalar_formula(dimension: int, tau: Tau) -> Fraction:
    value = prod(comb(dimension + degree - 1, degree) for degree in tau)
    return Fraction(value if sum(tau) % 2 == 0 else 0)


def wreath_sum_formula(base: tuple[Matrix, ...], k: int, tau: Tau, max_degree: int) -> Fraction:
    answer = Fraction(0)
    for parts in integer_partitions(k):
        for cycle_products in product(base, repeat=len(parts)):
            traces = {
                r: sum(
                    length * matrix_trace(matrix_power(label, r // length))
                    for length, label in zip(parts, cycle_products, strict=True)
                    if r % length == 0
                )
                for r in range(1, max_degree + 1)
            }
            answer += Fraction(
                coefficient_from_trace_vector(traces, tau, max_degree),
                cycle_z(parts) * len(base) ** len(parts),
            )
    return answer


def tensor_local_formula(base: tuple[Matrix, ...], k: int, tau: Tau, max_degree: int) -> Fraction:
    numerator = 0
    for labels in product(base, repeat=k):
        traces = {
            r: prod(matrix_trace(matrix_power(label, r)) for label in labels)
            for r in range(1, max_degree + 1)
        }
        numerator += coefficient_from_trace_vector(traces, tau, max_degree)
    return Fraction(numerator, len(base) ** k)


def tensor_wreath_formula(base: tuple[Matrix, ...], k: int, tau: Tau, max_degree: int) -> Fraction:
    answer = Fraction(0)
    for parts in integer_partitions(k):
        for cycle_products in product(base, repeat=len(parts)):
            traces = {}
            for r in range(1, max_degree + 1):
                value = 1
                for length, label in zip(parts, cycle_products, strict=True):
                    divisor = gcd(length, r)
                    value *= matrix_trace(matrix_power(label, r // divisor)) ** divisor
                traces[r] = value
            answer += Fraction(
                coefficient_from_trace_vector(traces, tau, max_degree),
                cycle_z(parts) * len(base) ** len(parts),
            )
    return answer


def compositions(total: int, length: int):
    if length == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in compositions(total - first, length - 1):
            yield (first, *tail)


def symmetric_power_matrix(matrix: Matrix, degree: int) -> Matrix:
    dimension = len(matrix)
    basis = tuple(compositions(degree, dimension))
    location = {monomial: index for index, monomial in enumerate(basis)}
    zero = (0,) * dimension
    answer = [[0] * len(basis) for _ in basis]
    for column, monomial in enumerate(basis):
        polynomial: dict[tuple[int, ...], int] = {zero: 1}
        for source, repetitions in enumerate(monomial):
            for _ in range(repetitions):
                updated: Counter[tuple[int, ...]] = Counter()
                for exponent, coefficient in polynomial.items():
                    for target in range(dimension):
                        entry = matrix[target][source]
                        if entry:
                            new_exponent = list(exponent)
                            new_exponent[target] += 1
                            updated[tuple(new_exponent)] += coefficient * entry
                polynomial = dict(updated)
        for exponent, coefficient in polynomial.items():
            answer[location[exponent]][column] = coefficient
    return tuple(tuple(row) for row in answer)


def representation_matrix(matrix: Matrix, tau: Tau) -> Matrix:
    return kron_all(tuple(symmetric_power_matrix(matrix, degree) for degree in tau))


def rational_rank(rows: list[list[int]]) -> int:
    if not rows:
        return 0
    matrix = [[Fraction(value) for value in row] for row in rows if any(row)]
    if not matrix:
        return 0
    row = 0
    columns = len(matrix[0])
    for column in range(columns):
        pivot = next((candidate for candidate in range(row, len(matrix)) if matrix[candidate][column]), None)
        if pivot is None:
            continue
        matrix[row], matrix[pivot] = matrix[pivot], matrix[row]
        pivot_value = matrix[row][column]
        matrix[row] = [entry / pivot_value for entry in matrix[row]]
        for other in range(len(matrix)):
            if other != row and matrix[other][column]:
                scale = matrix[other][column]
                matrix[other] = [
                    matrix[other][j] - scale * matrix[row][j]
                    for j in range(columns)
                ]
        row += 1
        if row == len(matrix):
            break
    return row


def fixed_space_dimension(generators: tuple[Matrix, ...], tau: Tau) -> int:
    represented = tuple(representation_matrix(generator, tau) for generator in generators)
    dimension = len(represented[0])
    constraints: list[list[int]] = []
    for matrix in represented:
        constraints.extend(
            [matrix[i][j] - int(i == j) for j in range(dimension)]
            for i in range(dimension)
        )
    return dimension - rational_rank(constraints)


def cycle_permutation(n: int) -> tuple[int, ...]:
    return tuple((i + 1) % n for i in range(n))


def transposition(n: int) -> tuple[int, ...]:
    answer = list(range(n))
    answer[0], answer[1] = answer[1], answer[0]
    return tuple(answer)


def a_generators(n: int) -> tuple[Matrix, ...]:
    d = n - 1
    return (
        permutation_matrix_on_standard(transposition(n)),
        permutation_matrix_on_standard(cycle_permutation(n)),
        scalar_matrix(-1, d),
    )


def signed_generators(n: int) -> tuple[Matrix, ...]:
    permutation_generators = [transposition(n)]
    if n > 2:
        permutation_generators.append(cycle_permutation(n))
    answer = []
    for permutation in permutation_generators:
        matrix = [[0] * n for _ in range(n)]
        for column, target in enumerate(permutation):
            matrix[target][column] = 1
        answer.append(tuple(tuple(row) for row in matrix))
    sign_flip = [list(row) for row in identity(n)]
    sign_flip[0][0] = -1
    answer.append(tuple(tuple(row) for row in sign_flip))
    return tuple(answer)


def wreath_sum_group(base: tuple[Matrix, ...], k: int) -> tuple[Matrix, ...]:
    return tuple(
        block_action(labels, permutation)
        for permutation in permutations(range(k))
        for labels in product(base, repeat=k)
    )


def tensor_local_group(base: tuple[Matrix, ...], k: int) -> tuple[Matrix, ...]:
    return tuple(kron_all(labels) for labels in product(base, repeat=k))


def tensor_wreath_group(base: tuple[Matrix, ...], k: int) -> tuple[Matrix, ...]:
    return tuple(
        tensor_wreath_action(labels, permutation)
        for permutation in permutations(range(k))
        for labels in product(base, repeat=k)
    )


def embed_block(matrix: Matrix, block: int, k: int) -> Matrix:
    labels = [identity(len(matrix)) for _ in range(k)]
    labels[block] = matrix
    return block_action(tuple(labels), tuple(range(k)))


def direct_sum_generators(base_generators: tuple[Matrix, ...], k: int) -> tuple[Matrix, ...]:
    generators = [embed_block(generator, 0, k) for generator in base_generators]
    if k > 1:
        generators.append(block_action(tuple(identity(len(base_generators[0])) for _ in range(k)), transposition(k)))
    return tuple(generators)


def tensor_generators(
    base_generators: tuple[Matrix, ...], k: int, include_permutations: bool
) -> tuple[Matrix, ...]:
    d = len(base_generators[0])
    generators = []
    for block in range(k):
        for generator in base_generators:
            labels = tuple(generator if j == block else identity(d) for j in range(k))
            generators.append(kron_all(labels))
    if include_permutations and k > 1:
        generators.append(tensor_factor_permutation(d, transposition(k)))
    return tuple(generators)


@dataclass(frozen=True)
class VerificationCase:
    section: str
    name: str
    matrices: tuple[Matrix, ...]
    generators: tuple[Matrix, ...]
    formula: object
    lattice_check: bool


def _make_cases() -> tuple[VerificationCase, ...]:
    cases: list[VerificationCase] = []
    for n in (3, 4, 5):
        gram = tuple(
            tuple(2 if i == j else 1 for j in range(n - 1))
            for i in range(n - 1)
        )
        group = a_lattice_group(n)
        cases.append(
            VerificationCase(
                "Root lattices A",
                f"Aut(A_{n - 1}) = +/-S_{n}",
                group,
                a_generators(n),
                lambda tau, degree, n=n: a_cycle_formula(n, tau, degree),
                all(gram_preserved(matrix, gram) for matrix in group),
            )
        )
    for n in (2, 3, 4):
        group = signed_permutation_group(n)
        cases.append(
            VerificationCase(
                "Signed permutation B/C/D",
                f"C2 wr S_{n} on Z^{n}",
                group,
                signed_generators(n),
                lambda tau, degree, n=n: signed_cycle_formula(n, tau, degree),
                all(gram_preserved(matrix, identity(n)) for matrix in group),
            )
        )
    for dimension, gram in SMALL_SYMMETRY_GRAMS.items():
        group = enumerate_lattice_automorphisms(gram)
        cases.append(
            VerificationCase(
                "Small symmetry",
                f"Aut(L_{dimension}) = {{+/-I_{dimension}}}",
                group,
                (scalar_matrix(-1, dimension),),
                lambda tau, degree, dimension=dimension: scalar_formula(dimension, tau),
                set(group) == {identity(dimension), scalar_matrix(-1, dimension)},
            )
        )
    base = a_lattice_group(3)
    base_generators = a_generators(3)
    k = 2
    direct_sum = wreath_sum_group(base, k)
    cases.append(
        VerificationCase(
            "Repeated orthogonal sums",
            "Aut(A2) wr S2 on A2 direct-sum A2",
            direct_sum,
            direct_sum_generators(base_generators, k),
            lambda tau, degree: wreath_sum_formula(base, k, tau, degree),
            all(
                gram_preserved(
                    matrix,
                    tuple(
                        tuple(
                            (2 if i % 2 == j % 2 else 1) if i // 2 == j // 2 else 0
                            for j in range(4)
                        )
                        for i in range(4)
                    ),
                )
                for matrix in direct_sum
            ),
        )
    )
    local_tensor = tensor_local_group(base, k)
    cases.append(
        VerificationCase(
            "Tensor products",
            "Aut(A2)^2 on V(A2) tensor V(A2)",
            local_tensor,
            tensor_generators(base_generators, k, False),
            lambda tau, degree: tensor_local_formula(base, k, tau, degree),
            True,
        )
    )
    tensor_wreath = tensor_wreath_group(base, k)
    cases.append(
        VerificationCase(
            "Tensor products",
            "Aut(A2) wr S2 on V(A2) tensor V(A2)",
            tensor_wreath,
            tensor_generators(base_generators, k, True),
            lambda tau, degree: tensor_wreath_formula(base, k, tau, degree),
            True,
        )
    )
    return tuple(cases)


def verify_case(case: VerificationCase, max_degree: int = MAX_DEGREE) -> dict[str, object]:
    unique_matrices = tuple(dict.fromkeys(case.matrices))
    frame_checks = 0
    frame_ok = True
    for matrix in unique_matrices:
        frame = frame_shape(matrix)
        frame_ok &= frame_determinant_identity(matrix, frame)
        direct_complete = matrix_complete_characters(matrix, max_degree)
        frame_complete = complete_from_frame(frame, max_degree)
        frame_ok &= direct_complete == frame_complete
        frame_checks += max_degree + 2

    rows = []
    for tau in partitions_up_to(max_degree):
        direct = direct_coefficient(unique_matrices, tau, max_degree)
        predicted = case.formula(tau, max_degree)
        frame_numerator = 0
        for matrix in unique_matrices:
            complete = complete_from_frame(frame_shape(matrix), max_degree)
            frame_numerator += prod(complete[degree] for degree in tau)
        frame_value = Fraction(frame_numerator, len(unique_matrices))
        rows.append(
            {
                "tau": "()" if not tau else str(tau),
                "direct matrix average": direct,
                "frame-shape sum": int(frame_value) if frame_value.denominator == 1 else str(frame_value),
                "family formula": int(predicted) if predicted.denominator == 1 else str(predicted),
                "pass": Fraction(direct) == frame_value == predicted,
            }
        )

    fixed_rows = []
    for tau in FIXED_SPACE_TAUS:
        direct = direct_coefficient(unique_matrices, tau, max_degree)
        fixed = fixed_space_dimension(case.generators, tau)
        fixed_rows.append(
            {
                "tau": str(tau),
                "common fixed-space dimension": fixed,
                "Reynolds trace": direct,
                "pass": fixed == direct,
            }
        )
    return {
        "section": case.section,
        "case": case.name,
        "dimension": len(unique_matrices[0]),
        "group order": len(unique_matrices),
        "lattice preservation": case.lattice_check,
        "frame checks": frame_checks,
        "frame pass": frame_ok,
        "rows": tuple(rows),
        "fixed-space rows": tuple(fixed_rows),
        "passed": case.lattice_check
        and frame_ok
        and all(row["pass"] for row in rows)
        and all(row["pass"] for row in fixed_rows),
    }


def run_suite(max_degree: int = MAX_DEGREE) -> dict[str, object]:
    cases = tuple(verify_case(case, max_degree) for case in _make_cases())
    return {
        "max degree": max_degree,
        "cases": cases,
        "case count": len(cases),
        "coefficient comparisons": sum(len(case["rows"]) for case in cases) * 3,
        "fixed-space comparisons": sum(len(case["fixed-space rows"]) for case in cases),
        "frame comparisons": sum(case["frame checks"] for case in cases),
        "passed": all(case["passed"] for case in cases),
    }


if __name__ == "__main__":
    suite = run_suite()
    print(
        f"{'PASS' if suite['passed'] else 'FAIL'}: {suite['case count']} cases, "
        f"{suite['coefficient comparisons']} coefficient-route comparisons, "
        f"{suite['fixed-space comparisons']} fixed-space comparisons, and "
        f"{suite['frame comparisons']} frame checks"
    )
    for case in suite["cases"]:
        print(
            f"  {'PASS' if case['passed'] else 'FAIL'} {case['case']}: "
            f"dim={case['dimension']}, |G|={case['group order']}"
        )
    raise SystemExit(0 if suite["passed"] else 1)
