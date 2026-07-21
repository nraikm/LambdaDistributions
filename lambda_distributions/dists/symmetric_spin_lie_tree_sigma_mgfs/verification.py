"""Exact and matrix-level checks for the proposed finite sigma-MGF formulas.

The module keeps the computational routes deliberately separate.

* Ordinary and spin-cover representations are constructed as honest matrices.
  Their invariant dimensions are found as common fixed spaces of generators,
  while the proposed Molien expression is evaluated by a full group average.
* Foulkes characters are independently expanded from ``h_b[h_a]`` in the
  power-sum basis and compared with zero-one block-partition matrices.
* Free-Lie matrices are obtained from a concrete bracket basis inside the
  multilinear associative algebra and compared with Witt's power-sum formula.
* Rooted-tree groups are constructed as leaf-permutation matrices.  Direct
  cycle averages and pair orbits are compared with wreath/cycle-index
  recurrences and the predicted branching moments.

All combinatorial arithmetic is exact.  Floating point is used only for the
small Clifford realization of a Schur double cover and for numerical ranks of
explicit fixed-space matrices; the reported residuals make that use visible.
"""

from __future__ import annotations

from collections import Counter, deque
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, combinations_with_replacement, permutations, product
from math import comb, factorial, gcd, isclose, prod, sqrt
from typing import Callable, Iterable, Sequence

import numpy as np


Permutation = tuple[int, ...]
Partition = tuple[int, ...]
Matrix = np.ndarray
TAUS = ((1,), (2,), (1, 1), (3,), (2, 1))


# ---------------------------------------------------------------------------
# Common finite-group and symmetric-power utilities


def identity(size: int) -> Permutation:
    return tuple(range(size))


def compose(left: Permutation, right: Permutation) -> Permutation:
    """Return left after right."""

    return tuple(left[right[index]] for index in range(len(left)))


def inverse(permutation: Permutation) -> Permutation:
    answer = [0] * len(permutation)
    for source, target in enumerate(permutation):
        answer[target] = source
    return tuple(answer)


def perm_power(permutation: Permutation, exponent: int) -> Permutation:
    answer = identity(len(permutation))
    base = permutation
    while exponent:
        if exponent & 1:
            answer = compose(answer, base)
        base = compose(base, base)
        exponent //= 2
    return answer


@lru_cache(maxsize=None)
def symmetric_group(degree: int) -> tuple[Permutation, ...]:
    return tuple(permutations(range(degree)))


def adjacent_generators(degree: int) -> tuple[Permutation, ...]:
    answer = []
    for index in range(degree - 1):
        p = list(range(degree))
        p[index], p[index + 1] = p[index + 1], p[index]
        answer.append(tuple(p))
    return tuple(answer)


def cycle_counts(permutation: Permutation) -> Counter[int]:
    seen: set[int] = set()
    answer: Counter[int] = Counter()
    for start in range(len(permutation)):
        if start in seen:
            continue
        point = start
        length = 0
        while point not in seen:
            seen.add(point)
            point = permutation[point]
            length += 1
        answer[length] += 1
    return answer


def fixed_points(permutation: Permutation) -> int:
    return sum(index == image for index, image in enumerate(permutation))


def permutation_matrix(permutation: Permutation) -> Matrix:
    matrix = np.zeros((len(permutation), len(permutation)), dtype=float)
    matrix[list(permutation), np.arange(len(permutation))] = 1.0
    return matrix


def matrix_key(matrix: Matrix, digits: int = 10) -> tuple[tuple[float, float], ...]:
    flat = np.round(matrix.reshape(-1), digits)
    return tuple((float(value.real), float(value.imag)) for value in flat)


def matrix_power(matrix: Matrix, exponent: int) -> Matrix:
    return np.linalg.matrix_power(matrix, exponent)


def complete_from_power_traces(traces: Sequence[complex], maximum: int) -> list[complex]:
    """Newton recursion: d h_d = sum_{r<=d} p_r h_{d-r}."""

    values: list[complex] = [1.0 + 0.0j]
    for degree in range(1, maximum + 1):
        values.append(
            sum(traces[r] * values[degree - r] for r in range(1, degree + 1))
            / degree
        )
    return values


def molien_from_matrices(matrices: Sequence[Matrix], tau: Sequence[int]) -> int:
    maximum = max(tau, default=0)
    total = 0.0 + 0.0j
    for matrix in matrices:
        traces = [0.0 + 0.0j] + [
            complex(np.trace(matrix_power(matrix, exponent)))
            for exponent in range(1, maximum + 1)
        ]
        h = complete_from_power_traces(traces, maximum)
        total += prod(h[degree] for degree in tau)
    value = total / len(matrices)
    if abs(value.imag) > 1e-7 or abs(value.real - round(value.real)) > 1e-7:
        raise AssertionError(f"nonintegral Molien coefficient {value}")
    return int(round(value.real))


def symmetric_square_matrix(matrix: Matrix) -> Matrix:
    dimension = matrix.shape[0]
    basis = tuple(combinations_with_replacement(range(dimension), 2))
    locations = {item: index for index, item in enumerate(basis)}
    answer = np.zeros((len(basis), len(basis)), dtype=complex)
    for column, (left, right) in enumerate(basis):
        for a in range(dimension):
            for b in range(dimension):
                coefficient = matrix[a, left] * matrix[b, right]
                if coefficient == 0:
                    continue
                answer[locations[tuple(sorted((a, b)))], column] += coefficient
    return answer


def symmetric_cube_matrix(matrix: Matrix) -> Matrix:
    dimension = matrix.shape[0]
    basis = tuple(combinations_with_replacement(range(dimension), 3))
    locations = {item: index for index, item in enumerate(basis)}
    answer = np.zeros((len(basis), len(basis)), dtype=complex)
    for column, source in enumerate(basis):
        for target in product(range(dimension), repeat=3):
            coefficient = prod(matrix[target[j], source[j]] for j in range(3))
            if coefficient:
                answer[locations[tuple(sorted(target))], column] += coefficient
    return answer


def represented_matrix(matrix: Matrix, tau: Sequence[int]) -> Matrix:
    factors = []
    for degree in tau:
        if degree == 1:
            factors.append(matrix.astype(complex))
        elif degree == 2:
            factors.append(symmetric_square_matrix(matrix))
        elif degree == 3:
            factors.append(symmetric_cube_matrix(matrix))
        else:
            raise ValueError("direct fixed-space route supports degrees through three")
    answer = np.array([[1.0 + 0.0j]])
    for factor in factors:
        answer = np.kron(answer, factor)
    return answer


def fixed_space_dimension(generator_matrices: Sequence[Matrix], tau: Sequence[int]) -> tuple[int, float]:
    represented = [represented_matrix(matrix, tau) for matrix in generator_matrices]
    dimension = represented[0].shape[0]
    constraints = np.vstack([matrix - np.eye(dimension) for matrix in represented])
    singular_values = np.linalg.svd(constraints, compute_uv=False)
    tolerance = max(constraints.shape) * np.finfo(float).eps * max(singular_values, default=1.0) * 50
    rank = int(np.sum(singular_values > tolerance))
    residual = float(singular_values[rank] if rank < len(singular_values) else 0.0)
    return dimension - rank, residual


def direct_vs_molien_rows(
    matrices: Sequence[Matrix], generator_matrices: Sequence[Matrix], taus: Sequence[tuple[int, ...]]
) -> list[dict[str, object]]:
    rows = []
    for tau in taus:
        direct, residual = fixed_space_dimension(generator_matrices, tau)
        formula = molien_from_matrices(matrices, tau)
        rows.append(
            {
                "tau": str(tuple(tau)),
                "direct fixed space": direct,
                "Molien formula": formula,
                "rank residual": f"{residual:.1e}",
                "passed": direct == formula,
            }
        )
    return rows


def integer_determinant(matrix: Sequence[Sequence[int]]) -> int:
    size = len(matrix)
    if size == 0:
        return 1
    return sum(
        (-1) ** column * matrix[0][column]
        * integer_determinant([row[:column] + row[column + 1 :] for row in matrix[1:]])
        for column in range(size)
    )


# ---------------------------------------------------------------------------
# I. Symmetric groups: standard, hook, and two-row representations


def standard_matrix(permutation: Permutation) -> Matrix:
    """S^(n-1,1) on the integral basis e_i-e_(n-1)."""

    n = len(permutation)
    answer = np.zeros((n - 1, n - 1), dtype=float)
    anchor = permutation[n - 1]
    for column in range(n - 1):
        image = permutation[column]
        for row in range(n - 1):
            answer[row, column] = (1 if image == row else 0) - (1 if anchor == row else 0)
    return answer


def exterior_matrix(matrix: Matrix, degree: int) -> Matrix:
    basis = tuple(combinations(range(matrix.shape[0]), degree))
    answer = np.zeros((len(basis), len(basis)), dtype=float)
    integer = np.rint(matrix).astype(int)
    for column, source in enumerate(basis):
        for row, target in enumerate(basis):
            minor = [[int(integer[i, j]) for j in source] for i in target]
            answer[row, column] = integer_determinant(minor)
    return answer


def polynomial_multiply(left: list[int], right: list[int], maximum: int) -> list[int]:
    answer = [0] * (maximum + 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if i + j <= maximum:
                answer[i + j] += a * b
    return answer


def hook_character_from_cycles(counts: Counter[int], k: int) -> int:
    polynomial = [1] + [(-1) ** degree for degree in range(1, k + 1)]
    for length, multiplicity in counts.items():
        factor = [0] * (k + 1)
        factor[0] = 1
        if length <= k:
            factor[length] = -((-1) ** length)
        for _ in range(multiplicity):
            polynomial = polynomial_multiply(polynomial, factor, k)
    return polynomial[k]


def subset_character(counts: Counter[int], k: int) -> int:
    polynomial = [1] + [0] * k
    for length, multiplicity in counts.items():
        factor = [0] * (k + 1)
        factor[0] = 1
        if length <= k:
            factor[length] = 1
        for _ in range(multiplicity):
            polynomial = polynomial_multiply(polynomial, factor, k)
    return polynomial[k]


def rref_pivot_columns(matrix: list[list[Fraction]]) -> tuple[int, ...]:
    values = [row[:] for row in matrix]
    rows = len(values)
    columns = len(values[0]) if rows else 0
    pivot_row = 0
    pivots = []
    for column in range(columns):
        selected = next((row for row in range(pivot_row, rows) if values[row][column]), None)
        if selected is None:
            continue
        values[pivot_row], values[selected] = values[selected], values[pivot_row]
        pivot = values[pivot_row][column]
        values[pivot_row] = [value / pivot for value in values[pivot_row]]
        for row in range(rows):
            if row == pivot_row or not values[row][column]:
                continue
            scale = values[row][column]
            values[row] = [a - scale * b for a, b in zip(values[row], values[pivot_row], strict=True)]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break
    return tuple(pivots)


def invert_fraction_matrix(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    size = len(matrix)
    augmented = [row[:] + [Fraction(int(i == j)) for j in range(size)] for i, row in enumerate(matrix)]
    for column in range(size):
        selected = next(row for row in range(column, size) if augmented[row][column])
        augmented[column], augmented[selected] = augmented[selected], augmented[column]
        pivot = augmented[column][column]
        augmented[column] = [value / pivot for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            scale = augmented[row][column]
            if scale:
                augmented[row] = [a - scale * b for a, b in zip(augmented[row], augmented[column], strict=True)]
    return [row[size:] for row in augmented]


def matvec_fraction(matrix: Sequence[Sequence[Fraction]], vector: Sequence[Fraction]) -> list[Fraction]:
    return [sum(entry * value for entry, value in zip(row, vector, strict=True)) for row in matrix]


def nullspace_basis(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    values = [row[:] for row in matrix]
    rows, columns = len(values), len(values[0])
    pivot_row = 0
    pivot_columns = []
    for column in range(columns):
        selected = next((row for row in range(pivot_row, rows) if values[row][column]), None)
        if selected is None:
            continue
        values[pivot_row], values[selected] = values[selected], values[pivot_row]
        pivot = values[pivot_row][column]
        values[pivot_row] = [value / pivot for value in values[pivot_row]]
        for row in range(rows):
            if row != pivot_row and values[row][column]:
                scale = values[row][column]
                values[row] = [a - scale * b for a, b in zip(values[row], values[pivot_row], strict=True)]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break
    free = [column for column in range(columns) if column not in pivot_columns]
    basis = []
    for free_column in free:
        vector = [Fraction(0) for _ in range(columns)]
        vector[free_column] = 1
        for row, pivot_column in enumerate(pivot_columns):
            vector[pivot_column] = -values[row][free_column]
        basis.append(vector)
    return basis


def two_row_matrices(degree: int) -> dict[Permutation, Matrix]:
    """Realize S^(n-2,2) as the vertex-sum-zero subspace of edge space."""

    edges = tuple(combinations(range(degree), 2))
    locations = {edge: index for index, edge in enumerate(edges)}
    incidence = [
        [Fraction(int(vertex in edge)) for edge in edges]
        for vertex in range(degree)
    ]
    basis_vectors = nullspace_basis(incidence)
    dimension = len(basis_vectors)
    basis_matrix = [[basis_vectors[column][row] for column in range(dimension)] for row in range(len(edges))]
    pivot_rows = rref_pivot_columns([[basis_matrix[row][column] for row in range(len(edges))] for column in range(dimension)])
    square = [[basis_matrix[row][column] for column in range(dimension)] for row in pivot_rows]
    inverse_square = invert_fraction_matrix(square)
    answer: dict[Permutation, Matrix] = {}
    for permutation in symmetric_group(degree):
        columns = []
        for vector in basis_vectors:
            image = [Fraction(0) for _ in edges]
            for edge, coefficient in zip(edges, vector, strict=True):
                target = tuple(sorted((permutation[edge[0]], permutation[edge[1]])))
                image[locations[target]] += coefficient
            coordinates = matvec_fraction(inverse_square, [image[row] for row in pivot_rows])
            columns.append(coordinates)
        answer[permutation] = np.array(columns, dtype=float).T
    return answer


def symmetric_representation_suite() -> dict[str, object]:
    cases = []
    character_checks = 0
    for n, kind, k in ((4, "standard hook", 1), (5, "hook", 2), (4, "two-row", 2), (5, "two-row", 2)):
        group = symmetric_group(n)
        generators = adjacent_generators(n)
        if kind in {"standard hook", "hook"}:
            matrices_by_element = {
                g: exterior_matrix(standard_matrix(g), k) for g in group
            }
            formula_character: Callable[[Counter[int]], int] = lambda counts, kk=k: hook_character_from_cycles(counts, kk)
            label = f"S_{n}, S^({n-k},1^{k})"
        else:
            matrices_by_element = two_row_matrices(n)
            formula_character = lambda counts, kk=k: subset_character(counts, kk) - subset_character(counts, kk - 1)
            label = f"S_{n}, S^({n-k},{k})"
        for g, matrix in matrices_by_element.items():
            for exponent in range(1, 5):
                observed = int(round(float(np.trace(matrix_power(matrix, exponent)).real)))
                proposed = formula_character(cycle_counts(perm_power(g, exponent)))
                if observed != proposed:
                    raise AssertionError((label, g, exponent, observed, proposed))
                character_checks += 1
        matrices = tuple(matrices_by_element[g] for g in group)
        generator_matrices = tuple(matrices_by_element[g] for g in generators)
        rows = direct_vs_molien_rows(matrices, generator_matrices, ((1,), (2,), (1, 1), (3,)))
        cases.append(
            {
                "representation": label,
                "group order": len(group),
                "dimension": matrices[0].shape[0],
                "rows": rows,
                "passed": all(row["passed"] for row in rows),
            }
        )
    return {
        "cases": cases,
        "character-power checks": character_checks,
        "rows": [{"representation": case["representation"], **row} for case in cases for row in case["rows"]],
        "passed": all(case["passed"] for case in cases),
    }


# ---------------------------------------------------------------------------
# II. A Schur double cover inside Pin(n)


def gamma_matrices(dimension: int) -> tuple[Matrix, ...]:
    identity2 = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron_all(factors: Sequence[Matrix]) -> Matrix:
        answer = np.array([[1.0 + 0.0j]])
        for factor in factors:
            answer = np.kron(answer, factor)
        return answer

    rank = dimension // 2
    gammas = []
    for index in range(rank):
        gammas.append(kron_all([z] * index + [x] + [identity2] * (rank - index - 1)))
        gammas.append(kron_all([z] * index + [y] + [identity2] * (rank - index - 1)))
    if dimension % 2:
        gammas.append(kron_all([z] * rank))
    return tuple(gammas)


def generated_matrix_group(generators: Sequence[Matrix], maximum_order: int = 1000) -> tuple[Matrix, ...]:
    dimension = generators[0].shape[0]
    identity_matrix = np.eye(dimension, dtype=complex)
    known = {matrix_key(identity_matrix): identity_matrix}
    queue = deque([identity_matrix])
    while queue:
        current = queue.popleft()
        for generator in generators:
            candidate = current @ generator
            key = matrix_key(candidate)
            if key not in known:
                known[key] = candidate
                queue.append(candidate)
                if len(known) > maximum_order:
                    raise AssertionError("generated matrix group exceeded expected finite bound")
    return tuple(known.values())


def schur_cover_suite() -> dict[str, object]:
    n = 4
    gammas = gamma_matrices(n)
    generators = tuple((gammas[index] - gammas[index + 1]) / sqrt(2) for index in range(n - 1))
    matrices = generated_matrix_group(generators)
    if len(matrices) != 2 * factorial(n):
        raise AssertionError(f"Pin preimage has order {len(matrices)}, expected {2*factorial(n)}")
    central = -np.eye(matrices[0].shape[0], dtype=complex)
    if matrix_key(central) not in {matrix_key(matrix) for matrix in matrices}:
        raise AssertionError("central involution is missing")
    rows = direct_vs_molien_rows(
        matrices,
        generators + (central,),
        ((1,), (2,), (1, 1), (3,), (1, 1, 1, 1)),
    )
    odd_zero = all(
        row["Molien formula"] == 0
        for row in rows
        if sum(int(value) for value in row["tau"].strip("()").replace(",", " ").split()) % 2
    )
    clifford_residual = max(
        float(np.linalg.norm(left @ right + right @ left - (2 * np.eye(left.shape[0]) if i == j else 0)))
        for i, left in enumerate(gammas)
        for j, right in enumerate(gammas)
    )
    return {
        "group": "Pin preimage 2.S_4",
        "group order": len(matrices),
        "dimension": matrices[0].shape[0],
        "Clifford residual": clifford_residual,
        "rows": rows,
        "odd-degree vanishing": odd_zero,
        "passed": all(row["passed"] for row in rows) and odd_zero and clifford_residual < 1e-12,
    }


# ---------------------------------------------------------------------------
# III. Foulkes modules and multilinear free-Lie modules


def set_partitions_equal_blocks(size: int, block_size: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if size % block_size:
        raise ValueError("block size must divide set size")

    def recurse(remaining: tuple[int, ...]) -> Iterable[tuple[tuple[int, ...], ...]]:
        if not remaining:
            yield ()
            return
        first = remaining[0]
        for tail in combinations(remaining[1:], block_size - 1):
            block = tuple(sorted((first,) + tail))
            rest = tuple(value for value in remaining if value not in block)
            for other in recurse(rest):
                yield (block,) + other

    return tuple(tuple(sorted(blocks)) for blocks in recurse(tuple(range(size))))


def action_on_points(permutation: Permutation, points: Sequence, action: Callable) -> Permutation:
    locations = {point: index for index, point in enumerate(points)}
    return tuple(locations[action(permutation, point)] for point in points)


def integer_partitions(total: int, maximum: int | None = None) -> tuple[Partition, ...]:
    if total == 0:
        return ((),)
    maximum = min(total, maximum if maximum is not None else total)
    answer = []
    for first in range(maximum, 0, -1):
        for tail in integer_partitions(total - first, first):
            answer.append((first,) + tail)
    return tuple(answer)


def z_partition(partition: Partition) -> int:
    counts = Counter(partition)
    return prod(length ** multiplicity * factorial(multiplicity) for length, multiplicity in counts.items())


def power_sum_multiply(
    left: dict[Partition, Fraction], right: dict[Partition, Fraction]
) -> dict[Partition, Fraction]:
    answer: Counter[Partition] = Counter()
    for a, left_value in left.items():
        for b, right_value in right.items():
            answer[tuple(sorted(a + b, reverse=True))] += left_value * right_value
    return dict(answer)


def foulkes_power_sum(a: int, b: int) -> dict[Partition, Fraction]:
    h_a = {partition: Fraction(1, z_partition(partition)) for partition in integer_partitions(a)}
    answer: Counter[Partition] = Counter()
    for outer in integer_partitions(b):
        term = {(): Fraction(1, z_partition(outer))}
        for scale in outer:
            substituted = {
                tuple(sorted((scale * part for part in partition), reverse=True)): coefficient
                for partition, coefficient in h_a.items()
            }
            term = power_sum_multiply(term, substituted)
        answer.update(term)
    return dict(answer)


def direct_configuration_orbits(group: Sequence[Permutation], tau: Sequence[int]) -> int:
    blocks = tuple(tuple(combinations_with_replacement(range(len(group[0])), degree)) for degree in tau)
    remaining = set(product(*blocks))
    number = 0
    while remaining:
        representative = remaining.pop()
        orbit = {
            tuple(tuple(sorted(g[index] for index in block)) for block in representative)
            for g in group
        }
        remaining.difference_update(orbit)
        number += 1
    return number


def h_from_cycles(permutation: Permutation, maximum: int) -> list[int]:
    values = [0] * (maximum + 1)
    values[0] = 1
    for length, multiplicity in cycle_counts(permutation).items():
        for _ in range(multiplicity):
            for degree in range(length, maximum + 1):
                values[degree] += values[degree - length]
    return values


def permutation_molien(group: Sequence[Permutation], tau: Sequence[int]) -> int:
    numerator = sum(prod(h_from_cycles(g, max(tau))[degree] for degree in tau) for g in group)
    if numerator % len(group):
        raise AssertionError("permutation Molien average not integral")
    return numerator // len(group)


def foulkes_suite() -> dict[str, object]:
    cases = []
    character_checks = 0
    for a, b in ((2, 2), (2, 3)):
        n = a * b
        points = set_partitions_equal_blocks(n, a)
        group = symmetric_group(n)

        def act(g: Permutation, partition):
            return tuple(sorted(tuple(sorted(g[value] for value in block)) for block in partition))

        actions = tuple(action_on_points(g, points, act) for g in group)
        frobenius = foulkes_power_sum(a, b)
        for g, action in zip(group, actions, strict=True):
            mu = tuple(sorted(cycle_counts(g).elements(), reverse=True))
            proposed = frobenius.get(mu, Fraction(0)) * z_partition(mu)
            if proposed.denominator != 1 or fixed_points(action) != proposed.numerator:
                raise AssertionError((a, b, mu, fixed_points(action), proposed))
            character_checks += 1
        rows = []
        for tau in ((1,), (2,), (1, 1), (3,), (2, 1)):
            formula = permutation_molien(actions, tau)
            direct = direct_configuration_orbits(actions, tau)
            rows.append({"tau": str(tau), "direct orbits": direct, "Frobenius/Molien": formula, "passed": direct == formula})
        pair_formula = contingency_table_orbits(a, b)
        pair_observed = next(row["direct orbits"] for row in rows if row["tau"] == str((1, 1)))
        cases.append(
            {
                "module": f"H^({a}^{b})",
                "ambient S_n": f"S_{n}",
                "dimension": len(points),
                "rows": rows,
                "contingency formula": pair_formula,
                "passed": all(row["passed"] for row in rows) and pair_formula == pair_observed,
            }
        )
    return {
        "cases": cases,
        "character checks": character_checks,
        "rows": [{"module": case["module"], **row} for case in cases for row in case["rows"]],
        "passed": all(case["passed"] for case in cases),
    }


def contingency_table_orbits(a: int, b: int) -> int:
    """Count b-by-b nonnegative tables with all margins a modulo row/column permutations."""

    rows = tuple(compositions(a, b))
    tables = []
    for candidate in product(rows, repeat=b):
        if all(sum(candidate[i][j] for i in range(b)) == a for j in range(b)):
            tables.append(candidate)
    row_column = symmetric_group(b)

    def canonical(table):
        images = []
        for row_perm in row_column:
            for column_perm in row_column:
                images.append(tuple(tuple(table[row_perm[i]][column_perm[j]] for j in range(b)) for i in range(b)))
        return min(images)

    return len({canonical(table) for table in tables})


def compositions(total: int, parts: int) -> tuple[tuple[int, ...], ...]:
    if parts == 1:
        return ((total,),)
    return tuple((first,) + tail for first in range(total + 1) for tail in compositions(total - first, parts - 1))


def bracket(left: dict[tuple[int, ...], int], right: dict[tuple[int, ...], int]) -> dict[tuple[int, ...], int]:
    answer: Counter[tuple[int, ...]] = Counter()
    for a, av in left.items():
        for b, bv in right.items():
            answer[a + b] += av * bv
            answer[b + a] -= av * bv
    return {word: coefficient for word, coefficient in answer.items() if coefficient}


def lie_bracket_basis(n: int) -> tuple[dict[tuple[int, ...], int], ...]:
    answer = []
    for order in permutations(range(n - 1)):
        value = {(n - 1,): 1}
        for letter in order:
            value = bracket(value, {(letter,): 1})
        answer.append(value)
    return tuple(answer)


def lie_matrices(n: int) -> dict[Permutation, Matrix]:
    words = tuple(permutations(range(n)))
    word_locations = {word: index for index, word in enumerate(words)}
    basis = lie_bracket_basis(n)
    dimension = len(basis)
    basis_matrix = [[Fraction(vector.get(word, 0)) for vector in basis] for word in words]
    pivot_rows = rref_pivot_columns([[basis_matrix[row][column] for row in range(len(words))] for column in range(dimension)])
    square = [[basis_matrix[row][column] for column in range(dimension)] for row in pivot_rows]
    inverse_square = invert_fraction_matrix(square)
    answer = {}
    for g in symmetric_group(n):
        columns = []
        for vector in basis:
            image = {tuple(g[letter] for letter in word): coefficient for word, coefficient in vector.items()}
            selected = [Fraction(image.get(words[row], 0)) for row in pivot_rows]
            columns.append(matvec_fraction(inverse_square, selected))
        matrix = np.array(columns, dtype=float).T
        if np.max(np.abs(matrix - np.rint(matrix))) > 1e-12:
            raise AssertionError("Lie action matrix should be integral in the bracket basis")
        answer[g] = matrix
    return answer


def mobius(value: int) -> int:
    remaining = value
    primes = 0
    candidate = 2
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


def lie_formula_character(cycle_type: Partition) -> int:
    n = sum(cycle_type)
    counts = Counter(cycle_type)
    if len(counts) != 1:
        return 0
    d, multiplicity = next(iter(counts.items()))
    if d * multiplicity != n:
        return 0
    coefficient = Fraction(mobius(d), n)
    return int(coefficient * z_partition(cycle_type))


def lie_second_moment_formula(n: int) -> int:
    value = sum(
        mobius(d) ** 2 * d ** (n // d) * factorial(n // d)
        for d in range(1, n + 1)
        if n % d == 0
    )
    if value % (n * n):
        raise AssertionError("Witt norm formula should be integral")
    return value // (n * n)


def free_lie_suite() -> dict[str, object]:
    cases = []
    character_checks = 0
    for n in (3, 4):
        matrices_by_element = lie_matrices(n)
        group = symmetric_group(n)
        for g, matrix in matrices_by_element.items():
            observed = int(round(float(np.trace(matrix))))
            cycle_type = tuple(sorted(cycle_counts(g).elements(), reverse=True))
            proposed = lie_formula_character(cycle_type)
            if observed != proposed:
                raise AssertionError((n, g, observed, proposed))
            character_checks += 1
        matrices = tuple(matrices_by_element[g] for g in group)
        generators = tuple(matrices_by_element[g] for g in adjacent_generators(n))
        rows = direct_vs_molien_rows(matrices, generators, ((1,), (2,), (1, 1), (3,)))
        observed_second = next(row["Molien formula"] for row in rows if row["tau"] == str((1, 1)))
        predicted_second = lie_second_moment_formula(n)
        cases.append(
            {
                "module": f"Lie_{n}",
                "group order": factorial(n),
                "dimension": factorial(n - 1),
                "rows": rows,
                "second moment formula": predicted_second,
                "passed": all(row["passed"] for row in rows) and observed_second == predicted_second,
            }
        )
    return {
        "cases": cases,
        "character checks": character_checks,
        "rows": [{"module": case["module"], **row} for case in cases for row in case["rows"]],
        "passed": all(case["passed"] for case in cases),
    }


# ---------------------------------------------------------------------------
# IV-V. Rooted-tree permutation groups


def cyclic_regular_group(degree: int) -> tuple[Permutation, ...]:
    return tuple(tuple((index + shift) % degree for index in range(degree)) for shift in range(degree))


def wreath_action_group(
    base_group: Sequence[Permutation], local_group: Sequence[Permutation]
) -> tuple[Permutation, ...]:
    branches = len(local_group[0])
    base_size = len(base_group[0])
    answer = []
    for top in local_group:
        for sections in product(base_group, repeat=branches):
            action = [0] * (branches * base_size)
            for branch in range(branches):
                for point in range(base_size):
                    action[branch * base_size + point] = top[branch] * base_size + sections[branch][point]
            answer.append(tuple(action))
    if len(set(answer)) != len(answer):
        raise AssertionError("wreath action is not faithful")
    return tuple(answer)


def iterated_local_group(local_group: Sequence[Permutation], height: int) -> tuple[Permutation, ...]:
    group: tuple[Permutation, ...] = ((0,),)
    for _ in range(height):
        group = wreath_action_group(group, local_group)
    return group


def cycle_index_recurrence_coefficient(
    base_group: Sequence[Permutation], local_group: Sequence[Permutation], tau: Sequence[int]
) -> Fraction:
    bounds = tuple(tau)
    zero = (0,) * len(bounds)
    base_polynomial = molien_polynomial(base_group, bounds)
    total = Fraction(0)
    for top in local_group:
        polynomial = {zero: Fraction(1)}
        for length, multiplicity in cycle_counts(top).items():
            factor = substitute_power_polynomial(base_polynomial, length, bounds)
            for _ in range(multiplicity):
                polynomial = multiply_polynomials(polynomial, factor, bounds)
        total += polynomial.get(bounds, Fraction(0))
    return total / len(local_group)


def molien_polynomial(group: Sequence[Permutation], bounds: Sequence[int]) -> dict[tuple[int, ...], Fraction]:
    result: Counter[tuple[int, ...]] = Counter()
    for action in group:
        h = h_from_cycles(action, max(bounds, default=0))
        for alpha in product(*(range(bound + 1) for bound in bounds)):
            result[alpha] += prod(h[degree] for degree in alpha)
    return {alpha: Fraction(value, len(group)) for alpha, value in result.items()}


def multiply_polynomials(
    left: dict[tuple[int, ...], Fraction],
    right: dict[tuple[int, ...], Fraction],
    bounds: Sequence[int],
) -> dict[tuple[int, ...], Fraction]:
    result: Counter[tuple[int, ...]] = Counter()
    for alpha, av in left.items():
        for beta, bv in right.items():
            exponent = tuple(a + b for a, b in zip(alpha, beta, strict=True))
            if all(value <= bound for value, bound in zip(exponent, bounds, strict=True)):
                result[exponent] += av * bv
    return dict(result)


def substitute_power_polynomial(
    polynomial: dict[tuple[int, ...], Fraction], exponent: int, bounds: Sequence[int]
) -> dict[tuple[int, ...], Fraction]:
    return {
        tuple(exponent * value for value in alpha): coefficient
        for alpha, coefficient in polynomial.items()
        if all(exponent * value <= bound for value, bound in zip(alpha, bounds, strict=True))
    }


def pair_orbits(group: Sequence[Permutation]) -> int:
    remaining = set(product(range(len(group[0])), repeat=2))
    number = 0
    while remaining:
        point = remaining.pop()
        orbit = {(g[point[0]], g[point[1]]) for g in group}
        remaining.difference_update(orbit)
        number += 1
    return number


def local_rank(local_group: Sequence[Permutation]) -> int:
    return pair_orbits(local_group)


def verify_tree_case(local_name: str, local_group: Sequence[Permutation], height: int) -> dict[str, object]:
    base = iterated_local_group(local_group, height - 1)
    group = wreath_action_group(base, local_group)
    rows = []
    for tau in ((1,), (2,), (1, 1), (3,), (2, 1), (2, 2)):
        direct = permutation_molien(group, tau)
        proposed = cycle_index_recurrence_coefficient(base, local_group, tau)
        rows.append({"tau": str(tau), "direct matrix/cycle average": direct, "wreath recurrence": str(proposed), "passed": proposed == direct})
    observed_second = sum(fixed_points(g) ** 2 for g in group) // len(group)
    predicted_second = 1 + height * (local_rank(local_group) - 1)
    orbits = pair_orbits(group)
    return {
        "family": local_name,
        "height": height,
        "dimension/leaves": len(group[0]),
        "group order": len(group),
        "rows": rows,
        "pair moment direct": observed_second,
        "pair moment formula": predicted_second,
        "pair orbits": orbits,
        "universal lower bound": height + 1,
        "passed": all(row["passed"] for row in rows) and observed_second == predicted_second and orbits >= height + 1,
    }


Tree = tuple["Tree", ...]
LEAF: Tree = ()


def tree_leaves(tree: Tree) -> int:
    return 1 if not tree else sum(tree_leaves(child) for child in tree)


@lru_cache(maxsize=None)
def automorphism_group(tree: Tree) -> tuple[Permutation, ...]:
    if not tree:
        return ((0,),)
    types = Counter(tree)
    offset = 0
    global_group: tuple[Permutation, ...] = ((),)
    for child, multiplicity in sorted(types.items(), key=repr):
        child_group = automorphism_group(child)
        local = symmetric_group(multiplicity)
        component = wreath_action_group(child_group, local)
        shifted_component = tuple(tuple(offset + value for value in action) for action in component)
        combined = []
        for left in global_group:
            for right in shifted_component:
                combined.append(left + right)
        global_group = tuple(combined)
        offset += multiplicity * tree_leaves(child)
    return global_group


def tree_recursive_coefficient(tree: Tree, tau: Sequence[int]) -> Fraction:
    if not tree:
        return Fraction(1)
    bounds = tuple(tau)
    zero = (0,) * len(bounds)
    total_polynomial = {zero: Fraction(1)}
    for child, multiplicity in Counter(tree).items():
        child_group = automorphism_group(child)
        component_polynomial: Counter[tuple[int, ...]] = Counter()
        for top in symmetric_group(multiplicity):
            polynomial = {zero: Fraction(1)}
            child_series = molien_polynomial(child_group, bounds)
            for length, count in cycle_counts(top).items():
                factor = substitute_power_polynomial(child_series, length, bounds)
                for _ in range(count):
                    polynomial = multiply_polynomials(polynomial, factor, bounds)
            for exponent, coefficient in polynomial.items():
                component_polynomial[exponent] += coefficient / factorial(multiplicity)
        total_polynomial = multiply_polynomials(total_polynomial, dict(component_polynomial), bounds)
    return total_polynomial.get(bounds, Fraction(0))


def arbitrary_tree_suite() -> dict[str, object]:
    cherry = (LEAF, LEAF)
    ternary_star = (LEAF, LEAF, LEAF)
    trees = (
        (LEAF, LEAF, cherry),
        (LEAF, cherry, cherry),
        (cherry, ternary_star),
    )
    cases = []
    for index, tree in enumerate(trees, 1):
        group = automorphism_group(tree)
        rows = []
        for tau in ((1,), (2,), (1, 1), (3,), (2, 1)):
            direct = permutation_molien(group, tau)
            proposed = tree_recursive_coefficient(tree, tau)
            rows.append({"tau": str(tau), "direct": direct, "tree cycle index": str(proposed), "passed": direct == proposed})
        cases.append(
            {
                "tree": f"mixed tree {index}",
                "leaves": tree_leaves(tree),
                "group order": len(group),
                "rows": rows,
                "passed": all(row["passed"] for row in rows),
            }
        )
    return {
        "cases": cases,
        "rows": [{"tree": case["tree"], **row} for case in cases for row in case["rows"]],
        "passed": all(case["passed"] for case in cases),
    }


def rooted_tree_suite() -> dict[str, object]:
    cases = []
    for p, height in ((2, 1), (2, 2), (2, 3), (3, 1), (3, 2)):
        cases.append(verify_tree_case(f"C_{p} local action", cyclic_regular_group(p), height))
    for degree, height in ((3, 1), (3, 2)):
        cases.append(verify_tree_case(f"S_{degree} local action", symmetric_group(degree), height))
    arbitrary = arbitrary_tree_suite()
    return {
        "iterated cases": cases,
        "rows": [{"family": case["family"], "height": case["height"], **row} for case in cases for row in case["rows"]],
        "arbitrary": arbitrary,
        "passed": all(case["passed"] for case in cases) and arbitrary["passed"],
    }


# ---------------------------------------------------------------------------
# Full audit


def run_suite() -> dict[str, object]:
    symmetric = symmetric_representation_suite()
    schur = schur_cover_suite()
    foulkes = foulkes_suite()
    lie = free_lie_suite()
    trees = rooted_tree_suite()
    checks = (
        symmetric["character-power checks"]
        + len(symmetric["rows"])
        + len(schur["rows"])
        + foulkes["character checks"]
        + len(foulkes["rows"])
        + lie["character checks"]
        + len(lie["rows"])
        + len(trees["rows"])
        + len(trees["arbitrary"]["rows"])
        + 2 * len(trees["iterated cases"])
    )
    passed = all(section["passed"] for section in (symmetric, schur, foulkes, lie, trees))
    if not passed:
        raise AssertionError("at least one verification section failed")
    return {
        "symmetric groups": symmetric,
        "Schur cover": schur,
        "Foulkes": foulkes,
        "free Lie": lie,
        "rooted trees": trees,
        "exact checks": checks,
        "passed": passed,
    }


if __name__ == "__main__":
    report = run_suite()
    print(f"PASS: {report['exact checks']} formula, character, matrix, and orbit checks")
    for name in ("symmetric groups", "Schur cover", "Foulkes", "free Lie", "rooted trees"):
        print(f"  {name}: PASS")
