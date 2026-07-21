"""Transparent finite-field and permutation-action verification tools.

The code is intentionally limited to prime fields and GF(4), which is enough
for the representative cases in the accompanying notebooks.  Every group is
constructed as finite-field matrices before its permutation representation is
formed.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import combinations_with_replacement, permutations, product
from math import comb, prod

import numpy as np


def determinant_mod(matrix: tuple[int, ...], n: int, q: int) -> int:
    a = [list(matrix[i * n : (i + 1) * n]) for i in range(n)]
    answer = 1
    for column in range(n):
        pivot = next((row for row in range(column, n) if a[row][column] % q), None)
        if pivot is None:
            return 0
        if pivot != column:
            a[column], a[pivot] = a[pivot], a[column]
            answer = -answer
        value = a[column][column] % q
        answer = answer * value % q
        inverse = pow(value, -1, q)
        a[column] = [(entry * inverse) % q for entry in a[column]]
        for row in range(column + 1, n):
            factor = a[row][column] % q
            if factor:
                a[row] = [
                    (a[row][j] - factor * a[column][j]) % q for j in range(n)
                ]
    return answer % q


@lru_cache(maxsize=None)
def general_linear_group(n: int, q: int, determinant: int | None = None):
    """Enumerate GL_n(q), or its requested determinant fiber, for prime q."""

    matrices = []
    for entries in product(range(q), repeat=n * n):
        det = determinant_mod(entries, n, q)
        if det and (determinant is None or det == determinant % q):
            matrices.append(entries)
    return tuple(matrices)


def mat_vec(matrix: tuple[int, ...], vector: tuple[int, ...], q: int):
    n = len(vector)
    return tuple(
        sum(matrix[row * n + column] * vector[column] for column in range(n)) % q
        for row in range(n)
    )


def inverse_mod(matrix: tuple[int, ...], n: int, q: int):
    left = [list(matrix[i * n : (i + 1) * n]) for i in range(n)]
    right = [[int(i == j) for j in range(n)] for i in range(n)]
    for column in range(n):
        pivot = next(row for row in range(column, n) if left[row][column] % q)
        left[column], left[pivot] = left[pivot], left[column]
        right[column], right[pivot] = right[pivot], right[column]
        scale = pow(left[column][column] % q, -1, q)
        left[column] = [(x * scale) % q for x in left[column]]
        right[column] = [(x * scale) % q for x in right[column]]
        for row in range(n):
            if row == column:
                continue
            factor = left[row][column] % q
            if factor:
                left[row] = [
                    (left[row][j] - factor * left[column][j]) % q for j in range(n)
                ]
                right[row] = [
                    (right[row][j] - factor * right[column][j]) % q for j in range(n)
                ]
    return tuple(entry for row in right for entry in row)


def nonzero_vectors(n: int, q: int):
    return tuple(vector for vector in product(range(q), repeat=n) if any(vector))


def canonical_line(vector: tuple[int, ...], q: int):
    first = next(entry for entry in vector if entry)
    inverse = pow(first, -1, q)
    return tuple(entry * inverse % q for entry in vector)


def projective_points(n: int, q: int):
    return tuple(sorted({canonical_line(vector, q) for vector in nonzero_vectors(n, q)}))


def induced_permutations(matrices, points, action):
    position = {point: index for index, point in enumerate(points)}
    permutations = {
        tuple(position[action(matrix, point)] for point in points) for matrix in matrices
    }
    return tuple(sorted(permutations))


def linear_action(matrices, points, q: int, projective: bool = False):
    return induced_permutations(
        matrices,
        points,
        lambda matrix, point: canonical_line(mat_vec(matrix, point, q), q)
        if projective
        else mat_vec(matrix, point, q),
    )


def permutation_matrix(permutation):
    matrix = np.zeros((len(permutation), len(permutation)), dtype=float)
    matrix[permutation, np.arange(len(permutation))] = 1.0
    return matrix


def cycle_counts(permutation):
    seen = set()
    counts = Counter()
    for start in range(len(permutation)):
        if start in seen:
            continue
        current = start
        length = 0
        while current not in seen:
            seen.add(current)
            current = permutation[current]
            length += 1
        counts[length] += 1
    return counts


def permutation_power(permutation, exponent: int):
    result = tuple(range(len(permutation)))
    base = tuple(permutation)
    while exponent:
        if exponent & 1:
            result = tuple(base[result[i]] for i in range(len(result)))
        base = tuple(base[base[i]] for i in range(len(base)))
        exponent //= 2
    return result


def mobius(value: int) -> int:
    primes = 0
    candidate = 2
    remaining = value
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


def reconstructed_cycle_counts(permutation):
    actual = cycle_counts(permutation)
    maximum = max(actual)
    result = {}
    for d in range(1, maximum + 1):
        numerator = 0
        for r in range(1, d + 1):
            if d % r == 0:
                powered = permutation_power(permutation, r)
                fixed = sum(powered[i] == i for i in range(len(powered)))
                numerator += mobius(d // r) * fixed
        value = numerator // d
        if value:
            result[d] = value
    return Counter(result)


def complete_from_cycles(counts: Counter, degree: int) -> int:
    coefficients = [0] * (degree + 1)
    coefficients[0] = 1
    for length, count in counts.items():
        updated = [0] * (degree + 1)
        for old_degree, old_value in enumerate(coefficients):
            for copies in range((degree - old_degree) // length + 1):
                updated[old_degree + copies * length] += old_value * comb(
                    count + copies - 1, copies
                )
        coefficients = updated
    return coefficients[degree]


def formula_coefficient(permutations, tau: tuple[int, ...]) -> int:
    numerator = sum(
        prod(complete_from_cycles(cycle_counts(permutation), degree) for degree in tau)
        for permutation in permutations
    )
    assert numerator % len(permutations) == 0
    return numerator // len(permutations)


@lru_cache(maxsize=None)
def block_multisets(number_of_points: int, degree: int):
    return tuple(combinations_with_replacement(range(number_of_points), degree))


def direct_orbit_count(permutations, tau: tuple[int, ...]) -> int:
    """Count orbits on tuples of multisets, without cycle-index extraction."""

    configurations = set(product(*(block_multisets(len(permutations[0]), d) for d in tau)))
    orbits = 0
    while configurations:
        representative = next(iter(configurations))
        orbit = {
            tuple(tuple(sorted(permutation[x] for x in block)) for block in representative)
            for permutation in permutations
        }
        configurations.difference_update(orbit)
        orbits += 1
    return orbits


def numerical_molien_comparison(permutations, t_values=(0.07, 0.11)):
    direct = 0.0
    cycles = 0.0
    for permutation in permutations:
        matrix = permutation_matrix(permutation)
        direct += prod(1 / np.linalg.det(np.eye(len(permutation)) - t * matrix) for t in t_values)
        counts = cycle_counts(permutation)
        cycles += prod(
            (1 - t**length) ** (-count)
            for t in t_values
            for length, count in counts.items()
        )
    direct /= len(permutations)
    cycles /= len(permutations)
    return float(direct), float(cycles), float(abs(direct - cycles))


def verify_action(name, permutations, expected: dict[tuple[int, ...], int]):
    rows = []
    for tau, predicted in expected.items():
        formula = formula_coefficient(permutations, tau)
        direct = direct_orbit_count(permutations, tau)
        rows.append(
            {
                "action": name,
                "degree": sum(tau),
                "tau": str(tau),
                "group image order": len(permutations),
                "set size": len(permutations[0]),
                "direct orbit count": direct,
                "cycle formula": formula,
                "claimed value": predicted,
                "pass": direct == formula == predicted,
            }
        )
    return rows


def all_cycle_reconstructions_pass(permutations):
    return all(reconstructed_cycle_counts(p) == cycle_counts(p) for p in permutations)


def complete_flags_f2(n: int):
    if n == 2:
        return projective_points(2, 2), lambda matrix, point: mat_vec(matrix, point, 2)
    if n != 3:
        raise ValueError("the transparent flag model is implemented for n=2,3")
    vectors = nonzero_vectors(3, 2)
    flags = tuple((point, covector) for point in vectors for covector in vectors if sum(a*b for a, b in zip(point, covector)) % 2 == 0)

    def action(matrix, flag):
        point, covector = flag
        moved_point = mat_vec(matrix, point, 2)
        inverse = inverse_mod(matrix, 3, 2)
        inverse_transpose = tuple(inverse[column * 3 + row] for row in range(3) for column in range(3))
        return moved_point, mat_vec(inverse_transpose, covector, 2)

    return flags, action


def symplectic_group_4_2():
    vectors = nonzero_vectors(4, 2)

    def pairing(x, y):
        return (x[0]*y[1] + x[1]*y[0] + x[2]*y[3] + x[3]*y[2]) % 2

    return tuple(
        matrix
        for matrix in general_linear_group(4, 2)
        if all(pairing(mat_vec(matrix, x, 2), mat_vec(matrix, y, 2)) == pairing(x, y) for x in vectors for y in vectors)
    )


def orthogonal_plus_group_4_2():
    vectors = tuple(product(range(2), repeat=4))
    quadratic = lambda x: (x[0]*x[1] + x[2]*x[3]) % 2
    return tuple(
        matrix
        for matrix in general_linear_group(4, 2)
        if all(quadratic(mat_vec(matrix, x, 2)) == quadratic(x) for x in vectors)
    )


def gf4_multiply(left: int, right: int) -> int:
    answer = 0
    a, b = left, right
    while b:
        if b & 1:
            answer ^= a
        b >>= 1
        a <<= 1
        if a & 4:
            a ^= 7  # x^2 + x + 1
    return answer & 3


def gf4_conjugate(value: int) -> int:
    return gf4_multiply(value, value)


def gf4_inverse(value: int) -> int:
    if not value:
        raise ZeroDivisionError
    return gf4_multiply(value, value)  # every nonzero element has order three


def hermitian_inner(left, right):
    answer = 0
    for x, y in zip(left, right):
        answer ^= gf4_multiply(gf4_conjugate(x), y)
    return answer


def gf4_mat_vec(matrix, vector):
    n = len(vector)
    result = []
    for row in range(n):
        value = 0
        for column in range(n):
            value ^= gf4_multiply(matrix[row * n + column], vector[column])
        result.append(value)
    return tuple(result)


def canonical_gf4_line(vector):
    first = next(value for value in vector if value)
    scale = gf4_inverse(first)
    return tuple(gf4_multiply(scale, value) for value in vector)


@lru_cache(maxsize=None)
def unitary_group_4_2():
    vectors = tuple(product(range(4), repeat=4))
    unit = tuple(vector for vector in vectors if hermitian_inner(vector, vector) == 1)
    matrices = []

    def extend(columns):
        if len(columns) == 4:
            matrices.append(tuple(columns[column][row] for row in range(4) for column in range(4)))
            return
        for candidate in unit:
            if all(hermitian_inner(previous, candidate) == 0 for previous in columns):
                extend((*columns, candidate))

    extend(())
    return tuple(matrices)


def unitary_isotropic_points_4_2():
    vectors = tuple(vector for vector in product(range(4), repeat=4) if any(vector))
    return tuple(sorted({canonical_gf4_line(v) for v in vectors if hermitian_inner(v, v) == 0}))


def _f2_polar_points(family: str, rank: int):
    points = tuple(vector for vector in product(range(2), repeat=2 * rank) if any(vector))
    if family == "orthogonal_plus":
        points = tuple(
            vector
            for vector in points
            if sum(vector[2 * i] * vector[2 * i + 1] for i in range(rank)) % 2 == 0
        )
    return points


def _f2_pairing(left, right):
    return sum(
        left[2 * i] * right[2 * i + 1] + left[2 * i + 1] * right[2 * i]
        for i in range(len(left) // 2)
    ) % 2


def _f2_triple_type(triple):
    candidates = []
    for order in permutations(range(3)):
        ordered = tuple(triple[index] for index in order)
        equality = tuple(
            int(ordered[i] == ordered[j]) for i in range(3) for j in range(i + 1, 3)
        )
        gram = tuple(
            _f2_pairing(ordered[i], ordered[j])
            for i in range(3)
            for j in range(i + 1, 3)
        )
        candidates.append(equality + gram)
    return min(candidates)


def _unitary_isotropic_points(rank: int):
    vectors = (
        vector for vector in product(range(4), repeat=2 * rank) if any(vector)
    )
    return tuple(
        sorted(
            {
                canonical_gf4_line(vector)
                for vector in vectors
                if hermitian_inner(vector, vector) == 0
            }
        )
    )


def _unitary_triple_type(triple):
    """Canonical formed-space data for an unordered triple of isotropic lines."""

    candidates = []
    for order in permutations(range(3)):
        left, middle, right = (triple[index] for index in order)
        a = hermitian_inner(left, middle)
        b = hermitian_inner(left, right)
        c = hermitian_inner(middle, right)
        # a*c*conj(b) is invariant under rescaling the three line vectors.
        triangle = gf4_multiply(gf4_multiply(a, c), gf4_conjugate(b))
        candidates.append(
            (
                int(left == middle),
                int(left == right),
                int(middle == right),
                int(a != 0),
                int(b != 0),
                int(c != 0),
                triangle,
            )
        )
    return min(candidates)


def polar_degree_three_type_counts():
    """Compare exact unordered-triple formed-space types in Witt ranks 2 and 3."""

    results = {}
    for family in ("symplectic", "orthogonal_plus"):
        counts = []
        for rank in (2, 3):
            points = _f2_polar_points(family, rank)
            fixed = points[0]
            counts.append(
                len({_f2_triple_type((fixed, left, right)) for left in points for right in points})
            )
        results[family] = tuple(counts)
    counts = []
    for rank in (2, 3):
        points = _unitary_isotropic_points(rank)
        fixed = points[0]
        counts.append(
            len({_unitary_triple_type((fixed, left, right)) for left in points for right in points})
        )
    results["unitary"] = tuple(counts)
    return results


def linear_suite():
    rows = []
    gl22 = general_linear_group(2, 2)
    gl32 = general_linear_group(3, 2)
    gl23 = general_linear_group(2, 3)
    sl23 = general_linear_group(2, 3, determinant=1)

    p2 = projective_points(2, 2)
    p3 = projective_points(3, 2)
    p23 = projective_points(2, 3)
    v23 = nonzero_vectors(2, 3)

    psl22 = linear_action(gl22, p2, 2, projective=True)
    psl32 = linear_action(gl32, p3, 2, projective=True)
    gl23_vec = linear_action(gl23, v23, 3)
    sl23_vec = linear_action(sl23, v23, 3)
    pgl23 = linear_action(gl23, p23, 3, projective=True)
    psl23 = linear_action(sl23, p23, 3, projective=True)

    stable_projective = {(1,): 1, (2,): 2, (1, 1): 2, (3,): 4, (2, 1): 5, (1, 1, 1): 6}
    rows += verify_action("PSL_3(2) on P^2(F_2)", psl32, stable_projective)
    rows += verify_action("PSL_2(2) on P^1(F_2), below stable degree 3", psl22, {(3,): 3})
    rows += verify_action("GL_2(3) on nonzero vectors", gl23_vec, {(2,): 3, (1, 1): 3})
    rows += verify_action(
        "SL_2(3) on nonzero vectors (boundary n=degree)",
        sl23_vec,
        {(2,): 3, (1, 1): 4},
    )
    rows += verify_action("PGL_2(3) on projective points", pgl23, {(2,): 2, (1, 1): 2})
    rows += verify_action("PSL_2(3) on projective points", psl23, {(2,): 2, (1, 1): 2})

    flag_checks = []
    for n, group, expected in ((2, gl22, 2), (3, gl32, 6)):
        flags, action = complete_flags_f2(n)
        permutations = induced_permutations(group, flags, action)
        result = verify_action(f"GL_{n}(2) on complete flags", permutations, {(1, 1): expected})
        rows += result
        flag_checks.append((n, len(flags), expected))

    numerical = numerical_molien_comparison(psl32)
    return {
        "rows": tuple(rows),
        "passed": all(row["pass"] for row in rows),
        "cycle reconstruction": all_cycle_reconstructions_pass(psl32),
        "numerical determinant": numerical,
        "flag checks": tuple(flag_checks),
        "group orders": {
            "GL_3(2)": len(gl32), "GL_2(3)": len(gl23), "SL_2(3)": len(sl23),
            "PGL_2(3) image": len(pgl23), "PSL_2(3) image": len(psl23),
        },
    }


def polar_suite(include_unitary: bool = True):
    rows = []
    sp = symplectic_group_4_2()
    sp_points = nonzero_vectors(4, 2)
    sp_permutations = linear_action(sp, sp_points, 2)
    rows += verify_action("Sp_4(2) on isotropic points", sp_permutations, {(2,): 3, (1, 1): 3})

    orthogonal = orthogonal_plus_group_4_2()
    singular = tuple(point for point in nonzero_vectors(4, 2) if (point[0]*point[1] + point[2]*point[3]) % 2 == 0)
    orthogonal_permutations = linear_action(orthogonal, singular, 2)
    rows += verify_action("O_4^+(2) on singular points", orthogonal_permutations, {(2,): 3, (1, 1): 3})

    unitary_order = None
    unitary_image_order = None
    unitary_reconstruction = None
    if include_unitary:
        unitary = unitary_group_4_2()
        points = unitary_isotropic_points_4_2()
        unitary_permutations = induced_permutations(
            unitary, points, lambda matrix, point: canonical_gf4_line(gf4_mat_vec(matrix, point))
        )
        rows += verify_action("U_4(2) on isotropic points", unitary_permutations, {(2,): 3, (1, 1): 3})
        unitary_order = len(unitary)
        unitary_image_order = len(unitary_permutations)
        unitary_reconstruction = all_cycle_reconstructions_pass(unitary_permutations)

    numerical = numerical_molien_comparison(sp_permutations)
    return {
        "rows": tuple(rows),
        "passed": all(row["pass"] for row in rows),
        "cycle reconstruction": all_cycle_reconstructions_pass(sp_permutations) and unitary_reconstruction is not False,
        "numerical determinant": numerical,
        "group orders": {
            "Sp_4(2)": len(sp), "O_4^+(2)": len(orthogonal), "U_4(2)": unitary_order,
            "projective U_4(2) image": unitary_image_order,
        },
    }
