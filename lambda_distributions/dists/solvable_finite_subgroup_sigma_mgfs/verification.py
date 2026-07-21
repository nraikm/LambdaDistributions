"""Exact and high-precision checks for the XI--XII sigma-MGF formulas.

The direct route always constructs the represented matrices.  Permutation
representations are stored losslessly by their nonzero column positions;
small defining complex representations are stored as dense matrices.  The
comparison route uses weighted cycles, fixed-power/image formulas, conjugacy
class spectra, or the wreath exponential recurrence.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import permutations, product
from math import comb, factorial, gcd, lcm, prod, sqrt

import numpy as np


Permutation = tuple[int, ...]
Vector = tuple[int, ...]
Matrix = tuple[int, ...]
TAUS = ((1,), (2,), (1, 1), (3,), (2, 1))


def _compose(left: Permutation, right: Permutation) -> Permutation:
    return tuple(left[right[index]] for index in range(len(right)))


def _perm_power(permutation: Permutation, exponent: int) -> Permutation:
    answer = tuple(range(len(permutation)))
    base = permutation
    while exponent:
        if exponent & 1:
            answer = _compose(answer, base)
        base = _compose(base, base)
        exponent //= 2
    return answer


def _cycle_lengths(permutation: Permutation) -> tuple[int, ...]:
    seen: set[int] = set()
    answer = []
    for start in range(len(permutation)):
        if start in seen:
            continue
        current = start
        length = 0
        while current not in seen:
            seen.add(current)
            current = permutation[current]
            length += 1
        answer.append(length)
    return tuple(sorted(answer))


def _cycle_complete(lengths: tuple[int, ...], degree: int) -> int:
    counts = Counter(lengths)
    coefficients = [0] * (degree + 1)
    coefficients[0] = 1
    for length, multiplicity in counts.items():
        updated = [0] * (degree + 1)
        for old_degree, value in enumerate(coefficients):
            for copies in range((degree - old_degree) // length + 1):
                updated[old_degree + copies * length] += value * comb(
                    multiplicity + copies - 1, copies
                )
        coefficients = updated
    return coefficients[degree]


def _permutation_dimension(actions: tuple[Permutation, ...], tau: tuple[int, ...]) -> int:
    numerator = sum(
        prod(_cycle_complete(_cycle_lengths(action), degree) for degree in tau)
        for action in actions
    )
    assert numerator % len(actions) == 0
    return numerator // len(actions)


def _pair_orbits(actions: tuple[Permutation, ...]) -> int:
    size = len(actions[0])
    unseen = {(left, right) for left in range(size) for right in range(size)}
    count = 0
    while unseen:
        seed = next(iter(unseen))
        orbit = {(action[seed[0]], action[seed[1]]) for action in actions}
        unseen.difference_update(orbit)
        count += 1
    return count


def _fixed(permutation: Permutation) -> int:
    return sum(index == image for index, image in enumerate(permutation))


def _complete_from_eigenvalues(eigenvalues, maximum_degree: int) -> np.ndarray:
    coefficients = np.zeros(maximum_degree + 1, dtype=complex)
    coefficients[0] = 1
    for eigenvalue in eigenvalues:
        for degree in range(1, maximum_degree + 1):
            coefficients[degree] += eigenvalue * coefficients[degree - 1]
    return coefficients


def _matrix_dimension(matrices: tuple[np.ndarray, ...], tau: tuple[int, ...]) -> int:
    values = []
    for matrix in matrices:
        complete = _complete_from_eigenvalues(np.linalg.eigvals(matrix), max(tau))
        values.append(prod(complete[degree] for degree in tau))
    average = sum(values) / len(values)
    rounded = int(round(float(average.real)))
    if abs(average.imag) > 2e-6 or abs(average.real - rounded) > 2e-6:
        raise ArithmeticError((tau, average))
    return rounded


def _spectral_dimension(weighted_spectra, tau: tuple[int, ...]) -> int:
    numerator = 0j
    total = 0
    for multiplicity, eigenvalues in weighted_spectra:
        complete = _complete_from_eigenvalues(eigenvalues, max(tau))
        numerator += multiplicity * prod(complete[degree] for degree in tau)
        total += multiplicity
    average = numerator / total
    rounded = int(round(float(average.real)))
    if abs(average.imag) > 2e-6 or abs(average.real - rounded) > 2e-6:
        raise ArithmeticError((tau, average))
    return rounded


def _matrix_molien(matrices: tuple[np.ndarray, ...], t_values=(0.031, -0.047)) -> complex:
    dimension = len(matrices[0])
    identity = np.eye(dimension, dtype=complex)
    return sum(
        prod(1 / np.linalg.det(identity - t * matrix) for t in t_values)
        for matrix in matrices
    ) / len(matrices)


def _spectral_molien(weighted_spectra, t_values=(0.031, -0.047)) -> complex:
    total = sum(multiplicity for multiplicity, _ in weighted_spectra)
    return sum(
        multiplicity
        * prod(1 / prod(1 - t * eigenvalue for eigenvalue in eigenvalues) for t in t_values)
        for multiplicity, eigenvalues in weighted_spectra
    ) / total


def _matrix_rows(name, matrices, formula_dimension):
    rows = []
    for tau in TAUS:
        direct = _matrix_dimension(matrices, tau)
        formula = formula_dimension(tau)
        rows.append(
            {
                "group / representation": name,
                "dimension": len(matrices[0]),
                "order / image order": len(matrices),
                "tau": str(tau),
                "direct matrix average": direct,
                "formula": formula,
                "pass": direct == formula,
            }
        )
    return tuple(rows)


def _convolve_polynomials(left, right, bound):
    answer: dict[tuple[int, ...], Fraction] = {}
    for alpha, left_value in left.items():
        for beta, right_value in right.items():
            degree = tuple(a + b for a, b in zip(alpha, beta, strict=True))
            if all(degree[index] <= bound[index] for index in range(len(bound))):
                answer[degree] = answer.get(degree, Fraction()) + left_value * right_value
    return answer


def _power_polynomial(base, exponent: int, bound):
    zero = (0,) * len(bound)
    answer = {zero: Fraction(1)}
    for _ in range(exponent):
        answer = _convolve_polynomials(answer, base, bound)
    return answer


def _lamplighter_matrices(r: int, n: int) -> tuple[np.ndarray, ...]:
    root = np.exp(2j * np.pi / r)
    matrices = []
    for labels in product(range(r), repeat=n):
        diagonal = np.diag([root**label for label in labels])
        for shift in range(n):
            permutation = np.zeros((n, n), dtype=complex)
            for column in range(n):
                permutation[(column + shift) % n, column] = 1
            matrices.append(diagonal @ permutation)
    return tuple(matrices)


def _lamplighter_formula_coefficient(r: int, n: int, tau: tuple[int, ...]) -> int:
    bound = tau
    accumulated = Fraction()
    for ell in (divisor for divisor in range(1, n + 1) if n % divisor == 0):
        base = {}
        for degree in product(*(range(entry + 1) for entry in bound)):
            if all(entry % ell == 0 for entry in degree) and sum(degree) % (r * ell) == 0:
                base[degree] = Fraction(1)
        value = _power_polynomial(base, n // ell, bound).get(bound, Fraction())
        phi = sum(gcd(k, ell) == 1 for k in range(1, ell + 1))
        accumulated += Fraction(phi, n) * value
    assert accumulated.denominator == 1
    return accumulated.numerator


def _lamplighter_formula_value(r: int, n: int, t_values=(0.031, -0.047)) -> complex:
    root = np.exp(2j * np.pi / r)
    answer = 0j
    for ell in (divisor for divisor in range(1, n + 1) if n % divisor == 0):
        a_value = sum(
            prod(1 / (1 - root**power * t**ell) for t in t_values)
            for power in range(r)
        ) / r
        phi = sum(gcd(k, ell) == 1 for k in range(1, ell + 1))
        answer += phi * a_value ** (n // ell) / n
    return answer


def _lamplighter_monomial_suite():
    cases = []
    for r, n in ((2, 3), (3, 2), (2, 4)):
        matrices = _lamplighter_matrices(r, n)
        rows = _matrix_rows(
            f"L_{{{r},{n}}} natural monomial",
            matrices,
            lambda tau, r=r, n=n: _lamplighter_formula_coefficient(r, n, tau),
        )
        numeric_error = abs(_matrix_molien(matrices) - _lamplighter_formula_value(r, n))
        cases.append(
            {
                "r": r,
                "n": n,
                "group order": len(matrices),
                "rows": rows,
                "Molien error": float(numeric_error),
                "passed": all(row["pass"] for row in rows) and numeric_error < 2e-9,
            }
        )
    return {
        "name": "finite lamplighter natural monomial representations",
        "cases": tuple(cases),
        "passed": all(case["passed"] for case in cases),
    }


def _cyclic_shift(value: Vector, shift: int) -> Vector:
    n = len(value)
    return tuple(value[(index - shift) % n] for index in range(n))


def _lamp_affine_suite():
    rows = []
    checks = 0
    passed = True
    for r, n in ((2, 3), (2, 4), (3, 3)):
        points = tuple(product(range(r), repeat=n))
        locations = {point: index for index, point in enumerate(points)}
        actions = []
        representatives = []
        for translation in points:
            for shift in range(n):
                action = tuple(
                    locations[tuple((translation[j] + _cyclic_shift(point, shift)[j]) % r for j in range(n))]
                    for point in points
                )
                actions.append(action)
                representatives.append((translation, shift))
        actions_tuple = tuple(actions)
        fixed_ok = True
        for action, (translation, shift) in zip(actions_tuple, representatives, strict=True):
            for exponent in range(1, lcm(*_cycle_lengths(action)) + 1):
                b_k = tuple(
                    sum(_cyclic_shift(translation, j * shift)[coordinate] for j in range(exponent)) % r
                    for coordinate in range(n)
                )
                powered_shift = exponent * shift % n
                image = {
                    tuple((point[j] - _cyclic_shift(point, powered_shift)[j]) % r for j in range(n))
                    for point in points
                }
                predicted = r ** gcd(n, exponent * shift) if b_k in image else 0
                observed = _fixed(_perm_power(action, exponent))
                fixed_ok &= predicted == observed
                checks += 1
        direct_pair = _pair_orbits(actions_tuple)
        necklace = sum(
            sum(gcd(k, divisor) == 1 for k in range(1, divisor + 1)) * r ** (n // divisor)
            for divisor in range(1, n + 1)
            if n % divisor == 0
        ) // n
        rows.append(
            {
                "action": f"L_{{{r},{n}}} on C_{r}^{n}",
                "degree": len(points),
                "group order": len(actions_tuple),
                "fixed-power formula": fixed_ok,
                "direct pair orbits": direct_pair,
                "necklace formula": necklace,
                "pass": fixed_ok and direct_pair == necklace,
            }
        )
        passed &= rows[-1]["pass"]
    return {"name": "lamplighter affine lamp actions", "rows": tuple(rows), "checks": checks, "passed": passed}


def _metacyclic_case(m: int, n: int, u: int):
    root = np.exp(2j * np.pi / m)
    u_inverse = pow(u, -1, m)
    a = np.diag([root ** pow(u_inverse, j, m) for j in range(n)])
    b = np.zeros((n, n), dtype=complex)
    for column in range(n):
        b[(column + 1) % n, column] = 1
    matrices = tuple(np.linalg.matrix_power(a, x) @ np.linalg.matrix_power(b, y) for x in range(m) for y in range(n))

    def spectra():
        answer = []
        for x in range(m):
            for y in range(n):
                seen = set()
                eigenvalues = []
                for start in range(n):
                    if start in seen:
                        continue
                    cycle = []
                    current = start
                    while current not in seen:
                        seen.add(current)
                        cycle.append(current)
                        current = (current + y) % n
                    alpha = root ** (x * sum(pow(u_inverse, j, m) for j in cycle))
                    ell = len(cycle)
                    chosen = np.exp(1j * np.angle(alpha) / ell)
                    eigenvalues.extend(chosen * np.exp(2j * np.pi * k / ell) for k in range(ell))
                answer.append((1, tuple(eigenvalues)))
        return tuple(answer)

    weighted = spectra()
    relation_error = max(
        np.linalg.norm(np.linalg.matrix_power(b, n) - np.eye(n)),
        np.linalg.norm(b @ a @ b.conj().T - np.linalg.matrix_power(a, u)),
    )
    rows = _matrix_rows(
        f"M({m},{n},{u}) induced monomial",
        matrices,
        lambda tau: _spectral_dimension(weighted, tau),
    )
    numeric_error = abs(_matrix_molien(matrices) - _spectral_molien(weighted))
    return {
        "m": m,
        "n": n,
        "u": u,
        "rows": rows,
        "presentation error": float(relation_error),
        "Molien error": float(numeric_error),
        "passed": all(row["pass"] for row in rows) and relation_error < 2e-9 and numeric_error < 2e-9,
    }


def _metacyclic_suite():
    cases = (_metacyclic_case(5, 2, 4), _metacyclic_case(7, 3, 2))
    return {"name": "split metacyclic induced representations", "cases": cases, "passed": all(case["passed"] for case in cases)}


def _mat_mul(left: Matrix, right: Matrix, n: int, q: int) -> Matrix:
    return tuple(
        sum(left[n * i + k] * right[n * k + j] for k in range(n)) % q
        for i in range(n)
        for j in range(n)
    )


def _mat_vec(matrix: Matrix, vector: Vector, n: int, q: int) -> Vector:
    return tuple(sum(matrix[n * i + j] * vector[j] for j in range(n)) % q for i in range(n))


def _rank(matrix: Matrix, rows: int, columns: int, q: int) -> int:
    work = [list(matrix[row * columns : (row + 1) * columns]) for row in range(rows)]
    rank = 0
    for column in range(columns):
        pivot = next((row for row in range(rank, rows) if work[row][column] % q), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = pow(work[rank][column], -1, q)
        work[rank] = [(value * inverse) % q for value in work[rank]]
        for row in range(rows):
            if row != rank and work[row][column] % q:
                factor = work[row][column]
                work[row] = [(a - factor * b) % q for a, b in zip(work[row], work[rank], strict=True)]
        rank += 1
    return rank


def _general_linear(n: int, q: int) -> tuple[Matrix, ...]:
    return tuple(matrix for matrix in product(range(q), repeat=n * n) if _rank(matrix, n, n, q) == n)


def _linear_action(matrix: Matrix, points: tuple[Vector, ...], q: int) -> Permutation:
    locations = {point: index for index, point in enumerate(points)}
    n = len(points[0])
    return tuple(locations[_mat_vec(matrix, point, n, q)] for point in points)


def _gaussian_binomial(n: int, k: int, q: int) -> int:
    if k < 0 or k > n:
        return 0
    numerator = prod(q ** (n - j) - 1 for j in range(k))
    denominator = prod(q ** (k - j) - 1 for j in range(k))
    return numerator // denominator


def _orbit_count(actions: tuple[Permutation, ...], size: int, power: int) -> int:
    unseen = set(product(range(size), repeat=power))
    count = 0
    while unseen:
        seed = next(iter(unseen))
        orbit = {tuple(action[index] for index in seed) for action in actions}
        unseen.difference_update(orbit)
        count += 1
    return count


def _affine_suite():
    case_rows = []
    fixed_checks = 0
    passed = True
    for q, d, maximum_s in ((2, 2, 4), (2, 3, 3)):
        points = tuple(product(range(q), repeat=d))
        locations = {point: index for index, point in enumerate(points)}
        linear_group = _general_linear(d, q)
        linear_actions = tuple(_linear_action(matrix, points, q) for matrix in linear_group)
        affine_actions = []
        representatives = []
        for translation in points:
            for matrix, linear in zip(linear_group, linear_actions, strict=True):
                action = tuple(
                    locations[tuple((translation[j] + points[linear[index]][j]) % q for j in range(d))]
                    for index in range(len(points))
                )
                affine_actions.append(action)
                representatives.append((translation, matrix))
        affine_actions_tuple = tuple(affine_actions)
        fixed_ok = True
        identity = tuple(int(i == j) for i in range(d) for j in range(d))
        zero = (0,) * d
        for action, (translation, matrix) in zip(affine_actions_tuple, representatives, strict=True):
            order = lcm(*_cycle_lengths(action))
            matrix_power = identity
            accumulated = zero
            current_translation = translation
            for exponent in range(1, order + 1):
                accumulated = tuple((a + b) % q for a, b in zip(accumulated, current_translation, strict=True))
                current_translation = _mat_vec(matrix, current_translation, d, q)
                matrix_power = _mat_mul(matrix_power, matrix, d, q)
                difference_values = tuple(
                    tuple((point[j] - _mat_vec(matrix_power, point, d, q)[j]) % q for j in range(d))
                    for point in points
                )
                predicted = (
                    sum(value == zero for value in difference_values)
                    if accumulated in set(difference_values)
                    else 0
                )
                observed = _fixed(_perm_power(action, exponent))
                fixed_ok &= predicted == observed
                fixed_checks += 1
        moment_rows = []
        for s in range(2, maximum_s + 1):
            direct = _orbit_count(linear_actions, len(points), s - 1)
            formula = sum(_gaussian_binomial(s - 1, j, q) for j in range(min(d, s - 1) + 1))
            burnside = sum(_fixed(action) ** s for action in affine_actions_tuple) // len(affine_actions_tuple)
            moment_rows.append({"s": s, "direct H-orbits": direct, "affine Burnside": burnside, "Gaussian formula": formula, "pass": direct == burnside == formula})
        row = {
            "action": f"AGL({d},{q}) on F_{q}^{d}",
            "degree": len(points),
            "group order": len(affine_actions_tuple),
            "fixed-power formula": fixed_ok,
            "moments": tuple(moment_rows),
            "pass": fixed_ok and all(item["pass"] for item in moment_rows),
        }
        case_rows.append(row)
        passed &= row["pass"]
    return {"name": "general affine groups", "cases": tuple(case_rows), "fixed checks": fixed_checks, "passed": passed}


def _frobenius_suite():
    rows = []
    for prime, multipliers in ((5, (1, 2, 3, 4)), (7, (1, 6))):
        actions = tuple(
            tuple((translation + multiplier * point) % prime for point in range(prime))
            for translation in range(prime)
            for multiplier in multipliers
        )
        direct_types = Counter(_cycle_lengths(action) for action in actions)
        predicted = Counter({(1,) * prime: 1, (prime,): prime - 1})
        for multiplier in multipliers:
            if multiplier == 1:
                continue
            order = next(k for k in range(1, prime) if pow(multiplier, k, prime) == 1)
            predicted[(1, *([order] * ((prime - 1) // order)))] += prime
        coefficient_rows = []
        pair_formula = 1 + (prime - 1) // len(multipliers)
        for tau in TAUS:
            direct = _permutation_dimension(actions, tau)
            formula_numerator = sum(
                count * prod(_cycle_complete(lengths, degree) for degree in tau)
                for lengths, count in predicted.items()
            )
            formula = formula_numerator // len(actions)
            coefficient_rows.append({"tau": str(tau), "direct": direct, "cycle formula": formula, "pass": direct == formula})
        direct_pair = _pair_orbits(actions)
        rows.append(
            {
                "group": f"F_{prime} semidirect H_{len(multipliers)}",
                "degree": prime,
                "group order": len(actions),
                "cycle inventory": direct_types == predicted,
                "pair direct": direct_pair,
                "pair formula": pair_formula,
                "coefficients": tuple(coefficient_rows),
                "pass": direct_types == predicted and direct_pair == pair_formula and all(item["pass"] for item in coefficient_rows),
            }
        )
    return {"name": "finite Frobenius natural actions", "rows": tuple(rows), "passed": all(row["pass"] for row in rows)}


def _triangular_group(n: int, q: int, unitriangular: bool) -> tuple[Matrix, ...]:
    matrices = []
    upper_positions = tuple((i, j) for i in range(n) for j in range(i, n))
    for values in product(range(q), repeat=len(upper_positions)):
        matrix = [0] * (n * n)
        for (i, j), value in zip(upper_positions, values, strict=True):
            matrix[n * i + j] = value
        diagonal = tuple(matrix[n * i + i] for i in range(n))
        if unitriangular and diagonal != (1,) * n:
            continue
        if not unitriangular and any(value == 0 for value in diagonal):
            continue
        matrices.append(tuple(matrix))
    return tuple(matrices)


def _projective_points(n: int, q: int) -> tuple[Vector, ...]:
    representatives = set()
    for vector in product(range(q), repeat=n):
        if not any(vector):
            continue
        first = next(value for value in vector if value)
        inverse = pow(first, -1, q)
        representatives.add(tuple(value * inverse % q for value in vector))
    return tuple(sorted(representatives))


def _normalize_projective(vector: Vector, q: int) -> Vector:
    first = next(value for value in vector if value)
    inverse = pow(first, -1, q)
    return tuple(value * inverse % q for value in vector)


def _projective_action(matrix: Matrix, points: tuple[Vector, ...], q: int) -> Permutation:
    locations = {point: index for index, point in enumerate(points)}
    n = len(points[0])
    return tuple(
        locations[_normalize_projective(_mat_vec(matrix, point, n, q), q)]
        for point in points
    )


def _triangular_suite():
    rows = []
    fixed_checks = 0
    for name, n, q, unitriangular in (("UT_3(2)", 3, 2, True), ("B_2(3)", 2, 3, False)):
        group = _triangular_group(n, q, unitriangular)
        vectors = tuple(product(range(q), repeat=n))
        nonzero = tuple(vector for vector in vectors if any(vector))
        projective = _projective_points(n, q)
        actions_nonzero = tuple(_linear_action(matrix, nonzero, q) for matrix in group)
        actions_projective = tuple(_projective_action(matrix, projective, q) for matrix in group)
        formula_nonzero = n * (q - 1) if unitriangular else n
        formula_projective = n
        orbit_nonzero = _orbit_count(actions_nonzero, len(nonzero), 1)
        orbit_projective = _orbit_count(actions_projective, len(projective), 1)
        fixed_ok = True
        identity = tuple(int(i == j) for i in range(n) for j in range(n))
        for matrix, vector_action, projective_action in zip(group, actions_nonzero, actions_projective, strict=True):
            power = identity
            maximum = lcm(*_cycle_lengths(vector_action), *_cycle_lengths(projective_action))
            for exponent in range(1, maximum + 1):
                power = _mat_mul(power, matrix, n, q)
                kernel = sum(_mat_vec(power, vector, n, q) == vector for vector in vectors)
                predicted_nonzero = kernel - 1
                predicted_projective = sum(
                    (sum(_mat_vec(power, vector, n, q) == tuple(lam * x % q for x in vector) for vector in vectors) - 1) // (q - 1)
                    for lam in range(1, q)
                )
                fixed_ok &= _fixed(_perm_power(vector_action, exponent)) == predicted_nonzero
                fixed_ok &= _fixed(_perm_power(projective_action, exponent)) == predicted_projective
                fixed_checks += 2
        rows.append(
            {
                "group": name,
                "order": len(group),
                "nonzero-vector orbits": orbit_nonzero,
                "nonzero formula": formula_nonzero,
                "projective orbits": orbit_projective,
                "projective formula": formula_projective,
                "fixed-power formulas": fixed_ok,
                "pass": fixed_ok and orbit_nonzero == formula_nonzero and orbit_projective == formula_projective,
            }
        )

    # B_3(2) on complete flags of F_2^3; compare direct fixed flags with the
    # centralizer/class-intersection formula inside GL_3(2).
    n, q = 3, 2
    ambient = _general_linear(n, q)
    borel = _triangular_group(n, q, False)
    nonzero = tuple(vector for vector in product(range(q), repeat=n) if any(vector))
    flags = tuple((line, covector) for line in nonzero for covector in nonzero if sum(a * b for a, b in zip(line, covector, strict=True)) % q == 0)
    flag_locations = {flag: index for index, flag in enumerate(flags)}

    def inverse_matrix(matrix):
        identity = tuple(int(i == j) for i in range(n) for j in range(n))
        return next(candidate for candidate in ambient if _mat_mul(matrix, candidate, n, q) == identity)

    inverses = {matrix: inverse_matrix(matrix) for matrix in borel}
    actions = []
    for matrix in borel:
        inverse = inverses[matrix]
        inverse_transpose = tuple(inverse[n * j + i] for i in range(n) for j in range(n))
        actions.append(tuple(flag_locations[(_mat_vec(matrix, line, n, q), _mat_vec(inverse_transpose, covector, n, q))] for line, covector in flags))
    actions_tuple = tuple(actions)
    flag_fixed_ok = True
    identity = tuple(int(i == j) for i in range(n) for j in range(n))
    for matrix, action in zip(borel, actions_tuple, strict=True):
        power = identity
        for exponent in range(1, lcm(*_cycle_lengths(action)) + 1):
            power = _mat_mul(power, matrix, n, q)
            centralizer = sum(_mat_mul(candidate, power, n, q) == _mat_mul(power, candidate, n, q) for candidate in ambient)
            conjugates = {
                _mat_mul(_mat_mul(inverse_matrix(candidate), power, n, q), candidate, n, q)
                for candidate in ambient
            }
            intersection = sum(candidate in conjugates for candidate in borel)
            predicted = centralizer * intersection // len(borel)
            flag_fixed_ok &= predicted == _fixed(_perm_power(action, exponent))
            fixed_checks += 1
    flag_orbits = _orbit_count(actions_tuple, len(flags), 1)
    rows.append(
        {
            "group": "B_3(2) on GL_3(2)/B_3(2)",
            "order": len(borel),
            "degree": len(flags),
            "flag orbits": flag_orbits,
            "Weyl formula": factorial(3),
            "class-intersection fixed powers": flag_fixed_ok,
            "pass": flag_fixed_ok and flag_orbits == factorial(3),
        }
    )
    return {"name": "unitriangular, Borel, and flag actions", "rows": tuple(rows), "fixed checks": fixed_checks, "passed": all(row["pass"] for row in rows)}


def _matrix_closure(generators: tuple[np.ndarray, ...], digits: int = 9) -> tuple[np.ndarray, ...]:
    identity = np.eye(len(generators[0]), dtype=complex)

    def key(matrix):
        cleaned = matrix.copy()
        cleaned[abs(cleaned) < 10 ** (-(digits - 1))] = 0
        return tuple((round(value.real, digits), round(value.imag, digits)) for value in cleaned.flat)

    known = {key(identity): identity}
    frontier = [identity]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            candidate = current @ generator
            candidate_key = key(candidate)
            if candidate_key not in known:
                known[candidate_key] = candidate
                frontier.append(candidate)
                if len(known) > 10000:
                    raise ArithmeticError("matrix closure did not terminate")
    return tuple(known.values())


def _a5_rotation_group():
    phi = (1 + sqrt(5)) / 2
    axis = np.array([0.0, 1.0, phi])
    axis /= np.linalg.norm(axis)
    angle = 2 * np.pi / 5
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.cos(angle) * np.eye(3) + (1 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * cross
    cyclic = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)
    return _matrix_closure((rotation.astype(complex), cyclic))


def _psl27_group():
    eta = np.exp(2j * np.pi / 7)
    diagonal = np.diag([eta, eta**2, eta**4])
    a, b, c = eta**2 - eta**5, eta - eta**6, eta**4 - eta**3
    involution = 1j / sqrt(7) * np.array([[a, b, c], [b, c, a], [c, a, b]], dtype=complex)
    return _matrix_closure((diagonal, involution))


def _finite_su3_suite():
    omega = np.exp(2j * np.pi / 3)
    zeta5 = np.exp(2j * np.pi / 5)
    zeta7 = np.exp(2j * np.pi / 7)
    a5_spectra = (
        (1, (1, 1, 1)),
        (15, (1, -1, -1)),
        (20, (1, omega, omega**2)),
        (12, (1, zeta5, zeta5**4)),
        (12, (1, zeta5**2, zeta5**3)),
    )
    psl_spectra = (
        (1, (1, 1, 1)),
        (21, (1, -1, -1)),
        (56, (1, omega, omega**2)),
        (42, (1, 1j, -1j)),
        (24, (zeta7, zeta7**2, zeta7**4)),
        (24, (zeta7**3, zeta7**5, zeta7**6)),
    )
    groups = (("Sigma(60) = A5 rotation", _a5_rotation_group(), a5_spectra), ("Sigma(168) = PSL_2(7)", _psl27_group(), psl_spectra))
    cases = []
    for name, matrices, spectra in groups:
        rows = _matrix_rows(name, matrices, lambda tau, spectra=spectra: _spectral_dimension(spectra, tau))
        unitary_error = max(np.linalg.norm(matrix.conj().T @ matrix - np.eye(3)) for matrix in matrices)
        determinant_error = max(abs(np.linalg.det(matrix) - 1) for matrix in matrices)
        numeric_error = abs(_matrix_molien(matrices) - _spectral_molien(spectra))
        cases.append({"group": name, "order": len(matrices), "rows": rows, "unitary error": float(unitary_error), "determinant error": float(determinant_error), "Molien error": float(numeric_error), "passed": all(row["pass"] for row in rows) and unitary_error < 2e-7 and determinant_error < 2e-7 and numeric_error < 2e-8})

    # Delta(12) = (C_2 x C_2) semidirect C_3, an SU(3) type-C monomial group.
    shift = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)
    diagonals = tuple(np.diag([a, b, a * b]).astype(complex) for a in (1, -1) for b in (1, -1))
    matrices = tuple(diagonal @ np.linalg.matrix_power(shift, power) for diagonal in diagonals for power in range(3))
    spectra = tuple((1, tuple(np.linalg.eigvals(matrix))) for matrix in matrices)
    rows = _matrix_rows("Delta(12) monomial SU(3)", matrices, lambda tau: _spectral_dimension(spectra, tau))
    factor_errors = []
    for diagonal in diagonals:
        labels = np.diag(diagonal)
        for power in range(3):
            matrix = diagonal @ np.linalg.matrix_power(shift, power)
            for t in (0.021, -0.037):
                cycles = _cycle_lengths(tuple((index + power) % 3 for index in range(3)))
                if power == 0:
                    formula = prod(1 - labels[index] * t for index in range(3))
                else:
                    formula = 1 - prod(labels) * t ** cycles[0]
                factor_errors.append(abs(np.linalg.det(np.eye(3) - t * matrix) - formula))
    cases.append({"group": "Delta(12) determinant-one monomial", "order": 12, "rows": rows, "weighted-cycle determinant error": float(max(factor_errors)), "passed": all(row["pass"] for row in rows) and max(factor_errors) < 2e-9})
    return {"name": "finite SU(3) defining representations", "cases": tuple(cases), "passed": all(case["passed"] for case in cases)}


def _a5_su4_group():
    basis = np.zeros((5, 4), dtype=complex)
    basis[:4, :] = np.eye(4)
    basis[4, :] = -1
    orthonormal, _ = np.linalg.qr(basis)
    matrices = []
    for permutation in permutations(range(5)):
        inversions = sum(permutation[i] > permutation[j] for i in range(5) for j in range(i + 1, 5))
        if inversions % 2:
            continue
        p_matrix = np.zeros((5, 5), dtype=complex)
        p_matrix[permutation, np.arange(5)] = 1
        matrices.append(orthonormal.conj().T @ p_matrix @ orthonormal)
    return tuple(matrices)


def _q8():
    identity = np.eye(2, dtype=complex)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.diag([1, -1]).astype(complex)
    generators = (1j * sigma_x, 1j * sigma_y, 1j * sigma_z)
    return tuple(sign * matrix for matrix in (identity, *generators) for sign in (1, -1))


def _wreath_group(base: tuple[np.ndarray, ...], n: int) -> tuple[np.ndarray, ...]:
    block = len(base[0])
    matrices = []
    for permutation in permutations(range(n)):
        for labels in product(base, repeat=n):
            matrix = np.zeros((n * block, n * block), dtype=complex)
            for column, row in enumerate(permutation):
                matrix[row * block : (row + 1) * block, column * block : (column + 1) * block] = labels[column]
            matrices.append(matrix)
    return tuple(matrices)


def _wreath_formula_dimension(base, n: int, tau: tuple[int, ...]) -> int:
    zero = (0,) * len(tau)
    a_polynomials = {}
    for ell in range(1, n + 1):
        polynomial = {}
        for degree in product(*(range(entry + 1) for entry in tau)):
            if all(entry % ell == 0 for entry in degree):
                reduced = tuple(entry // ell for entry in degree)
                polynomial[degree] = Fraction(_matrix_dimension(base, reduced)) if reduced else Fraction(1)
        a_polynomials[ell] = polynomial
    coefficients = [{zero: Fraction(1)}]
    for block_count in range(1, n + 1):
        accumulated = {}
        for ell in range(1, block_count + 1):
            term = _convolve_polynomials(a_polynomials[ell], coefficients[block_count - ell], tau)
            for degree, value in term.items():
                accumulated[degree] = accumulated.get(degree, Fraction()) + value
        coefficients.append({degree: value / block_count for degree, value in accumulated.items()})
    answer = coefficients[n].get(tau, Fraction())
    assert answer.denominator == 1
    return answer.numerator


def _su4_sp4_so4_suite():
    cases = []
    a5 = _a5_su4_group()
    order_counts = Counter()
    identity = np.eye(4)
    for matrix in a5:
        power = identity
        for order in range(1, 7):
            power = power @ matrix
            if np.allclose(power, identity, atol=2e-8):
                order_counts[order] += 1
                break
    spectra_by_order = {
        1: (1, 1, 1, 1),
        2: (1, 1, -1, -1),
        3: (1, 1, np.exp(2j * np.pi / 3), np.exp(4j * np.pi / 3)),
        5: tuple(np.exp(2j * np.pi * k / 5) for k in range(1, 5)),
    }
    spectra = tuple((count, spectra_by_order[order]) for order, count in order_counts.items())
    rows = _matrix_rows("A5 standard SU(4) representation", a5, lambda tau: _spectral_dimension(spectra, tau))
    cases.append({"group": "A5 in SU(4)", "order": len(a5), "rows": rows, "class counts": dict(order_counts), "pass": all(row["pass"] for row in rows)})

    q8 = _q8()
    sp4 = _wreath_group(q8, 2)
    rows = _matrix_rows("Q8 wreath S2 in Sp(4)", sp4, lambda tau: _wreath_formula_dimension(q8, 2, tau))
    odd_vanishing = all(_matrix_dimension(q8, tau) == 0 for tau in ((1,), (3,), (2, 1)))
    cases.append({"group": "Q8 wreath S2 in Sp(4)", "order": len(sp4), "rows": rows, "central -I odd selection": odd_vanishing, "pass": all(row["pass"] for row in rows) and odd_vanishing})

    cover = tuple(np.kron(left, right) for left in q8 for right in q8)
    image_by_key = {}
    for matrix in cover:
        key = tuple((round(value.real, 10), round(value.imag, 10)) for value in matrix.flat)
        image_by_key.setdefault(key, matrix)
    image = tuple(image_by_key.values())
    trace_power_error = max(
        abs(np.trace(np.linalg.matrix_power(np.kron(left, right), exponent)) - np.trace(np.linalg.matrix_power(left, exponent)) * np.trace(np.linalg.matrix_power(right, exponent)))
        for left in q8
        for right in q8
        for exponent in range(1, 5)
    )
    rows = _matrix_rows("Q8 x Q8 spin-pair SO(4) image", image, lambda tau: _matrix_dimension(cover, tau))
    cover_molien_error = abs(_matrix_molien(image) - _matrix_molien(cover))
    cases.append({"group": "(Q8 x Q8)/{+- (I,I)} in SO(4)", "cover order": len(cover), "image order": len(image), "rows": rows, "trace-product error": float(trace_power_error), "cover/image Molien error": float(cover_molien_error), "pass": all(row["pass"] for row in rows) and len(image) == 32 and trace_power_error < 2e-9 and cover_molien_error < 2e-9})
    return {"name": "finite SU(4), Sp(4), and SO(4) defining representations", "cases": tuple(cases), "passed": all(case["pass"] for case in cases)}


def run_suite():
    sections = {
        "lamplighter monomial": _lamplighter_monomial_suite(),
        "lamplighter affine": _lamp_affine_suite(),
        "metacyclic": _metacyclic_suite(),
        "affine": _affine_suite(),
        "Frobenius": _frobenius_suite(),
        "triangular and parabolic": _triangular_suite(),
        "finite SU3": _finite_su3_suite(),
        "finite SU4 Sp4 SO4": _su4_sp4_so4_suite(),
    }
    exact_checks = 0
    coefficient_comparisons = 0
    for section in sections.values():
        exact_checks += int(section.get("checks", section.get("fixed checks", 0)))
        containers = section.get("cases", section.get("rows", ()))
        for case in containers:
            coefficient_comparisons += len(case.get("rows", case.get("coefficients", ())))
            coefficient_comparisons += len(case.get("moments", ()))
    passed = all(section["passed"] for section in sections.values())
    return {**sections, "fixed-power checks": exact_checks, "coefficient comparisons": coefficient_comparisons, "passed": passed}


if __name__ == "__main__":
    result = run_suite()
    print(f"{'PASS' if result['passed'] else 'FAIL'}: {result['fixed-power checks']:,} fixed-power checks and {result['coefficient comparisons']} coefficient comparisons")
    for key, section in result.items():
        if isinstance(section, dict) and "passed" in section:
            print(f"  {'PASS' if section['passed'] else 'FAIL'} {key}")
    raise SystemExit(0 if result["passed"] else 1)
