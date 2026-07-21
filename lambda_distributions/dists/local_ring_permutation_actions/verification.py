"""Exact finite checks for matrix groups over ``Z / p^a Z``.

The implementation deliberately constructs the matrix groups and their finite
sets from first principles.  It then compares direct configuration orbits with
the fixed-point/cycle formula.  The largest genuinely local example is
``GL_2(Z/9)``; a generator computation also checks the rank-two Grassmann
pair formula over ``Z/4`` without attempting to enumerate ``GL_4(Z/4)``.
"""

from __future__ import annotations

from collections import Counter, deque
from functools import lru_cache
from itertools import combinations, combinations_with_replacement, permutations, product
from math import comb, prod


Matrix = tuple[int, ...]
Vector = tuple[int, ...]
Permutation = tuple[int, ...]


def determinant(matrix: Matrix, n: int, modulus: int) -> int:
    """Leibniz determinant; the verification dimensions are at most four."""

    total = 0
    for sigma in permutations(range(n)):
        inversions = sum(
            sigma[i] > sigma[j] for i in range(n) for j in range(i + 1, n)
        )
        term = prod(matrix[i * n + sigma[i]] for i in range(n))
        total += (-1 if inversions % 2 else 1) * term
    return total % modulus


@lru_cache(maxsize=None)
def general_linear_group(n: int, p: int, a: int, determinant_one: bool = False):
    modulus = p**a
    answer = []
    for entries in product(range(modulus), repeat=n * n):
        det = determinant(entries, n, modulus)
        if det % p and (not determinant_one or det == 1):
            answer.append(entries)
    return tuple(answer)


def identity_matrix(n: int) -> Matrix:
    return tuple(int(i == j) for i in range(n) for j in range(n))


def matrix_multiply(left: Matrix, right: Matrix, n: int, modulus: int) -> Matrix:
    return tuple(
        sum(left[i * n + k] * right[k * n + j] for k in range(n)) % modulus
        for i in range(n)
        for j in range(n)
    )


def matrix_power(matrix: Matrix, exponent: int, n: int, modulus: int) -> Matrix:
    answer = identity_matrix(n)
    base = matrix
    while exponent:
        if exponent & 1:
            answer = matrix_multiply(answer, base, n, modulus)
        base = matrix_multiply(base, base, n, modulus)
        exponent //= 2
    return answer


def matrix_inverse(matrix: Matrix, n: int, modulus: int) -> Matrix:
    """Gauss-Jordan inversion over a local principal ideal ring."""

    left = [list(matrix[i * n : (i + 1) * n]) for i in range(n)]
    right = [[int(i == j) for j in range(n)] for i in range(n)]
    for column in range(n):
        pivot = next(
            row for row in range(column, n) if __import__("math").gcd(left[row][column], modulus) == 1
        )
        left[column], left[pivot] = left[pivot], left[column]
        right[column], right[pivot] = right[pivot], right[column]
        scale = pow(left[column][column], -1, modulus)
        left[column] = [(scale * x) % modulus for x in left[column]]
        right[column] = [(scale * x) % modulus for x in right[column]]
        for row in range(n):
            if row == column:
                continue
            factor = left[row][column]
            left[row] = [
                (left[row][j] - factor * left[column][j]) % modulus
                for j in range(n)
            ]
            right[row] = [
                (right[row][j] - factor * right[column][j]) % modulus
                for j in range(n)
            ]
    return tuple(x for row in right for x in row)


def matrix_vector(matrix: Matrix, vector: Vector, n: int, modulus: int) -> Vector:
    return tuple(
        sum(matrix[i * n + j] * vector[j] for j in range(n)) % modulus
        for i in range(n)
    )


def primitive_vectors(n: int, p: int, a: int):
    modulus = p**a
    return tuple(v for v in product(range(modulus), repeat=n) if any(x % p for x in v))


def projective_points(n: int, p: int, a: int):
    """Free rank-one summands, represented by their full underlying sets."""

    modulus = p**a
    lines = {
        frozenset(tuple((u * x) % modulus for x in vector) for u in range(modulus))
        for vector in primitive_vectors(n, p, a)
    }
    return tuple(sorted(lines, key=lambda line: sorted(line)))


def act_on_submodule(matrix: Matrix, submodule: frozenset[Vector], n: int, modulus: int):
    return frozenset(matrix_vector(matrix, vector, n, modulus) for vector in submodule)


def induced_permutations(group, points, action):
    positions = {point: i for i, point in enumerate(points)}
    return tuple(tuple(positions[action(g, point)] for point in points) for g in group)


def permutation_power(permutation: Permutation, exponent: int) -> Permutation:
    answer = tuple(range(len(permutation)))
    base = permutation
    while exponent:
        if exponent & 1:
            answer = tuple(base[answer[i]] for i in range(len(base)))
        base = tuple(base[base[i]] for i in range(len(base)))
        exponent //= 2
    return answer


def cycle_counts(permutation: Permutation):
    seen: set[int] = set()
    answer = Counter()
    for start in range(len(permutation)):
        if start in seen:
            continue
        current = start
        length = 0
        while current not in seen:
            seen.add(current)
            current = permutation[current]
            length += 1
        answer[length] += 1
    return answer


def mobius(value: int) -> int:
    primes = 0
    divisor = 2
    remaining = value
    while divisor * divisor <= remaining:
        if remaining % divisor == 0:
            remaining //= divisor
            primes += 1
            if remaining % divisor == 0:
                return 0
            while remaining % divisor == 0:
                remaining //= divisor
        divisor += 1
    if remaining > 1:
        primes += 1
    return -1 if primes % 2 else 1


def reconstructed_cycles(permutation: Permutation):
    maximum = max(cycle_counts(permutation))
    answer = Counter()
    for r in range(1, maximum + 1):
        numerator = 0
        for d in range(1, r + 1):
            if r % d == 0:
                powered = permutation_power(permutation, d)
                fixed = sum(powered[i] == i for i in range(len(powered)))
                numerator += mobius(r // d) * fixed
        if numerator:
            assert numerator % r == 0
            answer[r] = numerator // r
    return answer


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


def formula_coefficient(permutations_: tuple[Permutation, ...], tau: tuple[int, ...]):
    numerator = sum(
        prod(complete_from_cycles(cycle_counts(g), degree) for degree in tau)
        for g in permutations_
    )
    assert numerator % len(permutations_) == 0
    return numerator // len(permutations_)


@lru_cache(maxsize=None)
def block_multisets(number_of_points: int, degree: int):
    return tuple(combinations_with_replacement(range(number_of_points), degree))


def direct_orbit_count(permutations_: tuple[Permutation, ...], tau: tuple[int, ...]):
    configurations = set(
        product(*(block_multisets(len(permutations_[0]), degree) for degree in tau))
    )
    orbits = 0
    while configurations:
        representative = next(iter(configurations))
        orbit = {
            tuple(tuple(sorted(g[x] for x in block)) for block in representative)
            for g in permutations_
        }
        configurations.difference_update(orbit)
        orbits += 1
    return orbits


def kernel_size(matrix: Matrix, rows: int, columns: int, p: int, b: int) -> int:
    if b == 0:
        return 1
    modulus = p**b
    reduced = tuple(x % modulus for x in matrix)
    return sum(
        all(
            sum(reduced[i * columns + j] * vector[j] for j in range(columns))
            % modulus
            == 0
            for i in range(rows)
        )
        for vector in product(range(modulus), repeat=columns)
    )


def smith_exponents(matrix: Matrix, rows: int, columns: int, p: int, a: int):
    """Recover truncated Smith exponents from exact kernel cardinalities."""

    logs = [0]
    for b in range(1, a + 1):
        size = kernel_size(matrix, rows, columns, p, b)
        exponent = 0
        while size > 1:
            assert size % p == 0
            size //= p
            exponent += 1
        logs.append(exponent)
    at_least = [logs[b] - logs[b - 1] for b in range(1, a + 1)]
    exponents = [0] * (columns - at_least[0])
    for value in range(1, a):
        exponents.extend([value] * (at_least[value - 1] - at_least[value]))
    exponents.extend([a] * at_least[-1])
    assert len(exponents) == columns
    return tuple(exponents)


def smith_kernel_size(exponents: tuple[int, ...], p: int, b: int) -> int:
    return p ** sum(min(value, b) for value in exponents)


def shifted_power(matrix: Matrix, exponent: int, scalar: int, n: int, modulus: int):
    powered = list(matrix_power(matrix, exponent, n, modulus))
    for i in range(n):
        powered[i * n + i] = (powered[i * n + i] - scalar) % modulus
    return tuple(powered)


def primitive_fixed_formula(matrix: Matrix, exponent: int, n: int, p: int, a: int):
    modulus = p**a
    difference = shifted_power(matrix, exponent, 1, n, modulus)
    smith = smith_exponents(difference, n, n, p, a)
    return smith_kernel_size(smith, p, a) - smith_kernel_size(smith, p, a - 1)


def projective_fixed_formula(matrix: Matrix, exponent: int, n: int, p: int, a: int):
    modulus = p**a
    units = [u for u in range(modulus) if u % p]
    numerator = 0
    for unit in units:
        difference = shifted_power(matrix, exponent, unit, n, modulus)
        smith = smith_exponents(difference, n, n, p, a)
        numerator += smith_kernel_size(smith, p, a) - smith_kernel_size(smith, p, a - 1)
    assert numerator % len(units) == 0
    return numerator // len(units)


def representative_pairs(group, permutations_, limit: int | None = None):
    if limit is None or len(group) <= limit:
        return tuple(zip(group, permutations_))
    indices = sorted({round(i * (len(group) - 1) / (limit - 1)) for i in range(limit)})
    return tuple((group[i], permutations_[i]) for i in indices)


def all_fixed_formula_checks(group, permutations_, formula, limit: int | None = None):
    for matrix, permutation in representative_pairs(group, permutations_, limit):
        for exponent in range(1, max(cycle_counts(permutation)) + 1):
            powered = permutation_power(permutation, exponent)
            direct = sum(powered[i] == i for i in range(len(powered)))
            if direct != formula(matrix, exponent):
                return False
    return True


def homogeneous_space_fixed_check_gl2(
    group, projective_permutations, p: int, a: int, limit: int | None = None
):
    """Check |C(g)| |g^G cap P| / |P| against fixed projective lines."""

    modulus = p**a
    parabolic = tuple(g for g in group if g[2] == 0)
    parabolic_set = set(parabolic)
    inverses = {g: matrix_inverse(g, 2, modulus) for g in group}
    for g, action in representative_pairs(group, projective_permutations, limit):
        centralizer = sum(
            matrix_multiply(g, x, 2, modulus) == matrix_multiply(x, g, 2, modulus)
            for x in group
        )
        conjugates_in_p = {
            matrix_multiply(
                matrix_multiply(inverses[x], g, 2, modulus), x, 2, modulus
            )
            for x in group
        } & parabolic_set
        numerator = centralizer * len(conjugates_in_p)
        assert numerator % len(parabolic) == 0
        if numerator // len(parabolic) != sum(action[i] == i for i in range(len(action))):
            return False
    return True


def span(vectors: tuple[Vector, ...], p: int):
    if not vectors:
        return frozenset({(0,) * 0})
    n = len(vectors[0])
    return frozenset(
        tuple(sum(c * vectors[j][i] for j, c in enumerate(coeffs)) % p for i in range(n))
        for coeffs in product(range(p), repeat=len(vectors))
    )


def field_subspaces(n: int, k: int, p: int):
    vectors = tuple(product(range(p), repeat=n))
    nonzero = tuple(v for v in vectors if any(v))
    spaces = set()
    for basis in combinations(nonzero, k):
        space = span(basis, p)
        if len(space) == p**k:
            spaces.add(space)
    return tuple(sorted(spaces, key=lambda space: sorted(space)))


def basis_of_space(space: frozenset[Vector], dimension: int, p: int):
    basis: list[Vector] = []
    current = frozenset({(0,) * len(next(iter(space)))})
    for vector in sorted(space):
        candidate = span(tuple(basis + [vector]), p)
        if len(candidate) > len(current):
            basis.append(vector)
            current = candidate
        if len(basis) == dimension:
            return tuple(basis)
    raise AssertionError("space has the wrong dimension")


def free_summands(n: int, k: int, p: int, a: int):
    """All free rank-k summands, using unique graph lifts of mod-p subspaces."""

    modulus = p**a
    answer: set[frozenset[Vector]] = set()
    standard = [tuple(int(i == j) for i in range(n)) for j in range(n)]
    for residue_space in field_subspaces(n, k, p):
        base = list(basis_of_space(residue_space, k, p))
        full = list(base)
        for vector in standard:
            if len(span(tuple(full + [vector]), p)) > len(span(tuple(full), p)):
                full.append(vector)
        complement = full[k:]
        for graph_entries in product(range(p ** (a - 1)), repeat=(n - k) * k):
            lifted = []
            for column in range(k):
                lifted.append(
                    tuple(
                        (
                            base[column][row]
                            + p
                            * sum(
                                graph_entries[i * k + column] * complement[i][row]
                                for i in range(n - k)
                            )
                        )
                        % modulus
                        for row in range(n)
                    )
                )
            submodule = frozenset(
                tuple(
                    sum(coeffs[j] * lifted[j][i] for j in range(k)) % modulus
                    for i in range(n)
                )
                for coeffs in product(range(modulus), repeat=k)
            )
            answer.add(submodule)
    return tuple(sorted(answer, key=lambda space: sorted(space)))


def parabolic_generators(n: int, k: int, modulus: int):
    generators = []

    def elementary(i: int, j: int, value: int):
        matrix = list(identity_matrix(n))
        matrix[i * n + j] = value % modulus
        return tuple(matrix)

    for start, stop in ((0, k), (k, n)):
        for i in range(start, stop):
            for j in range(start, stop):
                if i != j:
                    generators.append(elementary(i, j, 1))
        if stop - start >= 2:
            swap = list(identity_matrix(n))
            i, j = start, start + 1
            swap[i * n + i] = swap[j * n + j] = 0
            swap[i * n + j] = swap[j * n + i] = 1
            generators.append(tuple(swap))
        diagonal = list(identity_matrix(n))
        diagonal[start * n + start] = modulus - 1
        generators.append(tuple(diagonal))
    for i in range(k):
        for j in range(k, n):
            generators.append(elementary(i, j, 1))
    return tuple(dict.fromkeys(generators))


def summand_projection_exponents(submodule, n: int, k: int, p: int, a: int):
    residues = frozenset(tuple(x % p for x in vector) for vector in submodule)
    residue_basis = basis_of_space(residues, k, p)
    lifts = []
    for residue in residue_basis:
        lifts.append(next(vector for vector in sorted(submodule) if tuple(x % p for x in vector) == residue))
    projection = tuple(lifts[column][k + row] for row in range(n - k) for column in range(k))
    return smith_exponents(projection, n - k, k, p, a)


def grassmann_pair_check():
    n, k, p, a = 4, 2, 2, 2
    modulus = p**a
    summands = free_summands(n, k, p, a)
    expected_size = p ** ((a - 1) * k * (n - k)) * 35
    assert len(summands) == expected_size == 560
    remaining = set(summands)
    generators = parabolic_generators(n, k, modulus)
    orbit_invariants = []
    while remaining:
        representative = next(iter(remaining))
        orbit = {representative}
        queue = deque([representative])
        while queue:
            current = queue.popleft()
            for generator in generators:
                image = act_on_submodule(generator, current, n, modulus)
                if image not in orbit:
                    orbit.add(image)
                    queue.append(image)
        remaining.difference_update(orbit)
        invariants = {summand_projection_exponents(w, n, k, p, a) for w in orbit}
        assert len(invariants) == 1
        orbit_invariants.append(next(iter(invariants)))
    predicted = comb(a + k, k)
    return {
        "ambient": "(Z/4)^4",
        "rank": k,
        "summands": len(summands),
        "direct pair orbits": len(orbit_invariants),
        "Smith types": tuple(sorted(orbit_invariants)),
        "proposed formula": predicted,
        "passed": len(orbit_invariants) == predicted,
    }


def conjugation_action(group, p: int, c: int):
    modulus = p**c
    points = tuple(product(range(modulus), repeat=4))

    def action(g, x):
        reduced = tuple(value % modulus for value in g)
        inverse = matrix_inverse(reduced, 2, modulus)
        return matrix_multiply(matrix_multiply(reduced, x, 2, modulus), inverse, 2, modulus)

    return points, induced_permutations(group, points, action)


def adjoint_fixed_formula(matrix: Matrix, exponent: int, p: int, c: int):
    modulus = p**c
    g = matrix_power(tuple(x % modulus for x in matrix), exponent, 2, modulus)
    inverse = matrix_inverse(g, 2, modulus)
    return sum(
        matrix_multiply(matrix_multiply(g, x, 2, modulus), inverse, 2, modulus) == x
        for x in product(range(modulus), repeat=4)
    )


def action_rows(label, permutations_, expectations):
    rows = []
    for tau, expected in expectations.items():
        direct = direct_orbit_count(permutations_, tau)
        formula = formula_coefficient(permutations_, tau)
        rows.append(
            {
                "action": label,
                "tau": str(tau),
                "direct orbits": direct,
                "cycle formula": formula,
                "proposed": expected,
                "passed": direct == formula == expected,
            }
        )
    return rows


@lru_cache(maxsize=None)
def local_linear_suite(p: int, a: int):
    n = 2
    modulus = p**a
    group = general_linear_group(n, p, a)
    primitive = primitive_vectors(n, p, a)
    projective = projective_points(n, p, a)
    primitive_perms = induced_permutations(
        group, primitive, lambda g, v: matrix_vector(g, v, n, modulus)
    )
    projective_perms = induced_permutations(
        group, projective, lambda g, line: act_on_submodule(g, line, n, modulus)
    )
    expected_group_order = p ** (n * n * (a - 1)) * (p**n - 1) * (p**n - p)
    expected_primitive_size = p ** (a * n) - p ** ((a - 1) * n)
    expected_projective_size = p ** ((a - 1) * (n - 1)) * (p**n - 1) // (p - 1)
    assert len(group) == expected_group_order
    assert len(primitive) == expected_primitive_size
    assert len(projective) == expected_projective_size
    rows = action_rows(
        f"GL_2(Z/{modulus}) primitive vectors",
        primitive_perms,
        {(1, 1): p**a},
    )
    rows += action_rows(
        f"GL_2(Z/{modulus}) projective points",
        projective_perms,
        {(2,): a + 1, (1, 1): a + 1},
    )
    verification_limit = None if len(group) <= 100 else 32
    primitive_fixed = all_fixed_formula_checks(
        group,
        primitive_perms,
        lambda g, d: primitive_fixed_formula(g, d, n, p, a),
        verification_limit,
    )
    projective_fixed = all_fixed_formula_checks(
        group,
        projective_perms,
        lambda g, d: projective_fixed_formula(g, d, n, p, a),
        verification_limit,
    )
    homogeneous = homogeneous_space_fixed_check_gl2(
        group, projective_perms, p, a, verification_limit
    )
    cycles = all(
        reconstructed_cycles(g) == cycle_counts(g)
        for _, g in representative_pairs(group, projective_perms, verification_limit)
    )
    return {
        "p": p,
        "a": a,
        "group order": len(group),
        "primitive size": len(primitive),
        "projective size": len(projective),
        "matrices checked pointwise": len(
            representative_pairs(group, projective_perms, verification_limit)
        ),
        "rows": tuple(rows),
        "primitive Smith fixed points": primitive_fixed,
        "projective Smith fixed points": projective_fixed,
        "homogeneous-space fixed points": homogeneous,
        "Mobius cycle reconstruction": cycles,
        "passed": all(row["passed"] for row in rows)
        and primitive_fixed
        and projective_fixed
        and homogeneous
        and cycles,
    }


@lru_cache(maxsize=None)
def symplectic_rank_one_suite():
    p, a, n = 2, 2, 2
    modulus = p**a
    group = general_linear_group(n, p, a, determinant_one=True)
    points = projective_points(n, p, a)
    actions = induced_permutations(
        group, points, lambda g, line: act_on_submodule(g, line, n, modulus)
    )
    rows = action_rows("Sp_2(Z/4) projective isotropic lines", actions, {(1, 1): 3})
    fixed = all_fixed_formula_checks(
        group, actions, lambda g, d: projective_fixed_formula(g, d, n, p, a)
    )
    return {
        "group order": len(group),
        "set size": len(points),
        "rows": tuple(rows),
        "Smith fixed points": fixed,
        "passed": fixed and all(row["passed"] for row in rows),
    }


@lru_cache(maxsize=None)
def adjoint_suite():
    p, a, c = 2, 2, 1
    group = general_linear_group(2, p, a)
    points, actions = conjugation_action(group, p, c)
    expectations = {
        tau: direct_orbit_count(actions, tau) for tau in ((1,), (2,), (1, 1))
    }
    rows = action_rows("GL_2(Z/4) on M_2(F_2) by conjugation", actions, expectations)
    fixed = all_fixed_formula_checks(
        group, actions, lambda g, d: adjoint_fixed_formula(g, d, p, c)
    )
    return {
        "group order": len(group),
        "set size": len(points),
        "rows": tuple(rows),
        "adjoint-kernel fixed points": fixed,
        "passed": fixed and all(row["passed"] for row in rows),
    }


@lru_cache(maxsize=None)
def run_suite():
    try:
        from for_this_guy.finite_lie_type_permutation_actions.verification_core import (
            polar_suite,
        )
    except ModuleNotFoundError:  # Support direct execution by file path.
        from proved_matrix_groups.packages.finite_lie_type_permutation_actions.verification_core import (
            polar_suite,
        )

    local_cases = (local_linear_suite(2, 2), local_linear_suite(3, 2))
    grassmann = grassmann_pair_check()
    symplectic = symplectic_rank_one_suite()
    adjoint = adjoint_suite()
    formed_field = polar_suite(include_unitary=False)
    rows = tuple(
        row
        for section in (*local_cases, symplectic, adjoint)
        for row in section["rows"]
    )
    passed = (
        all(section["passed"] for section in local_cases)
        and grassmann["passed"]
        and symplectic["passed"]
        and adjoint["passed"]
        and formed_field["passed"]
        and formed_field["cycle reconstruction"]
    )
    return {
        "passed": passed,
        "rows": rows,
        "local cases": local_cases,
        "grassmann": grassmann,
        "symplectic local": symplectic,
        "adjoint": adjoint,
        "formed field": formed_field,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    workspace_root = Path(__file__).resolve().parents[3]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    result = run_suite()
    print(f"{'PASS' if result['passed'] else 'FAIL'}: {len(result['rows'])} coefficient checks")
    for row in result["rows"]:
        print(row)
    print(result["grassmann"])
