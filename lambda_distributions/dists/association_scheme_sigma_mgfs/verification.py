"""Exact matrix/permutation checks for association-scheme sigma-MGFs.

Every represented matrix is a zero-one permutation matrix, stored by the row
of its nonzero entry in each column.  The checks compare four routes whenever
the size permits: direct matrix traces, cycle factors, literal configuration
orbits, and a family-specific fixed-point formula.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import combinations, permutations, product
from math import prod

from lambda_distributions.proofs.association_scheme_permutation_representations.finite_field import (
    general_linear_group,
    mat_mul,
    mat_vec,
    move_subspace,
    subspaces,
)
from lambda_distributions.proofs.association_scheme_permutation_representations.grassmann.verification import (
    run_suite as run_grassmann,
)
from lambda_distributions.proofs.association_scheme_permutation_representations.hamming.verification import (
    hamming_action,
    run_suite as run_hamming,
)
from lambda_distributions.proofs.association_scheme_permutation_representations.johnson.verification import (
    run_suite as run_johnson,
)
from lambda_distributions.proofs.association_scheme_permutation_representations.polar.verification import (
    run_suite as run_polar,
)
from lambda_distributions.proofs.association_scheme_permutation_representations.shared import (
    complete_from_cycles,
    cycle_counts,
    cycle_formula_coefficient,
    direct_orbit_count,
    fixed_formula_reconstructs,
    permutation_power,
)


Permutation = tuple[int, ...]
Vector = tuple[int, ...]
Matrix = tuple[int, ...]
TAUS = ((1,), (2,), (1, 1))


def _identity(size: int) -> Permutation:
    return tuple(range(size))


def _compose(left: Permutation, right: Permutation) -> Permutation:
    return tuple(left[right[index]] for index in range(len(right)))


def _inverse_permutation(permutation: Permutation) -> Permutation:
    answer = [0] * len(permutation)
    for source, target in enumerate(permutation):
        answer[target] = source
    return tuple(answer)


def _pairwise_add(left: Vector, right: Vector, q: int) -> Vector:
    return tuple((x + y) % q for x, y in zip(left, right, strict=True))


def _induced_linear_permutation(points, transform) -> Permutation:
    locations = {point: index for index, point in enumerate(points)}
    return tuple(locations[transform(point)] for point in points)


def _trace_character(permutation: Permutation, degree: int) -> int:
    """Trace of Sym^degree(P), obtained from traces of actual matrix powers."""

    fixed = [len(permutation)] + [
        sum(image == index for index, image in enumerate(permutation_power(permutation, r)))
        for r in range(1, degree + 1)
    ]
    complete = [1]
    for d in range(1, degree + 1):
        numerator = sum(fixed[r] * complete[d - r] for r in range(1, d + 1))
        assert numerator % d == 0
        complete.append(numerator // d)
    return complete[degree]


def _matrix_trace_average(actions: tuple[Permutation, ...], tau: tuple[int, ...]) -> int:
    numerator = sum(prod(_trace_character(action, degree) for degree in tau) for action in actions)
    assert numerator % len(actions) == 0
    return numerator // len(actions)


def _coefficient_rows(name: str, actions, pair_prediction: int | None = None):
    actions = tuple(dict.fromkeys(actions))
    rows = []
    for tau in TAUS:
        matrix = _matrix_trace_average(actions, tau)
        cycles = cycle_formula_coefficient(actions, tau)
        direct = direct_orbit_count(actions, tau)
        prediction = pair_prediction if tau == (1, 1) else None
        rows.append(
            {
                "action": name,
                "|G image|": len(actions),
                "dimension": len(actions[0]),
                "tau": str(tau),
                "matrix trace": matrix,
                "cycle formula": cycles,
                "direct orbits": direct,
                "prediction": prediction,
                "pass": matrix == cycles == direct
                and (prediction is None or matrix == prediction),
            }
        )
    return tuple(rows)


def _affine_structural_fixed(
    points: tuple[Vector, ...],
    add,
    translation: Vector,
    linear: Permutation,
    exponent: int,
) -> int:
    """Formula F_r=|ker(I-L^r)| or zero according to the image test."""

    locations = {point: index for index, point in enumerate(points)}
    zero = points[0]
    accumulated = zero
    current = translation
    for _ in range(exponent):
        accumulated = add(accumulated, current)
        current = points[linear[locations[current]]]
    powered = permutation_power(linear, exponent)
    differences = tuple(
        add(point, _additive_inverse(points[powered[index]], add, points))
        for index, point in enumerate(points)
    )
    kernel_size = sum(value == zero for value in differences)
    return kernel_size if accumulated in set(differences) else 0


def _additive_inverse(value: Vector, add, points: tuple[Vector, ...]) -> Vector:
    zero = points[0]
    return next(candidate for candidate in points if add(value, candidate) == zero)


def _affine_action(points, add, translation, linear: Permutation) -> Permutation:
    locations = {point: index for index, point in enumerate(points)}
    return tuple(locations[add(translation, points[linear[index]])] for index in range(len(points)))


def _verify_affine(name, points, add, linear_actions, pair_prediction: int):
    representatives: dict[Permutation, tuple[Vector, Permutation]] = {}
    for translation in points:
        for linear in dict.fromkeys(linear_actions):
            action = _affine_action(points, add, translation, linear)
            representatives.setdefault(action, (translation, linear))
    actions = tuple(representatives)
    fixed_ok = True
    for action, (translation, linear) in representatives.items():
        maximum = max(cycle_counts(action), default=1)
        predicted = lambda exponent, t=translation, ell=linear: _affine_structural_fixed(
            points, add, t, ell, exponent
        )
        fixed_ok &= fixed_formula_reconstructs(action, predicted)
        fixed_ok &= all(
            predicted(exponent)
            == sum(
                image == index
                for index, image in enumerate(permutation_power(action, exponent))
            )
            for exponent in range(1, maximum + 1)
        )
    rows = _coefficient_rows(name, actions, pair_prediction)
    return {
        "name": name,
        "|A|": len(points),
        "|H image|": len(tuple(dict.fromkeys(linear_actions))),
        "|A semidirect H image|": len(actions),
        "fixed-point formula": fixed_ok,
        "rows": rows,
        "passed": fixed_ok and all(row["pass"] for row in rows),
    }


def _matrix_inverse_from_group(matrix: Matrix, group, n: int, q: int) -> Matrix:
    identity = tuple(int(i == j) for i in range(n) for j in range(n))
    return next(candidate for candidate in group if mat_mul(matrix, candidate, n, q) == identity)


def _bilinear_case():
    q = 2
    group = general_linear_group(2, q)
    inverses = {matrix: _matrix_inverse_from_group(matrix, group, 2, q) for matrix in group}
    points = tuple(product(range(q), repeat=4))

    def transform(left, right, value):
        first = mat_mul(left, value, 2, q)
        return mat_mul(first, inverses[right], 2, q)

    linear = tuple(
        _induced_linear_permutation(points, lambda value, p=p, r=r: transform(p, r, value))
        for p in group
        for r in group
    )
    return _verify_affine(
        "Bilinear forms M_2(F_2)", points, lambda x, y: _pairwise_add(x, y, 2), linear, 3
    )


def _alternating_case():
    q = 2
    group = general_linear_group(3, q)
    points = tuple(product(range(q), repeat=3))

    def expand(value):
        a, b, c = value
        return (0, a, b, a, 0, c, b, c, 0)

    def compress(matrix):
        return matrix[1], matrix[2], matrix[5]

    def transpose(matrix):
        return tuple(matrix[3 * j + i] for i in range(3) for j in range(3))

    linear = tuple(
        _induced_linear_permutation(
            points,
            lambda value, p=p: compress(mat_mul(mat_mul(p, expand(value), 3, q), transpose(p), 3, q)),
        )
        for p in group
    )
    return _verify_affine(
        "Alternating 3x3 forms over F_2", points, lambda x, y: _pairwise_add(x, y, 2), linear, 2
    )


def _quadratic_case():
    q = 3
    group = general_linear_group(2, q)
    inverses = {matrix: _matrix_inverse_from_group(matrix, group, 2, q) for matrix in group}
    points = tuple(product(range(q), repeat=3))

    def transform(matrix, value):
        u, v, w, z = inverses[matrix]
        a, b, c = value
        return (
            (a * u * u + b * u * w + c * w * w) % q,
            (2 * a * u * v + b * (u * z + v * w) + 2 * c * w * z) % q,
            (a * v * v + b * v * z + c * z * z) % q,
        )

    linear = tuple(
        _induced_linear_permutation(points, lambda value, matrix=matrix: transform(matrix, value))
        for matrix in group
    )
    return _verify_affine(
        "Binary quadratic forms over F_3", points, lambda x, y: _pairwise_add(x, y, 3), linear, 5
    )


def _gf4_add(left: int, right: int) -> int:
    return left ^ right


def _gf4_mul(left: int, right: int) -> int:
    answer = 0
    a, b = left, right
    while b:
        if b & 1:
            answer ^= a
        b >>= 1
        a <<= 1
        if a & 4:
            a ^= 0b111
    return answer


def _gf4_pow(value: int, exponent: int) -> int:
    answer = 1
    while exponent:
        if exponent & 1:
            answer = _gf4_mul(answer, value)
        value = _gf4_mul(value, value)
        exponent //= 2
    return answer


def _gf4_matrix_mul(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        _gf4_add(_gf4_mul(left[2 * i], right[j]), _gf4_mul(left[2 * i + 1], right[2 + j]))
        for i in range(2)
        for j in range(2)
    )


def _hermitian_case():
    matrices = tuple(
        matrix
        for matrix in product(range(4), repeat=4)
        if _gf4_add(_gf4_mul(matrix[0], matrix[3]), _gf4_mul(matrix[1], matrix[2])) != 0
    )
    points = tuple((a, z, d) for a in range(2) for z in range(4) for d in range(2))

    def expand(value):
        a, z, d = value
        return a, z, _gf4_pow(z, 2), d

    def star(matrix):
        return tuple(_gf4_pow(matrix[2 * j + i], 2) for i in range(2) for j in range(2))

    def transform(matrix, value):
        result = _gf4_matrix_mul(_gf4_matrix_mul(matrix, expand(value)), star(matrix))
        assert result[0] in (0, 1) and result[3] in (0, 1)
        assert result[2] == _gf4_pow(result[1], 2)
        return result[0], result[1], result[3]

    linear = tuple(
        _induced_linear_permutation(points, lambda value, matrix=matrix: transform(matrix, value))
        for matrix in matrices
    )
    add = lambda x, y: (x[0] ^ y[0], x[1] ^ y[1], x[2] ^ y[2])
    return _verify_affine("Hermitian 2x2 forms over F_4/F_2", points, add, linear, 3)


def _orthogonal_plus_case():
    points = tuple(product(range(2), repeat=4))
    group = general_linear_group(4, 2)

    def quadratic(value):
        return (value[0] * value[1] + value[2] * value[3]) % 2

    orthogonal = tuple(
        matrix
        for matrix in group
        if all(quadratic(mat_vec(matrix, value, 2)) == quadratic(value) for value in points)
    )
    linear = tuple(
        _induced_linear_permutation(points, lambda value, matrix=matrix: mat_vec(matrix, value, 2))
        for matrix in orthogonal
    )
    return _verify_affine(
        "Affine polar O_4^+(2) on F_2^4", points, lambda x, y: _pairwise_add(x, y, 2), linear, 3
    )


def _coordinate_permutation(value: Vector, sigma: Permutation) -> Vector:
    moved = [0] * len(value)
    for source, target in enumerate(sigma):
        moved[target] = value[source]
    return tuple(moved)


def _folded_case(n: int = 4):
    one = (1,) * n

    def canonical(value):
        complement = tuple(x ^ 1 for x in value)
        return min(value, complement)

    points = tuple(sorted({canonical(value) for value in product(range(2), repeat=n)}))
    add = lambda x, y: canonical(tuple(a ^ b for a, b in zip(x, y, strict=True)))
    linear = tuple(
        _induced_linear_permutation(points, lambda value, sigma=sigma: canonical(_coordinate_permutation(value, sigma)))
        for sigma in permutations(range(n))
    )
    return _verify_affine(
        f"Folded {n}-cube", points, add, linear, n // 2 + 1
    )


def _halved_case(n: int = 4):
    points = tuple(value for value in product(range(2), repeat=n) if sum(value) % 2 == 0)
    add = lambda x, y: tuple(a ^ b for a, b in zip(x, y, strict=True))
    linear = tuple(
        _induced_linear_permutation(points, lambda value, sigma=sigma: _coordinate_permutation(value, sigma))
        for sigma in permutations(range(n))
    )
    return _verify_affine(
        f"Halved {n}-cube", points, add, linear, n // 2 + 1
    )


def _matrix_inverse_binary(matrix: Matrix, n: int, group) -> Matrix:
    return _matrix_inverse_from_group(matrix, group, n, 2)


def _building_case():
    group = general_linear_group(3, 2)
    point_spaces = subspaces(3, 1, 2)
    line_spaces = subspaces(3, 2, 2)

    def contained(point, line):
        return all(vector in _span(line, 2) for vector in point)

    flags = tuple((point, line) for point in point_spaces for line in line_spaces if contained(point, line))
    locations = {flag: index for index, flag in enumerate(flags)}
    actions = tuple(
        tuple(
            locations[(move_subspace(matrix, point, 2), move_subspace(matrix, line, 2))]
            for point, line in flags
        )
        for matrix in group
    )
    standard = flags[0]
    borel = tuple(
        matrix
        for matrix in group
        if (move_subspace(matrix, standard[0], 2), move_subspace(matrix, standard[1], 2)) == standard
    )

    @lru_cache(maxsize=None)
    def coset_formula(element):
        centralizer = sum(
            mat_mul(candidate, element, 3, 2) == mat_mul(element, candidate, 3, 2)
            for candidate in group
        )
        conjugates = set()
        for candidate in group:
            inverse = _matrix_inverse_binary(candidate, 3, group)
            conjugates.add(mat_mul(mat_mul(inverse, element, 3, 2), candidate, 3, 2))
        intersection = sum(value in set(borel) for value in conjugates)
        numerator = centralizer * intersection
        assert numerator % len(borel) == 0
        return numerator // len(borel)

    representative = dict(zip(actions, group, strict=True))
    fixed_ok = True
    for action, matrix in representative.items():
        order_bound = max(cycle_counts(action), default=1)
        power = tuple(int(i == j) for i in range(3) for j in range(3))
        for exponent in range(1, order_bound + 1):
            power = mat_mul(matrix, power, 3, 2)
            predicted = coset_formula(power)
            actual = sum(
                image == index
                for index, image in enumerate(permutation_power(action, exponent))
            )
            fixed_ok &= predicted == actual
    rows = _coefficient_rows("GL_3(2) on complete flags", actions, 6)
    return {
        "name": "Type A_2 building chambers",
        "|G|": len(group),
        "|B|": len(borel),
        "dimension": len(flags),
        "coset fixed-point formula": fixed_ok,
        "rows": rows,
        "passed": fixed_ok and all(row["pass"] for row in rows),
    }


def _span(rows, q: int):
    return {
        tuple(sum(coefficients[i] * rows[i][j] for i in range(len(rows))) % q for j in range(len(rows[0])))
        for coefficients in product(range(q), repeat=len(rows))
    }


def _shrikhande_actions():
    vertices = tuple(product(range(4), repeat=2))
    connection = {(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)}

    def determinant(matrix):
        return (matrix[0] * matrix[3] - matrix[1] * matrix[2]) % 4

    stabilizer = tuple(
        matrix
        for matrix in product(range(4), repeat=4)
        if determinant(matrix) % 2
        and {
            ((matrix[0] * x + matrix[1] * y) % 4, (matrix[2] * x + matrix[3] * y) % 4)
            for x, y in connection
        }
        == connection
    )
    locations = {point: index for index, point in enumerate(vertices)}
    actions = tuple(
        tuple(
            locations[((b[0] + matrix[0] * x + matrix[1] * y) % 4,
                       (b[1] + matrix[2] * x + matrix[3] * y) % 4)]
            for x, y in vertices
        )
        for b in vertices
        for matrix in stabilizer
    )
    return tuple(dict.fromkeys(actions)), len(stabilizer)


def _graph_cases():
    shrikhande, stabilizer_size = _shrikhande_actions()
    rook = tuple(element[2] for element in hamming_action(4, 2))
    sh_rows = _coefficient_rows("Shrikhande graph automorphism action", shrikhande, 4)
    rook_rows = _coefficient_rows("4x4 rook graph automorphism action", rook, 3)

    s4 = tuple(tuple(sigma) for sigma in permutations(range(4)))
    doob_actions = []
    for left in shrikhande:
        for right in s4:
            doob_actions.append(tuple(4 * left[x // 4] + right[x % 4] for x in range(64)))
    doob_actions = tuple(doob_actions)
    doob_rows = _coefficient_rows("Doob D(1,1)", doob_actions, 8)
    product_fixed = all(
        sum(image == index for index, image in enumerate(permutation_power(action, exponent)))
        == sum(image == index for index, image in enumerate(permutation_power(left, exponent)))
        * sum(image == index for index, image in enumerate(permutation_power(right, exponent)))
        for left in shrikhande
        for right in s4
        for exponent in range(1, 7)
        for action in (tuple(4 * left[x // 4] + right[x % 4] for x in range(64)),)
    )
    return {
        "Shrikhande": {
            "|Aut image|": len(shrikhande),
            "vertex stabilizer": stabilizer_size,
            "rows": sh_rows,
        },
        "rook": {"|Aut image|": len(tuple(dict.fromkeys(rook))), "rows": rook_rows},
        "Doob": {"|G image|": len(doob_actions), "product fixed-point formula": product_fixed, "rows": doob_rows},
        "passed": product_fixed
        and all(row["pass"] for row in sh_rows + rook_rows + doob_rows),
    }


def run_suite():
    """Run all grouped exact checks and return notebook-friendly records."""

    homogeneous = {
        "Johnson": run_johnson(),
        "Grassmann": run_grassmann(),
        "polar and generalized quadrangle": run_polar(),
        "building": _building_case(),
    }
    wreath = {"Hamming": run_hamming()}
    affine = {
        case["name"]: case
        for case in (
            _bilinear_case(),
            _alternating_case(),
            _hermitian_case(),
            _quadratic_case(),
            _orthogonal_plus_case(),
            _folded_case(),
            _halved_case(),
        )
    }
    graphs = _graph_cases()
    passed = (
        all(case["passed"] for case in homogeneous.values())
        and all(case["passed"] for case in wreath.values())
        and all(case["passed"] for case in affine.values())
        and graphs["passed"]
    )
    comparisons = (
        sum(len(case["rows"]) for case in homogeneous.values())
        + sum(len(case["rows"]) for case in wreath.values())
        + sum(len(case["rows"]) for case in affine.values())
        + sum(len(graphs[name]["rows"]) for name in ("Shrikhande", "rook", "Doob"))
    )
    return {
        "passed": passed,
        "coefficient comparisons": comparisons,
        "homogeneous": homogeneous,
        "wreath": wreath,
        "affine": affine,
        "graphs": graphs,
    }


if __name__ == "__main__":
    result = run_suite()
    print(
        f"Association schemes: {'PASS' if result['passed'] else 'FAIL'} "
        f"({result['coefficient comparisons']} coefficient comparisons)"
    )
