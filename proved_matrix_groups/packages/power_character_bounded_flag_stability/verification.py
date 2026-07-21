"""Exact checks for the power-character and bounded-support theorem.

The suite deliberately separates four mathematical routes:

* explicit symmetric-power matrices and a Reynolds projector;
* class-power characters and Newton's recurrence;
* permutation matrices, cycle factors, parabolic intersections, and orbits;
* generator-orbit enumeration in consecutive stable ranks.

All finite-field constructions in this file use prime fields.  This keeps the
matrix arithmetic transparent; the proved statements are not restricted to
prime fields.
"""

from __future__ import annotations

from collections import deque
from functools import reduce
from itertools import permutations, product
from math import comb, prod

import numpy as np

from ..association_scheme_permutation_representations.finite_field import (
    general_linear_group,
    induced_subspace_permutations,
    mat_mul,
    mat_vec,
    matrix_power,
    move_subspace,
    subspaces,
)
from ..association_scheme_permutation_representations.shared import (
    TAUS,
    compose,
    cycle_counts,
    direct_orbit_count,
    fixed_formula_reconstructs,
    multisets,
    permutation_power,
    verify_action,
)


def inverse_permutation(permutation: tuple[int, ...]) -> tuple[int, ...]:
    inverse = [0] * len(permutation)
    for source, target in enumerate(permutation):
        inverse[target] = source
    return tuple(inverse)


def symmetric_power_matrix(matrix: np.ndarray, degree: int) -> np.ndarray:
    """Matrix of Sym^degree(matrix) in the binary-monomial basis."""

    if degree == 0:
        return np.ones((1, 1), dtype=np.int64)
    answer = np.zeros((degree + 1, degree + 1), dtype=np.int64)
    a, b = int(matrix[0, 0]), int(matrix[0, 1])
    c, d = int(matrix[1, 0]), int(matrix[1, 1])
    for source_twos in range(degree + 1):
        first_degree = degree - source_twos
        for target_twos in range(degree + 1):
            coefficient = 0
            for from_first in range(first_degree + 1):
                from_second = target_twos - from_first
                if 0 <= from_second <= source_twos:
                    coefficient += (
                        comb(first_degree, from_first)
                        * a ** (first_degree - from_first)
                        * c**from_first
                        * comb(source_twos, from_second)
                        * b ** (source_twos - from_second)
                        * d**from_second
                    )
            answer[target_twos, source_twos] = coefficient
    return answer


def tensor_symmetric_matrix(matrix: np.ndarray, tau: tuple[int, ...]) -> np.ndarray:
    return reduce(np.kron, (symmetric_power_matrix(matrix, degree) for degree in tau))


def standard_s3_matrices() -> tuple[np.ndarray, ...]:
    """The two-dimensional standard representation in an integral basis."""

    basis = (
        np.array((1, 0, -1), dtype=np.int64),
        np.array((0, 1, -1), dtype=np.int64),
    )
    matrices = []
    for permutation in permutations(range(3)):
        columns = []
        for vector in basis:
            moved = np.zeros(3, dtype=np.int64)
            for source, value in enumerate(vector):
                moved[permutation[source]] = value
            columns.append((int(moved[0]), int(moved[1])))
        matrices.append(np.array(columns, dtype=np.int64).T)
    return tuple(matrices)


def complete_characters(matrix: np.ndarray, maximum: int) -> tuple[int, ...]:
    traces = [0]
    power = np.eye(matrix.shape[0], dtype=np.int64)
    for _ in range(maximum):
        power = power @ matrix
        traces.append(int(np.trace(power)))
    complete = [1]
    for degree in range(1, maximum + 1):
        numerator = sum(traces[r] * complete[degree - r] for r in range(1, degree + 1))
        assert numerator % degree == 0
        complete.append(numerator // degree)
    return tuple(complete)


def verify_s3_standard() -> tuple[dict, ...]:
    """Compare Reynolds ranks with the proposed class-power coefficient."""

    matrices = standard_s3_matrices()
    maximum = max(max(tau) for tau in TAUS)
    complete = tuple(complete_characters(matrix, maximum) for matrix in matrices)
    rows = []
    for tau in TAUS:
        represented = tuple(tensor_symmetric_matrix(matrix, tau) for matrix in matrices)
        reynolds = sum(represented) / len(represented)
        rank = int(np.linalg.matrix_rank(reynolds, tol=1e-10))
        formula_numerator = sum(prod(character[d] for d in tau) for character in complete)
        assert formula_numerator % len(matrices) == 0
        formula = formula_numerator // len(matrices)
        idempotence_error = float(np.max(np.abs(reynolds @ reynolds - reynolds)))
        rows.append(
            {
                "group": "S3 standard = Steinberg of GL_2(2)",
                "tau": str(tau),
                "Reynolds rank": rank,
                "power-character formula": formula,
                "projector error": idempotence_error,
                "pass": rank == formula and idempotence_error < 1e-10,
            }
        )
    return tuple(rows)


def conjugacy_data(matrices, n: int, q: int):
    """Return class and centralizer data for a small explicitly built GL."""

    order = len(matrices)
    unassigned = set(matrices)
    class_of = {}
    classes = []
    while unassigned:
        representative = next(iter(unassigned))
        conjugates = {
            mat_mul(
                mat_mul(actor, representative, n, q),
                matrix_power(actor, n, q, order - 1),
                n,
                q,
            )
            for actor in matrices
        }
        class_index = len(classes)
        classes.append(frozenset(conjugates))
        for element in conjugates:
            class_of[element] = class_index
        unassigned.difference_update(conjugates)
    centralizer = {
        element: order // len(classes[class_of[element]]) for element in matrices
    }
    return tuple(classes), class_of, centralizer


def verify_gl3_parabolic():
    """Check GL_3(2)/P by matrices, intersections, cycles, and orbits."""

    n, q = 3, 2
    matrices = general_linear_group(n, q)
    points = subspaces(n, 1, q)
    actions = induced_subspace_permutations(matrices, points, q)
    rows = verify_action("GL_3(2) on projective points GL_3(2)/P", actions)

    classes, class_of, centralizer = conjugacy_data(matrices, n, q)
    base_point = points[0]
    parabolic = tuple(matrix for matrix in matrices if move_subspace(matrix, base_point, q) == base_point)
    class_parabolic_counts = tuple(len(conjugacy_class.intersection(parabolic)) for conjugacy_class in classes)

    fixed_equalities = 0
    cycle_reconstructions = 0
    for matrix, action in zip(matrices, actions, strict=True):
        def parabolic_fixed(exponent: int) -> int:
            powered = matrix_power(matrix, n, q, exponent)
            numerator = centralizer[powered] * class_parabolic_counts[class_of[powered]]
            assert numerator % len(parabolic) == 0
            return numerator // len(parabolic)

        maximum = max(cycle_counts(action))
        for exponent in range(1, maximum + 1):
            actual = sum(
                permutation_power(action, exponent)[point] == point
                for point in range(len(points))
            )
            fixed_equalities += int(actual == parabolic_fixed(exponent))
        cycle_reconstructions += int(fixed_formula_reconstructs(action, parabolic_fixed))

    diagnostics = {
        "group": "GL_3(2)",
        "|G|": len(matrices),
        "|G/P|": len(points),
        "|P|": len(parabolic),
        "classes": len(classes),
        "fixed-point equalities": fixed_equalities,
        "cycle reconstructions": cycle_reconstructions,
        "expected cycle reconstructions": len(matrices),
        "pass": all(row["pass"] for row in rows) and cycle_reconstructions == len(matrices),
    }
    return tuple(rows), diagnostics


def verify_s3_conjugation():
    group = tuple(permutations(range(3)))
    position = {element: index for index, element in enumerate(group)}
    actions = []
    fixed_checks = 0
    reconstruction_checks = 0
    for actor in group:
        inverse = inverse_permutation(actor)
        action = tuple(
            position[compose(compose(actor, element), inverse)] for element in group
        )
        actions.append(action)

        def centralizer_fixed(exponent: int) -> int:
            powered = permutation_power(actor, exponent)
            return sum(compose(powered, x) == compose(x, powered) for x in group)

        maximum = max(cycle_counts(action))
        fixed_checks += sum(
            sum(permutation_power(action, exponent)[x] == x for x in range(len(group)))
            == centralizer_fixed(exponent)
            for exponent in range(1, maximum + 1)
        )
        reconstruction_checks += int(fixed_formula_reconstructs(action, centralizer_fixed))

    rows = list(verify_action("S3 acting on itself by conjugation", tuple(actions)))
    # The general helper assumes a transitive action in degree one.  Conjugacy
    # has one orbit per conjugacy class, so retain the three independent
    # calculations and replace only that helper-specific expectation.
    for row in rows:
        row["low-degree prediction"] = 3 if row["tau"] == "(1,)" else None
        row["pass"] = (
            row["matrix average"] == row["cycle formula"] == row["direct orbits"]
        )
    diagnostics = {
        "group": "S3 conjugation action",
        "|G|": len(group),
        "fixed-point equalities": fixed_checks,
        "cycle reconstructions": reconstruction_checks,
        "expected cycle reconstructions": len(group),
        "pass": all(row["pass"] for row in rows) and reconstruction_checks == len(group),
    }
    return tuple(rows), diagnostics


def transvection_generators(n: int, q: int = 2):
    matrices = []
    for row in range(n):
        for column in range(n):
            if row == column:
                continue
            matrix = [int(i == j) for i in range(n) for j in range(n)]
            matrix[row * n + column] = 1
            matrices.append(tuple(entry % q for entry in matrix))
    return tuple(matrices)


def generator_actions_on_subspaces(n: int, k: int, q: int = 2):
    spaces = subspaces(n, k, q)
    position = {space: index for index, space in enumerate(spaces)}
    actions = tuple(
        tuple(position[move_subspace(matrix, space, q)] for space in spaces)
        for matrix in transvection_generators(n, q)
    )
    return spaces, actions


def generator_orbit_count(generators, point_count: int, tau: tuple[int, ...]) -> int:
    """Count configuration orbits from a generating set, without listing GL_n."""

    remaining = set(product(*(multisets(point_count, degree) for degree in tau)))
    orbit_count = 0
    while remaining:
        start = remaining.pop()
        orbit_count += 1
        queue = deque((start,))
        while queue:
            configuration = queue.popleft()
            for generator in generators:
                moved = tuple(
                    tuple(sorted(generator[point] for point in block))
                    for block in configuration
                )
                if moved in remaining:
                    remaining.remove(moved)
                    queue.append(moved)
    return orbit_count


def verify_type_a_stability():
    rows = []
    projective_values = {}
    for n in (1, 2, 3, 4):
        points, generators = generator_actions_on_subspaces(n, 1)
        for tau in TAUS:
            projective_values[n, tau] = generator_orbit_count(generators, len(points), tau)
    for tau in TAUS:
        degree = sum(tau)
        eligible = tuple(n for n in (1, 2, 3, 4) if n >= degree)
        stable = len({projective_values[n, tau] for n in eligible}) == 1
        rows.append(
            {
                "family": "GL_n(2) on projective points",
                "tau": str(tau),
                "proved threshold n>=d|tau|": degree,
                "n=1": projective_values[1, tau],
                "n=2": projective_values[2, tau],
                "n=3": projective_values[3, tau],
                "n=4": projective_values[4, tau],
                "pass": stable,
            }
        )

    grassmann_values = {}
    for n in (4, 5):
        spaces, generators = generator_actions_on_subspaces(n, 2)
        for tau in ((2,), (1, 1)):
            grassmann_values[n, tau] = generator_orbit_count(generators, len(spaces), tau)
    for tau in ((2,), (1, 1)):
        rows.append(
            {
                "family": "GL_n(2) on Gr_2(n)",
                "tau": str(tau),
                "proved threshold n>=d|tau|": 4,
                "n=4": grassmann_values[4, tau],
                "n=5": grassmann_values[5, tau],
                "expected": 3,
                "pass": grassmann_values[4, tau] == grassmann_values[5, tau] == 3,
            }
        )
    return tuple(rows)


def run_suite():
    power_rows = verify_s3_standard()
    parabolic_rows, parabolic_diagnostics = verify_gl3_parabolic()
    conjugation_rows, conjugation_diagnostics = verify_s3_conjugation()
    stability_rows = verify_type_a_stability()
    passed = (
        all(row["pass"] for row in power_rows)
        and all(row["pass"] for row in parabolic_rows)
        and parabolic_diagnostics["pass"]
        and all(row["pass"] for row in conjugation_rows)
        and conjugation_diagnostics["pass"]
        and all(row["pass"] for row in stability_rows)
    )
    return {
        "family": "Power-character and bounded-support stability",
        "power-character": power_rows,
        "parabolic": parabolic_rows,
        "parabolic diagnostics": parabolic_diagnostics,
        "conjugation": conjugation_rows,
        "conjugation diagnostics": conjugation_diagnostics,
        "stability": stability_rows,
        "comparison count": len(power_rows) + len(parabolic_rows) + len(conjugation_rows) + len(stability_rows),
        "passed": passed,
    }


if __name__ == "__main__":
    result = run_suite()
    print(f"{result['family']}: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"{result['comparison count']} coefficient/stability rows")
    print(result["parabolic diagnostics"])
    print(result["conjugation diagnostics"])
