"""Construct W(H3) from the icosahedron and test the explicit spectral formula."""

from __future__ import annotations

from itertools import permutations, product
from math import prod

import numpy as np

from for_this_guy.matrix_group_formula_verification.common import (
    coefficient_rows,
    complete_characters,
    projector_check,
)


def icosahedron_vertices() -> np.ndarray:
    phi = (1 + np.sqrt(5.0)) / 2
    vertices = []
    for signs in product((-1.0, 1.0), repeat=2):
        vertices.append((0.0, signs[0], signs[1] * phi))
        vertices.append((signs[0], signs[1] * phi, 0.0))
        vertices.append((signs[0] * phi, 0.0, signs[1]))
    return np.array(vertices, dtype=float)


def _matrix_key(matrix: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(matrix.real, 9).ravel())


def h3_rotation_group() -> tuple[np.ndarray, ...]:
    """Enumerate the 60 orientation-preserving symmetries of an icosahedron."""

    vertices = icosahedron_vertices()
    base_indices = next(
        triple
        for triple in permutations(range(len(vertices)), 3)
        if abs(np.linalg.det(vertices[list(triple)].T)) > 0.5
    )
    base = vertices[list(base_indices)].T
    base_gram = base.T @ base
    inverse_base = np.linalg.inv(base)
    vertex_keys = {_matrix_key(vertex.reshape(3, 1)) for vertex in vertices}
    rotations: dict[tuple[float, ...], np.ndarray] = {}
    for target_indices in permutations(range(len(vertices)), 3):
        target = vertices[list(target_indices)].T
        if not np.allclose(target.T @ target, base_gram, atol=1e-9):
            continue
        candidate = target @ inverse_base
        if np.linalg.det(candidate) < 0 or not np.allclose(candidate.T @ candidate, np.eye(3), atol=1e-8):
            continue
        images = candidate @ vertices.T
        if all(_matrix_key(images[:, index].reshape(3, 1)) in vertex_keys for index in range(12)):
            rotations[_matrix_key(candidate)] = candidate.astype(complex)
    if len(rotations) != 60:
        raise AssertionError(f"expected 60 rotations, found {len(rotations)}")
    return tuple(rotations.values())


def h3_group() -> tuple[np.ndarray, ...]:
    rotations = h3_rotation_group()
    matrices = rotations + tuple(-matrix for matrix in rotations)
    if len({_matrix_key(matrix) for matrix in matrices}) != 120:
        raise AssertionError("the full H3 group should have order 120")
    return matrices


def spectral_classes() -> tuple[tuple[int, tuple[complex, ...]], ...]:
    omega = np.exp(2j * np.pi / 3)
    xi = np.exp(2j * np.pi / 5)
    positive = (
        (1, (1, 1, 1)),
        (15, (1, -1, -1)),
        (20, (1, omega, omega**2)),
        (12, (1, xi, xi**-1)),
        (12, (1, xi**2, xi**-2)),
    )
    return positive + tuple((weight, tuple(-value for value in spectrum)) for weight, spectrum in positive)


def octahedral_rotation_group() -> tuple[np.ndarray, ...]:
    """The 24 determinant-one signed permutation matrices."""

    matrices = []
    for permutation in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3), dtype=complex)
            for column, row in enumerate(permutation):
                matrix[row, column] = signs[column]
            if round(np.linalg.det(matrix).real) == 1:
                matrices.append(matrix)
    return tuple(matrices)


def tetrahedral_rotation_group() -> tuple[np.ndarray, ...]:
    vertices = {
        (1, 1, 1),
        (1, -1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
    }
    return tuple(
        matrix
        for matrix in octahedral_rotation_group()
        if all(tuple(np.rint(matrix.real @ vertex).astype(int)) in vertices for vertex in vertices)
    )


def rotation_class_formula(
    classes: tuple[tuple[int, float], ...], tau: tuple[int, ...]
) -> int:
    total = 0j
    order = sum(size for size, _ in classes)
    for size, angle in classes:
        spectrum = (1, np.exp(1j * angle), np.exp(-1j * angle))
        total += size * prod(_h_from_spectrum(spectrum, degree) for degree in tau)
    total /= order
    rounded = int(round(float(total.real)))
    if abs(total.imag) > 5e-7 or abs(total.real - rounded) > 5e-7:
        raise ArithmeticError(f"nonintegral rotation-class result {total}")
    return rounded


def polyhedral_cases():
    return (
        (
            "A4 tetrahedral rotations",
            tetrahedral_rotation_group(),
            ((1, 0.0), (8, 2 * np.pi / 3), (3, np.pi)),
        ),
        (
            "S4 octahedral rotations",
            octahedral_rotation_group(),
            ((1, 0.0), (8, 2 * np.pi / 3), (6, np.pi / 2), (9, np.pi)),
        ),
        (
            "A5 icosahedral rotations",
            h3_rotation_group(),
            (
                (1, 0.0),
                (20, 2 * np.pi / 3),
                (15, np.pi),
                (12, 2 * np.pi / 5),
                (12, 4 * np.pi / 5),
            ),
        ),
    )


def run_polyhedral_suite(maximum_degree: int = 7):
    results = []
    for name, matrices, classes in polyhedral_cases():
        rows = coefficient_rows(
            matrices,
            lambda tau, classes=classes: rotation_class_formula(classes, tau),
            maximum_degree,
        )
        results.append(
            {
                "group": name,
                "order": len(matrices),
                "rows": rows,
                "checks": len(rows),
                "passed": all(row["pass"] for row in rows),
            }
        )
    return tuple(results)


def _h_from_spectrum(spectrum: tuple[complex, ...], degree: int) -> complex:
    coefficients = np.zeros(degree + 1, dtype=complex)
    coefficients[0] = 1
    for eigenvalue in spectrum:
        for index in range(1, degree + 1):
            coefficients[index] += eigenvalue * coefficients[index - 1]
    return coefficients[degree]


def spectral_formula(tau: tuple[int, ...]) -> int:
    total = sum(
        weight * prod(_h_from_spectrum(spectrum, degree) for degree in tau)
        for weight, spectrum in spectral_classes()
    ) / 120
    rounded = int(round(float(total.real)))
    if abs(total.imag) > 5e-7 or abs(total.real - rounded) > 5e-7:
        raise ArithmeticError(f"nonintegral H3 spectral result {total}")
    return rounded


def invariant_degree_formula(degree: int) -> int:
    return sum(
        1
        for a in range(degree // 2 + 1)
        for b in range(degree // 6 + 1)
        for c in range(degree // 10 + 1)
        if 2 * a + 6 * b + 10 * c == degree
    )


def group_diagnostics(matrices: tuple[np.ndarray, ...]) -> dict[str, float | int | bool]:
    keys = {_matrix_key(matrix) for matrix in matrices}
    closure = all(_matrix_key(left @ right) in keys for left in matrices for right in matrices)
    identity = np.eye(3)
    flattened = np.array([matrix.real.ravel() for matrix in matrices])
    minimum_separation = min(
        float(np.linalg.norm(flattened[i] - flattened[j]))
        for i in range(len(flattened))
        for j in range(i)
    )
    maximum_closure_distance = max(
        min(float(np.linalg.norm((left @ right).real - candidate.real)) for candidate in matrices)
        for left in matrices
        for right in matrices
    )
    vertices = icosahedron_vertices()
    base_indices = next(
        triple
        for triple in permutations(range(len(vertices)), 3)
        if abs(np.linalg.det(vertices[list(triple)].T)) > 0.5
    )
    base_condition = float(np.linalg.cond(vertices[list(base_indices)].T))
    return {
        "order": len(matrices),
        "closed under multiplication": closure,
        "equality tolerance": 5.0e-10,
        "minimum matrix separation": minimum_separation,
        "maximum closure matching distance": maximum_closure_distance,
        "base triple condition number": base_condition,
        "maximum orthogonality error": max(
            float(np.linalg.norm(matrix.T @ matrix - identity)) for matrix in matrices
        ),
        "determinants are signs": all(abs(abs(np.linalg.det(matrix)) - 1) < 1e-8 for matrix in matrices),
    }


def run_suite(maximum_degree: int = 8):
    matrices = h3_group()
    coefficient_checks = coefficient_rows(matrices, spectral_formula, maximum_degree)
    one_variable_checks = tuple(
        {
            "degree": degree,
            "direct matrix average": int(round(np.mean([
                complete_characters(matrix, maximum_degree)[degree] for matrix in matrices
            ]).real)),
            "degree-product formula": invariant_degree_formula(degree),
        }
        for degree in range(maximum_degree + 1)
    )
    return {
        "diagnostics": group_diagnostics(matrices),
        "coefficient checks": coefficient_checks,
        "one-variable checks": one_variable_checks,
        "projector": projector_check(matrices, (2,)),
        "passed": all(row["pass"] for row in coefficient_checks)
        and all(row["direct matrix average"] == row["degree-product formula"] for row in one_variable_checks),
    }


if __name__ == "__main__":
    result = run_suite(10)
    polyhedral = run_polyhedral_suite()
    print("H3 diagnostics:", result["diagnostics"])
    print("H3 projector:", result["projector"])
    print(f"{'PASS' if result['passed'] else 'FAIL'} H3: {len(result['coefficient checks'])} multivariate checks")
    for item in polyhedral:
        print(f"{'PASS' if item['passed'] else 'FAIL'} {item['group']}: {item['checks']} checks")
    raise SystemExit(0 if result["passed"] and all(item["passed"] for item in polyhedral) else 1)
