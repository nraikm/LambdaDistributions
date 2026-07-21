"""Construct the binary polyhedral matrix groups and test the lift formula."""

from __future__ import annotations

from math import prod

import numpy as np

from lambda_distributions.proofs.matrix_group_formula_verification.common import (
    coefficient_rows,
    projector_check,
)
from lambda_distributions.proofs.matrix_group_formula_verification.h3_reflection.verification import (
    h3_rotation_group,
    octahedral_rotation_group,
    tetrahedral_rotation_group,
)


def rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Return one unit quaternion lift (w,x,y,z) of a 3D rotation."""

    matrix = np.asarray(rotation.real, dtype=float)
    trace = np.trace(matrix)
    if trace > 0:
        scale = 2 * np.sqrt(trace + 1)
        w = scale / 4
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        largest = int(np.argmax(diagonal))
        if largest == 0:
            scale = 2 * np.sqrt(max(0.0, 1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
            x = scale / 4
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif largest == 1:
            scale = 2 * np.sqrt(max(0.0, 1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
            y = scale / 4
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = 2 * np.sqrt(max(0.0, 1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
            z = scale / 4
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
    quaternion = np.array((w, x, y, z), dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def quaternion_to_su2(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    """The standard faithful two-dimensional complex quaternion representation."""

    w, x, y, z = quaternion
    return np.array(
        [[w + 1j * z, y + 1j * x], [-y + 1j * x, w - 1j * z]],
        dtype=complex,
    )


def binary_lift(rotations: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    positive = tuple(quaternion_to_su2(rotation_to_quaternion(r)) for r in rotations)
    matrices = positive + tuple(-matrix for matrix in positive)
    keys = {tuple(np.round(matrix, 9).ravel()) for matrix in matrices}
    if len(keys) != 2 * len(rotations):
        raise AssertionError("binary lift did not have twice the parent order")
    return matrices


def _h_from_spectrum(spectrum: tuple[complex, ...], degree: int) -> complex:
    coefficients = np.zeros(degree + 1, dtype=complex)
    coefficients[0] = 1
    for eigenvalue in spectrum:
        for index in range(1, degree + 1):
            coefficients[index] += eigenvalue * coefficients[index - 1]
    return coefficients[degree]


def binary_class_formula(
    angle_counts: tuple[tuple[int, float], ...], tau: tuple[int, ...]
) -> int:
    parent_order = sum(count for count, _ in angle_counts)
    total = 0j
    for count, angle in angle_counts:
        positive = (np.exp(0.5j * angle), np.exp(-0.5j * angle))
        negative = tuple(-value for value in positive)
        positive_character = prod(_h_from_spectrum(positive, degree) for degree in tau)
        negative_character = prod(_h_from_spectrum(negative, degree) for degree in tau)
        total += count * (positive_character + negative_character) / 2
    total /= parent_order
    rounded = int(round(float(total.real)))
    if abs(total.imag) > 8e-7 or abs(total.real - rounded) > 8e-7:
        raise ArithmeticError(f"nonintegral binary class result {total} for tau={tau}")
    return rounded


def binary_cases():
    return (
        (
            "2T binary tetrahedral",
            binary_lift(tetrahedral_rotation_group()),
            ((1, 0.0), (8, 2 * np.pi / 3), (3, np.pi)),
        ),
        (
            "2O binary octahedral",
            binary_lift(octahedral_rotation_group()),
            ((1, 0.0), (6, np.pi / 2), (8, 2 * np.pi / 3), (9, np.pi)),
        ),
        (
            "2I binary icosahedral",
            binary_lift(h3_rotation_group()),
            (
                (1, 0.0),
                (15, np.pi),
                (20, 2 * np.pi / 3),
                (12, 2 * np.pi / 5),
                (12, 4 * np.pi / 5),
            ),
        ),
    )


def group_diagnostics(matrices: tuple[np.ndarray, ...]):
    keys = {tuple(np.round(matrix, 8).ravel()) for matrix in matrices}
    closure = all(
        tuple(np.round(left @ right, 8).ravel()) in keys
        for left in matrices
        for right in matrices
    )
    return {
        "order": len(matrices),
        "closed under multiplication": closure,
        "maximum unitarity error": max(
            float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(2)))
            for matrix in matrices
        ),
        "maximum determinant error": max(
            float(abs(np.linalg.det(matrix) - 1)) for matrix in matrices
        ),
        "contains -I": any(np.allclose(matrix, -np.eye(2)) for matrix in matrices),
    }


def run_suite(maximum_degree: int = 8):
    results = []
    for name, matrices, angle_counts in binary_cases():
        rows = coefficient_rows(
            matrices,
            lambda tau, angle_counts=angle_counts: binary_class_formula(angle_counts, tau),
            maximum_degree,
        )
        odd_rows = tuple(row for row in rows if row["degree"] % 2)
        results.append(
            {
                "group": name,
                "diagnostics": group_diagnostics(matrices),
                "rows": rows,
                "odd-degree filter passed": all(
                    row["direct matrix average"] == 0 for row in odd_rows
                ),
                "projector V tensor V": projector_check(matrices, (1, 1)),
                "projector Sym^2": projector_check(matrices, (2,)),
                "passed": all(row["pass"] for row in rows)
                and all(row["direct matrix average"] == 0 for row in odd_rows),
            }
        )
    return tuple(results)


if __name__ == "__main__":
    suite = run_suite()
    for result in suite:
        print(
            f"{'PASS' if result['passed'] else 'FAIL'} {result['group']}: "
            f"{len(result['rows'])} coefficient checks"
        )
        print("  diagnostics:", result["diagnostics"])
        print("  V tensor V projector:", result["projector V tensor V"])
    raise SystemExit(0 if all(result["passed"] for result in suite) else 1)
