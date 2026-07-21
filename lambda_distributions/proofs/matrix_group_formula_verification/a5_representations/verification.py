"""Direct matrix checks for the three A5 representations in the proposal."""

from __future__ import annotations

from itertools import permutations
from math import prod

import numpy as np

from lambda_distributions.proofs.matrix_group_formula_verification.common import (
    coefficient_rows,
    projector_check,
)
from lambda_distributions.proofs.matrix_group_formula_verification.h3_reflection.verification import (
    h3_rotation_group,
)


SpectrumClass = tuple[int, tuple[complex, ...]]


def permutation_matrix(permutation: tuple[int, ...]) -> np.ndarray:
    matrix = np.zeros((len(permutation), len(permutation)), dtype=complex)
    for column, row in enumerate(permutation):
        matrix[row, column] = 1
    return matrix


def permutation_sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def a5_permutations() -> tuple[tuple[int, ...], ...]:
    answer = tuple(p for p in permutations(range(5)) if permutation_sign(p) == 1)
    if len(answer) != 60:
        raise AssertionError(f"expected 60 even permutations, found {len(answer)}")
    return answer


def a5_permutation_representation() -> tuple[np.ndarray, ...]:
    return tuple(permutation_matrix(p) for p in a5_permutations())


def a5_deleted_representation() -> tuple[np.ndarray, ...]:
    spanning = np.column_stack(
        [np.eye(5)[:, index] - np.eye(5)[:, 4] for index in range(4)]
    )
    basis, _ = np.linalg.qr(spanning)
    return tuple(
        basis.conj().T @ permutation_matrix(p) @ basis for p in a5_permutations()
    )


def _h_from_spectrum(spectrum: tuple[complex, ...], degree: int) -> complex:
    coefficients = np.zeros(degree + 1, dtype=complex)
    coefficients[0] = 1
    for eigenvalue in spectrum:
        for index in range(1, degree + 1):
            coefficients[index] += eigenvalue * coefficients[index - 1]
    return coefficients[degree]


def class_formula(classes: tuple[SpectrumClass, ...], tau: tuple[int, ...]) -> int:
    order = sum(size for size, _ in classes)
    value = sum(
        size * prod(_h_from_spectrum(spectrum, degree) for degree in tau)
        for size, spectrum in classes
    ) / order
    rounded = int(round(float(value.real)))
    if abs(value.imag) > 5e-7 or abs(value.real - rounded) > 5e-7:
        raise ArithmeticError(f"nonintegral A5 class result {value} for tau={tau}")
    return rounded


def spectral_classes():
    omega = np.exp(2j * np.pi / 3)
    xi = np.exp(2j * np.pi / 5)
    permutation = (
        (1, (1, 1, 1, 1, 1)),
        (15, (1, 1, 1, -1, -1)),
        (20, (1, 1, 1, omega, omega**-1)),
        (24, (1, xi, xi**2, xi**3, xi**4)),
    )
    deleted = (
        (1, (1, 1, 1, 1)),
        (15, (1, 1, -1, -1)),
        (20, (1, 1, omega, omega**-1)),
        (24, (xi, xi**2, xi**3, xi**4)),
    )
    rotation = (
        (1, (1, 1, 1)),
        (15, (1, -1, -1)),
        (20, (1, omega, omega**-1)),
        (12, (1, xi, xi**-1)),
        (12, (1, xi**2, xi**-2)),
    )
    return permutation, deleted, rotation


def representation_cases():
    permutation_classes, deleted_classes, rotation_classes = spectral_classes()
    return (
        ("A5 permutation", a5_permutation_representation(), permutation_classes),
        ("A5 deleted permutation", a5_deleted_representation(), deleted_classes),
        ("A5 rotation", h3_rotation_group(), rotation_classes),
    )


def run_suite(maximum_degree: int = 7):
    results = []
    for name, matrices, classes in representation_cases():
        rows = coefficient_rows(
            matrices,
            lambda tau, classes=classes: class_formula(classes, tau),
            maximum_degree,
        )
        selected = {
            tau: class_formula(classes, tau)
            for tau in ((1,), (2,), (1, 1), (1, 1, 1))
        }
        results.append(
            {
                "representation": name,
                "group order": len(matrices),
                "dimension": matrices[0].shape[0],
                "rows": rows,
                "selected coefficients": selected,
                "projector Sym^2": projector_check(matrices, (2,)),
                "projector V tensor V": projector_check(matrices, (1, 1)),
                "passed": all(row["pass"] for row in rows),
            }
        )
    return tuple(results)


if __name__ == "__main__":
    suite = run_suite()
    for result in suite:
        print(
            f"{'PASS' if result['passed'] else 'FAIL'} {result['representation']}: "
            f"order={result['group order']}, dimension={result['dimension']}, "
            f"coefficients={len(result['rows'])}"
        )
        print("  selected:", result["selected coefficients"])
        print("  Sym^2 projector:", result["projector Sym^2"])
    raise SystemExit(0 if all(result["passed"] for result in suite) else 1)
