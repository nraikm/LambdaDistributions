"""Matrix-level verification of the finite braid-image sigma-MGF.

The deliberately independent routes are:

* an explicit Reynolds projector on tensor products of symmetric powers;
* the elementwise power-trace recurrence implied by the Molien exponential;
* the same character calculation grouped into matrix conjugacy classes.

Only small dimensions are intended.  The symmetric-power implementation uses
an explicit orthonormal occupation basis inside ``V**tensor k``; this makes the
direct projector easy to audit and avoids symbolic representation software.
"""

from __future__ import annotations

from collections import deque
from functools import reduce
from itertools import product
from math import comb

import numpy as np


TOLERANCE = 2.0e-8


def matrix_key(matrix: np.ndarray, digits: int = 10) -> tuple[float, ...]:
    """A stable hash key for the cyclotomic matrices used in the sweeps."""

    real = np.round(matrix.real, digits)
    imag = np.round(matrix.imag, digits)
    real[np.abs(real) < 10 ** (-digits)] = 0.0
    imag[np.abs(imag) < 10 ** (-digits)] = 0.0
    return tuple(real.flat) + tuple(imag.flat)


def close_matrix_group(generators: list[np.ndarray], maximum_order: int = 20_000) -> list[np.ndarray]:
    """Close a finite unitary matrix group by breadth-first multiplication."""

    dimension = generators[0].shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    known = {matrix_key(identity): identity}
    queue = deque([identity])
    while queue:
        left = queue.popleft()
        for generator in generators:
            candidate = left @ generator
            key = matrix_key(candidate)
            if key not in known:
                known[key] = candidate
                queue.append(candidate)
                if len(known) > maximum_order:
                    raise RuntimeError("matrix closure exceeded the declared finite bound")
    return list(known.values())


def cyclic_scalar_braid_image(order: int, dimension: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The B_3 quotient with both generators equal to zeta times identity."""

    if order < 2 or dimension < 1:
        raise ValueError("order must be at least 2 and dimension must be positive")
    zeta = np.exp(2j * np.pi / order)
    generator = zeta * np.eye(dimension, dtype=np.complex128)
    return close_matrix_group([generator]), [generator, generator]


def s3_permutation_braid_image() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The standard B_3 -> S_3 quotient in its 3D permutation representation."""

    sigma_1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.complex128)
    sigma_2 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.complex128)
    return close_matrix_group([sigma_1, sigma_2]), [sigma_1, sigma_2]


def ising_b3_braid_image() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The 2D Ising/Jones B_3 image, with its conventional finite scalar lift."""

    hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    phase_gate = np.diag([1, 1j]).astype(np.complex128)
    ribbon_phase = np.exp(-1j * np.pi / 8)
    sigma_1 = ribbon_phase * phase_gate
    sigma_2 = ribbon_phase * hadamard @ phase_gate @ hadamard
    return close_matrix_group([sigma_1, sigma_2]), [sigma_1, sigma_2]


def adjoint_projective_image(lift: list[np.ndarray]) -> list[np.ndarray]:
    """Deduplicate the conjugation action on End(V), killing scalar ambiguity."""

    actions: dict[tuple[float, ...], np.ndarray] = {}
    for matrix in lift:
        action = np.kron(matrix, matrix.conjugate())
        actions.setdefault(matrix_key(action), action)
    return list(actions.values())


def braid_diagnostics(generators: list[np.ndarray]) -> dict[str, float | bool]:
    """Check unitarity and the B_3 braid relation."""

    sigma_1, sigma_2 = generators
    identity = np.eye(sigma_1.shape[0], dtype=np.complex128)
    unitary_error = max(
        np.linalg.norm(generator.conjugate().T @ generator - identity)
        for generator in generators
    )
    braid_error = np.linalg.norm(sigma_1 @ sigma_2 @ sigma_1 - sigma_2 @ sigma_1 @ sigma_2)
    return {
        "unitary_error": float(unitary_error),
        "braid_error": float(braid_error),
        "passed": bool(max(unitary_error, braid_error) < TOLERANCE),
    }


def full_twist(generators: list[np.ndarray]) -> np.ndarray:
    """Return rho((sigma_1 sigma_2)^3) for B_3."""

    return np.linalg.matrix_power(generators[0] @ generators[1], 3)


def _occupation_isometry(dimension: int, degree: int) -> np.ndarray:
    """Columns form the normalized symmetric occupation basis in V^tensor degree."""

    if degree == 0:
        return np.ones((1, 1), dtype=np.complex128)
    words = list(product(range(dimension), repeat=degree))
    occupations = sorted({tuple(word.count(letter) for letter in range(dimension)) for word in words})
    result = np.zeros((dimension**degree, len(occupations)), dtype=np.complex128)
    column_for = {occupation: column for column, occupation in enumerate(occupations)}
    multiplicities = {occupation: 0 for occupation in occupations}
    for word in words:
        occupation = tuple(word.count(letter) for letter in range(dimension))
        multiplicities[occupation] += 1
    for row, word in enumerate(words):
        occupation = tuple(word.count(letter) for letter in range(dimension))
        result[row, column_for[occupation]] = multiplicities[occupation] ** -0.5
    return result


def symmetric_power_matrix(matrix: np.ndarray, degree: int) -> np.ndarray:
    """Matrix of Sym^degree(matrix) in the normalized occupation basis."""

    if degree == 0:
        return np.ones((1, 1), dtype=np.complex128)
    tensor_power = reduce(np.kron, [matrix] * degree)
    isometry = _occupation_isometry(matrix.shape[0], degree)
    return isometry.conjugate().T @ tensor_power @ isometry


def tensor_symmetric_matrix(matrix: np.ndarray, tau: tuple[int, ...]) -> np.ndarray:
    factors = [symmetric_power_matrix(matrix, degree) for degree in tau]
    return reduce(np.kron, factors, np.ones((1, 1), dtype=np.complex128))


def reynolds_projector(group: list[np.ndarray], tau: tuple[int, ...]) -> np.ndarray:
    return sum((tensor_symmetric_matrix(matrix, tau) for matrix in group)) / len(group)


def complete_homogeneous_from_power_traces(matrix: np.ndarray, maximum_degree: int) -> list[complex]:
    """Compute h_r from r h_r = sum_{k=1}^r Tr(g^k) h_{r-k}."""

    traces = [0j] + [np.trace(np.linalg.matrix_power(matrix, k)) for k in range(1, maximum_degree + 1)]
    values = [1.0 + 0j]
    for degree in range(1, maximum_degree + 1):
        values.append(sum(traces[k] * values[degree - k] for k in range(1, degree + 1)) / degree)
    return values


def molien_coefficient(group: list[np.ndarray], tau: tuple[int, ...]) -> complex:
    maximum_degree = max(tau, default=0)
    total = 0j
    for matrix in group:
        h = complete_homogeneous_from_power_traces(matrix, maximum_degree)
        total += np.prod([h[degree] for degree in tau])
    return total / len(group)


def conjugacy_classes(group: list[np.ndarray]) -> list[list[np.ndarray]]:
    """Construct conjugacy classes directly from the closed matrix group."""

    by_key = {matrix_key(matrix): matrix for matrix in group}
    unseen = set(by_key)
    classes: list[list[np.ndarray]] = []
    while unseen:
        representative = by_key[next(iter(unseen))]
        class_keys = {
            matrix_key(element @ representative @ element.conjugate().T)
            for element in group
        }
        classes.append([by_key[key] for key in class_keys])
        unseen -= class_keys
    return classes


def class_molien_coefficient(group: list[np.ndarray], tau: tuple[int, ...]) -> tuple[complex, int]:
    maximum_degree = max(tau, default=0)
    total = 0j
    classes = conjugacy_classes(group)
    for conjugacy_class in classes:
        representative = conjugacy_class[0]
        h = complete_homogeneous_from_power_traces(representative, maximum_degree)
        total += len(conjugacy_class) * np.prod([h[degree] for degree in tau])
    return total / len(group), len(classes)


def clean_integer(value: complex) -> int:
    rounded = int(round(value.real))
    if abs(value.imag) > TOLERANCE or abs(value.real - rounded) > TOLERANCE:
        raise AssertionError(f"expected an integer, got {value!r}")
    return rounded


def verify_case(group: list[np.ndarray], tau: tuple[int, ...], label: str) -> dict[str, object]:
    """Compare direct projector rank with both versions of the proposed formula."""

    projector = reynolds_projector(group, tau)
    singular_values = np.linalg.eigvalsh((projector + projector.conjugate().T) / 2)
    rank = int(np.count_nonzero(singular_values > 0.5))
    element_value = molien_coefficient(group, tau)
    class_value, class_count = class_molien_coefficient(group, tau)
    target_dimension = int(np.prod([comb(group[0].shape[0] + degree - 1, degree) for degree in tau]))
    idempotence = float(np.linalg.norm(projector @ projector - projector))
    hermitian = float(np.linalg.norm(projector - projector.conjugate().T))
    formula_integer = clean_integer(element_value)
    class_integer = clean_integer(class_value)
    passed = rank == formula_integer == class_integer and max(idempotence, hermitian) < TOLERANCE
    return {
        "representation": label,
        "dimension": group[0].shape[0],
        "group_order": len(group),
        "class_count": class_count,
        "tau": str(tau),
        "total_degree": sum(tau),
        "target_dimension": target_dimension,
        "projector_rank": rank,
        "element_formula": formula_integer,
        "class_formula": class_integer,
        "projector_error": max(idempotence, hermitian),
        "passed": passed,
    }

