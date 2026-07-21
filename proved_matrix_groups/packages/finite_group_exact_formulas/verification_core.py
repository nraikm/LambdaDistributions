"""Independent finite-group checks for the exact sigma-MGF formulas.

The direct route constructs matrices on tensor products of symmetric powers and
forms the Reynolds projector.  The formula routes use cycle types, weight
counts, or the displayed spectra, so they do not reuse the projector code.
The routines are intentionally small-case tools: they favor transparency over
asymptotic performance.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import permutations, product
from math import comb, factorial, gcd, prod, sqrt

import numpy as np


def integer_partitions(total: int, ceiling: int | None = None):
    if total == 0:
        yield ()
        return
    ceiling = min(total, total if ceiling is None else ceiling)
    for first in range(ceiling, 0, -1):
        for tail in integer_partitions(total - first, first):
            yield (first, *tail)


def z_partition(partition: tuple[int, ...]) -> int:
    counts = Counter(partition)
    return prod(part**count * factorial(count) for part, count in counts.items())


@lru_cache(maxsize=None)
def weak_compositions(total: int, length: int) -> tuple[tuple[int, ...], ...]:
    if total < 0 or length < 1:
        raise ValueError("total must be nonnegative and length must be positive")
    if length == 1:
        return ((total,),)
    return tuple(
        (first,) + tail
        for first in range(total + 1)
        for tail in weak_compositions(total - first, length - 1)
    )


def _kron_power(matrix: np.ndarray, degree: int) -> np.ndarray:
    result = np.ones((1, 1), dtype=complex)
    for _ in range(degree):
        result = np.kron(result, matrix)
    return result


@lru_cache(maxsize=None)
def _symmetric_isometry(dimension: int, degree: int) -> np.ndarray:
    """Orthonormal occupancy basis embedded in V tensor ... tensor V."""

    if degree == 0:
        return np.ones((1, 1), dtype=complex)
    compositions = weak_compositions(degree, dimension)
    columns = {composition: index for index, composition in enumerate(compositions)}
    result = np.zeros((dimension**degree, len(compositions)), dtype=complex)
    words = tuple(product(range(dimension), repeat=degree))
    counts_by_word = [tuple(word.count(index) for index in range(dimension)) for word in words]
    multiplicities = Counter(counts_by_word)
    for row, counts in enumerate(counts_by_word):
        result[row, columns[counts]] = 1 / sqrt(multiplicities[counts])
    return result


def symmetric_power_matrix(matrix: np.ndarray, degree: int) -> np.ndarray:
    dimension = matrix.shape[0]
    isometry = _symmetric_isometry(dimension, degree)
    return isometry.conj().T @ _kron_power(matrix, degree) @ isometry


def target_matrix(matrix: np.ndarray, tau: tuple[int, ...]) -> np.ndarray:
    result = np.ones((1, 1), dtype=complex)
    for degree in tau:
        result = np.kron(result, symmetric_power_matrix(matrix, degree))
    return result


def target_dimension(dimension: int, tau: tuple[int, ...]) -> int:
    return prod(comb(dimension + degree - 1, degree) for degree in tau)


def reynolds_check(
    matrices: tuple[np.ndarray, ...], tau: tuple[int, ...], dimension_cap: int = 2500
) -> dict[str, float | int]:
    ambient = target_dimension(matrices[0].shape[0], tau)
    if ambient > dimension_cap:
        raise ValueError(f"target dimension {ambient} exceeds safety cap {dimension_cap}")
    projector = sum(
        (target_matrix(matrix, tau) for matrix in matrices),
        start=np.zeros((ambient, ambient), dtype=complex),
    ) / len(matrices)
    rank = int(np.count_nonzero(np.linalg.svd(projector, compute_uv=False) > 1e-8))
    trace = np.trace(projector)
    return {
        "group_order": len(matrices),
        "representation_dimension": matrices[0].shape[0],
        "target_dimension": ambient,
        "projector_rank": rank,
        "projector_trace": float(np.real_if_close(trace).real),
        "idempotence_error": float(np.linalg.norm(projector @ projector - projector)),
        "self_adjoint_error": float(np.linalg.norm(projector.conj().T - projector)),
    }


def character_average(matrices: tuple[np.ndarray, ...], tau: tuple[int, ...]) -> complex:
    return sum(
        prod(np.trace(symmetric_power_matrix(matrix, degree)) for degree in tau)
        for matrix in matrices
    ) / len(matrices)


def _block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    size = sum(block.shape[0] for block in blocks)
    result = np.zeros((size, size), dtype=complex)
    offset = 0
    for block in blocks:
        width = block.shape[0]
        result[offset : offset + width, offset : offset + width] = block
        offset += width
    return result


def dicyclic_representation(
    m: int,
    two_d_counts: dict[int, int],
    one_d_characters: tuple[tuple[complex, complex, int], ...] = (),
) -> tuple[np.ndarray, ...]:
    """Matrices of a^j and x a^j for a direct sum of stated irreducibles."""

    if m < 2:
        raise ValueError("m must be at least 2")
    zeta = np.exp(1j * np.pi / m)
    matrices = []
    for coset in (0, 1):
        for exponent in range(2 * m):
            blocks: list[np.ndarray] = []
            for alpha, beta, count in one_d_characters:
                if not np.isclose(alpha**2, 1) or not np.isclose(beta**2, alpha**m):
                    raise ValueError("one-dimensional character violates beta^2=alpha^m")
                scalar = alpha**exponent if coset == 0 else beta * alpha**exponent
                blocks.extend(np.array([[scalar]], dtype=complex) for _ in range(count))
            for k, count in sorted(two_d_counts.items()):
                if not 1 <= k <= m - 1:
                    raise ValueError("two-dimensional indices must satisfy 1 <= k <= m-1")
                a_power = np.diag([zeta ** (k * exponent), zeta ** (-k * exponent)])
                x = np.array([[0, (-1) ** k], [1, 0]], dtype=complex)
                block = a_power if coset == 0 else x @ a_power
                blocks.extend(block.copy() for _ in range(count))
            if not blocks:
                raise ValueError("the representation must contain at least one summand")
            matrices.append(_block_diag(blocks))
    return tuple(matrices)


def dicyclic_relation_error(
    m: int,
    two_d_counts: dict[int, int],
    one_d_characters: tuple[tuple[complex, complex, int], ...] = (),
) -> float:
    matrices = dicyclic_representation(m, two_d_counts, one_d_characters)
    a, x = matrices[1], matrices[2 * m]
    identity = np.eye(a.shape[0], dtype=complex)
    residuals = [
        np.linalg.norm(np.linalg.matrix_power(a, 2 * m) - identity),
        np.linalg.norm(x @ x - np.linalg.matrix_power(a, m)),
        np.linalg.norm(x @ a @ np.linalg.inv(x) - np.linalg.inv(a)),
    ]
    residuals.extend(np.linalg.norm(g.conj().T @ g - identity) for g in matrices)
    return float(max(residuals))


def dicyclic_single_formula(m: int, k: int, tau: tuple[int, ...]) -> int:
    q = 2 * m // gcd(2 * m, k)
    zero_weight_count = sum(
        (sum(tau) - 2 * sum(rows)) % q == 0
        for rows in product(*(range(degree + 1) for degree in tau))
    )
    coset = int(all(degree % 2 == 0 for degree in tau)) * (-1) ** (
        k * sum(tau) // 2
    )
    numerator = zero_weight_count + coset
    assert numerator % 2 == 0
    return numerator // 2


def _complete_homogeneous_from_eigenvalues(eigenvalues, degree: int) -> complex:
    coefficients = np.zeros(degree + 1, dtype=complex)
    coefficients[0] = 1
    for eigenvalue in eigenvalues:
        for power in range(1, degree + 1):
            coefficients[power] += eigenvalue * coefficients[power - 1]
    return coefficients[degree]


def dicyclic_spectral_formula(
    m: int,
    two_d_counts: dict[int, int],
    one_d_characters: tuple[tuple[complex, complex, int], ...],
    tau: tuple[int, ...],
) -> complex:
    """Coefficient extracted from the two displayed spectral sums."""

    zeta = np.exp(1j * np.pi / m)
    total = 0j
    for exponent in range(2 * m):
        cyclic_eigenvalues: list[complex] = []
        coset_eigenvalues: list[complex] = []
        for alpha, beta, count in one_d_characters:
            cyclic_eigenvalues.extend([alpha**exponent] * count)
            coset_eigenvalues.extend([beta * alpha**exponent] * count)
        for k, count in two_d_counts.items():
            for _ in range(count):
                cyclic_eigenvalues.extend([zeta ** (k * exponent), zeta ** (-k * exponent)])
                root = 1 if k % 2 == 0 else 1j
                coset_eigenvalues.extend([root, -root])
        total += prod(
            _complete_homogeneous_from_eigenvalues(cyclic_eigenvalues, degree)
            for degree in tau
        )
        total += prod(
            _complete_homogeneous_from_eigenvalues(coset_eigenvalues, degree)
            for degree in tau
        )
    return total / (4 * m)


def _dic_multiply(m: int, left: tuple[int, int], right: tuple[int, int]):
    e, j = left
    f, ell = right
    modulus = 2 * m
    if (e, f) == (0, 0):
        return 0, (j + ell) % modulus
    if (e, f) == (0, 1):
        return 1, (ell - j) % modulus
    if (e, f) == (1, 0):
        return 1, (j + ell) % modulus
    return 0, (m + ell - j) % modulus


def dicyclic_regular_representation(m: int) -> tuple[np.ndarray, ...]:
    labels = tuple((e, j) for e in (0, 1) for j in range(2 * m))
    indices = {label: index for index, label in enumerate(labels)}
    matrices = []
    for left in labels:
        matrix = np.zeros((4 * m, 4 * m), dtype=complex)
        for column, right in enumerate(labels):
            matrix[indices[_dic_multiply(m, left, right)], column] = 1
        matrices.append(matrix)
    return tuple(matrices)


def _totient(value: int) -> int:
    return sum(gcd(value, candidate) == 1 for candidate in range(1, value + 1))


def _cycle_complete_homogeneous(cycle_counts: dict[int, int], degree: int) -> int:
    coefficients = [0] * (degree + 1)
    coefficients[0] = 1
    for length, count in cycle_counts.items():
        updated = [0] * (degree + 1)
        for old_degree, old_value in enumerate(coefficients):
            for copies in range((degree - old_degree) // length + 1):
                updated[old_degree + copies * length] += old_value * comb(
                    count + copies - 1, copies
                )
        coefficients = updated
    return coefficients[degree]


def dicyclic_regular_formula(m: int, tau: tuple[int, ...]) -> int:
    cyclic = 0
    for order in range(1, 2 * m + 1):
        if 2 * m % order == 0:
            cyclic += _totient(order) * prod(
                _cycle_complete_homogeneous({order: 4 * m // order}, degree)
                for degree in tau
            )
    coset = 2 * m * prod(
        _cycle_complete_homogeneous({4: m}, degree) for degree in tau
    )
    return (cyclic + coset) // (4 * m)


def permutation_matrices(n: int, alternating: bool = False) -> tuple[np.ndarray, ...]:
    matrices = []
    for permutation in permutations(range(n)):
        inversions = sum(
            permutation[i] > permutation[j]
            for i in range(n)
            for j in range(i + 1, n)
        )
        if alternating and inversions % 2:
            continue
        matrix = np.zeros((n, n), dtype=complex)
        matrix[permutation, np.arange(n)] = 1
        matrices.append(matrix)
    return tuple(matrices)


def permutation_cycle_formula(n: int, tau: tuple[int, ...], alternating=False) -> int:
    total = 0.0
    for cycle_type in integer_partitions(n):
        if alternating and (n - len(cycle_type)) % 2:
            continue
        cycle_counts = Counter(cycle_type)
        weight = (2 if alternating else 1) / z_partition(cycle_type)
        total += weight * prod(
            _cycle_complete_homogeneous(cycle_counts, degree) for degree in tau
        )
    return int(round(total))


def vector_partition_counts(tau: tuple[int, ...], n: int) -> tuple[int, int]:
    """Return the S_n orbit count and A_n splitting correction."""

    vectors = tuple(
        vector
        for vector in product(*(range(value + 1) for value in tau))
        if any(vector)
    )
    target = tuple(tau)
    # dp[(sum-vector, number of nonzero parts)] = number of multisets.
    unrestricted = {(tuple(0 for _ in tau), 0): 1}
    distinct = {(tuple(0 for _ in tau), 0): 1}
    for vector in vectors:
        next_unrestricted = dict(unrestricted)
        for (subtotal, count), ways in unrestricted.items():
            max_copies = min(
                ((target[i] - subtotal[i]) // vector[i])
                for i in range(len(tau))
                if vector[i]
            )
            for copies in range(1, max_copies + 1):
                new_sum = tuple(subtotal[i] + copies * vector[i] for i in range(len(tau)))
                next_unrestricted[(new_sum, count + copies)] = (
                    next_unrestricted.get((new_sum, count + copies), 0) + ways
                )
        unrestricted = next_unrestricted

        next_distinct = dict(distinct)
        for (subtotal, count), ways in distinct.items():
            new_sum = tuple(subtotal[i] + vector[i] for i in range(len(tau)))
            if all(new_sum[i] <= target[i] for i in range(len(tau))):
                next_distinct[(new_sum, count + 1)] = (
                    next_distinct.get((new_sum, count + 1), 0) + ways
                )
        distinct = next_distinct

    symmetric = sum(
        unrestricted.get((target, count), 0) for count in range(n + 1)
    )
    correction = distinct.get((target, n), 0) + distinct.get((target, n - 1), 0)
    return symmetric, correction


def representative_sweeps() -> dict[str, list[dict[str, object]]]:
    """Small cases used both by the notebooks and the written reports."""

    results: dict[str, list[dict[str, object]]] = {
        "dicyclic": [],
        "symmetric": [],
        "alternating": [],
    }
    dicyclic_cases = [
        (2, 1, (2,)),
        (3, 1, (2, 2)),
        (4, 2, (3, 1)),
        (5, 3, (4,)),
    ]
    for m, k, tau in dicyclic_cases:
        matrices = dicyclic_representation(m, {k: 1})
        direct = reynolds_check(matrices, tau)
        formula = dicyclic_single_formula(m, k, tau)
        character = character_average(matrices, tau)
        results["dicyclic"].append(
            {
                "case": f"Dic_{m}, rho_{k}, tau={tau}",
                "target_dim": direct["target_dimension"],
                "direct": direct["projector_rank"],
                "character": int(round(character.real)),
                "formula": formula,
                "error": direct["idempotence_error"],
            }
        )

    for n, tau in [(2, (3,)), (3, (2, 1)), (4, (2, 2)), (5, (2, 1))]:
        matrices = permutation_matrices(n)
        direct = reynolds_check(matrices, tau)
        cycle = permutation_cycle_formula(n, tau)
        orbit, _ = vector_partition_counts(tau, n)
        results["symmetric"].append(
            {
                "case": f"S_{n}, tau={tau}",
                "target_dim": direct["target_dimension"],
                "direct": direct["projector_rank"],
                "cycle_formula": cycle,
                "orbit_formula": orbit,
                "error": direct["idempotence_error"],
            }
        )

    for n, tau in [
        (3, (2,)),
        (3, (3,)),
        (4, (2, 1)),
        (5, (2, 2)),
        (5, (3, 1)),
    ]:
        matrices = permutation_matrices(n, alternating=True)
        direct = reynolds_check(matrices, tau)
        cycle = permutation_cycle_formula(n, tau, alternating=True)
        symmetric, correction = vector_partition_counts(tau, n)
        results["alternating"].append(
            {
                "case": f"A_{n}, tau={tau}",
                "target_dim": direct["target_dimension"],
                "direct": direct["projector_rank"],
                "cycle_formula": cycle,
                "orbit_formula": symmetric + correction,
                "error": direct["idempotence_error"],
            }
        )
    return results


if __name__ == "__main__":
    sweeps = representative_sweeps()
    for group_name, rows in sweeps.items():
        print(group_name)
        for row in rows:
            print(row)
