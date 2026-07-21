"""Exact and numerical checks for U(n) acting on End(C^n) by conjugation.

The numerical route computes the common kernel of the derived u(n)-action on
tensor products of symmetric powers.  The combinatorial route evaluates the
finite-rank Littlewood--Richardson formula.  In stable rank, two further
independent routes count H_tau-conjugacy orbits and colored necklaces.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations, product
from math import comb, prod

import numpy as np


TOLERANCE = 2.0e-8


@lru_cache(maxsize=None)
def weak_compositions(total: int, length: int) -> tuple[tuple[int, ...], ...]:
    if total < 0 or length < 1:
        raise ValueError("total must be nonnegative and length must be positive")
    if length == 1:
        return ((total,),)
    return tuple(
        (first,) + rest
        for first in range(total + 1)
        for rest in weak_compositions(total - first, length - 1)
    )


@lru_cache(maxsize=None)
def partitions(total: int, max_length: int, ceiling: int | None = None):
    """Partitions of ``total`` in weakly decreasing order."""

    if total == 0:
        return ((),)
    if total < 0 or max_length < 1:
        return ()
    upper = min(total, total if ceiling is None else ceiling)
    result = []
    for first in range(upper, 0, -1):
        for rest in partitions(total - first, max_length - 1, first):
            result.append((first,) + rest)
    return tuple(result)


def _contains(outer: tuple[int, ...], inner: tuple[int, ...]) -> bool:
    return all(
        (inner[row] if row < len(inner) else 0)
        <= (outer[row] if row < len(outer) else 0)
        for row in range(max(len(outer), len(inner)))
    )


@lru_cache(maxsize=None)
def littlewood_richardson(
    lam: tuple[int, ...], mu: tuple[int, ...], nu: tuple[int, ...]
) -> int:
    """Count LR tableaux of shape nu/lam and content mu."""

    if sum(lam) + sum(mu) != sum(nu) or not _contains(nu, lam):
        return 0
    if not mu:
        return int(lam == nu)

    # Reading order: rows top-to-bottom and each row right-to-left.
    cells = tuple(
        (row, column)
        for row, width in enumerate(nu)
        for column in range(width - 1, (lam[row] if row < len(lam) else 0) - 1, -1)
    )
    remaining = list(mu)
    counts = [0] * len(mu)
    filling: dict[tuple[int, int], int] = {}
    answer = 0

    def visit(position: int) -> None:
        nonlocal answer
        if position == len(cells):
            answer += 1
            return
        row, column = cells[position]
        right = filling.get((row, column + 1))
        above = filling.get((row - 1, column))
        for value in range(1, len(mu) + 1):
            index = value - 1
            if remaining[index] == 0:
                continue
            # Rows weakly increase left-to-right; columns strictly increase.
            if right is not None and value > right:
                continue
            if above is not None and value <= above:
                continue
            counts[index] += 1
            lattice_ok = all(counts[k] >= counts[k + 1] for k in range(len(mu) - 1))
            if lattice_ok:
                remaining[index] -= 1
                filling[(row, column)] = value
                visit(position + 1)
                del filling[(row, column)]
                remaining[index] += 1
            counts[index] -= 1

    visit(0)
    return answer


def schur_product_coefficients(factors, max_length: int):
    coefficients = {(): 1}
    total = 0
    for factor in factors:
        total += sum(factor)
        updated: dict[tuple[int, ...], int] = {}
        for source, multiplicity in coefficients.items():
            for target in partitions(total, max_length):
                coefficient = littlewood_richardson(source, tuple(factor), target)
                if coefficient:
                    updated[target] = updated.get(target, 0) + multiplicity * coefficient
        coefficients = updated
    return coefficients


def finite_rank_lr_coefficient(n: int, tau: tuple[int, ...]) -> int:
    """Evaluate equation (6), including all finite-rank length cutoffs."""

    if n < 1 or any(degree < 1 for degree in tau):
        raise ValueError("n must be positive and tau must have positive parts")
    choices = tuple(partitions(degree, n) for degree in tau)
    total = 0
    for factors in product(*choices):
        coefficients = schur_product_coefficients(factors, n)
        total += sum(value * value for value in coefficients.values())
    return total


def u_n_lie_basis(n: int) -> tuple[np.ndarray, ...]:
    """A real basis of skew-Hermitian matrices for u(n)."""

    if n < 1:
        raise ValueError("n must be positive")
    basis = []
    for row in range(n):
        diagonal = np.zeros((n, n), dtype=complex)
        diagonal[row, row] = 1j
        basis.append(diagonal)
    for row in range(n):
        for column in range(row + 1, n):
            real_skew = np.zeros((n, n), dtype=complex)
            real_skew[row, column] = 1.0
            real_skew[column, row] = -1.0
            basis.append(real_skew)

            imaginary_symmetric = np.zeros((n, n), dtype=complex)
            imaginary_symmetric[row, column] = 1j
            imaginary_symmetric[column, row] = 1j
            basis.append(imaginary_symmetric)
    assert len(basis) == n * n
    return tuple(basis)


def unitary_generator_matrices(n: int, angle: float = 0.371) -> tuple[np.ndarray, ...]:
    """Exponentials of the standard u(n) basis, written without SciPy."""

    generators = []
    phase = np.exp(1j * angle)
    for row in range(n):
        matrix = np.eye(n, dtype=complex)
        matrix[row, row] = phase
        generators.append(matrix)
    cosine, sine = np.cos(angle), np.sin(angle)
    for row in range(n):
        for column in range(row + 1, n):
            first = np.eye(n, dtype=complex)
            first[row, row] = first[column, column] = cosine
            first[row, column] = first[column, row] = 1j * sine
            generators.append(first)

            second = np.eye(n, dtype=complex)
            second[row, row] = second[column, column] = cosine
            second[row, column] = sine
            second[column, row] = -sine
            generators.append(second)
    assert len(generators) == n * n
    return tuple(generators)


def conjugation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Matrix of A -> g A g^* in column-major vectorization."""

    return np.kron(matrix.conjugate(), matrix)


def conjugation_lie_matrix(lie_matrix: np.ndarray) -> np.ndarray:
    """Matrix of A -> XA-AX in column-major vectorization."""

    n = lie_matrix.shape[0]
    identity = np.eye(n, dtype=complex)
    return np.kron(identity, lie_matrix) - np.kron(lie_matrix.T, identity)


def symmetric_power_lie_matrix(matrix: np.ndarray, degree: int) -> np.ndarray:
    """Derived action on Sym^degree, in the monomial basis."""

    dimension = matrix.shape[0]
    basis = weak_compositions(degree, dimension)
    row_for = {exponent: index for index, exponent in enumerate(basis)}
    result = np.zeros((len(basis), len(basis)), dtype=complex)
    for column, exponent in enumerate(basis):
        for source in range(dimension):
            if exponent[source] == 0:
                continue
            for target in range(dimension):
                coefficient = exponent[source] * matrix[target, source]
                if abs(coefficient) < 1.0e-14:
                    continue
                updated = list(exponent)
                updated[source] -= 1
                updated[target] += 1
                result[row_for[tuple(updated)], column] += coefficient
    return result


def tensor_product_lie_matrix(matrix: np.ndarray, tau: tuple[int, ...]) -> np.ndarray:
    factors = tuple(symmetric_power_lie_matrix(matrix, degree) for degree in tau)
    dimensions = tuple(factor.shape[0] for factor in factors)
    target_dimension = prod(dimensions) if dimensions else 1
    result = np.zeros((target_dimension, target_dimension), dtype=complex)
    for selected, factor in enumerate(factors):
        term = np.ones((1, 1), dtype=complex)
        for index, dimension in enumerate(dimensions):
            term = np.kron(term, factor if index == selected else np.eye(dimension))
        result += term
    return result


def target_dimension(n: int, tau: tuple[int, ...]) -> int:
    end_dimension = n * n
    return prod(comb(end_dimension + degree - 1, degree) for degree in tau)


def direct_invariant_dimension(
    n: int, tau: tuple[int, ...], dimension_cap: int = 500
) -> dict[str, float | int]:
    """Compute invariants as the common kernel of the derived u(n)-action."""

    dimension = target_dimension(n, tau)
    if dimension > dimension_cap:
        raise ValueError(f"target dimension {dimension} exceeds safety cap {dimension_cap}")
    gram = np.zeros((dimension, dimension), dtype=complex)
    for generator in u_n_lie_basis(n):
        action = tensor_product_lie_matrix(conjugation_lie_matrix(generator), tau)
        gram += action.conjugate().T @ action
    eigenvalues = np.linalg.eigvalsh(gram)
    scale = max(1.0, float(eigenvalues[-1]))
    rank_threshold = TOLERANCE * scale
    rank = int(np.count_nonzero(eigenvalues > rank_threshold))
    smallest_positive = float(eigenvalues[dimension - rank]) if rank else 0.0
    largest_null = (
        float(eigenvalues[dimension - rank - 1]) if dimension - rank else 0.0
    )
    null_residual = max(0.0, abs(largest_null))
    gap_denominator = max(rank_threshold, null_residual, np.finfo(float).eps * scale)
    return {
        "n": n,
        "degree": sum(tau),
        "target_dimension": dimension,
        "lie_generators": n * n,
        "invariant_dimension": dimension - rank,
        "rank_threshold": rank_threshold,
        "smallest_positive_gram_eigenvalue": smallest_positive,
        "largest_null_gram_eigenvalue": largest_null,
        "spectral_gap_ratio": smallest_positive / gap_denominator if rank else float("inf"),
        "nullspace_residual": null_residual,
    }


def matrix_group_diagnostics(n: int) -> dict[str, float | int]:
    """Check unitary generators, Ad matrices, multiplication, and trace identity."""

    generators = unitary_generator_matrices(n)
    identity = np.eye(n, dtype=complex)
    unitary_error = max(np.linalg.norm(g.conjugate().T @ g - identity) for g in generators)
    g, h = generators[0], generators[-1]
    homomorphism_error = np.linalg.norm(
        conjugation_matrix(g @ h) - conjugation_matrix(g) @ conjugation_matrix(h)
    )
    trace_errors = []
    ad = conjugation_matrix(g @ h)
    for power in range(1, 5):
        lhs = np.trace(np.linalg.matrix_power(ad, power))
        rhs = abs(np.trace(np.linalg.matrix_power(g @ h, power))) ** 2
        trace_errors.append(abs(lhs - rhs))
    return {
        "n": n,
        "group_generators": len(generators),
        "ad_dimension": n * n,
        "unitarity_error": float(unitary_error),
        "homomorphism_error": float(homomorphism_error),
        "trace_identity_error": float(max(trace_errors)),
    }


def _block_permutations(tau: tuple[int, ...]):
    offsets = []
    start = 0
    for size in tau:
        offsets.append(tuple(range(start, start + size)))
        start += size
    for local_permutations in product(*(tuple(permutations(block)) for block in offsets)):
        result = list(range(start))
        for block, image in zip(offsets, local_permutations):
            for source, target in zip(block, image):
                result[source] = target
        yield tuple(result)


def _compose(left, right):
    return tuple(left[right[index]] for index in range(len(left)))


def _inverse(permutation):
    result = [0] * len(permutation)
    for source, target in enumerate(permutation):
        result[target] = source
    return tuple(result)


def stable_orbit_count(tau: tuple[int, ...]) -> int:
    """Count H_tau-conjugacy orbits in S_d directly."""

    degree = sum(tau)
    symmetric_group = tuple(permutations(range(degree)))
    subgroup = tuple(_block_permutations(tau))
    unseen = set(symmetric_group)
    orbit_count = 0
    while unseen:
        sigma = next(iter(unseen))
        orbit = {
            _compose(_compose(h, sigma), _inverse(h))
            for h in subgroup
        }
        unseen.difference_update(orbit)
        orbit_count += 1
    return orbit_count


def _canonical_rotation(word: tuple[int, ...]) -> tuple[int, ...]:
    return min(word[index:] + word[:index] for index in range(len(word)))


def necklace_euler_coefficient(tau: tuple[int, ...]) -> int:
    """Extract t^tau from the colored-necklace Euler product (9)."""

    colors = len(tau)
    degree = sum(tau)
    necklaces: set[tuple[int, ...]] = set()
    for length in range(1, degree + 1):
        for word in product(range(colors), repeat=length):
            weight = tuple(word.count(color) for color in range(colors))
            if all(weight[color] <= tau[color] for color in range(colors)):
                necklaces.add(_canonical_rotation(tuple(word)))

    weights = sorted({tuple(word.count(color) for color in range(colors)) for word in necklaces})
    # Keep multiplicity: distinct necklaces can have the same multidegree.
    weighted_necklaces = [
        tuple(word.count(color) for color in range(colors)) for word in sorted(necklaces)
    ]
    coefficients = {(0,) * colors: 1}
    for weight in weighted_necklaces:
        updated = dict(coefficients)
        for exponent, coefficient in coefficients.items():
            multiple = 1
            while True:
                target = tuple(exponent[color] + multiple * weight[color] for color in range(colors))
                if any(target[color] > tau[color] for color in range(colors)):
                    break
                updated[target] = updated.get(target, 0) + coefficient
                multiple += 1
        coefficients = updated
    assert weights  # Every nonempty tau has at least one admissible necklace.
    return coefficients.get(tuple(tau), 0)


def verify_case(n: int, tau: tuple[int, ...], dimension_cap: int = 500):
    direct = direct_invariant_dimension(n, tau, dimension_cap=dimension_cap)
    finite_formula = finite_rank_lr_coefficient(n, tau)
    result: dict[str, object] = {
        "n": n,
        "tau": tau,
        "degree": sum(tau),
        "target_dimension": direct["target_dimension"],
        "direct": direct["invariant_dimension"],
        "finite_lr": finite_formula,
        "finite_match": direct["invariant_dimension"] == finite_formula,
        "rank_threshold": direct["rank_threshold"],
        "largest_null_gram_eigenvalue": direct["largest_null_gram_eigenvalue"],
        "smallest_positive_gram_eigenvalue": direct["smallest_positive_gram_eigenvalue"],
        "spectral_gap_ratio": direct["spectral_gap_ratio"],
        "nullspace_residual": direct["nullspace_residual"],
    }
    if n >= sum(tau):
        orbit = stable_orbit_count(tau)
        necklace = necklace_euler_coefficient(tau)
        result.update(
            {
                "stable_orbits": orbit,
                "necklace_euler": necklace,
                "stable_match": finite_formula == orbit == necklace,
            }
        )
    else:
        result.update({"stable_orbits": None, "necklace_euler": None, "stable_match": None})
    return result


REPRESENTATIVE_CASES = (
    (1, (1,)),
    (1, (2,)),
    (1, (1, 1)),
    (1, (3,)),
    (1, (2, 1)),
    (2, (1,)),
    (2, (2,)),
    (2, (1, 1)),
    (2, (3,)),
    (2, (2, 1)),
    (2, (1, 1, 1)),
    (3, (1,)),
    (3, (2,)),
    (3, (1, 1)),
    (3, (3,)),
    (3, (2, 1)),
    (4, (1,)),
    (4, (2,)),
)


def run_representative_checks():
    rows = [verify_case(n, tau) for n, tau in REPRESENTATIVE_CASES]
    diagnostics = [matrix_group_diagnostics(n) for n in (1, 2, 3, 4)]
    if not all(row["finite_match"] for row in rows):
        raise AssertionError("a direct/LR comparison failed")
    if not all(row["stable_match"] in (None, True) for row in rows):
        raise AssertionError("a stable orbit/necklace comparison failed")
    if not all(
        max(
            row["unitarity_error"],
            row["homomorphism_error"],
            row["trace_identity_error"],
        )
        < 1.0e-10
        for row in diagnostics
    ):
        raise AssertionError("a matrix-group diagnostic failed")
    return rows, diagnostics


if __name__ == "__main__":
    verification_rows, group_diagnostics = run_representative_checks()
    for row in verification_rows:
        print(row)
    for row in group_diagnostics:
        print(row)
