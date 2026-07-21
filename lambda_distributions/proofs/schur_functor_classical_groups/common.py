"""Shared numerical checks for Schur functors of U(n), O(n), and USp(2n).

The exact checks deliberately use two independent routes:

* construct a Schur-functor matrix with a Young symmetrizer in ``W**tensor d``;
* evaluate the plethystic/Jacobi--Trudi formula using traces of powers of the
  defining matrix only.

The Haar checks are statistical diagnostics.  They are not used to turn an
asymptotic assertion into an exact finite-dimensional claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product
from math import prod, sqrt
from typing import Callable, Iterable

import numpy as np


Partition = tuple[int, ...]


def _validate_partition(partition: Partition) -> None:
    if any(part <= 0 for part in partition):
        raise ValueError("partition parts must be positive")
    if any(a < b for a, b in zip(partition, partition[1:])):
        raise ValueError("partition must be weakly decreasing")


def schur_dimension(partition: Partition, dimension: int) -> int:
    """Hook-content formula for dim S_partition(C^dimension)."""

    _validate_partition(partition)
    if len(partition) > dimension:
        return 0
    numerator = 1
    denominator = 1
    conjugate = tuple(sum(part >= column for part in partition) for column in range(1, partition[0] + 1))
    for row, length in enumerate(partition, start=1):
        for column in range(1, length + 1):
            numerator *= dimension + column - row
            denominator *= length - column + conjugate[column - 1] - row + 1
    return numerator // denominator


def _block_permutations(blocks: tuple[tuple[int, ...], ...], degree: int):
    choices = [tuple(permutations(block)) for block in blocks]
    for selected in product(*choices):
        permutation = list(range(degree))
        sign = 1
        for block, rearranged in zip(blocks, selected):
            for position, image in zip(block, rearranged):
                permutation[position] = image
            order = {value: index for index, value in enumerate(block)}
            encoded = [order[value] for value in rearranged]
            inversions = sum(encoded[i] > encoded[j] for i in range(len(encoded)) for j in range(i + 1, len(encoded)))
            sign *= -1 if inversions % 2 else 1
        yield tuple(permutation), sign


@lru_cache(maxsize=None)
def schur_basis(partition: Partition, dimension: int) -> np.ndarray:
    """Orthonormal basis of one Young-symmetrizer copy of S_lambda(C^n)."""

    _validate_partition(partition)
    degree = sum(partition)
    if len(partition) > dimension:
        return np.zeros((dimension**degree, 0), dtype=complex)

    rows: list[tuple[int, ...]] = []
    columns: list[list[int]] = [[] for _ in range(partition[0])]
    label = 0
    for length in partition:
        row = tuple(range(label, label + length))
        rows.append(row)
        for column, box in enumerate(row):
            columns[column].append(box)
        label += length
    row_group = tuple(_block_permutations(tuple(rows), degree))
    column_group = tuple(_block_permutations(tuple(tuple(column) for column in columns), degree))

    symmetrizer = np.zeros((dimension**degree, dimension**degree), dtype=complex)
    for source_flat in range(dimension**degree):
        source = np.unravel_index(source_flat, (dimension,) * degree)
        for row_permutation, _ in row_group:
            for column_permutation, column_sign in column_group:
                combined = tuple(column_permutation[row_permutation[index]] for index in range(degree))
                target = tuple(source[combined[index]] for index in range(degree))
                target_flat = np.ravel_multi_index(target, (dimension,) * degree)
                symmetrizer[target_flat, source_flat] += column_sign

    u, singular_values, _ = np.linalg.svd(symmetrizer, full_matrices=False)
    rank = int(np.sum(singular_values > 1e-9))
    expected = schur_dimension(partition, dimension)
    if rank != expected:
        raise AssertionError(f"Young symmetrizer rank {rank}, expected {expected}")
    return np.asarray(u[:, :rank], dtype=complex)


def tensor_power(matrix: np.ndarray, degree: int) -> np.ndarray:
    answer = np.ones((1, 1), dtype=complex)
    for _ in range(degree):
        answer = np.kron(answer, matrix)
    return answer


def schur_matrix(matrix: np.ndarray, partition: Partition) -> np.ndarray:
    """Construct S_lambda(matrix) by compression to a Young-symmetrizer image."""

    basis = schur_basis(partition, matrix.shape[0])
    return basis.conjugate().T @ tensor_power(np.asarray(matrix, dtype=complex), sum(partition)) @ basis


def complete_from_power_sums(power_sums: dict[int, complex], maximum_degree: int) -> list[complex]:
    """Newton recurrence: m h_m = sum_{k=1}^m p_k h_{m-k}."""

    complete = [1.0 + 0.0j]
    for degree in range(1, maximum_degree + 1):
        complete.append(sum(power_sums[k] * complete[degree - k] for k in range(1, degree + 1)) / degree)
    return complete


def schur_from_power_sums(partition: Partition, power_sums: dict[int, complex]) -> complex:
    """Evaluate s_lambda by Jacobi--Trudi from the supplied p_k values."""

    _validate_partition(partition)
    length = len(partition)
    maximum = partition[0] + length - 1
    complete = complete_from_power_sums(power_sums, maximum)
    jacobi_trudi = np.empty((length, length), dtype=complex)
    for row in range(length):
        for column in range(length):
            index = partition[row] - row + column
            jacobi_trudi[row, column] = 0 if index < 0 else complete[index]
    return complex(np.linalg.det(jacobi_trudi))


def trace_formula(matrix: np.ndarray, partition: Partition, power: int) -> complex:
    """The proposed formula p_power[s_lambda][X] = s_lambda[X**power]."""

    maximum = partition[0] + len(partition) - 1
    power_sums = {k: complex(np.trace(np.linalg.matrix_power(matrix, power * k))) for k in range(1, maximum + 1)}
    return schur_from_power_sums(partition, power_sums)


def complete_character_from_traces(traces: dict[int, complex], degree: int) -> complex:
    return complete_from_power_sums(traces, degree)[degree]


def sigma_integrand_direct(representation: np.ndarray, tau: Partition) -> complex:
    if not tau:
        return 1.0 + 0.0j
    maximum = max(tau)
    traces = {power: complex(np.trace(np.linalg.matrix_power(representation, power))) for power in range(1, maximum + 1)}
    complete = complete_from_power_sums(traces, maximum)
    return complex(prod(complete[part] for part in tau))


def sigma_integrand_formula(matrix: np.ndarray, partition: Partition, tau: Partition) -> complex:
    if not tau:
        return 1.0 + 0.0j
    maximum = max(tau)
    schur_traces = {power: trace_formula(matrix, partition, power) for power in range(1, maximum + 1)}
    complete = complete_from_power_sums(schur_traces, maximum)
    return complex(prod(complete[part] for part in tau))


def haar_unitary(dimension: int, rng: np.random.Generator) -> np.ndarray:
    sample = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(sample)
    diagonal = np.diag(r)
    phases = np.divide(diagonal, np.abs(diagonal), out=np.ones_like(diagonal), where=np.abs(diagonal) > 0)
    return np.asarray(q * phases.conjugate(), dtype=complex)


def haar_orthogonal(dimension: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.where(np.diag(r) >= 0, 1.0, -1.0)
    return np.asarray(q * signs, dtype=float)


def haar_compact_symplectic(rank: int, rng: np.random.Generator) -> np.ndarray:
    """Quaternionic Gram--Schmidt sampler for the compact group USp(2*rank)."""

    size = 2 * rank
    identity = np.eye(rank)
    zero = np.zeros((rank, rank))
    form = np.block([[zero, identity], [-identity, zero]]).astype(complex)
    first_columns: list[np.ndarray] = []
    paired_columns: list[np.ndarray] = []
    for _ in range(rank):
        vector = rng.normal(size=size) + 1j * rng.normal(size=size)
        for column in first_columns + paired_columns:
            vector -= column * np.vdot(column, vector)
        vector /= np.linalg.norm(vector)
        pair = -form @ vector.conjugate()
        first_columns.append(vector)
        paired_columns.append(pair)
    return np.column_stack(first_columns + paired_columns)


def group_residual(matrix: np.ndarray, group: str) -> float:
    size = matrix.shape[0]
    if group in {"unitary", "orthogonal"}:
        return float(np.linalg.norm(matrix.conjugate().T @ matrix - np.eye(size), ord=np.inf))
    rank = size // 2
    identity = np.eye(rank)
    zero = np.zeros((rank, rank))
    form = np.block([[zero, identity], [-identity, zero]])
    return float(max(
        np.linalg.norm(matrix.conjugate().T @ matrix - np.eye(size), ord=np.inf),
        np.linalg.norm(matrix.T @ form @ matrix - form, ord=np.inf),
    ))


@dataclass(frozen=True)
class ExactCase:
    dimension: int
    partition: Partition
    powers: tuple[int, ...] = (1, 2, 3)
    taus: tuple[Partition, ...] = ((), (1,), (2,), (1, 1), (3,))


def sampler_for(group: str) -> Callable[[int, np.random.Generator], np.ndarray]:
    if group == "unitary":
        return haar_unitary
    if group == "orthogonal":
        return haar_orthogonal
    if group == "symplectic":
        return lambda dimension, rng: haar_compact_symplectic(dimension // 2, rng)
    raise ValueError(f"unknown group {group!r}")


def exact_rows(group: str, cases: Iterable[ExactCase], seed: int = 20260717) -> tuple[dict[str, object], ...]:
    rng = np.random.default_rng(seed)
    sampler = sampler_for(group)
    rows: list[dict[str, object]] = []
    for case in cases:
        matrix = sampler(case.dimension, rng)
        representation = schur_matrix(matrix, case.partition)
        residual = group_residual(matrix, group)
        representation_residual = float(
            np.linalg.norm(
                representation.conjugate().T @ representation
                - np.eye(representation.shape[0]),
                ord=np.inf,
            )
        )
        for power in case.powers:
            direct = complex(np.trace(np.linalg.matrix_power(representation, power)))
            formula = trace_formula(matrix, case.partition, power)
            error = abs(direct - formula)
            rows.append({
                "kind": "trace", "group": group, "n": case.dimension,
                "lambda": str(case.partition), "test": f"r={power}",
                "direct": direct, "formula": formula, "error": error,
                "group residual": residual, "representation residual": representation_residual,
                "pass": error < 2e-9 and residual < 2e-10 and representation_residual < 2e-9,
            })
        for tau in case.taus:
            direct = sigma_integrand_direct(representation, tau)
            formula = sigma_integrand_formula(matrix, case.partition, tau)
            error = abs(direct - formula)
            rows.append({
                "kind": "sigma integrand", "group": group, "n": case.dimension,
                "lambda": str(case.partition), "test": f"tau={tau}",
                "direct": direct, "formula": formula, "error": error,
                "group residual": residual, "representation residual": representation_residual,
                "pass": error < 5e-8 and residual < 2e-10 and representation_residual < 2e-9,
            })
    return tuple(rows)


def character(matrix: np.ndarray, partition: Partition) -> complex:
    return trace_formula(matrix, partition, 1)


def structural_moment_rows(group: str, dimension: int, samples: int = 1200, seed: int = 314159) -> tuple[dict[str, object], ...]:
    """Haar diagnostics for central-character/parity consequences and means."""

    rng = np.random.default_rng(seed)
    sampler = sampler_for(group)
    if group == "unitary":
        specifications = (((2,), 1, 0.0), ((2,), 2, 0.0), ((1, 1), 1, 0.0))
    elif group == "orthogonal":
        specifications = (((2,), 1, 1.0), ((1, 1), 1, 0.0), ((3,), 1, 0.0))
    else:
        specifications = (((2,), 1, 0.0), ((1, 1), 1, 1.0), ((3,), 1, 0.0))
    values = {spec[:2]: [] for spec in specifications}
    worst_residual = 0.0
    for _ in range(samples):
        matrix = sampler(dimension, rng)
        worst_residual = max(worst_residual, group_residual(matrix, group))
        cached = {partition: character(matrix, partition) for partition, _, _ in specifications}
        for partition, exponent, _ in specifications:
            values[(partition, exponent)].append(cached[partition] ** exponent)
    rows = []
    for partition, exponent, target in specifications:
        data = np.asarray(values[(partition, exponent)], dtype=complex)
        estimate = complex(np.mean(data))
        standard_error = float(np.std(data, ddof=1) / sqrt(samples))
        tolerance = max(0.12, 4.5 * standard_error)
        rows.append({
            "group": group, "n": dimension, "lambda": str(partition),
            "moment": exponent, "samples": samples, "estimate": estimate,
            "target": target, "standard error": standard_error,
            "tolerance": tolerance, "group residual": worst_residual,
            "pass": abs(estimate - target) <= tolerance and worst_residual < 5e-10,
        })
    return tuple(rows)


def run_group_suite(group: str, cases: tuple[ExactCase, ...], moment_dimension: int) -> dict[str, object]:
    exact = exact_rows(group, cases)
    moments = structural_moment_rows(group, moment_dimension)
    return {
        "group": group,
        "exact rows": exact,
        "moment rows": moments,
        "exact checks": len(exact),
        "moment checks": len(moments),
        "maximum exact error": max(float(row["error"]) for row in exact),
        "passed": all(bool(row["pass"]) for row in exact + moments),
    }
