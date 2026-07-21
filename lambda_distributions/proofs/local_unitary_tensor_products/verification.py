"""Direct and exact checks for balanced sigma-MGFs of local unitary groups.

The represented group is

    U(d_1) x ... x U(d_k) -> U(d_1 ... d_k),
    (g_1, ..., g_k) |-> g_1 kron ... kron g_k.

Exact combinatorics are kept independent of the numerical Haar experiment.
The latter literally constructs the tensor-product matrices and evaluates
their traces and complete homogeneous characters.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial, prod, sqrt
from typing import Iterable, Iterator

import numpy as np


Partition = tuple[int, ...]


@dataclass(frozen=True)
class HaarCase:
    label: str
    dimensions: tuple[int, ...]
    maximum_degree: int


DEFAULT_CASES = (
    HaarCase("global U(3)", (3,), 3),
    HaarCase("bipartite U(2) x U(3)", (2, 3), 2),
    HaarCase("bipartite boundary U(2) x U(2)", (2, 2), 3),
    HaarCase("tripartite U(2)^3", (2, 2, 2), 2),
)


def partitions(total: int, ceiling: int | None = None) -> Iterator[Partition]:
    if total == 0:
        yield ()
        return
    ceiling = min(total, total if ceiling is None else ceiling)
    for first in range(ceiling, 0, -1):
        for rest in partitions(total - first, first):
            yield (first,) + rest


def z_partition(partition: Partition) -> int:
    """Return z_lambda = product_i i^(m_i) m_i!."""
    value = 1
    for part in set(partition):
        multiplicity = partition.count(part)
        value *= part**multiplicity * factorial(multiplicity)
    return value


def hook_dimension(partition: Partition) -> int:
    """Dimension f^lambda of the symmetric-group Specht module."""
    size = sum(partition)
    hooks = 1
    for row, row_length in enumerate(partition):
        for column in range(row_length):
            below = sum(column < later_length for later_length in partition[row + 1 :])
            hooks *= row_length - column + below
    return factorial(size) // hooks


def unitary_tensor_centralizer_dimension(dimension: int, tensor_power: int) -> int:
    return sum(
        hook_dimension(shape) ** 2
        for shape in partitions(tensor_power)
        if len(shape) <= dimension
    )


def multilinear_coefficient(dimensions: tuple[int, ...], tensor_power: int) -> int:
    """Exact [m_(1^r) m_(1^r)] balanced coefficient."""
    return prod(
        unitary_tensor_centralizer_dimension(dimension, tensor_power)
        for dimension in dimensions
    )


def stable_power_moment(
    dimensions: tuple[int, ...], left: Partition, right: Partition
) -> int | None:
    """Stable E[p_left(rho) conjugate(p_right(rho))], or None out of range."""
    if sum(left) != sum(right):
        return 0
    degree = sum(left)
    if degree > min(dimensions):
        return None
    return z_partition(left) ** len(dimensions) if left == right else 0


def symmetric_coefficient(dimensions: tuple[int, ...], degree: int) -> int | None:
    """Exact where known here: global, bipartite, or stable local range."""
    factors = len(dimensions)
    if factors == 1:
        return 1
    if factors == 2:
        cutoff = min(dimensions)
        return sum(len(shape) <= cutoff for shape in partitions(degree))
    if degree <= min(dimensions):
        return sum(z_partition(shape) ** (factors - 2) for shape in partitions(degree))
    return None


def stable_symmetric_coefficient(number_of_factors: int, degree: int) -> int:
    return sum(
        z_partition(shape) ** (number_of_factors - 2)
        for shape in partitions(degree)
    )


def haar_unitary(dimension: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    q, r = np.linalg.qr(z)
    diagonal = np.diag(r)
    phases = np.where(np.abs(diagonal) > 0, diagonal / np.abs(diagonal), 1.0)
    return q * phases.conjugate()


def tensor_representation(matrices: Iterable[np.ndarray]) -> np.ndarray:
    result = np.array([[1.0 + 0.0j]])
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


def traces_of_powers(matrix: np.ndarray, maximum_degree: int) -> tuple[complex, ...]:
    traces: list[complex] = [complex(matrix.shape[0])]
    power = np.eye(matrix.shape[0], dtype=complex)
    for _ in range(maximum_degree):
        power = power @ matrix
        traces.append(complex(np.trace(power)))
    return tuple(traces)


def complete_characters(power_traces: tuple[complex, ...]) -> tuple[complex, ...]:
    """Newton recurrence: m h_m = sum_{j=1}^m p_j h_{m-j}."""
    values = [1.0 + 0.0j]
    for degree in range(1, len(power_traces)):
        values.append(
            sum(power_traces[j] * values[degree - j] for j in range(1, degree + 1))
            / degree
        )
    return tuple(values)


def _estimate(values: list[complex], expected: int) -> dict[str, object]:
    data = np.asarray(values, dtype=complex)
    mean = complex(data.mean())
    standard_error = float(np.sqrt(np.mean(np.abs(data - mean) ** 2) / len(data)))
    error = abs(mean - expected)
    tolerance = 6.0 * standard_error + 0.025 * (1.0 + abs(expected))
    return {
        "estimate": mean,
        "expected": expected,
        "standard error": standard_error,
        "absolute error": float(error),
        "tolerance": float(tolerance),
        "passed": bool(error <= tolerance),
    }


def construction_check(case: HaarCase, seed: int = 11) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    local = tuple(haar_unitary(dimension, rng) for dimension in case.dimensions)
    rho = tensor_representation(local)
    identity = np.eye(rho.shape[0], dtype=complex)
    unitarity_error = float(np.linalg.norm(rho.conj().T @ rho - identity, ord=np.inf))
    trace_errors = []
    for power in range(1, case.maximum_degree + 1):
        direct = np.trace(np.linalg.matrix_power(rho, power))
        factored = prod(np.trace(np.linalg.matrix_power(matrix, power)) for matrix in local)
        trace_errors.append(abs(direct - factored))
    maximum_trace_error = float(max(trace_errors, default=0.0))
    return {
        "label": case.label,
        "dimensions": case.dimensions,
        "matrix dimension": rho.shape[0],
        "unitarity error": unitarity_error,
        "trace-factorization error": maximum_trace_error,
        "passed": unitarity_error < 1e-11 and maximum_trace_error < 1e-11,
    }


def center_witness(dimensions: tuple[int, ...], degree: int) -> dict[str, object]:
    """Construct a central group element that excludes degree-d invariants."""
    z = np.exp(2j * np.pi / (degree + 1))
    local = [np.eye(dimension, dtype=complex) for dimension in dimensions]
    local[0] = z * local[0]
    rho = tensor_representation(local)
    scalar_error = float(
        np.linalg.norm(rho - z * np.eye(rho.shape[0], dtype=complex), ord=np.inf)
    )
    action_scalar = z**degree
    return {
        "dimensions": dimensions,
        "degree": degree,
        "rho scalar error": scalar_error,
        "Sym action scalar": action_scalar,
        "distance from 1": float(abs(action_scalar - 1)),
        "passed": scalar_error < 1e-12 and abs(action_scalar - 1) > 1e-6,
    }


def monte_carlo_case(
    case: HaarCase, samples: int = 6000, seed: int = 20260717
) -> dict[str, object]:
    """Build Haar tensor-product matrices and estimate selected coefficients."""
    rng = np.random.default_rng(seed)
    targets: dict[tuple[str, object], list[complex]] = {}

    ordinary_partitions = tuple(
        shape
        for total in range(1, min(case.maximum_degree, 2) + 1)
        for shape in partitions(total)
    )
    power_pairs: list[tuple[Partition, Partition]] = [((1,), (1,))]
    if case.maximum_degree >= 2 and min(case.dimensions) >= 2:
        power_pairs.extend(
            [((2,), (2,)), ((1, 1), (1, 1)), ((2,), (1, 1))]
        )
    symmetric_degrees = tuple(range(1, case.maximum_degree + 1))
    multilinear_degrees = tuple(range(1, min(case.maximum_degree, 2) + 1))

    for shape in ordinary_partitions:
        targets[("ordinary", shape)] = []
    for pair in power_pairs:
        targets[("power", pair)] = []
    for degree in symmetric_degrees:
        if symmetric_coefficient(case.dimensions, degree) is not None:
            targets[("symmetric", degree)] = []
    for degree in multilinear_degrees:
        targets[("multilinear", degree)] = []

    for _ in range(samples):
        local = tuple(haar_unitary(dimension, rng) for dimension in case.dimensions)
        rho = tensor_representation(local)
        traces = traces_of_powers(rho, case.maximum_degree)
        complete = complete_characters(traces)
        for shape in ordinary_partitions:
            targets[("ordinary", shape)].append(prod(complete[part] for part in shape))
        for left, right in power_pairs:
            left_value = prod(traces[part] for part in left)
            right_value = prod(traces[part] for part in right)
            targets[("power", (left, right))].append(left_value * right_value.conjugate())
        for degree in symmetric_degrees:
            if ("symmetric", degree) in targets:
                targets[("symmetric", degree)].append(abs(complete[degree]) ** 2)
        for degree in multilinear_degrees:
            targets[("multilinear", degree)].append(abs(traces[1]) ** (2 * degree))

    rows: list[dict[str, object]] = []
    for (sector, index), values in targets.items():
        if sector == "ordinary":
            expected = 0
        elif sector == "power":
            left, right = index  # type: ignore[misc]
            expected = stable_power_moment(case.dimensions, left, right)
            assert expected is not None
        elif sector == "symmetric":
            expected = symmetric_coefficient(case.dimensions, int(index))
            assert expected is not None
        else:
            expected = multilinear_coefficient(case.dimensions, int(index))
        row = _estimate(values, expected)
        row.update({"sector": sector, "index": index})
        rows.append(row)

    return {
        "label": case.label,
        "dimensions": case.dimensions,
        "samples": samples,
        "rows": tuple(rows),
        "passed": all(row["passed"] for row in rows),
    }


def exact_tables(maximum_degree: int = 5) -> dict[str, tuple[dict[str, object], ...]]:
    stable_rows = []
    for factors in (1, 2, 3):
        for degree in range(1, maximum_degree + 1):
            stable_rows.append(
                {
                    "k": factors,
                    "degree": degree,
                    "coefficient": stable_symmetric_coefficient(factors, degree),
                }
            )
    bipartite_rows = []
    for dimensions in ((2, 2), (2, 3), (3, 3)):
        for degree in range(1, maximum_degree + 1):
            bipartite_rows.append(
                {
                    "dimensions": dimensions,
                    "degree": degree,
                    "coefficient": symmetric_coefficient(dimensions, degree),
                }
            )
    multilinear_rows = []
    for dimensions in ((3,), (2, 2), (2, 3), (2, 2, 2)):
        for degree in range(1, min(maximum_degree, 4) + 1):
            value = multilinear_coefficient(dimensions, degree)
            stable = factorial(degree) ** len(dimensions)
            multilinear_rows.append(
                {
                    "dimensions": dimensions,
                    "degree": degree,
                    "exact": value,
                    "stable formula": stable,
                    "in stable range": degree <= min(dimensions),
                }
            )
    return {
        "stable symmetric": tuple(stable_rows),
        "bipartite": tuple(bipartite_rows),
        "multilinear": tuple(multilinear_rows),
    }


def run_all(samples: int = 6000, seed: int = 20260717) -> dict[str, object]:
    construction = tuple(construction_check(case, seed + i) for i, case in enumerate(DEFAULT_CASES))
    center = tuple(center_witness((2, 3), degree) for degree in range(1, 6))
    numerical = tuple(
        monte_carlo_case(case, samples=samples, seed=seed + 1009 * i)
        for i, case in enumerate(DEFAULT_CASES)
    )
    tables = exact_tables()
    passed = (
        all(row["passed"] for row in construction)
        and all(row["passed"] for row in center)
        and all(result["passed"] for result in numerical)
    )
    return {
        "construction": construction,
        "center": center,
        "numerical": numerical,
        "exact tables": tables,
        "passed": passed,
    }


if __name__ == "__main__":
    report = run_all()
    print("PASS" if report["passed"] else "FAIL")
    for result in report["numerical"]:
        failures = [row for row in result["rows"] if not row["passed"]]
        print(result["label"], "PASS" if not failures else f"FAIL ({len(failures)})")

