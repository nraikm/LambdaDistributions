"""Exact-enumeration checks for nonuniform measures on permutation matrices."""

from __future__ import annotations

from itertools import permutations
from math import comb, prod

import numpy as np

from lambda_distributions.proofs.nonuniform_lambda_distributions.common import (
    clean,
    cycle_counts,
    cycle_formula_coefficient,
    direct_matrix_coefficient,
    partitions_up_to,
    permutation_matrix,
    tensor_symmetric_character,
)

Permutation = tuple[int, ...]


def compose(left: Permutation, right: Permutation) -> Permutation:
    return tuple(left[right[index]] for index in range(len(left)))


def inverse(permutation: Permutation) -> Permutation:
    result = [0] * len(permutation)
    for source, target in enumerate(permutation):
        result[target] = source
    return tuple(result)


def parity(permutation: Permutation) -> int:
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def transposition(n: int, left: int, right: int) -> Permutation:
    result = list(range(n))
    result[left], result[right] = result[right], result[left]
    return tuple(result)


def distribution_convolution(
    labels: tuple[Permutation, ...], left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    positions = {label: index for index, label in enumerate(labels)}
    result = np.zeros(len(labels), dtype=float)
    for left_index, left_probability in enumerate(left):
        if left_probability == 0:
            continue
        for right_index, right_probability in enumerate(right):
            if right_probability:
                result[positions[compose(labels[left_index], labels[right_index])]] += (
                    left_probability * right_probability
                )
    return result


def distribution_power(labels: tuple[Permutation, ...], step: np.ndarray, steps: int) -> np.ndarray:
    result = np.zeros(len(labels), dtype=float)
    result[labels.index(tuple(range(len(labels[0]))))] = 1.0
    for _ in range(steps):
        result = distribution_convolution(labels, result, step)
    return result


def lazy_transposition_step(labels: tuple[Permutation, ...], laziness: float) -> np.ndarray:
    n = len(labels[0])
    positions = {label: index for index, label in enumerate(labels)}
    result = np.zeros(len(labels), dtype=float)
    result[positions[tuple(range(n))]] = laziness
    probability = (1 - laziness) / comb(n, 2)
    for left in range(n):
        for right in range(left + 1, n):
            result[positions[transposition(n, left, right)]] = probability
    return result


def lazy_k_cycle_step(
    labels: tuple[Permutation, ...], cycle_length: int, laziness: float
) -> np.ndarray:
    n = len(labels[0])
    positions = {label: index for index, label in enumerate(labels)}
    selected = [label for label in labels if cycle_counts(label) == {cycle_length: 1, 1: n - cycle_length}]
    result = np.zeros(len(labels), dtype=float)
    result[positions[tuple(range(n))]] = laziness
    for label in selected:
        result[positions[label]] = (1 - laziness) / len(selected)
    return result


def lazy_adjacent_step(labels: tuple[Permutation, ...], laziness: float) -> np.ndarray:
    n = len(labels[0])
    positions = {label: index for index, label in enumerate(labels)}
    result = np.zeros(len(labels), dtype=float)
    result[positions[tuple(range(n))]] = laziness
    for left in range(n - 1):
        result[positions[transposition(n, left, left + 1)]] = (1 - laziness) / (n - 1)
    return result


def riffle_distribution(labels: tuple[Permutation, ...], packets: int) -> np.ndarray:
    n = len(labels[0])
    probabilities = np.array(
        [comb(n + packets - sum(label[i] > label[i + 1] for i in range(n - 1)) - 1, n) / packets**n for label in labels],
        dtype=float,
    )
    total = float(probabilities.sum())
    if abs(total - 1.0) > 2e-12:
        raise ArithmeticError(f"inverse riffle formula is not normalized: {total}")
    return probabilities


def ewens_distribution(labels: tuple[Permutation, ...], theta: float) -> np.ndarray:
    weights = np.array([theta ** sum(cycle_counts(label).values()) for label in labels], dtype=float)
    return weights / weights.sum()


def conjugacy_weighted_distribution(labels: tuple[Permutation, ...], weights: dict[int, float]) -> np.ndarray:
    values = np.array(
        [prod(weights.get(length, 1.0) ** count for length, count in cycle_counts(label).items()) for label in labels],
        dtype=float,
    )
    return values / values.sum()


def direct_cycle_comparison(
    labels: tuple[Permutation, ...], probabilities: np.ndarray, tau: tuple[int, ...]
) -> tuple[complex, float]:
    matrices = tuple(permutation_matrix(label) for label in labels)
    direct = direct_matrix_coefficient(matrices, probabilities, tau)
    formula = float(
        sum(
            probability * cycle_formula_coefficient(label, tau)
            for label, probability in zip(labels, probabilities, strict=True)
        )
    )
    return direct, formula


def standard_s3_matrices(labels: tuple[Permutation, ...]) -> tuple[np.ndarray, ...]:
    spanning = np.column_stack((np.array([1.0, -1.0, 0.0]), np.array([1.0, 0.0, -1.0])))
    basis, _ = np.linalg.qr(spanning)
    return tuple(basis.conj().T @ permutation_matrix(label) @ basis for label in labels)


def s3_irreducibles(labels: tuple[Permutation, ...]) -> dict[str, tuple[np.ndarray, ...]]:
    return {
        "trivial": tuple(np.ones((1, 1), dtype=complex) for _ in labels),
        "sign": tuple(np.array([[parity(label)]], dtype=complex) for label in labels),
        "standard": standard_s3_matrices(labels),
    }


def multiplicities_in_tensor_symmetric(
    permutation_matrices: tuple[np.ndarray, ...],
    irreducibles: dict[str, tuple[np.ndarray, ...]],
    tau: tuple[int, ...],
) -> dict[str, int]:
    character = np.array([tensor_symmetric_character(matrix, tau) for matrix in permutation_matrices])
    result = {}
    for name, matrices in irreducibles.items():
        irreducible_character = np.array([np.trace(matrix) for matrix in matrices])
        value = np.mean(character * irreducible_character.conj())
        rounded = int(round(float(value.real)))
        if abs(value.imag) > 2e-9 or abs(value.real - rounded) > 2e-9:
            raise ArithmeticError(f"nonintegral multiplicity {name}, {tau}: {value}")
        result[name] = rounded
    return result


def fourier_operator_coefficient(
    probabilities: np.ndarray,
    irreducibles: dict[str, tuple[np.ndarray, ...]],
    multiplicities: dict[str, int],
) -> complex:
    total = 0j
    for name, matrices in irreducibles.items():
        average = sum(
            probability * matrix
            for probability, matrix in zip(probabilities, matrices, strict=True)
        )
        total += multiplicities[name] * np.trace(average)
    return total


def run_suite(maximum_degree: int = 4, tolerance: float = 3e-9) -> dict:
    cycle_rows: list[dict] = []
    measure_builders = (
        ("lazy transpositions, r=3", lambda labels: distribution_power(labels, lazy_transposition_step(labels, 0.35), 3)),
        ("lazy 3-cycles, r=2", lambda labels: distribution_power(labels, lazy_k_cycle_step(labels, 3, 0.4), 2)),
        ("lazy adjacent, r=4", lambda labels: distribution_power(labels, lazy_adjacent_step(labels, 0.25), 4)),
        ("inverse 2-riffle", lambda labels: riffle_distribution(labels, 2)),
        ("Ewens theta=1.7", lambda labels: ewens_distribution(labels, 1.7)),
        ("cycle weights", lambda labels: conjugacy_weighted_distribution(labels, {1: 1.2, 2: 0.7, 3: 1.6})),
    )
    for n in (3, 4, 5):
        labels = tuple(permutations(range(n)))
        for measure_name, builder in measure_builders:
            probabilities = builder(labels)
            for tau in partitions_up_to(maximum_degree):
                direct, formula = direct_cycle_comparison(labels, probabilities, tau)
                error = abs(direct - formula)
                cycle_rows.append(
                    {
                        "group": f"S{n}",
                        "dimension": n,
                        "measure": measure_name,
                        "tau": str(tau),
                        "matrix average": clean(direct),
                        "cycle formula": formula,
                        "error": float(error),
                        "pass": bool(error < tolerance),
                    }
                )

    labels = tuple(permutations(range(3)))
    permutation_matrices = tuple(permutation_matrix(label) for label in labels)
    irreducibles = s3_irreducibles(labels)
    fourier_measures = (
        ("arbitrary", np.array([1, 2, 3, 4, 5, 6], dtype=float) / 21),
        ("lazy transpositions, r=5", distribution_power(labels, lazy_transposition_step(labels, 0.35), 5)),
        ("lazy adjacent, r=5", distribution_power(labels, lazy_adjacent_step(labels, 0.35), 5)),
        ("inverse 2-riffle", riffle_distribution(labels, 2)),
    )
    fourier_rows: list[dict] = []
    for tau in partitions_up_to(maximum_degree):
        multiplicities = multiplicities_in_tensor_symmetric(permutation_matrices, irreducibles, tau)
        for measure_name, probabilities in fourier_measures:
            direct = direct_matrix_coefficient(permutation_matrices, probabilities, tau)
            formula = fourier_operator_coefficient(probabilities, irreducibles, multiplicities)
            error = abs(direct - formula)
            fourier_rows.append(
                {
                    "group": "S3",
                    "measure": measure_name,
                    "tau": str(tau),
                    "multiplicities (1,sgn,std)": str(tuple(multiplicities[name] for name in ("trivial", "sign", "standard"))),
                    "direct": clean(direct),
                    "Fourier operator": clean(formula),
                    "error": float(error),
                    "pass": bool(error < tolerance),
                }
            )

    # Independent operator diagnostics for the central/noncentral distinction.
    central = lazy_transposition_step(labels, 0.35)
    adjacent = lazy_adjacent_step(labels, 0.35)
    operator_rows = []
    for name, matrices in irreducibles.items():
        dimension = matrices[0].shape[0]
        central_operator = sum(p * m for p, m in zip(central, matrices, strict=True))
        transposition_index = next(i for i, label in enumerate(labels) if cycle_counts(label) == {2: 1, 1: 1})
        ratio = np.trace(matrices[transposition_index]) / dimension
        beta = 0.35 + 0.65 * ratio
        scalar_error = np.linalg.norm(central_operator - beta * np.eye(dimension))
        adjacent_operator = sum(p * m for p, m in zip(adjacent, matrices, strict=True))
        operator_rows.append(
            {
                "irrep": name,
                "dimension": dimension,
                "central beta": clean(complex(beta)),
                "central scalar error": float(scalar_error),
                "adjacent eigenvalues": str(tuple(np.round(np.linalg.eigvals(adjacent_operator), 10))),
                "pass": bool(scalar_error < tolerance),
            }
        )

    all_rows = cycle_rows + fourier_rows
    return {
        "cycle_rows": cycle_rows,
        "fourier_rows": fourier_rows,
        "operator_rows": operator_rows,
        "checks": len(all_rows) + len(operator_rows),
        "maximum_error": max(
            max(row["error"] for row in all_rows),
            max(row["central scalar error"] for row in operator_rows),
        ),
        "passed": all(row["pass"] for row in all_rows + operator_rows),
    }


if __name__ == "__main__":
    result = run_suite()
    print(
        f"{'PASS' if result['passed'] else 'FAIL'} symmetric groups: "
        f"checks={result['checks']}, maximum_error={result['maximum_error']:.3e}"
    )
    raise SystemExit(0 if result["passed"] else 1)
