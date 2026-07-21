"""Direct matrix checks of nonuniform averaging and convolution on C_m."""

from __future__ import annotations

from math import prod

import numpy as np

from for_this_guy.nonuniform_lambda_distributions.common import (
    clean,
    direct_matrix_coefficient,
    partitions_up_to,
    tensor_weight_multiplicities,
)


def cyclic_matrices(order: int, weights: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    root = np.exp(2j * np.pi / order)
    return tuple(
        np.diag([root ** (element * weight) for weight in weights]).astype(complex)
        for element in range(order)
    )


def cyclic_convolution(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    order = len(left)
    return np.array(
        [sum(left[x] * right[(target - x) % order] for x in range(order)) for target in range(order)]
    )


def convolution_power(step: np.ndarray, steps: int) -> np.ndarray:
    result = np.zeros_like(step)
    result[0] = 1.0
    for _ in range(steps):
        result = cyclic_convolution(result, step)
    return result


def fourier_weight_coefficient(
    order: int,
    representation_weights: tuple[int, ...],
    tau: tuple[int, ...],
    probabilities: np.ndarray,
) -> complex:
    multiplicities = tensor_weight_multiplicities(representation_weights, tau, order)
    root = np.exp(2j * np.pi / order)
    fourier = {
        weight: sum(probabilities[element] * root ** (element * weight) for element in range(order))
        for weight in multiplicities
    }
    return sum(multiplicity * fourier[weight] for weight, multiplicity in multiplicities.items())


def convolution_fourier_coefficient(
    order: int,
    representation_weights: tuple[int, ...],
    tau: tuple[int, ...],
    step: np.ndarray,
    steps: int,
) -> complex:
    multiplicities = tensor_weight_multiplicities(representation_weights, tau, order)
    root = np.exp(2j * np.pi / order)
    return sum(
        multiplicity
        * sum(step[element] * root ** (element * weight) for element in range(order)) ** steps
        for weight, multiplicity in multiplicities.items()
    )


def direct_sigma_value(
    matrices: tuple[np.ndarray, ...], probabilities: np.ndarray, variables: tuple[float, ...]
) -> complex:
    identity = np.eye(matrices[0].shape[0], dtype=complex)
    return sum(
        probability
        * prod(1.0 / np.linalg.det(identity - variable * matrix) for variable in variables)
        for matrix, probability in zip(matrices, probabilities, strict=True)
    )


CASES = (
    (3, (1,)),
    (4, (1, -1)),
    (5, (0, 1, 2)),
)


def run_suite(maximum_degree: int = 5, tolerance: float = 2e-9) -> dict:
    all_rows: list[dict] = []
    sigma_rows: list[dict] = []
    for order, weights in CASES:
        matrices = cyclic_matrices(order, weights)
        arbitrary = np.arange(1, order + 1, dtype=float)
        arbitrary /= arbitrary.sum()
        step = np.zeros(order, dtype=float)
        step[0] = 0.35
        step[1] = 0.40
        step[-1] += 0.25

        for tau in partitions_up_to(maximum_degree):
            direct = direct_matrix_coefficient(matrices, arbitrary, tau)
            formula = fourier_weight_coefficient(order, weights, tau, arbitrary)
            error = abs(direct - formula)
            all_rows.append(
                {
                    "group": f"C{order}",
                    "dimension": len(weights),
                    "measure": "arbitrary",
                    "tau": str(tau),
                    "direct": clean(direct),
                    "formula": clean(formula),
                    "error": float(error),
                    "pass": bool(error < tolerance),
                }
            )

        for steps in (0, 1, 2, 5):
            probabilities = convolution_power(step, steps)
            for tau in partitions_up_to(maximum_degree):
                direct = direct_matrix_coefficient(matrices, probabilities, tau)
                formula = convolution_fourier_coefficient(order, weights, tau, step, steps)
                error = abs(direct - formula)
                all_rows.append(
                    {
                        "group": f"C{order}",
                        "dimension": len(weights),
                        "measure": f"walk r={steps}",
                        "tau": str(tau),
                        "direct": clean(direct),
                        "formula": clean(formula),
                        "error": float(error),
                        "pass": bool(error < tolerance),
                    }
                )

        variables = (0.07, -0.04)
        determinant_value = direct_sigma_value(matrices, arbitrary, variables)
        spectral_value = sum(
            arbitrary[element]
            * prod(
                1.0
                / prod(
                    1 - variable * np.exp(2j * np.pi * element * weight / order)
                    for weight in weights
                )
                for variable in variables
            )
            for element in range(order)
        )
        sigma_rows.append(
            {
                "group": f"C{order}",
                "dimension": len(weights),
                "variables": str(variables),
                "direct determinant": clean(determinant_value),
                "spectral product": clean(spectral_value),
                "error": float(abs(determinant_value - spectral_value)),
                "pass": bool(abs(determinant_value - spectral_value) < tolerance),
            }
        )

    return {
        "rows": all_rows,
        "sigma_rows": sigma_rows,
        "checks": len(all_rows) + len(sigma_rows),
        "maximum_error": max(row["error"] for row in all_rows + sigma_rows),
        "passed": all(row["pass"] for row in all_rows + sigma_rows),
    }


if __name__ == "__main__":
    result = run_suite()
    print(
        f"{'PASS' if result['passed'] else 'FAIL'} cyclic groups: "
        f"checks={result['checks']}, maximum_error={result['maximum_error']:.3e}"
    )
    raise SystemExit(0 if result["passed"] else 1)

