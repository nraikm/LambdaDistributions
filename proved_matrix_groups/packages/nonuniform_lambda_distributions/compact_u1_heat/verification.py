"""Quadrature checks of the U(1) heat-kernel coefficient formula."""

from __future__ import annotations

from math import prod

import numpy as np

from for_this_guy.nonuniform_lambda_distributions.common import (
    clean,
    complete_characters,
    partitions_up_to,
    tensor_weight_multiplicities,
)


CASES = (
    ("defining", (1,)),
    ("weight pair", (1, -1)),
    ("mixed weights", (0, 1, -2)),
)


def heat_kernel(theta: np.ndarray, time: float, cutoff: int) -> np.ndarray:
    values = np.ones_like(theta, dtype=complex)
    for frequency in range(1, cutoff + 1):
        values += 2 * np.exp(-0.5 * frequency * frequency * time) * np.cos(frequency * theta)
    return values.real


def direct_quadrature_coefficient(
    weights: tuple[int, ...], tau: tuple[int, ...], time: float, grid_size: int, cutoff: int
) -> complex:
    if time == 0:
        return complex(prod(
            complete_characters(np.eye(len(weights), dtype=complex), max(tau, default=0))[degree]
            for degree in tau
        ))
    theta = 2 * np.pi * np.arange(grid_size) / grid_size
    density = heat_kernel(theta, time, cutoff)
    character = np.empty(grid_size, dtype=complex)
    for index, angle in enumerate(theta):
        matrix = np.diag([np.exp(1j * weight * angle) for weight in weights])
        h = complete_characters(matrix, max(tau, default=0))
        character[index] = prod(h[degree] for degree in tau)
    return complex(np.mean(density * character))


def casimir_formula_coefficient(weights: tuple[int, ...], tau: tuple[int, ...], time: float) -> float:
    multiplicities = tensor_weight_multiplicities(weights, tau)
    return float(
        sum(multiplicity * np.exp(-0.5 * weight * weight * time) for weight, multiplicity in multiplicities.items())
    )


def run_suite(
    maximum_degree: int = 5,
    grid_size: int = 512,
    cutoff: int = 32,
    tolerance: float = 3e-9,
) -> dict:
    rows: list[dict] = []
    for name, weights in CASES:
        for time in (0.0, 0.25, 0.8, 3.0, 20.0):
            for tau in partitions_up_to(maximum_degree):
                direct = direct_quadrature_coefficient(weights, tau, time, grid_size, cutoff)
                formula = casimir_formula_coefficient(weights, tau, time)
                error = abs(direct - formula)
                rows.append(
                    {
                        "representation": name,
                        "weights": str(weights),
                        "dimension": len(weights),
                        "time": time,
                        "tau": str(tau),
                        "quadrature": clean(direct),
                        "Casimir formula": formula,
                        "error": float(error),
                        "pass": bool(error < tolerance),
                    }
                )
    return {
        "rows": rows,
        "checks": len(rows),
        "maximum_error": max(row["error"] for row in rows),
        "passed": all(row["pass"] for row in rows),
        "grid_size": grid_size,
        "cutoff": cutoff,
    }


if __name__ == "__main__":
    result = run_suite()
    print(
        f"{'PASS' if result['passed'] else 'FAIL'} U(1) heat: "
        f"checks={result['checks']}, maximum_error={result['maximum_error']:.3e}, "
        f"grid={result['grid_size']}, cutoff={result['cutoff']}"
    )
    raise SystemExit(0 if result["passed"] else 1)
