"""Exact tests of the Johnson-scheme fixed-subset formulas (9)--(15)."""

from __future__ import annotations

from collections import Counter
from itertools import combinations, permutations

from ..shared import (
    cycle_counts,
    fixed_formula_reconstructs,
    numerical_determinant_check,
    permutation_power,
    verify_action,
)


def _subset_fixed_count(base_permutation, k: int, exponent: int) -> int:
    """Equation (9), evaluated by coefficient extraction."""

    powered = permutation_power(base_permutation, exponent)
    coefficients = [0] * (k + 1)
    coefficients[0] = 1
    for length, multiplicity in cycle_counts(powered).items():
        for _ in range(multiplicity):
            for degree in range(k, length - 1, -1):
                coefficients[degree] += coefficients[degree - length]
    return coefficients[k]


def _induced(base_permutation, v: int, k: int, complement: bool = False):
    points = tuple(combinations(range(v), k))
    position = {point: index for index, point in enumerate(points)}
    universe = set(range(v))
    result = []
    for point in points:
        moved = {base_permutation[x] for x in point}
        if complement:
            moved = universe - moved
        result.append(position[tuple(sorted(moved))])
    return tuple(result)


def johnson_action(v: int, k: int, full_middle_automorphism: bool = True):
    elements = []
    for sigma in permutations(range(v)):
        sigma = tuple(sigma)
        elements.append((sigma, False, _induced(sigma, v, k)))
        if full_middle_automorphism and v == 2 * k and k > 1:
            elements.append((sigma, True, _induced(sigma, v, k, complement=True)))
    return tuple(elements)


def _claimed_fixed_count(sigma, complement: bool, v: int, k: int, exponent: int) -> int:
    if not complement or exponent % 2 == 0:
        return _subset_fixed_count(sigma, k, exponent)
    powered = permutation_power(sigma, exponent)
    counts = cycle_counts(powered)
    if any(length % 2 for length in counts):
        return 0
    return 2 ** sum(counts.values())


def _case(v: int, k: int, full_middle: bool):
    elements = johnson_action(v, k, full_middle)
    induced = tuple(element[2] for element in elements)
    reconstruction = all(
        fixed_formula_reconstructs(
            action,
            lambda exponent, sigma=sigma, complement=complement: _claimed_fixed_count(
                sigma, complement, v, k, exponent
            ),
        )
        for sigma, complement, action in elements
    )
    rows = verify_action(
        f"Aut J({v},{k})" if full_middle and v == 2 * k else f"S_{v} on {k}-subsets",
        induced,
        pair_orbits=min(k, v - k) + 1,
    )
    return rows, reconstruction, tuple(dict.fromkeys(induced))


def run_suite():
    rows = []
    reconstructions = []
    actions = []
    for parameters in ((4, 2, True), (5, 2, False), (5, 1, False)):
        case_rows, reconstruction, action = _case(*parameters)
        rows.extend(case_rows)
        reconstructions.append(reconstruction)
        actions.append(action)
    numerical = numerical_determinant_check(actions[0])
    return {
        "family": "Johnson",
        "cases": ((4, 2, "full automorphism group including complementation"), (5, 2, "S5"), (5, 1, "S5")),
        "rows": tuple(rows),
        "fixed-point reconstruction": all(reconstructions),
        "numerical determinant": numerical,
        "passed": all(row["pass"] for row in rows) and all(reconstructions) and numerical[2] < 1e-10,
    }


if __name__ == "__main__":
    result = run_suite()
    print(f"Johnson: {'PASS' if result['passed'] else 'FAIL'} ({len(result['rows'])} comparisons)")

