"""Matrix-level tests for symplectic polar points and a dual-polar scheme."""

from __future__ import annotations

from functools import lru_cache

from ..finite_field import (
    general_linear_group,
    induced_subspace_permutations,
    invariant_subspace_count,
    matrix_power,
    subspaces,
)
from ..shared import fixed_formula_reconstructs, numerical_determinant_check, verify_action


def alternating_pair(left, right):
    return (left[0] * right[1] + left[1] * right[0] + left[2] * right[3] + left[3] * right[2]) % 2


@lru_cache(maxsize=None)
def symplectic_group_4_2():
    basis = tuple(tuple(int(i == j) for i in range(4)) for j in range(4))
    from ..finite_field import mat_vec

    return tuple(
        matrix
        for matrix in general_linear_group(4, 2)
        if all(
            alternating_pair(mat_vec(matrix, x, 2), mat_vec(matrix, y, 2))
            == alternating_pair(x, y)
            for x in basis
            for y in basis
        )
    )


def lagrangians_4_2():
    return tuple(
        space
        for space in subspaces(4, 2, 2)
        if all(alternating_pair(x, y) == 0 for x in space for y in space)
    )


def _case(name: str, points, pair_orbits: int):
    matrices = symplectic_group_4_2()
    actions = induced_subspace_permutations(matrices, points, 2)
    reconstruction = all(
        fixed_formula_reconstructs(
            action,
            lambda exponent, matrix=matrix: invariant_subspace_count(
                matrix_power(matrix, 4, 2, exponent), points, 2
            ),
        )
        for action, matrix in zip(actions, matrices, strict=True)
    )
    rows = verify_action(name, actions, pair_orbits=pair_orbits)
    return rows, reconstruction, actions


def run_suite():
    group = symplectic_group_4_2()
    isotropic_points = subspaces(4, 1, 2)
    lagrangians = lagrangians_4_2()
    point_rows, point_reconstruction, point_action = _case(
        "Sp_4(2) on isotropic points", isotropic_points, 3
    )
    dual_rows, dual_reconstruction, dual_action = _case(
        "Sp_4(2) on maximal isotropic 2-spaces", lagrangians, 3
    )
    rows = point_rows + dual_rows
    numerical = numerical_determinant_check(dual_action)
    return {
        "family": "Polar",
        "cases": (
            {"group": "Sp_4(2)", "|G|": len(group), "object": "isotropic points", "|X|": len(isotropic_points)},
            {"group": "Sp_4(2)", "|G|": len(group), "object": "maximal isotropic 2-spaces", "|X|": len(lagrangians)},
        ),
        "rows": tuple(rows),
        "fixed-point reconstruction": point_reconstruction and dual_reconstruction,
        "numerical determinant": numerical,
        "passed": all(row["pass"] for row in rows)
        and point_reconstruction
        and dual_reconstruction
        and numerical[2] < 1e-10,
    }


if __name__ == "__main__":
    result = run_suite()
    print(f"Polar: {'PASS' if result['passed'] else 'FAIL'} ({len(result['rows'])} comparisons)")

