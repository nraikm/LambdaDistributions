"""Matrix-level tests of the Grassmann formulas (21)--(25)."""

from __future__ import annotations

from ..finite_field import (
    general_linear_group,
    induced_subspace_permutations,
    invariant_subspace_count,
    matrix_power,
    subspaces,
)
from ..shared import fixed_formula_reconstructs, numerical_determinant_check, verify_action


def _case(q: int, n: int, k: int):
    matrices = general_linear_group(n, q)
    points = subspaces(n, k, q)
    actions = induced_subspace_permutations(matrices, points, q)
    # Scalar matrices may have the same action.  Use one representative matrix
    # for each element of the permutation image, as Molien averaging requires.
    representative = {}
    position = {space: index for index, space in enumerate(points)}
    from ..finite_field import move_subspace

    for matrix in matrices:
        action = tuple(position[move_subspace(matrix, space, q)] for space in points)
        representative.setdefault(action, matrix)
    reconstruction = all(
        fixed_formula_reconstructs(
            action,
            lambda exponent, matrix=matrix: invariant_subspace_count(
                matrix_power(matrix, n, q, exponent), points, q
            ),
        )
        for action, matrix in representative.items()
    )
    rows = verify_action(
        f"PGL_{n}({q}) image on Gr_{q}({n},{k})",
        actions,
        pair_orbits=min(k, n - k) + 1,
    )
    return rows, reconstruction, actions, len(matrices), len(points)


def run_suite():
    rows = []
    reconstructions = []
    metadata = []
    actions = []
    for q, n, k in ((2, 3, 1), (2, 3, 2), (3, 2, 1)):
        case_rows, reconstruction, action, matrix_order, point_count = _case(q, n, k)
        rows.extend(case_rows)
        reconstructions.append(reconstruction)
        actions.append(action)
        metadata.append({"q": q, "N": n, "k": k, "|GL|": matrix_order, "|X|": point_count, "|image|": len(action)})
    numerical = numerical_determinant_check(actions[0])
    return {
        "family": "Grassmann",
        "cases": tuple(metadata),
        "rows": tuple(rows),
        "fixed-point reconstruction": all(reconstructions),
        "numerical determinant": numerical,
        "passed": all(row["pass"] for row in rows) and all(reconstructions) and numerical[2] < 1e-10,
    }


if __name__ == "__main__":
    result = run_suite()
    print(f"Grassmann: {'PASS' if result['passed'] else 'FAIL'} ({len(result['rows'])} comparisons)")

