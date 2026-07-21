from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    assert_real_orthogonal_representation,
    barnes_wall_group,
    barnes_wall_lattice_basis,
    clean_integer,
    coefficient_integrand,
    matrix_key,
    preserves_lattice,
    real_clifford_group,
    target_coefficient,
)


PARTITIONS = [(1,), (2,), (4,), (6,), (8,), (1, 1), (2, 2), (4, 2)]


def run_sweep() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for m in (1, 2):
        clifford = real_clifford_group(m)
        barnes_wall = barnes_wall_group(m)
        assert_real_orthogonal_representation(clifford)
        assert_real_orthogonal_representation(barnes_wall)
        lattice_basis = barnes_wall_lattice_basis(m)
        barnes_wall_keys = {matrix_key(matrix) for matrix in barnes_wall}
        assert len(clifford) == 2 * len(barnes_wall)

        # This independently identifies B_m as the Clifford elements preserving
        # Z^2 (m=1) or D4 (m=2), rather than merely trusting the generators.
        preserving_keys = {
            matrix_key(matrix)
            for matrix in clifford
            if preserves_lattice(matrix, lattice_basis)
        }
        assert preserving_keys == barnes_wall_keys

        epsilon = [1 if matrix_key(matrix) in barnes_wall_keys else -1 for matrix in clifford]
        for tau in PARTITIONS:
            direct = clean_integer(target_coefficient(barnes_wall, tau))
            clifford_part = clean_integer(target_coefficient(clifford, tau))
            twisted_part = clean_integer(target_coefficient(clifford, tau, epsilon))
            formula = clifford_part + twisted_part
            rows.append(
                {
                    "group": f"B_{m}=Aut(BW_{2**m})",
                    "m": m,
                    "N": 2**m,
                    "order": len(barnes_wall),
                    "tau": str(tau),
                    "direct B_m": direct,
                    "C_m part": clifford_part,
                    "twisted part": twisted_part,
                    "formula (11.19)": formula,
                    "passed": direct == formula,
                }
            )
    assert all(row["passed"] for row in rows)
    return rows


if __name__ == "__main__":
    for row in run_sweep():
        print(row)
