from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    assert_real_orthogonal_representation,
    clean_integer,
    extraspecial_formula_coefficient,
    real_pauli_group,
    target_coefficient,
)


PARTITIONS = [(1,), (2,), (1, 1), (3, 1), (4,), (2, 2), (6,), (4, 2)]


def run_sweep() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for m in (1, 2, 3):
        matrices = real_pauli_group(m)
        expected_order = 2 ** (1 + 2 * m)
        assert len(matrices) == expected_order
        assert_real_orthogonal_representation(matrices)
        for tau in PARTITIONS:
            direct = clean_integer(target_coefficient(matrices, tau))
            formula = extraspecial_formula_coefficient(m, tau)
            rows.append(
                {
                    "group": f"E_{m}",
                    "m": m,
                    "N": 2**m,
                    "order": expected_order,
                    "tau": str(tau),
                    "direct": direct,
                    "formula (11.5)": formula,
                    "passed": direct == formula,
                }
            )
    assert all(row["passed"] for row in rows)
    return rows


if __name__ == "__main__":
    for row in run_sweep():
        print(row)
