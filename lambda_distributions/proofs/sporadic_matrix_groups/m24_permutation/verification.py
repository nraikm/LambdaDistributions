from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    M24_CYCLE_DATA,
    M24_ORDER,
    bell_number,
    m24_generators,
    m24_cycle_data_sha256,
    m24_symmetric_coefficient,
    m24_tensor_coefficient,
    multiset_orbit_count,
    partition_number,
    permutation_matrix,
    permutation_order,
    subset_orbit_count,
    tuple_orbit_count,
    word,
    compose,
)


def run_sweep() -> list[dict[str, object]]:
    a, b = m24_generators()
    matrices = (permutation_matrix(a), permutation_matrix(b))
    assert matrices[0].shape == (24, 24)
    assert permutation_order(a) == 2
    assert permutation_order(b) == 3
    assert permutation_order(compose(a, b)) == 23
    assert permutation_order(word({"a": a, "b": b}, "abababbababbabb")) == 4
    assert sum(size for _, size, _ in M24_CYCLE_DATA) == M24_ORDER
    assert m24_cycle_data_sha256() == (
        "103ba576555f01bea74df73f8e2f788b02bee4e8b20df3ed198eb005d16de044"
    )

    rows: list[dict[str, object]] = []
    for degree in range(1, 7):
        direct = multiset_orbit_count(degree)
        class_formula = m24_symmetric_coefficient(degree)
        symmetric_reference = partition_number(degree)
        assert direct == class_formula
        rows.append(
            {
                "coefficient": f"m_({degree})",
                "direct orbit count": direct,
                "class-sum formula": class_formula,
                "S_24 reference": symmetric_reference,
                "agreement": direct == class_formula,
            }
        )

    for degree in range(1, 5):
        direct = tuple_orbit_count(degree)
        class_formula = m24_tensor_coefficient(degree)
        symmetric_reference = bell_number(degree)
        assert direct == class_formula
        rows.append(
            {
                "coefficient": f"m_(1^{degree})",
                "direct orbit count": direct,
                "class-sum formula": class_formula,
                "S_24 reference": symmetric_reference,
                "agreement": direct == class_formula,
            }
        )

    six_subset_orbits = subset_orbit_count(6)
    assert six_subset_orbits == 2
    tensor_six = bell_number(6) + six_subset_orbits - 1
    assert tensor_six == m24_tensor_coefficient(6) == 204
    rows.append(
        {
            "coefficient": "m_(1^6)",
            "direct orbit count": tensor_six,
            "class-sum formula": m24_tensor_coefficient(6),
            "S_24 reference": bell_number(6),
            "agreement": True,
        }
    )
    return rows


if __name__ == "__main__":
    for row in run_sweep():
        print(row)
