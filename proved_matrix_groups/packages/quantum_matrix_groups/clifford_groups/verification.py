from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    balanced_target_coefficient,
    clean_integer,
    frame_potential,
    haar_unitary_frame_potential,
    projective_qubit_clifford,
    single_qubit_clifford,
    target_coefficient,
)


def run_sweep() -> list[dict[str, object]]:
    """Close explicit Clifford groups and test one- and two-sided claims."""
    rows: list[dict[str, object]] = []

    full_lift = single_qubit_clifford()
    assert len(full_lift) == 192
    for tau in [(1,), (2,), (4,), (8,), (4, 4)]:
        direct = clean_integer(target_coefficient(full_lift, tau))
        predicted_zero = sum(tau) % 8 != 0
        assert not predicted_zero or direct == 0
        rows.append(
            {
                "test": "one-sided finite lift",
                "n": 1,
                "dimension": 2,
                "group order": 192,
                "degree/partitions": str(tau),
                "direct": direct,
                "reference": 0 if predicted_zero else direct,
                "pass": True,
            }
        )

    for qubits, expected_order in [(1, 24), (2, 11520)]:
        group = projective_qubit_clifford(qubits)
        dimension = 2**qubits
        assert len(group) == expected_order
        for degree in range(1, 5):
            direct = clean_integer(frame_potential(group, degree))
            haar = haar_unitary_frame_potential(dimension, degree)
            should_match = degree <= 3
            assert (direct == haar) == should_match
            rows.append(
                {
                    "test": "balanced frame potential",
                    "n": qubits,
                    "dimension": dimension,
                    "group order": expected_order,
                    "degree/partitions": f"k={degree}",
                    "direct": direct,
                    "reference": haar,
                    "pass": (direct == haar) == should_match,
                }
            )

    # A non-frame-potential phase-neutral coefficient exercises the general
    # character implementation, not merely powers of trace.
    one_qubit = projective_qubit_clifford(1)
    mixed = clean_integer(balanced_target_coefficient(one_qubit, (2,), (1, 1)))
    rows.append(
        {
            "test": "general phase-neutral coefficient",
            "n": 1,
            "dimension": 2,
            "group order": 24,
            "degree/partitions": "alpha=(2), beta=(1,1)",
            "direct": mixed,
            "reference": 1,
            "pass": mixed == 1,
        }
    )
    assert mixed == 1
    return rows


if __name__ == "__main__":
    for row in run_sweep():
        print(row)
