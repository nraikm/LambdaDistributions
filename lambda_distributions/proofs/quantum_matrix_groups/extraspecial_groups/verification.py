from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    clean_integer,
    d8_matrices,
    extraspecial_2_prediction,
    extraspecial_p2_prediction,
    modular_extraspecial_matrices,
    pauli_prediction,
    prime_pauli_matrices,
    q8_matrices,
    target_coefficient,
)


def run_sweep():
    rows = []
    cases = [
        ("extraspecial exponent 3", prime_pauli_matrices(3), lambda tau: pauli_prediction(3, 3, 1, tau), [(3,), (2, 1), (6,), (3, 3)]),
        ("extraspecial exponent 9", modular_extraspecial_matrices(3), lambda tau: extraspecial_p2_prediction(3, 1, tau), [(3,), (2, 1), (6,), (3, 3)]),
        ("D8 (plus type)", d8_matrices(), lambda tau: extraspecial_2_prediction(+1, 1, tau), [(1,), (2,), (1, 1), (4,), (2, 2)]),
        ("Q8 (minus type)", q8_matrices(), lambda tau: extraspecial_2_prediction(-1, 1, tau), [(1,), (2,), (1, 1), (4,), (2, 2)]),
    ]
    for name, matrices, prediction, partitions in cases:
        for tau in partitions:
            direct = clean_integer(target_coefficient(matrices, tau))
            predicted = prediction(tau)
            rows.append({
                "group": name,
                "dimension": matrices[0].shape[0],
                "order": len(matrices),
                "tau": str(tau),
                "direct": direct,
                "formula": round(predicted, 10),
                "passed": abs(direct - predicted) < 2e-8,
            })
    return rows


if __name__ == "__main__":
    rows = run_sweep()
    for row in rows:
        print(row)
    assert all(row["passed"] for row in rows)

