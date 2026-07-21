from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import clean_integer, gf9_pauli_matrices, pauli_prediction, prime_pauli_matrices, target_coefficient


CASES = [
    ("P_{3,1}", 3, 3, 1, prime_pauli_matrices(3), [(1,), (3,), (2, 1), (3, 3), (6,)]),
    ("P_{5,1}", 5, 5, 1, prime_pauli_matrices(5), [(1,), (5,), (4, 1), (5, 5)]),
    ("P_{9,1}", 9, 3, 1, gf9_pauli_matrices(), [(1,), (3,), (2, 1), (3, 3), (6,)]),
    ("H_3(F_9)", 9, 3, 1, gf9_pauli_matrices(heisenberg=True), [(3,), (2, 1), (3, 3)]),
]


def run_sweep():
    rows = []
    for name, q, p, n, matrices, partitions in CASES:
        for tau in partitions:
            direct_raw = target_coefficient(matrices, tau)
            direct = clean_integer(direct_raw)
            predicted = pauli_prediction(q, p, n, tau)
            rows.append({
                "group": name,
                "dimension": q**n,
                "matrix count": len(matrices),
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

