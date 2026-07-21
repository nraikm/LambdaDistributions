from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import clean_integer, pauli_prediction, prime_pauli_matrices, qubit_pauli_matrices, target_coefficient


CASES = [
    ("mu_4 P_{2,1}", 2, 2, 1, 4, qubit_pauli_matrices(1), [(2,), (4,), (2, 2), (3, 1), (8,)]),
    ("mu_4 P_{2,2}", 2, 2, 2, 4, qubit_pauli_matrices(2), [(2,), (4,), (2, 2), (4, 4)]),
    ("mu_6 P_{3,1}", 3, 3, 1, 6, prime_pauli_matrices(3, scalar_order=6), [(3,), (6,), (3, 3), (5, 1)]),
]


def run_sweep():
    rows = []
    for name, q, p, n, s, matrices, partitions in CASES:
        for tau in partitions:
            direct = clean_integer(target_coefficient(matrices, tau))
            predicted = pauli_prediction(q, p, n, tau, scalar_order=s)
            rows.append({
                "group": name,
                "dimension": q**n,
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

