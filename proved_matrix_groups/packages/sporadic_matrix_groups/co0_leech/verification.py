from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    m24_generators,
    multiset_orbit_count,
    permutation_matrix,
    signed_m24_symmetric_coefficient,
)


def run_sweep() -> list[dict[str, object]]:
    a, b = m24_generators()
    generators = (permutation_matrix(a), permutation_matrix(b), -np.eye(24, dtype=np.int8))
    assert all(matrix.shape == (24, 24) for matrix in generators)
    assert np.array_equal(generators[2] @ generators[2], np.eye(24, dtype=np.int8))

    rows: list[dict[str, object]] = []
    for degree in range(1, 7):
        # Direct representation-theoretic calculation: -I acts by (-1)^degree;
        # in even degree its fixed space is the M24 multiset-orbit space.
        direct = 0 if degree % 2 else multiset_orbit_count(degree)
        formula = signed_m24_symmetric_coefficient(degree)
        assert direct == formula
        rows.append(
            {
                "degree": degree,
                "direct invariant dimension": direct,
                "signed class-sum formula": formula,
                "central parity": "forced zero" if degree % 2 else "even sector",
                "agreement": direct == formula,
            }
        )
    return rows


if __name__ == "__main__":
    for row in run_sweep():
        print(row)
