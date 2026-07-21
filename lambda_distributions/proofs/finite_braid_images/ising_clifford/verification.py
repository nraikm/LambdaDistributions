from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import (
    adjoint_projective_image,
    braid_diagnostics,
    full_twist,
    ising_b3_braid_image,
    verify_case,
)


def run_sweep() -> list[dict[str, object]]:
    lift, generators = ising_b3_braid_image()
    diagnostics = braid_diagnostics(generators)
    twist = full_twist(generators)
    twist_scalar = np.trace(twist) / 2
    assert diagnostics["passed"] and len(lift) == 192
    assert np.linalg.norm(twist - twist_scalar * np.eye(2)) < 2e-8
    assert abs(twist_scalar**8 - 1) < 2e-8

    rows = [verify_case(lift, tau, "Ising B_3 finite lift") for tau in [(1,), (4,), (8,), (4, 4)]]
    for row in rows:
        if row["total_degree"] % 8:
            assert row["projector_rank"] == 0
        assert row["passed"]

    projective = adjoint_projective_image(lift)
    assert len(projective) == 24
    projective_rows = [
        verify_case(projective, tau, "Ising projective adjoint on End(C^2)")
        for tau in [(1,), (2,), (1, 1), (3,)]
    ]
    assert all(row["passed"] for row in projective_rows)
    return rows + projective_rows


if __name__ == "__main__":
    for result in run_sweep():
        print(result)

