from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import braid_diagnostics, s3_permutation_braid_image, verify_case


def run_sweep() -> list[dict[str, object]]:
    group, generators = s3_permutation_braid_image()
    diagnostics = braid_diagnostics(generators)
    assert diagnostics["passed"] and len(group) == 6
    rows = [
        verify_case(group, tau, "S_3 permutation B_3 image")
        for tau in [(1,), (2,), (1, 1), (3,), (2, 1), (2, 2)]
    ]
    assert all(row["passed"] and row["class_count"] == 3 for row in rows)
    return rows


if __name__ == "__main__":
    for result in run_sweep():
        print(result)

