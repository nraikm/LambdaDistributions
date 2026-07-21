from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import braid_diagnostics, cyclic_scalar_braid_image, verify_case


def run_sweep() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cases = [
        (3, 1, (1,)),
        (3, 1, (3,)),
        (3, 2, (2, 1)),
        (4, 2, (4,)),
        (4, 3, (2, 2)),
        (5, 4, (5,)),
    ]
    for order, dimension, tau in cases:
        group, generators = cyclic_scalar_braid_image(order, dimension)
        diagnostics = braid_diagnostics(generators)
        assert diagnostics["passed"] and len(group) == order
        row = verify_case(group, tau, f"C_{order} scalar B_3 image")
        expected = row["target_dimension"] if sum(tau) % order == 0 else 0
        assert row["projector_rank"] == expected and row["passed"]
        row.update({"scalar_order": order, "selection_prediction": expected})
        rows.append(row)
    return rows


if __name__ == "__main__":
    for result in run_sweep():
        print(result)

