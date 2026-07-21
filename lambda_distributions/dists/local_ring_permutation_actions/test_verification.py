import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lambda_distributions.dists.local_ring_permutation_actions.verification import (
    adjoint_suite,
    grassmann_pair_check,
    local_linear_suite,
    run_suite,
    symplectic_rank_one_suite,
)


def test_local_linear_cases():
    assert local_linear_suite(2, 2)["passed"]
    assert local_linear_suite(3, 2)["passed"]


def test_rank_two_summands_over_z4():
    result = grassmann_pair_check()
    assert result["summands"] == 560
    assert result["direct pair orbits"] == 6
    assert result["Smith types"] == (
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 2),
    )
    assert result["passed"]


def test_formed_and_adjoint_cases():
    assert symplectic_rank_one_suite()["passed"]
    assert adjoint_suite()["passed"]


def test_full_suite():
    assert run_suite()["passed"]
