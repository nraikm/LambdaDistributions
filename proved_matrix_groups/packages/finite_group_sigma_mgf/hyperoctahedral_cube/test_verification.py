import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from for_this_guy.finite_group_sigma_mgf.hyperoctahedral_cube.verification import run_suite


def test_hyperoctahedral_cube_formulas():
    assert run_suite()["passed"]
