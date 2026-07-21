from .cyclic_scalar.verification import run_sweep as cyclic_sweep
from .ising_clifford.verification import run_sweep as ising_sweep
from .s3_permutation.verification import run_sweep as s3_sweep


def test_cyclic_scalar_braid_image() -> None:
    assert all(row["passed"] for row in cyclic_sweep())


def test_s3_permutation_braid_image() -> None:
    assert all(row["passed"] for row in s3_sweep())


def test_ising_and_projective_adjoint() -> None:
    assert all(row["passed"] for row in ising_sweep())

