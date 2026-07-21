from for_this_guy.schur_functor_classical_groups.common import schur_dimension
from for_this_guy.schur_functor_classical_groups.orthogonal.verification import (
    run_suite as run_orthogonal,
)
from for_this_guy.schur_functor_classical_groups.symplectic.verification import (
    run_suite as run_symplectic,
)
from for_this_guy.schur_functor_classical_groups.unitary.verification import (
    run_suite as run_unitary,
)


def test_hook_content_dimensions_used_by_the_matrix_constructor():
    assert schur_dimension((2,), 3) == 6
    assert schur_dimension((1, 1), 4) == 6
    assert schur_dimension((2, 1), 3) == 8


def test_unitary_schur_functor_suite():
    assert run_unitary()["passed"]


def test_orthogonal_schur_functor_suite():
    assert run_orthogonal()["passed"]


def test_symplectic_schur_functor_suite():
    assert run_symplectic()["passed"]
