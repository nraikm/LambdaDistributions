from for_this_guy.finite_lie_type_permutation_actions.verification_core import (
    linear_suite,
    polar_degree_three_type_counts,
    polar_suite,
)


def test_linear_suite():
    result = linear_suite()
    assert result["passed"]
    assert result["cycle reconstruction"]
    assert result["numerical determinant"][2] < 1e-10


def test_polar_suite():
    result = polar_suite(include_unitary=True)
    assert result["passed"]
    assert result["cycle reconstruction"]
    assert result["numerical determinant"][2] < 1e-10


def test_polar_degree_three_types_stabilize_across_two_ranks():
    assert polar_degree_three_type_counts() == {
        "symplectic": (7, 7),
        "orthogonal_plus": (7, 7),
        "unitary": (8, 8),
    }
