from lambda_distributions.dists.verification import (
    run_all,
    verify_clifford_casimirs,
    verify_half_spin_coefficients,
    verify_pin_components,
    verify_spin_coefficients,
)


def test_odd_spin_weyl_coefficients_through_rank_five():
    assert all(row["passed"] for row in verify_spin_coefficients(5))


def test_half_spin_weyl_coefficients_through_rank_five():
    assert all(row["passed"] for row in verify_half_spin_coefficients(2, 5))


def test_clifford_matrix_and_pin_component_checks():
    assert all(row["passed"] for row in verify_clifford_casimirs(4))
    assert all(row["passed"] for row in verify_pin_components())


def test_full_spin_pin_report():
    report = run_all(5)
    assert report["checks"] == 76
    assert report["passed"]
