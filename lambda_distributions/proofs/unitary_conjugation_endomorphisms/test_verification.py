from .unitary_conjugation_core import (
    REPRESENTATIVE_CASES,
    finite_rank_lr_coefficient,
    matrix_group_diagnostics,
    run_representative_checks,
    stable_orbit_count,
    necklace_euler_coefficient,
)


def test_representative_direct_matrix_checks():
    rows, diagnostics = run_representative_checks()
    assert len(rows) == len(REPRESENTATIVE_CASES) == 18
    assert all(row["finite_match"] for row in rows)
    assert all(row["stable_match"] in (None, True) for row in rows)
    assert all(row["target_dimension"] <= 405 for row in rows)
    assert all(row["rank_threshold"] > 0 for row in rows)
    assert all(row["spectral_gap_ratio"] >= 1.0e4 for row in rows)
    assert len(diagnostics) == 4


def test_expected_finite_rank_correction():
    # At degree three, n=2 is below stable rank: one of the six contraction
    # diagrams is removed by the second fundamental theorem.
    assert finite_rank_lr_coefficient(2, (1, 1, 1)) == 5
    assert finite_rank_lr_coefficient(3, (1, 1, 1)) == 6


def test_stable_orbits_equal_necklace_euler_coefficients():
    for tau, expected in (((3,), 3), ((2, 1), 4), ((1, 1, 1), 6), ((2, 2), 10)):
        assert stable_orbit_count(tau) == expected
        assert necklace_euler_coefficient(tau) == expected


def test_constructed_group_matrices():
    for n in range(1, 5):
        diagnostics = matrix_group_diagnostics(n)
        assert diagnostics["group_generators"] == n * n
        assert diagnostics["ad_dimension"] == n * n
        assert diagnostics["unitarity_error"] < 1.0e-12
        assert diagnostics["homomorphism_error"] < 1.0e-12
        assert diagnostics["trace_identity_error"] < 1.0e-12
