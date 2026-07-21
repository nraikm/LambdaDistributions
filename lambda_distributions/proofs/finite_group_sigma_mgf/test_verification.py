from lambda_distributions.proofs.finite_group_sigma_mgf.verification_core import (
    REGULAR_CASES,
    SUBSET_CASES,
    SYMMETRIC_REP_CASES,
    derived_character_errors,
    induced_verification_row,
    representative_suites,
    verify_regular_case,
    verify_sign_case,
    verify_standard_case,
    verify_subset_case,
)


def test_regular_representations():
    assert all(verify_regular_case(*case).passed for case in REGULAR_CASES)


def test_symmetric_group_representations():
    for kind, n, tau in SYMMETRIC_REP_CASES:
        result = verify_sign_case(n, tau) if kind == "sign" else verify_standard_case(n, tau)
        assert result.passed
    assert induced_verification_row().passed


def test_subset_actions():
    assert all(verify_subset_case(*case).passed for case in SUBSET_CASES)


def test_stable_subset_quadratic_coefficient():
    for k in (1, 2, 3):
        n = 2 * k
        assert verify_subset_case(n, k, (2,)).formula == k + 1
        assert verify_subset_case(n, k, (1, 1)).formula == k + 1


def test_alternating_subset_quadratic_boundaries():
    assert verify_subset_case(4, 2, (2,), alternating=True).formula == 3
    assert verify_subset_case(4, 2, (1, 1), alternating=True).formula == 4
    assert verify_subset_case(5, 2, (1, 1), alternating=True).formula == 3
    assert verify_subset_case(6, 2, (1, 1), alternating=True).formula == 3


def test_derived_character_formulas():
    assert max(derived_character_errors(3).values()) < 1e-9


def test_reported_rows_pass():
    assert all(row["pass"] for rows in representative_suites().values() for row in rows)
