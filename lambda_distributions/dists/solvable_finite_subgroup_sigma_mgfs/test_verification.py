from .verification import run_suite


def test_solvable_and_finite_subgroup_sigma_mgfs():
    result = run_suite()
    assert result["passed"]
    assert result["fixed-power checks"] >= 1_000
    assert result["coefficient comparisons"] >= 70
