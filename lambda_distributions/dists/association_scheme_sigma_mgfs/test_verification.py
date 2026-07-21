from .verification import run_suite


def test_association_scheme_sigma_mgfs():
    result = run_suite()
    assert result["passed"]
    assert result["coefficient comparisons"] >= 80
