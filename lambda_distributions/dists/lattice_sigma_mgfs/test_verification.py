from .verification import run_suite


def test_lattice_sigma_mgfs():
    result = run_suite()
    assert result["passed"]
    assert result["case count"] == 12
    assert result["coefficient comparisons"] >= 400
    assert result["fixed-space comparisons"] == 48
    assert result["frame comparisons"] >= 7000
