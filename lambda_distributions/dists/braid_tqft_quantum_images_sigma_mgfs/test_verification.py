from .verification import run_suite


def test_braid_tqft_quantum_image_sigma_mgfs():
    result = run_suite(haar_samples=1_500)
    assert result["passed"]
    assert result["finite coefficient comparisons"] >= 35
    assert result["fixed-power checks"] >= 1_000
    assert result["haar diagnostics"] == 16
