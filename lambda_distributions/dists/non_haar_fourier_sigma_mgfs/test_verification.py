from lambda_distributions.dists.non_haar_fourier_sigma_mgfs.verification import (
    ewens_suite,
    run_all,
    s3_walk_suite,
    u1_heat_suite,
)


def test_s3_walks():
    result = s3_walk_suite()
    assert result["group order"] == 6
    assert result["matrix dimensions"] == (3, 2)
    assert result["comparisons"] == 220
    assert not result["noncentral transform is scalar"]
    assert result["passed"]


def test_u1_heat_kernel():
    result = u1_heat_suite()
    assert result["representation dimension"] == 2
    assert result["maximum error"] < 1e-11
    assert result["endpoint checks"]
    assert result["passed"]


def test_ewens_symmetric_groups():
    result = ewens_suite()
    assert result["dimensions"] == (3, 4, 5, 6)
    assert result["coefficient comparisons"] == 132
    assert result["full determinant comparisons"] == 12
    assert result["passed"]


def test_full_suite():
    assert run_all()["passed"]
