from lambda_distributions.dists.central_extensions_projective_representations.verification import run_suite


def test_central_extension_and_projective_suite() -> None:
    result = run_suite(max_degree=6)
    assert result["passed"]
    assert result["groups"] == 9
    assert result["coefficient comparisons"] == 180
    assert result["determinant comparisons"] == 5
    assert result["moment comparisons"] == 9
    assert result["maximum error"] < 2.0e-9
