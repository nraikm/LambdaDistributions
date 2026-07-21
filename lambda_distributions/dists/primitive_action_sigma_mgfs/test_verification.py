from lambda_distributions.dists.primitive_action_sigma_mgfs.verification import (
    almost_simple_suite,
    compound_diagonal_suite,
    diagonal_and_holomorph_suite,
    holomorph_compound_suite,
    product_action_suite,
    run_suite,
    twisted_wreath_suite,
)


def test_almost_simple_class_power_and_coefficients():
    result = almost_simple_suite()
    assert result["representation law"]
    assert result["class-power checks"] == 360
    assert result["passed"]


def test_simple_and_compound_diagonal_actions():
    diagonal = diagonal_and_holomorph_suite()
    assert diagonal["conjugacy classes"] == 5
    assert diagonal["rows"][1]["direct [m_(1,1)]"] == 77
    assert diagonal["passed"]
    compound = compound_diagonal_suite()
    assert tuple(row["direct"] for row in compound["rows"]) == (25, 15)
    assert compound["passed"]


def test_product_action_wreath_cases():
    result = product_action_suite()
    assert tuple(case["group order"] for case in result["cases"]) == (72, 1296)
    assert tuple(case["rows"][1]["formula"] for case in result["cases"]) == (15, 35)
    assert result["passed"]


def test_twisted_and_holomorph_compound_cases():
    twisted = twisted_wreath_suite()
    assert twisted["pair direct"] == 21
    assert twisted["rows"][-1]["direct average"] == 666
    assert twisted["passed"]
    compound = holomorph_compound_suite()
    assert compound["direct pair orbits"] == 6
    assert compound["passed"]


def test_full_primitive_action_suite():
    result = run_suite()
    assert result["exact checks"] == 43642
    assert result["passed"]
