from lambda_distributions.dists.multiplicative_wreath_sigma_mgfs.verification import (
    diagonal_suite,
    fixed_tail_bipartition_suite,
    fock_suite,
    run_suite,
    tensor_induced_suite,
    wreath_class_power_suite,
)


def test_wreath_types_and_class_weights():
    result = wreath_class_power_suite()
    assert tuple(case["order"] for case in result["cases"]) == (48, 72)
    assert result["passed"]


def test_tensor_induced_matrix_groups():
    result = tensor_induced_suite()
    assert tuple(case["dimension"] for case in result["cases"]) == (4, 8)
    assert result["cases"][1]["rows"][-1]["moment theorem"] == 120
    assert result["passed"]


def test_fixed_tail_bipartition_character_polynomial():
    result = fixed_tail_bipartition_suite()
    assert result["character polynomial"] == "X_(1,+) - X_(1,-)"
    assert result["cases"][-1]["rows"][-1]["class formula"] == 4
    assert result["passed"]


def test_diagonal_and_fock_formulas():
    diagonal = diagonal_suite()
    assert diagonal["cases"][1]["rows"][1]["class formula"] == 11
    assert diagonal["passed"]
    assert fock_suite()["passed"]


def test_full_multiplicative_wreath_suite():
    result = run_suite()
    assert result["exact checks"] > 10_000
    assert result["passed"]
