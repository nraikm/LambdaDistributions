import pytest

from .code_monomial.verification import CodeCase, run_suite as run_code, validate_case
from .delta_su3.verification import run_suite as run_delta
from .psl2_projective_line.verification import run_suite as run_psl2


def test_code_monomial_formula():
    assert run_code(max_degree=4)["passed"]


def test_code_monomial_rejects_nonpermutation_automorphism():
    case = CodeCase(
        "invalid monomial automorphism",
        3,
        2,
        ((0, 0), (1, 1), (2, 2)),
        ((1, -1),),
    )
    with pytest.raises(ValueError, match="coordinate permutations"):
        validate_case(case)


def test_delta_su3_formulas():
    assert run_delta(max_degree=4, n_values=(2, 3, 4))["passed"]


def test_psl2_projective_line_formula():
    assert run_psl2(q_values=(2, 3, 5), max_degree=4)["passed"]
