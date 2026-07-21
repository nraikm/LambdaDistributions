from dataclasses import replace

import pytest

from .verification import representative_cases, run_suite, validate_case


def test_code_monomial_suite():
    result = run_suite(max_degree=5)
    assert result["passed"]
    assert result["groups"] == 5
    assert result["coefficient comparisons"] == 95


def test_composite_cyclic_alphabet_is_included():
    case = representative_cases()[-1]
    assert case.modulus == 4
    assert len(case.code) * len(case.automorphisms) == 64


def test_nonautomorphism_is_rejected():
    case = representative_cases()[-1]
    invalid = replace(case, automorphisms=((0, 1, 2, 3), (0, 2, 1, 3)))
    with pytest.raises(ValueError, match="does not preserve"):
        validate_case(invalid)
