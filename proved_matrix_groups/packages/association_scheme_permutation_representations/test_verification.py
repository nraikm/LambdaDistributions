from for_this_guy.association_scheme_permutation_representations.grassmann.verification import run_suite as grassmann_suite
from for_this_guy.association_scheme_permutation_representations.hamming.verification import run_suite as hamming_suite
from for_this_guy.association_scheme_permutation_representations.johnson.verification import run_suite as johnson_suite
from for_this_guy.association_scheme_permutation_representations.polar.verification import run_suite as polar_suite


def test_johnson_suite():
    result = johnson_suite()
    assert result["passed"]
    assert result["fixed-point reconstruction"]


def test_hamming_suite():
    result = hamming_suite()
    assert result["passed"]
    assert result["fixed-point reconstruction"]


def test_grassmann_suite():
    result = grassmann_suite()
    assert result["passed"]
    assert result["fixed-point reconstruction"]


def test_polar_suite():
    result = polar_suite()
    assert result["passed"]
    assert result["fixed-point reconstruction"]

