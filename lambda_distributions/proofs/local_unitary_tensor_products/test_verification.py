from math import factorial

from lambda_distributions.proofs.local_unitary_tensor_products.verification import (
    DEFAULT_CASES,
    construction_check,
    hook_dimension,
    multilinear_coefficient,
    partitions,
    stable_power_moment,
    stable_symmetric_coefficient,
    symmetric_coefficient,
    z_partition,
)


def test_partition_data():
    assert list(partitions(4)) == [(4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)]
    assert z_partition((3, 1, 1)) == 6
    assert [hook_dimension(shape) for shape in partitions(4)] == [1, 3, 2, 3, 1]


def test_stable_power_formula():
    assert stable_power_moment((3, 4), (2, 1), (2, 1)) == 2**2
    assert stable_power_moment((3, 4), (3,), (2, 1)) == 0
    assert stable_power_moment((2, 4), (3,), (3,)) is None


def test_symmetric_coefficients():
    assert [symmetric_coefficient((2, 2), degree) for degree in range(1, 6)] == [1, 2, 2, 3, 3]
    assert [stable_symmetric_coefficient(2, degree) for degree in range(1, 6)] == [1, 2, 3, 5, 7]
    assert stable_symmetric_coefficient(3, 3) == 11


def test_multilinear_coefficients_and_stable_range():
    for dimensions in ((3,), (2, 3), (2, 2, 2)):
        for degree in range(1, min(dimensions) + 1):
            assert multilinear_coefficient(dimensions, degree) == factorial(degree) ** len(dimensions)
    assert multilinear_coefficient((2, 2), 3) == 25


def test_constructed_tensor_product_matrices():
    assert all(construction_check(case)["passed"] for case in DEFAULT_CASES)
