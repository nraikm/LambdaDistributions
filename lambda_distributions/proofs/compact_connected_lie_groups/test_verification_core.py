from .verification_core import (
    exceptional_representations,
    su_adjoint,
    verify_expected,
    verify_monomial_expected,
)


def test_exceptional_weight_data():
    reps = exceptional_representations()
    assert (reps["g2_7"].dimension, len(reps["g2_7"].root_system)) == (7, 12)
    assert (reps["spin7_8"].dimension, len(reps["spin7_8"].root_system)) == (8, 18)
    assert (reps["f4_26"].dimension, len(reps["f4_26"].root_system)) == (26, 48)
    assert (reps["e6_27"].dimension, len(reps["e6_27"].root_system)) == (27, 72)
    assert (reps["e7_56"].dimension, len(reps["e7_56"].root_system)) == (56, 126)
    assert (reps["e8_248"].dimension, len(reps["e8_248"].root_system)) == (248, 240)


def test_exact_manageable_exceptional_coefficients():
    reps = exceptional_representations()
    cases = (
        (reps["g2_7"], {(): 1, (2,): 1, (1, 1, 1): 1}, 3),
        (
            reps["spin7_8"],
            {(): 1, (2,): 1, (4,): 1, (2, 2): 1, (1, 1, 1, 1): 1},
            4,
        ),
        (reps["f4_26"], {(): 1, (2,): 1, (3,): 1}, 3),
        (reps["e6_27"], {(): 1, (3,): 1}, 3),
        (reps["e7_56"], {(): 1, (1, 1): 1}, 2),
    )
    for rep, expected, maximum_degree in cases:
        assert all(row["pass"] for row in verify_expected(rep, expected, maximum_degree))


def test_exact_monomial_coefficients_in_manageable_degrees():
    reps = exceptional_representations()
    cases = (
        (
            reps["g2_7"],
            {(): 1, (2,): 1, (1, 1): 1, (1, 1, 1): 1},
            3,
        ),
        (
            reps["spin7_8"],
            {(): 1, (2,): 1, (1, 1): 1},
            3,
        ),
        (
            reps["f4_26"],
            {(): 1, (2,): 1, (1, 1): 1, (3,): 1, (2, 1): 1, (1, 1, 1): 1},
            3,
        ),
        (
            reps["e6_27"],
            {(): 1, (3,): 1, (2, 1): 1, (1, 1, 1): 1},
            3,
        ),
        (
            reps["e7_56"],
            {(): 1, (1, 1): 1},
            2,
        ),
        (
            reps["e8_248"],
            {(): 1, (2,): 1, (1, 1): 1},
            2,
        ),
    )
    for rep, expected, maximum_degree in cases:
        assert all(
            row["pass"]
            for row in verify_monomial_expected(rep, expected, maximum_degree)
        )


def test_su_adjoint_small_n():
    for n in (2, 3, 4):
        expected = {(): 1, (2,): 1, (1, 1, 1): 1}
        if n >= 3:
            expected[(3,)] = 1
        assert all(row["pass"] for row in verify_expected(su_adjoint(n), expected, 3))
