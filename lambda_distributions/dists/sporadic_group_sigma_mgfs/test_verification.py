from lambda_distributions.dists.sporadic_group_sigma_mgfs.verification import (
    M11_ORDER,
    coefficient_sweep,
    numerical_sweep,
    orbit_sweep,
    representation_checks,
    run_suite,
)


def test_m11_matrix_groups_and_atlas_fingerprint():
    result = representation_checks()
    assert result["group order"] == M11_ORDER
    assert result["dimensions"] == (11, 10)
    assert result["aggregated spectra"] == 8
    assert result["fingerprint"] == {
        "o(a)": 2,
        "o(b)": 4,
        "o(ab)": 11,
        "o(ababbabbb)": 5,
    }
    assert result["passed"]


def test_exact_matrix_class_and_power_character_coefficients():
    rows = coefficient_sweep(6)
    assert len(rows) == 58
    assert all(row["passed"] for row in rows)
    values = {
        (row["representation"], row["tau"]): row["direct matrix average"]
        for row in rows
    }
    assert values[("permutation", (1,))] == 1
    assert values[("permutation", (1, 1, 1, 1))] == 15
    assert values[("deleted", (1,))] == 0
    assert values[("deleted", (2,))] == 1


def test_generator_orbits_and_direct_function_values():
    assert all(row["passed"] for row in orbit_sweep())
    numerical = numerical_sweep()
    assert all(row["passed"] for row in numerical)
    assert max(row["absolute error"] for row in numerical) < 5e-13


def test_full_suite():
    result = run_suite()
    assert result["exact coefficient checks"] == 58
    assert result["passed"]

