import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verification_core import (  # noqa: E402
    character_average,
    dicyclic_regular_formula,
    dicyclic_regular_representation,
    dicyclic_relation_error,
    dicyclic_representation,
    dicyclic_single_formula,
    dicyclic_spectral_formula,
    permutation_cycle_formula,
    permutation_matrices,
    representative_sweeps,
    reynolds_check,
    vector_partition_counts,
)


@pytest.mark.parametrize(
    ("m", "k", "tau"),
    [(2, 1, (2,)), (3, 1, (2, 2)), (4, 2, (3, 1)), (5, 3, (4,))],
)
def test_single_dicyclic_irrep(m, k, tau):
    matrices = dicyclic_representation(m, {k: 1})
    direct = reynolds_check(matrices, tau)
    formula = dicyclic_single_formula(m, k, tau)
    spectral = dicyclic_spectral_formula(m, {k: 1}, (), tau)
    assert direct["projector_rank"] == formula
    assert character_average(matrices, tau) == pytest.approx(formula, abs=1e-9)
    assert spectral == pytest.approx(formula, abs=1e-9)
    assert dicyclic_relation_error(m, {k: 1}) < 1e-9


def test_arbitrary_dicyclic_direct_sum_with_one_dimensional_characters():
    characters = ((1, -1, 1), (-1, 1j, 1))
    matrices = dicyclic_representation(3, {2: 1}, characters)
    for tau in [(1,), (2,), (2, 1), (3,)]:
        direct = reynolds_check(matrices, tau)
        spectral = dicyclic_spectral_formula(3, {2: 1}, characters, tau)
        assert direct["projector_rank"] == pytest.approx(spectral.real, abs=1e-9)
    assert dicyclic_relation_error(3, {2: 1}, characters) < 1e-9


def test_dicyclic_direct_sum_of_two_dimensional_irreps():
    matrices = dicyclic_representation(4, {1: 1, 2: 1})
    for tau in [(2,), (2, 1)]:
        direct = reynolds_check(matrices, tau)
        spectral = dicyclic_spectral_formula(4, {1: 1, 2: 1}, (), tau)
        assert direct["projector_rank"] == pytest.approx(spectral.real, abs=1e-9)


def test_dicyclic_regular_representation_formula():
    matrices = dicyclic_regular_representation(2)
    for tau in [(), (1,), (2,), (4,)]:
        direct = reynolds_check(matrices, tau)
        formula = dicyclic_regular_formula(2, tau)
        assert direct["projector_rank"] == formula
        assert character_average(matrices, tau) == pytest.approx(formula, abs=1e-9)


@pytest.mark.parametrize(
    ("n", "tau"), [(2, (3,)), (3, (2, 1)), (4, (2, 2)), (5, (2, 1))]
)
def test_symmetric_cycle_and_orbit_formulas(n, tau):
    matrices = permutation_matrices(n)
    direct = reynolds_check(matrices, tau)
    cycle = permutation_cycle_formula(n, tau)
    orbit, _ = vector_partition_counts(tau, n)
    assert direct["projector_rank"] == cycle == orbit
    assert character_average(matrices, tau) == pytest.approx(cycle, abs=1e-9)


@pytest.mark.parametrize(
    ("n", "tau"),
    [(3, (2,)), (3, (3,)), (4, (2, 1)), (5, (2, 2)), (5, (3, 1))],
)
def test_alternating_cycle_and_split_orbit_formulas(n, tau):
    matrices = permutation_matrices(n, alternating=True)
    direct = reynolds_check(matrices, tau)
    cycle = permutation_cycle_formula(n, tau, alternating=True)
    symmetric, correction = vector_partition_counts(tau, n)
    assert direct["projector_rank"] == cycle == symmetric + correction
    assert character_average(matrices, tau) == pytest.approx(cycle, abs=1e-9)


def test_every_reported_sweep_row_agrees():
    for group_rows in representative_sweeps().values():
        for row in group_rows:
            direct = row["direct"]
            for key, value in row.items():
                if key.endswith("formula"):
                    assert value == direct
            assert row["error"] < 1e-9
