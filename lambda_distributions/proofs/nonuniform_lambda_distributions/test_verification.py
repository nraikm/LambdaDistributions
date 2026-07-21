from lambda_distributions.proofs.nonuniform_lambda_distributions.compact_u1_heat.verification import run_suite as run_u1
from lambda_distributions.proofs.nonuniform_lambda_distributions.cyclic_groups.verification import run_suite as run_cyclic
from lambda_distributions.proofs.nonuniform_lambda_distributions.symmetric_groups.verification import run_suite as run_symmetric


def test_cyclic_groups():
    assert run_cyclic(maximum_degree=4)["passed"]


def test_symmetric_groups():
    assert run_symmetric(maximum_degree=3)["passed"]


def test_compact_u1_heat():
    assert run_u1(maximum_degree=3)["passed"]
