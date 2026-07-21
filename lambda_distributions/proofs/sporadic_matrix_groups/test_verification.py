from .co0_leech.verification import run_sweep as run_co0
from .common import M24_CYCLE_DATA, M24_ORDER, m24_cycle_data_sha256
from .m24_permutation.verification import run_sweep as run_m24


def test_m24_permutation_verification_and_external_data():
    rows = run_m24()
    assert rows and all(row["agreement"] for row in rows)
    assert sum(size for _, size, _ in M24_CYCLE_DATA) == M24_ORDER
    assert m24_cycle_data_sha256() == (
        "103ba576555f01bea74df73f8e2f788b02bee4e8b20df3ed198eb005d16de044"
    )


def test_co0_signed_coordinate_stress_test():
    rows = run_co0()
    assert rows and all(row["agreement"] for row in rows)
