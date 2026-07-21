from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    TOL,
    assert_real_orthogonal_representation,
    clean_integer,
    coset_coefficient_means,
    coset_representatives,
    matrix_key,
    real_clifford_group,
    real_pauli_group,
    target_coefficient,
)


PARTITIONS = [(1,), (2,), (4,), (6,), (8,), (1, 1), (2, 2), (4, 2)]


def stable_code_reference(m: int, tau: tuple[int, ...]) -> int | None:
    """Return only values covered by the stated stable-range theorem."""
    if tau == (1,):
        return 0
    known = {(2,): 1, (4,): 1, (6,): 1, (8,): 2}
    if tau not in known:
        return None
    k = tau[0] // 2
    return known[tau] if m >= k - 1 else None


def run_sweep() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for m, quotient_order in ((1, 2), (2, 72)):
        clifford = real_clifford_group(m)
        pauli = real_pauli_group(m)
        assert_real_orthogonal_representation(clifford)
        assert_real_orthogonal_representation(pauli)
        representatives = coset_representatives(clifford, pauli)
        assert len(representatives) == quotient_order

        # Equation (11.7) must not depend on the chosen lift.  Replace every
        # representative by e_0 times that representative and compare.
        shifted = [
            pauli[index % len(pauli)] @ representative
            for index, representative in enumerate(representatives)
        ]
        for tau in PARTITIONS:
            direct = clean_integer(target_coefficient(clifford, tau))
            coset_means = coset_coefficient_means(pauli, representatives, tau)
            shifted_means = coset_coefficient_means(pauli, shifted, tau)
            lift_independent = all(
                abs(original - replacement) < TOL
                for original, replacement in zip(
                    coset_means, shifted_means, strict=True
                )
            )
            quotient_formula = clean_integer(
                sum(coset_means) / len(coset_means)
            )
            reference = stable_code_reference(m, tau)
            passed = direct == quotient_formula and lift_independent
            if reference is not None:
                passed = passed and direct == reference
            rows.append(
                {
                    "group": f"C_{m}",
                    "m": m,
                    "N": 2**m,
                    "order": len(clifford),
                    "quotient order": quotient_order,
                    "tau": str(tau),
                    "direct": direct,
                    "coset formula (11.8)": quotient_formula,
                    "every lift pair agrees": lift_independent,
                    "stable-code reference": (
                        "out of range" if reference is None else reference
                    ),
                    "passed": passed,
                }
            )

        # Normalizer sanity check on the explicit matrices.
        pauli_keys = {matrix_key(matrix) for matrix in pauli}
        for generator_like in representatives:
            inverse = generator_like.conj().T
            for element in pauli:
                assert matrix_key(generator_like @ element @ inverse) in pauli_keys

    assert all(row["passed"] for row in rows)
    return rows


if __name__ == "__main__":
    for row in run_sweep():
        print(row)
