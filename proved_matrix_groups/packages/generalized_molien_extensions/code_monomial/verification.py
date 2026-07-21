"""Direct matrix checks of the code-monomial generalized Molien formula."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod

import numpy as np

from ..common import (
    all_permutations,
    matrix_average_coefficient,
    numerical_molien_average,
    partitions_through,
    permutation_matrix,
    weak_compositions,
)


@dataclass(frozen=True)
class CodeCase:
    name: str
    modulus: int
    dimension: int
    code: tuple[tuple[int, ...], ...]
    automorphisms: tuple[tuple[int, ...], ...]


def _permute_word(word: tuple[int, ...], permutation: tuple[int, ...]):
    result = [0] * len(word)
    for source, target in enumerate(permutation):
        result[target] = word[source]
    return tuple(result)


def validate_case(case: CodeCase):
    if case.modulus < 2 or case.dimension < 1:
        raise ValueError("the modulus and dimension must be positive")
    code = set(case.code)
    if (0,) * case.dimension not in code:
        raise ValueError("the code must contain zero")
    for left in code:
        for right in code:
            if tuple((a + b) % case.modulus for a, b in zip(left, right)) not in code:
                raise ValueError("the supplied words are not an additive code")
    for permutation in case.automorphisms:
        if (
            len(permutation) != case.dimension
            or any(not isinstance(value, int) for value in permutation)
            or set(permutation) != set(range(case.dimension))
        ):
            raise ValueError(
                "code automorphisms must be coordinate permutations; "
                "monomial and semilinear entries require the generalized formula"
            )
        if {_permute_word(word, permutation) for word in code} != code:
            raise ValueError("a supplied permutation does not preserve the code")


def matrices(case: CodeCase):
    root = np.exp(2j * np.pi / case.modulus)
    answer = []
    for word in case.code:
        diagonal = np.diag([root**entry for entry in word])
        for permutation in case.automorphisms:
            answer.append(diagonal @ permutation_matrix(permutation))
    return tuple(answer)


def dual_contains(case: CodeCase, word: tuple[int, ...]):
    return all(
        sum(a * b for a, b in zip(codeword, word)) % case.modulus == 0
        for codeword in case.code
    )


def orbit_formula_coefficient(case: CodeCase, tau: tuple[int, ...]) -> int:
    """Theorem 2: automorphism orbits of dual-code-constrained arrays."""

    rows = [tuple(weak_compositions(degree, case.dimension)) for degree in tau]
    allowed = set()
    for array in product(*rows) if rows else [()]:
        coordinate_totals = tuple(
            sum(row[column] for row in array) % case.modulus
            for column in range(case.dimension)
        )
        if dual_contains(case, coordinate_totals):
            allowed.add(array)

    orbits = 0
    while allowed:
        representative = next(iter(allowed))
        orbit = set()
        for permutation in case.automorphisms:
            transformed = tuple(
                tuple(row[permutation[column]] for column in range(case.dimension))
                for row in representative
            )
            orbit.add(transformed)
        allowed.difference_update(orbit)
        orbits += 1
    return orbits


def cycle_block_average(case: CodeCase, t_values=(0.07, 0.11)) -> float:
    """Numerically evaluate formula (1.1) from weighted permutation cycles."""

    root = np.exp(2j * np.pi / case.modulus)
    total = 0j
    for word in case.code:
        for permutation in case.automorphisms:
            seen = set()
            factors = []
            for start in range(case.dimension):
                if start in seen:
                    continue
                current = start
                cycle = []
                while current not in seen:
                    seen.add(current)
                    cycle.append(current)
                    current = permutation[current]
                phase = root ** (sum(word[index] for index in cycle) % case.modulus)
                factors.append((len(cycle), phase))
            total += prod(
                1 / (1 - phase * t ** length)
                for t in t_values
                for length, phase in factors
            )
    return float((total / (len(case.code) * len(case.automorphisms))).real)


def representative_cases():
    s2 = all_permutations(2)
    s3 = all_permutations(3)
    binary_full_2 = tuple(product(range(2), repeat=2))
    binary_even_3 = tuple(
        word for word in product(range(2), repeat=3) if sum(word) % 2 == 0
    )
    ternary_repetition_3 = tuple((a, a, a) for a in range(3))
    return (
        CodeCase("W(B2): full binary code", 2, 2, binary_full_2, s2),
        CodeCase("W(D3): even binary code", 2, 3, binary_even_3, s3),
        CodeCase("ternary repetition code semidirect S3", 3, 3, ternary_repetition_3, s3),
    )


def run_suite(max_degree: int = 5):
    rows = []
    numerical_rows = []
    for case in representative_cases():
        validate_case(case)
        group = matrices(case)
        assert len(group) == len(case.code) * len(case.automorphisms)
        for tau in partitions_through(max_degree):
            direct = matrix_average_coefficient(group, tau)
            proposed = orbit_formula_coefficient(case, tau)
            rows.append(
                {
                    "case": case.name,
                    "dimension": case.dimension,
                    "group order": len(group),
                    "tau": str(tau),
                    "matrix average": direct,
                    "dual-code orbit formula": proposed,
                    "pass": direct == proposed,
                }
            )
        direct_value = numerical_molien_average(group)
        block_value = cycle_block_average(case)
        numerical_rows.append(
            {
                "case": case.name,
                "direct determinant average": direct_value,
                "weighted-cycle formula": block_value,
                "absolute error": abs(direct_value - block_value),
            }
        )
    passed = all(row["pass"] for row in rows) and all(
        row["absolute error"] < 1e-10 for row in numerical_rows
    )
    return {"passed": passed, "rows": rows, "numerical": numerical_rows}


if __name__ == "__main__":
    suite = run_suite()
    print(f"code-monomial: {'PASS' if suite['passed'] else 'FAIL'} ({len(suite['rows'])} coefficients)")
    for row in suite["numerical"]:
        print(row)
