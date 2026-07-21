"""Exact and numerical checks for cyclic-alphabet code-monomial groups.

The direct route constructs every dense complex matrix ``D(c) P_pi``.  It
compares matrix determinant averages and symmetric-power trace averages with
two independent combinatorial routes: weighted coordinate cycles and orbits
of dual-code-constrained exponent arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from math import prod

import numpy as np


Word = tuple[int, ...]
Permutation = tuple[int, ...]


@dataclass(frozen=True)
class CodeCase:
    name: str
    family: str
    modulus: int
    dimension: int
    code: tuple[Word, ...]
    automorphisms: tuple[Permutation, ...]


def _compose(left: Permutation, right: Permutation) -> Permutation:
    """Composition matching permutation-matrix multiplication."""

    return tuple(left[right[index]] for index in range(len(right)))


def _permute_word(word: Word, permutation: Permutation) -> Word:
    answer = [0] * len(word)
    for source, target in enumerate(permutation):
        answer[target] = word[source]
    return tuple(answer)


def _permutation_matrix(permutation: Permutation) -> np.ndarray:
    matrix = np.zeros((len(permutation), len(permutation)), dtype=complex)
    for source, target in enumerate(permutation):
        matrix[target, source] = 1
    return matrix


def _cyclic_group(generator: Permutation) -> tuple[Permutation, ...]:
    identity = tuple(range(len(generator)))
    answer = []
    current = identity
    while current not in answer:
        answer.append(current)
        current = _compose(generator, current)
    return tuple(answer)


def validate_case(case: CodeCase) -> None:
    if case.modulus < 2 or case.dimension < 1:
        raise ValueError("modulus must be at least two and dimension positive")
    code = set(case.code)
    zero = (0,) * case.dimension
    if zero not in code or any(len(word) != case.dimension for word in code):
        raise ValueError("code words must have the declared dimension and include zero")
    for left in code:
        for right in code:
            total = tuple((a + b) % case.modulus for a, b in zip(left, right, strict=True))
            if total not in code:
                raise ValueError("the supplied words are not an additive code")

    automorphisms = set(case.automorphisms)
    identity = tuple(range(case.dimension))
    if identity not in automorphisms:
        raise ValueError("the automorphisms must contain the identity")
    for permutation in automorphisms:
        if len(permutation) != case.dimension or set(permutation) != set(range(case.dimension)):
            raise ValueError("automorphisms must be coordinate permutations")
        if {_permute_word(word, permutation) for word in code} != code:
            raise ValueError("an automorphism does not preserve the code")
    for left in automorphisms:
        for right in automorphisms:
            if _compose(left, right) not in automorphisms:
                raise ValueError("the supplied automorphisms are not a group")


def dual_code(case: CodeCase) -> tuple[Word, ...]:
    return tuple(
        word
        for word in product(range(case.modulus), repeat=case.dimension)
        if all(
            sum(a * b for a, b in zip(codeword, word, strict=True)) % case.modulus == 0
            for codeword in case.code
        )
    )


def matrix_group(case: CodeCase) -> tuple[np.ndarray, ...]:
    root = np.exp(2j * np.pi / case.modulus)
    return tuple(
        np.diag([root**entry for entry in word]) @ _permutation_matrix(permutation)
        for word in case.code
        for permutation in case.automorphisms
    )


def matrix_group_audit(case: CodeCase, matrices: tuple[np.ndarray, ...]) -> dict[str, object]:
    identity = np.eye(case.dimension, dtype=complex)
    unitary_error = max(float(np.max(np.abs(matrix.conj().T @ matrix - identity))) for matrix in matrices)
    unique = {
        tuple((round(value.real, 12), round(value.imag, 12)) for value in matrix.flat)
        for matrix in matrices
    }
    return {
        "expected order": len(case.code) * len(case.automorphisms),
        "constructed matrices": len(matrices),
        "distinct matrices": len(unique),
        "maximum unitarity error": unitary_error,
        "pass": len(unique) == len(matrices) and unitary_error < 1e-12,
    }


def partitions_through(maximum: int) -> tuple[tuple[int, ...], ...]:
    def partitions(total: int, cap: int | None = None):
        if total == 0:
            yield ()
            return
        cap = total if cap is None else min(cap, total)
        for first in range(cap, 0, -1):
            for rest in partitions(total - first, first):
                yield (first,) + rest

    return tuple(partition for total in range(maximum + 1) for partition in partitions(total))


def weak_compositions(total: int, length: int):
    if length == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in weak_compositions(total - first, length - 1):
            yield (first,) + rest


def _symmetric_character(matrix: np.ndarray, degree: int) -> complex:
    """Compute Tr(Sym^degree(matrix)) from traces of actual matrix powers."""

    complete = [1 + 0j]
    power = np.eye(matrix.shape[0], dtype=complex)
    traces = [0j]
    for _ in range(degree):
        power = power @ matrix
        traces.append(np.trace(power))
    for current_degree in range(1, degree + 1):
        complete.append(
            sum(traces[k] * complete[current_degree - k] for k in range(1, current_degree + 1))
            / current_degree
        )
    return complete[degree]


def direct_matrix_coefficient(matrices: tuple[np.ndarray, ...], tau: tuple[int, ...]):
    value = sum(
        prod(_symmetric_character(matrix, degree) for degree in tau)
        for matrix in matrices
    ) / len(matrices)
    rounded = int(round(value.real))
    return rounded, float(abs(value - rounded))


def orbit_formula_coefficient(case: CodeCase, tau: tuple[int, ...]) -> int:
    """Count P-orbits of dual-code-constrained exponent arrays exactly."""

    allowed = set()
    rows = tuple(tuple(weak_compositions(degree, case.dimension)) for degree in tau)
    for array in product(*rows) if rows else [()]:
        residue = tuple(
            sum(row[column] for row in array) % case.modulus
            for column in range(case.dimension)
        )
        if all(
            sum(a * b for a, b in zip(codeword, residue, strict=True)) % case.modulus == 0
            for codeword in case.code
        ):
            allowed.add(array)

    number_of_orbits = 0
    while allowed:
        representative = next(iter(allowed))
        orbit = {
            tuple(
                tuple(row[permutation[column]] for column in range(case.dimension))
                for row in representative
            )
            for permutation in case.automorphisms
        }
        allowed.difference_update(orbit)
        number_of_orbits += 1
    return number_of_orbits


def _cycles(permutation: Permutation) -> tuple[tuple[int, ...], ...]:
    seen: set[int] = set()
    answer = []
    for start in range(len(permutation)):
        if start in seen:
            continue
        cycle = []
        current = start
        while current not in seen:
            seen.add(current)
            cycle.append(current)
            current = permutation[current]
        answer.append(tuple(cycle))
    return tuple(answer)


def direct_determinant_average(case: CodeCase, t_values: tuple[float, ...]) -> complex:
    identity = np.eye(case.dimension, dtype=complex)
    return sum(
        prod(1 / np.linalg.det(identity - t * matrix) for t in t_values)
        for matrix in matrix_group(case)
    ) / (len(case.code) * len(case.automorphisms))


def weighted_cycle_average(case: CodeCase, t_values: tuple[float, ...]) -> complex:
    root = np.exp(2j * np.pi / case.modulus)
    total = 0j
    for word in case.code:
        for permutation in case.automorphisms:
            factors = tuple(
                (len(cycle), root ** (sum(word[index] for index in cycle) % case.modulus))
                for cycle in _cycles(permutation)
            )
            total += prod(
                1 / (1 - phase * t**length)
                for t in t_values
                for length, phase in factors
            )
    return total / (len(case.code) * len(case.automorphisms))


def residue_series(modulus: int, residue: int, powers: tuple[float, ...]) -> complex:
    root = np.exp(2j * np.pi / modulus)
    return sum(
        root ** (-residue * character)
        * prod(1 / (1 - root**character * value) for value in powers)
        for character in range(modulus)
    ) / modulus


def dual_cycle_average(case: CodeCase, t_values: tuple[float, ...]) -> complex:
    dual = dual_code(case)
    total = 0j
    for permutation in case.automorphisms:
        cycles = _cycles(permutation)
        for word in dual:
            if all(len({word[index] for index in cycle}) == 1 for cycle in cycles):
                total += prod(
                    residue_series(
                        case.modulus,
                        word[cycle[0]],
                        tuple(t**len(cycle) for t in t_values),
                    )
                    for cycle in cycles
                )
    return total / len(case.automorphisms)


def representative_cases() -> tuple[CodeCase, ...]:
    identity_2 = ((0, 1),)
    s2 = tuple(permutations(range(2)))
    s3 = tuple(permutations(range(3)))
    full_binary_2 = tuple(product(range(2), repeat=2))
    even_binary_3 = tuple(word for word in product(range(2), repeat=3) if sum(word) % 2 == 0)
    ternary_repetition_3 = tuple((value,) * 3 for value in range(3))
    quaternary_pair_4 = tuple((left, right, left, right) for left in range(4) for right in range(4))
    c4 = _cyclic_group((1, 2, 3, 0))
    return (
        CodeCase("D_(Z/2)^2 (pure diagonal)", "Pure diagonal", 2, 2, full_binary_2, identity_2),
        CodeCase("W(B2) = D_(Z/2)^2 semidirect S2", "Binary semidirect", 2, 2, full_binary_2, s2),
        CodeCase("W(D3) = D_even semidirect S3", "Binary semidirect", 2, 3, even_binary_3, s3),
        CodeCase("D_rep(Z/3) semidirect S3", "Ternary semidirect", 3, 3, ternary_repetition_3, s3),
        CodeCase("D_pair(Z/4) semidirect C4", "Composite cyclic alphabet", 4, 4, quaternary_pair_4, c4),
    )


def run_case(case: CodeCase, max_degree: int = 5, t_values=(0.07, 0.11)) -> dict[str, object]:
    validate_case(case)
    matrices = matrix_group(case)
    audit = matrix_group_audit(case, matrices)
    coefficient_rows = []
    for tau in partitions_through(max_degree):
        direct, numerical_error = direct_matrix_coefficient(matrices, tau)
        formula = orbit_formula_coefficient(case, tau)
        coefficient_rows.append(
            {
                "group": case.name,
                "tau": str(tau),
                "matrix trace average": direct,
                "dual-code orbit formula": formula,
                "rounding error": numerical_error,
                "pass": direct == formula and numerical_error < 1e-9,
            }
        )

    direct_value = direct_determinant_average(case, tuple(t_values))
    weighted_value = weighted_cycle_average(case, tuple(t_values))
    dual_value = dual_cycle_average(case, tuple(t_values))
    determinant_row = {
        "group": case.name,
        "direct matrix average": float(direct_value.real),
        "weighted-cycle average": float(weighted_value.real),
        "dual-cycle H average": float(dual_value.real),
        "weighted error": float(abs(direct_value - weighted_value)),
        "dual error": float(abs(direct_value - dual_value)),
    }
    determinant_row["pass"] = determinant_row["weighted error"] < 1e-10 and determinant_row["dual error"] < 1e-10
    return {
        "name": case.name,
        "family": case.family,
        "modulus": case.modulus,
        "dimension": case.dimension,
        "code order": len(case.code),
        "permutation order": len(case.automorphisms),
        "group order": len(matrices),
        "dual order": len(dual_code(case)),
        "matrix audit": audit,
        "coefficient rows": tuple(coefficient_rows),
        "determinant row": determinant_row,
        "passed": audit["pass"]
        and all(row["pass"] for row in coefficient_rows)
        and determinant_row["pass"],
    }


def run_suite(max_degree: int = 5) -> dict[str, object]:
    cases = tuple(run_case(case, max_degree=max_degree) for case in representative_cases())
    rows = tuple(row for case in cases for row in case["coefficient rows"])
    return {
        "passed": all(case["passed"] for case in cases),
        "maximum degree": max_degree,
        "coefficient comparisons": len(rows),
        "groups": len(cases),
        "cases": cases,
        "rows": rows,
        "determinant rows": tuple(case["determinant row"] for case in cases),
        "maximum coefficient rounding error": max(row["rounding error"] for row in rows),
        "maximum determinant error": max(
            max(row["weighted error"], row["dual error"])
            for row in (case["determinant row"] for case in cases)
        ),
    }


if __name__ == "__main__":
    suite = run_suite()
    print(
        f"code-monomial: {'PASS' if suite['passed'] else 'FAIL'}; "
        f"{suite['coefficient comparisons']} coefficients, "
        f"max determinant error {suite['maximum determinant error']:.3e}"
    )
    for row in suite["determinant rows"]:
        print(row)
