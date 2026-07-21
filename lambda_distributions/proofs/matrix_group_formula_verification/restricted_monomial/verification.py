"""Independent checks for D(A) semidirect H and G(r,p,n)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from math import factorial

import numpy as np

from lambda_distributions.proofs.matrix_group_formula_verification.common import coefficient_rows, projector_check


Permutation = tuple[int, ...]
Codeword = tuple[int, ...]


@dataclass(frozen=True)
class RestrictedCase:
    name: str
    r: int
    code: tuple[Codeword, ...]
    permutations: tuple[Permutation, ...]
    matrices: tuple[np.ndarray, ...]


def symmetric_group(n: int) -> tuple[Permutation, ...]:
    return tuple(permutations(range(n)))


def cyclic_permutation_group(n: int) -> tuple[Permutation, ...]:
    return tuple(tuple((column + shift) % n for column in range(n)) for shift in range(n))


def constrained_code(r: int, n: int, modulus: int) -> tuple[Codeword, ...]:
    return tuple(x for x in product(range(r), repeat=n) if sum(x) % modulus == 0)


def all_code(r: int, n: int) -> tuple[Codeword, ...]:
    return tuple(product(range(r), repeat=n))


def monomial_matrices(
    r: int, code: tuple[Codeword, ...], permutation_group: tuple[Permutation, ...]
) -> tuple[np.ndarray, ...]:
    n = len(code[0])
    root = np.exp(2j * np.pi / r)
    matrices = []
    for permutation in permutation_group:
        for x in code:
            matrix = np.zeros((n, n), dtype=complex)
            for column, row in enumerate(permutation):
                matrix[row, column] = root ** x[row]
            matrices.append(matrix)
    return tuple(matrices)


def make_case(
    name: str,
    r: int,
    code: tuple[Codeword, ...],
    permutation_group: tuple[Permutation, ...],
) -> RestrictedCase:
    return RestrictedCase(name, r, code, permutation_group, monomial_matrices(r, code, permutation_group))


def cycles(permutation: Permutation) -> tuple[tuple[int, ...], ...]:
    unseen = set(range(len(permutation)))
    answer = []
    while unseen:
        start = min(unseen)
        cycle = []
        current = start
        while current in unseen:
            unseen.remove(current)
            cycle.append(current)
            current = permutation[current]
        answer.append(tuple(cycle))
    return tuple(answer)


def _cycle_assignments(cycle_lengths: tuple[int, ...], tau: tuple[int, ...]):
    """Yield arrays m[C,a] satisfying sum_C |C|m[C,a] = tau[a]."""

    if not tau:
        yield tuple(() for _ in cycle_lengths)
        return

    choices_by_variable = []
    for degree in tau:
        valid = []
        for assignment in product(*(range(degree // length + 1) for length in cycle_lengths)):
            if sum(length * value for length, value in zip(cycle_lengths, assignment)) == degree:
                valid.append(assignment)
        choices_by_variable.append(tuple(valid))
    for by_variable in product(*choices_by_variable):
        yield tuple(
            tuple(by_variable[a][cycle_index] for a in range(len(tau)))
            for cycle_index in range(len(cycle_lengths))
        )


def dual_cycle_formula(case: RestrictedCase, tau: tuple[int, ...]) -> int:
    """Coefficient in equation (5), using only cycles and code orthogonality."""

    total = 0
    for permutation in case.permutations:
        cycle_list = cycles(permutation)
        lengths = tuple(map(len, cycle_list))
        for multiplicities in _cycle_assignments(lengths, tau):
            exponent = [0] * len(permutation)
            for cycle, row in zip(cycle_list, multiplicities):
                residue = sum(row) % case.r
                for coordinate in cycle:
                    exponent[coordinate] = residue
            if all(
                sum(x_i * e_i for x_i, e_i in zip(codeword, exponent)) % case.r == 0
                for codeword in case.code
            ):
                total += 1
    quotient, remainder = divmod(total, len(case.permutations))
    if remainder:
        raise ArithmeticError("cycle formula did not give an integer")
    return quotient


def g_r_p_n_formula(r: int, p: int, n: int, tau: tuple[int, ...]) -> int:
    """Coefficient of equation (10), counted as multisets of residue labels."""

    zero = (0,) * len(tau)
    answer = 0
    all_labels = tuple(product(*(range(degree + 1) for degree in tau))) if tau else ((),)
    for q in range(p):
        residue = q * r // p
        labels = tuple(label for label in all_labels if sum(label) % r == residue)
        states = {(0, zero): 1}
        for label in labels:
            updated = {}
            for (used, degrees), count in states.items():
                for copies in range(n - used + 1):
                    candidate = tuple(a + copies * b for a, b in zip(degrees, label))
                    if all(candidate[index] <= tau[index] for index in range(len(tau))):
                        key = (used + copies, candidate)
                        updated[key] = updated.get(key, 0) + count
            states = updated
        answer += states.get((n, tau), 0)
    return answer


def g_r_p_n_case(r: int, p: int, n: int) -> RestrictedCase:
    if r % p:
        raise ValueError("p must divide r")
    code = constrained_code(r, n, p)
    case = make_case(f"G({r},{p},{n})", r, code, symmetric_group(n))
    expected = r**n * factorial(n) // p
    if len(case.matrices) != expected:
        raise AssertionError((len(case.matrices), expected))
    return case


def representative_cases() -> tuple[tuple[RestrictedCase, str], ...]:
    parity_cyclic = make_case(
        "even-sign code semidirect C3",
        2,
        constrained_code(2, 3, 2),
        cyclic_permutation_group(3),
    )
    return (
        (parity_cyclic, "dual-code cycle formula"),
        (g_r_p_n_case(2, 1, 2), "G(r,p,n) coefficient formula"),
        (g_r_p_n_case(2, 2, 3), "G(r,p,n) coefficient formula"),
        (g_r_p_n_case(3, 1, 2), "G(r,p,n) coefficient formula"),
        (g_r_p_n_case(4, 2, 2), "G(r,p,n) coefficient formula"),
        (g_r_p_n_case(4, 4, 2), "G(r,p,n) coefficient formula"),
    )


def verify_case(case: RestrictedCase, maximum_degree: int = 6):
    if case.name.startswith("G("):
        r, p, n = (int(value) for value in case.name[2:-1].split(","))
        formula = lambda tau: g_r_p_n_formula(r, p, n, tau)
    else:
        formula = lambda tau: dual_cycle_formula(case, tau)
    return coefficient_rows(case.matrices, formula, maximum_degree)


def run_suite(maximum_degree: int = 6):
    results = []
    for case, formula_name in representative_cases():
        rows = verify_case(case, maximum_degree)
        results.append(
            {
                "case": case.name,
                "order": len(case.matrices),
                "dimension": case.matrices[0].shape[0],
                "formula": formula_name,
                "checks": len(rows),
                "passed": all(row["pass"] for row in rows),
                "rows": rows,
            }
        )
    return tuple(results)


if __name__ == "__main__":
    suite = run_suite()
    for result in suite:
        print(
            f"{'PASS' if result['passed'] else 'FAIL'} {result['case']}: "
            f"order={result['order']}, coefficients={result['checks']}"
        )
    d3 = g_r_p_n_case(2, 2, 3)
    print("D3 projector:", projector_check(d3.matrices, (2, 1)))
    raise SystemExit(0 if all(result["passed"] for result in suite) else 1)
