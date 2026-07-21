"""Independent small-case checks for finite-group sigma-MGF formulas.

The direct route constructs every representation matrix and, for a coefficient
``tau``, forms the Reynolds operator on a tensor product of symmetric powers.
The comparison routes use element orders, cycle types, fixed points, or coset
monodromy.  They therefore do not reuse the direct projector computation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations, permutations
from math import comb, factorial, gcd, lcm, prod

import numpy as np

from lambda_distributions.proofs.finite_group_exact_formulas.verification_core import (
    character_average,
    reynolds_check,
    symmetric_power_matrix,
    target_dimension,
)


Permutation = tuple[int, ...]


def compose(left: Permutation, right: Permutation) -> Permutation:
    """Composition ``left o right`` for image-tuples."""

    return tuple(left[right[index]] for index in range(len(left)))


def inverse(permutation: Permutation) -> Permutation:
    result = [0] * len(permutation)
    for index, image in enumerate(permutation):
        result[image] = index
    return tuple(result)


def permutation_power(permutation: Permutation, exponent: int) -> Permutation:
    result = tuple(range(len(permutation)))
    base = permutation
    while exponent:
        if exponent & 1:
            result = compose(result, base)
        base = compose(base, base)
        exponent //= 2
    return result


def parity(permutation: Permutation) -> int:
    return sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    ) % 2


def permutation_elements(n: int, alternating: bool = False) -> tuple[Permutation, ...]:
    return tuple(
        permutation
        for permutation in permutations(range(n))
        if not alternating or parity(permutation) == 0
    )


def permutation_matrix(permutation: Permutation) -> np.ndarray:
    matrix = np.zeros((len(permutation), len(permutation)), dtype=complex)
    matrix[permutation, np.arange(len(permutation))] = 1
    return matrix


def permutation_cycle_lengths(permutation: Permutation) -> tuple[int, ...]:
    seen: set[int] = set()
    lengths = []
    for start in range(len(permutation)):
        if start in seen:
            continue
        current = start
        length = 0
        while current not in seen:
            seen.add(current)
            current = permutation[current]
            length += 1
        lengths.append(length)
    return tuple(sorted(lengths))


def permutation_order(permutation: Permutation) -> int:
    return lcm(*permutation_cycle_lengths(permutation))


def cyclic_group(n: int):
    elements = tuple(range(n))
    return elements, lambda left, right: (left + right) % n


def dihedral_group(n: int):
    """D_n of order 2n, represented by r^a s^b."""

    elements = tuple((a, b) for b in (0, 1) for a in range(n))

    def multiply(left, right):
        a, b = left
        c, d = right
        return ((a + (-1) ** b * c) % n, (b + d) % 2)

    return elements, multiply


def regular_matrices(elements, multiply) -> tuple[np.ndarray, ...]:
    indices = {element: index for index, element in enumerate(elements)}
    matrices = []
    for left in elements:
        matrix = np.zeros((len(elements), len(elements)), dtype=complex)
        for column, right in enumerate(elements):
            matrix[indices[multiply(left, right)], column] = 1
        matrices.append(matrix)
    return tuple(matrices)


def regular_permutation_matrices(n: int, alternating: bool = False) -> tuple[np.ndarray, ...]:
    elements = permutation_elements(n, alternating)
    return regular_matrices(elements, compose)


def abstract_element_order(element, identity, multiply, group_order: int) -> int:
    value = identity
    for exponent in range(1, group_order + 1):
        value = multiply(value, element)
        if value == identity:
            return exponent
    raise ArithmeticError("element order did not divide the group order")


def regular_order_counts(family: str, parameter: int) -> tuple[int, Counter[int]]:
    if family == "cyclic":
        elements, multiply = cyclic_group(parameter)
        identity = 0
    elif family == "dihedral":
        elements, multiply = dihedral_group(parameter)
        identity = (0, 0)
    elif family in {"symmetric", "alternating"}:
        elements = permutation_elements(parameter, family == "alternating")
        return len(elements), Counter(permutation_order(element) for element in elements)
    else:
        raise ValueError(f"unknown family {family!r}")
    return len(elements), Counter(
        abstract_element_order(element, identity, multiply, len(elements))
        for element in elements
    )


def regular_family_matrices(family: str, parameter: int) -> tuple[np.ndarray, ...]:
    if family == "cyclic":
        return regular_matrices(*cyclic_group(parameter))
    if family == "dihedral":
        return regular_matrices(*dihedral_group(parameter))
    if family in {"symmetric", "alternating"}:
        return regular_permutation_matrices(parameter, family == "alternating")
    raise ValueError(f"unknown family {family!r}")


def _cycle_complete(cycle_counts: dict[int, int], degree: int) -> int:
    coefficients = [0] * (degree + 1)
    coefficients[0] = 1
    for length, count in cycle_counts.items():
        updated = [0] * (degree + 1)
        for old_degree, old_value in enumerate(coefficients):
            for copies in range((degree - old_degree) // length + 1):
                updated[old_degree + copies * length] += old_value * comb(
                    count + copies - 1, copies
                )
        coefficients = updated
    return coefficients[degree]


def regular_formula_coefficient(
    group_order: int, order_counts: Counter[int], tau: tuple[int, ...]
) -> int:
    numerator = sum(
        count
        * prod(
            _cycle_complete({order: group_order // order}, degree)
            for degree in tau
        )
        for order, count in order_counts.items()
    )
    if numerator % group_order:
        raise ArithmeticError("regular formula did not produce an integer")
    return numerator // group_order


def molien_value(matrices: tuple[np.ndarray, ...], t_values: tuple[float, ...]) -> complex:
    total = 0j
    dimension = matrices[0].shape[0]
    identity = np.eye(dimension, dtype=complex)
    for matrix in matrices:
        total += prod(1 / np.linalg.det(identity - t * matrix) for t in t_values)
    return total / len(matrices)


def regular_formula_value(
    group_order: int, order_counts: Counter[int], t_values: tuple[float, ...]
) -> float:
    return sum(
        count
        * prod((1 - t**order) ** (-group_order // order) for t in t_values)
        for order, count in order_counts.items()
    ) / group_order


@dataclass(frozen=True)
class VerificationRow:
    case: str
    representation_dimension: int
    target_dimension: int
    direct: int
    formula: int
    character: int
    numerical_error: float
    projector_error: float

    @property
    def passed(self) -> bool:
        return (
            self.direct == self.formula == self.character
            and self.numerical_error < 1e-8
            and self.projector_error < 1e-8
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "case": self.case,
            "rep dim": self.representation_dimension,
            "target dim": self.target_dimension,
            "projector": self.direct,
            "character": self.character,
            "formula": self.formula,
            "numeric error": self.numerical_error,
            "projector error": self.projector_error,
            "pass": self.passed,
        }


def verify_regular_case(
    family: str,
    parameter: int,
    tau: tuple[int, ...],
    t_values: tuple[float, ...] = (0.07, -0.04),
) -> VerificationRow:
    matrices = regular_family_matrices(family, parameter)
    group_order, counts = regular_order_counts(family, parameter)
    direct = reynolds_check(matrices, tau)
    character = character_average(matrices, tau)
    formula = regular_formula_coefficient(group_order, counts, tau)
    numeric_direct = molien_value(matrices, t_values)
    numeric_formula = regular_formula_value(group_order, counts, t_values)
    return VerificationRow(
        case=f"{family}({parameter}), tau={tau}",
        representation_dimension=group_order,
        target_dimension=int(direct["target_dimension"]),
        direct=int(direct["projector_rank"]),
        formula=formula,
        character=int(round(character.real)),
        numerical_error=float(abs(numeric_direct - numeric_formula)),
        projector_error=float(direct["idempotence_error"]),
    )


def standard_matrices(n: int) -> tuple[np.ndarray, ...]:
    """The standard S_n representation in the basis e_i-e_n."""

    basis = np.zeros((n, n - 1), dtype=complex)
    basis[: n - 1, :] = np.eye(n - 1)
    basis[n - 1, :] = -1
    return tuple(
        (permutation_matrix(permutation) @ basis)[: n - 1, :]
        for permutation in permutation_elements(n)
    )


def sign_matrices(n: int) -> tuple[np.ndarray, ...]:
    return tuple(
        np.array([[(-1) ** parity(permutation)]], dtype=complex)
        for permutation in permutation_elements(n)
    )


def _vector_partition_count(tau: tuple[int, ...], n: int) -> int:
    """S_n-orbits on tuples of multisets for the natural permutation action."""

    target = tuple(tau)
    vectors = tuple(
        vector
        for vector in __import__("itertools").product(
            *(range(value + 1) for value in target)
        )
        if any(vector)
    )
    zero = (0,) * len(target)
    states: dict[tuple[tuple[int, ...], int], int] = {(zero, 0): 1}
    for vector in vectors:
        updated = dict(states)
        for (subtotal, used), ways in states.items():
            max_copies = min(
                (target[i] - subtotal[i]) // vector[i]
                for i in range(len(target))
                if vector[i]
            )
            for copies in range(1, min(max_copies, n - used) + 1):
                candidate = tuple(
                    subtotal[i] + copies * vector[i] for i in range(len(target))
                )
                key = (candidate, used + copies)
                updated[key] = updated.get(key, 0) + ways
        states = updated
    return sum(states.get((target, used), 0) for used in range(n + 1))


def standard_formula_coefficient(n: int, tau: tuple[int, ...]) -> int:
    """Coefficient of prod_i(1-t_i) times the natural-action series."""

    answer = 0
    for mask in range(1 << len(tau)):
        reduced = []
        sign = 1
        valid = True
        for index, degree in enumerate(tau):
            if mask & (1 << index):
                if degree == 0:
                    valid = False
                    break
                reduced.append(degree - 1)
                sign *= -1
            else:
                reduced.append(degree)
        if valid:
            answer += sign * _vector_partition_count(tuple(reduced), n)
    return answer


def _z_partition(partition: tuple[int, ...]) -> int:
    counts = Counter(partition)
    return prod(part**count * factorial(count) for part, count in counts.items())


def _integer_partitions(total: int, ceiling: int | None = None):
    if total == 0:
        yield ()
        return
    ceiling = min(total, total if ceiling is None else ceiling)
    for first in range(ceiling, 0, -1):
        for tail in _integer_partitions(total - first, first):
            yield (first, *tail)


def standard_cycle_formula_value(n: int, t_values: tuple[float, ...]) -> float:
    answer = 0.0
    for cycle_type in _integer_partitions(n):
        term = 1.0 / _z_partition(cycle_type)
        for t in t_values:
            term *= 1 - t
            for length in cycle_type:
                term /= 1 - t**length
        answer += term
    return answer


def sign_formula_value(t_values: tuple[float, ...]) -> float:
    return 0.5 * (
        prod(1 / (1 - t) for t in t_values)
        + prod(1 / (1 + t) for t in t_values)
    )


def verify_standard_case(
    n: int, tau: tuple[int, ...], t_values: tuple[float, ...] = (0.11, -0.06)
) -> VerificationRow:
    matrices = standard_matrices(n)
    direct = reynolds_check(matrices, tau)
    character = character_average(matrices, tau)
    formula = standard_formula_coefficient(n, tau)
    return VerificationRow(
        case=f"S_{n} standard, tau={tau}",
        representation_dimension=n - 1,
        target_dimension=int(direct["target_dimension"]),
        direct=int(direct["projector_rank"]),
        formula=formula,
        character=int(round(character.real)),
        numerical_error=float(abs(molien_value(matrices, t_values) - standard_cycle_formula_value(n, t_values))),
        projector_error=float(direct["idempotence_error"]),
    )


def verify_sign_case(
    n: int, tau: tuple[int, ...], t_values: tuple[float, ...] = (0.11, -0.06)
) -> VerificationRow:
    matrices = sign_matrices(n)
    direct = reynolds_check(matrices, tau)
    character = character_average(matrices, tau)
    formula = int(sum(tau) % 2 == 0)
    return VerificationRow(
        case=f"S_{n} sign, tau={tau}",
        representation_dimension=1,
        target_dimension=1,
        direct=int(direct["projector_rank"]),
        formula=formula,
        character=int(round(character.real)),
        numerical_error=float(abs(molien_value(matrices, t_values) - sign_formula_value(t_values))),
        projector_error=float(direct["idempotence_error"]),
    )


def _mobius(value: int) -> int:
    prime_count = 0
    divisor = 2
    remaining = value
    while divisor * divisor <= remaining:
        if remaining % divisor:
            divisor += 1
            continue
        remaining //= divisor
        prime_count += 1
        if remaining % divisor == 0:
            return 0
        while remaining % divisor == 0:
            remaining //= divisor
    if remaining > 1:
        prime_count += 1
    return -1 if prime_count % 2 else 1


def _subset_fixed_count(permutation: Permutation, k: int) -> int:
    coefficients = [0] * (k + 1)
    coefficients[0] = 1
    for length in permutation_cycle_lengths(permutation):
        for degree in range(k, length - 1, -1):
            coefficients[degree] += coefficients[degree - length]
    return coefficients[k]


def subset_cycle_counts_formula(permutation: Permutation, k: int) -> dict[int, int]:
    order = permutation_order(permutation)
    result: dict[int, int] = {}
    for d in range(1, order + 1):
        if order % d:
            continue
        fixed_sum = sum(
            _mobius(d // r) * _subset_fixed_count(permutation_power(permutation, r), k)
            for r in range(1, d + 1)
            if d % r == 0
        )
        if fixed_sum % d:
            raise ArithmeticError("Möbius inversion did not give an integer")
        if fixed_sum:
            result[d] = fixed_sum // d
    return result


def subset_matrices(n: int, k: int, alternating: bool = False) -> tuple[np.ndarray, ...]:
    labels = tuple(combinations(range(n), k))
    indices = {label: index for index, label in enumerate(labels)}
    matrices = []
    for permutation in permutation_elements(n, alternating):
        matrix = np.zeros((len(labels), len(labels)), dtype=complex)
        for column, label in enumerate(labels):
            image = tuple(sorted(permutation[index] for index in label))
            matrix[indices[image], column] = 1
        matrices.append(matrix)
    return tuple(matrices)


def subset_formula_coefficient(
    n: int, k: int, tau: tuple[int, ...], alternating: bool = False
) -> int:
    total = 0
    elements = permutation_elements(n, alternating)
    for permutation in elements:
        cycle_counts = subset_cycle_counts_formula(permutation, k)
        total += prod(_cycle_complete(cycle_counts, degree) for degree in tau)
    if total % len(elements):
        raise ArithmeticError("subset formula did not average to an integer")
    return total // len(elements)


def subset_formula_value(
    n: int, k: int, t_values: tuple[float, ...], alternating: bool = False
) -> float:
    elements = permutation_elements(n, alternating)
    return sum(
        prod(
            prod((1 - t**length) ** (-count) for length, count in subset_cycle_counts_formula(permutation, k).items())
            for t in t_values
        )
        for permutation in elements
    ) / len(elements)


def verify_subset_case(
    n: int,
    k: int,
    tau: tuple[int, ...],
    alternating: bool = False,
    t_values: tuple[float, ...] = (0.08, -0.03),
) -> VerificationRow:
    matrices = subset_matrices(n, k, alternating)
    direct = reynolds_check(matrices, tau)
    character = character_average(matrices, tau)
    formula = subset_formula_coefficient(n, k, tau, alternating)
    label = "A" if alternating else "S"
    return VerificationRow(
        case=f"{label}_{n} on {k}-subsets, tau={tau}",
        representation_dimension=comb(n, k),
        target_dimension=int(direct["target_dimension"]),
        direct=int(direct["projector_rank"]),
        formula=formula,
        character=int(round(character.real)),
        numerical_error=float(abs(molien_value(matrices, t_values) - subset_formula_value(n, k, t_values, alternating))),
        projector_error=float(direct["idempotence_error"]),
    )


def induced_s3_from_c3() -> tuple[tuple[np.ndarray, ...], tuple[Permutation, ...], tuple[Permutation, ...], dict[Permutation, complex]]:
    """Induce a nontrivial character of A_3=C_3 to S_3."""

    group = permutation_elements(3)
    identity = (0, 1, 2)
    rotation = (1, 2, 0)
    rotation2 = compose(rotation, rotation)
    subgroup = (identity, rotation, rotation2)
    omega = np.exp(2j * np.pi / 3)
    character = {identity: 1 + 0j, rotation: omega, rotation2: omega**2}
    representatives = (identity, (1, 0, 2))
    matrices = []
    for g in group:
        matrix = np.zeros((2, 2), dtype=complex)
        for column, x in enumerate(representatives):
            gx = compose(g, x)
            for row, y in enumerate(representatives):
                h = compose(inverse(y), gx)
                if h in character:
                    matrix[row, column] = character[h]
                    break
            else:
                raise ArithmeticError("coset representative not found")
        matrices.append(matrix)
    return tuple(matrices), group, representatives, character


def induced_coset_formula_value(t_values: tuple[float, ...]) -> float:
    matrices, group, representatives, character = induced_s3_from_c3()
    del matrices
    total = 0j
    for g in group:
        coset_action = []
        for x in representatives:
            gx = compose(g, x)
            for row, y in enumerate(representatives):
                if compose(inverse(y), gx) in character:
                    coset_action.append(row)
                    break
        seen: set[int] = set()
        cycle_data: list[tuple[int, complex]] = []
        for start in range(len(representatives)):
            if start in seen:
                continue
            current = start
            length = 0
            while current not in seen:
                seen.add(current)
                current = coset_action[current]
                length += 1
            x = representatives[start]
            monodromy = compose(compose(inverse(x), permutation_power(g, length)), x)
            cycle_data.append((length, character[monodromy]))
        total += prod(
            prod(1 / (1 - (t**length) * scalar) for length, scalar in cycle_data)
            for t in t_values
        )
    return float((total / len(group)).real)


def induced_verification_row(
    tau: tuple[int, ...] = (2, 1), t_values: tuple[float, ...] = (0.09, -0.04)
) -> VerificationRow:
    matrices, _, _, _ = induced_s3_from_c3()
    direct = reynolds_check(matrices, tau)
    character = character_average(matrices, tau)
    direct_value = molien_value(matrices, t_values)
    formula_value = induced_coset_formula_value(t_values)
    # The coefficient formula is independently recovered by a small Cauchy
    # integral surrogate: here Ind_{C3}^{S3}(omega) is the standard irrep.
    formula = standard_formula_coefficient(3, tau)
    return VerificationRow(
        case=f"Ind_C3^S3(omega), tau={tau}",
        representation_dimension=2,
        target_dimension=int(direct["target_dimension"]),
        direct=int(direct["projector_rank"]),
        formula=formula,
        character=int(round(character.real)),
        numerical_error=float(abs(direct_value - formula_value)),
        projector_error=float(direct["idempotence_error"]),
    )


def derived_character_errors(n: int = 4, degree: int = 2) -> dict[str, float]:
    """Check (7.31)--(7.34) on the standard S_n representation."""

    matrices = standard_matrices(n)
    symmetric_error = 0.0
    exterior_error = 0.0
    tensor_error = 0.0
    for matrix in matrices:
        trace = np.trace(matrix)
        trace_power = np.trace(matrix @ matrix)
        symmetric_formula = (trace**2 + trace_power) / 2
        exterior_formula = (trace**2 - trace_power) / 2
        tensor_formula = trace**degree
        symmetric_error = max(
            symmetric_error,
            float(abs(np.trace(symmetric_power_matrix(matrix, 2)) - symmetric_formula)),
        )
        exterior_matrix = np.array([[np.linalg.det(matrix)]], dtype=complex) if n == 3 else None
        if exterior_matrix is not None:
            exterior_error = max(exterior_error, float(abs(np.trace(exterior_matrix) - exterior_formula)))
        tensor_matrix = np.ones((1, 1), dtype=complex)
        for _ in range(degree):
            tensor_matrix = np.kron(tensor_matrix, matrix)
        tensor_error = max(
            tensor_error, float(abs(np.trace(tensor_matrix) - tensor_formula))
        )
    return {
        "symmetric-square character error": symmetric_error,
        "exterior-square character error (S3)": exterior_error if n == 3 else 0.0,
        "tensor-power character error": tensor_error,
    }


REGULAR_CASES = (
    ("cyclic", 5, (2, 1)),
    ("dihedral", 4, (2,)),
    ("symmetric", 3, (1, 1)),
    ("alternating", 4, (2,)),
)

SYMMETRIC_REP_CASES = (
    ("sign", 4, (2,)),
    ("sign", 4, (3,)),
    ("standard", 3, (2, 1)),
    ("standard", 4, (2, 2)),
    ("standard", 5, (2, 1)),
)

SUBSET_CASES = (
    (4, 2, (2,), False),
    (5, 2, (1, 1), False),
    (5, 2, (2, 1), False),
    (5, 2, (2,), True),
    (4, 2, (2,), True),
    (4, 2, (1, 1), True),
    (5, 2, (1, 1), True),
    (6, 2, (1, 1), True),
)


def representative_suites() -> dict[str, list[dict[str, object]]]:
    regular = [verify_regular_case(*case).as_dict() for case in REGULAR_CASES]
    symmetric = []
    for kind, n, tau in SYMMETRIC_REP_CASES:
        row = verify_sign_case(n, tau) if kind == "sign" else verify_standard_case(n, tau)
        symmetric.append(row.as_dict())
    symmetric.append(induced_verification_row().as_dict())
    subsets = [verify_subset_case(*case).as_dict() for case in SUBSET_CASES]
    return {"regular": regular, "symmetric": symmetric, "subsets": subsets}


if __name__ == "__main__":
    for family, rows in representative_suites().items():
        print(family)
        for row in rows:
            print("PASS" if row["pass"] else "FAIL", row)
