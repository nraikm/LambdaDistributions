"""Exact and matrix-level checks for compact Spin/Pin sigma-MGFs.

The exact route uses integer coordinates equal to twice the usual orthonormal
weight coordinates.  It evaluates Weyl integrals with the alternant identity

    (1 / |W|) CT(D_Phi f) = sum_w det(w) [x^(rho-w rho)] f(x)

for a Weyl-invariant Laurent character ``f``.  The matrix route constructs
Clifford gamma matrices, exponentiates the bivector generators, and inspects
quadratic Casimirs on tensor squares.  The two routes share no numerical
integration.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product
from math import comb, factorial
from typing import Iterable, Sequence

import numpy as np

Weight = tuple[int, ...]
Character = dict[Weight, int]


def add_weights(left: Weight, right: Weight) -> Weight:
    return tuple(a + b for a, b in zip(left, right))


def subtract_weights(left: Weight, right: Weight) -> Weight:
    return tuple(a - b for a, b in zip(left, right))


def scale_weight(scale: int, weight: Weight) -> Weight:
    return tuple(scale * value for value in weight)


def multiply_characters(left: Character, right: Character) -> Character:
    answer: Counter[Weight] = Counter()
    for left_weight, left_multiplicity in left.items():
        for right_weight, right_multiplicity in right.items():
            answer[add_weights(left_weight, right_weight)] += (
                left_multiplicity * right_multiplicity
            )
    return dict(answer)


@lru_cache(maxsize=None)
def symmetric_power_character(weights: tuple[Weight, ...], degree: int) -> Character:
    """Return the character of Sym^degree for a list of weight spaces."""

    rank = len(weights[0])
    zero = (0,) * rank
    levels: list[Character] = [{zero: 1}] + [{} for _ in range(degree)]
    for weight in weights:
        updated: list[Counter[Weight]] = [Counter() for _ in range(degree + 1)]
        for old_degree, character in enumerate(levels):
            for repetitions in range(degree - old_degree + 1):
                shift = scale_weight(repetitions, weight)
                for old_weight, multiplicity in character.items():
                    updated[old_degree + repetitions][
                        add_weights(old_weight, shift)
                    ] += multiplicity
        levels = [dict(character) for character in updated]
    return levels[degree]


def complete_product_character(
    weights: tuple[Weight, ...], degrees: Sequence[int]
) -> Character:
    zero = (0,) * len(weights[0])
    answer: Character = {zero: 1}
    for degree in degrees:
        answer = multiply_characters(
            answer, symmetric_power_character(weights, degree)
        )
    return answer


def permutation_sign(permutation: Sequence[int]) -> int:
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


@dataclass(frozen=True)
class WeylElement:
    permutation: tuple[int, ...]
    signs: tuple[int, ...]
    determinant: int

    def act(self, weight: Weight) -> Weight:
        return tuple(
            self.signs[index] * weight[self.permutation[index]]
            for index in range(len(weight))
        )


@lru_cache(maxsize=None)
def weyl_group(kind: str, rank: int) -> tuple[WeylElement, ...]:
    """Signed-permutation Weyl groups for B_rank and D_rank."""

    if kind not in {"B", "D"}:
        raise ValueError("kind must be 'B' or 'D'")
    elements: list[WeylElement] = []
    for permutation in permutations(range(rank)):
        sign_of_permutation = permutation_sign(permutation)
        for signs in product((-1, 1), repeat=rank):
            sign_product = int(np.prod(signs))
            if kind == "D" and sign_product != 1:
                continue
            elements.append(
                WeylElement(
                    permutation,
                    signs,
                    sign_of_permutation * sign_product,
                )
            )
    return tuple(elements)


def doubled_rho(kind: str, rank: int) -> Weight:
    if kind == "B":
        return tuple(2 * rank - 2 * index - 1 for index in range(rank))
    if kind == "D":
        return tuple(2 * rank - 2 * index - 2 for index in range(rank))
    raise ValueError("kind must be 'B' or 'D'")


def weyl_invariant_multiplicity(character: Character, kind: str) -> int:
    """Integrate a Weyl-invariant character over Spin(2n+1) or Spin(2n)."""

    rank = len(next(iter(character)))
    rho = doubled_rho(kind, rank)
    return sum(
        element.determinant
        * character.get(subtract_weights(rho, element.act(rho)), 0)
        for element in weyl_group(kind, rank)
    )


def monomial_coefficient(
    weights: tuple[Weight, ...], tau: Sequence[int], kind: str
) -> int:
    """Compute [m_tau] S as an exact Weyl character inner product."""

    return weyl_invariant_multiplicity(
        complete_product_character(weights, tuple(tau)), kind
    )


def spin_weights(rank: int) -> tuple[Weight, ...]:
    """Weights of the 2^rank-dimensional B_rank spin module."""

    return tuple(product((-1, 1), repeat=rank))


def half_spin_weights(rank: int, chirality: int = 1) -> tuple[Weight, ...]:
    """One D_rank half-spin weight orbit in doubled coordinates.

    ``chirality=1`` selects an even number of minus signs and ``-1`` selects
    an odd number.  The labels + and - may be interchanged without changing
    any coefficient tested here.
    """

    if chirality not in {-1, 1}:
        raise ValueError("chirality must be +1 or -1")
    return tuple(
        signs
        for signs in product((-1, 1), repeat=rank)
        if int(np.prod(signs)) == chirality
    )


def vector_weights(kind: str, rank: int) -> tuple[Weight, ...]:
    weights: list[Weight] = []
    for index in range(rank):
        for sign in (-2, 2):
            weight = [0] * rank
            weight[index] = sign
            weights.append(tuple(weight))
    if kind == "B":
        weights.append((0,) * rank)
    elif kind != "D":
        raise ValueError("kind must be 'B' or 'D'")
    return tuple(weights)


def root_weights(kind: str, rank: int) -> tuple[Weight, ...]:
    roots: list[Weight] = []
    for i in range(rank):
        if kind == "B":
            for sign in (-2, 2):
                root = [0] * rank
                root[i] = sign
                roots.append(tuple(root))
        for j in range(i + 1, rank):
            for sign_i, sign_j in product((-2, 2), repeat=2):
                root = [0] * rank
                root[i] = sign_i
                root[j] = sign_j
                roots.append(tuple(root))
    if kind not in {"B", "D"}:
        raise ValueError("kind must be 'B' or 'D'")
    return tuple(roots)


def adjoint_weights(kind: str, rank: int) -> tuple[Weight, ...]:
    return root_weights(kind, rank) + ((0,) * rank,) * rank


def tensor_product_weights(*weight_lists: Iterable[Weight]) -> tuple[Weight, ...]:
    lists = [tuple(weights) for weights in weight_lists]
    zero = (0,) * len(lists[0][0])
    totals = [zero]
    for weights in lists:
        totals = [add_weights(total, weight) for total in totals for weight in weights]
    return tuple(totals)


def torus_matrix(weights: Sequence[Weight], angles: Sequence[float]) -> np.ndarray:
    """Construct a represented maximal-torus matrix from doubled weights."""

    phases = [
        0.5 * sum(weight[index] * angles[index] for index in range(len(angles)))
        for weight in weights
    ]
    return np.diag(np.exp(1j * np.asarray(phases)))


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kronecker_all(factors: Sequence[np.ndarray]) -> np.ndarray:
    answer = np.array([[1]], dtype=complex)
    for factor in factors:
        answer = np.kron(answer, factor)
    return answer


@lru_cache(maxsize=None)
def gamma_matrices(dimension: int) -> tuple[np.ndarray, ...]:
    """Hermitian complex gamma matrices with gamma_a gamma_b + ... = 2 delta."""

    if dimension < 2:
        raise ValueError("dimension must be at least 2")
    rank = dimension // 2
    gammas: list[np.ndarray] = []
    for index in range(rank):
        prefix = [Z] * index
        suffix = [I2] * (rank - index - 1)
        gammas.append(kronecker_all(prefix + [X] + suffix))
        gammas.append(kronecker_all(prefix + [Y] + suffix))
    if dimension % 2:
        gammas.append(kronecker_all([Z] * rank))
    return tuple(gammas)


def clifford_residual(dimension: int) -> float:
    gammas = gamma_matrices(dimension)
    identity = np.eye(gammas[0].shape[0], dtype=complex)
    return max(
        float(
            np.linalg.norm(
                left @ right + right @ left
                - (2 * identity if i == j else 0 * identity)
            )
        )
        for i, left in enumerate(gammas)
        for j, right in enumerate(gammas)
    )


def spin_group_matrix(dimension: int, rotations: Sequence[tuple[int, int, float]]) -> np.ndarray:
    """Construct a product of exponentiated Clifford bivectors in Spin(dimension)."""

    gammas = gamma_matrices(dimension)
    answer = np.eye(gammas[0].shape[0], dtype=complex)
    for left, right, angle in rotations:
        bivector = gammas[left] @ gammas[right]
        # For distinct Clifford generators, bivector^2=-I, so the exponential
        # is available in closed form and needs no matrix-exponential package.
        factor = np.cos(0.5 * angle) * np.eye(bivector.shape[0], dtype=complex)
        factor += np.sin(0.5 * angle) * bivector
        answer = answer @ factor
    return answer


def chirality_operator(even_dimension: int) -> np.ndarray:
    if even_dimension % 2:
        raise ValueError("chirality is defined here only in even dimension")
    gammas = gamma_matrices(even_dimension)
    product_matrix = kronecker_all([])
    product_matrix = np.eye(gammas[0].shape[0], dtype=complex)
    for gamma in gammas:
        product_matrix = product_matrix @ gamma
    # Choose the phase algorithmically so that the square is +I and the
    # eigenvalues are numerically real.  This avoids a convention-dependent
    # formula while preserving the two half-spin eigenspaces.
    phase = (1j) ** (even_dimension // 2)
    operator = phase * product_matrix
    if np.linalg.norm(operator @ operator - np.eye(operator.shape[0])) > 1e-10:
        operator = -phase * product_matrix
    return operator


def half_spin_block(matrix: np.ndarray, even_dimension: int, chirality: int = 1) -> np.ndarray:
    operator = chirality_operator(even_dimension)
    diagonal = np.real_if_close(np.diag(operator))
    indices = np.flatnonzero(np.real(diagonal) * chirality > 0)
    return matrix[np.ix_(indices, indices)]


def spin_lie_generators(dimension: int) -> tuple[np.ndarray, ...]:
    gammas = gamma_matrices(dimension)
    return tuple(
        0.5 * gammas[left] @ gammas[right]
        for left in range(dimension)
        for right in range(left + 1, dimension)
    )


def tensor_square_casimir(
    dimension: int, chirality: int | None = None
) -> np.ndarray:
    generators = spin_lie_generators(dimension)
    if chirality is not None:
        generators = tuple(
            half_spin_block(generator, dimension, chirality) for generator in generators
        )
    module_dimension = generators[0].shape[0]
    identity = np.eye(module_dimension, dtype=complex)
    casimir = np.zeros((module_dimension**2, module_dimension**2), dtype=complex)
    for generator in generators:
        total = np.kron(generator, identity) + np.kron(identity, generator)
        casimir -= total @ total
    return np.real_if_close(casimir)


def spectral_clusters(matrix: np.ndarray, tolerance: float = 1e-7) -> list[tuple[float, int]]:
    eigenvalues = np.linalg.eigvalsh(matrix)
    clusters: list[list[float]] = []
    for value in eigenvalues:
        if not clusters or abs(value - clusters[-1][-1]) > tolerance:
            clusters.append([float(value)])
        else:
            clusters[-1].append(float(value))
    return [(float(np.mean(cluster)), len(cluster)) for cluster in clusters]


def multilinear_prediction(family: str, rank: int) -> int:
    if family == "B-spin":
        return rank + 1
    if family == "D-half":
        return rank // 2 + 1 if rank % 2 == 0 else (rank - 1) // 2
    if family == "D-balanced":
        return rank // 2 + 1
    raise ValueError("unknown family")


def bilinear_prediction(family: str, rank: int, symmetric: bool = False) -> int:
    if family == "B-spin":
        if not symmetric:
            return 1
        return int(rank % 4 in {0, 3})
    if family == "D-half":
        if not symmetric:
            return int(rank % 2 == 0)
        return int(rank % 4 == 0)
    raise ValueError("unknown family")


def verify_spin_coefficients(maximum_rank: int = 5) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank in range(1, maximum_rank + 1):
        weights = spin_weights(rank)
        targets = {
            (1,): 0,
            (2,): bilinear_prediction("B-spin", rank, symmetric=True),
            (1, 1): 1,
            (1, 1, 1): 0,
            (1, 1, 1, 1): rank + 1,
        }
        for tau, predicted in targets.items():
            exact = monomial_coefficient(weights, tau, "B")
            rows.append(
                {
                    "group": f"Spin({2 * rank + 1})",
                    "module": f"Delta_{rank}",
                    "dimension": 2**rank,
                    "tau": tau,
                    "Weyl coefficient": exact,
                    "formula": predicted,
                    "passed": exact == predicted,
                }
            )
    return rows


def verify_half_spin_coefficients(
    minimum_rank: int = 2, maximum_rank: int = 5
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank in range(minimum_rank, maximum_rank + 1):
        weights = half_spin_weights(rank)
        targets = {
            (1,): 0,
            (2,): bilinear_prediction("D-half", rank, symmetric=True),
            (1, 1): bilinear_prediction("D-half", rank, symmetric=False),
            (1, 1, 1, 1): multilinear_prediction("D-half", rank),
        }
        for tau, predicted in targets.items():
            exact = monomial_coefficient(weights, tau, "D")
            rows.append(
                {
                    "group": f"Spin({2 * rank})",
                    "module": f"Delta_{rank}^+",
                    "dimension": 2 ** (rank - 1),
                    "tau": tau,
                    "Weyl coefficient": exact,
                    "formula": predicted,
                    "passed": exact == predicted,
                }
            )
    return rows


def verify_balanced_half_spin(maximum_rank: int = 5) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank in range(2, maximum_rank + 1):
        plus = half_spin_weights(rank, 1)
        dual = plus if rank % 2 == 0 else half_spin_weights(rank, -1)
        end_weights = tensor_product_weights(plus, dual)
        # End(Delta+) is self-dual.  Its tensor-square invariant dimension is
        # the mixed fourth coefficient in the supplied two-sided formula.
        exact = weyl_invariant_multiplicity(
            multiply_characters(dict(Counter(end_weights)), dict(Counter(end_weights))),
            "D",
        )
        predicted = multilinear_prediction("D-balanced", rank)
        rows.append(
            {
                "group": f"Spin({2 * rank})",
                "target": "dim ((Delta+ tensor Delta+*)^tensor2)^G",
                "exact Weyl coefficient": exact,
                "formula": predicted,
                "passed": exact == predicted,
            }
        )
    return rows


def verify_clifford_casimirs(maximum_rank: int = 4) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank in range(1, maximum_rank + 1):
        dimension = 2 * rank + 1
        clusters = spectral_clusters(tensor_square_casimir(dimension))
        predicted_multiplicities = [comb(dimension, degree) for degree in range(rank + 1)]
        observed_multiplicities = sorted(multiplicity for _, multiplicity in clusters)
        rows.append(
            {
                "group": f"Spin({dimension})",
                "spin dimension": 2**rank,
                "Clifford residual": clifford_residual(dimension),
                "Casimir blocks": len(clusters),
                "formula blocks": rank + 1,
                "block dimensions": observed_multiplicities,
                "formula dimensions": sorted(predicted_multiplicities),
                "passed": len(clusters) == rank + 1
                and observed_multiplicities == sorted(predicted_multiplicities),
            }
        )
    for rank in (2, 4):
        if rank > maximum_rank:
            continue
        dimension = 2 * rank
        clusters = spectral_clusters(tensor_square_casimir(dimension, chirality=1))
        predicted = rank // 2 + 1
        rows.append(
            {
                "group": f"Spin({dimension})",
                "spin dimension": 2 ** (rank - 1),
                "Clifford residual": clifford_residual(dimension),
                "Casimir blocks": len(clusters),
                "formula blocks": predicted,
                "block dimensions": sorted(multiplicity for _, multiplicity in clusters),
                "formula dimensions": "multiplicity-free half-spin tensor square",
                "passed": len(clusters) == predicted,
            }
        )
    return rows


def verify_pin_components() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rotations_by_dimension = {
        2: ((0, 1, 0.37),),
        4: ((0, 1, 0.37), (2, 3, -0.21)),
        6: ((0, 1, 0.37), (2, 3, -0.21), (4, 5, 0.29)),
    }
    for dimension, rotations in rotations_by_dimension.items():
        spin_matrix = spin_group_matrix(dimension, rotations)
        gamma = gamma_matrices(dimension)[0]
        chirality = chirality_operator(dimension)
        for sign, reflection in (("+", gamma), ("-", 1j * gamma)):
            odd_matrix = reflection @ spin_matrix
            odd_traces = [np.trace(np.linalg.matrix_power(odd_matrix, power)) for power in (1, 3, 5)]
            square_target = np.eye(reflection.shape[0]) * (1 if sign == "+" else -1)
            swaps_chirality = np.linalg.norm(reflection @ chirality + chirality @ reflection)
            rows.append(
                {
                    "group": f"Pin^{sign}({dimension})",
                    "pinor dimension": reflection.shape[0],
                    "u^2 residual": float(np.linalg.norm(reflection @ reflection - square_target)),
                    "chirality-swap residual": float(swaps_chirality),
                    "max odd-power trace": float(max(abs(value) for value in odd_traces)),
                    "unitarity residual": float(
                        np.linalg.norm(odd_matrix.conj().T @ odd_matrix - np.eye(odd_matrix.shape[0]))
                    ),
                    "passed": np.linalg.norm(reflection @ reflection - square_target) < 1e-11
                    and swaps_chirality < 1e-11
                    and max(abs(value) for value in odd_traces) < 1e-11,
                }
            )
    return rows


def verify_derived_representations() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    angles = (0.23, -0.31, 0.17)
    for kind, rank in (("B", 3), ("D", 3)):
        vector = vector_weights(kind, rank)
        spinor = spin_weights(rank) if kind == "B" else half_spin_weights(rank)
        vector_matrix = torus_matrix(vector, angles)
        spinor_matrix = torus_matrix(spinor, angles)
        mixed_matrix = np.kron(spinor_matrix, vector_matrix)
        mixed_weights = tensor_product_weights(spinor, vector)
        expected_diagonal = np.diag(torus_matrix(mixed_weights, angles))
        actual_diagonal = np.diag(mixed_matrix)
        trace_residual = float(
            abs(np.trace(mixed_matrix) - np.trace(spinor_matrix) * np.trace(vector_matrix))
        )
        spectral_residual = float(np.linalg.norm(actual_diagonal - expected_diagonal))
        rows.append(
            {
                "group": f"type {kind}_{rank}",
                "identity": "chi_(Delta tensor V) = chi_Delta chi_V",
                "residual": trace_residual,
                "spectral residual": spectral_residual,
                "passed": trace_residual < 1e-11 and spectral_residual < 1e-11,
            }
        )

        adjoint = torus_matrix(adjoint_weights(kind, rank), angles)
        for power in (1, 2, 3, 4):
            direct = np.trace(np.linalg.matrix_power(adjoint, power))
            vector_power_trace = np.trace(np.linalg.matrix_power(vector_matrix, power))
            vector_double_trace = np.trace(np.linalg.matrix_power(vector_matrix, 2 * power))
            predicted = 0.5 * (vector_power_trace**2 - vector_double_trace)
            rows.append(
                {
                    "group": f"type {kind}_{rank}",
                    "identity": f"adjoint trace power r={power}",
                    "residual": float(abs(direct - predicted)),
                    "spectral residual": 0.0,
                    "passed": abs(direct - predicted) < 1e-11,
                }
            )
    for kind, rank in (("B", 2), ("B", 3), ("D", 4)):
        weights = adjoint_weights(kind, rank)
        for tau, predicted in (((2,), 1), ((1, 1), 1), ((1, 1, 1), 1)):
            exact = monomial_coefficient(weights, tau, kind)
            rows.append(
                {
                    "group": f"type {kind}_{rank} adjoint",
                    "identity": f"[m_{tau}]",
                    "residual": abs(exact - predicted),
                    "spectral residual": 0.0,
                    "passed": exact == predicted,
                }
            )
    return rows


def run_all(maximum_rank: int = 5) -> dict[str, object]:
    spin_rows = verify_spin_coefficients(maximum_rank)
    half_spin_rows = verify_half_spin_coefficients(2, maximum_rank)
    balanced_rows = verify_balanced_half_spin(maximum_rank)
    casimir_rows = verify_clifford_casimirs(min(maximum_rank, 4))
    pin_rows = verify_pin_components()
    derived_rows = verify_derived_representations()
    all_rows = spin_rows + half_spin_rows + balanced_rows + casimir_rows + pin_rows + derived_rows
    return {
        "maximum rank": maximum_rank,
        "spin rows": spin_rows,
        "half-spin rows": half_spin_rows,
        "balanced rows": balanced_rows,
        "casimir rows": casimir_rows,
        "pin rows": pin_rows,
        "derived rows": derived_rows,
        "checks": len(all_rows),
        "passed": all(bool(row["passed"]) for row in all_rows),
    }


def print_report(report: dict[str, object]) -> None:
    print(
        f"Spin/Pin verification: {report['checks']} checks; "
        f"passed={report['passed']}; maximum rank={report['maximum rank']}"
    )
    for key in (
        "spin rows",
        "half-spin rows",
        "balanced rows",
        "casimir rows",
        "pin rows",
        "derived rows",
    ):
        rows = report[key]
        passed = sum(bool(row["passed"]) for row in rows)
        print(f"  {key}: {passed}/{len(rows)} passed")


if __name__ == "__main__":
    result = run_all()
    print_report(result)
    if not result["passed"]:
        for section in (
            "spin rows",
            "half-spin rows",
            "balanced rows",
            "casimir rows",
            "pin rows",
            "derived rows",
        ):
            for row in result[section]:
                if not row["passed"]:
                    print("FAIL", section, row)
        raise SystemExit(1)
