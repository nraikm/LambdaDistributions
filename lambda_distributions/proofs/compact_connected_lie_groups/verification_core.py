"""Exact low-degree checks for compact Lie-group sigma-MGFs.

Weights and roots use simple-root coordinates.  The Weyl integral is evaluated
with the one-sided Weyl denominator; its coefficients are recognized by the
Weyl-denominator identity, so no floating-point integration is involved.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, combinations_with_replacement, permutations
from math import factorial
from typing import Iterable

Vector = tuple[Fraction, ...]
Character = dict[Vector, int]


def _v(values: Iterable[int | Fraction]) -> Vector:
    return tuple(Fraction(x) for x in values)


def _add(left: Vector, right: Vector) -> Vector:
    return tuple(a + b for a, b in zip(left, right))


def _sub(left: Vector, right: Vector) -> Vector:
    return tuple(a - b for a, b in zip(left, right))


def _scale(value: int, vector: Vector) -> Vector:
    return tuple(value * x for x in vector)


def solve(matrix: tuple[tuple[int, ...], ...], rhs: Vector) -> Vector:
    """Solve a small rational linear system by Gauss-Jordan elimination."""

    size = len(matrix)
    augmented = [list(map(Fraction, row)) + [rhs[i]] for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = next(row for row in range(column, size) if augmented[row][column])
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [entry / divisor for entry in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            multiple = augmented[row][column]
            if multiple:
                augmented[row] = [
                    entry - multiple * pivot_entry
                    for entry, pivot_entry in zip(augmented[row], augmented[column])
                ]
    return tuple(augmented[row][-1] for row in range(size))


def dynkin_labels(weight: Vector, cartan: tuple[tuple[int, ...], ...]) -> Vector:
    """Return (<weight, alpha_i^vee>) in the chosen simple-root basis."""

    return tuple(
        sum(weight[j] * cartan[j][i] for j in range(len(weight)))
        for i in range(len(weight))
    )


def reflect(weight: Vector, node: int, cartan: tuple[tuple[int, ...], ...]) -> Vector:
    labels = dynkin_labels(weight, cartan)
    result = list(weight)
    result[node] -= labels[node]
    return tuple(result)


def orbit(seed: Vector, cartan: tuple[tuple[int, ...], ...]) -> tuple[Vector, ...]:
    seen = {seed}
    frontier = [seed]
    while frontier:
        current = frontier.pop()
        for node in range(len(cartan)):
            image = reflect(current, node, cartan)
            if image not in seen:
                seen.add(image)
                frontier.append(image)
    return tuple(sorted(seen))


def roots(cartan: tuple[tuple[int, ...], ...]) -> tuple[Vector, ...]:
    all_roots: set[Vector] = set()
    rank = len(cartan)
    for node in range(rank):
        seed = tuple(Fraction(int(i == node)) for i in range(rank))
        all_roots.update(orbit(seed, cartan))
    return tuple(sorted(all_roots))


def fundamental_weight(node: int, cartan: tuple[tuple[int, ...], ...]) -> Vector:
    # A^T c = e_node for c in simple-root coordinates.
    transpose = tuple(tuple(cartan[j][i] for j in range(len(cartan))) for i in range(len(cartan)))
    return solve(transpose, tuple(Fraction(int(i == node)) for i in range(len(cartan))))


def add_characters(left: Character, right: Character, factor: int = 1) -> Character:
    result = Counter(left)
    for weight, multiplicity in right.items():
        result[weight] += factor * multiplicity
        if not result[weight]:
            del result[weight]
    return dict(result)


def multiply_characters(left: Character, right: Character) -> Character:
    result: Counter[Vector] = Counter()
    for left_weight, left_mult in left.items():
        for right_weight, right_mult in right.items():
            result[_add(left_weight, right_weight)] += left_mult * right_mult
    return dict(result)


def symmetric_power_character(weights: tuple[Vector, ...], degree: int) -> Character:
    """Character of Sym^degree(V), including repeated weights correctly."""

    rank = len(weights[0])
    levels: list[Character] = [{(Fraction(0),) * rank: 1}] + [{} for _ in range(degree)]
    for weight in weights:
        updated: list[Character] = [{} for _ in range(degree + 1)]
        for old_degree, old_character in enumerate(levels):
            for repetitions in range(degree - old_degree + 1):
                shift = _scale(repetitions, weight)
                target = updated[old_degree + repetitions]
                for old_weight, multiplicity in old_character.items():
                    new_weight = _add(old_weight, shift)
                    target[new_weight] = target.get(new_weight, 0) + multiplicity
        levels = updated
    return levels[degree]


def schur_character(weights: tuple[Vector, ...], partition: tuple[int, ...]) -> Character:
    """Jacobi-Trudi evaluation of s_partition on the weight eigenvalues."""

    if not partition:
        return {(Fraction(0),) * len(weights[0]): 1}
    length = len(partition)
    largest_h = max(partition[i] - i + length - 1 for i in range(length))
    complete = [symmetric_power_character(weights, degree) for degree in range(largest_h + 1)]
    zero = (Fraction(0),) * len(weights[0])

    result: Character = {}
    for perm in permutations(range(length)):
        inversions = sum(perm[i] > perm[j] for i in range(length) for j in range(i + 1, length))
        term: Character = {zero: 1}
        valid = True
        for i, column in enumerate(perm):
            index = partition[i] - i + column
            if index < 0:
                valid = False
                break
            term = multiply_characters(term, complete[index])
        if valid:
            result = add_characters(result, term, -1 if inversions % 2 else 1)
    return result


@lru_cache(maxsize=None)
def denominator_coefficient(weight: Vector, cartan: tuple[tuple[int, ...], ...], rho: Vector) -> int:
    """Coefficient of e^-weight in product_{alpha>0}(1-e^-alpha)."""

    if any(value.denominator != 1 for value in weight):
        return 0
    candidate = _sub(rho, weight)
    parity = 0
    # Repeatedly reflect across a negative simple-coroot pairing to reach the
    # unique dominant point in the Weyl orbit.
    for _ in range(10000):
        labels = dynkin_labels(candidate, cartan)
        negative = next((i for i, value in enumerate(labels) if value < 0), None)
        if negative is None:
            return (-1) ** parity if candidate == rho else 0
        candidate = reflect(candidate, negative, cartan)
        parity ^= 1
    raise RuntimeError("dominantization did not terminate")


def invariant_multiplicity(character: Character, rep: "Representation") -> int:
    """Apply the exact Weyl integral to a torus character."""

    return sum(
        multiplicity * denominator_coefficient(weight, rep.cartan, rep.rho)
        for weight, multiplicity in character.items()
    )


def _complete_weight_sums(weights: tuple[Vector, ...], degree: int):
    zero = (Fraction(0),) * len(weights[0])
    if degree == 0:
        yield zero
        return
    for indices in combinations_with_replacement(range(len(weights)), degree):
        total = zero
        for index in indices:
            total = _add(total, weights[index])
        yield total


def complete_product_invariant(rep: "Representation", degrees: tuple[int, ...]) -> int:
    """Integrate a product h_d1...h_dr by streaming monomial weights."""

    zero = (Fraction(0),) * rep.rank

    def visit(block: int, total: Vector) -> int:
        if block == len(degrees):
            return denominator_coefficient(total, rep.cartan, rep.rho)
        return sum(
            visit(block + 1, _add(total, block_weight))
            for block_weight in _complete_weight_sums(rep.weights, degrees[block])
        )

    return visit(0, zero)


def exterior_invariant(rep: "Representation", degree: int) -> int:
    """Integrate e_degree by streaming distinct eigenvalue products."""

    zero = (Fraction(0),) * rep.rank
    total = 0
    for indices in combinations(range(rep.dimension), degree):
        weight = zero
        for index in indices:
            weight = _add(weight, rep.weights[index])
        total += denominator_coefficient(weight, rep.cartan, rep.rho)
    return total


def schur_invariant_streaming(rep: "Representation", partition: tuple[int, ...]) -> int:
    """Low-memory Jacobi-Trudi Weyl integral for larger representations."""

    if not partition:
        return 1
    if all(part == 1 for part in partition):
        return exterior_invariant(rep, len(partition))
    length = len(partition)
    answer = 0
    for perm in permutations(range(length)):
        degrees = tuple(partition[i] - i + perm[i] for i in range(length))
        if any(degree < 0 for degree in degrees):
            continue
        inversions = sum(perm[i] > perm[j] for i in range(length) for j in range(i + 1, length))
        answer += (-1 if inversions % 2 else 1) * complete_product_invariant(rep, degrees)
    return answer


def partitions(total: int, maximum: int | None = None) -> tuple[tuple[int, ...], ...]:
    if total == 0:
        return ((),)
    maximum = total if maximum is None else min(maximum, total)
    result = []
    for first in range(maximum, 0, -1):
        for rest in partitions(total - first, first):
            result.append((first,) + rest)
    return tuple(result)


@dataclass(frozen=True)
class Representation:
    name: str
    cartan: tuple[tuple[int, ...], ...]
    weights: tuple[Vector, ...]
    weyl_order: int

    @property
    def rank(self) -> int:
        return len(self.cartan)

    @property
    def dimension(self) -> int:
        return len(self.weights)

    @property
    def rho(self) -> Vector:
        transpose = tuple(tuple(self.cartan[j][i] for j in range(self.rank)) for i in range(self.rank))
        return solve(transpose, (Fraction(1),) * self.rank)

    @property
    def root_system(self) -> tuple[Vector, ...]:
        return roots(self.cartan)

    def torus_matrix(self, angles: tuple[float, ...]):
        """Explicit diagonal rho(t); imported lazily to keep exact core light."""

        import numpy as np

        phases = [sum(float(a * b) for a, b in zip(weight, angles)) for weight in self.weights]
        return np.diag(np.exp(1j * np.asarray(phases)))

    def schur_invariant(self, partition: tuple[int, ...]) -> int:
        # Dictionary characters are fast for small modules.  Streaming avoids a
        # large memory spike for quartics of V_56 and similar cases.
        if self.dimension > 40 and sum(partition) >= 3:
            return schur_invariant_streaming(self, partition)
        return invariant_multiplicity(schur_character(self.weights, partition), self)

    def ordinary_coefficients(self, maximum_degree: int) -> tuple[int, ...]:
        return tuple(
            invariant_multiplicity(symmetric_power_character(self.weights, degree), self)
            for degree in range(maximum_degree + 1)
        )


G2 = ((2, -1), (-3, 2))
B3 = ((2, -1, 0), (-1, 2, -2), (0, -1, 2))
F4 = ((2, -1, 0, 0), (-1, 2, -2, 0), (0, -1, 2, -1), (0, 0, -1, 2))


def simply_laced(rank: int, edges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, ...], ...]:
    matrix = [[2 if i == j else 0 for j in range(rank)] for i in range(rank)]
    for left, right in edges:
        matrix[left][right] = matrix[right][left] = -1
    return tuple(tuple(row) for row in matrix)


E6 = simply_laced(6, ((0, 2), (2, 3), (3, 1), (3, 4), (4, 5)))
E7 = simply_laced(7, ((0, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 6)))
E8 = simply_laced(8, ((0, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 6), (6, 7)))


def adjoint_representation(name: str, cartan, weyl_order: int) -> Representation:
    root_weights = roots(cartan)
    zero = (Fraction(0),) * len(cartan)
    return Representation(name, cartan, root_weights + (zero,) * len(cartan), weyl_order)


def exceptional_representations() -> dict[str, Representation]:
    zero2 = (Fraction(0),) * 2
    zero4 = (Fraction(0),) * 4
    g2_short = orbit((Fraction(1), Fraction(0)), G2)
    f4_short = orbit((Fraction(0), Fraction(0), Fraction(0), Fraction(1)), F4)
    return {
        "g2_7": Representation("G2 minimal 7", G2, g2_short + (zero2,), 12),
        "spin7_8": Representation(
            "Spin(7) spin 8", B3, orbit(fundamental_weight(2, B3), B3), 48
        ),
        "f4_26": Representation("F4 minimal 26", F4, f4_short + (zero4, zero4), 1152),
        "e6_27": Representation("E6 minimal 27", E6, orbit(fundamental_weight(0, E6), E6), 51840),
        "e7_56": Representation("E7 minimal 56", E7, orbit(fundamental_weight(6, E7), E7), 2903040),
        "e8_248": adjoint_representation("E8 adjoint 248", E8, 696729600),
    }


def cartan_a(rank: int) -> tuple[tuple[int, ...], ...]:
    return simply_laced(rank, tuple((i, i + 1) for i in range(rank - 1)))


def su_adjoint(n: int) -> Representation:
    if n < 2:
        raise ValueError("n must be at least 2")
    return adjoint_representation(f"SU({n}) adjoint", cartan_a(n - 1), factorial(n))


def low_degree_table(rep: Representation, maximum_degree: int) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for degree in range(maximum_degree + 1):
        for partition in partitions(degree):
            rows.append(
                {
                    "degree": degree,
                    "partition": str(partition),
                    "multiplicity": rep.schur_invariant(partition),
                }
            )
    return rows


def verify_expected(rep: Representation, expected: dict[tuple[int, ...], int], maximum_degree: int) -> list[dict[str, object]]:
    rows = []
    for degree in range(maximum_degree + 1):
        for partition in partitions(degree):
            actual = rep.schur_invariant(partition)
            wanted = expected.get(partition, 0)
            rows.append(
                {
                    "partition": partition,
                    "expected": wanted,
                    "exact": actual,
                    "pass": actual == wanted,
                }
            )
    return rows


def monomial_invariant(rep: Representation, partition: tuple[int, ...]) -> int:
    """Return the coefficient of m_partition in the sigma-MGF.

    Expanding each determinant inverse in the variables t_a shows that this
    is the Haar integral of the product h_{partition[0]}(rho(g)) ... .  The
    exact Weyl-denominator calculation is deliberately independent of the
    tensor-product decompositions quoted in the proof notes.
    """

    return complete_product_invariant(rep, partition)


def verify_monomial_expected(
    rep: Representation,
    expected: dict[tuple[int, ...], int],
    maximum_degree: int,
) -> list[dict[str, object]]:
    """Compare the claimed m_tau coefficients with exact Weyl integrals."""

    rows = []
    for degree in range(maximum_degree + 1):
        for partition in partitions(degree):
            actual = monomial_invariant(rep, partition)
            wanted = expected.get(partition, 0)
            rows.append(
                {
                    "partition tau": partition,
                    "expected [m_tau]": wanted,
                    "exact Weyl integral": actual,
                    "pass": actual == wanted,
                }
            )
    return rows
