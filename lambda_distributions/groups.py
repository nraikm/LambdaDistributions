"""Concrete finite matrix-group representations.

Every constructor returns all represented group elements, including repeats
when the representation has a kernel.  That is important: expectations are
averages over the abstract group, not over the set of distinct image matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from math import factorial

import numpy as np
from numpy.typing import NDArray

ComplexMatrix = NDArray[np.complex128]


@dataclass(frozen=True)
class FiniteMatrixGroup:
    """A named finite group together with a fixed matrix representation."""

    name: str
    elements: tuple[ComplexMatrix, ...]

    def __post_init__(self) -> None:
        if not self.elements:
            raise ValueError("a finite matrix group must contain at least one element")
        dimension = self.elements[0].shape[0]
        if dimension == 0 or any(matrix.shape != (dimension, dimension) for matrix in self.elements):
            raise ValueError("all group elements must be square matrices of the same size")

    @property
    def order(self) -> int:
        return len(self.elements)

    @property
    def dimension(self) -> int:
        return self.elements[0].shape[0]


def _positive_integer(value: int, name: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def cyclic_character(n: int, k: int = 1) -> FiniteMatrixGroup:
    """The one-dimensional character ``m -> exp(2 pi i k m / n)`` of C_n. Note n is not dimension here; it represents the length of the cyclic group."""

    n = _positive_integer(n, "n")
    roots = tuple(
        np.array([[np.exp(2j * np.pi * k * m / n)]], dtype=complex)
        for m in range(n)
    )  # needs to be a matrix, even in dimension 1, for consistency.
    return FiniteMatrixGroup(f"C_{n} character k={k}", roots)


def cyclic_permutation_group(n: int) -> FiniteMatrixGroup:
    """The regular cyclic shift representation of C_n in dimension ``n``."""

    n = _positive_integer(n, "n")
    generator = np.zeros((n, n), dtype=complex)
    columns = np.arange(n)
    generator[(columns + 1) % n, columns] = 1
    elements = tuple(np.linalg.matrix_power(generator, exponent) for exponent in range(n))
    return FiniteMatrixGroup(f"C_{n} permutation representation", elements)


def cyclic_real_2d(n: int, k: int = 1) -> FiniteMatrixGroup:
    """The complex diagonal form of the two-dimensional real C_n representation."""

    n = _positive_integer(n, "n")
    elements = []
    for m in range(n):
        root = np.exp(2j * np.pi * k * m / n)
        elements.append(np.diag([root, root.conjugate()]).astype(complex))
    return FiniteMatrixGroup(f"C_{n} real 2D representation k={k}", tuple(elements))


def dihedral_group(n: int, k: int = 1) -> FiniteMatrixGroup:
    """The standard two-dimensional representation of D_n, of order ``2n``."""

    n = _positive_integer(n, "n", 2)
    rotations: list[ComplexMatrix] = []
    reflections: list[ComplexMatrix] = []
    for m in range(n):
        root = np.exp(2j * np.pi * k * m / n)
        rotations.append(np.diag([root, root.conjugate()]).astype(complex))
        reflections.append(np.array([[0, root], [root.conjugate(), 0]], dtype=complex))
    return FiniteMatrixGroup(f"D_{n} representation k={k}", tuple(rotations + reflections))


def dicyclic_group(n: int) -> FiniteMatrixGroup:
    """The faithful two-dimensional representation of Dic_n, of order ``4n``."""

    n = _positive_integer(n, "n", 2)
    root = np.exp(1j * np.pi / n)
    a = np.diag([root, root.conjugate()]).astype(complex)
    x = np.array([[0, 1], [-1, 0]], dtype=complex)
    powers = [np.linalg.matrix_power(a, exponent) for exponent in range(2 * n)]
    return FiniteMatrixGroup(f"Dic_{n}", tuple(powers + [x @ power for power in powers]))


def _permutation_is_even(permutation: tuple[int, ...]) -> bool:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return inversions % 2 == 0


def _permutation_matrix(permutation: tuple[int, ...]) -> ComplexMatrix:
    size = len(permutation)
    matrix = np.zeros((size, size), dtype=complex)
    matrix[permutation, np.arange(size)] = 1
    return matrix


def alternating_group(n: int) -> FiniteMatrixGroup:
    """The permutation representation of the alternating group A_n."""

    n = _positive_integer(n, "n")
    elements = tuple(
        _permutation_matrix(permutation)
        for permutation in permutations(range(n))
        if _permutation_is_even(permutation)
    )
    return FiniteMatrixGroup(f"A_{n} permutation representation", elements)


def generalized_symmetric_group(n: int, level: int) -> FiniteMatrixGroup:
    """The monomial representation of ``C_level wreath S_n``.

    The group has order ``level**n * n!`` and is therefore intended for small
    formula-checking examples.
    """

    n = _positive_integer(n, "n")
    level = _positive_integer(level, "level")
    roots = np.exp(2j * np.pi * np.arange(level) / level)
    elements: list[ComplexMatrix] = []
    for permutation in permutations(range(n)):
        permutation_matrix = _permutation_matrix(permutation)
        for exponents in product(range(level), repeat=n):
            diagonal = np.diag([roots[exponent] for exponent in exponents])
            elements.append(diagonal @ permutation_matrix)
    expected_order = level**n * factorial(n)
    assert len(elements) == expected_order
    return FiniteMatrixGroup(f"C_{level} wreath S_{n}", tuple(elements))


def pauli_group(n: int) -> FiniteMatrixGroup:
    """The n-qubit Pauli group in its ``2**n``-dimensional representation."""

    n = _positive_integer(n, "n")
    identity = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1, -1]).astype(complex)
    single_qubit = (identity, x, y, z)
    phases = (1, -1, 1j, -1j)
    elements: list[ComplexMatrix] = []
    for factors in product(single_qubit, repeat=n):
        tensor = np.array([[1]], dtype=complex)
        for factor in factors:
            tensor = np.kron(tensor, factor)
        elements.extend(phase * tensor for phase in phases)
    return FiniteMatrixGroup(f"Pauli group P_{n}", tuple(elements))
