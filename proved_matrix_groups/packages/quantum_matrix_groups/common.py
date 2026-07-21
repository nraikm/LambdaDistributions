"""Small, transparent numerical routines shared by the group verifications."""

from __future__ import annotations

from itertools import product
from math import comb

import numpy as np


TOL = 2.0e-8


def complete_homogeneous(eigenvalues: np.ndarray, degree: int) -> complex:
    """Return h_degree(eigenvalues) via the generating product."""
    coeff = np.zeros(degree + 1, dtype=np.complex128)
    coeff[0] = 1.0
    for value in eigenvalues:
        updated = coeff.copy()
        for d in range(1, degree + 1):
            updated[d] = coeff[d] + value * updated[d - 1]
        coeff = updated
    return complex(coeff[degree])


def target_coefficient(matrices: list[np.ndarray], tau: tuple[int, ...]) -> complex:
    """Compute dim((tensor_i Sym^{tau_i} V)^G) from explicit matrices."""
    total = 0.0j
    max_degree = max(tau, default=0)
    for matrix in matrices:
        eigenvalues = np.linalg.eigvals(matrix)
        h = [complete_homogeneous(eigenvalues, d) for d in range(max_degree + 1)]
        total += np.prod([h[d] for d in tau])
    return total / len(matrices)


def balanced_target_coefficient(
    matrices: list[np.ndarray],
    alpha: tuple[int, ...],
    beta: tuple[int, ...],
) -> complex:
    """Compute the phase-neutral two-sided coefficient by character averaging."""
    total = 0.0j
    max_degree = max((*alpha, *beta), default=0)
    for matrix in matrices:
        eigenvalues = np.linalg.eigvals(matrix)
        h = [complete_homogeneous(eigenvalues, d) for d in range(max_degree + 1)]
        left = np.prod([h[d] for d in alpha])
        right = np.prod([h[d] for d in beta])
        total += left * right.conjugate()
    return total / len(matrices)


def frame_potential(matrices: list[np.ndarray], degree: int) -> float:
    """Return |G|^{-1} sum_g |Tr(g)|^(2 degree)."""
    return float(np.mean([abs(np.trace(matrix)) ** (2 * degree) for matrix in matrices]))


def partitions_of(total: int, max_part: int | None = None):
    """Yield integer partitions in weakly decreasing order."""
    if total == 0:
        yield ()
        return
    upper = total if max_part is None else min(total, max_part)
    for first in range(upper, 0, -1):
        for tail in partitions_of(total - first, first):
            yield (first, *tail)


def standard_tableau_count(shape: tuple[int, ...]) -> int:
    """Hook-length formula for the number of standard tableaux of ``shape``."""
    from math import factorial

    hooks = 1
    for row, width in enumerate(shape):
        for column in range(width):
            below = sum(column < other_width for other_width in shape[row + 1 :])
            hooks *= width - column + below
    return factorial(sum(shape)) // hooks


def haar_unitary_frame_potential(dimension: int, degree: int) -> int:
    """Haar U(d) value sum_{lambda |- degree, ell(lambda)<=d} (f^lambda)^2."""
    return sum(
        standard_tableau_count(shape) ** 2
        for shape in partitions_of(degree)
        if len(shape) <= dimension
    )


def clean_integer(value: complex, tol: float = TOL) -> int:
    rounded = int(round(value.real))
    if abs(value.imag) > tol or abs(value.real - rounded) > tol:
        raise AssertionError(f"expected an integer, received {value!r}")
    return rounded


def a_term(dimension: int, tau: tuple[int, ...]) -> int:
    return int(np.prod([comb(dimension + d - 1, d) for d in tau]))


def b_term(dimension: int, prime: int, tau: tuple[int, ...]) -> int:
    if any(d % prime for d in tau):
        return 0
    multiplicity = dimension // prime
    return int(np.prod([comb(multiplicity + d // prime - 1, d // prime) for d in tau]))


def pauli_prediction(q: int, prime: int, n: int, tau: tuple[int, ...], scalar_order: int | None = None) -> float:
    """Coefficient formula for odd-p Pauli, or a scalar-extended lift."""
    selection = prime if scalar_order is None else scalar_order
    if sum(tau) % selection:
        return 0.0
    dimension = q**n
    central_weight = q ** (-2 * n)
    return central_weight * a_term(dimension, tau) + (1.0 - central_weight) * b_term(dimension, prime, tau)


def prime_pauli_matrices(prime: int, n: int = 1, scalar_order: int | None = None) -> list[np.ndarray]:
    """Construct phase-normalized Weyl matrices over the prime field F_p."""
    p = prime
    if p < 3 or any(p % d == 0 for d in range(2, int(p**0.5) + 1)):
        raise ValueError("prime_pauli_matrices expects an odd prime")
    s = p if scalar_order is None else scalar_order
    if s % p:
        raise ValueError("scalar order must be divisible by p")
    points = list(product(range(p), repeat=n))
    index = {x: i for i, x in enumerate(points)}
    omega = np.exp(2j * np.pi / p)
    scalar_root = np.exp(2j * np.pi / s)
    inv_two = pow(2, -1, p)
    matrices: list[np.ndarray] = []
    for scalar_power in range(s):
        scalar = scalar_root**scalar_power
        for a in points:
            for b in points:
                dot_ab = sum(x * y for x, y in zip(a, b)) % p
                matrix = np.zeros((p**n, p**n), dtype=np.complex128)
                phase0 = omega ** ((inv_two * dot_ab) % p)
                for column, x in enumerate(points):
                    target = tuple((x[k] + a[k]) % p for k in range(n))
                    phase = omega ** (sum(b[k] * x[k] for k in range(n)) % p)
                    matrix[index[target], column] = scalar * phase0 * phase
                matrices.append(matrix)
    return matrices


def qubit_pauli_matrices(n: int = 1, scalar_order: int = 4) -> list[np.ndarray]:
    """Construct the full phase-extended n-qubit Pauli group."""
    if scalar_order % 4:
        raise ValueError("the qubit lift needs scalar order divisible by four")
    identity = np.eye(2, dtype=np.complex128)
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    single = {(0, 0): identity, (1, 0): x, (0, 1): z, (1, 1): 1j * x @ z}
    scalar_root = np.exp(2j * np.pi / scalar_order)
    matrices: list[np.ndarray] = []
    for k in range(scalar_order):
        for labels in product(single, repeat=n):
            matrix = np.array([[scalar_root**k]], dtype=np.complex128)
            for label in labels:
                matrix = np.kron(matrix, single[label])
            matrices.append(matrix)
    return matrices


def gf9_pauli_matrices(heisenberg: bool = False) -> list[np.ndarray]:
    """Construct q=9, n=1 Pauli matrices; optionally average over H_3(F_9)."""
    # F_9 = F_3[u]/(u^2+1); elements are pairs a+b*u.
    elements = list(product(range(3), repeat=2))

    def add(x, y):
        return ((x[0] + y[0]) % 3, (x[1] + y[1]) % 3)

    def mul(x, y):
        return ((x[0] * y[0] + 2 * x[1] * y[1]) % 3, (x[0] * y[1] + x[1] * y[0]) % 3)

    def trace(x):
        return (2 * x[0]) % 3

    omega = np.exp(2j * np.pi / 3)
    index = {x: i for i, x in enumerate(elements)}
    matrices: list[np.ndarray] = []
    center_values = elements if heisenberg else [(0, 0), (1, 0), (2, 0)]
    for z0 in center_values:
        scalar = omega ** trace(z0)
        for a in elements:
            for b in elements:
                half_dot = mul((2, 0), mul(a, b))
                phase0 = omega ** trace(half_dot)
                matrix = np.zeros((9, 9), dtype=np.complex128)
                for column, x0 in enumerate(elements):
                    matrix[index[add(x0, a)], column] = scalar * phase0 * omega ** trace(mul(b, x0))
                matrices.append(matrix)
    return matrices


def modular_extraspecial_matrices(prime: int) -> list[np.ndarray]:
    """Faithful p-dimensional representation of C_{p^2} semidirect C_p."""
    p = prime
    eta = np.exp(2j * np.pi / (p * p))
    # x e_j = eta^((1+p)^j) e_j, y e_j = e_(j-1), so y^-1 x y=x^(1+p).
    x = np.diag([eta ** pow(1 + p, j, p * p) for j in range(p)])
    y = np.zeros((p, p), dtype=np.complex128)
    for j in range(p):
        y[(j - 1) % p, j] = 1.0
    return [np.linalg.matrix_power(x, a) @ np.linalg.matrix_power(y, b) for a in range(p * p) for b in range(p)]


def extraspecial_p2_prediction(prime: int, n: int, tau: tuple[int, ...]) -> float:
    p = prime
    if sum(tau) % p:
        return 0.0
    dimension = p**n
    a = a_term(dimension, tau)
    b = b_term(dimension, p, tau)
    if b == 0:
        return (p ** (-2 * n)) * a
    root_sum = p - 1 if (sum(tau) // p) % p == 0 else -1
    return (p ** (-2 * n)) * a + (1 / p - p ** (-2 * n)) * b + (root_sum / p) * b


def d8_matrices() -> list[np.ndarray]:
    r = np.array([[0, -1], [1, 0]], dtype=np.complex128)
    s = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return [np.linalg.matrix_power(r, a) @ np.linalg.matrix_power(s, b) for a in range(4) for b in range(2)]


def q8_matrices() -> list[np.ndarray]:
    i = np.array([[1j, 0], [0, -1j]], dtype=np.complex128)
    j = np.array([[0, 1], [-1, 0]], dtype=np.complex128)
    return [sign * matrix for sign in (1, -1) for matrix in (np.eye(2), i, j, i @ j)]


def extraspecial_2_prediction(sign: int, n: int, tau: tuple[int, ...]) -> float:
    if sum(tau) % 2:
        return 0.0
    dimension = 2**n
    central = 2 ** (-2 * n) * a_term(dimension, tau)
    if any(d % 2 for d in tau):
        return central
    base = int(np.prod([comb(dimension // 2 + d // 2 - 1, d // 2) for d in tau]))
    parity = (-1) ** (sum(tau) // 2)
    plus_weight = 1 / 2 + sign * 2 ** (-n - 1) - 2 ** (-2 * n)
    minus_weight = 1 / 2 - sign * 2 ** (-n - 1)
    return central + base * (plus_weight + parity * minus_weight)


def matrix_key(matrix: np.ndarray, phase_free: bool = False) -> tuple[float, ...]:
    candidate = matrix.copy()
    if phase_free:
        entry = next(value for value in candidate.flat if abs(value) > 1.0e-9)
        candidate *= np.exp(-1j * np.angle(entry))
    candidate.real[abs(candidate.real) < 1.0e-10] = 0.0
    candidate.imag[abs(candidate.imag) < 1.0e-10] = 0.0
    return tuple(np.round(candidate.real, 10).flat) + tuple(np.round(candidate.imag, 10).flat)


def single_qubit_clifford(phase_free: bool = False) -> list[np.ndarray]:
    h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    s = np.diag([1, 1j]).astype(np.complex128)
    generators = (h, s)
    identity = np.eye(2, dtype=np.complex128)
    seen = {matrix_key(identity, phase_free): identity}
    frontier = [identity]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            candidate = current @ generator
            key = matrix_key(candidate, phase_free)
            if key not in seen:
                seen[key] = candidate
                frontier.append(candidate)
        if len(seen) > 256:
            raise RuntimeError("Clifford closure did not terminate at the expected size")
    return list(seen.values())


def projective_qubit_clifford(qubits: int) -> list[np.ndarray]:
    """Close the projective Clifford group for one or two qubits."""
    if qubits == 1:
        return single_qubit_clifford(phase_free=True)
    if qubits != 2:
        raise ValueError("the explicit sweep is intentionally limited to one or two qubits")

    identity_2 = np.eye(2, dtype=np.complex128)
    h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    s = np.diag([1, 1j]).astype(np.complex128)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=np.complex128,
    )
    generators = (
        np.kron(h, identity_2),
        np.kron(s, identity_2),
        np.kron(identity_2, h),
        np.kron(identity_2, s),
        cnot,
    )
    identity = np.eye(4, dtype=np.complex128)
    seen = {matrix_key(identity, phase_free=True): identity}
    frontier = [identity]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            candidate = current @ generator
            key = matrix_key(candidate, phase_free=True)
            if key not in seen:
                seen[key] = candidate
                frontier.append(candidate)
        if len(seen) > 12000:
            raise RuntimeError("two-qubit Clifford closure exceeded its expected size")
    return list(seen.values())


def adjoint_matrices(projective_representatives: list[np.ndarray]) -> list[np.ndarray]:
    return [np.kron(matrix, matrix.conj()) for matrix in projective_representatives]
