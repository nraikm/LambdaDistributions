"""Numerical matrix samples retained from the exploratory notebooks."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def random_unitary(dimension: int, rng: np.random.Generator | None = None) -> NDArray[np.complex128]:
    """Draw a Haar-distributed unitary matrix using complex QR decomposition."""

    if dimension < 1:
        raise ValueError("dimension must be positive")
    rng = rng or np.random.default_rng()
    sample = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(sample)
    diagonal = np.diag(r)
    phases = np.divide(diagonal, np.abs(diagonal), out=np.ones_like(diagonal), where=np.abs(diagonal) > 0)
    return np.asarray(q * phases, dtype=complex)


def random_symplectic(half_dimension: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
    """Construct a random real symplectic matrix of size ``2*half_dimension``.

    For an invertible matrix ``A``, ``diag(A, A^{-T})`` is symplectic.  This is
    a convenient numerical sampler, not a claim of a uniform distribution on
    the non-compact symplectic group.
    """

    if half_dimension < 1:
        raise ValueError("half_dimension must be positive")
    rng = rng or np.random.default_rng()
    while True:
        a = rng.normal(size=(half_dimension, half_dimension))
        if abs(np.linalg.det(a)) > 1e-8:
            break
    inverse_transpose = np.linalg.inv(a).T
    zero = np.zeros_like(a)
    return np.block([[a, zero], [zero, inverse_transpose]])


def log_sigma_mgf(matrix: ArrayLike, t: complex, max_power: int = 20) -> complex:
    """Compute ``sum_{r=1}^max_power t^r Tr(M^r)/r``."""

    square = np.asarray(matrix, dtype=complex)
    if square.ndim != 2 or square.shape[0] != square.shape[1]:
        raise ValueError("matrix must be square")
    if max_power < 1:
        raise ValueError("max_power must be positive")
    power = np.eye(square.shape[0], dtype=complex)
    result = 0j
    for degree in range(1, max_power + 1):
        power = power @ square
        result += t**degree * np.trace(power) / degree
    return complex(result)


def sigma_mgf(matrix: ArrayLike, t: complex, max_power: int = 20) -> complex:
    """Exponentiate :func:`log_sigma_mgf`."""

    return complex(np.exp(log_sigma_mgf(matrix, t, max_power)))


def is_unitary(matrix: ArrayLike, tolerance: float = 1e-10) -> bool:
    square = np.asarray(matrix, dtype=complex)
    if square.ndim != 2 or square.shape[0] != square.shape[1]:
        return False
    identity = np.eye(square.shape[0], dtype=complex)
    return bool(np.allclose(square.conjugate().T @ square, identity, atol=tolerance, rtol=0))


def is_symplectic(matrix: ArrayLike, tolerance: float = 1e-10) -> bool:
    square = np.asarray(matrix, dtype=float)
    if square.ndim != 2 or square.shape[0] != square.shape[1] or square.shape[0] % 2:
        return False
    half = square.shape[0] // 2
    identity = np.eye(half)
    zero = np.zeros((half, half))
    form = np.block([[zero, identity], [-identity, zero]])
    return bool(np.allclose(square.T @ form @ square, form, atol=tolerance, rtol=0))
