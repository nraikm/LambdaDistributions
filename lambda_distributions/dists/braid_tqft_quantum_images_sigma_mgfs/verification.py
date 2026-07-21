"""Matrix-level checks for braid/TQFT image sigma-MGF formulas.

The suite deliberately separates the represented groups.  It constructs:

* the finite Ising braid lift in dimension two;
* the qutrit Heisenberg group in its monomial Schrödinger representation;
* a rank-one qutrit Weil lift together with SL(2,F_3);
* the one-qubit projective Clifford group for balanced design moments;
* regular and affine phase-space permutation groups; and
* seeded Haar matrices for U, SU, O, and USp diagnostics.

Finite-group comparisons are exhaustive.  Haar rows are statistical
diagnostics with reported standard errors and are not used as proofs.
"""

from __future__ import annotations

from collections import deque
from itertools import product
from math import comb, factorial, prod, sqrt

import numpy as np


Matrix = np.ndarray
Permutation = tuple[int, ...]
Tau = tuple[int, ...]
T_VALUES = (0.023, -0.031, 0.041)


def matrix_key(matrix: Matrix, *, phase_free: bool = False) -> tuple[float, ...]:
    candidate = np.array(matrix, dtype=np.complex128, copy=True)
    if phase_free:
        pivot = next(value for value in candidate.flat if abs(value) > 1.0e-9)
        candidate *= np.exp(-1j * np.angle(pivot))
    candidate.real[abs(candidate.real) < 1.0e-10] = 0.0
    candidate.imag[abs(candidate.imag) < 1.0e-10] = 0.0
    return tuple(np.round(candidate.real, 10).flat) + tuple(
        np.round(candidate.imag, 10).flat
    )


def close_matrix_group(
    generators: tuple[Matrix, ...], *, phase_free: bool = False, bound: int = 20_000
) -> tuple[Matrix, ...]:
    dimension = generators[0].shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    seen = {matrix_key(identity, phase_free=phase_free): identity}
    queue = deque((identity,))
    while queue:
        current = queue.popleft()
        for generator in generators:
            candidate = current @ generator
            key = matrix_key(candidate, phase_free=phase_free)
            if key not in seen:
                seen[key] = candidate
                queue.append(candidate)
                if len(seen) > bound:
                    raise RuntimeError("matrix closure exceeded its safety bound")
    return tuple(seen.values())


def complete_from_eigenvalues(eigenvalues: np.ndarray, maximum: int) -> tuple[complex, ...]:
    coefficients = np.zeros(maximum + 1, dtype=np.complex128)
    coefficients[0] = 1.0
    for eigenvalue in eigenvalues:
        for degree in range(1, maximum + 1):
            coefficients[degree] += eigenvalue * coefficients[degree - 1]
    return tuple(complex(value) for value in coefficients)


def complete_from_power_traces(matrix: Matrix, maximum: int) -> tuple[complex, ...]:
    traces = [0j] * (maximum + 1)
    current = np.eye(matrix.shape[0], dtype=np.complex128)
    for exponent in range(1, maximum + 1):
        current = current @ matrix
        traces[exponent] = complex(np.trace(current))
    complete = [1.0 + 0j]
    for degree in range(1, maximum + 1):
        complete.append(
            sum(traces[r] * complete[degree - r] for r in range(1, degree + 1))
            / degree
        )
    return tuple(complete)


def coefficient(matrices: tuple[Matrix, ...], tau: Tau, *, power_route: bool) -> complex:
    maximum = max(tau, default=0)
    total = 0j
    for matrix in matrices:
        if power_route:
            complete = complete_from_power_traces(matrix, maximum)
        else:
            complete = complete_from_eigenvalues(np.linalg.eigvals(matrix), maximum)
        total += prod(complete[degree] for degree in tau)
    return total / len(matrices)


def clean_integer(value: complex, tolerance: float = 3.0e-7) -> int:
    rounded = int(round(value.real))
    if abs(value.imag) > tolerance or abs(value.real - rounded) > tolerance:
        raise ArithmeticError(f"expected an integer, got {value!r}")
    return rounded


def determinant_mgf(matrices: tuple[Matrix, ...], values=T_VALUES) -> complex:
    dimension = matrices[0].shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    return sum(
        prod(1 / np.linalg.det(identity - t * matrix) for t in values)
        for matrix in matrices
    ) / len(matrices)


def power_character_mgf(
    matrices: tuple[Matrix, ...], values=T_VALUES, truncation: int = 60
) -> complex:
    power_sums = [0j] + [sum(t**r for t in values) for r in range(1, truncation + 1)]
    total = 0j
    for matrix in matrices:
        current = np.eye(matrix.shape[0], dtype=np.complex128)
        exponent = 0j
        for r in range(1, truncation + 1):
            current = current @ matrix
            exponent += power_sums[r] * np.trace(current) / r
        total += np.exp(exponent)
    return total / len(matrices)


def _coefficient_rows(name: str, matrices: tuple[Matrix, ...], taus: tuple[Tau, ...]):
    rows = []
    for tau in taus:
        direct = clean_integer(coefficient(matrices, tau, power_route=False))
        formula = clean_integer(coefficient(matrices, tau, power_route=True))
        rows.append(
            {
                "group / representation": name,
                "dimension": matrices[0].shape[0],
                "image order": len(matrices),
                "tau": str(tau),
                "direct matrices": direct,
                "power-character formula": formula,
                "passed": direct == formula,
            }
        )
    return tuple(rows)


def ising_braid_suite() -> dict[str, object]:
    """Finite Ising braid lift generated by two exact braid matrices."""

    hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / sqrt(2)
    phase = np.exp(-1j * np.pi / 8)
    sigma_1 = phase * np.diag([1, 1j]).astype(np.complex128)
    sigma_2 = hadamard @ sigma_1 @ hadamard
    matrices = close_matrix_group((sigma_1, sigma_2))
    projective = close_matrix_group((sigma_1, sigma_2), phase_free=True)
    braid_residual = float(
        np.linalg.norm(sigma_1 @ sigma_2 @ sigma_1 - sigma_2 @ sigma_1 @ sigma_2)
    )
    rows = _coefficient_rows(
        "Ising braid finite lift",
        matrices,
        ((1,), (2,), (4,), (8,), (7, 1), (4, 4), (1,) * 8),
    )
    direct_value = determinant_mgf(matrices)
    power_value = power_character_mgf(matrices)
    numeric_error = float(abs(direct_value - power_value))
    scalar_eighth_roots = sum(
        np.linalg.norm(matrix - matrix[0, 0] * np.eye(2)) < 2.0e-8
        for matrix in matrices
    )
    selection_ok = all(
        clean_integer(coefficient(matrices, tau, power_route=False)) == 0
        for tau in ((1,), (2,), (4,))
    )
    passed = (
        len(matrices) == 192
        and len(projective) == 24
        and scalar_eighth_roots == 8
        and braid_residual < 2.0e-12
        and numeric_error < 2.0e-12
        and selection_ok
        and all(row["passed"] for row in rows)
    )
    return {
        "name": "Ising braid / finite Clifford lift",
        "lift order": len(matrices),
        "projective order": len(projective),
        "scalar center order": scalar_eighth_roots,
        "braid residual": braid_residual,
        "coefficient rows": rows,
        "determinant MGF": direct_value,
        "power-character MGF": power_value,
        "numeric error": numeric_error,
        "passed": passed,
    }


def qutrit_pauli_group() -> tuple[Matrix, ...]:
    p = 3
    omega = np.exp(2j * np.pi / p)
    shift = np.zeros((p, p), dtype=np.complex128)
    for column in range(p):
        shift[(column + 1) % p, column] = 1
    clock = np.diag([omega**column for column in range(p)])
    return tuple(
        omega**center * np.linalg.matrix_power(shift, a) @ np.linalg.matrix_power(clock, b)
        for center, a, b in product(range(p), repeat=3)
    )


def monomial_cycles(matrix: Matrix) -> tuple[tuple[int, complex], ...]:
    dimension = matrix.shape[0]
    target = tuple(int(np.argmax(np.abs(matrix[:, column]))) for column in range(dimension))
    phase = tuple(matrix[target[column], column] for column in range(dimension))
    seen: set[int] = set()
    cycles = []
    for start in range(dimension):
        if start in seen:
            continue
        current = start
        cycle_phase = 1.0 + 0j
        length = 0
        while current not in seen:
            seen.add(current)
            cycle_phase *= phase[current]
            current = target[current]
            length += 1
        cycles.append((length, cycle_phase))
    return tuple(cycles)


def monomial_mgf(matrices: tuple[Matrix, ...], values=T_VALUES) -> complex:
    return sum(
        prod(
            1 / (1 - phase * t**length)
            for t in values
            for length, phase in monomial_cycles(matrix)
        )
        for matrix in matrices
    ) / len(matrices)


def pauli_prediction(tau: Tau) -> int:
    if sum(tau) % 3:
        return 0
    central = prod(comb(3 + degree - 1, degree) for degree in tau) / 9
    noncentral = (
        8
        * prod(1 if degree % 3 == 0 else 0 for degree in tau)
        / 9
    )
    value = central + noncentral
    rounded = round(value)
    if abs(value - rounded) > 1.0e-9:
        raise ArithmeticError(value)
    return int(rounded)


def heisenberg_suite() -> dict[str, object]:
    matrices = qutrit_pauli_group()
    taus = ((1,), (2,), (3,), (2, 1), (4,), (3, 3), (6,))
    rows = []
    for tau in taus:
        direct = clean_integer(coefficient(matrices, tau, power_route=False))
        formula = pauli_prediction(tau)
        rows.append(
            {
                "group / representation": "H_3 Schrödinger monomial",
                "dimension": 3,
                "image order": 27,
                "tau": str(tau),
                "direct matrices": direct,
                "closed spectral formula": formula,
                "passed": direct == formula,
            }
        )
    dense_value = determinant_mgf(matrices)
    cycle_value = monomial_mgf(matrices)
    error = float(abs(dense_value - cycle_value))
    return {
        "name": "qutrit finite Heisenberg image",
        "order": len(matrices),
        "coefficient rows": tuple(rows),
        "dense determinant MGF": dense_value,
        "cycle-phase MGF": cycle_value,
        "numeric error": error,
        "passed": len(matrices) == 27 and error < 2.0e-12 and all(row["passed"] for row in rows),
    }


def _matrix2_product(left: tuple[int, ...], right: tuple[int, ...], p: int):
    product_matrix = (
        np.array(left, dtype=int).reshape(2, 2)
        @ np.array(right, dtype=int).reshape(2, 2)
    ) % p
    return tuple(int(value) for value in product_matrix.flat)


def _kernel_size_a_minus_i(matrix: tuple[int, ...], p: int) -> int:
    candidate = (np.array(matrix, dtype=int).reshape(2, 2) - np.eye(2, dtype=int)) % p
    return sum(
        np.all(candidate @ np.array(vector, dtype=int) % p == 0)
        for vector in product(range(p), repeat=2)
    )


def projective_weil_qutrit() -> tuple[tuple[Matrix, tuple[int, ...]], ...]:
    p = 3
    omega = np.exp(2j * np.pi / p)
    fourier = np.array(
        [[omega ** (row * column) for column in range(p)] for row in range(p)],
        dtype=np.complex128,
    ) / sqrt(p)
    inv_two = pow(2, -1, p)
    chirp = np.diag([omega ** ((inv_two * x * x) % p) for x in range(p)])
    symplectic_s = (0, -1 % p, 1, 0)
    symplectic_t = (1, 0, 1, 1)
    identity_u = np.eye(p, dtype=np.complex128)
    identity_a = (1, 0, 0, 1)
    seen = {matrix_key(identity_u, phase_free=True): (identity_u, identity_a)}
    queue = deque((matrix_key(identity_u, phase_free=True),))
    for_key = ((fourier, symplectic_s), (chirp, symplectic_t))
    while queue:
        key = queue.popleft()
        unitary, symplectic = seen[key]
        for generator_u, generator_a in for_key:
            candidate_u = unitary @ generator_u
            candidate_a = _matrix2_product(symplectic, generator_a, p)
            candidate_key = matrix_key(candidate_u, phase_free=True)
            if candidate_key in seen:
                if seen[candidate_key][1] != candidate_a:
                    raise AssertionError("projective Weil phase tracking is inconsistent")
            else:
                seen[candidate_key] = (candidate_u, candidate_a)
                queue.append(candidate_key)
    return tuple(seen.values())


def weil_suite() -> dict[str, object]:
    representatives = projective_weil_qutrit()
    p = 3
    omega = np.exp(2j * np.pi / p)
    fourier = np.array(
        [[omega ** (row * column) for column in range(p)] for row in range(p)],
        dtype=np.complex128,
    ) / sqrt(p)
    chirp = np.diag([omega ** ((pow(2, -1, p) * x * x) % p) for x in range(p)])
    lift = close_matrix_group((fourier, chirp))
    magnitude_rows = []
    for unitary, symplectic in representatives:
        direct = float(abs(np.trace(unitary)) ** 2)
        formula = _kernel_size_a_minus_i(symplectic, p)
        magnitude_rows.append(
            {
                "A": str(symplectic),
                "|Tr omega(A)|^2": direct,
                "|ker(A-I)|": formula,
                "passed": abs(direct - formula) < 2.0e-9,
            }
        )
    rows = _coefficient_rows(
        "q=3 rank-one Weil lift", lift, ((1,), (2,), (4,), (3, 1), (4, 4))
    )
    dense_value = determinant_mgf(lift)
    power_value = power_character_mgf(lift)
    error = float(abs(dense_value - power_value))
    return {
        "name": "rank-one qutrit Weil / torus mapping-class quotient",
        "projective symplectic order": len(representatives),
        "lift order": len(lift),
        "magnitude rows": tuple(magnitude_rows),
        "coefficient rows": rows,
        "numeric error": error,
        "passed": (
            len(representatives) == 24
            and len(lift) == 96
            and all(row["passed"] for row in magnitude_rows)
            and all(row["passed"] for row in rows)
            and error < 2.0e-12
        ),
    }


def _partitions(total: int, maximum: int | None = None):
    if total == 0:
        yield ()
        return
    maximum = total if maximum is None else min(total, maximum)
    for first in range(maximum, 0, -1):
        for tail in _partitions(total - first, first):
            yield (first, *tail)


def _tableau_count(shape: tuple[int, ...]) -> int:
    hooks = 1
    for row, width in enumerate(shape):
        for column in range(width):
            below = sum(column < other for other in shape[row + 1 :])
            hooks *= width - column + below
    return factorial(sum(shape)) // hooks


def haar_frame_potential(dimension: int, degree: int) -> int:
    return sum(
        _tableau_count(shape) ** 2
        for shape in _partitions(degree)
        if len(shape) <= dimension
    )


def clifford_design_suite() -> dict[str, object]:
    hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / sqrt(2)
    phase = np.diag([1, 1j]).astype(np.complex128)
    group = close_matrix_group((hadamard, phase), phase_free=True)
    rows = []
    for degree in range(1, 5):
        direct = round(float(np.mean([abs(np.trace(matrix)) ** (2 * degree) for matrix in group])))
        haar = haar_frame_potential(2, degree)
        expected_equality = degree <= 3
        rows.append(
            {
                "degree": degree,
                "direct Clifford frame potential": direct,
                "Haar U(2)": haar,
                "expected relation": "equal" if expected_equality else "different",
                "passed": (direct == haar) == expected_equality,
            }
        )
    return {
        "name": "one-qubit projective Clifford balanced series",
        "projective order": len(group),
        "rows": tuple(rows),
        "passed": len(group) == 24 and all(row["passed"] for row in rows),
    }


def compose_permutations(left: Permutation, right: Permutation) -> Permutation:
    return tuple(left[right[index]] for index in range(len(right)))


def permutation_power(permutation: Permutation, exponent: int) -> Permutation:
    answer = tuple(range(len(permutation)))
    base = permutation
    while exponent:
        if exponent & 1:
            answer = compose_permutations(answer, base)
        base = compose_permutations(base, base)
        exponent //= 2
    return answer


def permutation_matrix(permutation: Permutation) -> Matrix:
    matrix = np.zeros((len(permutation), len(permutation)), dtype=np.complex128)
    for source, target in enumerate(permutation):
        matrix[target, source] = 1
    return matrix


def cycle_lengths(permutation: Permutation) -> tuple[int, ...]:
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


def cycle_complete(lengths: tuple[int, ...], maximum: int) -> tuple[int, ...]:
    coefficients = [0] * (maximum + 1)
    coefficients[0] = 1
    for length in lengths:
        for degree in range(length, maximum + 1):
            coefficients[degree] += coefficients[degree - length]
    return tuple(coefficients)


def regular_translations(p: int) -> tuple[Permutation, ...]:
    points = tuple(product(range(p), repeat=2))
    index = {point: position for position, point in enumerate(points)}
    return tuple(
        tuple(index[((x + a) % p, (y + b) % p)] for x, y in points)
        for a, b in points
    )


def regular_formula_coefficient(p: int, tau: Tau) -> int:
    size = p * p
    identity = prod(comb(size + degree - 1, degree) for degree in tau)
    nonidentity = prod(
        comb(p + degree // p - 1, degree // p) if degree % p == 0 else 0
        for degree in tau
    )
    numerator = identity + (size - 1) * nonidentity
    assert numerator % size == 0
    return numerator // size


def _permutation_coefficient(permutations: tuple[Permutation, ...], tau: Tau) -> int:
    numerator = 0
    for permutation in permutations:
        complete = cycle_complete(cycle_lengths(permutation), max(tau))
        numerator += prod(complete[degree] for degree in tau)
    assert numerator % len(permutations) == 0
    return numerator // len(permutations)


def sl2_matrices(p: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        (a, b, c, d)
        for a, b, c, d in product(range(p), repeat=4)
        if (a * d - b * c) % p == 1
    )


def affine_actions(p: int):
    points = tuple(product(range(p), repeat=2))
    index = {point: position for position, point in enumerate(points)}
    answer = []
    for matrix in sl2_matrices(p):
        a = np.array(matrix, dtype=int).reshape(2, 2)
        for translation in points:
            b = np.array(translation, dtype=int)
            permutation = tuple(
                index[tuple(int(value) for value in ((a @ np.array(point) + b) % p))]
                for point in points
            )
            answer.append((permutation, matrix, translation))
    return tuple(answer)


def affine_fixed_formula(
    matrix: tuple[int, ...], translation: tuple[int, int], exponent: int, p: int
) -> int:
    a = np.array(matrix, dtype=int).reshape(2, 2)
    current = np.eye(2, dtype=int)
    geometric = np.zeros((2, 2), dtype=int)
    for _ in range(exponent):
        geometric = (geometric + current) % p
        current = current @ a % p
    right = geometric @ np.array(translation, dtype=int) % p
    linear = (np.eye(2, dtype=int) - current) % p
    kernel = 0
    image = set()
    for vector in product(range(p), repeat=2):
        value = tuple(int(entry) for entry in (linear @ np.array(vector, dtype=int) % p))
        image.add(value)
        if value == (0, 0):
            kernel += 1
    return kernel if tuple(int(entry) for entry in right) in image else 0


def phase_space_suite() -> dict[str, object]:
    regular_rows = []
    numeric_rows = []
    for p in (2, 3, 5):
        permutations = regular_translations(p)
        matrices = tuple(permutation_matrix(permutation) for permutation in permutations)
        for tau in ((1, 1), (2,), (p, 1), (p, p)):
            direct = _permutation_coefficient(permutations, tau)
            formula = regular_formula_coefficient(p, tau)
            regular_rows.append(
                {
                    "p": p,
                    "degree": p * p,
                    "tau": str(tau),
                    "direct permutation matrices": direct,
                    "regular formula": formula,
                    "passed": direct == formula,
                }
            )
        dense = determinant_mgf(matrices)
        identity_factor = prod((1 - t) ** (-(p * p)) for t in T_VALUES)
        moving_factor = prod((1 - t**p) ** (-p) for t in T_VALUES)
        formula_value = (identity_factor + (p * p - 1) * moving_factor) / (p * p)
        numeric_rows.append(
            {
                "p": p,
                "direct": dense,
                "formula": formula_value,
                "error": float(abs(dense - formula_value)),
                "passed": abs(dense - formula_value) < 3.0e-12,
            }
        )

    affine_rows = []
    fixed_checks = 0
    for p in (2, 3):
        actions = affine_actions(p)
        permutations = tuple(row[0] for row in actions)
        for permutation, matrix, translation in actions:
            for exponent in range(1, 7):
                actual = sum(
                    index == image
                    for index, image in enumerate(permutation_power(permutation, exponent))
                )
                predicted = affine_fixed_formula(matrix, translation, exponent, p)
                if actual != predicted:
                    raise AssertionError((p, matrix, translation, exponent, actual, predicted))
                fixed_checks += 1
        pair_coefficient = sum(
            sum(index == image for index, image in enumerate(permutation)) ** 2
            for permutation in permutations
        ) // len(permutations)
        affine_rows.append(
            {
                "p": p,
                "permutation degree": p * p,
                "affine group order": len(actions),
                "direct pair coefficient": pair_coefficient,
                "orbit formula": 2,
                "passed": pair_coefficient == 2,
            }
        )
    return {
        "name": "Weyl-Heisenberg regular and affine phase-space actions",
        "regular rows": tuple(regular_rows),
        "numeric rows": tuple(numeric_rows),
        "affine rows": tuple(affine_rows),
        "fixed-power checks": fixed_checks,
        "passed": all(
            row["passed"] for row in (*regular_rows, *numeric_rows, *affine_rows)
        ),
    }


def haar_unitary(dimension: int, rng: np.random.Generator) -> Matrix:
    sample = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    q, r = np.linalg.qr(sample)
    diagonal = np.diag(r)
    phases = diagonal / np.abs(diagonal)
    return q * phases.conjugate()


def haar_special_unitary(dimension: int, rng: np.random.Generator) -> Matrix:
    unitary = haar_unitary(dimension, rng)
    return unitary / np.linalg.det(unitary) ** (1 / dimension)


def haar_orthogonal(dimension: int, rng: np.random.Generator) -> Matrix:
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.where(np.diag(r) >= 0, 1.0, -1.0)
    return q * signs


def haar_symplectic(rank: int, rng: np.random.Generator) -> Matrix:
    size = 2 * rank
    identity = np.eye(rank)
    zero = np.zeros((rank, rank))
    form = np.block([[zero, identity], [-identity, zero]]).astype(complex)
    first: list[Matrix] = []
    paired: list[Matrix] = []
    for _ in range(rank):
        vector = rng.normal(size=size) + 1j * rng.normal(size=size)
        for column in first + paired:
            vector -= column * np.vdot(column, vector)
        vector /= np.linalg.norm(vector)
        pair = -form @ vector.conjugate()
        first.append(vector)
        paired.append(pair)
    return np.column_stack(first + paired)


def _haar_row(
    group: str,
    dimension: int,
    tau: Tau,
    target: int,
    samples: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    if group == "U":
        sampler = lambda: haar_unitary(dimension, rng)
    elif group == "SU":
        sampler = lambda: haar_special_unitary(dimension, rng)
    elif group == "O":
        sampler = lambda: haar_orthogonal(dimension, rng)
    elif group == "USp":
        sampler = lambda: haar_symplectic(dimension // 2, rng)
    else:
        raise ValueError(group)
    values = []
    worst_residual = 0.0
    for _ in range(samples):
        matrix = sampler()
        complete = complete_from_power_traces(matrix, max(tau))
        values.append(prod(complete[degree] for degree in tau))
        worst_residual = max(
            worst_residual,
            float(np.linalg.norm(matrix.conjugate().T @ matrix - np.eye(dimension))),
        )
    data = np.asarray(values, dtype=np.complex128)
    estimate = complex(np.mean(data))
    standard_error = float(
        sqrt(np.sum(np.abs(data - estimate) ** 2) / (samples - 1)) / sqrt(samples)
    )
    tolerance = max(0.10, 5.0 * standard_error)
    return {
        "group": f"{group}({dimension})",
        "tau": str(tau),
        "samples": samples,
        "direct Haar estimate": estimate,
        "formula target": target,
        "standard error": standard_error,
        "unitarity residual": worst_residual,
        "passed": abs(estimate - target) <= tolerance and worst_residual < 2.0e-12,
    }


def compact_closure_suite(samples: int = 3_000) -> dict[str, object]:
    specifications = (
        ("U", 4, (1,), 0),
        ("U", 4, (2,), 0),
        ("U", 4, (1, 1), 0),
        ("SU", 2, (1,), 0),
        ("SU", 2, (2,), 0),
        ("SU", 2, (1, 1), 1),
        ("SU", 3, (1, 1), 0),
        ("SU", 3, (1, 1, 1), 1),
        ("O", 6, (2,), 1),
        ("O", 6, (1, 1), 1),
        ("O", 6, (1, 1, 1, 1), 3),
        ("O", 6, (2, 2), 2),
        ("USp", 4, (2,), 0),
        ("USp", 4, (1, 1), 1),
        ("USp", 4, (1, 1, 1, 1), 3),
        ("USp", 4, (2, 2), 1),
    )
    rows = tuple(
        _haar_row(group, dimension, tau, target, samples, 20260720 + index)
        for index, (group, dimension, tau, target) in enumerate(specifications)
    )
    return {
        "name": "compact-closure Haar diagnostics",
        "rows": rows,
        "samples per row": samples,
        "passed": all(row["passed"] for row in rows),
    }


def run_suite(haar_samples: int = 3_000) -> dict[str, object]:
    ising = ising_braid_suite()
    heisenberg = heisenberg_suite()
    weil = weil_suite()
    design = clifford_design_suite()
    phase_space = phase_space_suite()
    compact = compact_closure_suite(haar_samples)
    sections = (ising, heisenberg, weil, design, phase_space, compact)
    return {
        "sections": sections,
        "ising": ising,
        "heisenberg": heisenberg,
        "weil": weil,
        "design": design,
        "phase space": phase_space,
        "compact closures": compact,
        "finite coefficient comparisons": (
            len(ising["coefficient rows"])
            + len(heisenberg["coefficient rows"])
            + len(weil["coefficient rows"])
            + len(phase_space["regular rows"])
            + len(design["rows"])
        ),
        "fixed-power checks": phase_space["fixed-power checks"],
        "haar diagnostics": len(compact["rows"]),
        "passed": all(section["passed"] for section in sections),
    }


if __name__ == "__main__":
    report = run_suite()
    print("PASS" if report["passed"] else "FAIL")
    print(
        report["finite coefficient comparisons"],
        "finite comparisons;",
        report["fixed-power checks"],
        "affine fixed-power checks;",
        report["haar diagnostics"],
        "Haar diagnostics",
    )
