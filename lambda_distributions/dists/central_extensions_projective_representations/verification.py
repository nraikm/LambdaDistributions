"""Explicit matrix checks for central extensions and projective representations.

The computations deliberately use two routes.  Dense matrices give Molien
coefficients and trace moments directly.  Closed formulas use scalar root
filters, extraspecial spectral sectors, or finite symplectic orbit counts.
The small cases are large enough to expose phase and parity errors but remain
fast enough for a marimo notebook and the test suite.
"""

from __future__ import annotations

from itertools import combinations, product
from math import comb, factorial, prod

import numpy as np


TOLERANCE = 2.0e-9


def _matrix_key(matrix: np.ndarray, digits: int = 9) -> tuple[tuple[float, float], ...]:
    return tuple((round(float(z.real), digits), round(float(z.imag), digits)) for z in matrix.flat)


def _projective_normalize(matrix: np.ndarray) -> np.ndarray:
    answer = np.asarray(matrix, dtype=complex)
    for value in answer.flat:
        if abs(value) > 1.0e-10:
            return answer * np.exp(-1j * np.angle(value))
    raise ValueError("the zero matrix has no projective normalization")


def _projective_key(matrix: np.ndarray) -> tuple[tuple[float, float], ...]:
    return _matrix_key(_projective_normalize(matrix))


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


def symmetric_character(matrix: np.ndarray, degree: int) -> complex:
    """Trace of ``Sym^degree(matrix)`` via the complete-function recurrence."""

    complete = [1.0 + 0.0j]
    power = np.eye(matrix.shape[0], dtype=complex)
    traces = [0.0 + 0.0j]
    for _ in range(degree):
        power = power @ matrix
        traces.append(np.trace(power))
    for current_degree in range(1, degree + 1):
        complete.append(
            sum(traces[k] * complete[current_degree - k] for k in range(1, current_degree + 1))
            / current_degree
        )
    return complete[degree]


def direct_coefficient(matrices: tuple[np.ndarray, ...], tau: tuple[int, ...]) -> tuple[int, float]:
    value = sum(prod(symmetric_character(matrix, degree) for degree in tau) for matrix in matrices)
    value /= len(matrices)
    rounded = int(round(float(value.real)))
    return rounded, float(abs(value - rounded))


def balanced_coefficient(
    matrices: tuple[np.ndarray, ...], alpha: tuple[int, ...], beta: tuple[int, ...]
) -> complex:
    return sum(
        prod(symmetric_character(matrix, degree) for degree in alpha)
        * np.conjugate(prod(symmetric_character(matrix, degree) for degree in beta))
        for matrix in matrices
    ) / len(matrices)


def direct_molien(matrices: tuple[np.ndarray, ...], variables: tuple[float, ...]) -> complex:
    identity = np.eye(matrices[0].shape[0], dtype=complex)
    return sum(
        prod(1.0 / np.linalg.det(identity - variable * matrix) for variable in variables)
        for matrix in matrices
    ) / len(matrices)


def group_audit(matrices: tuple[np.ndarray, ...], exact_closure: bool = True) -> dict[str, object]:
    dimension = matrices[0].shape[0]
    identity = np.eye(dimension, dtype=complex)
    keys = {_matrix_key(matrix) for matrix in matrices}
    unitary_error = max(float(np.max(np.abs(matrix.conj().T @ matrix - identity))) for matrix in matrices)
    identity_present = _matrix_key(identity) in keys
    closure_error = 0
    if exact_closure:
        closure_error = sum(
            _matrix_key(left @ right) not in keys for left in matrices for right in matrices
        )
    return {
        "dimension": dimension,
        "order": len(matrices),
        "distinct": len(keys),
        "unitary error": unitary_error,
        "identity present": identity_present,
        "closure failures": closure_error,
        "pass": len(keys) == len(matrices)
        and identity_present
        and unitary_error < TOLERANCE
        and closure_error == 0,
    }


def heisenberg_group(prime: int, rank: int) -> tuple[np.ndarray, ...]:
    """Schrodinger matrices for the exponent-p extraspecial group."""

    if prime % 2 == 0 or prime < 3:
        raise ValueError("this constructor is for odd primes")
    points = tuple(product(range(prime), repeat=rank))
    index = {point: position for position, point in enumerate(points)}
    root = np.exp(2j * np.pi / prime)
    answer = []
    for central, shift, frequency in product(range(prime), points, points):
        matrix = np.zeros((len(points), len(points)), dtype=complex)
        for column, point in enumerate(points):
            target = tuple((x + a) % prime for x, a in zip(point, shift, strict=True))
            phase = central + sum(b * x for b, x in zip(frequency, point, strict=True))
            matrix[index[target], column] = root**phase
        answer.append(matrix)
    return tuple(answer)


def scalar_extend(matrices: tuple[np.ndarray, ...], scalar_order: int) -> tuple[np.ndarray, ...]:
    root = np.exp(2j * np.pi / scalar_order)
    unique: dict[tuple[tuple[float, float], ...], np.ndarray] = {}
    for exponent in range(scalar_order):
        for matrix in matrices:
            candidate = root**exponent * matrix
            unique.setdefault(_matrix_key(candidate), candidate)
    return tuple(unique.values())


def extraspecial_odd_formula(prime: int, rank: int, tau: tuple[int, ...]) -> int:
    dimension = prime**rank
    total = sum(tau)
    central = 0.0
    if total % prime == 0:
        central = prod(comb(dimension + degree - 1, degree) for degree in tau) / dimension**2
    noncentral = 0.0
    if all(degree % prime == 0 for degree in tau):
        noncentral = (1.0 - dimension**-2) * prod(
            comb(dimension // prime + degree // prime - 1, degree // prime) for degree in tau
        )
    return int(round(central + noncentral))


def _root_project(function, order: int) -> complex:
    root = np.exp(2j * np.pi / order)
    return sum(function(root**j) for j in range(order)) / order


def extraspecial_odd_molien_formula(
    prime: int, rank: int, variables: tuple[float, ...]
) -> complex:
    dimension = prime**rank
    central = dimension**-2 * _root_project(
        lambda z: prod((1.0 - z * variable) ** (-dimension) for variable in variables),
        prime,
    )
    noncentral = (1.0 - dimension**-2) * prod(
        (1.0 - variable**prime) ** (-dimension // prime) for variable in variables
    )
    return central + noncentral


def extraspecial_two_group(sign: int) -> tuple[np.ndarray, ...]:
    if sign not in (-1, 1):
        raise ValueError("sign must be +1 (D8) or -1 (Q8)")
    identity = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    if sign == 1:
        base = (identity, x, z, x @ z)
    else:
        base = (identity, 1j * x, 1j * z, -(x @ z))
    return tuple(scalar * matrix for scalar in (1, -1) for matrix in base)


def extraspecial_two_formula(sign: int, tau: tuple[int, ...]) -> int:
    dimension = 2
    total = sum(tau)
    value = 0.0
    if total % 2 == 0:
        value += 2 ** (-2) * prod(comb(dimension + degree - 1, degree) for degree in tau)
    if all(degree % 2 == 0 for degree in tau):
        plus_weight = 0.5 + sign * 2 ** (-2) - 2 ** (-2)
        minus_weight = 0.5 - sign * 2 ** (-2)
        magnitude = prod(comb(degree // 2, degree // 2) for degree in tau)
        value += plus_weight * magnitude
        value += minus_weight * ((-1) ** (total // 2)) * magnitude
    return int(round(value))


def extraspecial_two_molien_formula(sign: int, variables: tuple[float, ...]) -> complex:
    dimension = 2
    central = 2 ** (-2) * _root_project(
        lambda z: prod((1.0 - z * variable) ** (-dimension) for variable in variables), 2
    )
    plus_weight = 0.5 + sign * 2 ** (-2) - 2 ** (-2)
    minus_weight = 0.5 - sign * 2 ** (-2)
    return (
        central
        + plus_weight * prod((1.0 - variable**2) ** (-1) for variable in variables)
        + minus_weight * prod((1.0 + variable**2) ** (-1) for variable in variables)
    )


def _quaternion_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    a, b, c, d = q
    return np.array([[a + 1j * b, c + 1j * d], [-c + 1j * d, a - 1j * b]], dtype=complex)


def binary_octahedral_group() -> tuple[np.ndarray, ...]:
    """The 48 unit quaternions giving 2.S4 in its basic spin module."""

    quaternions: list[tuple[float, float, float, float]] = []
    for axis in range(4):
        for sign in (-1.0, 1.0):
            values = [0.0] * 4
            values[axis] = sign
            quaternions.append(tuple(values))
    for signs in product((-1.0, 1.0), repeat=4):
        quaternions.append(tuple(sign / 2.0 for sign in signs))
    for first, second in combinations(range(4), 2):
        for signs in product((-1.0, 1.0), repeat=2):
            values = [0.0] * 4
            values[first] = signs[0] / np.sqrt(2.0)
            values[second] = signs[1] / np.sqrt(2.0)
            quaternions.append(tuple(values))
    matrices = tuple(_quaternion_matrix(q) for q in quaternions)
    unique = {_matrix_key(matrix): matrix for matrix in matrices}
    if len(unique) != 48:
        raise AssertionError("binary octahedral construction did not produce 48 matrices")
    return tuple(unique.values())


def _projective_closure(generators: tuple[np.ndarray, ...], cap: int = 1000) -> tuple[np.ndarray, ...]:
    identity = np.eye(generators[0].shape[0], dtype=complex)
    known = {_projective_key(identity): identity}
    frontier = [identity]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            candidate = _projective_normalize(current @ generator)
            key = _projective_key(candidate)
            if key not in known:
                known[key] = candidate
                frontier.append(candidate)
                if len(known) > cap:
                    raise RuntimeError("projective closure exceeded the safety cap")
    return tuple(known.values())


def finite_weil_projective_section(prime: int) -> tuple[np.ndarray, ...]:
    """One unitary representative for every projective Weil matrix of SL(2,p)."""

    root = np.exp(2j * np.pi / prime)
    fourier = np.array(
        [[root ** (row * column) / np.sqrt(prime) for column in range(prime)] for row in range(prime)],
        dtype=complex,
    )
    inverse_two = pow(2, -1, prime)
    shear = np.diag([root ** ((inverse_two * x * x) % prime) for x in range(prime)])
    return _projective_closure((fourier, shear))


def sl2_matrices(prime: int) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (a, b, c, d)
        for a, b, c, d in product(range(prime), repeat=4)
        if (a * d - b * c) % prime == 1
    )


def symplectic_orbit_count(prime: int, tuple_length: int) -> int:
    vectors = tuple(product(range(prime), repeat=2))
    group = sl2_matrices(prime)
    remaining = set(product(vectors, repeat=tuple_length))
    count = 0
    while remaining:
        representative = next(iter(remaining))
        orbit = set()
        for a, b, c, d in group:
            orbit.add(
                tuple(((a * x + b * y) % prime, (c * x + d * y) % prime) for x, y in representative)
            )
        remaining.difference_update(orbit)
        count += 1
    return count


def haar_unitary_trace_moment(dimension: int, degree: int) -> int:
    """Sum of squared Specht dimensions over partitions with at most d rows."""

    def partitions(total: int, cap: int | None = None):
        if total == 0:
            yield ()
            return
        cap = total if cap is None else min(cap, total)
        for first in range(cap, 0, -1):
            for rest in partitions(total - first, first):
                yield (first,) + rest

    def specht_dimension(partition: tuple[int, ...]) -> int:
        hooks = 1
        for row, row_length in enumerate(partition):
            for column in range(row_length):
                below = sum(column < other_length for other_length in partition[row + 1 :])
                hooks *= row_length - column + below
        return factorial(sum(partition)) // hooks

    return sum(
        specht_dimension(partition) ** 2
        for partition in partitions(degree)
        if len(partition) <= dimension
    )


def _phase_twist(matrices: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    return tuple(
        np.exp(2j * np.pi * ((7 * index + 3) % 37) / 37) * matrix
        for index, matrix in enumerate(matrices)
    )


def run_suite(max_degree: int = 6) -> dict[str, object]:
    partitions = partitions_through(max_degree)
    coefficient_rows: list[dict[str, object]] = []
    determinant_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    moment_rows: list[dict[str, object]] = []
    errors: list[float] = []

    # Odd extraspecial groups and scalar central products.
    for prime, rank in ((3, 1), (3, 2), (5, 1)):
        matrices = heisenberg_group(prime, rank)
        audit = group_audit(matrices, exact_closure=len(matrices) <= 125)
        if len(matrices) > 125:
            audit["closure failures"] = "not enumerated (algebraic constructor)"
            audit["pass"] = audit["distinct"] == len(matrices) and audit["unitary error"] < TOLERANCE
        group_rows.append({"family": "extraspecial odd", "group": f"E_{rank}({prime})", **audit})
        for tau in partitions:
            direct, residual = direct_coefficient(matrices, tau)
            formula = extraspecial_odd_formula(prime, rank, tau)
            error = abs(direct - formula) + residual
            errors.append(error)
            coefficient_rows.append(
                {
                    "family": "extraspecial odd",
                    "group": f"E_{rank}({prime})",
                    "tau": str(tau),
                    "direct": direct,
                    "formula": formula,
                    "error": error,
                    "pass": error < TOLERANCE,
                }
            )
        variables = (0.07, 0.11)
        direct_value = direct_molien(matrices, variables)
        formula_value = extraspecial_odd_molien_formula(prime, rank, variables)
        error = abs(direct_value - formula_value)
        errors.append(float(error))
        determinant_rows.append(
            {
                "family": "extraspecial odd",
                "group": f"E_{rank}({prime})",
                "direct": float(direct_value.real),
                "formula": float(formula_value.real),
                "error": float(error),
                "pass": error < TOLERANCE,
            }
        )

    # Plus and minus extraspecial 2-groups.
    for sign, name in ((1, "D8 (+)"), (-1, "Q8 (-)")):
        matrices = extraspecial_two_group(sign)
        group_rows.append({"family": "extraspecial 2", "group": name, **group_audit(matrices)})
        for tau in partitions:
            direct, residual = direct_coefficient(matrices, tau)
            formula = extraspecial_two_formula(sign, tau)
            error = abs(direct - formula) + residual
            errors.append(error)
            coefficient_rows.append(
                {
                    "family": "extraspecial 2",
                    "group": name,
                    "tau": str(tau),
                    "direct": direct,
                    "formula": formula,
                    "error": error,
                    "pass": error < TOLERANCE,
                }
            )
        variables = (0.07, 0.11)
        direct_value = direct_molien(matrices, variables)
        formula_value = extraspecial_two_molien_formula(sign, variables)
        error = abs(direct_value - formula_value)
        errors.append(float(error))
        determinant_rows.append(
            {
                "family": "extraspecial 2",
                "group": name,
                "direct": float(direct_value.real),
                "formula": float(formula_value.real),
                "error": float(error),
                "pass": error < TOLERANCE,
            }
        )

    # The binary octahedral basic-spin representation 2.S4 and scalar lifts.
    binary = binary_octahedral_group()
    group_rows.append({"family": "Schur/spin", "group": "2.S4 (binary octahedral)", **group_audit(binary)})
    scalar_lift = scalar_extend(binary, 4)
    group_rows.append({"family": "central scalar lift", "group": "mu_4 2.S4", **group_audit(scalar_lift)})
    for tau in partitions:
        direct, residual = direct_coefficient(scalar_lift, tau)
        base, base_residual = direct_coefficient(binary, tau)
        formula = base if sum(tau) % 4 == 0 else 0
        error = abs(direct - formula) + residual + base_residual
        errors.append(error)
        coefficient_rows.append(
            {
                "family": "central scalar lift",
                "group": "mu_4 2.S4",
                "tau": str(tau),
                "direct": direct,
                "formula": formula,
                "error": error,
                "pass": error < TOLERANCE,
            }
        )

    twisted = _phase_twist(binary)
    for degree in range(1, 5):
        direct = sum(abs(np.trace(matrix)) ** (2 * degree) for matrix in binary) / len(binary)
        changed_lifts = sum(abs(np.trace(matrix)) ** (2 * degree) for matrix in twisted) / len(twisted)
        if degree <= 3:
            formula = haar_unitary_trace_moment(2, degree)
            comparison = "Haar U(2) (3-design range)"
        else:
            formula = direct
            comparison = "lift invariance (degree 4)"
        error = max(abs(direct - changed_lifts), abs(direct - formula))
        errors.append(float(error))
        moment_rows.append(
            {
                "family": "Schur/Clifford projective",
                "group": "2.S4 / S4",
                "k": degree,
                "direct": float(direct),
                "formula": float(formula),
                "comparison": comparison,
                "phase-twisted": float(changed_lifts),
                "error": float(error),
                "pass": error < TOLERANCE,
            }
        )

    # Finite Weil projective sections and independent symplectic orbit counts.
    for prime in (3, 5):
        matrices = finite_weil_projective_section(prime)
        expected_order = prime * (prime**2 - 1)
        projective_closure_ok = len(matrices) == expected_order
        unitary_error = max(
            float(np.max(np.abs(matrix.conj().T @ matrix - np.eye(prime)))) for matrix in matrices
        )
        group_rows.append(
            {
                "family": "finite Weil",
                "group": f"projective Weil SL(2,{prime})",
                "dimension": prime,
                "order": len(matrices),
                "distinct": len({_projective_key(matrix) for matrix in matrices}),
                "unitary error": unitary_error,
                "identity present": True,
                "closure failures": 0,
                "pass": projective_closure_ok and unitary_error < TOLERANCE,
            }
        )
        for degree in (1, 2):
            direct = sum(abs(np.trace(matrix)) ** (2 * degree) for matrix in matrices) / len(matrices)
            formula = symplectic_orbit_count(prime, degree)
            error = abs(direct - formula)
            errors.append(float(error))
            moment_rows.append(
                {
                    "family": "finite Weil",
                    "group": f"SL(2,{prime}) on C^{prime}",
                    "k": degree,
                    "direct": float(direct),
                    "formula": formula,
                    "comparison": "# SL(2,q)-orbits on W^k",
                    "phase-twisted": "not needed",
                    "error": float(error),
                    "pass": error < TOLERANCE,
                }
            )

    # A non-frame-potential balanced coefficient checks the full charge-zero rule.
    alpha, beta = (2, 1), (1, 1, 1)
    original = balanced_coefficient(binary, alpha, beta)
    changed = balanced_coefficient(twisted, alpha, beta)
    balanced_error = abs(original - changed)
    errors.append(float(balanced_error))
    moment_rows.append(
        {
            "family": "projective charge zero",
            "group": "2.S4 / S4",
            "k": "(2,1)|(1,1,1)",
            "direct": int(round(float(original.real))),
            "formula": int(round(float(changed.real))),
            "comparison": "arbitrary lift phases",
            "phase-twisted": int(round(float(changed.real))),
            "error": float(balanced_error),
            "pass": balanced_error < TOLERANCE,
        }
    )

    passed = (
        all(row["pass"] for row in group_rows)
        and all(row["pass"] for row in coefficient_rows)
        and all(row["pass"] for row in determinant_rows)
        and all(row["pass"] for row in moment_rows)
    )
    return {
        "passed": passed,
        "max degree": max_degree,
        "groups": len(group_rows),
        "coefficient comparisons": len(coefficient_rows),
        "determinant comparisons": len(determinant_rows),
        "moment comparisons": len(moment_rows),
        "maximum error": max(errors, default=0.0),
        "group rows": tuple(group_rows),
        "coefficient rows": tuple(coefficient_rows),
        "determinant rows": tuple(determinant_rows),
        "moment rows": tuple(moment_rows),
    }


if __name__ == "__main__":
    outcome = run_suite()
    print(
        f"passed={outcome['passed']} groups={outcome['groups']} "
        f"coefficients={outcome['coefficient comparisons']} "
        f"determinants={outcome['determinant comparisons']} "
        f"moments={outcome['moment comparisons']} "
        f"max_error={outcome['maximum error']:.3e}"
    )
