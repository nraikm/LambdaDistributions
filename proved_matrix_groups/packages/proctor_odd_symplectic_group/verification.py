"""Exact checks for Proctor's odd symplectic group and valid substitutes.

The complex odd symplectic group is noncompact, so this module deliberately
does not manufacture a normalized Haar average for it.  Instead it checks:

* the block-matrix description and determinant factorization;
* the maximal-compact formulas in rank one by exact constant terms;
* finite odd symplectic groups over prime fields, acting on projective points;
* direct permutation-matrix character averages versus cycle and orbit formulas.

Only the Python standard library is required.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement, product
from math import comb, prod
from typing import Iterable, Iterator, Sequence


Matrix = tuple[tuple[int, ...], ...]
Permutation = tuple[int, ...]
Partition = tuple[int, ...]
LaurentPolynomial = dict[tuple[int, int], int]


def matmul(left: Matrix, right: Matrix, q: int) -> Matrix:
    rows, middle, columns = len(left), len(right), len(right[0])
    return tuple(
        tuple(
            sum(left[i][k] * right[k][j] for k in range(middle)) % q
            for j in range(columns)
        )
        for i in range(rows)
    )


def transpose(matrix: Matrix) -> Matrix:
    return tuple(zip(*matrix))


def identity(size: int) -> Matrix:
    return tuple(tuple(int(i == j) for j in range(size)) for i in range(size))


def symplectic_form(n: int, q: int) -> Matrix:
    """Return J = [[0,I],[-I,0]] on a 2n-dimensional space."""
    size = 2 * n
    return tuple(
        tuple(
            (1 if i < n and j == i + n else -1 if i >= n and j == i - n else 0)
            % q
            for j in range(size)
        )
        for i in range(size)
    )


def alternating_form(n: int, q: int) -> Matrix:
    j = symplectic_form(n, q)
    size = 2 * n + 1
    return tuple(
        tuple(j[i][k] if i < size - 1 and k < size - 1 else 0 for k in range(size))
        for i in range(size)
    )


def all_matrices(size: int, q: int) -> Iterator[Matrix]:
    for entries in product(range(q), repeat=size * size):
        yield tuple(
            tuple(entries[size * row + column] for column in range(size))
            for row in range(size)
        )


def symplectic_matrices(n: int, q: int) -> tuple[Matrix, ...]:
    j = symplectic_form(n, q)
    return tuple(
        matrix
        for matrix in all_matrices(2 * n, q)
        if matmul(matmul(transpose(matrix), j, q), matrix, q) == j
    )


def odd_matrix(a_matrix: Matrix, ell: Sequence[int], scalar: int, q: int) -> Matrix:
    size = len(a_matrix)
    return tuple(
        tuple(a_matrix[i][j] for j in range(size)) + (0,)
        for i in range(size)
    ) + (tuple(value % q for value in ell) + (scalar % q,),)


def odd_symplectic_group(n: int, q: int) -> tuple[Matrix, ...]:
    """Enumerate W* semidirect (Sp_2n(q) x F_q^*) for prime q."""
    symplectic = symplectic_matrices(n, q)
    covectors = tuple(product(range(q), repeat=2 * n))
    return tuple(
        odd_matrix(a_matrix, ell, scalar, q)
        for a_matrix in symplectic
        for ell in covectors
        for scalar in range(1, q)
    )


def symplectic_order(n: int, q: int) -> int:
    return q ** (n * n) * prod(q ** (2 * j) - 1 for j in range(1, n + 1))


def odd_symplectic_order(n: int, q: int) -> int:
    return q ** (2 * n) * (q - 1) * symplectic_order(n, q)


def canonical_projective(vector: Sequence[int], q: int) -> tuple[int, ...]:
    reduced = tuple(value % q for value in vector)
    pivot = next(index for index, value in enumerate(reduced) if value)
    inverse = pow(reduced[pivot], -1, q)
    return tuple(value * inverse % q for value in reduced)


def projective_points(dimension: int, q: int) -> tuple[tuple[int, ...], ...]:
    points = {
        canonical_projective(vector, q)
        for vector in product(range(q), repeat=dimension)
        if any(vector)
    }
    return tuple(sorted(points))


def matvec(matrix: Matrix, vector: Sequence[int], q: int) -> tuple[int, ...]:
    return tuple(
        sum(matrix[i][j] * vector[j] for j in range(len(vector))) % q
        for i in range(len(matrix))
    )


def projective_permutation(
    matrix: Matrix, points: Sequence[tuple[int, ...]], q: int
) -> Permutation:
    lookup = {point: index for index, point in enumerate(points)}
    return tuple(
        lookup[canonical_projective(matvec(matrix, point, q), q)] for point in points
    )


def compose(left: Permutation, right: Permutation) -> Permutation:
    return tuple(left[right[index]] for index in range(len(right)))


def cycle_lengths(permutation: Permutation) -> tuple[int, ...]:
    unseen = set(range(len(permutation)))
    lengths: list[int] = []
    while unseen:
        start = next(iter(unseen))
        current = start
        length = 0
        while current in unseen:
            unseen.remove(current)
            current = permutation[current]
            length += 1
        lengths.append(length)
    return tuple(sorted(lengths))


def direct_complete_characters(permutation: Permutation, maximum_degree: int) -> tuple[int, ...]:
    """Compute h_d from traces of explicit powers of the permutation matrix.

    Newton's recurrence d h_d = sum_{k=1}^d tr(P^k) h_{d-k} is the direct
    matrix-representation route; it does not use a cycle generating product.
    """
    power = tuple(range(len(permutation)))
    traces = [0]
    for _ in range(maximum_degree):
        power = compose(permutation, power)
        traces.append(sum(power[index] == index for index in range(len(power))))
    values = [1]
    for degree in range(1, maximum_degree + 1):
        numerator = sum(traces[k] * values[degree - k] for k in range(1, degree + 1))
        assert numerator % degree == 0
        values.append(numerator // degree)
    return tuple(values)


def cycle_complete_characters(permutation: Permutation, maximum_degree: int) -> tuple[int, ...]:
    """Compute coefficients of prod_cycles (1-t^length)^(-1)."""
    coefficients = [0] * (maximum_degree + 1)
    coefficients[0] = 1
    for length in cycle_lengths(permutation):
        updated = [0] * (maximum_degree + 1)
        for degree, value in enumerate(coefficients):
            for extra in range(0, maximum_degree - degree + 1, length):
                updated[degree + extra] += value
        coefficients = updated
    return tuple(coefficients)


def partitions_through(maximum_degree: int) -> tuple[Partition, ...]:
    def partitions(total: int, ceiling: int | None = None) -> Iterator[Partition]:
        if total == 0:
            yield ()
            return
        ceiling = min(total, ceiling if ceiling is not None else total)
        for first in range(ceiling, 0, -1):
            for rest in partitions(total - first, first):
                yield (first,) + rest

    return tuple(partition for total in range(maximum_degree + 1) for partition in partitions(total))


def coefficient_averages(
    permutations: Sequence[Permutation], maximum_degree: int
) -> tuple[dict[Partition, Fraction], dict[Partition, Fraction]]:
    partitions = partitions_through(maximum_degree)
    direct_totals = {partition: 0 for partition in partitions}
    cycle_totals = {partition: 0 for partition in partitions}
    for permutation in permutations:
        direct_h = direct_complete_characters(permutation, maximum_degree)
        cycle_h = cycle_complete_characters(permutation, maximum_degree)
        for partition in partitions:
            direct_totals[partition] += prod(direct_h[degree] for degree in partition)
            cycle_totals[partition] += prod(cycle_h[degree] for degree in partition)
    order = len(permutations)
    return (
        {partition: Fraction(total, order) for partition, total in direct_totals.items()},
        {partition: Fraction(total, order) for partition, total in cycle_totals.items()},
    )


def multiset_states(size: int, degree: int) -> tuple[tuple[int, ...], ...]:
    return tuple(combinations_with_replacement(range(size), degree))


def orbit_count(permutations: Sequence[Permutation], partition: Partition) -> int:
    """Explicitly enumerate G-orbits on a product of multiset spaces."""
    size = len(permutations[0])
    states = set(product(*(multiset_states(size, degree) for degree in partition)))
    count = 0
    while states:
        representative = next(iter(states))
        orbit = {
            tuple(tuple(sorted(permutation[index] for index in block)) for block in representative)
            for permutation in permutations
        }
        states.difference_update(orbit)
        count += 1
    return count


def determinant_mod(matrix: Matrix, q: int) -> int:
    work = [list(row) for row in matrix]
    determinant = 1
    for column in range(len(work)):
        pivot = next((row for row in range(column, len(work)) if work[row][column] % q), None)
        if pivot is None:
            return 0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            determinant = -determinant
        pivot_value = work[column][column] % q
        determinant = determinant * pivot_value % q
        inverse = pow(pivot_value, -1, q)
        for row in range(column + 1, len(work)):
            factor = work[row][column] * inverse % q
            for entry in range(column, len(work)):
                work[row][entry] = (work[row][entry] - factor * work[column][entry]) % q
    return determinant % q


def characteristic_determinant(matrix: Matrix, scalar: int, q: int) -> int:
    shifted = tuple(
        tuple((int(i == j) - scalar * matrix[i][j]) % q for j in range(len(matrix)))
        for i in range(len(matrix))
    )
    return determinant_mod(shifted, q)


def _poly_add(left: LaurentPolynomial, right: LaurentPolynomial) -> LaurentPolynomial:
    result = Counter(left)
    result.update(right)
    return {weight: value for weight, value in result.items() if value}


def _poly_mul(left: LaurentPolynomial, right: LaurentPolynomial) -> LaurentPolynomial:
    result: Counter[tuple[int, int]] = Counter()
    for (z_left, u_left), left_value in left.items():
        for (z_right, u_right), right_value in right.items():
            result[(z_left + z_right, u_left + u_right)] += left_value * right_value
    return dict(result)


def complete_weight_character(weights: Sequence[tuple[int, int]], degree: int) -> LaurentPolynomial:
    coefficients: list[LaurentPolynomial] = [{(0, 0): 1}] + [{} for _ in range(degree)]
    for weight in weights:
        updated: list[LaurentPolynomial] = [{} for _ in range(degree + 1)]
        for old_degree, polynomial in enumerate(coefficients):
            monomial: LaurentPolynomial = {(0, 0): 1}
            for extra in range(degree - old_degree + 1):
                updated[old_degree + extra] = _poly_add(updated[old_degree + extra], _poly_mul(polynomial, monomial))
                monomial = _poly_mul(monomial, {weight: 1})
        coefficients = updated
    return coefficients[degree]


def product_character(weights: Sequence[tuple[int, int]], partition: Partition) -> LaurentPolynomial:
    result: LaurentPolynomial = {(0, 0): 1}
    for degree in partition:
        result = _poly_mul(result, complete_weight_character(weights, degree))
    return result


def su2_u1_haar_constant_term(polynomial: LaurentPolynomial) -> int:
    """Weyl constant term CT_u CT_z ((1-z^2) f(z,u))."""
    return polynomial.get((0, 0), 0) - polynomial.get((-2, 0), 0)


def compact_substitute_checks(maximum_degree: int = 6) -> tuple[dict[str, object], ...]:
    w_weights = ((1, 0), (-1, 0))
    k_weights = w_weights + ((0, 1),)
    fixed_radical_weights = w_weights + ((0, 0),)
    rows = []
    for partition in partitions_through(maximum_degree):
        w_value = su2_u1_haar_constant_term(product_character(w_weights, partition))
        k_value = su2_u1_haar_constant_term(product_character(k_weights, partition))
        fixed_value = su2_u1_haar_constant_term(product_character(fixed_radical_weights, partition))

        convolution: LaurentPolynomial = {(0, 0): 1}
        for degree in partition:
            factor: LaurentPolynomial = {}
            for smaller_degree in range(degree + 1):
                factor = _poly_add(factor, complete_weight_character(w_weights, smaller_degree))
            convolution = _poly_mul(convolution, factor)
        predicted_fixed = su2_u1_haar_constant_term(convolution)
        rows.append(
            {
                "partition": partition,
                "K direct": k_value,
                "Sp(1), W": w_value,
                "fixed-radical direct": fixed_value,
                "fixed-radical formula": predicted_fixed,
                "passed": k_value == w_value and fixed_value == predicted_fixed,
            }
        )
    return tuple(rows)


@dataclass(frozen=True)
class FiniteCase:
    n: int
    q: int
    maximum_degree: int
    orbit_partitions: tuple[Partition, ...] = ()


DEFAULT_CASES = (
    FiniteCase(1, 2, 4, ((1,), (2,), (1, 1))),
    FiniteCase(1, 3, 4, ((1,), (2,), (1, 1))),
    FiniteCase(1, 5, 3),
    FiniteCase(2, 2, 3),
)


def verify_finite_case(case: FiniteCase) -> dict[str, object]:
    group = odd_symplectic_group(case.n, case.q)
    expected_order = odd_symplectic_order(case.n, case.q)
    omega = alternating_form(case.n, case.q)
    form_preserved = all(matmul(matmul(transpose(g), omega, case.q), g, case.q) == omega for g in group)

    points = projective_points(2 * case.n + 1, case.q)
    permutations = tuple(projective_permutation(g, points, case.q) for g in group)
    direct, cycle = coefficient_averages(permutations, case.maximum_degree)

    determinant_checks = 0
    determinant_passed = True
    sample_step = max(1, len(group) // 97)
    for g in group[::sample_step]:
        a_matrix = tuple(row[:-1] for row in g[:-1])
        scalar_a = g[-1][-1]
        for t in range(case.q):
            left = characteristic_determinant(g, t, case.q)
            right = (1 - scalar_a * t) * characteristic_determinant(a_matrix, t, case.q) % case.q
            determinant_checks += 1
            determinant_passed &= left == right

    closure_checks = 0
    closure_passed = True
    group_set = set(group)
    stride = max(1, len(group) // 31)
    sampled = group[::stride]
    for left in sampled:
        for right in sampled:
            closure_checks += 1
            closure_passed &= matmul(left, right, case.q) in group_set

    orbit_rows = []
    for partition in case.orbit_partitions:
        count = orbit_count(permutations, partition)
        orbit_rows.append(
            {
                "partition": partition,
                "explicit orbits": count,
                "matrix average": direct[partition],
                "passed": Fraction(count) == direct[partition],
            }
        )

    coefficient_rows = tuple(
        {
            "partition": partition,
            "matrix average": direct[partition],
            "cycle formula": cycle[partition],
            "passed": direct[partition] == cycle[partition],
        }
        for partition in direct
    )
    radical_index = points.index((0,) * (2 * case.n) + (1,))
    radical_fixed = all(permutation[radical_index] == radical_index for permutation in permutations)

    return {
        "n": case.n,
        "q": case.q,
        "dimension": 2 * case.n + 1,
        "order": len(group),
        "expected order": expected_order,
        "symplectic Levi order": len(symplectic_matrices(case.n, case.q)),
        "expected symplectic Levi order": symplectic_order(case.n, case.q),
        "projective points": len(points),
        "expected projective points": (case.q ** (2 * case.n + 1) - 1) // (case.q - 1),
        "form preserved": form_preserved,
        "radical point fixed": radical_fixed,
        "m_(1)": direct[(1,)],
        "determinant checks": determinant_checks,
        "determinant passed": determinant_passed,
        "closure checks": closure_checks,
        "closure passed": closure_passed,
        "coefficient rows": coefficient_rows,
        "orbit rows": tuple(orbit_rows),
        "passed": (
            len(group) == expected_order
            and len(symplectic_matrices(case.n, case.q)) == symplectic_order(case.n, case.q)
            and len(points) == (case.q ** (2 * case.n + 1) - 1) // (case.q - 1)
            and form_preserved
            and radical_fixed
            and direct[(1,)] == 2
            and determinant_passed
            and closure_passed
            and all(row["passed"] for row in coefficient_rows)
            and all(row["passed"] for row in orbit_rows)
        ),
    }


def invariant_vanishing_certificate(n_values: Iterable[int] = (1, 2), maximum_degree: int = 4) -> tuple[dict[str, int | bool], ...]:
    """Dimension/rank certificate for the two-stage invariant proof.

    The C* subgroup first removes every tensor with a radical-line factor. On
    W^tensor d, stack the contractions at one position against a basis of W*.
    This stacked map is a coordinate permutation, hence has full rank (2n)^d.
    """
    def rank_mod(matrix: list[list[int]], prime: int = 101) -> int:
        work = [[entry % prime for entry in row] for row in matrix]
        row = 0
        for column in range(len(work[0])):
            pivot = next((candidate for candidate in range(row, len(work)) if work[candidate][column]), None)
            if pivot is None:
                continue
            work[row], work[pivot] = work[pivot], work[row]
            inverse = pow(work[row][column], -1, prime)
            work[row] = [entry * inverse % prime for entry in work[row]]
            for candidate in range(len(work)):
                if candidate != row and work[candidate][column]:
                    factor = work[candidate][column]
                    work[candidate] = [
                        (left - factor * right) % prime
                        for left, right in zip(work[candidate], work[row])
                    ]
            row += 1
            if row == len(work):
                break
        return row

    rows = []
    for n in n_values:
        dimension = 2 * n
        for degree in range(1, maximum_degree + 1):
            basis = tuple(product(range(dimension), repeat=degree))
            row_labels = tuple((dual_index, tail) for dual_index in range(dimension) for tail in product(range(dimension), repeat=degree - 1))
            row_lookup = {label: index for index, label in enumerate(row_labels)}
            matrix = [[0] * len(basis) for _ in row_labels]
            for column, tensor_index in enumerate(basis):
                matrix[row_lookup[(tensor_index[0], tensor_index[1:])]][column] = 1
            rank = rank_mod(matrix)
            rows.append(
                {
                    "n": n,
                    "tensor degree": degree,
                    "domain dimension": len(basis),
                    "contraction rank": rank,
                    "kernel dimension": len(basis) - rank,
                    "passed": rank == len(basis),
                }
            )
    return tuple(rows)


def run_all(cases: Sequence[FiniteCase] = DEFAULT_CASES) -> dict[str, object]:
    compact_rows = compact_substitute_checks()
    invariant_rows = invariant_vanishing_certificate()
    finite_results = tuple(verify_finite_case(case) for case in cases)
    passed = (
        all(row["passed"] for row in compact_rows)
        and all(row["passed"] for row in invariant_rows)
        and all(result["passed"] for result in finite_results)
    )
    return {
        "passed": passed,
        "compact rows": compact_rows,
        "invariant rows": invariant_rows,
        "finite results": finite_results,
    }


def _format_fraction(value: object) -> str:
    if isinstance(value, Fraction):
        return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    return str(value)


def print_report(report: dict[str, object]) -> None:
    print("PASS" if report["passed"] else "FAIL", "Proctor odd symplectic verification")
    print("compact coefficients checked:", len(report["compact rows"]))
    print("invariant vanishing certificates:", len(report["invariant rows"]))
    for result in report["finite results"]:
        print(
            f"n={result['n']}, q={result['q']}, dim={result['dimension']}: "
            f"|G|={result['order']}, |P(E)|={result['projective points']}, "
            f"coefficients={len(result['coefficient rows'])}, "
            f"[m_(1)]={_format_fraction(result['m_(1)'])}, "
            f"status={'PASS' if result['passed'] else 'FAIL'}"
        )


if __name__ == "__main__":
    report = run_all()
    print_report(report)
    raise SystemExit(0 if report["passed"] else 1)
