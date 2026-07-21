"""Transparent prime-field matrix and subspace construction."""

from __future__ import annotations

from functools import lru_cache
from itertools import product


def rref(rows, q: int):
    matrix = [list(row) for row in rows if any(entry % q for entry in row)]
    if not matrix:
        return ()
    width = len(matrix[0])
    pivot_row = 0
    for column in range(width):
        pivot = next(
            (row for row in range(pivot_row, len(matrix)) if matrix[row][column] % q),
            None,
        )
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        scale = pow(matrix[pivot_row][column] % q, -1, q)
        matrix[pivot_row] = [(entry * scale) % q for entry in matrix[pivot_row]]
        for row in range(len(matrix)):
            if row == pivot_row:
                continue
            factor = matrix[row][column] % q
            if factor:
                matrix[row] = [
                    (matrix[row][j] - factor * matrix[pivot_row][j]) % q
                    for j in range(width)
                ]
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    matrix = [row for row in matrix if any(row)]
    matrix.sort(key=lambda row: next(index for index, value in enumerate(row) if value))
    return tuple(tuple(row) for row in matrix)


def rank(rows, q: int) -> int:
    return len(rref(rows, q))


def determinant(matrix: tuple[int, ...], n: int, q: int) -> int:
    rows = [list(matrix[n * i : n * (i + 1)]) for i in range(n)]
    answer = 1
    for column in range(n):
        pivot = next((row for row in range(column, n) if rows[row][column] % q), None)
        if pivot is None:
            return 0
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            answer = -answer
        value = rows[column][column] % q
        answer = answer * value % q
        inverse = pow(value, -1, q)
        rows[column] = [(entry * inverse) % q for entry in rows[column]]
        for row in range(column + 1, n):
            factor = rows[row][column] % q
            if factor:
                rows[row] = [
                    (rows[row][j] - factor * rows[column][j]) % q for j in range(n)
                ]
    return answer % q


@lru_cache(maxsize=None)
def general_linear_group(n: int, q: int):
    return tuple(
        entries
        for entries in product(range(q), repeat=n * n)
        if determinant(entries, n, q)
    )


def mat_vec(matrix: tuple[int, ...], vector: tuple[int, ...], q: int):
    n = len(vector)
    return tuple(
        sum(matrix[row * n + column] * vector[column] for column in range(n)) % q
        for row in range(n)
    )


def mat_mul(left: tuple[int, ...], right: tuple[int, ...], n: int, q: int):
    return tuple(
        sum(left[row * n + middle] * right[middle * n + column] for middle in range(n)) % q
        for row in range(n)
        for column in range(n)
    )


def matrix_power(matrix: tuple[int, ...], n: int, q: int, exponent: int):
    answer = tuple(int(row == column) for row in range(n) for column in range(n))
    base = matrix
    while exponent:
        if exponent & 1:
            answer = mat_mul(base, answer, n, q)
        base = mat_mul(base, base, n, q)
        exponent //= 2
    return answer


@lru_cache(maxsize=None)
def subspaces(n: int, k: int, q: int):
    spaces = set()
    for entries in product(range(q), repeat=k * n):
        rows = tuple(entries[n * row : n * (row + 1)] for row in range(k))
        canonical = rref(rows, q)
        if len(canonical) == k:
            spaces.add(canonical)
    return tuple(sorted(spaces))


def move_subspace(matrix, space, q: int):
    return rref(tuple(mat_vec(matrix, vector, q) for vector in space), q)


def induced_subspace_permutations(matrices, spaces, q: int):
    position = {space: index for index, space in enumerate(spaces)}
    return tuple(
        dict.fromkeys(
            tuple(position[move_subspace(matrix, space, q)] for space in spaces)
            for matrix in matrices
        )
    )


def invariant_subspace_count(matrix, spaces, q: int) -> int:
    return sum(move_subspace(matrix, space, q) == space for space in spaces)

