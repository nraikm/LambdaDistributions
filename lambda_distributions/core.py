"""Partitions and their standard symmetric-function constants."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from math import factorial

Partition = tuple[int, ...]


def normalize_partition(parts: Iterable[int]) -> Partition:
    """Return ``parts`` as a weakly decreasing integer partition.

    The empty partition is allowed and represents degree zero.
    """

    partition = tuple(int(part) for part in parts)
    if any(part <= 0 for part in partition):
        raise ValueError("partition parts must be positive integers")
    return tuple(sorted(partition, reverse=True))


def z_partition(partition: Iterable[int]) -> int:
    """Compute ``z_lambda = product(i^m_i m_i!)``."""

    counts = Counter(normalize_partition(partition))
    result = 1
    for part, multiplicity in counts.items():
        result *= part**multiplicity * factorial(multiplicity)
    return result


def integer_partitions(total: int, max_part: int | None = None) -> Iterator[Partition]:
    """Yield integer partitions of ``total`` in reverse lexicographic order."""

    if total < 0:
        raise ValueError("total must be non-negative")
    if total == 0:
        yield ()
        return

    upper = min(total, max_part if max_part is not None else total)
    for first in range(upper, 0, -1):
        for tail in integer_partitions(total - first, first):
            yield (first, *tail)


def partitions_up_to(max_degree: int, *, include_empty: bool = True) -> Iterator[Partition]:
    """Yield all partitions of degree at most ``max_degree``."""

    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")
    start = 0 if include_empty else 1
    for degree in range(start, max_degree + 1):
        yield from integer_partitions(degree)
