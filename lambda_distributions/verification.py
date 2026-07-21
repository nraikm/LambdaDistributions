"""Reusable checks for proposed Lambda-distribution formulas."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .core import Partition, normalize_partition, partitions_up_to
from .distribution import average_homogeneous, power_sum_moment, scalar_sigma_monomial_coefficient
from .groups import FiniteMatrixGroup


@dataclass(frozen=True)
class Check:
    partition: Partition
    explicit: complex
    predicted: complex
    error: float
    passed: bool


@dataclass(frozen=True)
class VerificationReport:
    group_name: str
    basis: str
    tolerance: float
    checks: tuple[Check, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def max_error(self) -> float:
        return max((check.error for check in self.checks), default=0.0)

    def records(self) -> list[dict[str, object]]:
        """Return rows suitable for ``mo.ui.table`` or a dataframe."""

        return [
            {
                "partition": str(check.partition),
                "explicit": check.explicit,
                "predicted": check.predicted,
                "error": check.error,
                "passed": check.passed,
            }
            for check in self.checks
        ]


def check_formula(
    group: FiniteMatrixGroup,
    formula: Callable[[Partition], complex],
    *,
    partitions: Iterable[Iterable[int]] | None = None,
    max_degree: int | None = None,
    basis: str = "power_sum",
    tolerance: float = 1e-9,
) -> VerificationReport:
    """Compare enumeration with ``formula`` on a collection of partitions."""

    if partitions is None:
        if max_degree is None:
            raise ValueError("provide partitions or max_degree")
        selected = tuple(partitions_up_to(max_degree))
    else:
        selected = tuple(normalize_partition(partition) for partition in partitions)

    if basis == "power_sum":
        evaluator = power_sum_moment
    elif basis == "homogeneous":
        evaluator = average_homogeneous
    elif basis == "scalar_sigma_monomial":
        evaluator = scalar_sigma_monomial_coefficient
    else:
        raise ValueError(
            "basis must be 'power_sum', 'homogeneous', or 'scalar_sigma_monomial'"
        )

    checks = []
    for partition in selected:
        explicit = complex(evaluator(group, partition))
        predicted = complex(formula(partition))
        error = abs(explicit - predicted)
        checks.append(Check(partition, explicit, predicted, error, error <= tolerance))
    return VerificationReport(group.name, basis, tolerance, tuple(checks))
