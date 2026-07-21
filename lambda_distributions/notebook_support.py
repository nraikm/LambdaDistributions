"""Small presentation helpers shared by the Marimo applications."""

from __future__ import annotations

from collections.abc import Mapping

from .core import Partition, normalize_partition


def parse_partition(text: str) -> Partition:
    """Parse comma- or whitespace-separated partition notation."""

    pieces = text.replace(",", " ").split()
    return normalize_partition(int(piece) for piece in pieces)


def display_number(value: complex, tolerance: float = 1e-10) -> str:
    """Format numerical noise cleanly for notebook tables."""

    number = complex(value)
    real = 0.0 if abs(number.real) < tolerance else number.real
    imaginary = 0.0 if abs(number.imag) < tolerance else number.imag
    if imaginary == 0:
        return f"{real:.10g}"
    if real == 0:
        return f"{imaginary:.10g}j"
    return f"{real:.10g}{imaginary:+.10g}j"


def coefficient_records(coefficients: Mapping[Partition, complex], tolerance: float = 1e-10):
    """Create table rows for nonzero power-sum coefficients."""

    return [
        {"partition": str(partition), "coefficient": display_number(coefficient, tolerance)}
        for partition, coefficient in coefficients.items()
        if abs(coefficient) > tolerance
    ]


def verification_records(report):
    """Create display-safe rows from a :class:`VerificationReport`."""

    return [
        {
            "partition": str(check.partition),
            "explicit": display_number(check.explicit),
            "predicted": display_number(check.predicted),
            "error": f"{check.error:.3g}",
            "passed": check.passed,
        }
        for check in report.checks
    ]
