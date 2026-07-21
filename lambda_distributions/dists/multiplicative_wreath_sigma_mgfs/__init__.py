"""Exact checks for multiplicative wreath-product sigma-MGF formulas."""


def run_suite():
    """Run the audit without importing the computational module eagerly."""

    from .verification import run_suite as _run_suite

    return _run_suite()


__all__ = ["run_suite"]
