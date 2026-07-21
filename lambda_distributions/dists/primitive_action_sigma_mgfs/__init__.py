"""Exact checks for primitive-action and O'Nan--Scott sigma-MGF formulas."""


def run_suite():
    """Run the exact audit without importing the heavy module at package load."""

    from .verification import run_suite as _run_suite

    return _run_suite()


__all__ = ["run_suite"]
