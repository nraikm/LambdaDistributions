from __future__ import annotations

from for_this_guy.schur_functor_classical_groups.common import ExactCase, run_group_suite


CASES = (
    ExactCase(2, (1,)),
    ExactCase(2, (2,)),
    ExactCase(3, (1, 1)),
    ExactCase(3, (3,)),
    ExactCase(3, (2, 1)),
)


def run_suite():
    return run_group_suite("unitary", CASES, moment_dimension=8)


if __name__ == "__main__":
    suite = run_suite()
    print(f"{'PASS' if suite['passed'] else 'FAIL'} U(n): {suite['exact checks']} exact, {suite['moment checks']} Haar checks")
    print(f"maximum exact error: {suite['maximum exact error']:.3e}")
    raise SystemExit(0 if suite["passed"] else 1)

