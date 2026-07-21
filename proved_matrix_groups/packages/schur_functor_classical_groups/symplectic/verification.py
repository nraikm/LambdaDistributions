from __future__ import annotations

from for_this_guy.schur_functor_classical_groups.common import ExactCase, run_group_suite


CASES = (
    ExactCase(2, (1,)),
    ExactCase(4, (2,)),
    ExactCase(4, (1, 1)),
    ExactCase(4, (3,)),
    ExactCase(4, (2, 1)),
)


def run_suite():
    return run_group_suite("symplectic", CASES, moment_dimension=8)


if __name__ == "__main__":
    suite = run_suite()
    print(f"{'PASS' if suite['passed'] else 'FAIL'} USp(2n): {suite['exact checks']} exact, {suite['moment checks']} Haar checks")
    print(f"maximum exact error: {suite['maximum exact error']:.3e}")
    raise SystemExit(0 if suite["passed"] else 1)

