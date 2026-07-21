"""Command-line entry point for all three generalized Molien suites."""

from .code_monomial.verification import run_suite as run_code
from .delta_su3.verification import run_suite as run_delta
from .psl2_projective_line.verification import run_suite as run_psl2


def run_all():
    return {
        "code-monomial": run_code(),
        "Delta SU(3)": run_delta(),
        "PSL2 projective line": run_psl2(),
    }


if __name__ == "__main__":
    suites = run_all()
    for name, suite in suites.items():
        print(f"{name}: {'PASS' if suite['passed'] else 'FAIL'} ({len(suite['rows'])} coefficient checks)")
    if not all(suite["passed"] for suite in suites.values()):
        raise SystemExit(1)

