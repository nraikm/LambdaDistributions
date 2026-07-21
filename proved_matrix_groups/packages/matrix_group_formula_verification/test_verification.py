"""Smoke tests for the three group-family verification suites."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_family(name: str):
    path = ROOT / name / "verification.py"
    spec = importlib.util.spec_from_file_location(f"matrix_formula_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_restricted_monomial_suite():
    module = load_family("restricted_monomial")
    assert all(result["passed"] for result in module.run_suite(maximum_degree=4))


def test_h3_suite():
    module = load_family("h3_reflection")
    result = module.run_suite(maximum_degree=6)
    assert result["passed"]
    assert result["diagnostics"]["order"] == 120
    assert result["diagnostics"]["closed under multiplication"]


def test_representation_variants_suite():
    module = load_family("representation_variants")
    suite = module.run_suite(maximum_degree=4)
    assert all(result["passed"] for result in suite.values())
