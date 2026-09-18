"""Focused contract tests for the canonical DARKMATTER planning seam."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "tools" / "accelerator", ROOT / "tools" / "odyssey"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import physical_graph_compiler as compiler  # noqa: E402


def test_darkmatter_physical_plan_reuses_validated_plan_owner() -> None:
    result = compiler.build_darkmatter_physical_plan()

    assert result["schema"] == "hawking.odyssey.darkmatter_physical_plan.v1"
    assert result["pass_name"] == "DARKMATTER_PHYSICAL_PLAN_PASS"
    assert result["owner"] == "tools/odyssey/physical_graph_compiler.py"
    assert result["status"] == "PLAN_VALIDATED"
    assert result["qualification"] == "PLAN_ONLY"
    assert result["evidence_tier"] == "COST_MODEL"
    assert result["physical_qualification"] == "WITHHELD"
    assert result["execution"]["status"] == "WITHHELD"
    assert result["pass"] is True


def test_darkmatter_plan_cannot_be_misread_as_hardware_measurement() -> None:
    result = compiler.build_darkmatter_physical_plan()

    assert result["evidence_tier"] != "HARDWARE_MEASURED"
    assert "hardware qualification" in result["claim_boundary"]
    assert "runtime receipt" in result["execution"]["reason"]
