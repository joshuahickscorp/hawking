"""Invariants for CLEAN SLATE Stage A: controller integration, giant adapters, derived readiness."""
from __future__ import annotations

import json
import os
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import succ_gravity as sgv          # noqa: E402
import succ_gravity_policy as gp    # noqa: E402
from eco_common import sealed       # noqa: E402


def test_forge_program_materializes_sealed_and_launch_refused():
    """Section 8 + Section 1 law: the one controller materializes a sealed Forge program and REFUSES
    to launch it (default-off) - a higher/any-rate program cannot launch before Gravity authorizes."""
    prog = sgv.materialize_forge_program("120B", rate=Fraction(4, 5), family="transform_pq",
                                         source_manifest_sha256="deadbeef")
    assert sealed(prog, "program_sha256")
    assert prog["kind"] == "forge_subbit"
    assert prog["representation_family"].startswith("gravity_forge:")
    assert prog["is_subbit"] is True
    assert prog["launch_gate"]["gravity_enabled"] is False
    launchable, reasons = sgv.program_launchable(
        prog, policy=None, heavy_lock=sgv.HeavyLock(held_by=None), admission_passed=False)
    assert not launchable and any("default-off" in r for r in reasons)


def test_forge_program_carries_required_section8_fields():
    prog = sgv.materialize_forge_program("120B", rate=Fraction(4, 5), family="transform_pq",
                                         source_manifest_sha256="abc")
    for f in ("f_sequence", "doctor_budget_bpw", "resource_request", "checkpoint_rules",
              "telegram_events", "escape_receipt_rules", "stop_conditions", "source_manifest_sha256"):
        assert f in prog
    assert prog["escape_receipt_rules"]["authorizes_escape"] is False


def test_pre_run_readiness_receipt_is_derived_not_static():
    p = "reports/condense/gravity_forge/FORGE_PRE_RUN_READINESS.json"
    if not os.path.exists(p):
        return  # gate not yet derived in this checkout
    d = json.load(open(p))
    assert d["derived"] is True and d["operator_declared"] is False
    # every condition must carry an evidence dict (a live probe result), never a bare bool
    assert all(isinstance(c, dict) and "value" in c for c in d["conditions"].values())


def test_giant_adapter_contracts_stable_and_composed_from_authority():
    p = "reports/condense/gravity_forge/giant_adapters/STABLE.json"
    if not os.path.exists(p):
        return
    s = json.load(open(p))
    assert s["all_contracts_valid"] is True
    assert set(s["adapters"]) >= {"deepseek-v3.2-685b", "kimi-k2.6-1t", "deepseek-v4-pro-1.6t"}
