"""Tests for the resident-callability audit and invoke surface.

A guard nobody has watched fail is not a guard: invoke() must RAISE on an
unknown capability and on a module that failed to import, and a module
lacking a receipt or a frontier must be scored NOT_OPERATIONAL rather than
credited.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import resident_api as ra
from tools.future._common import HARDWARE_FIELDS, RECEIPTS


def _synthetic_inspection(**patch):
    base = {
        "filename": "synthetic_mod.py",
        "stem": "synthetic_mod",
        "relpath": "tools/future/synthetic_mod.py",
        "kind": "production",
        "parse_ok": True,
        "import_ok": True,
        "import_error": None,
        "public_callables": ["build"],
        "preferred_callable": "build",
        "has_main": True,
        "has_if_main": True,
        "cli_flags": [{"flags": ["--build"], "dest": None}],
        "receipt": "SYNTHETIC.json",
        "write_receipt_names": ["SYNTHETIC.json"],
        "workunit_constructs": ["WorkUnit"],
        "raise_count": 3,
        "uses_write_receipt": True,
    }
    base.update(patch)
    return base


def test_audit_entry_point_seals_receipt():
    out = ra.selftest()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "RESIDENT_API_AUDIT.json"
    assert doc["schema"] == "hawking.future.resident_api.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["gpu_authority"] is False
    assert doc["measurement_class"] == "STATIC_ONLY"
    assert doc["bench_state"] == "UNKNOWN"
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert "resident_callable" in doc
    for field in HARDWARE_FIELDS:
        assert field not in doc


def test_discovers_modules_by_introspection_not_a_roster():
    live = sorted(p.name for p in Path("tools/future").glob("*.py"))
    snap = ra.inspect_all()
    discovered = [row["filename"] for row in snap["inspections"]]
    assert discovered == live
    scored_names = [row["filename"] for row in snap["audits"]]
    expected_scored = [name for name in live if not name.startswith("test_")]
    assert scored_names == expected_scored
    assert snap["counts"]["discovered_python"] == len(live)
    assert snap["counts"]["scored"] == len(expected_scored)
    assert snap["counts"]["tests"] == sum(1 for name in live if name.startswith("test_"))
    assert "resident_api.py" in scored_names
    assert "workunit_species.py" in scored_names


def test_five_question_fields_and_honest_partition():
    doc = json.loads(ra.audit().read_text())
    assert doc["five_questions"] == list(ra.FIVE_QUESTIONS)
    audits = doc["modules"]
    assert audits
    operational = 0
    not_operational = 0
    for row in audits:
        for name in ra.FIVE_QUESTIONS:
            assert name in row["questions"]
            assert "pass" in row["questions"][name]
            assert "evidence" in row["questions"][name]
        if row["gaps"]:
            assert row["status"] == "NOT_OPERATIONAL"
            not_operational += 1
        else:
            assert row["status"] == "OPERATIONAL"
            assert all(row["questions"][name]["pass"] for name in ra.FIVE_QUESTIONS)
            operational += 1
    counts = doc["counts"]
    assert counts["operational"] == operational
    assert counts["not_operational"] == not_operational
    assert operational + not_operational == counts["scored"]
    assert counts["scored"] == len(audits)
    # Do not round up: the substrate is present, not fully resident-operable.
    assert not_operational == counts["not_operational"]
    assert "human_cli_only" in counts


def test_invoke_audit_returns_receipt_path():
    path = ra.invoke("future.resident_api")
    assert path == RECEIPTS / "RESIDENT_API_AUDIT.json"
    doc = json.loads(path.read_text())
    assert doc["schema"] == ra.SCHEMA
    assert "future.resident_api" in doc["registry"]
    cap = doc["registry"]["future.resident_api"]
    assert cap["module"] == "tools/future/resident_api.py"
    assert cap["callable"] in {"audit", "build"}
    assert cap["receipt"] == "RESIDENT_API_AUDIT.json"


def test_invoke_unknown_capability_raises():
    with pytest.raises(ra.UnknownCapabilityError, match="unknown capability"):
        ra.invoke("future.capability-that-must-not-exist")
    with pytest.raises(ra.UnknownCapabilityError, match="empty name"):
        ra.invoke("")


def test_invoke_failed_import_raises():
    registry = {
        "future.synthetic_failed": {
            "name": "future.synthetic_failed",
            "module": "tools/future/synthetic_failed.py",
            "import_name": "tools.future.synthetic_failed",
            "callable": "build",
            "argument_contract": {"bindable": False, "params": [], "cli_flags": []},
            "receipt": None,
            "frontier": [],
            "import_error": "ImportError: injected failure for negative control",
            "status": "NOT_OPERATIONAL",
            "gaps": ["hcli_invoke"],
        }
    }
    with pytest.raises(ra.CapabilityImportError, match="failed to import"):
        ra.invoke("future.synthetic_failed", registry=registry)


def test_module_lacking_a_receipt_is_not_operational():
    inspection = _synthetic_inspection(
        receipt=None,
        write_receipt_names=[],
        uses_write_receipt=False,
    )
    verdict = ra.evaluate_five_questions(
        inspection,
        hcli={"tool_names": [], "future_tools": []},
        frontier={"present": True, "by_integration_module": {}, "by_probe_receipt": {}},
    )
    assert verdict["status"] == "NOT_OPERATIONAL"
    assert "does_not_produce_receipt" in verdict["gaps"]
    assert verdict["questions"]["produces_receipt"]["pass"] is False


def test_module_lacking_a_frontier_is_not_operational():
    inspection = _synthetic_inspection(
        receipt="HAS_RECEIPT.json",
        write_receipt_names=["HAS_RECEIPT.json"],
        uses_write_receipt=True,
        workunit_constructs=["WorkUnit"],
        preferred_callable="build",
        raise_count=2,
    )
    verdict = ra.evaluate_five_questions(
        inspection,
        hcli={"tool_names": [], "future_tools": []},
        frontier={
            "present": True,
            "by_integration_module": {},
            "by_probe_receipt": {},
            "writes_frontier_modules": ["tools/future/global_frontier.py"],
        },
    )
    assert verdict["status"] == "NOT_OPERATIONAL"
    assert "result_does_not_feed_a_named_frontier" in verdict["gaps"]
    assert verdict["questions"]["feeds_named_frontier"]["pass"] is False
    # Receipt, WorkUnit, callable, fail-closed all hold — frontier is the gap.
    assert verdict["questions"]["produces_receipt"]["pass"] is True
    assert verdict["questions"]["emits_workunit"]["pass"] is True
    assert verdict["questions"]["hcli_invoke"]["pass"] is True
    assert verdict["questions"]["fail_closed"]["pass"] is True


def test_named_as_integration_target_is_not_credited_as_feeding_frontier():
    inspection = _synthetic_inspection(
        relpath="tools/future/synthetic_mod.py",
        receipt="HAS_RECEIPT.json",
        workunit_constructs=["WorkUnit"],
    )
    verdict = ra.evaluate_five_questions(
        inspection,
        hcli={"tool_names": [], "future_tools": []},
        frontier={
            "present": True,
            "by_integration_module": {"tools/future/synthetic_mod.py": ["F999"]},
            "by_probe_receipt": {},
            "writes_frontier_modules": ["tools/future/global_frontier.py"],
        },
    )
    assert verdict["questions"]["feeds_named_frontier"]["pass"] is False
    assert verdict["status"] == "NOT_OPERATIONAL"
    assert "F999" in verdict["frontier_ids"]


def test_live_modules_without_receipt_or_frontier_are_not_credited():
    doc = json.loads(ra.audit().read_text())
    lacking_receipt = [
        row for row in doc["modules"] if not row["questions"]["produces_receipt"]["pass"]
    ]
    for row in lacking_receipt:
        assert row["status"] == "NOT_OPERATIONAL"
        assert "does_not_produce_receipt" in row["gaps"]
    lacking_frontier = [
        row
        for row in doc["modules"]
        if not row["questions"]["feeds_named_frontier"]["pass"]
    ]
    assert lacking_frontier, "expected live modules that do not feed a named frontier"
    for row in lacking_frontier:
        assert row["status"] == "NOT_OPERATIONAL"
        assert "result_does_not_feed_a_named_frontier" in row["gaps"]


def test_hcli_registry_has_no_future_tools():
    doc = json.loads(ra.audit().read_text())
    hcli = doc["hcli"]
    assert hcli["future_tools"] == []
    assert hcli["future_source_refs"] == []
    assert hcli["hcli_unknown_tool_is_fail_open"] is True
    resident = doc["resident_callable"]
    assert resident["entry_point"]
    assert resident["receipt"] == "receipts/future/RESIDENT_API_AUDIT.json"
    assert "UnknownCapabilityError" in resident["fail_closed"]
    assert "CapabilityImportError" in resident["fail_closed"]


def test_work_units_are_emitted_and_hcli_shaped():
    doc = json.loads(ra.audit().read_text())
    units = doc["work_units"]
    assert units
    ids = [row["id"] for row in units]
    assert len(ids) == len(set(ids))
    for row in units:
        assert row["id"]
        assert row["role"]
        assert row["description"]
        assert row["verifier"]
        assert row["claim_boundary"]
        assert row.get("gpu_authority") is not True


def test_invoke_rejects_unexpected_kwargs():
    with pytest.raises(ra.InvocationError, match="rejected arguments"):
        ra.invoke("future.resident_api", this_kwarg_does_not_exist=True)


def test_cli_audit_writes_receipt(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["resident_api.py", "--audit"])
    assert ra.main() == 0
    captured = capsys.readouterr()
    assert "RESIDENT_API_AUDIT.json" in captured.out
    assert "operational=" in captured.out
    assert (RECEIPTS / "RESIDENT_API_AUDIT.json").is_file()
