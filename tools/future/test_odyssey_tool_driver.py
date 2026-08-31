"""Odyssey tool driver: invoke isolation, refusals, SLEEPING units.

Negative controls the launch gate named: absent parent refuses; invoke never
writes receipts/headless; a run with no receipt raises; an unrunnable tool
emits SLEEPING, not pending. A validator nobody has watched reject is not
a validator.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tools.future import odyssey_tool_driver as otd
from tools.future import orchestration as orch
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims
from tools.future.mutation_surface import intersects_codex


def _fixture_tool(tmp_path: Path, *, body: str, receipt_name: str = "HARDCODED.json") -> dict:
    src = tmp_path / "hardcoded_tool.py"
    src.write_text(body)
    return {
        "id": "hardcoded_tool",
        "rel": "hardcoded_tool.py",
        "family": "doctor",
        "source_path": str(src),
        "declared_inputs": [],
        "extra_inputs": [],
        "output_policy": {
            "kind": "hardcoded_rh",
            "cli_flags": [],
            "env_keys": [],
            "hardcoded_rh": True,
            "requires_gpu": False,
            "has_main": True,
            "headless_receipt": receipt_name,
            "writes_headless_literal": True,
            "writes_ascent_literal": False,
            "isolatable": True,
        },
        "frontier": "FT.MODEL_CAPABILITY.hard-gates",
        "sidecar_receipt": "HARDCODED_DRIVEN.json",
        "requires_modules": [],
        "missing_modules": [],
        "cli_args": [],
        "reads_from_headless": [],
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
    }


def test_build_seals_receipt_and_invokes_doctor_seal():
    out = otd.build()
    assert out.parent == RECEIPTS
    assert out.name == otd.RECEIPT
    doc = json.loads(out.read_text())
    assert doc["schema"] == otd.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        assert key not in doc
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["resident_callable"]["entry_point"]
    assert doc["resident_callable"]["workunit"]
    assert doc["resident_callable"]["receipt"]
    assert doc["resident_callable"]["frontier"]
    assert doc["resident_callable"]["fails_closed"]
    executed = doc["executed_invocation"]
    assert executed is not None, doc.get("executed_error")
    assert executed["invoked"] is True
    assert executed["refused"] is False
    assert Path(REPO_JOIN(executed["receipt"])).is_file()
    seal = json.loads(Path(REPO_JOIN(executed["receipt"])).read_text())
    assert seal["schema"] == "hawking.nos.doctor_seal.v1"
    assert seal["refusals"] == 4


def REPO_JOIN(rel: str) -> Path:
    return otd.REPO / rel


def test_tools_recover_owned_surfaces():
    rows = otd.tools()
    ids = {r["id"] for r in rows}
    assert "doctor_tournament" in ids
    assert "doctor_seal" in ids
    assert "decoding_gravity" in ids
    assert "state_gravity" in ids
    assert "gravity_package" in ids
    by = {r["id"]: r for r in rows}
    parents = by["doctor_tournament"]["declared_inputs"]
    assert any(i["name"] == "PARENT" for i in parents)
    assert by["doctor_seal"]["output_policy"]["kind"] == "cli_out"
    assert by["doctor_tournament"]["output_policy"]["kind"] == "hardcoded_rh"
    assert by["decoding_gravity"]["output_policy"]["kind"] == "hardcoded_rh"
    assert by["gravity_package"]["output_policy"]["kind"] == "package_marker"
    assert by["gravity_doctor_capability"]["output_policy"]["requires_gpu"] is True


def test_absent_declared_parent_refuses_invoke(monkeypatch):
    """A tool whose declared parent is absent must refuse, not run."""
    monkeypatch.setattr(otd, "_present", lambda p: False)
    with pytest.raises(otd.InvokeRefused) as exc:
        otd.invoke("doctor_tournament")
    msg = str(exc.value).lower()
    assert "absent" in msg or "declared" in msg or "parent" in msg
    unit = otd.emit_workunit("doctor_tournament")
    assert unit["status"] == "sleeping"
    assert unit["classification"] == "SLEEPING"
    assert unit["blocked_reason"]


def test_invoke_never_writes_headless(tmp_path):
    before = otd._headless_snapshot()
    result = otd.invoke("doctor_seal", sidecar_dir=tmp_path)
    after = otd._headless_snapshot()
    assert after == before
    assert result["headless_untouched"] is True
    assert "headless" not in result["receipt"]
    assert (tmp_path / otd.DRIVEN_SEAL).is_file()
    assert not (otd.HEADLESS / "DOCTOR_SEAL.json").exists()
    assert not intersects_codex("receipts/future/" + otd.DRIVEN_SEAL)


def test_hardcoded_rh_is_isolated_to_sidecar(tmp_path):
    """A tool that writes RH / file.json must not create receipts/headless/."""
    body = (
        "from pathlib import Path\n"
        "RH = Path('/this/must/not/be/used')\n"
        "def main():\n"
        "    RH.mkdir(parents=True, exist_ok=True)\n"
        "    (RH / 'HARDCODED.json').write_text('{\"ok\": true}')\n"
        "    return 0\n"
    )
    tool = _fixture_tool(tmp_path, body=body)
    before = otd._headless_snapshot()
    result = otd.invoke(tool, sidecar_dir=tmp_path / "out")
    after = otd._headless_snapshot()
    assert after == before
    assert result["invoked"] is True
    dest = tmp_path / "out" / "HARDCODED_DRIVEN.json"
    assert dest.is_file()
    assert json.loads(dest.read_text())["ok"] is True
    assert otd.HEADLESS.exists() is before["exists"]


def test_invocation_with_no_receipt_raises(tmp_path):
    body = (
        "from pathlib import Path\n"
        "RH = Path('/unused')\n"
        "def main():\n"
        "    return 0\n"
    )
    tool = _fixture_tool(tmp_path, body=body)
    before = otd._headless_snapshot()
    with pytest.raises(otd.NoReceipt):
        otd.invoke(tool, sidecar_dir=tmp_path / "out")
    assert otd._headless_snapshot() == before


def test_unrunnable_workunit_is_sleeping_not_pending():
    """Cope either way: absent /tmp measurement → SLEEPING; present → pending.

    The negative we always assert: a unit that cannot run is never pending.
    """
    tool = otd.tool_by_id("decoding_gravity")
    runnable, why = otd.can_run(tool)
    unit = otd.emit_workunit("decoding_gravity")
    if not runnable:
        assert unit["status"] == "sleeping"
        assert unit["classification"] == "SLEEPING"
        assert unit["blocked_reason"]
        assert unit["wake_condition"]["all_of"]
        assert "write receipts/headless" in unit["wake_condition"]["never"]
        assert why
    else:
        assert unit["status"] == "pending"
        assert unit["classification"] == "STATIC_ONLY"


def test_sleeping_is_reachable_via_package_marker():
    """Negative control that cannot flip with host state: package marker."""
    unit = otd.emit_workunit("gravity_package")
    assert unit["status"] == "sleeping"
    assert unit["classification"] == "SLEEPING"
    assert unit["blocked_reason"]
    assert unit["status"] != "pending"


def test_doctor_seal_workunit_is_pending():
    unit = otd.emit_workunit("doctor_seal")
    assert unit["status"] == "pending"
    assert unit["classification"] == "STATIC_ONLY"
    assert unit["gpu_authority"] is False
    assert unit["resource_class"] in {"STATIC_ANALYSIS", "static_analysis"}
    assert "ODYSSEY" in (unit.get("required_lanes") or [unit.get("lane")]) or unit.get("lane") == "ODYSSEY"


def test_gpu_doctor_tools_emit_sleeping():
    for tid in ("gravity_doctor_capability", "gravity_doctor_dimensions"):
        unit = otd.emit_workunit(tid)
        assert unit["status"] == "sleeping", tid
        assert "gpu" in (unit.get("blocked_reason") or "").lower()
        with pytest.raises(otd.InvokeRefused):
            otd.invoke(tid)


def test_package_marker_refuses_invoke():
    with pytest.raises(otd.InvokeRefused) as exc:
        otd.invoke("gravity_package")
    assert "package" in str(exc.value).lower() or "marker" in str(exc.value).lower()
    unit = otd.emit_workunit("gravity_package")
    assert unit["status"] == "sleeping"


def test_route_missing_receipt_raises(tmp_path):
    with pytest.raises(otd.DriverError):
        otd.route(tmp_path / "no-such.json")


def test_route_headless_receipt_refuses(tmp_path):
    fake = tmp_path / "headless_copy.json"
    fake.write_text("{}")
    # A path under receipts/headless/ is refused even if we only have the name
    # in a relative string the router sees after resolve. Plant one if the
    # directory already exists; otherwise the missing-file raise is the cope.
    planted = otd.HEADLESS / "PLANTED_ROUTE.json"
    if otd.HEADLESS.exists():
        planted.write_text("{}")
        try:
            with pytest.raises(otd.DriverError) as exc:
                otd.route(planted)
            assert "headless" in str(exc.value).lower() or "codex" in str(exc.value).lower()
        finally:
            planted.unlink()
    else:
        with pytest.raises(otd.DriverError):
            otd.route(otd.HEADLESS / "DOCTOR_TOURNAMENT.json")


def test_route_driven_seal(tmp_path):
    result = otd.invoke("doctor_seal", sidecar_dir=tmp_path)
    routed = otd.route(tmp_path / otd.DRIVEN_SEAL)
    assert routed["frontier_item"] == "FT.MODEL_CAPABILITY.hard-gates"
    assert routed["tool_id"] == "doctor_seal"
    assert result["receipt"]


def test_plan_does_not_run_or_claim_success_for_blocked_tool():
    p = otd.plan("decoding_gravity")
    assert p["evidence_class"] == "STATIC_ONLY"
    assert p["gpu_authority"] is False
    runnable, _ = otd.can_run(otd.tool_by_id("decoding_gravity"))
    if not runnable:
        assert p["runnable"] is False
        assert p["produces"] is None
    p_seal = otd.plan("doctor_seal")
    assert p_seal["runnable"] is True
    assert p_seal["produces"]["receipt"].startswith("receipts/future/")
    assert "headless" not in p_seal["produces"]["receipt"]


def test_driver_calls_owned_paths_not_just_lists_them():
    found = otd.drives_owned_paths()
    missing = [p for p, ok in found.items() if not ok]
    assert missing == [], f"gate AST probe would miss Calls for {missing}"
    tree = ast.parse(Path(otd.__file__).read_text())
    assigns_owned = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and "owned" in ast.dump(node).lower():
            assigns_owned = True
    # An owned = [...] literal must not be the only mention; Calls must exist.
    assert all(found.values())
    _ = assigns_owned


def test_this_driver_is_bound_and_the_binding_describes_disk():
    """The lane could not edit BINDINGS, so it asserted the honest state then:
    schedule was false because nothing named this driver. It is bound now, and
    what must hold is not the boolean but the reason behind it -- the gate may
    only credit a module that CALLS the tool, never one that merely names it.
    """
    assert "odyssey_tool_driver.py" in orch.BINDINGS
    frontier_id, species = orch.BINDINGS["odyssey_tool_driver.py"]
    assert frontier_id.startswith("FT.")
    assert species

    sched = otd._gate_measured()
    for side in ("doctor", "gravity"):
        assert sched[side].get("driver_module") != "odyssey_launch.py", (
            "the gate certified itself again"
        )
        if sched[side]["schedule"]:
            assert sched[side]["driver_module"] == "odyssey_tool_driver.py"


def test_refill_copes_and_does_not_invent_units():
    doc = otd.refill()
    assert "units" in doc
    assert "informed_ids" in doc
    if doc.get("ok"):
        for uid in doc["informed_ids"]:
            assert uid in set(otd.FRONTIER_OF.values())
    else:
        assert doc.get("reason")


def test_write_paths_are_sidecar_owned():
    for rel in (
        "tools/future/odyssey_tool_driver.py",
        "tools/future/test_odyssey_tool_driver.py",
        "receipts/future/ODYSSEY_TOOL_DRIVER.json",
    ):
        assert intersects_codex(rel) is False
    assert intersects_codex("receipts/headless/DOCTOR_TOURNAMENT.json") is True
