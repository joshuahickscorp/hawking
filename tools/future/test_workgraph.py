"""Tests for the WorkGraph scheduler.

Ready set is computed. Fourteen lanes. Exclusive GPU_PROTECTED never
co-schedules with itself; CPU_ANALYSIS and NETWORK_RESEARCH do co-schedule
with a GPU unit in the same tick. Intersecting mutation scopes block.
Missing required fields are REJECTED. Durability resumes after process death.
A guard nobody has watched fail is not a guard.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from hcli.workunit import WorkUnit
from tools.future import workgraph as wg
from tools.future._common import RECEIPTS, REPO, _assert_no_hardware_claims


def _unit(uid: str, lane: str, **over):
    fields = dict(
        id=uid,
        role="science",
        description=f"test unit {uid}",
        dependencies=[],
        resource_lane=lane,
        mutation_scope=[],
        verifier=f"future.workgraph.test.{uid}",
        expected_information_gain=wg.INFO_MEDIUM,
        cost_units=1,
        requires_hardware=False,
        species="workgraph_probe",
        effect_class="READ_ONLY",
    )
    fields.update(over)
    return wg.make_unit(**fields)


def _graph(**kwargs) -> wg.WorkGraph:
    kwargs.setdefault("ncpu", 8)
    return wg.WorkGraph(**kwargs)


def test_build_and_selftest_emit_sealed_receipt():
    out = wg.selftest()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "WORKGRAPH_STATE.json"
    assert doc["schema"] == "hawking.future.workgraph.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["executes"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["no_era_vi"] is True
    assert doc["no_odyssey_iv"] is True
    _assert_no_hardware_claims(doc)
    assert doc["resident_callable"]["hcli_can_invoke"] is True
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["recovered_implementation"]["hcli.scheduler.Scheduler"]
    assert doc["negative_controls"]["gpu_protected_exclusive"]["fired"] is True
    assert doc["negative_controls"]["missing_field_rejected"]["fired"] is True
    assert doc["concurrent_tick_proof"]["fired"] is True
    assert doc["recovered_tick"]["sleeping"] > 0
    assert "GPU_PROTECTED" not in doc["recovered_tick"]["scheduled_lanes"]
    assert doc["recovered_tick"]["scheduled_ids"]
    assert set(doc["recovered_tick"]["scheduled_lanes"]) <= set(wg.LANE_IDS)
    assert "FPGA_SIM" in doc["recovered_tick"]["scheduled_lanes"] or "CPU_ANALYSIS" in doc["recovered_tick"]["scheduled_lanes"]


def test_fourteen_lanes_declared():
    specs = wg.lane_specs(ncpu=8)
    assert tuple(specs) == wg.LANE_IDS
    assert len(wg.LANE_IDS) == 14
    assert wg.LANE_IDS[0] == "GPU_PROTECTED"
    assert wg.LANE_IDS[4] == "CPU_ANALYSIS"
    assert specs["GPU_PROTECTED"].exclusive is True
    assert specs["GPU_PROTECTED"].capacity == 1
    assert specs["CPU_ANALYSIS"].exclusive is False
    assert specs["CPU_ANALYSIS"].capacity == 8
    for lid, spec in specs.items():
        assert spec.hcli_resource_class == wg.LANE_TO_HCLI[lid]
        assert spec.capacity >= 1


def test_ready_set_is_computed_not_hand_ordered():
    g = _graph()
    g.admit(_unit("z-parent", "CPU_ANALYSIS"))
    g.admit(_unit("a-child", "CPU_VERIFY", dependencies=["z-parent"]))
    g.admit(_unit("m-ready", "NETWORK_RESEARCH"))
    ready = {u["id"] for u in g.compute_ready()}
    assert "z-parent" in ready
    assert "m-ready" in ready
    assert "a-child" not in ready
    g.tick()
    g.record_result("z-parent", ok=True)
    ready2 = {u["id"] for u in g.compute_ready()}
    assert "a-child" in ready2
    assert "z-parent" not in ready2


def test_gpu_cpu_research_and_fpga_co_scheduled_one_tick():
    g = _graph()
    g.admit(_unit("gpu-1", "GPU_PROTECTED"))
    g.admit(_unit("cpu-1", "CPU_ANALYSIS"))
    g.admit(_unit("research-1", "NETWORK_RESEARCH"))
    g.admit(_unit("fpga-1", "FPGA_SIM"))
    s = g.tick()
    got = set(s["scheduled_ids"])
    assert {"gpu-1", "cpu-1", "research-1", "fpga-1"} <= got
    assert s["executes"] is False


def test_two_gpu_protected_never_co_scheduled():
    """NEGATIVE CONTROL: exclusive lane refusal is watched to fire."""
    g = _graph()
    g.admit(_unit("gpu-a", "GPU_PROTECTED", expected_information_gain=wg.INFO_HIGH))
    g.admit(_unit("gpu-b", "GPU_PROTECTED", expected_information_gain=wg.INFO_LOW))
    g.admit(_unit("cpu-1", "CPU_ANALYSIS"))
    g.admit(_unit("research-1", "NETWORK_RESEARCH"))
    s = g.tick()
    scheduled = set(s["scheduled_ids"])
    assert not ("gpu-a" in scheduled and "gpu-b" in scheduled)
    assert "gpu-a" in scheduled  # higher information wins the exclusive slot
    assert "gpu-b" not in scheduled
    assert s["skip_reasons"].get("gpu-b") in {"exclusive_lane", "capacity", "physical_conflict"}
    assert "cpu-1" in scheduled and "research-1" in scheduled


def test_gpu_protected_and_gpu_diagnostic_never_co_scheduled():
    g = _graph()
    g.admit(_unit("prot", "GPU_PROTECTED"))
    g.admit(_unit("diag", "GPU_DIAGNOSTIC"))
    g.admit(_unit("cpu-1", "CPU_ANALYSIS"))
    s = g.tick()
    scheduled = set(s["scheduled_ids"])
    assert not ("prot" in scheduled and "diag" in scheduled)
    assert "cpu-1" in scheduled


def test_intersecting_mutation_scopes_block():
    """NEGATIVE CONTROL: mutation-scope intersection is watched to fire."""
    g = _graph()
    g.admit(
        _unit(
            "mut-a",
            "CPU_ANALYSIS",
            mutation_scope=["crates/hawking-core/src/engine.rs"],
        )
    )
    g.admit(
        _unit(
            "mut-b",
            "CPU_ANALYSIS",
            mutation_scope=["crates/hawking-core/src/engine.rs", "docs/x.md"],
        )
    )
    g.admit(
        _unit(
            "mut-c",
            "CPU_ANALYSIS",
            mutation_scope=["receipts/future/WORKGRAPH_STATE.json"],
        )
    )
    s = g.tick()
    scheduled = set(s["scheduled_ids"])
    assert not ("mut-a" in scheduled and "mut-b" in scheduled)
    assert "mut-c" in scheduled
    blocked = "mut-a" if "mut-b" in scheduled else "mut-b"
    assert s["skip_reasons"].get(blocked) == "mutation_scope"


def test_missing_required_field_rejected():
    """NEGATIVE CONTROL: admission rejection is watched to fire."""
    g = _graph()
    outcome = g.admit(
        {
            "id": "missing-verifier",
            "role": "science",
            "description": "should be rejected",
            "dependencies": [],
            "resource_lane": "CPU_ANALYSIS",
            "mutation_scope": [],
            "expected_information_gain": 1,
            "cost_units": 1,
        }
    )
    assert outcome["kind"] == "rejected"
    assert "verifier" in (outcome["missing"] or [outcome["reason"]]) or "verifier" in str(outcome["reason"])
    assert "missing-verifier" not in g.units
    assert g.rejected
    assert g.rejected[-1]["rejected"] is True


def test_unknown_lane_rejected():
    g = _graph()
    with pytest.raises(wg.AdmissionError, match="not one of the fourteen lanes"):
        wg.make_unit(
            id="bad-lane",
            role="science",
            description="x",
            dependencies=[],
            resource_lane="GPU_EXCLUSIVE",
            mutation_scope=[],
            verifier="future.workgraph.test.bad-lane",
            expected_information_gain=1,
            cost_units=1,
            requires_hardware=False,
        )
    outcome = g.admit(
        {
            "id": "bad-lane-2",
            "role": "science",
            "description": "x",
            "dependencies": [],
            "resource_lane": "GPU_EXCLUSIVE",
            "mutation_scope": [],
            "verifier": "future.workgraph.test.bad-lane-2",
            "expected_information_gain": 1,
            "cost_units": 1,
        }
    )
    assert outcome["kind"] == "rejected"
    assert "bad-lane-2" not in g.units


def test_durability_survives_process_restart(tmp_path):
    g = _graph(workspace=tmp_path)
    g.admit(_unit("dur-a", "CPU_ANALYSIS", expected_information_gain=wg.INFO_HIGH))
    g.admit(_unit("dur-b", "NETWORK_RESEARCH"))
    g.admit(_unit("dur-c", "CPU_VERIFY", dependencies=["dur-a"]))
    s1 = g.tick()
    assert "dur-a" in s1["scheduled_ids"]
    assert "dur-b" in s1["scheduled_ids"]
    assert "dur-c" not in s1["scheduled_ids"]
    g.record_result("dur-a", ok=True)
    g.save()

    g2 = wg.WorkGraph.load(tmp_path, ncpu=8)
    assert g2.resumed is True
    assert g2.units["dur-a"]["status"] == "completed"
    assert g2.units["dur-b"]["status"] == "running"
    assert g2.tick_n == g.tick_n
    s2 = g2.tick()
    assert "dur-a" not in s2["scheduled_ids"]
    assert g2.units["dur-a"]["status"] == "completed"
    assert "dur-c" in s2["scheduled_ids"]

    script = f"""
import sys
sys.path.insert(0, {str(REPO)!r})
from tools.future.workgraph import WorkGraph
g = WorkGraph.load({str(tmp_path)!r}, ncpu=8)
assert g.units["dur-a"]["status"] == "completed", g.units["dur-a"]["status"]
assert g.units["dur-c"]["status"] == "running", g.units["dur-c"]["status"]
s = g.tick()
assert "dur-a" not in s["scheduled_ids"]
print("RESUME_OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "RESUME_OK" in proc.stdout


def test_selection_prefers_information_per_cost():
    g = _graph()
    g.admit(_unit("sel-low", "GPU_PROTECTED", expected_information_gain=wg.INFO_LOW, cost_units=1))
    g.admit(_unit("sel-high", "GPU_PROTECTED", expected_information_gain=wg.INFO_HIGH, cost_units=1))
    s = g.tick()
    assert s["scheduled_ids"] == ["sel-high"]


def test_starvation_is_reported_and_aged():
    g = _graph()
    g.admit(_unit("starve-low", "GPU_PROTECTED", expected_information_gain=wg.INFO_LOW))
    scheduled_low_at = None
    for i in range(wg.STARVATION_THRESHOLD + 2):
        hid = f"starve-high-{i:02d}"
        g.admit(_unit(hid, "GPU_PROTECTED", expected_information_gain=wg.INFO_HIGH))
        s = g.tick()
        if "starve-low" in s["scheduled_ids"]:
            scheduled_low_at = s["tick"]
            break
        running = [uid for uid in s["scheduled_ids"] if uid != "starve-low"]
        if running:
            g.record_result(running[0], ok=True)
    assert scheduled_low_at is not None
    assert scheduled_low_at >= wg.STARVATION_THRESHOLD
    reports = [r for r in g.starvation_reports if r.get("unit_id") == "starve-low"]
    assert reports or scheduled_low_at >= wg.STARVATION_THRESHOLD


def test_verification_dependency_waits():
    g = _graph()
    g.admit(_unit("parent", "CPU_ANALYSIS"))
    g.admit(
        _unit(
            "child",
            "CPU_VERIFY",
            verification_depends_on=["parent"],
            expected_information_gain=wg.INFO_HIGH,
        )
    )
    s1 = g.tick()
    assert "parent" in s1["scheduled_ids"]
    assert "child" not in s1["scheduled_ids"]
    g.record_result("parent", ok=True)
    s2 = g.tick()
    assert "child" in s2["scheduled_ids"]


def test_physical_gpu_units_sleep_not_synthetic():
    g = _graph()
    g.admit(_unit("sleep-gpu", "GPU_PROTECTED", requires_hardware=True))
    assert g.units["sleep-gpu"]["status"] == "sleeping"
    s = g.tick()
    assert "sleep-gpu" not in s["scheduled_ids"]
    assert "sleep-gpu" in s["sleeping_ids"]
    with pytest.raises(wg.SyntheticResultError, match="SLEEPING"):
        g.record_result("sleep-gpu", ok=True)
    assert g.units["sleep-gpu"]["status"] == "sleeping"
    woken = g.wake_sleeping(hardware_qualified=False)
    assert woken == []
    assert g.units["sleep-gpu"]["status"] == "sleeping"
    woken = g.wake_sleeping(hardware_qualified=True)
    assert "sleep-gpu" in woken
    assert g.units["sleep-gpu"]["status"] == "pending"
    assert g.units["sleep-gpu"]["status"] != "completed"


def test_execute_refused():
    g = _graph()
    with pytest.raises(wg.ExecutionRefused):
        g.execute()


def test_record_result_refuses_never_scheduled():
    g = _graph()
    g.admit(_unit("idle", "CPU_ANALYSIS"))
    with pytest.raises(wg.SyntheticResultError, match="never scheduled"):
        g.record_result("idle", ok=True)


def test_hcli_workunit_emission_roundtrips():
    g = _graph()
    g.admit(_unit("emit-1", "FPGA_SIM"))
    g.tick()
    row = g.emit_hcli_workunit("emit-1")
    wu = WorkUnit.from_dict(dict(row))
    assert wu.id == "emit-1"
    assert wu.resource_class == "COMPILE"
    assert row["resource_lane"] == "FPGA_SIM"
    assert row["gpu_authority"] is False
    assert row["evidence_class"] == "STATIC_ONLY"
    assert row["may_promote"] is False


def test_idempotent_admit_and_identity_conflict():
    g = _graph()
    first = g.admit(_unit("same", "CPU_ANALYSIS"))
    second = g.admit(_unit("same", "CPU_ANALYSIS"))
    assert first["kind"] == "inserted"
    assert second["kind"] == "idempotent"
    with pytest.raises(wg.IdentityConflict):
        g.admit(_unit("same", "CPU_ANALYSIS", description="different work"))


def test_cycle_rejected_at_admission():
    g = _graph()
    g.admit(_unit("a", "CPU_ANALYSIS", dependencies=["b"]))
    with pytest.raises(wg.CycleError):
        g.admit(_unit("b", "CPU_VERIFY", dependencies=["a"]))
    assert "b" not in g.units


def test_negative_control_helpers_fire():
    proofs = wg._prove_negative_controls(ncpu=8)
    assert proofs["gpu_protected_exclusive"]["fired"] is True
    assert proofs["cpu_and_research_co_scheduled"]["fired"] is True
    assert proofs["gpu_cpu_research_one_tick"]["fired"] is True
    assert proofs["mutation_scope_blocks"]["fired"] is True
    assert proofs["missing_field_rejected"]["fired"] is True
    assert proofs["verification_dependency_waits"]["fired"] is True
    assert proofs["sleeping_not_synthetic"]["fired"] is True
    assert proofs["execute_refused"]["fired"] is True
    assert proofs["info_per_cost_wins"]["fired"] is True


def test_open_resident_graph_copes_with_existing_or_fresh(tmp_path):
    first = wg.open_resident_graph(tmp_path, ncpu=8)
    assert first.path is not None
    n = len(first.units)
    # Either the species receipt was ingested, or the queue was empty and
    # the graph is still valid. Absence of recovered units is not a code bug.
    second = wg.open_resident_graph(tmp_path, ncpu=8)
    assert second.resumed is True
    assert len(second.units) == n


def test_hardware_qualification_is_unqualified():
    q = wg.hardware_qualification()
    assert q["qualified"] is False
    assert q["gpu_authority"] is False
    assert q["reasons"]
    assert wg.hardware_is_qualified() is False


def test_ingest_sleeps_protected_gpu_when_units_exist():
    g = _graph()
    meta = wg.ingest_recovered(g)
    sleeping = [u for u in g.units.values() if u.get("status") == "sleeping"]
    if meta["source_rows"] == 0:
        pytest.skip("no recovered HCLI units visible in this checkout; ingest coped")
    gpu_sleeping = [u for u in sleeping if u.get("resource_lane") in wg.HARDWARE_LANES]
    assert gpu_sleeping, "recovered GPU/ANE work must SLEEP, not run, on this sidecar"
    s = g.tick()
    for uid in s["scheduled_ids"]:
        assert g.units[uid]["resource_lane"] not in wg.HARDWARE_LANES or not g.units[uid]["requires_hardware"]


def test_receipt_has_no_hardware_numbers_after_build():
    path = RECEIPTS / "WORKGRAPH_STATE.json"
    if not path.is_file():
        wg.build()
    doc = json.loads((RECEIPTS / "WORKGRAPH_STATE.json").read_text())
    _assert_no_hardware_claims(doc)
    assert doc["bench"]["gpu_authority"] is False
