"""Negative controls for tools/future/restart_supervisor.py.

A guard nobody has watched fail is not a guard. These tests actually fire:

- restore from a checkpoint whose named artifact no longer exists REFUSES
- a resident-issued RestartRequest does not perform a restart on its own
- rediscover does not double-count a job in both the checkpoint and the live table
- a checkpoint truncated mid-write is refused, not half-read

No pytest.skip: absent inputs are asserted as refusals.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hcli.resources import pid_is_alive
from tools.future import restart_supervisor as rs
from tools.future._common import RECEIPTS, _assert_no_hardware_claims
from tools.future.detached import DetachedSupervisor
from tools.future.wakeup import PARTIAL_TRUNCATED, classify_receipt_bytes


def _mission() -> dict:
    return {
        "mission_id": "TEST.RESTART",
        "phase": "running",
        "next_action": "drain",
        "units": ["WU.T.1"],
        "queue": [{"id": "WU.T.2", "frontier_id": "FT.HCLI_SELF.no-launch"}],
        "frontier_ids": ["FT.HCLI_SELF.no-launch"],
        "scar_consultations": ["SCAR.T"],
    }


def _frontier() -> dict:
    return {"status": "SUPPLIED", "item_ids": ["FT.HCLI_SELF.no-launch"], "items": []}


def _sleep_unit(name: str, seconds: int, **extra):
    return {
        "id": name,
        "role": "science",
        "description": f"sleep {seconds}",
        "command": ["/bin/sleep", str(seconds)],
        "resource_class": "LIGHT_CONTROL",
        "verifier": "future.restart_supervisor.rediscover",
        "classification": "STATIC_ONLY",
        **extra,
    }


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


@pytest.fixture
def artifact(workspace: Path) -> dict:
    workspace.mkdir(parents=True, exist_ok=True)
    return rs._write_artifact(workspace, rs._RESIDENT_SCRIPT)


@pytest.fixture
def dsup(workspace: Path) -> DetachedSupervisor:
    workspace.mkdir(parents=True, exist_ok=True)
    sup = DetachedSupervisor(workspace)
    yield sup
    sup.reap_all()


# ---------------------------------------------------------------------------
# Receipt / entry point
# ---------------------------------------------------------------------------


def test_build_emits_sealed_receipt():
    out = rs.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "RESTART_SUPERVISOR.json"
    assert doc["schema"] == rs.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["status"] == "BUILT_NOT_PROMOTED"
    assert doc["promoted"] is False
    assert doc["proofs_all_passed"] is True
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert "resident_callable" in doc
    callable_ = doc["resident_callable"]
    assert callable_["entry_point"]
    assert callable_["workunit"]
    assert callable_["receipt"].endswith("RESTART_SUPERVISOR.json")
    assert callable_["frontier"] == "FT.HCLI_SELF.no-launch"
    assert callable_["fails_closed"]
    assert doc["authority_boundary"]["request_executable"] is False
    assert doc["authority_boundary"]["user_constructible_decision"] is False
    for name, row in doc["negative_controls"].items():
        assert row["fires"] is True, name
    _assert_no_hardware_claims(doc)
    assert "VI" not in "".join(doc["eras"])
    assert all("Odyssey IV" not in item and not item.startswith("IV ") for item in doc["odysseys"])


def test_ast_module_is_parseable():
    src = Path(rs.__file__).read_text()
    compile(src, rs.__file__, "exec")
    for needle in ("TODO", "NotImplementedError", "pytest.skip"):
        assert needle not in src


# ---------------------------------------------------------------------------
# Truncated checkpoint
# ---------------------------------------------------------------------------


def test_truncated_checkpoint_is_refused_not_half_read(workspace: Path):
    workspace.mkdir(parents=True, exist_ok=True)
    torn = workspace / "RESTART_CHECKPOINT.json"
    torn.write_bytes(
        b'{"schema": "hawking.future.restart_supervisor.checkpoint.v1", "complete": true, "artifact": {'
    )
    state, why = classify_receipt_bytes(
        torn.read_bytes(), required_schema=rs.CHECKPOINT_SCHEMA
    )
    assert state == PARTIAL_TRUNCATED
    assert "artifact" in why.lower() or "json" in why.lower() or "truncated" in why.lower()
    with pytest.raises(rs.CheckpointCorrupt) as exc:
        rs.load_checkpoint(torn)
    assert exc.value.fault == "truncated_mid_write"
    with pytest.raises(rs.CheckpointCorrupt):
        rs.restore(torn, workspace=workspace)


def test_leftover_tmp_sibling_is_truncated_mid_write(workspace: Path):
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / "ghost.json"
    sibling = workspace / f".{target.name}.9.deadbeef.tmp"
    sibling.write_text("{")
    with pytest.raises(rs.CheckpointCorrupt) as exc:
        rs.load_checkpoint(target)
    assert exc.value.fault == "truncated_mid_write"


def test_seal_mismatch_is_refused(workspace: Path, artifact: dict):
    ckpt = rs.checkpoint(
        _mission(), workspace=workspace, artifact=artifact, frontier=_frontier()
    )
    path = Path(ckpt["path"])
    doc = json.loads(path.read_text())
    doc["queue"] = [{"id": "forged"}]
    path.write_text(json.dumps(doc))
    with pytest.raises(rs.CheckpointCorrupt) as exc:
        rs.load_checkpoint(path)
    assert exc.value.fault == "seal_mismatch"


# ---------------------------------------------------------------------------
# Artifact existence
# ---------------------------------------------------------------------------


def test_restore_missing_artifact_refuses(workspace: Path, artifact: dict):
    ckpt = rs.checkpoint(
        _mission(), workspace=workspace, artifact=artifact, frontier=_frontier()
    )
    Path(artifact["path"]).unlink()
    with pytest.raises(rs.ArtifactMissing):
        rs.restore(ckpt["path"], workspace=workspace)
    with pytest.raises(rs.ArtifactMissing):
        rs.restart(artifact, workspace=workspace)


def test_restore_hash_mismatch_refuses(workspace: Path, artifact: dict):
    ckpt = rs.checkpoint(
        _mission(), workspace=workspace, artifact=artifact, frontier=_frontier()
    )
    Path(artifact["path"]).write_text("# tampered\n" + Path(artifact["path"]).read_text())
    with pytest.raises(rs.ArtifactMissing):
        rs.restore(ckpt["path"], workspace=workspace)


def test_restore_rebuilds_queue_and_frontier(workspace: Path, artifact: dict):
    mission = _mission()
    ckpt = rs.checkpoint(
        mission,
        workspace=workspace,
        artifact=artifact,
        frontier=_frontier(),
        queue=mission["queue"],
        in_flight=mission["units"],
        scar_consultations=mission["scar_consultations"],
    )
    restored = rs.restore(ckpt["path"], workspace=workspace)
    assert restored["result"] == "ok"
    assert restored["queue"] == mission["queue"]
    assert restored["in_flight"] == mission["units"]
    assert restored["frontier"]["item_ids"] == ["FT.HCLI_SELF.no-launch"]
    assert restored["queue_restored"] is True
    assert restored["scar_consultations"] == ["SCAR.T"]


def test_checkpoint_absent_mission_refuses(workspace: Path, artifact: dict):
    with pytest.raises(rs.RestartRefused) as exc:
        rs.checkpoint(workspace / "no-mission.json", workspace=workspace, artifact=artifact)
    assert exc.value.fault == "mission_required"


def test_checkpoint_none_without_live_mission_refuses(workspace, artifact, monkeypatch):
    """With no mission on disk, checkpoint must refuse rather than invent one.

    This asserted the live AUTONOMY_MISSION_STATE.json does not exist, which is
    true only in a sparse lane worktree -- the running loop writes it. Point the
    module at a path that really is absent instead of asserting the repo is.
    """
    monkeypatch.setattr(rs, "MISSION_STATE", workspace / "no-such-mission.json")
    assert not rs.MISSION_STATE.is_file()
    with pytest.raises(rs.RestartRefused) as exc:
        rs.checkpoint(None, workspace=workspace, artifact=artifact)
    assert exc.value.fault == "mission_required"


# ---------------------------------------------------------------------------
# Authority boundary
# ---------------------------------------------------------------------------


def test_resident_request_cannot_perform_restart(workspace: Path, artifact: dict):
    req = rs.request_restart("pathological event loop", pathology="hung")
    assert req.to_dict()["executable"] is False
    with pytest.raises(rs.RestartAuthorityError):
        req.restart
    with pytest.raises(rs.RestartAuthorityError):
        req.execute
    with pytest.raises(rs.RestartAuthorityError):
        req.restart_cycle
    with pytest.raises(rs.RestartAuthorityError):
        rs.restart_cycle(
            req,
            workspace=workspace,
            mission=_mission(),
            artifact=artifact,
            already_dead=True,
        )
    with pytest.raises(rs.RestartAuthorityError):
        rs.SupervisorDecision()


def test_empty_reason_decision_does_not_restart(workspace: Path, artifact: dict):
    dec = rs.decide(rs.request_restart(""))
    assert dec.action == "REFUSE"
    cycle = rs.restart_cycle(
        dec,
        workspace=workspace,
        mission=_mission(),
        artifact=artifact,
        already_dead=True,
    )
    assert cycle["status"] == "REFUSED"
    assert cycle["steps"]["checkpoint"]["result"] == "not_run"
    assert cycle["steps"]["restart"]["result"] == "not_run"
    assert not (workspace / rs.CHECKPOINT_NAME).exists()


def test_relative_argv_is_not_resolved_at_runtime(artifact: dict):
    rel = dict(artifact)
    rel["argv"] = ["python3", artifact["path"]]
    with pytest.raises(rs.RestartRefused) as exc:
        rs._artifact_spec(rel)
    assert exc.value.fault == "runtime_resolution_refused"


# ---------------------------------------------------------------------------
# rediscover: ingest / adopt / unknown / no double-count
# ---------------------------------------------------------------------------


def test_rediscover_does_not_double_count(dsup: DetachedSupervisor):
    rec = dsup.launch(_sleep_unit("dup", 20))
    try:
        live = dsup.list()
        handle = {
            "job_id": rec["job_id"],
            "pid": rec.get("pid"),
            "start_token": rec.get("start_token"),
        }
        out = rs.rediscover_detached(
            detached_handles=[handle],
            detached=dsup,
            live_table=live,
        )
        assert out["n_jobs"] == 1
        assert out["n_input_mentions"] >= 2
        assert out["n_duplicates_dropped"] == out["n_input_mentions"] - 1
        assert out["relaunched"] is False
        assert out["jobs"][0]["job_id"] == rec["job_id"]
        assert out["jobs"][0]["relaunched"] is False
    finally:
        dsup.cancel(rec["job_id"])


def test_rediscover_ingests_finished_and_does_not_relaunch(
    dsup: DetachedSupervisor, workspace: Path
):
    receipt = workspace / "results" / "done.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    rec = dsup.launch(
        {
            "id": "done",
            "role": "science",
            "description": "write receipt and exit",
            "command": [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{\"ok\":true}')",
                str(receipt),
            ],
            "resource_class": "LIGHT_CONTROL",
            "verifier": "future.restart_supervisor.ingest",
            "classification": "STATIC_ONLY",
            "output_receipt_path": str(receipt),
        }
    )
    terminal = dsup.wait_terminal(rec["job_id"], timeout_s=4.0)
    before = {j["job_id"] for j in dsup.list()}
    out = rs.rediscover_detached(
        detached_handles=[rs._handle_snapshot(terminal)],
        detached=dsup,
    )
    after = {j["job_id"] for j in dsup.list()}
    assert out["jobs"][0]["fate"] == "INGESTED"
    assert out["jobs"][0]["relaunched"] is False
    assert out["relaunched"] is False
    assert rec["job_id"] in before
    assert rec["job_id"] in after
    assert after == before


def test_rediscover_adopts_running(dsup: DetachedSupervisor):
    rec = dsup.launch(_sleep_unit("live", 20))
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not rec.get("pid"):
            rec = dsup.inspect(rec["job_id"])
            time.sleep(0.02)
        out = rs.rediscover_detached(
            detached_handles=[rs._handle_snapshot(rec)],
            detached=dsup,
        )
        assert out["n_jobs"] == 1
        assert out["jobs"][0]["fate"] == "ADOPTED"
        assert out["jobs"][0]["adopted"] is True
        assert out["jobs"][0]["relaunched"] is False
        assert out["relaunched"] is False
        live = dsup.inspect(rec["job_id"])
        if rec.get("pid") and rec.get("start_token"):
            assert pid_is_alive(int(rec["pid"]))
            assert live.get("terminal") is None
    finally:
        dsup.cancel(rec["job_id"])


def test_rediscover_unknown_when_fate_undetermined():
    out = rs.rediscover_detached(
        detached_handles=[{"job_id": "ghost-no-record", "pid": 999999, "start_token": "nope"}]
    )
    assert out["jobs"][0]["fate"] == "UNKNOWN"
    assert out["jobs"][0]["assumed_complete"] is False
    assert out["jobs"][0]["relaunched"] is False
    assert out["assumed_complete"] is False


def test_completed_receipt_missing_is_unknown_not_complete(
    dsup: DetachedSupervisor, workspace: Path
):
    receipt = workspace / "results" / "vanished.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    rec = dsup.launch(
        {
            "id": "vanish",
            "role": "science",
            "description": "write then we delete",
            "command": [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{\"ok\":true}')",
                str(receipt),
            ],
            "resource_class": "LIGHT_CONTROL",
            "verifier": "future.restart_supervisor.ingest",
            "classification": "STATIC_ONLY",
            "output_receipt_path": str(receipt),
        }
    )
    terminal = dsup.wait_terminal(rec["job_id"], timeout_s=4.0)
    if receipt.is_file():
        receipt.unlink()
    handle = rs._handle_snapshot(terminal)
    handle["terminal"] = "completed-with-receipt"
    handle["expected_receipt_path"] = str(receipt)
    out = rs.rediscover_detached(detached_handles=[handle])
    assert out["jobs"][0]["fate"] == "UNKNOWN"
    assert out["jobs"][0]["assumed_complete"] is False
    assert out["jobs"][0].get("observed_complete_with_receipt") is False


# ---------------------------------------------------------------------------
# stop / restart
# ---------------------------------------------------------------------------


def test_stop_cooperative(workspace: Path, artifact: dict):
    handle = rs._spawn_handle(artifact, workspace)
    try:
        stopped = rs.stop(handle, grace_s=0.5)
        assert stopped["result"] == "ok"
        assert stopped["needed"] == "cooperative"
        assert stopped["stopped"] is True
        assert stopped["signal"] == "SIGTERM"
        assert not pid_is_alive(int(handle["pid"]))
    finally:
        rs._reap_quietly(handle.get("pid"))


def test_stop_escalates_when_sigterm_ignored(workspace: Path):
    ign = rs._write_artifact(workspace, rs._IGNORING_SCRIPT, name="ignore_term.py")
    handle = rs._spawn_handle(ign, workspace)
    try:
        stopped = rs.stop(handle, grace_s=0.25)
        assert stopped["result"] == "ok"
        assert stopped["needed"] == "escalated"
        assert stopped["stopped"] is True
        assert stopped["signal"] == "SIGKILL"
        assert not pid_is_alive(int(handle["pid"]))
    finally:
        rs._reap_quietly(handle.get("pid"))


def test_stop_refuses_unproven_identity():
    foreign = subprocess.Popen(
        ["/bin/sleep", "20"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        forged = {"pid": foreign.pid, "start_token": "not-the-real-token"}
        stopped = rs.stop(forged, grace_s=0.2)
        assert stopped["needed"] == "refused"
        assert stopped["stopped"] is False
        assert pid_is_alive(foreign.pid)
    finally:
        rs._reap_quietly(foreign.pid)


def test_stop_none_refuses():
    with pytest.raises(rs.RestartRefused) as exc:
        rs.stop(None)
    assert exc.value.fault == "resident_required"


def test_restart_starts_exact_named_artifact(workspace: Path, artifact: dict):
    handle = rs.restart(artifact, workspace=workspace)
    try:
        assert handle["artifact_path"] == artifact["path"]
        assert handle["artifact_sha256"] == artifact["sha256"]
        assert handle["argv"] == artifact["argv"]
        assert handle["alive"] is True
        assert Path(artifact["argv"][0]).is_absolute()
        marker = Path(artifact["argv"][2])
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not (
            marker.is_file() and marker.read_text() == "alive"
        ):
            time.sleep(0.02)
        assert marker.read_text() == "alive"
    finally:
        rs._reap_quietly(handle.get("pid"))


def test_restart_cycle_records_each_step(workspace: Path, artifact: dict, dsup: DetachedSupervisor):
    live = rs._spawn_handle(artifact, workspace)
    decision = rs.decide(rs.request_restart("hung event loop", pathology="hung"))
    assert decision.action == "RESTART"
    cycle = rs.restart_cycle(
        decision,
        workspace=workspace,
        mission=_mission(),
        artifact=artifact,
        resident=live,
        detached=dsup,
        frontier=_frontier(),
        queue=_mission()["queue"],
        in_flight=_mission()["units"],
        scar_consultations=_mission()["scar_consultations"],
        grace_s=0.5,
    )
    new_pid = (cycle.get("resident") or {}).get("pid")
    try:
        assert cycle["status"] == "PASSED"
        assert cycle["result"] == "ok"
        for step in ("checkpoint", "stop", "restart", "restore", "rediscover_detached"):
            assert cycle["steps"][step]["result"] == "ok", step
        assert cycle["steps"]["stop"]["needed"] in {"cooperative", "escalated"}
        assert cycle["steps"]["restore"]["queue_restored"] is True
        assert cycle["steps"]["restore"]["queue"] == _mission()["queue"]
        assert cycle["steps"]["rediscover_detached"]["relaunched"] is False
        if new_pid:
            assert pid_is_alive(int(new_pid))
            assert int(new_pid) != int(live["pid"])
    finally:
        rs._reap_quietly(new_pid)
        rs._reap_quietly(live.get("pid"))
