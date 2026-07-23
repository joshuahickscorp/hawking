#!/usr/bin/env python3.12
"""Durability proofs for the Second Light PQ Gravity controller.

Proves the invariants that keep the campaign honest and resumable:
  * SINGLETON        - a second controller cannot acquire the lease while one holds it.
  * RESUME           - sealed rows are never redone; the queue continues from the next pending row.
  * CRASH/RESUME x5  - every HAWKING_SL_KILL_AT point (fitting, packing, eval,
                       after_write_before_receipt, after_receipt_before_transition) resumes with
                       no duplicate work, no partial output, budget preserved, still singleton.
  * OVER_BUDGET      - a row forced over its exact budget is marked FAILED_OVER_BUDGET.
  * STATUS           - NOT_STARTED with no controller; a stale PID / historical JSON is never RUNNING.

The tests bind the REAL sealed program (sha verified) but keep all mutable state under a temp
root, so they never touch the live campaign. Packs are bounded (max_experts 1-4, a handful of
rows) so the suite runs in a couple of minutes on M3 Ultra MPS.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from second_light_controller import (  # noqa: E402
    ControllerConfig, SecondLightController, ControllerError, KILL_POINTS,
    STATUS_SEALED, STATUS_FAILED_OVER_BUDGET,
)
import second_light_status as sl_status  # noqa: E402

REPO = Path(_PKG).resolve().parents[1]
PROGRAM = REPO / "reports" / "condense" / "second_light" / "GPT_OSS_120B_PQ_GRAVITY_PROGRAM.json"
MANIFEST = "reports/condense/subbit_frontier/GRAVITY_120B_PROVENANCE.json"
CONTROLLER = str(Path(_PKG) / "second_light_controller.py")
STATUS_TOOL = str(Path(_PKG) / "second_light_status.py")

_SOURCE_PRESENT = (REPO / "models" / "gpt-oss-120b" / "original" /
                   "model--00001-of-00007.safetensors").exists()
needs_source = pytest.mark.skipif(not _SOURCE_PRESENT, reason="120B source shards absent")


def _cfg(root: Path, **kw) -> ControllerConfig:
    kw.setdefault("max_experts", 4)
    kw.setdefault("manifest_path", MANIFEST)
    return ControllerConfig(campaign_root=root, program_path=PROGRAM, **kw)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_cli(args: list[str], root: Path, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    argv = [sys.executable, CONTROLLER, *args, "--root", str(root),
            "--program", str(PROGRAM), "--manifest", MANIFEST]
    return subprocess.run(argv, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300)


# ── singleton ──────────────────────────────────────────────────────────────────────────
def test_singleton_lease_refuses_second_controller(tmp_path):
    a = SecondLightController(_cfg(tmp_path))
    a.acquire_lease()
    try:
        b = SecondLightController(_cfg(tmp_path))
        with pytest.raises(ControllerError):
            b.acquire_lease()
    finally:
        a.release_lease()
    # once released, a fresh controller may acquire it.
    c = SecondLightController(_cfg(tmp_path))
    c.acquire_lease()
    c.release_lease()


# ── program integrity ───────────────────────────────────────────────────────────────────
def test_program_sha256_verifies(tmp_path):
    ctl = SecondLightController(_cfg(tmp_path))
    ctl.load_program()
    assert ctl.program_sha256 and len(ctl.program_sha256) == 64
    assert len(ctl.rows) == 183


# ── resume: 3 sealed rows, then continue without redo ────────────────────────────────────
@needs_source
def test_resume_does_not_redo_sealed_rows(tmp_path):
    ctl = SecondLightController(_cfg(tmp_path, max_experts=4))
    summary = ctl.run(max_rows=3)
    assert summary["processed_this_invocation"] == 3
    assert summary["completed_rows"] == 3

    ckpt_dir = tmp_path / "checkpoints"
    sealed_ids = ["r0000", "r0001", "r0002"]
    for rid in sealed_ids:
        assert (ckpt_dir / f"{rid}.json").exists()
    # cursor reflects 3 done
    cursor = json.loads((tmp_path / "controller" / "checkpoint.json").read_text())
    assert cursor["completed_rows"] == 3 and cursor["pending_rows"] == 180

    shas_before = {rid: _file_sha(ckpt_dir / f"{rid}.json") for rid in sealed_ids}

    # resume continues from r0003; the three sealed rows must be untouched.
    ctl2 = SecondLightController(_cfg(tmp_path, max_experts=4))
    resume = ctl2.resume(max_rows=2)
    assert resume["processed_this_invocation"] == 2
    assert resume["completed_rows"] == 5
    for rid in sealed_ids:
        assert _file_sha(ckpt_dir / f"{rid}.json") == shas_before[rid], f"{rid} was redone"
    assert (ckpt_dir / "r0003.json").exists() and (ckpt_dir / "r0004.json").exists()


# ── crash / resume for all five kill points ──────────────────────────────────────────────
@needs_source
@pytest.mark.parametrize("point", KILL_POINTS)
def test_crash_and_resume_every_kill_point(tmp_path, point):
    common = ["--only", "r0000", "--max-rows", "1", "--max-experts", "1"]
    row_ckpt = tmp_path / "checkpoints" / "r0000.json"

    # 1) crash mid-row at `point` (a real subprocess that dies).
    crashed = _run_cli(["run", *common], tmp_path, env_extra={"HAWKING_SL_KILL_AT": point})
    assert crashed.returncode != 0, f"kill at {point} should exit non-zero"

    post_write = point in ("after_write_before_receipt", "after_receipt_before_transition")
    if post_write:
        assert row_ckpt.exists(), f"{point}: durable row checkpoint must survive the crash"
        sha_after_crash = _file_sha(row_ckpt)
    else:
        assert not row_ckpt.exists(), f"{point}: no partial row output before the durable write"

    # lease must not be held after the crash (OS released the flock) -> singleton preserved.
    snap = sl_status.snapshot(_cfg(tmp_path, only_rows=("r0000",)))
    assert snap["lease"]["live"] is False
    assert snap["state"] != "RUNNING"

    # 2) resume without the kill env.
    resumed = _run_cli(["resume", *common], tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    out = json.loads(resumed.stdout)

    if post_write:
        # adopted, not recomputed: byte-identical checkpoint, zero work this invocation.
        assert _file_sha(row_ckpt) == sha_after_crash, f"{point}: sealed row was recomputed"
        assert out["processed_this_invocation"] == 0
    else:
        # recomputed exactly once from scratch.
        assert out["processed_this_invocation"] == 1
        assert row_ckpt.exists()

    # exactly one sealed row, budget preserved, within budget, still singleton.
    assert out["completed_rows"] == 1
    cp = json.loads(row_ckpt.read_text())
    assert cp["status"] == STATUS_SEALED
    m = cp["metrics"]
    assert m["within_budget"] is True
    assert m["physical_bits"] <= m["budget_bits"]

    snap2 = sl_status.snapshot(_cfg(tmp_path, only_rows=("r0000",)))
    assert snap2["lease"]["live"] is False
    assert snap2["state"] == "COMPLETE"
    # a fresh controller can still take the lease (no leak).
    fresh = SecondLightController(_cfg(tmp_path))
    fresh.acquire_lease()
    fresh.release_lease()


# ── over budget is a hard row failure, never silently accepted ────────────────────────────
@needs_source
def test_over_budget_marks_failed(tmp_path):
    ctl = SecondLightController(_cfg(tmp_path, max_experts=1, only_rows=("r0000",),
                                     shrink_budget={"r0000": 0.001}))
    summary = ctl.run(max_rows=1)
    assert summary["failed_rows"] == 1 and summary["completed_rows"] == 0

    cp = json.loads((tmp_path / "checkpoints" / "r0000.json").read_text())
    assert cp["status"] == STATUS_FAILED_OVER_BUDGET
    m = cp["metrics"]
    assert m["within_budget"] is False
    assert m["physical_bits"] > m["budget_bits"]

    snap = sl_status.snapshot(_cfg(tmp_path, only_rows=("r0000",)))
    assert snap["state"] == "FAILED"


# ── status truth: NOT_STARTED, and a stale PID never says RUNNING ─────────────────────────
def test_status_not_started(tmp_path):
    snap = sl_status.snapshot(_cfg(tmp_path))
    assert snap["state"] == "NOT_STARTED"
    assert snap["lease"]["live"] is False
    assert snap["lease"]["heavy_controller_count"] == 0


def test_stale_pid_and_historical_json_never_running(tmp_path):
    # Fabricate a historical controller checkpoint.json with a live-looking but bogus PID,
    # and a stale lease file with a dead PID stamp. No process holds the flock.
    ctl = SecondLightController(_cfg(tmp_path))
    ctl.controller_dir.mkdir(parents=True, exist_ok=True)
    ctl.lease_path.parent.mkdir(parents=True, exist_ok=True)
    ctl.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    from eco_common import seal_field, atomic_write_json
    bogus = seal_field({
        "schema": "hawking.second_light.controller.v1",
        "controller_pid": 999999, "process_start_time": "2020-01-01T00:00:00+00:00",
        "program_sha256": "0" * 64, "current_row": "r0000", "state_hint": "running",
        "completed_rows": 5, "failed_rows": 0, "pending_rows": 178, "total_working_rows": 183,
        "completed_row_ids": [], "failed_row_ids": [],
    }, "checkpoint_sha256")
    atomic_write_json(ctl.checkpoint_path, bogus)
    ctl.lease_path.write_text(json.dumps({"pid": 999999, "acquired_at": "2020-01-01T00:00:00+00:00"}) + "\n")

    snap = sl_status.snapshot(_cfg(tmp_path))
    assert snap["state"] != "RUNNING", "a stale PID / historical JSON must never report RUNNING"
    assert snap["lease"]["live"] is False
    assert snap["lease"]["holder_alive"] is False


@needs_source
def test_redo_bounded_recomputes_bounded_seal(tmp_path):
    # Seal r0000 at bounded fidelity (1 of 128 experts).
    ctl = SecondLightController(_cfg(tmp_path, max_experts=1, only_rows=("r0000",)))
    ctl.run(max_rows=1)
    ckpt = tmp_path / "checkpoints" / "r0000.json"
    m = json.loads(ckpt.read_text())["metrics"]
    assert m["bounded_experts"] is True and m["n_experts_packed"] == 1
    sha_before = _file_sha(ckpt)

    # A default resume adopts the bounded seal (never redo a sealed row).
    ctl_adopt = SecondLightController(_cfg(tmp_path, max_experts=8, only_rows=("r0000",)))
    ctl_adopt.load_program()
    assert ctl_adopt.needs_pack("r0000") is False

    # With redo_bounded, a higher-fidelity run recomputes it.
    ctl_redo = SecondLightController(_cfg(tmp_path, max_experts=8, only_rows=("r0000",),
                                         redo_bounded=True))
    ctl_redo.load_program()
    assert ctl_redo.needs_pack("r0000") is True
    ctl_redo.run(max_rows=1)
    m2 = json.loads(ckpt.read_text())["metrics"]
    assert m2["n_experts_packed"] == 8
    assert _file_sha(ckpt) != sha_before


def test_reset_clears_state(tmp_path):
    ctl = SecondLightController(_cfg(tmp_path))
    ctl.controller_dir.mkdir(parents=True, exist_ok=True)
    (ctl.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    (ctl.checkpoints_dir / "r0000.json").write_text("{}")
    ctl.checkpoint_path.write_text("{}")
    ctl.reset()
    assert not ctl.checkpoint_path.exists()
    assert not (ctl.checkpoints_dir / "r0000.json").exists()
    snap = sl_status.snapshot(_cfg(tmp_path))
    assert snap["state"] == "NOT_STARTED"
