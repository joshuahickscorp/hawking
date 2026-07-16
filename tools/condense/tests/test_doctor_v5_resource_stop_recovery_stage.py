from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

import doctor_v5_remaining_scratch_ledger as ledger
import doctor_v5_resource_stop_recovery_stage as recovery


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def semantic(value: dict, field: str) -> dict:
    value[field] = recovery._hash_value(recovery._without(value, field))
    return value


def raw_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path: Path) -> dict:
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


class Probe:
    def __init__(self, *, swap=(2600.0, 2599.0, 2598.0),
                 free=96_000_000_000, disk=200_000_000_000,
                 pressure=1) -> None:
        self.swap = list(swap)
        self.index = 0
        self.free = free; self.disk = disk; self.pressure = pressure
        self.base = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=10)

    def __call__(self, _paths: recovery.Paths) -> dict:
        index = min(self.index, len(self.swap) - 1)
        self.index += 1
        return {
            "sampled_at": (self.base + dt.timedelta(seconds=5 * index)).isoformat(),
            "memory_pressure_level": self.pressure,
            "swap_used_mb": self.swap[index], "disk_free_bytes": self.disk,
            "thermal_nominal": True, "ac_power": True,
            "physical_memory_bytes": 103_000_000_000,
            "free_memory_percent": 93,
            "available_memory_bytes": self.free,
            "probe_commands": ["/usr/sbin/sysctl", "/usr/bin/pmset",
                               "/usr/bin/memory_pressure", "statvfs"],
        }


class Fixture:
    def __init__(self) -> None:
        (ROOT / "scratch").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "scratch")
        base = Path(self.temp.name)
        ultra = base / "reports/condense/doctor_v5_ultra"
        result = ultra / "results" / recovery.TARGET_CELL_ID
        stage = ultra / "staged_acceleration" / recovery.STAGE_DIRNAME
        self.paths = recovery.Paths(
            root=ROOT, ultra=ultra, plan=ultra / "campaign_plan.json",
            state=ultra / "queue_state.json", control=ultra / "control.json",
            pid_file=ultra / "queue.pid.json", queue_lock=ultra / "queue.lock",
            heavy_lock=base / "reports/cron/studio_heavy.lock",
            runtime_spec=ultra / "runtime_specs" / f"{recovery.TARGET_CELL_ID}.json",
            result_dir=result, request=result / "request.json",
            registry_snapshot=result / "adapter_registry.json",
            live_registry=ultra / "adapter_registry.json",
            adapter_checkpoint=result / "checkpoint.json",
            worker_checkpoint=result / "strand_ladder/checkpoint.json",
            worker_request=result / "strand_ladder/request.json",
            resource_stop=result / "resource_stop.json",
            result=result / "result.json", execution_receipt=result / "execution_receipt.json",
            active_marker=ultra / "staged_acceleration/active_stack.json",
            aggressive_overlay=ultra / "staged_acceleration/aggressive_v2/aggressive_admission_overlay.json",
            ledger_tool=ROOT / "tools/condense/doctor_v5_remaining_scratch_ledger.py",
            remaining_scratch_gate_adapter=ROOT / (
                "tools/condense/doctor_v5_remaining_scratch_gate_adapter.py"),
            swap_policy_tool=ROOT / "tools/condense/doctor_v5_aggressive_admission_policy.py",
            legacy_recovery_tool=ROOT / "tools/condense/doctor_v5_blocked_cell_recovery.py",
            stage_root=stage, packet=stage / "recovery_packet.json",
            stage_lock=stage / "stage.lock", swap_proof_dir=stage / "swap_promotions",
        )
        for directory in (self.paths.runtime_spec.parent, result / "strand_ladder",
                          self.paths.heavy_lock.parent, stage):
            directory.mkdir(parents=True, exist_ok=True)
        self.paths.queue_lock.touch(); self.paths.heavy_lock.touch()
        write_json(self.paths.active_marker, {})
        write_json(self.paths.aggressive_overlay, {})

        output = self.paths.worker_request.parent
        source = base / "source"
        (output / "bundle/shards").mkdir(parents=True)
        (output / "evaluation/reconstruction").mkdir(parents=True)
        source.mkdir()
        source_rows = []
        for ordinal in range(3):
            path = source / f"model-{ordinal:05d}.safetensors"
            path.write_bytes(bytes([ordinal + 1]) * (3 + ordinal))
            source_rows.append({"bytes": path.stat().st_size, "name": path.name,
                                "ordinal": ordinal, "path": str(path),
                                "sha256": raw_sha(path)})
        worker_request = {
            "schema": ledger.REQUEST_SCHEMA, "request_id": "3bpw-test",
            "label": "14B", "model_family": "qwen2.5-dense",
            "campaign_binding": {"cell_id": recovery.TARGET_CELL_ID},
            "codec": {}, "source": {"census_path": str(base / "census.json"),
                "census_sha256": "1" * 64, "model_dir": str(source),
                "shards": source_rows, "source_manifest_sha256": "2" * 64},
            "parameter_manifest": {}, "execution": {},
            "evaluation": {"mode": "resident", "retain_dense_reconstruction": False},
            "doctor_hook": {}, "resources": {
                "disk_reserve_bytes": ledger.DISK_RESERVE_BYTES,
                "scratch_budget_bytes": 48_000_000_000},
            "output_root": str(output), "evidence_policy": {},
        }
        write_json(self.paths.worker_request, worker_request)

        packed: list[Path] = []
        recon: list[Path] = []
        passthrough: list[Path] = []
        for ordinal, (packed_size, recon_size) in enumerate(((5, 13), (7, 17), (8, 19))):
            p = output / f"bundle/shards/{ordinal:05d}.strand"
            p.write_bytes(bytes([20 + ordinal]) * packed_size); packed.append(p)
            q = output / f"evaluation/reconstruction/{ordinal:05d}.safetensors"
            if ordinal < 2: q.write_bytes(bytes([30 + ordinal]) * recon_size)
            recon.append(q)
            t = output / f"bundle/shards/{ordinal:05d}.passthrough.safetensors"
            t.write_bytes(bytes([40 + ordinal]) * 2); passthrough.append(t)
        plan_units = ["preflight", "metadata"]
        for ordinal in range(3):
            plan_units.extend([f"passthrough:{ordinal:05d}", f"encode:{ordinal:05d}",
                               f"attest:{ordinal:05d}", f"decode:{ordinal:05d}"])
        plan_units.extend(ledger.RESIDENT_SUFFIX)
        completed = list(recovery.EXPECTED_COMPLETED_UNITS)
        units: dict[str, dict] = {}
        for unit in completed:
            units[unit] = {"completed_at": "2026-07-14T00:00:00+00:00"}
            match = ledger.ORDINAL_UNIT_RE.fullmatch(unit)
            if match:
                phase, raw = match.groups(); ordinal = int(raw)
                if phase == "passthrough": units[unit]["artifact"] = ref(passthrough[ordinal])
                if phase == "encode": units[unit]["artifact"] = ref(packed[ordinal])
                if phase == "attest": units[unit]["archive"] = ref(packed[ordinal])
                if phase == "decode": units[unit]["artifact"] = ref(recon[ordinal])
        worker_checkpoint = {
            "schema": ledger.CHECKPOINT_SCHEMA,
            "request_sha256": raw_sha(self.paths.worker_request),
            "created_at": "2026-07-14T00:00:00+00:00",
            "updated_at": "2026-07-14T00:01:00+00:00", "status": "running",
            "plan": plan_units, "completed_units": completed, "units": units,
            "stop_requested": False,
        }
        write_json(self.paths.worker_checkpoint, worker_checkpoint)

        registry = semantic({"schema": "registry", "entries": []}, "registry_sha256")
        write_json(self.paths.registry_snapshot, registry); write_json(self.paths.live_registry, registry)
        request = semantic({"schema": "adapter-request",
                            "registry_sha256": registry["registry_sha256"]}, "request_sha256")
        write_json(self.paths.request, request)
        outer = semantic({"schema": "hawking.doctor_v5_adapter_exact_resume_checkpoint.v1",
                          "request_sha256": request["request_sha256"],
                          "resume_state_sha256": raw_sha(self.paths.worker_checkpoint),
                          "completed_units": completed, "status": "running"},
                         "checkpoint_sha256")
        write_json(self.paths.adapter_checkpoint, outer)

        tools = base / "tools"; tools.mkdir()
        binaries = {}
        input_rows = []
        for role, payload in (("quantizer", b"quantizer"), ("attestor", b"attestor"),
                              ("decoder", b"decoder")):
            path = tools / role; path.write_bytes(payload)
            binaries[role] = (raw_sha(path), path.stat().st_size)
            input_rows.append({"role": role, **ref(path)})
        runtime = {"schema": "runtime", "inputs": input_rows}
        write_json(self.paths.runtime_spec, runtime)
        cell_identity = "8" * 64
        plan = semantic({"schema": "plan", "cells": [{
            "cell_id": recovery.TARGET_CELL_ID,
            "cell_identity_sha256": cell_identity,
            "projected_output_bytes": 20, "source_deletion_permitted": False,
            "lifecycle": {"parent_source_cleanup": "disabled_separate_operator_action_only"},
        }]}, "plan_sha256")
        write_json(self.paths.plan, plan)
        target = {
            "attempts": 10, "blockers": [
                "resource-stop ceiling reached: its residency alone reaches the process RAM budget as the sole live lane"],
            "completed_at": None, "disposition_sha256": None,
            "error": "resource-stop ceiling reached: its residency alone reaches the process RAM budget as the sole live lane",
            "execution_receipt_sha256": None, "last_exit_code": 75,
            "packed_gc_receipt_sha256": None, "payload_released_at": None,
            "registry_sha256": registry["registry_sha256"], "released_payload_bytes": 0,
            "request_sha256": request["request_sha256"], "result_sha256": None,
            "runtime_spec_sha256": raw_sha(self.paths.runtime_spec),
            "started_at": "2026-07-14T00:00:00+00:00", "status": "blocked-execution",
        }
        other = {"status": "complete", "attempts": 1, "result_sha256": "9" * 64}
        state = semantic({"schema": "state", "plan_sha256": plan["plan_sha256"],
                          "status": "waiting-prerequisites", "supervisor_pid": None,
                          "active_cells": [], "active_children": {},
                          "cells": {recovery.TARGET_CELL_ID: target, "other": other}},
                         "state_sha256")
        write_json(self.paths.state, state)
        control = semantic({"schema": "control", "plan_sha256": plan["plan_sha256"],
                            "mode": "run"}, "control_sha256")
        write_json(self.paths.control, control)
        owner = semantic({"schema": "hawking.doctor_v5_ultra_queue_pid.v1",
                          "pid": 999999, "process_started": "never",
                          "process_command_sha256": "3" * 64}, "pid_record_sha256")
        write_json(self.paths.pid_file, owner)
        stop = semantic({
            "schema": "hawking.doctor_v5_ultra_resource_stop.v1",
            "cell_id": recovery.TARGET_CELL_ID, "cell_identity_sha256": cell_identity,
            "plan_sha256": plan["plan_sha256"], "request_sha256": request["request_sha256"],
            "reason": "system_memory_pressure_or_swap",
            "resume_policy": "retry_exact_checkpoint_after_resource_gate_recovers",
            "parent_source_deleted": False,
            "checkpoint": {"path": str(self.paths.adapter_checkpoint),
                           "sha256": raw_sha(self.paths.adapter_checkpoint),
                           "bytes": self.paths.adapter_checkpoint.stat().st_size}},
            "receipt_sha256")
        write_json(self.paths.resource_stop, stop)

        self.pins = recovery.Pins(
            plan_sha256=plan["plan_sha256"], cell_identity_sha256=cell_identity,
            runtime_file_sha256=raw_sha(self.paths.runtime_spec),
            request_sha256=request["request_sha256"],
            registry_sha256=registry["registry_sha256"],
            adapter_checkpoint_file_sha256=raw_sha(self.paths.adapter_checkpoint),
            worker_checkpoint_file_sha256=raw_sha(self.paths.worker_checkpoint),
            worker_request_file_sha256=raw_sha(self.paths.worker_request),
            resource_stop_file_sha256=raw_sha(self.paths.resource_stop),
            resource_stop_receipt_sha256=stop["receipt_sha256"],
            target_row_sha256=recovery._hash_value(target), attempts=10,
            projected_packed_output_bytes=20, binaries=binaries,
        )
        self.probe = Probe()

    def close(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def legacy(_paths, _owners):
        return {"target_cell_id": "qwen2-5-14b__4bpw__codec-control",
                "structurally_ready": True, "target_row_sha256": "4" * 64,
                "plan_sha256": "5" * 64, "checkpoint": {}, "bindings": {},
                "errors": []}

    def kwargs(self, **overrides):
        values = {"owner_observer": lambda: [], "lock_observer": lambda _paths: [],
                  "resource_probe": self.probe, "legacy_inspector": self.legacy}
        values.update(overrides); return values

    def seal(self, probe: Probe | None = None) -> Path:
        return recovery.seal_swap_promotion(
            self.paths, self.pins, owner_observer=lambda: [],
            lock_observer=lambda _paths: [], resource_probe=probe or self.probe,
            sleep=lambda _seconds: None, interval_seconds=0,
        )


class ResourceStopRecoveryStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Fixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_green_generation_stages_and_verifies_but_live_queue_remains_unwired(self) -> None:
        self.fixture.seal()
        packet = recovery.stage(self.fixture.paths, self.fixture.pins,
                                **self.fixture.kwargs())
        self.assertFalse(packet["future_commit_gates_ready_at_stage"])
        self.assertIn("CAS-only recovery would re-block", " ".join(packet["blockers_at_stage"]))
        self.assertFalse(packet["activation_permitted"])
        self.assertFalse(packet["apply_implementation_present"])
        self.assertEqual([], recovery.validate_packet(
            packet, self.fixture.paths, self.fixture.pins, **self.fixture.kwargs()))
        self.assertNotIn("apply", recovery.main.__code__.co_consts)

    def test_resealed_receipt_cannot_claim_live_consumer_or_erase_blocker(self) -> None:
        self.fixture.seal()
        packet = recovery.stage(self.fixture.paths, self.fixture.pins,
                                **self.fixture.kwargs())
        compatibility = packet["queue_generation_admission_compatibility"]
        compatibility["live_queue_remaining_scratch_consumption_absent"] = False
        compatibility["active_queue_phase_aware_receipt_consumer_bound"] = True
        compatibility["separate_adapter_wired_into_live_queue"] = True
        compatibility["bypass_permitted"] = True
        packet["blockers_at_stage"] = ["stable swap proof is absent"]
        packet["future_commit_gates_ready_at_stage"] = True
        packet["packet_sha256"] = recovery._hash_value(
            recovery._without(packet, "packet_sha256"))
        errors = recovery.validate_packet(
            packet, self.fixture.paths, self.fixture.pins,
            require_current=False)
        self.assertTrue(any("live-queue" in row or "mandatory" in row
                            for row in errors), errors)

    def test_owner_and_open_lock_block_future_commit(self) -> None:
        status = recovery.inspect(
            self.fixture.paths, self.fixture.pins,
            **self.fixture.kwargs(owner_observer=lambda: [{"pid": 7}],
                                  lock_observer=lambda _paths: [{"pid": 8}]))
        self.assertIn("one or more heavy owners are active", status["blockers"])
        self.assertIn("one or more live campaign locks have open holders", status["blockers"])

    def test_stale_pin_and_other_row_drift_are_rejected(self) -> None:
        self.fixture.seal(); packet = recovery.stage(
            self.fixture.paths, self.fixture.pins, **self.fixture.kwargs())
        state = json.loads(self.fixture.paths.state.read_text())
        state["cells"]["other"]["result_sha256"] = "a" * 64
        semantic(state, "state_sha256"); write_json(self.fixture.paths.state, state)
        errors = recovery.validate_packet(packet, self.fixture.paths, self.fixture.pins,
                                          **self.fixture.kwargs())
        self.assertTrue(any("stale" in row or "state" in row for row in errors), errors)

    def test_checkpoint_loss_is_rejected(self) -> None:
        checkpoint = json.loads(self.fixture.paths.worker_checkpoint.read_text())
        checkpoint["completed_units"].pop()
        checkpoint["units"].pop("passthrough:00002")
        write_json(self.fixture.paths.worker_checkpoint, checkpoint)
        status = recovery.inspect(self.fixture.paths, self.fixture.pins,
                                  **self.fixture.kwargs())
        self.assertFalse(status["structurally_ready"])
        self.assertTrue(any("checkpoint" in row for row in status["errors"]))

    def test_attempt_or_extra_field_clearing_is_rejected(self) -> None:
        self.fixture.seal(); packet = recovery.stage(
            self.fixture.paths, self.fixture.pins, **self.fixture.kwargs())
        packet["proposed_transaction"]["attempts_after"] = 0
        packet["proposed_transaction"]["allowed_patch"]["attempts"] = 0
        packet["packet_sha256"] = recovery._hash_value(
            recovery._without(packet, "packet_sha256"))
        errors = recovery.validate_packet(packet, self.fixture.paths, self.fixture.pins,
                                          require_current=False)
        self.assertTrue(any("transition" in row or "transaction" in row for row in errors))

    def test_symlink_and_observation_race_fail_closed(self) -> None:
        checkpoint = json.loads(self.fixture.paths.worker_checkpoint.read_text())
        path = Path(checkpoint["units"]["decode:00000"]["artifact"]["path"])
        payload = path.read_bytes(); path.unlink(); os.symlink(self.fixture.paths.worker_request, path)
        status = recovery.inspect(self.fixture.paths, self.fixture.pins,
                                  **self.fixture.kwargs())
        self.assertFalse(status["structurally_ready"])
        self.assertTrue(any("symlink" in row for row in status["errors"]), status["errors"])
        path.unlink(); path.write_bytes(payload)
        stable = (1, 2, 1, len(payload), 3, 4)
        changed = (1, 2, 1, len(payload), 5, 6)
        with mock.patch.object(recovery, "_identity",
                               side_effect=[stable, stable, stable, changed]):
            with self.assertRaisesRegex(recovery.StageError, "changed while"):
                recovery._stable_size(path, ROOT, len(payload))

    def test_high_or_rising_swap_generation_is_rejected(self) -> None:
        high = Probe(swap=(4097.0, 4096.0, 4095.0))
        with self.assertRaisesRegex(recovery.StageError, "emergency"):
            self.fixture.seal(high)
        rising = Probe(swap=(100.0, 200.0, 300.0))
        with self.assertRaisesRegex(recovery.StageError, "rising"):
            self.fixture.seal(rising)

    def test_disk_shortfall_blocks_without_invalidating_checkpoint(self) -> None:
        self.fixture.seal()
        low = Probe(disk=1)
        status = recovery.inspect(self.fixture.paths, self.fixture.pins,
                                  **self.fixture.kwargs(resource_probe=low))
        self.assertTrue(status["structurally_ready"], status["errors"])
        self.assertIn("phase-aware disk/lifecycle admission has a shortfall",
                      status["blockers"])

    def test_caller_supplied_swap_path_is_not_a_cli_option(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                recovery.main(["status", "--swap-proof", "/tmp/untrusted.json"])

    def test_legacy_4bpw_drift_invalidates_3bpw_stage(self) -> None:
        status = recovery.inspect(
            self.fixture.paths, self.fixture.pins,
            **self.fixture.kwargs(legacy_inspector=lambda _p, _o: {
                "target_cell_id": "qwen2-5-14b__4bpw__codec-control",
                "structurally_ready": False, "errors": ["drift"]}))
        self.assertFalse(status["structurally_ready"])
        self.assertTrue(any("4bpw" in row for row in status["errors"]))


if __name__ == "__main__":
    unittest.main()
