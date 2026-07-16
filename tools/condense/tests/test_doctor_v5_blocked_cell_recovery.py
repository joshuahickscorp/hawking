from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "tools/condense/doctor_v5_blocked_cell_recovery.py"
SPEC = importlib.util.spec_from_file_location("doctor_v5_blocked_cell_recovery_test", MODULE_PATH)
assert SPEC and SPEC.loader
recovery = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = recovery
SPEC.loader.exec_module(recovery)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def semantic(document, field):
    document[field] = hashlib.sha256(canonical({
        key: value for key, value in document.items() if key != field
    })).hexdigest()
    return document


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class Fixture:
    KEY_A = "first-independent-recovery-key-0001"
    KEY_B = "second-independent-recovery-key-0002"

    def __init__(self):
        scratch = ROOT / "scratch"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        base = Path(self.temporary.name)
        ultra = base / "reports/condense/doctor_v5_ultra"
        result = ultra / "results" / recovery.TARGET_CELL_ID
        stage = ultra / "staged_acceleration" / recovery.STAGE_DIRNAME
        self.paths = recovery.RecoveryPaths(
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
            execution_log=result / "execution.log", result=result / "result.json",
            execution_receipt=result / "execution_receipt.json",
            active_marker=ultra / "staged_acceleration/active_stack.json",
            accelerated_autoresume=base / "tools/autoresume.py",
            accelerated_queue=base / "tools/accelerated_queue.py",
            stage_root=stage, packet=stage / "recovery_packet.json",
            recovery_lock=stage / "recovery.lock", intent=stage / "apply_intent.json",
            receipt=stage / "apply_receipt.json",
            resume_receipt=stage / "resume_receipt.json",
            resume_log=stage / "autoresume.log",
        )
        for directory in (
            ultra / "runtime_specs", result / "strand_ladder", stage,
            self.paths.heavy_lock.parent, self.paths.accelerated_queue.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.paths.queue_lock.touch(); self.paths.heavy_lock.touch()
        self.paths.accelerated_autoresume.write_text("# autoresume\n", encoding="utf-8")
        self.paths.accelerated_queue.write_text("# accelerated queue\n", encoding="utf-8")

        tools = base / "bound-tools"
        tools.mkdir()
        self.attestor = tools / "attest-strand"
        self.decoder = tools / "archive-to-safetensors"
        self.source = tools / "source.safetensors"
        self.helper = tools / "adapter.py"
        for path, payload in (
            (self.attestor, b"attestor"), (self.decoder, b"decoder"),
            (self.source, b"source"), (self.helper, b"helper"),
        ):
            path.write_bytes(payload)

        registry = semantic({"schema": "registry", "entries": []}, "registry_sha256")
        write_json(self.paths.registry_snapshot, registry)
        write_json(self.paths.live_registry, registry)
        request = semantic({
            "schema": "request", "registry_sha256": registry["registry_sha256"]
        }, "request_sha256")
        write_json(self.paths.request, request)

        worker_request = {"schema": "worker-request", "target": recovery.TARGET_CELL_ID}
        write_json(self.paths.worker_request, worker_request)
        worker_request_sha = recovery._hash_regular(self.paths.worker_request, ROOT)[0]

        unit_artifact = result / "strand_ladder/bundle/unit.bin"
        unit_artifact.parent.mkdir(parents=True)
        unit_artifact.write_bytes(b"durable-checkpoint-artifact")
        unit_ref = recovery._artifact(unit_artifact, ROOT)
        plan_units = ["preflight", "metadata"]
        for index in range(8):
            plan_units.extend([
                f"passthrough:{index:05d}", f"encode:{index:05d}",
                f"attest:{index:05d}", f"decode:{index:05d}",
            ])
        plan_units.extend([
            "bundle_manifest", "override_manifest", "baseline_ppl",
            "reconstruction_ppl", "baseline_capability",
            "reconstruction_capability", "ephemeral_cleanup", "receipt",
        ])
        completed = plan_units[:plan_units.index("attest:00007")]
        worker_checkpoint = {
            "schema": "hawking.doctor_v5_strand_ladder_checkpoint.v1",
            "request_sha256": worker_request_sha, "plan": plan_units,
            "completed_units": completed,
            "units": {unit: {"artifact": unit_ref} for unit in completed},
            "status": "running", "stop_requested": False,
        }
        write_json(self.paths.worker_checkpoint, worker_checkpoint)
        outer = semantic({
            "schema": "adapter-checkpoint", "request_sha256": request["request_sha256"],
            "status": "running", "artifact_hashes": [],
        }, "checkpoint_sha256")
        write_json(self.paths.adapter_checkpoint, outer)

        def input_row(role, path):
            row = recovery._artifact(path, ROOT); row["role"] = role; return row

        runtime = {"schema": "runtime", "inputs": [
            input_row("attestor", self.attestor), input_row("decoder", self.decoder),
            input_row("adapter_source", self.helper),
            input_row("source_shard:00000", self.source),
        ]}
        write_json(self.paths.runtime_spec, runtime)
        runtime_sha = recovery._hash_regular(self.paths.runtime_spec, ROOT)[0]

        cell_identity = "1" * 64
        plan = semantic({
            "schema": "plan", "cells": [
                {"cell_id": recovery.TARGET_CELL_ID,
                 "cell_identity_sha256": cell_identity},
                {"cell_id": "completed-control", "cell_identity_sha256": "2" * 64},
            ],
        }, "plan_sha256")
        write_json(self.paths.plan, plan)
        target_row = {
            "status": "blocked-execution", "attempts": 14,
            "started_at": "2026-07-14T00:00:00+00:00", "completed_at": None,
            "last_exit_code": 2, "blockers": ["typed adapter exited with status 2"],
            "runtime_spec_sha256": runtime_sha,
            "registry_sha256": registry["registry_sha256"],
            "request_sha256": request["request_sha256"], "result_sha256": None,
            "execution_receipt_sha256": None, "disposition_sha256": None,
            "packed_gc_receipt_sha256": None, "payload_released_at": None,
            "released_payload_bytes": 0,
            "error": "typed adapter exited with status 2",
        }
        completed_row = copy.deepcopy(target_row)
        completed_row.update({"status": "complete", "attempts": 1,
                              "last_exit_code": 0, "blockers": [], "error": None,
                              "result_sha256": "3" * 64,
                              "execution_receipt_sha256": "4" * 64})
        state = semantic({
            "schema": "state", "plan_sha256": plan["plan_sha256"],
            "status": "waiting-prerequisites", "control_mode": "run",
            "supervisor_pid": None, "active_cells": [], "active_children": {},
            "updated_at": "2026-07-14T00:00:00+00:00",
            "cells": {recovery.TARGET_CELL_ID: target_row,
                      "completed-control": completed_row},
            "source_deletion_permitted": False,
        }, "state_sha256")
        write_json(self.paths.state, state)
        control = semantic({
            "schema": "control", "plan_sha256": plan["plan_sha256"],
            "mode": "run", "sequence": 1,
        }, "control_sha256")
        write_json(self.paths.control, control)
        owner_record = semantic({
            "schema": "hawking.doctor_v5_ultra_queue_pid.v1", "version": "test",
            "pid": 999_999, "process_started": "never",
            "process_command_sha256": "6" * 64,
            "ownership_nonce": "7" * 32, "plan_sha256": plan["plan_sha256"],
            "recorded_at": "2026-07-14T00:00:00+00:00",
        }, "pid_record_sha256")
        write_json(self.paths.pid_file, owner_record)
        self.paths.execution_log.write_text(json.dumps({
            "status": "refused",
            "error": f"[Errno 2] No such file or directory: '{self.attestor}'",
        }) + "\n", encoding="utf-8")

        overlay = semantic({"schema": "overlay", "enabled": True}, "overlay_sha256")
        overlay_path = stage.parent / "overlay.json"
        write_json(overlay_path, overlay)
        marker = {
            "schema": "hawking.doctor_v5_acceleration_active_marker.v1",
            "activated_at": "2026-07-14T00:00:00+00:00",
            "overlay_path": str(overlay_path),
            "overlay_sha256": overlay["overlay_sha256"],
            "pending_runtime_generation_sha256": "5" * 64,
            "accelerated_queue": recovery._artifact(self.paths.accelerated_queue, ROOT),
            "accelerated_autoresume": recovery._artifact(
                self.paths.accelerated_autoresume, ROOT),
        }
        semantic(marker, "marker_sha256")
        write_json(self.paths.active_marker, marker)

        self.pins = recovery.RecoveryPins(
            plan_sha256=plan["plan_sha256"], cell_identity_sha256=cell_identity,
            runtime_file_sha256=runtime_sha,
            request_sha256=request["request_sha256"],
            registry_sha256=registry["registry_sha256"],
            adapter_checkpoint_file_sha256=recovery._hash_regular(
                self.paths.adapter_checkpoint, ROOT)[0],
            worker_checkpoint_file_sha256=recovery._hash_regular(
                self.paths.worker_checkpoint, ROOT)[0],
            worker_request_file_sha256=worker_request_sha,
            autoresume_sha256=recovery._hash_regular(
                self.paths.accelerated_autoresume, ROOT)[0],
            accelerated_queue_sha256=recovery._hash_regular(
                self.paths.accelerated_queue, ROOT)[0],
            binaries={
                "attestor": {"sha256": recovery._hash_regular(self.attestor, ROOT)[0],
                             "bytes": self.attestor.stat().st_size},
                "decoder": {"sha256": recovery._hash_regular(self.decoder, ROOT)[0],
                            "bytes": self.decoder.stat().st_size},
            }, attempts=14,
        )

    def close(self):
        self.temporary.cleanup()

    def stage(self):
        return recovery.stage_packet(
            self.paths, self.pins, key_a=self.KEY_A, key_b=self.KEY_B,
            owner_observer=lambda: [], probe_locks=True,
        )


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Fixture()

    def tearDown(self):
        self.fixture.close()

    def test_clean_fixture_is_structurally_and_activation_ready(self):
        status = recovery.inspect_recovery(
            self.fixture.paths, self.fixture.pins, owner_observer=lambda: []
        )
        self.assertTrue(status["structurally_ready"], status["errors"])
        self.assertTrue(status["activation_permitted"], status["blockers"])
        self.assertEqual(status["checkpoint"]["last_completed_unit"], "encode:00007")
        self.assertEqual(status["checkpoint"]["next_unit"], "attest:00007")

    def test_active_supervisor_refuses_activation(self):
        with mock.patch.object(recovery, "_supervisor_alive", return_value=True):
            status = recovery.inspect_recovery(
                self.fixture.paths, self.fixture.pins, owner_observer=lambda: [],
                probe_locks=False,
            )
        self.assertIn("detached Doctor supervisor is active", status["blockers"])
        self.assertFalse(status["activation_permitted"])

    def test_fake_or_external_owner_refuses_activation(self):
        status = recovery.inspect_recovery(
            self.fixture.paths, self.fixture.pins,
            owner_observer=lambda: [{"pid": 999, "command": "fake-heavy"}],
            probe_locks=False,
        )
        self.assertIn("one or more heavy owners are active", status["blockers"])

    def test_full_hashing_is_refused_before_payload_reads_while_owned(self):
        unit_artifact = (self.fixture.paths.result_dir
                         / "strand_ladder/bundle/unit.bin")
        original = unit_artifact.read_bytes()
        unit_artifact.write_bytes(b"X" * len(original))
        owned = recovery.inspect_recovery(
            self.fixture.paths, self.fixture.pins, full=True,
            owner_observer=lambda: [{"pid": 999, "command": "heavy-owner"}],
            probe_locks=False,
        )
        self.assertFalse(owned["full_checkpoint_verification"])
        self.assertIn("full payload hashing refused until the owner-free gate passes",
                      owned["blockers"])
        self.assertFalse(any("checkpoint artifact hash changed" in row
                             for row in owned["errors"]))
        unowned = recovery.inspect_recovery(
            self.fixture.paths, self.fixture.pins, full=True,
            owner_observer=lambda: [], probe_locks=False,
        )
        self.assertTrue(any("checkpoint artifact hash changed" in row
                            for row in unowned["errors"]))

    def test_stale_state_and_non_target_mutation_are_refused(self):
        packet = self.fixture.stage()
        state = json.loads(self.fixture.paths.state.read_text())
        state["cells"]["completed-control"]["result_sha256"] = "9" * 64
        semantic(state, "state_sha256")
        write_json(self.fixture.paths.state, state)
        errors = recovery.validate_packet(
            packet, self.fixture.paths, self.fixture.pins,
            owner_observer=lambda: [], probe_locks=False,
        )
        self.assertTrue(any("state" in row or "stale" in row for row in errors), errors)
        with self.assertRaisesRegex(recovery.RecoveryError, "state generation"):
            recovery.build_after_state(state, packet)

    def test_changed_binary_is_refused(self):
        packet = self.fixture.stage()
        self.fixture.attestor.write_bytes(b"changed")
        errors = recovery.validate_packet(
            packet, self.fixture.paths, self.fixture.pins,
            owner_observer=lambda: [], probe_locks=False,
        )
        self.assertTrue(any("attestor" in row for row in errors), errors)

    def test_changed_checkpoint_is_refused(self):
        packet = self.fixture.stage()
        checkpoint = json.loads(self.fixture.paths.worker_checkpoint.read_text())
        checkpoint["updated_at"] = "tampered"
        write_json(self.fixture.paths.worker_checkpoint, checkpoint)
        errors = recovery.validate_packet(
            packet, self.fixture.paths, self.fixture.pins,
            owner_observer=lambda: [], probe_locks=False,
        )
        self.assertTrue(any("worker_checkpoint" in row or "checkpoint" in row
                            for row in errors), errors)

    def test_path_escape_and_symlink_are_refused(self):
        with self.assertRaisesRegex(recovery.RecoveryError, "not confined"):
            recovery._confined(Path("/tmp"), self.fixture.paths.result_dir)
        link = self.fixture.paths.result_dir / "strand_ladder/symlink.bin"
        link.symlink_to(self.fixture.attestor)
        with self.assertRaisesRegex(recovery.RecoveryError, "symlink"):
            recovery._hash_regular(link, ROOT)

    def test_duplicate_apply_is_refused(self):
        self.fixture.stage()
        write_json(self.fixture.paths.receipt, {"already": True})
        with self.assertRaisesRegex(recovery.RecoveryError, "duplicate"):
            recovery.apply_packet(
                self.fixture.paths, self.fixture.paths.packet,
                key_a=self.fixture.KEY_A, key_b=self.fixture.KEY_B,
                pins=self.fixture.pins, owner_observer=lambda: [],
                resume_launcher=lambda _paths: {"returncode": 0},
            )

    def test_wrong_activation_key_is_refused_before_locks_or_mutation(self):
        self.fixture.stage()
        before = self.fixture.paths.state.read_bytes()
        with self.assertRaisesRegex(recovery.RecoveryError, "activation keys"):
            recovery.apply_packet(
                self.fixture.paths, self.fixture.paths.packet,
                key_a="wrong-independent-recovery-key-0000",
                key_b=self.fixture.KEY_B, pins=self.fixture.pins,
                owner_observer=lambda: [],
                resume_launcher=lambda _paths: {"returncode": 0},
            )
        self.assertEqual(self.fixture.paths.state.read_bytes(), before)

    def test_owner_appearing_during_full_verification_refuses_before_intent(self):
        self.fixture.stage()
        before = self.fixture.paths.state.read_bytes()
        observations = iter(([], [], [{"pid": 999, "command": "late-heavy-owner"}]))

        def observer():
            return next(observations)

        with self.assertRaisesRegex(recovery.RecoveryError, "appeared"):
            recovery.apply_packet(
                self.fixture.paths, self.fixture.paths.packet,
                key_a=self.fixture.KEY_A, key_b=self.fixture.KEY_B,
                pins=self.fixture.pins, owner_observer=observer,
                resume_launcher=lambda _paths: {"returncode": 0},
            )
        self.assertEqual(self.fixture.paths.state.read_bytes(), before)
        self.assertFalse(self.fixture.paths.intent.exists())

    def test_unavailable_queue_lock_is_refused(self):
        self.fixture.stage()
        code = (
            "import fcntl,sys,time; "
            "f=open(sys.argv[1],'a+'); fcntl.flock(f,fcntl.LOCK_EX); "
            "print('ready',flush=True); time.sleep(30)"
        )
        holder = subprocess.Popen(
            [sys.executable, "-c", code, str(self.fixture.paths.queue_lock)],
            stdout=subprocess.PIPE, text=True,
        )
        try:
            self.assertEqual(holder.stdout.readline().strip(), "ready")
            with self.assertRaisesRegex(recovery.RecoveryError, "lock is unavailable"):
                recovery.apply_packet(
                    self.fixture.paths, self.fixture.paths.packet,
                    key_a=self.fixture.KEY_A, key_b=self.fixture.KEY_B,
                    pins=self.fixture.pins, owner_observer=lambda: [],
                    resume_launcher=lambda _paths: {"returncode": 0},
                )
        finally:
            holder.terminate(); holder.wait(timeout=5)
            assert holder.stdout is not None
            holder.stdout.close()

    def test_stage_refuses_while_recovery_lock_is_held(self):
        self.fixture.paths.recovery_lock.touch()
        code = (
            "import fcntl,sys,time; "
            "f=open(sys.argv[1],'r+'); fcntl.flock(f,fcntl.LOCK_EX); "
            "print('ready',flush=True); time.sleep(30)"
        )
        holder = subprocess.Popen(
            [sys.executable, "-c", code, str(self.fixture.paths.recovery_lock)],
            stdout=subprocess.PIPE, text=True,
        )
        try:
            assert holder.stdout is not None
            self.assertEqual(holder.stdout.readline().strip(), "ready")
            with self.assertRaisesRegex(recovery.RecoveryError, "staging/apply lock"):
                self.fixture.stage()
        finally:
            holder.terminate(); holder.wait(timeout=5)
            assert holder.stdout is not None
            holder.stdout.close()

    def test_malformed_json_is_controlled_refusal(self):
        self.fixture.paths.state.write_text("{bad", encoding="utf-8")
        status = recovery.inspect_recovery(
            self.fixture.paths, self.fixture.pins, owner_observer=lambda: [],
            probe_locks=False,
        )
        self.assertFalse(status["structurally_ready"])
        self.assertTrue(any("invalid JSON" in row for row in status["errors"]))

    def test_apply_changes_only_target_and_writes_receipts_then_resumes(self):
        self.fixture.stage()
        before = json.loads(self.fixture.paths.state.read_text())
        result = recovery.apply_packet(
            self.fixture.paths, self.fixture.paths.packet,
            key_a=self.fixture.KEY_A, key_b=self.fixture.KEY_B,
            pins=self.fixture.pins, owner_observer=lambda: [],
            resume_launcher=lambda _paths: {
                "argv": ["fake-autoresume"], "argv_sha256": "a" * 64,
                "returncode": 0, "detached_supervisor_verified": True,
            },
        )
        after = json.loads(self.fixture.paths.state.read_text())
        self.assertEqual(after["cells"][recovery.TARGET_CELL_ID]["status"], "pending")
        self.assertEqual(after["cells"][recovery.TARGET_CELL_ID]["attempts"], 14)
        self.assertEqual(after["cells"]["completed-control"],
                         before["cells"]["completed-control"])
        self.assertTrue(result["apply"]["other_cells_unchanged"])
        self.assertTrue(self.fixture.paths.intent.is_file())
        self.assertTrue(self.fixture.paths.receipt.is_file())
        self.assertTrue(self.fixture.paths.resume_receipt.is_file())


if __name__ == "__main__":
    unittest.main()
