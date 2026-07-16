from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import plistlib
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

import doctor_v5_forward_recovery as recovery
from tools.condense.tests import doctor_v5_forward_recovery_fault_support as fault_support


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    bridge = "qwen-bridge"

    def __init__(self) -> None:
        (ROOT / "scratch").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "scratch")
        base = Path(self.temp.name)
        ultra = base / "reports/condense/doctor_v5_ultra"
        stage = ultra / "staged_acceleration" / recovery.STAGE_NAME
        tools = base / "tools/condense"
        self.paths = recovery.Paths(
            root=base, ultra=ultra, plan=ultra / "campaign_plan.json",
            state=ultra / "queue_state.json", campaign=ultra / "campaign.json",
            control=ultra / "control.json", registry=ultra / "adapter_registry.json",
            results=ultra / "results", runtime_specs=ultra / "runtime_specs",
            pid_file=ultra / "queue.pid.json", queue_lock=ultra / "queue.lock",
            heavy_lock=base / "reports/cron/studio_heavy.lock",
            launch_agent=base / "Library/LaunchAgents/autoresume.plist",
            active_marker=ultra / "staged_acceleration/active_stack.json",
            overlay=ultra / "staged_acceleration/stacked_admission_overlay.json",
            predecessor_journal=ultra / "staged_acceleration/activation_journal.json",
            canonical_reentry_packet=ultra / "staged_acceleration/pending_runtime_packet.json",
            gc_authority=ultra / "staged_acceleration/gc_runtime_transition_authority.json",
            accelerated_queue=tools / "doctor_v5_ultra_accelerated_queue.py",
            accelerated_autoresume=tools / "doctor_v5_ultra_accelerated_autoresume.py",
            stage_root=stage, packet=stage / "recovery_packet.json",
            journal=stage / "activation_journal.json",
            transaction_lock=stage / "transaction.lock",
        )
        for directory in (
            self.paths.results, self.paths.runtime_specs, stage,
            self.paths.heavy_lock.parent, self.paths.launch_agent.parent, tools,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.paths.queue_lock.touch(); self.paths.heavy_lock.touch()
        self.paths.accelerated_queue.write_text("# queue\n", encoding="utf-8")
        self.paths.accelerated_autoresume.write_text("# autoresume\n", encoding="utf-8")
        self.paths.launch_agent.write_bytes(plistlib.dumps({
            "Label": "test", "ProgramArguments": ["python", "old.py"]
        }))

        self.cells = [
            {"cell_id": "qwen-complete", "model_family": "qwen2.5-dense"},
            {"cell_id": "qwen-failed", "model_family": "qwen2.5-dense"},
            {"cell_id": "qwen-blocked", "model_family": "qwen2.5-dense"},
            {"cell_id": "qwen-clean", "model_family": "qwen2.5-dense"},
            {"cell_id": self.bridge, "model_family": "qwen2.5-dense"},
            {"cell_id": "gpt-failed", "model_family": "gpt-oss"},
        ]
        for index, cell in enumerate(self.cells):
            cell.update({
                "cell_identity_sha256": f"{index + 1:x}" * 64,
                "runtime_spec_path": f"reports/condense/doctor_v5_ultra/runtime_specs/{cell['cell_id']}.json",
            })
            write_json(self.paths.runtime_specs / f"{cell['cell_id']}.json",
                       {"cell_id": cell["cell_id"], "generation": "before"})
        plan = {"schema": "fixture-plan", "cells": self.cells}
        plan["plan_sha256"] = recovery._hash_value(plan)
        write_json(self.paths.plan, plan)
        state = recovery.queue._base_state(plan)
        state["status"] = "drained"; state["control_mode"] = "drain"
        state["last_resource_stop"] = {
            "cell_id": self.bridge,
            "path": str(self.paths.results / self.bridge / "resource_stop.json"),
        }
        state["cells"]["qwen-complete"].update({
            "status": "complete", "attempts": 1, "last_exit_code": 0,
            "result_sha256": "a" * 64, "execution_receipt_sha256": "b" * 64,
        })
        state["cells"]["qwen-failed"].update({
            "status": "pending", "attempts": 2, "last_exit_code": 1,
            "error": "reboot", "request_sha256": "c" * 64,
        })
        state["cells"]["qwen-blocked"].update({
            "status": "blocked-execution", "attempts": 0,
            "blockers": ["failed"], "error": "failed",
        })
        state["cells"][self.bridge].update({
            "status": "blocked-execution", "attempts": 3,
            "blockers": ["failed"], "error": "failed",
        })
        state["cells"]["gpt-failed"].update({
            "status": "blocked-execution", "attempts": 7,
            "blockers": ["failed"], "error": "failed",
        })
        state["state_sha256"] = recovery._hash_value(
            recovery._without(state, "state_sha256")
        )
        write_json(self.paths.state, state)
        self.state = state; self.plan = plan
        self.campaign = {
            "schema": "fixture-campaign", "plan_sha256": plan["plan_sha256"],
            "active_cells": [], "active_children": {}, "counts": {},
        }
        write_json(self.paths.campaign, self.campaign)
        self.control = {"schema": "fixture-control", "mode": "drain",
                        "plan_sha256": plan["plan_sha256"]}
        write_json(self.paths.control, self.control)
        write_json(self.paths.registry, {"schema": "registry", "generation": "before"})
        write_json(self.paths.active_marker, {"generation": "old"})
        write_json(self.paths.overlay, {"generation": "old"})
        write_json(self.paths.predecessor_journal, {"generation": "old"})
        write_json(self.paths.canonical_reentry_packet, {"generation": "old"})
        write_json(self.paths.pid_file, {"pid": 999999})
        self.authority = {"schema": "authority", "authority_sha256": "d" * 64}
        write_json(self.paths.gc_authority, self.authority)

        # Every extant nonterminal Qwen result root must be inventoried.  The
        # GPT result is intentionally excluded by the model-family boundary.
        for cell_id in ("qwen-failed", "qwen-blocked", "qwen-clean", self.bridge):
            root = self.paths.results / cell_id
            root.mkdir(parents=True)
            (root / "outer.json").write_text(cell_id, encoding="utf-8")
        ladder = self.paths.results / self.bridge / "strand_ladder"
        ladder.mkdir()
        (ladder / "request.json").write_text("request", encoding="utf-8")
        (ladder / "checkpoint.json").write_text("checkpoint", encoding="utf-8")
        gpt = self.paths.results / "gpt-failed"
        gpt.mkdir(); (gpt / "request.json").write_text("gpt", encoding="utf-8")

        staged_registry = stage / "prepared/registry.json"
        write_json(staged_registry, {"schema": "registry", "generation": "after"})
        spec_rows = []
        for cell in self.cells:
            if cell["model_family"] != "qwen2.5-dense" \
                    or self.state["cells"][cell["cell_id"]]["status"] in recovery.TERMINAL:
                continue
            target = self.paths.runtime_specs / f"{cell['cell_id']}.json"
            staged = stage / "prepared" / f"{cell['cell_id']}.json"
            write_json(staged, {"cell_id": cell["cell_id"], "generation": "after"})
            spec_rows.append({"cell_id": cell["cell_id"], "target": str(target),
                              "before": recovery._artifact(target),
                              "staged": recovery._artifact(staged)})
        self.prepared = {
            "schema": recovery.reentry.PACKET_SCHEMA,
            "registry": {"target": str(self.paths.registry),
                         "before": recovery._artifact(self.paths.registry),
                         "staged": recovery._artifact(staged_registry)},
            "pending_runtime_specs": spec_rows,
            "terminal_seal": {"count": 1, "rows": [],
                              "rows_sha256": recovery._hash_value([])},
        }
        self.prepared["packet_sha256"] = recovery._hash_value(self.prepared)

    def projection(self, _paths, plan, state, created_at):
        value = {"schema": "fixture-projection", "plan_sha256": plan["plan_sha256"],
                 "state_sha256": state["state_sha256"], "generated_at": created_at,
                 "active_cells": [], "active_children": {}}
        value["campaign_sha256"] = recovery._hash_value(value)
        return value

    def mocks(self):
        anchor = {
            "request": {"path": "r", "sha256": "1" * 64, "bytes": 1},
            "checkpoint": {"path": "c", "sha256": "2" * 64, "bytes": 1},
            "request_id": "fixture", "checkpoint_status": "running",
            "completed_units": ["preflight"], "completed_unit_count": 1,
            "completed_artifact_count": 0,
            "completed_artifact_binding_sha256": recovery._hash_value([]),
        }
        return (
            mock.patch.object(recovery, "_capture_prepared_packet",
                              return_value=copy.deepcopy(self.prepared)),
            mock.patch.object(recovery.reentry, "validate_packet", return_value=[]),
            mock.patch.object(recovery.reentry, "_validate_terminal_seal", return_value=[]),
            mock.patch.object(recovery.gc_transition, "validate_authority",
                              return_value=self.authority),
            mock.patch.object(recovery, "_campaign_projection", side_effect=self.projection),
            mock.patch.object(recovery, "_bridge_anchor", return_value=anchor),
        )

    def close(self) -> None:
        self.temp.cleanup()


class ForwardRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Fixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def _stage(self):
        patches = self.fixture.mocks()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            return recovery.stage(
                self.fixture.paths, production_checks=False,
                bridge_ids=(self.fixture.bridge,),
            )

    def test_reset_predicate_is_qwen_nonterminal_failed_attempt_only(self) -> None:
        rows = recovery._reset_rows(self.fixture.plan, self.fixture.state)
        self.assertEqual([row["cell_id"] for row in rows],
                         ["qwen-blocked", self.fixture.bridge, "qwen-failed"])
        for row in rows:
            self.assertEqual(row["after"], recovery.queue._state_row())
        self.assertNotIn("gpt-failed", [row["cell_id"] for row in rows])
        self.assertNotIn("qwen-clean", [row["cell_id"] for row in rows])
        self.assertNotIn("qwen-complete", [row["cell_id"] for row in rows])

    def test_lifecycle_evidence_cannot_be_erased(self) -> None:
        state = copy.deepcopy(self.fixture.state)
        state["cells"]["qwen-failed"].update({
            "packed_gc_receipt_sha256": "e" * 64,
            "payload_released_at": "2026-07-15T00:00:00+00:00",
            "released_payload_bytes": 1,
        })
        with self.assertRaisesRegex(recovery.ForwardRecoveryError, "lifecycle evidence"):
            recovery._reset_rows(self.fixture.plan, state)

    def test_stage_is_read_only_outside_its_generation(self) -> None:
        # Post-120B runtime specs are intentionally not promoted yet.  Their
        # absence is a first-class CAS binding, not a staging blocker.
        absent_runtime = self.fixture.paths.runtime_specs / "gpt-failed.json"
        absent_runtime.unlink()
        protected = [
            self.fixture.paths.plan, self.fixture.paths.state,
            self.fixture.paths.campaign, self.fixture.paths.control,
            self.fixture.paths.registry, self.fixture.paths.active_marker,
            self.fixture.paths.overlay, self.fixture.paths.predecessor_journal,
            self.fixture.paths.canonical_reentry_packet, self.fixture.paths.pid_file,
            self.fixture.paths.launch_agent,
            *self.fixture.paths.runtime_specs.glob("*.json"),
        ]
        before = {path: digest(path) for path in protected}
        result_before = recovery._tree_manifest(self.fixture.paths.results)
        packet = self._stage()
        self.assertEqual(before, {path: digest(path) for path in protected})
        self.assertEqual(result_before, recovery._tree_manifest(self.fixture.paths.results))
        self.assertEqual(packet["last_resource_stop_before"],
                         self.fixture.state["last_resource_stop"])
        absent_rows = [row for row in packet["runtime_spec_inventory"]
                       if row["cell_id"] == "gpt-failed"]
        self.assertEqual(len(absent_rows), 1)
        self.assertEqual(absent_rows[0]["binding"], {
            "path": str(absent_runtime.absolute()), "exists": False,
        })
        staged = recovery._read_json(Path(packet["staged_state"]["path"]))
        self.assertIsNone(staged["last_resource_stop"])

    def test_artifact_hashing_streams_and_json_limit_precedes_read(self) -> None:
        large = self.fixture.paths.stage_root / "large.bin"
        large.write_bytes(b"a" * (9 * 1024 * 1024 + 17))
        expected = hashlib.sha256(large.read_bytes()).hexdigest()
        with mock.patch.object(recovery, "_stable_file",
                               side_effect=AssertionError("must stream")):
            artifact = recovery._artifact(large)
        self.assertEqual(artifact["sha256"], expected)
        self.assertEqual(artifact["bytes"], large.stat().st_size)

        oversized_json = self.fixture.paths.stage_root / "oversized.json"
        with oversized_json.open("wb") as handle:
            handle.truncate(recovery.MAX_JSON_BYTES + 1)
        with self.assertRaisesRegex(recovery.ForwardRecoveryError, "too large"):
            recovery._read_json(oversized_json)

    def test_content_match_retries_first_ctime_only_churn(self) -> None:
        path = self.fixture.paths.ultra / "content-retry.bin"
        path.write_bytes(b"stable promoted bytes")
        expected = recovery._artifact(path)
        real_fstat = recovery.os.fstat
        calls = 0

        def first_attempt_ctime_churn(fd: int):
            nonlocal calls
            calls += 1
            observed = real_fstat(fd)
            if calls == 2:
                return SimpleNamespace(
                    st_dev=observed.st_dev, st_ino=observed.st_ino,
                    st_size=observed.st_size, st_mtime_ns=observed.st_mtime_ns,
                    st_ctime_ns=observed.st_ctime_ns + 1,
                )
            return observed

        with mock.patch.object(recovery.os, "fstat", side_effect=first_attempt_ctime_churn):
            self.assertTrue(recovery._artifact_content_matches(expected, path))
        self.assertEqual(calls, 4)
        self.assertEqual(digest(path), expected["sha256"])

    def test_content_match_rejects_stable_actual_byte_drift(self) -> None:
        path = self.fixture.paths.ultra / "content-drift.bin"
        path.write_bytes(b"expected")
        expected = recovery._artifact(path)
        path.write_bytes(b"different")
        with mock.patch.object(
            recovery, "_artifact", wraps=recovery._artifact
        ) as artifact:
            self.assertFalse(recovery._artifact_content_matches(expected, path))
        self.assertEqual(artifact.call_count, 1)

    def test_content_match_rejects_persistent_metadata_churn(self) -> None:
        path = self.fixture.paths.ultra / "content-persistent-churn.bin"
        path.write_bytes(b"stable promoted bytes")
        expected = recovery._artifact(path)
        real_fstat = recovery.os.fstat
        calls = 0

        def persistent_ctime_churn(fd: int):
            nonlocal calls
            calls += 1
            observed = real_fstat(fd)
            if calls % 2 == 0:
                return SimpleNamespace(
                    st_dev=observed.st_dev, st_ino=observed.st_ino,
                    st_size=observed.st_size, st_mtime_ns=observed.st_mtime_ns,
                    st_ctime_ns=observed.st_ctime_ns + 1,
                )
            return observed

        with mock.patch.object(recovery.os, "fstat", side_effect=persistent_ctime_churn):
            self.assertFalse(recovery._artifact_content_matches(expected, path))
        self.assertEqual(calls, 2 * recovery.CONTENT_VERIFY_ATTEMPTS)
        self.assertEqual(digest(path), expected["sha256"])

    def test_live_promotion_accepts_one_shot_metadata_churn(self) -> None:
        source = self.fixture.paths.stage_root / "promotion-source.json"
        target = self.fixture.paths.ultra / "promotion-target.json"
        write_json(source, {"generation": "after"})
        write_json(target, {"generation": "before"})
        expected = recovery._artifact(source)
        real_fstat = recovery.os.fstat
        target_fstats = 0

        def first_target_ctime_churn(fd: int):
            nonlocal target_fstats
            observed = real_fstat(fd)
            target_info = target.stat()
            if observed.st_dev == target_info.st_dev \
                    and observed.st_ino == target_info.st_ino:
                target_fstats += 1
                if target_fstats == 2:
                    return SimpleNamespace(
                        st_dev=observed.st_dev, st_ino=observed.st_ino,
                        st_size=observed.st_size,
                        st_mtime_ns=observed.st_mtime_ns,
                        st_ctime_ns=observed.st_ctime_ns + 1,
                    )
            return observed

        journal = {"forward_packet_sha256": "f" * 64}
        with mock.patch.object(
            recovery, "_journal_step", side_effect=lambda _paths, _backup, value,
            **_kwargs: value
        ), mock.patch.object(recovery, "_fault"), mock.patch.object(
            recovery.os, "fstat", side_effect=first_target_ctime_churn
        ), mock.patch.object(
            recovery, "_replace_file", wraps=recovery._replace_file
        ) as replace:
            recovery._live_step(
                self.fixture.paths, self.fixture.paths.stage_root, journal,
                phase="promotion", operation="promote-test",
                action=lambda: recovery._replace_file(source, target),
                verify=lambda: recovery._artifact_content_matches(expected, target),
            )
        self.assertEqual(replace.call_count, 1)
        self.assertEqual(target_fstats, 4)
        self.assertEqual(digest(target), expected["sha256"])

    def test_packet_and_adversarial_audit_pass(self) -> None:
        packet = self._stage()
        patches = self.fixture.mocks()
        with patches[1], patches[2], patches[3], patches[4], patches[5]:
            self.assertEqual(recovery.validate_packet(
                packet, paths=self.fixture.paths, production_checks=False,
                bridge_ids=(self.fixture.bridge,)), [])
            audit = recovery.adversarial_audit(
                packet, paths=self.fixture.paths, production_checks=False,
                bridge_ids=(self.fixture.bridge,))
        self.assertTrue(audit["ok"])
        self.assertEqual(audit["probe_count"], 6)

    def test_generation_identity_and_artifact_paths_are_exact(self) -> None:
        packet = self._stage()
        patches = self.fixture.mocks()
        generation = Path(packet["prepared_runtime_packet"]["path"]).parent
        redirected = generation / "redirected_runtime_packet.json"
        redirected.write_bytes(Path(packet["prepared_runtime_packet"]["path"]).read_bytes())
        altered = copy.deepcopy(packet)
        altered["prepared_runtime_packet"] = recovery._artifact(redirected)
        altered["packet_sha256"] = recovery._hash_value(
            recovery._without(altered, "packet_sha256")
        )
        with patches[1], patches[2], patches[3], patches[4], patches[5]:
            errors = recovery.validate_packet(
                altered, paths=self.fixture.paths, production_checks=False,
                bridge_ids=(self.fixture.bridge,),
            )
        self.assertIn("generation-specific prepared packet artifact changed", errors)

        altered = copy.deepcopy(packet)
        altered["generation_id"] = "0" * 64
        altered["packet_sha256"] = recovery._hash_value(
            recovery._without(altered, "packet_sha256")
        )
        with patches[1], patches[2], patches[3], patches[4], patches[5]:
            errors = recovery.validate_packet(
                altered, paths=self.fixture.paths, production_checks=False,
                bridge_ids=(self.fixture.bridge,),
            )
        self.assertIn("generation identity differs from exact staged inputs", errors)

    def test_broken_symlink_is_never_bound_as_absent(self) -> None:
        target = self.fixture.paths.runtime_specs / "gpt-failed.json"
        target.unlink()
        target.symlink_to(target.parent / "missing-target.json")
        with self.assertRaisesRegex(recovery.ForwardRecoveryError,
                                    "not a regular file"):
            recovery._optional_artifact(target)
        absent = {"path": str(target.absolute()), "exists": False}
        self.assertFalse(recovery._optional_matches(absent, target))

    def test_reentry_capture_never_overwrites_canonical_packet(self) -> None:
        before = self.fixture.paths.canonical_reentry_packet.read_bytes()
        original_packet_path = recovery.reentry.PACKET

        def prepare(*, packet_path):
            write_json(packet_path, self.fixture.prepared)
            return copy.deepcopy(self.fixture.prepared)

        with mock.patch.object(recovery.gc_transition, "validate_authority"), \
                mock.patch.object(recovery.reentry, "prepare", side_effect=prepare):
            observed = recovery._capture_prepared_packet(self.fixture.paths)
        self.assertEqual(observed, self.fixture.prepared)
        self.assertEqual(self.fixture.paths.canonical_reentry_packet.read_bytes(), before)
        self.assertEqual(recovery.reentry.PACKET, original_packet_path)
        self.assertFalse(any(self.fixture.paths.stage_root.glob(".pending-runtime-capture-*")))

    def test_result_split_and_baseline_aware_rollback_guard(self) -> None:
        with mock.patch.object(recovery, "_bridge_anchor", return_value={}):
            rows = recovery._result_inventory(
                self.fixture.plan, self.fixture.state, self.fixture.paths,
                (self.fixture.bridge,),
            )
        packet = {"result_archives": rows}
        backup = self.fixture.paths.stage_root / "rollback-test"
        backup.mkdir()
        moved: list[dict] = []
        recovery._archive_results(packet, self.fixture.paths, backup, moved)
        self.assertEqual(len(moved), 4)
        bridge_live = self.fixture.paths.results / self.fixture.bridge
        self.assertEqual([path.name for path in bridge_live.iterdir()], ["strand_ladder"])
        self.assertEqual(recovery._no_new_output(moved), [])
        (bridge_live / "resume_transition.json").write_text("new", encoding="utf-8")
        self.assertEqual(recovery._no_new_output(moved), [self.fixture.bridge])

        bridge_live.joinpath("resume_transition.json").unlink()
        absent_live = self.fixture.paths.results / "formerly-absent"
        absent = [{"cell_id": "formerly-absent", "live": str(absent_live)}]
        self.assertEqual(recovery._no_new_output(moved, absent), [])
        absent_live.mkdir()
        self.assertEqual(recovery._no_new_output(moved, absent), ["formerly-absent"])

    def test_full_wal_apply_rollback_and_repeat_are_exact(self) -> None:
        packet = self._stage()
        before = fault_support.snapshot_live_surface(self.fixture.paths)
        patches = self.fixture.mocks()
        overlay = {"schema": "fixture-overlay", "overlay_sha256": "e" * 64,
                   "source_bindings": {}}
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with patches[1], patches[2], patches[3], patches[4], patches[5], \
                mock.patch.object(recovery, "_build_staged_overlay",
                                  return_value=overlay), \
                mock.patch.object(recovery, "_start_detached", return_value=424242), \
                mock.patch.object(recovery.subprocess, "run", return_value=completed), \
                mock.patch.object(recovery, "_verified_owner", return_value=None), \
                mock.patch.object(recovery, "_reload_launch_agent"):
            activated = recovery.apply(
                packet_sha256=packet["packet_sha256"],
                plan_sha256=packet["plan_sha256"], paths=self.fixture.paths,
                production_checks=False, bridge_ids=(self.fixture.bridge,),
            )
            self.assertEqual(activated["status"], "active")
            journal = recovery._read_json(self.fixture.paths.journal)
            backup, _, _, wal = recovery._validate_transaction_chain(
                self.fixture.paths, packet, journal
            )
            marker_done = [row["index"] for row in wal
                           if row["operation"] == "promote-marker-last"
                           and row["phase"] == "promotion-done"]
            service = [row["index"] for row in wal
                       if row["phase"].startswith("service")]
            self.assertEqual(len(marker_done), 1)
            self.assertTrue(service and marker_done[0] < min(service))
            self.assertTrue((backup / "activation_receipt.json").is_file())

            rolled = recovery.rollback(paths=self.fixture.paths)
            self.assertEqual(rolled["status"], "rolled-back")
            fault_support.assert_live_surface_equal(self.fixture.paths, before)
            repeated = recovery.rollback(paths=self.fixture.paths)
            self.assertEqual(repeated["status"], "already-rolled-back")
            fault_support.assert_live_surface_equal(self.fixture.paths, before)

    def test_terminal_rolled_back_transaction_can_be_superseded_and_restaged(self) -> None:
        packet = self._stage()
        patches = self.fixture.mocks()
        overlay = {"schema": "fixture-overlay", "overlay_sha256": "e" * 64,
                   "source_bindings": {}}
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with patches[1], patches[2], patches[3], patches[4], patches[5], \
                mock.patch.object(recovery, "_build_staged_overlay",
                                  return_value=overlay), \
                mock.patch.object(recovery, "_start_detached", return_value=424242), \
                mock.patch.object(recovery.subprocess, "run", return_value=completed), \
                mock.patch.object(recovery, "_verified_owner", return_value=None), \
                mock.patch.object(recovery, "_reload_launch_agent"):
            recovery.apply(
                packet_sha256=packet["packet_sha256"],
                plan_sha256=packet["plan_sha256"], paths=self.fixture.paths,
                production_checks=False, bridge_ids=(self.fixture.bridge,),
            )
            recovery.rollback(paths=self.fixture.paths)
            receipt = recovery.supersede(
                reason="fixture terminal rollback retry", paths=self.fixture.paths
            )
        self.assertEqual(receipt["schema"], recovery.SUPERSESSION_SCHEMA)
        self.assertIsNotNone(receipt["archived_journal"])
        self.assertIsNotNone(receipt["terminal_rollback_receipt"])
        self.assertFalse(recovery._lexists(self.fixture.paths.packet))
        self.assertFalse(recovery._lexists(self.fixture.paths.journal))
        archived_packet = Path(receipt["archived_packet"]["path"])
        archived_journal = Path(receipt["archived_journal"]["path"])
        self.assertTrue(archived_packet.is_file())
        self.assertTrue(archived_journal.is_file())
        replacement = self._stage()
        self.assertEqual(replacement["schema"], recovery.SCHEMA)

    def test_terminal_supersession_recovers_after_journal_move_crash(self) -> None:
        packet = self._stage()
        patches = self.fixture.mocks()
        overlay = {"schema": "fixture-overlay", "overlay_sha256": "e" * 64,
                   "source_bindings": {}}
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with patches[1], patches[2], patches[3], patches[4], patches[5], \
                mock.patch.object(recovery, "_build_staged_overlay",
                                  return_value=overlay), \
                mock.patch.object(recovery, "_start_detached", return_value=424242), \
                mock.patch.object(recovery.subprocess, "run", return_value=completed), \
                mock.patch.object(recovery, "_verified_owner", return_value=None), \
                mock.patch.object(recovery, "_reload_launch_agent"):
            recovery.apply(
                packet_sha256=packet["packet_sha256"],
                plan_sha256=packet["plan_sha256"], paths=self.fixture.paths,
                production_checks=False, bridge_ids=(self.fixture.bridge,),
            )
            recovery.rollback(paths=self.fixture.paths)

        reason = "fixture terminal rollback crash retry"
        live_before = fault_support.snapshot_live_surface(self.fixture.paths)

        def crash_supersede() -> None:
            injector = fault_support.FaultInjector(
                "after:supersede:journal-move", mode="hard"
            )
            with mock.patch.object(recovery, "_fault", injector):
                recovery.supersede(reason=reason, paths=self.fixture.paths)

        outcome = fault_support.run_forked(crash_supersede)
        fault_support.assert_hard_exit(outcome)
        with self.assertRaisesRegex(recovery.ForwardRecoveryError, "supersession"):
            recovery.stage(
                self.fixture.paths, production_checks=False,
                bridge_ids=(self.fixture.bridge,),
            )
        with self.assertRaisesRegex(recovery.ForwardRecoveryError, "supersession"):
            recovery.apply(
                packet_sha256=packet["packet_sha256"],
                plan_sha256=packet["plan_sha256"], paths=self.fixture.paths,
                production_checks=False, bridge_ids=(self.fixture.bridge,),
            )
        fault_support.assert_live_surface_equal(
            self.fixture.paths, live_before,
            label="unfinished terminal journal supersession barrier",
        )
        receipt = recovery.supersede(reason=reason, paths=self.fixture.paths)
        self.assertEqual(receipt["status"], "superseded")
        self.assertIsNotNone(receipt["archived_journal"])
        self.assertFalse(recovery._lexists(self.fixture.paths.packet))
        self.assertFalse(recovery._lexists(self.fixture.paths.journal))

    def test_hard_crash_after_result_rename_recovers_exact_prestate(self) -> None:
        packet = self._stage()
        before = fault_support.snapshot_live_surface(self.fixture.paths)
        patches = self.fixture.mocks()
        overlay = {"schema": "fixture-overlay", "overlay_sha256": "e" * 64,
                   "source_bindings": {}}
        with patches[1], patches[2], patches[3], patches[4], patches[5], \
                mock.patch.object(recovery, "_build_staged_overlay",
                                  return_value=overlay), \
                mock.patch.object(recovery, "_verified_owner", return_value=None), \
                mock.patch.object(recovery, "_reload_launch_agent"):
            def crash_apply() -> None:
                injector = fault_support.FaultInjector(
                    "module:after:promotion:archive-result:qwen-blocked", mode="hard"
                )
                with fault_support.MutationBoundaryHarness(
                        recovery, injector, patch_replace=False, patch_atomic=False):
                    recovery.apply(
                        packet_sha256=packet["packet_sha256"],
                        plan_sha256=packet["plan_sha256"], paths=self.fixture.paths,
                        production_checks=False, bridge_ids=(self.fixture.bridge,),
                    )

            outcome = fault_support.run_forked(crash_apply)
            fault_support.assert_hard_exit(outcome)
            recovered = recovery.recover(paths=self.fixture.paths)
            self.assertEqual(recovered["status"], "rolled-back")
            fault_support.assert_live_surface_equal(self.fixture.paths, before)
            repeated = recovery.recover(paths=self.fixture.paths)
            self.assertEqual(repeated["status"], "already-rolled-back")
            fault_support.assert_live_surface_equal(self.fixture.paths, before)

    def test_wal_tail_overrides_stale_forward_journal_direction(self) -> None:
        packet = self._stage()
        before = fault_support.snapshot_live_surface(self.fixture.paths)
        patches = self.fixture.mocks()
        overlay = {"schema": "fixture-overlay", "overlay_sha256": "e" * 64,
                   "source_bindings": {}}
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with patches[1], patches[2], patches[3], patches[4], patches[5], \
                mock.patch.object(recovery, "_build_staged_overlay",
                                  return_value=overlay), \
                mock.patch.object(recovery, "_start_detached", return_value=424242), \
                mock.patch.object(recovery.subprocess, "run", return_value=completed), \
                mock.patch.object(recovery, "_verified_owner", return_value=None), \
                mock.patch.object(recovery, "_reload_launch_agent"):
            recovery.apply(
                packet_sha256=packet["packet_sha256"],
                plan_sha256=packet["plan_sha256"], paths=self.fixture.paths,
                production_checks=False, bridge_ids=(self.fixture.bridge,),
            )
            stale = recovery._read_json(self.fixture.paths.journal)
            backup = Path(stale["backup_root"])
            recovery._append_wal(
                backup, packet["packet_sha256"], phase="rollback-decision",
                operation="rollback-decision",
                details={"forward_only": True, "live_marker_cleared": True},
            )
            self.assertEqual(
                recovery._read_json(self.fixture.paths.journal)["status"], "active"
            )
            recovered = recovery.recover(paths=self.fixture.paths)
            self.assertEqual(recovered["status"], "rolled-back")
            fault_support.assert_live_surface_equal(self.fixture.paths, before)

    def test_activation_requires_both_exact_keys_before_mutation(self) -> None:
        packet = self._stage()
        before = digest(self.fixture.paths.state)
        with self.assertRaisesRegex(recovery.ForwardRecoveryError, "activation keys"):
            recovery.apply(
                packet_sha256="0" * 64, plan_sha256=packet["plan_sha256"],
                paths=self.fixture.paths, production_checks=False,
                bridge_ids=(self.fixture.bridge,),
            )
        self.assertEqual(digest(self.fixture.paths.state), before)

    def test_apply_rechecks_supersession_barrier_under_all_leases(self) -> None:
        packet = self._stage()
        before = fault_support.snapshot_live_surface(self.fixture.paths)
        patches = self.fixture.mocks()
        with patches[1], patches[2], patches[3], patches[4], patches[5], \
                mock.patch.object(
                    recovery, "_supersession_barrier",
                    side_effect=[None, recovery.ForwardRecoveryError(
                        "apply refused while packet supersession is unfinished"
                    )]):
            with self.assertRaisesRegex(recovery.ForwardRecoveryError, "supersession"):
                recovery.apply(
                    packet_sha256=packet["packet_sha256"],
                    plan_sha256=packet["plan_sha256"], paths=self.fixture.paths,
                    production_checks=False, bridge_ids=(self.fixture.bridge,),
                )
        fault_support.assert_live_surface_equal(
            self.fixture.paths, before, label="under-lock supersession barrier"
        )
        self.assertFalse(recovery._lexists(self.fixture.paths.journal))

    def test_supersession_converges_across_every_durable_cut(self) -> None:
        labels = (
            "after:supersede:intent",
            "after:supersede:locator",
            "after:supersede:move",
            "after:supersede:receipt",
        )
        for label in labels:
            with self.subTest(label=label):
                case = Fixture()
                try:
                    patches = case.mocks()
                    with patches[0], patches[1], patches[2], patches[3], \
                            patches[4], patches[5]:
                        packet = recovery.stage(
                            case.paths, production_checks=False,
                            bridge_ids=(case.bridge,),
                        )
                    packet_file_sha256 = digest(case.paths.packet)
                    live_before = fault_support.snapshot_live_surface(case.paths)
                    reason = f"fixture crash at {label}"

                    def crash_supersede() -> None:
                        injector = fault_support.FaultInjector(label, mode="hard")
                        with mock.patch.object(recovery, "_fault", injector):
                            recovery.supersede(reason=reason, paths=case.paths)

                    outcome = fault_support.run_forked(crash_supersede)
                    fault_support.assert_hard_exit(outcome)
                    if label != "after:supersede:receipt":
                        observed_status = recovery.status(case.paths)
                        self.assertFalse(observed_status["activation_permitted_now"])
                        self.assertIn(
                            "supersession", observed_status["supersession_blocker"]
                        )
                        with self.assertRaisesRegex(
                                recovery.ForwardRecoveryError, "supersession"):
                            recovery.stage(
                                case.paths, production_checks=False,
                                bridge_ids=(case.bridge,),
                            )
                        with self.assertRaisesRegex(
                                recovery.ForwardRecoveryError, "supersession"):
                            recovery.apply(
                                packet_sha256=packet["packet_sha256"],
                                plan_sha256=packet["plan_sha256"], paths=case.paths,
                                production_checks=False, bridge_ids=(case.bridge,),
                            )
                    fault_support.assert_live_surface_equal(
                        case.paths, live_before,
                        label=f"unfinished supersession barrier at {label}",
                    )
                    receipt = recovery.supersede(reason=reason, paths=case.paths)
                    self.assertEqual(receipt["status"], "superseded")
                    self.assertFalse(recovery._lexists(case.paths.packet))
                    archived = Path(receipt["archived_packet"]["path"])
                    self.assertEqual(digest(archived), packet_file_sha256)
                    self.assertEqual(
                        recovery._read_json(archived)["packet_sha256"],
                        packet["packet_sha256"],
                    )
                    repeated = recovery.supersede(reason=reason, paths=case.paths)
                    self.assertEqual(repeated, receipt)
                finally:
                    case.close()

    def test_supersession_refuses_archive_parent_symlink_before_escape(self) -> None:
        self._stage()
        outside = self.fixture.paths.root / "outside-supersession"
        outside.mkdir()
        redirected = self.fixture.paths.stage_root / "superseded"
        fault_support.replace_with_symlink(
            redirected, outside, fixture_root=self.fixture.paths.root
        )
        packet_before = digest(self.fixture.paths.packet)
        with self.assertRaisesRegex(
                recovery.ForwardRecoveryError, "directory.*unsafe"):
            recovery.supersede(reason="fixture path attack", paths=self.fixture.paths)
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual(digest(self.fixture.paths.packet), packet_before)
        self.assertFalse(recovery._lexists(
            self.fixture.paths.stage_root / "supersession_active.json"
        ))

    def test_supersession_refuses_to_orphan_unfinished_locator(self) -> None:
        self._stage()
        reason = "fixture unfinished transaction"
        injector = fault_support.FaultInjector(
            "after:supersede:locator", mode="soft"
        )
        with mock.patch.object(recovery, "_fault", injector), \
                self.assertRaises(fault_support.InjectedFault):
            recovery.supersede(reason=reason, paths=self.fixture.paths)
        write_json(self.fixture.paths.packet, {"different": "packet"})
        packet_before = digest(self.fixture.paths.packet)
        transaction_before = fault_support.snapshot_tree(
            self.fixture.paths.stage_root
        )
        with self.assertRaisesRegex(
                recovery.ForwardRecoveryError, "different supersession.*unfinished"):
            recovery.supersede(reason=reason, paths=self.fixture.paths)
        self.assertEqual(digest(self.fixture.paths.packet), packet_before)
        fault_support.assert_snapshot_equal(
            transaction_before,
            fault_support.snapshot_tree(self.fixture.paths.stage_root),
            label="unfinished supersession refusal",
        )

    def test_supersession_recovery_refuses_resealed_symlink_ancestry(self) -> None:
        self._stage()
        reason = "fixture resealed ancestry"
        injector = fault_support.FaultInjector(
            "after:supersede:move", mode="soft"
        )
        with mock.patch.object(recovery, "_fault", injector), \
                self.assertRaises(fault_support.InjectedFault):
            recovery.supersede(reason=reason, paths=self.fixture.paths)
        superseded = self.fixture.paths.stage_root / "superseded"
        relocated = self.fixture.paths.root / "relocated-superseded"
        superseded.rename(relocated)
        superseded.symlink_to(relocated, target_is_directory=True)
        locator_path = self.fixture.paths.stage_root / "supersession_active.json"
        locator = recovery._read_json(locator_path)
        intent_path = (
            superseded / locator["original_packet_sha256"]
            / "supersession_intent.json"
        )
        locator["intent"] = recovery._artifact(intent_path)
        locator["locator_sha256"] = recovery._hash_value(
            recovery._without(locator, "locator_sha256")
        )
        write_json(locator_path, locator)
        receipt_path = (
            relocated / locator["original_packet_sha256"]
            / "supersession_receipt.json"
        )
        with self.assertRaisesRegex(
                recovery.ForwardRecoveryError, "not lexical|directory is unsafe"):
            recovery.supersede(reason=reason, paths=self.fixture.paths)
        self.assertFalse(recovery._lexists(receipt_path))

    def test_supersession_durably_seals_each_new_directory_entry(self) -> None:
        packet = self._stage()
        superseded = self.fixture.paths.stage_root / "superseded"
        with mock.patch.object(
                recovery, "_fsync_dir", wraps=recovery._fsync_dir
        ) as fsync_dir:
            receipt = recovery.supersede(
                reason="fixture directory durability", paths=self.fixture.paths
            )
        flushed = [Path(call.args[0]).resolve()
                   for call in fsync_dir.call_args_list]
        self.assertIn(self.fixture.paths.stage_root.resolve(), flushed)
        self.assertIn(superseded.resolve(), flushed)
        self.assertEqual(
            recovery._read_json(Path(receipt["archived_packet"]["path"]))
            ["packet_sha256"],
            packet["packet_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
