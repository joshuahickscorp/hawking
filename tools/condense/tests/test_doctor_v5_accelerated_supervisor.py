#!/usr/bin/env python3.12
"""Focused no-model tests for accelerated supervisor re-entry and ramping."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


CONDENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONDENSE))
import doctor_v5_ultra_accelerated_queue as accelerated
import doctor_v5_ultra_accelerated_autoresume as autoresume


class AcceleratedSupervisorTests(unittest.TestCase):
    def test_acceleration_entrypoint_sources_are_hash_pinned(self) -> None:
        accelerated._verify_static_bindings()
        queue = {"path": str(autoresume.ACCEL_QUEUE.resolve()),
                 "sha256": autoresume._hash_regular(autoresume.ACCEL_QUEUE),
                 "bytes": autoresume.ACCEL_QUEUE.stat().st_size}
        resume = {"path": str(Path(autoresume.__file__).resolve()),
                  "sha256": autoresume._hash_regular(Path(autoresume.__file__)),
                  "bytes": Path(autoresume.__file__).stat().st_size}
        self.assertTrue(autoresume._artifact_matches(queue, autoresume.ACCEL_QUEUE))
        self.assertTrue(autoresume._artifact_matches(resume, Path(autoresume.__file__)))

    def test_only_quiescent_checkpoint_uses_one_time_preflight(self) -> None:
        self.assertTrue(accelerated._requires_one_time_preflight(
            {"mode": "drain"}, {"status": "drained"}
        ))
        self.assertTrue(accelerated._requires_one_time_preflight(
            {"mode": "pause"}, {"status": "paused"}
        ))
        self.assertFalse(accelerated._requires_one_time_preflight(
            {"mode": "run"}, {"status": "running-cell"}
        ))
        self.assertFalse(accelerated._requires_one_time_preflight(
            {"mode": "run"}, {"status": "waiting-resources"}
        ))

    def test_resume_preflight_does_not_pin_mutable_snapshot_hashes(self) -> None:
        overlay = {
            "source_bindings": {
                "plan_sha256": "p", "observer_source": {"fixture": True},
            },
            "immutable_terminal_seal": {"rows": [],
                                        "rows_sha256": accelerated.stacked._hash_value([])},
        }
        saved = {
            "validate_overlay": accelerated.stacked.validate_overlay,
            "read_json": accelerated.stacked._read_json,
            "validate_live": accelerated.stacked._validate_live_documents,
            "reference": accelerated.stacked._reference_matches,
            "observer": accelerated.stacked._observer_structure_errors,
            "terminal": accelerated._terminal_subset_errors,
            "generation": accelerated._active_generation_errors,
            "snapshot": accelerated.stacked.ram_scheduler.resource_snapshot,
            "thermal": accelerated.stacked._thermal_probe,
        }
        try:
            accelerated.stacked.validate_overlay = lambda _overlay: []
            accelerated.stacked._reference_matches = lambda _reference: True
            accelerated.stacked._observer_structure_errors = lambda: []
            accelerated.stacked._read_json = lambda path: (
                {"plan_sha256": "p"} if path == accelerated.stacked.PLAN else {}
            )
            accelerated.stacked._validate_live_documents = lambda *args: None
            accelerated._terminal_subset_errors = lambda *args: []
            accelerated._active_generation_errors = lambda *args: []
            accelerated.stacked.ram_scheduler.resource_snapshot = lambda _root: {
                "pressure_level": 1, "swap_used_mb": 0,
                "power_source": "Now drawing from 'AC Power'",
            }
            accelerated.stacked._thermal_probe = lambda: {"ok": True}
            result = accelerated._resume_preflight(overlay)
        finally:
            accelerated.stacked.validate_overlay = saved["validate_overlay"]
            accelerated.stacked._read_json = saved["read_json"]
            accelerated.stacked._validate_live_documents = saved["validate_live"]
            accelerated.stacked._reference_matches = saved["reference"]
            accelerated.stacked._observer_structure_errors = saved["observer"]
            accelerated._terminal_subset_errors = saved["terminal"]
            accelerated._active_generation_errors = saved["generation"]
            accelerated.stacked.ram_scheduler.resource_snapshot = saved["snapshot"]
            accelerated.stacked._thermal_probe = saved["thermal"]
        self.assertTrue(result["ready"])
        self.assertEqual("resume-safe", result["mode"])

    def test_resume_preflight_rejects_structurally_invalid_live_observer(self) -> None:
        overlay = {
            "source_bindings": {
                "plan_sha256": "p", "observer_source": {"fixture": True},
            },
            "immutable_terminal_seal": {"rows": [],
                                        "rows_sha256": accelerated.stacked._hash_value([])},
        }
        saved = {
            "validate_overlay": accelerated.stacked.validate_overlay,
            "read_json": accelerated.stacked._read_json,
            "validate_live": accelerated.stacked._validate_live_documents,
            "reference": accelerated.stacked._reference_matches,
            "observer": accelerated.stacked._observer_structure_errors,
            "terminal": accelerated._terminal_subset_errors,
            "generation": accelerated._active_generation_errors,
            "snapshot": accelerated.stacked.ram_scheduler.resource_snapshot,
            "thermal": accelerated.stacked._thermal_probe,
        }
        try:
            accelerated.stacked.validate_overlay = lambda _overlay: []
            accelerated.stacked._read_json = lambda path: (
                {"plan_sha256": "p"} if path == accelerated.stacked.PLAN else {}
            )
            accelerated.stacked._validate_live_documents = lambda *args: None
            accelerated.stacked._reference_matches = lambda _reference: True
            accelerated.stacked._observer_structure_errors = lambda: [
                "live observer self-hash is invalid"
            ]
            accelerated._terminal_subset_errors = lambda *args: []
            accelerated._active_generation_errors = lambda *args: []
            accelerated.stacked.ram_scheduler.resource_snapshot = lambda _root: {
                "pressure_level": 1, "swap_used_mb": 0,
                "power_source": "Now drawing from 'AC Power'",
            }
            accelerated.stacked._thermal_probe = lambda: {"ok": True}
            result = accelerated._resume_preflight(overlay)
        finally:
            accelerated.stacked.validate_overlay = saved["validate_overlay"]
            accelerated.stacked._read_json = saved["read_json"]
            accelerated.stacked._validate_live_documents = saved["validate_live"]
            accelerated.stacked._reference_matches = saved["reference"]
            accelerated.stacked._observer_structure_errors = saved["observer"]
            accelerated._terminal_subset_errors = saved["terminal"]
            accelerated._active_generation_errors = saved["generation"]
            accelerated.stacked.ram_scheduler.resource_snapshot = saved["snapshot"]
            accelerated.stacked._thermal_probe = saved["thermal"]
        self.assertFalse(result["ready"])
        self.assertIn("live observer self-hash is invalid", result["blockers"])

    def test_empty_pool_ramps_only_one_block_parallel_lane_per_scan(self) -> None:
        heads = [self._execution(f"cell-{index}") for index in range(4)]
        original_scan = accelerated._ORIGINAL_SCAN_HEADS
        original_reservation = accelerated._reservation_from_evidence
        original_evidence = accelerated._evidence
        original_global_cpu = accelerated._global_host_cpu_cores
        accelerated._CPU_SAMPLES.clear()
        accelerated._SCAN_RESERVATIONS.clear()
        try:
            accelerated._ORIGINAL_SCAN_HEADS = lambda plan, state: (list(heads), {})
            accelerated._reservation_from_evidence = (
                lambda cell, evidence: 4_000_000_000
            )
            accelerated._evidence = lambda: {}
            accelerated._global_host_cpu_cores = lambda: 0.0
            selected, _ = accelerated._accelerated_scan_heads(
                {"cells": [row["cell"] for row in heads]}, {"active_children": {}}
            )
        finally:
            accelerated._ORIGINAL_SCAN_HEADS = original_scan
            accelerated._reservation_from_evidence = original_reservation
            accelerated._evidence = original_evidence
            accelerated._global_host_cpu_cores = original_global_cpu
            accelerated._CPU_SAMPLES.clear()
            accelerated._SCAN_RESERVATIONS.clear()
        self.assertEqual(1, len(selected))

    def test_core_token_blocks_second_launch_when_headroom_is_under_twenty(self) -> None:
        heads = [self._execution("candidate")]
        original_scan = accelerated._ORIGINAL_SCAN_HEADS
        original_reservation = accelerated._reservation_from_evidence
        original_evidence = accelerated._evidence
        original_global_cpu = accelerated._global_host_cpu_cores
        accelerated._CPU_SAMPLES.clear()
        accelerated._SCAN_RESERVATIONS.clear()
        try:
            accelerated._ORIGINAL_SCAN_HEADS = lambda plan, state: (list(heads), {})
            accelerated._reservation_from_evidence = (
                lambda cell, evidence: 4_000_000_000
            )
            accelerated._evidence = lambda: {}
            accelerated._global_host_cpu_cores = lambda: 7.0
            selected, _ = accelerated._accelerated_scan_heads(
                {"cells": [row["cell"] for row in heads]},
                {"active_children": {"live": {"pgid": 1,
                                                "reserved_bytes": 4_000_000_000}}},
            )
        finally:
            accelerated._ORIGINAL_SCAN_HEADS = original_scan
            accelerated._reservation_from_evidence = original_reservation
            accelerated._evidence = original_evidence
            accelerated._global_host_cpu_cores = original_global_cpu
            accelerated._CPU_SAMPLES.clear()
            accelerated._SCAN_RESERVATIONS.clear()
        self.assertEqual([], selected)

    def test_residency_evidence_is_loaded_once_and_reused_for_launch_charge(self) -> None:
        heads = [self._execution(f"cell-{index}") for index in range(6)]
        original_scan = accelerated._ORIGINAL_SCAN_HEADS
        original_reservation = accelerated._reservation_from_evidence
        original_evidence = accelerated._evidence
        original_global_cpu = accelerated._global_host_cpu_cores
        evidence_calls = 0
        reservation_calls: list[str] = []

        def evidence() -> dict[str, dict[str, int]]:
            nonlocal evidence_calls
            evidence_calls += 1
            return {"0.5B": {"samples": evidence_calls, "peak_bytes": 1}}

        def reserve(cell: dict, observed: dict) -> int:
            self.assertEqual(evidence_calls,
                             observed["0.5B"]["samples"])
            reservation_calls.append(cell["cell_id"])
            return 4_000_000_000

        accelerated._CPU_SAMPLES.clear()
        accelerated._SCAN_RESERVATIONS.clear()
        try:
            accelerated._ORIGINAL_SCAN_HEADS = lambda plan, state: (list(heads), {})
            accelerated._reservation_from_evidence = reserve
            accelerated._evidence = evidence
            accelerated._global_host_cpu_cores = lambda: 0.0
            selected, _ = accelerated._accelerated_scan_heads(
                {"cells": [row["cell"] for row in heads]}, {"active_children": {}}
            )
            self.assertEqual(1, evidence_calls)
            self.assertEqual(len(heads), len(reservation_calls))
            # This is the base queue's post-scan launch-charge call. It must use
            # the exact per-scan integer and never re-open the 12.5MB evidence log.
            self.assertEqual(4_000_000_000,
                             accelerated._accelerated_reservation(selected[0]["cell"]))
            self.assertEqual(1, evidence_calls)
            self.assertEqual(len(heads), len(reservation_calls))
        finally:
            accelerated._ORIGINAL_SCAN_HEADS = original_scan
            accelerated._reservation_from_evidence = original_reservation
            accelerated._evidence = original_evidence
            accelerated._global_host_cpu_cores = original_global_cpu
            accelerated._CPU_SAMPLES.clear()
            accelerated._SCAN_RESERVATIONS.clear()

    def test_global_twenty_four_core_owner_blocks_an_empty_doctor_pool(self) -> None:
        heads = [self._execution("candidate")]
        original_scan = accelerated._ORIGINAL_SCAN_HEADS
        original_global_cpu = accelerated._global_host_cpu_cores
        original_evidence = accelerated._evidence
        evidence_calls = 0

        def evidence() -> dict:
            nonlocal evidence_calls
            evidence_calls += 1
            return {}

        accelerated._CPU_SAMPLES.clear()
        accelerated._SCAN_RESERVATIONS.clear()
        try:
            accelerated._ORIGINAL_SCAN_HEADS = lambda plan, state: (list(heads), {})
            accelerated._global_host_cpu_cores = lambda: 24.0
            accelerated._evidence = evidence
            selected, _ = accelerated._accelerated_scan_heads(
                {"cells": [heads[0]["cell"]]}, {"active_children": {}}
            )
        finally:
            accelerated._ORIGINAL_SCAN_HEADS = original_scan
            accelerated._global_host_cpu_cores = original_global_cpu
            accelerated._evidence = original_evidence
            accelerated._CPU_SAMPLES.clear()
            accelerated._SCAN_RESERVATIONS.clear()
        self.assertEqual([], selected)
        self.assertEqual(0, evidence_calls)

    def test_exact_adaptive_mop_contract_allows_one_launch_only_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "mop"
            script = repo / "scripts/mop_generation1_campaign.py"
            program_path = repo / "configs/campaign/program.json"
            config_path = repo / "configs/experiment/generation1_context_routing.json"
            script.parent.mkdir(parents=True)
            program_path.parent.mkdir(parents=True)
            config_path.parent.mkdir(parents=True)
            script.write_text("# bound owner\n", encoding="utf-8")
            plan_sha = "a" * 64
            config = {
                "schema": accelerated.COOPERATIVE_MOP_CONFIG_SCHEMA,
                "activation_allowed": False,
                "adaptive_resources": {
                    "idle_workers": accelerated.COOPERATIVE_MOP_IDLE_WORKERS,
                    "hawking_workers": accelerated.COOPERATIVE_MOP_HAWKING_WORKERS,
                    "hawking_queue_state": str(accelerated._BASE.STATE.resolve()),
                    "hawking_plan_sha256": plan_sha,
                },
            }
            config_raw = json.dumps(config, sort_keys=True).encode()
            config_artifact = {
                "path": str(config_path),
                "sha256": hashlib.sha256(config_raw).hexdigest(),
                "bytes": len(config_raw),
            }
            program = {
                "schema": accelerated.COOPERATIVE_MOP_PROGRAM_SCHEMA,
                "authorities": [{
                    "path": "configs/experiment/generation1_context_routing.json",
                    "sha256": config_artifact["sha256"],
                }],
            }
            program["program_sha256"] = accelerated._hash_value(program)
            program_raw = json.dumps(program, sort_keys=True).encode()
            program_artifact = {
                "path": str(program_path),
                "sha256": hashlib.sha256(program_raw).hexdigest(),
                "bytes": len(program_raw),
            }
            rows = [
                {"pid": 100, "ppid": 1,
                 "command": (f"{sys.executable} {script} run --program "
                             f"{program_path} --execute")},
                {"pid": 101, "ppid": 100,
                 "command": (f"{sys.executable} {repo / 'scripts/run_shard.py'} "
                             "--idle-workers 25 --hawking-workers 6")},
            ]
            original_rows = accelerated._command_rows
            original_json = accelerated._stable_external_json

            def stable(path: Path) -> tuple[dict, dict]:
                if path.resolve() == program_path.resolve():
                    return program, program_artifact
                if path.resolve() == config_path.resolve():
                    return config, config_artifact
                raise AssertionError(path)

            try:
                accelerated._command_rows = lambda: rows
                accelerated._stable_external_json = stable
                decision = accelerated._cooperative_mop_handoff(
                    {"plan_sha256": plan_sha},
                    {"plan_sha256": plan_sha, "state_sha256": "b" * 64,
                     "active_cells": [], "active_children": {}},
                    charged_global_cpu_cores=25.0,
                )
                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertEqual(1, decision["launch_limit"])
                self.assertFalse(decision["external_signal_or_mutation_permitted"])
                self.assertFalse(decision["stop_or_shed_authority"])
                config["adaptive_resources"]["hawking_workers"] = 7
                self.assertIsNone(accelerated._cooperative_mop_handoff(
                    {"plan_sha256": plan_sha},
                    {"plan_sha256": plan_sha, "state_sha256": "b" * 64,
                     "active_cells": [], "active_children": {}},
                    charged_global_cpu_cores=25.0,
                ))
                config["adaptive_resources"]["hawking_workers"] = 6
                rows.append({
                    "pid": 102, "ppid": 100,
                    "command": (f"{sys.executable} "
                                f"{repo / 'scripts/run_shard.py'} \""),
                })
                self.assertIsNone(accelerated._cooperative_mop_handoff(
                    {"plan_sha256": plan_sha},
                    {"plan_sha256": plan_sha, "state_sha256": "b" * 64,
                     "active_cells": [], "active_children": {}},
                    charged_global_cpu_cores=25.0,
                ))
            finally:
                accelerated._command_rows = original_rows
                accelerated._stable_external_json = original_json

    def test_exact_labeled_caffeinate_mop_topology_and_lookalikes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "mop"
            script = repo / "scripts/mop_generation1_campaign.py"
            shard_script = repo / "scripts/generation1_context_routing/run_shard.py"
            python = repo / ".venv/bin/python"
            program_path = repo / "configs/campaign/program.json"
            config_path = repo / "configs/experiment/generation1_context_routing.json"
            for path in (script, shard_script, python, program_path, config_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            script.write_text("# labeled owner\n", encoding="utf-8")
            shard_script.write_text("# sealed shard\n", encoding="utf-8")
            python.write_text("# exact interpreter path\n", encoding="utf-8")
            plan_sha = "c" * 64
            config = {
                "schema": accelerated.COOPERATIVE_MOP_CONFIG_SCHEMA,
                "activation_allowed": False,
                "adaptive_resources": {
                    "idle_workers": accelerated.COOPERATIVE_MOP_IDLE_WORKERS,
                    "hawking_workers": accelerated.COOPERATIVE_MOP_HAWKING_WORKERS,
                    "hawking_queue_state": str(accelerated._BASE.STATE.resolve()),
                    "hawking_plan_sha256": plan_sha,
                },
            }
            config_raw = json.dumps(config, sort_keys=True).encode()
            config_path.write_bytes(config_raw)
            config_artifact = {
                "path": str(config_path),
                "sha256": hashlib.sha256(config_raw).hexdigest(),
                "bytes": len(config_raw),
            }
            program_id = "generation1-c2-context-routing-v1-adaptive25-labeled"
            capsules = []
            for shard_index in range(accelerated.COOPERATIVE_MOP_SHARD_COUNT):
                capsule = {
                    "schema": accelerated.COOPERATIVE_MOP_CAPSULE_SCHEMA,
                    "id": f"g1_c2_context_routing_shard_{shard_index:02d}",
                    "kind": "corpus",
                    "cwd": ".",
                    "command": [
                        ".venv/bin/python",
                        "scripts/generation1_context_routing/run_shard.py",
                        "--config",
                        "configs/experiment/generation1_context_routing.json",
                        "--shard-index", str(shard_index),
                        "--idle-workers", "25",
                        "--hawking-workers", "6",
                    ],
                    "resources": {
                        "cpu_cores": 25,
                        "lane": "cpu",
                        "accelerator": "none",
                        "process_marker": "run_shard.py",
                    },
                }
                capsule["capsule_sha256"] = accelerated._hash_value(capsule)
                capsules.append(capsule)
            program = {
                "schema": accelerated.COOPERATIVE_MOP_PROGRAM_SCHEMA,
                "program_id": program_id,
                "authorities": [{
                    "path": "configs/experiment/generation1_context_routing.json",
                    "sha256": config_artifact["sha256"],
                }],
                "capsules": capsules,
            }
            program["program_sha256"] = accelerated._hash_value(program)
            program_raw = json.dumps(program, sort_keys=True).encode()
            program_path.write_bytes(program_raw)
            program_artifact = {
                "path": str(program_path),
                "sha256": hashlib.sha256(program_raw).hexdigest(),
                "bytes": len(program_raw),
            }
            supervisor = {
                "pid": 100, "ppid": 1,
                "command": f"mop-supervisor:{program_id}",
            }
            launcher = {
                "pid": 101, "ppid": 100,
                "command": (f"/usr/bin/caffeinate -ims {python} {script} run "
                            f"--program {program_path} --execute"),
            }
            capsule = {
                "pid": 102, "ppid": 100,
                "command": "mop-capsule:g1_c2_context_routing_shard_03",
            }
            workers = [
                {"pid": 200 + index, "ppid": 102,
                 "command": "mop-c2-s03-worker"}
                for index in range(accelerated.COOPERATIVE_MOP_IDLE_WORKERS)
            ]
            rows = [supervisor, launcher, capsule, *workers]
            original_rows = accelerated._command_rows
            original_json = accelerated._stable_external_json

            def stable(path: Path) -> tuple[dict, dict]:
                if path.resolve() == program_path.resolve():
                    return program, program_artifact
                if path.resolve() == config_path.resolve():
                    return config, config_artifact
                raise AssertionError(path)

            def decision(candidate_rows: list[dict]) -> dict | None:
                accelerated._command_rows = lambda: candidate_rows
                return accelerated._cooperative_mop_handoff(
                    {"plan_sha256": plan_sha},
                    {"plan_sha256": plan_sha, "state_sha256": "d" * 64,
                     "active_cells": [], "active_children": {}},
                    charged_global_cpu_cores=25.0,
                )

            try:
                accelerated._stable_external_json = stable
                accepted = decision(rows)
                self.assertIsNotNone(accepted)
                assert accepted is not None
                self.assertEqual("setproctitle-caffeinate",
                                 accepted["owner_topology"])
                self.assertEqual(100, accepted["owner_pid"])
                self.assertEqual(101, accepted["launcher_pid"])
                self.assertEqual(102, accepted["capsule_pid"])
                self.assertEqual(3, accepted["active_shard_index"])
                self.assertEqual(25, accepted["observed_idle_worker_count"])

                mutations = {
                    "missing worker": rows[:-1],
                    "wrong capsule ancestry": [
                        *rows[:2], {**capsule, "ppid": 101}, *workers,
                    ],
                    "worker label lookalike": [
                        *rows[:3], {**workers[0],
                                    "command": "mop-c2-s03-worker-lookalike"},
                        *workers[1:],
                    ],
                    "extra labeled worker elsewhere": [
                        *rows, {"pid": 999, "ppid": 1,
                                "command": "mop-c2-s03-worker"},
                    ],
                    "supervisor not launchd-owned": [
                        {**supervisor, "ppid": 99}, *rows[1:],
                    ],
                    "malformed launcher lookalike": [
                        *rows, {"pid": 998, "ppid": 1,
                                "command": (f"{sys.executable} "
                                            f"{script} \"")},
                    ],
                }
                for label, candidate_rows in mutations.items():
                    with self.subTest(label=label):
                        self.assertIsNone(decision(candidate_rows))
            finally:
                accelerated._command_rows = original_rows
                accelerated._stable_external_json = original_json

    def test_cooperative_handoff_never_adds_a_second_doctor_lane(self) -> None:
        self.assertIsNone(accelerated._cooperative_mop_handoff(
            {"plan_sha256": "p"},
            {"plan_sha256": "p", "active_cells": ["live"],
             "active_children": {"live": {}}},
            charged_global_cpu_cores=25.0,
        ))

    def test_global_cpu_recovery_uses_three_sample_hysteresis(self) -> None:
        heads = [self._execution("candidate")]
        original_scan = accelerated._ORIGINAL_SCAN_HEADS
        original_global_cpu = accelerated._global_host_cpu_cores
        original_reservation = accelerated._reservation_from_evidence
        original_evidence = accelerated._evidence
        samples = iter((24.0, 0.0, 0.0, 0.0))
        accelerated._CPU_SAMPLES.clear()
        accelerated._SCAN_RESERVATIONS.clear()
        try:
            accelerated._ORIGINAL_SCAN_HEADS = lambda plan, state: (list(heads), {})
            accelerated._global_host_cpu_cores = lambda: next(samples)
            accelerated._reservation_from_evidence = (
                lambda cell, evidence: 4_000_000_000
            )
            accelerated._evidence = lambda: {}
            decisions = [accelerated._accelerated_scan_heads(
                {"cells": [heads[0]["cell"]]}, {"active_children": {}}
            )[0] for _ in range(4)]
        finally:
            accelerated._ORIGINAL_SCAN_HEADS = original_scan
            accelerated._global_host_cpu_cores = original_global_cpu
            accelerated._reservation_from_evidence = original_reservation
            accelerated._evidence = original_evidence
            accelerated._CPU_SAMPLES.clear()
            accelerated._SCAN_RESERVATIONS.clear()
        self.assertEqual([[], [], []], decisions[:3])
        self.assertEqual(1, len(decisions[3]))

    @staticmethod
    def _execution(cell_id: str) -> dict:
        return {"cell": {"cell_id": cell_id, "model_label": "0.5B",
                         "nominal_params_b": 0.5}}


if __name__ == "__main__":
    unittest.main()
