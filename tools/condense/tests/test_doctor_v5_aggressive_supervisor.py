#!/usr/bin/env python3.12
"""Focused no-model tests for the default-off aggressive-v2 live consumer."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


CONDENSE = Path(__file__).resolve().parents[1]
ROOT = CONDENSE.parents[1]
sys.path.insert(0, str(CONDENSE))
import doctor_v5_aggressive_admission_policy as policy
import doctor_v5_ultra_aggressive_autoresume as autoresume
import doctor_v5_ultra_aggressive_queue as queue


class AggressiveSupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temporary.name)
        self.old_stage = queue.STAGE_ROOT
        queue.STAGE_ROOT = self.root
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        queue.STAGE_ROOT = self.old_stage
        queue._OVERLAY = None
        queue._MARKER = None
        queue._PROFILE_BY_CELL = {}
        queue._SWAP_STATE_PATH = None
        queue._CPU_STATE_PATH = None
        queue._LAST_SWAP_DECISION = None
        queue._LAST_SWAP_DECISION_EPOCH = 0.0
        queue._LAST_CPU_DECISION = None
        queue._LAST_CPU_DECISION_EPOCH = 0.0
        self.temporary.cleanup()

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")

    def _overlay(self, *, profiles: list[dict] | None = None) -> dict:
        binary = self.root / "quantizer"
        binary.write_bytes(b"qualified-production-binary-fixture")
        profile_path = self.root / "thread-profile.json"
        profile_path.write_text('{"fixture":true}\n', encoding="utf-8")
        profiles = profiles or [self._profile("cell-3b", threads=8, wall=10.0)]
        selections = {}
        for row in profiles:
            if row["model_family"] != queue.QWEN_FAMILY:
                continue
            key = json.dumps([row["model_label"], row["rate_id"]],
                             separators=(",", ":"))
            selection = {
                "tier": row["model_label"], "rate": row["rate_id"],
                "selected_threads": row["threads"],
                "selected_wall_seconds": row["projected_wall_seconds"],
                "selected_peak_rss_bytes": row["selected_peak_rss_bytes"],
                "scratch_budget_bytes": 12_000_000_000,
                "mode": "fixture", "source_sha256": "a" * 64,
                "canonical_output_sha256": "b" * 64,
                "candidate_measurements": row["candidate_measurements"],
                "all_candidates_eligible": True,
                "selection_source": "qualified-vendor-thread-profile-contract",
            }
            selection["selection_sha256"] = policy._hash_value(selection)
            row["thread_selection_sha256"] = selection["selection_sha256"]
            selections[key] = selection
        qualification = {
            "status": "qualified",
            "required_threads": list(policy.REQUIRED_THREAD_PARITY),
            "contract": policy._file_reference(policy.THREAD_PROFILE_CONTRACT_PATH),
            "profile": policy._file_reference(profile_path),
            "binary": policy._file_reference(binary),
            "binary_sha256": policy._file_reference(binary)["sha256"],
            "selections": selections, "blockers": [],
        }
        qualification["qualification_sha256"] = policy._hash_value(qualification)
        evidence = {
            "plan_sha256": "1" * 64, "state_sha256": "2" * 64,
            "accepted_sample_count": 0, "rejected_sample_count": 0,
            "rejected_reasons": {}, "accepted_sample_sha256s": [],
            "profiles": {}, "authentication_rule": "fixture",
        }
        evidence["evidence_sha256"] = policy._hash_value(evidence)
        initial = policy.initial_swap_state(
            {"pressure_level": 1, "swap_used_mb": 0.0}, now_epoch=100.0
        )
        overlay = {
            "schema": policy.SCHEMA, "version": policy.VERSION,
            "created_at": "2026-07-15T00:00:00+00:00",
            "mode": "unbound-pending-only-generation",
            "plan_sha256": "1" * 64, "state_sha256_at_stage": "2" * 64,
            "source_bindings": {
                "policy_module": policy._file_reference(Path(policy.__file__)),
                "thread_profile_contract": policy._file_reference(
                    policy.THREAD_PROFILE_CONTRACT_PATH
                ),
                "plan": None, "queue_state": None,
            },
            "resource_policy": {
                "process_budget_bytes": policy.PROCESS_BUDGET_BYTES,
                "global_reserve_bytes": policy.GLOBAL_RESERVE_BYTES,
                "admission_ceiling_bytes": policy.ADMISSION_CEILING_BYTES,
                "reservation_rule": "fixture",
                "cpu_budget_cores": policy.CPU_BUDGET_CORES,
                "candidate_thread_profiles": {
                    policy.THREAD_PROFILE_CONTRACT[threads]["name"]: threads
                    for threads in policy.REQUIRED_THREAD_PARITY
                },
                "required_exact_parity_thread_counts": list(
                    policy.REQUIRED_THREAD_PARITY
                ),
                "thread_selection_rule": (
                    "exact per-tier/rate selected_threads from the hash-bound qualified "
                    "vendor contract; no nominal-tier default or fallback"
                ),
                "pack_objective": "maximize measured sum(1/selected_wall_seconds)",
                "swap": policy.swap_policy(), "sealed_swap_baseline_mb": 0.0,
            },
            "evidence": evidence, "initial_swap_state": initial,
            "thread_profile_qualification": qualification,
            "pending_profiles": profiles,
            "promotion": {
                "automatic_live_mutation_permitted": False,
                "requires_quiescent_paused_or_drained_state": True,
                "requires_zero_active_children": True,
                "requires_restage_at_checkpoint": True,
                "requires_atomic_pending_runtime_generation": True,
                "requires_exact_parity_for_all_selected_thread_counts": True,
                "requires_bound_production_profiles_8_12_16_20": True,
                "requires_exact_vendor_selected_threads_per_tier_rate": True,
                "completed_evidence_mutation_permitted": False,
                "rollback": "fixture",
            },
        }
        overlay["overlay_sha256"] = policy._hash_value(overlay)
        return overlay

    @staticmethod
    def _profile(cell_id: str, *, threads: int, wall: float,
                 reserve: int = 10_000_000_000, priority: int = 1,
                 model_label: str = "3B", rate_id: str = "0.1",
                 model_family: str = queue.QWEN_FAMILY) -> dict:
        measurements = [
            {"threads": candidate, "wall_seconds": wall + candidate,
             "peak_rss_bytes": reserve - 1_000_000,
             "receipt_sha256": f"{candidate:064x}"}
            for candidate in policy.REQUIRED_THREAD_PARITY
        ]
        return {
            "cell_id": cell_id, "priority": priority,
            "model_family": model_family,
            "model_label": model_label, "rate_id": rate_id,
            "branch": "codec_control", "current_status": "pending",
            "profile_key": f"{model_label}|codec_control|whole-parent",
            "reservation_bytes": reserve, "exclusive_canary": False,
            "source": "authenticated-process-tree-high-water",
            "high_water_bytes": reserve - 2_000_000_000,
            "deterministic_margin_bytes": 2_000_000_000,
            "threads": threads,
            "profile": policy.THREAD_PROFILE_CONTRACT[threads]["name"],
            "exclusive_cpu_profile": threads == 20,
            "projected_wall_seconds": wall,
            "selected_peak_rss_bytes": reserve - 1_000_000,
            "candidate_measurements": measurements,
            "exact_parity_approved": True,
            "all_four_candidates_eligible": True,
            "thread_selection_sha256": "0" * 64,
            "selection_source": "qualified-vendor-thread-profile-contract",
        }

    @staticmethod
    def _gpt_profile(cell_id: str = "gpt-oss-120b__4bpw__codec-control",
                     *, reserve: int = 60_000_000_000,
                     priority: int = 280) -> dict:
        return {
            "cell_id": cell_id, "priority": priority,
            "model_family": queue.GPTOSS_FAMILY,
            "model_label": "120B", "rate_id": "4",
            "branch": "codec_control", "current_status": "pending",
            "profile_key": "120B|codec_control|streaming",
            "reservation_bytes": reserve, "exclusive_canary": True,
            "source": "unmeasured-exclusive-canary", "high_water_bytes": None,
            "deterministic_margin_bytes": None, "threads": None,
            "profile": None, "exclusive_cpu_profile": None,
            "projected_wall_seconds": None, "selected_peak_rss_bytes": None,
            "candidate_measurements": [], "exact_parity_approved": False,
            "all_four_candidates_eligible": False,
            "thread_selection_sha256": None, "selection_source": None,
            "thread_profile_blocker": (
                "GPT-OSS uses its separately reviewed source-bound contract"
            ),
        }

    def _generation(self, overlay: dict) -> tuple[dict, dict, dict, dict]:
        overlay_path = self.root / "overlay.json"
        self._write_json(overlay_path, overlay)
        swap_path = self.root / "swap-state.json"
        self._write_json(swap_path, overlay["initial_swap_state"])
        cpu_path = self.root / "cpu-state.json"
        cpu_state = queue._initial_cpu_state(
            {"ok": True, "logical_cores": 28, "occupied_cores": 0.0},
            now_epoch=100.0,
        )
        self._write_json(cpu_path, cpu_state)
        registry = self.root / "adapter_registry.json"
        registry.write_text('{"registry":"fixture"}\n', encoding="utf-8")
        binary = overlay["thread_profile_qualification"]["binary"]
        runtime_rows = []
        runtime_paths: dict[str, Path] = {}
        specs: dict[str, dict] = {}
        for profile in overlay["pending_profiles"]:
            cell_id = profile["cell_id"]
            runtime = self.root / f"{cell_id}.runtime.json"
            runtime_paths[cell_id] = runtime
            if profile["model_family"] == queue.QWEN_FAMILY:
                spec = {
                    "model_family": queue.QWEN_FAMILY,
                    "campaign_binding": {"cell_id": cell_id},
                    "resources": {"threads": profile["threads"]},
                    "inputs": [{"role": "quantizer", "path": binary["path"],
                                "sha256": binary["sha256"],
                                "bytes": binary["bytes"]}],
                }
                self._write_json(runtime, spec)
                runtime_rows.append({
                    "cell_id": cell_id, "model_family": queue.QWEN_FAMILY,
                    "runtime_spec": queue._artifact(runtime),
                    "selected_threads": profile["threads"],
                    "thread_selection_sha256": profile[
                        "thread_selection_sha256"
                    ],
                })
            else:
                source = self.root / f"{cell_id}.source-work-plan.json"
                source.write_text('{"source":"fixture"}\n', encoding="utf-8")
                source_artifact = queue._artifact(source)
                source_artifact["role"] = "source_work_plan"
                spec = {
                    "model_family": queue.GPTOSS_FAMILY,
                    "campaign_binding": {"cell_id": cell_id},
                    "resources": {"threads": 12}, "inputs": [source_artifact],
                }
                self._write_json(runtime, spec)
                runtime_artifact = queue._artifact(runtime)
                parity = self.root / f"{cell_id}.exact-output-receipt.json"
                parity.write_text('{"exact_output":true}\n', encoding="utf-8")
                contract = {
                    "schema": queue.SOURCE_BOUND_CONTRACT_SCHEMA,
                    "version": queue.VERSION, "cell_id": cell_id,
                    "model_family": queue.GPTOSS_FAMILY,
                    "runtime_spec_sha256": runtime_artifact["sha256"],
                    "runtime_inputs_sha256": queue._hash_value(spec["inputs"]),
                    "selected_threads": 12, "projected_wall_seconds": 100.0,
                    "exclusive_cpu": False,
                    "review_status": "approved-source-bound-exact-output",
                    "exact_output_receipts": [queue._artifact(parity)],
                }
                contract["contract_sha256"] = queue._hash_value(contract)
                contract_path = self.root / f"{cell_id}.thread-contract.json"
                self._write_json(contract_path, contract)
                runtime_rows.append({
                    "cell_id": cell_id, "model_family": queue.GPTOSS_FAMILY,
                    "runtime_spec": runtime_artifact,
                    "source_bound_execution_contract": queue._artifact(contract_path),
                })
            specs[cell_id] = spec
        rollback = {
            "schema": queue.ROLLBACK_SCHEMA, "version": queue.VERSION,
            "generation_id": "generation-fixture",
            "predecessor_marker_sha256": None,
            "activation_cas": {
                "plan_sha256": "1" * 64, "state_sha256": "2" * 64,
                "campaign_sha256": "3" * 64, "registry_sha256": "4" * 64,
            },
            "restore_inventory": [{
                "role": "registry", "target": str(registry),
                "backup": str(self.root / "registry.backup"),
                "sha256": "4" * 64, "bytes": 1,
            }],
            "preserved_result_directories": [],
            "completed_evidence_mutation_permitted": False,
            "result_directory_deletion_permitted": False,
            "source_deletion_permitted": False,
        }
        rollback["rollback_manifest_sha256"] = queue._hash_value(rollback)
        rollback_path = self.root / "rollback.json"
        self._write_json(rollback_path, rollback)
        packet = {
            "schema": queue.PACKET_SCHEMA, "version": queue.VERSION,
            "created_at": "2026-07-15T00:00:00+00:00",
            "generation_id": "generation-fixture", "plan_sha256": "1" * 64,
            "overlay_sha256": overlay["overlay_sha256"],
            "thread_profile_qualification_sha256": overlay[
                "thread_profile_qualification"
            ]["qualification_sha256"],
            "registry": queue._artifact(registry),
            "runtime_specs": runtime_rows,
            "terminal_seal": {"rows": [], "rows_sha256": queue._hash_value([])},
            "rollback_manifest": queue._artifact(rollback_path),
            "completed_evidence_mutation_permitted": False,
            "runtime_defaults_mutation_permitted": False,
            "source_deletion_permitted": False,
        }
        packet["packet_sha256"] = queue._hash_value(packet)
        packet_path = self.root / "packet.json"
        self._write_json(packet_path, packet)
        marker = {
            "schema": queue.MARKER_SCHEMA, "version": queue.VERSION,
            "activated_at": "2026-07-15T00:00:00+00:00",
            "generation_id": packet["generation_id"],
            "activation_state_sha256": "2" * 64,
            "overlay": queue._artifact(overlay_path),
            "overlay_sha256": overlay["overlay_sha256"],
            "generation": queue._artifact(packet_path),
            "generation_sha256": packet["packet_sha256"],
            "aggressive_queue": queue._artifact(Path(queue.__file__)),
            "aggressive_autoresume": queue._artifact(Path(autoresume.__file__)),
            "aggressive_policy": queue._artifact(Path(policy.__file__)),
            "accelerated_resource_policy": queue._artifact(
                Path(queue.resource_policy.__file__)
            ),
            "rollback_manifest": queue._artifact(rollback_path),
            "swap_state": {"path": str(swap_path),
                           "sealed_baseline_swap_mb": 0.0,
                           "initial_state_sha256": overlay[
                               "initial_swap_state"]["state_sha256"]},
            "cpu_state": {"path": str(cpu_path),
                          "target_cores": queue.CPU_TARGET_CORES,
                          "initial_state_sha256": cpu_state[
                              "initial_state_sha256"]},
            "completed_evidence_mutation_permitted": False,
            "runtime_defaults_mutation_permitted": False,
        }
        marker["marker_sha256"] = queue._hash_value(marker)
        first_cell = overlay["pending_profiles"][0]["cell_id"]
        return marker, packet, specs[first_cell], {
            "registry": registry, "runtime": runtime_paths[first_cell],
            "runtimes": runtime_paths,
        }

    def test_static_sources_are_hash_pinned(self) -> None:
        queue._verify_static_bindings()
        self.assertEqual(queue.BASE_SHA256, queue.accel_loader.hash_file(queue.BASE_PATH)[0])

    def test_missing_profile_cannot_become_live(self) -> None:
        overlay = self._overlay()
        qualification = overlay["thread_profile_qualification"]
        qualification.update({"status": "missing", "profile": None, "binary": None,
                              "binary_sha256": None, "selections": {},
                              "blockers": ["missing"]})
        qualification["qualification_sha256"] = policy._hash_value(
            {key: value for key, value in qualification.items()
             if key != "qualification_sha256"}
        )
        overlay["overlay_sha256"] = policy._hash_value(
            {key: value for key, value in overlay.items() if key != "overlay_sha256"}
        )
        errors = queue._strict_overlay_errors(overlay)
        self.assertTrue(any("qualified exact" in row for row in errors))

    def test_runtime_must_use_exact_selected_threads_and_binary(self) -> None:
        overlay = self._overlay()
        profile = overlay["pending_profiles"][0]
        binary = overlay["thread_profile_qualification"]["binary"]
        cell = {"cell_id": profile["cell_id"],
                "model_family": queue.QWEN_FAMILY}
        spec = {"model_family": queue.QWEN_FAMILY,
                "resources": {"threads": profile["threads"]}, "inputs": [{
            "role": "quantizer", "path": binary["path"],
            "sha256": binary["sha256"], "bytes": binary["bytes"],
        }]}
        self.assertEqual([], queue._runtime_profile_errors(
            cell, spec, profile, overlay["thread_profile_qualification"]
        ))
        spec["resources"]["threads"] = 20
        self.assertTrue(any("selected thread" in row for row in
                            queue._runtime_profile_errors(
                                cell, spec, profile,
                                overlay["thread_profile_qualification"])))
        spec["resources"]["threads"] = profile["threads"]
        spec["inputs"][0]["sha256"] = "f" * 64
        self.assertTrue(any("qualified profile binary" in row for row in
                            queue._runtime_profile_errors(
                                cell, spec, profile,
                                overlay["thread_profile_qualification"])))

    def test_consumer_selects_measured_16_plus_8_under_24_cores(self) -> None:
        profiles = [
            self._profile("cell-16", threads=16, wall=10.0, priority=1,
                          model_label="14B", rate_id="0.1"),
            self._profile("cell-8", threads=8, wall=10.0, priority=2,
                          model_label="3B", rate_id="0.25"),
            self._profile("cell-12", threads=12, wall=30.0, priority=3,
                          model_label="7B", rate_id="0.33"),
        ]
        queue._PROFILE_BY_CELL = {row["cell_id"]: row for row in profiles}
        heads = [{"cell": {"cell_id": row["cell_id"]}} for row in profiles]
        with mock.patch.object(queue, "_ORIGINAL_SCAN_HEADS",
                               return_value=(heads, {})), \
                mock.patch.object(queue, "_advance_swap", return_value={
                    "allow_launch": True, "launch_limit": 8}), \
                mock.patch.object(queue, "_advance_cpu", return_value={
                    "allow_launch": True, "charged_global_cpu_cores": 0.0}):
            selected, _ = queue._aggressive_scan_heads(
                {"cells": [row["cell"] for row in heads]},
                {"active_children": {}},
            )
        self.assertEqual(["cell-16", "cell-8"],
                         [row["cell"]["cell_id"] for row in selected])

    def test_persistent_swap_hysteresis_rate_limits_emergency_shed(self) -> None:
        overlay = self._overlay()
        state_path = self.root / "swap.json"
        self._write_json(state_path, overlay["initial_swap_state"])
        queue._OVERLAY = overlay
        queue._SWAP_STATE_PATH = state_path
        soft = queue._advance_swap(
            {"pressure_level": 1, "swap_used_mb": 600.0},
            now_epoch=160.0, force=True,
        )
        hard = queue._advance_swap(
            {"pressure_level": 1, "swap_used_mb": 1800.0},
            now_epoch=220.0, force=True,
        )
        emergency = queue._advance_swap(
            {"pressure_level": 1, "swap_used_mb": 4000.0},
            now_epoch=280.0, force=True,
        )
        repeated = queue._advance_swap(
            {"pressure_level": 4, "swap_used_mb": 4000.0},
            now_epoch=281.0, force=True,
        )
        self.assertEqual("soft_throttle", soft["mode"])
        self.assertEqual("hard_stop", hard["mode"])
        self.assertEqual("emergency_shed", emergency["mode"])
        self.assertTrue(emergency["shed_one"])
        self.assertFalse(repeated["shed_one"])

    def test_checkpoint_shed_releases_only_largest_lane(self) -> None:
        output = self.root / "result"
        output.mkdir()
        checkpoint = output / "checkpoint.json"
        checkpoint.write_text('{"checkpoint":true}\n', encoding="utf-8")
        victim = types.SimpleNamespace(
            execution={"checkpoint_path": checkpoint, "output_dir": output,
                       "request": {"request_sha256": "a" * 64},
                       "cell": {"cell_id": "large"}},
            max_tree_rss_bytes=20_000, process_identity=("cmd", "start"),
            process_pgid=101, process=types.SimpleNamespace(pid=101),
        )
        survivor = types.SimpleNamespace()
        live = {"large": victim, "small": survivor}
        state = {"cells": {
            "large": {"status": "running"}, "small": {"status": "running"}}}
        samples = {"large": {"tree_rss_bytes": 20_000, "pgid": 101},
                   "small": {"tree_rss_bytes": 10_000, "pgid": 102}}
        decision = {"decision_sha256": "b" * 64, "state_sha256": "c" * 64}

        def release(_state: dict, live_cells: dict, lane: object) -> None:
            live_cells.pop("large" if lane is victim else "small")

        with mock.patch.object(queue, "_request_checkpoint_exit"), \
                mock.patch.object(queue._BASE, "_resource_stop_receipt",
                                  return_value={"path": "resource_stop.json",
                                                "sha256": "d" * 64,
                                                "bytes": 1,
                                                "receipt_sha256": "e" * 64,
                                                "reason": "fixture"}), \
                mock.patch.object(queue._BASE, "_append_event"), \
                mock.patch.object(queue._BASE, "_release_cell", side_effect=release), \
                mock.patch.object(queue._BASE, "_save_state"):
            selected = queue._shed_one_checkpoint_lane(
                {"plan_sha256": "f" * 64}, state, live, samples, decision
            )
        self.assertEqual("large", selected)
        self.assertEqual({"small"}, set(live))
        self.assertEqual("pending", state["cells"]["large"]["status"])
        self.assertTrue((output / "aggressive_swap_shed_intent.json").is_file())

    def test_marker_and_active_generation_are_hash_bound(self) -> None:
        overlay = self._overlay()
        marker, _packet, _spec, paths = self._generation(overlay)
        self.assertEqual([], queue._marker_errors(marker, verify_files=True))
        plan = {"plan_sha256": "1" * 64, "cells": [{
            "cell_id": "cell-3b", "cell_identity_sha256": "5" * 64,
            "model_family": queue.QWEN_FAMILY, "priority": 1,
            "runtime_spec_path": str(paths["runtime"].relative_to(ROOT)),
        }]}
        state = {"cells": {"cell-3b": {"status": "pending"}}}
        old_registry = queue._BASE.REGISTRY_PATH
        old_validate = queue._BASE._validate_runtime_spec
        try:
            queue._BASE.REGISTRY_PATH = paths["registry"]
            queue._BASE._validate_runtime_spec = lambda *args, **kwargs: ({}, {}, [])
            self.assertEqual([], queue._active_generation_errors(
                marker, overlay, plan=plan, state=state
            ))
        finally:
            queue._BASE.REGISTRY_PATH = old_registry
            queue._BASE._validate_runtime_spec = old_validate
        tampered = copy.deepcopy(marker)
        tampered["aggressive_queue"]["sha256"] = "0" * 64
        tampered["marker_sha256"] = queue._hash_value(
            {key: value for key, value in tampered.items() if key != "marker_sha256"}
        )
        self.assertTrue(any("queue source" in row for row in
                            queue._marker_errors(tampered, verify_files=True)))

    def test_mixed_generation_keeps_gptoss_off_qwen_profile_contract(self) -> None:
        qwen = self._profile("cell-qwen", threads=12, wall=10.0,
                             reserve=8_000_000_000)
        gptoss = self._gpt_profile(reserve=8_000_000_000)
        overlay = self._overlay(profiles=[qwen, gptoss])
        self.assertEqual([], queue._strict_overlay_errors(overlay))
        self.assertIsNone(gptoss["threads"])
        self.assertIsNone(gptoss["thread_selection_sha256"])
        marker, packet, _spec, paths = self._generation(overlay)
        plan = {"plan_sha256": "1" * 64, "cells": [
            {"cell_id": qwen["cell_id"], "cell_identity_sha256": "5" * 64,
             "model_family": queue.QWEN_FAMILY, "priority": qwen["priority"],
             "runtime_spec_path": str(
                 paths["runtimes"][qwen["cell_id"]].relative_to(ROOT))},
            {"cell_id": gptoss["cell_id"], "cell_identity_sha256": "6" * 64,
             "model_family": queue.GPTOSS_FAMILY,
             "priority": gptoss["priority"],
             "runtime_spec_path": str(
                 paths["runtimes"][gptoss["cell_id"]].relative_to(ROOT))},
        ]}
        state = {"cells": {row["cell_id"]: {"status": "pending"}
                           for row in plan["cells"]}}
        old_registry = queue._BASE.REGISTRY_PATH
        old_validate = queue._BASE._validate_runtime_spec
        try:
            queue._BASE.REGISTRY_PATH = paths["registry"]
            queue._BASE._validate_runtime_spec = lambda *args, **kwargs: ({}, {}, [])
            self.assertEqual([], queue._active_generation_errors(
                marker, overlay, plan=plan, state=state
            ))
            profiles = queue._bound_pack_profiles(marker, overlay, plan)
        finally:
            queue._BASE.REGISTRY_PATH = old_registry
            queue._BASE._validate_runtime_spec = old_validate
        self.assertEqual(12, profiles[gptoss["cell_id"]]["threads"])
        self.assertEqual(
            "reviewed-gptoss-source-bound-thread-contract",
            profiles[gptoss["cell_id"]]["selection_source"],
        )
        packet_row = next(row for row in packet["runtime_specs"]
                          if row["cell_id"] == gptoss["cell_id"])
        self.assertNotIn("thread_selection_sha256", packet_row)
        selected = queue._choose_mixed_pack(
            list(profiles.values()), active_reserved_bytes=0,
            active_threads=0, active_lanes=0, launch_limit=8,
        )
        self.assertEqual({qwen["cell_id"], gptoss["cell_id"]},
                         set(selected["selected_cell_ids"]))

    def test_gptoss_cannot_activate_without_real_review_contract_receipts(self) -> None:
        gptoss = self._gpt_profile(reserve=8_000_000_000)
        overlay = self._overlay(profiles=[gptoss])
        marker, packet, _spec, paths = self._generation(overlay)
        packet_row = packet["runtime_specs"][0]
        contract_path = Path(packet_row["source_bound_execution_contract"]["path"])
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["exact_output_receipts"] = []
        contract["contract_sha256"] = queue._hash_value(
            {key: value for key, value in contract.items()
             if key != "contract_sha256"}
        )
        self._write_json(contract_path, contract)
        plan = {"plan_sha256": "1" * 64, "cells": [{
            "cell_id": gptoss["cell_id"], "cell_identity_sha256": "6" * 64,
            "model_family": queue.GPTOSS_FAMILY,
            "priority": gptoss["priority"],
            "runtime_spec_path": str(paths["runtime"].relative_to(ROOT)),
        }]}
        state = {"cells": {gptoss["cell_id"]: {"status": "pending"}}}
        old_registry = queue._BASE.REGISTRY_PATH
        old_validate = queue._BASE._validate_runtime_spec
        try:
            queue._BASE.REGISTRY_PATH = paths["registry"]
            queue._BASE._validate_runtime_spec = lambda *args, **kwargs: ({}, {}, [])
            errors = queue._active_generation_errors(
                marker, overlay, plan=plan, state=state
            )
        finally:
            queue._BASE.REGISTRY_PATH = old_registry
            queue._BASE._validate_runtime_spec = old_validate
        self.assertTrue(any("contract changed" in row for row in errors))

    def test_total_host_cpu_saturation_blocks_immediately_and_recovers_hysteretically(
            self) -> None:
        state_path = self.root / "cpu-hysteresis.json"
        initial = queue._initial_cpu_state(
            {"ok": True, "logical_cores": 28, "occupied_cores": 0.0},
            now_epoch=100.0,
        )
        self._write_json(state_path, initial)
        queue._CPU_STATE_PATH = state_path
        queue._MARKER = {"cpu_state": {
            "initial_state_sha256": initial["initial_state_sha256"]}}
        saturated = queue._advance_cpu(
            {"ok": True, "logical_cores": 28, "occupied_cores": 24.0},
            now_epoch=101.0, force=True,
        )
        held_one = queue._advance_cpu(
            {"ok": True, "logical_cores": 28, "occupied_cores": 0.0},
            now_epoch=102.0, force=True,
        )
        held_two = queue._advance_cpu(
            {"ok": True, "logical_cores": 28, "occupied_cores": 0.0},
            now_epoch=103.0, force=True,
        )
        recovered = queue._advance_cpu(
            {"ok": True, "logical_cores": 28, "occupied_cores": 0.0},
            now_epoch=104.0, force=True,
        )
        unknown = queue._advance_cpu(
            {"ok": False, "error": "probe unavailable"},
            now_epoch=105.0, force=True,
        )
        self.assertFalse(saturated["allow_launch"])
        self.assertFalse(held_one["allow_launch"])
        self.assertFalse(held_two["allow_launch"])
        self.assertTrue(recovered["allow_launch"])
        self.assertFalse(unknown["allow_launch"])
        self.assertEqual("unknown", unknown["mode"])
        self.assertFalse(unknown["shed_or_stop_authority"])

    def test_saturated_unrelated_owner_prevents_launch_without_shedding(self) -> None:
        profile = self._profile("cell-qwen", threads=8, wall=10.0)
        queue._PROFILE_BY_CELL = {profile["cell_id"]: profile}
        heads = [{"cell": {"cell_id": profile["cell_id"]}}]
        with mock.patch.object(queue, "_ORIGINAL_SCAN_HEADS",
                               return_value=(heads, {})), \
                mock.patch.object(queue, "_advance_swap", return_value={
                    "allow_launch": True, "launch_limit": 8}), \
                mock.patch.object(queue, "_advance_cpu", return_value={
                    "allow_launch": False, "mode": "saturated",
                    "shed_or_stop_authority": False}), \
                mock.patch.object(queue, "_choose_mixed_pack") as pack:
            selected, _ = queue._aggressive_scan_heads(
                {"cells": [heads[0]["cell"]]}, {"active_children": {}},
            )
        self.assertEqual([], selected)
        pack.assert_not_called()

    def test_autoresume_does_not_fallback_when_marker_is_missing(self) -> None:
        with mock.patch.object(autoresume, "_read", side_effect=[
                {"mode": "run"}, {"status": "running"}]), \
                mock.patch.object(autoresume, "_marker",
                                  side_effect=ValueError("missing")):
            self.assertEqual(2, autoresume.main())


if __name__ == "__main__":
    unittest.main()
