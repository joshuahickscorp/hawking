#!/usr/bin/env python3.12
"""No-model tests for the controlled-swap successor entrypoint."""
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
import doctor_v5_controlled_swap_successor as successor


class ControlledSwapSuccessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predecessor_fixture = tempfile.TemporaryDirectory(dir=successor.ROOT)
        fixture = Path(cls.predecessor_fixture.name)
        cls.saved_marker = successor.v1.MARKER
        cls.saved_overlay = successor.stacked.DEFAULT_OVERLAY
        overlay_path = fixture / "stacked_overlay.json"
        marker_path = fixture / "accelerated_marker.json"
        reference = {"path": "fixture", "sha256": "a" * 64, "bytes": 1}
        overlay = {
            "schema": successor.stacked.SCHEMA,
            "version": successor.stacked.VERSION,
            "mode": "pending_only_opt_in",
            "policy": {
                "process_budget_bytes": successor.stacked.PROCESS_BUDGET_BYTES,
                "minimum_margin_bytes": successor.stacked.MIN_DYNAMIC_MARGIN_BYTES,
            },
            "promotion": {
                "automatic_live_redeployment_permitted": False,
                "requires_zero_active_children": True,
                "requires_terminal_seal_subset_match": True,
                "requires_live_observer_structural_readiness": True,
            },
            "source_bindings": {
                "plan": reference, "campaign": reference,
                "queue_state": reference, "child_resources": reference,
                "observer_source": reference,
                "observer_state_at_stage": reference, "plan_sha256": "fixture",
            },
            "simulation": {
                "schema": successor.stacked.SIMULATION_SCHEMA,
                "gpt_oss_120b_execution_ready": False,
                "projection_only_120b": True,
            },
        }
        overlay["overlay_sha256"] = successor.stacked._hash_value(overlay)
        overlay_path.write_text(json.dumps(overlay, sort_keys=True), encoding="utf-8")
        marker = {
            "schema": successor.v1.MARKER_SCHEMA,
            "overlay_path": str(overlay_path.resolve()),
            "overlay_sha256": overlay["overlay_sha256"],
        }
        marker["marker_sha256"] = successor._hash_value(marker)
        marker_path.write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")
        successor.v1.MARKER = marker_path
        successor.stacked.DEFAULT_OVERLAY = overlay_path
        cls.marker, cls.overlay = successor._predecessor_documents()

    @classmethod
    def tearDownClass(cls) -> None:
        successor.v1.MARKER = cls.saved_marker
        successor.stacked.DEFAULT_OVERLAY = cls.saved_overlay
        cls.predecessor_fixture.cleanup()

    def policy(self) -> dict:
        stage = successor.STAGE_ROOT
        predecessor = {
            "accelerated_queue": successor._artifact(successor.V1_PATH),
            "stacked_admission": successor._artifact(successor.STACKED_PATH),
            "active_marker": successor._artifact(successor.v1.MARKER),
            "admission_overlay": successor._artifact(
                successor.stacked.DEFAULT_OVERLAY
            ),
            "marker_sha256": self.marker["marker_sha256"],
            "overlay_sha256": self.overlay["overlay_sha256"],
        }
        policy = {
            "schema": successor.POLICY_SCHEMA,
            "version": successor.POLICY_VERSION,
            "created_at": "2026-07-15T00:00:00+00:00",
            "mode": successor.POLICY_MODE,
            "operational_root": str(stage.relative_to(successor.ROOT)),
            "predecessor": predecessor,
            "policy": {
                "swap_used_mb_max": 512.0,
                "swap_boundary": "absolute_inclusive",
                "required_pressure": "normal",
                "pressure_level_required": 1,
                "ram_capacity_credit_bytes": 0,
                "preserve_ac_power_gate": True,
                "preserve_thermal_gate": True,
            },
            "phase_gate": None,
            "phase_aware_disk_gate": None,
            "promotion": {
                "automatic_activation_permitted": False,
                "completed_evidence_mutation_permitted": False,
                "runtime_spec_mutation_permitted": False,
                "result_mutation_permitted": False,
                "pending_cells_only": True,
            },
        }
        policy["policy_sha256"] = successor._hash_value(policy)
        return policy

    def reseal(self, policy: dict) -> dict:
        policy["policy_sha256"] = successor._hash_value(
            successor._without(policy, "policy_sha256")
        )
        return policy

    def test_exact_predecessor_sources_are_hash_bound(self) -> None:
        successor._verify_static_bindings()
        self.assertEqual([], successor._predecessor_errors(
            self.policy(), self.marker, self.overlay, deep=False
        ))

    def test_policy_is_exact_pending_only_and_zero_credit(self) -> None:
        policy = self.policy()
        self.assertEqual([], successor.validate_policy(policy))
        self.assertEqual(0, policy["policy"]["ram_capacity_credit_bytes"])
        self.assertTrue(policy["promotion"]["pending_cells_only"])
        self.assertFalse(policy["promotion"]["result_mutation_permitted"])

    def test_marker_loader_requires_both_activation_keys(self) -> None:
        with self.assertRaises(successor.SuccessorError):
            successor.load_successor_marker(self.policy(), Path("/tmp/policy"), {})
        with self.assertRaises(successor.SuccessorError):
            successor.load_successor_marker(
                self.policy(), Path("/tmp/policy"),
                {successor.ENV_MARKER: "/tmp/missing"},
            )

    def test_old_cap_and_growth_semantics_are_rejected(self) -> None:
        old = copy.deepcopy(self.policy())
        old["policy"]["swap_used_mb_max"] = 1.0
        self.reseal(old)
        self.assertIn("successor controlled-swap guard is not exact",
                      successor.validate_policy(old))
        growth = copy.deepcopy(self.policy())
        growth["policy"]["swap_growth_mb_max"] = 512.0
        self.reseal(growth)
        self.assertIn("successor controlled-swap guard is not exact",
                      successor.validate_policy(growth))

    def test_marker_overlay_mixtures_fail_closed(self) -> None:
        mixed = copy.deepcopy(self.policy())
        mixed["predecessor"]["overlay_sha256"] = "0" * 64
        self.reseal(mixed)
        errors = successor._predecessor_errors(
            mixed, self.marker, self.overlay, deep=False
        )
        self.assertIn("predecessor marker/overlay mixture is invalid", errors)

        wrong_source = copy.deepcopy(self.policy())
        wrong_source["predecessor"]["accelerated_queue"]["sha256"] = "f" * 64
        self.reseal(wrong_source)
        errors = successor._predecessor_errors(
            wrong_source, self.marker, self.overlay, deep=False
        )
        self.assertTrue(any("accelerated_queue" in row for row in errors))

    def test_successor_marker_binds_activation_and_service_generation(self) -> None:
        def write_json(path: Path, value: dict) -> None:
            path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

        def bound(path: Path) -> dict:
            raw = path.read_bytes()
            return {"path": str(path.resolve()),
                    "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

        with tempfile.TemporaryDirectory(dir=successor.ROOT) as raw:
            root = Path(raw)
            packet_path, staged_marker = root / "packet.json", root / "staged.json"
            phase_gate, service = root / "phase.json", root / "service.plist"
            launch_agent, policy_path = root / "live.plist", root / "policy.json"
            receipt_root = root / "phase_receipts"
            receipt_root.mkdir()
            write_json(phase_gate, {"sealed": True})
            service.write_bytes(b"reviewed-service")
            launch_agent.write_bytes(service.read_bytes())
            policy = self.policy()
            write_json(policy_path, policy)
            old_marker = successor._read_json(successor.v1.MARKER)
            old_overlay = successor._read_json(successor.stacked.DEFAULT_OVERLAY)
            snapshot = {
                "old_marker_sha256": old_marker["marker_sha256"],
                "overlay_sha256": old_overlay["overlay_sha256"],
            }
            sources = {
                "activation_source": bound(successor.ACTIVATION_SOURCE),
                "successor_queue": bound(successor.SCRIPT),
                "successor_autoresume": bound(successor.SUCCESSOR_AUTORESUME),
                "phase_gate_declaration": policy["phase_gate"],
            }
            packet = {
                "schema": successor.PACKET_SCHEMA,
                "version": successor.POLICY_VERSION,
                "created_at": "now", "generation_id": "9" * 64,
                "snapshot": snapshot, "policy": policy["policy"],
                "policy_sha256": policy["policy_sha256"],
                "successor_policy": bound(policy_path), "sources": sources,
                "phase_gate": bound(phase_gate),
                "phase_receipt_root": str(receipt_root.resolve()),
                "service": {
                    "label": "com.hawking.doctorv5ultra.autoresume",
                    "target": str(launch_agent.resolve()),
                    "candidate": bound(service), "preexisting": None,
                    "was_loaded": True,
                },
                "mutation_boundary": successor.MUTATION_BOUNDARY,
            }
            packet["packet_sha256"] = successor._hash_value(packet)
            write_json(packet_path, packet)
            marker = {
                "schema": successor.MARKER_SCHEMA,
                "version": successor.POLICY_VERSION,
                "generation_id": packet["generation_id"], "prepared_at": "now",
                "packet": bound(packet_path),
                "successor_policy": bound(policy_path),
                "policy_sha256": policy["policy_sha256"],
                "predecessor_marker": bound(successor.v1.MARKER),
                "predecessor_marker_sha256": old_marker["marker_sha256"],
                "predecessor_overlay": bound(successor.stacked.DEFAULT_OVERLAY),
                "predecessor_overlay_sha256": old_overlay["overlay_sha256"],
                "activation_source": sources["activation_source"],
                "successor_queue": sources["successor_queue"],
                "successor_autoresume": sources["successor_autoresume"],
                "phase_gate": packet["phase_gate"],
                "phase_gate_declaration": policy["phase_gate"],
                "phase_aware_disk_gate": policy["phase_aware_disk_gate"],
                "phase_receipt_root": packet["phase_receipt_root"],
                "service_candidate": packet["service"]["candidate"],
                "activation_snapshot_sha256": successor._hash_value(snapshot),
                "result_mutation_permitted": False,
                "evidence_mutation_permitted": False,
            }
            marker["marker_sha256"] = successor._hash_value(marker)
            write_json(staged_marker, marker)
            saved = {
                "packet": successor.GENERATION_PACKET,
                "staged": successor.STAGED_MARKER,
                "phase": successor.PHASE_GATE_RECEIPT,
                "receipts": successor.PHASE_RECEIPT_ROOT,
                "service": successor.SERVICE_CANDIDATE,
                "launch": successor.LAUNCH_AGENT,
            }
            try:
                successor.GENERATION_PACKET = packet_path
                successor.STAGED_MARKER = staged_marker
                successor.PHASE_GATE_RECEIPT = phase_gate
                successor.PHASE_RECEIPT_ROOT = receipt_root
                successor.SERVICE_CANDIDATE = service
                successor.LAUNCH_AGENT = launch_agent
                self.assertEqual([], successor._successor_generation_errors(
                    marker, packet, policy, policy_path
                ))
                mixed = copy.deepcopy(marker)
                mixed["activation_source"]["sha256"] = "0" * 64
                mixed["marker_sha256"] = successor._hash_value(
                    successor._without(mixed, "marker_sha256")
                )
                write_json(staged_marker, mixed)
                self.assertTrue(any("activation" in row for row in
                                    successor._successor_generation_errors(
                                        mixed, packet, policy, policy_path
                                    )))
            finally:
                successor.GENERATION_PACKET = saved["packet"]
                successor.STAGED_MARKER = saved["staged"]
                successor.PHASE_GATE_RECEIPT = saved["phase"]
                successor.PHASE_RECEIPT_ROOT = saved["receipts"]
                successor.SERVICE_CANDIDATE = saved["service"]
                successor.LAUNCH_AGENT = saved["launch"]

    def test_phase_declaration_accepts_only_reviewed_exact_path(self) -> None:
        saved = successor.PHASE_GATE_PATH
        try:
            # Use an existing regular source as the reviewed-path fixture; the
            # declaration helper does not import or execute it.
            successor.PHASE_GATE_PATH = successor.SCRIPT
            declaration = successor.phase_gate_declaration(successor.SCRIPT)
            self.assertEqual(successor.PHASE_GATE_API_SCHEMA,
                             declaration["api_schema"])
            with self.assertRaises(successor.SuccessorError):
                successor.phase_gate_declaration(successor.V1_PATH)
        finally:
            successor.PHASE_GATE_PATH = saved

    def test_final_phase_module_loads_through_exact_api(self) -> None:
        declaration = successor.phase_gate_declaration(
            successor.PHASE_GATE_PATH
        )
        module = successor._load_phase_module(declaration)
        self.assertEqual(successor.PHASE_GATE_API_SCHEMA,
                         module.PHASE_GATE_API_SCHEMA)
        self.assertTrue(callable(module.install_phase_gate))

    def test_phase_policy_binds_exact_module_ledger_and_zero_ram(self) -> None:
        policy = self.policy()
        declaration = successor.phase_gate_declaration(
            successor.PHASE_GATE_PATH
        )
        root = successor.ROOT
        bindings = {
            "plan_path": str(successor._BASE.PLAN.resolve()),
            "plan_file_sha256": "1" * 64,
            "plan_sha256": "2" * 64,
            "plan_cell_sha256": "3" * 64,
            "cell_id": "fixture-cell",
            "cell_identity_sha256": "4" * 64,
            "runtime_spec_path": str(root / "fixture-runtime.json"),
            "runtime_spec_file_sha256": "5" * 64,
            "program_spec_sha256": "6" * 64,
            "execution_output_root": str(root / "fixture"),
            "disk_reserve_bytes": successor._BASE.DISK_RESERVE_BYTES,
            "declared_scratch_bytes": successor._BASE.MIN_SCRATCH_BYTES,
            "frozen_projected_output_bytes": 0,
        }
        config = {
            "schema": successor.PHASE_GATE_API_SCHEMA,
            "enabled": True,
            "module_sha256": declaration["module"]["sha256"],
            "ledger_module_sha256": successor._artifact(
                successor.LEDGER_PATH,
                ceiling=successor.MAX_PHASE_MODULE_BYTES,
            )["sha256"],
            "ram_credit_bytes": 0,
            "bindings": [bindings],
        }
        policy["phase_gate"] = declaration
        policy["phase_aware_disk_gate"] = config
        self.reseal(policy)
        self.assertEqual([], successor.validate_policy(policy))

        tampered = copy.deepcopy(policy)
        tampered["phase_aware_disk_gate"]["ram_credit_bytes"] = 1
        self.reseal(tampered)
        self.assertTrue(any("zero-RAM" in row
                            for row in successor.validate_policy(tampered)))

    def test_swap_boundary_is_inclusive_and_nonnegative(self) -> None:
        base = {"pressure_level": 1, "power_source": "AC Power"}
        for swap in (0, 6.44, 511.999, 512.0):
            self.assertTrue(successor.resource_health(
                {**base, "swap_used_mb": swap}, {"ok": True}
            )["ok"], swap)
        for swap in (-0.001, 512.000001, float("inf"), float("nan"), True, None):
            self.assertFalse(successor.resource_health(
                {**base, "swap_used_mb": swap}, {"ok": True}
            )["ok"], swap)

    def test_pressure_is_exact_numeric_normal_and_old_guards_remain(self) -> None:
        healthy = {"pressure_level": 1, "swap_used_mb": 6.44,
                   "power_source": "AC Power"}
        self.assertTrue(successor.resource_health(healthy, {"ok": True})["ok"])
        for mutation, thermal in (
            ({"pressure_level": True}, {"ok": True}),
            ({"pressure_level": 2}, {"ok": True}),
            ({"power_source": "Battery Power"}, {"ok": True}),
            ({}, {"ok": False}),
        ):
            self.assertFalse(successor.resource_health(
                {**healthy, **mutation}, thermal
            )["ok"])

    def test_owner_identity_accepts_only_successor_entrypoint(self) -> None:
        plan = {"plan_sha256": "a" * 64}
        nonce = "b" * 32
        started = "started"
        saved = successor._BASE._process_identity
        try:
            def record_for(command: str) -> dict:
                successor._BASE._process_identity = lambda _pid: (command, started)
                row = {
                    "schema": successor._BASE.PID_SCHEMA,
                    "version": successor._BASE.VERSION,
                    "pid": 42, "process_started": started,
                    "process_command_sha256": hashlib.sha256(
                        command.encode("utf-8")
                    ).hexdigest(),
                    "ownership_nonce": nonce,
                    "plan_sha256": plan["plan_sha256"], "recorded_at": "now",
                }
                row["pid_record_sha256"] = successor._BASE._hash_value(row)
                return row

            current = f"python {successor.SCRIPT} run --nonce {nonce}"
            self.assertTrue(successor._owner_alive(record_for(current), plan))
            old = f"python {successor.V1_PATH} run --nonce {nonce}"
            self.assertFalse(successor._owner_alive(record_for(old), plan))
            lookalike = (
                f"python /tmp/x{successor.SCRIPT.name} run --nonce {nonce}"
            )
            self.assertFalse(successor._owner_alive(record_for(lookalike), plan))
        finally:
            successor._BASE._process_identity = saved

    def test_configure_calls_v1_first_and_changes_no_ram_capacity(self) -> None:
        policy = self.policy()
        saved = {
            "configure": successor.v1.configure,
            "install": successor._install_phase_gate,
            "policy_root": successor._policy_root,
            "validate": successor.validate_policy,
            "budget": successor._BASE.PROCESS_BUDGET_BYTES,
            "swap": successor._BASE.SWAP_TOLERANCE_MB,
            "snapshot": successor._BASE.ram_scheduler.resource_snapshot,
            "health": successor.stacked.resource_health,
            "gate": successor._BASE._execution_resource_gate,
            "owner": successor._BASE._owner_alive,
            "start": successor._BASE.start_queue,
        }
        calls: list[str] = []
        try:
            successor.v1.configure = lambda overlay: calls.append(
                overlay["overlay_sha256"]
            )
            successor._install_phase_gate = (
                lambda _policy, _root, gate: gate
            )
            successor._policy_root = lambda _path: successor.STAGE_ROOT
            successor.validate_policy = lambda *_args, **_kwargs: []
            successor.configure(
                self.overlay, policy,
                policy_path=successor.STAGE_ROOT / "successor_policy.json",
            )
            self.assertEqual([self.overlay["overlay_sha256"]], calls)
            self.assertEqual(512.0, successor._BASE.SWAP_TOLERANCE_MB)
            self.assertEqual(saved["budget"], successor._BASE.PROCESS_BUDGET_BYTES)
        finally:
            successor.v1.configure = saved["configure"]
            successor._install_phase_gate = saved["install"]
            successor._policy_root = saved["policy_root"]
            successor.validate_policy = saved["validate"]
            successor._BASE.PROCESS_BUDGET_BYTES = saved["budget"]
            successor._BASE.SWAP_TOLERANCE_MB = saved["swap"]
            successor._BASE.ram_scheduler.resource_snapshot = saved["snapshot"]
            successor.stacked.resource_health = saved["health"]
            successor._BASE._execution_resource_gate = saved["gate"]
            successor._BASE._owner_alive = saved["owner"]
            successor._BASE.start_queue = saved["start"]


if __name__ == "__main__":
    unittest.main()
