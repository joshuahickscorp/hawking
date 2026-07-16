#!/usr/bin/env python3.12
"""Focused gates for the inert Doctor V5 stacked-admission overlay."""
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
import doctor_v5_stacked_admission as stacked


class StackedAdmissionTests(unittest.TestCase):
    def test_resource_health_fails_closed(self) -> None:
        healthy = {
            "pressure_level": 1, "swap_used_mb": 0.25,
            "power_source": "Now drawing from 'AC Power'",
        }
        self.assertTrue(stacked.resource_health(healthy, {"ok": True})["ok"])
        for mutation in (
            {"pressure_level": None}, {"pressure_level": 2},
            {"swap_used_mb": None},
            {"swap_used_mb": 1.1},
            {"power_source": "Battery Power"},
        ):
            sample = {**healthy, **mutation}
            self.assertFalse(stacked.resource_health(sample, {"ok": True})["ok"])
        self.assertFalse(stacked.resource_health(healthy, {"ok": False})["ok"])

    def test_dynamic_margin_requires_large_tier_measurement(self) -> None:
        nominal = {"3B": 3.0, "72B": 72.0}
        self.assertEqual(
            stacked.CURRENT_MARGIN_BYTES,
            stacked.dynamic_margin(["3B", "72B"], {}, nominal),
        )
        self.assertEqual(
            20_000_000_000,
            stacked.dynamic_margin(
                ["72B"], {"72B": {"peak_bytes": 32_000_000_000, "samples": 1}},
                nominal,
            ),
        )
        self.assertEqual(
            stacked.DYNAMIC_BASE_MARGIN_BYTES,
            stacked.dynamic_margin(
                ["72B"], {"72B": {"peak_bytes": 32_000_000_000, "samples": 3}},
                nominal,
            ),
        )

    def test_cpu_gate_uses_hysteresis_and_never_kills(self) -> None:
        self.assertTrue(stacked.cpu_launch_gate([27.0, 1.0, 27.0], logical_cores=28)["ok"])
        blocked = stacked.cpu_launch_gate([26.0, 27.0, 26.5], logical_cores=28)
        self.assertFalse(blocked["ok"])
        decision = stacked.admission_decision(
            active_reservations=[32_000_000_000],
            candidate_reservation=32_000_000_000,
            margin_bytes=12_000_000_000,
            health={"ok": True, "blockers": []}, cpu_samples=[2.0, 3.0, 4.0],
            live_labels=["72B"], candidate_label="72B", uncalibrated_large=set(),
        )
        self.assertTrue(decision["admit"])

    def test_two_key_loader_never_silently_degrades(self) -> None:
        self.assertIsNone(stacked.load_activated_overlay({}))
        with self.assertRaises(stacked.OverlayError):
            stacked.load_activated_overlay({stacked.ENV_OVERLAY: "/tmp/nope"})

    def test_overlay_hash_and_policy_tamper_are_rejected(self) -> None:
        reference = {"path": "fixture", "sha256": "a" * 64, "bytes": 1}
        overlay = {
            "schema": stacked.SCHEMA, "version": stacked.VERSION,
            "mode": "pending_only_opt_in",
            "policy": {
                "process_budget_bytes": stacked.PROCESS_BUDGET_BYTES,
                "minimum_margin_bytes": stacked.MIN_DYNAMIC_MARGIN_BYTES,
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
                "observer_state_at_stage": reference, "plan_sha256": "p",
            },
            "simulation": {
                "schema": stacked.SIMULATION_SCHEMA,
                "gpt_oss_120b_execution_ready": False,
                "projection_only_120b": True,
            },
        }
        overlay["overlay_sha256"] = stacked._hash_value(overlay)
        self.assertEqual([], stacked.validate_overlay(overlay))
        tampered = copy.deepcopy(overlay)
        tampered["policy"]["minimum_margin_bytes"] = 1
        self.assertTrue(stacked.validate_overlay(tampered))
        missing_simulation = copy.deepcopy(overlay)
        missing_simulation.pop("simulation")
        missing_simulation["overlay_sha256"] = stacked._hash_value({
            key: value for key, value in missing_simulation.items()
            if key != "overlay_sha256"
        })
        self.assertIn(
            "overlay 120B simulation/readiness anchor is invalid",
            stacked.validate_overlay(missing_simulation),
        )

    def test_uncalibrated_large_tier_is_exclusive_canary(self) -> None:
        # A compact synthetic graph exercises the packing rule without depending
        # on the live campaign's changing completion frontier.
        plan = {"cells": [
            self._cell("small", "3B", 3.0, 0),
            self._cell("large-a", "120B", 120.0, 1),
            self._cell("large-b", "120B", 120.0, 2),
        ]}
        campaign = {"cells": [
            self._state("small"), self._state("large-a"), self._state("large-b"),
        ], "active_children": {}}
        original = stacked._remaining_durations
        try:
            stacked._remaining_durations = lambda *args, **kwargs: {
                "small": 10.0, "large-a": 20.0, "large-b": 20.0,
            }
            result = stacked.simulate(
                plan, campaign, margin_bytes=stacked.DYNAMIC_BASE_MARGIN_BYTES,
                include_unready_120b=True,
                evidence={"3B": {"peak_bytes": 5_000_000_000, "samples": 1}},
            )
        finally:
            stacked._remaining_durations = original
        self.assertTrue(result["ok"])
        self.assertIn("120B", result["uncalibrated_large_tiers"])
        self.assertEqual(1, result["peak_lanes"])

    def test_calibrated_large_lanes_stack_below_budget(self) -> None:
        plan = {"cells": [
            self._cell("a", "72B", 72.0, 0), self._cell("b", "72B", 72.0, 1),
        ]}
        campaign = {"cells": [self._state("a"), self._state("b")],
                    "active_children": {}}
        original = stacked._remaining_durations
        try:
            stacked._remaining_durations = lambda *args, **kwargs: {"a": 10.0, "b": 10.0}
            result = stacked.simulate(
                plan, campaign, margin_bytes=stacked.DYNAMIC_BASE_MARGIN_BYTES,
                evidence={"72B": {"peak_bytes": 32_000_000_000, "samples": 5}},
            )
        finally:
            stacked._remaining_durations = original
        self.assertTrue(result["ok"])
        self.assertEqual(2, result["peak_lanes"])
        self.assertLessEqual(
            result["peak_reserved_bytes"] + stacked.DYNAMIC_BASE_MARGIN_BYTES,
            stacked.PROCESS_BUDGET_BYTES,
        )

    def test_activation_requires_quiescence_and_preserves_terminal_subset(self) -> None:
        terminal = {
            "cell_id": "done", "status": "complete", "cell_identity_sha256": "a",
            "runtime_spec_sha256": "b", "result_sha256": "c",
            "execution_receipt_sha256": "d", "disposition_sha256": None,
        }
        overlay = {
            "schema": stacked.SCHEMA, "version": stacked.VERSION,
            "mode": "pending_only_opt_in",
            "policy": {"process_budget_bytes": stacked.PROCESS_BUDGET_BYTES,
                       "minimum_margin_bytes": stacked.MIN_DYNAMIC_MARGIN_BYTES},
            "promotion": {"automatic_live_redeployment_permitted": False,
                          "requires_zero_active_children": True,
                          "requires_terminal_seal_subset_match": True,
                          "requires_live_observer_structural_readiness": True},
            "source_bindings": {"plan_sha256": "p"},
            "immutable_terminal_seal": {"rows": [terminal]},
        }
        overlay["overlay_sha256"] = stacked._hash_value(overlay)
        plan = {"plan_sha256": "p"}
        campaign = {"active_children": {"x": {}}, "cells": [terminal]}
        state = {"status": "running-cell"}
        result = stacked.activation_preflight(
            overlay, plan, campaign, state, {"ok": True, "blockers": []},
            singleton_lease_available=True, heavy_lease_available=True,
        )
        self.assertFalse(result["ready"])
        self.assertIn("campaign is not quiescent", result["blockers"])

    def test_live_observer_is_structurally_checked_not_byte_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            snapshot = snapshots / "observation.json"
            snapshot_doc = {
                "schema": "hawking.doctor_v5_post_120b_observation.v1",
                "generation_id": "a" * 64,
                "source_deletion_permitted": False,
                "eta": {"confidence": "blocked"},
                "final_gate": {
                    "ready": False, "blockers": ["campaign is not terminal"],
                },
                "gpt_oss_120b": {
                    "execution_ready": False,
                    "execution_readiness": {"currently_disk_admissible": True},
                },
            }
            snapshot_doc["packet_sha256"] = stacked._hash_value(snapshot_doc)
            snapshot.write_text(json.dumps(snapshot_doc), encoding="utf-8")
            snapshot_raw = snapshot.read_bytes()
            source = Path(stacked.post_120b.__file__).resolve()
            source_raw = source.read_bytes()
            state = {
                "schema": stacked.OBSERVER_STATE_SCHEMA,
                "version": stacked.post_120b.VERSION,
                "updated_at": "2026-07-15T00:00:00+00:00",
                "observer_source_sha256": hashlib.sha256(source_raw).hexdigest(),
                "observer_source_bytes": len(source_raw),
                "latest_generation_id": "a" * 64,
                "latest_observation_id": snapshot_doc["packet_sha256"],
                "latest_observation": {
                    "path": "snapshots/observation.json",
                    "sha256": hashlib.sha256(snapshot_raw).hexdigest(),
                    "bytes": len(snapshot_raw),
                },
                "final_interpretation_ready": False,
                "final_interpretation_packet": None,
                "final_interpretation_handoff": None,
                "eta": {"confidence": "blocked"},
                "gpt_oss_120b_execution_ready": False,
                "gpt_oss_120b_currently_disk_admissible": True,
                "final_gate_blockers": ["campaign is not terminal"],
                "source_deletion_permitted": False,
            }
            state["state_sha256"] = stacked._hash_value(state)
            observer = root / "observer_state.json"
            observer.write_text(json.dumps(state), encoding="utf-8")
            staged_reference = stacked._file_reference(observer)

            self.assertEqual([], stacked._observer_structure_errors(observer))
            state["updated_at"] = "2026-07-15T00:01:00+00:00"
            state["state_sha256"] = stacked._hash_value(
                {key: value for key, value in state.items() if key != "state_sha256"}
            )
            observer.write_text(json.dumps(state), encoding="utf-8")
            self.assertFalse(stacked._reference_matches(staged_reference))
            self.assertEqual([], stacked._observer_structure_errors(observer))

            state["eta"] = {"confidence": "forged"}
            state["state_sha256"] = stacked._hash_value(
                {key: value for key, value in state.items() if key != "state_sha256"}
            )
            observer.write_text(json.dumps(state), encoding="utf-8")
            self.assertIn(
                "live observer latest snapshot artifact is invalid",
                stacked._observer_structure_errors(observer),
            )
            state["eta"] = {"confidence": "blocked"}
            state["state_sha256"] = "0" * 64
            observer.write_text(json.dumps(state), encoding="utf-8")
            self.assertIn(
                "live observer self-hash is invalid",
                stacked._observer_structure_errors(observer),
            )

    def test_activation_accepts_valid_observer_rewrite_but_not_bound_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            snapshot_doc = {
                "schema": "hawking.doctor_v5_post_120b_observation.v1",
                "generation_id": "c" * 64,
                "source_deletion_permitted": False,
                "eta": {"confidence": "blocked"},
                "final_gate": {"ready": False, "blockers": ["not terminal"]},
                "gpt_oss_120b": {
                    "execution_ready": False,
                    "execution_readiness": {"currently_disk_admissible": None},
                },
            }
            snapshot_doc["packet_sha256"] = stacked._hash_value(snapshot_doc)
            snapshot = snapshots / "observation.json"
            snapshot.write_text(json.dumps(snapshot_doc), encoding="utf-8")
            snapshot_raw = snapshot.read_bytes()
            observer_source = Path(stacked.post_120b.__file__).resolve()
            source_raw = observer_source.read_bytes()
            observer_state = {
                "schema": stacked.OBSERVER_STATE_SCHEMA,
                "version": stacked.post_120b.VERSION,
                "updated_at": "2026-07-15T00:00:00+00:00",
                "observer_source_sha256": hashlib.sha256(source_raw).hexdigest(),
                "observer_source_bytes": len(source_raw),
                "latest_generation_id": "c" * 64,
                "latest_observation_id": snapshot_doc["packet_sha256"],
                "latest_observation": {
                    "path": "snapshots/observation.json",
                    "sha256": hashlib.sha256(snapshot_raw).hexdigest(),
                    "bytes": len(snapshot_raw),
                },
                "final_interpretation_ready": False,
                "final_interpretation_packet": None,
                "final_interpretation_handoff": None,
                "eta": {"confidence": "blocked"},
                "gpt_oss_120b_execution_ready": False,
                "gpt_oss_120b_currently_disk_admissible": None,
                "final_gate_blockers": ["not terminal"],
                "source_deletion_permitted": False,
            }
            observer_state["state_sha256"] = stacked._hash_value(observer_state)
            observer = root / "observer_state.json"
            observer.write_text(json.dumps(observer_state), encoding="utf-8")

            plan_file = root / "plan.json"
            campaign_file = root / "campaign.json"
            queue_file = root / "queue.json"
            child_file = root / "children.jsonl"
            for path, value in (
                (plan_file, "plan"), (campaign_file, "campaign"),
                (queue_file, "queue"), (child_file, "children"),
            ):
                path.write_text(value, encoding="utf-8")
            plan = {"plan_sha256": "p", "cells": []}
            campaign = {"active_children": {}, "cells": []}
            state = {"status": "drained"}
            reference_at_stage = stacked._file_reference(observer)
            overlay = {
                "schema": stacked.SCHEMA, "version": stacked.VERSION,
                "mode": "pending_only_opt_in",
                "policy": {
                    "process_budget_bytes": stacked.PROCESS_BUDGET_BYTES,
                    "minimum_margin_bytes": stacked.MIN_DYNAMIC_MARGIN_BYTES,
                },
                "promotion": {
                    "automatic_live_redeployment_permitted": False,
                    "requires_zero_active_children": True,
                    "requires_terminal_seal_subset_match": True,
                    "requires_live_observer_structural_readiness": True,
                },
                "source_bindings": {
                    "plan": stacked._file_reference(plan_file),
                    "campaign": stacked._file_reference(campaign_file),
                    "queue_state": stacked._file_reference(queue_file),
                    "child_resources": stacked._file_reference(child_file),
                    "observer_source": stacked._file_reference(observer_source),
                    "observer_state_at_stage": reference_at_stage,
                    "plan_sha256": "p",
                },
                "immutable_terminal_seal": stacked._terminal_seal(campaign),
                "simulation": {
                    "schema": stacked.SIMULATION_SCHEMA,
                    "gpt_oss_120b_execution_ready": False,
                    "projection_only_120b": True,
                },
            }
            overlay["overlay_sha256"] = stacked._hash_value(overlay)
            old_observer = stacked.OBSERVER_STATE
            try:
                stacked.OBSERVER_STATE = observer
                observer_state["updated_at"] = "2026-07-15T00:01:00+00:00"
                observer_state["state_sha256"] = stacked._hash_value({
                    key: value for key, value in observer_state.items()
                    if key != "state_sha256"
                })
                observer.write_text(json.dumps(observer_state), encoding="utf-8")
                self.assertFalse(stacked._reference_matches(reference_at_stage))
                result = stacked.activation_preflight(
                    overlay, plan, campaign, state, {"ok": True, "blockers": []},
                    singleton_lease_available=True, heavy_lease_available=True,
                )
                self.assertTrue(result["ready"], result["blockers"])

                plan_file.write_text("drifted", encoding="utf-8")
                result = stacked.activation_preflight(
                    overlay, plan, campaign, state, {"ok": True, "blockers": []},
                    singleton_lease_available=True, heavy_lease_available=True,
                )
                self.assertIn(
                    "source binding changed or is invalid: plan", result["blockers"]
                )
                plan_file.write_text("plan", encoding="utf-8")
                source_drift = copy.deepcopy(overlay)
                source_drift["source_bindings"]["observer_source"]["sha256"] = "0" * 64
                source_drift["overlay_sha256"] = stacked._hash_value({
                    key: value for key, value in source_drift.items()
                    if key != "overlay_sha256"
                })
                result = stacked.activation_preflight(
                    source_drift, plan, campaign, state,
                    {"ok": True, "blockers": []},
                    singleton_lease_available=True, heavy_lease_available=True,
                )
                self.assertIn(
                    "source binding changed or is invalid: observer_source",
                    result["blockers"],
                )
            finally:
                stacked.OBSERVER_STATE = old_observer

    @staticmethod
    def _cell(cell_id: str, label: str, nominal: float, priority: int) -> dict:
        return {
            "cell_id": cell_id, "model_label": label, "nominal_params_b": nominal,
            "priority": priority, "dependencies": [], "branch": "codec_control",
            "rate_id": "4", "exact_stored_parameter_count": int(nominal * 1e9),
            "parameter_manifest": {
                "source_weight_bytes": int(nominal * 1e8),
                "largest_source_shard_bytes": 4_000_000_000,
            },
            "admission": {"whole_parent_residency_assumed": nominal <= 16},
        }

    @staticmethod
    def _state(cell_id: str) -> dict:
        return {"cell_id": cell_id, "status": "pending", "started_at": None}


if __name__ == "__main__":
    unittest.main()
