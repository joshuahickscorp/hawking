from __future__ import annotations

import contextlib
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import doctor_v5_single_device_benchmark as sprint


ETA = sprint.eta_contract


def artifact(label: str) -> dict[str, object]:
    return {"sha256": hashlib.sha256(label.encode()).hexdigest(), "bytes": len(label)}


def run(role: str, index: int, seconds: float,
        components: list[str] | None = None) -> dict[str, object]:
    return {
        "role": role, "repeat_index": index, "status": "complete", "exit_code": 0,
        "skipped": False, "source_files_deleted": False,
        "exercised_components": components or [],
        "runtime_defaults_changed": False, "program": artifact(role),
        "benchmark_runner": artifact("trusted-runner"),
        "input_bundle": artifact("input"), "output_bundle": artifact("output"),
        "receipt_bundle": artifact("receipt"),
        "invocation_sha256": hashlib.sha256(f"{role}-{index}".encode()).hexdigest(),
        "semantic_contract_sha256": hashlib.sha256(b"semantics").hexdigest(),
        "wall_seconds": seconds, "cpu_seconds": seconds * 4,
        "gpu_seconds": 0.0, "peak_rss_bytes": 1024,
        "scratch_peak_bytes": 0, "disk_free_start_bytes": 10_000,
        "disk_free_end_bytes": 10_000,
        "swap_start_mb": 0.25, "swap_end_mb": 0.25,
        "memory_pressure_start": "normal", "memory_pressure_end": "normal",
        "thermal_start": "nominal", "thermal_end": "fair",
    }


def environment(*, production: bool) -> dict[str, object]:
    return {
        "machine_identity_sha256": hashlib.sha256(b"m3-ultra").hexdigest(),
        "same_machine_both_arms": True, "randomized_interleaved_order": True,
        "owner_free": production, "active_heavy_owner_count": 0 if production else 1,
        "real_artifact": production, "warmup_complete": production,
        "physical_counters_recorded": production,
        "workload_segment": sprint.DOCTOR_WORKLOAD_SEGMENT if production else None,
    }


def eta_calibration() -> dict[str, object]:
    return {
        "cell_id": "fixture-cell",
        "cell_identity_sha256": "e" * 64,
        "model_label": "3B",
        "branch_rate": "codec_control@4",
        "log_artifacts": [{
            "path": "/fixture/encode.log",
            "sha256": "f" * 64,
            "bytes": 1,
            "source_path": "/fixture/model.safetensors",
            "attempt_count": 1,
            "completed_tensor_count": 1,
            "completed_weights": 1,
        }],
        "queue_started_at": "2026-07-14T19:00:00+00:00",
        "first_encode_started_at": "2026-07-14T19:01:00+00:00",
        "queue_attempts": 1,
        "completed_weights": 1,
        "total_two_dimensional_weights": 2,
        "progress_fraction": 0.5,
        "elapsed_seconds": 3600.0,
        "observed_weights_per_second": 1.0,
        "projected_full_cell_seconds": 7200.0,
        "legacy_cell_seconds": 14_400.0,
        "sub_120b_observed_speedup": 2.0,
        "transferable_to_gpt_oss_120b": False,
    }


def eta_document(*, bindings: dict[str, object], low_days: float = 11,
                 high_days: float = 12,
                 status: str = "provisional-live-production-calibration") \
        -> dict[str, object]:
    created_at = "2026-07-14T20:00:00+00:00"
    created = dt.datetime.fromisoformat(created_at)
    blockers = ["one or more cells are blocked-execution"]
    eta: dict[str, object] = {
        "schema": ETA.SCHEMA,
        "created_at": created_at,
        "status": status,
        "eta_scope": "sub-120b-only",
        "input_bindings": bindings,
        "through_120b": ETA._gated_segment(appendix=False),
        "through_120b_plus_appendix": ETA._gated_segment(appendix=True),
    }
    if status == "provisional-live-production-calibration":
        seconds = [low_days * 86_400, high_days * 86_400]
        eta.update({
            "calibration_available": True,
            "calibration": eta_calibration(),
            "eta_blocked": False,
            "sub_120b": {
                "available": True,
                "seconds_range": seconds,
                "days_range": [value / 86_400 for value in seconds],
                "date_range": [ETA._date(created, value) for value in seconds],
            },
            "mechanical_sensitivity": ETA._mechanical_sensitivity(
                observed_sub_120b_speedup=2.0,
                through_range=[13 * 86_400, 20 * 86_400],
            ),
            "claim_limits": ETA.CLAIM_LIMITS[status],
        })
    elif status == "blocked-live-production-calibration":
        eta.update({
            "calibration_available": True,
            "calibration": eta_calibration(),
            "eta_blocked": True,
            "blockers": blockers,
            "failed_simulations": ["sub_120b_one_lane"],
            "sub_120b": ETA._empty_sub_120b(),
            "mechanical_sensitivity": ETA._mechanical_sensitivity(
                observed_sub_120b_speedup=2.0,
                through_range=None,
                blockers=blockers,
            ),
            "claim_limits": ETA.CLAIM_LIMITS[status],
        })
    elif status == "unavailable-live-production-calibration":
        eta.update({
            "calibration_available": False,
            "eta_blocked": True,
            "blockers": blockers,
            "sub_120b": ETA._empty_sub_120b(),
            "mechanical_sensitivity": ETA._mechanical_sensitivity(
                observed_sub_120b_speedup=None,
                through_range=None,
                blockers=blockers,
            ),
            "claim_limits": ETA.CLAIM_LIMITS[status],
        })
    else:
        raise ValueError(f"unsupported ETA fixture status: {status}")
    eta["document_sha256"] = sprint._hash_value(eta)
    return eta


@contextlib.contextmanager
def fresh_eta_document(*, low_days: float = 11, high_days: float = 12,
                       status: str = "provisional-live-production-calibration"):
    with tempfile.TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        paths = {
            "plan": root / "plan.json",
            "campaign": root / "campaign.json",
            "observer_state": root / "observer.json",
        }
        plan: dict[str, object] = {"kind": "plan"}
        plan["plan_sha256"] = ETA._hash_value(plan)
        campaign: dict[str, object] = {
            "kind": "campaign", "plan_sha256": plan["plan_sha256"]
        }
        campaign["campaign_sha256"] = ETA._hash_value(campaign)
        observer: dict[str, object] = {"kind": "observer"}
        observer["state_sha256"] = ETA._hash_value(observer)
        documents = (plan, campaign, observer)
        for path, document in zip(paths.values(), documents, strict=True):
            path.write_bytes(ETA._canonical(document))
        references = {
            name: ETA._read_bound_json(path)[1]
            for name, path in paths.items()
        }
        bindings: dict[str, object] = {
            **references,
            "plan_sha256": plan["plan_sha256"],
            "campaign_sha256": campaign["campaign_sha256"],
            "observer_state_sha256": observer["state_sha256"],
        }
        eta = eta_document(
            bindings=bindings, low_days=low_days, high_days=high_days,
            status=status,
        )
        with mock.patch.object(ETA, "INPUT_BINDING_PATHS", paths), \
                mock.patch.object(
                    ETA, "_read_inputs",
                    return_value=(plan, campaign, observer, bindings),
                ), mock.patch.object(
                    ETA, "_calibration", side_effect=lambda *_: eta_calibration()
                ):
            yield eta, paths


def production_authority(eta_sha256: str, baseline: list[dict[str, object]],
                         candidate: list[dict[str, object]]) -> dict[str, object]:
    owners = []
    for role, rows in (("baseline", baseline), ("candidate", candidate)):
        for index, row in enumerate(rows):
            owner = {
                "role": role, "repeat_index": index, "owner_free": True,
                "active_heavy_owner_count": 0,
                "before": artifact(f"{role}-{index}-owners-before"),
                "after": artifact(f"{role}-{index}-owners-after"),
            }
            row["owner_inventory_sha256"] = sprint._hash_value(owner)
            row["started_at_epoch_ns"] = 200 + index * 10 + (5 if role == "candidate" else 0)
            row["ended_at_epoch_ns"] = row["started_at_epoch_ns"] + 2
            owners.append(owner)
    order = []
    for index in range(len(baseline)):
        if index % 2:
            order.extend((f"candidate:{index}", f"baseline:{index}"))
        else:
            order.extend((f"baseline:{index}", f"candidate:{index}"))
    slices = []
    for tier in sprint.EXPECTED_TIERS:
        for rate in sprint.EXPECTED_RATES:
            for branch in sprint.EXPECTED_BRANCHES:
                label = f"{tier}:{rate}:{branch}"
                slices.append({
                    "tier": tier, "rate": rate, "branch": branch,
                    "source_cell_identity_sha256": hashlib.sha256(
                        f"cell:{label}".encode()
                    ).hexdigest(),
                    "slice_artifact": artifact(f"slice:{label}"),
                })
    identities = [row["source_cell_identity_sha256"] for row in slices]
    authority: dict[str, object] = {
        "schema": sprint.PRODUCTION_AUTHORITY_SCHEMA,
        "production_eta_sha256": eta_sha256,
        "workload_segment": sprint.DOCTOR_WORKLOAD_SEGMENT,
        "selection_frozen_before_execution": True,
        "canonical_baseline_source_bound": True,
        "every_remaining_workload_class_represented": True,
        "frozen_at_epoch_ns": 100,
        "representative_matrix": {
            "tiers": list(sprint.EXPECTED_TIERS),
            "rates": list(sprint.EXPECTED_RATES),
            "branches": list(sprint.EXPECTED_BRANCHES),
            "matrix_frozen_before_execution": True,
            "remaining_cell_ids_sha256": sprint._hash_value(identities),
            "representative_slices": slices,
            "representative_slices_sha256": sprint._hash_value(slices),
        },
        "workload_manifest": artifact("input"),
        "baseline_program": artifact("baseline"),
        "candidate_program": artifact("candidate"),
        "benchmark_runner": artifact("trusted-runner"),
        "baseline_invocation_sha256s": [row["invocation_sha256"] for row in baseline],
        "candidate_invocation_sha256s": [row["invocation_sha256"] for row in candidate],
        "owner_inventory_receipts": owners,
        "execution_order": order,
    }
    authority["authority_sha256"] = sprint._hash_value(authority)
    return authority


class SingleDeviceBenchmarkTests(unittest.TestCase):
    def test_synthetic_exact_pair_is_valid_but_cannot_drive_eta(self) -> None:
        receipt = sprint.build_receipt(
            scope=sprint.SYNTHETIC_SCOPE, components=["component"],
            baseline_runs=[run("baseline", 0, 10)],
            candidate_runs=[run("candidate", 0, 5, ["component"])],
            environment=environment(production=False), full_stack_end_to_end=False,
        )
        self.assertEqual([], sprint.validate_receipt(receipt, require_production=False))
        errors = sprint.validate_receipt(receipt, require_production=True)
        self.assertTrue(any("production owner-free" in row for row in errors))

    def test_production_structure_is_conservative_but_cannot_promote_eta(self) -> None:
        components = sorted(sprint.REQUIRED_COMPONENTS)
        with fresh_eta_document() as (eta, _paths):
            baseline = [run("baseline", i, value)
                        for i, value in enumerate((12, 10, 11))]
            candidate = [run("candidate", i, value, components)
                         for i, value in enumerate((6, 5, 6))]
            authority = production_authority(
                eta["document_sha256"], baseline, candidate
            )
            receipt = sprint.build_receipt(
                scope=sprint.PRODUCTION_SCOPE, components=components,
                baseline_runs=baseline, candidate_runs=candidate,
                environment=environment(production=True),
                full_stack_end_to_end=True,
                production_authority=authority,
            )
            self.assertAlmostEqual(11 / 6, receipt["summary"][
                "paired_speedup_conservative"
            ])
            self.assertEqual([], sprint.validate_receipt(
                receipt, require_production=False, production_authority=authority
            ))
            with self.assertRaisesRegex(sprint.SprintBenchmarkError,
                                        "trusted physical runner"):
                sprint.build_projection(
                    production_eta=eta, receipt=receipt,
                    production_authority=authority,
                )

    def test_output_or_receipt_mismatch_is_rejected(self) -> None:
        candidate = run("candidate", 0, 5, ["component"])
        candidate["output_bundle"] = artifact("different")
        with self.assertRaises(sprint.SprintBenchmarkError):
            sprint.build_receipt(
                scope=sprint.SYNTHETIC_SCOPE, components=["component"],
                baseline_runs=[run("baseline", 0, 10)], candidate_runs=[candidate],
                environment=environment(production=False), full_stack_end_to_end=False,
            )

    def test_tamper_and_skipped_run_are_rejected(self) -> None:
        receipt = sprint.build_receipt(
            scope=sprint.SYNTHETIC_SCOPE, components=["component"],
            baseline_runs=[run("baseline", 0, 10)],
            candidate_runs=[run("candidate", 0, 5, ["component"])],
            environment=environment(production=False), full_stack_end_to_end=False,
        )
        damaged = copy.deepcopy(receipt)
        damaged["candidate_runs"][0]["skipped"] = True
        errors = sprint.validate_receipt(damaged, require_production=False)
        self.assertTrue(any("hash mismatch" in row for row in errors))
        self.assertTrue(any("skipped" in row for row in errors))

    def test_component_microbenchmark_cannot_drive_projection(self) -> None:
        with fresh_eta_document(
                low_days=1 / 86_400, high_days=2 / 86_400) as (eta, _paths):
            baseline = [run("baseline", i, 10) for i in range(3)]
            candidate = [run("candidate", i, 5, ["qualified-thread-profile"])
                         for i in range(3)]
            authority = production_authority(
                eta["document_sha256"], baseline, candidate
            )
            receipt = sprint.build_receipt(
                scope=sprint.PRODUCTION_SCOPE,
                components=["qualified-thread-profile"],
                baseline_runs=baseline, candidate_runs=candidate,
                environment=environment(production=True),
                full_stack_end_to_end=True,
                production_authority=authority,
            )
            with self.assertRaisesRegex(sprint.SprintBenchmarkError,
                                        "trusted physical runner"):
                sprint.build_projection(
                    production_eta=eta, receipt=receipt,
                    production_authority=authority,
                )

    def test_caller_invented_thousand_x_receipt_cannot_promote(self) -> None:
        components = sorted(sprint.REQUIRED_COMPONENTS)
        with fresh_eta_document() as (eta, _paths):
            baseline = [run("baseline", i, 1_000) for i in range(3)]
            candidate = [run("candidate", i, 1, components) for i in range(3)]
            authority = production_authority(
                eta["document_sha256"], baseline, candidate
            )
            receipt = sprint.build_receipt(
                scope=sprint.PRODUCTION_SCOPE, components=components,
                baseline_runs=baseline, candidate_runs=candidate,
                environment=environment(production=True),
                full_stack_end_to_end=True,
                production_authority=authority,
            )
            self.assertEqual(1_000, receipt["summary"][
                "paired_speedup_conservative"
            ])
            with self.assertRaisesRegex(sprint.SprintBenchmarkError,
                                        "non-caller-declared campaign attestation"):
                sprint.build_projection(
                    production_eta=eta, receipt=receipt,
                    production_authority=authority,
                )

    def test_production_receipt_needs_external_prefrozen_authority(self) -> None:
        components = sorted(sprint.REQUIRED_COMPONENTS)
        with self.assertRaisesRegex(sprint.SprintBenchmarkError, "external frozen"):
            sprint.build_receipt(
                scope=sprint.PRODUCTION_SCOPE, components=components,
                baseline_runs=[run("baseline", i, 10) for i in range(3)],
                candidate_runs=[run("candidate", i, 5, components)
                                for i in range(3)],
                environment=environment(production=True), full_stack_end_to_end=True,
            )

    def test_malformed_run_and_forged_summary_refuse_without_crashing(self) -> None:
        receipt = sprint.build_receipt(
            scope=sprint.SYNTHETIC_SCOPE, components=["component"],
            baseline_runs=[run("baseline", 0, 10)],
            candidate_runs=[run("candidate", 0, 5, ["component"])],
            environment=environment(production=False), full_stack_end_to_end=False,
        )
        malformed = copy.deepcopy(receipt)
        malformed["baseline_runs"] = [None]
        malformed["receipt_sha256"] = sprint._hash_value(
            sprint._without(malformed, "receipt_sha256")
        )
        self.assertTrue(sprint.validate_receipt(malformed, require_production=False))
        malformed_components = copy.deepcopy(receipt)
        malformed_components["candidate_runs"][0]["exercised_components"] = [{}]
        malformed_components["receipt_sha256"] = sprint._hash_value(
            sprint._without(malformed_components, "receipt_sha256")
        )
        self.assertTrue(sprint.validate_receipt(
            malformed_components, require_production=False
        ))
        missing_wall = copy.deepcopy(receipt)
        del missing_wall["candidate_runs"][0]["wall_seconds"]
        missing_wall["receipt_sha256"] = sprint._hash_value(
            sprint._without(missing_wall, "receipt_sha256")
        )
        self.assertTrue(sprint.validate_receipt(
            missing_wall, require_production=False
        ))
        for role, field in (("baseline_runs", "output_bundle"),
                            ("candidate_runs", "receipt_bundle")):
            missing_artifact = copy.deepcopy(receipt)
            del missing_artifact[role][0][field]
            missing_artifact["receipt_sha256"] = sprint._hash_value(
                sprint._without(missing_artifact, "receipt_sha256")
            )
            self.assertTrue(sprint.validate_receipt(
                missing_artifact, require_production=False
            ))
        malformed_thermal = copy.deepcopy(receipt)
        malformed_thermal["baseline_runs"][0]["thermal_start"] = []
        malformed_thermal["receipt_sha256"] = sprint._hash_value(
            sprint._without(malformed_thermal, "receipt_sha256")
        )
        self.assertTrue(sprint.validate_receipt(
            malformed_thermal, require_production=False
        ))
        malformed_scope = copy.deepcopy(receipt)
        malformed_scope["scope"] = []
        malformed_scope["receipt_sha256"] = sprint._hash_value(
            sprint._without(malformed_scope, "receipt_sha256")
        )
        self.assertTrue(sprint.validate_receipt(
            malformed_scope, require_production=False
        ))
        forged = copy.deepcopy(receipt)
        forged["summary"]["repeat_count"] = 999
        forged["summary"]["baseline_median_seconds"] = 1
        forged["receipt_sha256"] = sprint._hash_value(
            sprint._without(forged, "receipt_sha256")
        )
        self.assertTrue(any("summary differs" in row for row in
                            sprint.validate_receipt(forged, require_production=False)))

    def test_malformed_external_authority_refuses_without_crashing(self) -> None:
        components = sorted(sprint.REQUIRED_COMPONENTS)
        with fresh_eta_document() as (eta, _paths):
            baseline = [run("baseline", i, 2) for i in range(3)]
            candidate = [run("candidate", i, 1, components) for i in range(3)]
            authority = production_authority(
                eta["document_sha256"], baseline, candidate
            )
            receipt = sprint.build_receipt(
                scope=sprint.PRODUCTION_SCOPE, components=components,
                baseline_runs=baseline, candidate_runs=candidate,
                environment=environment(production=True),
                full_stack_end_to_end=True,
                production_authority=authority,
            )
            for malformed_order in ([{}] * 6, [None] * 6):
                malformed = copy.deepcopy(authority)
                malformed["execution_order"] = malformed_order
                malformed["authority_sha256"] = sprint._hash_value(
                    sprint._without(malformed, "authority_sha256")
                )
                errors = sprint.validate_receipt(
                    receipt, require_production=True,
                    production_authority=malformed,
                    production_eta_sha256=eta["document_sha256"],
                )
                self.assertTrue(any("execution order" in row for row in errors))

    def test_threshold_refuses_to_accelerate_unmeasured_segments(self) -> None:
        with fresh_eta_document() as (eta, _paths):
            threshold = sprint.build_threshold(
                production_eta=eta, target_days=7
            )
        required = threshold[
            "sub_120b_required_additional_speedup_for_entire_range"
        ]
        self.assertAlmostEqual(12 / 7, required)
        self.assertEqual([11 / 7, 12 / 7], threshold[
            "sub_120b_required_additional_speedup_by_endpoint"
        ])
        self.assertEqual(sprint.STRICT_TARGET_CONTRACT,
                         threshold["target_contract"])
        self.assertEqual(sprint.STRICT_SPEEDUP_RELATION,
                         threshold["strict_inequality"])
        self.assertFalse(threshold["threshold_equality_is_sufficient"])
        self.assertAlmostEqual(7.0, 12 / required)
        self.assertIsNone(threshold[
            "unchanged_120b_plus_appendix_increment_days_by_endpoint"
        ])
        self.assertIsNone(threshold[
            "full_campaign_required_sub_120b_speedup_by_endpoint_if_other_segments_unchanged"
        ])
        self.assertFalse(threshold["gpt_oss_120b_threshold_available"])
        self.assertFalse(threshold["appendix_threshold_available"])
        self.assertFalse(threshold[
            "full_campaign_entire_range_possible_with_sub_120b_speedup_only"
        ])
        self.assertFalse(threshold["unmeasured_segment_speedup_applied"])

    def test_unavailable_or_blocked_eta_yields_no_threshold_or_projection(self) -> None:
        statuses = (
            "unavailable-live-production-calibration",
            "blocked-live-production-calibration",
        )
        numeric_threshold_fields = (
            "sub_120b_baseline_days_range",
            "sub_120b_required_additional_speedup_by_endpoint",
            "sub_120b_required_additional_speedup_for_entire_range",
            "unchanged_120b_plus_appendix_increment_days_by_endpoint",
            "full_campaign_required_sub_120b_speedup_by_endpoint_if_other_segments_unchanged",
        )
        for status in statuses:
            with self.subTest(status=status), fresh_eta_document(
                    status=status) as (eta, _paths):
                threshold = sprint.build_threshold(
                    production_eta=eta, target_days=7
                )
                self.assertFalse(threshold["available"])
                self.assertTrue(all(
                    threshold[field] is None for field in numeric_threshold_fields
                ))
                self.assertFalse(threshold["threshold_equality_is_sufficient"])
                self.assertFalse(threshold["gpt_oss_120b_threshold_available"])
                self.assertFalse(threshold["appendix_threshold_available"])
                self.assertFalse(threshold["unmeasured_segment_speedup_applied"])
                with self.assertRaisesRegex(
                        sprint.SprintBenchmarkError, "production ETA is blocked"):
                    sprint.build_projection(
                        production_eta=eta, receipt={}, production_authority={}
                    )

    def test_projection_never_transfers_sub_120b_speedup_to_other_segments(self) -> None:
        components = sorted(sprint.REQUIRED_COMPONENTS)
        with fresh_eta_document(low_days=11, high_days=14) as (eta, _paths):
            baseline = [run("baseline", index, 10, []) for index in range(3)]
            candidate = [run("candidate", index, 5, components)
                         for index in range(3)]
            authority = production_authority(
                eta["document_sha256"], baseline, candidate
            )
            receipt = sprint.build_receipt(
                scope=sprint.PRODUCTION_SCOPE, components=components,
                baseline_runs=baseline, candidate_runs=candidate,
                environment=environment(production=True),
                full_stack_end_to_end=True,
                production_authority=authority,
            )
            with mock.patch.object(
                    sprint, "TRUSTED_PRODUCTION_PROMOTION_AVAILABLE", True):
                projection = sprint.build_projection(
                    production_eta=eta, receipt=receipt,
                    production_authority=authority,
                )
        self.assertEqual([5.5, 7.0], projection["sub_120b_days_range"])
        self.assertFalse(projection["sub_120b_under_seven_days"])
        self.assertFalse(projection["threshold_equality_is_sufficient"])
        self.assertFalse(projection["through_120b_available"])
        self.assertIsNone(projection["through_120b_days_range"])
        self.assertFalse(projection["through_120b_plus_appendix_available"])
        self.assertIsNone(projection["through_120b_plus_appendix_days_range"])
        self.assertIsNone(projection["gptoss_120b_speedup_credit"])
        self.assertIsNone(projection["appendix_speedup_credit"])
        self.assertFalse(projection[
            "sub_120b_speedup_transferable_to_gpt_oss_120b"
        ])
        self.assertFalse(projection[
            "sub_120b_speedup_transferable_to_appendix"
        ])
        self.assertFalse(projection["unmeasured_segment_speedup_applied"])

    def test_stale_eta_input_binding_is_rejected(self) -> None:
        with fresh_eta_document() as (eta, paths):
            paths["campaign"].write_text(
                json.dumps({"changed": True}), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                    sprint.SprintBenchmarkError,
                    "ETA input binding is stale: campaign"):
                sprint.build_threshold(production_eta=eta, target_days=7)

    def test_old_v1_eta_document_fails_closed(self) -> None:
        eta = {
            "schema": "hawking.doctor_v5_production_eta.v1",
            "sub_120b": {"seconds_range": [11 * 86_400, 12 * 86_400]},
        }
        eta["document_sha256"] = sprint._hash_value(eta)
        with self.assertRaisesRegex(
                sprint.SprintBenchmarkError, "production ETA schema differs"):
            sprint.build_threshold(production_eta=eta, target_days=7)

    def test_threshold_refuses_malformed_authority_cleanly(self) -> None:
        with self.assertRaisesRegex(sprint.SprintBenchmarkError,
                                    "production ETA authority is invalid"):
            sprint.build_threshold(production_eta=[], target_days=7)  # type: ignore[arg-type]
        with fresh_eta_document() as (eta, _paths):
            eta["sub_120b"] = []
            eta["document_sha256"] = sprint._hash_value(
                sprint._without(eta, "document_sha256")
            )
            with self.assertRaisesRegex(
                    sprint.SprintBenchmarkError,
                    "production ETA authority is invalid"):
                sprint.build_threshold(production_eta=eta, target_days=7)


if __name__ == "__main__":
    unittest.main()
