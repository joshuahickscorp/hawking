#!/usr/bin/env python3.12
"""Offline crash/restart, lease, signal, and adversarial worker tests."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import pathlib
import plistlib
import subprocess
import sys
from dataclasses import replace

import pytest


TOOLS = pathlib.Path(__file__).resolve().parents[2]
CONDENSE = TOOLS / "condense"
for directory in (TOOLS, CONDENSE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import glm52_gravity as gravity  # noqa: E402
import glm52_grounding as grounding  # noqa: E402
import glm52_notifications as notifications  # noqa: E402
import glm52_state as state  # noqa: E402
import glm52_worker as worker  # noqa: E402
import glm52_xet_live as xet_live  # noqa: E402
from glm52_common import atomic_json, canonical, seal  # noqa: E402


REVISION = worker.OFFICIAL_REVISION
CAMPAIGN = "glm52-worker-offline-test"
CONTROLLER_EPOCH = "glm52-worker-test-epoch"
SHARD = "model-00001-of-00001.safetensors"
TENSOR = "model.layers.0.self_attn.q_proj.weight"
CHAT_ID = 100424242
CHAT_DIGEST = state.telegram_chat_identity_digest(CHAT_ID)
TELEGRAM_KEY = b"worker-test-telegram-hmac-key-material-0001"
EVIDENCE_KEY = b"worker-test-evidence-hmac-key-material-0002"
GROUNDING_KEY = b"worker-test-grounding-hmac-key-material-0003"
CREATED_AT = "2026-07-21T00:00:00Z"
BOT_ID = 424242424
BOT_DIGEST = notifications.telegram_bot_identity_digest(BOT_ID)


def _policy(path: str, schema: str) -> dict:
    return {
        "path": path,
        "expected_seal_sha256": None,
        "expected_schema": schema,
        "allowed_statuses": ["PASS"],
        "validator_id": "campaign_artifact_v1",
        "validator_source_sha256": state.EVIDENCE_VALIDATOR_SOURCE_SHA256[
            "campaign_artifact_v1"
        ],
        "require_producer_hmac": True,
    }


def _gates() -> dict:
    return {
        "ASSEMBLE_ARTIFACT": {
            "require_source_complete": True,
            "require_tensor_complete": True,
            "require_final_source_eviction": True,
            "require_telegram_delivery": True,
            "require_phone_status": False,
            "required_phone_status_path": None,
            "required_artifacts": {},
            "required_checklist": {},
        },
        "COMPLETE": {
            "require_source_complete": True,
            "require_tensor_complete": True,
            "require_final_source_eviction": True,
            "require_telegram_delivery": True,
            "require_phone_status": True,
            "required_phone_status_path": "GLM52_PHONE_STATUS.json",
            "required_artifacts": {
                "final": _policy("GLM52_FINAL.json", "test.worker.final.v1")
            },
            "required_checklist": {
                "test_stop": _policy("TEST_STOP.json", "test.worker.stop.v1")
            },
        },
    }


def _contract(*, gates: dict | None = None) -> dict:
    return state.make_expected_campaign_contract(
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
        expected_chat_identity_digest=CHAT_DIGEST,
        source_shards=[{
            "path": SHARD,
            "logical_bytes": 4096,
            "xet_hash": "a" * 64,
            "lfs_sha256": "b" * 64,
        }],
        expected_tensors=[TENSOR],
        window_schedule=[{
            "schedule_index": 0,
            "window_id": "window-0001",
            "source_shards": [SHARD],
            "carry_in_shards": [],
            "new_fetch_shards": [SHARD],
            "refetch_shards": [],
            "carry_out_shards": [],
            "evict_shards": [SHARD],
            "tensor_set": [TENSOR],
        }],
        state_gates=gates or _gates(),
        source_profile="SYNTHETIC_TEST_ONLY",
        created_at=CREATED_AT,
    )


def _runtime(
    tmp_path: pathlib.Path,
    *,
    contract: dict | None = None,
) -> gravity.Runtime:
    contract = contract or _contract()
    artifact_root = tmp_path / "artifacts"
    controller_root = tmp_path / "controller"
    phone_root = artifact_root / "phone"
    for path in (artifact_root, controller_root, phone_root):
        path.mkdir(parents=True, exist_ok=True)
    contract_path = artifact_root / "contract.json"
    atomic_json(contract_path, contract)
    config_path = tmp_path / "controller-config.json"
    config_doc = gravity.make_config(
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
        controller_root=controller_root,
        artifact_root=artifact_root,
        expected_contract_path=contract_path,
        phone_status_directory=phone_root,
        expected_chat_identity_digest=CHAT_DIGEST,
        allow_synthetic_contract=True,
        controller_epoch=CONTROLLER_EPOCH,
    )
    atomic_json(config_path, config_doc)
    return gravity.Runtime(
        config_path=config_path,
        config_sha256=config_doc["seal_sha256"],
        controller_root=controller_root,
        artifact_root=artifact_root,
        expected_contract_path=contract_path,
        phone_status_directory=phone_root,
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
        controller_epoch=CONTROLLER_EPOCH,
        allow_synthetic_contract=True,
        expected_contract=contract,
        telegram_auth=state.TelegramAuthConfig(
            hmac_key=TELEGRAM_KEY,
            expected_chat_identity_digest=CHAT_DIGEST,
        ),
        evidence_auth=state.EvidenceAuthConfig(
            hmac_key=EVIDENCE_KEY,
            campaign_id=CAMPAIGN,
            source_revision=REVISION,
        ),
        grounding_auth=grounding.ProducerAuthenticator(GROUNDING_KEY),
        contract_phone_status_path="phone/GLM52_PHONE_STATUS.json",
    )


def _boot(runtime: gravity.Runtime) -> None:
    controller = runtime.controller()
    with controller:
        intent = controller.prepare_transition(
            "PRECHECK", claim_id="worker-test:boot:0001"
        )
        receipt = state.make_telegram_delivery_receipt(
            intent,
            auth=runtime.telegram_auth,
            bot_api_response={
                "ok": True,
                "result": {
                    "message_id": 1,
                    "chat": {"id": CHAT_ID},
                    "text": intent["rendered_message"],
                },
            },
            http_status=200,
            delivered_at=CREATED_AT,
        )
        checkpoint = controller.boot(
            transition_intent=intent,
            telegram_delivery=receipt,
        )
        assert checkpoint["state"] == "PRECHECK"


def _checkpoint(runtime: gravity.Runtime) -> dict:
    value, _status = worker._read_only_checkpoint_snapshot(runtime.controller())
    return value


def _persistent_worker(
    runtime: gravity.Runtime,
    **kwargs,
) -> worker.PersistentWorker:
    """Explicit synthetic-only bypass for control-spine tests without an outbox."""
    return worker.PersistentWorker(
        runtime,
        test_only_notification_audit_bypass=True,
        **kwargs,
    )


def _worker_config(tmp_path: pathlib.Path, runtime: gravity.Runtime) -> worker.WorkerConfig:
    readiness = {
        kind: {
            "path": f"readiness/{kind}.json",
            "expected_file_sha256": "0" * 64,
        }
        for kind in worker.READINESS_KINDS
    }
    document = worker.make_worker_config(
        profile=worker.SYNTHETIC_CONFIG_PROFILE,
        campaign_id=runtime.campaign_id,
        source_revision=runtime.source_revision,
        controller_config_path=runtime.config_path,
        workspace_root=tmp_path,
        expected_contract_sha256=runtime.expected_contract["seal_sha256"],
        expected_contract_created_at=CREATED_AT,
        heartbeat_interval_seconds=0.1,
        readiness=readiness,
        evidence_auth=runtime.evidence_auth,
    )
    return worker._parse_worker_config_document(
        document,
        tmp_path / "GLM52_WORKER_CONFIG.json",
    )


def _xet_gate(required_artifacts: dict) -> dict:
    return {
        "require_source_complete": False,
        "require_tensor_complete": False,
        "require_final_source_eviction": False,
        "require_telegram_delivery": True,
        "require_phone_status": False,
        "required_phone_status_path": None,
        "required_artifacts": required_artifacts,
        "required_checklist": {},
    }


def _xet_contract(plan: dict) -> dict:
    gates = _gates()
    gates["AUTOTUNE_XET"] = _xet_gate({
        "xet_autotune_plan": {
            "path": "GLM52_XET_AUTOTUNE_PLAN.json",
            "expected_seal_sha256": plan["seal_sha256"],
            "expected_schema": "hawking.glm52.xet_autotune_plan.v2",
            "allowed_statuses": ["PASS_OFFLINE_PLAN_BODY_NOT_READ"],
            "validator_id": "sealed_exact_v1",
            "validator_source_sha256": state.EVIDENCE_VALIDATOR_SOURCE_SHA256[
                "sealed_exact_v1"
            ],
            "require_producer_hmac": False,
        },
    })
    gates["BUILD_ADAPTER"] = _xet_gate({
        "xet_autotune_result": {
            "path": "GLM52_XET_AUTOTUNE.json",
            "expected_seal_sha256": None,
            "expected_schema": "hawking.glm52.xet_autotune_result.v1",
            "allowed_statuses": [
                "PASS_LIVE_XET_AUTOTUNE_COMPLETE_SCHEDULE_REFREEZE_REQUIRED"
            ],
            "validator_id": "xet_autotune_result_v1",
            "validator_source_sha256": state.EVIDENCE_VALIDATOR_SOURCE_SHA256[
                "xet_autotune_result_v1"
            ],
            "require_producer_hmac": True,
        },
    })
    return _contract(gates=gates)


def _make_xet_result(
    plan: dict,
    contract: dict,
    auth: state.EvidenceAuthConfig,
) -> dict:
    trial_ids = [row["trial_id"] for row in plan["trial_matrix"]]
    selected = plan["trial_matrix"][0]

    def selection(lane: str) -> dict:
        return {
            "lane": lane,
            "status": "SELECTED",
            "trial_id": selected["trial_id"],
            "selected_trial": {"trial_id": selected["trial_id"]},
            "selected_caller_concurrent_shard_streams": selected[
                "caller_concurrent_shard_streams"
            ],
            "post_autotune_schedule_refreeze_required": True,
        }

    selections = {
        lane: selection(lane) for lane in ("acquisition", "steady")
    }
    source_refs = [
        dict(item) for item in plan["inputs"]
        if item.get("path") in {
            "GLM52_OFFICIAL_MANIFEST.json",
            "GLM52_SOURCE_FORMAT_LEDGER.json",
            "GLM52_SHARD_DEPENDENCY_GRAPH.json",
            "GLM52_SOURCE_ADMISSION.json",
        }
    ]
    budget = plan["network_budget"]
    raw = seal({
        "schema": xet_live.AUTOTUNE_RESULT_SCHEMA,
        "status": "PASS_LIVE_XET_AUTOTUNE_COMPLETE_SCHEDULE_REFREEZE_REQUIRED",
        "repo": xet_live.REPO_ID,
        "revision": xet_live.REVISION,
        "bindings": {
            "plan_seal_sha256": plan["seal_sha256"],
            "plan_toolchain_binding_sha256": hashlib.sha256(
                canonical(plan["toolchain_binding"])
            ).hexdigest(),
            "plan_input_refs_sha256": hashlib.sha256(
                canonical(plan["inputs"])
            ).hexdigest(),
            "source_refs": source_refs,
            "live_executor_sha256": hashlib.sha256(
                pathlib.Path(xet_live.__file__).read_bytes()
            ).hexdigest(),
            "resource_reserve_policy": copy.deepcopy(
                plan["resource_reserve_policy"]
            ),
        },
        "coverage": {
            "trial_ids_in_plan_order": trial_ids,
            "trial_results": [
                {
                    "trial_id": trial_id,
                    "trial_result_seal_sha256": hashlib.sha256(
                        f"trial:{trial_id}".encode()
                    ).hexdigest(),
                    "resource_verdict": {"status": "PASS", "measured": {}},
                    "selection_candidate_sha256": hashlib.sha256(
                        f"candidate:{trial_id}".encode()
                    ).hexdigest(),
                }
                for trial_id in trial_ids
            ],
            "required_file_settings": [8, 16, 24, 32, 48],
            "required_file_settings_measured": [8, 16, 24, 32, 48],
            "fixed_profiles_measured": ["FIXED_16", "FIXED_32", "FIXED_64"],
            "high_performance_measured": True,
            "cache_profiles_measured": ["CACHE_1G_COLD", "CACHE_1G_REPLAY"],
            "unique_sealed_ranges_hashed": len(plan["range_strategy"]["body_ranges"]),
            "repeated_range_sha256_consistent": True,
        },
        "selections": selections,
        "selected_profile": copy.deepcopy(selections),
        "largest_shard_validations": [
            {
                "lane": lane,
                "evidence_seal_sha256": hashlib.sha256(
                    f"largest:{lane}".encode()
                ).hexdigest(),
                "selected_trial_id": selected["trial_id"],
                "observed_sha256": plan["largest_shard_validation"]["lfs_sha256"],
            }
            for lane in ("acquisition", "steady")
        ],
        "network_budget": {
            "planned_range_payload_bytes": budget["bounded_range_payload_bytes"],
            "planned_full_validation_payload_bytes": budget[
                "largest_shard_validation_bytes"
            ],
            "planned_total_payload_bytes": budget["planned_maximum_bytes"],
            "actual_network_bytes": 1,
            "hard_cap_bytes": budget["hard_cap_bytes"],
            "remaining_bytes": budget["hard_cap_bytes"] - 1,
            "protocol_overhead_and_retries_included": True,
        },
        "claim_boundary": {
            "xet_autotune_complete": True,
            "all_12_trials_measured": True,
            "two_largest_shard_full_hash_passes": True,
            "model_body_files_created_by_executor": 0,
            "full_model_downloaded": False,
            "model_capability_claimed": False,
            "streaming_schedule_refreeze_required": True,
        },
    })
    raw_body = {key: copy.deepcopy(value) for key, value in raw.items() if key != "seal_sha256"}
    evidence = {
        "raw_xet_autotune_result_seal_sha256": raw["seal_sha256"],
        "controller_anchor_sha256": "b" * 64,
    }
    raw_body.update({
        "campaign_id": contract["campaign_id"],
        "source_revision": contract["source_revision"],
        "expected_contract_sha256": contract["seal_sha256"],
        "evidence": evidence,
        "evidence_sha256": hashlib.sha256(canonical(evidence)).hexdigest(),
    })
    return state.seal_producer_authenticated_evidence(raw_body, auth=auth)


def _xet_runtime(tmp_path: pathlib.Path) -> tuple[gravity.Runtime, dict]:
    plan = json.loads(
        (TOOLS.parent / "GLM52_XET_AUTOTUNE_PLAN.json").read_text(encoding="utf-8")
    )
    contract = _xet_contract(plan)
    runtime = _runtime(tmp_path, contract=contract)
    result = _make_xet_result(plan, contract, runtime.evidence_auth)
    atomic_json(runtime.artifact_root / "GLM52_XET_AUTOTUNE_PLAN.json", plan)
    atomic_json(runtime.artifact_root / "GLM52_XET_AUTOTUNE.json", result)
    return runtime, result


def _advance_to_build_adapter(runtime: gravity.Runtime) -> None:
    _boot(runtime)
    controller = runtime.controller()
    with controller:
        for message_id, target in enumerate((
            "CLOSE_KIMI",
            "RELEASE_KIMI_SOURCE",
            "ADMIT_GLM_SOURCE",
            "BUILD_MANIFEST",
            "BUILD_DEPENDENCY_GRAPH",
            "AUTOTUNE_XET",
            "BUILD_ADAPTER",
        ), start=2):
            claim = f"worker-test:xet:{message_id:04d}"
            intent = controller.prepare_transition(target, claim_id=claim)
            receipt = state.make_telegram_delivery_receipt(
                intent,
                auth=runtime.telegram_auth,
                bot_api_response={
                    "ok": True,
                    "result": {
                        "message_id": message_id,
                        "chat": {"id": CHAT_ID, "type": "private"},
                        "text": intent["rendered_message"],
                    },
                },
                http_status=200,
                delivered_at=CREATED_AT,
            )
            checkpoint = controller.transition(
                target,
                transition_intent=intent,
                telegram_delivery=receipt,
            )
            assert checkpoint["state"] == target


def _write_final_schedule(
    runtime: gravity.Runtime,
    xet_result: dict,
    *,
    binding: dict,
) -> tuple[dict, pathlib.Path, dict]:
    expected_schedule_hash = hashlib.sha256(
        canonical(runtime.expected_contract["window_schedule"])
    ).hexdigest()
    body = {
        "schema": "hawking.glm52.streaming_schedule.v2",
        "status": "FROZEN_AFTER_XET_AUTOTUNE",
        "campaign_id": runtime.campaign_id,
        "source_revision": runtime.source_revision,
        "expected_contract_sha256": runtime.expected_contract["seal_sha256"],
        "window_count": 1,
        "windows": copy.deepcopy(runtime.expected_contract["window_schedule"]),
        "dependency_freeze": {
            "window_schedule_sha256": expected_schedule_hash,
            "source_dependency_membership_changed": False,
            "tensor_ownership_changed": False,
        },
        "autotune_binding": {
            "xet_autotune_result_seal_sha256": xet_result["seal_sha256"]
        },
        "resource_policy_binding": {
            "path": worker.OFFICIAL_RESOURCE_POLICY_PATH,
            "seal_sha256": worker.OFFICIAL_RESOURCE_POLICY_SEAL,
            "required_free_disk_bytes": worker.OFFICIAL_REQUIRED_FREE_DISK_BYTES,
            "expected_contract_sha256": runtime.expected_contract["seal_sha256"],
        },
    }
    schedule = state.seal_producer_authenticated_evidence(
        body, auth=runtime.evidence_auth
    )
    path = runtime.artifact_root / "final-schedule.json"
    atomic_json(path, schedule)
    result_path = runtime.artifact_root / "GLM52_XET_AUTOTUNE.json"
    subject = {
        "schedule_path": path.name,
        "schedule_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "schedule_seal_sha256": schedule["seal_sha256"],
        "window_schedule_sha256": expected_schedule_hash,
        "live_xet_result_path": result_path.name,
        "live_xet_result_file_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "live_xet_result_seal_sha256": xet_result["seal_sha256"],
        "controller_autotune_event_chain_sha256": binding[
            "controller_event_chain_sha256"
        ],
        "controller_autotune_terminal_evidence_seal_sha256": binding[
            "terminal_evidence_seal_sha256"
        ],
        "post_autotune_final": True,
    }
    return subject, path, body


def test_crash_restart_replays_checkpoint_and_authenticates_heartbeat(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    _boot(runtime)

    def crash_after_first_wait(_seconds: float) -> bool:
        raise RuntimeError("simulated process crash")

    first = _persistent_worker(
        runtime,
        heartbeat_interval_seconds=0.0,
        waiter=crash_after_first_wait,
        instance_id="worker-crash-instance-0001",
    )
    with pytest.raises(RuntimeError, match="simulated process crash"):
        first.run(install_signals=False)
    assert first.controller.lease.held is False

    checkpoint_after_crash = _checkpoint(runtime)
    authenticated = worker._validate_authenticated_heartbeat(
        checkpoint_after_crash["heartbeat"], runtime=runtime
    )
    assert authenticated is not None
    assert authenticated["lifecycle"] == (
        "CONTROL_PLANE_STANDBY_NO_SCIENTIFIC_DISPATCH"
    )

    restarted = _persistent_worker(
        runtime,
        heartbeat_interval_seconds=0.0,
        waiter=lambda _seconds: False,
        instance_id="worker-restart-instance-0002",
    )
    result = restarted.run(max_cycles=1, install_signals=False)
    assert result["status"] == "TEST_CYCLE_LIMIT"
    final = _checkpoint(runtime)
    assert final["state"] == "PRECHECK"
    assert final["event_count"] == checkpoint_after_crash["event_count"] + 1
    assert final["heartbeat"]["telemetry"]["worker_instance_id"] == (
        "worker-restart-instance-0002"
    )


def test_restart_recovers_exact_event_tail_after_checkpoint_write_crash(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    _boot(runtime)
    instance = _persistent_worker(
        runtime,
        heartbeat_interval_seconds=0.0,
        instance_id="worker-tail-crash-instance-0001",
    )

    def crash_before_checkpoint():
        raise RuntimeError("crash after durable event append")

    instance.controller._write_checkpoint = crash_before_checkpoint
    with pytest.raises(RuntimeError, match="after durable event append"):
        instance.run(max_cycles=1, install_signals=False)
    assert instance.controller.lease.held is False

    config = _worker_config(tmp_path, runtime)
    report = worker.evaluate_preflight(
        config,
        runtime=runtime,
        contract=runtime.expected_contract,
        allow_test_profile=True,
    )
    checkpoint_check = report["checks"]["durable_checkpoint_replay"]
    assert checkpoint_check["status"] == "PASS"
    assert checkpoint_check["detail"]["recovery_required"] is True
    assert checkpoint_check["detail"]["recoverable_tail"] == {
        "controller_events": 1,
        "window_events": 0,
    }

    recovered = _persistent_worker(
        runtime,
        heartbeat_interval_seconds=0.0,
        instance_id="worker-tail-recovery-instance-0002",
    ).run(max_cycles=1, install_signals=False)
    assert recovered["status"] == "TEST_CYCLE_LIMIT"
    checkpoint = _checkpoint(runtime)
    assert checkpoint["event_count"] == 3
    assert checkpoint["heartbeat"]["telemetry"]["worker_instance_id"] == (
        "worker-tail-recovery-instance-0002"
    )


def test_singleton_lease_refuses_second_worker_and_releases_after_failure(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    _boot(runtime)
    first = _persistent_worker(runtime, instance_id="worker-lease-owner-0001")
    second = _persistent_worker(runtime, instance_id="worker-lease-rival-0002")
    first.controller.acquire()
    try:
        with pytest.raises(state.StateError, match="already-running"):
            second.run(max_cycles=1, install_signals=False)
    finally:
        first.controller.close()
    assert second.controller.lease.held is False
    assert _persistent_worker(
        runtime, instance_id="worker-lease-successor-0003"
    ).run(max_cycles=1, install_signals=False)["status"] == "TEST_CYCLE_LIMIT"


def test_static_symlink_lease_attack_is_rejected_before_target_write(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    _boot(runtime)
    lease_path = runtime.controller_root / "GLM52_CONTROLLER.lease"
    lease_path.unlink()
    victim = tmp_path / "victim.txt"
    victim.write_text("do-not-touch", encoding="utf-8")
    lease_path.symlink_to(victim)
    instance = _persistent_worker(runtime, instance_id="worker-symlink-test-0001")
    with pytest.raises(worker.WorkerError, match="lease target"):
        instance.run(max_cycles=1, install_signals=False)
    assert victim.read_text(encoding="utf-8") == "do-not-touch"


def test_signal_latches_until_boundary_then_emits_safe_stop_heartbeat(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    _boot(runtime)
    latch = worker.SignalLatch()
    latch.request(signal_number := 15)
    instance = _persistent_worker(
        runtime,
        signal_latch=latch,
        instance_id="worker-signal-test-0001",
    )
    result = instance.run(install_signals=False)
    assert result == {
        "status": "STOPPED",
        "reason": "SIGNAL_STOP",
        "cycles": 0,
        "checkpoint_seal_sha256": result["checkpoint_seal_sha256"],
    }
    heartbeat = _checkpoint(runtime)["heartbeat"]["telemetry"]
    assert heartbeat["safe_point"] == "BETWEEN_WINDOWS"
    assert heartbeat["lifecycle"] == "STOPPED_AT_SAFE_BOUNDARY"
    assert heartbeat["control"]["worker_directive"] == "SIGNAL_STOP"
    assert signal_number == 15


def test_signal_latch_wakes_persistent_wait_without_applying_mid_phase() -> None:
    wake_calls: list[str] = []
    latch = worker.SignalLatch(wake=lambda: wake_calls.append("wake"))
    latch.request(15)
    latch.request(2)
    assert latch.requested_signal == 15
    assert wake_calls == ["wake"]


def test_pause_and_stop_directives_are_consumed_only_by_boundary(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    _boot(runtime)
    instance = _persistent_worker(runtime, instance_id="worker-control-test-0001")

    class FakeControls:
        def __init__(self) -> None:
            self.points: list[str] = []

        def poll(self, controller, *, safe_point):
            controller.lease.assert_held()
            self.points.append(safe_point)
            return {
                "worker_directive": "PAUSE" if safe_point == "BETWEEN_WINDOWS" else "CONTINUE_CURRENT_WINDOW",
                "applied_action": gravity.ACTION_PAUSE,
                "applied_request_sequence": 0,
                "application_state": "IN_SYNC",
            }

    fake = FakeControls()
    instance.controls = fake
    instance.controller.acquire()
    try:
        assert instance.boundary("BETWEEN_RANGE_READS")[0] == "CONTINUE_CURRENT_WINDOW"
        assert instance.boundary("BETWEEN_WINDOWS")[0] == "PAUSE"
        with pytest.raises(worker.WorkerError, match="unknown worker safe point"):
            instance.boundary("MID_KERNEL")
    finally:
        instance.controller.close()
    assert fake.points == ["BETWEEN_RANGE_READS", "BETWEEN_WINDOWS"]


def test_worker_heartbeat_rejects_forged_hmac_and_identity(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    _boot(runtime)
    _persistent_worker(
        runtime, instance_id="worker-heartbeat-test-0001"
    ).run(max_cycles=1, install_signals=False)
    heartbeat = _checkpoint(runtime)["heartbeat"]
    forged = copy.deepcopy(heartbeat)
    forged["telemetry"]["lifecycle"] = "FORGED_PROGRESS"
    with pytest.raises(worker.WorkerError, match="producer HMAC"):
        worker._validate_authenticated_heartbeat(forged, runtime=runtime)
    foreign = copy.deepcopy(heartbeat)
    foreign["telemetry"]["campaign_id"] = "foreign-campaign"
    with pytest.raises(worker.WorkerError, match="campaign_id mismatch"):
        worker._validate_authenticated_heartbeat(foreign, runtime=runtime)


def test_preflight_reports_each_production_blocker_without_side_effects(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    _boot(runtime)
    config = _worker_config(tmp_path, runtime)
    report = worker.evaluate_preflight(
        config,
        runtime=runtime,
        contract=runtime.expected_contract,
        allow_test_profile=True,
    )
    codes = {item["code"] for item in report["blockers"]}
    assert {
        "rotated_telegram_identity",
        "notification_outbox_ready",
        "final_xet_schedule",
        "resource_policy_digest",
        "execution_adapters_grounded",
        "scientific_dispatch_bound",
    }.issubset(codes)
    assert "contract_v3" not in codes
    assert "deterministic_created_at" not in codes
    assert "authentication_role_separation" not in codes
    assert report["checks"]["authentication_role_separation"]["detail"] == {
        "telegram_auth_loaded": True,
        "evidence_auth_loaded": True,
        "grounding_auth_loaded": True,
        "pairwise_distinct": True,
        "identity_fingerprints_serialized": False,
    }
    assert "durable_checkpoint_replay" not in codes
    assert report["production_start_authorized"] is False
    assert report["checks"]["scientific_dispatch_bound"]["reason"].startswith(
        "NO_SCIENTIFIC_DISPATCH:"
    )
    assert report["side_effects"] == {
        "network_access": False,
        "xet_body_bytes_read": 0,
        "telegram_messages_sent": 0,
        "scientific_phases_executed": 0,
        "files_deleted": 0,
    }


def test_telegram_rotation_receipt_binds_live_distinct_keys(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    credential_status = {
        "ready": True,
        "token_configured": True,
        "private_chat_configured": True,
        "hmac_key_configured": True,
        "token_identity_digest": "c" * 64,
        "notification_bot_identity_digest": BOT_DIGEST,
        "chat_identity_digest": CHAT_DIGEST,
    }
    subject = {
        "expected_chat_identity_digest": CHAT_DIGEST,
        "token_identity_digest": credential_status["token_identity_digest"],
        "notification_bot_identity_digest": BOT_DIGEST,
        "telegram_key_identity_sha256": runtime.telegram_auth._key_material_identity(),
        "evidence_key_identity_sha256": runtime.evidence_auth._key_material_identity(),
        "rotation_completed_at": CREATED_AT,
        "historical_credential_revoked": True,
    }
    receipt = worker.make_readiness_receipt(
        worker.TELEGRAM_ROTATION,
        subject,
        runtime=runtime,
        created_at=CREATED_AT,
    )
    raw = canonical(receipt)
    envelope = worker._validate_receipt_envelope(
        raw,
        expected_kind=worker.TELEGRAM_ROTATION,
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        runtime=runtime,
    )
    assert worker._validate_telegram_rotation(
        envelope["subject"], runtime, credential_status
    )[
        "historical_credential_revoked"
    ] is True
    bad = dict(envelope["subject"])
    bad["telegram_key_identity_sha256"] = "0" * 64
    with pytest.raises(worker.WorkerError, match="key identities"):
        worker._validate_telegram_rotation(bad, runtime, credential_status)


def _notification_fixture(
    tmp_path: pathlib.Path,
    runtime: gravity.Runtime,
    *,
    rotation_seal: str = "d" * 64,
):
    module_target = tmp_path / "tools/condense/glm52_notifications.py"
    module_target.parent.mkdir(parents=True, exist_ok=True)
    module_target.write_bytes((CONDENSE / "glm52_notifications.py").read_bytes())
    producer = notifications.NotificationProducerSigner(b"P" * 32)
    receipt = notifications.NotificationReceiptSigner(
        b"R" * 32,
        expected_chat_identity_digest=CHAT_DIGEST,
        expected_bot_identity_digest=BOT_DIGEST,
        producer_verifier=producer.verifier,
    )
    reconciliation = notifications.NotificationReconciliationSigner(b"O" * 32)
    subject = {
        "journal_path": "GLM52_NOTIFICATION_OUTBOX.jsonl",
        "notification_module_path": "tools/condense/glm52_notifications.py",
        "notification_module_sha256": hashlib.sha256(
            module_target.read_bytes()
        ).hexdigest(),
        "producer_public_key": producer.verifier.public_key_hex,
        "receipt_public_key": receipt.verifier.public_key_hex,
        "reconciliation_public_key": reconciliation.verifier.public_key_hex,
        "expected_chat_identity_digest": CHAT_DIGEST,
        "expected_bot_identity_digest": BOT_DIGEST,
        "telegram_rotation_readiness_seal_sha256": rotation_seal,
        "network_sender_embedded_in_worker": False,
    }
    rotation = {
        "chat_identity_digest": CHAT_DIGEST,
        "notification_bot_identity_digest": BOT_DIGEST,
    }
    return subject, rotation, producer, receipt, reconciliation


def _anchored_notification_audit_readiness(
    tmp_path: pathlib.Path,
    runtime: gravity.Runtime,
) -> worker.NotificationAuditReadiness:
    subject, rotation, producer, receipt, reconciliation = _notification_fixture(
        tmp_path, runtime
    )
    controller = runtime.controller()
    controller.acquire()
    try:
        notifications.NotificationJournal(
            runtime.controller_root / subject["journal_path"],
            controller=controller,
            producer_signer=producer,
            receipt_verifier=receipt.verifier,
            reconciliation_verifier=reconciliation.verifier,
            expected_bot_identity_digest=BOT_DIGEST,
        )
    finally:
        controller.close()
    return worker._freeze_notification_audit_readiness(
        workspace_root=tmp_path,
        subject=subject,
        runtime=runtime,
        telegram_rotation_receipt_seal_sha256="d" * 64,
        telegram_rotation=rotation,
    )


def test_notification_static_readiness_never_claims_semantic_replay(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    subject, rotation, producer, receipt, reconciliation = _notification_fixture(
        tmp_path, runtime
    )
    detail = worker._validate_notification_outbox_static(
        subject,
        workspace_store=state.TrustedArtifactStore(tmp_path),
        runtime=runtime,
        telegram_rotation_receipt_seal_sha256="d" * 64,
        telegram_rotation=rotation,
    )
    assert detail["static_binding_verified"] is True
    assert detail["lease_bound_semantic_audit_verified"] is False
    assert detail["derived_head_path"] == "GLM52_NOTIFICATION_OUTBOX.jsonl.head.json"
    assert detail["producer_key_id"] == producer.key_id
    assert detail["receipt_key_id"] == receipt.key_id
    assert detail["reconciliation_key_id"] == reconciliation.key_id
    assert "journal_sha256" not in detail and "head_sha256" not in detail

    reused = dict(subject)
    reused["receipt_public_key"] = reused["producer_public_key"]
    with pytest.raises(worker.WorkerError, match="three distinct"):
        worker._validate_notification_outbox_static(
            reused,
            workspace_store=state.TrustedArtifactStore(tmp_path),
            runtime=runtime,
            telegram_rotation_receipt_seal_sha256="d" * 64,
            telegram_rotation=rotation,
        )

    wrong_bot = dict(subject)
    wrong_bot["expected_bot_identity_digest"] = "e" * 64
    with pytest.raises(worker.WorkerError, match="rotated Telegram identity"):
        worker._validate_notification_outbox_static(
            wrong_bot,
            workspace_store=state.TrustedArtifactStore(tmp_path),
            runtime=runtime,
            telegram_rotation_receipt_seal_sha256="d" * 64,
            telegram_rotation=rotation,
        )


def test_notification_semantic_audit_requires_and_uses_exact_held_lease(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    _boot(runtime)
    subject, rotation, producer, receipt, reconciliation = _notification_fixture(
        tmp_path, runtime
    )
    controller = runtime.controller()
    controller.acquire()
    try:
        notifications.NotificationJournal(
            runtime.controller_root / subject["journal_path"],
            controller=controller,
            producer_signer=producer,
            receipt_verifier=receipt.verifier,
            reconciliation_verifier=reconciliation.verifier,
            expected_bot_identity_digest=BOT_DIGEST,
        )
        detail = worker._audit_notification_outbox_under_lease(
            subject,
            workspace_store=state.TrustedArtifactStore(tmp_path),
            runtime=runtime,
            controller=controller,
            telegram_rotation_receipt_seal_sha256="d" * 64,
            telegram_rotation=rotation,
        )
        assert detail["lease_bound_semantic_audit_verified"] is True
        assert detail["ambiguous_intent_count"] == 0
        assert detail["entry_count"] == 0
    finally:
        controller.close()
    with pytest.raises(worker.WorkerError, match="lease is not held"):
        worker._audit_notification_outbox_under_lease(
            subject,
            workspace_store=state.TrustedArtifactStore(tmp_path),
            runtime=runtime,
            controller=controller,
            telegram_rotation_receipt_seal_sha256="d" * 64,
            telegram_rotation=rotation,
        )


def test_preflight_never_calls_unleased_notification_audit(
    tmp_path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path)
    _boot(runtime)
    config = _worker_config(tmp_path, runtime)

    monkeypatch.setattr(
        worker,
        "_validate_all_readiness",
        lambda *_args, **_kwargs: {kind: {} for kind in worker.READINESS_KINDS},
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("preflight attempted a lease-bound outbox audit")

    monkeypatch.setattr(notifications, "audit_notification_outbox", forbidden)
    report = worker.evaluate_preflight(
        config,
        runtime=runtime,
        contract=runtime.expected_contract,
        allow_test_profile=True,
    )
    readiness = report["checks"]["notification_outbox_ready"]
    assert readiness["status"] == "BLOCKED"
    assert readiness["reason"].startswith(
        "LEASE_BOUND_NOTIFICATION_OUTBOX_AUDIT_REQUIRED:"
    )
    assert report["production_start_authorized"] is False


def test_persistent_run_refuses_missing_audit_before_first_heartbeat(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    _boot(runtime)
    before = _checkpoint(runtime)
    instance = worker.PersistentWorker(
        runtime,
        instance_id="worker-missing-audit-0001",
    )
    with pytest.raises(worker.WorkerError, match="LEASE_BOUND_NOTIFICATION"):
        instance.run(max_cycles=1, install_signals=False)
    after = _checkpoint(runtime)
    assert after["event_count"] == before["event_count"]
    assert after.get("heartbeat") == before.get("heartbeat")
    assert instance.last_notification_audit is None
    assert instance.controller.lease.held is False


def test_persistent_run_audits_under_lease_at_each_dispatch_boundary(
    tmp_path,
    monkeypatch,
) -> None:
    runtime = _runtime(tmp_path)
    _boot(runtime)
    readiness = _anchored_notification_audit_readiness(tmp_path, runtime)
    event_count_before = _checkpoint(runtime)["event_count"]
    calls: list[int] = []
    real_audit = notifications.audit_notification_outbox

    def observed_audit(*args, **kwargs):
        controller = kwargs["controller"]
        controller.lease.assert_held()
        calls.append(controller.resume(recover_single_tail=False)["event_count"])
        return real_audit(*args, **kwargs)

    monkeypatch.setattr(notifications, "audit_notification_outbox", observed_audit)
    instance = worker.PersistentWorker(
        runtime,
        heartbeat_interval_seconds=0.0,
        waiter=lambda _seconds: False,
        instance_id="worker-audited-loop-0001",
        notification_audit_readiness=readiness,
    )
    result = instance.run(max_cycles=2, install_signals=False)
    assert result["status"] == "TEST_CYCLE_LIMIT"
    assert calls == [event_count_before, event_count_before + 1]
    assert instance.last_notification_audit is not None
    assert instance.last_notification_audit[
        "lease_bound_semantic_audit_verified"
    ] is True


def test_tampered_frozen_audit_readiness_blocks_before_heartbeat(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    _boot(runtime)
    readiness = _anchored_notification_audit_readiness(tmp_path, runtime)
    subject = json.loads(readiness.notification_subject_canonical)
    subject["expected_bot_identity_digest"] = "e" * 64
    tampered = replace(
        readiness,
        notification_subject_canonical=canonical(subject),
    )
    before = _checkpoint(runtime)
    instance = worker.PersistentWorker(
        runtime,
        instance_id="worker-tampered-audit-0001",
        notification_audit_readiness=tampered,
    )
    with pytest.raises(worker.WorkerError, match="rotated Telegram identity"):
        instance.run(max_cycles=1, install_signals=False)
    after = _checkpoint(runtime)
    assert after["event_count"] == before["event_count"]
    assert after.get("heartbeat") == before.get("heartbeat")


def test_notification_audit_bypass_is_synthetic_and_explicit_only(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    production_like = replace(runtime, allow_synthetic_contract=False)
    with pytest.raises(worker.WorkerError, match="LEASE_BOUND_NOTIFICATION"):
        worker.PersistentWorker(production_like)
    with pytest.raises(worker.WorkerError, match="restricted to synthetic"):
        worker.PersistentWorker(
            production_like,
            test_only_notification_audit_bypass=True,
        )


def test_worker_authentication_role_check_requires_distinct_grounding_without_leak(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    detail = worker._validate_runtime_authentication_roles(runtime)
    assert detail["pairwise_distinct"] is True
    rendered = json.dumps(detail, sort_keys=True)
    for authenticator in (
        runtime.telegram_auth,
        runtime.evidence_auth,
        runtime.grounding_auth,
    ):
        assert authenticator._key_material_identity() not in rendered

    object.__setattr__(
        runtime,
        "grounding_auth",
        grounding.ProducerAuthenticator(TELEGRAM_KEY),
    )
    with pytest.raises(worker.WorkerError, match="pairwise distinct"):
        worker._validate_runtime_authentication_roles(runtime)


def test_empty_reviewed_registry_hard_blocks_every_claimed_adapter(tmp_path) -> None:
    source = tmp_path / "adapter.py"
    source.write_text(
        "def execute_window(ctx):\n"
        "    result = ctx.execute_grounded_operation()\n"
        "    if result is None:\n"
        "        raise RuntimeError('grounded operation failed')\n"
        "    return result\n",
        encoding="utf-8",
    )
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    policy_digest = "9" * 64
    adapters = [{
        "adapter_id": "offline-grounded-adapter",
        "interface_version": worker.EXECUTION_ADAPTER_INTERFACE,
        "roles": sorted(worker.REQUIRED_EXECUTION_ROLES),
        "entry_points": {
            role: "execute_window" for role in sorted(worker.REQUIRED_EXECUTION_ROLES)
        },
        "source_path": "adapter.py",
        "source_sha256": source_sha,
        "resource_policy_digest": policy_digest,
    }]
    subject = {
        "adapters": adapters,
        "registered_dispatch_digest": hashlib.sha256(canonical(adapters)).hexdigest(),
        "resource_policy_digest": policy_digest,
    }
    assert dict(worker.PRODUCTION_EXECUTION_ADAPTER_REGISTRY) == {}
    with pytest.raises(
        worker.WorkerError,
        match="NO_REVIEWED_EXECUTION_ADAPTER_REGISTRY",
    ):
        worker._validate_execution_adapters(
            subject,
            workspace_store=state.TrustedArtifactStore(str(tmp_path)),
            resource_policy_digest=policy_digest,
        )


@pytest.mark.parametrize(
    ("source", "entry_point", "reason"),
    [
        (
            "def execute_window(ctx):\n    pass\n",
            "execute_window",
            "disabled/no-op",
        ),
        (
            "def execute_window(ctx):\n    return ctx\n",
            "execute_window",
            "disabled/no-op",
        ),
        (
            "def execute_window(ctx):\n    value = ctx\n    return value\n",
            "execute_window",
            "no grounded operation",
        ),
        (
            "record_import_side_effect()\n"
            "def execute_window(ctx):\n    return ctx.run()\n",
            "execute_window",
            "top-level effects",
        ),
        (
            "@register\n"
            "def execute_window(ctx):\n    return ctx.run()\n",
            "execute_window",
            "import-time defaults/decorators/annotations",
        ),
        (
            "def execute_window(ctx=construct_context()):\n    return ctx.run()\n",
            "execute_window",
            "import-time defaults/decorators/annotations",
        ),
        (
            "def execute_window(ctx: resolve_type()):\n    return ctx.run()\n",
            "execute_window",
            "import-time defaults/decorators/annotations",
        ),
        (
            "def _execute_eviction_test_only(ctx):\n    return ctx.evict()\n",
            "_execute_eviction_test_only",
            "non-production entry point",
        ),
        (
            "def execute_window(ctx):\n"
            "    result = _execute_eviction_test_only(ctx)\n"
            "    if result is None:\n"
            "        raise RuntimeError('eviction failed')\n"
            "    return result\n",
            "execute_window",
            "private/test code",
        ),
    ],
)
def test_adapter_source_inspection_rejects_noop_effects_and_test_eviction(
    source,
    entry_point,
    reason,
) -> None:
    with pytest.raises(worker.WorkerError, match=reason):
        worker._inspect_registered_adapter_source(
            source.encode("utf-8"),
            source_path="adapter.py",
            adapter_id="adversarial-adapter",
            entry_points={"EVICT_WINDOW": entry_point},
        )


def test_public_eviction_entry_point_remains_a_hard_blocker() -> None:
    source_path = TOOLS / "condense/glm52_window_execution.py"
    module_source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(module_source, filename=str(source_path))
    function = next(
        node for node in tree.body
        if getattr(node, "name", None) == "execute_eviction"
    )
    segment = ast.get_source_segment(module_source, function)
    assert function.name == "execute_eviction"
    assert segment is not None
    with pytest.raises(worker.WorkerError, match="disabled/no-op"):
        worker._inspect_registered_adapter_source(
            segment.encode("utf-8"),
            source_path=str(source_path),
            adapter_id="window-execution",
            entry_points={"EVICT_WINDOW": "execute_eviction"},
        )


def test_resource_policy_requires_exact_contract_binding_and_frozen_floor(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    source_policy = TOOLS.parent / worker.OFFICIAL_RESOURCE_POLICY_PATH
    destination = runtime.artifact_root / worker.OFFICIAL_RESOURCE_POLICY_PATH
    destination.write_bytes(source_policy.read_bytes())
    subject = {
        "policy_path": worker.OFFICIAL_RESOURCE_POLICY_PATH,
        "policy_file_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "policy_seal_sha256": worker.OFFICIAL_RESOURCE_POLICY_SEAL,
        "policy_schema": worker.OFFICIAL_RESOURCE_POLICY_SCHEMA,
        "resource_policy_digest": worker.OFFICIAL_RESOURCE_POLICY_SEAL,
        "enforced_at_every_adapter_boundary": True,
    }
    store = state.TrustedArtifactStore(str(runtime.artifact_root))
    with pytest.raises(worker.WorkerError, match="expected contract v3 does not bind"):
        worker._validate_resource_policy(subject, artifact_store=store, runtime=runtime)

    gates = _gates()
    gates["ASSEMBLE_ARTIFACT"]["required_artifacts"]["resource_reserve_policy"] = {
        "path": worker.OFFICIAL_RESOURCE_POLICY_PATH,
        "expected_seal_sha256": worker.OFFICIAL_RESOURCE_POLICY_SEAL,
        "expected_schema": worker.OFFICIAL_RESOURCE_POLICY_SCHEMA,
        "allowed_statuses": [worker.OFFICIAL_RESOURCE_POLICY_STATUS],
        "validator_id": "sealed_exact_v1",
        "validator_source_sha256": state.EVIDENCE_VALIDATOR_SOURCE_SHA256[
            "sealed_exact_v1"
        ],
        "require_producer_hmac": False,
    }
    bound_runtime = replace(runtime, expected_contract=_contract(gates=gates))
    detail = worker._validate_resource_policy(
        subject,
        artifact_store=state.TrustedArtifactStore(str(bound_runtime.artifact_root)),
        runtime=bound_runtime,
    )
    assert detail["policy_seal_sha256"] == worker.OFFICIAL_RESOURCE_POLICY_SEAL
    assert detail["required_free_disk_bytes"] == 416_036_394_619
    assert detail["expected_contract_bindings"] == [
        "ASSEMBLE_ARTIFACT.resource_reserve_policy"
    ]


def test_final_schedule_reopens_semantic_xet_and_binds_controller_commit(tmp_path) -> None:
    runtime, xet_result = _xet_runtime(tmp_path)
    _advance_to_build_adapter(runtime)
    binding = worker._controller_committed_xet_binding(runtime)
    subject, path, body = _write_final_schedule(
        runtime,
        xet_result,
        binding=binding,
    )
    detail = worker._validate_final_schedule(
        subject,
        artifact_store=state.TrustedArtifactStore(str(runtime.artifact_root)),
        runtime=runtime,
    )
    assert detail["live_xet_result_seal_sha256"] == xet_result["seal_sha256"]
    assert detail["controller_autotune_event_chain_sha256"] == binding[
        "controller_event_chain_sha256"
    ]
    assert detail["resource_policy_seal_sha256"] == worker.OFFICIAL_RESOURCE_POLICY_SEAL

    forged_body = dict(body)
    forged_body["resource_policy_binding"] = {
        **body["resource_policy_binding"],
        "required_free_disk_bytes": worker.OFFICIAL_REQUIRED_FREE_DISK_BYTES - 1,
    }
    forged = state.seal_producer_authenticated_evidence(
        forged_body, auth=runtime.evidence_auth
    )
    atomic_json(path, forged)
    forged_raw = path.read_bytes()
    forged_subject = {
        **subject,
        "schedule_file_sha256": hashlib.sha256(forged_raw).hexdigest(),
        "schedule_seal_sha256": forged["seal_sha256"],
    }
    with pytest.raises(worker.WorkerError, match="resource reserve policy"):
        worker._validate_final_schedule(
            forged_subject,
            artifact_store=state.TrustedArtifactStore(str(runtime.artifact_root)),
            runtime=runtime,
        )


def test_final_schedule_rejects_absent_tampered_and_arbitrary_xet_evidence(
    tmp_path,
) -> None:
    runtime, xet_result = _xet_runtime(tmp_path)
    _advance_to_build_adapter(runtime)
    binding = worker._controller_committed_xet_binding(runtime)
    subject, _path, _body = _write_final_schedule(
        runtime,
        xet_result,
        binding=binding,
    )
    store = state.TrustedArtifactStore(str(runtime.artifact_root))

    absent = {
        **subject,
        "live_xet_result_path": "absent-live-xet-result.json",
    }
    with pytest.raises((worker.WorkerError, state.StateError)):
        worker._validate_final_schedule(absent, artifact_store=store, runtime=runtime)

    arbitrary = {
        **subject,
        "live_xet_result_seal_sha256": "6" * 64,
    }
    with pytest.raises(worker.WorkerError, match="semantic snapshot"):
        worker._validate_final_schedule(arbitrary, artifact_store=store, runtime=runtime)

    result_path = runtime.artifact_root / subject["live_xet_result_path"]
    original = result_path.read_bytes()
    tampered = copy.deepcopy(xet_result)
    tampered["status"] = "FORGED_PASS"
    atomic_json(result_path, tampered)
    with pytest.raises(worker.WorkerError, match="bytes differ"):
        worker._validate_final_schedule(subject, artifact_store=store, runtime=runtime)
    result_path.write_bytes(original)


def test_final_schedule_rejects_semantic_result_without_controller_commit(tmp_path) -> None:
    runtime, xet_result = _xet_runtime(tmp_path)
    _boot(runtime)
    subject, _path, _body = _write_final_schedule(
        runtime,
        xet_result,
        binding={
            "controller_event_chain_sha256": "d" * 64,
            "terminal_evidence_seal_sha256": "e" * 64,
        },
    )
    with pytest.raises(worker.WorkerError, match="has not committed exactly one"):
        worker._validate_final_schedule(
            subject,
            artifact_store=state.TrustedArtifactStore(str(runtime.artifact_root)),
            runtime=runtime,
        )


def test_worker_config_rejects_symlink_hardlink_and_nondeterministic_timestamp(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    readiness = {
        kind: {"path": f"readiness/{kind}.json", "expected_file_sha256": "0" * 64}
        for kind in worker.READINESS_KINDS
    }
    auth = state.EvidenceAuthConfig(
        hmac_key=EVIDENCE_KEY,
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
    )
    document = worker.make_worker_config(
        profile=worker.SYNTHETIC_CONFIG_PROFILE,
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
        controller_config_path=runtime.config_path,
        workspace_root=tmp_path,
        expected_contract_sha256=runtime.expected_contract["seal_sha256"],
        expected_contract_created_at=CREATED_AT,
        readiness=readiness,
        evidence_auth=auth,
    )
    config_path = tmp_path / "worker.json"
    atomic_json(config_path, document)
    assert worker.load_worker_config(config_path).seal_sha256 == document["seal_sha256"]

    symlink = tmp_path / "worker-link.json"
    symlink.symlink_to(config_path)
    with pytest.raises(worker.WorkerError, match="regular file"):
        worker.load_worker_config(symlink)
    hardlink = tmp_path / "worker-hardlink.json"
    hardlink.hardlink_to(config_path)
    with pytest.raises(worker.WorkerError, match="one hard link"):
        worker.load_worker_config(config_path)
    hardlink.unlink()

    bad = dict(document)
    bad["expected_contract_created_at"] = "NOW"
    bad.pop("seal_sha256")
    with pytest.raises(worker.WorkerError, match="deterministic UTC"):
        worker._parse_worker_config_document(seal(bad), config_path)


def test_worker_config_replacement_cannot_reseal_an_alternate_contract(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    config = _worker_config(tmp_path, runtime)
    valid = worker.evaluate_preflight(
        config,
        runtime=runtime,
        contract=runtime.expected_contract,
        allow_test_profile=True,
    )
    assert valid["checks"]["worker_config_authority"]["status"] == "PASS"

    replacement = copy.deepcopy(config.document)
    replacement.pop("seal_sha256")
    replacement["expected_contract_sha256"] = "a" * 64
    attacker_resealed = worker._parse_worker_config_document(
        seal(replacement),
        tmp_path / "attacker-resealed-worker.json",
    )
    report = worker.evaluate_preflight(
        attacker_resealed,
        runtime=runtime,
        contract=runtime.expected_contract,
        allow_test_profile=True,
    )
    authority = report["checks"]["worker_config_authority"]
    assert authority["status"] == "BLOCKED"
    assert "producer HMAC failed" in authority["reason"]
    assert report["production_start_authorized"] is False


def test_worker_config_binds_exact_controller_bytes_seal_and_runtime_targets(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    config = _worker_config(tmp_path, runtime)
    raw = runtime.config_path.read_bytes()
    assert config.controller_config_file_sha256 == hashlib.sha256(raw).hexdigest()
    assert config.controller_config_seal_sha256 == runtime.config_sha256
    assert config.controller_root == runtime.controller_root
    assert config.artifact_root == runtime.artifact_root
    assert config.expected_contract_path == runtime.expected_contract_path

    spoofed_runtime = replace(runtime, config_sha256="f" * 64)
    report = worker.evaluate_preflight(
        config,
        runtime=spoofed_runtime,
        contract=runtime.expected_contract,
        allow_test_profile=True,
    )
    authority = report["checks"]["worker_config_authority"]
    assert authority["status"] == "BLOCKED"
    assert "loaded runtime differs" in authority["reason"]


def test_same_path_controller_reseal_and_root_redirection_is_rejected(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    config = _worker_config(tmp_path, runtime)
    redirected_controller = tmp_path / "redirected-controller"
    redirected_artifacts = tmp_path / "redirected-artifacts"
    redirected_contract = tmp_path / "redirected-contract.json"
    attacker_config = gravity.make_config(
        campaign_id=runtime.campaign_id,
        source_revision=runtime.source_revision,
        controller_root=redirected_controller,
        artifact_root=redirected_artifacts,
        expected_contract_path=redirected_contract,
        phone_status_directory=redirected_artifacts / "phone",
        expected_chat_identity_digest=CHAT_DIGEST,
        allow_synthetic_contract=True,
        controller_epoch=runtime.controller_epoch,
    )
    atomic_json(runtime.config_path, attacker_config)
    attacker_runtime = replace(
        runtime,
        config_sha256=attacker_config["seal_sha256"],
        controller_root=redirected_controller,
        artifact_root=redirected_artifacts,
        expected_contract_path=redirected_contract,
        phone_status_directory=redirected_artifacts / "phone",
    )
    report = worker.evaluate_preflight(
        config,
        runtime=attacker_runtime,
        contract=runtime.expected_contract,
        allow_test_profile=True,
    )
    authority = report["checks"]["worker_config_authority"]
    assert authority["status"] == "BLOCKED"
    assert "controller config bytes/targets differ" in authority["reason"]
    assert "file_sha256" in authority["reason"]
    assert report["production_start_authorized"] is False


def test_authenticated_worker_target_substitution_cannot_override_controller_file(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    config = _worker_config(tmp_path, runtime)
    body = {
        key: value for key, value in config.document.items()
        if key not in {"seal_sha256", "producer_hmac_sha256"}
    }
    body["controller_root"] = os.fspath(tmp_path / "forged-controller")
    substituted = state.seal_producer_authenticated_evidence(
        body,
        auth=runtime.evidence_auth,
    )
    parsed = worker._parse_worker_config_document(
        substituted,
        tmp_path / "substituted-worker.json",
    )
    report = worker.evaluate_preflight(
        parsed,
        runtime=runtime,
        contract=runtime.expected_contract,
        allow_test_profile=True,
    )
    authority = report["checks"]["worker_config_authority"]
    assert authority["status"] == "BLOCKED"
    assert "controller config bytes/targets differ" in authority["reason"]
    assert "controller_root" in authority["reason"]


def test_unset_official_contract_pin_blocks_any_authenticated_alternate_hash(
    tmp_path,
) -> None:
    readiness = {
        kind: {
            "path": f"readiness/{kind}.json",
            "expected_file_sha256": "0" * 64,
        }
        for kind in worker.READINESS_KINDS
    }
    official_auth = state.EvidenceAuthConfig(
        hmac_key=b"official-worker-config-test-hmac-material-0004",
        campaign_id=worker.OFFICIAL_CAMPAIGN_ID,
        source_revision=worker.OFFICIAL_REVISION,
    )
    official_controller_path = tmp_path / "official-controller.json"
    atomic_json(
        official_controller_path,
        gravity.make_config(
            campaign_id=worker.OFFICIAL_CAMPAIGN_ID,
            source_revision=worker.OFFICIAL_REVISION,
            controller_root=tmp_path / "official-controller",
            artifact_root=tmp_path / "official-artifacts",
            expected_contract_path=tmp_path / "official-contract.json",
            phone_status_directory=tmp_path / "official-phone",
            expected_chat_identity_digest=CHAT_DIGEST,
        ),
    )
    document = worker.make_worker_config(
        profile=worker.OFFICIAL_CONFIG_PROFILE,
        campaign_id=worker.OFFICIAL_CAMPAIGN_ID,
        source_revision=worker.OFFICIAL_REVISION,
        controller_config_path=official_controller_path,
        workspace_root=tmp_path,
        expected_contract_sha256="c" * 64,
        expected_contract_created_at=CREATED_AT,
        readiness=readiness,
        evidence_auth=official_auth,
    )
    config = worker._parse_worker_config_document(
        document,
        tmp_path / "official-worker.json",
    )
    assert worker.OFFICIAL_CONTRACT_SEAL is None
    report = worker.evaluate_preflight(
        config,
        runtime=None,
        runtime_error="offline test",
    )
    pin = report["checks"]["official_contract_authority"]
    assert pin["status"] == "BLOCKED"
    assert pin["reason"].startswith("OFFICIAL_CONTRACT_SEAL_UNSET")
    assert report["production_start_authorized"] is False


@pytest.mark.parametrize(
    ("command", "ok", "reason"),
    [
        ("/usr/bin/caffeinate -dimsu /usr/bin/python3 worker.py", True, None),
        ("/usr/bin/caffeinate -di /usr/bin/python3 worker.py", False, "lacks required flags"),
        ("/usr/bin/python3 worker.py", False, "not /usr/bin/caffeinate"),
        ("/tmp/caffeinate -dimsu worker.py", False, "path is not /usr/bin/caffeinate"),
    ],
)
def test_caffeinate_parent_validation(command, ok, reason) -> None:
    def runner(args, **_kwargs):
        is_comm = args[-1] == "comm="
        comm = pathlib.Path(command.split()[0]).name
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=f"{comm if is_comm else command}\n",
            stderr="",
        )

    result = worker.inspect_caffeinate_parent(parent_pid=4242, runner=runner)
    assert result["ok"] is ok
    if reason:
        assert reason in result["reason"]


def test_caffeinate_parent_exact_argv_rejects_any_program_drift() -> None:
    actual = "/usr/bin/caffeinate -dimsu /usr/bin/python3 /tmp/worker.py"

    def runner(args, **_kwargs):
        value = "caffeinate" if args[-1] == "comm=" else actual
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=f"{value}\n", stderr=""
        )

    assert worker.inspect_caffeinate_parent(
        parent_pid=4242,
        runner=runner,
        expected_argv=actual.split(),
    )["ok"] is True
    rejected = worker.inspect_caffeinate_parent(
        parent_pid=4242,
        runner=runner,
        expected_argv=[
            "/usr/bin/caffeinate",
            "-dimsu",
            "/usr/bin/python3",
            "/approved/worker.py",
        ],
    )
    assert rejected["ok"] is False
    assert "exact approved invocation" in rejected["reason"]


def test_launchd_plist_is_caffeinated_restartable_and_secret_free() -> None:
    plist_path = TOOLS.parent / "deploy/launchd/com.hawking.glm52.gravity.plist"
    value = worker.validate_launchd_plist(plist_path)
    assert value["KeepAlive"] == {"Crashed": True, "SuccessfulExit": False}
    assert value["RunAtLoad"] is True
    assert value["ProgramArguments"] == list(worker.PINNED_LAUNCHD_ARGUMENTS)
    assert set(value) == worker.PINNED_LAUNCHD_KEYS
    assert value["ProgramArguments"][2] == str(worker.PINNED_PYTHON)
    assert value["ProgramArguments"][3] == str(worker.PINNED_WORKER)
    assert value["ProgramArguments"][5] == str(worker.PINNED_WORKER_CONFIG)
    raw = plist_path.read_text(encoding="utf-8").lower()
    assert "telegram" not in raw
    assert "token" not in raw
    assert "secret" not in raw
    plistlib.loads(plist_path.read_bytes())


def test_launchd_plist_rejects_injection_and_every_argument_drift(tmp_path) -> None:
    original_path = TOOLS.parent / "deploy/launchd/com.hawking.glm52.gravity.plist"
    original = plistlib.loads(original_path.read_bytes())

    def environment_injection(value):
        value["EnvironmentVariables"] = {
            "DYLD_INSERT_LIBRARIES": "/tmp/inject.dylib",
            "PYTHONPATH": "/tmp/import-shadow",
        }

    def wrong_flags(value):
        value["ProgramArguments"][1] = "-di"

    def unpinned_python(value):
        value["ProgramArguments"][2] = "/usr/bin/python3"

    def wrong_worker(value):
        value["ProgramArguments"][3] = "/tmp/worker.py"

    def wrong_config(value):
        value["ProgramArguments"][5] = "/tmp/replacement-config.json"

    def append_python_injection(value):
        value["ProgramArguments"].extend(["-c", "raise SystemExit(0)"])

    def extra_launchd_key(value):
        value["LimitLoadToSessionType"] = "Aqua"

    def integer_boolean_substitution(value):
        value["RunAtLoad"] = 1
        value["KeepAlive"]["Crashed"] = 1
        value["KeepAlive"]["SuccessfulExit"] = 0

    for index, mutate in enumerate((
        environment_injection,
        wrong_flags,
        unpinned_python,
        wrong_worker,
        wrong_config,
        append_python_injection,
        extra_launchd_key,
        integer_boolean_substitution,
    )):
        malicious = copy.deepcopy(original)
        mutate(malicious)
        candidate = tmp_path / f"mutated-{index}.plist"
        candidate.write_bytes(plistlib.dumps(malicious, sort_keys=True))
        with pytest.raises(worker.WorkerError):
            worker.validate_launchd_plist(candidate)

    marker = "  <key>ProgramArguments</key>"
    malicious_duplicate = (
        "  <key>ProgramArguments</key>\n"
        "  <array><string>/tmp/injected-worker</string></array>\n\n"
        f"{marker}"
    )
    duplicate_path = tmp_path / "duplicate-program-arguments.plist"
    duplicate_path.write_text(
        original_path.read_text(encoding="utf-8").replace(
            marker,
            malicious_duplicate,
            1,
        ),
        encoding="utf-8",
    )
    assert plistlib.loads(duplicate_path.read_bytes())["ProgramArguments"] == (
        original["ProgramArguments"]
    )
    with pytest.raises(worker.WorkerError, match="duplicate or unapproved"):
        worker.validate_launchd_plist(duplicate_path)


def test_status_documents_logout_and_reboot_limitations(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    config = _worker_config(tmp_path, runtime)
    report = {
        "production_start_authorized": False,
        "blockers": [{"code": "test", "reason": "blocked"}],
    }
    status = worker.build_status(config, report, runtime=None)
    assert status["status"] == "BLOCKED"
    assert "logout" in status["session_limitations"]
    assert "reboot" in status["session_limitations"]
    assert "LaunchAgent" in status["session_limitations"]
