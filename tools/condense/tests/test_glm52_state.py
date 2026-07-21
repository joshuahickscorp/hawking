#!/usr/bin/env python3.12
"""Offline adversarial tests for the GLM-5.2 durable state spine."""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import sys

import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_state as gs  # noqa: E402
from glm52_common import atomic_json, seal  # noqa: E402


REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
CAMPAIGN = "glm52-bf16-xet-gravity-test"
HASH_A = "a" * 64
HASH_B = "b" * 64
SHARD = "model-00001-of-00282.safetensors"
TENSOR = "model.layers.0.self_attn.q_proj.weight"
CHAT_ID = -100424242
CHAT_DIGEST = gs.telegram_chat_identity_digest(CHAT_ID)
HMAC_KEY = b"offline-test-only-hmac-key-32-bytes-minimum!!"


def _source_shard(path=SHARD, logical_bytes=5_000_000_000, content_hash=HASH_A):
    return {
        "path": path,
        "logical_bytes": logical_bytes,
        "content_hash": content_hash,
        "content_hash_kind": "xet",
    }


def _state_gates():
    def policy(
        path, schema, statuses=("PASS",), expected=None, validator="campaign_artifact_v1"
    ):
        return {
            "path": path,
            "expected_seal_sha256": expected,
            "expected_schema": schema,
            "allowed_statuses": list(statuses),
            "validator_id": validator,
            "validator_source_sha256": gs.EVIDENCE_VALIDATOR_SOURCE_SHA256[validator],
            "require_producer_hmac": validator in {
                "campaign_artifact_v1", "stop_condition_v1"
            },
        }

    return {
        "ASSEMBLE_ARTIFACT": {
            "require_source_complete": True,
            "require_tensor_complete": True,
            "require_final_source_eviction": True,
            "require_telegram_delivery": True,
            "require_phone_status": False,
            "required_phone_status_path": None,
            "required_artifacts": {
                "source_manifest": policy(
                    "reports/GLM52_OFFICIAL_MANIFEST.json", "test.source_manifest.v1"
                )
            },
            "required_checklist": {
                item: policy(
                    f"evidence/{item}.json",
                    gs.STOP_CONDITION_EVIDENCE_SCHEMA,
                    validator="stop_condition_v1",
                )
                for item in ("source_stream_complete", "tensor_coverage_complete")
            },
        },
        "COMPLETE": {
            "require_source_complete": True,
            "require_tensor_complete": True,
            "require_final_source_eviction": True,
            "require_telegram_delivery": True,
            "require_phone_status": True,
            "required_phone_status_path": "reports/GLM52_PHONE_STATUS.json",
            "required_artifacts": {
                "final_result": policy(
                    "reports/GLM52_GRAVITY_FINAL.json", "test.final_result.v1"
                ),
                "gravity_audit": policy(
                    "reports/GRAVITY_COMPLETENESS_AUDIT_GLM52_FINAL.json",
                    "test.gravity_audit.v1",
                ),
            },
            "required_checklist": {
                item: policy(
                    f"evidence/{item}.json",
                    gs.STOP_CONDITION_EVIDENCE_SCHEMA,
                    validator="stop_condition_v1",
                )
                for item in (
                    "source_evicted", "all_artifacts_sealed", "phone_current",
                    "stop_conditions_met",
                )
            },
        },
    }


def _contract():
    shard = _source_shard()
    return gs.make_expected_campaign_contract(
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
        expected_chat_identity_digest=CHAT_DIGEST,
        source_shards=[shard],
        expected_tensors=[TENSOR],
        window_schedule=[
            {
                "schedule_index": 0,
                "window_id": "window-0001",
                "source_shards": [SHARD],
                "carry_in_shards": [],
                "new_fetch_shards": [SHARD],
                "refetch_shards": [],
                "carry_out_shards": [],
                "evict_shards": [SHARD],
                "tensor_set": [TENSOR],
            }
        ],
        state_gates=_state_gates(),
        source_profile="SYNTHETIC_TEST_ONLY",
        created_at="2026-07-21T00:00:00Z",
    )


def _auth():
    return gs.TelegramAuthConfig(
        hmac_key=HMAC_KEY,
        expected_chat_identity_digest=CHAT_DIGEST,
    )


def _wrap_xet_raw_result(raw, contract):
    semantic = {
        "raw_xet_autotune_result_seal_sha256": raw["seal_sha256"],
        "controller_anchor_sha256": HASH_B,
    }
    body = {
        key: copy.deepcopy(value)
        for key, value in raw.items()
        if key != "seal_sha256"
    }
    body.update({
        "campaign_id": contract["campaign_id"],
        "source_revision": contract["source_revision"],
        "expected_contract_sha256": contract["seal_sha256"],
        "evidence": semantic,
        "evidence_sha256": gs._sha256(semantic),
    })
    return gs.seal_producer_authenticated_evidence(body, auth=_auth())


def _xet_result_for_plan(plan, contract):
    import glm52_xet_live as live

    trial_ids = [row["trial_id"] for row in plan["trial_matrix"]]
    selected_trial = plan["trial_matrix"][0]

    def selection(lane):
        return {
            "lane": lane,
            "status": "SELECTED",
            "trial_id": selected_trial["trial_id"],
            "selected_trial": {"trial_id": selected_trial["trial_id"]},
            "selected_caller_concurrent_shard_streams": selected_trial[
                "caller_concurrent_shard_streams"
            ],
            "post_autotune_schedule_refreeze_required": True,
        }

    selections = {
        "acquisition": selection("acquisition"),
        "steady": selection("steady"),
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
        "schema": live.AUTOTUNE_RESULT_SCHEMA,
        "status": "PASS_LIVE_XET_AUTOTUNE_COMPLETE_SCHEDULE_REFREEZE_REQUIRED",
        "repo": live.REPO_ID,
        "revision": live.REVISION,
        "bindings": {
            "plan_seal_sha256": plan["seal_sha256"],
            "plan_toolchain_binding_sha256": gs._sha256(plan["toolchain_binding"]),
            "plan_input_refs_sha256": gs._sha256(plan["inputs"]),
            "source_refs": source_refs,
            "live_executor_sha256": hashlib.sha256(
                pathlib.Path(live.__file__).read_bytes()
            ).hexdigest(),
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
                "selected_trial_id": selected_trial["trial_id"],
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
    return raw, _wrap_xet_raw_result(raw, contract)


def _controller(tmp_path):
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    return gs.Controller(
        tmp_path / "controller",
        artifact_root=tmp_path / "artifacts",
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
        expected_contract=_contract(),
        telegram_auth=_auth(),
        allow_synthetic_contract=True,
    )


def _delivery(controller, state, claim, payload=None, message_id=None):
    intent = controller.prepare_transition(state, claim_id=claim, payload=payload)
    key = intent["dedupe_key"]
    if message_id is None:
        import hashlib
        message_id = int(hashlib.sha256(claim.encode("utf-8")).hexdigest()[:12], 16) + 1
    response = {
        "ok": True,
        "result": {
            "message_id": message_id,
            "chat": {"id": CHAT_ID, "type": "private"},
            "text": intent["rendered_message"],
        },
    }
    return gs.make_telegram_delivery_receipt(
        intent,
        auth=controller.telegram_auth,
        bot_api_response=response,
        http_status=200,
        delivered_at="2026-07-21T00:00:01Z",
    )


def _boot(controller):
    claim = "test:boot:0001"
    return controller.boot(
        claim_id=claim,
        telegram_delivery=_delivery(controller, "PRECHECK", claim),
    )


def _transition(controller, state, number, payload=None):
    claim = f"test:transition:{number:04d}"
    return controller.transition(
        state,
        claim_id=claim,
        payload=payload,
        telegram_delivery=_delivery(controller, state, claim, payload),
    )


def _reach_freeze(controller):
    _boot(controller)
    for number, state in enumerate(
        [
            "CLOSE_KIMI", "RELEASE_KIMI_SOURCE", "ADMIT_GLM_SOURCE", "BUILD_MANIFEST",
            "BUILD_DEPENDENCY_GRAPH", "AUTOTUNE_XET", "BUILD_ADAPTER", "BUILD_REFERENCE",
            "BUILD_CORPUS", "PILOT_ORACLES", "FREEZE_PROGRAM",
        ],
        1,
    ):
        _transition(controller, state, number)


def _window_args(claim="test:window:declare:0001", tensor=TENSOR, window="window-0001"):
    return {
        "window_id": window,
        "schedule_index": 0,
        "source_shards": [_source_shard()],
        "carry_in_shards": [],
        "new_fetch_shards": [SHARD],
        "refetch_shards": [],
        "carry_out_shards": [],
        "evict_shards": [SHARD],
        "tensor_set": [tensor],
        "layer_organ_dependencies": [
            {"layer": 0, "organ": "attention", "dependency": "q_projection"}
        ],
        "disk_before": {"free_bytes": 600_000_000_000, "allocated_bytes": 20_000_000},
        "claim_id": claim,
    }


def _advance(ledger, phase, number, *, patch=None, source=None, tensor=None):
    return ledger.advance(
        "window-0001",
        phase,
        claim_id=f"test:window:advance:{number:04d}",
        patch=patch,
        source_coverage=source,
        tensor_coverage=tensor,
    )


def _complete_window(ledger):
    _advance(
        ledger,
        "FETCHING",
        1,
        patch={"download_start": "2026-07-21T12:00:00Z"},
        source={SHARD: "FETCHING"},
    )
    _advance(
        ledger,
        "FETCHED",
        2,
        patch={
            "download_end": "2026-07-21T12:01:00Z",
            "bytes_transferred": 5_000_000_100,
            "transfer_accounting": {
                "new_fetch_network_bytes": 5_000_000_000,
                "refetch_network_bytes": 0,
                "protocol_overhead_bytes": 100,
            },
        },
        source={SHARD: "FETCHED"},
    )
    _advance(
        ledger,
        "VERIFIED",
        3,
        patch={"hash_verification": {"status": "VERIFIED", "verified_shards": [SHARD]}},
        source={SHARD: "HASH_VERIFIED"},
        tensor={TENSOR: "SOURCE_VERIFIED"},
    )
    _advance(
        ledger,
        "TEACHER_CAPTURED",
        4,
        patch={"teacher_evidence_produced": [{"path": "teacher/w1.json", "sha256": HASH_A}]},
        tensor={TENSOR: "TEACHER_EVIDENCED"},
    )
    _advance(ledger, "CANDIDATES_FIT", 5, patch={"metrics": {"fit_loss": 0.01}})
    _advance(
        ledger,
        "CANDIDATES_PACKED",
        6,
        patch={"candidate_payloads_produced": [{"path": "candidate/w1.bin", "sha256": HASH_B}]},
        tensor={TENSOR: "CANDIDATE_PACKED"},
    )
    _advance(
        ledger,
        "FORWARD_COMPLETE",
        7,
        patch={"metrics": {"fit_loss": 0.01, "forward_cosine": 0.999}},
        tensor={TENSOR: "FORWARD_VERIFIED"},
    )
    _advance(
        ledger,
        "SEALED",
        8,
        patch={
            "compact_shard_hashes": {"compact/core-0001.bin": HASH_B},
            "terminal_coverage_evidence": {
                TENSOR: {
                    "disposition": "PACKED_IN_CORE_ARTIFACT",
                    "evidence_sha256": HASH_A,
                }
            },
        },
        source={SHARD: "CONSUMED"},
        tensor={TENSOR: "PACKED_IN_CORE_ARTIFACT"},
    )
    return _advance(
        ledger,
        "EVICTED",
        9,
        patch={
            "source_eviction": {
                "status": "EVICTED", "evicted_shards": [SHARD], "receipt_sha256": HASH_B,
            },
            "disk_after": {"free_bytes": 605_000_000_000, "allocated_bytes": 20_000_000},
        },
        source={SHARD: "EVICTED"},
    )


def _terminal_evidence(controller, state):
    gate = controller.expected_contract["state_gates"][state]
    root = controller.artifact_root
    for name, spec in gate["required_artifacts"].items():
        semantic = {"artifact": name, "test": True}
        body = {
            "schema": spec["expected_schema"],
            "status": spec["allowed_statuses"][0],
            "campaign_id": controller.campaign_id,
            "source_revision": controller.source_revision,
            "expected_contract_sha256": controller.expected_contract_sha256,
            "evidence": semantic,
            "evidence_sha256": gs._sha256(semantic),
        }
        value = (
            gs.seal_producer_authenticated_evidence(body, auth=controller.telegram_auth)
            if spec["require_producer_hmac"] else seal(body)
        )
        atomic_json(root / spec["path"], value)
    for item, spec in gate["required_checklist"].items():
        semantic = {"test": True}
        body = {
            "schema": spec["expected_schema"],
            "status": "PASS",
            "campaign_id": controller.campaign_id,
            "source_revision": controller.source_revision,
            "expected_contract_sha256": controller.expected_contract_sha256,
            "stop_condition": item,
            "evidence": semantic,
            "evidence_sha256": gs._sha256(semantic),
        }
        value = gs.seal_producer_authenticated_evidence(
            body, auth=controller.telegram_auth
        )
        atomic_json(root / spec["path"], value)
    checkpoint = controller.resume()
    anchor = controller._controller_anchor(checkpoint)
    if gate["require_phone_status"]:
        phone_body = {
            "schema": "hawking.glm52.phone_status.v2",
            "status": "GREEN",
            "overall_status": "GREEN",
            "campaign_id": controller.campaign_id,
            "source_revision": controller.source_revision,
            "controller_epoch": controller.controller_epoch,
            "expected_contract_sha256": controller.expected_contract_sha256,
            "cli_config_sha256": HASH_A,
            "controller": {
                "durable_state_ok": True,
                "live_worker_lease_ok": True,
                "heartbeat_fresh_ok": True,
                "heartbeat_max_age_seconds": gs.CONTROLLER_HEARTBEAT_MAX_AGE_SECONDS,
                "heartbeat_at": checkpoint["heartbeat"]["at"],
            },
            "operator_control": {
                "application_state": "IN_SYNC",
                "requested_sequence": 7,
                "requested_action": "RESUME",
                "applied": {
                    "applied_request_sequence": 7,
                    "applied_action": "RESUME",
                },
            },
            "checkpoint_anchor": anchor["checkpoint"],
        }
        atomic_json(
            root / gate["required_phone_status_path"],
            gs.seal_producer_authenticated_evidence(
                phone_body, auth=controller.telegram_auth
            ),
        )
    return gs.make_state_terminal_evidence(
        controller.expected_contract,
        state,
        artifact_root=root,
        controller_anchor=anchor,
        evidence_auth=controller.telegram_auth,
        created_at="2026-07-21T12:10:00Z",
    )


def _advance_controller_window(controller, phase, number, *, patch=None, source=None, tensor=None):
    return controller.advance_window(
        "window-0001",
        phase,
        claim_id=f"test:controller-window:advance:{number:04d}",
        patch=patch,
        source_coverage=source,
        tensor_coverage=tensor,
    )


def _run_single_controller_window(controller):
    _transition(controller, "FETCH_WINDOW", 100, {"window_id": "window-0001"})
    _advance_controller_window(
        controller,
        "FETCHING",
        1,
        patch={"download_start": "2026-07-21T12:00:00Z"},
        source={SHARD: "FETCHING"},
    )
    _advance_controller_window(
        controller,
        "FETCHED",
        2,
        patch={
            "download_end": "2026-07-21T12:01:00Z",
            "bytes_transferred": 5_000_000_100,
            "transfer_accounting": {
                "new_fetch_network_bytes": 5_000_000_000,
                "refetch_network_bytes": 0,
                "protocol_overhead_bytes": 100,
            },
        },
        source={SHARD: "FETCHED"},
    )
    _transition(controller, "VERIFY_WINDOW", 101, {"window_id": "window-0001"})
    _advance_controller_window(
        controller,
        "VERIFIED",
        3,
        patch={"hash_verification": {"status": "VERIFIED", "verified_shards": [SHARD]}},
        source={SHARD: "HASH_VERIFIED"},
        tensor={TENSOR: "SOURCE_VERIFIED"},
    )
    _transition(controller, "CAPTURE_TEACHER", 102, {"window_id": "window-0001"})
    _advance_controller_window(
        controller,
        "TEACHER_CAPTURED",
        4,
        patch={"teacher_evidence_produced": [{"path": "teacher/w1.json", "sha256": HASH_A}]},
        tensor={TENSOR: "TEACHER_EVIDENCED"},
    )
    _transition(controller, "FIT_CANDIDATES", 103, {"window_id": "window-0001"})
    _advance_controller_window(
        controller, "CANDIDATES_FIT", 5, patch={"metrics": {"fit_loss": 0.01}}
    )
    _transition(controller, "PACK_CANDIDATES", 104, {"window_id": "window-0001"})
    _advance_controller_window(
        controller,
        "CANDIDATES_PACKED",
        6,
        patch={"candidate_payloads_produced": [{"path": "candidate/w1.bin", "sha256": HASH_B}]},
        tensor={TENSOR: "CANDIDATE_PACKED"},
    )
    _transition(controller, "RUN_WINDOW_FORWARD", 105, {"window_id": "window-0001"})
    _advance_controller_window(
        controller,
        "FORWARD_COMPLETE",
        7,
        patch={"metrics": {"fit_loss": 0.01, "forward_cosine": 0.999}},
        tensor={TENSOR: "FORWARD_VERIFIED"},
    )
    _transition(controller, "SEAL_WINDOW", 106, {"window_id": "window-0001"})
    _advance_controller_window(
        controller,
        "SEALED",
        8,
        patch={
            "compact_shard_hashes": {"compact/core-0001.bin": HASH_B},
            "terminal_coverage_evidence": {
                TENSOR: {
                    "disposition": "PACKED_IN_CORE_ARTIFACT",
                    "evidence_sha256": HASH_A,
                }
            },
        },
        source={SHARD: "CONSUMED"},
        tensor={TENSOR: "PACKED_IN_CORE_ARTIFACT"},
    )
    _transition(controller, "EVICT_WINDOW", 107, {"window_id": "window-0001"})
    _advance_controller_window(
        controller,
        "EVICTED",
        9,
        patch={
            "source_eviction": {
                "status": "EVICTED", "evicted_shards": [SHARD], "receipt_sha256": HASH_B,
            },
            "disk_after": {"free_bytes": 605_000_000_000, "allocated_bytes": 20_000_000},
        },
        source={SHARD: "EVICTED"},
    )


def test_exact_part_vi_state_list_and_selfcheck():
    assert gs.STATES == (
        "PRECHECK", "CLOSE_KIMI", "RELEASE_KIMI_SOURCE", "ADMIT_GLM_SOURCE",
        "BUILD_MANIFEST", "BUILD_DEPENDENCY_GRAPH", "AUTOTUNE_XET", "BUILD_ADAPTER",
        "BUILD_REFERENCE", "BUILD_CORPUS", "PILOT_ORACLES", "FREEZE_PROGRAM",
        "FETCH_WINDOW", "VERIFY_WINDOW", "CAPTURE_TEACHER", "FIT_CANDIDATES",
        "PACK_CANDIDATES", "RUN_WINDOW_FORWARD", "SEAL_WINDOW", "EVICT_WINDOW",
        "ASSEMBLE_ARTIFACT", "VERIFY_ARTIFACT", "RUN_FULL_COMPACT",
        "RUN_DOCTOR_REFINEMENT", "RUN_RATE_DESCENT", "SEAL_GLM_RESULT",
        "FINAL_GRAVITY_AUDIT", "COMPLETE", "BLOCKED",
    )
    assert gs.selfcheck()["status"] == "PASS"


def test_xet_result_validator_reconstructs_raw_result_and_rejects_signed_fabrication(
    tmp_path,
):
    repo_root = CONDENSE.parents[1]
    plan = json.loads(
        (repo_root / "GLM52_XET_AUTOTUNE_PLAN.json").read_text(encoding="utf-8")
    )
    plan_policy = {
        "path": "GLM52_XET_AUTOTUNE_PLAN.json",
        "expected_seal_sha256": plan["seal_sha256"],
        "expected_schema": "hawking.glm52.xet_autotune_plan.v2",
        "allowed_statuses": ["PASS_OFFLINE_PLAN_BODY_NOT_READ"],
        "validator_id": "sealed_exact_v1",
        "validator_source_sha256": gs.EVIDENCE_VALIDATOR_SOURCE_SHA256[
            "sealed_exact_v1"
        ],
        "require_producer_hmac": False,
    }
    contract_body = copy.deepcopy(_contract())
    contract_body.pop("seal_sha256")
    contract_body["state_gates"]["AUTOTUNE_XET"] = {
        "require_source_complete": False,
        "require_tensor_complete": False,
        "require_final_source_eviction": False,
        "require_telegram_delivery": True,
        "require_phone_status": False,
        "required_phone_status_path": None,
        "required_artifacts": {"xet_autotune_plan": plan_policy},
        "required_checklist": {},
    }
    contract = seal(contract_body)
    policy = {
        "path": "GLM52_XET_AUTOTUNE.json",
        "expected_seal_sha256": None,
        "expected_schema": "hawking.glm52.xet_autotune_result.v1",
        "allowed_statuses": [
            "PASS_LIVE_XET_AUTOTUNE_COMPLETE_SCHEDULE_REFREEZE_REQUIRED"
        ],
        "validator_id": "xet_autotune_result_v1",
        "validator_source_sha256": gs.EVIDENCE_VALIDATOR_SOURCE_SHA256[
            "xet_autotune_result_v1"
        ],
        "require_producer_hmac": True,
    }
    atomic_json(tmp_path / plan_policy["path"], plan)
    raw, wrapped = _xet_result_for_plan(plan, contract)
    atomic_json(tmp_path / policy["path"], wrapped)
    snapshot = gs._snapshot_policy_evidence(
        gs.TrustedArtifactStore(tmp_path),
        policy,
        label="Xet result",
        contract=contract,
        evidence_auth=_auth(),
    )
    assert snapshot["seal_sha256"] == wrapped["seal_sha256"]

    unanchored_body = {
        key: copy.deepcopy(item)
        for key, item in wrapped.items()
        if key not in {"seal_sha256", "producer_hmac_sha256"}
    }
    unanchored_body["evidence"].pop("controller_anchor_sha256")
    unanchored_body["evidence_sha256"] = gs._sha256(
        unanchored_body["evidence"]
    )
    atomic_json(
        tmp_path / policy["path"],
        gs.seal_producer_authenticated_evidence(unanchored_body, auth=_auth()),
    )
    with pytest.raises(gs.StateError, match="producing controller anchor"):
        gs._snapshot_policy_evidence(
            gs.TrustedArtifactStore(tmp_path),
            policy,
            label="Xet result",
            contract=contract,
            evidence_auth=_auth(),
        )

    fabricated_body = copy.deepcopy(raw)
    fabricated_body.pop("seal_sha256")
    fabricated_body["coverage"]["required_file_settings_measured"] = [8, 16, 24, 32]
    fabricated_raw = seal(fabricated_body)
    atomic_json(
        tmp_path / policy["path"],
        _wrap_xet_raw_result(fabricated_raw, contract),
    )
    with pytest.raises(gs.StateError, match="raw live-Xet result validation failed.*coverage"):
        gs._snapshot_policy_evidence(
            gs.TrustedArtifactStore(tmp_path),
            policy,
            label="Xet result",
            contract=contract,
            evidence_auth=_auth(),
        )


def test_frozen_schedule_validator_reaches_window_membership_and_checks_producer_hmac(
    tmp_path,
):
    contract_body = copy.deepcopy(_contract())
    contract_body.pop("seal_sha256")
    contract_body["state_gates"]["AUTOTUNE_XET"] = {
        "required_artifacts": {
            "xet_autotune_plan": {"expected_seal_sha256": HASH_A},
        }
    }
    contract_body["state_gates"]["ASSEMBLE_ARTIFACT"]["required_artifacts"][
        "preliminary_streaming_schedule"
    ] = {"expected_seal_sha256": HASH_B}
    contract = seal(contract_body)
    profile = {
        "acquisition": {"trial_id": "FILES_16"},
        "steady": {"trial_id": "FILES_08"},
    }
    schedule_evidence = {
        "producer": "synthetic-schedule-freezer",
        "window_schedule_sha256": gs._sha256(contract["window_schedule"]),
    }
    schedule = gs.seal_producer_authenticated_evidence(
        {
            "schema": "hawking.glm52.streaming_schedule.v2",
            "status": "FROZEN_AFTER_XET_AUTOTUNE",
            "repo": "zai-org/GLM-5.2",
            "revision": contract["source_revision"],
            "campaign_id": contract["campaign_id"],
            "source_revision": contract["source_revision"],
            "expected_contract_sha256": contract["seal_sha256"],
            "autotune_binding": {
                "xet_autotune_result_seal_sha256": "c" * 64,
                "xet_autotune_plan_seal_sha256": HASH_A,
                "preliminary_schedule_seal_sha256": HASH_B,
                "selected_profile_sha256": gs._sha256(profile),
            },
            "selected_profile": profile,
            "windows": copy.deepcopy(contract["window_schedule"]),
            "evidence": schedule_evidence,
            "evidence_sha256": gs._sha256(schedule_evidence),
        },
        auth=_auth(),
    )
    policy = {
        "path": "GLM52_STREAMING_SCHEDULE.json",
        "expected_seal_sha256": None,
        "expected_schema": "hawking.glm52.streaming_schedule.v2",
        "allowed_statuses": ["FROZEN_AFTER_XET_AUTOTUNE"],
        "validator_id": "frozen_schedule_v2",
        "validator_source_sha256": gs.EVIDENCE_VALIDATOR_SOURCE_SHA256[
            "frozen_schedule_v2"
        ],
        "require_producer_hmac": True,
    }
    atomic_json(tmp_path / policy["path"], schedule)
    snapshot = gs._snapshot_policy_evidence(
        gs.TrustedArtifactStore(tmp_path),
        policy,
        label="frozen schedule",
        contract=contract,
        evidence_auth=_auth(),
    )
    assert snapshot["seal_sha256"] == schedule["seal_sha256"]

    wrong_identity_body = {
        key: copy.deepcopy(item)
        for key, item in schedule.items()
        if key not in {"seal_sha256", "producer_hmac_sha256"}
    }
    wrong_identity_body["campaign_id"] = "different-campaign"
    atomic_json(
        tmp_path / policy["path"],
        gs.seal_producer_authenticated_evidence(wrong_identity_body, auth=_auth()),
    )
    with pytest.raises(gs.StateError, match="campaign_id identity mismatch"):
        gs._snapshot_policy_evidence(
            gs.TrustedArtifactStore(tmp_path),
            policy,
            label="frozen schedule",
            contract=contract,
            evidence_auth=_auth(),
        )

    tampered = copy.deepcopy(schedule)
    tampered.pop("seal_sha256")
    tampered["producer_hmac_sha256"] = "d" * 64
    atomic_json(tmp_path / policy["path"], seal(tampered))
    with pytest.raises(gs.StateError, match="producer HMAC authentication failed"):
        gs._snapshot_policy_evidence(
            gs.TrustedArtifactStore(tmp_path),
            policy,
            label="frozen schedule",
            contract=contract,
            evidence_auth=_auth(),
        )


@pytest.mark.parametrize(
    ("validator_id", "message"),
    (
        ("future_artifact_blocked_v1", "no registered schema-specific"),
        ("stop_condition_blocked_v1", "not yet derived from grounded"),
    ),
)
def test_unimplemented_official_evidence_cannot_be_satisfied_by_signed_pass(
    tmp_path, validator_id, message
):
    contract = _contract()
    semantic = {"test": True}
    body = {
        "schema": "hawking.glm52.unimplemented_test.v1",
        "status": "PASS",
        "campaign_id": contract["campaign_id"],
        "source_revision": contract["source_revision"],
        "expected_contract_sha256": contract["seal_sha256"],
        "evidence": semantic,
        "evidence_sha256": gs._sha256(semantic),
    }
    path = "future.json"
    atomic_json(
        tmp_path / path,
        gs.seal_producer_authenticated_evidence(body, auth=_auth()),
    )
    policy = {
        "path": path,
        "expected_seal_sha256": None,
        "expected_schema": body["schema"],
        "allowed_statuses": ["PASS"],
        "validator_id": validator_id,
        "validator_source_sha256": gs.EVIDENCE_VALIDATOR_SOURCE_SHA256[
            validator_id
        ],
        "require_producer_hmac": True,
    }
    with pytest.raises(gs.StateError, match=message):
        gs._snapshot_policy_evidence(
            gs.TrustedArtifactStore(tmp_path),
            policy,
            label="unimplemented evidence",
            contract=contract,
            evidence_auth=_auth(),
        )


def test_official_source_profile_enforces_exact_282_shards_bytes_and_grounding(
    tmp_path,
):
    per_shard = 5_000_000_000
    shards = [
        _source_shard(
            path=f"model-{index + 1:05d}-of-00282.safetensors",
            logical_bytes=(
                per_shard if index < 281
                else gs.OFFICIAL_WEIGHT_LOGICAL_BYTES - (281 * per_shard)
            ),
            content_hash=f"{index:064x}",
        )
        for index in range(gs.OFFICIAL_WEIGHT_SHARD_COUNT)
    ]
    paths = [item["path"] for item in shards]
    official_gates = json.loads(json.dumps(_state_gates()))
    official_gates["ASSEMBLE_ARTIFACT"]["required_artifacts"] = {
        name: {
            "path": f"reports/{name}.json",
            "expected_seal_sha256": HASH_A,
            "expected_schema": f"test.{name}.v1",
            "allowed_statuses": ["PASS"],
            "validator_id": "sealed_exact_v1",
            "validator_source_sha256": gs.EVIDENCE_VALIDATOR_SOURCE_SHA256[
                "sealed_exact_v1"
            ],
            "require_producer_hmac": False,
        }
        for name in gs.OFFICIAL_ASSEMBLY_REQUIRED_ARTIFACTS
    }
    official_gates["COMPLETE"]["required_artifacts"] = {
        name: {
            "path": f"reports/{name}.json",
            "expected_seal_sha256": None,
            "expected_schema": f"test.{name}.v1",
            "allowed_statuses": ["PASS"],
            "validator_id": "campaign_artifact_v1",
            "validator_source_sha256": gs.EVIDENCE_VALIDATOR_SOURCE_SHA256[
                "campaign_artifact_v1"
            ],
            "require_producer_hmac": True,
        }
        for name in gs.OFFICIAL_COMPLETE_REQUIRED_ARTIFACTS
    }
    official_gates["COMPLETE"]["required_checklist"] = {
        item: {
            "path": f"evidence/{item}.json",
            "expected_seal_sha256": None,
            "expected_schema": gs.STOP_CONDITION_EVIDENCE_SCHEMA,
            "allowed_statuses": ["PASS"],
            "validator_id": "stop_condition_v1",
            "validator_source_sha256": gs.EVIDENCE_VALIDATOR_SOURCE_SHA256[
                "stop_condition_v1"
            ],
            "require_producer_hmac": True,
        }
        for item in gs.MANDATORY_COMPLETE_STOP_CONDITIONS
    }
    contract = gs.make_expected_campaign_contract(
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
        expected_chat_identity_digest=CHAT_DIGEST,
        source_shards=shards,
        expected_tensors=[TENSOR],
        window_schedule=[{
            "schedule_index": 0,
            "window_id": "official-window",
            "source_shards": paths,
            "carry_in_shards": [],
            "new_fetch_shards": paths,
            "refetch_shards": [],
            "carry_out_shards": [],
            "evict_shards": paths,
            "tensor_set": [TENSOR],
        }],
        state_gates=official_gates,
        created_at="2026-07-21T00:00:00Z",
    )
    assert contract["source"]["expected_shard_count"] == 282
    assert contract["source"]["expected_logical_bytes"] == 1_506_667_387_408
    ledger = gs.WindowLedger(
        tmp_path / "official-window-ledger.jsonl",
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
        expected_contract=contract,
        lease_guard=lambda: None,
    )
    with pytest.raises(gs.StateError, match="filesystem-executor-v1 grounding"):
        ledger.declare_window(
            window_id="official-window",
            schedule_index=0,
            source_shards=shards,
            carry_in_shards=[],
            new_fetch_shards=paths,
            refetch_shards=[],
            carry_out_shards=[],
            evict_shards=paths,
            tensor_set=[TENSOR],
            layer_organ_dependencies=[],
            disk_before={"free_bytes": 1},
            claim_id="official:window:declare:0001",
        )
    with pytest.raises(gs.StateError, match="exactly 282"):
        gs.make_expected_campaign_contract(
            campaign_id=CAMPAIGN,
            source_revision=REVISION,
            expected_chat_identity_digest=CHAT_DIGEST,
            source_shards=[_source_shard()],
            expected_tensors=[TENSOR],
            window_schedule=_contract()["window_schedule"],
            state_gates=official_gates,
            created_at="2026-07-21T00:00:00Z",
        )


def test_controller_requires_explicit_opt_in_for_synthetic_contract(tmp_path):
    with pytest.raises(gs.StateError, match="test-only opt-in"):
        gs.Controller(
            tmp_path / "controller",
            artifact_root=tmp_path,
            campaign_id=CAMPAIGN,
            source_revision=REVISION,
            expected_contract=_contract(),
            telegram_auth=_auth(),
        )


def test_transition_is_claimed_idempotent_heartbeat_and_telegram_bound(tmp_path):
    with _controller(tmp_path) as controller:
        checkpoint = _boot(controller)
        payload = {"kimi_release_seal": HASH_A}
        claim = "test:transition:close:0001"
        receipt = _delivery(controller, "CLOSE_KIMI", claim, payload)
        first = controller.transition(
            "CLOSE_KIMI", claim_id=claim, payload=payload, telegram_delivery=receipt
        )
        second = controller.transition(
            "CLOSE_KIMI", claim_id=claim, payload=payload, telegram_delivery=receipt
        )
        assert first == second
        assert first["state"] == "CLOSE_KIMI"
        assert first["heartbeat"]["state"] == "CLOSE_KIMI"
        assert first["telegram"]["status"] == "DELIVERED"
        assert first["event_count"] == checkpoint["event_count"] + 1
        assert controller.events.verify_chain() == (True, [])
        with pytest.raises(gs.StateError, match="claim reused"):
            controller.transition(
                "RELEASE_KIMI_SOURCE",
                claim_id=claim,
                telegram_delivery=_delivery(controller, "RELEASE_KIMI_SOURCE", claim),
            )


def test_illegal_transition_and_missing_or_wrong_delivery_fail_closed(tmp_path):
    with _controller(tmp_path) as controller:
        _boot(controller)
        claim = "test:transition:illegal:0001"
        with pytest.raises(gs.StateError, match="illegal controller transition"):
            controller.transition(
                "BUILD_MANIFEST",
                claim_id=claim,
                telegram_delivery=_delivery(controller, "BUILD_MANIFEST", claim),
            )
        with pytest.raises(gs.StateError, match="Telegram"):
            controller.transition(
                "CLOSE_KIMI", claim_id="test:transition:nodelivery:0001", telegram_delivery={}
            )
        wrong = _delivery(
            controller,
            "CLOSE_KIMI",
            "test:transition:different-intent:0001",
            message_id=999,
        )
        with pytest.raises(gs.StateError, match="claim differs"):
            controller.transition(
                "CLOSE_KIMI", claim_id="test:transition:wrongkey:0001", telegram_delivery=wrong
            )
        assert controller.resume()["state"] == "PRECHECK"


def test_telegram_receipt_is_hmac_authenticated_response_and_chat_bound(tmp_path):
    with _controller(tmp_path) as controller:
        _boot(controller)
        claim = "test:transition:hmac:0001"
        receipt = _delivery(controller, "CLOSE_KIMI", claim)
        tampered = dict(receipt)
        tampered["bot_api_response_sha256"] = HASH_B
        with pytest.raises(gs.StateError, match="HMAC authentication failed"):
            controller.transition(
                "CLOSE_KIMI", claim_id=claim, telegram_delivery=tampered
            )
        with pytest.raises(gs.StateError, match="chat identity"):
            intent = controller.prepare_transition("CLOSE_KIMI", claim_id=claim)
            gs.make_telegram_delivery_receipt(
                intent,
                auth=controller.telegram_auth,
                bot_api_response={
                    "ok": True,
                    "result": {
                        "message_id": 7,
                        "chat": {"id": 999999},
                        "text": intent["rendered_message"],
                    },
                },
                http_status=200,
            )
        with pytest.raises(gs.StateError, match="validated success"):
            intent = controller.prepare_transition("CLOSE_KIMI", claim_id=claim)
            gs.make_telegram_delivery_receipt(
                intent,
                auth=controller.telegram_auth,
                bot_api_response={"ok": False, "description": "failure"},
                http_status=200,
            )
        persisted = b"".join(
            path.read_bytes() for path in controller.root.iterdir() if path.is_file()
        )
        assert HMAC_KEY not in persisted
        assert HMAC_KEY.hex().encode("ascii") not in persisted


def test_authenticated_operator_confirmation_cannot_advance_state(tmp_path):
    with _controller(tmp_path) as controller:
        _boot(controller)
        intent = controller.prepare_transition(
            "CLOSE_KIMI", claim_id="test:transition:operator-proof:0001"
        )
        body = {
            "schema": "hawking.glm52.telegram_operator_delivery_confirmation.v1",
            "status": "OPERATOR_CONFIRMED_DELIVERED",
            "algorithm": "HMAC-SHA256",
            "event_kind": intent["event_kind"],
            "claim_id": intent["claim_id"],
            "from_state": intent["from_state"],
            "to_state": intent["to_state"],
            "dedupe_key": intent["dedupe_key"],
            "canonical_status": intent["canonical_status"],
            "canonical_status_sha256": intent["canonical_status_sha256"],
            "rendered_message": intent["rendered_message"],
            "rendered_message_sha256": intent["rendered_message_sha256"],
            "controller_anchor": intent["controller_anchor"],
            "controller_anchor_sha256": intent["controller_anchor"]["anchor_sha256"],
            "transition_intent": intent,
            "transition_intent_sha256": intent["seal_sha256"],
            "message_id": 77,
            "confirmation_claim_id": "operator-confirmation-0001",
            "confirmed_at": "2026-07-21T12:00:00Z",
            "chat_identity_digest": controller.telegram_auth.expected_chat_identity_digest,
            "response_validated": False,
            "delivery_proof": "HUMAN_CONFIRMED_EXISTING_EXACT_MESSAGE",
        }
        signed = {**body, "operator_confirmation_sha256": gs._sha256(body)}
        proof = {
            **signed,
            "hmac_sha256": controller.telegram_auth.authenticate(signed),
        }
        assert not hasattr(gs, "make_operator_confirmed_delivery_receipt")
        with pytest.raises(gs.StateError, match="receipt schema/fields invalid"):
            controller.commit_transition(intent, telegram_delivery=proof)
        assert controller.resume()["state"] == "PRECHECK"


def test_status_requires_fresh_strict_utc_heartbeat_for_live_worker(
    tmp_path, monkeypatch
):
    clock = {"now": "2026-07-21T12:00:00Z"}
    monkeypatch.setattr(gs, "utc_now", lambda: clock["now"])
    with _controller(tmp_path) as controller:
        _boot(controller)
        fresh = controller.status()
        assert fresh["heartbeat_at"] == "2026-07-21T12:00:00Z"
        assert fresh["heartbeat_fresh_ok"] is True
        assert fresh["heartbeat_max_age_seconds"] == 120
        assert fresh["live_worker_lease_ok"] is True

        clock["now"] = "2026-07-21T12:02:01Z"
        stale = controller.status()
        assert stale["durable_state_ok"] is True
        assert stale["heartbeat_fresh_ok"] is False
        assert stale["heartbeat_freshness_reason"] == (
            "checkpoint heartbeat exceeded the maximum age"
        )
        assert stale["live_worker_lease_ok"] is False

        clock["now"] = "not-an-rfc3339-time"
        controller.heartbeat(
            claim_id="test:heartbeat:invalid-utc:0001", telemetry={"sample": 1}
        )
        clock["now"] = "2026-07-21T12:02:02Z"
        invalid = controller.status()
        assert invalid["durable_state_ok"] is True
        assert invalid["heartbeat_fresh_ok"] is False
        assert "RFC3339 UTC timestamp" in invalid["heartbeat_freshness_reason"]
        assert invalid["live_worker_lease_ok"] is False


def test_status_never_reports_stale_checkpoint_green(tmp_path, monkeypatch):
    with _controller(tmp_path) as controller:
        _boot(controller)
        assert controller.status()["durable_state_ok"] is True
        original = controller._write_checkpoint

        def simulated_checkpoint_loss():
            raise OSError("checkpoint write lost")

        monkeypatch.setattr(controller, "_write_checkpoint", simulated_checkpoint_loss)
        with pytest.raises(OSError, match="checkpoint write lost"):
            controller.heartbeat(
                claim_id="test:heartbeat:stale:0001", telemetry={"source_pct": 1}
            )
        stale = controller.status()
        assert stale["controller_event_chain_ok"] is True
        assert stale["checkpoint_seal_ok"] is True
        assert stale["checkpoint_anchor_ok"] is False
        assert stale["checkpoint_replay_ok"] is False
        assert stale["durable_state_ok"] is False
        assert stale["state"] is None
        monkeypatch.setattr(controller, "_write_checkpoint", original)
        controller.resume()
        assert controller.status()["durable_state_ok"] is True


def test_blocked_can_only_resume_exact_prior_state_with_resolution(tmp_path):
    with _controller(tmp_path) as controller:
        _boot(controller)
        payload = {"reason": "immutable source manifest unavailable"}
        blocked = _transition(controller, "BLOCKED", 1, payload)
        assert blocked["blocked_from"] == "PRECHECK"
        wrong_payload = {"resolution_receipt_sha256": HASH_A}
        claim = "test:transition:wrongresume:0001"
        with pytest.raises(gs.StateError, match="resume only"):
            controller.transition(
                "CLOSE_KIMI",
                claim_id=claim,
                payload=wrong_payload,
                telegram_delivery=_delivery(controller, "CLOSE_KIMI", claim, wrong_payload),
            )
        resolved = {"resolution_receipt_sha256": HASH_A}
        resumed = _transition(controller, "PRECHECK", 2, resolved)
        assert resumed["state"] == "PRECHECK"
        assert resumed["blocked_from"] is None


def test_singleton_lease_refuses_second_controller(tmp_path):
    first = _controller(tmp_path)
    second = _controller(tmp_path)
    first.acquire()
    try:
        with pytest.raises(gs.StateError, match="already-running"):
            second.acquire()
    finally:
        first.close()
    second.acquire()
    second.close()


def test_resume_rejects_checkpoint_tamper_and_log_truncation(tmp_path):
    controller = _controller(tmp_path)
    with controller:
        _boot(controller)
        _transition(controller, "CLOSE_KIMI", 1)
        path = controller.checkpoint_path
        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["state"] = "BUILD_MANIFEST"
        atomic_json(path, tampered)
        with pytest.raises(gs.StateError, match="seal mismatch"):
            controller.resume()
    # Restore a valid run, then truncate its log below the checkpoint anchor.
    controller = _controller(tmp_path / "truncated")
    with controller:
        _boot(controller)
        _transition(controller, "CLOSE_KIMI", 1)
        lines = controller.events.path.read_text(encoding="utf-8").splitlines()
        controller.events.path.write_text(lines[0] + "\n", encoding="utf-8")
        with pytest.raises(gs.StateError, match="ahead of a log"):
            controller.resume()


def test_single_valid_crash_tail_is_recovered_but_fork_anchor_is_rejected(tmp_path, monkeypatch):
    controller = _controller(tmp_path)
    controller.acquire()
    try:
        _boot(controller)
        original = controller._write_checkpoint

        def simulated_power_loss():
            raise OSError("simulated power loss after fsync(event)")

        monkeypatch.setattr(controller, "_write_checkpoint", simulated_power_loss)
        claim = "test:transition:crash:0001"
        with pytest.raises(OSError, match="power loss"):
            controller.transition(
                "CLOSE_KIMI",
                claim_id=claim,
                telegram_delivery=_delivery(controller, "CLOSE_KIMI", claim),
            )
        monkeypatch.setattr(controller, "_write_checkpoint", original)
        recovered = controller.resume()
        assert recovered["state"] == "CLOSE_KIMI"
        assert recovered["event_count"] == 2
        forked = dict(recovered)
        forked.pop("seal_sha256")
        forked["event_head_hash"] = HASH_A
        atomic_json(controller.checkpoint_path, seal(forked))
        with pytest.raises(gs.StateError, match="fork/split-brain"):
            controller.resume()
    finally:
        controller.close()


def test_boot_transition_is_idempotent_after_checkpoint_power_loss(tmp_path, monkeypatch):
    controller = _controller(tmp_path)
    controller.acquire()
    try:
        original = controller._write_checkpoint

        def simulated_power_loss():
            raise OSError("simulated genesis checkpoint power loss")

        monkeypatch.setattr(controller, "_write_checkpoint", simulated_power_loss)
        claim = "test:boot:crash:0001"
        receipt = _delivery(controller, "PRECHECK", claim)
        with pytest.raises(OSError, match="genesis checkpoint power loss"):
            controller.boot(claim_id=claim, telegram_delivery=receipt)
        monkeypatch.setattr(controller, "_write_checkpoint", original)
        recovered = controller.boot(claim_id=claim, telegram_delivery=receipt)
        assert recovered["state"] == "PRECHECK"
        assert recovered["event_count"] == 1
    finally:
        controller.close()


def test_resume_rejects_multiple_valid_orphan_events_as_ambiguous(tmp_path):
    with _controller(tmp_path) as controller:
        _boot(controller)
        for index in (1, 2):
            claim = f"test:heartbeat:orphan:{index:04d}"
            telemetry = {"sample": index}
            request_sha = gs._sha256(
                {
                    "kind": "HEARTBEAT",
                    "campaign_id": CAMPAIGN,
                    "source_revision": REVISION,
                    "controller_epoch": controller.controller_epoch,
                    "expected_contract_sha256": controller.expected_contract_sha256,
                    "claim_id": claim,
                    "state": "PRECHECK",
                    "telemetry": telemetry,
                }
            )
            controller.events.append(
                "HEARTBEAT",
                claim,
                {
                    "campaign_id": CAMPAIGN,
                    "source_revision": REVISION,
                    "controller_epoch": controller.controller_epoch,
                    "expected_contract_sha256": controller.expected_contract_sha256,
                    "state": "PRECHECK",
                    "heartbeat": {
                        "at": f"2026-07-21T12:00:0{index}Z",
                        "pid": 123,
                        "telemetry": telemetry,
                    },
                    "request_sha256": request_sha,
                },
            )
        with pytest.raises(gs.StateError, match="ambiguous uncheckpointed tails"):
            controller.resume()


def test_hash_chain_detects_in_place_edit_and_torn_tail(tmp_path):
    with _controller(tmp_path) as controller:
        _boot(controller)
        line = json.loads(controller.events.path.read_text(encoding="utf-8").splitlines()[0])
        line["payload"]["state_payload"] = {"forged": True}
        controller.events.path.write_text(
            json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        ok, reasons = controller.events.verify_chain()
        assert not ok and any("hash mismatch" in reason for reason in reasons)
    with _controller(tmp_path / "torn") as controller:
        _boot(controller)
        raw = controller.events.path.read_bytes()
        controller.events.path.write_bytes(raw[:-1])
        ok, reasons = controller.events.verify_chain()
        assert not ok and any("torn" in reason for reason in reasons)


def test_window_ledger_full_monotonic_lifecycle_and_exact_terminal_coverage(tmp_path):
    lease = gs.SingletonLease(
        tmp_path / "lease", campaign_id=CAMPAIGN, controller_epoch="test-window-ledger"
    )
    with lease:
        ledger = gs.WindowLedger(
            tmp_path / gs.WINDOW_LEDGER_FILENAME,
            campaign_id=CAMPAIGN,
            source_revision=REVISION,
            expected_contract=_contract(),
            lease_guard=lease.assert_held,
        )
        ledger.declare_window(**_window_args())
        initial_plan = ledger.resume_plan()
        assert initial_plan["earliest_incomplete_window_id"] == "window-0001"
        assert initial_plan["next_controller_state"] == "FETCH_WINDOW"
        terminal = _complete_window(ledger)
        assert terminal["phase"] == "EVICTED"
        assert terminal["source_coverage"][SHARD] == "EVICTED"
        assert terminal["tensor_coverage"][TENSOR] == "PACKED_IN_CORE_ARTIFACT"
        coverage = ledger.assert_complete_tensor_coverage([TENSOR])
        assert coverage["status"] == "COMPLETE"
        assert coverage["terminal_tensor_count"] == 1
        source = ledger.assert_complete_source_coverage()
        assert source["expected_shard_count"] == 1
        assert source["expected_logical_bytes"] == 5_000_000_000
        assert source["all_eventually_evicted"] is True
        assert source["transfer_amplification"]["numerator_network_bytes"] == 5_000_000_100
        assert ledger.resume_plan()["campaign_windows_complete"] is True
        assert ledger.log.verify_chain() == (True, [])


def test_window_ledger_rejects_backward_skipped_and_off_schedule_declaration(tmp_path):
    with gs.SingletonLease(
        tmp_path / "lease", campaign_id=CAMPAIGN, controller_epoch="test-window-ledger"
    ) as lease:
        ledger = gs.WindowLedger(
            tmp_path / gs.WINDOW_LEDGER_FILENAME,
            campaign_id=CAMPAIGN,
            source_revision=REVISION,
            expected_contract=_contract(),
            lease_guard=lease.assert_held,
        )
        ledger.declare_window(**_window_args())
        with pytest.raises(gs.StateError, match="backward/skipped"):
            _advance(
                ledger,
                "FETCHED",
                1,
                patch={
                    "download_start": "2026-07-21T12:00:00Z",
                    "download_end": "2026-07-21T12:01:00Z",
                },
                source={SHARD: "FETCHED"},
            )
        second = _window_args(
            claim="test:window:declare:0002", tensor=TENSOR, window="window-0002"
        )
        second["source_shards"][0]["path"] = "model-00002-of-00282.safetensors"
        with pytest.raises(gs.StateError, match="authoritative schedule"):
            ledger.declare_window(**second)


def test_window_retry_is_exactly_incremented_and_idempotent(tmp_path):
    with gs.SingletonLease(
        tmp_path / "lease", campaign_id=CAMPAIGN, controller_epoch="test-window-ledger"
    ) as lease:
        ledger = gs.WindowLedger(
            tmp_path / gs.WINDOW_LEDGER_FILENAME,
            campaign_id=CAMPAIGN,
            source_revision=REVISION,
            expected_contract=_contract(),
            lease_guard=lease.assert_held,
        )
        ledger.declare_window(**_window_args())
        claim = "test:window:retry:0001"
        first = ledger.record_retry("window-0001", claim_id=claim, metrics={"fault": "timeout"})
        second = ledger.record_retry("window-0001", claim_id=claim, metrics={"fault": "timeout"})
        assert first == second
        assert first["retry_count"] == 1
        with pytest.raises(gs.StateError, match="different window operation"):
            ledger.record_retry("window-0001", claim_id=claim, metrics={"fault": "other"})


def test_sequential_carry_refetch_and_eventual_eviction_accounting(tmp_path):
    shard_b = "model-00002-of-00282.safetensors"
    shard_c = "model-00003-of-00282.safetensors"
    tensor_b = "model.layers.1.self_attn.q_proj.weight"
    identities = {
        SHARD: _source_shard(),
        shard_b: _source_shard(shard_b, 3_000_000_000, HASH_B),
        shard_c: _source_shard(shard_c, 2_000_000_000, "c" * 64),
    }
    schedule = [
        {
            "schedule_index": 0, "window_id": "carry-window-0",
            "source_shards": [SHARD, shard_b], "carry_in_shards": [],
            "new_fetch_shards": [SHARD, shard_b], "refetch_shards": [],
            "carry_out_shards": [shard_b], "evict_shards": [SHARD],
            "tensor_set": [TENSOR],
        },
        {
            "schedule_index": 1, "window_id": "carry-window-1",
            "source_shards": [shard_b, shard_c], "carry_in_shards": [shard_b],
            "new_fetch_shards": [shard_c], "refetch_shards": [],
            "carry_out_shards": [], "evict_shards": [shard_b, shard_c],
            "tensor_set": [tensor_b],
        },
        {
            "schedule_index": 2, "window_id": "carry-window-2",
            "source_shards": [SHARD], "carry_in_shards": [],
            "new_fetch_shards": [], "refetch_shards": [SHARD],
            "carry_out_shards": [], "evict_shards": [SHARD],
            "tensor_set": [],
        },
    ]
    contract = gs.make_expected_campaign_contract(
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
        expected_chat_identity_digest=CHAT_DIGEST,
        source_shards=list(identities.values()),
        expected_tensors=[TENSOR, tensor_b],
        window_schedule=schedule,
        state_gates=_state_gates(),
        source_profile="SYNTHETIC_TEST_ONLY",
        created_at="2026-07-21T00:00:00Z",
    )
    bad_schedule = json.loads(json.dumps(schedule))
    bad_schedule[1]["carry_in_shards"] = []
    with pytest.raises(gs.StateError, match="carry_in != prior carry_out"):
        gs.make_expected_campaign_contract(
            campaign_id=CAMPAIGN,
            source_revision=REVISION,
            expected_chat_identity_digest=CHAT_DIGEST,
            source_shards=list(identities.values()),
            expected_tensors=[TENSOR, tensor_b],
            window_schedule=bad_schedule,
            state_gates=_state_gates(),
            source_profile="SYNTHETIC_TEST_ONLY",
            created_at="2026-07-21T00:00:00Z",
        )

    def declaration(spec, index):
        return {
            **{key: json.loads(json.dumps(value)) for key, value in spec.items()
               if key not in {"source_shards"}},
            "source_shards": [identities[path] for path in spec["source_shards"]],
            "layer_organ_dependencies": [{"schedule_index": index}],
            "disk_before": {"free_bytes": 600_000_000_000, "allocated_bytes": index},
            "claim_id": f"carry:declare:{index:04d}",
        }

    def finish(ledger, spec, index):
        window_id = spec["window_id"]
        fetched = spec["new_fetch_shards"] + spec["refetch_shards"]
        tensors = spec["tensor_set"]

        def advance(phase, step, **kwargs):
            return ledger.advance(
                window_id,
                phase,
                claim_id=f"carry:advance:{index:04d}:{step:04d}",
                **kwargs,
            )

        advance(
            "FETCHING", 1,
            patch={"download_start": f"2026-07-21T12:0{index}:00Z"},
            source_coverage={path: "FETCHING" for path in fetched},
        )
        new_bytes = 1000 * len(spec["new_fetch_shards"])
        refetch_bytes = 500 * len(spec["refetch_shards"])
        advance(
            "FETCHED", 2,
            patch={
                "download_end": f"2026-07-21T12:0{index}:10Z",
                "bytes_transferred": new_bytes + refetch_bytes + 10,
                "transfer_accounting": {
                    "new_fetch_network_bytes": new_bytes,
                    "refetch_network_bytes": refetch_bytes,
                    "protocol_overhead_bytes": 10,
                },
            },
            source_coverage={path: "FETCHED" for path in fetched},
        )
        advance(
            "VERIFIED", 3,
            patch={
                "hash_verification": {
                    "status": "VERIFIED", "verified_shards": spec["source_shards"],
                }
            },
            source_coverage={path: "HASH_VERIFIED" for path in fetched},
            tensor_coverage={tensor: "SOURCE_VERIFIED" for tensor in tensors},
        )
        advance(
            "TEACHER_CAPTURED", 4,
            patch={
                "teacher_evidence_produced": [
                    {"path": f"teacher/{window_id}.json", "sha256": HASH_A}
                ]
            },
            tensor_coverage={tensor: "TEACHER_EVIDENCED" for tensor in tensors},
        )
        advance("CANDIDATES_FIT", 5, patch={"metrics": {"fit_loss": index}})
        advance(
            "CANDIDATES_PACKED", 6,
            patch={
                "candidate_payloads_produced": [
                    {"path": f"candidate/{window_id}.bin", "sha256": HASH_B}
                ]
            },
            tensor_coverage={tensor: "CANDIDATE_PACKED" for tensor in tensors},
        )
        advance(
            "FORWARD_COMPLETE", 7,
            patch={"metrics": {"fit_loss": index, "forward_cosine": 0.99}},
            tensor_coverage={tensor: "FORWARD_VERIFIED" for tensor in tensors},
        )
        advance(
            "SEALED", 8,
            patch={
                "compact_shard_hashes": (
                    {f"compact/{window_id}.bin": HASH_B} if tensors else {}
                ),
                "terminal_coverage_evidence": {
                    tensor: {
                        "disposition": "PACKED_IN_CORE_ARTIFACT",
                        "evidence_sha256": HASH_A,
                    }
                    for tensor in tensors
                },
            },
            source_coverage={path: "CONSUMED" for path in spec["source_shards"]},
            tensor_coverage={tensor: "PACKED_IN_CORE_ARTIFACT" for tensor in tensors},
        )
        return advance(
            "EVICTED", 9,
            patch={
                "source_eviction": {
                    "status": "EVICTED",
                    "evicted_shards": spec["evict_shards"],
                    "receipt_sha256": HASH_B,
                },
                "disk_after": {"free_bytes": 610_000_000_000, "allocated_bytes": index},
            },
            source_coverage={path: "EVICTED" for path in spec["evict_shards"]},
        )

    with gs.SingletonLease(
        tmp_path / "lease", campaign_id=CAMPAIGN, controller_epoch="carry-test"
    ) as lease:
        ledger = gs.WindowLedger(
            tmp_path / gs.WINDOW_LEDGER_FILENAME,
            campaign_id=CAMPAIGN,
            source_revision=REVISION,
            expected_contract=contract,
            lease_guard=lease.assert_held,
        )
        ledger.declare_window(**declaration(schedule[0], 0))
        first = finish(ledger, schedule[0], 0)
        assert first["source_coverage"][SHARD] == "EVICTED"
        assert first["source_coverage"][shard_b] == "CONSUMED"
        second_declared = ledger.declare_window(**declaration(schedule[1], 1))
        assert second_declared["source_coverage"][shard_b] == "HASH_VERIFIED"
        assert second_declared["source_coverage"][shard_c] == "DECLARED"
        finish(ledger, schedule[1], 1)
        third_declared = ledger.declare_window(**declaration(schedule[2], 2))
        assert third_declared["source_coverage"][SHARD] == "DECLARED"
        finish(ledger, schedule[2], 2)
        complete = ledger.assert_complete_source_coverage()
        assert complete["expected_shard_count"] == 3
        assert complete["new_fetch_shard_count"] == 3
        assert complete["refetch_event_count"] == 1
        assert complete["refetch_logical_bytes"] == 5_000_000_000
        assert complete["all_eventually_evicted"] is True
        assert ledger.assert_complete_tensor_coverage()["terminal_tensor_count"] == 2


def test_window_ledger_requires_lease_and_controller_binds_ledger_checkpoint(tmp_path):
    standalone = gs.WindowLedger(
        tmp_path / "bare.jsonl",
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
        expected_contract=_contract(),
        lease_guard=lambda: (_ for _ in ()).throw(gs.StateError("lease missing")),
    )
    with pytest.raises(gs.StateError, match="lease missing"):
        standalone.declare_window(**_window_args())

    with _controller(tmp_path / "bound") as controller:
        _boot(controller)
        # Reach FREEZE_PROGRAM through the exact pre-window sequence.
        path = [
            "CLOSE_KIMI", "RELEASE_KIMI_SOURCE", "ADMIT_GLM_SOURCE", "BUILD_MANIFEST",
            "BUILD_DEPENDENCY_GRAPH", "AUTOTUNE_XET", "BUILD_ADAPTER", "BUILD_REFERENCE",
            "BUILD_CORPUS", "PILOT_ORACLES", "FREEZE_PROGRAM",
        ]
        for number, state in enumerate(path, 1):
            _transition(controller, state, number)
        controller.declare_window(**_window_args())
        checkpoint = controller.resume()
        assert checkpoint["window_event_count"] == 1
        assert checkpoint["window_summary"]["owned_tensor_count"] == 1
        fetch_payload = {"window_id": "window-0001"}
        _transition(controller, "FETCH_WINDOW", 20, fetch_payload)
        controller.advance_window(
            "window-0001",
            "FETCHING",
            claim_id="test:window:bound:advance:0001",
            patch={"download_start": "2026-07-21T12:00:00Z"},
            source_coverage={SHARD: "FETCHING"},
        )
        checkpoint = controller.resume()
        assert checkpoint["state"] == "FETCH_WINDOW"
        assert checkpoint["window_event_count"] == 2


def test_authoritative_terminal_gates_block_incomplete_and_require_closure_evidence(tmp_path):
    with _controller(tmp_path) as controller:
        _reach_freeze(controller)
        controller.declare_window(**_window_args())
        _run_single_controller_window(controller)

        claim = "test:terminal:assembly-missing:0001"
        with pytest.raises(gs.StateError, match="trusted evidence"):
            controller.prepare_transition("ASSEMBLE_ARTIFACT", claim_id=claim)
        _terminal_evidence(controller, "ASSEMBLE_ARTIFACT")
        _transition(controller, "ASSEMBLE_ARTIFACT", 109)
        checkpoint = controller.resume()
        assert checkpoint["campaign_completeness"]["source"]["expected_shard_count"] == 1
        assert checkpoint["campaign_completeness"]["source"]["all_eventually_evicted"] is True
        assert checkpoint["campaign_completeness"]["tensors"]["terminal_tensor_count"] == 1

        _transition(controller, "VERIFY_ARTIFACT", 110)
        _transition(controller, "RUN_FULL_COMPACT", 111)
        _transition(controller, "RUN_RATE_DESCENT", 112)
        _transition(controller, "SEAL_GLM_RESULT", 113)
        _transition(controller, "FINAL_GRAVITY_AUDIT", 114)
        _terminal_evidence(controller, "COMPLETE")
        (controller.artifact_root / "reports/GLM52_PHONE_STATUS.json").unlink()
        bad_claim = "test:terminal:complete-missing-phone:0001"
        with pytest.raises(gs.StateError, match="trusted evidence"):
            controller.prepare_transition("COMPLETE", claim_id=bad_claim)
        _terminal_evidence(controller, "COMPLETE")
        final = _transition(controller, "COMPLETE", 116)
        assert final["state"] == "COMPLETE"
        assert final["telegram"]["status"] == "DELIVERED"
        assert final["campaign_completeness"]["source"]["expected_logical_bytes"] == 5_000_000_000
        assert controller.status()["durable_state_ok"] is True


def test_controller_rejects_claim_reuse_across_logs(tmp_path):
    with _controller(tmp_path) as controller:
        _boot(controller)
        for number, state in enumerate(
            [
                "CLOSE_KIMI", "RELEASE_KIMI_SOURCE", "ADMIT_GLM_SOURCE", "BUILD_MANIFEST",
                "BUILD_DEPENDENCY_GRAPH", "AUTOTUNE_XET", "BUILD_ADAPTER", "BUILD_REFERENCE",
                "BUILD_CORPUS", "PILOT_ORACLES", "FREEZE_PROGRAM",
            ],
            1,
        ):
            _transition(controller, state, number)
        reused = "test:transition:0001"
        args = _window_args(claim=reused)
        with pytest.raises(gs.StateError, match="other durable log"):
            controller.declare_window(**args)
