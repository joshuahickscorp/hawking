#!/usr/bin/env python3.12
"""Offline adversarial tests for the GLM-5.2 post-Xet schedule freeze."""
from __future__ import annotations

import copy
import hashlib
import pathlib
import sys
from typing import Any, Mapping

import pytest


CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_schedule_freeze as freezer  # noqa: E402
import glm52_state as state  # noqa: E402
from glm52_common import canonical, seal, verify_sealed  # noqa: E402


REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
CAMPAIGN = "glm52-schedule-freeze-offline-test"
CHAT_DIGEST = state.telegram_chat_identity_digest(-100525252)
HMAC_KEY = b"glm52-schedule-freeze-test-hmac-key-32-bytes!!"
SHARD = "model-00001-of-00282.safetensors"
TENSOR = "model.layers.0.self_attn.q_proj.weight"
TRIAL_IDS = [
    "DEFAULT_UNSET",
    "FILES_08",
    "FILES_16",
    "FILES_24",
    "FILES_32",
    "FILES_48",
    "FIXED_16",
    "FIXED_32",
    "FIXED_64",
    "HIGH_PERFORMANCE",
    "CACHE_1G_COLD",
    "CACHE_1G_REPLAY",
]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _auth() -> state.TelegramAuthConfig:
    return state.TelegramAuthConfig(
        hmac_key=HMAC_KEY,
        expected_chat_identity_digest=CHAT_DIGEST,
    )


def _exact_policy(
    path: str,
    schema: str,
    status: str,
    expected_seal: str,
) -> dict[str, Any]:
    validator = "sealed_exact_v1"
    return {
        "path": path,
        "expected_seal_sha256": expected_seal,
        "expected_schema": schema,
        "allowed_statuses": [status],
        "validator_id": validator,
        "validator_source_sha256": state.EVIDENCE_VALIDATOR_SOURCE_SHA256[validator],
        "require_producer_hmac": False,
    }


def _future_policy(path: str, schema: str) -> dict[str, Any]:
    validator = "campaign_artifact_v1"
    return {
        "path": path,
        "expected_seal_sha256": None,
        "expected_schema": schema,
        "allowed_statuses": ["PASS"],
        "validator_id": validator,
        "validator_source_sha256": state.EVIDENCE_VALIDATOR_SOURCE_SHA256[validator],
        "require_producer_hmac": True,
    }


def _gate(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    source: bool = False,
    tensor: bool = False,
    eviction: bool = False,
    phone: bool = False,
) -> dict[str, Any]:
    return {
        "require_source_complete": source,
        "require_tensor_complete": tensor,
        "require_final_source_eviction": eviction,
        "require_telegram_delivery": True,
        "require_phone_status": phone,
        "required_phone_status_path": "GLM52_PHONE_STATUS.json" if phone else None,
        "required_artifacts": dict(artifacts),
        "required_checklist": {},
    }


def _fake_plan() -> dict[str, Any]:
    matrix = [
        {
            "trial_id": trial_id,
            "caller_concurrent_shard_streams": index + 1,
        }
        for index, trial_id in enumerate(TRIAL_IDS)
    ]
    source_paths = (
        "GLM52_OFFICIAL_MANIFEST.json",
        "GLM52_SOURCE_FORMAT_LEDGER.json",
        "GLM52_SHARD_DEPENDENCY_GRAPH.json",
        "GLM52_SOURCE_ADMISSION.json",
    )
    return seal({
        "schema": "hawking.glm52.xet_autotune_plan.v2",
        "status": "PASS_OFFLINE_PLAN_BODY_NOT_READ",
        "repo": freezer.xet_live.REPO_ID,
        "revision": REVISION,
        "inputs": [
            {
                "path": path,
                "schema": "offline.test.v1",
                "status": "PASS",
                "seal_sha256": _sha(path),
            }
            for path in source_paths
        ],
        "toolchain_binding": {"schema": "offline.toolchain.v1", "id": _sha("tool")},
        "range_strategy": {"body_ranges": [{"range_id_sha256": _sha("range")}]},
        "trial_matrix": matrix,
        "largest_shard_validation": {
            "lfs_sha256": _sha("largest"),
        },
        "network_budget": {
            "bounded_range_payload_bytes": 10,
            "largest_shard_validation_bytes": 20,
            "planned_maximum_bytes": 30,
            "hard_cap_bytes": 1_000,
        },
    })


def _preliminary() -> dict[str, Any]:
    return seal({
        "schema": freezer.PRELIMINARY_SCHEMA,
        "status": freezer.PRELIMINARY_STATUS,
        "repo": freezer.xet_live.REPO_ID,
        "revision": REVISION,
        "planner_inputs": {
            "free_disk_bytes": 1_000,
            "hard_floor_bytes": 100,
            "largest_two_source_shards_bytes": 10,
            "operational_reserve_bytes": 100,
            "projected_three_complete_artifacts_bytes_0_98_plus_0_75_plus_0_50": 20,
            "projected_evidence_bytes": 20,
            "active_scratch_bytes": 0,
            "usable_raw_window_bytes": 760,
            "safety_fraction": 0.7,
            "p99_allocated_proxy_uses_remote_logical_shard_bytes": 10,
            "two_simultaneous_complete_raw_windows": True,
            "disk_limited_shards_per_window": 1,
            "preliminary_target_shards_per_window": 1,
        },
        "pipeline": copy.deepcopy(freezer._PIPELINE),
        "window_count": 1,
        "maximum_resident_shards_in_one_window": 1,
        "maximum_simultaneous_shards_active_plus_prefetch_upper_bound": 2,
        "source_shards_scheduled": 1,
        "planned_refetches": 0,
        "windows": [{
            "window_id": "W000",
            "organ_ids": ["text_layer_00"],
            "source_shards": [SHARD],
            "source_shard_count": 1,
            "carry_in_shards": [],
            "new_fetch_shards": [SHARD],
            "refetch_shards": [],
            "carry_out_shards": [],
            "evict_after_seal_shards": [SHARD],
            "new_fetch_logical_bytes": 64,
            "resident_logical_bytes": 64,
        }],
        "freeze_boundary": freezer._FREEZE_BOUNDARY,
    })


def _contract(plan: Mapping[str, Any], preliminary: Mapping[str, Any]) -> dict[str, Any]:
    autotune_gate = _gate({
        "xet_autotune_plan": _exact_policy(
            "GLM52_XET_AUTOTUNE_PLAN.json",
            plan["schema"],
            plan["status"],
            plan["seal_sha256"],
        ),
    })
    assembly_gate = _gate(
        {
            "preliminary_streaming_schedule": _exact_policy(
                "GLM52_STREAMING_SCHEDULE_PRE_AUTOTUNE.json",
                preliminary["schema"],
                preliminary["status"],
                preliminary["seal_sha256"],
            ),
        },
        source=True,
        tensor=True,
        eviction=True,
    )
    complete_gate = _gate(
        {"final": _future_policy("GLM52_FINAL_TEST.json", "offline.final.v1")},
        source=True,
        tensor=True,
        eviction=True,
        phone=True,
    )
    complete_gate["required_checklist"] = {
        "offline_complete": _future_policy(
            "evidence/offline_complete.json",
            state.STOP_CONDITION_EVIDENCE_SCHEMA,
        ),
    }
    return state.make_expected_campaign_contract(
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
        expected_chat_identity_digest=CHAT_DIGEST,
        source_shards=[{
            "path": SHARD,
            "logical_bytes": 64,
            "content_hash": _sha("source"),
            "content_hash_kind": "xet",
        }],
        expected_tensors=[TENSOR],
        window_schedule=[{
            "schedule_index": 0,
            "window_id": "W000",
            "source_shards": [SHARD],
            "carry_in_shards": [],
            "new_fetch_shards": [SHARD],
            "refetch_shards": [],
            "carry_out_shards": [],
            "evict_shards": [SHARD],
            "tensor_set": [TENSOR],
        }],
        state_gates={
            "AUTOTUNE_XET": autotune_gate,
            "ASSEMBLE_ARTIFACT": assembly_gate,
            "COMPLETE": complete_gate,
        },
        source_profile="SYNTHETIC_TEST_ONLY",
        created_at="2026-07-21T00:00:00Z",
    )


def _raw_result(plan: Mapping[str, Any]) -> dict[str, Any]:
    selected_trial = plan["trial_matrix"][0]

    def selection(lane: str) -> dict[str, Any]:
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
    source_refs = [dict(item) for item in plan["inputs"]]
    budget = plan["network_budget"]
    return seal({
        "schema": freezer.xet_live.AUTOTUNE_RESULT_SCHEMA,
        "status": "PASS_LIVE_XET_AUTOTUNE_COMPLETE_SCHEDULE_REFREEZE_REQUIRED",
        "repo": freezer.xet_live.REPO_ID,
        "revision": REVISION,
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
                pathlib.Path(freezer.xet_live.__file__).read_bytes()
            ).hexdigest(),
        },
        "coverage": {
            "trial_ids_in_plan_order": list(TRIAL_IDS),
            "trial_results": [
                {
                    "trial_id": trial_id,
                    "trial_result_seal_sha256": _sha(f"trial:{trial_id}"),
                    "resource_verdict": {"status": "PASS", "measured": {}},
                    "selection_candidate_sha256": _sha(f"candidate:{trial_id}"),
                }
                for trial_id in TRIAL_IDS
            ],
            "required_file_settings": [8, 16, 24, 32, 48],
            "required_file_settings_measured": [8, 16, 24, 32, 48],
            "fixed_profiles_measured": ["FIXED_16", "FIXED_32", "FIXED_64"],
            "high_performance_measured": True,
            "cache_profiles_measured": ["CACHE_1G_COLD", "CACHE_1G_REPLAY"],
            "unique_sealed_ranges_hashed": 1,
            "repeated_range_sha256_consistent": True,
        },
        "selections": selections,
        "selected_profile": copy.deepcopy(selections),
        "largest_shard_validations": [
            {
                "lane": lane,
                "evidence_seal_sha256": _sha(f"largest:{lane}"),
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


def _reauth(value: Mapping[str, Any], auth: state.TelegramAuthConfig) -> dict[str, Any]:
    body = {
        key: copy.deepcopy(item) for key, item in value.items()
        if key not in {"seal_sha256", "producer_hmac_sha256"}
    }
    return state.seal_producer_authenticated_evidence(body, auth=auth)


@pytest.fixture
def bundle(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    plan = _fake_plan()

    def accept_plan(
        candidate: Mapping[str, Any],
        *,
        root: pathlib.Path = freezer.REPO_ROOT,
        rebuild: bool = True,
    ) -> dict[str, Any]:
        del root, rebuild
        return dict(verify_sealed(dict(candidate), label="fake plan"))

    monkeypatch.setattr(freezer.xet_live, "validate_live_plan", accept_plan)
    preliminary = _preliminary()
    contract = _contract(plan, preliminary)
    auth = _auth()
    raw = _raw_result(plan)
    anchor = _sha("controller-anchor")
    attested = freezer.attest_xet_autotune_result(
        raw,
        plan,
        contract,
        auth=auth,
        controller_anchor_sha256=anchor,
        rebuild_plan=False,
    )
    schedule = freezer.freeze_schedule(
        preliminary,
        plan,
        attested,
        contract,
        auth=auth,
        rebuild_plan=False,
    )
    return {
        "plan": plan,
        "preliminary": preliminary,
        "contract": contract,
        "auth": auth,
        "raw": raw,
        "attested": attested,
        "anchor": anchor,
        "schedule": schedule,
    }


def test_attestation_and_freeze_bind_every_immutable_input(
    bundle: Mapping[str, Any], tmp_path: pathlib.Path
) -> None:
    schedule = bundle["schedule"]
    verify_sealed(schedule)
    assert schedule["schema"] == freezer.FINAL_SCHEMA
    assert schedule["status"] == freezer.FINAL_STATUS
    assert schedule["selected_profile"] == bundle["raw"]["selections"]
    assert schedule["autotune_binding"] == {
        "xet_autotune_result_seal_sha256": bundle["attested"]["seal_sha256"],
        "xet_autotune_plan_seal_sha256": bundle["plan"]["seal_sha256"],
        "preliminary_schedule_seal_sha256": bundle["preliminary"]["seal_sha256"],
        "selected_profile_sha256": hashlib.sha256(
            canonical(schedule["selected_profile"])
        ).hexdigest(),
    }
    assert bundle["attested"]["evidence"]["controller_anchor_sha256"] == bundle["anchor"]
    assert schedule["evidence"]["network_access"] is False
    assert schedule["evidence"]["model_body_bytes_read"] == 0
    assert schedule["evidence"]["production_artifact_written"] is False
    assert schedule["windows"][0]["tensor_set"] == [TENSOR]
    assert schedule["windows"][0]["evict_shards"] == [SHARD]
    assert not (tmp_path / "GLM52_STREAMING_SCHEDULE.json").exists()
    assert freezer.validate_frozen_schedule(
        schedule,
        bundle["preliminary"],
        bundle["plan"],
        bundle["attested"],
        bundle["contract"],
        auth=bundle["auth"],
        rebuild_plan=False,
    ) == schedule


def test_freeze_does_not_mutate_any_input(bundle: Mapping[str, Any]) -> None:
    before = {
        key: canonical(bundle[key])
        for key in ("preliminary", "plan", "attested", "contract")
    }
    freezer.freeze_schedule(
        bundle["preliminary"],
        bundle["plan"],
        bundle["attested"],
        bundle["contract"],
        auth=bundle["auth"],
        rebuild_plan=False,
    )
    assert before == {
        key: canonical(bundle[key])
        for key in ("preliminary", "plan", "attested", "contract")
    }


def test_attestation_rejects_semantically_tampered_raw_result(
    bundle: Mapping[str, Any]
) -> None:
    tampered = copy.deepcopy(bundle["raw"])
    tampered["coverage"]["required_file_settings_measured"] = [8, 16, 24, 32]
    tampered = seal(tampered)
    with pytest.raises(freezer.ScheduleFreezeError, match="coverage"):
        freezer.attest_xet_autotune_result(
            tampered,
            bundle["plan"],
            bundle["contract"],
            auth=bundle["auth"],
            controller_anchor_sha256=bundle["anchor"],
            rebuild_plan=False,
        )


def test_attested_result_rejects_changed_raw_seal_even_with_fresh_hmac(
    bundle: Mapping[str, Any]
) -> None:
    tampered = copy.deepcopy(bundle["attested"])
    tampered["evidence"][freezer.RAW_RESULT_SEAL_EVIDENCE_KEY] = _sha("not-raw")
    tampered["evidence_sha256"] = hashlib.sha256(canonical(tampered["evidence"])).hexdigest()
    tampered = _reauth(tampered, bundle["auth"])
    with pytest.raises(freezer.ScheduleFreezeError, match="raw seal reconstruction"):
        freezer.freeze_schedule(
            bundle["preliminary"],
            bundle["plan"],
            tampered,
            bundle["contract"],
            auth=bundle["auth"],
            rebuild_plan=False,
        )


def test_attested_result_rejects_wrong_plan_evidence_with_fresh_hmac(
    bundle: Mapping[str, Any]
) -> None:
    tampered = copy.deepcopy(bundle["attested"])
    tampered["evidence"]["xet_autotune_plan_seal_sha256"] = _sha("wrong-plan")
    tampered["evidence_sha256"] = hashlib.sha256(canonical(tampered["evidence"])).hexdigest()
    tampered = _reauth(tampered, bundle["auth"])
    with pytest.raises(freezer.ScheduleFreezeError, match="evidence plan seal"):
        freezer.freeze_schedule(
            bundle["preliminary"],
            bundle["plan"],
            tampered,
            bundle["contract"],
            auth=bundle["auth"],
            rebuild_plan=False,
        )


def test_attested_result_rejects_unknown_field_and_stale_hmac(
    bundle: Mapping[str, Any]
) -> None:
    unknown = copy.deepcopy(bundle["attested"])
    unknown["untrusted_extension"] = True
    unknown = _reauth(unknown, bundle["auth"])
    with pytest.raises(freezer.ScheduleFreezeError, match="fields differ"):
        freezer.freeze_schedule(
            bundle["preliminary"],
            bundle["plan"],
            unknown,
            bundle["contract"],
            auth=bundle["auth"],
            rebuild_plan=False,
        )

    stale = copy.deepcopy(bundle["attested"])
    stale["campaign_id"] = "attacker-campaign"
    stale = seal(stale)
    with pytest.raises(freezer.ScheduleFreezeError, match="campaign_id|HMAC"):
        freezer.freeze_schedule(
            bundle["preliminary"],
            bundle["plan"],
            stale,
            bundle["contract"],
            auth=bundle["auth"],
            rebuild_plan=False,
        )


def test_freeze_rejects_plan_and_preliminary_seal_substitution(
    bundle: Mapping[str, Any]
) -> None:
    plan = copy.deepcopy(bundle["plan"])
    plan["toolchain_binding"]["id"] = _sha("other-tool")
    plan = seal(plan)
    with pytest.raises(freezer.ScheduleFreezeError, match="expected contract"):
        freezer.freeze_schedule(
            bundle["preliminary"],
            plan,
            bundle["attested"],
            bundle["contract"],
            auth=bundle["auth"],
            rebuild_plan=False,
        )

    preliminary = copy.deepcopy(bundle["preliminary"])
    preliminary["windows"][0]["new_fetch_logical_bytes"] += 1
    preliminary = seal(preliminary)
    with pytest.raises(freezer.ScheduleFreezeError, match="expected contract"):
        freezer.freeze_schedule(
            preliminary,
            bundle["plan"],
            bundle["attested"],
            bundle["contract"],
            auth=bundle["auth"],
            rebuild_plan=False,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["windows"][0]["tensor_set"].append("forged.tensor"), "differs"),
        (
            lambda value: value["selected_profile"]["steady"].update({"trial_id": "FILES_48"}),
            "differs",
        ),
        (
            lambda value: value["autotune_binding"].update({
                "xet_autotune_result_seal_sha256": _sha("other-result")
            }),
            "differs",
        ),
    ],
)
def test_validator_rejects_reauthenticated_final_schedule_tampering(
    bundle: Mapping[str, Any], mutation: Any, message: str
) -> None:
    tampered = copy.deepcopy(bundle["schedule"])
    mutation(tampered)
    tampered = _reauth(tampered, bundle["auth"])
    with pytest.raises(freezer.ScheduleFreezeError, match=message):
        freezer.validate_frozen_schedule(
            tampered,
            bundle["preliminary"],
            bundle["plan"],
            bundle["attested"],
            bundle["contract"],
            auth=bundle["auth"],
            rebuild_plan=False,
        )


def test_attestation_rejects_invalid_controller_anchor(bundle: Mapping[str, Any]) -> None:
    with pytest.raises(freezer.ScheduleFreezeError, match="controller_anchor_sha256"):
        freezer.attest_xet_autotune_result(
            bundle["raw"],
            bundle["plan"],
            bundle["contract"],
            auth=bundle["auth"],
            controller_anchor_sha256="not-a-digest",
            rebuild_plan=False,
        )


def test_freeze_rejects_reauthenticated_result_without_controller_anchor(
    bundle: Mapping[str, Any],
) -> None:
    body = {
        key: copy.deepcopy(item)
        for key, item in bundle["attested"].items()
        if key not in {"seal_sha256", "producer_hmac_sha256"}
    }
    body["evidence"].pop("controller_anchor_sha256")
    body["evidence_sha256"] = hashlib.sha256(
        canonical(body["evidence"])
    ).hexdigest()
    unanchored = state.seal_producer_authenticated_evidence(
        body, auth=bundle["auth"]
    )
    with pytest.raises(freezer.ScheduleFreezeError, match="controller anchor"):
        freezer.freeze_schedule(
            bundle["preliminary"],
            bundle["plan"],
            unanchored,
            bundle["contract"],
            auth=bundle["auth"],
            rebuild_plan=False,
        )


def test_freeze_rejects_auth_for_another_chat(bundle: Mapping[str, Any]) -> None:
    wrong = state.TelegramAuthConfig(
        hmac_key=HMAC_KEY,
        expected_chat_identity_digest=_sha("another-chat"),
    )
    with pytest.raises(freezer.ScheduleFreezeError, match="chat identity"):
        freezer.freeze_schedule(
            bundle["preliminary"],
            bundle["plan"],
            bundle["attested"],
            bundle["contract"],
            auth=wrong,
            rebuild_plan=False,
        )
