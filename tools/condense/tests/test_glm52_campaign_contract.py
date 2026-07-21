#!/usr/bin/env python3.12
"""Offline adversarial tests for the official expected-campaign contract builder."""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import sys

import pytest


CONDENSE = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = CONDENSE.parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_campaign_contract as campaign  # noqa: E402
import glm52_state as state  # noqa: E402
from glm52_common import canonical, read_sealed_json, seal, verify_sealed  # noqa: E402


CHAT_DIGEST = hashlib.sha256(b"offline-test-rotated-private-chat").hexdigest()
CREATED_AT = "2026-07-21T00:00:00Z"


@pytest.fixture(scope="module")
def artifacts() -> dict[str, dict]:
    """Use real sealed inputs, rebinding only a concurrently regenerated Xet index.

    Adapter/parity/corpus generators can be running elsewhere in the shared worktree.
    The production preflight correctly rejects a stale Xet-plan seal.  Unit tests need
    an internally consistent immutable snapshot, so this fixture updates those plan
    references in memory and reseals the test-only copy.
    """
    loaded = campaign.load_inputs(REPO_ROOT)
    values = dict(loaded.artifacts)
    plan = copy.deepcopy(values["xet_autotune_plan"])
    for row in plan["inputs"]:
        spec = campaign.INPUT_BY_FILENAME[row["path"]]
        row["seal_sha256"] = values[spec.key]["seal_sha256"]
    plan.pop("seal_sha256")
    values["xet_autotune_plan"] = seal(plan)
    return values


@pytest.fixture(scope="module")
def official_contract(artifacts: dict[str, dict]) -> dict:
    return campaign.build_contract_from_artifacts(
        artifacts,
        chat_identity_digest=CHAT_DIGEST,
        created_at=CREATED_AT,
    )


def test_builds_exact_official_contract_and_losslessly_maps_eviction(
    artifacts: dict[str, dict], official_contract: dict
) -> None:
    contract = verify_sealed(official_contract)
    assert contract["schema"] == state.EXPECTED_CONTRACT_SCHEMA
    assert contract["campaign_id"] == "glm52-bf16-xet-gravity"
    assert contract["source_revision"] == campaign.OFFICIAL_REVISION
    assert contract["expected_chat_identity_digest"] == CHAT_DIGEST
    assert contract["created_at"] == CREATED_AT
    assert contract["source"]["profile"] == "OFFICIAL_GLM52_BF16"
    assert contract["source"]["expected_shard_count"] == 282
    assert contract["source"]["expected_logical_bytes"] == 1_506_667_387_408
    assert len({row["path"] for row in contract["source"]["shards"]}) == 282
    assert contract["tensors"]["expected_tensor_count"] == 59_585
    assert len(set(contract["tensors"]["names"])) == 59_585
    assert len(contract["window_schedule"]) == 20

    source_windows = artifacts["streaming_schedule"]["windows"]
    for index, mapped in enumerate(contract["window_schedule"]):
        source = source_windows[index]
        assert mapped["schedule_index"] == index
        assert mapped["window_id"] == source["window_id"]
        assert mapped["evict_shards"] == source["evict_after_seal_shards"]
        assert "evict_after_seal_shards" not in mapped
    assert contract["window_schedule"][-1]["carry_out_shards"] == []


def test_state_gates_freeze_inputs_and_require_all_terminal_evidence(
    artifacts: dict[str, dict], official_contract: dict
) -> None:
    gates = official_contract["state_gates"]
    assert set(gates) == {
        "AUTOTUNE_XET",
        "BUILD_ADAPTER",
        "BUILD_REFERENCE",
        "BUILD_CORPUS",
        "PILOT_ORACLES",
        "FREEZE_PROGRAM",
        "FETCH_WINDOW",
        "ASSEMBLE_ARTIFACT",
        "VERIFY_ARTIFACT",
        "RUN_FULL_COMPACT",
        "RUN_DOCTOR_REFINEMENT",
        "RUN_RATE_DESCENT",
        "SEAL_GLM_RESULT",
        "FINAL_GRAVITY_AUDIT",
        "COMPLETE",
    }
    assembly = gates["ASSEMBLE_ARTIFACT"]
    complete = gates["COMPLETE"]
    assert all(assembly[key] is True for key in (
        "require_source_complete",
        "require_tensor_complete",
        "require_final_source_eviction",
        "require_telegram_delivery",
    ))
    for mandatory in state.OFFICIAL_ASSEMBLY_REQUIRED_ARTIFACTS:
        assert mandatory in assembly["required_artifacts"]
        path = assembly["required_artifacts"][mandatory]["path"]
        spec = campaign.INPUT_BY_FILENAME[path]
        if mandatory == "streaming_schedule":
            assert assembly["required_artifacts"][mandatory]["expected_seal_sha256"] is None
            assert assembly["required_artifacts"][mandatory]["validator_id"] == (
                "frozen_schedule_v2"
            )
            assert assembly["required_artifacts"]["preliminary_streaming_schedule"][
                "expected_seal_sha256"
            ] == artifacts[spec.key]["seal_sha256"]
        else:
            assert assembly["required_artifacts"][mandatory]["expected_seal_sha256"] == (
                artifacts[spec.key]["seal_sha256"]
            )
    assert set(assembly["required_checklist"]) == set(
        state.MANDATORY_COMPLETE_STOP_CONDITIONS[:16]
    )
    assert complete["required_phone_status_path"] == "GLM52_PHONE_STATUS.json"
    assert set(complete["required_checklist"]) == set(
        state.MANDATORY_COMPLETE_STOP_CONDITIONS
    )
    assert len(complete["required_checklist"]) == 30
    assert set(state.OFFICIAL_COMPLETE_REQUIRED_ARTIFACTS).issubset(
        complete["required_artifacts"]
    )
    assert set(campaign.COMPLETE_ARTIFACT_PATHS) == set(complete["required_artifacts"])
    assert complete["required_artifacts"]["xet_autotune_result"]["path"] == (
        "GLM52_XET_AUTOTUNE.json"
    )
    assert complete["required_artifacts"]["xet_autotune_result"][
        "expected_seal_sha256"
    ] is None
    blocked_future = {
        label
        for label, policy in complete["required_artifacts"].items()
        if policy["expected_seal_sha256"] is None
        and label not in {"streaming_schedule", "xet_autotune_result"}
    }
    assert blocked_future
    assert all(
        complete["required_artifacts"][label]["validator_id"]
        == "future_artifact_blocked_v1"
        for label in blocked_future
    )
    assert all(
        policy["validator_id"] == "stop_condition_blocked_v1"
        for policy in complete["required_checklist"].values()
    )


def test_existing_closure_inputs_are_frozen_and_fetch_has_prerequisite_gates(
    artifacts: dict[str, dict], official_contract: dict
) -> None:
    assembly = official_contract["state_gates"]["ASSEMBLE_ARTIFACT"]
    expected_existing = {
        "handoff_precheck": "handoff_precheck",
        "kimi_source_release": "kimi_source_release",
        "gravity_pre_audit": "gravity_pre_audit",
        "external_baseline_matrix": "external_baseline_matrix",
    }
    for label, artifact_key in expected_existing.items():
        assert assembly["required_artifacts"][label]["expected_seal_sha256"] == (
            artifacts[artifact_key]["seal_sha256"]
        )

    gates = official_contract["state_gates"]
    assert gates["AUTOTUNE_XET"]["required_artifacts"]["xet_autotune_plan"][
        "expected_seal_sha256"
    ] == artifacts["xet_autotune_plan"]["seal_sha256"]
    xet_result = gates["BUILD_ADAPTER"]["required_artifacts"]["xet_autotune_result"]
    assert xet_result["path"] == "GLM52_XET_AUTOTUNE.json"
    assert xet_result["expected_seal_sha256"] is None
    assert xet_result["validator_id"] == "xet_autotune_result_v1"
    assert xet_result["require_producer_hmac"] is True
    assert gates["FETCH_WINDOW"]["required_artifacts"]["xet_autotune_result"][
        "validator_id"
    ] == "xet_autotune_result_v1"
    assert gates["BUILD_CORPUS"]["required_artifacts"]["bf16_reference_forward"][
        "path"
    ] == "GLM52_BF16_REFERENCE_FORWARD.json"
    fetch = gates["FETCH_WINDOW"]
    assert {
        "xet_autotune_result",
        "bf16_reference_forward",
        "corpus_integrity",
        "oracle_bandwidth",
        "causal_atlas",
        "frozen_program",
    }.issubset(fetch["required_artifacts"])
    assert {
        "xet_selected_profile_sealed",
        "bf16_reference_forward_validated",
        "corpus_integrity_green",
        "oracle_causal_pilot_complete",
        "full_candidate_program_frozen",
    }.issubset(fetch["required_checklist"])
    for state_name in (
        "VERIFY_ARTIFACT",
        "RUN_FULL_COMPACT",
        "RUN_DOCTOR_REFINEMENT",
        "RUN_RATE_DESCENT",
        "SEAL_GLM_RESULT",
        "FINAL_GRAVITY_AUDIT",
    ):
        gate = gates[state_name]
        assert gate["required_artifacts"]
        assert gate["required_checklist"]
        assert all(
            policy["validator_id"] == "future_artifact_blocked_v1"
            for policy in gate["required_artifacts"].values()
        )
        assert all(
            policy["validator_id"] == "stop_condition_blocked_v1"
            for policy in gate["required_checklist"].values()
        )


def test_pre_xet_authority_is_non_circular_and_post_xet_freeze_is_mandatory(
    artifacts: dict[str, dict], official_contract: dict
) -> None:
    gates = official_contract["state_gates"]
    assert "xet_autotune_result" not in gates["AUTOTUNE_XET"]["required_artifacts"]
    assert "frozen_streaming_schedule" not in gates["AUTOTUNE_XET"]["required_artifacts"]
    adapter = gates["BUILD_ADAPTER"]["required_artifacts"]
    assert set(adapter) == {"xet_autotune_result", "frozen_streaming_schedule"}
    frozen = adapter["frozen_streaming_schedule"]
    assert frozen["expected_schema"] == "hawking.glm52.streaming_schedule.v2"
    assert frozen["allowed_statuses"] == ["FROZEN_AFTER_XET_AUTOTUNE"]
    assert frozen["validator_id"] == "frozen_schedule_v2"
    assert frozen["require_producer_hmac"] is True
    assert gates["ASSEMBLE_ARTIFACT"]["required_artifacts"][
        "preliminary_streaming_schedule"
    ]["expected_seal_sha256"] == artifacts["streaming_schedule"]["seal_sha256"]


def test_build_is_deterministic_for_explicit_timestamp(
    artifacts: dict[str, dict], official_contract: dict
) -> None:
    repeated = campaign.build_contract_from_artifacts(
        artifacts,
        chat_identity_digest=CHAT_DIGEST,
        created_at=CREATED_AT,
    )
    assert canonical(repeated) == canonical(official_contract)


def test_omitted_or_unsealed_input_is_rejected(artifacts: dict[str, dict]) -> None:
    omitted = dict(artifacts)
    omitted.pop("dependency_graph")
    with pytest.raises(campaign.CampaignContractError, match="inventory mismatch"):
        campaign.validate_and_derive(omitted)

    tampered = dict(artifacts)
    manifest = dict(tampered["official_manifest"])
    manifest["weight_shards"] = 281  # retain old seal intentionally
    tampered["official_manifest"] = manifest
    with pytest.raises(campaign.CampaignContractError, match="seal mismatch"):
        campaign.validate_and_derive(tampered)


def test_count_and_schedule_tampering_fail_even_when_resealed(
    artifacts: dict[str, dict]
) -> None:
    wrong_count = dict(artifacts)
    logical = dict(wrong_count["logical_weight_ledger"])
    logical["tensor_count"] = 59_584
    logical.pop("seal_sha256")
    wrong_count["logical_weight_ledger"] = seal(logical)
    with pytest.raises(campaign.CampaignContractError, match="tensor ledger mismatch"):
        campaign.validate_and_derive(wrong_count)

    wrong_schedule = dict(artifacts)
    schedule = copy.deepcopy(wrong_schedule["streaming_schedule"])
    first = schedule["windows"][0]
    first["evict_shards"] = first.pop("evict_after_seal_shards")
    schedule.pop("seal_sha256")
    wrong_schedule["streaming_schedule"] = seal(schedule)
    with pytest.raises(campaign.CampaignContractError, match="window 0 schema mismatch"):
        campaign.validate_and_derive(wrong_schedule)


def test_stale_xet_input_binding_is_rejected(artifacts: dict[str, dict]) -> None:
    stale = dict(artifacts)
    plan = copy.deepcopy(stale["xet_autotune_plan"])
    plan["inputs"][0]["seal_sha256"] = "f" * 64
    plan.pop("seal_sha256")
    stale["xet_autotune_plan"] = seal(plan)
    with pytest.raises(campaign.CampaignContractError, match="stale or malformed binding"):
        campaign.validate_and_derive(stale)


@pytest.mark.parametrize(
    "digest",
    [None, "123456789", "A" * 64, "0" * 64, "a" * 63],
)
def test_only_safe_nonplaceholder_chat_digest_is_accepted(
    artifacts: dict[str, dict], digest: str | None
) -> None:
    with pytest.raises(campaign.CampaignContractError, match="digest is required") as caught:
        campaign.build_contract_from_artifacts(
            artifacts,
            chat_identity_digest=digest,  # type: ignore[arg-type]
            created_at=CREATED_AT,
        )
    assert str(digest) not in str(caught.value)


@pytest.mark.parametrize("created_at", [None, "", "now", "2026-07-21T00:00:00+00:00"])
def test_created_at_must_be_explicit_and_deterministic(
    artifacts: dict[str, dict], created_at: str | None
) -> None:
    with pytest.raises(campaign.CampaignContractError, match="created_at"):
        campaign.build_contract_from_artifacts(
            artifacts,
            chat_identity_digest=CHAT_DIGEST,
            created_at=created_at,  # type: ignore[arg-type]
        )


def test_preflight_and_build_command_refuse_missing_rotated_digest(
    tmp_path, capsys
) -> None:
    report = campaign.preflight(
        REPO_ROOT,
        chat_identity_digest=None,
        created_at=CREATED_AT,
    )
    assert report["status"] == "BLOCKED"
    assert report["build_authorized"] is False
    assert "ROTATED_CHAT_IDENTITY_DIGEST_REQUIRED_64_LOWERCASE_HEX" in report["blockers"]
    assert report["output_created"] is False

    output = tmp_path / campaign.OUTPUT_FILENAME
    code = campaign.main([
        "build",
        "--root",
        str(REPO_ROOT),
        "--created-at",
        CREATED_AT,
        "--output",
        str(output),
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert not output.exists()
    error = json.loads(captured.err)
    assert error["output_created"] is False
    assert "digest" in error["message"]


def test_write_is_atomic_idempotent_and_refuses_different_contract(
    tmp_path, official_contract: dict
) -> None:
    path = tmp_path / campaign.OUTPUT_FILENAME
    assert campaign.write_contract(path, official_contract) == path
    first = path.read_bytes()
    assert campaign.write_contract(path, official_contract) == path
    assert path.read_bytes() == first
    assert read_sealed_json(path) == official_contract

    different = dict(official_contract)
    different["created_at"] = "2026-07-21T00:00:01Z"
    different.pop("seal_sha256")
    different = seal(different)
    with pytest.raises(campaign.CampaignContractError, match="different frozen contract"):
        campaign.write_contract(path, different)
    assert path.read_bytes() == first
