#!/usr/bin/env python3.12
"""Storage-law tests for the non-invocable parent restream launcher."""
from __future__ import annotations

import pytest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lab.operators.glm52_common import canonical, seal
from ramanujan.restream_guard import (
    ACCOUNTING_COMPONENTS,
    ALIGNMENT_BYTES,
    ENVELOPE_BYTES,
    FRAMED_PROTOCOL,
    OWNER_AUTHORIZATION_SCHEMA,
    OPERATOR_APPROVAL_SCHEMA,
    PRODUCTION_LEASE_SCHEMA,
    RestreamGuardError,
    claim_single_use_owner_authorization,
    evaluate_green_light_transition,
    owner_authorization_claim_path,
    persist_green_light_transition,
    start_claimed_owner_authorization,
    validate_bounded_restream,
    validate_external_source_freeze_receipt,
    verify_owner_green_light_authorization,
)


_OPERATOR_RECEIPT_PRIVATE = Ed25519PrivateKey.generate()
_OPERATOR_RECEIPT_PUBLIC = _OPERATOR_RECEIPT_PRIVATE.public_key().public_bytes_raw()
_RANGE_EXECUTOR_SHA256 = "d" * 64


def _schedule(*, source: int = 1_048_576, active_model_limit: int = 1) -> dict[str, object]:
    accounting = {name: 0 for name in ACCOUNTING_COMPONENTS}
    accounting["source_range_rounded_bytes"] = source
    accounting["resident_incremental_bytes"] = source
    return seal(
        {
            "schema": "hawking.glm52.streaming_range_schedule.v1",
            "active_model_limit": active_model_limit,
            "incremental_accounting_contract": {
                "alignment_bytes": ALIGNMENT_BYTES,
                "components": list(ACCOUNTING_COMPONENTS),
            },
            "windows": [{"window_id": "W0", "incremental_accounting": accounting}],
        }
    )


def _policy(schedule: dict[str, object], *, active_model_limit: int = 1) -> dict[str, object]:
    return seal(
        {
            "schema": "hawking.glm52.resource_reserve_policy_90gb.v1",
            "input_seals": {"streaming_schedule": schedule["seal_sha256"]},
            "policy": {
                "incremental_storage_ceiling_bytes": ENVELOPE_BYTES,
                "active_model_limit": active_model_limit,
                "incremental_accounting_components": list(ACCOUNTING_COMPONENTS),
                "protected_filesystem_floor_bytes": 2_000_000,
            },
        }
    )


def _external_receipt() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for source_id in ("D5", "D8", "D9"):
        row: dict[str, object] = {
            "id": source_id,
            "status": "FROZEN_PENDING_INDEPENDENT_EVALUATION",
            "source_path_or_item_ids_serialized": False,
        }
        if source_id == "D5":
            row["no_leak_audit"] = {
                "direction": "D5_training_candidate_against_sealed_odyssey_evaluation",
                "exact_or_near_matches": 0,
            }
        elif source_id == "D8":
            row["no_leak_audit"] = {
                "direction": "all_current_D1_D7_training_against_D8_hidden_membership",
                "exact_or_near_matches": 0,
            }
        else:
            row["variant_generator"] = {
                "executed_by_freeze": False,
                "executable_sha256": "a" * 64,
                "seed_commitment_sha256": "b" * 64,
            }
        rows.append(row)
    return seal({
        "schema": "hawking.ramanujan.external_source_freeze_receipt.v1",
        "status": "PASS_INPUTS_FROZEN_RESEARCH_AND_CANDIDATE_AUTHORITY_FALSE",
        "RAMANUJAN_RESEARCH_AUTHORIZED": False,
        "candidate_launch_authorized": False,
        "independent_adjudication_complete": False,
        "counterexample_search_complete": False,
        "sources": rows,
        "training_visible": {"D8_hidden_item_ids": None, "D8_commitment_only": True},
    })


def _production_lease(now_ns: int) -> dict[str, object]:
    return {
        "schema": PRODUCTION_LEASE_SCHEMA,
        "production_authority": True,
        "fixture_only": False,
        "contention_label": "CLEAN",
        "foreign_processes": [],
        "lease_id": "owner-supplied-production-lease",
        "hardware_identity": "fixture-hardware-identity-not-a-production-certification",
        "pid": 42,
        "heartbeat_unix_ns": now_ns - 1,
        "expires_unix_ns": now_ns + 1,
    }


def _operator_approval(executable_sha256: str = "c" * 64) -> dict[str, object]:
    return {
        "schema": OPERATOR_APPROVAL_SCHEMA,
        "owner_approved": True,
        "production_authority": True,
        "fixture_only": False,
        "protocol": FRAMED_PROTOCOL,
        "executable_sha256": executable_sha256,
        "observed_executable_sha256": executable_sha256,
        "receipt_public_key_ed25519_hex": _OPERATOR_RECEIPT_PUBLIC.hex(),
        "receipt_public_key_sha256": __import__("hashlib").sha256(_OPERATOR_RECEIPT_PUBLIC).hexdigest(),
    }


def _owner_authorization(
    schedule: dict[str, object],
    policy: dict[str, object],
    external: dict[str, object],
    lease: dict[str, object],
    operator: dict[str, object],
    now_ns: int,
    range_executor_sha256: str = _RANGE_EXECUTOR_SHA256,
) -> tuple[dict[str, object], bytes]:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes_raw()
    payload = {
        "schedule_seal_sha256": schedule["seal_sha256"],
        "policy_seal_sha256": policy["seal_sha256"],
        "external_source_receipt_seal_sha256": external["seal_sha256"],
        "parent_restream_authorized": True,
        "production_lease_id": lease["lease_id"],
        "hardware_identity": lease["hardware_identity"],
        "production_lease_receipt_sha256": __import__("hashlib").sha256(canonical(lease)).hexdigest(),
        "operator_executable_sha256": operator["executable_sha256"],
        "operator_receipt_public_key_sha256": operator["receipt_public_key_sha256"],
        "range_executor_executable_sha256": range_executor_sha256,
        "operator_protocol": FRAMED_PROTOCOL,
        "issued_unix_ns": now_ns - 1,
        "expires_unix_ns": now_ns + 1,
        "nonce": "fixture-only-owner-nonce-0000000000000001",
        "max_uses": 1,
    }
    return ({
        "schema": OWNER_AUTHORIZATION_SCHEMA,
        "owner_public_key_sha256": __import__("hashlib").sha256(public_key).hexdigest(),
        "payload": payload,
        "signature_ed25519_hex": private_key.sign(canonical(payload)).hex(),
    }, public_key)


def test_bounded_schedule_requires_every_storage_class_and_one_model() -> None:
    schedule = _schedule()
    result = validate_bounded_restream(schedule, _policy(schedule))
    assert result == {"peak_incremental_bytes": 1_048_576, "window_count": 1}


@pytest.mark.parametrize("mutator", ["missing_teacher", "bad_sum", "parallel_model", "policy_unbound"])
def test_bounded_schedule_refuses_omission_or_unbound_policy(mutator: str) -> None:
    schedule = _schedule()
    policy = _policy(schedule)
    if mutator == "missing_teacher":
        del schedule["windows"][0]["incremental_accounting"]["teacher_evidence_bytes"]
    elif mutator == "bad_sum":
        schedule["windows"][0]["incremental_accounting"]["retained_artifact_bytes"] = 1
    elif mutator == "parallel_model":
        schedule = _schedule(active_model_limit=2)
        policy = _policy(schedule, active_model_limit=2)
    else:
        policy["input_seals"]["streaming_schedule"] = "0" * 64
    with pytest.raises(RestreamGuardError):
        validate_bounded_restream(schedule, policy)


def test_bounded_schedule_refuses_a_peak_above_the_envelope() -> None:
    schedule = _schedule(source=(ENVELOPE_BYTES // ALIGNMENT_BYTES + 1) * ALIGNMENT_BYTES)
    with pytest.raises(RestreamGuardError, match="exceeds"):
        validate_bounded_restream(schedule, _policy(schedule))


def test_parent_launcher_cannot_substitute_the_whole_shard_fetcher() -> None:
    launcher = Path("ramanujan/scaffold/guards/RAMANUJAN_FINAL_PARENT_NEXT_COMMAND.sh").read_text(encoding="utf-8")
    assert "HAWKING_GLM52_RANGE_STREAM_EXECUTOR" in launcher
    assert "ramanujan.status --require-hawking-complete" in launcher
    assert "HAWKING_EVOLUTION_COMPLETE" in launcher
    assert "tools/condense/glm52_range_stream_executor.py" in launcher
    assert 'tools/condense/glm52_source_fetch.py run' not in launcher


def test_external_owner_receipt_is_non_authorizing_and_privacy_bound() -> None:
    receipt = _external_receipt()
    assert validate_external_source_freeze_receipt(receipt) == receipt
    exposed = seal({**{key: value for key, value in receipt.items() if key != "seal_sha256"}})
    exposed["training_visible"]["D8_hidden_item_ids"] = ["secret"]
    exposed = seal(exposed)
    with pytest.raises(RestreamGuardError, match="D8 privacy"):
        validate_external_source_freeze_receipt(exposed)


def test_green_light_transition_reports_current_storage_but_keeps_owner_fences() -> None:
    schedule = _schedule()
    policy = _policy(schedule)
    result = evaluate_green_light_transition(
        schedule,
        policy,
        free_bytes=4_000_000,
        model_lane_file_count=0,
        external_source_receipt=None,
        parent_authorized=False,
        production_lease_receipt=None,
        operator_approval_receipt=None,
        range_executor_executable_sha256=_RANGE_EXECUTOR_SHA256,
        source_hashes_verified=False,
        runtime_pins_verified=False,
        now_ns=100,
    )
    assert result["status"] == "STORAGE_GREEN"
    assert result["restream_started"] is False
    assert result["production_authority"] is False
    assert result["satisfied_gates"] == ["OWNER_GATE_READY", "STORAGE_GREEN"]
    assert result["missing_gates"][:4] == [
        "OWNER_D5_APPROVAL",
        "OWNER_D8_APPROVAL",
        "OWNER_D9_APPROVAL",
        "OWNER_PARENT_RESTREAM_AUTHORIZATION",
    ]
    assert "OWNER_SIGNED_GREEN_LIGHT_AUTHORIZATION" in result["missing_gates"]


def test_storage_or_fixture_lease_cannot_be_promoted() -> None:
    schedule = _schedule()
    policy = _policy(schedule)
    external = _external_receipt()
    lease = _production_lease(100)
    operator = _operator_approval()
    authorization, public_key = _owner_authorization(schedule, policy, external, lease, operator, 100)
    below = evaluate_green_light_transition(
        schedule,
        policy,
        free_bytes=3_048_575,
        model_lane_file_count=0,
        external_source_receipt=external,
        parent_authorized=True,
        owner_authorization_receipt=authorization,
        owner_public_key_bytes=public_key,
        production_lease_receipt=lease,
        operator_approval_receipt=operator,
        range_executor_executable_sha256=_RANGE_EXECUTOR_SHA256,
        source_hashes_verified=True,
        runtime_pins_verified=True,
        now_ns=100,
    )
    assert below["status"] == "OWNER_GATE_READY"
    fixture = dict(lease)
    fixture["fixture_only"] = True
    fixture["production_authority"] = False
    refused = evaluate_green_light_transition(
        schedule,
        policy,
        free_bytes=4_000_000,
        model_lane_file_count=0,
        external_source_receipt=external,
        parent_authorized=True,
        owner_authorization_receipt=authorization,
        owner_public_key_bytes=public_key,
        production_lease_receipt=fixture,
        operator_approval_receipt=operator,
        range_executor_executable_sha256=_RANGE_EXECUTOR_SHA256,
        source_hashes_verified=True,
        runtime_pins_verified=True,
        now_ns=100,
    )
    # The owner signature binds the *full* production lease receipt, so a
    # fixture substitution invalidates authorization before it can reach the
    # lease gate.
    assert refused["status"] == "STORAGE_GREEN"
    assert "PRODUCTION_CLEAN_GPU_LEASE_IDENTITY" in refused["missing_gates"]


def test_full_transition_can_only_simulate_admission_in_fixture_test() -> None:
    schedule = _schedule()
    policy = _policy(schedule)
    external = _external_receipt()
    lease = _production_lease(100)
    operator = _operator_approval()
    authorization, public_key = _owner_authorization(schedule, policy, external, lease, operator, 100)
    result = evaluate_green_light_transition(
        schedule,
        policy,
        free_bytes=4_000_000,
        model_lane_file_count=0,
        external_source_receipt=external,
        parent_authorized=True,
        owner_authorization_receipt=authorization,
        owner_public_key_bytes=public_key,
        production_lease_receipt=lease,
        operator_approval_receipt=operator,
        range_executor_executable_sha256=_RANGE_EXECUTOR_SHA256,
        source_hashes_verified=True,
        runtime_pins_verified=True,
        now_ns=100,
        simulate_admission=True,
    )
    assert result["status"] == "SIMULATED_RESTREAM_ADMITTED"
    assert result["satisfied_gates"][-1] == "FINAL_PREFLIGHT"
    assert result["fixture_only"] is True
    assert result["production_authority"] is False
    assert result["restream_started"] is False
    assert result["exact_next_action"] == "SIMULATION_COMPLETE_NO_EXEC"


def test_transition_receipt_is_atomic_restart_safe_and_regresses_closed(tmp_path: Path) -> None:
    schedule = _schedule()
    policy = _policy(schedule)
    external = _external_receipt()
    lease = _production_lease(100)
    operator = _operator_approval()
    authorization, public_key = _owner_authorization(schedule, policy, external, lease, operator, 100)
    path = tmp_path / "green-light.json"
    ready = evaluate_green_light_transition(
        schedule,
        policy,
        free_bytes=4_000_000,
        model_lane_file_count=0,
        external_source_receipt=external,
        parent_authorized=True,
        owner_authorization_receipt=authorization,
        owner_public_key_bytes=public_key,
        production_lease_receipt=lease,
        operator_approval_receipt=operator,
        range_executor_executable_sha256=_RANGE_EXECUTOR_SHA256,
        source_hashes_verified=True,
        runtime_pins_verified=True,
        now_ns=100,
    )
    first = persist_green_light_transition(path, ready)
    assert persist_green_light_transition(path, ready) == first
    regressed = evaluate_green_light_transition(
        schedule,
        policy,
        free_bytes=4_000_000,
        model_lane_file_count=1,
        external_source_receipt=None,
        parent_authorized=False,
        production_lease_receipt=None,
        operator_approval_receipt=None,
        range_executor_executable_sha256=_RANGE_EXECUTOR_SHA256,
        source_hashes_verified=False,
        runtime_pins_verified=False,
        now_ns=101,
    )
    second = persist_green_light_transition(path, regressed)
    assert second["status"] == "OWNER_GATE_READY"
    assert second["previous_transition_seal_sha256"] == first["seal_sha256"]
    assert second["seal_sha256"] != first["seal_sha256"]


def test_owner_authorization_is_signature_expiry_and_launch_identity_bound() -> None:
    schedule = _schedule()
    policy = _policy(schedule)
    external = _external_receipt()
    lease = _production_lease(100)
    operator = _operator_approval()
    authorization, public_key = _owner_authorization(schedule, policy, external, lease, operator, 100)
    verified = verify_owner_green_light_authorization(
        authorization,
        public_key_bytes=public_key,
        schedule_seal_sha256=str(schedule["seal_sha256"]),
        policy_seal_sha256=str(policy["seal_sha256"]),
        external_source_receipt_seal_sha256=str(external["seal_sha256"]),
        production_lease_id=str(lease["lease_id"]),
        hardware_identity=str(lease["hardware_identity"]),
        production_lease_receipt_sha256=__import__("hashlib").sha256(canonical(lease)).hexdigest(),
        operator_executable_sha256=str(operator["executable_sha256"]),
        operator_receipt_public_key_sha256=str(operator["receipt_public_key_sha256"]),
        range_executor_executable_sha256=_RANGE_EXECUTOR_SHA256,
        now_ns=100,
    )
    assert verified == authorization

    alternate_key = Ed25519PrivateKey.generate().public_key().public_bytes_raw()
    with pytest.raises(RestreamGuardError, match="fingerprint differs"):
        verify_owner_green_light_authorization(
            authorization,
            public_key_bytes=alternate_key,
            schedule_seal_sha256=str(schedule["seal_sha256"]),
            policy_seal_sha256=str(policy["seal_sha256"]),
            external_source_receipt_seal_sha256=str(external["seal_sha256"]),
            production_lease_id=str(lease["lease_id"]),
            hardware_identity=str(lease["hardware_identity"]),
            production_lease_receipt_sha256=__import__("hashlib").sha256(canonical(lease)).hexdigest(),
            operator_executable_sha256=str(operator["executable_sha256"]),
            operator_receipt_public_key_sha256=str(operator["receipt_public_key_sha256"]),
            range_executor_executable_sha256=_RANGE_EXECUTOR_SHA256,
            now_ns=100,
        )

    tampered = {**authorization, "payload": {**authorization["payload"], "production_lease_id": "attacker"}}
    with pytest.raises(RestreamGuardError, match="exact launch inputs"):
        verify_owner_green_light_authorization(
            tampered,
            public_key_bytes=public_key,
            schedule_seal_sha256=str(schedule["seal_sha256"]),
            policy_seal_sha256=str(policy["seal_sha256"]),
            external_source_receipt_seal_sha256=str(external["seal_sha256"]),
            production_lease_id=str(lease["lease_id"]),
            hardware_identity=str(lease["hardware_identity"]),
            production_lease_receipt_sha256=__import__("hashlib").sha256(canonical(lease)).hexdigest(),
            operator_executable_sha256=str(operator["executable_sha256"]),
            operator_receipt_public_key_sha256=str(operator["receipt_public_key_sha256"]),
            range_executor_executable_sha256=_RANGE_EXECUTOR_SHA256,
            now_ns=100,
        )
    with pytest.raises(RestreamGuardError, match="expired"):
        verify_owner_green_light_authorization(
            authorization,
            public_key_bytes=public_key,
            schedule_seal_sha256=str(schedule["seal_sha256"]),
            policy_seal_sha256=str(policy["seal_sha256"]),
            external_source_receipt_seal_sha256=str(external["seal_sha256"]),
            production_lease_id=str(lease["lease_id"]),
            hardware_identity=str(lease["hardware_identity"]),
            production_lease_receipt_sha256=__import__("hashlib").sha256(canonical(lease)).hexdigest(),
            operator_executable_sha256=str(operator["executable_sha256"]),
            operator_receipt_public_key_sha256=str(operator["receipt_public_key_sha256"]),
            range_executor_executable_sha256=_RANGE_EXECUTOR_SHA256,
            now_ns=102,
        )


def test_owner_authorization_nonce_is_atomic_single_use(tmp_path: Path) -> None:
    schedule = _schedule()
    policy = _policy(schedule)
    external = _external_receipt()
    lease = _production_lease(100)
    operator = _operator_approval()
    authorization, public_key = _owner_authorization(schedule, policy, external, lease, operator, 100)
    transition = evaluate_green_light_transition(
        schedule,
        policy,
        free_bytes=4_000_000,
        model_lane_file_count=0,
        external_source_receipt=external,
        parent_authorized=True,
        owner_authorization_receipt=authorization,
        owner_public_key_bytes=public_key,
        production_lease_receipt=lease,
        operator_approval_receipt=operator,
        range_executor_executable_sha256=_RANGE_EXECUTOR_SHA256,
        source_hashes_verified=True,
        runtime_pins_verified=True,
        now_ns=100,
    )
    ledger = tmp_path / "uses"
    claim_args = {
        "public_key_bytes": public_key,
        "schedule": schedule,
        "policy": policy,
        "external_source_receipt": external,
        "production_lease_receipt": lease,
        "operator_approval_receipt": operator,
        "range_executor_executable_sha256": _RANGE_EXECUTOR_SHA256,
        "now_ns": 100,
    }
    claimed = claim_single_use_owner_authorization(ledger, authorization, transition, **claim_args)
    assert claimed["status"] == "CLAIMED_SINGLE_USE_BEFORE_EXEC"
    assert len(list(ledger.glob("*.json"))) == 1
    with pytest.raises(RestreamGuardError, match="already consumed"):
        claim_single_use_owner_authorization(ledger, authorization, transition, **claim_args)


def test_claim_is_atomically_carried_into_executor_and_preflight_replacement_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator_path = tmp_path / "operator"
    executor_path = tmp_path / "executor"
    operator_path.write_bytes(b"#!/bin/sh\nexit 0\n")
    executor_path.write_bytes(b"#!/bin/sh\nexit 0\n")
    operator_path.chmod(0o700)
    executor_path.chmod(0o700)
    operator_sha = __import__("hashlib").sha256(operator_path.read_bytes()).hexdigest()
    executor_sha = __import__("hashlib").sha256(executor_path.read_bytes()).hexdigest()

    schedule = _schedule()
    policy = _policy(schedule)
    external = _external_receipt()
    lease = _production_lease(100)
    operator = _operator_approval(operator_sha)
    authorization, public_key = _owner_authorization(
        schedule, policy, external, lease, operator, 100, executor_sha
    )
    transition = evaluate_green_light_transition(
        schedule,
        policy,
        free_bytes=4_000_000,
        model_lane_file_count=0,
        external_source_receipt=external,
        parent_authorized=True,
        owner_authorization_receipt=authorization,
        owner_public_key_bytes=public_key,
        production_lease_receipt=lease,
        operator_approval_receipt=operator,
        range_executor_executable_sha256=executor_sha,
        source_hashes_verified=True,
        runtime_pins_verified=True,
        now_ns=100,
    )
    ledger = tmp_path / "uses"
    claim_single_use_owner_authorization(
        ledger,
        authorization,
        transition,
        public_key_bytes=public_key,
        schedule=schedule,
        policy=policy,
        external_source_receipt=external,
        production_lease_receipt=lease,
        operator_approval_receipt=operator,
        range_executor_executable_sha256=executor_sha,
        now_ns=100,
    )
    claim_path = owner_authorization_claim_path(ledger, authorization)
    monkeypatch.setenv("HAWKING_PARENT_RESTREAM_AUTHORIZED", "YES")
    monkeypatch.setenv("HAWKING_CLEAN_GPU_LEASE_ID", str(lease["lease_id"]))

    # A sealed but caller-created STARTED record is not evidence that the
    # nonce's O_EXCL CLAIMED predecessor existed.  Only the executor's start
    # transition carries the complete sealed predecessor.
    claimed_before_start = __import__("lab.operators.glm52_common", fromlist=["read_sealed_json"]).read_sealed_json(claim_path)
    forged_started = seal({
        **{key: value for key, value in claimed_before_start.items() if key not in {"seal_sha256", "status"}},
        "status": "STARTED_SINGLE_USE_IN_EXECUTOR",
        "claimed_receipt_seal_sha256": "f" * 64,
        "started_unix_ns": 100,
    })
    from lab.operators.glm52_common import atomic_json
    atomic_json(claim_path, forged_started)
    with pytest.raises(RestreamGuardError, match="sealed CLAIMED predecessor"):
        start_claimed_owner_authorization(
            claim_path,
            schedule=schedule,
            policy=policy,
            final_preflight=transition,
            operator_path=operator_path,
            range_executor_path=executor_path,
            public_key_bytes=public_key,
            now_ns=100,
            allow_started_resume=True,
        )
    atomic_json(claim_path, claimed_before_start)

    forged_preflight = seal({
        **{key: value for key, value in transition.items() if key != "seal_sha256"},
        "input_commitment_sha256": "0" * 64,
    })
    with pytest.raises(RestreamGuardError, match="changed after"):
        start_claimed_owner_authorization(
            claim_path,
            schedule=schedule,
            policy=policy,
            final_preflight=forged_preflight,
            operator_path=operator_path,
            range_executor_path=executor_path,
            public_key_bytes=public_key,
            now_ns=100,
        )

    started = start_claimed_owner_authorization(
        claim_path,
        schedule=schedule,
        policy=policy,
        final_preflight=transition,
        operator_path=operator_path,
        range_executor_path=executor_path,
        public_key_bytes=public_key,
        now_ns=100,
    )
    assert started["status"] == "STARTED_SINGLE_USE_IN_EXECUTOR"
    with pytest.raises(RestreamGuardError, match="CLAIMED state"):
        start_claimed_owner_authorization(
            claim_path,
            schedule=schedule,
            policy=policy,
            final_preflight=transition,
            operator_path=operator_path,
            range_executor_path=executor_path,
            public_key_bytes=public_key,
            now_ns=100,
        )
    resumed = start_claimed_owner_authorization(
        claim_path,
        schedule=schedule,
        policy=policy,
        final_preflight=transition,
        operator_path=operator_path,
        range_executor_path=executor_path,
        public_key_bytes=public_key,
        now_ns=100,
        allow_started_resume=True,
    )
    assert resumed == started
    with pytest.raises(RestreamGuardError, match="expired"):
        start_claimed_owner_authorization(
            claim_path,
            schedule=schedule,
            policy=policy,
            final_preflight=transition,
            operator_path=operator_path,
            range_executor_path=executor_path,
            public_key_bytes=public_key,
            now_ns=102,
            allow_started_resume=True,
        )
