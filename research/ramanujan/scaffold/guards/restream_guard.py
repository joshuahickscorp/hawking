"""Fail-closed structural validation for the future bounded parent restream."""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from lab.operators.glm52_common import atomic_json, canonical, read_sealed_json, seal, verify_sealed
from ramanujan.layout import BOUNDARY_ROOT, REPO_ROOT, resolve_ramanujan_path


ENVELOPE_BYTES = 90_000_000_000
ALIGNMENT_BYTES = 64 * 1024
ACCOUNTING_COMPONENTS = (
    "source_range_rounded_bytes",
    "source_scratch_bytes",
    "retained_artifact_bytes",
    "teacher_evidence_bytes",
    "metadata_bytes",
    "carry_bytes",
    "prefetch_bytes",
)


class RestreamGuardError(ValueError):
    """A proposed restream schedule or policy omits bounded storage accounting."""


GREEN_LIGHT_STATES = (
    "OWNER_GATE_READY",
    "STORAGE_GREEN",
    "AUTHORIZATION_GREEN",
    "PRODUCTION_LEASE_GREEN",
    "OPERATOR_APPROVED",
    "FINAL_PREFLIGHT",
)
TRANSITION_SCHEMA = "hawking.ramanujan.green_light_transition.v1"
EXTERNAL_RECEIPT_SCHEMA = "hawking.ramanujan.external_source_freeze_receipt.v1"
PRODUCTION_LEASE_SCHEMA = "hawking.gpu.production_lease.v1"
OPERATOR_APPROVAL_SCHEMA = "hawking.glm52.operator_approval.v1"
OWNER_AUTHORIZATION_SCHEMA = "hawking.ramanujan.owner_green_light_authorization.v1"
OWNER_AUTHORIZATION_USE_SCHEMA = "hawking.ramanujan.owner_authorization_use.v2"
FRAMED_PROTOCOL = "hawking.glm52.window_stream.framed.v1"


def pinned_owner_public_key_path() -> Path:
    """Return the machine-admin-owned owner trust anchor.

    A key below the executor user's home is not an owner boundary: any process
    running as that user can replace it and sign a counterfeit authorization.
    This intentionally uses a root-owned system location.  If installation has
    not happened, the owner gate remains false rather than accepting a weaker
    per-user substitute.
    """
    return Path("/Library/Application Support/Hawking/owner_ed25519_public_key")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    """Hash an exact signed/attested mapping without trusting presentation bytes."""
    return hashlib.sha256(canonical(dict(value))).hexdigest()


def _nonnegative(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RestreamGuardError(f"{label} must be a non-negative integer")
    return value


def _components(value: object, label: str) -> None:
    if not isinstance(value, list) or tuple(value) != ACCOUNTING_COMPONENTS:
        raise RestreamGuardError(f"{label} must declare the complete ordered incremental accounting components")


def validate_bounded_restream(schedule: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, int]:
    """Validate the sealed schedule/policy shape required before a restream.

    Callers must verify the two seals before this function. This checks the
    semantic linkage and makes every storage class explicit in every window;
    logical source bytes alone are deliberately insufficient.
    """
    if not isinstance(schedule, Mapping) or not isinstance(policy, Mapping):
        raise RestreamGuardError("schedule and policy must be mappings")
    if schedule.get("active_model_limit") != 1:
        raise RestreamGuardError("schedule must bind active_model_limit=1")
    contract = schedule.get("incremental_accounting_contract")
    if not isinstance(contract, Mapping):
        raise RestreamGuardError("schedule lacks incremental_accounting_contract")
    if contract.get("alignment_bytes") != ALIGNMENT_BYTES:
        raise RestreamGuardError("schedule must bind 64-KiB conservative alignment")
    _components(contract.get("components"), "schedule incremental_accounting_contract.components")
    schedule_seal = schedule.get("seal_sha256")
    if not isinstance(schedule_seal, str) or len(schedule_seal) != 64:
        raise RestreamGuardError("schedule must carry a seal_sha256")

    policy_fields = policy.get("policy")
    input_seals = policy.get("input_seals")
    if not isinstance(policy_fields, Mapping) or not isinstance(input_seals, Mapping):
        raise RestreamGuardError("policy lacks policy or input_seals mapping")
    if policy_fields.get("incremental_storage_ceiling_bytes") != ENVELOPE_BYTES:
        raise RestreamGuardError("policy must bind incremental_storage_ceiling_bytes=90000000000")
    if policy_fields.get("active_model_limit") != 1:
        raise RestreamGuardError("policy must bind active_model_limit=1")
    _components(policy_fields.get("incremental_accounting_components"), "policy incremental_accounting_components")
    if input_seals.get("streaming_schedule") != schedule_seal:
        raise RestreamGuardError("policy is not sealed against this exact streaming schedule")

    windows = schedule.get("windows")
    if not isinstance(windows, list) or not windows:
        raise RestreamGuardError("schedule must contain at least one window")
    peak = 0
    allowed = set(ACCOUNTING_COMPONENTS) | {"resident_incremental_bytes"}
    for number, window in enumerate(windows):
        if not isinstance(window, Mapping):
            raise RestreamGuardError(f"schedule window {number} is not a mapping")
        accounting = window.get("incremental_accounting")
        if not isinstance(accounting, Mapping):
            raise RestreamGuardError(f"schedule window {number} lacks incremental_accounting")
        if set(accounting) != allowed:
            raise RestreamGuardError(
                f"schedule window {number} must account for exactly the declared storage components"
            )
        values = {
            name: _nonnegative(accounting[name], f"window {number} {name}") for name in ACCOUNTING_COMPONENTS
        }
        if any(value % ALIGNMENT_BYTES for value in values.values()):
            raise RestreamGuardError(f"schedule window {number} has a component not conservatively 64-KiB rounded")
        subtotal = sum(values.values())
        declared = _nonnegative(accounting["resident_incremental_bytes"], f"window {number} resident_incremental_bytes")
        if declared != subtotal:
            raise RestreamGuardError(
                f"schedule window {number} resident_incremental_bytes={declared} does not equal component sum={subtotal}"
            )
        peak = max(peak, declared)
    if peak > ENVELOPE_BYTES:
        raise RestreamGuardError(f"schedule incremental peak {peak} exceeds 90-GB envelope")
    return {"peak_incremental_bytes": peak, "window_count": len(windows)}


def validate_external_source_freeze_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the public non-authorizing D5/D8/D9 owner receipt.

    Raw D8 membership and private source paths are deliberately absent from
    this boundary.  The receipt can satisfy an owner-input gate; it can never
    authorize research, a candidate or a parent restream.
    """
    try:
        value = verify_sealed(dict(receipt), label="external source freeze receipt")
    except Exception as exc:  # noqa: BLE001 - normalize the shared seal authority.
        raise RestreamGuardError(f"external source freeze receipt is not sealed: {exc}") from exc
    if value.get("schema") != EXTERNAL_RECEIPT_SCHEMA:
        raise RestreamGuardError("external source freeze receipt schema mismatch")
    if value.get("status") != "PASS_INPUTS_FROZEN_RESEARCH_AND_CANDIDATE_AUTHORITY_FALSE":
        raise RestreamGuardError("external source receipt is not a non-authorizing PASS freeze")
    false_fields = (
        "RAMANUJAN_RESEARCH_AUTHORIZED",
        "candidate_launch_authorized",
        "independent_adjudication_complete",
        "counterexample_search_complete",
    )
    if any(value.get(name) is not False for name in false_fields):
        raise RestreamGuardError("external source receipt grants or claims unfinished authority")
    rows = value.get("sources")
    if not isinstance(rows, list) or len(rows) != 3 or any(not isinstance(row, Mapping) for row in rows):
        raise RestreamGuardError("external source receipt must contain exactly D5, D8 and D9")
    if [row.get("id") for row in rows] != ["D5", "D8", "D9"]:
        raise RestreamGuardError("external source receipt source order must be D5, D8, D9")
    if any(row.get("status") != "FROZEN_PENDING_INDEPENDENT_EVALUATION" for row in rows):
        raise RestreamGuardError("external source receipt contains a non-frozen source")
    if any(row.get("source_path_or_item_ids_serialized") is not False for row in rows):
        raise RestreamGuardError("external source receipt serializes a forbidden source path or item id")
    by_id = {str(row["id"]): row for row in rows}
    d5 = by_id["D5"].get("no_leak_audit")
    if not isinstance(d5, Mapping) or d5.get("direction") != "D5_training_candidate_against_sealed_odyssey_evaluation" or d5.get("exact_or_near_matches") != 0:
        raise RestreamGuardError("D5 Odyssey evaluation no-leak audit is absent or nonzero")
    d8 = by_id["D8"].get("no_leak_audit")
    if not isinstance(d8, Mapping) or d8.get("direction") != "all_current_D1_D7_training_against_D8_hidden_membership" or d8.get("exact_or_near_matches") != 0:
        raise RestreamGuardError("D8 hidden-membership no-leak audit is absent or nonzero")
    visible = value.get("training_visible")
    if not isinstance(visible, Mapping) or visible.get("D8_hidden_item_ids") is not None or visible.get("D8_commitment_only") is not True:
        raise RestreamGuardError("D8 privacy boundary is absent")
    generator = by_id["D9"].get("variant_generator")
    if not isinstance(generator, Mapping) or generator.get("executed_by_freeze") is not False:
        raise RestreamGuardError("D9 generator binding is absent or was executed by freeze")
    for name in ("executable_sha256", "seed_commitment_sha256"):
        digest = generator.get(name)
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise RestreamGuardError(f"D9 generator {name} is not an exact lowercase SHA-256")
    return value


def _production_lease_green(value: object, *, now_ns: int) -> tuple[bool, str]:
    if not isinstance(value, Mapping):
        return False, "production lease receipt is absent"
    if value.get("schema") != PRODUCTION_LEASE_SCHEMA or value.get("production_authority") is not True or value.get("fixture_only") is not False:
        return False, "production lease receipt is not production-authoritative"
    if value.get("contention_label") != "CLEAN" or value.get("foreign_processes") != []:
        return False, "production lease is contended or names foreign work"
    if any(not isinstance(value.get(name), str) or not value[name].strip() for name in ("lease_id", "hardware_identity")):
        return False, "production lease lacks owner identity or hardware identity"
    pid = value.get("pid")
    heartbeat = value.get("heartbeat_unix_ns")
    expires = value.get("expires_unix_ns")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False, "production lease pid is invalid"
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (heartbeat, expires)):
        return False, "production lease heartbeat or expiry is invalid"
    if not heartbeat <= now_ns <= expires:
        return False, "production lease heartbeat window is stale or not yet valid"
    return True, "production lease is CLEAN, current and hardware-bound"


def _operator_approval_green(value: object) -> tuple[bool, str]:
    if not isinstance(value, Mapping):
        return False, "operator approval receipt is absent"
    if value.get("schema") != OPERATOR_APPROVAL_SCHEMA or value.get("owner_approved") is not True:
        return False, "physical operator is not owner-approved"
    if value.get("protocol") != FRAMED_PROTOCOL or value.get("production_authority") is not True or value.get("fixture_only") is not False:
        return False, "operator approval does not bind the production framed protocol"
    expected = value.get("executable_sha256")
    observed = value.get("observed_executable_sha256")
    if not isinstance(expected, str) or len(expected) != 64 or expected != observed:
        return False, "operator executable hash is absent or differs from approval"
    receipt_key_hex = value.get("receipt_public_key_ed25519_hex")
    receipt_key_hash = value.get("receipt_public_key_sha256")
    try:
        receipt_key = bytes.fromhex(receipt_key_hex) if isinstance(receipt_key_hex, str) else b""
    except ValueError:
        receipt_key = b""
    if len(receipt_key) != 32 or hashlib.sha256(receipt_key).hexdigest() != receipt_key_hash:
        return False, "operator approval lacks a valid receipt-attestation public key"
    return True, "operator approval binds the observed executable SHA-256"


def verify_owner_green_light_authorization(
    value: Mapping[str, Any],
    *,
    public_key_bytes: bytes,
    schedule_seal_sha256: str,
    policy_seal_sha256: str,
    external_source_receipt_seal_sha256: str,
    production_lease_id: str,
    hardware_identity: str,
    production_lease_receipt_sha256: str,
    operator_executable_sha256: str,
    operator_receipt_public_key_sha256: str,
    range_executor_executable_sha256: str,
    now_ns: int,
) -> dict[str, Any]:
    """Verify an owner Ed25519 authorization bound to every launch identity.

    The repository's canonical SHA-256 seals provide integrity linkage, not
    owner authenticity.  This signature is the independent owner-controlled
    authorization boundary and is never generated by production code.
    """
    if not isinstance(public_key_bytes, bytes) or len(public_key_bytes) != 32:
        raise RestreamGuardError("owner Ed25519 public key must be exactly 32 raw bytes")
    if set(value) != {
        "schema", "owner_public_key_sha256", "payload", "signature_ed25519_hex",
    } or value.get("schema") != OWNER_AUTHORIZATION_SCHEMA:
        raise RestreamGuardError("owner green-light authorization schema or fields differ")
    expected_key_hash = hashlib.sha256(public_key_bytes).hexdigest()
    if value.get("owner_public_key_sha256") != expected_key_hash:
        raise RestreamGuardError("owner authorization public-key fingerprint differs")
    payload = value.get("payload")
    required_payload = {
        "schedule_seal_sha256", "policy_seal_sha256", "external_source_receipt_seal_sha256",
        "parent_restream_authorized", "production_lease_id", "hardware_identity",
        "production_lease_receipt_sha256",
        "operator_executable_sha256", "operator_receipt_public_key_sha256",
        "range_executor_executable_sha256", "operator_protocol", "issued_unix_ns", "expires_unix_ns",
        "nonce", "max_uses",
    }
    if not isinstance(payload, Mapping) or set(payload) != required_payload:
        raise RestreamGuardError("owner authorization payload fields differ")
    expected = {
        "schedule_seal_sha256": schedule_seal_sha256,
        "policy_seal_sha256": policy_seal_sha256,
        "external_source_receipt_seal_sha256": external_source_receipt_seal_sha256,
        "parent_restream_authorized": True,
        "production_lease_id": production_lease_id,
        "hardware_identity": hardware_identity,
        "production_lease_receipt_sha256": production_lease_receipt_sha256,
        "operator_executable_sha256": operator_executable_sha256,
        "operator_receipt_public_key_sha256": operator_receipt_public_key_sha256,
        "range_executor_executable_sha256": range_executor_executable_sha256,
        "operator_protocol": FRAMED_PROTOCOL,
    }
    if any(payload.get(name) != item for name, item in expected.items()):
        raise RestreamGuardError("owner authorization does not bind the exact launch inputs")
    issued = payload.get("issued_unix_ns")
    expires = payload.get("expires_unix_ns")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (issued, expires)):
        raise RestreamGuardError("owner authorization issue or expiry time is invalid")
    if not issued <= now_ns <= expires:
        raise RestreamGuardError("owner authorization is expired or not yet valid")
    nonce = payload.get("nonce")
    if not isinstance(nonce, str) or len(nonce) < 32 or not nonce.strip():
        raise RestreamGuardError("owner authorization nonce is absent or too short")
    if payload.get("max_uses") != 1:
        raise RestreamGuardError("owner authorization must be single-use")
    signature_hex = value.get("signature_ed25519_hex")
    if not isinstance(signature_hex, str) or len(signature_hex) != 128:
        raise RestreamGuardError("owner Ed25519 signature is not 64-byte hex")
    try:
        signature = bytes.fromhex(signature_hex)
        Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(signature, canonical(dict(payload)))
    except (ValueError, InvalidSignature) as exc:
        raise RestreamGuardError("owner Ed25519 signature verification failed") from exc
    return dict(value)


def evaluate_green_light_transition(
    schedule: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    free_bytes: int,
    model_lane_file_count: int,
    external_source_receipt: Mapping[str, Any] | None,
    parent_authorized: bool,
    owner_authorization_receipt: Mapping[str, Any] | None = None,
    owner_public_key_bytes: bytes | None = None,
    production_lease_receipt: Mapping[str, Any] | None,
    operator_approval_receipt: Mapping[str, Any] | None,
    range_executor_executable_sha256: str = "",
    source_hashes_verified: bool,
    runtime_pins_verified: bool,
    now_ns: int | None = None,
    simulate_admission: bool = False,
) -> dict[str, Any]:
    """Evaluate every green-light gate in fixed order and return a sealed receipt.

    ``simulate_admission`` exercises the final edge but always emits a
    non-production ``SIMULATED_RESTREAM_ADMITTED`` state.  Actual admission is
    the launcher's successful ``exec`` after a separately persisted
    ``FINAL_PREFLIGHT`` receipt.
    """
    bounded = validate_bounded_restream(schedule, policy)
    policy_fields = policy.get("policy")
    if not isinstance(policy_fields, Mapping):
        raise RestreamGuardError("policy mapping disappeared after bounded validation")
    floor = _nonnegative(policy_fields.get("protected_filesystem_floor_bytes"), "protected filesystem floor")
    free = _nonnegative(free_bytes, "free bytes")
    files = _nonnegative(model_lane_file_count, "model lane file count")
    current_ns = time.time_ns() if now_ns is None else _nonnegative(now_ns, "now_ns")

    satisfied: list[str] = ["OWNER_GATE_READY"]
    missing: list[str] = []
    details: dict[str, Any] = {}
    storage_green = files == 0 and free - bounded["peak_incremental_bytes"] >= floor
    details["storage"] = {
        "green": storage_green,
        "free_bytes": free,
        "protected_floor_bytes": floor,
        "required_incremental_bytes": bounded["peak_incremental_bytes"],
        "residual_after_incremental_bytes": free - bounded["peak_incremental_bytes"] - floor,
        "model_lane_file_count": files,
    }
    if storage_green:
        satisfied.append("STORAGE_GREEN")
    else:
        missing.append("STORAGE_GREEN")

    source_receipt_error: str | None = None
    try:
        sources_green = external_source_receipt is not None and bool(validate_external_source_freeze_receipt(external_source_receipt))
    except RestreamGuardError as exc:
        sources_green = False
        source_receipt_error = str(exc)
    owner_authorization_error: str | None = None
    try:
        owner_authorization_green = owner_authorization_receipt is not None and owner_public_key_bytes is not None and bool(
            verify_owner_green_light_authorization(
                owner_authorization_receipt,
                public_key_bytes=owner_public_key_bytes,
                schedule_seal_sha256=str(schedule.get("seal_sha256", "")),
                policy_seal_sha256=str(policy.get("seal_sha256", "")),
                external_source_receipt_seal_sha256="" if external_source_receipt is None else str(external_source_receipt.get("seal_sha256", "")),
                production_lease_id="" if production_lease_receipt is None else str(production_lease_receipt.get("lease_id", "")),
                hardware_identity="" if production_lease_receipt is None else str(production_lease_receipt.get("hardware_identity", "")),
                production_lease_receipt_sha256="" if production_lease_receipt is None else _canonical_sha256(production_lease_receipt),
                operator_executable_sha256="" if operator_approval_receipt is None else str(operator_approval_receipt.get("executable_sha256", "")),
                operator_receipt_public_key_sha256="" if operator_approval_receipt is None else str(operator_approval_receipt.get("receipt_public_key_sha256", "")),
                range_executor_executable_sha256=range_executor_executable_sha256,
                now_ns=current_ns,
            )
        )
    except RestreamGuardError as exc:
        owner_authorization_green = False
        owner_authorization_error = str(exc)
    authorization_green = storage_green and sources_green and parent_authorized is True and owner_authorization_green
    details["authorization"] = {
        "green": authorization_green,
        "D5_D8_D9_public_receipt_green": sources_green,
        "parent_authorized": parent_authorized is True,
        "owner_signature_green": owner_authorization_green,
        "owner_signature_error": owner_authorization_error,
        "receipt_error": source_receipt_error,
    }
    if authorization_green:
        satisfied.append("AUTHORIZATION_GREEN")
    else:
        if not sources_green:
            missing.extend(["OWNER_D5_APPROVAL", "OWNER_D8_APPROVAL", "OWNER_D9_APPROVAL"])
        if parent_authorized is not True:
            missing.append("OWNER_PARENT_RESTREAM_AUTHORIZATION")
        if not owner_authorization_green:
            missing.append("OWNER_SIGNED_GREEN_LIGHT_AUTHORIZATION")

    lease_ok, lease_reason = _production_lease_green(production_lease_receipt, now_ns=current_ns)
    lease_green = authorization_green and lease_ok
    details["production_lease"] = {"green": lease_green, "mechanism_green": lease_ok, "reason": lease_reason}
    if lease_green:
        satisfied.append("PRODUCTION_LEASE_GREEN")
    else:
        missing.append("PRODUCTION_CLEAN_GPU_LEASE_IDENTITY")

    operator_ok, operator_reason = _operator_approval_green(operator_approval_receipt)
    operator_green = lease_green and operator_ok
    details["operator"] = {"green": operator_green, "mechanism_green": operator_ok, "reason": operator_reason}
    if operator_green:
        satisfied.append("OPERATOR_APPROVED")
    else:
        missing.append("OWNER_APPROVAL_OF_PRODUCTION_PHYSICAL_GLM52_WINDOW_OPERATOR")

    final_green = operator_green and source_hashes_verified is True and runtime_pins_verified is True
    details["final_preflight"] = {
        "green": final_green,
        "source_hashes_verified": source_hashes_verified is True,
        "runtime_pins_verified": runtime_pins_verified is True,
    }
    if final_green:
        satisfied.append("FINAL_PREFLIGHT")
    else:
        if not source_hashes_verified:
            missing.append("SOURCE_HASH_REVALIDATION")
        if not runtime_pins_verified:
            missing.append("RUNTIME_PIN_REVALIDATION")

    current_state = satisfied[-1]
    if simulate_admission and final_green:
        current_state = "SIMULATED_RESTREAM_ADMITTED"
    deduped_missing = list(dict.fromkeys(missing))
    input_commitment = hashlib.sha256(canonical({
        "schedule": schedule.get("seal_sha256"),
        "policy": policy.get("seal_sha256"),
        "free_bytes": free,
        "model_lane_file_count": files,
        "external_source_receipt": None if external_source_receipt is None else external_source_receipt.get("seal_sha256"),
        "parent_authorized": parent_authorized is True,
        "owner_authorization": None if owner_authorization_receipt is None else dict(owner_authorization_receipt),
        "owner_public_key_sha256": None if owner_public_key_bytes is None else hashlib.sha256(owner_public_key_bytes).hexdigest(),
        "production_lease": None if production_lease_receipt is None else dict(production_lease_receipt),
        "operator_approval": None if operator_approval_receipt is None else dict(operator_approval_receipt),
        "range_executor_executable_sha256": range_executor_executable_sha256,
        "source_hashes_verified": source_hashes_verified is True,
        "runtime_pins_verified": runtime_pins_verified is True,
        "now_ns": current_ns,
        "simulate_admission": simulate_admission,
    })).hexdigest()
    launch_binding = {
        "owner_public_key_sha256": None if owner_public_key_bytes is None else hashlib.sha256(owner_public_key_bytes).hexdigest(),
        "owner_authorization_signature_sha256": None if owner_authorization_receipt is None else hashlib.sha256(
            str(owner_authorization_receipt.get("signature_ed25519_hex", "")).encode("ascii")
        ).hexdigest(),
        "owner_authorization_nonce_sha256": None if owner_authorization_receipt is None else hashlib.sha256(
            str(owner_authorization_receipt.get("payload", {}).get("nonce", "")).encode("utf-8")
        ).hexdigest(),
        "external_source_receipt_seal_sha256": None if external_source_receipt is None else external_source_receipt.get("seal_sha256"),
        "production_lease_id": None if production_lease_receipt is None else production_lease_receipt.get("lease_id"),
        "hardware_identity": None if production_lease_receipt is None else production_lease_receipt.get("hardware_identity"),
        "production_lease_receipt_sha256": None if production_lease_receipt is None else _canonical_sha256(production_lease_receipt),
        "operator_executable_sha256": None if operator_approval_receipt is None else operator_approval_receipt.get("executable_sha256"),
        "operator_receipt_public_key_sha256": None if operator_approval_receipt is None else operator_approval_receipt.get("receipt_public_key_sha256"),
        "range_executor_executable_sha256": range_executor_executable_sha256 or None,
    }
    return seal({
        "schema": TRANSITION_SCHEMA,
        "status": current_state,
        "fixture_only": simulate_admission,
        "production_authority": False if simulate_admission else final_green,
        "restream_started": False,
        "schedule_seal_sha256": schedule.get("seal_sha256"),
        "policy_seal_sha256": policy.get("seal_sha256"),
        "input_commitment_sha256": input_commitment,
        "satisfied_gates": satisfied,
        "missing_gates": deduped_missing,
        "details": details,
        "launch_binding": launch_binding,
        "exact_next_action": "EXEC_RANGE_RESTREAM" if final_green and not simulate_admission else (deduped_missing[0] if deduped_missing else "SIMULATION_COMPLETE_NO_EXEC"),
    })


def persist_green_light_transition(path: Path, transition: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically persist or resume one exact transition snapshot.

    Equal input commitments resume byte-for-byte.  A changed gate snapshot
    replaces the stale state and binds the predecessor seal, including safe
    regressions when an approval, lease heartbeat or storage gate disappears.
    """
    current = verify_sealed(dict(transition), label="green-light transition")
    if current.get("schema") != TRANSITION_SCHEMA:
        raise RestreamGuardError("green-light transition schema mismatch")
    previous: dict[str, Any] | None = None
    if path.is_file():
        try:
            previous = read_sealed_json(path)
        except Exception as exc:  # noqa: BLE001
            raise RestreamGuardError(f"existing green-light transition is invalid: {exc}") from exc
        if previous.get("input_commitment_sha256") == current.get("input_commitment_sha256"):
            return previous
    persisted = seal({
        **{key: value for key, value in current.items() if key != "seal_sha256"},
        "previous_transition_seal_sha256": None if previous is None else previous.get("seal_sha256"),
    })
    atomic_json(path, persisted)
    return persisted


def _load_json_if_file(raw_path: str, *, sealed: bool = False) -> dict[str, Any] | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_file() or path.is_symlink():
        return None
    if sealed:
        return read_sealed_json(path)
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RestreamGuardError(f"{path} is not a JSON object")
    return value


def load_pinned_owner_public_key() -> bytes | None:
    """Load only a root-owned, non-replaceable-in-user-space trust anchor."""
    import os
    import stat

    path = pinned_owner_public_key_path()
    if not path.is_file() or path.is_symlink():
        return None
    info = path.stat()
    if info.st_uid != 0 or stat.S_IMODE(info.st_mode) & 0o022:
        return None
    # Verify every mutable path component down from /Library.  A safe leaf is
    # insufficient if a same-user writable parent can be atomically replaced.
    for parent in (Path("/Library"), Path("/Library/Application Support"), path.parent):
        try:
            parent_info = parent.lstat()
        except OSError:
            return None
        if (
            stat.S_ISLNK(parent_info.st_mode)
            or not stat.S_ISDIR(parent_info.st_mode)
            or parent_info.st_uid != 0
            or stat.S_IMODE(parent_info.st_mode) & 0o022
        ):
            return None
    raw = path.read_bytes()
    if len(raw) == 32:
        return raw
    try:
        decoded = bytes.fromhex(raw.decode("ascii").strip())
    except (UnicodeDecodeError, ValueError):
        return None
    return decoded if len(decoded) == 32 else None


def _regular_model_lane_files(repo_root: Path) -> list[str]:
    # The primary live staging lane is operational state, not the immutable
    # evidence catalog.  Retain the old root lane as a read-only compatibility
    # check so an un-migrated artifact cannot be silently ignored.
    roots = [
        repo_root / "workspace/ops/local/models",
        repo_root / "models",
        Path("/tmp/hawking-tg-artifacts"),
    ]
    files: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_symlink():
            files.append(f"REFUSED_SYMLINK:{root}")
            continue
        for path in root.rglob("*"):
            if path.is_symlink():
                files.append(f"REFUSED_SYMLINK:{path}")
            elif path.is_file():
                files.append(str(path.resolve()))
    return sorted(files)


def claim_single_use_owner_authorization(
    ledger_dir: Path,
    authorization: Mapping[str, Any],
    transition: Mapping[str, Any],
    *,
    public_key_bytes: bytes,
    schedule: Mapping[str, Any],
    policy: Mapping[str, Any],
    external_source_receipt: Mapping[str, Any],
    production_lease_receipt: Mapping[str, Any],
    operator_approval_receipt: Mapping[str, Any],
    range_executor_executable_sha256: str,
    now_ns: int | None = None,
) -> dict[str, Any]:
    """Verify and atomically consume one signed owner nonce before launch."""
    import json
    import os

    verified_transition = verify_sealed(dict(transition), label="green-light final preflight")
    if verified_transition.get("status") != "FINAL_PREFLIGHT" or verified_transition.get("production_authority") is not True:
        raise RestreamGuardError("owner authorization cannot be consumed before FINAL_PREFLIGHT")
    validate_external_source_freeze_receipt(external_source_receipt)
    lease_ok, lease_reason = _production_lease_green(
        production_lease_receipt,
        now_ns=time.time_ns() if now_ns is None else _nonnegative(now_ns, "now_ns"),
    )
    if not lease_ok:
        raise RestreamGuardError(f"owner authorization launch lease is not green: {lease_reason}")
    operator_ok, operator_reason = _operator_approval_green(operator_approval_receipt)
    if not operator_ok:
        raise RestreamGuardError(f"owner authorization launch operator is not green: {operator_reason}")
    verified_authorization = verify_owner_green_light_authorization(
        authorization,
        public_key_bytes=public_key_bytes,
        schedule_seal_sha256=str(schedule.get("seal_sha256", "")),
        policy_seal_sha256=str(policy.get("seal_sha256", "")),
        external_source_receipt_seal_sha256=str(external_source_receipt.get("seal_sha256", "")),
        production_lease_id=str(production_lease_receipt.get("lease_id", "")),
        hardware_identity=str(production_lease_receipt.get("hardware_identity", "")),
        production_lease_receipt_sha256=_canonical_sha256(production_lease_receipt),
        operator_executable_sha256=str(operator_approval_receipt.get("executable_sha256", "")),
        operator_receipt_public_key_sha256=str(operator_approval_receipt.get("receipt_public_key_sha256", "")),
        range_executor_executable_sha256=range_executor_executable_sha256,
        now_ns=time.time_ns() if now_ns is None else _nonnegative(now_ns, "now_ns"),
    )
    binding = verified_transition.get("launch_binding")
    if not isinstance(binding, Mapping):
        raise RestreamGuardError("FINAL_PREFLIGHT lacks an exact signed launch binding")
    payload = verified_authorization.get("payload")
    nonce = payload.get("nonce") if isinstance(payload, Mapping) else None
    if not isinstance(nonce, str) or len(nonce) < 32:
        raise RestreamGuardError("owner authorization nonce is absent")
    ledger_dir.mkdir(parents=True, exist_ok=True)
    if ledger_dir.is_symlink():
        raise RestreamGuardError("owner authorization ledger may not be a symlink")
    nonce_hash = hashlib.sha256(nonce.encode("utf-8")).hexdigest()
    path = ledger_dir / f"{nonce_hash}.json"
    expected_binding = {
        "owner_public_key_sha256": hashlib.sha256(public_key_bytes).hexdigest(),
        "owner_authorization_signature_sha256": hashlib.sha256(
            str(verified_authorization.get("signature_ed25519_hex", "")).encode("ascii")
        ).hexdigest(),
        "owner_authorization_nonce_sha256": nonce_hash,
        "external_source_receipt_seal_sha256": external_source_receipt.get("seal_sha256"),
        "production_lease_id": production_lease_receipt.get("lease_id"),
        "hardware_identity": production_lease_receipt.get("hardware_identity"),
        "production_lease_receipt_sha256": _canonical_sha256(production_lease_receipt),
        "operator_executable_sha256": operator_approval_receipt.get("executable_sha256"),
        "operator_receipt_public_key_sha256": operator_approval_receipt.get("receipt_public_key_sha256"),
        "range_executor_executable_sha256": range_executor_executable_sha256,
    }
    if dict(binding) != expected_binding:
        raise RestreamGuardError("FINAL_PREFLIGHT launch binding differs from signed authorization")
    claimed = seal({
        "schema": OWNER_AUTHORIZATION_USE_SCHEMA,
        "status": "CLAIMED_SINGLE_USE_BEFORE_EXEC",
        "nonce_sha256": nonce_hash,
        "owner_public_key_sha256": authorization.get("owner_public_key_sha256"),
        "authorization_signature_sha256": hashlib.sha256(
            str(authorization.get("signature_ed25519_hex", "")).encode("ascii")
        ).hexdigest(),
        "transition_seal_sha256": verified_transition.get("seal_sha256"),
        "transition_input_commitment_sha256": verified_transition.get("input_commitment_sha256"),
        "schedule_seal_sha256": schedule.get("seal_sha256"),
        "policy_seal_sha256": policy.get("seal_sha256"),
        "range_executor_executable_sha256": range_executor_executable_sha256,
        "operator_executable_sha256": operator_approval_receipt.get("executable_sha256"),
        "operator_receipt_public_key_sha256": operator_approval_receipt.get("receipt_public_key_sha256"),
        "lease_identity_sha256": hashlib.sha256(str(production_lease_receipt.get("lease_id", "")).encode()).hexdigest(),
        "authorization": verified_authorization,
        "external_source_receipt": dict(external_source_receipt),
        "production_lease_receipt": dict(production_lease_receipt),
        "operator_approval_receipt": dict(operator_approval_receipt),
        "pid": os.getpid(),
        "claimed_unix_ns": time.time_ns(),
        "recovery": "a consumed launch attempt requires a fresh owner-signed nonce",
    })
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise RestreamGuardError("owner authorization nonce was already consumed") from exc
    try:
        encoded = (json.dumps(claimed, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RestreamGuardError("short owner authorization claim write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_fd = os.open(ledger_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return claimed


def owner_authorization_claim_path(ledger_dir: Path, authorization: Mapping[str, Any]) -> Path:
    payload = authorization.get("payload")
    nonce = payload.get("nonce") if isinstance(payload, Mapping) else None
    if not isinstance(nonce, str) or len(nonce) < 32:
        raise RestreamGuardError("owner authorization nonce is absent")
    return ledger_dir / f"{hashlib.sha256(nonce.encode('utf-8')).hexdigest()}.json"


def owner_authorization_ledger_dir(repo_root: Path) -> Path:
    """Return the compact operational state lane for single-use launch claims."""
    return repo_root / "workspace/ops/local/state/glm52/owner-authorization-uses"


def start_claimed_owner_authorization(
    claim_path: Path,
    *,
    schedule: Mapping[str, Any],
    policy: Mapping[str, Any],
    final_preflight: Mapping[str, Any],
    operator_path: Path,
    range_executor_path: Path,
    public_key_bytes: bytes,
    now_ns: int | None = None,
    allow_started_resume: bool = False,
) -> dict[str, Any]:
    """Atomically carry or revalidate one consumed claim in the executor.

    ``allow_started_resume`` is reserved for the range executor while it holds
    the nonce-specific campaign flock for its entire process lifetime.  It
    resumes the same consumed campaign; it does not mint another authorization
    use or a new nonce.
    """
    import fcntl
    import os

    if claim_path.is_symlink() or not claim_path.is_file():
        raise RestreamGuardError("owner launch claim is absent or not a regular file")
    lock_path = claim_path.with_suffix(".lock")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    lock_fd = os.open(lock_path, flags, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        claim = read_sealed_json(claim_path)
        status = claim.get("status")
        if claim.get("schema") != OWNER_AUTHORIZATION_USE_SCHEMA or status not in {
            "CLAIMED_SINGLE_USE_BEFORE_EXEC",
            "STARTED_SINGLE_USE_IN_EXECUTOR",
        }:
            raise RestreamGuardError("owner launch claim is not in the single-use CLAIMED state")
        resuming = status == "STARTED_SINGLE_USE_IN_EXECUTOR"
        if resuming and not allow_started_resume:
            raise RestreamGuardError("owner launch claim is not in the single-use CLAIMED state")
        preflight = verify_sealed(dict(final_preflight), label="green-light final preflight")
        if (
            preflight.get("schema") != TRANSITION_SCHEMA
            or preflight.get("status") != "FINAL_PREFLIGHT"
            or preflight.get("production_authority") is not True
            or claim.get("transition_seal_sha256") != preflight.get("seal_sha256")
            or claim.get("transition_input_commitment_sha256") != preflight.get("input_commitment_sha256")
        ):
            raise RestreamGuardError("FINAL_PREFLIGHT changed after the owner launch claim")
        if claim.get("schedule_seal_sha256") != schedule.get("seal_sha256") or claim.get("policy_seal_sha256") != policy.get("seal_sha256"):
            raise RestreamGuardError("owner launch claim is bound to different schedule/policy inputs")
        if operator_path.is_symlink() or not operator_path.is_file() or not os.access(operator_path, os.X_OK):
            raise RestreamGuardError("claimed physical operator is absent or unsafe")
        if range_executor_path.is_symlink() or not range_executor_path.is_file() or not os.access(range_executor_path, os.X_OK):
            raise RestreamGuardError("claimed range executor is absent or unsafe")
        operator_sha = hashlib.sha256(operator_path.read_bytes()).hexdigest()
        executor_sha = hashlib.sha256(range_executor_path.read_bytes()).hexdigest()
        if claim.get("operator_executable_sha256") != operator_sha or claim.get("range_executor_executable_sha256") != executor_sha:
            raise RestreamGuardError("operator or range-executor bytes changed after owner claim")
        external = claim.get("external_source_receipt")
        lease = claim.get("production_lease_receipt")
        approval = claim.get("operator_approval_receipt")
        authorization = claim.get("authorization")
        if any(not isinstance(item, Mapping) for item in (external, lease, approval, authorization)):
            raise RestreamGuardError("owner launch claim lacks complete authority snapshots")
        validate_external_source_freeze_receipt(external)
        current_ns = time.time_ns() if now_ns is None else _nonnegative(now_ns, "now_ns")
        lease_ok, lease_reason = _production_lease_green(lease, now_ns=current_ns)
        if not lease_ok:
            raise RestreamGuardError(f"production lease expired after claim: {lease_reason}")
        approval_with_observation = {**dict(approval), "observed_executable_sha256": operator_sha}
        approval_ok, approval_reason = _operator_approval_green(approval_with_observation)
        if not approval_ok:
            raise RestreamGuardError(f"operator approval failed after claim: {approval_reason}")
        verified_auth = verify_owner_green_light_authorization(
            authorization,
            public_key_bytes=public_key_bytes,
            schedule_seal_sha256=str(schedule.get("seal_sha256", "")),
            policy_seal_sha256=str(policy.get("seal_sha256", "")),
            external_source_receipt_seal_sha256=str(external.get("seal_sha256", "")),
            production_lease_id=str(lease.get("lease_id", "")),
            hardware_identity=str(lease.get("hardware_identity", "")),
            production_lease_receipt_sha256=_canonical_sha256(lease),
            operator_executable_sha256=operator_sha,
            operator_receipt_public_key_sha256=str(approval.get("receipt_public_key_sha256", "")),
            range_executor_executable_sha256=executor_sha,
            # A resume is a fresh live execution attempt.  Never validate an
            # owner signature at a caller-controlled historical timestamp.
            now_ns=current_ns,
        )
        if hashlib.sha256(str(verified_auth["payload"]["nonce"]).encode()).hexdigest() != claim.get("nonce_sha256"):
            raise RestreamGuardError("owner launch claim nonce differs from signed authorization")
        if owner_authorization_claim_path(claim_path.parent, authorization) != claim_path:
            raise RestreamGuardError("owner launch claim path is not the canonical nonce ledger path")
        if os.environ.get("HAWKING_PARENT_RESTREAM_AUTHORIZED") != "YES" or os.environ.get("HAWKING_CLEAN_GPU_LEASE_ID") != lease.get("lease_id"):
            raise RestreamGuardError("live parent authorization or production lease identity disappeared after claim")
        if resuming:
            claimed_seal = claim.get("claimed_receipt_seal_sha256")
            claimed_receipt = claim.get("claimed_receipt")
            try:
                original_claim = verify_sealed(
                    dict(claimed_receipt) if isinstance(claimed_receipt, Mapping) else {},
                    label="claimed owner authorization receipt",
                )
            except Exception as exc:  # noqa: BLE001 - normalize the durability boundary.
                raise RestreamGuardError("started owner launch claim lacks its sealed CLAIMED predecessor") from exc
            if (
                not isinstance(claimed_seal, str)
                or original_claim.get("status") != "CLAIMED_SINGLE_USE_BEFORE_EXEC"
                or claimed_seal != original_claim.get("seal_sha256")
                or original_claim.get("nonce_sha256") != claim.get("nonce_sha256")
                or claim.get("nonce_sha256") != hashlib.sha256(
                    str(verified_auth["payload"]["nonce"]).encode()
                ).hexdigest()
            ):
                raise RestreamGuardError("started owner launch claim is not a valid consumed campaign")
            return claim
        started = seal({
            **{key: value for key, value in claim.items() if key not in {"seal_sha256", "status"}},
            "status": "STARTED_SINGLE_USE_IN_EXECUTOR",
            "claimed_receipt_seal_sha256": claim["seal_sha256"],
            # Keep the exact sealed CLAIMED predecessor.  The STARTED state is
            # an append-only transition, not a caller-assembled replacement
            # that can merely assert a plausible predecessor digest.
            "claimed_receipt": claim,
            "executor_pid": os.getpid(),
            "started_unix_ns": current_ns,
            "recovery": (
                "STARTED is terminal as an authorization use; restart may only resume this exact nonce "
                "under the nonce-specific exclusive campaign lock and live lease"
            ),
        })
        atomic_json(claim_path, started)
        return started
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def green_light_status_main(argv: list[str] | None = None) -> int:
    """Print and persist the one fail-closed owner/green-light status plane."""
    import argparse
    import importlib.metadata
    import json
    import os
    import shutil

    parser = argparse.ArgumentParser(prog="python -m ramanujan.restream_guard")
    parser.add_argument("status", nargs="?")
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument(
        "--transition-receipt",
        default=str(BOUNDARY_ROOT / "RAMANUJAN_GREEN_LIGHT_TRANSITION.json"),
    )
    parser.add_argument("--simulate-admission", action="store_true")
    parser.add_argument("--claim-launch", action="store_true")
    args = parser.parse_args(argv)
    # Ramanujan's local scaffold is deliberately not an independent launch
    # lane.  Do not even evaluate a parent restream until Hawking's handoff
    # milestone changes this explicit, read-only boundary.
    from ramanujan.status import local_status

    try:
        hawking_boundary = local_status()
    except ValueError as exc:
        print(f"REFUSED_HAWKING_COMPLETION_BOUNDARY_INVALID: {exc}")
        return 78
    if hawking_boundary.get("status") == "BUILDABLE_BUT_BLOCKED_ON_HAWKING_COMPLETION":
        print("REFUSED_HAWKING_EVOLUTION_COMPLETE: Ramanujan remains a local scaffold until the handoff gate opens.")
        return 78
    repo_root = Path(args.repo_root).resolve()
    schedule = read_sealed_json(Path(args.schedule))
    policy = read_sealed_json(Path(args.policy))
    model_lane = _regular_model_lane_files(repo_root)

    external = _load_json_if_file(os.environ.get("HAWKING_EXTERNAL_SOURCE_FREEZE_RECEIPT", ""), sealed=True)
    lease = _load_json_if_file(os.environ.get("HAWKING_PRODUCTION_GPU_LEASE_RECEIPT", ""))
    operator_approval = _load_json_if_file(os.environ.get("HAWKING_GLM52_OPERATOR_APPROVAL_RECEIPT", ""))
    authorization = _load_json_if_file(os.environ.get("HAWKING_OWNER_GREEN_LIGHT_AUTHORIZATION", ""))
    owner_key = load_pinned_owner_public_key()

    operator_path_raw = os.environ.get("HAWKING_GLM52_WINDOW_OPERATOR", "")
    operator_path = Path(operator_path_raw) if operator_path_raw else None
    operator_observed = False
    if operator_path is not None and operator_path.is_file() and not operator_path.is_symlink() and os.access(operator_path, os.X_OK):
        actual_operator_sha = hashlib.sha256(operator_path.read_bytes()).hexdigest()
        if operator_approval is not None:
            operator_approval = {**operator_approval, "observed_executable_sha256": actual_operator_sha}
        operator_observed = True
    range_executor_path = Path(os.environ.get(
        "HAWKING_GLM52_RANGE_STREAM_EXECUTOR",
        str(repo_root / "tools/condense/glm52_range_stream_executor.py"),
    ))
    range_executor_sha256 = ""
    if (
        range_executor_path.is_file()
        and not range_executor_path.is_symlink()
        and os.access(range_executor_path, os.X_OK)
    ):
        range_executor_sha256 = hashlib.sha256(range_executor_path.read_bytes()).hexdigest()

    source_hashes_verified = False
    try:
        from lab.operators.glm52_range_stream_executor import rebuild_schedule_ranges

        rebuilt = rebuild_schedule_ranges(schedule)
        source_hashes_verified = bool(rebuilt)
    except Exception:  # noqa: BLE001 - status records the missing gate below.
        source_hashes_verified = False
    expected_runtime = {"hf_xet": "1.5.2", "huggingface-hub": "1.24.0"}
    try:
        observed_runtime = {name: importlib.metadata.version(name) for name in expected_runtime}
    except importlib.metadata.PackageNotFoundError:
        observed_runtime = {}
    runtime_pins_verified = observed_runtime == expected_runtime

    # A lease id passed separately to the old launcher cannot substitute for
    # the signed, hardware-bound production receipt.
    if lease is not None and os.environ.get("HAWKING_CLEAN_GPU_LEASE_ID", "") != lease.get("lease_id"):
        lease = None

    transition = evaluate_green_light_transition(
        schedule,
        policy,
        free_bytes=shutil.disk_usage(repo_root).free,
        model_lane_file_count=len(model_lane),
        external_source_receipt=external,
        parent_authorized=os.environ.get("HAWKING_PARENT_RESTREAM_AUTHORIZED") == "YES",
        owner_authorization_receipt=authorization,
        owner_public_key_bytes=owner_key,
        production_lease_receipt=lease,
        operator_approval_receipt=operator_approval,
        range_executor_executable_sha256=range_executor_sha256,
        source_hashes_verified=source_hashes_verified,
        runtime_pins_verified=runtime_pins_verified,
        simulate_admission=args.simulate_admission,
    )
    receipt_path = Path(args.transition_receipt)
    if not receipt_path.is_absolute():
        receipt_path = resolve_ramanujan_path(receipt_path, repo_root=repo_root)
    persisted = persist_green_light_transition(receipt_path, transition)
    launch_claim = None
    launch_claim_error = None
    if args.claim_launch:
        if persisted.get("status") != "FINAL_PREFLIGHT" or authorization is None:
            launch_claim_error = "launch claim refused because FINAL_PREFLIGHT is absent"
        else:
            try:
                launch_claim = claim_single_use_owner_authorization(
                    owner_authorization_ledger_dir(repo_root),
                    authorization,
                    persisted,
                    public_key_bytes=owner_key or b"",
                    schedule=schedule,
                    policy=policy,
                    external_source_receipt=external or {},
                    production_lease_receipt=lease or {},
                    operator_approval_receipt=operator_approval or {},
                    range_executor_executable_sha256=range_executor_sha256,
                )
            except RestreamGuardError as exc:
                launch_claim_error = str(exc)
    output = {
        "status": persisted["status"],
        "satisfied_gates": persisted["satisfied_gates"],
        "missing_gates": persisted["missing_gates"],
        "current_free_bytes": persisted["details"]["storage"]["free_bytes"],
        "protected_floor_bytes": persisted["details"]["storage"]["protected_floor_bytes"],
        "model_lane_contents": model_lane,
        "operator_executable_observed": operator_observed,
        "runtime_versions": observed_runtime,
        "pinned_owner_public_key_path": str(pinned_owner_public_key_path()),
        "range_executor_sha256": range_executor_sha256 or None,
        "exact_next_owner_action": persisted["exact_next_action"],
        "transition_receipt": str(receipt_path),
        "transition_seal_sha256": persisted["seal_sha256"],
        "restream_started": False,
        "owner_authorization_use": None if launch_claim is None else launch_claim["nonce_sha256"],
        "owner_authorization_use_path": None if launch_claim is None else str(
            owner_authorization_claim_path(owner_authorization_ledger_dir(repo_root), authorization or {})
        ),
        "owner_authorization_use_error": launch_claim_error,
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    claim_green = not args.claim_launch or launch_claim is not None
    return 0 if persisted["status"] == "FINAL_PREFLIGHT" and claim_green else 78


if __name__ == "__main__":
    raise SystemExit(green_light_status_main())
