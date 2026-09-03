#!/usr/bin/env python3
"""Seal the earned Qwen80 L0 handoff assessment as joint-capture provenance.

This is a deliberately narrow, CPU-only bridge for the future same-runtime
L0(23)+L1(9) child.  It reads four explicitly supplied, already-existing
receipts: the independent assessor result, the outer terminal, its inner L0
receipt, and the actual lease-release receipt.  The four documents are pinned
to this one immutable historical chain by both canonical-document identity and
seal.  No model payload is opened and no device, lease, process, server,
watcher, or child action can occur here.

The output is evidence only.  In particular, a historical PinnedBuffer cannot
cross into a future process: any positive L1 work must re-encode L0 and append
the nine L1 prefix dispatches in one fresh runtime and one fresh TCB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.receipts import SealIntegrityError, seal, verify


WRAPPER_SCHEMA = "hawking.ascension.qwen80_l0_state_handoff_post_capture_assessor_binding.v1"
WRAPPER_STATUS = (
    "REQUIRED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_POST_CAPTURE_ASSESSMENT_"
    "BEFORE_L1_JOINT_CAPTURE"
)
ASSESSMENT_SCHEMA = "hawking.ascension.qwen80_l0_state_handoff_post_capture_assessment.v1"
ASSESSMENT_STATUS = "EARNED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_COMPONENT_L1_BINDING_NOT_EXECUTED"
OUTER_SCHEMA = "hawking.ascension.qwen80_source_token_l0_state_handoff_outer_capture.v1"
OUTER_STATUS = "CAPTURED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_OUTER_TERMINAL_PRE_L1_COMPONENT_ONLY"
INNER_SCHEMA = "hawking.ascension.qwen80_source_token_l0_state_handoff_capture.v1"
INNER_STATUS = (
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_POST_STATE_ROLLBACK_RETAINED_OUTPUT_"
    "L1_BINDING_NOT_EXECUTED_COMPONENT_ONLY"
)
RELEASE_SCHEMA = "hawking.ascension.qwen80_source_token_l0_state_handoff_quiet_metal_lease_release.v1"
RELEASE_STATUS = "RELEASED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_COMPONENT_QUIET_METAL_LEASE_AFTER_TERMINAL_CAPTURE"
RECOMMENDATION_SCHEMA = "hawking.ascension.qwen80_source_token_l0_state_handoff_recommended_lease_release_contract.v1"
RECOMMENDATION_STATUS = "RECOMMENDED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_LEASE_RELEASE_AFTER_OUTER_TERMINAL"
HANDOFF_AUTHORITY_SCHEMA = "hawking.ascension.qwen80_l0_to_layer1_handoff_authority.v1"
HANDOFF_AUTHORITY_STATUS = (
    "ASSESSED_QWEN80_SOURCE_TOKEN_L0_TO_LAYER1_HANDOFF_INCOMPLETE_MISSING_"
    "RETAINED_DEVICE_OUTPUT_AND_POST_STATE_WITNESSES"
)
HISTORICAL_CONTINUATION_SCHEMA = "hawking.ascension.qwen80_l1_source_token_continuation_readiness_contract.v1"
HISTORICAL_CONTINUATION_STATUS = "INCOMPLETE_QWEN80_SOURCE_TOKEN_L1_CONTINUATION_MISSING_TRUSTED_L0_HANDOFF_OR_AUTHORITY"

MODEL_KEY = "qwen80"
SOURCE_TOKEN_ID = 1
SESSION_ID = "qwen80-source-token-l0-next-layer"
HIDDEN_ELEMENTS = 2_048
HIDDEN_BYTES = 8_192
L0_SLOT = 0
L1_LAYER = 1
L1_SLOT = 1
L0_PREFIX_DISPATCHES = 9
L0_SUFFIX_DISPATCHES = 14
L0_TOTAL_DISPATCHES = 23
L1_PREFIX_DISPATCHES = 9
JOINT_TOTAL_DISPATCHES = 32
L0_CONV_BYTES = 98_304
L0_RECURRENT_BYTES = 2_097_152
L1_CONV_CAPACITY_BYTES = 196_608
L1_RECURRENT_CAPACITY_BYTES = 4_194_304
LEASE_ID = "d7a4e7d6d4b5e5b9204dd3d47d779e1ca77f49ba2813fb21a7c0263fc8616f1a"
LEASE_SEAL = "d4d6a45081ecef7a9fb5ba00f234624ebc7e9019c3089777b7b72ebdd6e7457b"

RECORD_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-runtime"
)
CAPTURE_ROOT = RECORD_ROOT / "QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_OUTER_CAPTURE_20260809T081620Z"


@dataclass(frozen=True)
class ExpectedEvidence:
    path: Path
    schema: str
    status: str
    seal_sha256: str
    document_sha256: str
    file_sha256: str
    bytes: int


CANONICAL_EVIDENCE: Mapping[str, ExpectedEvidence] = {
    "assessment": ExpectedEvidence(
        RECORD_ROOT / "QWEN80_L0_STATE_HANDOFF_POST_CAPTURE_ASSESSMENT_20260809T083200Z.json",
        ASSESSMENT_SCHEMA,
        ASSESSMENT_STATUS,
        "23b6021b8403b9403a9b11044d43b2ba712fbcb2b99c431936d93b16e75ddba5",
        "d85851dce965d95b98faaa3bf687b75adfdbc8c4b7ec539e0a48354303f4b7f9",
        "391b2e22ecc1f39a5b3422d1e23871eb0d69d549ec0399caf6cc02779766b407",
        3_438,
    ),
    "outer": ExpectedEvidence(
        CAPTURE_ROOT / "outer-terminal-receipt.json",
        OUTER_SCHEMA,
        OUTER_STATUS,
        "0268f5642684a95488269f88df04bab70e3b9a8e876b598aa8032d8e633bd29c",
        "4063a539206d9e3899a8edd142d12825ad31581d6f46919952e9044ec3394966",
        "5422db7dd7bf1135146566441a53a6e33fdd26ffb9a069c1fc40738117962d2e",
        11_483,
    ),
    "inner": ExpectedEvidence(
        CAPTURE_ROOT / "inner/receipt.json",
        INNER_SCHEMA,
        INNER_STATUS,
        "4389a5252bb787307c0e909c88e016eacf9e25c5d501c999db407695e7ee1171",
        "1d1ea7625f1914d53f10033be234cec78fa938cc2215f535ff7ece171ce708b1",
        "7a45ae017c1332cf4cc53a3ac3000842ec85e8916db22f4ff25622cfd5b5f71a",
        25_418,
    ),
    "release": ExpectedEvidence(
        RECORD_ROOT / "QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_QUIET_METAL_LEASE_RELEASE_20260809T081925Z.json",
        RELEASE_SCHEMA,
        RELEASE_STATUS,
        "5b49a1a85441bedd6e986b5ab2bfcdb1ac8560cc6fc30a31081b898a12a02820",
        "ad8f13579a09a560a070d9878b3f65c3608ca7f07c219cc830027333b6ffb286",
        "13f13bfba789c4bc10a745c19fe93c61497c695bbad17202e59d650c07cf761c",
        2_178,
    ),
}

HANDOFF_AUTHORITY_IDENTITY = {
    "schema": HANDOFF_AUTHORITY_SCHEMA,
    "status": HANDOFF_AUTHORITY_STATUS,
    "document_sha256": "0d17df0a225cf763c7d80f043479a0378bfbc35ed81354a1ed4e9a8ab09cb43f",
    "document_seal_sha256": "ec31baec387a1065692b7fd0c2350f54db689112e77b50046225a37d316c40ca",
}
HISTORICAL_CONTINUATION_IDENTITY = {
    "schema": HISTORICAL_CONTINUATION_SCHEMA,
    "status": HISTORICAL_CONTINUATION_STATUS,
    "document_sha256": "944ca0199f9de618478044c687c13ee2e7dbb2ebc7c2ce791fb8459a39bec590",
    "document_seal_sha256": "2c7da055d3e969cd72e6fdebf32c234d48f9614dbc0449e52e6c919a1c6e5c4c",
}
RECOMMENDATION_IDENTITY = {
    "schema": RECOMMENDATION_SCHEMA,
    "status": RECOMMENDATION_STATUS,
    "document_sha256": "0987b7aa3848b1942ae029fdea279884668bb822c54d48757e55a11b6685358f",
    "document_seal_sha256": "820d078856f3f4d356c88fcd6741142e68de819a651f2fe97d359b73d1229378",
}
MANIFEST_FILE_SHA256 = "a0fcac0401a7962402bb8cb87d5055c83667b39575f9e0f4c7470d080758aa10"
MANIFEST_SEAL_SHA256 = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b"
ADMISSION_FILE_SHA256 = "0b601b57ef459bbfba69c4adba8c9fd2b069b3cbbf6ad7beaf3cee26556e9e53"
ADMISSION_SEAL_SHA256 = "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628"


class AssessmentBindingError(ValueError):
    """The supplied records are not the one immutable L0 provenance chain."""


@dataclass(frozen=True)
class BoundEvidence:
    label: str
    path: Path
    document: dict[str, Any]
    document_sha256: str
    file_sha256: str
    bytes: int
    seal_sha256: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(
        "utf-8"
    )


def _sha256(value: bytes | object) -> str:
    raw = value if isinstance(value, bytes) else _canonical_json(value)
    return hashlib.sha256(raw).hexdigest()


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssessmentBindingError(f"{label} must be an object")
    return value


def _expect(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise AssessmentBindingError(f"{label} mismatch: expected {expected!r}, observed {observed!r}")


def _expect_bool(document: Mapping[str, Any], field: str, expected: bool, label: str) -> None:
    _expect(document.get(field), expected, f"{label}.{field}")


def _expect_int(document: Mapping[str, Any], field: str, expected: int, label: str) -> None:
    _expect(document.get(field), expected, f"{label}.{field}")


def _canonical_regular(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise AssessmentBindingError(f"{label} must be an absolute path")
    try:
        before_resolve = os.lstat(path)
    except OSError as exc:
        raise AssessmentBindingError(f"cannot stat {label}: {path}: {exc}") from exc
    if stat.S_ISLNK(before_resolve.st_mode) or not stat.S_ISREG(before_resolve.st_mode):
        raise AssessmentBindingError(f"{label} must be a regular non-symlink file")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise AssessmentBindingError(f"cannot resolve {label}: {path}: {exc}") from exc


def _read_exact(label: str, supplied: Path) -> BoundEvidence:
    expected = CANONICAL_EVIDENCE[label]
    path = _canonical_regular(supplied, label)
    if path != expected.path.resolve():
        raise AssessmentBindingError(
            f"{label} path is not the immutable canonical record; historical substitution refused"
        )
    raw = path.read_bytes()
    _expect(len(raw), expected.bytes, f"{label} byte count")
    file_sha256 = _sha256(raw)
    _expect(file_sha256, expected.file_sha256, f"{label} raw file SHA-256")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssessmentBindingError(f"{label} is not JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise AssessmentBindingError(f"{label} root must be an object")
    try:
        verified = verify(parsed, label=label)
    except SealIntegrityError as exc:
        raise AssessmentBindingError(f"{label} has an invalid seal: {exc}") from exc
    _expect(verified.get("schema"), expected.schema, f"{label} schema")
    _expect(verified.get("status"), expected.status, f"{label} status")
    _expect(verified.get("seal_sha256"), expected.seal_sha256, f"{label} seal")
    document_sha256 = _sha256(verified)
    _expect(document_sha256, expected.document_sha256, f"{label} canonical document SHA-256")
    return BoundEvidence(
        label=label,
        path=path,
        document=verified,
        document_sha256=document_sha256,
        file_sha256=file_sha256,
        bytes=len(raw),
        seal_sha256=str(verified["seal_sha256"]),
    )


def _identity(evidence: BoundEvidence) -> dict[str, Any]:
    return {
        "present": True,
        "document_sha256": evidence.document_sha256,
        "document_seal_sha256": evidence.seal_sha256,
    }


def _expect_document_identity(reference: object, evidence: BoundEvidence, label: str) -> None:
    value = _object(reference, label)
    _expect(value.get("present"), True, f"{label}.present")
    _expect(value.get("document_sha256"), evidence.document_sha256, f"{label}.document_sha256")
    observed_seal = value.get("document_seal_sha256", value.get("seal_sha256"))
    _expect(observed_seal, evidence.seal_sha256, f"{label}.document_seal_sha256")


def _expect_fixed_identity(reference: object, expected: Mapping[str, str], label: str) -> None:
    value = _object(reference, label)
    _expect(value.get("present"), True, f"{label}.present")
    _expect(value.get("document_sha256"), expected["document_sha256"], f"{label}.document_sha256")
    observed_seal = value.get("document_seal_sha256", value.get("seal_sha256"))
    _expect(observed_seal, expected["document_seal_sha256"], f"{label}.document_seal_sha256")


def _expect_file_reference(reference: object, evidence: BoundEvidence, label: str) -> None:
    value = _object(reference, label)
    _expect(value.get("present"), True, f"{label}.present")
    _expect(value.get("path"), str(evidence.path), f"{label}.path")
    _expect(value.get("bytes"), evidence.bytes, f"{label}.bytes")
    _expect(value.get("sha256"), evidence.file_sha256, f"{label}.raw file SHA-256")
    _expect(value.get("seal_sha256"), evidence.seal_sha256, f"{label}.seal_sha256")


def _expect_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise AssessmentBindingError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_state(
    state: object,
    *,
    label: str,
    allocation: str,
    capacity: int,
    hash_field: str,
) -> dict[str, Any]:
    value = _object(state, label)
    _expect(value.get("allocation_id"), allocation, f"{label}.allocation_id")
    _expect_int(value, "slot", L0_SLOT, label)
    _expect_int(value, "offset_bytes", 0, label)
    _expect_int(value, "capacity_bytes", capacity, label)
    buffer = _expect_sha256(value.get("device_buffer_id"), f"{label}.device_buffer_id")
    state_hash = _expect_sha256(value.get(hash_field), f"{label}.{hash_field}")
    return {"allocation_id": allocation, "device_buffer_id": buffer, hash_field: state_hash}


def _validate_inner(inner: BoundEvidence) -> dict[str, Any]:
    root = inner.document
    _expect(root.get("mode"), "metal", "inner.mode")
    _expect_bool(root, "metal_device_or_dispatch_performed", True, "inner")
    _expect_bool(root, "component_only", True, "inner")
    _expect_bool(root, "l1_binding_not_executed", True, "inner")
    _expect_int(root, "l1_prefix_dispatches", 0, "inner")
    _expect_bool(root, "complete_layer_or_token_performed", False, "inner")
    handoff = _object(root.get("l0_state_handoff"), "inner.l0_state_handoff")
    _expect(handoff.get("schema"), INNER_SCHEMA, "inner.l0_state_handoff.schema")
    _expect(handoff.get("status"), INNER_STATUS, "inner.l0_state_handoff.status")
    _expect(handoff.get("session_id"), SESSION_ID, "inner.l0_state_handoff.session_id")
    _expect_int(handoff, "source_token_id", SOURCE_TOKEN_ID, "inner.l0_state_handoff")
    _expect_bool(handoff, "same_command_graph_retained", True, "inner.l0_state_handoff")
    _expect_bool(handoff, "l1_binding_not_executed", True, "inner.l0_state_handoff")
    _expect_int(handoff, "l1_prefix_dispatches", 0, "inner.l0_state_handoff")

    residual = _object(handoff.get("retained_l0_second_residual"), "inner.retained_l0_second_residual")
    _expect_int(residual, "elements", HIDDEN_ELEMENTS, "inner.retained_l0_second_residual")
    _expect_int(residual, "bytes", HIDDEN_BYTES, "inner.retained_l0_second_residual")
    _expect_bool(residual, "retained_for_future_layer1_encode", True, "inner.retained_l0_second_residual")
    output_hash = _expect_sha256(residual.get("f32le_sha256"), "inner.retained_l0_second_residual.f32le_sha256")
    output_buffer = _expect_sha256(residual.get("device_buffer_id"), "inner.retained_l0_second_residual.device_buffer_id")

    commit = _object(handoff.get("l0_post_state_commit"), "inner.l0_post_state_commit")
    _expect_int(commit, "layer", 0, "inner.l0_post_state_commit")
    _expect_int(commit, "linear_state_slot", L0_SLOT, "inner.l0_post_state_commit")
    _expect_bool(commit, "checkpoint_before_mutation", True, "inner.l0_post_state_commit")
    active_conv = _validate_state(
        commit.get("active_conv"),
        label="inner.l0_post_state_commit.active_conv",
        allocation=f"qwen80/session={SESSION_ID}/arena=active/domain=deltanet_conv_history",
        capacity=L0_CONV_BYTES,
        hash_field="post_state_f32le_sha256",
    )
    active_recurrent = _validate_state(
        commit.get("active_recurrent"),
        label="inner.l0_post_state_commit.active_recurrent",
        allocation=f"qwen80/session={SESSION_ID}/arena=active/domain=deltanet_recurrent",
        capacity=L0_RECURRENT_BYTES,
        hash_field="post_state_f32le_sha256",
    )
    rollback_conv = _validate_state(
        commit.get("rollback_conv"),
        label="inner.l0_post_state_commit.rollback_conv",
        allocation=f"qwen80/session={SESSION_ID}/arena=rollback/domain=deltanet_conv_history",
        capacity=L0_CONV_BYTES,
        hash_field="checkpoint_f32le_sha256",
    )
    rollback_recurrent = _validate_state(
        commit.get("rollback_recurrent"),
        label="inner.l0_post_state_commit.rollback_recurrent",
        allocation=f"qwen80/session={SESSION_ID}/arena=rollback/domain=deltanet_recurrent",
        capacity=L0_RECURRENT_BYTES,
        hash_field="checkpoint_f32le_sha256",
    )

    l1_input = _object(handoff.get("layer1_input_binding"), "inner.layer1_input_binding")
    _expect(l1_input.get("session_id"), SESSION_ID, "inner.layer1_input_binding.session_id")
    _expect_int(l1_input, "layer", L1_LAYER, "inner.layer1_input_binding")
    _expect_int(l1_input, "linear_state_slot", L1_SLOT, "inner.layer1_input_binding")
    _expect_bool(l1_input, "same_command_graph_retained", True, "inner.layer1_input_binding")
    _expect_bool(l1_input, "l1_binding_executed", False, "inner.layer1_input_binding")
    _expect(l1_input.get("input_f32le_sha256"), output_hash, "inner.layer1_input_binding.input_f32le_sha256")
    _expect(l1_input.get("input_device_buffer_id"), output_buffer, "inner.layer1_input_binding.input_device_buffer_id")
    for field, expected_capacity, expected_offset in (
        ("active_conv", L1_CONV_CAPACITY_BYTES, L0_CONV_BYTES),
        ("active_recurrent", L1_RECURRENT_CAPACITY_BYTES, L0_RECURRENT_BYTES),
    ):
        state = _object(l1_input.get(field), f"inner.layer1_input_binding.{field}")
        _expect_int(state, "slot", L1_SLOT, f"inner.layer1_input_binding.{field}")
        _expect_int(state, "capacity_bytes", expected_capacity, f"inner.layer1_input_binding.{field}")
        _expect_int(state, "offset_bytes", expected_offset, f"inner.layer1_input_binding.{field}")
        _expect_sha256(state.get("device_buffer_id"), f"inner.layer1_input_binding.{field}.device_buffer_id")

    claim = _object(handoff.get("claim_boundary"), "inner.l0_state_handoff.claim_boundary")
    for field in (
        "component_only",
        "layer1_not_encoded",
        "retention_binding_is_not_a_layer1_execution_claim",
        "may_not_satisfy_next_layer_execution_dependency",
    ):
        _expect_bool(claim, field, True, "inner.l0_state_handoff.claim_boundary")

    return {
        "captured_session_id": SESSION_ID,
        "source_token_id": SOURCE_TOKEN_ID,
        "retained_l0_second_residual": {
            "elements": HIDDEN_ELEMENTS,
            "bytes": HIDDEN_BYTES,
            "f32le_sha256": output_hash,
            "device_buffer_id": output_buffer,
        },
        "l0_active_and_rollback_state": {
            "active_conv": active_conv,
            "active_recurrent": active_recurrent,
            "rollback_conv": rollback_conv,
            "rollback_recurrent": rollback_recurrent,
        },
        "reserved_l1_slot": {
            "layer": L1_LAYER,
            "linear_state_slot": L1_SLOT,
            "active_conv_offset_bytes": L0_CONV_BYTES,
            "active_conv_capacity_bytes": L1_CONV_CAPACITY_BYTES,
            "active_recurrent_offset_bytes": L0_RECURRENT_BYTES,
            "active_recurrent_capacity_bytes": L1_RECURRENT_CAPACITY_BYTES,
        },
        "l1_binding_not_executed": True,
        "l1_prefix_dispatches": 0,
    }


def _validate_outer(outer: BoundEvidence, inner: BoundEvidence) -> None:
    root = outer.document
    _expect(root.get("lease_id"), LEASE_ID, "outer.lease_id")
    nested = _object(root.get("inner_probe_capture"), "outer.inner_probe_capture")
    _expect_bool(nested, "present", True, "outer.inner_probe_capture")
    _expect_bool(nested, "binding_valid", True, "outer.inner_probe_capture")
    _expect(nested.get("schema"), INNER_SCHEMA, "outer.inner_probe_capture.schema")
    _expect(nested.get("status"), INNER_STATUS, "outer.inner_probe_capture.status")
    _expect_file_reference(nested.get("receipt"), inner, "outer.inner_probe_capture.receipt")
    source_binding = _object(root.get("source_binding"), "outer.source_binding")
    lease_receipt = _object(source_binding.get("lease_receipt"), "outer.source_binding.lease_receipt")
    _expect(lease_receipt.get("seal_sha256"), LEASE_SEAL, "outer.source_binding.lease_receipt.seal_sha256")
    handoff_contract = _object(source_binding.get("handoff_contract"), "outer.source_binding.handoff_contract")
    _expect_int(handoff_contract, "source_token_id", SOURCE_TOKEN_ID, "outer.source_binding.handoff_contract")
    _expect_int(handoff_contract, "prefix_dispatches", L0_PREFIX_DISPATCHES, "outer.source_binding.handoff_contract")
    _expect_int(handoff_contract, "suffix_dispatches", L0_SUFFIX_DISPATCHES, "outer.source_binding.handoff_contract")
    _expect_int(handoff_contract, "total_dispatches", L0_TOTAL_DISPATCHES, "outer.source_binding.handoff_contract")
    _expect_bool(handoff_contract, "same_tcb_fence_required", True, "outer.source_binding.handoff_contract")
    _expect_bool(handoff_contract, "l1_binding_not_executed", True, "outer.source_binding.handoff_contract")
    _expect_int(handoff_contract, "l1_prefix_dispatches", 0, "outer.source_binding.handoff_contract")
    for field in (
        "automatic_retry_disabled",
        "lease_reuse_prohibited_after_terminal",
        "outer_reaped_child",
        "same_capture_dir_never_starts_a_second_child",
        "terminal_receipt_written_last",
    ):
        _expect_bool(_object(root.get("one_shot"), "outer.one_shot"), field, True, "outer.one_shot")
    child_terminal = _object(_object(root.get("child"), "outer.child").get("terminal"), "outer.child.terminal")
    _expect_int(child_terminal, "exit_code", 0, "outer.child.terminal")
    _expect_bool(child_terminal, "reaped", True, "outer.child.terminal")
    _expect_bool(child_terminal, "timed_out", False, "outer.child.terminal")
    boundary = _object(root.get("claim_boundary"), "outer.claim_boundary")
    _expect_bool(boundary, "l1_binding_not_executed", True, "outer.claim_boundary")
    _expect_bool(boundary, "l1_prefix_executed", False, "outer.claim_boundary")
    _expect_bool(boundary, "watcher_or_server_transition_not_authorized", True, "outer.claim_boundary")
    recommendation = _object(root.get("recommended_release_contract"), "outer.recommended_release_contract")
    _expect(recommendation.get("seal_sha256"), RECOMMENDATION_IDENTITY["document_seal_sha256"], "outer.recommended_release_contract.seal_sha256")
    admission_chain = _object(root.get("versioned_current_admission"), "outer.versioned_current_admission")
    _expect_bool(
        admission_chain,
        "terminal_current_pointer_valid",
        True,
        "outer.versioned_current_admission",
    )
    for phase in ("historical_preflight", "terminal"):
        phase_record = _object(admission_chain.get(phase), f"outer.versioned_current_admission.{phase}")
        manifest = _object(
            phase_record.get("immutable_manifest"),
            f"outer.versioned_current_admission.{phase}.immutable_manifest",
        )
        admission = _object(
            phase_record.get("immutable_admission_receipt"),
            f"outer.versioned_current_admission.{phase}.immutable_admission_receipt",
        )
        _expect(
            manifest.get("sha256"),
            MANIFEST_FILE_SHA256,
            f"outer.versioned_current_admission.{phase}.immutable_manifest.sha256",
        )
        _expect(
            manifest.get("seal_sha256"),
            MANIFEST_SEAL_SHA256,
            f"outer.versioned_current_admission.{phase}.immutable_manifest.seal_sha256",
        )
        _expect(
            admission.get("sha256"),
            ADMISSION_FILE_SHA256,
            f"outer.versioned_current_admission.{phase}.immutable_admission_receipt.sha256",
        )
        _expect(
            admission.get("seal_sha256"),
            ADMISSION_SEAL_SHA256,
            f"outer.versioned_current_admission.{phase}.immutable_admission_receipt.seal_sha256",
        )


def _validate_release(release: BoundEvidence, outer: BoundEvidence) -> None:
    root = release.document
    _expect_file_reference(root.get("outer_terminal"), outer, "release.outer_terminal")
    lease = _object(root.get("lease"), "release.lease")
    _expect(lease.get("lease_id"), LEASE_ID, "release.lease.lease_id")
    _expect(lease.get("seal_sha256"), LEASE_SEAL, "release.lease.seal_sha256")
    recommendation = _object(root.get("recommended_release_contract"), "release.recommended_release_contract")
    _expect(recommendation.get("seal_sha256"), RECOMMENDATION_IDENTITY["document_seal_sha256"], "release.recommended_release_contract.seal_sha256")
    coordination = _object(root.get("coordination"), "release.coordination")
    for field in (
        "automatic_retry_prohibited",
        "quiet_qwen80_component_lease_released",
        "watcher_hold_remains_active",
    ):
        _expect_bool(coordination, field, True, "release.coordination")
    _expect_bool(coordination, "watcher_restart_or_transition_authorized", False, "release.coordination")
    boundary = _object(root.get("claim_boundary"), "release.claim_boundary")
    _expect_bool(boundary, "release_is_gpu_coordination_only", True, "release.claim_boundary")
    _expect_bool(
        boundary,
        "does_not_promote_component_to_layer_token_decoder_hcli_tps_tg_or_tournament",
        True,
        "release.claim_boundary",
    )
    outer_time = str(outer.document.get("recorded_at") or "")
    release_time = str(root.get("recorded_at") or "")
    try:
        if datetime.fromisoformat(release_time.replace("Z", "+00:00")) <= datetime.fromisoformat(
            outer_time.replace("Z", "+00:00")
        ):
            raise AssessmentBindingError("actual release must occur after the outer terminal")
    except ValueError as exc:
        raise AssessmentBindingError("outer/release timestamps must be ISO-8601") from exc


def _validate_assessment(
    assessment: BoundEvidence,
    *,
    outer: BoundEvidence,
    inner: BoundEvidence,
    release: BoundEvidence,
    facts: Mapping[str, Any],
) -> None:
    root = assessment.document
    for field in (
        "earned_l0_state_handoff_component",
        "l1_binding_not_executed",
        "l0_handoff_is_evidence_baseline_only",
        "future_l1_requires_fresh_same_runtime_same_tcb_joint_l0_to_l1_capture",
    ):
        _expect_bool(root, field, True, "assessment")
    _expect_bool(root, "cross_process_or_prior_capture_pinned_buffer_reuse_authorized", False, "assessment")
    _expect_int(root, "l1_prefix_dispatches", 0, "assessment")
    _expect_document_identity(root.get("l0_outer_terminal"), outer, "assessment.l0_outer_terminal")
    _expect_document_identity(root.get("l0_inner_receipt"), inner, "assessment.l0_inner_receipt")
    _expect_document_identity(root.get("lease_release_receipt"), release, "assessment.lease_release_receipt")
    _expect_fixed_identity(root.get("handoff_authority"), HANDOFF_AUTHORITY_IDENTITY, "assessment.handoff_authority")
    _expect_fixed_identity(
        root.get("lease_release_recommendation_contract"),
        RECOMMENDATION_IDENTITY,
        "assessment.lease_release_recommendation_contract",
    )
    _expect_fixed_identity(
        root.get("l1_continuation_contract"),
        HISTORICAL_CONTINUATION_IDENTITY,
        "assessment.l1_continuation_contract",
    )
    _expect_bool(root, "l1_continuation_prepared", False, "assessment")
    _expect_bool(root, "l1_continuation_remains_non_executing", True, "assessment")
    validated = _object(root.get("validated_l0_handoff"), "assessment.validated_l0_handoff")
    _expect(validated.get("session_id"), facts["captured_session_id"], "assessment.validated_l0_handoff.session_id")
    _expect_int(validated, "bytes", HIDDEN_BYTES, "assessment.validated_l0_handoff")
    _expect_int(validated, "elements", HIDDEN_ELEMENTS, "assessment.validated_l0_handoff")
    _expect_int(validated, "l1_layer", L1_LAYER, "assessment.validated_l0_handoff")
    _expect_int(validated, "l1_linear_state_slot", L1_SLOT, "assessment.validated_l0_handoff")
    residual = _object(facts.get("retained_l0_second_residual"), "retained facts")
    _expect(
        validated.get("retained_l0_second_residual_f32le_sha256"),
        residual.get("f32le_sha256"),
        "assessment.validated_l0_handoff.retained_l0_second_residual_f32le_sha256",
    )
    _expect(
        validated.get("retained_l0_second_residual_device_buffer_id"),
        residual.get("device_buffer_id"),
        "assessment.validated_l0_handoff.retained_l0_second_residual_device_buffer_id",
    )
    for field in (
        "l1_dispatches_authorized",
        "lease_actions_authorized",
        "metal_or_gpu_actions_authorized",
        "new_model_processes_authorized",
        "server_or_watcher_actions_authorized",
        "tournament_actions_authorized",
        "tps_or_tg_measurements_authorized",
    ):
        _expect_int(_object(root.get("authority_boundary"), "assessment.authority_boundary"), field, 0, "assessment.authority_boundary")


def build_wrapper(*, assessment_path: Path, outer_path: Path, inner_path: Path, release_path: Path) -> dict[str, Any]:
    """Return a sealed, non-executing binding for the one canonical L0 chain."""
    assessment = _read_exact("assessment", assessment_path)
    outer = _read_exact("outer", outer_path)
    inner = _read_exact("inner", inner_path)
    release = _read_exact("release", release_path)
    facts = _validate_inner(inner)
    _validate_outer(outer, inner)
    _validate_release(release, outer)
    _validate_assessment(assessment, outer=outer, inner=inner, release=release, facts=facts)
    return seal(
        {
            "schema": WRAPPER_SCHEMA,
            "status": WRAPPER_STATUS,
            "assessment_result_bound": True,
            "assessment_required_before_joint_child_launch": True,
            "baseline_l0_evidence_is_provenance_only": True,
            "cross_process_pinned_buffer_transfer_allowed": False,
            "joint_l0_reencode_required": True,
            "future_l1_requires_fresh_same_runtime_same_tcb_joint_l0_to_l1_capture": True,
            "joint_child_execution_authorized_by_this_wrapper": False,
            "l1_execution_authorized_by_this_wrapper": False,
            "l0_outer_terminal": _identity(outer),
            "l0_inner_capture": _identity(inner),
            "post_capture_assessment": _identity(assessment),
            "lease_release_receipt": _identity(release),
            "required_assessment": {
                "schema": ASSESSMENT_SCHEMA,
                "earned_status": ASSESSMENT_STATUS,
                "must_be_sealed": True,
                "must_bind_l0_outer_and_inner": True,
                "must_bind_actual_release": True,
                "must_remain_l1_not_executed": True,
                "assessment_document_sha256": assessment.document_sha256,
                "assessment_document_seal_sha256": assessment.seal_sha256,
            },
            "retained_l0_state_handoff": facts,
            "immutable_authority_chain": {
                "handoff_authority": HANDOFF_AUTHORITY_IDENTITY,
                "lease_release_recommendation_contract": RECOMMENDATION_IDENTITY,
                "assessment_historical_continuation_only": {
                    **HISTORICAL_CONTINUATION_IDENTITY,
                    "assessment_did_not_revalidate_or_authorize_a_follow_on_continuation": True,
                },
                "versioned_manifest_and_admission": {
                    "model_key": MODEL_KEY,
                    "manifest_file_sha256": MANIFEST_FILE_SHA256,
                    "manifest_seal_sha256": MANIFEST_SEAL_SHA256,
                    "admission_file_sha256": ADMISSION_FILE_SHA256,
                    "admission_seal_sha256": ADMISSION_SEAL_SHA256,
                },
                "l0_dispatch_plan": {
                    "prefix_dispatches": L0_PREFIX_DISPATCHES,
                    "suffix_dispatches": L0_SUFFIX_DISPATCHES,
                    "total_dispatches": L0_TOTAL_DISPATCHES,
                    "l1_prefix_dispatches_in_historical_capture": 0,
                },
            },
            "future_joint_capture_requirement": {
                "same_session_required": True,
                "same_runtime_required": True,
                "same_tcb_required": True,
                "fresh_l0_reencode_dispatches": L0_TOTAL_DISPATCHES,
                "future_l1_slot1_prefix_dispatches": L1_PREFIX_DISPATCHES,
                "future_joint_total_dispatches": JOINT_TOTAL_DISPATCHES,
                "historical_pinned_buffer_or_state_import_allowed": False,
                "historical_receipts_are_provenance_only": True,
            },
            "authority_boundary": {
                "new_model_processes_authorized": 0,
                "metal_or_gpu_actions_authorized": 0,
                "lease_actions_authorized": 0,
                "server_or_watcher_actions_authorized": 0,
                "l1_dispatches_authorized": 0,
                "tps_or_tg_measurements_authorized": 0,
                "tournament_actions_authorized": 0,
            },
            "claim_boundary": {
                "cpu_only_sealed_binding": True,
                "does_not_open_or_scan_artifacts": True,
                "does_not_construct_metal_or_dispatch": True,
                "does_not_issue_or_consume_a_lease": True,
                "does_not_start_runtime_server_or_watcher": True,
                "does_not_execute_l0_or_l1": True,
                "does_not_measure_tps_or_tg": True,
                "does_not_claim_complete_layer_token_decoder_or_tournament": True,
            },
        }
    )


def write_new(path: Path, document: Mapping[str, Any]) -> Path:
    if not path.is_absolute():
        raise AssessmentBindingError("--out must be an absolute path")
    parent = path.parent
    try:
        metadata = os.lstat(parent)
    except OSError as exc:
        raise AssessmentBindingError(f"cannot stat --out parent: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise AssessmentBindingError("--out parent must be an existing non-symlink directory")
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(dict(document), handle, sort_keys=True, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite {path}") from exc
    return path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--outer-terminal", type=Path, required=True)
    parser.add_argument("--inner-receipt", type=Path, required=True)
    parser.add_argument("--lease-release", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        document = build_wrapper(
            assessment_path=args.assessment,
            outer_path=args.outer_terminal,
            inner_path=args.inner_receipt,
            release_path=args.lease_release,
        )
        output = write_new(args.out, document)
    except (OSError, ValueError, AssessmentBindingError) as exc:
        print(f"ascension_qwen80_l0_state_handoff_assessment_binding_wrapper: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(output), "seal_sha256": document["seal_sha256"], "status": document["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
