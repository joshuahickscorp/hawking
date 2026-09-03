"""One-shot future Qwen80 layer-3 GQA K/V component capture launcher.

This module is intentionally an *outer* CPU-only controller.  It can bind a
future source-hidden / caller-owned active-and-rollback-state authority to the
exact compact layer-3 Q/K/V/O ABI, then reap exactly one already-built child.
It never opens Metal, scans model payloads, registers a shader, issues a
lease, or changes a watcher, server, or runtime registry.

The current repository does not yet contain the required source-hidden and
device-readback authority.  Consequently this launcher is fail-closed today:
it refuses before creating a capture directory or starting a child.  A future
valid terminal receipt remains component-only and cannot establish a complete
layer, token, decoder, HCLI, TPS, TG, or tournament result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "hawking.ascension.qwen80_gqa_kv_component_outer_launcher.v1"
ACTIVE_FILENAME = "active.json"
CHILD_FILENAME = "child.json"
TERMINAL_FILENAME = "outer-terminal-receipt.json"
OUTER_STDOUT = "outer.stdout.log"
OUTER_STDERR = "outer.stderr.log"
INNER_CAPTURE = "inner"

EXPECTED_PROBE_BASENAME = "ascension_qwen80_gqa_kv_cache_device_preflight"
EXPECTED_INNER_SCHEMA = "hawking.ascension.qwen80_gqa_kv_cache_component_device_capture.v1"
EXPECTED_INNER_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_LAYER3_SLOT0_GQA_KV_CACHE_APPEND_CAUSAL_"
    "READ_ROLLBACK_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN"
)

MANIFEST_SCHEMA = "hawking.ascension.qwen80_complete_binary_gravity.v1"
ADMISSION_POINTER_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1"
ADMISSION_POINTER_STATUS = "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED"
ADMISSION_RECEIPT_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1"
ADMISSION_RECEIPT_STATUS = (
    "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
)
SOURCE_HIDDEN_AUTHORITY_SCHEMA = "hawking.ascension.qwen80_source_bound_hidden_input_authority.v1"
SOURCE_HIDDEN_AUTHORITY_STATUS = "EARNED_QWEN80_SOURCE_BOUND_HIDDEN_INPUT_COMPONENT_AUTHORITY"
COMPACT_ABI_SCHEMA = "hawking.ascension.qwen80_gqa_kv_cache_compact_abi_contract.v1"
COMPACT_ABI_STATUS = "PREPARED_QWEN80_GQA_KV_CACHE_LAYER3_SLOT0_COMPACT_ABI_NOT_EXECUTED"
LEASE_SCHEMA = "hawking.ascension.qwen80_gqa_kv_cache_component_quiet_lease.v1"
LEASE_STATUS = (
    "GRANTED_QWEN80_GQA_KV_CACHE_APPEND_CAUSAL_READ_ROLLBACK_COMPONENT_ONLY_"
    "NON_TIMED_LEASE"
)
LEASE_COMPONENT = "qwen80_gqa_kv_cache_append_causal_read_rollback"

MODEL_ID = "Qwen3-Coder-Next-80B"
MODEL_KEY = "qwen80"
SOURCE_REPOSITORY = "Qwen/Qwen3-Coder-Next"
SOURCE_REVISION = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb"
SELECTED_LAYER = 3
SELECTED_SLOT = 0
HIDDEN = 2_048
GQA_KV_HEADS = 2
GQA_HEAD_DIM = 256
KV_ROW_ELEMENTS = GQA_KV_HEADS * GQA_HEAD_DIM
MAX_NATIVE_CONTEXT = 4_096
DIRECT_PACKED_GROUP_SIZE = 128

EXPECTED_PROJECTION_ABI = (
    {
        "role": "q_projection_query_and_gate_rows",
        "tensor_name": "model.layers.3.self_attn.q_proj.weight",
        "shape": [8_192, HIDDEN],
        "group_size": DIRECT_PACKED_GROUP_SIZE,
        "compact_payload_format": "direct_binary_sign_bits_plus_fp16_group_scales",
        "direct_packed_only": True,
        "bf16_shadow_allowed": False,
        "projection_output_elements": 8_192,
    },
    {
        "role": "k_projection",
        "tensor_name": "model.layers.3.self_attn.k_proj.weight",
        "shape": [KV_ROW_ELEMENTS, HIDDEN],
        "group_size": DIRECT_PACKED_GROUP_SIZE,
        "compact_payload_format": "direct_binary_sign_bits_plus_fp16_group_scales",
        "direct_packed_only": True,
        "bf16_shadow_allowed": False,
        "projection_output_elements": KV_ROW_ELEMENTS,
    },
    {
        "role": "v_projection",
        "tensor_name": "model.layers.3.self_attn.v_proj.weight",
        "shape": [KV_ROW_ELEMENTS, HIDDEN],
        "group_size": DIRECT_PACKED_GROUP_SIZE,
        "compact_payload_format": "direct_binary_sign_bits_plus_fp16_group_scales",
        "direct_packed_only": True,
        "bf16_shadow_allowed": False,
        "projection_output_elements": KV_ROW_ELEMENTS,
    },
    {
        "role": "o_projection",
        "tensor_name": "model.layers.3.self_attn.o_proj.weight",
        "shape": [HIDDEN, 4_096],
        "group_size": DIRECT_PACKED_GROUP_SIZE,
        "compact_payload_format": "direct_binary_sign_bits_plus_fp16_group_scales",
        "direct_packed_only": True,
        "bf16_shadow_allowed": False,
        "projection_output_elements": HIDDEN,
    },
)
EXPECTED_COMMAND_ORDER = (
    "direct_packed_q_projection",
    "direct_packed_k_projection",
    "direct_packed_v_projection",
    "snapshot_active_kv_to_rollback",
    "append_key_at_current_position",
    "append_value_at_current_position",
    "causal_mask_including_current_position",
    "causal_read_including_current_position",
    "direct_packed_o_projection",
    "copy_readback_parity_ledger",
    "rollback_active_kv_from_caller_owned_snapshot",
)
EXPECTED_READBACKS = {
    "active_key_slot_row_after_append": KV_ROW_ELEMENTS,
    "active_value_slot_row_after_append": KV_ROW_ELEMENTS,
    "q_projection_rows": 8_192,
    "o_projection_output": HIDDEN,
}


class GqaKvComponentLauncherError(RuntimeError):
    """The future component capture cannot safely start or be promoted."""


@dataclass(frozen=True)
class LaunchConfig:
    probe_bin: Path
    manifest: Path
    admission_current: Path
    source_hidden_authority: Path
    compact_abi_contract: Path
    lease_receipt: Path
    capture_dir: Path
    timeout_seconds: float


@dataclass(frozen=True)
class LaunchContext:
    probe_binary: dict[str, Any]
    manifest: dict[str, Any]
    manifest_seal_sha256: str
    admission_current: dict[str, Any]
    admission_pointer_seal_sha256: str
    admission_receipt_seal_sha256: str
    source_revision: str
    source_hidden_authority: dict[str, Any]
    source_hidden_authority_seal_sha256: str
    source_state: dict[str, Any]
    compact_abi_contract: dict[str, Any]
    lease_receipt: dict[str, Any]
    lease_seal_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise GqaKvComponentLauncherError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GqaKvComponentLauncherError(f"{label} must be an object")
    return dict(value)


def _require_bool(value: Mapping[str, Any], key: str, label: str, expected: bool) -> None:
    if value.get(key) is not expected:
        raise GqaKvComponentLauncherError(f"{label}.{key} must be {expected}")


def _require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise GqaKvComponentLauncherError(f"{label} must be a non-empty string")
    return value


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    if not path.is_absolute():
        raise GqaKvComponentLauncherError(f"{label} must be absolute: {path}")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise GqaKvComponentLauncherError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise GqaKvComponentLauncherError(f"{label} must be a regular non-symlink file")
    if executable and not os.access(path, os.X_OK):
        raise GqaKvComponentLauncherError(f"{label} must be executable")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise GqaKvComponentLauncherError(f"cannot canonicalize {label}: {exc}") from exc


def _file_evidence(path: Path, label: str, *, executable: bool = False) -> dict[str, Any]:
    clean = _canonical_regular(path, label, executable=executable)
    return {
        "path": str(clean),
        "present": True,
        "bytes": clean.stat().st_size,
        "sha256": _file_sha256(clean),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    clean = _canonical_regular(path, label)
    try:
        document = json.loads(clean.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GqaKvComponentLauncherError(f"cannot read {label}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise GqaKvComponentLauncherError(f"{label} must be a JSON object")
    return dict(document)


def _sealed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    document = _read_json(path, label)
    try:
        verify(document, label=str(path))
    except ValueError as exc:
        raise GqaKvComponentLauncherError(f"{label} is not sealed: {exc}") from exc
    return document, _require_sha256(document.get("seal_sha256"), f"{label}.seal_sha256")


def _evidence_matches(evidence: object, expected: Mapping[str, Any], label: str) -> None:
    row = _mapping(evidence, label)
    if row.get("present") is not True:
        raise GqaKvComponentLauncherError(f"{label} does not attest a present file")
    observed = _canonical_regular(Path(str(row.get("path"))), f"{label}.path")
    if observed != Path(str(expected["path"])):
        raise GqaKvComponentLauncherError(f"{label} path drifted")
    if row.get("bytes") != expected["bytes"] or row.get("sha256") != expected["sha256"]:
        raise GqaKvComponentLauncherError(f"{label} byte/digest drifted")


def _atomic_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise GqaKvComponentLauncherError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise GqaKvComponentLauncherError(f"refusing to overwrite {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _bind_manifest(path: Path) -> tuple[dict[str, Any], str]:
    document, document_seal = _sealed_json(path, "--manifest")
    if document.get("schema") != MANIFEST_SCHEMA:
        raise GqaKvComponentLauncherError("--manifest schema drifted")
    return _file_evidence(path, "--manifest"), document_seal


def _bind_admission(
    path: Path, manifest: Mapping[str, Any], manifest_seal: str
) -> tuple[dict[str, Any], str, str, str]:
    pointer, pointer_seal = _sealed_json(path, "--admission-current")
    if (
        pointer.get("schema") != ADMISSION_POINTER_SCHEMA
        or pointer.get("status") != ADMISSION_POINTER_STATUS
    ):
        raise GqaKvComponentLauncherError("admission pointer schema/status drifted")
    selected_manifest = _mapping(pointer.get("complete_manifest"), "admission complete_manifest")
    if _canonical_regular(Path(str(selected_manifest.get("path"))), "admission manifest path") != Path(
        str(manifest["path"])
    ):
        raise GqaKvComponentLauncherError("admission selects another manifest")
    if (
        selected_manifest.get("document_sha256") != manifest["sha256"]
        or selected_manifest.get("seal_sha256") != manifest_seal
    ):
        raise GqaKvComponentLauncherError("admission manifest identity drifted")
    selected_receipt = _mapping(pointer.get("admission_receipt"), "admission receipt selection")
    receipt_path = _canonical_regular(
        Path(str(selected_receipt.get("path"))), "admission receipt path"
    )
    receipt, receipt_seal = _sealed_json(receipt_path, "admission receipt")
    receipt_evidence = _file_evidence(receipt_path, "admission receipt")
    if (
        receipt.get("schema") != ADMISSION_RECEIPT_SCHEMA
        or receipt.get("status") != ADMISSION_RECEIPT_STATUS
        or selected_receipt.get("document_sha256") != receipt_evidence["sha256"]
        or selected_receipt.get("seal_sha256") != receipt_seal
    ):
        raise GqaKvComponentLauncherError("admission receipt schema/status/identity drifted")
    receipt_manifest = _mapping(receipt.get("complete_manifest"), "admission receipt manifest")
    if (
        _canonical_regular(Path(str(receipt_manifest.get("path"))), "admission receipt manifest path")
        != Path(str(manifest["path"]))
        or receipt_manifest.get("document_sha256") != manifest["sha256"]
        or receipt_manifest.get("seal_sha256") != manifest_seal
    ):
        raise GqaKvComponentLauncherError("admission receipt manifest authority drifted")
    revalidation = _mapping(receipt.get("current_source_revalidation"), "admission source revalidation")
    revision = _require_nonempty_string(revalidation.get("revision"), "admission source revision")
    if revision != SOURCE_REVISION:
        raise GqaKvComponentLauncherError("admission source revision drifted")
    _require_sha256(revalidation.get("source_audit_seal_sha256"), "admission source audit seal")
    return _file_evidence(path, "--admission-current"), pointer_seal, receipt_seal, revision


def _state_buffer(
    value: object, *, label: str, max_seq_len: int
) -> dict[str, Any]:
    row = _mapping(value, label)
    _require_nonempty_string(row.get("handle"), f"{label}.handle")
    if row.get("shape") != [max_seq_len, GQA_KV_HEADS, GQA_HEAD_DIM]:
        raise GqaKvComponentLauncherError(f"{label}.shape drifted")
    expected_elements = max_seq_len * KV_ROW_ELEMENTS
    if row.get("elements") != expected_elements or row.get("bytes") != expected_elements * 4:
        raise GqaKvComponentLauncherError(f"{label} geometry drifted")
    _require_sha256(row.get("f32le_sha256"), f"{label}.f32le_sha256")
    return row


def _bind_source_hidden_authority(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission_pointer_seal: str,
    admission_receipt_seal: str,
    source_revision: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    authority, authority_seal = _sealed_json(path, "--source-hidden-authority")
    if (
        authority.get("schema") != SOURCE_HIDDEN_AUTHORITY_SCHEMA
        or authority.get("status") != SOURCE_HIDDEN_AUTHORITY_STATUS
    ):
        raise GqaKvComponentLauncherError("source-hidden authority schema/status drifted")
    source = _mapping(authority.get("source_binding"), "source-hidden authority source_binding")
    if (
        source.get("model_id") != MODEL_ID
        or source.get("model_key") != MODEL_KEY
        or source.get("source_repository") != SOURCE_REPOSITORY
        or source.get("source_revision") != source_revision
        or source.get("manifest_document_sha256") != manifest["sha256"]
        or source.get("manifest_seal_sha256") != manifest_seal
        or source.get("admission_pointer_seal_sha256") != admission_pointer_seal
        or source.get("admission_receipt_seal_sha256") != admission_receipt_seal
    ):
        raise GqaKvComponentLauncherError("source-hidden authority artifact identity drifted")
    hidden = _mapping(authority.get("source_hidden"), "source-hidden authority source_hidden")
    if (
        hidden.get("layer") != SELECTED_LAYER
        or hidden.get("gqa_slot") != SELECTED_SLOT
        or hidden.get("elements") != HIDDEN
        or hidden.get("source_bound") is not True
        or hidden.get("synthetic_or_fixture") is not False
    ):
        raise GqaKvComponentLauncherError("source-hidden authority layer3/slot0 hidden binding drifted")
    _require_sha256(hidden.get("f32le_sha256"), "source-hidden authority hidden hash")
    state = _mapping(authority.get("caller_owned_active_and_rollback_state"), "source-hidden authority state")
    session_id = _require_nonempty_string(state.get("session_id"), "source-hidden authority session_id")
    token_position = state.get("token_position")
    max_seq_len = state.get("max_seq_len")
    if (
        isinstance(token_position, bool)
        or not isinstance(token_position, int)
        or isinstance(max_seq_len, bool)
        or not isinstance(max_seq_len, int)
        or not 2 <= max_seq_len <= MAX_NATIVE_CONTEXT
        or not 0 <= token_position < max_seq_len - 1
    ):
        raise GqaKvComponentLauncherError("source-hidden authority token position/context drifted")
    if (
        state.get("selected_layer") != SELECTED_LAYER
        or state.get("selected_slot") != SELECTED_SLOT
        or state.get("caller_owned_by_upstream") is not True
        or state.get("active_and_rollback_disjoint") is not True
    ):
        raise GqaKvComponentLauncherError("source-hidden authority active/rollback state ownership drifted")
    state_buffers = {
        name: _state_buffer(state.get(name), label=f"source-hidden authority {name}", max_seq_len=max_seq_len)
        for name in ("active_key", "active_value", "rollback_key", "rollback_value")
    }
    handles = [row["handle"] for row in state_buffers.values()]
    if len(set(handles)) != len(handles):
        raise GqaKvComponentLauncherError("source-hidden authority state handles alias")
    future_evidence = _mapping(authority.get("upstream_evidence"), "source-hidden authority upstream_evidence")
    if (
        future_evidence.get("source_hidden_parity_evidence_earned") is not True
        or future_evidence.get("state_readback_authority_earned") is not True
        or future_evidence.get("receipt_written_last_is_completion_marker") is not True
        or future_evidence.get("complete_layer_or_token_performed") is not False
        or future_evidence.get("decoder_or_generation_performed") is not False
    ):
        raise GqaKvComponentLauncherError(
            "source-hidden authority lacks required future upstream parity/state evidence"
        )
    return _file_evidence(path, "--source-hidden-authority"), authority_seal, {
        "session_id": session_id,
        "token_position": token_position,
        "max_seq_len": max_seq_len,
        "source_hidden_f32le_sha256": hidden["f32le_sha256"],
        "active_key_f32le_sha256": state_buffers["active_key"]["f32le_sha256"],
        "active_value_f32le_sha256": state_buffers["active_value"]["f32le_sha256"],
        "rollback_key_f32le_sha256": state_buffers["rollback_key"]["f32le_sha256"],
        "rollback_value_f32le_sha256": state_buffers["rollback_value"]["f32le_sha256"],
    }


def _bind_compact_abi(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission_receipt_seal: str,
    source_revision: str,
) -> dict[str, Any]:
    document = _read_json(path, "--compact-abi-contract")
    if document.get("schema") != COMPACT_ABI_SCHEMA or document.get("status") != COMPACT_ABI_STATUS:
        raise GqaKvComponentLauncherError("compact ABI schema/status drifted")
    source = _mapping(document.get("source_binding"), "compact ABI source_binding")
    if (
        source.get("model_id") != MODEL_ID
        or source.get("model_key") != MODEL_KEY
        or source.get("source_repository") != SOURCE_REPOSITORY
        or source.get("source_revision") != source_revision
        or source.get("manifest_document_sha256") != manifest["sha256"]
        or source.get("manifest_seal_sha256") != manifest_seal
        or source.get("admission_receipt_seal_sha256") != admission_receipt_seal
    ):
        raise GqaKvComponentLauncherError("compact ABI artifact identity drifted")
    geometry = _mapping(document.get("geometry"), "compact ABI geometry")
    if geometry != {
        "layer": SELECTED_LAYER,
        "slot": SELECTED_SLOT,
        "hidden": HIDDEN,
        "kv_heads": GQA_KV_HEADS,
        "head_dim": GQA_HEAD_DIM,
        "kv_row_elements": KV_ROW_ELEMENTS,
        "minimum_context": 2,
        "maximum_context": MAX_NATIVE_CONTEXT,
    }:
        raise GqaKvComponentLauncherError("compact ABI layer3/slot0 cache geometry drifted")
    if document.get("direct_packed_projection_abi") != list(EXPECTED_PROJECTION_ABI):
        raise GqaKvComponentLauncherError("compact ABI Q/K/V/O direct-packed contract drifted")
    if document.get("component_command_order") != list(EXPECTED_COMMAND_ORDER):
        raise GqaKvComponentLauncherError("compact ABI command order drifted")
    boundary = _mapping(document.get("claim_boundary"), "compact ABI claim_boundary")
    for key in (
        "artifact_scan_or_payload_open_performed",
        "metal_context_or_dispatch_performed",
        "runtime_watcher_server_registry_or_hcli_changed",
        "complete_layer_or_token_performed",
        "decoder_or_generation_performed",
        "tps_or_tg_claim",
    ):
        _require_bool(boundary, key, "compact ABI claim_boundary", False)
    return _file_evidence(path, "--compact-abi-contract")


def _bind_lease(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission_receipt_seal: str,
    authority: Mapping[str, Any],
    authority_seal: str,
    compact_abi: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    lease, lease_seal = _sealed_json(path, "--lease-receipt")
    if lease.get("schema") != LEASE_SCHEMA or lease.get("status") != LEASE_STATUS:
        raise GqaKvComponentLauncherError("quiet lease schema/status drifted")
    _require_nonempty_string(lease.get("lease_id"), "quiet lease ID")
    lifecycle = _mapping(lease.get("lifecycle"), "quiet lease lifecycle")
    if (
        lifecycle.get("fresh_for_this_exact_launch") is not True
        or lifecycle.get("automatic_retry_prohibited") is not True
        or lifecycle.get("outer_reaped_capture_required") is not True
        or lifecycle.get("prior_terminal_receipt") is not None
    ):
        raise GqaKvComponentLauncherError("quiet lease is not fresh one-shot authority")
    policy = _mapping(lease.get("execution_policy"), "quiet lease execution_policy")
    if (
        policy.get("component") != LEASE_COMPONENT
        or policy.get("quiet_qwen80_device_lease") is not True
        or policy.get("strict_math") is not True
        or policy.get("timing_or_benchmarking_allowed") is not False
        or policy.get("complete_layer_or_token_allowed") is not False
        or policy.get("tps_or_tg_claim_allowed") is not False
    ):
        raise GqaKvComponentLauncherError("quiet lease policy is not component-only non-timed")
    artifact = _mapping(lease.get("artifact_binding"), "quiet lease artifact_binding")
    if (
        artifact.get("manifest_document_sha256") != manifest["sha256"]
        or artifact.get("manifest_seal_sha256") != manifest_seal
        or artifact.get("admission_receipt_seal_sha256") != admission_receipt_seal
        or artifact.get("selected_layer") != SELECTED_LAYER
        or artifact.get("selected_slot") != SELECTED_SLOT
    ):
        raise GqaKvComponentLauncherError("quiet lease artifact/layer/slot binding drifted")
    authority_binding = _mapping(
        lease.get("source_hidden_authority_binding"), "quiet lease source-hidden authority"
    )
    _evidence_matches(authority_binding, authority, "quiet lease source-hidden authority")
    if (
        authority_binding.get("schema") != SOURCE_HIDDEN_AUTHORITY_SCHEMA
        or authority_binding.get("status") != SOURCE_HIDDEN_AUTHORITY_STATUS
        or authority_binding.get("seal_sha256") != authority_seal
    ):
        raise GqaKvComponentLauncherError("quiet lease source-hidden authority seal/schema drifted")
    abi_binding = _mapping(lease.get("compact_abi_contract_binding"), "quiet lease compact ABI")
    _evidence_matches(abi_binding, compact_abi, "quiet lease compact ABI")
    if (
        abi_binding.get("schema") != COMPACT_ABI_SCHEMA
        or abi_binding.get("status") != COMPACT_ABI_STATUS
    ):
        raise GqaKvComponentLauncherError("quiet lease compact ABI schema/status drifted")
    return _file_evidence(path, "--lease-receipt"), lease_seal


def _validate_config(config: LaunchConfig) -> LaunchContext:
    if not config.capture_dir.is_absolute():
        raise GqaKvComponentLauncherError("--capture-dir must be absolute")
    if not config.timeout_seconds > 0:
        raise GqaKvComponentLauncherError("--timeout-seconds must be positive")
    probe = _file_evidence(config.probe_bin, "--probe-bin", executable=True)
    if Path(str(probe["path"])).name != EXPECTED_PROBE_BASENAME:
        raise GqaKvComponentLauncherError(
            f"--probe-bin must name {EXPECTED_PROBE_BASENAME}, got {Path(str(probe['path'])).name!r}"
        )
    manifest, manifest_seal = _bind_manifest(config.manifest)
    admission, admission_pointer_seal, admission_receipt_seal, source_revision = _bind_admission(
        config.admission_current, manifest, manifest_seal
    )
    authority, authority_seal, source_state = _bind_source_hidden_authority(
        config.source_hidden_authority,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission_pointer_seal=admission_pointer_seal,
        admission_receipt_seal=admission_receipt_seal,
        source_revision=source_revision,
    )
    compact_abi = _bind_compact_abi(
        config.compact_abi_contract,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission_receipt_seal=admission_receipt_seal,
        source_revision=source_revision,
    )
    lease, lease_seal = _bind_lease(
        config.lease_receipt,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission_receipt_seal=admission_receipt_seal,
        authority=authority,
        authority_seal=authority_seal,
        compact_abi=compact_abi,
    )
    return LaunchContext(
        probe_binary=probe,
        manifest=manifest,
        manifest_seal_sha256=manifest_seal,
        admission_current=admission,
        admission_pointer_seal_sha256=admission_pointer_seal,
        admission_receipt_seal_sha256=admission_receipt_seal,
        source_revision=source_revision,
        source_hidden_authority=authority,
        source_hidden_authority_seal_sha256=authority_seal,
        source_state=source_state,
        compact_abi_contract=compact_abi,
        lease_receipt=lease,
        lease_seal_sha256=lease_seal,
    )


def _launch_identity(config: LaunchConfig, context: LaunchContext) -> str:
    payload = {
        "probe_binary_sha256": context.probe_binary["sha256"],
        "manifest_document_sha256": context.manifest["sha256"],
        "admission_pointer_seal_sha256": context.admission_pointer_seal_sha256,
        "admission_receipt_seal_sha256": context.admission_receipt_seal_sha256,
        "source_hidden_authority_sha256": context.source_hidden_authority["sha256"],
        "source_hidden_authority_seal_sha256": context.source_hidden_authority_seal_sha256,
        "compact_abi_contract_sha256": context.compact_abi_contract["sha256"],
        "lease_receipt_sha256": context.lease_receipt["sha256"],
        "lease_seal_sha256": context.lease_seal_sha256,
        "timeout_seconds": config.timeout_seconds,
    }
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def _child_command(config: LaunchConfig, inner_capture_dir: Path) -> list[str]:
    return [
        str(config.probe_bin),
        "--mode",
        "component-child",
        "--manifest",
        str(config.manifest),
        "--admission-current",
        str(config.admission_current),
        "--source-hidden-authority",
        str(config.source_hidden_authority),
        "--compact-abi-contract",
        str(config.compact_abi_contract),
        "--lease-receipt",
        str(config.lease_receipt),
        "--capture-dir",
        str(inner_capture_dir),
    ]


def _readback_vector(value: object, *, label: str, expected_elements: int) -> None:
    row = _mapping(value, label)
    if row.get("elements") != expected_elements or row.get("exact") is not True:
        raise GqaKvComponentLauncherError(f"{label} is not exact source-shaped parity")
    max_abs_error = row.get("max_abs_error")
    if not isinstance(max_abs_error, (int, float)) or isinstance(max_abs_error, bool) or float(max_abs_error) != 0.0:
        raise GqaKvComponentLauncherError(f"{label}.max_abs_error must be exactly zero")
    _require_sha256(row.get("f32le_sha256"), f"{label}.f32le_sha256")


def _validate_inner(document: Mapping[str, Any], context: LaunchContext) -> None:
    if document.get("schema") != EXPECTED_INNER_SCHEMA or document.get("status") != EXPECTED_INNER_STATUS:
        raise GqaKvComponentLauncherError("inner child schema/status drifted")
    if (
        document.get("component_only") is not True
        or document.get("complete_layer_or_token_performed") is not False
        or document.get("decoder_or_generation_performed") is not False
        or document.get("metal_device_or_dispatch_performed") is not True
    ):
        raise GqaKvComponentLauncherError("inner child scope/device boundary drifted")
    execution = _mapping(document.get("metal_execution_policy"), "inner metal_execution_policy")
    if (
        execution.get("strict_math") is not True
        or execution.get("timing_or_benchmarking_allowed") is not False
        or execution.get("complete_layer_or_token_allowed") is not False
        or execution.get("tps_or_tg_claim_allowed") is not False
    ):
        raise GqaKvComponentLauncherError("inner child strict component policy drifted")
    durable = _mapping(document.get("durable_capture"), "inner durable_capture")
    if (
        durable.get("receipt_written_last_is_completion_marker") is not True
        or durable.get("source_hidden_and_state_readbacks_written_before_receipt") is not True
        or durable.get("outer_reaped_capture_required") is not True
    ):
        raise GqaKvComponentLauncherError("inner child lacks receipt-last durability")
    artifact = _mapping(document.get("artifact_binding"), "inner artifact_binding")
    if (
        artifact.get("manifest_document_sha256") != context.manifest["sha256"]
        or artifact.get("manifest_seal_sha256") != context.manifest_seal_sha256
        or artifact.get("admission_pointer_seal_sha256") != context.admission_pointer_seal_sha256
        or artifact.get("admission_receipt_seal_sha256") != context.admission_receipt_seal_sha256
    ):
        raise GqaKvComponentLauncherError("inner child artifact binding drifted")
    authority = _mapping(document.get("source_hidden_authority_binding"), "inner source-hidden authority")
    _evidence_matches(authority, context.source_hidden_authority, "inner source-hidden authority")
    if (
        authority.get("schema") != SOURCE_HIDDEN_AUTHORITY_SCHEMA
        or authority.get("status") != SOURCE_HIDDEN_AUTHORITY_STATUS
        or authority.get("seal_sha256") != context.source_hidden_authority_seal_sha256
    ):
        raise GqaKvComponentLauncherError("inner source-hidden authority binding drifted")
    abi = _mapping(document.get("compact_abi_contract_binding"), "inner compact ABI")
    _evidence_matches(abi, context.compact_abi_contract, "inner compact ABI")
    if abi.get("schema") != COMPACT_ABI_SCHEMA or abi.get("status") != COMPACT_ABI_STATUS:
        raise GqaKvComponentLauncherError("inner compact ABI schema/status drifted")
    lease = _mapping(document.get("lease_binding"), "inner quiet lease")
    _evidence_matches(lease, context.lease_receipt, "inner quiet lease")
    if (
        lease.get("schema") != LEASE_SCHEMA
        or lease.get("status") != LEASE_STATUS
        or lease.get("seal_sha256") != context.lease_seal_sha256
    ):
        raise GqaKvComponentLauncherError("inner quiet lease binding drifted")
    state = _mapping(document.get("caller_owned_state_binding"), "inner caller-owned state")
    for key in ("session_id", "token_position", "max_seq_len"):
        if state.get(key) != context.source_state[key]:
            raise GqaKvComponentLauncherError(f"inner caller-owned state {key} drifted")
    for key in (
        "source_hidden_f32le_sha256",
        "active_key_f32le_sha256",
        "active_value_f32le_sha256",
        "rollback_key_f32le_sha256",
        "rollback_value_f32le_sha256",
    ):
        if state.get(key) != context.source_state[key]:
            raise GqaKvComponentLauncherError(f"inner caller-owned state {key} identity drifted")
    if state.get("selected_layer") != SELECTED_LAYER or state.get("selected_slot") != SELECTED_SLOT:
        raise GqaKvComponentLauncherError("inner caller-owned state layer/slot drifted")
    readbacks = _mapping(document.get("readback_parity"), "inner readback_parity")
    if set(readbacks) != set(EXPECTED_READBACKS):
        raise GqaKvComponentLauncherError("inner readback parity vector set drifted")
    for name, elements in EXPECTED_READBACKS.items():
        _readback_vector(readbacks.get(name), label=f"inner readback {name}", expected_elements=elements)
    rollback = _mapping(document.get("rollback_readback"), "inner rollback_readback")
    if (
        rollback.get("restored_active_key_exact") is not True
        or rollback.get("restored_active_value_exact") is not True
        or rollback.get("active_key_after_rollback_f32le_sha256")
        != context.source_state["active_key_f32le_sha256"]
        or rollback.get("active_value_after_rollback_f32le_sha256")
        != context.source_state["active_value_f32le_sha256"]
    ):
        raise GqaKvComponentLauncherError("inner rollback readback parity failed")


def _inner_evidence(config: LaunchConfig, context: LaunchContext) -> dict[str, Any]:
    capture_dir = config.capture_dir / INNER_CAPTURE
    receipt_path = capture_dir / "receipt.json"
    result: dict[str, Any] = {"capture_dir": str(capture_dir), "receipt": _missing_evidence(receipt_path)}
    if not receipt_path.is_file():
        result["present"] = False
        result["invocation"] = _missing_evidence(capture_dir / "invocation.json")
        return result
    try:
        document, inner_seal = _sealed_json(receipt_path, "inner receipt")
        evidence = _file_evidence(receipt_path, "inner receipt")
        _validate_inner(document, context)
    except GqaKvComponentLauncherError as exc:
        result["present"] = True
        result["binding_valid"] = False
        result["binding_error"] = str(exc)
        result["receipt"] = _missing_evidence(receipt_path)
        return result
    result.update(
        {
            "present": True,
            "binding_valid": True,
            "schema": document.get("schema"),
            "status": document.get("status"),
            "seal_sha256": inner_seal,
            "receipt": evidence,
        }
    )
    return result


def _missing_evidence(path: Path) -> dict[str, Any]:
    if path.is_file() and not path.is_symlink():
        return {
            "path": str(path.resolve()),
            "present": True,
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    return {"path": str(path), "present": False}


def _sync_evidence(path: Path) -> dict[str, Any]:
    with path.open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    return _file_evidence(path, f"outer stream {path.name}")


def _terminal(returncode: int | None, *, timed_out: bool, spawn_error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reaped": returncode is not None,
        "timed_out": timed_out,
        "returncode": returncode,
        "exit_code": returncode if isinstance(returncode, int) and returncode >= 0 else None,
        "signal": -returncode if isinstance(returncode, int) and returncode < 0 else None,
    }
    if spawn_error is not None:
        result["spawn_error"] = spawn_error
        result["reaped"] = False
    return result


def _terminate_group(child: subprocess.Popen[bytes]) -> int | None:
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return child.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return child.wait(timeout=10.0)


def _terminal_status(terminal: Mapping[str, Any], inner: Mapping[str, Any]) -> str:
    if terminal.get("spawn_error"):
        return "REFUSED_QWEN80_GQA_KV_COMPONENT_OUTER_CHILD_SPAWN_ERROR"
    if terminal.get("timed_out"):
        return "REFUSED_QWEN80_GQA_KV_COMPONENT_OUTER_CHILD_TIMEOUT"
    if terminal.get("signal") is not None:
        return "REFUSED_QWEN80_GQA_KV_COMPONENT_OUTER_CHILD_SIGNAL"
    if terminal.get("exit_code") != 0:
        return "REFUSED_QWEN80_GQA_KV_COMPONENT_OUTER_CHILD_NONZERO"
    if inner.get("binding_valid") is not True:
        return "REFUSED_QWEN80_GQA_KV_COMPONENT_OUTER_ZERO_EXIT_WITHOUT_STRICT_INNER_RECEIPT"
    return "CAPTURED_QWEN80_GQA_KV_COMPONENT_OUTER_TERMINAL_COMPONENT_ONLY"


def _terminal_receipt(
    config: LaunchConfig,
    context: LaunchContext,
    *,
    identity: str,
    command: Sequence[str],
    child_pid: int | None,
    started_at: str,
    terminal: Mapping[str, Any],
    capture_error: str | None,
) -> dict[str, Any]:
    inner = _inner_evidence(config, context)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": _terminal_status(terminal, inner),
        "recorded_at": _utc_now(),
        "one_shot": {
            "exactly_one_child_per_capture_directory": True,
            "automatic_retry_disabled": True,
            "terminal_receipt_written_last": True,
        },
        "launch_identity_sha256": identity,
        "source_binding": {
            "probe_binary": context.probe_binary,
            "manifest": context.manifest,
            "manifest_seal_sha256": context.manifest_seal_sha256,
            "admission_current": context.admission_current,
            "admission_pointer_seal_sha256": context.admission_pointer_seal_sha256,
            "admission_receipt_seal_sha256": context.admission_receipt_seal_sha256,
            "source_hidden_authority": context.source_hidden_authority,
            "source_hidden_authority_seal_sha256": context.source_hidden_authority_seal_sha256,
            "compact_abi_contract": context.compact_abi_contract,
            "lease_receipt": context.lease_receipt,
            "lease_receipt_seal_sha256": context.lease_seal_sha256,
        },
        "child": {
            "pid": child_pid,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "command": list(command),
            "terminal": dict(terminal),
        },
        "outer_capture": {
            "directory": str(config.capture_dir),
            "stdout": _sync_evidence(config.capture_dir / OUTER_STDOUT),
            "stderr": _sync_evidence(config.capture_dir / OUTER_STDERR),
        },
        "inner_probe_capture": inner,
        "claim_boundary": {
            "outer_controller_is_cpu_only": True,
            "outer_did_not_open_metal_or_dispatch": True,
            "outer_did_not_scan_model_artifacts": True,
            "outer_did_not_issue_or_mutate_a_lease": True,
            "outer_did_not_change_registry_watcher_or_server": True,
            "component_only_not_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament": True,
        },
    }
    if capture_error is not None:
        receipt["capture_error"] = capture_error
    return seal(receipt)


def _replay(config: LaunchConfig, identity: str) -> dict[str, Any]:
    terminal_path = config.capture_dir / TERMINAL_FILENAME
    if not terminal_path.is_file():
        raise GqaKvComponentLauncherError("capture directory exists without terminal receipt")
    receipt, _ = _sealed_json(terminal_path, "outer terminal receipt")
    if receipt.get("schema") != SCHEMA or receipt.get("launch_identity_sha256") != identity:
        raise GqaKvComponentLauncherError("capture directory belongs to another launch identity")
    return receipt


def run_attempt(config: LaunchConfig) -> dict[str, Any]:
    """Run exactly one future child, or replay its sealed terminal record."""

    context = _validate_config(config)
    identity = _launch_identity(config, context)
    if config.capture_dir.exists():
        return _replay(config, identity)
    try:
        config.capture_dir.mkdir(mode=0o750)
    except FileExistsError:
        return _replay(config, identity)
    command = _child_command(config, config.capture_dir / INNER_CAPTURE)
    started_at = _utc_now()
    _atomic_json_new(
        config.capture_dir / ACTIVE_FILENAME,
        seal(
            {
                "schema": SCHEMA,
                "status": "STARTED_QWEN80_GQA_KV_COMPONENT_OUTER_ONE_SHOT",
                "recorded_at": started_at,
                "launch_identity_sha256": identity,
                "command": command,
                "claim_boundary": {
                    "component_only": True,
                    "outer_cpu_only": True,
                    "automatic_retry_disabled": True,
                },
            }
        ),
    )
    child_pid: int | None = None
    capture_error: str | None = None
    with (config.capture_dir / OUTER_STDOUT).open("xb") as stdout, (
        config.capture_dir / OUTER_STDERR
    ).open("xb") as stderr:
        try:
            child = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            terminal = _terminal(None, timed_out=False, spawn_error=f"{type(exc).__name__}: {exc}")
        else:
            child_pid = child.pid
            try:
                _atomic_json_new(
                    config.capture_dir / CHILD_FILENAME,
                    seal(
                        {
                            "schema": SCHEMA,
                            "status": "RUNNING_QWEN80_GQA_KV_COMPONENT_OUTER_ONE_SHOT",
                            "recorded_at": _utc_now(),
                            "launch_identity_sha256": identity,
                            "pid": child_pid,
                            "parent_pid": os.getpid(),
                            "command": command,
                            "component_only": True,
                            "strict_quiet_lease_required": True,
                        }
                    ),
                )
            except GqaKvComponentLauncherError as exc:
                capture_error = str(exc)
                terminal = _terminal(_terminate_group(child), timed_out=False)
            else:
                try:
                    terminal = _terminal(child.wait(timeout=config.timeout_seconds), timed_out=False)
                except subprocess.TimeoutExpired:
                    terminal = _terminal(_terminate_group(child), timed_out=True)
    receipt = _terminal_receipt(
        config,
        context,
        identity=identity,
        command=command,
        child_pid=child_pid,
        started_at=started_at,
        terminal=terminal,
        capture_error=capture_error,
    )
    _atomic_json_new(config.capture_dir / TERMINAL_FILENAME, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-bin", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--admission-current", type=Path, required=True)
    parser.add_argument("--source-hidden-authority", type=Path, required=True)
    parser.add_argument("--compact-abi-contract", type=Path, required=True)
    parser.add_argument("--lease-receipt", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7_200.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(argv)
    config = LaunchConfig(
        probe_bin=parsed.probe_bin,
        manifest=parsed.manifest,
        admission_current=parsed.admission_current,
        source_hidden_authority=parsed.source_hidden_authority,
        compact_abi_contract=parsed.compact_abi_contract,
        lease_receipt=parsed.lease_receipt,
        capture_dir=parsed.capture_dir,
        timeout_seconds=parsed.timeout_seconds,
    )
    try:
        receipt = run_attempt(config)
    except GqaKvComponentLauncherError as exc:
        print(json.dumps({"status": "REFUSED_QWEN80_GQA_KV_COMPONENT_OUTER_LAUNCHER_ERROR", "error": str(exc)}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"].startswith("CAPTURED_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
