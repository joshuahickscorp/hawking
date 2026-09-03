"""One-shot outer launcher for a future Qwen80 terminal-head component capture.

This module is deliberately a CPU-only controller.  It validates future,
sealed authority for one source-bound post-48 hidden vector, the sealed CPU
baseline that anchors the terminal component, the exact direct-packed terminal
ABI, and one fresh component-only quiet lease.  Only then can it outer-reap
one already-built child and write a receipt-last terminal envelope.

It does *not* create a Metal context, register a shader, issue or mutate a
lease, open a model payload, scan an artifact, start a model/server, contact
HCLI, generate a token, or measure TPS/TG.  The current Q80 terminal evidence
is intentionally partial and is hard-refused by this boundary.  A successful
future receipt is still only a terminal-head component receipt; it is not a
decoder, generation, HCLI, TPS, TG, tournament, or token claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "hawking.ascension.qwen80_terminal_head_component_outer_launcher.v1"
CAPTURED_STATUS = "CAPTURED_QWEN80_TERMINAL_HEAD_COMPONENT_OUTER_TERMINAL_NO_TOKEN"
REFUSED_CURRENT_PARTIAL_STATUS = "REFUSED_QWEN80_TERMINAL_HEAD_CURRENT_PARTIAL_EVIDENCE"
ACTIVE_FILENAME = "active.json"
CHILD_FILENAME = "child.json"
TERMINAL_FILENAME = "outer-terminal-receipt.json"
OUTER_STDOUT = "outer.stdout.log"
OUTER_STDERR = "outer.stderr.log"
INNER_CAPTURE = "inner"

EXPECTED_PROBE_BASENAME = "ascension_qwen80_terminal_head_component_device"
EXPECTED_INNER_SCHEMA = "hawking.ascension.qwen80_terminal_head_component_device_capture.v1"
EXPECTED_INNER_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_POST48_TERMINAL_HEAD_FULL_ROW_STRICT_MATH_"
    "METAL_COMPONENT_NOT_TOKEN_OR_DECODER"
)

MANIFEST_SCHEMA = "hawking.ascension.qwen80_complete_binary_gravity.v1"
ADMISSION_POINTER_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1"
ADMISSION_POINTER_STATUS = "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED"
ADMISSION_RECEIPT_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1"
ADMISSION_RECEIPT_STATUS = (
    "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
)
POST48_HIDDEN_SCHEMA = "hawking.ascension.qwen80_post48_layer_hidden_device_buffer.v1"
POST48_HIDDEN_STATUS = "EARNED_QWEN80_SOURCE_BOUND_POST48_HIDDEN_DEVICE_PARITY_AUTHORITY"
BASELINE_SCHEMA = "hawking.ascension.qwen80_terminal_head_cpu_baseline_wrapper.v1"
BASELINE_STATUS = "SEALED_CURRENT_ADMITTED_QWEN80_TERMINAL_HEAD_AND_SAMPLER_CPU_BASELINE"
TERMINAL_CONTRACT_SCHEMA = "hawking.ascension.qwen80_terminal_head_component_contract.v1"
TERMINAL_CONTRACT_STATUS = "SEALED_QWEN80_TERMINAL_HEAD_FULL_ROW_COMPONENT_ABI_AUTHORITY"
LEASE_SCHEMA = "hawking.ascension.qwen80_terminal_head_component_quiet_lease.v1"
LEASE_STATUS = "GRANTED_QWEN80_TERMINAL_HEAD_STRICT_MATH_COMPONENT_ONLY_NON_TIMED_LEASE"
LEASE_COMPONENT = "qwen80_terminal_head_full_row_mask_sample_feedback"

TERMINAL_CPU_RECEIPT_SCHEMA = "hawking.ascension.qwen80_direct_packed_terminal_head_cpu.v1"
TERMINAL_CPU_RECEIPT_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_TERMINAL_COMPONENT_CPU_ONLY_NOT_RUNTIME_OR_TOKEN"
)
TOKENIZER_RECEIPT_SCHEMA = "hawking.ascension.qwen80_tokenizer_sampler_handoff_contract.v1"
TOKENIZER_RECEIPT_STATUS = (
    "EARNED_SOURCE_BOUND_TOKENIZER_TEMPLATE_SAMPLER_HANDOFF_COMPONENT_NOT_RUNTIME_OR_TOKEN"
)

MODEL_ID = "Qwen3-Coder-Next-80B"
MODEL_KEY = "qwen80"
SOURCE_REPOSITORY = "Qwen/Qwen3-Coder-Next"
SOURCE_REVISION = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb"
HIDDEN = 2_048
POST48_HIDDEN_BYTES = HIDDEN * 4
LM_HEAD_ROWS = 151_936
TOKENIZER_VOCAB = 151_669
FIRST_RESERVED_ID = TOKENIZER_VOCAB
LAST_RESERVED_ID = LM_HEAD_ROWS - 1
RESERVED_TAIL_ROWS = LM_HEAD_ROWS - TOKENIZER_VOCAB
GROUP_SIZE = 128
RMS_EPSILON_BITS = 897_988_541
DIRECT_PACKED_FORMAT = "direct_binary_sign_bits_plus_fp16_group_scales"
DETERMINISTIC_SAMPLER = "greedy_argmax_lowest_token_id_tie_break"
EXPECTED_STAGE_ORDER = (
    "bind_real_post48_hidden",
    "final_rms_norm",
    "all_row_lm_head",
    "mask_reserved_tail",
    "deterministic_sample",
    "validate_feedback",
)


class TerminalHeadComponentLauncherError(RuntimeError):
    """A future terminal-head component capture cannot safely start."""


@dataclass(frozen=True)
class LaunchConfig:
    probe_bin: Path
    manifest: Path
    admission_current: Path
    post48_hidden_authority: Path
    sealed_terminal_baseline: Path
    terminal_head_contract: Path
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
    source_audit_seal_sha256: str
    source_revision: str
    post48_hidden_authority: dict[str, Any]
    post48_hidden_authority_seal_sha256: str
    post48_hidden: dict[str, Any]
    sealed_terminal_baseline: dict[str, Any]
    sealed_terminal_baseline_seal_sha256: str
    terminal_head_contract: dict[str, Any]
    terminal_head_contract_seal_sha256: str
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
        raise TerminalHeadComponentLauncherError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TerminalHeadComponentLauncherError(f"{label} must be a non-empty string")
    return value


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TerminalHeadComponentLauncherError(f"{label} must be an integer")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TerminalHeadComponentLauncherError(f"{label} must be an object")
    return dict(value)


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    if not path.is_absolute():
        raise TerminalHeadComponentLauncherError(f"{label} must be absolute: {path}")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TerminalHeadComponentLauncherError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise TerminalHeadComponentLauncherError(f"{label} must be a regular non-symlink file")
    if executable and not os.access(path, os.X_OK):
        raise TerminalHeadComponentLauncherError(f"{label} must be executable")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise TerminalHeadComponentLauncherError(f"cannot canonicalize {label}: {exc}") from exc


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
        raise TerminalHeadComponentLauncherError(f"cannot read {label}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise TerminalHeadComponentLauncherError(f"{label} must be a JSON object")
    return dict(document)


def _sealed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    document = _read_json(path, label)
    try:
        verify(document, label=str(path))
    except ValueError as exc:
        raise TerminalHeadComponentLauncherError(f"{label} is not sealed: {exc}") from exc
    return document, _require_sha256(document.get("seal_sha256"), f"{label}.seal_sha256")


def _evidence_matches(evidence: object, expected: Mapping[str, Any], label: str) -> None:
    observed = _mapping(evidence, label)
    if observed.get("present") is not True:
        raise TerminalHeadComponentLauncherError(f"{label} must attest a present file")
    observed_path = _canonical_regular(Path(str(observed.get("path"))), f"{label}.path")
    if observed_path != Path(str(expected["path"])):
        raise TerminalHeadComponentLauncherError(f"{label} path drifted")
    if observed.get("bytes") != expected["bytes"] or observed.get("sha256") != expected["sha256"]:
        raise TerminalHeadComponentLauncherError(f"{label} byte/digest drifted")


def _atomic_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise TerminalHeadComponentLauncherError(f"refusing to overwrite {path}")
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
        raise TerminalHeadComponentLauncherError(f"refusing to overwrite {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _bind_manifest(path: Path) -> tuple[dict[str, Any], str]:
    document, document_seal = _sealed_json(path, "--manifest")
    if document.get("schema") != MANIFEST_SCHEMA:
        raise TerminalHeadComponentLauncherError("--manifest schema drifted")
    return _file_evidence(path, "--manifest"), document_seal


def _bind_admission(
    path: Path, manifest: Mapping[str, Any], manifest_seal: str
) -> tuple[dict[str, Any], str, str, str, str]:
    pointer, pointer_seal = _sealed_json(path, "--admission-current")
    if (
        pointer.get("schema") != ADMISSION_POINTER_SCHEMA
        or pointer.get("status") != ADMISSION_POINTER_STATUS
    ):
        raise TerminalHeadComponentLauncherError("admission pointer schema/status drifted")
    selected_manifest = _mapping(pointer.get("complete_manifest"), "admission complete_manifest")
    if _canonical_regular(Path(str(selected_manifest.get("path"))), "admission manifest path") != Path(
        str(manifest["path"])
    ):
        raise TerminalHeadComponentLauncherError("admission selects another manifest")
    if (
        selected_manifest.get("document_sha256") != manifest["sha256"]
        or selected_manifest.get("seal_sha256") != manifest_seal
    ):
        raise TerminalHeadComponentLauncherError("admission manifest identity drifted")
    selected_receipt = _mapping(pointer.get("admission_receipt"), "admission receipt selection")
    receipt_path = _canonical_regular(Path(str(selected_receipt.get("path"))), "admission receipt path")
    receipt, receipt_seal = _sealed_json(receipt_path, "admission receipt")
    receipt_evidence = _file_evidence(receipt_path, "admission receipt")
    if (
        receipt.get("schema") != ADMISSION_RECEIPT_SCHEMA
        or receipt.get("status") != ADMISSION_RECEIPT_STATUS
        or selected_receipt.get("document_sha256") != receipt_evidence["sha256"]
        or selected_receipt.get("seal_sha256") != receipt_seal
    ):
        raise TerminalHeadComponentLauncherError("admission receipt schema/status/identity drifted")
    receipt_manifest = _mapping(receipt.get("complete_manifest"), "admission receipt manifest")
    if (
        _canonical_regular(Path(str(receipt_manifest.get("path"))), "admission receipt manifest path")
        != Path(str(manifest["path"]))
        or receipt_manifest.get("document_sha256") != manifest["sha256"]
        or receipt_manifest.get("seal_sha256") != manifest_seal
    ):
        raise TerminalHeadComponentLauncherError("admission receipt manifest authority drifted")
    revalidation = _mapping(receipt.get("current_source_revalidation"), "admission source revalidation")
    revision = _require_string(revalidation.get("revision"), "admission source revision")
    if revision != SOURCE_REVISION:
        raise TerminalHeadComponentLauncherError("admission source revision drifted")
    source_audit = _require_sha256(
        revalidation.get("source_audit_seal_sha256"), "admission source audit seal"
    )
    return _file_evidence(path, "--admission-current"), pointer_seal, receipt_seal, source_audit, revision


def _validate_source_binding(
    value: object,
    label: str,
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission_pointer_seal: str,
    admission_receipt_seal: str,
    source_audit_seal: str,
    source_revision: str,
) -> dict[str, Any]:
    source = _mapping(value, label)
    if (
        source.get("model_id") != MODEL_ID
        or source.get("model_key") != MODEL_KEY
        or source.get("source_repository") != SOURCE_REPOSITORY
        or source.get("source_revision") != source_revision
        or source.get("manifest_document_sha256") != manifest["sha256"]
        or source.get("manifest_seal_sha256") != manifest_seal
        or source.get("admission_pointer_seal_sha256") != admission_pointer_seal
        or source.get("admission_receipt_seal_sha256") != admission_receipt_seal
        or source.get("source_audit_seal_sha256") != source_audit_seal
    ):
        raise TerminalHeadComponentLauncherError(f"{label} artifact/admission identity drifted")
    return source


def _bind_post48_hidden_authority(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission_pointer_seal: str,
    admission_receipt_seal: str,
    source_audit_seal: str,
    source_revision: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    authority, authority_seal = _sealed_json(path, "--post48-hidden-authority")
    if (
        authority.get("schema") != POST48_HIDDEN_SCHEMA
        or authority.get("status") != POST48_HIDDEN_STATUS
    ):
        raise TerminalHeadComponentLauncherError("post-48 hidden authority schema/status is not future source-bound evidence")
    _validate_source_binding(
        authority.get("source_binding"),
        "post-48 hidden authority source_binding",
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission_pointer_seal=admission_pointer_seal,
        admission_receipt_seal=admission_receipt_seal,
        source_audit_seal=source_audit_seal,
        source_revision=source_revision,
    )
    digests = {
        name: _require_sha256(authority.get(name), f"post-48 hidden authority {name}")
        for name in (
            "buffer_id_sha256",
            "command_graph_capture_id_sha256",
            "all_layer_hidden_sha256",
            "device_parity_receipt_seal_sha256",
            "source_token_or_feedback_provenance_sha256",
        )
    }
    if (
        authority.get("shape") != [HIDDEN]
        or authority.get("byte_length") != POST48_HIDDEN_BYTES
        or authority.get("produced_by_exact_48_layer_schedule") is not True
        or authority.get("all_48_layers_physically_completed") is not True
        or authority.get("source_bound") is not True
        or authority.get("artifact_bound") is not True
        or authority.get("synthetic_or_component_fixture") is not False
        or authority.get("fallback_used") is not False
        or authority.get("buffer_owned_by_logical_session") is not True
        or authority.get("retained_until_terminal_feedback_fence") is not True
        or authority.get("receipt_written_last_is_completion_marker") is not True
        or authority.get("token_or_generation_claim") is not False
    ):
        raise TerminalHeadComponentLauncherError(
            "post-48 hidden authority is not a retained real full-path source-bound component input"
        )
    return _file_evidence(path, "--post48-hidden-authority"), authority_seal, digests


def _packed_abi(value: object, label: str, *, tensor_name: str, shape: list[int]) -> None:
    abi = _mapping(value, label)
    if (
        abi.get("tensor_name") != tensor_name
        or abi.get("shape") != shape
        or abi.get("group_size") != GROUP_SIZE
        or abi.get("packed_format") != DIRECT_PACKED_FORMAT
        or abi.get("direct_packed_only") is not True
        or abi.get("bf16_shadow_allowed") is not False
    ):
        raise TerminalHeadComponentLauncherError(f"{label} direct-packed geometry drifted")


def _bind_baseline(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission_pointer_seal: str,
    admission_receipt_seal: str,
    source_audit_seal: str,
    source_revision: str,
) -> tuple[dict[str, Any], str]:
    baseline, baseline_seal = _sealed_json(path, "--sealed-terminal-baseline")
    if (
        baseline.get("schema") != BASELINE_SCHEMA
        or baseline.get("status") != BASELINE_STATUS
        or baseline.get("integrity_verified") is not True
    ):
        raise TerminalHeadComponentLauncherError("sealed terminal CPU baseline schema/status/integrity drifted")
    _validate_source_binding(
        baseline.get("source_binding"),
        "sealed terminal baseline source_binding",
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission_pointer_seal=admission_pointer_seal,
        admission_receipt_seal=admission_receipt_seal,
        source_audit_seal=source_audit_seal,
        source_revision=source_revision,
    )
    terminal = _mapping(baseline.get("terminal_head_cpu_receipt"), "sealed terminal baseline terminal receipt")
    if (
        terminal.get("schema") != TERMINAL_CPU_RECEIPT_SCHEMA
        or terminal.get("status") != TERMINAL_CPU_RECEIPT_STATUS
        or terminal.get("full_row_cpu_oracle") is not True
    ):
        raise TerminalHeadComponentLauncherError("sealed terminal baseline does not bind the full-row CPU terminal receipt")
    _require_sha256(terminal.get("document_sha256"), "sealed terminal baseline terminal document")
    _require_sha256(terminal.get("unsealed_preimage_sha256"), "sealed terminal baseline terminal preimage")
    tokenizer = _mapping(baseline.get("tokenizer_sampler_receipt"), "sealed terminal baseline tokenizer receipt")
    if (
        tokenizer.get("schema") != TOKENIZER_RECEIPT_SCHEMA
        or tokenizer.get("status") != TOKENIZER_RECEIPT_STATUS
        or tokenizer.get("tail_mask_before_sampler") is not True
        or tokenizer.get("tokenizer_feedback_validation_required") is not True
    ):
        raise TerminalHeadComponentLauncherError("sealed terminal baseline tokenizer/sampler binding drifted")
    _require_sha256(tokenizer.get("document_sha256"), "sealed terminal baseline tokenizer document")
    _require_sha256(tokenizer.get("unsealed_preimage_sha256"), "sealed terminal baseline tokenizer preimage")
    _packed_abi(terminal.get("final_norm_abi"), "sealed terminal baseline final norm", tensor_name="model.norm.weight", shape=[HIDDEN])
    _packed_abi(terminal.get("lm_head_abi"), "sealed terminal baseline lm head", tensor_name="lm_head.weight", shape=[LM_HEAD_ROWS, HIDDEN])
    if (
        terminal.get("rms_epsilon_bits") != RMS_EPSILON_BITS
        or terminal.get("all_lm_head_rows") != LM_HEAD_ROWS
        or terminal.get("tokenizer_addressable_rows") != TOKENIZER_VOCAB
        or terminal.get("first_reserved_id") != FIRST_RESERVED_ID
        or terminal.get("last_reserved_id") != LAST_RESERVED_ID
        or terminal.get("reserved_tail_rows") != RESERVED_TAIL_ROWS
    ):
        raise TerminalHeadComponentLauncherError("sealed terminal baseline full-row/tail geometry drifted")
    return _file_evidence(path, "--sealed-terminal-baseline"), baseline_seal


def _expected_terminal_abi() -> dict[str, Any]:
    return {
        "ordered_stages": list(EXPECTED_STAGE_ORDER),
        "final_norm": {
            "tensor_name": "model.norm.weight",
            "shape": [HIDDEN],
            "group_size": GROUP_SIZE,
            "packed_format": DIRECT_PACKED_FORMAT,
            "direct_packed_only": True,
            "bf16_shadow_allowed": False,
            "rms_epsilon_bits": RMS_EPSILON_BITS,
        },
        "lm_head": {
            "tensor_name": "lm_head.weight",
            "shape": [LM_HEAD_ROWS, HIDDEN],
            "group_size": GROUP_SIZE,
            "packed_format": DIRECT_PACKED_FORMAT,
            "direct_packed_only": True,
            "bf16_shadow_allowed": False,
            "all_rows_required": LM_HEAD_ROWS,
            "selected_row_shortcut_allowed": False,
        },
        "tail_mask": {
            "first_reserved_id": FIRST_RESERVED_ID,
            "last_reserved_id": LAST_RESERVED_ID,
            "reserved_tail_rows": RESERVED_TAIL_ROWS,
            "mask_value": "negative_infinity",
            "must_run_after_all_row_lm_head": True,
        },
        "deterministic_sample_feedback": {
            "policy": DETERMINISTIC_SAMPLER,
            "sample_must_follow_tail_mask": True,
            "sampled_token_must_be_tokenizer_addressable": True,
            "feedback_must_equal_sample": True,
            "feedback_must_be_validated_before_next_embedding_or_state_step": True,
        },
    }


def _bind_terminal_contract(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission_pointer_seal: str,
    admission_receipt_seal: str,
    source_audit_seal: str,
    source_revision: str,
    baseline: Mapping[str, Any],
    baseline_seal: str,
) -> tuple[dict[str, Any], str]:
    contract, contract_seal = _sealed_json(path, "--terminal-head-contract")
    if (
        contract.get("schema") != TERMINAL_CONTRACT_SCHEMA
        or contract.get("status") != TERMINAL_CONTRACT_STATUS
        or contract.get("component_contract_only") is not True
    ):
        raise TerminalHeadComponentLauncherError("terminal-head contract schema/status/scope drifted")
    _validate_source_binding(
        contract.get("source_binding"),
        "terminal-head contract source_binding",
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission_pointer_seal=admission_pointer_seal,
        admission_receipt_seal=admission_receipt_seal,
        source_audit_seal=source_audit_seal,
        source_revision=source_revision,
    )
    baseline_binding = _mapping(
        contract.get("sealed_terminal_baseline_binding"), "terminal-head contract baseline binding"
    )
    _evidence_matches(baseline_binding, baseline, "terminal-head contract baseline binding")
    if (
        baseline_binding.get("schema") != BASELINE_SCHEMA
        or baseline_binding.get("status") != BASELINE_STATUS
        or baseline_binding.get("seal_sha256") != baseline_seal
    ):
        raise TerminalHeadComponentLauncherError("terminal-head contract baseline seal/schema drifted")
    if contract.get("terminal_head_abi") != _expected_terminal_abi():
        raise TerminalHeadComponentLauncherError("terminal-head contract norm/head/tail/sample ABI drifted")
    boundary = _mapping(contract.get("claim_boundary"), "terminal-head contract claim_boundary")
    for key in (
        "artifact_scan_or_payload_open_performed",
        "metal_context_or_dispatch_performed",
        "model_runtime_or_server_started",
        "hcli_execution_performed",
        "tps_or_tg_measurement_performed",
        "token_or_generation_claim",
    ):
        if boundary.get(key) is not False:
            raise TerminalHeadComponentLauncherError(f"terminal-head contract {key} must be false")
    return _file_evidence(path, "--terminal-head-contract"), contract_seal


def _binding_matches(
    value: object,
    expected: Mapping[str, Any],
    label: str,
    *,
    schema: str,
    status: str,
    seal_sha256: str,
) -> None:
    binding = _mapping(value, label)
    _evidence_matches(binding, expected, label)
    if (
        binding.get("schema") != schema
        or binding.get("status") != status
        or binding.get("seal_sha256") != seal_sha256
    ):
        raise TerminalHeadComponentLauncherError(f"{label} seal/schema/status drifted")


def _bind_lease(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission_receipt_seal: str,
    post48: Mapping[str, Any],
    post48_seal: str,
    baseline: Mapping[str, Any],
    baseline_seal: str,
    contract: Mapping[str, Any],
    contract_seal: str,
) -> tuple[dict[str, Any], str]:
    lease, lease_seal = _sealed_json(path, "--lease-receipt")
    if lease.get("schema") != LEASE_SCHEMA or lease.get("status") != LEASE_STATUS:
        raise TerminalHeadComponentLauncherError("terminal-head quiet lease schema/status drifted")
    _require_string(lease.get("lease_id"), "terminal-head quiet lease ID")
    lifecycle = _mapping(lease.get("lifecycle"), "terminal-head quiet lease lifecycle")
    if (
        lifecycle.get("fresh_for_this_exact_launch") is not True
        or lifecycle.get("automatic_retry_prohibited") is not True
        or lifecycle.get("outer_reaped_capture_required") is not True
        or lifecycle.get("receipt_written_last_required") is not True
        or lifecycle.get("prior_terminal_receipt") is not None
    ):
        raise TerminalHeadComponentLauncherError("terminal-head quiet lease is not fresh/one-shot/receipt-last")
    policy = _mapping(lease.get("execution_policy"), "terminal-head quiet lease policy")
    if (
        policy.get("component") != LEASE_COMPONENT
        or policy.get("quiet_qwen80_device_lease") is not True
        or policy.get("strict_math") is not True
        or policy.get("timing_or_benchmarking_allowed") is not False
        or policy.get("complete_layer_or_token_allowed") is not False
        or policy.get("tps_or_tg_claim_allowed") is not False
        or policy.get("hcli_or_server_allowed") is not False
        or policy.get("cpu_or_bf16_fallback_allowed") is not False
        or policy.get("selected_row_lm_head_allowed") is not False
    ):
        raise TerminalHeadComponentLauncherError("terminal-head quiet lease policy is not strict component-only")
    artifact = _mapping(lease.get("artifact_binding"), "terminal-head quiet lease artifact binding")
    if (
        artifact.get("manifest_document_sha256") != manifest["sha256"]
        or artifact.get("manifest_seal_sha256") != manifest_seal
        or artifact.get("admission_receipt_seal_sha256") != admission_receipt_seal
    ):
        raise TerminalHeadComponentLauncherError("terminal-head quiet lease artifact binding drifted")
    _binding_matches(
        lease.get("post48_hidden_authority_binding"),
        post48,
        "terminal-head quiet lease post-48 authority",
        schema=POST48_HIDDEN_SCHEMA,
        status=POST48_HIDDEN_STATUS,
        seal_sha256=post48_seal,
    )
    _binding_matches(
        lease.get("sealed_terminal_baseline_binding"),
        baseline,
        "terminal-head quiet lease baseline",
        schema=BASELINE_SCHEMA,
        status=BASELINE_STATUS,
        seal_sha256=baseline_seal,
    )
    _binding_matches(
        lease.get("terminal_head_contract_binding"),
        contract,
        "terminal-head quiet lease contract",
        schema=TERMINAL_CONTRACT_SCHEMA,
        status=TERMINAL_CONTRACT_STATUS,
        seal_sha256=contract_seal,
    )
    return _file_evidence(path, "--lease-receipt"), lease_seal


def _validate_config(config: LaunchConfig) -> LaunchContext:
    if not config.capture_dir.is_absolute() or not config.capture_dir.parent.is_dir():
        raise TerminalHeadComponentLauncherError("--capture-dir must be absolute with an existing parent")
    if not config.timeout_seconds > 0:
        raise TerminalHeadComponentLauncherError("--timeout-seconds must be positive")
    probe = _file_evidence(config.probe_bin, "--probe-bin", executable=True)
    if Path(str(probe["path"])).name != EXPECTED_PROBE_BASENAME:
        raise TerminalHeadComponentLauncherError(
            f"--probe-bin must name {EXPECTED_PROBE_BASENAME}, got {Path(str(probe['path'])).name!r}"
        )
    manifest, manifest_seal = _bind_manifest(config.manifest)
    admission, pointer_seal, receipt_seal, source_audit, source_revision = _bind_admission(
        config.admission_current, manifest, manifest_seal
    )
    post48, post48_seal, post48_hidden = _bind_post48_hidden_authority(
        config.post48_hidden_authority,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission_pointer_seal=pointer_seal,
        admission_receipt_seal=receipt_seal,
        source_audit_seal=source_audit,
        source_revision=source_revision,
    )
    baseline, baseline_seal = _bind_baseline(
        config.sealed_terminal_baseline,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission_pointer_seal=pointer_seal,
        admission_receipt_seal=receipt_seal,
        source_audit_seal=source_audit,
        source_revision=source_revision,
    )
    contract, contract_seal = _bind_terminal_contract(
        config.terminal_head_contract,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission_pointer_seal=pointer_seal,
        admission_receipt_seal=receipt_seal,
        source_audit_seal=source_audit,
        source_revision=source_revision,
        baseline=baseline,
        baseline_seal=baseline_seal,
    )
    lease, lease_seal = _bind_lease(
        config.lease_receipt,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission_receipt_seal=receipt_seal,
        post48=post48,
        post48_seal=post48_seal,
        baseline=baseline,
        baseline_seal=baseline_seal,
        contract=contract,
        contract_seal=contract_seal,
    )
    return LaunchContext(
        probe_binary=probe,
        manifest=manifest,
        manifest_seal_sha256=manifest_seal,
        admission_current=admission,
        admission_pointer_seal_sha256=pointer_seal,
        admission_receipt_seal_sha256=receipt_seal,
        source_audit_seal_sha256=source_audit,
        source_revision=source_revision,
        post48_hidden_authority=post48,
        post48_hidden_authority_seal_sha256=post48_seal,
        post48_hidden=post48_hidden,
        sealed_terminal_baseline=baseline,
        sealed_terminal_baseline_seal_sha256=baseline_seal,
        terminal_head_contract=contract,
        terminal_head_contract_seal_sha256=contract_seal,
        lease_receipt=lease,
        lease_seal_sha256=lease_seal,
    )


def _launch_identity(config: LaunchConfig, context: LaunchContext) -> str:
    payload = {
        "schema": SCHEMA,
        "probe_binary_sha256": context.probe_binary["sha256"],
        "manifest_document_sha256": context.manifest["sha256"],
        "manifest_seal_sha256": context.manifest_seal_sha256,
        "admission_pointer_seal_sha256": context.admission_pointer_seal_sha256,
        "admission_receipt_seal_sha256": context.admission_receipt_seal_sha256,
        "post48_hidden_authority_sha256": context.post48_hidden_authority["sha256"],
        "post48_hidden_authority_seal_sha256": context.post48_hidden_authority_seal_sha256,
        "sealed_terminal_baseline_sha256": context.sealed_terminal_baseline["sha256"],
        "sealed_terminal_baseline_seal_sha256": context.sealed_terminal_baseline_seal_sha256,
        "terminal_head_contract_sha256": context.terminal_head_contract["sha256"],
        "terminal_head_contract_seal_sha256": context.terminal_head_contract_seal_sha256,
        "lease_receipt_sha256": context.lease_receipt["sha256"],
        "lease_seal_sha256": context.lease_seal_sha256,
        "timeout_seconds": config.timeout_seconds,
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _child_command(config: LaunchConfig, inner_capture_dir: Path) -> list[str]:
    return [
        str(_canonical_regular(config.probe_bin, "--probe-bin", executable=True)),
        "--mode",
        "component-child",
        "--manifest",
        str(_canonical_regular(config.manifest, "--manifest")),
        "--admission-current",
        str(_canonical_regular(config.admission_current, "--admission-current")),
        "--post48-hidden-authority",
        str(_canonical_regular(config.post48_hidden_authority, "--post48-hidden-authority")),
        "--sealed-terminal-baseline",
        str(_canonical_regular(config.sealed_terminal_baseline, "--sealed-terminal-baseline")),
        "--terminal-head-contract",
        str(_canonical_regular(config.terminal_head_contract, "--terminal-head-contract")),
        "--lease-receipt",
        str(_canonical_regular(config.lease_receipt, "--lease-receipt")),
        "--capture-dir",
        str(inner_capture_dir),
    ]


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
    if child.poll() is not None:
        return child.returncode
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return child.wait(timeout=10)


def _validate_inner_binding(
    value: object,
    expected: Mapping[str, Any],
    label: str,
    *,
    schema: str,
    status: str,
    seal_sha256: str,
) -> None:
    _binding_matches(value, expected, label, schema=schema, status=status, seal_sha256=seal_sha256)


def _exact_vector_parity(value: object, label: str, *, elements: int, expected_sha256: str | None = None) -> None:
    parity = _mapping(value, label)
    if (
        parity.get("elements") != elements
        or parity.get("source_device_parity_passed") is not True
        or parity.get("all_finite") is not True
    ):
        raise TerminalHeadComponentLauncherError(f"{label} vector parity is incomplete")
    digest = _require_sha256(parity.get("f32le_sha256"), f"{label}.f32le_sha256")
    if expected_sha256 is not None and digest != expected_sha256:
        raise TerminalHeadComponentLauncherError(f"{label} source buffer digest drifted")


def _validate_inner(document: Mapping[str, Any], context: LaunchContext) -> None:
    if (
        document.get("schema") != EXPECTED_INNER_SCHEMA
        or document.get("status") != EXPECTED_INNER_STATUS
        or document.get("component_only") is not True
        or document.get("complete_layer_or_token_performed") is not False
        or document.get("decoder_or_generation_performed") is not False
        or document.get("hcli_execution_performed") is not False
        or document.get("tps_or_tg_measurement_performed") is not False
        or document.get("metal_device_or_dispatch_performed") is not True
    ):
        raise TerminalHeadComponentLauncherError("inner schema/status/scope boundary drifted")
    durable = _mapping(document.get("durable_capture"), "inner durable_capture")
    if (
        durable.get("receipt_written_last_is_completion_marker") is not True
        or durable.get("post48_input_and_all_terminal_readbacks_written_before_receipt") is not True
        or durable.get("outer_reaped_capture_required") is not True
        or durable.get("replay_guarded") is not True
    ):
        raise TerminalHeadComponentLauncherError("inner receipt lacks receipt-last outer-reaped durability")
    _validate_source_binding(
        document.get("source_binding"),
        "inner source_binding",
        manifest=context.manifest,
        manifest_seal=context.manifest_seal_sha256,
        admission_pointer_seal=context.admission_pointer_seal_sha256,
        admission_receipt_seal=context.admission_receipt_seal_sha256,
        source_audit_seal=context.source_audit_seal_sha256,
        source_revision=context.source_revision,
    )
    _validate_inner_binding(
        document.get("post48_hidden_authority_binding"),
        context.post48_hidden_authority,
        "inner post-48 authority",
        schema=POST48_HIDDEN_SCHEMA,
        status=POST48_HIDDEN_STATUS,
        seal_sha256=context.post48_hidden_authority_seal_sha256,
    )
    _validate_inner_binding(
        document.get("sealed_terminal_baseline_binding"),
        context.sealed_terminal_baseline,
        "inner terminal baseline",
        schema=BASELINE_SCHEMA,
        status=BASELINE_STATUS,
        seal_sha256=context.sealed_terminal_baseline_seal_sha256,
    )
    _validate_inner_binding(
        document.get("terminal_head_contract_binding"),
        context.terminal_head_contract,
        "inner terminal contract",
        schema=TERMINAL_CONTRACT_SCHEMA,
        status=TERMINAL_CONTRACT_STATUS,
        seal_sha256=context.terminal_head_contract_seal_sha256,
    )
    _validate_inner_binding(
        document.get("lease_binding"),
        context.lease_receipt,
        "inner quiet lease",
        schema=LEASE_SCHEMA,
        status=LEASE_STATUS,
        seal_sha256=context.lease_seal_sha256,
    )
    execution = _mapping(document.get("terminal_head_execution"), "inner terminal_head_execution")
    if (
        execution.get("ordered_stages") != list(EXPECTED_STAGE_ORDER)
        or execution.get("backend") != "metal"
        or execution.get("actual_device_execution") is not True
        or _require_int(execution.get("device_dispatches"), "inner device dispatches") < 5
        or execution.get("final_fence_before_capture_receipt") is not True
        or execution.get("fixture_only") is not False
        or execution.get("fallback_used") is not False
        or execution.get("selected_row_shortcut_used") is not False
    ):
        raise TerminalHeadComponentLauncherError("inner terminal execution order/scope drifted")
    readback = _mapping(document.get("readback_parity"), "inner readback_parity")
    _exact_vector_parity(
        readback.get("post48_hidden"),
        "inner post-48 hidden parity",
        elements=HIDDEN,
        expected_sha256=context.post48_hidden["all_layer_hidden_sha256"],
    )
    _exact_vector_parity(readback.get("final_norm"), "inner final norm parity", elements=HIDDEN)
    all_rows = _mapping(readback.get("lm_head_all_rows"), "inner all-row lm-head parity")
    raw_logits_sha256 = _require_sha256(all_rows.get("raw_logits_sha256"), "inner raw logits")
    if (
        all_rows.get("rows_evaluated") != LM_HEAD_ROWS
        or all_rows.get("all_rows_evaluated") != LM_HEAD_ROWS
        or all_rows.get("full_row_cpu_device_parity_passed") is not True
        or all_rows.get("all_logits_finite_before_mask") is not True
        or all_rows.get("selected_row_shortcut_used") is not False
        or execution.get("raw_logits_sha256") != raw_logits_sha256
    ):
        raise TerminalHeadComponentLauncherError("inner full 151936-row lm-head parity is incomplete")
    tail = _mapping(readback.get("reserved_tail_mask"), "inner reserved-tail mask")
    if (
        tail.get("first_reserved_id") != FIRST_RESERVED_ID
        or tail.get("last_reserved_id") != LAST_RESERVED_ID
        or tail.get("reserved_tail_rows") != RESERVED_TAIL_ROWS
        or tail.get("every_reserved_logit_negative_infinity") is not True
        or tail.get("mask_applied_after_all_row_lm_head") is not True
    ):
        raise TerminalHeadComponentLauncherError("inner exact reserved-tail mask proof is incomplete")
    feedback = _mapping(readback.get("deterministic_sample_feedback"), "inner deterministic sample feedback")
    sampled = _require_int(feedback.get("sampled_token_id"), "inner sampled token ID")
    feedback_id = _require_int(feedback.get("feedback_token_id"), "inner feedback token ID")
    if (
        feedback.get("policy") != DETERMINISTIC_SAMPLER
        or sampled < 0
        or sampled >= TOKENIZER_VOCAB
        or feedback_id != sampled
        or feedback.get("sampled_token_is_tokenizer_addressable") is not True
        or feedback.get("sample_after_tail_mask") is not True
        or feedback.get("feedback_matches_sample") is not True
        or feedback.get("feedback_validated_before_next_embedding_or_state_step") is not True
        or feedback.get("sample_feedback_is_component_proof_not_token_claim") is not True
    ):
        raise TerminalHeadComponentLauncherError("inner deterministic masked-tail sample/feedback proof is incomplete")


def _inner_evidence(config: LaunchConfig, context: LaunchContext) -> dict[str, Any]:
    capture = config.capture_dir / INNER_CAPTURE
    receipt_path = capture / "receipt.json"
    result: dict[str, Any] = {
        "capture_dir": str(capture),
        "receipt": {"path": str(receipt_path), "present": receipt_path.is_file()},
    }
    if not receipt_path.is_file():
        return result
    try:
        receipt, receipt_seal = _sealed_json(receipt_path, "inner receipt")
        result.update(_file_evidence(receipt_path, "inner receipt"))
        result["schema"] = receipt.get("schema")
        result["status"] = receipt.get("status")
        result["seal_sha256"] = receipt_seal
        _validate_inner(receipt, context)
    except TerminalHeadComponentLauncherError as exc:
        result["binding_valid"] = False
        result["binding_error"] = str(exc)
    else:
        result["binding_valid"] = True
    return result


def _terminal_status(terminal: Mapping[str, Any], inner: Mapping[str, Any]) -> str:
    if terminal.get("spawn_error"):
        return "REFUSED_QWEN80_TERMINAL_HEAD_OUTER_CHILD_SPAWN_ERROR"
    if terminal.get("timed_out"):
        return "REFUSED_QWEN80_TERMINAL_HEAD_OUTER_CHILD_TIMEOUT"
    if terminal.get("signal") is not None:
        return "REFUSED_QWEN80_TERMINAL_HEAD_OUTER_CHILD_SIGNAL"
    if terminal.get("exit_code") != 0:
        return "REFUSED_QWEN80_TERMINAL_HEAD_OUTER_CHILD_NONZERO"
    if inner.get("binding_valid") is not True:
        return "REFUSED_QWEN80_TERMINAL_HEAD_OUTER_ZERO_EXIT_WITHOUT_FULL_ROW_PROOF"
    return CAPTURED_STATUS


def _sync_evidence(path: Path) -> dict[str, Any]:
    with path.open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    return _file_evidence(path, f"outer stream {path.name}")


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
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": _terminal_status(terminal, inner),
        "recorded_at": _utc_now(),
        "one_shot": {
            "automatic_retry_disabled": True,
            "same_capture_dir_never_starts_a_second_child": True,
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
            "post48_hidden_authority": context.post48_hidden_authority,
            "post48_hidden_authority_seal_sha256": context.post48_hidden_authority_seal_sha256,
            "sealed_terminal_baseline": context.sealed_terminal_baseline,
            "sealed_terminal_baseline_seal_sha256": context.sealed_terminal_baseline_seal_sha256,
            "terminal_head_contract": context.terminal_head_contract,
            "terminal_head_contract_seal_sha256": context.terminal_head_contract_seal_sha256,
            "lease_receipt": context.lease_receipt,
            "lease_seal_sha256": context.lease_seal_sha256,
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
            "outer_did_not_register_or_compile_a_shader": True,
            "outer_did_not_open_or_scan_model_artifacts": True,
            "outer_did_not_create_a_metal_context_or_dispatch": True,
            "outer_did_not_issue_or_mutate_a_lease": True,
            "outer_did_not_start_a_model_server_or_watcher": True,
            "outer_did_not_contact_hcli": True,
            "outer_did_not_measure_tps_or_tg": True,
            "captured_child_evidence_is_component_only_not_decoder_generation_hcli_tps_tg_tournament_or_token_claim": True,
        },
    }
    if capture_error is not None:
        payload["capture_error"] = capture_error
    return seal(payload)


def _replay(config: LaunchConfig, identity: str) -> dict[str, Any]:
    terminal_path = config.capture_dir / TERMINAL_FILENAME
    if not terminal_path.is_file():
        raise TerminalHeadComponentLauncherError("capture directory exists without terminal receipt")
    receipt, _ = _sealed_json(terminal_path, "outer terminal receipt")
    if receipt.get("schema") != SCHEMA or receipt.get("launch_identity_sha256") != identity:
        raise TerminalHeadComponentLauncherError("capture directory belongs to another launch identity")
    return receipt


def run_attempt(config: LaunchConfig) -> dict[str, Any]:
    """Outer-reap exactly one future child, or sealed-replay its terminal record."""

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
                "status": "STARTED_QWEN80_TERMINAL_HEAD_COMPONENT_OUTER_ONE_SHOT",
                "recorded_at": started_at,
                "launch_identity_sha256": identity,
                "command": command,
                "claim_boundary": {"component_only": True, "automatic_retry_disabled": True},
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
                            "status": "RUNNING_QWEN80_TERMINAL_HEAD_COMPONENT_OUTER_ONE_SHOT",
                            "recorded_at": _utc_now(),
                            "launch_identity_sha256": identity,
                            "pid": child_pid,
                            "parent_pid": os.getpid(),
                            "command": command,
                            "component_only_quiet_lease_required": True,
                        }
                    ),
                )
            except TerminalHeadComponentLauncherError as exc:
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


def assess_current_partial_evidence() -> dict[str, Any]:
    """Return the explicit hard refusal for the evidence that exists today.

    This helper intentionally performs no filesystem, subprocess, device, or
    lease activity.  It gives orchestration a stable negative answer rather
    than allowing a partial CPU fixture or unsealed preflight to look runnable.
    """

    return {
        "schema": SCHEMA,
        "status": REFUSED_CURRENT_PARTIAL_STATUS,
        "future_child_launch_eligible": False,
        "current_partial_evidence_hard_refused": True,
        "blockers": [
            "no sealed source-bound post-48 hidden device-parity authority exists",
            "no sealed terminal CPU baseline wrapper binds the full-row CPU receipt and tokenizer/sampler receipt",
            "no sealed full-row terminal ABI contract binds that baseline",
            "no fresh component-only terminal-head quiet lease is available",
        ],
        "claim_boundary": {
            "no_capture_directory_created": True,
            "no_child_started": True,
            "no_metal_or_gpu_touched": True,
            "no_model_server_or_watcher_started": True,
            "no_hcli_tps_tg_or_token_claim": True,
        },
    }


def _parse_args(arguments: Sequence[str]) -> LaunchConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-bin", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--admission-current", type=Path, required=True)
    parser.add_argument("--post48-hidden-authority", type=Path, required=True)
    parser.add_argument("--sealed-terminal-baseline", type=Path, required=True)
    parser.add_argument("--terminal-head-contract", type=Path, required=True)
    parser.add_argument("--lease-receipt", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parsed = parser.parse_args(arguments)
    return LaunchConfig(
        probe_bin=parsed.probe_bin,
        manifest=parsed.manifest,
        admission_current=parsed.admission_current,
        post48_hidden_authority=parsed.post48_hidden_authority,
        sealed_terminal_baseline=parsed.sealed_terminal_baseline,
        terminal_head_contract=parsed.terminal_head_contract,
        lease_receipt=parsed.lease_receipt,
        capture_dir=parsed.capture_dir,
        timeout_seconds=parsed.timeout_seconds,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        receipt = run_attempt(_parse_args(sys.argv[1:] if arguments is None else arguments))
    except TerminalHeadComponentLauncherError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt.get("recorded_at", ""))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
