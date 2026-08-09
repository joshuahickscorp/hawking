"""CPU/file-only Q30 range-admission bootstrap outer/reaper preflight.

This module validates only sealed metadata receipts. It cannot spawn a child,
open a source root, issue a lease, or carry out a hash scan. A PREPARED result
is a receipt-last/replay grammar for a later independently authorized
non-inference bootstrap scan; it is never permission to execute that scan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab.receipts import SealIntegrityError, seal, verify

SCHEMA = "hawking.ascension.qwen30_streamed_source_range_admission_bootstrap_outer_preflight.v1"
PREPARED_STATUS = (
    "PREPARED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_OUTER_RECEIPT_LAST_RESERVATION_NOT_SPAWNED"
)
REFUSED_STATUS = (
    "REFUSED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_OUTER_PREREQUISITES_ABSENT_OR_INVALID"
)

PREFLIGHT_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_bootstrap_preflight.v1"
)
PREFLIGHT_STATUS = (
    "PREPARED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_NOT_EXECUTED"
)
BINARY_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_bootstrap_binary_binding.v1"
)
BINARY_STATUS = (
    "COMPILED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_CPU_ONLY_NOT_EXECUTED"
)
RESOURCE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_bootstrap_resource_admission.v1"
)
RESOURCE_STATUS = (
    "ADMITTED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_ZERO_SWAP_RESOURCE_WINDOW"
)
LEASE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_bootstrap_quiet_lease.v1"
)
LEASE_STATUS = (
    "GRANTED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_HASH_SCAN_ONE_SHOT"
)
CAPTURE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_bootstrap_capture.v1"
)
CAPTURE_STATUS = (
    "CAPTURED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_HASH_SCAN_NOT_SOURCE_TEACHER"
)
REPLAY_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_bootstrap_replay_reservation.v1"
)
REPLAY_STATUS = (
    "RESERVED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_ONE_SHOT_CAPTURE_NOT_SPAWNED"
)
OUTER_TERMINAL_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_bootstrap_outer_terminal.v1"
)
OUTER_TERMINAL_STATUS = (
    "CAPTURED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_OUTER_TERMINAL_NOT_SOURCE_TEACHER"
)
RELEASE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_bootstrap_quiet_lease_release.v1"
)
RELEASE_STATUS = (
    "RELEASED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_LEASE_AFTER_OUTER_TERMINAL"
)

MAX_METADATA_BYTES = 64 * 1024 * 1024
MAX_WINDOW_BYTES = 1024 * 1024
PRODUCTION_SHARDS = 16
PRODUCTION_TENSORS = 18_867


class BootstrapOuterError(RuntimeError):
    """A supplied record cannot reserve the future bootstrap lifecycle."""


@dataclass(frozen=True)
class Document:
    path: Path
    document: dict[str, Any]
    raw_document_sha256: str
    seal_sha256: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BootstrapOuterError(f"{label} must be an object")
    return dict(value)


def _text(value: object, *, label: str, sha256: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise BootstrapOuterError(f"{label} must be non-empty text")
    if sha256 and (
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BootstrapOuterError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BootstrapOuterError(f"{label} must be an integer >= {minimum}")
    return value


def _require(value: object, *, expected: bool, label: str) -> None:
    if value is not expected:
        raise BootstrapOuterError(f"{label} must be {expected}")


def _schema_status(
    document: Mapping[str, Any], *, schema: str, status: str, label: str
) -> None:
    if document.get("schema") != schema or document.get("status") != status:
        raise BootstrapOuterError(f"{label} schema/status drifted")


def _regular_json(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.suffix != ".json":
        raise BootstrapOuterError(f"{label} must be an absolute .json path")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise BootstrapOuterError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise BootstrapOuterError(f"{label} must be a regular non-symlink file")
    if metadata.st_size <= 0 or metadata.st_size > MAX_METADATA_BYTES:
        raise BootstrapOuterError(f"{label} has invalid metadata size")
    return path.resolve(strict=True)


def _sealed(path: Path, *, label: str) -> Document:
    clean = _regular_json(path, label=label)
    try:
        raw_bytes = clean.read_bytes()
        raw = json.loads(raw_bytes)
        checked = verify(raw, label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise BootstrapOuterError(f"{label} is absent or invalid: {exc}") from exc
    document = _mapping(checked, label=label)
    return Document(
        path=clean,
        document=document,
        raw_document_sha256=_sha256_bytes(raw_bytes),
        seal_sha256=_text(document.get("seal_sha256"), label=f"{label} seal", sha256=True),
    )


def _evidence(document: Document) -> dict[str, Any]:
    return {
        "path": str(document.path),
        "raw_document_sha256": document.raw_document_sha256,
        "seal_sha256": document.seal_sha256,
    }


def _pointer(value: object, *, expected: Document, label: str) -> None:
    pointer = _mapping(value, label=label)
    if (
        _text(pointer.get("raw_document_sha256"), label=f"{label}.raw_document_sha256", sha256=True)
        != expected.raw_document_sha256
        or _text(pointer.get("seal_sha256"), label=f"{label}.seal_sha256", sha256=True)
        != expected.seal_sha256
    ):
        raise BootstrapOuterError(f"{label} does not bind the supplied sealed document")


def _validate_preflight(document: Document) -> None:
    root = document.document
    _schema_status(
        root, schema=PREFLIGHT_SCHEMA, status=PREFLIGHT_STATUS, label="bootstrap preflight"
    )
    _require(root.get("prepared"), expected=True, label="bootstrap preflight.prepared")
    _require(
        root.get("execution_authorized"),
        expected=False,
        label="bootstrap preflight.execution_authorized",
    )
    bindings = _mapping(root.get("metadata_bindings"), label="bootstrap preflight metadata")
    for field in (
        "range_authority_document_sha256",
        "range_authority_content_sha256",
        "source_index_sha256",
    ):
        _text(bindings.get(field), label=f"bootstrap preflight metadata.{field}", sha256=True)
    if _integer(
        bindings.get("maximum_declared_bf16_window_bytes"),
        label="bootstrap preflight maximum row window",
        minimum=1,
    ) > MAX_WINDOW_BYTES:
        raise BootstrapOuterError("bootstrap preflight max row window exceeds <=1 MiB")
    lease = _mapping(root.get("future_bootstrap_lease"), label="bootstrap preflight lease")
    _schema_status(
        lease, schema=LEASE_SCHEMA, status=LEASE_STATUS, label="bootstrap preflight future lease"
    )
    for field in (
        "one_shot",
        "separate_from_source_teacher_lease",
        "non_inference_only",
    ):
        _require(lease.get(field), expected=True, label=f"bootstrap preflight lease.{field}")
    _require(
        lease.get("model_server_gpu_hcli_or_tps_allowed"),
        expected=False,
        label="bootstrap preflight lease disallowed surface",
    )
    if _integer(
        lease.get("maximum_positioned_read_bytes"),
        label="bootstrap preflight max read",
        minimum=1,
    ) != MAX_WINDOW_BYTES:
        raise BootstrapOuterError("bootstrap preflight max positioned read drifted")
    if _integer(
        lease.get("maximum_live_raw_bf16_windows"),
        label="bootstrap preflight live windows",
        minimum=1,
    ) != 1:
        raise BootstrapOuterError("bootstrap preflight must reserve exactly one raw window")
    outputs = _mapping(
        root.get("future_outputs_required_before_source_teacher"),
        label="bootstrap preflight future outputs",
    )
    flat = _mapping(outputs.get("flat_runtime_range_map"), label="bootstrap flat output")
    if (
        flat.get("schema") != "hawking.ascension.qwen30_source_bf16_range_map.v1"
        or _integer(flat.get("shards"), label="bootstrap flat shards") != PRODUCTION_SHARDS
        or _integer(flat.get("tensors"), label="bootstrap flat tensors") != PRODUCTION_TENSORS
    ):
        raise BootstrapOuterError("bootstrap preflight flat map scope drifted")
    runtime = _mapping(outputs.get("runtime_admission"), label="bootstrap runtime admission")
    _schema_status(
        runtime,
        schema="hawking.ascension.qwen30_streamed_source_teacher_runtime_range_admission.v1",
        status="EARNED_QWEN30_STREAMED_SOURCE_TEACHER_RUNTIME_RANGE_ADMISSION_NO_MODEL_RESIDENCY",
        label="bootstrap future runtime admission",
    )
    boundary = _mapping(root.get("execution_boundary"), label="bootstrap preflight boundary")
    for field in (
        "source_root_opened_or_statted",
        "source_tensor_payload_opened",
        "flat_runtime_range_map_emitted",
        "two_attestations_emitted",
        "runtime_admission_earned",
        "source_teacher_started",
        "model_gpu_server_hcli_or_tps_action",
        "lease_issued_or_consumed",
    ):
        _require(boundary.get(field), expected=False, label=f"bootstrap preflight boundary.{field}")


def _validate_binary(document: Document, *, preflight: Document) -> str:
    root = document.document
    _schema_status(root, schema=BINARY_SCHEMA, status=BINARY_STATUS, label="bootstrap binary")
    _require(root.get("cpu_only"), expected=True, label="bootstrap binary.cpu_only")
    _require(root.get("scan_or_runtime_executed"), expected=False, label="bootstrap binary execution")
    binary_sha = _text(root.get("binary_sha256"), label="bootstrap binary SHA", sha256=True)
    _text(root.get("source_sha256"), label="bootstrap source SHA", sha256=True)
    _pointer(
        root.get("bootstrap_preflight"),
        expected=preflight,
        label="bootstrap binary.bootstrap_preflight",
    )
    return binary_sha


def _validate_resource(document: Document, *, preflight: Document, binary_sha: str) -> str:
    root = document.document
    _schema_status(root, schema=RESOURCE_SCHEMA, status=RESOURCE_STATUS, label="bootstrap resource")
    for field in (
        "fresh_observation",
        "exclusive_clean_window",
        "zero_swap",
        "zero_swapouts",
        "resource_admitted_for_one_future_child",
    ):
        _require(root.get(field), expected=True, label=f"bootstrap resource.{field}")
    for field in (
        "source_payload_opened",
        "source_model_loaded",
        "gpu_server_hcli_or_tps_action",
        "lease_issued_or_consumed",
        "child_started",
    ):
        _require(root.get(field), expected=False, label=f"bootstrap resource.{field}")
    if _integer(root.get("swap_used_bytes"), label="bootstrap resource swap") != 0:
        raise BootstrapOuterError("bootstrap resource must show zero swap")
    if _integer(root.get("swapouts_pages_delta"), label="bootstrap resource swapouts") != 0:
        raise BootstrapOuterError("bootstrap resource must show zero swapout growth")
    if _integer(root.get("reclaimable_bytes"), label="bootstrap resource reclaimable", minimum=1) < _integer(
        root.get("minimum_reclaimable_bytes_required"),
        label="bootstrap resource reclaimable floor",
        minimum=1,
    ):
        raise BootstrapOuterError("bootstrap resource reclaimable floor is not met")
    if _text(root.get("bootstrap_binary_sha256"), label="bootstrap resource binary SHA", sha256=True) != binary_sha:
        raise BootstrapOuterError("bootstrap resource binary binding drifted")
    _pointer(
        root.get("bootstrap_preflight"),
        expected=preflight,
        label="bootstrap resource.bootstrap_preflight",
    )
    return _text(root.get("resource_window_identity_sha256"), label="bootstrap resource window ID", sha256=True)


def _validate_lease(
    document: Document,
    *,
    preflight: Document,
    binary_sha: str,
    resource: Document,
    resource_window_id: str,
) -> str:
    root = document.document
    _schema_status(root, schema=LEASE_SCHEMA, status=LEASE_STATUS, label="bootstrap lease")
    for field in (
        "fresh_for_this_exact_launch",
        "one_shot",
        "non_inference_only",
        "new_capture_root_required",
        "existing_output_reuse_forbidden",
        "replay_or_relaunch_forbidden",
        "separate_from_source_teacher_lease",
    ):
        _require(root.get(field), expected=True, label=f"bootstrap lease.{field}")
    for field in (
        "source_teacher_or_logits_authorized",
        "model_gpu_server_hcli_or_tps_authorized",
        "lease_consumed_by_this_preflight",
    ):
        _require(root.get(field), expected=False, label=f"bootstrap lease.{field}")
    if _text(root.get("bootstrap_binary_sha256"), label="bootstrap lease binary SHA", sha256=True) != binary_sha:
        raise BootstrapOuterError("bootstrap lease binary binding drifted")
    if _text(
        root.get("resource_window_identity_sha256"),
        label="bootstrap lease resource ID",
        sha256=True,
    ) != resource_window_id:
        raise BootstrapOuterError("bootstrap lease resource window binding drifted")
    _pointer(
        root.get("bootstrap_preflight"),
        expected=preflight,
        label="bootstrap lease.bootstrap_preflight",
    )
    _pointer(
        root.get("resource_admission"),
        expected=resource,
        label="bootstrap lease.resource_admission",
    )
    return _text(root.get("lease_id"), label="bootstrap lease ID", sha256=True)


def _future_lifecycle() -> dict[str, Any]:
    return {
        "future_child_command": [
            "ascension_qwen30_streamed_source_range_admission_bootstrap",
            "--mode",
            "bootstrap-scan",
            "--range-authority",
            "ABSOLUTE_METADATA_RANGE_AUTHORITY_JSON",
            "--semantics",
            "ABSOLUTE_METADATA_SEMANTICS_JSON",
            "--bootstrap-lease",
            "ABSOLUTE_FRESH_ONE_SHOT_BOOTSTRAP_LEASE_JSON",
            "--source-root",
            "ABSOLUTE_CANONICAL_QWEN30_SOURCE_ROOT",
            "--capture-dir",
            "NEW_ABSOLUTE_BOOTSTRAP_CAPTURE_DIR",
            "--out",
            "NEW_ABSOLUTE_BOOTSTRAP_CAPTURE_RECEIPT_JSON",
        ],
        "replay_reservation": {
            "schema": REPLAY_SCHEMA,
            "status": REPLAY_STATUS,
            "create_new_before_child": True,
            "one_child_maximum": True,
            "replay_or_relaunch_forbidden": True,
            "existing_capture_without_matching_terminal_must_refuse": True,
        },
        "child_capture": {
            "schema": CAPTURE_SCHEMA,
            "status": CAPTURE_STATUS,
            "one_bounded_window": True,
            "flat_map_and_two_attestations_required": True,
            "source_teacher_or_logits_forbidden": True,
            "receipt_written_last": True,
        },
        "outer_terminal": {
            "schema": OUTER_TERMINAL_SCHEMA,
            "status": OUTER_TERMINAL_STATUS,
            "child_must_be_reaped_before_terminal": True,
        },
        "release": {
            "schema": RELEASE_SCHEMA,
            "status": RELEASE_STATUS,
            "separate_release_after_outer_terminal": True,
        },
    }


def build_outer_preflight(
    *,
    preflight_path: Path,
    binary_path: Path | None = None,
    resource_path: Path | None = None,
    lease_path: Path | None = None,
) -> dict[str, Any]:
    """Produce a sealed PREPARED reservation or sealed refusal before any spawn.

    This has no process, source-root, or lease issuer surface. Every optional
    input is a metadata receipt and is read only when its path is supplied.
    """
    preflight = _sealed(preflight_path, label="bootstrap preflight")
    _validate_preflight(preflight)
    blockers: list[str] = []
    binary: Document | None = None
    resource: Document | None = None
    lease: Document | None = None
    binary_sha: str | None = None
    resource_window_id: str | None = None

    if binary_path is None:
        blockers.append("bootstrap_binary_binding_absent")
    else:
        try:
            binary = _sealed(binary_path, label="bootstrap binary")
            binary_sha = _validate_binary(binary, preflight=preflight)
        except BootstrapOuterError as exc:
            blockers.append(f"bootstrap_binary_binding_invalid:{exc}")

    if resource_path is None:
        blockers.append("fresh_zero_swap_resource_admission_absent")
    elif binary_sha is None:
        blockers.append("resource_admission_not_evaluated_without_valid_binary")
    else:
        try:
            resource = _sealed(resource_path, label="bootstrap resource")
            resource_window_id = _validate_resource(
                resource, preflight=preflight, binary_sha=binary_sha
            )
        except BootstrapOuterError as exc:
            blockers.append(f"fresh_zero_swap_resource_admission_invalid:{exc}")

    if lease_path is None:
        blockers.append("fresh_non_inference_bootstrap_lease_absent")
    elif binary_sha is None or resource is None or resource_window_id is None:
        blockers.append("bootstrap_lease_not_evaluated_without_valid_binary_and_resource")
    else:
        try:
            lease = _sealed(lease_path, label="bootstrap lease")
            _validate_lease(
                lease,
                preflight=preflight,
                binary_sha=binary_sha,
                resource=resource,
                resource_window_id=resource_window_id,
            )
        except BootstrapOuterError as exc:
            blockers.append(f"fresh_non_inference_bootstrap_lease_invalid:{exc}")

    prepared = not blockers
    return seal(
        {
            "schema": SCHEMA,
            "status": PREPARED_STATUS if prepared else REFUSED_STATUS,
            "prepared": prepared,
            "spawn_permitted": False,
            "bootstrap_preflight": _evidence(preflight),
            "bootstrap_binary": _evidence(binary) if binary else {"present": False},
            "zero_swap_resource_admission": _evidence(resource)
            if resource
            else {"present": False},
            "bootstrap_lease": _evidence(lease) if lease else {"present": False},
            "future_lifecycle": _future_lifecycle(),
            "blockers": blockers,
            "execution_boundary": {
                "source_root_opened_or_statted": False,
                "source_tensor_payload_opened": False,
                "source_model_loaded": False,
                "gpu_server_hcli_or_tps_action": False,
                "lease_issued_or_consumed": False,
                "replay_reservation_created": False,
                "child_spawned": False,
                "outer_terminal_written": False,
                "lease_released": False,
            },
            "claim_boundary": "CPU/file-only outer preflight. It does not create a replay guard, spawn/reap a child, open source payloads, issue/consume/release a lease, or claim a range map, attestation, runtime admission, source teacher, GPU, server, TPS, or tournament result.",
        }
    )


def validate_fake_child_and_replay(
    *,
    reservation: Mapping[str, Any],
    child_capture: Mapping[str, Any],
    outer_terminal: Mapping[str, Any],
    lease_id: str,
    preflight_seal_sha256: str,
    binary_sha256: str,
) -> None:
    """Validate future lifecycle-shaped mappings for tests or later reaper code.

    This takes mappings only and has no child or filesystem launch surface.
    """
    _schema_status(reservation, schema=REPLAY_SCHEMA, status=REPLAY_STATUS, label="replay")
    for field in (
        "create_new_before_child",
        "one_child_maximum",
        "replay_or_relaunch_forbidden",
    ):
        _require(reservation.get(field), expected=True, label=f"replay.{field}")
    if _integer(reservation.get("attempt"), label="replay.attempt", minimum=1) != 1:
        raise BootstrapOuterError("replay attempt must be exactly one")
    if _text(reservation.get("lease_id"), label="replay lease ID", sha256=True) != lease_id:
        raise BootstrapOuterError("replay lease ID drifted")
    if _text(
        reservation.get("preflight_seal_sha256"), label="replay preflight seal", sha256=True
    ) != preflight_seal_sha256:
        raise BootstrapOuterError("replay preflight binding drifted")

    _schema_status(
        child_capture, schema=CAPTURE_SCHEMA, status=CAPTURE_STATUS, label="fake child capture"
    )
    for field in (
        "non_inference_only",
        "one_bounded_window",
        "flat_runtime_range_map_emitted",
        "operator_attestation_emitted",
        "range_reader_attestation_emitted",
        "receipt_written_last",
    ):
        _require(child_capture.get(field), expected=True, label=f"fake child capture.{field}")
    for field in (
        "source_teacher_started",
        "logits_or_vectors_written",
        "source_model_loaded",
        "gpu_server_hcli_or_tps_action",
    ):
        _require(child_capture.get(field), expected=False, label=f"fake child capture.{field}")
    if _text(child_capture.get("lease_id"), label="fake child lease ID", sha256=True) != lease_id:
        raise BootstrapOuterError("fake child lease binding drifted")
    if _text(
        child_capture.get("binary_sha256"), label="fake child binary SHA", sha256=True
    ) != binary_sha256:
        raise BootstrapOuterError("fake child binary binding drifted")

    _schema_status(
        outer_terminal,
        schema=OUTER_TERMINAL_SCHEMA,
        status=OUTER_TERMINAL_STATUS,
        label="fake outer terminal",
    )
    for field in (
        "child_reaped",
        "terminal_receipt_written_last",
        "automatic_retry_disabled",
        "lease_reuse_prohibited",
    ):
        _require(outer_terminal.get(field), expected=True, label=f"fake outer terminal.{field}")
    _require(
        outer_terminal.get("child_timed_out"),
        expected=False,
        label="fake outer terminal.child_timed_out",
    )
    if _integer(outer_terminal.get("child_exit_code"), label="fake outer exit") != 0:
        raise BootstrapOuterError("fake outer terminal child exit must be zero")
    if _text(outer_terminal.get("lease_id"), label="fake outer lease ID", sha256=True) != lease_id:
        raise BootstrapOuterError("fake outer lease binding drifted")


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise BootstrapOuterError("--out must be a new absolute path below an existing parent")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--resource-admission", type=Path)
    parser.add_argument("--bootstrap-lease", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_outer_preflight(
            preflight_path=args.preflight,
            binary_path=args.binary,
            resource_path=args.resource_admission,
            lease_path=args.bootstrap_lease,
        )
        _write_new(args.out, result)
    except BootstrapOuterError as exc:
        print(f"Q30 range-admission bootstrap outer preflight refused: {exc}")
        return 2
    print(
        json.dumps(
            {
                "output": str(args.out.resolve()),
                "status": result["status"],
                "seal_sha256": result["seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

