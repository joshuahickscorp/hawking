"""Fail-closed outer controller preflight for Q30's six-vector oracle.

This controller deliberately has no subprocess, Metal, MPS, HCLI, server, or
watcher control.  It prepares and validates the two separately leased future
captures needed for the source/control/candidate numerical discriminator:

1. a source-BF16 teacher capture of two F32LE logits, then durable eviction;
2. a native direct-packed raw-logit successor capture of four F32LE logits.

The two phases may never coexist in unified memory.  A future process runner
can consume this controller's immutable launch plan only after the source
memory preflight is READY and both fresh one-shot leases/terminal evidence
exist.  The present memory receipt is blocked, so this module cannot launch a
child and records that refusal as evidence rather than guessing at eviction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.operators import ascension_qwen30_quality_repack_raw_final_logit_retention_build_binding as build_binding
from lab.operators import ascension_qwen30_quality_repack_raw_final_logit_retention_contract as retention_contract
from lab.operators import ascension_qwen30_quality_repack_source_bf16_memory_lease_preflight as memory_preflight
from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates"
    / "gate-up-residual-v1"
)
DEFAULT_CONTRACT = (
    DEFAULT_ROOT / "raw-final-logit-retention-successor/receipts"
    / "QWEN30_HQ30GR2_RAW_FINAL_LOGIT_RETENTION_SUCCESSOR_"
    "07260bb96d09dab6ba7b0955c4f72da541404dfb5c38117dffe944173a9e8e34.json"
)
DEFAULT_BUILD_BINDING = (
    DEFAULT_ROOT / "raw-final-logit-retention-successor/build-bindings"
    / "QWEN30_HQ30GR2_RAW_FINAL_LOGIT_RETENTION_EXECUTOR_"
    "88286a683085ea5ab35cd0812aee1bb2c4a16682a00a5a3f8f4dd150b3865c79.json"
)
DEFAULT_MEMORY_PREFLIGHT = (
    DEFAULT_ROOT / "source-bf16-three-way-memory-preflight/receipts"
    / "QWEN30_HQ30GR2_SOURCE_BF16_MEMORY_PREFLIGHT_"
    "efdacf5952583dc03d2aee37b73a0af284f2d865a3e36b9739b16150efe3f726.json"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "raw-final-logit-retention-successor/outer-controller-preflights"

SCHEMA = "hawking.ascension.qwen30_hq30gr2_raw_final_logit_retention_outer_controller.v1"
BLOCKED_STATUS = "BLOCKED_RAW_FINAL_LOGIT_RETENTION_OUTER_CONTROLLER_SOURCE_MEMORY_PREFLIGHT"
READY_STATUS = "PREPARED_RAW_FINAL_LOGIT_RETENTION_OUTER_CONTROLLER_AWAITING_FRESH_SEPARATE_LEASES"

PINNED_EXECUTOR_SHA256 = "88286a683085ea5ab35cd0812aee1bb2c4a16682a00a5a3f8f4dd150b3865c79"
SOURCE_LEASE_SCHEMA = "hawking.ascension.qwen30_hq30gr2_source_bf16_teacher_quiet_lease.v1"
SOURCE_LEASE_STATUS = "GRANTED_QWEN30_HQ30GR2_SOURCE_BF16_TEACHER_RAW_LOGIT_CAPTURE_ONE_SHOT"
NATIVE_LEASE_SCHEMA = "hawking.ascension.qwen30_hq30gr2_raw_final_logit_retention_quiet_lease.v1"
NATIVE_LEASE_STATUS = "GRANTED_QWEN30_HQ30GR2_RAW_FINAL_LOGIT_RETENTION_ONE_SHOT"
SOURCE_TERMINAL_SCHEMA = "hawking.ascension.qwen30_hq30gr2_source_bf16_teacher_raw_logit_capture.v1"
SOURCE_TERMINAL_STATUS = "CAPTURED_QWEN30_HQ30GR2_SOURCE_BF16_TWO_RAW_FINAL_LOGITS_TEACHER_ONLY"
SOURCE_EVICTION_SCHEMA = "hawking.ascension.qwen30_hq30gr2_source_bf16_teacher_eviction.v1"
SOURCE_EVICTION_STATUS = "EARNED_QWEN30_HQ30GR2_SOURCE_BF16_TEACHER_EVICTED_BEFORE_NATIVE_CAPTURE"
NATIVE_TERMINAL_SCHEMA = "hawking.ascension.qwen30_hq30gr2_raw_final_logit_retention_capture.v1"
NATIVE_TERMINAL_STATUS = "EARNED_NEW_DIAGNOSTIC_RAW_FINAL_LOGITS_RETAINED_NOT_THREE_WAY_ORACLE"


class RawFinalLogitOuterControllerError(RuntimeError):
    """The six-vector capture sequence is not safe to prepare or consume."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, *, label: str, executable: bool = False) -> Path:
    if not path.is_absolute():
        raise RawFinalLogitOuterControllerError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RawFinalLogitOuterControllerError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RawFinalLogitOuterControllerError(f"{label} must be a regular non-symlink file")
    if executable and not os.access(path, os.X_OK):
        raise RawFinalLogitOuterControllerError(f"{label} must be executable")
    return path.resolve(strict=True)


def _sealed(path: Path, *, label: str) -> tuple[dict[str, Any], Path]:
    path = _regular(path, label=label)
    try:
        checked = verify(json.loads(path.read_text(encoding="utf-8")), label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise RawFinalLogitOuterControllerError(f"{label} is absent or invalid: {exc}") from exc
    if not isinstance(checked, Mapping):
        raise RawFinalLogitOuterControllerError(f"{label} is not an object")
    return dict(checked), path


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RawFinalLogitOuterControllerError(f"{label} must be an object")
    return dict(value)


def _text(value: object, *, label: str, sha256: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise RawFinalLogitOuterControllerError(f"{label} must be a non-empty string")
    if sha256 and (len(value) != 64 or any(char not in "0123456789abcdef" for char in value)):
        raise RawFinalLogitOuterControllerError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RawFinalLogitOuterControllerError(f"{label} must be an integer >= {minimum}")
    return value


def _evidence(path: Path, document: Mapping[str, Any] | None = None) -> dict[str, Any]:
    stat_result = path.stat()
    result = {
        "path": str(path),
        "bytes": stat_result.st_size,
        "sha256": _sha256_file(path),
    }
    if document is not None:
        result["seal_sha256"] = _text(document.get("seal_sha256"), label="evidence seal", sha256=True)
    return result


def _write_new_json(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute():
        raise RawFinalLogitOuterControllerError("output must be absolute")
    if path.exists():
        raise RawFinalLogitOuterControllerError(f"refusing to overwrite immutable preflight {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False)
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
        raise RawFinalLogitOuterControllerError(f"refusing to overwrite immutable preflight {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _schema_status(document: Mapping[str, Any], *, schema: str, status: str, label: str) -> None:
    if document.get("schema") != schema or document.get("status") != status:
        raise RawFinalLogitOuterControllerError(f"{label} schema/status drifted")


def _vector_rules(contract: Mapping[str, Any]) -> dict[str, Any]:
    rules = _mapping(contract.get("six_vector_retention_contract"), label="raw retention six-vector contract")
    expected = retention_contract.raw_vector_plan()
    for key in ("dtype", "vocab_rows", "bytes_per_vector", "required_payload_count", "required_total_payload_bytes"):
        if rules.get(key) != expected.get(key):
            raise RawFinalLogitOuterControllerError(f"six-vector contract {key} drifted")
    if rules.get("required_payloads") != expected["required_payloads"]:
        raise RawFinalLogitOuterControllerError("six-vector contract payload ordering/names drifted")
    return expected


def _native_expected_hashes(contract: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    replay = _mapping(contract.get("replay_binding"), label="raw retention replay binding")
    rows = _mapping(
        replay.get("native_raw_hashes_must_replay_prior_98db_witness"),
        label="raw retention native 98db replay hashes",
    )
    expected: dict[str, dict[str, str]] = {}
    for endpoint in retention_contract.ENDPOINTS:
        row = _mapping(rows.get(endpoint), label=f"native 98db replay {endpoint}")
        expected[endpoint] = {
            "scalar_control": _text(row.get("scalar_control"), label=f"scalar {endpoint} expected SHA", sha256=True),
            "hq30gr2_candidate": _text(row.get("hq30gr2_candidate"), label=f"candidate {endpoint} expected SHA", sha256=True),
        }
    return expected


def _validated_vector_metadata(
    *,
    rules: Mapping[str, Any],
    row: Mapping[str, Any],
    expected_filename: str,
    label: str,
) -> dict[str, Any]:
    path = Path(_text(row.get("path"), label=f"{label} path"))
    if path.name != expected_filename:
        raise RawFinalLogitOuterControllerError(f"{label} filename drifted")
    if row.get("dtype") != rules["dtype"]:
        raise RawFinalLogitOuterControllerError(f"{label} dtype drifted")
    if _integer(row.get("vocab_rows"), label=f"{label} rows") != rules["vocab_rows"]:
        raise RawFinalLogitOuterControllerError(f"{label} row count drifted")
    if _integer(row.get("bytes"), label=f"{label} bytes") != rules["bytes_per_vector"]:
        raise RawFinalLogitOuterControllerError(f"{label} byte count drifted")
    _text(row.get("sha256"), label=f"{label} SHA", sha256=True)
    if row.get("all_values_finite") is not True:
        raise RawFinalLogitOuterControllerError(f"{label} lacks finite-value witness")
    return dict(row)


def validate_source_payload_retention(
    *, contract: Mapping[str, Any], source_payloads: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Validate future source-teacher payload geometry before native may start."""
    rules = _vector_rules(contract)
    result: dict[str, dict[str, Any]] = {}
    for endpoint in retention_contract.ENDPOINTS:
        row = _mapping(source_payloads.get(endpoint), label=f"source payload {endpoint}")
        result[endpoint] = _validated_vector_metadata(
            rules=rules,
            row=row,
            expected_filename=f"source_bf16_{endpoint}_logits.f32le",
            label=f"source payload {endpoint}",
        )
    return result


def validate_native_payload_replay(
    *, contract: Mapping[str, Any], native_payloads: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Validate future native terminal payload metadata without opening payloads.

    The terminal controller additionally hashes and decodes the payload files;
    this preflight-level guard is intentionally strict about the immutable
    expected names, full geometry, and old-capture hash replay.
    """
    expected = _native_expected_hashes(contract)
    rules = _vector_rules(contract)
    result: dict[str, dict[str, Any]] = {}
    for model in retention_contract.NATIVE_RAW_VECTOR_MODELS:
        model_rows = _mapping(native_payloads.get(model), label=f"native payloads {model}")
        result[model] = {}
        for endpoint in retention_contract.ENDPOINTS:
            row = _mapping(model_rows.get(endpoint), label=f"native payload {model}/{endpoint}")
            checked = _validated_vector_metadata(
                rules=rules,
                row=row,
                expected_filename=f"{model}_{endpoint}_logits.f32le",
                label=f"native payload {model}/{endpoint}",
            )
            observed = _text(checked.get("sha256"), label=f"native payload {model}/{endpoint} SHA", sha256=True)
            if observed != expected[endpoint][model]:
                raise RawFinalLogitOuterControllerError(f"native payload {model}/{endpoint} did not replay 98db hash")
            result[model][endpoint] = checked
    return result


def validate_six_vector_terminal_set(
    *, contract: Mapping[str, Any], source_payloads: Mapping[str, Any], native_payloads: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate all six terminal metadata rows before a metric reader opens them."""
    checked_source = validate_source_payload_retention(contract=contract, source_payloads=source_payloads)
    checked_native = validate_native_payload_replay(contract=contract, native_payloads=native_payloads)
    rows = [*checked_source.values()]
    rows.extend(
        checked_native[model][endpoint]
        for model in retention_contract.NATIVE_RAW_VECTOR_MODELS
        for endpoint in retention_contract.ENDPOINTS
    )
    paths = [str(row["path"]) for row in rows]
    hashes = [str(row["sha256"]) for row in rows]
    if len(paths) != 6 or len(set(paths)) != 6:
        raise RawFinalLogitOuterControllerError("six-vector terminal set must retain six distinct payload paths")
    if len(hashes) != 6:
        raise RawFinalLogitOuterControllerError("six-vector terminal set must contain six hashes")
    return {
        "source_payloads": checked_source,
        "native_payloads": checked_native,
        "payload_count": len(rows),
        "total_bytes": sum(_integer(row["bytes"], label="validated payload bytes") for row in rows),
        "metric_not_evaluated_by_controller": True,
    }


def validate_source_eviction_before_native(
    *, source_terminal: Mapping[str, Any], source_eviction: Mapping[str, Any], native_lease: Mapping[str, Any]
) -> None:
    """Future terminal guard: source capture must be evicted before native lease."""
    _schema_status(source_terminal, schema=SOURCE_TERMINAL_SCHEMA, status=SOURCE_TERMINAL_STATUS, label="source teacher terminal")
    _schema_status(source_eviction, schema=SOURCE_EVICTION_SCHEMA, status=SOURCE_EVICTION_STATUS, label="source teacher eviction")
    _schema_status(native_lease, schema=NATIVE_LEASE_SCHEMA, status=NATIVE_LEASE_STATUS, label="native raw retention lease")
    eviction = _mapping(source_eviction.get("eviction"), label="source teacher eviction facts")
    for key in (
        "source_weights_evicted",
        "source_backend_shutdown",
        "source_model_residency_released",
        "swap_remained_zero",
        "pre_native_lease_process_tree_checked",
    ):
        if eviction.get(key) is not True:
            raise RawFinalLogitOuterControllerError(f"source eviction lacks {key}=true")
    lifecycle = _mapping(native_lease.get("one_shot_lifecycle"), label="native raw retention lease lifecycle")
    if lifecycle.get("fresh_for_this_exact_launch") is not True or lifecycle.get("prior_terminal_receipt") is not None:
        raise RawFinalLogitOuterControllerError("native raw retention lease is not fresh one-shot evidence")
    if lifecycle.get("automatic_retry_allowed") is not False:
        raise RawFinalLogitOuterControllerError("native raw retention lease must forbid automatic retry")
    source_receipt = _mapping(source_eviction.get("source_teacher_terminal"), label="source eviction source terminal pointer")
    if source_receipt.get("seal_sha256") != source_terminal.get("seal_sha256"):
        raise RawFinalLogitOuterControllerError("source eviction does not bind source teacher terminal seal")


def build_preflight(*, contract_path: Path, build_binding_path: Path, memory_preflight_path: Path) -> dict[str, Any]:
    contract, contract_path = _sealed(contract_path, label="raw final-logit retention contract")
    _schema_status(contract, schema=retention_contract.SCHEMA, status=retention_contract.STATUS, label="raw final-logit retention contract")
    build, build_binding_path = _sealed(build_binding_path, label="raw final-logit retention build binding")
    _schema_status(build, schema=build_binding.SCHEMA, status=build_binding.STATUS, label="raw final-logit retention build binding")
    memory, memory_preflight_path = _sealed(memory_preflight_path, label="strict source-BF16 memory preflight")
    if memory.get("schema") != memory_preflight.SCHEMA or memory.get("status") not in {
        memory_preflight.READY_STATUS,
        memory_preflight.BLOCKED_STATUS,
    }:
        raise RawFinalLogitOuterControllerError("source-BF16 memory preflight schema/status drifted")
    _vector_rules(contract)
    _native_expected_hashes(contract)
    build_contract = _mapping(build.get("raw_final_logit_retention_contract"), label="build binding contract pointer")
    if Path(_text(build_contract.get("path"), label="build binding contract path")).resolve() != contract_path:
        raise RawFinalLogitOuterControllerError("build binding names a different raw retention contract")
    if build_contract.get("seal_sha256") != contract.get("seal_sha256"):
        raise RawFinalLogitOuterControllerError("build binding raw retention contract seal drifted")
    executable = _mapping(build.get("executor_binary"), label="raw retention executor binary")
    if _text(executable.get("sha256"), label="raw retention executor SHA", sha256=True) != PINNED_EXECUTOR_SHA256:
        raise RawFinalLogitOuterControllerError("raw retention executor SHA differs from the prepared 88286 binary")
    memory_pointer = _mapping(contract.get("strict_source_bf16_memory_preflight"), label="raw retention contract memory pointer")
    if Path(_text(memory_pointer.get("path"), label="raw retention memory preflight path")).resolve() != memory_preflight_path:
        raise RawFinalLogitOuterControllerError("raw retention contract names a different memory preflight")
    if memory_pointer.get("seal_sha256") != memory.get("seal_sha256"):
        raise RawFinalLogitOuterControllerError("raw retention contract memory preflight seal drifted")
    gate = _mapping(contract.get("source_memory_and_eviction_gate"), label="raw retention source memory gate")
    blocked = memory.get("status") == memory_preflight.BLOCKED_STATUS
    if gate.get("source_teacher_capture_is_currently_blocked") is not blocked:
        raise RawFinalLogitOuterControllerError("raw retention contract/source memory block state differs from current preflight")
    headroom = _mapping(memory.get("headroom_assessment"), label="source memory headroom")
    status = BLOCKED_STATUS if blocked else READY_STATUS
    controller_source = Path(__file__).resolve()
    return seal(
        {
            "schema": SCHEMA,
            "status": status,
            "recorded_at": _utc_now(),
            "raw_final_logit_retention_contract": _evidence(contract_path, contract),
            "executor_build_binding": _evidence(build_binding_path, build),
            "source_bf16_memory_preflight": _evidence(memory_preflight_path, memory),
            "controller_source": _evidence(controller_source),
            "executor": {
                "path": executable.get("path"),
                "sha256": executable.get("sha256"),
                "mode": retention_contract.NATIVE_MODE,
                "binary_must_be_rehashed_by_future_outer_launcher_before_child_start": True,
            },
            "six_vector_rules": _vector_rules(contract),
            "native_replay_rules": _native_expected_hashes(contract),
            "required_future_sequence": {
                "source_lease": {"schema": SOURCE_LEASE_SCHEMA, "status": SOURCE_LEASE_STATUS},
                "source_terminal": {"schema": SOURCE_TERMINAL_SCHEMA, "status": SOURCE_TERMINAL_STATUS},
                "source_eviction": {"schema": SOURCE_EVICTION_SCHEMA, "status": SOURCE_EVICTION_STATUS},
                "native_lease": {"schema": NATIVE_LEASE_SCHEMA, "status": NATIVE_LEASE_STATUS},
                "native_terminal": {"schema": NATIVE_TERMINAL_SCHEMA, "status": NATIVE_TERMINAL_STATUS},
                "source_then_evict_then_native_is_mandatory": True,
                "one_child_process_group_per_phase": True,
                "fresh_output_roots_and_receipt_last_terminal_writes": True,
                "automatic_retry_forbidden": True,
                "six_vector_metric_requires_immutable_two_source_plus_four_native_payloads": True,
            },
            "current_source_memory_refusal": {
                "blocked": blocked,
                "measured_reclaimable_bytes": headroom.get("measured_reclaimable_bytes"),
                "required_reclaimable_bytes": headroom.get("minimum_reclaimable_bytes_required_before_source_load"),
                "deficit_bytes": headroom.get("measured_reclaimable_deficit_bytes"),
                "swap_used_bytes": headroom.get("measured_swap_used_bytes"),
                "no_child_launch_performed": True,
                "no_model_gpu_or_endpoint_action_performed": True,
            },
            "claim_boundary": {
                "controller_preflight_only": True,
                "does_not_launch_or_prepare_a_child_process": True,
                "does_not_open_source_weight_payloads_or_load_any_model": True,
                "does_not_create_metal_or_mps_context": True,
                "does_not_touch_qwen30_server_watcher_adapter_hcli_or_tps": True,
                "does_not_emit_source_oracle_coherence_or_quality_result": True,
            },
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--build-binding", type=Path, default=DEFAULT_BUILD_BINDING)
    parser.add_argument("--memory-preflight", type=Path, default=DEFAULT_MEMORY_PREFLIGHT)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_preflight(
            contract_path=args.contract,
            build_binding_path=args.build_binding,
            memory_preflight_path=args.memory_preflight,
        )
        if args.output is None:
            output = DEFAULT_OUTPUT_ROOT / (
                "QWEN30_HQ30GR2_RAW_LOGIT_OUTER_PREFLIGHT_"
                f"{result['raw_final_logit_retention_contract']['seal_sha256'][:12]}_"
                f"{result['executor']['sha256'][:12]}_"
                f"{result['controller_source']['sha256'][:12]}.json"
            )
        else:
            output = args.output
        _write_new_json(output.resolve(), result)
    except RawFinalLogitOuterControllerError as exc:
        print(f"Q30 raw-final-logit outer controller refused: {exc}")
        return 2
    print(json.dumps({"output": str(output.resolve()), "status": result["status"], "seal_sha256": result["seal_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
