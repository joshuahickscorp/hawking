"""One-shot future strict-Metal capture for Qwen80's true L0 first residual.

This outer launcher is deliberately fail-closed.  It accepts a new, retained
source/direct-packed CPU baseline only as a parity authority; it cannot reuse
the older synthetic L0 mixer receipt.  A future child must prove that an
actual 2048-float device first-residual buffer was produced from the same
source-token input and zeroed DeltaNet state in the same command graph, then
the outer receipt becomes the only first-residual input accepted by the
all-ten true-input MoE launcher.

The module itself never opens Metal, starts a watcher/server, or claims a
complete Qwen80 layer, decoder, token, HCLI, TPS, TG, or tournament result.
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
SCHEMA = "hawking.ascension.qwen80_first_residual_outer_capture.v1"
CAPTURED_STATUS = "CAPTURED_QWEN80_FIRST_RESIDUAL_STRICT_MATH_COMPONENT_ONLY"
ACTIVE_FILENAME = "active.json"
CHILD_FILENAME = "child.json"
TERMINAL_FILENAME = "outer-terminal-receipt.json"
OUTER_STDOUT = "outer.stdout.log"
OUTER_STDERR = "outer.stderr.log"
INNER_CAPTURE = "inner"

EXPECTED_PROBE_BASENAME = "ascension_qwen80_first_residual_bridge_device"
CPU_BASELINE_SCHEMA = "hawking.ascension.qwen80_first_residual_bridge_inner.v1"
CPU_BASELINE_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_LAYER0_FIRST_RESIDUAL_CPU_ORACLE_BASELINE_"
    "METAL_LEASE_REQUIRED"
)
DEVICE_INNER_SCHEMA = "hawking.ascension.qwen80_first_residual_bridge_device.v1"
DEVICE_INNER_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_MIXER_FIRST_RESIDUAL_"
    "STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN"
)
MANIFEST_SCHEMA = "hawking.ascension.qwen80_complete_binary_gravity.v1"
ADMISSION_POINTER_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1"
ADMISSION_POINTER_STATUS = "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED"
ADMISSION_RECEIPT_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1"
ADMISSION_RECEIPT_STATUS = "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
LEASE_SCHEMA = "hawking.ascension.qwen80_first_residual_quiet_metal_lease.v1"
LEASE_STATUS = "GRANTED_QWEN80_FIRST_RESIDUAL_NON_TIMED_DEVICE_PARITY_LEASE"
LEASE_COMPONENT = "qwen80_first_residual_bridge"
HIDDEN = 2_048
FIRST_RESIDUAL_BYTES = HIDDEN * 4
CONV_STATE_ELEMENTS = 8_192 * 3
RECURRENT_STATE_ELEMENTS = 32 * 128 * 128


class FirstResidualBridgeLauncherError(RuntimeError):
    """The strict first-residual component capture cannot safely start."""


@dataclass(frozen=True)
class LaunchConfig:
    probe_bin: Path
    manifest: Path
    admission_current: Path
    cpu_baseline_receipt: Path
    lease_receipt: Path | None
    capture_dir: Path
    workers: int
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
    cpu_baseline: dict[str, Any]
    cpu_baseline_input: dict[str, Any]
    cpu_baseline_output: dict[str, Any]
    lease: dict[str, Any]
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
        raise FirstResidualBridgeLauncherError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FirstResidualBridgeLauncherError(f"{label} must be an object")
    return dict(value)


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    if not path.is_absolute():
        raise FirstResidualBridgeLauncherError(f"{label} must be absolute: {path}")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise FirstResidualBridgeLauncherError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise FirstResidualBridgeLauncherError(f"{label} must be a regular non-symlink file")
    if executable and not os.access(path, os.X_OK):
        raise FirstResidualBridgeLauncherError(f"{label} must be executable")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise FirstResidualBridgeLauncherError(f"cannot canonicalize {label}: {exc}") from exc


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
        raise FirstResidualBridgeLauncherError(f"cannot read {label}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise FirstResidualBridgeLauncherError(f"{label} must be a JSON object")
    return dict(document)


def _sealed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    document = _read_json(path, label)
    try:
        verify(document, label=str(path))
    except ValueError as exc:
        raise FirstResidualBridgeLauncherError(f"{label} is not sealed: {exc}") from exc
    return document, _require_sha256(document.get("seal_sha256"), f"{label}.seal_sha256")


def _evidence_matches(evidence: object, expected: Mapping[str, Any], label: str) -> None:
    value = _mapping(evidence, label)
    if value.get("present") is not True:
        raise FirstResidualBridgeLauncherError(f"{label} must attest a present file")
    observed = _canonical_regular(Path(str(value.get("path"))), f"{label}.path")
    if observed != Path(str(expected["path"])):
        raise FirstResidualBridgeLauncherError(f"{label} path drifted")
    if value.get("bytes") != expected["bytes"] or value.get("sha256") != expected["sha256"]:
        raise FirstResidualBridgeLauncherError(f"{label} byte/digest drifted")


def _atomic_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FirstResidualBridgeLauncherError(f"refusing to overwrite {path}")
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
        raise FirstResidualBridgeLauncherError(f"refusing to overwrite {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _bind_manifest(path: Path) -> tuple[dict[str, Any], str]:
    document, seal_sha256 = _sealed_json(path, "--manifest")
    if document.get("schema") != MANIFEST_SCHEMA:
        raise FirstResidualBridgeLauncherError("--manifest schema drifted")
    return _file_evidence(path, "--manifest"), seal_sha256


def _bind_admission(
    path: Path, manifest: Mapping[str, Any], manifest_seal: str
) -> tuple[dict[str, Any], str, str, str, str]:
    pointer, pointer_seal = _sealed_json(path, "--admission-current")
    if pointer.get("schema") != ADMISSION_POINTER_SCHEMA or pointer.get("status") != ADMISSION_POINTER_STATUS:
        raise FirstResidualBridgeLauncherError("admission pointer schema/status drifted")
    selected = _mapping(pointer.get("complete_manifest"), "admission complete_manifest")
    if _canonical_regular(Path(str(selected.get("path"))), "admission manifest path") != Path(str(manifest["path"])):
        raise FirstResidualBridgeLauncherError("admission selects another manifest")
    if selected.get("document_sha256") != manifest["sha256"] or selected.get("seal_sha256") != manifest_seal:
        raise FirstResidualBridgeLauncherError("admission manifest identity drifted")
    selected_receipt = _mapping(pointer.get("admission_receipt"), "admission receipt selection")
    receipt_path = _canonical_regular(Path(str(selected_receipt.get("path"))), "admission receipt path")
    receipt, receipt_seal = _sealed_json(receipt_path, "admission receipt")
    if (
        receipt.get("schema") != ADMISSION_RECEIPT_SCHEMA
        or receipt.get("status") != ADMISSION_RECEIPT_STATUS
        or selected_receipt.get("seal_sha256") != receipt_seal
    ):
        raise FirstResidualBridgeLauncherError("admission receipt schema/status/seal drifted")
    receipt_manifest = _mapping(receipt.get("complete_manifest"), "admission receipt manifest")
    if (
        receipt_manifest.get("document_sha256") != manifest["sha256"]
        or receipt_manifest.get("seal_sha256") != manifest_seal
        or _canonical_regular(Path(str(receipt_manifest.get("path"))), "admission receipt manifest path")
        != Path(str(manifest["path"]))
    ):
        raise FirstResidualBridgeLauncherError("admission receipt manifest authority drifted")
    revalidation = _mapping(receipt.get("current_source_revalidation"), "admission source revalidation")
    source_audit = _require_sha256(revalidation.get("source_audit_seal_sha256"), "source audit seal")
    revision = revalidation.get("revision")
    if not isinstance(revision, str) or not revision:
        raise FirstResidualBridgeLauncherError("admission source revision is missing")
    return _file_evidence(path, "--admission-current"), pointer_seal, receipt_seal, source_audit, revision


def _bind_cpu_baseline(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    source_audit_seal: str,
    source_revision: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    document = _read_json(path, "--cpu-baseline-receipt")
    evidence = _file_evidence(path, "--cpu-baseline-receipt")
    if (
        document.get("schema") != CPU_BASELINE_SCHEMA
        or document.get("status") != CPU_BASELINE_STATUS
        or document.get("mode") != "cpu-oracle"
        or document.get("metal_device_or_dispatch_performed") is not False
        or document.get("component_only") is not True
        or document.get("complete_layer_or_token_performed") is not False
    ):
        raise FirstResidualBridgeLauncherError("CPU baseline is not a retained non-device first-residual witness")
    artifact = _mapping(document.get("artifact_binding"), "CPU baseline artifact binding")
    if (
        _canonical_regular(Path(str(artifact.get("manifest_path"))), "CPU baseline manifest path")
        != Path(str(manifest["path"]))
        or artifact.get("manifest_document_sha256") != manifest["sha256"]
        or artifact.get("manifest_seal_sha256") != manifest_seal
        or artifact.get("source_audit_seal_sha256") != source_audit_seal
        or artifact.get("source_revision") != source_revision
        or artifact.get("layer") != 0
        or artifact.get("linear_state_slot") != 0
        or artifact.get("admission_scan_performed_once_before_catalog_reuse") is not True
        or artifact.get("direct_packed_payloads_only") is not True
    ):
        raise FirstResidualBridgeLauncherError("CPU baseline artifact authority/geometry drifted")
    input_provenance = _mapping(document.get("same_input_provenance"), "CPU baseline input provenance")
    input_evidence = _mapping(input_provenance.get("input_hidden"), "CPU baseline input hidden")
    input_path = _canonical_regular(Path(str(input_evidence.get("path"))), "CPU baseline input hidden path")
    actual_input = _file_evidence(input_path, "CPU baseline input hidden")
    _evidence_matches(input_evidence, actual_input, "CPU baseline input hidden")
    if (
        input_provenance.get("kind") != "source_direct_packed_embedding_with_zeroed_layer0_deltanet_state"
        or not isinstance(input_provenance.get("token_id"), int)
        or input_provenance.get("embedding_tensor") != "model.embed_tokens.weight"
        or input_provenance.get("input_hidden_f32le_sha256") != actual_input["sha256"]
        or actual_input["bytes"] != FIRST_RESIDUAL_BYTES
    ):
        raise FirstResidualBridgeLauncherError("CPU baseline same-input hidden provenance drifted")
    conv = _mapping(input_provenance.get("initial_conv_state"), "CPU baseline conv state")
    recurrent = _mapping(input_provenance.get("initial_recurrent_state"), "CPU baseline recurrent state")
    if (
        conv.get("elements") != CONV_STATE_ELEMENTS
        or conv.get("zero_initialized") is not True
        or not _is_sha256(conv.get("f32le_sha256"))
        or recurrent.get("elements") != RECURRENT_STATE_ELEMENTS
        or recurrent.get("zero_initialized") is not True
        or not _is_sha256(recurrent.get("f32le_sha256"))
        or input_provenance.get("future_strict_metal_child_must_retain_exact_input_and_state_identity") is not True
    ):
        raise FirstResidualBridgeLauncherError("CPU baseline state provenance drifted")
    output = _mapping(document.get("first_residual_output"), "CPU baseline first residual")
    output_file = _mapping(output.get("file"), "CPU baseline first residual file")
    output_path = _canonical_regular(Path(str(output_file.get("path"))), "CPU baseline first residual path")
    actual_output = _file_evidence(output_path, "CPU baseline first residual")
    _evidence_matches(output_file, actual_output, "CPU baseline first residual")
    if (
        output.get("layer") != 0
        or output.get("linear_state_slot") != 0
        or output.get("elements") != HIDDEN
        or output.get("bytes") != FIRST_RESIDUAL_BYTES
        or actual_output["bytes"] != FIRST_RESIDUAL_BYTES
        or output.get("sha256") != actual_output["sha256"]
        or output.get("f32le_sha256") != actual_output["sha256"]
        or output.get("same_command_graph_required_for_future_strict_metal_bridge") is not True
    ):
        raise FirstResidualBridgeLauncherError("CPU baseline first-residual buffer/hash drifted")
    durable = _mapping(document.get("durable_capture"), "CPU baseline durable capture")
    if (
        durable.get("input_hidden_written_before_receipt") is not True
        or durable.get("first_residual_written_before_receipt") is not True
        or durable.get("receipt_written_last_is_completion_marker") is not True
        or durable.get("outer_reaped_strict_metal_capture_required_before_any_device_or_layer_promotion") is not True
    ):
        raise FirstResidualBridgeLauncherError("CPU baseline lacks receipt-last device-promotion boundary")
    return evidence, input_provenance, output


def _bind_lease(
    path: Path | None,
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission_receipt_seal: str,
    cpu_baseline: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if path is None:
        raise FirstResidualBridgeLauncherError("--lease-receipt is required before any child starts")
    document, seal_sha256 = _sealed_json(path, "--lease-receipt")
    if document.get("schema") != LEASE_SCHEMA or document.get("status") != LEASE_STATUS:
        raise FirstResidualBridgeLauncherError("lease schema/status does not authorize this component")
    lifecycle = _mapping(document.get("lifecycle"), "lease lifecycle")
    if (
        lifecycle.get("fresh_for_this_exact_launch") is not True
        or lifecycle.get("automatic_retry_prohibited") is not True
        or lifecycle.get("outer_reaped_capture_required") is not True
    ):
        raise FirstResidualBridgeLauncherError("lease is not fresh/one-shot/outer-reaped")
    policy = _mapping(document.get("execution_policy"), "lease execution policy")
    if (
        policy.get("component") != LEASE_COMPONENT
        or policy.get("quiet_qwen80_device_lease") is not True
        or policy.get("strict_math") is not True
        or policy.get("timing_or_benchmarking_allowed") is not False
        or policy.get("complete_layer_or_token_allowed") is not False
        or policy.get("tps_or_tg_claim_allowed") is not False
    ):
        raise FirstResidualBridgeLauncherError("lease policy is not strict non-timed component-only")
    artifact = _mapping(document.get("artifact_binding"), "lease artifact binding")
    if (
        artifact.get("manifest_document_sha256") != manifest["sha256"]
        or artifact.get("manifest_seal_sha256") != manifest_seal
        or artifact.get("admission_receipt_seal_sha256") != admission_receipt_seal
    ):
        raise FirstResidualBridgeLauncherError("lease artifact authority drifted")
    baseline = _mapping(document.get("cpu_baseline_binding"), "lease CPU baseline binding")
    if (
        baseline.get("receipt_path") != cpu_baseline["path"]
        or baseline.get("receipt_document_sha256") != cpu_baseline["sha256"]
        or baseline.get("schema") != CPU_BASELINE_SCHEMA
        or baseline.get("status") != CPU_BASELINE_STATUS
    ):
        raise FirstResidualBridgeLauncherError("lease CPU baseline identity drifted")
    return _file_evidence(path, "--lease-receipt"), seal_sha256


def _validate_config(config: LaunchConfig) -> LaunchContext:
    probe = _canonical_regular(config.probe_bin, "--probe-bin", executable=True)
    if probe.name != EXPECTED_PROBE_BASENAME:
        raise FirstResidualBridgeLauncherError(
            f"--probe-bin must be {EXPECTED_PROBE_BASENAME}, got {probe.name!r}"
        )
    if config.workers < 1 or config.workers > 4:
        raise FirstResidualBridgeLauncherError("--workers must be 1..4")
    if not config.timeout_seconds > 0:
        raise FirstResidualBridgeLauncherError("--timeout-seconds must be positive")
    if not config.capture_dir.is_absolute() or not config.capture_dir.parent.is_dir():
        raise FirstResidualBridgeLauncherError("--capture-dir must be absolute with an existing parent")
    manifest, manifest_seal = _bind_manifest(config.manifest)
    admission, pointer_seal, receipt_seal, source_audit, source_revision = _bind_admission(
        config.admission_current, manifest, manifest_seal
    )
    baseline, baseline_input, baseline_output = _bind_cpu_baseline(
        config.cpu_baseline_receipt,
        manifest=manifest,
        manifest_seal=manifest_seal,
        source_audit_seal=source_audit,
        source_revision=source_revision,
    )
    lease, lease_seal = _bind_lease(
        config.lease_receipt,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission_receipt_seal=receipt_seal,
        cpu_baseline=baseline,
    )
    return LaunchContext(
        probe_binary=_file_evidence(probe, "--probe-bin", executable=True),
        manifest=manifest,
        manifest_seal_sha256=manifest_seal,
        admission_current=admission,
        admission_pointer_seal_sha256=pointer_seal,
        admission_receipt_seal_sha256=receipt_seal,
        source_audit_seal_sha256=source_audit,
        source_revision=source_revision,
        cpu_baseline=baseline,
        cpu_baseline_input=baseline_input,
        cpu_baseline_output=baseline_output,
        lease=lease,
        lease_seal_sha256=lease_seal,
    )


def _launch_identity(config: LaunchConfig, context: LaunchContext) -> str:
    payload = {
        "schema": SCHEMA,
        "probe": context.probe_binary,
        "manifest": context.manifest,
        "manifest_seal": context.manifest_seal_sha256,
        "admission": context.admission_current,
        "admission_pointer_seal": context.admission_pointer_seal_sha256,
        "admission_receipt_seal": context.admission_receipt_seal_sha256,
        "cpu_baseline": context.cpu_baseline,
        "cpu_input": context.cpu_baseline_input,
        "cpu_output": context.cpu_baseline_output,
        "lease": context.lease,
        "lease_seal": context.lease_seal_sha256,
        "workers": config.workers,
        "timeout_seconds": config.timeout_seconds,
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _child_command(config: LaunchConfig, inner_capture: Path) -> list[str]:
    assert config.lease_receipt is not None
    return [
        str(_canonical_regular(config.probe_bin, "--probe-bin", executable=True)),
        "--manifest", str(_canonical_regular(config.manifest, "--manifest")),
        "--admission-current", str(_canonical_regular(config.admission_current, "--admission-current")),
        "--cpu-baseline-receipt", str(_canonical_regular(config.cpu_baseline_receipt, "--cpu-baseline-receipt")),
        "--lease-receipt", str(_canonical_regular(config.lease_receipt, "--lease-receipt")),
        "--outer-capture-dir", str(config.capture_dir),
        "--capture-dir", str(inner_capture),
        "--mode", "metal",
        "--workers", str(config.workers),
    ]


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


def _terminal(returncode: int | None, *, timed_out: bool, spawn_error: str | None = None) -> dict[str, Any]:
    terminal: dict[str, Any] = {
        "reaped": returncode is not None,
        "timed_out": timed_out,
        "returncode": returncode,
        "exit_code": returncode if isinstance(returncode, int) and returncode >= 0 else None,
        "signal": -returncode if isinstance(returncode, int) and returncode < 0 else None,
    }
    if spawn_error is not None:
        terminal["spawn_error"] = spawn_error
        terminal["reaped"] = False
    return terminal


def _validate_inner(receipt: Mapping[str, Any], context: LaunchContext) -> dict[str, Any]:
    if (
        receipt.get("schema") != DEVICE_INNER_SCHEMA
        or receipt.get("status") != DEVICE_INNER_STATUS
        or receipt.get("mode") != "metal"
        or receipt.get("metal_device_or_dispatch_performed") is not True
        or receipt.get("component_only") is not True
        or receipt.get("complete_layer_or_token_performed") is not False
        or receipt.get("synthetic_input") is not False
        or receipt.get("fixture_only") is not False
    ):
        raise FirstResidualBridgeLauncherError("inner schema/status/scope boundary drifted")
    durable = _mapping(receipt.get("durable_capture"), "inner durable capture")
    if (
        durable.get("receipt_written_last_is_completion_marker") is not True
        or durable.get("outer_reaped_capture_required") is not True
        or durable.get("replay_guarded") is not True
    ):
        raise FirstResidualBridgeLauncherError("inner lacks receipt-last/replay guard")
    artifact = _mapping(receipt.get("artifact_binding"), "inner artifact binding")
    if (
        artifact.get("manifest_document_sha256") != context.manifest["sha256"]
        or artifact.get("manifest_seal_sha256") != context.manifest_seal_sha256
        or artifact.get("admission_pointer_seal_sha256") != context.admission_pointer_seal_sha256
        or artifact.get("admission_receipt_seal_sha256") != context.admission_receipt_seal_sha256
        or artifact.get("source_audit_seal_sha256") != context.source_audit_seal_sha256
        or artifact.get("source_revision") != context.source_revision
        or artifact.get("layer") != 0
        or artifact.get("linear_state_slot") != 0
    ):
        raise FirstResidualBridgeLauncherError("inner artifact authority/geometry drifted")
    baseline = _mapping(receipt.get("cpu_baseline_binding"), "inner CPU baseline binding")
    if (
        baseline.get("receipt_path") != context.cpu_baseline["path"]
        or baseline.get("receipt_document_sha256") != context.cpu_baseline["sha256"]
        or baseline.get("schema") != CPU_BASELINE_SCHEMA
        or baseline.get("status") != CPU_BASELINE_STATUS
    ):
        raise FirstResidualBridgeLauncherError("inner CPU baseline identity drifted")
    provenance = _mapping(receipt.get("same_input_provenance"), "inner input provenance")
    source_provenance = context.cpu_baseline_input
    for key in ("kind", "token_id", "embedding_tensor", "input_hidden_f32le_sha256"):
        if provenance.get(key) != source_provenance.get(key):
            raise FirstResidualBridgeLauncherError(f"inner same-input {key} drifted")
    for state_name in ("initial_conv_state", "initial_recurrent_state"):
        observed = _mapping(provenance.get(state_name), f"inner {state_name}")
        expected = _mapping(source_provenance.get(state_name), f"CPU {state_name}")
        if observed != expected:
            raise FirstResidualBridgeLauncherError(f"inner {state_name} identity drifted")
    output = _mapping(receipt.get("first_residual_output"), "inner first residual")
    expected_output = context.cpu_baseline_output
    if (
        output.get("layer") != 0
        or output.get("linear_state_slot") != 0
        or output.get("elements") != HIDDEN
        or output.get("bytes") != FIRST_RESIDUAL_BYTES
        or not _is_sha256(output.get("sha256"))
        or output.get("cpu_reference_sha256") != expected_output.get("sha256")
        or output.get("all_finite") is not True
    ):
        raise FirstResidualBridgeLauncherError("inner first-residual output/CPU authority drifted")
    graph = _mapping(receipt.get("same_command_graph"), "inner same command graph")
    if (
        graph.get("same_command_graph_required") is not True
        or graph.get("same_command_graph_retained") is not True
        or not isinstance(graph.get("command_buffer_identity"), str)
        or not graph["command_buffer_identity"]
        or graph.get("device_first_residual_buffer_bytes") != FIRST_RESIDUAL_BYTES
        or graph.get("input_then_deltanet_then_first_residual_then_fence_order") is not True
        or graph.get("prefix_dispatches") != 9
        or graph.get("total_dispatches") != 9
        or graph.get("prefix_only") is not True
        or graph.get("suffix_dispatches") != 0
        or graph.get("no_true_moe_suffix_encoded") is not True
    ):
        raise FirstResidualBridgeLauncherError("inner did not retain a same-graph prefix-only device first-residual buffer")
    parity = _mapping(receipt.get("cpu_device_parity"), "inner CPU/device parity")
    if (
        parity.get("checked_elements") != HIDDEN
        or parity.get("passed") is not True
        or not isinstance(parity.get("max_abs_error"), (int, float))
        or not isinstance(parity.get("tolerance"), (int, float))
        or float(parity["max_abs_error"]) > float(parity["tolerance"])
    ):
        raise FirstResidualBridgeLauncherError("inner first-residual CPU/device parity failed")
    state = _mapping(receipt.get("state_witness"), "inner state witness")
    if (
        state.get("linear_state_slot") != 0
        or state.get("conv_state_elements") != CONV_STATE_ELEMENTS
        or state.get("recurrent_state_elements") != RECURRENT_STATE_ELEMENTS
        or state.get("state_commit_after_parity_fence") is not True
    ):
        raise FirstResidualBridgeLauncherError("inner DeltaNet state witness drifted")
    policy = _mapping(receipt.get("metal_execution_policy"), "inner execution policy")
    lease = _mapping(policy.get("lease_binding"), "inner lease binding")
    if (
        policy.get("strict_math_required") is not True
        or policy.get("timing_or_benchmarking_allowed") is not False
        or policy.get("complete_layer_or_token_allowed") is not False
        or policy.get("tps_or_tg_claim_allowed") is not False
        or lease.get("receipt_path") != context.lease["path"]
        or lease.get("receipt_document_sha256") != context.lease["sha256"]
        or lease.get("seal_sha256") != context.lease_seal_sha256
        or lease.get("schema") != LEASE_SCHEMA
        or lease.get("status") != LEASE_STATUS
    ):
        raise FirstResidualBridgeLauncherError("inner strict component lease drifted")
    return output


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
        receipt = _read_json(receipt_path, "inner receipt")
        result.update(_file_evidence(receipt_path, "inner receipt"))
        result["schema"] = receipt.get("schema")
        result["status"] = receipt.get("status")
        result["first_residual_output"] = _validate_inner(receipt, context)
    except FirstResidualBridgeLauncherError as exc:
        result["binding_valid"] = False
        result["binding_error"] = str(exc)
    else:
        result["binding_valid"] = True
    return result


def _terminal_status(terminal: Mapping[str, Any], inner: Mapping[str, Any]) -> str:
    if terminal.get("spawn_error"):
        return "REFUSED_QWEN80_FIRST_RESIDUAL_OUTER_CHILD_SPAWN_ERROR"
    if terminal.get("timed_out"):
        return "REFUSED_QWEN80_FIRST_RESIDUAL_OUTER_CHILD_TIMEOUT"
    if terminal.get("signal") is not None:
        return "REFUSED_QWEN80_FIRST_RESIDUAL_OUTER_CHILD_SIGNAL"
    if terminal.get("exit_code") != 0:
        return "REFUSED_QWEN80_FIRST_RESIDUAL_OUTER_CHILD_NONZERO"
    if inner.get("binding_valid") is not True:
        return "REFUSED_QWEN80_FIRST_RESIDUAL_OUTER_ZERO_EXIT_WITHOUT_STRICT_INNER_RECEIPT"
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
    output = inner.get("first_residual_output") if inner.get("binding_valid") is True else None
    source_binding: dict[str, Any] = {
        "manifest": context.manifest,
        "manifest_seal_sha256": context.manifest_seal_sha256,
        "admission_current": context.admission_current,
        "admission_pointer_seal_sha256": context.admission_pointer_seal_sha256,
        "admission_receipt_seal_sha256": context.admission_receipt_seal_sha256,
        "source_audit_seal_sha256": context.source_audit_seal_sha256,
        "source_revision": context.source_revision,
        "cpu_baseline_receipt": context.cpu_baseline,
        "lease_receipt": context.lease,
        "lease_receipt_seal_sha256": context.lease_seal_sha256,
        "workers": config.workers,
    }
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
        "source_binding": source_binding,
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
            "strict_metal_first_residual_component_only": True,
            "requires_same_input_state_command_graph_and_parity": True,
            "not_a_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_result": True,
            "watcher_or_server_transition_not_authorized": True,
        },
    }
    if isinstance(output, Mapping):
        payload["first_residual_output"] = {
            "layer": output["layer"],
            "linear_state_slot": output["linear_state_slot"],
            "elements": output["elements"],
            "same_command_graph_required": True,
            "sha256": output["sha256"],
        }
    if capture_error is not None:
        payload["capture_error"] = capture_error
    return seal(payload)


def _replay(config: LaunchConfig, identity: str) -> dict[str, Any]:
    terminal_path = config.capture_dir / TERMINAL_FILENAME
    if not terminal_path.is_file():
        raise FirstResidualBridgeLauncherError("capture directory exists without terminal receipt")
    receipt, _ = _sealed_json(terminal_path, "outer terminal receipt")
    if receipt.get("schema") != SCHEMA or receipt.get("launch_identity_sha256") != identity:
        raise FirstResidualBridgeLauncherError("capture directory belongs to another launch identity")
    return receipt


def run_attempt(config: LaunchConfig) -> dict[str, Any]:
    """Launch exactly one future device child, or replay its terminal evidence."""

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
        seal({
            "schema": SCHEMA,
            "status": "STARTED_QWEN80_FIRST_RESIDUAL_OUTER_ONE_SHOT",
            "recorded_at": started_at,
            "launch_identity_sha256": identity,
            "command": command,
            "claim_boundary": {"component_only": True, "automatic_retry_disabled": True},
        }),
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
                    seal({
                        "schema": SCHEMA,
                        "status": "RUNNING_QWEN80_FIRST_RESIDUAL_OUTER_ONE_SHOT",
                        "recorded_at": _utc_now(),
                        "launch_identity_sha256": identity,
                        "pid": child_pid,
                        "parent_pid": os.getpid(),
                        "command": command,
                        "mode": "metal",
                        "strict_component_lease_required": True,
                    }),
                )
            except FirstResidualBridgeLauncherError as exc:
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
    parser.add_argument("--cpu-baseline-receipt", type=Path, required=True)
    parser.add_argument("--lease-receipt", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(argv)
    config = LaunchConfig(
        probe_bin=parsed.probe_bin,
        manifest=parsed.manifest,
        admission_current=parsed.admission_current,
        cpu_baseline_receipt=parsed.cpu_baseline_receipt,
        lease_receipt=parsed.lease_receipt,
        capture_dir=parsed.capture_dir,
        workers=parsed.workers,
        timeout_seconds=parsed.timeout_seconds,
    )
    try:
        receipt = run_attempt(config)
    except FirstResidualBridgeLauncherError as exc:
        print(json.dumps({"status": "REFUSED_QWEN80_FIRST_RESIDUAL_OUTER_LAUNCHER_ERROR", "error": str(exc)}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == CAPTURED_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
