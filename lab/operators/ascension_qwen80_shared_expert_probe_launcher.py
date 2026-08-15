"""One-shot outer capture for a future Qwen80 layer-0 shared-expert device probe.

The existing shared-expert result is a CPU-only, unsigned inner receipt.  This
launcher intentionally refuses to consume that receipt directly: a later
operator must first publish a separately sealed baseline wrapper which binds
its exact bytes and current admitted artifact inputs.  A future strict-Metal
child may then be launched once under a component-specific quiet lease.

The launcher owns only durable outer terminal evidence and process lifecycle.
It is not a scheduler, a registry authorizer, a retry loop, or proof of a
complete MoE/layer/token/decoder/HCLI/TPS/TG/tournament result.  Creating this
file does not invoke Metal or change the current Qwen80 runtime.
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
SCHEMA = "hawking.ascension.qwen80_shared_expert_outer_launcher.v1"
ACTIVE_FILENAME = "active.json"
CHILD_FILENAME = "child.json"
TERMINAL_FILENAME = "outer-terminal-receipt.json"
OUTER_STDOUT = "outer.stdout.log"
OUTER_STDERR = "outer.stderr.log"
INNER_CAPTURE = "inner"

EXPECTED_PROBE_BASENAME = "ascension_qwen80_direct_packed_shared_expert_wave"
EXPECTED_INNER_SCHEMA = "hawking.ascension.qwen80_direct_packed_shared_expert_wave.v1"
EXPECTED_CPU_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_SHARED_EXPERT_"
    "CPU_ORACLE_READY_METAL_LEASE_REQUIRED"
)
EXPECTED_METAL_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_SHARED_EXPERT_"
    "STRICT_MATH_METAL_COMPONENT_NOT_ROUTED_MOE_OR_LAYER"
)

CURRENT_ADMISSION_SCHEMA = (
    "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1"
)
CURRENT_ADMISSION_STATUS = "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED"

# This wrapper does not exist yet.  It is deliberately a separate, sealed
# document because the already-earned CPU receipt is durable but unsigned.
CPU_BASELINE_WRAPPER_SCHEMA = (
    "hawking.ascension.qwen80_shared_expert_cpu_baseline_wrapper.v1"
)
CPU_BASELINE_WRAPPER_STATUS = (
    "SEALED_CURRENT_ADMITTED_QWEN80_SHARED_EXPERT_CPU_ORACLE_BASELINE"
)

SHARED_EXPERT_LEASE_SCHEMA = "hawking.ascension.qwen80_quiet_metal_lease.v1"
SHARED_EXPERT_LEASE_STATUS = (
    "GRANTED_QWEN80_SHARED_EXPERT_NON_TIMED_DEVICE_PARITY_LEASE"
)
SHARED_EXPERT_LEASE_COMPONENT = "qwen80_direct_packed_shared_expert_wave"


class SharedExpertProbeLauncherError(RuntimeError):
    """The one-shot shared-expert outer capture cannot safely continue."""


@dataclass(frozen=True)
class LaunchConfig:
    """Inputs for one future strict-Metal shared-expert component attempt."""

    probe_bin: Path
    manifest: Path
    admission_current: Path
    cpu_baseline_receipt: Path
    lease_receipt: Path | None
    capture_dir: Path
    mode: str
    workers: int
    timeout_seconds: float


@dataclass(frozen=True)
class LaunchContext:
    """Immutable file evidence collected before a child can be started."""

    probe_binary: dict[str, Any]
    manifest: dict[str, Any]
    admission_current: dict[str, Any]
    admission_pointer_seal_sha256: str
    admission_receipt_seal_sha256: str
    cpu_baseline_receipt: dict[str, Any]
    cpu_baseline_seal_sha256: str
    cpu_inner_receipt: dict[str, Any]
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
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _require_absolute(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise SharedExpertProbeLauncherError(f"{label} must be absolute: {path}")


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    _require_absolute(path, label)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SharedExpertProbeLauncherError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SharedExpertProbeLauncherError(f"{label} must be a regular non-symlink file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise SharedExpertProbeLauncherError(f"{label} must be executable: {path}")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise SharedExpertProbeLauncherError(f"cannot canonicalize {label} {path}: {exc}") from exc


def _canonical_from_document(value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise SharedExpertProbeLauncherError(f"{label} must be an absolute path string")
    return _canonical_regular(Path(value), label)


def _file_evidence(path: Path, label: str) -> dict[str, Any]:
    canonical = _canonical_regular(path, label)
    return {
        "path": str(canonical),
        "present": True,
        "bytes": canonical.stat().st_size,
        "sha256": _file_sha256(canonical),
    }


def _atomic_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one new receipt without replacing terminal evidence."""

    if path.exists():
        raise SharedExpertProbeLauncherError(f"refusing to overwrite {path}")
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
        raise SharedExpertProbeLauncherError(f"refusing to overwrite {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SharedExpertProbeLauncherError(f"cannot read JSON {label} at {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SharedExpertProbeLauncherError(f"JSON {label} at {path} is not an object")
    return dict(payload)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SharedExpertProbeLauncherError(f"{label} must be an object")
    return dict(value)


def _sealed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    document = _read_json(path, label)
    try:
        verify(document, label=str(path))
    except ValueError as exc:
        raise SharedExpertProbeLauncherError(f"{label} is not a valid sealed receipt: {exc}") from exc
    seal_sha256 = document.get("seal_sha256")
    if not _is_sha256(seal_sha256):
        raise SharedExpertProbeLauncherError(f"{label} has no lowercase SHA-256 seal")
    return document, str(seal_sha256)


def _assert_evidence_matches_file(
    evidence: object, expected: Mapping[str, Any], label: str, *, require_digest: bool = True
) -> None:
    row = _mapping(evidence, label)
    if row.get("present") is not True:
        raise SharedExpertProbeLauncherError(f"{label} does not attest a present file")
    observed_path = _canonical_from_document(row.get("path"), f"{label}.path")
    if observed_path != Path(str(expected["path"])):
        raise SharedExpertProbeLauncherError(
            f"{label} path drifted: expected {expected['path']}, got {observed_path}"
        )
    if row.get("bytes") != expected["bytes"]:
        raise SharedExpertProbeLauncherError(f"{label} byte count drifted")
    if require_digest and row.get("sha256") != expected["sha256"]:
        raise SharedExpertProbeLauncherError(f"{label} SHA-256 drifted")


def _bind_current_admission(
    admission_path: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], str, str]:
    document, pointer_seal = _sealed_json(admission_path, "--admission-current")
    if (
        document.get("schema") != CURRENT_ADMISSION_SCHEMA
        or document.get("status") != CURRENT_ADMISSION_STATUS
    ):
        raise SharedExpertProbeLauncherError("--admission-current schema/status drifted")
    complete_manifest = _mapping(document.get("complete_manifest"), "admission complete_manifest")
    if _canonical_from_document(
        complete_manifest.get("path"), "admission complete_manifest.path"
    ) != Path(str(manifest["path"])):
        raise SharedExpertProbeLauncherError("--admission-current does not select --manifest")
    if complete_manifest.get("document_sha256") != manifest["sha256"]:
        raise SharedExpertProbeLauncherError("--admission-current manifest document SHA-256 drifted")
    admission_receipt = _mapping(document.get("admission_receipt"), "admission admission_receipt")
    admission_receipt_seal = admission_receipt.get("seal_sha256")
    if not _is_sha256(admission_receipt_seal):
        raise SharedExpertProbeLauncherError("admission receipt lacks a lowercase SHA-256 seal")
    return _file_evidence(admission_path, "--admission-current"), pointer_seal, str(
        admission_receipt_seal
    )


def _bind_cpu_baseline(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    admission_current: Mapping[str, Any],
    admission_receipt_seal: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Require a sealed wrapper around the pre-existing unsigned CPU receipt."""

    wrapper, wrapper_seal = _sealed_json(path, "--cpu-baseline-receipt")
    if (
        wrapper.get("schema") != CPU_BASELINE_WRAPPER_SCHEMA
        or wrapper.get("status") != CPU_BASELINE_WRAPPER_STATUS
    ):
        raise SharedExpertProbeLauncherError("--cpu-baseline-receipt schema/status drifted")
    source_binding = _mapping(wrapper.get("source_binding"), "CPU baseline source_binding")
    _assert_evidence_matches_file(
        source_binding.get("manifest"), manifest, "CPU baseline manifest evidence"
    )
    # A current-pointer document can be refreshed while retaining the same
    # immutable admitted receipt.  The baseline must retain the same path and
    # stable admission receipt seal, while this launch binds its exact pointer
    # seal separately in the child contract.
    historical_admission = _mapping(
        source_binding.get("admission_current"), "CPU baseline admission evidence"
    )
    if historical_admission.get("present") is not True:
        raise SharedExpertProbeLauncherError("CPU baseline did not attest an admission pointer")
    if _canonical_from_document(
        historical_admission.get("path"), "CPU baseline admission evidence.path"
    ) != Path(str(admission_current["path"])):
        raise SharedExpertProbeLauncherError("CPU baseline admission pointer path drifted")
    if not _is_sha256(historical_admission.get("sha256")):
        raise SharedExpertProbeLauncherError("CPU baseline admission evidence lacks SHA-256")
    if source_binding.get("admission_receipt_seal_sha256") != admission_receipt_seal:
        raise SharedExpertProbeLauncherError("CPU baseline admission receipt seal drifted")

    inner_binding = _mapping(wrapper.get("cpu_inner_receipt"), "CPU baseline cpu_inner_receipt")
    inner_path = _canonical_from_document(inner_binding.get("path"), "CPU baseline inner path")
    inner_evidence = _file_evidence(inner_path, "CPU baseline inner receipt")
    _assert_evidence_matches_file(
        inner_binding, inner_evidence, "CPU baseline inner receipt evidence"
    )
    inner = _read_json(inner_path, "CPU baseline inner receipt")
    if (
        inner.get("schema") != EXPECTED_INNER_SCHEMA
        or inner.get("status") != EXPECTED_CPU_STATUS
        or inner.get("mode") != "cpu-oracle"
        or inner.get("metal_device_or_dispatch_performed") is not False
        or inner.get("shared_expert_only") is not True
    ):
        raise SharedExpertProbeLauncherError("CPU baseline inner receipt is not the required CPU-only component")
    durable_capture = _mapping(inner.get("durable_capture"), "CPU baseline inner durable_capture")
    if durable_capture.get("receipt_written_last_is_completion_marker") is not True:
        raise SharedExpertProbeLauncherError("CPU baseline inner receipt lacks receipt-last capture")
    artifact_binding = _mapping(inner.get("artifact_binding"), "CPU baseline inner artifact_binding")
    if _canonical_from_document(
        artifact_binding.get("manifest_path"), "CPU baseline inner manifest_path"
    ) != Path(str(manifest["path"])):
        raise SharedExpertProbeLauncherError("CPU baseline inner manifest path drifted")
    if artifact_binding.get("manifest_document_sha256") != manifest["sha256"]:
        raise SharedExpertProbeLauncherError("CPU baseline inner manifest SHA-256 drifted")
    if _canonical_from_document(
        artifact_binding.get("admission_current_path"), "CPU baseline inner admission_current_path"
    ) != Path(str(admission_current["path"])):
        raise SharedExpertProbeLauncherError("CPU baseline inner admission pointer path drifted")
    if artifact_binding.get("admission_receipt_seal_sha256") != admission_receipt_seal:
        raise SharedExpertProbeLauncherError("CPU baseline inner admission receipt seal drifted")
    return _file_evidence(path, "--cpu-baseline-receipt"), wrapper_seal, inner_evidence


def _bind_lease(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    admission_receipt_seal: str,
    cpu_baseline_receipt: Mapping[str, Any],
    cpu_baseline_seal: str,
) -> tuple[dict[str, Any], str]:
    document, lease_seal = _sealed_json(path, "--lease-receipt")
    if (
        document.get("schema") != SHARED_EXPERT_LEASE_SCHEMA
        or document.get("status") != SHARED_EXPERT_LEASE_STATUS
    ):
        raise SharedExpertProbeLauncherError("--lease-receipt schema/status does not authorize shared expert")
    policy = _mapping(document.get("execution_policy"), "shared-expert lease execution_policy")
    if (
        policy.get("component") != SHARED_EXPERT_LEASE_COMPONENT
        or policy.get("quiet_qwen80_device_lease") is not True
        or policy.get("strict_math") is not True
        or policy.get("timing_or_benchmarking_allowed") is not False
        or policy.get("complete_layer_or_token_allowed") is not False
        or policy.get("tps_or_tg_claim_allowed") is not False
    ):
        raise SharedExpertProbeLauncherError("--lease-receipt policy is not the strict non-timed shared-expert lease")
    artifact_binding = _mapping(document.get("artifact_binding"), "shared-expert lease artifact_binding")
    if artifact_binding.get("manifest_document_sha256") != manifest["sha256"]:
        raise SharedExpertProbeLauncherError("shared-expert lease manifest SHA-256 drifted")
    if artifact_binding.get("admission_receipt_seal_sha256") != admission_receipt_seal:
        raise SharedExpertProbeLauncherError("shared-expert lease admission receipt seal drifted")
    baseline_binding = _mapping(document.get("cpu_baseline_binding"), "shared-expert lease CPU baseline")
    if _canonical_from_document(
        baseline_binding.get("receipt_path"), "shared-expert lease CPU baseline receipt_path"
    ) != Path(str(cpu_baseline_receipt["path"])):
        raise SharedExpertProbeLauncherError("shared-expert lease CPU baseline path drifted")
    if baseline_binding.get("receipt_document_sha256") != cpu_baseline_receipt["sha256"]:
        raise SharedExpertProbeLauncherError("shared-expert lease CPU baseline SHA-256 drifted")
    if (
        baseline_binding.get("schema") != CPU_BASELINE_WRAPPER_SCHEMA
        or baseline_binding.get("status") != CPU_BASELINE_WRAPPER_STATUS
        or baseline_binding.get("seal_sha256") != cpu_baseline_seal
    ):
        raise SharedExpertProbeLauncherError("shared-expert lease CPU baseline identity drifted")
    return _file_evidence(path, "--lease-receipt"), lease_seal


def _validate_config(config: LaunchConfig) -> LaunchContext:
    probe = _canonical_regular(config.probe_bin, "--probe-bin", executable=True)
    if probe.name != EXPECTED_PROBE_BASENAME:
        raise SharedExpertProbeLauncherError(
            f"--probe-bin must name {EXPECTED_PROBE_BASENAME}, got {probe.name!r}"
        )
    for path, label in (
        (config.manifest, "--manifest"),
        (config.admission_current, "--admission-current"),
        (config.cpu_baseline_receipt, "--cpu-baseline-receipt"),
    ):
        _canonical_regular(path, label)
    _require_absolute(config.capture_dir, "--capture-dir")
    if config.mode != "metal":
        raise SharedExpertProbeLauncherError(
            "shared-expert outer launcher is metal-only; it never reruns the sealed CPU baseline"
        )
    if config.workers < 1:
        raise SharedExpertProbeLauncherError("--workers must be positive")
    if not config.timeout_seconds > 0:
        raise SharedExpertProbeLauncherError("--timeout-seconds must be positive")
    if config.lease_receipt is None:
        raise SharedExpertProbeLauncherError("--mode metal requires --lease-receipt")
    _canonical_regular(config.lease_receipt, "--lease-receipt")

    probe_evidence = _file_evidence(probe, "--probe-bin")
    manifest = _file_evidence(config.manifest, "--manifest")
    admission_current, pointer_seal, admission_receipt_seal = _bind_current_admission(
        config.admission_current, manifest
    )
    cpu_baseline, cpu_baseline_seal, cpu_inner = _bind_cpu_baseline(
        config.cpu_baseline_receipt,
        manifest=manifest,
        admission_current=admission_current,
        admission_receipt_seal=admission_receipt_seal,
    )
    lease, lease_seal = _bind_lease(
        config.lease_receipt,
        manifest=manifest,
        admission_receipt_seal=admission_receipt_seal,
        cpu_baseline_receipt=cpu_baseline,
        cpu_baseline_seal=cpu_baseline_seal,
    )
    return LaunchContext(
        probe_binary=probe_evidence,
        manifest=manifest,
        admission_current=admission_current,
        admission_pointer_seal_sha256=pointer_seal,
        admission_receipt_seal_sha256=admission_receipt_seal,
        cpu_baseline_receipt=cpu_baseline,
        cpu_baseline_seal_sha256=cpu_baseline_seal,
        cpu_inner_receipt=cpu_inner,
        lease_receipt=lease,
        lease_seal_sha256=lease_seal,
    )


def _launch_identity(config: LaunchConfig, context: LaunchContext) -> str:
    payload = {
        "probe_binary": context.probe_binary,
        "manifest": context.manifest,
        "admission_current": context.admission_current,
        "admission_pointer_seal_sha256": context.admission_pointer_seal_sha256,
        "admission_receipt_seal_sha256": context.admission_receipt_seal_sha256,
        "cpu_baseline_receipt": context.cpu_baseline_receipt,
        "cpu_baseline_seal_sha256": context.cpu_baseline_seal_sha256,
        "cpu_inner_receipt": context.cpu_inner_receipt,
        "lease_receipt": context.lease_receipt,
        "lease_seal_sha256": context.lease_seal_sha256,
        "mode": config.mode,
        "workers": config.workers,
        "timeout_seconds": config.timeout_seconds,
    }
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def _child_command(config: LaunchConfig, inner_capture: Path) -> list[str]:
    """The child ABI is intentionally future-only and cannot reuse old CPU code."""

    assert config.lease_receipt is not None
    return [
        str(_canonical_regular(config.probe_bin, "--probe-bin", executable=True)),
        "--manifest",
        str(_canonical_regular(config.manifest, "--manifest")),
        "--admission-current",
        str(_canonical_regular(config.admission_current, "--admission-current")),
        "--cpu-baseline-receipt",
        str(_canonical_regular(config.cpu_baseline_receipt, "--cpu-baseline-receipt")),
        "--lease-receipt",
        str(_canonical_regular(config.lease_receipt, "--lease-receipt")),
        "--capture-dir",
        str(inner_capture),
        "--mode",
        config.mode,
        "--workers",
        str(config.workers),
    ]


def _sync_evidence(path: Path) -> dict[str, Any]:
    with path.open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    return _file_evidence(path, f"outer stream {path.name}")


def _validate_inner_binding(
    receipt: Mapping[str, Any], config: LaunchConfig, context: LaunchContext
) -> None:
    if (
        receipt.get("schema") != EXPECTED_INNER_SCHEMA
        or receipt.get("status") != EXPECTED_METAL_STATUS
        or receipt.get("mode") != config.mode
        or receipt.get("metal_device_or_dispatch_performed") is not True
        or receipt.get("shared_expert_only") is not True
        or receipt.get("routed_expert_sum_performed") is not False
        or receipt.get("moe_combine_performed") is not False
        or receipt.get("second_residual_performed") is not False
    ):
        raise SharedExpertProbeLauncherError("inner shared-expert schema/status/scope boundary drifted")
    durable_capture = _mapping(receipt.get("durable_capture"), "inner durable_capture")
    if durable_capture.get("receipt_written_last_is_completion_marker") is not True:
        raise SharedExpertProbeLauncherError("inner receipt does not attest receipt-last capture")
    artifact_binding = _mapping(receipt.get("artifact_binding"), "inner artifact_binding")
    if _canonical_from_document(
        artifact_binding.get("manifest_path"), "inner artifact_binding.manifest_path"
    ) != Path(str(context.manifest["path"])):
        raise SharedExpertProbeLauncherError("inner manifest path drifted")
    if artifact_binding.get("manifest_document_sha256") != context.manifest["sha256"]:
        raise SharedExpertProbeLauncherError("inner manifest SHA-256 drifted")
    if _canonical_from_document(
        artifact_binding.get("admission_current_path"), "inner artifact_binding.admission_current_path"
    ) != Path(str(context.admission_current["path"])):
        raise SharedExpertProbeLauncherError("inner admission pointer path drifted")
    if artifact_binding.get("admission_pointer_seal_sha256") != context.admission_pointer_seal_sha256:
        raise SharedExpertProbeLauncherError("inner admission pointer seal drifted during launch")
    if artifact_binding.get("admission_receipt_seal_sha256") != context.admission_receipt_seal_sha256:
        raise SharedExpertProbeLauncherError("inner admission receipt seal drifted")

    baseline = _mapping(receipt.get("cpu_baseline_binding"), "inner cpu_baseline_binding")
    if _canonical_from_document(
        baseline.get("receipt_path"), "inner cpu_baseline_binding.receipt_path"
    ) != Path(str(context.cpu_baseline_receipt["path"])):
        raise SharedExpertProbeLauncherError("inner CPU baseline path drifted")
    if baseline.get("receipt_document_sha256") != context.cpu_baseline_receipt["sha256"]:
        raise SharedExpertProbeLauncherError("inner CPU baseline SHA-256 drifted")
    if (
        baseline.get("schema") != CPU_BASELINE_WRAPPER_SCHEMA
        or baseline.get("status") != CPU_BASELINE_WRAPPER_STATUS
        or baseline.get("seal_sha256") != context.cpu_baseline_seal_sha256
    ):
        raise SharedExpertProbeLauncherError("inner CPU baseline identity drifted")

    policy = _mapping(receipt.get("metal_execution_policy"), "inner metal_execution_policy")
    if (
        policy.get("strict_math_required") is not True
        or policy.get("timing_or_benchmarking_allowed") is not False
        or policy.get("complete_layer_or_token_allowed") is not False
        or policy.get("tps_or_tg_claim_allowed") is not False
    ):
        raise SharedExpertProbeLauncherError("inner Metal policy drifted")
    lease = _mapping(policy.get("lease_binding"), "inner metal_execution_policy.lease_binding")
    if _canonical_from_document(
        lease.get("receipt_path"), "inner lease_binding.receipt_path"
    ) != Path(str(context.lease_receipt["path"])):
        raise SharedExpertProbeLauncherError("inner lease receipt path drifted")
    if lease.get("receipt_document_sha256") != context.lease_receipt["sha256"]:
        raise SharedExpertProbeLauncherError("inner lease receipt SHA-256 drifted")
    if (
        lease.get("schema") != SHARED_EXPERT_LEASE_SCHEMA
        or lease.get("status") != SHARED_EXPERT_LEASE_STATUS
        or lease.get("seal_sha256") != context.lease_seal_sha256
    ):
        raise SharedExpertProbeLauncherError("inner lease schema/status drifted")


def _inner_evidence(config: LaunchConfig, context: LaunchContext) -> dict[str, Any]:
    inner_capture = config.capture_dir / INNER_CAPTURE
    receipt_path = inner_capture / "receipt.json"
    evidence: dict[str, Any] = {
        "capture_dir": str(inner_capture),
        "receipt": {"path": str(receipt_path), "present": receipt_path.is_file()},
    }
    if not receipt_path.is_file():
        evidence["invocation"] = {
            "path": str(inner_capture / "invocation.json"),
            "present": (inner_capture / "invocation.json").is_file(),
        }
        return evidence
    try:
        receipt = _read_json(receipt_path, "inner shared-expert receipt")
        evidence["receipt"] = _file_evidence(receipt_path, "inner shared-expert receipt")
        evidence["schema"] = receipt.get("schema")
        evidence["status"] = receipt.get("status")
        evidence["mode"] = receipt.get("mode")
        evidence["metal_performed"] = receipt.get("metal_device_or_dispatch_performed")
        _validate_inner_binding(receipt, config, context)
    except SharedExpertProbeLauncherError as exc:
        evidence["binding_valid"] = False
        evidence["binding_error"] = str(exc)
    else:
        evidence["binding_valid"] = True
    return evidence


def _terminate_process_group(process: subprocess.Popen[bytes]) -> int | None:
    if process.poll() is not None:
        return process.returncode
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait(timeout=10)


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


def _terminal_status(terminal: Mapping[str, Any], inner: Mapping[str, Any]) -> str:
    if terminal.get("spawn_error"):
        return "REFUSED_QWEN80_SHARED_EXPERT_OUTER_CHILD_SPAWN_ERROR"
    if terminal.get("timed_out"):
        return "REFUSED_QWEN80_SHARED_EXPERT_OUTER_CHILD_TIMEOUT"
    if terminal.get("signal") is not None:
        return "REFUSED_QWEN80_SHARED_EXPERT_OUTER_CHILD_SIGNAL"
    if terminal.get("exit_code") != 0:
        return "REFUSED_QWEN80_SHARED_EXPERT_OUTER_CHILD_NONZERO"
    if inner.get("binding_valid") is not True or inner.get("status") != EXPECTED_METAL_STATUS:
        return "REFUSED_QWEN80_SHARED_EXPERT_OUTER_ZERO_EXIT_WITHOUT_STRICTLY_BOUND_INNER_RECEIPT"
    return "CAPTURED_QWEN80_SHARED_EXPERT_OUTER_TERMINAL_COMPONENT_ONLY"


def _terminal_success(receipt: Mapping[str, Any]) -> bool:
    return receipt.get("status") == "CAPTURED_QWEN80_SHARED_EXPERT_OUTER_TERMINAL_COMPONENT_ONLY"


def _terminal_receipt(
    config: LaunchConfig,
    context: LaunchContext,
    *,
    identity: str,
    command: Sequence[str],
    child_pid: int | None,
    started_at: str,
    finished_at: str,
    terminal: Mapping[str, Any],
    capture_error: str | None = None,
) -> dict[str, Any]:
    inner = _inner_evidence(config, context)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": _terminal_status(terminal, inner),
        "recorded_at": finished_at,
        "one_shot": {
            "automatic_retry_disabled": True,
            "same_capture_dir_never_starts_a_second_child": True,
            "terminal_receipt_written_last": True,
        },
        "launch_identity_sha256": identity,
        "source_binding": {
            "probe_binary": context.probe_binary,
            "manifest": context.manifest,
            "admission_current": context.admission_current,
            "admission_pointer_seal_sha256": context.admission_pointer_seal_sha256,
            "admission_receipt_seal_sha256": context.admission_receipt_seal_sha256,
            "cpu_baseline_receipt": context.cpu_baseline_receipt,
            "cpu_baseline_seal_sha256": context.cpu_baseline_seal_sha256,
            "cpu_inner_receipt": context.cpu_inner_receipt,
            "lease_receipt": context.lease_receipt,
            "lease_seal_sha256": context.lease_seal_sha256,
            "mode": config.mode,
            "workers": config.workers,
        },
        "child": {
            "pid": child_pid,
            "started_at": started_at,
            "finished_at": finished_at,
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
            "outer_terminal_capture_only": True,
            "shared_expert_component_only": True,
            "does_not_validate_or_promote_inner_component_parity": True,
            "does_not_execute_routed_expert_sum_moe_combine_or_complete_layer": True,
            "does_not_generate_tokens_expose_hcli_or_measure_tps": True,
            "does_not_claim_tg10_tg3_or_tournament_qualification": True,
        },
    }
    if capture_error is not None:
        receipt["capture_error"] = capture_error
    return seal(receipt)


def _replay_existing(config: LaunchConfig, identity: str) -> dict[str, Any]:
    terminal_path = config.capture_dir / TERMINAL_FILENAME
    if not terminal_path.is_file():
        raise SharedExpertProbeLauncherError(
            f"capture directory exists without a terminal receipt: {config.capture_dir}"
        )
    receipt = _read_json(terminal_path, "outer terminal receipt")
    try:
        verify(receipt, label=str(terminal_path))
    except ValueError as exc:
        raise SharedExpertProbeLauncherError(f"outer terminal receipt is not sealed: {exc}") from exc
    if receipt.get("schema") != SCHEMA or receipt.get("launch_identity_sha256") != identity:
        raise SharedExpertProbeLauncherError("capture directory belongs to another launch identity")
    return receipt


def run_attempt(config: LaunchConfig) -> dict[str, Any]:
    """Run one reaped strict-Metal process group, or replay its terminal receipt."""

    context = _validate_config(config)
    identity = _launch_identity(config, context)
    if config.capture_dir.exists():
        return _replay_existing(config, identity)
    if not config.capture_dir.parent.is_dir():
        raise SharedExpertProbeLauncherError(
            f"capture parent does not exist: {config.capture_dir.parent}"
        )
    try:
        config.capture_dir.mkdir(mode=0o750)
    except FileExistsError:
        return _replay_existing(config, identity)
    command = _child_command(config, config.capture_dir / INNER_CAPTURE)
    started_at = _utc_now()
    _atomic_json_new(
        config.capture_dir / ACTIVE_FILENAME,
        seal(
            {
                "schema": SCHEMA,
                "status": "STARTED_QWEN80_SHARED_EXPERT_OUTER_ONE_SHOT",
                "recorded_at": started_at,
                "launch_identity_sha256": identity,
                "command": command,
                "claim_boundary": {
                    "automatic_retry_disabled": True,
                    "future_strict_metal_component_only": True,
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
                            "status": "RUNNING_QWEN80_SHARED_EXPERT_OUTER_ONE_SHOT",
                            "recorded_at": _utc_now(),
                            "launch_identity_sha256": identity,
                            "pid": child_pid,
                            "parent_pid": os.getpid(),
                            "command": command,
                            "inner_capture_dir": str(config.capture_dir / INNER_CAPTURE),
                        }
                    ),
                )
            except SharedExpertProbeLauncherError as exc:
                capture_error = str(exc)
                terminal = _terminal(_terminate_process_group(child), timed_out=False)
            else:
                try:
                    terminal = _terminal(child.wait(timeout=config.timeout_seconds), timed_out=False)
                except subprocess.TimeoutExpired:
                    terminal = _terminal(_terminate_process_group(child), timed_out=True)
    receipt = _terminal_receipt(
        config,
        context,
        identity=identity,
        command=command,
        child_pid=child_pid,
        started_at=started_at,
        finished_at=_utc_now(),
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
    parser.add_argument("--mode", choices=("metal",), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
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
        mode=parsed.mode,
        workers=parsed.workers,
        timeout_seconds=parsed.timeout_seconds,
    )
    try:
        receipt = run_attempt(config)
    except SharedExpertProbeLauncherError as exc:
        print(json.dumps({"status": "REFUSED_QWEN80_SHARED_EXPERT_OUTER_LAUNCHER_ERROR", "error": str(exc)}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0 if _terminal_success(receipt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
