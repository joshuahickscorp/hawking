"""One-shot outer capture for the Qwen80 route-0/expert-65 component.

The launcher deliberately owns process lifecycle and durable terminal evidence
only.  The child remains the authority for the CPU or strict-Metal component
math.  Before it starts a child, this launcher binds the current manifest and
admission pointer to the sealed upstream postnorm/router *outer* record and
its exact inner receipt.  Once started, it launches exactly once, captures
both streams, reaps the whole child process group, and writes its sealed
terminal record last.

This is not a retry loop, a scheduler, an admission authority, or a complete
Qwen80 layer/token/decoder/HCLI/TPS/TG/tournament result.  In particular, it
does not turn one source-selected expert into the remaining nine routed
experts, the shared expert, aggregation, or the second residual.
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
SCHEMA = "hawking.ascension.qwen80_routed_expert_65_outer_launcher.v1"
ACTIVE_FILENAME = "active.json"
CHILD_FILENAME = "child.json"
TERMINAL_FILENAME = "outer-terminal-receipt.json"
OUTER_STDOUT = "outer.stdout.log"
OUTER_STDERR = "outer.stderr.log"
INNER_CAPTURE = "inner"

EXPECTED_PROBE_BASENAME = "ascension_qwen80_direct_packed_routed_expert_wave"
EXPECTED_INNER_SCHEMA = "hawking.ascension.qwen80_direct_packed_routed_expert_wave.v1"
EXPECTED_CPU_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_ONE_ROUTED_EXPERT_"
    "CPU_ORACLE_READY_METAL_LEASE_REQUIRED"
)
EXPECTED_METAL_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_ONE_ROUTED_EXPERT_65_"
    "STRICT_MATH_METAL_COMPONENT_NOT_TEN_ROUTE_OR_LAYER"
)

CURRENT_ADMISSION_SCHEMA = (
    "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1"
)
CURRENT_ADMISSION_STATUS = "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED"
UPSTREAM_ROUTER_OUTER_SCHEMA = (
    "hawking.ascension.qwen80_direct_packed_postnorm_router_top10_outer_launcher.v1"
)
UPSTREAM_ROUTER_OUTER_STATUS = (
    "CAPTURED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_TERMINAL_COMPONENT_ONLY"
)
UPSTREAM_ROUTER_INNER_SCHEMA = (
    "hawking.ascension.qwen80_direct_packed_postnorm_router_top10.v1"
)
UPSTREAM_ROUTER_INNER_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_POSTNORM_ROUTER_TOP10_"
    "STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN"
)
ROUTED_EXPERT_LEASE_SCHEMA = (
    "hawking.ascension.qwen80_routed_expert_65_quiet_metal_lease.v1"
)
ROUTED_EXPERT_LEASE_STATUS = (
    "GRANTED_QWEN80_ROUTED_EXPERT_65_NON_TIMED_DEVICE_PARITY_LEASE"
)
ROUTED_EXPERT_LEASE_COMPONENT = "qwen80_direct_packed_routed_expert_65"


class RoutedExpertProbeLauncherError(RuntimeError):
    """The one-shot route-65 outer capture cannot safely continue."""


@dataclass(frozen=True)
class LaunchConfig:
    probe_bin: Path
    manifest: Path
    admission_current: Path
    router_receipt: Path
    router_outer_receipt: Path
    capture_dir: Path
    mode: str
    workers: int
    timeout_seconds: float
    lease_receipt: Path | None = None


@dataclass(frozen=True)
class LaunchContext:
    """Immutable evidence captured before the child can observe mutable inputs."""

    probe_binary: dict[str, Any]
    manifest: dict[str, Any]
    admission_current: dict[str, Any]
    admission_pointer_seal_sha256: str
    admission_receipt_seal_sha256: str
    router_receipt: dict[str, Any]
    router_outer_receipt: dict[str, Any]
    router_outer_seal_sha256: str
    upstream_admission_pointer_evidence: dict[str, Any]
    lease_receipt: dict[str, Any] | None


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
        raise RoutedExpertProbeLauncherError(f"{label} must be absolute: {path}")


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    _require_absolute(path, label)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RoutedExpertProbeLauncherError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RoutedExpertProbeLauncherError(f"{label} must be a regular non-symlink file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise RoutedExpertProbeLauncherError(f"{label} must be executable: {path}")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise RoutedExpertProbeLauncherError(f"cannot canonicalize {label} {path}: {exc}") from exc


def _canonical_from_document(value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise RoutedExpertProbeLauncherError(f"{label} must be an absolute path string")
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
    """Publish a receipt with create-new semantics and durable metadata."""

    if path.exists():
        raise RoutedExpertProbeLauncherError(f"refusing to overwrite {path}")
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
        raise RoutedExpertProbeLauncherError(f"refusing to overwrite {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RoutedExpertProbeLauncherError(f"cannot read JSON {label} at {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RoutedExpertProbeLauncherError(f"JSON {label} at {path} is not an object")
    return dict(payload)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RoutedExpertProbeLauncherError(f"{label} must be an object")
    return dict(value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise RoutedExpertProbeLauncherError(f"{label} must be a string")
    return value


def _sealed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    document = _read_json(path, label)
    try:
        verify(document, label=str(path))
    except ValueError as exc:
        raise RoutedExpertProbeLauncherError(f"{label} is not a valid sealed receipt: {exc}") from exc
    seal_sha256 = document.get("seal_sha256")
    if not _is_sha256(seal_sha256):
        raise RoutedExpertProbeLauncherError(f"{label} has no lowercase SHA-256 seal")
    return document, str(seal_sha256)


def _assert_evidence_matches_file(
    evidence: object, expected: dict[str, Any], label: str, *, require_digest: bool = True
) -> None:
    row = _mapping(evidence, label)
    if row.get("present") is not True:
        raise RoutedExpertProbeLauncherError(f"{label} does not attest a present file")
    observed_path = _canonical_from_document(row.get("path"), f"{label}.path")
    if observed_path != Path(expected["path"]):
        raise RoutedExpertProbeLauncherError(
            f"{label} path drifted: expected {expected['path']}, got {observed_path}"
        )
    if row.get("bytes") != expected["bytes"]:
        raise RoutedExpertProbeLauncherError(f"{label} byte count drifted")
    if require_digest and row.get("sha256") != expected["sha256"]:
        raise RoutedExpertProbeLauncherError(f"{label} SHA-256 drifted")


def _bind_current_admission(
    admission_path: Path, manifest: dict[str, Any]
) -> tuple[dict[str, Any], str, str]:
    document, pointer_seal = _sealed_json(admission_path, "--admission-current")
    if (
        document.get("schema") != CURRENT_ADMISSION_SCHEMA
        or document.get("status") != CURRENT_ADMISSION_STATUS
    ):
        raise RoutedExpertProbeLauncherError("--admission-current schema/status drifted")
    complete_manifest = _mapping(document.get("complete_manifest"), "admission complete_manifest")
    if _canonical_from_document(
        complete_manifest.get("path"), "admission complete_manifest.path"
    ) != Path(manifest["path"]):
        raise RoutedExpertProbeLauncherError("--admission-current does not select --manifest")
    if complete_manifest.get("document_sha256") != manifest["sha256"]:
        raise RoutedExpertProbeLauncherError("--admission-current manifest document SHA-256 drifted")
    admission_receipt = _mapping(document.get("admission_receipt"), "admission admission_receipt")
    admission_receipt_seal = admission_receipt.get("seal_sha256")
    if not _is_sha256(admission_receipt_seal):
        raise RoutedExpertProbeLauncherError("admission receipt lacks a lowercase SHA-256 seal")
    return _file_evidence(admission_path, "--admission-current"), pointer_seal, str(
        admission_receipt_seal
    )


def _bind_upstream_router(
    *,
    manifest: dict[str, Any],
    admission_current: dict[str, Any],
    admission_receipt_seal: str,
    router_receipt_path: Path,
    router_outer_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    """Bind the current input to the immutable router outer+inner chain.

    The current-pointer *document* may be refreshed by its watcher, so a
    historical router outer receipt is joined to the current pointer through
    the stable admitted receipt seal, rather than an obsolete pointer digest.
    The new child will still be required to echo this attempt's exact current
    pointer seal.
    """

    router_evidence = _file_evidence(router_receipt_path, "--router-receipt")
    outer_evidence = _file_evidence(router_outer_path, "--router-outer-receipt")
    outer, outer_seal = _sealed_json(router_outer_path, "--router-outer-receipt")
    if (
        outer.get("schema") != UPSTREAM_ROUTER_OUTER_SCHEMA
        or outer.get("status") != UPSTREAM_ROUTER_OUTER_STATUS
    ):
        raise RoutedExpertProbeLauncherError("--router-outer-receipt schema/status drifted")
    source_binding = _mapping(outer.get("source_binding"), "router outer source_binding")
    _assert_evidence_matches_file(
        source_binding.get("manifest"), manifest, "router outer manifest evidence"
    )
    # The outer record has a historical pointer digest.  The stable admission
    # receipt seal is checked below against the exact inner router receipt.
    upstream_admission = _mapping(
        source_binding.get("admission_current"), "router outer admission evidence"
    )
    if upstream_admission.get("present") is not True:
        raise RoutedExpertProbeLauncherError("router outer did not attest its admission pointer")
    if _canonical_from_document(
        upstream_admission.get("path"), "router outer admission evidence.path"
    ) != Path(admission_current["path"]):
        raise RoutedExpertProbeLauncherError("router outer admission pointer path drifted")
    if not _is_sha256(upstream_admission.get("sha256")):
        raise RoutedExpertProbeLauncherError("router outer admission evidence lacks SHA-256")

    inner_summary = _mapping(outer.get("inner_probe_capture"), "router outer inner_probe_capture")
    if (
        inner_summary.get("present") is not True
        or inner_summary.get("schema") != UPSTREAM_ROUTER_INNER_SCHEMA
        or inner_summary.get("status") != UPSTREAM_ROUTER_INNER_STATUS
        or inner_summary.get("mode") != "metal"
        or inner_summary.get("metal_performed") is not True
    ):
        raise RoutedExpertProbeLauncherError("router outer lacks the required strict-Metal inner receipt")
    if _canonical_from_document(
        inner_summary.get("path"), "router outer inner receipt path"
    ) != Path(router_evidence["path"]):
        raise RoutedExpertProbeLauncherError("router outer references another inner router receipt")
    if inner_summary.get("sha256") != router_evidence["sha256"]:
        raise RoutedExpertProbeLauncherError("router outer inner router SHA-256 drifted")

    router = _read_json(router_receipt_path, "--router-receipt")
    if (
        router.get("schema") != UPSTREAM_ROUTER_INNER_SCHEMA
        or router.get("status") != UPSTREAM_ROUTER_INNER_STATUS
        or router.get("mode") != "metal"
        or router.get("component_only") is not True
        or router.get("metal_device_or_dispatch_performed") is not True
    ):
        raise RoutedExpertProbeLauncherError("--router-receipt is not required strict-Metal component evidence")
    artifact_binding = _mapping(router.get("artifact_binding"), "router artifact_binding")
    if _canonical_from_document(
        artifact_binding.get("manifest_path"), "router artifact_binding.manifest_path"
    ) != Path(manifest["path"]):
        raise RoutedExpertProbeLauncherError("router inner manifest path drifted")
    if artifact_binding.get("manifest_document_sha256") != manifest["sha256"]:
        raise RoutedExpertProbeLauncherError("router inner manifest SHA-256 drifted")
    if _canonical_from_document(
        artifact_binding.get("admission_current_path"), "router artifact_binding.admission_current_path"
    ) != Path(admission_current["path"]):
        raise RoutedExpertProbeLauncherError("router inner admission pointer path drifted")
    if artifact_binding.get("admission_receipt_seal_sha256") != admission_receipt_seal:
        raise RoutedExpertProbeLauncherError("router inner admission receipt seal drifted")
    return router_evidence, outer_evidence, outer_seal, upstream_admission


def _bind_lease(path: Path) -> dict[str, Any]:
    document, _ = _sealed_json(path, "--lease-receipt")
    if (
        document.get("schema") != ROUTED_EXPERT_LEASE_SCHEMA
        or document.get("status") != ROUTED_EXPERT_LEASE_STATUS
    ):
        raise RoutedExpertProbeLauncherError("--lease-receipt schema/status does not authorize route-65")
    policy = _mapping(document.get("execution_policy"), "route-65 lease execution_policy")
    if (
        policy.get("component") != ROUTED_EXPERT_LEASE_COMPONENT
        or policy.get("quiet_qwen80_device_lease") is not True
        or policy.get("strict_math") is not True
        or policy.get("timing_or_benchmarking_allowed") is not False
        or policy.get("complete_layer_or_token_allowed") is not False
        or policy.get("tps_or_tg_claim_allowed") is not False
    ):
        raise RoutedExpertProbeLauncherError("--lease-receipt policy is not the strict non-timed route-65 lease")
    return _file_evidence(path, "--lease-receipt")


def _validate_config(config: LaunchConfig) -> LaunchContext:
    probe = _canonical_regular(config.probe_bin, "--probe-bin", executable=True)
    if probe.name != EXPECTED_PROBE_BASENAME:
        raise RoutedExpertProbeLauncherError(
            f"--probe-bin must name {EXPECTED_PROBE_BASENAME}, got {probe.name!r}"
        )
    for path, label in (
        (config.manifest, "--manifest"),
        (config.admission_current, "--admission-current"),
        (config.router_receipt, "--router-receipt"),
        (config.router_outer_receipt, "--router-outer-receipt"),
    ):
        _canonical_regular(path, label)
    _require_absolute(config.capture_dir, "--capture-dir")
    if config.mode not in {"cpu-oracle", "metal"}:
        raise RoutedExpertProbeLauncherError(f"unsupported --mode {config.mode!r}")
    if config.workers < 1:
        raise RoutedExpertProbeLauncherError("--workers must be positive")
    if not config.timeout_seconds > 0:
        raise RoutedExpertProbeLauncherError("--timeout-seconds must be positive")
    if config.mode == "metal":
        if config.lease_receipt is None:
            raise RoutedExpertProbeLauncherError("--mode metal requires --lease-receipt")
        lease_evidence = _bind_lease(config.lease_receipt)
    else:
        if config.lease_receipt is not None:
            raise RoutedExpertProbeLauncherError("--lease-receipt is valid only with --mode metal")
        lease_evidence = None

    probe_evidence = _file_evidence(config.probe_bin, "--probe-bin")
    manifest = _file_evidence(config.manifest, "--manifest")
    admission_current, pointer_seal, admission_receipt_seal = _bind_current_admission(
        config.admission_current, manifest
    )
    router_receipt, router_outer, router_outer_seal, upstream_admission = _bind_upstream_router(
        manifest=manifest,
        admission_current=admission_current,
        admission_receipt_seal=admission_receipt_seal,
        router_receipt_path=config.router_receipt,
        router_outer_path=config.router_outer_receipt,
    )
    return LaunchContext(
        probe_binary=probe_evidence,
        manifest=manifest,
        admission_current=admission_current,
        admission_pointer_seal_sha256=pointer_seal,
        admission_receipt_seal_sha256=admission_receipt_seal,
        router_receipt=router_receipt,
        router_outer_receipt=router_outer,
        router_outer_seal_sha256=router_outer_seal,
        upstream_admission_pointer_evidence=upstream_admission,
        lease_receipt=lease_evidence,
    )


def _launch_identity(config: LaunchConfig, context: LaunchContext) -> str:
    payload: dict[str, Any] = {
        "probe_binary": context.probe_binary,
        "manifest": context.manifest,
        "admission_current": context.admission_current,
        "admission_pointer_seal_sha256": context.admission_pointer_seal_sha256,
        "admission_receipt_seal_sha256": context.admission_receipt_seal_sha256,
        "router_receipt": context.router_receipt,
        "router_outer_receipt": context.router_outer_receipt,
        "router_outer_seal_sha256": context.router_outer_seal_sha256,
        "mode": config.mode,
        "workers": config.workers,
        "timeout_seconds": config.timeout_seconds,
        "lease_receipt": context.lease_receipt,
    }
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def _child_command(config: LaunchConfig, inner_capture: Path) -> list[str]:
    command = [
        str(_canonical_regular(config.probe_bin, "--probe-bin", executable=True)),
        "--manifest",
        str(_canonical_regular(config.manifest, "--manifest")),
        "--admission-current",
        str(_canonical_regular(config.admission_current, "--admission-current")),
        "--router-receipt",
        str(_canonical_regular(config.router_receipt, "--router-receipt")),
        "--router-outer-receipt",
        str(_canonical_regular(config.router_outer_receipt, "--router-outer-receipt")),
        "--capture-dir",
        str(inner_capture),
        "--mode",
        config.mode,
        "--workers",
        str(config.workers),
    ]
    if config.lease_receipt is not None:
        command.extend(
            ("--lease-receipt", str(_canonical_regular(config.lease_receipt, "--lease-receipt")))
        )
    return command


def _sync_evidence(path: Path) -> dict[str, Any]:
    with path.open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    return _file_evidence(path, f"outer stream {path.name}")


def _inner_evidence(config: LaunchConfig, context: LaunchContext) -> dict[str, Any]:
    inner_capture = config.capture_dir / INNER_CAPTURE
    receipt_path = inner_capture / "receipt.json"
    evidence: dict[str, Any] = {
        "capture_dir": str(inner_capture),
        "receipt": {
            "path": str(receipt_path),
            "present": receipt_path.is_file(),
        },
    }
    if not receipt_path.is_file():
        evidence["invocation"] = {
            "path": str(inner_capture / "invocation.json"),
            "present": (inner_capture / "invocation.json").is_file(),
        }
        return evidence
    try:
        receipt = _read_json(receipt_path, "inner routed-expert receipt")
        receipt_evidence = _file_evidence(receipt_path, "inner routed-expert receipt")
        evidence["receipt"] = receipt_evidence
        evidence["schema"] = receipt.get("schema")
        evidence["status"] = receipt.get("status")
        evidence["mode"] = receipt.get("mode")
        evidence["metal_performed"] = receipt.get("metal_device_or_dispatch_performed")
        _validate_inner_binding(receipt, config, context)
    except RoutedExpertProbeLauncherError as exc:
        evidence["binding_valid"] = False
        evidence["binding_error"] = str(exc)
    else:
        evidence["binding_valid"] = True
    return evidence


def _validate_inner_binding(
    receipt: Mapping[str, Any], config: LaunchConfig, context: LaunchContext
) -> None:
    expected_status = EXPECTED_METAL_STATUS if config.mode == "metal" else EXPECTED_CPU_STATUS
    expected_metal = config.mode == "metal"
    if (
        receipt.get("schema") != EXPECTED_INNER_SCHEMA
        or receipt.get("status") != expected_status
        or receipt.get("mode") != config.mode
        or receipt.get("one_selected_expert_only") is not True
        or receipt.get("metal_device_or_dispatch_performed") is not expected_metal
    ):
        raise RoutedExpertProbeLauncherError("inner route-65 schema/status/mode boundary drifted")
    durable_capture = _mapping(receipt.get("durable_capture"), "inner durable_capture")
    if durable_capture.get("receipt_written_last_is_completion_marker") is not True:
        raise RoutedExpertProbeLauncherError("inner receipt does not attest receipt-last capture")
    artifact_binding = _mapping(receipt.get("artifact_binding"), "inner artifact_binding")
    if _canonical_from_document(
        artifact_binding.get("manifest_path"), "inner artifact_binding.manifest_path"
    ) != Path(context.manifest["path"]):
        raise RoutedExpertProbeLauncherError("inner manifest path drifted")
    if artifact_binding.get("manifest_document_sha256") != context.manifest["sha256"]:
        raise RoutedExpertProbeLauncherError("inner manifest SHA-256 drifted")
    if _canonical_from_document(
        artifact_binding.get("admission_current_path"), "inner artifact_binding.admission_current_path"
    ) != Path(context.admission_current["path"]):
        raise RoutedExpertProbeLauncherError("inner admission pointer path drifted")
    if artifact_binding.get("admission_pointer_seal_sha256") != context.admission_pointer_seal_sha256:
        raise RoutedExpertProbeLauncherError("inner admission pointer seal drifted during launch")
    if artifact_binding.get("admission_receipt_seal_sha256") != context.admission_receipt_seal_sha256:
        raise RoutedExpertProbeLauncherError("inner admission receipt seal drifted")

    route = _mapping(receipt.get("route_evidence"), "inner route_evidence")
    if _canonical_from_document(
        route.get("router_receipt_path"), "inner route_evidence.router_receipt_path"
    ) != Path(context.router_receipt["path"]):
        raise RoutedExpertProbeLauncherError("inner router receipt path drifted")
    if route.get("router_receipt_sha256") != context.router_receipt["sha256"]:
        raise RoutedExpertProbeLauncherError("inner router receipt SHA-256 drifted")
    if _canonical_from_document(
        route.get("router_outer_receipt_path"), "inner route_evidence.router_outer_receipt_path"
    ) != Path(context.router_outer_receipt["path"]):
        raise RoutedExpertProbeLauncherError("inner router outer receipt path drifted")
    if route.get("router_outer_receipt_sha256") != context.router_outer_receipt["sha256"]:
        raise RoutedExpertProbeLauncherError("inner router outer receipt SHA-256 drifted")
    if route.get("router_outer_receipt_seal_sha256") != context.router_outer_seal_sha256:
        raise RoutedExpertProbeLauncherError("inner router outer receipt seal drifted")
    if route.get("selected_expert") != 65 or route.get("selected_route_index") != 0:
        raise RoutedExpertProbeLauncherError("inner result is not source route-0/expert-65")

    if config.mode == "metal":
        assert context.lease_receipt is not None  # established by _validate_config
        policy = _mapping(receipt.get("metal_execution_policy"), "inner metal_execution_policy")
        if (
            policy.get("strict_math_required") is not True
            or policy.get("timing_or_benchmarking_allowed") is not False
            or policy.get("complete_layer_or_token_allowed") is not False
            or policy.get("tps_or_tg_claim_allowed") is not False
        ):
            raise RoutedExpertProbeLauncherError("inner Metal policy drifted")
        lease_binding = _mapping(policy.get("lease_binding"), "inner lease_binding")
        if _canonical_from_document(
            lease_binding.get("receipt_path"), "inner lease_binding.receipt_path"
        ) != Path(context.lease_receipt["path"]):
            raise RoutedExpertProbeLauncherError("inner lease receipt path drifted")
        if lease_binding.get("receipt_document_sha256") != context.lease_receipt["sha256"]:
            raise RoutedExpertProbeLauncherError("inner lease receipt SHA-256 drifted")
        if (
            lease_binding.get("schema") != ROUTED_EXPERT_LEASE_SCHEMA
            or lease_binding.get("status") != ROUTED_EXPERT_LEASE_STATUS
        ):
            raise RoutedExpertProbeLauncherError("inner lease schema/status drifted")


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


def _terminal_status(config: LaunchConfig, terminal: Mapping[str, Any], inner: Mapping[str, Any]) -> str:
    if terminal.get("spawn_error"):
        return "REFUSED_QWEN80_ROUTED_EXPERT_65_OUTER_CHILD_SPAWN_ERROR"
    if terminal.get("timed_out"):
        return "REFUSED_QWEN80_ROUTED_EXPERT_65_OUTER_CHILD_TIMEOUT"
    if terminal.get("signal") is not None:
        return "REFUSED_QWEN80_ROUTED_EXPERT_65_OUTER_CHILD_SIGNAL"
    if terminal.get("exit_code") != 0:
        return "REFUSED_QWEN80_ROUTED_EXPERT_65_OUTER_CHILD_NONZERO"
    expected = EXPECTED_METAL_STATUS if config.mode == "metal" else EXPECTED_CPU_STATUS
    if inner.get("binding_valid") is not True or inner.get("status") != expected:
        return (
            "REFUSED_QWEN80_ROUTED_EXPERT_65_OUTER_ZERO_EXIT_"
            "WITHOUT_STRICTLY_BOUND_INNER_RECEIPT"
        )
    return "CAPTURED_QWEN80_ROUTED_EXPERT_65_OUTER_TERMINAL_COMPONENT_ONLY"


def _terminal_success(receipt: Mapping[str, Any]) -> bool:
    return receipt.get("status") == "CAPTURED_QWEN80_ROUTED_EXPERT_65_OUTER_TERMINAL_COMPONENT_ONLY"


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
        "status": _terminal_status(config, terminal, inner),
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
            "router_receipt": context.router_receipt,
            "router_outer_receipt": context.router_outer_receipt,
            "router_outer_seal_sha256": context.router_outer_seal_sha256,
            "router_outer_historical_admission_pointer": context.upstream_admission_pointer_evidence,
            "lease_receipt": context.lease_receipt,
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
            "one_source_selected_route_0_expert_65_only": True,
            "does_not_validate_or_promote_inner_component_parity": True,
            "does_not_execute_remaining_routes_shared_expert_or_complete_layer": True,
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
        raise RoutedExpertProbeLauncherError(
            f"capture directory exists without a terminal receipt: {config.capture_dir}"
        )
    receipt = _read_json(terminal_path, "outer terminal receipt")
    try:
        verify(receipt, label=str(terminal_path))
    except ValueError as exc:
        raise RoutedExpertProbeLauncherError(f"outer terminal receipt is not sealed: {exc}") from exc
    if receipt.get("schema") != SCHEMA or receipt.get("launch_identity_sha256") != identity:
        raise RoutedExpertProbeLauncherError("capture directory belongs to another launch identity")
    return receipt


def run_attempt(config: LaunchConfig) -> dict[str, Any]:
    """Run exactly one process group or sealed-replay its terminal record."""

    context = _validate_config(config)
    identity = _launch_identity(config, context)
    if config.capture_dir.exists():
        return _replay_existing(config, identity)
    if not config.capture_dir.parent.is_dir():
        raise RoutedExpertProbeLauncherError(
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
                "status": "STARTED_QWEN80_ROUTED_EXPERT_65_OUTER_ONE_SHOT",
                "recorded_at": started_at,
                "launch_identity_sha256": identity,
                "command": command,
                "claim_boundary": {"automatic_retry_disabled": True, "component_only": True},
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
                            "status": "RUNNING_QWEN80_ROUTED_EXPERT_65_OUTER_ONE_SHOT",
                            "recorded_at": _utc_now(),
                            "launch_identity_sha256": identity,
                            "pid": child_pid,
                            "parent_pid": os.getpid(),
                            "command": command,
                            "inner_capture_dir": str(config.capture_dir / INNER_CAPTURE),
                        }
                    ),
                )
            except RoutedExpertProbeLauncherError as exc:
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
    parser.add_argument("--router-receipt", type=Path, required=True)
    parser.add_argument("--router-outer-receipt", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("cpu-oracle", "metal"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lease-receipt", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(argv)
    config = LaunchConfig(
        probe_bin=parsed.probe_bin,
        manifest=parsed.manifest,
        admission_current=parsed.admission_current,
        router_receipt=parsed.router_receipt,
        router_outer_receipt=parsed.router_outer_receipt,
        capture_dir=parsed.capture_dir,
        mode=parsed.mode,
        workers=parsed.workers,
        timeout_seconds=parsed.timeout_seconds,
        lease_receipt=parsed.lease_receipt,
    )
    try:
        receipt = run_attempt(config)
    except RoutedExpertProbeLauncherError as exc:
        print(json.dumps({"status": "REFUSED_QWEN80_ROUTED_EXPERT_65_OUTER_LAUNCHER_ERROR", "error": str(exc)}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0 if _terminal_success(receipt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
