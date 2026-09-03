"""One-shot outer capture for a future Qwen80 MoE combine component.

This launcher is deliberately only preparation for the narrow component that
aggregates the ten source-selected routed-expert outputs, adds the shared
expert, and applies the second residual.  It never treats raw or unsigned CPU
output as baseline evidence: a future ``lab.receipts``-sealed wrapper must
attest the exact CPU inner receipt first.  A later strict-Metal child may run
once, under a component-specific quiet lease, after all immutable inputs are
bound.

The outer record owns lifecycle and terminal evidence only.  It is neither an
all-expert projection runner nor a complete layer/token/decoder/HCLI/TPS/TG or
tournament result.  Merely importing or creating this launcher does not open
Metal, start a watcher, or change the Qwen80 runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "hawking.ascension.qwen80_moe_combine_outer_launcher.v1"
ACTIVE_FILENAME = "active.json"
CHILD_FILENAME = "child.json"
TERMINAL_FILENAME = "outer-terminal-receipt.json"
OUTER_STDOUT = "outer.stdout.log"
OUTER_STDERR = "outer.stderr.log"
INNER_CAPTURE = "inner"

EXPECTED_PROBE_BASENAME = "ascension_qwen80_direct_packed_moe_combine"
EXPECTED_INNER_SCHEMA = "hawking.ascension.qwen80_direct_packed_moe_combine.v1"
EXPECTED_CPU_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_MOE_COMBINE_SHARED_ADD_SECOND_RESIDUAL_"
    "CPU_ORACLE_READY_METAL_LEASE_REQUIRED"
)
EXPECTED_METAL_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_MOE_COMBINE_SHARED_ADD_SECOND_RESIDUAL_"
    "STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN"
)

CURRENT_MANIFEST_SCHEMA = "hawking.ascension.qwen80_complete_binary_gravity.v1"
CURRENT_ADMISSION_SCHEMA = (
    "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1"
)
CURRENT_ADMISSION_STATUS = "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED"

# No such combine wrapper is created here.  Its schema makes the future CPU
# baseline an explicit, independently sealed hand-off rather than a raw inner
# receipt supplied to a device attempt.
CPU_BASELINE_WRAPPER_SCHEMA = "hawking.ascension.qwen80_moe_combine_cpu_baseline_wrapper.v1"
CPU_BASELINE_WRAPPER_STATUS = (
    "SEALED_CURRENT_ADMITTED_QWEN80_MOE_COMBINE_SHARED_ADD_SECOND_RESIDUAL_"
    "CPU_ORACLE_BASELINE"
)

MOE_COMBINE_LEASE_SCHEMA = "hawking.ascension.qwen80_moe_combine_quiet_metal_lease.v1"
MOE_COMBINE_LEASE_STATUS = (
    "GRANTED_QWEN80_MOE_COMBINE_SHARED_ADD_SECOND_RESIDUAL_"
    "NON_TIMED_DEVICE_PARITY_LEASE"
)
MOE_COMBINE_LEASE_COMPONENT = "qwen80_direct_packed_moe_combine"

UPSTREAM_ROUTER_OUTER_SCHEMA = (
    "hawking.ascension.qwen80_direct_packed_postnorm_router_top10_outer_launcher.v1"
)
UPSTREAM_ROUTER_OUTER_STATUS = (
    "CAPTURED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_TERMINAL_COMPONENT_ONLY"
)
UPSTREAM_ROUTER_INNER_SCHEMA = "hawking.ascension.qwen80_direct_packed_postnorm_router_top10.v1"
UPSTREAM_ROUTER_INNER_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_POSTNORM_ROUTER_TOP10_"
    "STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN"
)
SOURCE_TOP10_IDS = (65, 245, 227, 35, 189, 440, 298, 405, 109, 494)


class MoeCombineProbeLauncherError(RuntimeError):
    """The one-shot MoE-combine outer capture cannot safely continue."""


@dataclass(frozen=True)
class LaunchConfig:
    """Inputs for one future strict-Metal MoE-combine component attempt."""

    probe_bin: Path
    manifest: Path
    admission_current: Path
    router_receipt: Path
    router_outer_receipt: Path
    cpu_baseline_receipt: Path
    lease_receipt: Path | None
    capture_dir: Path
    mode: str
    workers: int
    timeout_seconds: float


@dataclass(frozen=True)
class LaunchContext:
    """Immutable evidence collected before a child can be started."""

    probe_binary: dict[str, Any]
    manifest: dict[str, Any]
    manifest_seal_sha256: str
    admission_current: dict[str, Any]
    admission_pointer_seal_sha256: str
    admission_receipt_seal_sha256: str
    router_receipt: dict[str, Any]
    router_outer_receipt: dict[str, Any]
    router_outer_seal_sha256: str
    router_outer_historical_admission_pointer: dict[str, Any]
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
        raise MoeCombineProbeLauncherError(f"{label} must be absolute: {path}")


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    _require_absolute(path, label)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise MoeCombineProbeLauncherError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise MoeCombineProbeLauncherError(f"{label} must be a regular non-symlink file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise MoeCombineProbeLauncherError(f"{label} must be executable: {path}")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise MoeCombineProbeLauncherError(f"cannot canonicalize {label} {path}: {exc}") from exc


def _canonical_from_document(value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise MoeCombineProbeLauncherError(f"{label} must be an absolute path string")
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
    """Publish a new record without replacing terminal evidence."""

    if path.exists():
        raise MoeCombineProbeLauncherError(f"refusing to overwrite {path}")
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
        raise MoeCombineProbeLauncherError(f"refusing to overwrite {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MoeCombineProbeLauncherError(f"cannot read JSON {label} at {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise MoeCombineProbeLauncherError(f"JSON {label} at {path} is not an object")
    return dict(payload)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MoeCombineProbeLauncherError(f"{label} must be an object")
    return dict(value)


def _sealed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    document = _read_json(path, label)
    try:
        verify(document, label=str(path))
    except ValueError as exc:
        raise MoeCombineProbeLauncherError(f"{label} is not a valid sealed receipt: {exc}") from exc
    seal_sha256 = document.get("seal_sha256")
    if not _is_sha256(seal_sha256):
        raise MoeCombineProbeLauncherError(f"{label} has no lowercase SHA-256 seal")
    return document, str(seal_sha256)


def _assert_evidence_matches_file(
    evidence: object, expected: Mapping[str, Any], label: str, *, require_digest: bool = True
) -> None:
    row = _mapping(evidence, label)
    if row.get("present") is not True:
        raise MoeCombineProbeLauncherError(f"{label} does not attest a present file")
    observed_path = _canonical_from_document(row.get("path"), f"{label}.path")
    if observed_path != Path(str(expected["path"])):
        raise MoeCombineProbeLauncherError(
            f"{label} path drifted: expected {expected['path']}, got {observed_path}"
        )
    if row.get("bytes") != expected["bytes"]:
        raise MoeCombineProbeLauncherError(f"{label} byte count drifted")
    if require_digest and row.get("sha256") != expected["sha256"]:
        raise MoeCombineProbeLauncherError(f"{label} SHA-256 drifted")


def _bind_manifest(path: Path) -> tuple[dict[str, Any], str]:
    document, seal_sha256 = _sealed_json(path, "--manifest")
    if document.get("schema") != CURRENT_MANIFEST_SCHEMA:
        raise MoeCombineProbeLauncherError("--manifest schema is not Qwen80 complete gravity")
    return _file_evidence(path, "--manifest"), seal_sha256


def _bind_current_admission(
    admission_path: Path, manifest: Mapping[str, Any], manifest_seal_sha256: str
) -> tuple[dict[str, Any], str, str]:
    document, pointer_seal = _sealed_json(admission_path, "--admission-current")
    if (
        document.get("schema") != CURRENT_ADMISSION_SCHEMA
        or document.get("status") != CURRENT_ADMISSION_STATUS
    ):
        raise MoeCombineProbeLauncherError("--admission-current schema/status drifted")
    complete_manifest = _mapping(document.get("complete_manifest"), "admission complete_manifest")
    if _canonical_from_document(
        complete_manifest.get("path"), "admission complete_manifest.path"
    ) != Path(str(manifest["path"])):
        raise MoeCombineProbeLauncherError("--admission-current does not select --manifest")
    if complete_manifest.get("document_sha256") != manifest["sha256"]:
        raise MoeCombineProbeLauncherError("--admission-current manifest document SHA-256 drifted")
    if complete_manifest.get("seal_sha256") != manifest_seal_sha256:
        raise MoeCombineProbeLauncherError("--admission-current manifest seal drifted")
    admission_receipt = _mapping(document.get("admission_receipt"), "admission admission_receipt")
    admission_receipt_seal = admission_receipt.get("seal_sha256")
    if not _is_sha256(admission_receipt_seal):
        raise MoeCombineProbeLauncherError("admission receipt lacks a lowercase SHA-256 seal")
    return _file_evidence(admission_path, "--admission-current"), pointer_seal, str(
        admission_receipt_seal
    )


def _source_top10_ids(value: object, label: str) -> None:
    if not isinstance(value, list) or tuple(value) != SOURCE_TOP10_IDS:
        raise MoeCombineProbeLauncherError(f"{label} is not the exact source top-10 ordering")


def _source_top10_weights(value: object, label: str) -> None:
    if not isinstance(value, list) or len(value) != len(SOURCE_TOP10_IDS):
        raise MoeCombineProbeLauncherError(f"{label} must contain ten source top-10 weights")
    if any(isinstance(weight, bool) or not isinstance(weight, (float, int)) for weight in value):
        raise MoeCombineProbeLauncherError(f"{label} must contain numeric weights")
    if any(not math.isfinite(float(weight)) or float(weight) <= 0.0 for weight in value):
        raise MoeCombineProbeLauncherError(f"{label} must contain finite positive weights")
    if not math.isclose(sum(float(weight) for weight in value), 1.0, rel_tol=0.0, abs_tol=2e-5):
        raise MoeCombineProbeLauncherError(f"{label} does not sum to one within source tolerance")


def _bind_upstream_router(
    *,
    manifest: Mapping[str, Any],
    manifest_seal_sha256: str,
    admission_current: Mapping[str, Any],
    admission_receipt_seal: str,
    router_receipt_path: Path,
    router_outer_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    """Bind the combine inputs to a sealed source top-10 outer+inner chain."""

    router_evidence = _file_evidence(router_receipt_path, "--router-receipt")
    outer_evidence = _file_evidence(router_outer_path, "--router-outer-receipt")
    outer, outer_seal = _sealed_json(router_outer_path, "--router-outer-receipt")
    if (
        outer.get("schema") != UPSTREAM_ROUTER_OUTER_SCHEMA
        or outer.get("status") != UPSTREAM_ROUTER_OUTER_STATUS
    ):
        raise MoeCombineProbeLauncherError("--router-outer-receipt schema/status drifted")
    source_binding = _mapping(outer.get("source_binding"), "router outer source_binding")
    _assert_evidence_matches_file(
        source_binding.get("manifest"), manifest, "router outer manifest evidence"
    )
    historical_admission = _mapping(
        source_binding.get("admission_current"), "router outer admission evidence"
    )
    if historical_admission.get("present") is not True:
        raise MoeCombineProbeLauncherError("router outer did not attest an admission pointer")
    if _canonical_from_document(
        historical_admission.get("path"), "router outer admission evidence.path"
    ) != Path(str(admission_current["path"])):
        raise MoeCombineProbeLauncherError("router outer admission pointer path drifted")
    if not _is_sha256(historical_admission.get("sha256")):
        raise MoeCombineProbeLauncherError("router outer admission evidence lacks SHA-256")

    inner_summary = _mapping(outer.get("inner_probe_capture"), "router outer inner_probe_capture")
    if (
        inner_summary.get("present") is not True
        or inner_summary.get("schema") != UPSTREAM_ROUTER_INNER_SCHEMA
        or inner_summary.get("status") != UPSTREAM_ROUTER_INNER_STATUS
        or inner_summary.get("mode") != "metal"
        or inner_summary.get("metal_performed") is not True
    ):
        raise MoeCombineProbeLauncherError("router outer lacks strict-Metal source top-10 evidence")
    if _canonical_from_document(
        inner_summary.get("path"), "router outer inner receipt path"
    ) != Path(str(router_evidence["path"])):
        raise MoeCombineProbeLauncherError("router outer references another inner router receipt")
    if inner_summary.get("sha256") != router_evidence["sha256"]:
        raise MoeCombineProbeLauncherError("router outer inner router SHA-256 drifted")

    router = _read_json(router_receipt_path, "--router-receipt")
    if (
        router.get("schema") != UPSTREAM_ROUTER_INNER_SCHEMA
        or router.get("status") != UPSTREAM_ROUTER_INNER_STATUS
        or router.get("mode") != "metal"
        or router.get("component_only") is not True
        or router.get("metal_device_or_dispatch_performed") is not True
    ):
        raise MoeCombineProbeLauncherError("--router-receipt is not strict-Metal source top-10 evidence")
    artifact_binding = _mapping(router.get("artifact_binding"), "router artifact_binding")
    if _canonical_from_document(
        artifact_binding.get("manifest_path"), "router artifact_binding.manifest_path"
    ) != Path(str(manifest["path"])):
        raise MoeCombineProbeLauncherError("router inner manifest path drifted")
    if artifact_binding.get("manifest_document_sha256") != manifest["sha256"]:
        raise MoeCombineProbeLauncherError("router inner manifest SHA-256 drifted")
    if artifact_binding.get("manifest_seal_sha256") != manifest_seal_sha256:
        raise MoeCombineProbeLauncherError("router inner manifest seal drifted")
    if _canonical_from_document(
        artifact_binding.get("admission_current_path"), "router artifact_binding.admission_current_path"
    ) != Path(str(admission_current["path"])):
        raise MoeCombineProbeLauncherError("router inner admission pointer path drifted")
    if artifact_binding.get("admission_receipt_seal_sha256") != admission_receipt_seal:
        raise MoeCombineProbeLauncherError("router inner admission receipt seal drifted")
    if artifact_binding.get("layer") != 0 or artifact_binding.get("experts_per_token") != 10:
        raise MoeCombineProbeLauncherError("router inner is not the layer-0/top-10 source binding")
    source_top10 = _mapping(router.get("source_stable_top10_router"), "router source_stable_top10_router")
    _source_top10_ids(source_top10.get("ids"), "router source top-10 ids")
    _source_top10_ids(source_top10.get("device_ids"), "router device top-10 ids")
    if (
        source_top10.get("device_ids_exact_match") is not True
        or source_top10.get("ids_unique_and_in_range") is not True
    ):
        raise MoeCombineProbeLauncherError("router source top-10 identity checks did not pass")
    _source_top10_weights(source_top10.get("renormalized_weights"), "router renormalized weights")
    return router_evidence, outer_evidence, outer_seal, historical_admission


def _validate_source_top10_binding(
    binding: object, context: LaunchContext, label: str
) -> None:
    row = _mapping(binding, label)
    if _canonical_from_document(row.get("router_receipt_path"), f"{label}.router_receipt_path") != Path(
        str(context.router_receipt["path"])
    ):
        raise MoeCombineProbeLauncherError(f"{label} router receipt path drifted")
    if row.get("router_receipt_sha256") != context.router_receipt["sha256"]:
        raise MoeCombineProbeLauncherError(f"{label} router receipt SHA-256 drifted")
    if _canonical_from_document(
        row.get("router_outer_receipt_path"), f"{label}.router_outer_receipt_path"
    ) != Path(str(context.router_outer_receipt["path"])):
        raise MoeCombineProbeLauncherError(f"{label} router outer receipt path drifted")
    if row.get("router_outer_receipt_sha256") != context.router_outer_receipt["sha256"]:
        raise MoeCombineProbeLauncherError(f"{label} router outer receipt SHA-256 drifted")
    if row.get("router_outer_receipt_seal_sha256") != context.router_outer_seal_sha256:
        raise MoeCombineProbeLauncherError(f"{label} router outer receipt seal drifted")
    _source_top10_ids(row.get("ids"), f"{label}.ids")


def _bind_cpu_baseline(
    path: Path, *, context_without_baseline: LaunchContext
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Require a sealed wrapper around a future CPU combine inner receipt."""

    wrapper, wrapper_seal = _sealed_json(path, "--cpu-baseline-receipt")
    if (
        wrapper.get("schema") != CPU_BASELINE_WRAPPER_SCHEMA
        or wrapper.get("status") != CPU_BASELINE_WRAPPER_STATUS
    ):
        raise MoeCombineProbeLauncherError("--cpu-baseline-receipt schema/status drifted")
    source_binding = _mapping(wrapper.get("source_binding"), "CPU baseline source_binding")
    _assert_evidence_matches_file(
        source_binding.get("manifest"), context_without_baseline.manifest, "CPU baseline manifest evidence"
    )
    if source_binding.get("manifest_seal_sha256") != context_without_baseline.manifest_seal_sha256:
        raise MoeCombineProbeLauncherError("CPU baseline manifest seal drifted")
    historical_admission = _mapping(
        source_binding.get("admission_current"), "CPU baseline admission evidence"
    )
    if historical_admission.get("present") is not True:
        raise MoeCombineProbeLauncherError("CPU baseline did not attest an admission pointer")
    if _canonical_from_document(
        historical_admission.get("path"), "CPU baseline admission evidence.path"
    ) != Path(str(context_without_baseline.admission_current["path"])):
        raise MoeCombineProbeLauncherError("CPU baseline admission pointer path drifted")
    if not _is_sha256(historical_admission.get("sha256")):
        raise MoeCombineProbeLauncherError("CPU baseline admission evidence lacks SHA-256")
    if (
        source_binding.get("admission_receipt_seal_sha256")
        != context_without_baseline.admission_receipt_seal_sha256
    ):
        raise MoeCombineProbeLauncherError("CPU baseline admission receipt seal drifted")
    _validate_source_top10_binding(
        source_binding.get("source_top10_binding"), context_without_baseline, "CPU baseline source top-10"
    )

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
        or inner.get("component_only") is not True
        or inner.get("routed_expert_aggregation_performed") is not True
        or inner.get("shared_expert_add_performed") is not True
        or inner.get("second_residual_performed") is not True
        or inner.get("complete_layer_or_token_performed") is not False
    ):
        raise MoeCombineProbeLauncherError("CPU baseline inner receipt is not the required CPU component")
    durable_capture = _mapping(inner.get("durable_capture"), "CPU baseline inner durable_capture")
    if durable_capture.get("receipt_written_last_is_completion_marker") is not True:
        raise MoeCombineProbeLauncherError("CPU baseline inner receipt lacks receipt-last capture")
    artifact_binding = _mapping(inner.get("artifact_binding"), "CPU baseline inner artifact_binding")
    if _canonical_from_document(
        artifact_binding.get("manifest_path"), "CPU baseline inner manifest_path"
    ) != Path(str(context_without_baseline.manifest["path"])):
        raise MoeCombineProbeLauncherError("CPU baseline inner manifest path drifted")
    if artifact_binding.get("manifest_document_sha256") != context_without_baseline.manifest["sha256"]:
        raise MoeCombineProbeLauncherError("CPU baseline inner manifest SHA-256 drifted")
    if artifact_binding.get("manifest_seal_sha256") != context_without_baseline.manifest_seal_sha256:
        raise MoeCombineProbeLauncherError("CPU baseline inner manifest seal drifted")
    if _canonical_from_document(
        artifact_binding.get("admission_current_path"), "CPU baseline inner admission_current_path"
    ) != Path(str(context_without_baseline.admission_current["path"])):
        raise MoeCombineProbeLauncherError("CPU baseline inner admission pointer path drifted")
    if (
        artifact_binding.get("admission_receipt_seal_sha256")
        != context_without_baseline.admission_receipt_seal_sha256
    ):
        raise MoeCombineProbeLauncherError("CPU baseline inner admission receipt seal drifted")
    _validate_source_top10_binding(
        inner.get("source_top10_binding"), context_without_baseline, "CPU baseline inner source top-10"
    )
    return _file_evidence(path, "--cpu-baseline-receipt"), wrapper_seal, inner_evidence


def _bind_lease(
    path: Path,
    *,
    context_without_lease: LaunchContext,
) -> tuple[dict[str, Any], str]:
    document, lease_seal = _sealed_json(path, "--lease-receipt")
    if (
        document.get("schema") != MOE_COMBINE_LEASE_SCHEMA
        or document.get("status") != MOE_COMBINE_LEASE_STATUS
    ):
        raise MoeCombineProbeLauncherError("--lease-receipt schema/status does not authorize MoE combine")
    policy = _mapping(document.get("execution_policy"), "MoE combine lease execution_policy")
    if (
        policy.get("component") != MOE_COMBINE_LEASE_COMPONENT
        or policy.get("quiet_qwen80_device_lease") is not True
        or policy.get("strict_math") is not True
        or policy.get("timing_or_benchmarking_allowed") is not False
        or policy.get("complete_layer_or_token_allowed") is not False
        or policy.get("tps_or_tg_claim_allowed") is not False
    ):
        raise MoeCombineProbeLauncherError("--lease-receipt is not a strict component-only quiet lease")
    artifact_binding = _mapping(document.get("artifact_binding"), "MoE combine lease artifact_binding")
    if artifact_binding.get("manifest_document_sha256") != context_without_lease.manifest["sha256"]:
        raise MoeCombineProbeLauncherError("MoE combine lease manifest SHA-256 drifted")
    if artifact_binding.get("manifest_seal_sha256") != context_without_lease.manifest_seal_sha256:
        raise MoeCombineProbeLauncherError("MoE combine lease manifest seal drifted")
    if (
        artifact_binding.get("admission_receipt_seal_sha256")
        != context_without_lease.admission_receipt_seal_sha256
    ):
        raise MoeCombineProbeLauncherError("MoE combine lease admission receipt seal drifted")
    _validate_source_top10_binding(
        document.get("source_top10_binding"), context_without_lease, "MoE combine lease source top-10"
    )
    baseline_binding = _mapping(document.get("cpu_baseline_binding"), "MoE combine lease CPU baseline")
    if _canonical_from_document(
        baseline_binding.get("receipt_path"), "MoE combine lease CPU baseline receipt_path"
    ) != Path(str(context_without_lease.cpu_baseline_receipt["path"])):
        raise MoeCombineProbeLauncherError("MoE combine lease CPU baseline path drifted")
    if (
        baseline_binding.get("receipt_document_sha256")
        != context_without_lease.cpu_baseline_receipt["sha256"]
    ):
        raise MoeCombineProbeLauncherError("MoE combine lease CPU baseline SHA-256 drifted")
    if (
        baseline_binding.get("schema") != CPU_BASELINE_WRAPPER_SCHEMA
        or baseline_binding.get("status") != CPU_BASELINE_WRAPPER_STATUS
        or baseline_binding.get("seal_sha256") != context_without_lease.cpu_baseline_seal_sha256
    ):
        raise MoeCombineProbeLauncherError("MoE combine lease CPU baseline identity drifted")
    return _file_evidence(path, "--lease-receipt"), lease_seal


def _validate_config(config: LaunchConfig) -> LaunchContext:
    probe = _canonical_regular(config.probe_bin, "--probe-bin", executable=True)
    if probe.name != EXPECTED_PROBE_BASENAME:
        raise MoeCombineProbeLauncherError(
            f"--probe-bin must name {EXPECTED_PROBE_BASENAME}, got {probe.name!r}"
        )
    for path, label in (
        (config.manifest, "--manifest"),
        (config.admission_current, "--admission-current"),
        (config.router_receipt, "--router-receipt"),
        (config.router_outer_receipt, "--router-outer-receipt"),
        (config.cpu_baseline_receipt, "--cpu-baseline-receipt"),
    ):
        _canonical_regular(path, label)
    _require_absolute(config.capture_dir, "--capture-dir")
    if config.mode != "metal":
        raise MoeCombineProbeLauncherError(
            "MoE-combine outer launcher is metal-only; it never reruns the sealed CPU baseline"
        )
    if config.workers < 1:
        raise MoeCombineProbeLauncherError("--workers must be positive")
    if not config.timeout_seconds > 0:
        raise MoeCombineProbeLauncherError("--timeout-seconds must be positive")
    if config.lease_receipt is None:
        raise MoeCombineProbeLauncherError("--mode metal requires --lease-receipt")
    _canonical_regular(config.lease_receipt, "--lease-receipt")

    probe_evidence = _file_evidence(probe, "--probe-bin")
    manifest, manifest_seal = _bind_manifest(config.manifest)
    admission_current, pointer_seal, admission_receipt_seal = _bind_current_admission(
        config.admission_current, manifest, manifest_seal
    )
    router_receipt, router_outer, router_outer_seal, historical_admission = _bind_upstream_router(
        manifest=manifest,
        manifest_seal_sha256=manifest_seal,
        admission_current=admission_current,
        admission_receipt_seal=admission_receipt_seal,
        router_receipt_path=config.router_receipt,
        router_outer_path=config.router_outer_receipt,
    )
    provisional = LaunchContext(
        probe_binary=probe_evidence,
        manifest=manifest,
        manifest_seal_sha256=manifest_seal,
        admission_current=admission_current,
        admission_pointer_seal_sha256=pointer_seal,
        admission_receipt_seal_sha256=admission_receipt_seal,
        router_receipt=router_receipt,
        router_outer_receipt=router_outer,
        router_outer_seal_sha256=router_outer_seal,
        router_outer_historical_admission_pointer=historical_admission,
        cpu_baseline_receipt={},
        cpu_baseline_seal_sha256="",
        cpu_inner_receipt={},
        lease_receipt={},
        lease_seal_sha256="",
    )
    baseline, baseline_seal, cpu_inner = _bind_cpu_baseline(
        config.cpu_baseline_receipt, context_without_baseline=provisional
    )
    with_baseline = replace(
        provisional,
        cpu_baseline_receipt=baseline,
        cpu_baseline_seal_sha256=baseline_seal,
        cpu_inner_receipt=cpu_inner,
    )
    lease, lease_seal = _bind_lease(config.lease_receipt, context_without_lease=with_baseline)
    return replace(with_baseline, lease_receipt=lease, lease_seal_sha256=lease_seal)


def _launch_identity(config: LaunchConfig, context: LaunchContext) -> str:
    payload = {
        "probe_binary": context.probe_binary,
        "manifest": context.manifest,
        "manifest_seal_sha256": context.manifest_seal_sha256,
        "admission_current": context.admission_current,
        "admission_pointer_seal_sha256": context.admission_pointer_seal_sha256,
        "admission_receipt_seal_sha256": context.admission_receipt_seal_sha256,
        "router_receipt": context.router_receipt,
        "router_outer_receipt": context.router_outer_receipt,
        "router_outer_seal_sha256": context.router_outer_seal_sha256,
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
    assert config.lease_receipt is not None
    return [
        str(_canonical_regular(config.probe_bin, "--probe-bin", executable=True)),
        "--manifest",
        str(_canonical_regular(config.manifest, "--manifest")),
        "--admission-current",
        str(_canonical_regular(config.admission_current, "--admission-current")),
        "--router-receipt",
        str(_canonical_regular(config.router_receipt, "--router-receipt")),
        "--router-outer-receipt",
        str(_canonical_regular(config.router_outer_receipt, "--router-outer-receipt")),
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
        or receipt.get("component_only") is not True
        or receipt.get("routed_expert_aggregation_performed") is not True
        or receipt.get("shared_expert_add_performed") is not True
        or receipt.get("second_residual_performed") is not True
        or receipt.get("complete_layer_or_token_performed") is not False
    ):
        raise MoeCombineProbeLauncherError("inner MoE-combine schema/status/scope boundary drifted")
    durable_capture = _mapping(receipt.get("durable_capture"), "inner durable_capture")
    if durable_capture.get("receipt_written_last_is_completion_marker") is not True:
        raise MoeCombineProbeLauncherError("inner receipt does not attest receipt-last capture")
    artifact_binding = _mapping(receipt.get("artifact_binding"), "inner artifact_binding")
    if _canonical_from_document(
        artifact_binding.get("manifest_path"), "inner artifact_binding.manifest_path"
    ) != Path(str(context.manifest["path"])):
        raise MoeCombineProbeLauncherError("inner manifest path drifted")
    if artifact_binding.get("manifest_document_sha256") != context.manifest["sha256"]:
        raise MoeCombineProbeLauncherError("inner manifest SHA-256 drifted")
    if artifact_binding.get("manifest_seal_sha256") != context.manifest_seal_sha256:
        raise MoeCombineProbeLauncherError("inner manifest seal drifted")
    if _canonical_from_document(
        artifact_binding.get("admission_current_path"), "inner artifact_binding.admission_current_path"
    ) != Path(str(context.admission_current["path"])):
        raise MoeCombineProbeLauncherError("inner admission pointer path drifted")
    if artifact_binding.get("admission_pointer_seal_sha256") != context.admission_pointer_seal_sha256:
        raise MoeCombineProbeLauncherError("inner admission pointer seal drifted during launch")
    if artifact_binding.get("admission_receipt_seal_sha256") != context.admission_receipt_seal_sha256:
        raise MoeCombineProbeLauncherError("inner admission receipt seal drifted")
    _validate_source_top10_binding(receipt.get("source_top10_binding"), context, "inner source top-10")

    baseline = _mapping(receipt.get("cpu_baseline_binding"), "inner cpu_baseline_binding")
    if _canonical_from_document(
        baseline.get("receipt_path"), "inner cpu_baseline_binding.receipt_path"
    ) != Path(str(context.cpu_baseline_receipt["path"])):
        raise MoeCombineProbeLauncherError("inner CPU baseline path drifted")
    if baseline.get("receipt_document_sha256") != context.cpu_baseline_receipt["sha256"]:
        raise MoeCombineProbeLauncherError("inner CPU baseline SHA-256 drifted")
    if (
        baseline.get("schema") != CPU_BASELINE_WRAPPER_SCHEMA
        or baseline.get("status") != CPU_BASELINE_WRAPPER_STATUS
        or baseline.get("seal_sha256") != context.cpu_baseline_seal_sha256
    ):
        raise MoeCombineProbeLauncherError("inner CPU baseline identity drifted")

    policy = _mapping(receipt.get("metal_execution_policy"), "inner metal_execution_policy")
    if (
        policy.get("strict_math_required") is not True
        or policy.get("timing_or_benchmarking_allowed") is not False
        or policy.get("complete_layer_or_token_allowed") is not False
        or policy.get("tps_or_tg_claim_allowed") is not False
    ):
        raise MoeCombineProbeLauncherError("inner Metal policy drifted")
    lease = _mapping(policy.get("lease_binding"), "inner metal_execution_policy.lease_binding")
    if _canonical_from_document(
        lease.get("receipt_path"), "inner lease_binding.receipt_path"
    ) != Path(str(context.lease_receipt["path"])):
        raise MoeCombineProbeLauncherError("inner lease receipt path drifted")
    if lease.get("receipt_document_sha256") != context.lease_receipt["sha256"]:
        raise MoeCombineProbeLauncherError("inner lease receipt SHA-256 drifted")
    if (
        lease.get("schema") != MOE_COMBINE_LEASE_SCHEMA
        or lease.get("status") != MOE_COMBINE_LEASE_STATUS
        or lease.get("seal_sha256") != context.lease_seal_sha256
    ):
        raise MoeCombineProbeLauncherError("inner lease schema/status drifted")


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
        receipt = _read_json(receipt_path, "inner MoE-combine receipt")
        evidence["receipt"] = _file_evidence(receipt_path, "inner MoE-combine receipt")
        evidence["schema"] = receipt.get("schema")
        evidence["status"] = receipt.get("status")
        evidence["mode"] = receipt.get("mode")
        evidence["metal_performed"] = receipt.get("metal_device_or_dispatch_performed")
        _validate_inner_binding(receipt, config, context)
    except MoeCombineProbeLauncherError as exc:
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
        return "REFUSED_QWEN80_MOE_COMBINE_OUTER_CHILD_SPAWN_ERROR"
    if terminal.get("timed_out"):
        return "REFUSED_QWEN80_MOE_COMBINE_OUTER_CHILD_TIMEOUT"
    if terminal.get("signal") is not None:
        return "REFUSED_QWEN80_MOE_COMBINE_OUTER_CHILD_SIGNAL"
    if terminal.get("exit_code") != 0:
        return "REFUSED_QWEN80_MOE_COMBINE_OUTER_CHILD_NONZERO"
    if inner.get("binding_valid") is not True or inner.get("status") != EXPECTED_METAL_STATUS:
        return "REFUSED_QWEN80_MOE_COMBINE_OUTER_ZERO_EXIT_WITHOUT_STRICTLY_BOUND_INNER_RECEIPT"
    return "CAPTURED_QWEN80_MOE_COMBINE_OUTER_TERMINAL_COMPONENT_ONLY"


def _terminal_success(receipt: Mapping[str, Any]) -> bool:
    return receipt.get("status") == "CAPTURED_QWEN80_MOE_COMBINE_OUTER_TERMINAL_COMPONENT_ONLY"


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
            "manifest_seal_sha256": context.manifest_seal_sha256,
            "admission_current": context.admission_current,
            "admission_pointer_seal_sha256": context.admission_pointer_seal_sha256,
            "admission_receipt_seal_sha256": context.admission_receipt_seal_sha256,
            "router_receipt": context.router_receipt,
            "router_outer_receipt": context.router_outer_receipt,
            "router_outer_seal_sha256": context.router_outer_seal_sha256,
            "router_outer_historical_admission_pointer": context.router_outer_historical_admission_pointer,
            "source_top10_ids": list(SOURCE_TOP10_IDS),
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
            "source_selected_ten_route_aggregation_shared_add_second_residual_component_only": True,
            "does_not_validate_or_promote_inner_component_parity": True,
            "does_not_execute_or_prove_all_routed_projection_waves_or_complete_layer": True,
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
        raise MoeCombineProbeLauncherError(
            f"capture directory exists without a terminal receipt: {config.capture_dir}"
        )
    receipt = _read_json(terminal_path, "outer terminal receipt")
    try:
        verify(receipt, label=str(terminal_path))
    except ValueError as exc:
        raise MoeCombineProbeLauncherError(f"outer terminal receipt is not sealed: {exc}") from exc
    if receipt.get("schema") != SCHEMA or receipt.get("launch_identity_sha256") != identity:
        raise MoeCombineProbeLauncherError("capture directory belongs to another launch identity")
    return receipt


def run_attempt(config: LaunchConfig) -> dict[str, Any]:
    """Run one reaped process group, or sealed-replay its terminal evidence."""

    context = _validate_config(config)
    identity = _launch_identity(config, context)
    if config.capture_dir.exists():
        return _replay_existing(config, identity)
    if not config.capture_dir.parent.is_dir():
        raise MoeCombineProbeLauncherError(
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
                "status": "STARTED_QWEN80_MOE_COMBINE_OUTER_ONE_SHOT",
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
                            "status": "RUNNING_QWEN80_MOE_COMBINE_OUTER_ONE_SHOT",
                            "recorded_at": _utc_now(),
                            "launch_identity_sha256": identity,
                            "pid": child_pid,
                            "parent_pid": os.getpid(),
                            "command": command,
                            "inner_capture_dir": str(config.capture_dir / INNER_CAPTURE),
                        }
                    ),
                )
            except MoeCombineProbeLauncherError as exc:
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
        router_receipt=parsed.router_receipt,
        router_outer_receipt=parsed.router_outer_receipt,
        cpu_baseline_receipt=parsed.cpu_baseline_receipt,
        lease_receipt=parsed.lease_receipt,
        capture_dir=parsed.capture_dir,
        mode=parsed.mode,
        workers=parsed.workers,
        timeout_seconds=parsed.timeout_seconds,
    )
    try:
        receipt = run_attempt(config)
    except MoeCombineProbeLauncherError as exc:
        print(json.dumps({"status": "REFUSED_QWEN80_MOE_COMBINE_OUTER_LAUNCHER_ERROR", "error": str(exc)}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0 if _terminal_success(receipt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
