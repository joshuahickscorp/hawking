#!/usr/bin/env python3
"""Read-only resource admission for one future Q30 range-bootstrap hash scan.

This is deliberately narrower than a lease issuer or outer runner.  It reads
only the supplied sealed bootstrap-preflight/binary records and small host
resource/process metadata.  It never accepts a source-root argument, stats or
opens source payloads, starts a child, or issues/consumes/releases a lease.

The successful document has the exact resource schema/status expected by the
existing receipt-last bootstrap outer preflight.  It is an observation that a
future, separately authorized single hash-scan child *could* be considered;
it is not authorization to run that child.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_bootstrap_outer_preflight as outer,
)
from lab.receipts import seal

SCHEMA = outer.RESOURCE_SCHEMA
PREPARED_STATUS = outer.RESOURCE_STATUS
REFUSED_STATUS = (
    "REFUSED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_"
    "RESOURCE_WINDOW_UNSAFE_OR_UNBOUND"
)

MAX_WINDOW_BYTES = outer.MAX_WINDOW_BYTES
PRODUCTION_SHARDS = outer.PRODUCTION_SHARDS
PRODUCTION_TENSORS = outer.PRODUCTION_TENSORS
MINIMUM_RECLAIMABLE_BYTES = 2 * 1024**3


class ResourceAdmissionError(RuntimeError):
    """Host metadata cannot support the bounded, non-inference reservation."""


@dataclass(frozen=True)
class HostSample:
    """Small, redacted host observation; command text is never emitted."""

    backend: str
    swap_used_bytes: int
    swapouts_pages: int
    reclaimable_bytes: int
    q30_capture_children: tuple[str, ...]
    q80_capture_children: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _run_read_only(command: Sequence[str]) -> str:
    """Run a short OS *reader* utility with no shell or mutation surface."""
    try:
        return subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ResourceAdmissionError(f"read-only host probe failed for {command[0]}: {exc}") from exc


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResourceAdmissionError(f"{label} must be a nonnegative integer")
    return value


def _parse_darwin_vm_stat(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in value.splitlines():
        match = re.match(r"^([^:]+):\s*([0-9]+)\.\s*$", line.strip())
        if match:
            result[match.group(1)] = int(match.group(2))
    required = ("Pages free", "Pages inactive", "Swapouts")
    missing = [field for field in required if field not in result]
    if missing:
        raise ResourceAdmissionError(f"vm_stat lacks required counters: {', '.join(missing)}")
    return result


def _parse_darwin_swap_used(value: str) -> int:
    match = re.search(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTP])", value, re.I)
    if not match:
        raise ResourceAdmissionError("sysctl vm.swapusage lacks a parsable used counter")
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
    return int(float(match.group(1)) * units[match.group(2).upper()])


def _linux_meminfo() -> dict[str, int]:
    try:
        raw = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError as exc:
        raise ResourceAdmissionError(f"cannot read /proc/meminfo: {exc}") from exc
    result: dict[str, int] = {}
    for line in raw.splitlines():
        key, _, remainder = line.partition(":")
        match = re.match(r"\s*([0-9]+)\s*kB\s*$", remainder)
        if match:
            result[key] = int(match.group(1)) * 1024
    for field in ("MemAvailable", "SwapTotal", "SwapFree"):
        if field not in result:
            raise ResourceAdmissionError(f"/proc/meminfo lacks {field}")
    return result


def _linux_swapouts_pages() -> int:
    try:
        raw = Path("/proc/vmstat").read_text(encoding="utf-8")
    except OSError as exc:
        raise ResourceAdmissionError(f"cannot read /proc/vmstat: {exc}") from exc
    for line in raw.splitlines():
        key, _, value = line.partition(" ")
        if key == "pswpout":
            try:
                return int(value.strip())
            except ValueError as exc:
                raise ResourceAdmissionError("/proc/vmstat pswpout is invalid") from exc
    raise ResourceAdmissionError("/proc/vmstat lacks pswpout")


def _capture_processes() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return redacted fingerprints of live capture children, not commands.

    We intentionally match only child-style command shapes.  A resident
    Gravity server or an editor mentioning a Qwen filename is not treated as a
    capture child; a later independent lease still controls any real launch.
    """
    q30: list[str] = []
    q80: list[str] = []
    for line in _run_read_only(("ps", "-axo", "pid=,command=")).splitlines():
        pid, separator, command = line.strip().partition(" ")
        if not separator or not pid.isdigit():
            continue
        lowered = command.lower()
        q30_child = (
            "ascension_qwen30_streamed_source_range_admission_bootstrap" in lowered
            and "bootstrap-scan" in lowered
        ) or (
            "ascension_qwen30_streamed_source_teacher_child" in lowered
            and "source-teacher" in lowered
        )
        q80_child = (
            "ascension_qwen80_" in lowered
            and "--capture-dir" in lowered
            and any(marker in lowered for marker in ("capture", "same_runtime", "strict_host"))
        )
        fingerprint = _sha256_json({"pid": int(pid), "command": command})
        if q30_child:
            q30.append(fingerprint)
        if q80_child:
            q80.append(fingerprint)
    return tuple(sorted(q30)), tuple(sorted(q80))


def collect_live_host_sample() -> HostSample:
    """Collect a local, read-only resource sample without inspecting source data."""
    q30, q80 = _capture_processes()
    system = platform.system()
    if system == "Darwin":
        vm_stat = _parse_darwin_vm_stat(_run_read_only(("vm_stat",)))
        try:
            page_size = int(_run_read_only(("sysctl", "-n", "hw.pagesize")).strip())
        except ValueError as exc:
            raise ResourceAdmissionError("sysctl hw.pagesize is invalid") from exc
        if page_size <= 0:
            raise ResourceAdmissionError("sysctl hw.pagesize must be positive")
        # Free plus inactive pages is intentionally conservative; purgeable /
        # speculative pages are not added because their accounting can overlap.
        reclaimable = (vm_stat["Pages free"] + vm_stat["Pages inactive"]) * page_size
        return HostSample(
            backend="darwin-vm_stat-sysctl-ps",
            swap_used_bytes=_parse_darwin_swap_used(
                _run_read_only(("sysctl", "-n", "vm.swapusage"))
            ),
            swapouts_pages=vm_stat["Swapouts"],
            reclaimable_bytes=reclaimable,
            q30_capture_children=q30,
            q80_capture_children=q80,
        )
    if system == "Linux":
        memory = _linux_meminfo()
        return HostSample(
            backend="linux-procfs-ps",
            swap_used_bytes=memory["SwapTotal"] - memory["SwapFree"],
            swapouts_pages=_linux_swapouts_pages(),
            reclaimable_bytes=memory["MemAvailable"],
            q30_capture_children=q30,
            q80_capture_children=q80,
        )
    raise ResourceAdmissionError(f"unsupported read-only resource backend: {system}")


def _evidence(document: outer.Document | None) -> dict[str, object]:
    if document is None:
        return {"present": False}
    return {
        "path": str(document.path),
        "raw_document_sha256": document.raw_document_sha256,
        "seal_sha256": document.seal_sha256,
    }


def _pointer(document: outer.Document) -> dict[str, str]:
    return {
        "raw_document_sha256": document.raw_document_sha256,
        "seal_sha256": document.seal_sha256,
    }


def _sample_fields(sample: HostSample) -> dict[str, object]:
    return {
        "backend": sample.backend,
        "swap_used_bytes": sample.swap_used_bytes,
        "swapouts_pages": sample.swapouts_pages,
        "reclaimable_bytes": sample.reclaimable_bytes,
        "q30_capture_child_count": len(sample.q30_capture_children),
        "q80_capture_child_count": len(sample.q80_capture_children),
        "q30_capture_child_fingerprints": list(sample.q30_capture_children),
        "q80_capture_child_fingerprints": list(sample.q80_capture_children),
    }


def _validate_sample(sample: HostSample) -> None:
    if not isinstance(sample.backend, str) or not sample.backend:
        raise ResourceAdmissionError("host sample backend is absent")
    _nonnegative_int(sample.swap_used_bytes, label="host sample swap usage")
    _nonnegative_int(sample.swapouts_pages, label="host sample swapouts")
    _nonnegative_int(sample.reclaimable_bytes, label="host sample reclaimable bytes")
    if any(len(value) != 64 for value in (*sample.q30_capture_children, *sample.q80_capture_children)):
        raise ResourceAdmissionError("host sample capture fingerprint is malformed")


def _bounded_profile(*, safety_floor_bytes: int) -> dict[str, object]:
    return {
        "exactly_one_future_non_inference_hash_scan_child": True,
        "maximum_concurrent_source_hash_scan_children": 1,
        "maximum_positioned_read_bytes": MAX_WINDOW_BYTES,
        "maximum_live_raw_bf16_windows": 1,
        "maximum_cached_raw_bf16_bytes": MAX_WINDOW_BYTES,
        "maximum_shards": PRODUCTION_SHARDS,
        "maximum_tensors": PRODUCTION_TENSORS,
        "minimum_reclaimable_bytes_required": safety_floor_bytes,
        "source_teacher_or_logits_allowed": False,
        "source_model_residency_allowed": False,
        "model_gpu_server_hcli_or_tps_allowed": False,
        "source_root_statted_or_opened": False,
    }


def build_resource_admission(
    *,
    preflight_path: Path,
    binary_path: Path,
    safety_floor_bytes: int = MINIMUM_RECLAIMABLE_BYTES,
    snapshot_provider: Callable[[], HostSample] = collect_live_host_sample,
) -> dict[str, Any]:
    """Seal a safe reservation or a refusal; no child/lease/source action exists."""
    safety_floor_bytes = _nonnegative_int(
        safety_floor_bytes, label="minimum reclaimable safety floor"
    )
    if safety_floor_bytes == 0:
        raise ResourceAdmissionError("minimum reclaimable safety floor must be positive")

    blockers: list[str] = []
    preflight: outer.Document | None = None
    binary: outer.Document | None = None
    binary_sha256: str | None = None
    try:
        preflight = outer._sealed(preflight_path, label="bootstrap preflight")
        outer._validate_preflight(preflight)
    except outer.BootstrapOuterError as exc:
        blockers.append(f"bootstrap_preflight_invalid:{exc}")

    if preflight is None:
        blockers.append("bootstrap_binary_not_evaluated_without_valid_preflight")
    else:
        try:
            binary = outer._sealed(binary_path, label="bootstrap binary")
            binary_sha256 = outer._validate_binary(binary, preflight=preflight)
        except outer.BootstrapOuterError as exc:
            blockers.append(f"bootstrap_binary_invalid:{exc}")

    before: HostSample | None = None
    after: HostSample | None = None
    if preflight is not None and binary_sha256 is not None:
        try:
            before = snapshot_provider()
            after = snapshot_provider()
            _validate_sample(before)
            _validate_sample(after)
        except ResourceAdmissionError as exc:
            blockers.append(f"read_only_resource_observation_invalid:{exc}")
        except Exception as exc:  # snapshot adapters are untrusted inputs too.
            blockers.append(f"read_only_resource_observation_failed:{type(exc).__name__}:{exc}")
    else:
        blockers.append("resource_observation_not_evaluated_without_valid_preflight_and_binary")

    if before is not None and after is not None:
        samples = (before, after)
        if any(sample.swap_used_bytes != 0 for sample in samples):
            blockers.append("nonzero_swap_used_bytes")
        swapout_delta = after.swapouts_pages - before.swapouts_pages
        if swapout_delta != 0:
            blockers.append("nonzero_swapout_pages_delta")
        measured_reclaimable = min(sample.reclaimable_bytes for sample in samples)
        if measured_reclaimable < safety_floor_bytes:
            blockers.append("reclaimable_bytes_below_explicit_safety_floor")
        if before.q30_capture_children or after.q30_capture_children:
            blockers.append("active_q30_capture_child_detected")
        if before.q80_capture_children or after.q80_capture_children:
            blockers.append("active_q80_capture_child_detected")
    else:
        swapout_delta = None
        measured_reclaimable = None

    blockers = sorted(set(blockers))
    admitted = not blockers
    observation = {
        "sample_count": 2 if before is not None and after is not None else 0,
        "before": _sample_fields(before) if before is not None else {"present": False},
        "after": _sample_fields(after) if after is not None else {"present": False},
        "swapouts_pages_delta": swapout_delta,
        "captures_observed_before_and_after": True,
        "source_root_or_payload_observed": False,
    }
    identity = _sha256_json(
        {
            "schema": SCHEMA,
            "preflight": _evidence(preflight),
            "binary_sha256": binary_sha256,
            "safety_floor_bytes": safety_floor_bytes,
            "observation": observation,
        }
    )
    return seal(
        {
            "schema": SCHEMA,
            "status": PREPARED_STATUS if admitted else REFUSED_STATUS,
            "recorded_at": _utc_now(),
            "prepared": admitted,
            "fresh_observation": before is not None and after is not None,
            "exclusive_clean_window": admitted,
            "zero_swap": before is not None
            and after is not None
            and before.swap_used_bytes == 0
            and after.swap_used_bytes == 0,
            "zero_swapouts": swapout_delta == 0,
            "resource_admitted_for_one_future_child": admitted,
            "source_payload_opened": False,
            "source_model_loaded": False,
            "gpu_server_hcli_or_tps_action": False,
            "lease_issued_or_consumed": False,
            "child_started": False,
            "swap_used_bytes": after.swap_used_bytes if after is not None else None,
            "swapouts_pages_delta": swapout_delta,
            "reclaimable_bytes": measured_reclaimable,
            "minimum_reclaimable_bytes_required": safety_floor_bytes,
            "bootstrap_binary_sha256": binary_sha256,
            "bootstrap_preflight": _pointer(preflight) if preflight is not None else {"present": False},
            "bootstrap_binary": _evidence(binary),
            "resource_window_identity_sha256": identity,
            "bounded_one_source_hash_scan_resource_profile": _bounded_profile(
                safety_floor_bytes=safety_floor_bytes
            ),
            "read_only_observation": observation,
            "blockers": blockers,
            "execution_boundary": {
                "source_root_argument_or_stat_performed": False,
                "source_payload_opened": False,
                "source_model_loaded": False,
                "capture_child_spawned": False,
                "gpu_server_hcli_or_tps_action": False,
                "lease_issued_or_consumed_or_released": False,
                "q30_or_q80_capture_child_started": False,
            },
            "claim_boundary": "Read-only resource admission only. A successful observation is not a lease, child launch, source scan, runtime admission, source-teacher, model, GPU, server, HCLI, TPS, TG, or tournament result.",
        }
    )


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise ResourceAdmissionError("--out must be a new absolute path below an existing parent")
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
    parser.add_argument("--bootstrap-preflight", type=Path, required=True)
    parser.add_argument("--bootstrap-binary", type=Path, required=True)
    parser.add_argument("--minimum-reclaimable-bytes", type=int, default=MINIMUM_RECLAIMABLE_BYTES)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = build_resource_admission(
            preflight_path=args.bootstrap_preflight,
            binary_path=args.bootstrap_binary,
            safety_floor_bytes=args.minimum_reclaimable_bytes,
        )
        _write_new(args.out, document)
    except ResourceAdmissionError as exc:
        print(f"Q30 range-bootstrap resource admission refused: {exc}")
        return 2
    print(
        json.dumps(
            {
                "output": str(args.out.resolve()),
                "status": document["status"],
                "seal_sha256": document["seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
