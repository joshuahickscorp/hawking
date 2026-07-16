#!/usr/bin/env python3.12
"""Inert recovery staging for the pinned Doctor V5 14B/3bpw resource stop.

Only ``status``, ``stage`` and ``verify`` exist.  The module has no apply,
resume, signal, queue import, live-lock acquisition, or source-deletion path.
It can write only an atomic packet below its dedicated staged-acceleration
directory.  That packet describes a future compare-and-swap transaction but
cannot execute it.

The incident pins are deliberately single-use.  Any changed plan, cell,
request, runtime spec, checkpoints, resource-stop receipt, binary, durable
unit, or target state row needs a new review rather than inheriting this
exception.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import time
from typing import Any, Callable

import doctor_v5_aggressive_admission_policy as swap_policy
import doctor_v5_blocked_cell_recovery as legacy_recovery
import doctor_v5_remaining_scratch_ledger as scratch_ledger


ROOT = Path(__file__).resolve().parents[2]
TARGET_CELL_ID = "qwen2-5-14b__3bpw__codec-control"
SCHEMA = "hawking.doctor_v5_resource_stop_recovery_status.v1"
PACKET_SCHEMA = "hawking.doctor_v5_resource_stop_recovery_packet.v1"
SWAP_PROOF_SCHEMA = "hawking.doctor_v5_stable_swap_baseline_proof.v1"
VERSION = "2026-07-14.1"
STAGE_DIRNAME = "resource_stop_recovery_3bpw_v1"
SHA_RE = re.compile(r"[0-9a-f]{64}")
INCIDENT_SWAP_BASELINE_MB = 0.25
SWAP_SOFT_GROWTH_MB = 512.0
SWAP_SAMPLE_MINIMUM = 3
SWAP_MINIMUM_SPAN_SECONDS = 10.0
SWAP_MAXIMUM_SPAN_SECONDS = 45.0
SWAP_SAMPLE_INTERVAL_SECONDS = 5.0
SWAP_PROOF_MAX_AGE_SECONDS = 120.0
LEDGER_MAX_AGE_SECONDS = 120.0
PROJECTED_PACKED_OUTPUT_BYTES = 7_092_638_887
PROCESS_BUDGET_BYTES = 78_000_000_000
PROMOTION_FREE_RAM_RESERVE_BYTES = 8_000_000_000

EXPECTED_PLAN_SHA256 = "3d254b5f7fcc5f02b55f2a71f306f7f6852839b699fd14ab4ddf5a05dbaa0106"
EXPECTED_CELL_IDENTITY_SHA256 = "8f795db9a669d6d36b14c928187478407fd1a9a1a236eab74adbbe2589d6394f"
EXPECTED_RUNTIME_FILE_SHA256 = "2af4515eb975f8c5db71923b99fd26a2625cedfc7e863b0ec2dd7f325eb7bde3"
EXPECTED_REQUEST_SHA256 = "aaae269cc75a8270ba2a2de6630a09fe4ee70ea91cac41f389a5dd2cafed13ac"
EXPECTED_REGISTRY_SHA256 = "27c8daeef9abb20dafe7a44d080914e34ff1fea87e763963ff7a3f0bc8f5209f"
EXPECTED_ADAPTER_CHECKPOINT_FILE_SHA256 = "07341b05f84c17a41314ea5e4ca1a1e22bf566a4cb7388fecb843b2866a3c8f6"
EXPECTED_WORKER_CHECKPOINT_FILE_SHA256 = "c69184954cc0e8b59cce9b0d1a81c9e54c860d4de70506c4441867a8ab2dbf14"
EXPECTED_RESOURCE_STOP_FILE_SHA256 = "c6c3f60b57253d4f2228a63603e751ac11f8e319ea21134e1d9b3cd514af0028"
EXPECTED_RESOURCE_STOP_RECEIPT_SHA256 = "06511224e2280e8957cbab1b90e0a3866a51237dad13c213344dd33a61cbe025"
EXPECTED_TARGET_ROW_SHA256 = "1ada8560c093c1691fdb5c35e97676f9f7cf0574c5050c77b9e6528427fb817d"
EXPECTED_COMPLETED_UNITS = (
    "preflight", "metadata", "passthrough:00000", "encode:00000",
    "attest:00000", "decode:00000", "passthrough:00001",
    "encode:00001", "attest:00001", "decode:00001", "passthrough:00002",
)
EXPECTED_BINARIES = {
    "quantizer": ("69ce7e09741e84a785604863f0fff369355c94185544646059baeeb08cabf4a9", 1_452_208),
    "attestor": ("d431a04f37ee45cb899f691bc5bae913e2ad8a9271d6db94b027bbe24c85787b", 827_904),
    "decoder": ("e1cec500c39fef02a02e63ed00c7a0971484da125454e793f06e9ba37054f676", 729_472),
}
# This is the SHA-256 of the complete worker request file, not a semantic field.
EXPECTED_WORKER_REQUEST_FILE_SHA256 = "a88ef27ddedcfc74af0e3e01f1f57cff5a4a7d9ee4531b0c1838ba606888a6a1"

HEAVY_PATTERNS = (
    "doctor_v5_ultra_queue.py run", "doctor_v5_ultra_accelerated_queue.py run",
    "doctor_v5_strand_ladder_adapter.py run",
    "doctor_v5_strand_ladder_block_parallel_adapter.py run",
    "doctor_v5_strand_ladder_worker.py",
    "doctor_v5_strand_ladder_block_parallel_worker.py",
    "quantize-model", "strand-quant", "hawking-quant", "condense_ladder",
    "audit_ladder.py", "processing_queue.py run", "studio_run.py",
    "appendix_device_runner.py", "spec_tq_runner.py", "hawking-tq-device-probe",
    "hawking-tq-spec-probe", "probe-metal-rht", "native_probe.py",
    "mop_generation1_campaign.py", "generation1_cognitive_corpus.py",
    "vllm-metal", "mlx_lm", "llama-server",
)


class StageError(RuntimeError):
    """A staged recovery invariant is absent or ambiguous."""


@dataclass(frozen=True)
class Pins:
    plan_sha256: str = EXPECTED_PLAN_SHA256
    cell_identity_sha256: str = EXPECTED_CELL_IDENTITY_SHA256
    runtime_file_sha256: str = EXPECTED_RUNTIME_FILE_SHA256
    request_sha256: str = EXPECTED_REQUEST_SHA256
    registry_sha256: str = EXPECTED_REGISTRY_SHA256
    adapter_checkpoint_file_sha256: str = EXPECTED_ADAPTER_CHECKPOINT_FILE_SHA256
    worker_checkpoint_file_sha256: str = EXPECTED_WORKER_CHECKPOINT_FILE_SHA256
    worker_request_file_sha256: str = EXPECTED_WORKER_REQUEST_FILE_SHA256
    resource_stop_file_sha256: str = EXPECTED_RESOURCE_STOP_FILE_SHA256
    resource_stop_receipt_sha256: str = EXPECTED_RESOURCE_STOP_RECEIPT_SHA256
    target_row_sha256: str = EXPECTED_TARGET_ROW_SHA256
    attempts: int = 10
    projected_packed_output_bytes: int = PROJECTED_PACKED_OUTPUT_BYTES
    binaries: dict[str, tuple[str, int]] | None = None

    def binary_pins(self) -> dict[str, tuple[str, int]]:
        return EXPECTED_BINARIES if self.binaries is None else self.binaries


PRODUCTION_PINS = Pins()


@dataclass(frozen=True)
class Paths:
    root: Path
    ultra: Path
    plan: Path
    state: Path
    control: Path
    pid_file: Path
    queue_lock: Path
    heavy_lock: Path
    runtime_spec: Path
    result_dir: Path
    request: Path
    registry_snapshot: Path
    live_registry: Path
    adapter_checkpoint: Path
    worker_checkpoint: Path
    worker_request: Path
    resource_stop: Path
    result: Path
    execution_receipt: Path
    active_marker: Path
    aggressive_overlay: Path
    ledger_tool: Path
    remaining_scratch_gate_adapter: Path
    swap_policy_tool: Path
    legacy_recovery_tool: Path
    stage_root: Path
    packet: Path
    stage_lock: Path
    swap_proof_dir: Path


def production_paths(root: Path = ROOT) -> Paths:
    ultra = root / "reports/condense/doctor_v5_ultra"
    result = ultra / "results" / TARGET_CELL_ID
    stage = ultra / "staged_acceleration" / STAGE_DIRNAME
    return Paths(
        root=root, ultra=ultra, plan=ultra / "campaign_plan.json",
        state=ultra / "queue_state.json", control=ultra / "control.json",
        pid_file=ultra / "queue.pid.json", queue_lock=ultra / "queue.lock",
        heavy_lock=root / "reports/cron/studio_heavy.lock",
        runtime_spec=ultra / "runtime_specs" / f"{TARGET_CELL_ID}.json",
        result_dir=result, request=result / "request.json",
        registry_snapshot=result / "adapter_registry.json",
        live_registry=ultra / "adapter_registry.json",
        adapter_checkpoint=result / "checkpoint.json",
        worker_checkpoint=result / "strand_ladder/checkpoint.json",
        worker_request=result / "strand_ladder/request.json",
        resource_stop=result / "resource_stop.json",
        result=result / "result.json",
        execution_receipt=result / "execution_receipt.json",
        active_marker=ultra / "staged_acceleration/active_stack.json",
        aggressive_overlay=ultra / "staged_acceleration/aggressive_v2/aggressive_admission_overlay.json",
        ledger_tool=root / "tools/condense/doctor_v5_remaining_scratch_ledger.py",
        remaining_scratch_gate_adapter=root / (
            "tools/condense/doctor_v5_remaining_scratch_gate_adapter.py"),
        swap_policy_tool=root / "tools/condense/doctor_v5_aggressive_admission_policy.py",
        legacy_recovery_tool=root / "tools/condense/doctor_v5_blocked_cell_recovery.py",
        stage_root=stage, packet=stage / "recovery_packet.json",
        stage_lock=stage / "stage.lock", swap_proof_dir=stage / "swap_promotions",
    )


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: child for name, child in value.items() if name != key}


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(value: dt.datetime | None = None) -> str:
    return (value or _now()).isoformat(timespec="seconds")


def _parse_time(value: Any) -> dt.datetime:
    if not isinstance(value, str):
        raise StageError("timestamp is not a string")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise StageError(f"invalid timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise StageError("timestamp has no timezone")
    return parsed.astimezone(dt.timezone.utc)


def _identity(row: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (row.st_dev, row.st_ino, row.st_nlink, row.st_size,
            row.st_mtime_ns, row.st_ctime_ns)


def _confined(path: Path, root: Path, *, must_exist: bool = True) -> Path:
    root = Path(os.path.abspath(root))
    candidate = Path(os.path.abspath(path))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise StageError(f"path escapes confinement root: {path}") from exc
    cursor = root
    try:
        root_info = os.lstat(root)
    except OSError as exc:
        raise StageError(f"confinement root unavailable: {root}: {exc}") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise StageError(f"confinement root is not a real directory: {root}")
    for index, component in enumerate(relative.parts):
        cursor /= component
        try:
            info = os.lstat(cursor)
        except FileNotFoundError:
            if not must_exist and index == len(relative.parts) - 1:
                return candidate
            raise StageError(f"required path is missing: {cursor}")
        except OSError as exc:
            raise StageError(f"cannot inspect path: {cursor}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise StageError(f"symlink component is forbidden: {cursor}")
    return candidate


def _read_stable(path: Path, root: Path, *, maximum: int = 64 * 1024 * 1024) -> tuple[bytes, os.stat_result]:
    resolved = _confined(path, root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        before = os.lstat(resolved)
        descriptor = os.open(resolved, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise StageError(f"bound input is not a single-link regular file: {path}")
        if _identity(before) != _identity(opened):
            raise StageError(f"bound input changed while opening: {path}")
        if opened.st_size > maximum:
            raise StageError(f"bound input exceeds size ceiling: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk); total += len(chunk)
            if total > maximum:
                raise StageError(f"bound input exceeds size ceiling: {path}")
        after_fd, after_path = os.fstat(descriptor), os.lstat(resolved)
        if _identity(opened) != _identity(after_fd) or _identity(opened) != _identity(after_path):
            raise StageError(f"bound input changed while reading: {path}")
        return b"".join(chunks), opened
    except StageError:
        raise
    except OSError as exc:
        raise StageError(f"cannot read stable input {path}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, info = _read_stable(path, root)
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise StageError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StageError(f"JSON root is not an object: {path}")
    return value, {
        "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw), "identity": {
            "device": info.st_dev, "inode": info.st_ino, "links": info.st_nlink,
            "size": info.st_size, "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
        },
    }


def _artifact(path: Path, root: Path, *, maximum: int = 64 * 1024 * 1024) -> dict[str, Any]:
    raw, info = _read_stable(path, root, maximum=maximum)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw), "identity": {
                "device": info.st_dev, "inode": info.st_ino, "links": info.st_nlink,
                "size": info.st_size, "mtime_ns": info.st_mtime_ns,
                "ctime_ns": info.st_ctime_ns,
            }}


def _stable_size(path: Path, root: Path, expected_bytes: int) -> dict[str, Any]:
    resolved = _confined(path, root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        before = os.lstat(resolved)
        descriptor = os.open(resolved, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise StageError(f"checkpoint artifact is not a single-link regular file: {path}")
        if opened.st_size != expected_bytes:
            raise StageError(f"checkpoint artifact byte count changed: {path}")
        after = os.lstat(resolved)
        if _identity(before) != _identity(opened) or _identity(opened) != _identity(after):
            raise StageError(f"checkpoint artifact changed while observing: {path}")
        return {"path": str(path), "bytes": opened.st_size,
                "device": opened.st_dev, "inode": opened.st_ino,
                "mtime_ns": opened.st_mtime_ns, "ctime_ns": opened.st_ctime_ns}
    except StageError:
        raise
    except OSError as exc:
        raise StageError(f"cannot observe checkpoint artifact {path}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _semantic(document: dict[str, Any], field: str, label: str) -> None:
    if document.get(field) != _hash_value(_without(document, field)):
        raise StageError(f"{label} semantic hash mismatch")


def _process_identity(pid: Any) -> tuple[str, str] | None:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return None
    try:
        command = subprocess.run(
            ["/bin/ps", "-ww", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        started = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return (command, started) if command and started else None


def _supervisor_alive(record: dict[str, Any]) -> bool:
    try:
        _semantic(record, "pid_record_sha256", "queue owner record")
    except StageError:
        return False
    identity = _process_identity(record.get("pid"))
    return identity is not None and identity[1] == record.get("process_started") \
        and hashlib.sha256(identity[0].encode()).hexdigest() \
        == record.get("process_command_sha256")


def observe_heavy_owners() -> list[dict[str, Any]]:
    try:
        output = subprocess.run(
            ["/bin/ps", "-axo", "pid=,ppid=,lstart=,command="],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise StageError(f"heavy-owner observation failed: {exc}") from exc
    owners: list[dict[str, Any]] = []
    for line in output.splitlines():
        match = re.match(r"\s*(\d+)\s+(\d+)\s+(.{24})\s+(.*)$", line)
        if match is None:
            raise StageError("heavy-owner observation contains an unparsed row")
        pid, ppid, started, command = int(match.group(1)), int(match.group(2)), \
            match.group(3).strip(), match.group(4)
        lowered = command.lower()
        if pid == os.getpid() or not any(pattern in lowered for pattern in HEAVY_PATTERNS):
            continue
        owners.append({"pid": pid, "ppid": ppid, "process_started": started,
                       "command_sha256": hashlib.sha256(command.encode()).hexdigest()})
    return sorted(owners, key=lambda row: (row["pid"], row["command_sha256"]))


def observe_lock_holders(paths: Paths) -> list[dict[str, Any]]:
    """Read-only lsof probe; this function never attempts either live lock."""
    targets = [str(paths.queue_lock), str(paths.heavy_lock)]
    try:
        process = subprocess.run(
            ["/usr/sbin/lsof", "-nP", "-Fpcfn", "--", *targets],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StageError(f"live-lock holder observation failed: {exc}") from exc
    if process.returncode not in {0, 1}:
        raise StageError(f"live-lock holder observation exited {process.returncode}")
    holders: list[dict[str, Any]] = []
    pid: int | None = None
    command: str | None = None
    for line in process.stdout.splitlines():
        if line.startswith("p"):
            try: pid = int(line[1:])
            except ValueError as exc: raise StageError("lsof emitted invalid pid") from exc
        elif line.startswith("c"):
            command = line[1:]
        elif line.startswith("n") and line[1:] in targets:
            if pid is None:
                raise StageError("lsof lock row lacks pid")
            holders.append({"pid": pid, "command": command, "path": line[1:]})
    return sorted(holders, key=lambda row: (row["path"], row["pid"]))


def probe_resources(paths: Paths) -> dict[str, Any]:
    def command(argv: list[str]) -> str:
        try:
            return subprocess.run(argv, capture_output=True, text=True, check=True,
                                  timeout=10).stdout.strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise StageError(f"resource probe failed: {' '.join(argv)}: {exc}") from exc
    pressure_raw = command(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
    swap_raw = command(["/usr/sbin/sysctl", "-n", "vm.swapusage"])
    thermal_raw = command(["/usr/bin/pmset", "-g", "therm"])
    power_raw = command(["/usr/bin/pmset", "-g", "batt"])
    memory_raw = command(["/usr/bin/memory_pressure", "-Q"])
    try:
        pressure = int(pressure_raw)
        match = re.search(r"used\s*=\s*([0-9.]+)([MG])", swap_raw)
        if match is None:
            raise ValueError("unparsed swap output")
        swap_mb = float(match.group(1)) * (1024.0 if match.group(2) == "G" else 1.0)
        total_match = re.search(r"system has\s+(\d+)", memory_raw)
        free_match = re.search(r"free percentage:\s*(\d+)%", memory_raw)
        if total_match is None or free_match is None:
            raise ValueError("unparsed memory-pressure output")
        physical = int(total_match.group(1))
        free_percent = int(free_match.group(1))
        available = physical * free_percent // 100
        disk = os.statvfs(paths.root)
    except (OSError, ValueError) as exc:
        raise StageError(f"resource probe output is invalid: {exc}") from exc
    warning = thermal_raw.lower()
    thermal_nominal = "warning level has been recorded" in warning \
        and not any(token in warning for token in ("warning level: 1", "warning level: 2"))
    return {"sampled_at": _iso(), "memory_pressure_level": pressure,
            "swap_used_mb": round(swap_mb, 3),
            "disk_free_bytes": disk.f_bavail * disk.f_frsize,
            "thermal_nominal": thermal_nominal,
            "ac_power": "AC Power" in power_raw,
            "physical_memory_bytes": physical,
            "free_memory_percent": free_percent,
            "available_memory_bytes": available,
            "probe_commands": ["/usr/sbin/sysctl", "/usr/bin/pmset",
                               "/usr/bin/memory_pressure", "statvfs"]}


def validate_swap_proof(proof: Any, pins: Pins, *, now: dt.datetime | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(proof, dict):
        return ["stable swap proof is absent"]
    if proof.get("schema") != SWAP_PROOF_SCHEMA or proof.get("version") != VERSION:
        errors.append("stable swap proof schema/version mismatch")
    if proof.get("proof_sha256") != _hash_value(_without(proof, "proof_sha256")):
        errors.append("stable swap proof self-hash mismatch")
    if proof.get("target_cell_id") != TARGET_CELL_ID \
            or proof.get("plan_sha256") != pins.plan_sha256:
        errors.append("stable swap proof incident binding mismatch")
    baseline = proof.get("sealed_baseline_swap_mb")
    if isinstance(baseline, bool) or not isinstance(baseline, (int, float)) \
            or not math.isfinite(float(baseline)) or float(baseline) < 0 \
            or float(baseline) >= swap_policy.SWAP_ABSOLUTE_EMERGENCY_MB:
        errors.append("stable swap proof baseline is outside the finite emergency envelope")
    if proof.get("incident_baseline_swap_mb") != INCIDENT_SWAP_BASELINE_MB:
        errors.append("stable swap proof lost the original incident baseline")
    generation = proof.get("promotion_generation")
    if not isinstance(generation, dict) or generation.get("mode") \
            != "new-owner-free-direct-probe-generation" \
            or generation.get("caller_supplied") is not False \
            or generation.get("in_place_rebaseline") is not False \
            or not _valid_sha(generation.get("generation_sha256")):
        errors.append("stable swap proof promotion generation is invalid")
    elif generation.get("generation_sha256") \
            != _hash_value(_without(generation, "generation_sha256")):
        errors.append("stable swap proof promotion generation hash mismatch")
    expected_policy_sha = _hash_value(swap_policy.swap_policy())
    if proof.get("controller_policy_sha256") != expected_policy_sha:
        errors.append("stable swap proof controller policy changed")
    samples = proof.get("samples")
    parsed: list[tuple[dt.datetime, float]] = []
    if not isinstance(samples, list) or len(samples) < SWAP_SAMPLE_MINIMUM:
        errors.append("stable swap proof has too few samples")
    else:
        for row in samples:
            try:
                if not isinstance(row, dict) or set(row) != {
                        "sampled_at", "memory_pressure_level", "swap_used_mb",
                        "thermal_nominal", "ac_power", "available_memory_bytes",
                        "physical_memory_bytes", "probe_commands"}:
                    raise StageError("swap sample keys are invalid")
                when = _parse_time(row["sampled_at"])
                swap = row["swap_used_mb"]
                if row["memory_pressure_level"] != 1 \
                        or row["thermal_nominal"] is not True \
                        or row["ac_power"] is not True \
                        or isinstance(row.get("available_memory_bytes"), bool) \
                        or not isinstance(row.get("available_memory_bytes"), int) \
                        or isinstance(row.get("physical_memory_bytes"), bool) \
                        or not isinstance(row.get("physical_memory_bytes"), int) \
                        or row["physical_memory_bytes"] <= 0 \
                        or row["available_memory_bytes"] > row["physical_memory_bytes"] \
                        or row["available_memory_bytes"] \
                        < PROCESS_BUDGET_BYTES + PROMOTION_FREE_RAM_RESERVE_BYTES \
                        or isinstance(swap, bool) \
                        or not isinstance(swap, (int, float)) or not math.isfinite(float(swap)) \
                        or float(swap) < 0:
                    raise StageError("swap sample lacks normal pressure/thermal/AC/free-RAM reserve")
                if row.get("probe_commands") != ["/usr/sbin/sysctl", "/usr/bin/pmset",
                                                  "/usr/bin/memory_pressure", "statvfs"]:
                    raise StageError("swap sample did not use the direct trusted probes")
                parsed.append((when, float(swap)))
            except StageError as exc:
                errors.append(str(exc)); break
    if parsed:
        if [row[0] for row in parsed] != sorted(set(row[0] for row in parsed)):
            errors.append("stable swap samples are duplicate or unordered")
        span = (parsed[-1][0] - parsed[0][0]).total_seconds()
        if span < SWAP_MINIMUM_SPAN_SECONDS or span > SWAP_MAXIMUM_SPAN_SECONDS:
            errors.append("stable swap proof sample span is outside the bounded window")
        age = ((now or _now()) - parsed[-1][0]).total_seconds()
        if age < -5 or age > SWAP_PROOF_MAX_AGE_SECONDS:
            errors.append("stable swap proof is stale or future-dated")
        if any(value >= swap_policy.SWAP_ABSOLUTE_EMERGENCY_MB for _, value in parsed):
            errors.append("stable swap proof samples reach the absolute emergency ceiling")
        if any(parsed[index][1] > parsed[index - 1][1] + 1.0
               for index in range(1, len(parsed))):
            errors.append("stable swap proof contains rising swap")
        if isinstance(baseline, (int, float)) and not isinstance(baseline, bool) \
                and round(float(baseline), 3) != round(parsed[-1][1], 3):
            errors.append("stable swap baseline is not the final direct observation")
        if proof.get("sealed_at") != samples[-1].get("sampled_at"):
            errors.append("stable swap seal time differs from the final direct observation")
        if isinstance(generation, dict) \
                and (generation.get("samples_sha256") != _hash_value(samples)
                     or generation.get("previous_incident_baseline_swap_mb")
                     != INCIDENT_SWAP_BASELINE_MB):
            errors.append("stable swap generation sample/history binding changed")
    state = proof.get("controller_state")
    try:
        state_errors = swap_policy.validate_swap_state(
            state, sealed_baseline_swap_mb=float(baseline))
    except (TypeError, ValueError):
        state_errors = ["swap controller state is invalid"]
    errors.extend(f"stable swap proof {row}" for row in state_errors)
    if isinstance(state, dict) and parsed:
        last_epoch = parsed[-1][0].timestamp()
        if state.get("mode") != "green" or state.get("recovered_from_invalid") is not False \
                or state.get("green_streak", -1) < SWAP_SAMPLE_MINIMUM \
                or state.get("previous_swap_mb") != round(parsed[-1][1], 3) \
                or state.get("previous_sample_epoch") != last_epoch \
                or state.get("hard_until_epoch", math.inf) > last_epoch:
            errors.append("stable swap proof controller is not sealed and stably green")
    decision = proof.get("controller_decision")
    if not isinstance(decision, dict) or decision.get("mode") != "green" \
            or decision.get("allow_launch") is not True \
            or decision.get("shed_one") is not False:
        errors.append("stable swap proof decision is not green")
    if proof.get("owner_free_at_every_sample") is not True \
            or proof.get("live_locks_unheld_at_every_sample") is not True \
            or proof.get("source_deletion_permitted") is not False:
        errors.append("stable swap proof weakens owner/lock/source isolation")
    return sorted(set(errors))


def _swap_sample(resource: dict[str, Any]) -> dict[str, Any]:
    return {key: resource.get(key) for key in (
        "sampled_at", "memory_pressure_level", "swap_used_mb", "thermal_nominal",
        "ac_power", "available_memory_bytes", "physical_memory_bytes",
        "probe_commands",
    )}


def seal_swap_promotion(
        paths: Paths, pins: Pins = PRODUCTION_PINS, *,
        owner_observer: Callable[[], list[dict[str, Any]]] = observe_heavy_owners,
        lock_observer: Callable[[Paths], list[dict[str, Any]]] = observe_lock_holders,
        resource_probe: Callable[[Paths], dict[str, Any]] = probe_resources,
        sleep: Callable[[float], None] = time.sleep,
        interval_seconds: float = SWAP_SAMPLE_INTERVAL_SECONDS) -> Path:
    """Create one immutable generation from direct probes; never overwrite one."""
    if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, (int, float)) \
            or interval_seconds < 0:
        raise StageError("swap promotion sample interval is invalid")
    before_state, before_artifact = _read_json(paths.state, paths.root)
    target = before_state.get("cells", {}).get(TARGET_CELL_ID)
    _target_transition(target, pins)
    if before_state.get("supervisor_pid") is not None \
            or before_state.get("active_cells") != [] \
            or before_state.get("active_children") != {} \
            or before_state.get("status") not in {"waiting-prerequisites", "drained"}:
        raise StageError("swap promotion requires a quiescent queue state")
    samples: list[dict[str, Any]] = []
    for index in range(SWAP_SAMPLE_MINIMUM):
        if owner_observer():
            raise StageError("swap promotion refuses active heavy owners")
        if lock_observer(paths):
            raise StageError("swap promotion refuses open live campaign lock holders")
        owner_record, _ = _read_json(paths.pid_file, paths.root)
        if _supervisor_alive(owner_record):
            raise StageError("swap promotion refuses a live Doctor supervisor")
        samples.append(_swap_sample(resource_probe(paths)))
        if index + 1 < SWAP_SAMPLE_MINIMUM:
            sleep(float(interval_seconds))
    after_state, after_artifact = _read_json(paths.state, paths.root)
    if (after_state, after_artifact["sha256"]) != (before_state, before_artifact["sha256"]):
        raise StageError("queue state changed during swap promotion sampling")
    parsed_times = [_parse_time(row["sampled_at"]) for row in samples]
    # Tests may inject a zero wall-clock sleep but must still provide an exact
    # bounded timestamp sequence through their trusted fake probe.
    span = (parsed_times[-1] - parsed_times[0]).total_seconds()
    if span < SWAP_MINIMUM_SPAN_SECONDS or span > SWAP_MAXIMUM_SPAN_SECONDS:
        raise StageError("direct swap promotion samples lack the bounded time span")
    baseline = samples[-1]["swap_used_mb"]
    if isinstance(baseline, bool) or not isinstance(baseline, (int, float)):
        raise StageError("direct swap promotion final baseline is invalid")
    last_epoch = parsed_times[-1].timestamp()
    state = swap_policy._swap_state_payload(
        baseline_swap_mb=float(baseline), mode="green",
        previous_swap_mb=float(baseline), previous_sample_epoch=last_epoch,
        green_streak=SWAP_SAMPLE_MINIMUM, hard_until_epoch=last_epoch,
        last_transition_epoch=parsed_times[0].timestamp(), last_shed_epoch=None,
        recovered_from_invalid=False,
    )
    tool = _artifact(Path(__file__), paths.root)
    generation: dict[str, Any] = {
        "mode": "new-owner-free-direct-probe-generation",
        "created_at": _iso(), "created_by_tool_sha256": tool["sha256"],
        "caller_supplied": False, "in_place_rebaseline": False,
        "nonce": secrets.token_hex(32), "samples_sha256": _hash_value(samples),
        "previous_incident_baseline_swap_mb": INCIDENT_SWAP_BASELINE_MB,
    }
    generation["generation_sha256"] = _hash_value(generation)
    proof: dict[str, Any] = {
        "schema": SWAP_PROOF_SCHEMA, "version": VERSION,
        "target_cell_id": TARGET_CELL_ID, "plan_sha256": pins.plan_sha256,
        "incident_baseline_swap_mb": INCIDENT_SWAP_BASELINE_MB,
        "sealed_baseline_swap_mb": round(float(baseline), 3),
        "controller_policy_sha256": _hash_value(swap_policy.swap_policy()),
        "promotion_generation": generation, "samples": samples,
        "controller_state": state,
        "controller_decision": {"mode": "green", "allow_launch": True,
                                "shed_one": False},
        "sealed_at": samples[-1]["sampled_at"],
        "owner_free_at_every_sample": True,
        "live_locks_unheld_at_every_sample": True,
        "source_deletion_permitted": False,
    }
    proof["proof_sha256"] = _hash_value(proof)
    errors = validate_swap_proof(proof, pins, now=parsed_times[-1])
    if errors:
        raise StageError("direct swap promotion proof failed: " + "; ".join(errors))
    paths.swap_proof_dir.mkdir(parents=True, exist_ok=True)
    _confined(paths.swap_proof_dir, paths.stage_root)
    destination = paths.swap_proof_dir / f"promotion-{generation['generation_sha256']}.json"
    if destination.exists():
        raise StageError("refusing in-place swap promotion rebaseline")
    _create_json_exclusive(destination, proof, paths)
    return destination


def _load_swap_proof(paths: Paths) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not paths.swap_proof_dir.exists():
        return None, None
    directory = _confined(paths.swap_proof_dir, paths.stage_root)
    candidates: list[Path] = []
    for candidate in directory.iterdir():
        if candidate.is_symlink() or not candidate.is_file() \
                or not re.fullmatch(r"promotion-[0-9a-f]{64}\.json", candidate.name):
            raise StageError(f"swap promotion directory contains an untrusted entry: {candidate}")
        candidates.append(candidate)
    if not candidates:
        return None, None
    current_tool = _artifact(Path(__file__), paths.root)
    receipts: list[tuple[dt.datetime, dict[str, Any], dict[str, Any]]] = []
    for candidate in candidates:
        proof, artifact = _read_json(candidate, paths.root)
        generation = proof.get("promotion_generation")
        expected_name = (f"promotion-{generation.get('generation_sha256')}.json"
                         if isinstance(generation, dict) else "")
        if candidate.name != expected_name:
            raise StageError("swap promotion filename is not content-addressed")
        if not isinstance(generation, dict) \
                or generation.get("created_by_tool_sha256") != current_tool["sha256"]:
            raise StageError("swap promotion was not created by this reviewed tool generation")
        receipts.append((_parse_time(proof.get("sealed_at")), proof, artifact))
    receipts.sort(key=lambda row: row[0])
    if len(receipts) > 1 and receipts[-1][0] == receipts[-2][0]:
        raise StageError("swap promotion generations have an ambiguous latest seal")
    return receipts[-1][1], receipts[-1][2]


def _checkpoint_summary(worker: dict[str, Any], outer: dict[str, Any],
                        paths: Paths, pins: Pins) -> dict[str, Any]:
    if worker.get("schema") != scratch_ledger.CHECKPOINT_SCHEMA:
        raise StageError("worker checkpoint schema mismatch")
    plan, completed, units = worker.get("plan"), worker.get("completed_units"), worker.get("units")
    if not isinstance(plan, list) or len(plan) != len(set(plan)) \
            or not all(isinstance(row, str) for row in plan):
        raise StageError("worker checkpoint plan is invalid")
    if completed != plan[:len(completed)] or tuple(completed) != EXPECTED_COMPLETED_UNITS:
        raise StageError("worker checkpoint exact completed prefix changed")
    if not isinstance(units, dict) or set(units) != set(completed):
        raise StageError("worker checkpoint unit evidence changed")
    if outer.get("schema") != "hawking.doctor_v5_adapter_exact_resume_checkpoint.v1":
        raise StageError("adapter checkpoint schema mismatch")
    _semantic(outer, "checkpoint_sha256", "adapter checkpoint")
    if outer.get("completed_units") != completed \
            or outer.get("resume_state_sha256") != pins.worker_checkpoint_file_sha256 \
            or outer.get("request_sha256") != pins.request_sha256:
        raise StageError("adapter checkpoint no longer binds exact worker progress")
    observations: list[dict[str, Any]] = []
    references: dict[str, tuple[str, int]] = {}
    for unit in completed:
        evidence = units[unit]
        if not isinstance(evidence, dict):
            raise StageError(f"checkpoint evidence is not an object: {unit}")
        for key in ("artifact", "archive"):
            row = evidence.get(key)
            if row is None:
                continue
            if not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"} \
                    or not isinstance(row.get("path"), str) \
                    or not _valid_sha(row.get("sha256")) \
                    or isinstance(row.get("bytes"), bool) \
                    or not isinstance(row.get("bytes"), int) or row["bytes"] < 0:
                raise StageError(f"checkpoint artifact reference is invalid: {unit}/{key}")
            resolved = _confined(Path(row["path"]), paths.result_dir)
            prior = references.get(str(resolved))
            identity = (row["sha256"], row["bytes"])
            if prior is not None and prior != identity:
                raise StageError("checkpoint aliases one path with conflicting identities")
            references[str(resolved)] = identity
    for path, (digest, size) in sorted(references.items()):
        observations.append({**_stable_size(Path(path), paths.root, size),
                             "checkpoint_sha256": digest})
    required = {
        "encode:00000", "attest:00000", "decode:00000",
        "encode:00001", "attest:00001", "decode:00001", "passthrough:00002",
    }
    if not required <= set(completed) or "encode:00002" in completed:
        raise StageError("durable shard boundary changed")
    return {"completed_units": completed,
            "completed_units_sha256": _hash_value(completed),
            "last_completed_unit": completed[-1], "next_unit": plan[len(completed)],
            "durable_complete_shards": [0, 1], "passthrough_only_shards": [2],
            "artifact_identity_observations": observations,
            "checkpoint_payloads_content_rehashed": False}


def _target_transition(row: dict[str, Any], pins: Pins) -> tuple[dict[str, Any], dict[str, Any]]:
    if _hash_value(row) != pins.target_row_sha256:
        raise StageError("target state row differs from the pinned resource-stop incident")
    expected = {
        "status": "blocked-execution", "attempts": pins.attempts,
        "last_exit_code": 75,
        "error": "resource-stop ceiling reached: its residency alone reaches the process RAM budget as the sole live lane",
        "blockers": ["resource-stop ceiling reached: its residency alone reaches the process RAM budget as the sole live lane"],
        "runtime_spec_sha256": pins.runtime_file_sha256,
        "request_sha256": pins.request_sha256,
        "registry_sha256": pins.registry_sha256,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise StageError("target state row incident fields changed")
    after = copy.deepcopy(row)
    patch = {"status": "pending", "blockers": [], "error": None,
             "last_exit_code": None}
    after.update(patch)
    changed = {key for key in set(row) | set(after) if row.get(key) != after.get(key)}
    if changed != set(patch) or after.get("attempts") != pins.attempts:
        raise StageError("proposed transition exceeds the four allowed target fields")
    return after, patch


def inspect_legacy_4bpw(paths: Paths, owners: list[dict[str, Any]]) -> dict[str, Any]:
    return legacy_recovery.inspect_recovery(
        legacy_recovery.production_paths(paths.root), legacy_recovery.PRODUCTION_PINS,
        full=False, owner_observer=lambda: owners, probe_locks=False,
    )


def inspect(paths: Paths, pins: Pins = PRODUCTION_PINS, *,
            owner_observer: Callable[[], list[dict[str, Any]]] = observe_heavy_owners,
            lock_observer: Callable[[Paths], list[dict[str, Any]]] = observe_lock_holders,
            resource_probe: Callable[[Paths], dict[str, Any]] = probe_resources,
            ledger_builder: Callable[..., dict[str, Any]] = scratch_ledger.build_ledger,
            legacy_inspector: Callable[[Paths, list[dict[str, Any]]], dict[str, Any]]
            = inspect_legacy_4bpw,
            now: dt.datetime | None = None) -> dict[str, Any]:
    errors: list[str] = []
    blockers: list[str] = []
    docs: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    required = {
        "plan": paths.plan, "state": paths.state, "control": paths.control,
        "pid_file": paths.pid_file, "runtime_spec": paths.runtime_spec,
        "request": paths.request, "registry_snapshot": paths.registry_snapshot,
        "live_registry": paths.live_registry,
        "adapter_checkpoint": paths.adapter_checkpoint,
        "worker_checkpoint": paths.worker_checkpoint,
        "worker_request": paths.worker_request, "resource_stop": paths.resource_stop,
        "active_marker": paths.active_marker, "aggressive_overlay": paths.aggressive_overlay,
    }
    try:
        for name, path in required.items():
            docs[name], bindings[name] = _read_json(path, paths.root)
        bindings["recovery_tool"] = _artifact(Path(__file__), paths.root)
        bindings["remaining_scratch_tool"] = _artifact(paths.ledger_tool, paths.root)
        bindings["remaining_scratch_gate_adapter"] = _artifact(
            paths.remaining_scratch_gate_adapter, paths.root)
        bindings["swap_policy_tool"] = _artifact(paths.swap_policy_tool, paths.root)
        bindings["legacy_4bpw_recovery_tool"] = _artifact(
            paths.legacy_recovery_tool, paths.root)
    except StageError as exc:
        errors.append(str(exc))
        return {"schema": SCHEMA, "version": VERSION, "target_cell_id": TARGET_CELL_ID,
                "inspected_at": _iso(now), "structurally_ready": False,
                "future_commit_gates_ready": False, "activation_permitted": False,
                "errors": errors, "blockers": ["required input is unreadable"],
                "apply_implementation_present": False,
                "source_deletion_permitted": False}
    try:
        plan, state, control = docs["plan"], docs["state"], docs["control"]
        runtime, request = docs["runtime_spec"], docs["request"]
        registry, live_registry = docs["registry_snapshot"], docs["live_registry"]
        outer, worker = docs["adapter_checkpoint"], docs["worker_checkpoint"]
        worker_request, stop = docs["worker_request"], docs["resource_stop"]
        _semantic(plan, "plan_sha256", "campaign plan")
        _semantic(state, "state_sha256", "queue state")
        _semantic(control, "control_sha256", "queue control")
        _semantic(request, "request_sha256", "adapter request")
        _semantic(registry, "registry_sha256", "registry snapshot")
        _semantic(live_registry, "registry_sha256", "live registry")
        _semantic(stop, "receipt_sha256", "resource-stop receipt")
        if plan.get("plan_sha256") != pins.plan_sha256 \
                or state.get("plan_sha256") != pins.plan_sha256 \
                or control.get("plan_sha256") != pins.plan_sha256:
            raise StageError("plan generation differs from the pinned incident")
        if bindings["runtime_spec"]["sha256"] != pins.runtime_file_sha256:
            raise StageError("runtime spec file changed")
        if request.get("request_sha256") != pins.request_sha256 \
                or request.get("registry_sha256") != pins.registry_sha256 \
                or registry.get("registry_sha256") != pins.registry_sha256 \
                or live_registry.get("registry_sha256") != pins.registry_sha256:
            raise StageError("request or registry generation changed")
        pin_files = {
            "adapter_checkpoint": pins.adapter_checkpoint_file_sha256,
            "worker_checkpoint": pins.worker_checkpoint_file_sha256,
            "worker_request": pins.worker_request_file_sha256,
            "resource_stop": pins.resource_stop_file_sha256,
        }
        for name, expected in pin_files.items():
            if bindings[name]["sha256"] != expected:
                raise StageError(f"{name.replace('_', ' ')} file changed")
        if stop.get("receipt_sha256") != pins.resource_stop_receipt_sha256 \
                or stop.get("cell_id") != TARGET_CELL_ID \
                or stop.get("cell_identity_sha256") != pins.cell_identity_sha256 \
                or stop.get("plan_sha256") != pins.plan_sha256 \
                or stop.get("request_sha256") != pins.request_sha256 \
                or stop.get("reason") != "system_memory_pressure_or_swap" \
                or stop.get("resume_policy") != "retry_exact_checkpoint_after_resource_gate_recovers" \
                or stop.get("parent_source_deleted") is not False:
            raise StageError("resource-stop receipt incident binding changed")
        stop_checkpoint = stop.get("checkpoint")
        if not isinstance(stop_checkpoint, dict) \
                or stop_checkpoint.get("sha256") != pins.adapter_checkpoint_file_sha256 \
                or stop_checkpoint.get("bytes") != bindings["adapter_checkpoint"]["bytes"]:
            raise StageError("resource-stop receipt checkpoint binding changed")
        cells = plan.get("cells")
        plan_cell = next((row for row in cells if isinstance(row, dict)
                          and row.get("cell_id") == TARGET_CELL_ID), None) \
            if isinstance(cells, list) else None
        if not isinstance(plan_cell, dict) \
                or plan_cell.get("cell_identity_sha256") != pins.cell_identity_sha256 \
                or plan_cell.get("projected_output_bytes") != pins.projected_packed_output_bytes \
                or plan_cell.get("source_deletion_permitted") is not False \
                or plan_cell.get("lifecycle", {}).get("parent_source_cleanup") \
                != "disabled_separate_operator_action_only":
            raise StageError("plan cell identity or lifecycle contract changed")
        state_cells = state.get("cells")
        if not isinstance(state_cells, dict) or not isinstance(state_cells.get(TARGET_CELL_ID), dict):
            raise StageError("target state row is absent")
        target_row = state_cells[TARGET_CELL_ID]
        proposed_row, patch = _target_transition(target_row, pins)
        checkpoint = _checkpoint_summary(worker, outer, paths, pins)
        if worker.get("request_sha256") != pins.worker_request_file_sha256:
            raise StageError("worker checkpoint request binding changed")
        if paths.result.exists() or paths.execution_receipt.exists():
            raise StageError("terminal result evidence already exists")
        roles: dict[str, dict[str, Any]] = {}
        inputs = runtime.get("inputs")
        if not isinstance(inputs, list):
            raise StageError("runtime input inventory is invalid")
        for row in inputs:
            role = row.get("role") if isinstance(row, dict) else None
            if not isinstance(role, str) or role in roles:
                raise StageError("runtime input roles are invalid or duplicated")
            roles[role] = row
        for role, (digest, size) in pins.binary_pins().items():
            row = roles.get(role)
            if not isinstance(row, dict) or row.get("sha256") != digest \
                    or row.get("bytes") != size:
                raise StageError(f"runtime binary declaration changed: {role}")
            binary = _artifact(Path(row["path"]), paths.root, maximum=16 * 1024 * 1024)
            if (binary["sha256"], binary["bytes"]) != (digest, size):
                raise StageError(f"runtime binary changed: {role}")
            bindings[f"binary:{role}"] = binary
        if worker_request.get("campaign_binding", {}).get("cell_id") != TARGET_CELL_ID \
                or worker_request.get("output_root") != str(paths.worker_request.parent):
            raise StageError("worker request target/output binding changed")
        ledger = ledger_builder(
            paths.worker_request,
            projected_packed_output_bytes=pins.projected_packed_output_bytes,
            workspace_root=paths.root,
        )
        ledger_errors = scratch_ledger.validate_receipt(ledger)
        if ledger_errors:
            raise StageError("remaining-scratch ledger invalid: " + "; ".join(ledger_errors))
        if ledger.get("checkpoint", {}).get("sha256") != pins.worker_checkpoint_file_sha256 \
                or ledger.get("request", {}).get("sha256") != pins.worker_request_file_sha256:
            raise StageError("remaining-scratch ledger is not bound to the pinned checkpoint")
        if _parse_time(ledger["observed_at"]) > (now or _now()) + dt.timedelta(seconds=5):
            raise StageError("remaining-scratch ledger is future-dated")
        # The currently activated queue still passes the frozen 48 GB runtime
        # scratch budget into its base resource gate.  It subtracts produced
        # packed archives, but has no source-bound consumer for this ledger's
        # durable reconstruction credit.  A row reset alone would therefore
        # immediately re-block.  This remains a hard generation blocker, not a
        # local staging override.
        active_generation_required_free = (
            ledger["disk_reserve_bytes"] + ledger["declared_total_scratch_bytes"]
            + ledger["projected_remaining_packed_output_bytes"]
        )
        queue_compatibility = {
            "live_queue_remaining_scratch_consumption_absent": True,
            "active_queue_uses_full_declared_scratch_bytes": True,
            "active_queue_phase_aware_receipt_consumer_bound": False,
            "separate_default_off_source_bound_adapter_present": True,
            "separate_adapter_wired_into_live_queue": False,
            "active_generation_required_free_bytes": active_generation_required_free,
            "phase_aware_required_free_bytes": ledger["required_free_bytes"],
            "double_charged_durable_materialized_bytes": ledger[
                "durable_materialized_bytes"],
            "required_future_generation": (
                "separately reviewed source-bound queue generation must validate "
                "and consume this exact ledger receipt before any 3bpw CAS"
            ),
            "bypass_permitted": False,
        }
        proof, proof_artifact = _load_swap_proof(paths)
        proof_errors = validate_swap_proof(proof, pins, now=now)
        if proof_artifact is not None:
            bindings["stable_swap_proof"] = proof_artifact
        try:
            owners = owner_observer()
            lock_holders = lock_observer(paths)
            resources = resource_probe(paths)
        except StageError:
            raise
        except Exception as exc:
            raise StageError(f"dynamic gate observer failed: {exc}") from exc
        if not isinstance(owners, list) or any(not isinstance(row, dict) for row in owners):
            raise StageError("heavy-owner observer returned invalid data")
        if not isinstance(lock_holders, list) or any(not isinstance(row, dict) for row in lock_holders):
            raise StageError("live-lock observer returned invalid data")
        legacy_status = legacy_inspector(paths, owners)
        if not legacy_status.get("structurally_ready") \
                or legacy_status.get("target_cell_id") \
                != "qwen2-5-14b__4bpw__codec-control":
            raise StageError("separate 4bpw recovery prerequisite drifted: "
                             + "; ".join(legacy_status.get("errors", [])))
        owner_record = docs["pid_file"]
        _semantic(owner_record, "pid_record_sha256", "queue owner record")
        supervisor_alive = _supervisor_alive(owner_record)
        if supervisor_alive: blockers.append("detached Doctor supervisor is active")
        if state.get("supervisor_pid") is not None:
            blockers.append("queue state records a supervisor")
        if state.get("active_cells") != [] or state.get("active_children") != {}:
            blockers.append("queue is not child-free")
        if state.get("status") not in {"waiting-prerequisites", "drained"}:
            blockers.append("queue status is not quiescent")
        if owners: blockers.append("one or more heavy owners are active")
        if lock_holders: blockers.append("one or more live campaign locks have open holders")
        if control.get("mode") != "run": blockers.append("queue control is not already run")
        blockers.append(
            "active queue generation does not consume the phase-aware remaining-scratch receipt; CAS-only recovery would re-block"
        )
        blockers.extend(proof_errors)
        if not isinstance(resources, dict):
            raise StageError("resource probe returned invalid data")
        if resources.get("memory_pressure_level") != 1:
            blockers.append("memory pressure is not normal")
        if resources.get("thermal_nominal") is not True:
            blockers.append("thermal state is not nominal")
        if resources.get("ac_power") is not True:
            blockers.append("machine is not on AC power")
        available_ram = resources.get("available_memory_bytes")
        if isinstance(available_ram, bool) or not isinstance(available_ram, int) \
                or available_ram < PROCESS_BUDGET_BYTES + PROMOTION_FREE_RAM_RESERVE_BYTES:
            blockers.append("full process RAM budget plus promotion reserve is unavailable")
        swap_now = resources.get("swap_used_mb")
        if isinstance(swap_now, bool) or not isinstance(swap_now, (int, float)) \
                or not math.isfinite(float(swap_now)):
            blockers.append("current swap observation is invalid")
        elif proof and isinstance(proof.get("sealed_baseline_swap_mb"), (int, float)) \
                and float(swap_now) - float(proof["sealed_baseline_swap_mb"]) \
                >= SWAP_SOFT_GROWTH_MB:
            blockers.append("current swap exceeds the promoted sealed growth envelope")
        if proof and isinstance(proof.get("samples"), list) and proof["samples"]:
            last_swap = proof["samples"][-1].get("swap_used_mb")
            if isinstance(last_swap, (int, float)) and float(swap_now) > float(last_swap) + 1.0:
                blockers.append("current swap is rising beyond the sealed proof")
        disk_free = resources.get("disk_free_bytes")
        if isinstance(disk_free, bool) or not isinstance(disk_free, int) \
                or disk_free < ledger["required_free_bytes"]:
            blockers.append("phase-aware disk/lifecycle admission has a shortfall")
        non_target = {name: row for name, row in state_cells.items() if name != TARGET_CELL_ID}
        return {
            "schema": SCHEMA, "version": VERSION, "target_cell_id": TARGET_CELL_ID,
            "inspected_at": _iso(now), "structurally_ready": True,
            "future_commit_gates_ready": not blockers,
            "activation_permitted": False, "apply_implementation_present": False,
            "errors": [], "blockers": sorted(set(blockers)),
            "plan_sha256": pins.plan_sha256, "state_sha256": state.get("state_sha256"),
            "target_row": target_row, "target_row_sha256": _hash_value(target_row),
            "proposed_target_row": proposed_row,
            "proposed_target_row_sha256": _hash_value(proposed_row),
            "allowed_patch": patch, "non_target_rows_sha256": _hash_value(non_target),
            "checkpoint": checkpoint, "remaining_scratch_ledger": ledger,
            "queue_generation_admission_compatibility": queue_compatibility,
            "legacy_4bpw_recovery": {
                "target_cell_id": legacy_status["target_cell_id"],
                "structurally_ready": legacy_status["structurally_ready"],
                "target_row_sha256": legacy_status["target_row_sha256"],
                "plan_sha256": legacy_status["plan_sha256"],
                "checkpoint": legacy_status["checkpoint"],
                "bindings": legacy_status["bindings"],
                "activation_permitted": False,
            },
            "ordered_owner_free_recovery_sequence": [
                "qwen2-5-14b__4bpw__codec-control via its separately reviewed exact-checkpoint recovery",
                "re-observe all gates and state generations",
                "qwen2-5-14b__3bpw__codec-control via a future separately authorized CAS executor",
            ],
            "stable_swap_proof": proof, "resource_sample": resources,
            "supervisor_alive": supervisor_alive, "heavy_owners": owners,
            "live_lock_holders": lock_holders, "bindings": bindings,
            "source_deletion_permitted": False,
        }
    except (StageError, scratch_ledger.ScratchLedgerError, OSError, ValueError, TypeError) as exc:
        errors.append(str(exc))
        return {"schema": SCHEMA, "version": VERSION, "target_cell_id": TARGET_CELL_ID,
                "inspected_at": _iso(now), "structurally_ready": False,
                "future_commit_gates_ready": False, "activation_permitted": False,
                "errors": sorted(set(errors)), "blockers": blockers,
                "apply_implementation_present": False,
                "source_deletion_permitted": False}


def build_packet(paths: Paths, pins: Pins = PRODUCTION_PINS, **kwargs: Any) -> dict[str, Any]:
    snapshot = inspect(paths, pins, **kwargs)
    if not snapshot.get("structurally_ready"):
        raise StageError("cannot stage structurally invalid recovery: "
                         + "; ".join(snapshot.get("errors", [])))
    packet: dict[str, Any] = {
        "schema": PACKET_SCHEMA, "version": VERSION, "created_at": _iso(),
        "target_cell_id": TARGET_CELL_ID,
        "mode": "default-off-inert-staging-only",
        "activation_permitted": False, "apply_implementation_present": False,
        "future_commit_gates_ready_at_stage": snapshot["future_commit_gates_ready"],
        "blockers_at_stage": snapshot["blockers"],
        "bindings": snapshot["bindings"], "plan_sha256": pins.plan_sha256,
        "state_sha256": snapshot["state_sha256"],
        "target_row": snapshot["target_row"],
        "target_row_sha256": snapshot["target_row_sha256"],
        "non_target_rows_sha256": snapshot["non_target_rows_sha256"],
        "checkpoint": snapshot["checkpoint"],
        "legacy_4bpw_recovery": snapshot["legacy_4bpw_recovery"],
        "ordered_owner_free_recovery_sequence": snapshot[
            "ordered_owner_free_recovery_sequence"],
        "remaining_scratch_ledger": snapshot["remaining_scratch_ledger"],
        "queue_generation_admission_compatibility": snapshot[
            "queue_generation_admission_compatibility"],
        "stable_swap_proof": snapshot["stable_swap_proof"],
        "resource_sample": snapshot["resource_sample"],
        "proposed_transaction": {
            "executor_present": False, "commit_permitted_by_this_packet": False,
            "requires_new_explicit_owner_free_authorization": True,
            "compare_and_swap": {
                "state_file_sha256": snapshot["bindings"]["state"]["sha256"],
                "state_sha256": snapshot["state_sha256"],
                "target_row_sha256": snapshot["target_row_sha256"],
                "non_target_rows_sha256": snapshot["non_target_rows_sha256"],
                "checkpoint_file_sha256": pins.worker_checkpoint_file_sha256,
                "resource_stop_receipt_sha256": pins.resource_stop_receipt_sha256,
                "exclusive_queue_and_heavy_locks_required_at_future_commit": True,
            },
            "only_cell": TARGET_CELL_ID,
            "allowed_patch": snapshot["allowed_patch"],
            "attempts_before": pins.attempts, "attempts_after": pins.attempts,
            "proposed_target_row": snapshot["proposed_target_row"],
            "proposed_target_row_sha256": snapshot["proposed_target_row_sha256"],
            "rollback": {
                "restore_exact_target_row": snapshot["target_row"],
                "restore_exact_target_row_sha256": snapshot["target_row_sha256"],
                "restore_state_generation_sha256": snapshot["state_sha256"],
                "atomic_intent_and_rollback_receipts_required_before_future_commit": True,
            },
            "other_rows_mutation_permitted": False,
            "completed_evidence_mutation_permitted": False,
            "result_mutation_permitted": False, "control_mutation_permitted": False,
            "source_deletion_permitted": False,
        },
        "live_campaign_mutated": False, "runtime_defaults_changed": False,
        "source_deletion_permitted": False,
    }
    packet["packet_sha256"] = _hash_value(packet)
    return packet


def _atomic_json(path: Path, value: dict[str, Any], paths: Paths) -> None:
    parent = _confined(path.parent, paths.stage_root)
    if path.exists() and path.is_symlink():
        raise StageError(f"refusing to replace symlink: {path}")
    temporary = parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                         allow_nan=False).encode() + b"\n"
    descriptor: int | None = None
    published = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | getattr(os, "O_CLOEXEC", 0), 0o600)
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0: raise StageError("short atomic packet write")
            offset += count
        os.fsync(descriptor); os.close(descriptor); descriptor = None
        os.replace(temporary, path); published = True
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if descriptor is not None: os.close(descriptor)
        if not published:
            try: temporary.unlink()
            except FileNotFoundError: pass


def _create_json_exclusive(path: Path, value: dict[str, Any], paths: Paths) -> None:
    parent = _confined(path.parent, paths.stage_root)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                         allow_nan=False).encode() + b"\n"
    temporary = parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    descriptor: int | None = None
    published = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | getattr(os, "O_CLOEXEC", 0)
                             | getattr(os, "O_NOFOLLOW", 0), 0o600)
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0: raise StageError("short exclusive receipt write")
            offset += count
        os.fsync(descriptor); os.close(descriptor); descriptor = None
        os.link(temporary, path, follow_symlinks=False)
        published = True
        temporary.unlink()
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try: os.fsync(directory)
        finally: os.close(directory)
    except FileExistsError as exc:
        raise StageError("refusing in-place swap promotion rebaseline") from exc
    finally:
        if descriptor is not None: os.close(descriptor)
        if not published:
            try: temporary.unlink()
            except FileNotFoundError: pass


def stage(paths: Paths, pins: Pins = PRODUCTION_PINS, *,
          seal_stable_swap_generation: bool = False,
          sleep: Callable[[float], None] = time.sleep,
          interval_seconds: float = SWAP_SAMPLE_INTERVAL_SECONDS,
          **kwargs: Any) -> dict[str, Any]:
    paths.stage_root.mkdir(parents=True, exist_ok=True)
    _confined(paths.stage_root, paths.ultra / "staged_acceleration")
    # This is a private staging sentinel, never either live campaign lock.
    if os.path.lexists(paths.stage_lock):
        info = os.lstat(paths.stage_lock)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise StageError("staging sentinel is not a regular non-symlink file")
    else:
        descriptor = os.open(
            paths.stage_lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), 0o600)
        os.close(descriptor)
    if seal_stable_swap_generation:
        preliminary = inspect(paths, pins, **kwargs)
        if not preliminary.get("structurally_ready"):
            raise StageError("swap promotion preflight is structurally invalid: "
                             + "; ".join(preliminary.get("errors", [])))
        allowed = (
            "stable swap proof", "current swap exceeds the promoted sealed",
            "current swap is rising beyond the sealed proof",
        )
        unsafe = [row for row in preliminary["blockers"]
                  if not row.startswith(allowed)]
        if unsafe:
            raise StageError("swap promotion is not owner-free/quiescent: "
                             + "; ".join(unsafe))
        seal_swap_promotion(
            paths, pins,
            owner_observer=kwargs.get("owner_observer", observe_heavy_owners),
            lock_observer=kwargs.get("lock_observer", observe_lock_holders),
            resource_probe=kwargs.get("resource_probe", probe_resources),
            sleep=sleep, interval_seconds=interval_seconds,
        )
    packet = build_packet(paths, pins, **kwargs)
    _atomic_json(paths.packet, packet, paths)
    return packet


def validate_packet(packet: Any, paths: Paths, pins: Pins = PRODUCTION_PINS, *,
                    require_current: bool = True, **kwargs: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(packet, dict): return ["packet is not an object"]
    if packet.get("schema") != PACKET_SCHEMA or packet.get("version") != VERSION \
            or packet.get("target_cell_id") != TARGET_CELL_ID:
        errors.append("packet schema/version/target mismatch")
    if packet.get("packet_sha256") != _hash_value(_without(packet, "packet_sha256")):
        errors.append("packet self-hash mismatch")
    if packet.get("mode") != "default-off-inert-staging-only" \
            or packet.get("activation_permitted") is not False \
            or packet.get("apply_implementation_present") is not False:
        errors.append("packet is not irrevocably inert")
    packet_bindings = packet.get("bindings")
    if not isinstance(packet_bindings, dict):
        errors.append("packet source bindings are missing")
        packet_bindings = {}
    for name, source_path in (
            ("remaining_scratch_tool", paths.ledger_tool),
            ("remaining_scratch_gate_adapter", paths.remaining_scratch_gate_adapter)):
        try:
            current_source = _artifact(source_path, paths.root)
            staged_source = packet_bindings.get(name)
            if not isinstance(staged_source, dict) \
                    or (staged_source.get("sha256"), staged_source.get("bytes")) \
                    != (current_source["sha256"], current_source["bytes"]):
                errors.append(f"packet source binding changed: {name}")
        except StageError as exc:
            errors.append(str(exc))
    mismatch_blocker = (
        "active queue generation does not consume the phase-aware remaining-scratch receipt; CAS-only recovery would re-block"
    )
    compatibility = packet.get("queue_generation_admission_compatibility")
    staged_ledger = packet.get("remaining_scratch_ledger")
    if not isinstance(staged_ledger, dict):
        errors.append("packet remaining-scratch ledger is missing")
        staged_ledger = {}
    expected_active_required = None
    try:
        expected_active_required = (
            staged_ledger["disk_reserve_bytes"]
            + staged_ledger["declared_total_scratch_bytes"]
            + staged_ledger["projected_remaining_packed_output_bytes"])
    except (KeyError, TypeError):
        errors.append("packet cannot derive active-generation disk requirement")
    expected_compatibility = {
        "live_queue_remaining_scratch_consumption_absent": True,
        "active_queue_uses_full_declared_scratch_bytes": True,
        "active_queue_phase_aware_receipt_consumer_bound": False,
        "separate_default_off_source_bound_adapter_present": True,
        "separate_adapter_wired_into_live_queue": False,
        "active_generation_required_free_bytes": expected_active_required,
        "phase_aware_required_free_bytes": staged_ledger.get("required_free_bytes"),
        "double_charged_durable_materialized_bytes": staged_ledger.get(
            "durable_materialized_bytes"),
        "required_future_generation": (
            "separately reviewed source-bound queue generation must validate "
            "and consume this exact ledger receipt before any 3bpw CAS"
        ),
        "bypass_permitted": False,
    }
    if compatibility != expected_compatibility:
        errors.append("packet live-queue remaining-scratch incompatibility contract changed")
    blockers_at_stage = packet.get("blockers_at_stage")
    if packet.get("future_commit_gates_ready_at_stage") is not False \
            or not isinstance(blockers_at_stage, list) \
            or mismatch_blocker not in blockers_at_stage:
        errors.append("packet erased the mandatory live-queue consumption blocker")
    transaction = packet.get("proposed_transaction")
    expected_patch = {"status": "pending", "blockers": [], "error": None,
                      "last_exit_code": None}
    if not isinstance(transaction, dict) or transaction.get("executor_present") is not False \
            or transaction.get("commit_permitted_by_this_packet") is not False \
            or transaction.get("only_cell") != TARGET_CELL_ID \
            or transaction.get("allowed_patch") != expected_patch \
            or transaction.get("attempts_before") != pins.attempts \
            or transaction.get("attempts_after") != pins.attempts \
            or any(transaction.get(key) is not False for key in (
                "other_rows_mutation_permitted", "completed_evidence_mutation_permitted",
                "result_mutation_permitted", "control_mutation_permitted",
                "source_deletion_permitted")):
        errors.append("packet transaction exceeds the reviewed transition")
    else:
        before, after = packet.get("target_row"), transaction.get("proposed_target_row")
        if not isinstance(before, dict) or not isinstance(after, dict):
            errors.append("packet target transition rows are missing")
        else:
            expected_after = copy.deepcopy(before); expected_after.update(expected_patch)
            changed = {key for key in set(before) | set(after)
                       if before.get(key) != after.get(key)}
            if after != expected_after or changed != set(expected_patch) \
                    or after.get("attempts") != pins.attempts:
                errors.append("packet transition clears or changes forbidden fields")
            if transaction.get("proposed_target_row_sha256") != _hash_value(after):
                errors.append("packet proposed target hash mismatch")
        rollback = transaction.get("rollback")
        if not isinstance(rollback, dict) or rollback.get("restore_exact_target_row") != before \
                or rollback.get("restore_exact_target_row_sha256") != packet.get("target_row_sha256") \
                or rollback.get("restore_state_generation_sha256") != packet.get("state_sha256") \
                or rollback.get("atomic_intent_and_rollback_receipts_required_before_future_commit") is not True:
            errors.append("packet rollback contract is invalid")
        cas = transaction.get("compare_and_swap")
        state_binding = packet.get("bindings", {}).get("state", {}) \
            if isinstance(packet.get("bindings"), dict) else {}
        if not isinstance(cas, dict) \
                or cas.get("state_file_sha256") != state_binding.get("sha256") \
                or cas.get("state_sha256") != packet.get("state_sha256") \
                or cas.get("target_row_sha256") != packet.get("target_row_sha256") \
                or cas.get("non_target_rows_sha256") \
                != packet.get("non_target_rows_sha256") \
                or cas.get("checkpoint_file_sha256") \
                != pins.worker_checkpoint_file_sha256 \
                or cas.get("resource_stop_receipt_sha256") \
                != pins.resource_stop_receipt_sha256 \
                or cas.get("exclusive_queue_and_heavy_locks_required_at_future_commit") \
                is not True:
            errors.append("packet compare-and-swap contract is invalid")
    legacy = packet.get("legacy_4bpw_recovery")
    sequence = packet.get("ordered_owner_free_recovery_sequence")
    if not isinstance(legacy, dict) or legacy.get("target_cell_id") \
            != "qwen2-5-14b__4bpw__codec-control" \
            or legacy.get("structurally_ready") is not True \
            or legacy.get("activation_permitted") is not False:
        errors.append("packet separate 4bpw prerequisite is invalid")
    if sequence != [
            "qwen2-5-14b__4bpw__codec-control via its separately reviewed exact-checkpoint recovery",
            "re-observe all gates and state generations",
            "qwen2-5-14b__3bpw__codec-control via a future separately authorized CAS executor",
    ]:
        errors.append("packet ordered two-cell recovery sequence changed")
    ledger_errors = scratch_ledger.validate_receipt(packet.get("remaining_scratch_ledger"))
    errors.extend(f"packet {row}" for row in ledger_errors)
    proof_errors = validate_swap_proof(packet.get("stable_swap_proof"), pins)
    if packet.get("future_commit_gates_ready_at_stage") is True:
        errors.extend(proof_errors)
    else:
        if not isinstance(blockers_at_stage, list) \
                or any(not isinstance(row, str) for row in blockers_at_stage):
            errors.append("packet blocked-stage gate inventory is invalid")
        elif any(row not in blockers_at_stage for row in proof_errors):
            errors.append("packet omits a sealed-swap blocker present at stage")
    if require_current and not errors:
        snapshot = inspect(paths, pins, **kwargs)
        if not snapshot.get("structurally_ready"):
            errors.extend(snapshot.get("errors", []))
        else:
            if snapshot.get("state_sha256") != packet.get("state_sha256") \
                    or snapshot.get("target_row_sha256") != packet.get("target_row_sha256") \
                    or snapshot.get("non_target_rows_sha256") != packet.get("non_target_rows_sha256"):
                errors.append("packet is stale relative to queue state or rows")
            current_bindings = snapshot.get("bindings", {})
            for name, staged in packet.get("bindings", {}).items():
                current = current_bindings.get(name)
                if not isinstance(staged, dict) or not isinstance(current, dict) \
                        or (staged.get("sha256"), staged.get("bytes")) \
                        != (current.get("sha256"), current.get("bytes")):
                    errors.append(f"packet artifact binding changed: {name}")
            staged_ledger = packet.get("remaining_scratch_ledger", {})
            current_ledger = snapshot.get("remaining_scratch_ledger", {})
            for field in ("request", "checkpoint", "remaining_scratch_bytes",
                          "durable_materialized_bytes", "durable_attested_packed_bytes",
                          "projected_remaining_packed_output_bytes", "required_free_bytes",
                          "artifact_identity_observations"):
                if staged_ledger.get(field) != current_ledger.get(field):
                    errors.append(f"packet remaining-scratch ledger changed: {field}")
            try:
                age = (_now() - _parse_time(staged_ledger.get("observed_at"))).total_seconds()
                if age < -5 or age > LEDGER_MAX_AGE_SECONDS:
                    errors.append("packet remaining-scratch ledger is stale")
            except StageError as exc:
                errors.append(str(exc))
    return sorted(set(errors))


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    stage_cmd = commands.add_parser("stage")
    stage_cmd.add_argument("--seal-stable-swap-generation", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--packet", type=Path)
    args = parser.parse_args(argv)
    paths = production_paths()
    try:
        if args.command == "status":
            value = inspect(paths)
            _print(value); return 0 if value["structurally_ready"] else 2
        if args.command == "stage":
            value = stage(paths,
                          seal_stable_swap_generation=args.seal_stable_swap_generation)
            _print({"ok": True, "packet": str(paths.packet),
                    "packet_sha256": value["packet_sha256"],
                    "future_commit_gates_ready_at_stage": value["future_commit_gates_ready_at_stage"],
                    "activation_permitted": False, "apply_implementation_present": False,
                    "blockers": value["blockers_at_stage"], "live_campaign_mutated": False})
            return 0
        if args.command == "verify":
            packet_path = args.packet or paths.packet
            packet, _ = _read_json(_confined(packet_path, paths.stage_root), paths.root)
            integrity_errors = validate_packet(packet, paths, require_current=False)
            current_errors = (validate_packet(packet, paths, require_current=True)
                              if not integrity_errors else [])
            current = inspect(paths)
            fresh_ready = (not integrity_errors and not current_errors
                           and current.get("future_commit_gates_ready") is True)
            _print({
                "ok": not integrity_errors,
                "historical_packet_integrity_ok": not integrity_errors,
                "historical_packet_integrity_errors": integrity_errors,
                "fresh_current_bindings_ok": not current_errors,
                "fresh_current_binding_errors": current_errors,
                "fresh_current_commit_readiness": fresh_ready,
                "current_blockers": current.get("blockers", []),
                "activation_permitted": False, "apply_implementation_present": False,
            })
            return 0 if not integrity_errors else 2
    except (StageError, OSError, ValueError, TypeError) as exc:
        print(f"[doctor-v5-resource-stop-recovery-stage] {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
