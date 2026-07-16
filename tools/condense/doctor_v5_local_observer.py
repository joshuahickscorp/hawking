#!/usr/bin/env python3.12
"""Trusted, read-only local observer for staged Doctor V5 admission gates.

The observer accepts paths and proposed process identities, never caller-made
process inventories, resource samples, or owner-free assertions.  It reads the
persisted elastic state while the same state lock is held, obtains process
identity from ``ps``, and samples topology, pressure, swap, power, and thermal
state directly.  It performs no model, GPU, corpus, queue, or runtime mutation.
"""
from __future__ import annotations

import datetime as dt
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OBSERVER_SCHEMA = "hawking.doctor_v5_local_observer.v1"
LOCK_LEASE_SCHEMA = "hawking.doctor_v5_local_observer_lock_lease.v1"
PROCESS_OBSERVATION_SCHEMA = "hawking.doctor_v5_local_process_observation.v1"
INVOCATION_OBSERVATION_SCHEMA = "hawking.doctor_v5_local_invocation_observation.v1"
VERSION = "2026-07-14.1"
OWNER_FIELDS = (
    "prepare_owner", "encoder_owner", "serial_finalizer_owner",
    "companion_owner",
)
TOPOLOGY_SYSCTLS = {
    "physical_cores": "hw.physicalcpu",
    "logical_cores": "hw.logicalcpu",
    "performance_cores": "hw.perflevel0.physicalcpu",
    "efficiency_cores": "hw.perflevel1.physicalcpu",
}
HEAVY_COMMAND_PATTERNS = (
    re.compile(r"doctor_v5_ultra_accelerated_queue\.py\s+run"),
    re.compile(r"doctor_v5_.*adapter\.py\s+run"),
    re.compile(r"doctor_v5_.*(?:queue|worker|supervisor)\.py(?:\s|$)"),
    re.compile(r"quantize-model(?:-[^ ]+)?(?:\s|$)"),
    re.compile(r"(?:strand-quant|hawking-quant|condense_ladder|studio_run\.py)"),
    re.compile(r"appendix_(?:device_runner|postrun)\.py.*(?:--run|--force)"),
    re.compile(r"(?:hawking-tq-(?:device|spec)-probe|tq_(?:device|spec)_probe)"),
    re.compile(r"(?:probe-metal-rht|native_probe\.py)"),
    re.compile(r"mop_generation1_campaign\.py\s+run"),
    re.compile(r"generation1_cognitive_corpus\.py"),
    # vLLM servers/experiments are heavy even when the command tail is merely
    # ``python -``.  Keep these path/entrypoint-specific so unrelated Python
    # stdin helpers and MLX utilities are not classified as heavy owners.
    re.compile(r"(?:^|\s)/[^\s]*vllm-metal(?:-cx)?/[^\s]*/python(?:3[^\s]*)?(?:\s|$)"),
    re.compile(r"(?:^|\s)vllm\s+serve(?:\s|$)"),
    re.compile(r"(?:^|\s)python(?:3[^\s]*)?\s+-m\s+vllm\.entrypoints\."),
    re.compile(r"vllm\.entrypoints\.(?:openai\.api_server|llm|launcher)"),
)
AUTHORITY_TOOL_PATHS = {
    "ps": Path("/bin/ps"),
    "lsof": Path("/usr/sbin/lsof"),
    "sysctl": Path("/usr/sbin/sysctl"),
    "pmset": Path("/usr/bin/pmset"),
}


class LocalObserverError(RuntimeError):
    """The trusted local observation could not be obtained exactly."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: child for name, child in value.items() if name != key}


def _file_reference(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    resolved = path.resolve(strict=True)
    try:
        display = str(resolved.relative_to(ROOT.resolve(strict=True)))
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


def _reference_from_raw(path: Path, raw: bytes) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        display = str(resolved.relative_to(ROOT.resolve(strict=True)))
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


def stable_artifact_reference(path: Path) -> dict[str, Any]:
    """Hash one regular file from a no-follow descriptor with stable fstat."""
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise LocalObserverError(f"observed artifact is not regular: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_stat = os.stat(path, follow_symlinks=False)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    identity_path = (
        path_stat.st_dev, path_stat.st_ino, path_stat.st_size, path_stat.st_mtime_ns,
    )
    raw = b"".join(chunks)
    if identity_before != identity_after or identity_after != identity_path \
            or len(raw) != after.st_size:
        raise LocalObserverError(f"observed artifact changed during read: {path}")
    return _reference_from_raw(path, raw)


def authority_tool_references() -> dict[str, dict[str, Any]]:
    """Bind every authority-bearing probe to its canonical system artifact."""
    references: dict[str, dict[str, Any]] = {}
    for name, expected in AUTHORITY_TOOL_PATHS.items():
        cursor = Path(expected.anchor)
        for component in expected.parts[1:]:
            cursor = cursor / component
            if cursor.is_symlink():
                raise LocalObserverError(
                    f"authority tool path contains a symlink: {name}"
                )
        try:
            resolved = expected.resolve(strict=True)
        except OSError as exc:
            raise LocalObserverError(
                f"authority tool is absent: {name}: {exc}"
            ) from exc
        if resolved != expected or not resolved.is_file():
            raise LocalObserverError(
                f"authority tool is not the exact canonical file: {name}"
            )
        references[name] = stable_artifact_reference(expected)
        if references[name].get("path") != str(expected):
            raise LocalObserverError(
                f"authority tool reference path changed: {name}"
            )
    return references


def _run(argv: list[str], *, timeout: float = 5.0) -> dict[str, Any]:
    try:
        process = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False,
        )
        return {
            "argv": argv, "returncode": process.returncode,
            "stdout": process.stdout, "stderr": process.stderr[-4000:],
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": argv, "returncode": None, "stdout": "",
            "stderr": str(exc)[-4000:], "timed_out": True,
        }
    except OSError as exc:
        return {
            "argv": argv, "returncode": None, "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}"[-4000:],
            "timed_out": False,
        }


def _normalize_start(value: str) -> str:
    return "ps-lstart:" + " ".join(value.split())


def _parse_ps_rows(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    if receipt.get("returncode") != 0 or receipt.get("timed_out") is not False:
        raise LocalObserverError("ps process inventory probe failed")
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+"
        r"(\S+\s+\S+\s+\d+\s+\d\d:\d\d:\d\d\s+\d{4})\s+(.*)$"
    )
    for line in str(receipt.get("stdout", "")).splitlines():
        if not line.strip():
            continue
        match = pattern.match(line)
        if match is None:
            raise LocalObserverError("ps process inventory contains an unparsed row")
        pid, ppid, started, command = match.groups()
        command = command.strip()
        if not command:
            continue
        row = {
            "pid": int(pid), "ppid": int(ppid),
            "start_identity": _normalize_start(started),
            "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
            "command": command,
        }
        row["process_generation_sha256"] = _hash_value({
            key: row[key] for key in ("pid", "start_identity", "command_sha256")
        })
        rows.append(row)
    return rows


def _identity_core(identity: Any) -> tuple[int | None, str | None, str | None]:
    if not isinstance(identity, dict):
        return None, None, None
    pid, started, command_sha = (
        identity.get("pid"), identity.get("start_identity"),
        identity.get("command_sha256"),
    )
    return (
        pid if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 else None,
        started if isinstance(started, str) and started else None,
        command_sha if isinstance(command_sha, str) and len(command_sha) == 64 else None,
    )


def _darwin_procargs(pid: int) -> tuple[str, list[str], dict[str, str]]:
    """Read exact executable/argv/environment through KERN_PROCARGS2."""
    if sys.platform != "darwin":
        raise LocalObserverError("exact process argv/environment requires Darwin")
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    mib = (ctypes.c_int * 3)(1, 49, pid)  # CTL_KERN, KERN_PROCARGS2, pid
    size = ctypes.c_size_t(0)
    if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0 \
            or size.value <= 4:
        raise LocalObserverError(
            f"KERN_PROCARGS2 size probe failed for PID {pid}: errno={ctypes.get_errno()}"
        )
    buffer = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0) != 0:
        raise LocalObserverError(
            f"KERN_PROCARGS2 read failed for PID {pid}: errno={ctypes.get_errno()}"
        )
    raw = bytes(buffer.raw[:size.value])
    argc = struct.unpack_from("=i", raw, 0)[0]
    if argc <= 0 or argc > 65536:
        raise LocalObserverError("KERN_PROCARGS2 argc is invalid")
    position = 4

    def next_cstring() -> str:
        nonlocal position
        end = raw.find(b"\0", position)
        if end < 0:
            raise LocalObserverError("KERN_PROCARGS2 contains an unterminated string")
        value = raw[position:end].decode("utf-8", errors="strict")
        position = end + 1
        return value

    executable = next_cstring()
    while position < len(raw) and raw[position] == 0:
        position += 1
    argv = [next_cstring() for _ in range(argc)]
    environment: dict[str, str] = {}
    while position < len(raw):
        # KERN_PROCARGS2 stores a NUL terminator after envp, followed by the
        # Darwin "apple" vector (executable_file, dyld_file, etc.).  Those are
        # loader metadata, not inherited environment authority.
        if raw[position] == 0:
            break
        item = next_cstring()
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if not key or key in environment:
            raise LocalObserverError("process environment contains an invalid/duplicate key")
        environment[key] = value
    if not executable or not argv:
        raise LocalObserverError("KERN_PROCARGS2 executable/argv is empty")
    return executable, argv, environment


def _lsof_cwd(pid: int) -> tuple[str, dict[str, Any]]:
    lsof_reference = authority_tool_references()["lsof"]
    lsof = str(AUTHORITY_TOOL_PATHS["lsof"])
    receipt = _run([lsof, "-a", "-p", str(pid), "-d", "cwd", "-Fn"])
    paths = [line[1:] for line in str(receipt.get("stdout", "")).splitlines()
             if line.startswith("n") and len(line) > 1]
    if receipt.get("returncode") != 0 or receipt.get("timed_out") is not False \
            or len(paths) != 1:
        raise LocalObserverError(f"cannot obtain exact cwd for PID {pid}")
    cwd = str(Path(paths[0]).resolve(strict=True))
    compact = {
        "tool": lsof_reference,
        "argv": receipt["argv"], "returncode": receipt["returncode"],
        "timed_out": receipt["timed_out"],
        "stdout_sha256": hashlib.sha256(
            str(receipt.get("stdout", "")).encode("utf-8")
        ).hexdigest(),
    }
    compact["receipt_sha256"] = _hash_value(compact)
    return cwd, compact


def observe_process_invocation(pid: int) -> dict[str, Any]:
    """Read executable bytes, exact argv, cwd, and environment directly."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise LocalObserverError("invocation PID is invalid")
    executable, argv, environment = _darwin_procargs(pid)
    executable_reference = stable_artifact_reference(Path(executable))
    cwd, cwd_receipt = _lsof_cwd(pid)
    environment_hashes = {
        key: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for key, value in sorted(environment.items())
    }
    value: dict[str, Any] = {
        "schema": INVOCATION_OBSERVATION_SCHEMA, "version": VERSION,
        "pid": pid, "method": "KERN_PROCARGS2+lsof-cwd+stable-executable-hash",
        "executable": executable_reference,
        "argv": argv, "argv_sha256": _hash_value(argv),
        "cwd": cwd, "cwd_receipt": cwd_receipt,
        "environment_keys": sorted(environment_hashes),
        "environment_value_sha256s": environment_hashes,
        "environment_sha256": _hash_value(environment_hashes),
    }
    value["invocation_observation_sha256"] = _hash_value(value)
    return value


def _observe_identity(identity: Any,
                      rows_by_pid: dict[int, dict[str, Any]], *,
                      include_invocation: bool = False) -> dict[str, Any]:
    pid, started, command_sha = _identity_core(identity)
    observed = rows_by_pid.get(pid) if pid is not None else None
    exact = bool(
        observed is not None and started is not None and command_sha is not None
        and observed["start_identity"] == started
        and observed["command_sha256"] == command_sha
    )
    result: dict[str, Any] = {
        "schema": PROCESS_OBSERVATION_SCHEMA, "version": VERSION,
        "requested_process_identity_sha256": (
            identity.get("process_identity_sha256")
            if isinstance(identity, dict) else None
        ),
        "requested_pid": pid, "requested_start_identity": started,
        "requested_command_sha256": command_sha,
        "pid_present": observed is not None, "exact_identity_running": exact,
        "observed_start_identity": (
            observed.get("start_identity") if observed is not None else None
        ),
        "observed_command_sha256": (
            observed.get("command_sha256") if observed is not None else None
        ),
        "observed_process_generation_sha256": (
            observed.get("process_generation_sha256")
            if observed is not None else None
        ),
    }
    result["invocation_observation"] = (
        observe_process_invocation(pid) if exact and include_invocation
        and pid is not None else None
    )
    result["observation_sha256"] = _hash_value(result)
    return result


def _descendant_rows(root_pid: int, rows: list[dict[str, Any]]) \
        -> list[dict[str, Any]]:
    """Return a PID/start/PPID-bound snapshot below one exact leased root."""
    descendants: list[dict[str, Any]] = []
    admitted = {root_pid}
    remaining = {row["pid"]: row for row in rows if row["pid"] != root_pid}
    while True:
        added = [
            row for row in remaining.values() if row.get("ppid") in admitted
        ]
        if not added:
            break
        for row in sorted(added, key=lambda item: item["pid"]):
            admitted.add(row["pid"])
            remaining.pop(row["pid"], None)
            bound = {
                key: row[key] for key in (
                    "pid", "ppid", "start_identity", "command_sha256",
                    "process_generation_sha256",
                )
            }
            bound["descendant_sha256"] = _hash_value(bound)
            descendants.append(bound)
    return descendants


def observe_process_identity(pid: int) -> dict[str, Any]:
    """Return direct ``ps`` identity fields for a PID (convenience, not authority)."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise LocalObserverError("PID is invalid")
    authority_tool_references()
    ps = str(AUTHORITY_TOOL_PATHS["ps"])
    receipt = _run([ps, "-axo", "pid=,ppid=,lstart=,command="])
    row = next((item for item in _parse_ps_rows(receipt) if item["pid"] == pid), None)
    if row is None:
        raise LocalObserverError(f"PID {pid} is not present in direct ps inventory")
    return {key: row[key] for key in (
        "pid", "start_identity", "command_sha256", "process_generation_sha256"
    )}


def _parse_swap_mb(value: str) -> float | None:
    match = re.search(r"used\s*=\s*([0-9.]+)([MGT])", value)
    if match is None:
        return None
    number, unit = float(match.group(1)), match.group(2)
    return round(number * {"M": 1.0, "G": 1024.0, "T": 1024.0 ** 2}[unit], 3)


def _thermal_green(receipt: dict[str, Any]) -> bool:
    if receipt.get("returncode") != 0:
        return False
    text = str(receipt.get("stdout", "") or receipt.get("stderr", ""))
    lowered = text.lower()
    if "no thermal warning level has been recorded" in lowered \
            and "no performance warning level has been recorded" in lowered:
        return True
    numbers = {
        key.lower(): int(value)
        for key, value in re.findall(r"([A-Za-z_]+)\s*[:=]\s*(\d+)", text)
    }
    return bool(
        {"cpu_speed_limit", "scheduler_limit", "available_cpus"}.issubset(numbers)
        and numbers["cpu_speed_limit"] >= 100
        and numbers["scheduler_limit"] >= 100
        and numbers["available_cpus"] > 0
    )


def _direct_resources() -> dict[str, Any]:
    sysctl = str(AUTHORITY_TOOL_PATHS["sysctl"])
    pmset = str(AUTHORITY_TOOL_PATHS["pmset"])
    topology_receipts = {
        field: _run([sysctl, "-n", key])
        for field, key in TOPOLOGY_SYSCTLS.items()
    }
    topology: dict[str, int | None] = {}
    for field, receipt in topology_receipts.items():
        try:
            value = int(str(receipt.get("stdout", "")).strip())
        except (TypeError, ValueError):
            value = 0
        topology[field] = value if receipt.get("returncode") == 0 and value > 0 else None
    pressure = _run([sysctl, "-n", "kern.memorystatus_vm_pressure_level"])
    swap = _run([sysctl, "-n", "vm.swapusage"])
    thermal = _run([pmset, "-g", "therm"])
    power = _run([pmset, "-g", "batt"])
    try:
        pressure_level = int(str(pressure.get("stdout", "")).strip())
    except (TypeError, ValueError):
        pressure_level = None
    swap_mb = _parse_swap_mb(str(swap.get("stdout", ""))) \
        if swap.get("returncode") == 0 else None
    thermal_green = _thermal_green(thermal)
    ac_power = "AC Power" in str(power.get("stdout", ""))
    resource: dict[str, Any] = {
        "source": "direct-local-subprocess",
        "topology": topology, "topology_receipts": topology_receipts,
        "pressure_level": pressure_level, "pressure_receipt": pressure,
        "swap_used_mb": swap_mb, "swap_receipt": swap,
        "thermal_receipt": thermal, "power_receipt": power,
        "thermal_green": thermal_green, "ac_power": ac_power,
        "probe_valid": (
            all(isinstance(value, int) and value > 0 for value in topology.values())
            and pressure_level in {1, 2, 4}
            and isinstance(swap_mb, (int, float)) and math.isfinite(float(swap_mb))
            and thermal_green and ac_power
        ),
    }
    resource["resource_sha256"] = _hash_value(resource)
    return resource


def _read_json_reference(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        reference = stable_artifact_reference(path)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference["sha256"] \
                or len(raw) != reference["bytes"]:
            raise LocalObserverError(f"observed JSON changed after stable read: {path}")
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalObserverError(f"cannot read observed JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LocalObserverError(f"observed JSON root is not an object: {path}")
    return value, reference


def _observer_receipt(state_path: Path, descriptor: int, *,
                      proposed_process_identities: Iterable[dict[str, Any]] = (),
                      extra_json_paths: dict[str, Path] | None = None,
                      extra_artifact_paths: dict[str, Path] | None = None) \
        -> dict[str, Any]:
    lock_stat = os.fstat(descriptor)
    if not stat.S_ISREG(lock_stat.st_mode):
        raise LocalObserverError("state lock descriptor is not a regular file")
    lock_path = state_path.with_name(state_path.name + ".lock")
    path_stat = os.stat(lock_path, follow_symlinks=False)
    if not stat.S_ISREG(path_stat.st_mode) \
            or (path_stat.st_dev, path_stat.st_ino) != (lock_stat.st_dev, lock_stat.st_ino):
        raise LocalObserverError("state lock path/descriptor identity differs")
    state, state_ref = _read_json_reference(state_path)
    state_sha = state.get("state_sha256")
    state_generation = state.get("state_generation")
    if not isinstance(state_sha, str) or len(state_sha) != 64 \
            or state_sha != _hash_value(_without(state, "state_sha256")) \
            or isinstance(state_generation, bool) \
            or not isinstance(state_generation, int) or state_generation < 0:
        raise LocalObserverError("persisted elastic state hash/generation is invalid")
    wall_epoch, monotonic_ns = time.time(), time.monotonic_ns()
    observer_source = _file_reference(Path(__file__))
    authority_tools = authority_tool_references()
    lock_lease: dict[str, Any] = {
        "schema": LOCK_LEASE_SCHEMA, "version": VERSION,
        "lock_path": str(lock_path.resolve(strict=True)),
        "lock_device": lock_stat.st_dev, "lock_inode": lock_stat.st_ino,
        "observer_pid": os.getpid(), "acquired_wall_epoch": wall_epoch,
        "acquired_monotonic_ns": monotonic_ns,
        "state_sha256": state_sha, "state_generation": state_generation,
        "observer_source_sha256": observer_source["sha256"],
    }
    lock_lease["lock_lease_sha256"] = _hash_value(lock_lease)

    ps = str(AUTHORITY_TOOL_PATHS["ps"])
    ps_receipt = _run([ps, "-axo", "pid=,ppid=,lstart=,command="])
    rows = _parse_ps_rows(ps_receipt)
    rows_by_pid = {row["pid"]: row for row in rows}
    owners: list[dict[str, Any]] = []
    for field in OWNER_FIELDS:
        owner = state.get(field)
        if not isinstance(owner, dict):
            continue
        observation = _observe_identity(owner.get("process_identity"), rows_by_pid)
        owners.append({
            "owner_field": field, "cell_id": owner.get("cell_id"),
            "process_identity": json.loads(json.dumps(owner.get("process_identity"))),
            "owner_lease": json.loads(json.dumps(owner.get("lease"))),
            "lease_sha256": (
                owner.get("lease", {}).get("lease_sha256")
                if isinstance(owner.get("lease"), dict) else None
            ),
            "process_observation": observation,
            "descendants": (
                _descendant_rows(
                    owner.get("process_identity", {}).get("pid"), rows
                ) if observation.get("exact_identity_running") is True else []
            ),
        })
    proposed = [
        _observe_identity(identity, rows_by_pid, include_invocation=True)
        for identity in proposed_process_identities
    ]
    heavy = []
    for row in rows:
        matches = sorted({
            pattern.pattern for pattern in HEAVY_COMMAND_PATTERNS
            if pattern.search(row["command"].lower())
        })
        if row["pid"] == os.getpid() or not matches:
            continue
        heavy.append({
            "pid": row["pid"], "ppid": row["ppid"],
            "start_identity": row["start_identity"],
            "command_sha256": row["command_sha256"],
            "process_generation_sha256": row["process_generation_sha256"],
            "matched_patterns": matches,
        })
    extras: dict[str, Any] = {}
    for label, path in sorted((extra_json_paths or {}).items()):
        if not isinstance(label, str) or not label:
            raise LocalObserverError("extra evidence label is invalid")
        value, reference = _read_json_reference(Path(path))
        extras[label] = {"reference": reference, "value": value}
    artifacts: dict[str, Any] = {}
    for label, path in sorted((extra_artifact_paths or {}).items()):
        if not isinstance(label, str) or not label:
            raise LocalObserverError("extra artifact label is invalid")
        path = Path(path)
        try:
            reference = stable_artifact_reference(path)
        except (OSError, LocalObserverError) as exc:
            raise LocalObserverError(
                f"cannot read observed artifact {path}: {exc}"
            ) from exc
        artifacts[label] = reference

    receipt: dict[str, Any] = {
        "schema": OBSERVER_SCHEMA, "version": VERSION,
        "authority": "trusted-local-observer-under-state-lock",
        "observer_source": observer_source,
        "authority_tools": authority_tools, "state_reference": state_ref,
        "state_sha256": state_sha, "state_generation": state_generation,
        "observed_wall_epoch": wall_epoch, "observed_monotonic_ns": monotonic_ns,
        "lock_lease": lock_lease,
        "process_probe": {
            "argv": ps_receipt["argv"], "returncode": ps_receipt["returncode"],
            "timed_out": ps_receipt["timed_out"],
            "stdout_sha256": hashlib.sha256(
                str(ps_receipt.get("stdout", "")).encode("utf-8")
            ).hexdigest(),
            "row_count": len(rows),
        },
        "persisted_owner_observations": owners,
        "proposed_process_observations": proposed,
        "heavy_owner_count": len(heavy), "heavy_owners": heavy,
        "resources": _direct_resources(), "extra_json": extras,
        "extra_artifacts": artifacts,
        "read_only": True, "model_or_gpu_work_attempted": False,
        "runtime_or_corpus_mutation_attempted": False,
        "stable_file_read_method": "open-fstat-before-after-no-follow",
    }
    receipt["observer_receipt_sha256"] = _hash_value(receipt)
    return receipt


def observe_under_lock(state_path: Path, descriptor: int, *,
                       proposed_process_identities: Iterable[dict[str, Any]] = (),
                       extra_json_paths: dict[str, Path] | None = None,
                       extra_artifact_paths: dict[str, Path] | None = None) \
        -> dict[str, Any]:
    """Observe persisted state/resources while the caller holds its state lock."""
    return _observer_receipt(
        Path(state_path), descriptor,
        proposed_process_identities=proposed_process_identities,
        extra_json_paths=extra_json_paths,
        extra_artifact_paths=extra_artifact_paths,
    )


def observe_with_state_lock(state_path: Path, *,
                            proposed_process_identities: Iterable[dict[str, Any]] = (),
                            extra_json_paths: dict[str, Path] | None = None,
                            extra_artifact_paths: dict[str, Path] | None = None) \
        -> dict[str, Any]:
    """Acquire the canonical state lock and return a trusted read-only receipt."""
    state_path = Path(state_path).resolve(strict=True)
    lock_path = state_path.with_name(state_path.name + ".lock")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return _observer_receipt(
            state_path, descriptor,
            proposed_process_identities=proposed_process_identities,
            extra_json_paths=extra_json_paths,
            extra_artifact_paths=extra_artifact_paths,
        )
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
