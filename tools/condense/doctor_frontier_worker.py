#!/usr/bin/env python3.12
"""Fail-closed preflight scaffold for one executable Doctor-v2 cell.

This file is deliberately *not* an experiment runner.  It has no start/daemon
command, imports no subprocess module, interprets no executor command line, and
contains no mechanism that can launch work.  It only proves that one immutable
HealerProgram/HealerCell pair would be eligible for a future in-process adapter.

Admission requires all of the following at the same instant:

* valid, executable healer ABI v2 program and cell identities;
* every operator wired to an explicitly registered in-process adapter;
* every input file confined to the workspace and verified by bytes + SHA-256;
* the exact checkpoint contract below;
* the already-held inherited Studio heavy lease, including device/inode identity;
* normal memory pressure, exactly zero swap, AC power, nominal thermal state;
* sufficient disk for the reserve plus declared scratch/output/checkpoint bytes.

The production adapter registry is intentionally empty.  Consequently this
scaffold refuses every current planned program and every executable program
until a reviewed adapter is explicitly added in source.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import sys
import tempfile
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import healer_abi as abi  # noqa: E402


PREFLIGHT_SCHEMA = "hawking.doctor_frontier_worker_preflight.v1"
EXECUTION_CONTRACT_SCHEMA = "hawking.doctor_frontier_cell_execution.v1"
CHECKPOINT_CONTRACT_SCHEMA = "hawking.healer_checkpoint_contract.v2"
HEAVY_LEASE_FD_ENV = "HAWKING_HEAVY_LEASE_FD"
ADAPTER_ARGV_MARKER = "hawking-internal-adapter"
MIN_DISK_RESERVE_BYTES = 50_000_000_000  # lowered 150->50 GB after campaign retirement (operator, 2026-07-17)
MAX_JSON_BYTES = 32 * 1024 * 1024

CHECKPOINT_REQUIRED_STATE = (
    "gradient_accumulation",
    "microstep",
    "operator_state",
    "optimizer",
    "partial_output_hashes",
    "resume_command_identity",
    "rng",
    "sampler_cursor",
    "source_shard_offset",
    "teacher_cache_identity",
)
EXACT_CHECKPOINT_CONTRACT: dict[str, Any] = {
    "schema": CHECKPOINT_CONTRACT_SCHEMA,
    "required_state": list(CHECKPOINT_REQUIRED_STATE),
    "atomic_replace": True,
    "fsync_file": True,
    "fsync_parent_directory": True,
    "validate_before_resume": True,
    "partial_outputs_hash_bound": True,
    "resource_gate_failure": "checkpoint_then_exit",
    "signal_boundary": "declared_microstep_or_operator_boundary",
}

EXECUTION_CONTRACT_KEYS = {
    "schema",
    "single_cell",
    "worker_capability",
    "heavy_lease",
    "disk_reserve_bytes",
    "checkpoint_reserve_bytes",
    "input_files",
}
HEAVY_LEASE_KEYS = {"fd_env", "path", "st_dev", "st_ino"}
INPUT_FILE_KEYS = {"binding", "path", "sha256", "bytes"}
EXECUTOR_KEYS = {"wired", "adapter_id", "source_sha256", "argv"}


@dataclass(frozen=True)
class AdapterSpec:
    """Identity-only registration for a future reviewed in-process adapter.

    There is intentionally no callable or command field.  Adding execution is a
    separate future change and cannot be inferred from an ABI ``argv`` value.
    """

    source_path: str
    source_sha256: str
    operator_kinds: frozenset[str]
    backends: frozenset[str]


# Fail closed by construction.  A future adapter needs an explicit source edit,
# source hash, operator-kind scope, backend scope, tests, and review.
ADAPTER_REGISTRY: Mapping[str, AdapterSpec] = {}


@dataclass(frozen=True)
class FileObservation:
    path: Path
    sha256: str
    bytes: int
    st_dev: int
    st_ino: int


def _gate(gate_id: str, errors: list[str], **observed: Any) -> dict[str, Any]:
    return {
        "id": gate_id,
        "ok": not errors,
        "errors": errors,
        "observed": observed,
    }


def _is_int(value: Any, *, positive: bool = False, nonnegative: bool = False) -> bool:
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    if positive and value <= 0:
        return False
    if nonnegative and value < 0:
        return False
    return True


def _confined_path(raw: Any, root: Path) -> tuple[Path | None, str | None]:
    if not isinstance(raw, str) or not raw:
        return None, "path must be a non-empty string"
    root = root.resolve()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None, f"path is missing, cyclic, or outside workspace: {raw!r}"
    if candidate.is_symlink():
        return None, f"symlink inputs are forbidden: {raw!r}"
    return resolved, None


def _observe_regular_file(raw: Any, root: Path) -> tuple[FileObservation | None, str | None]:
    path, error = _confined_path(raw, root)
    if error or path is None:
        return None, error
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        return None, f"cannot open input {path}: {exc}"
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            return None, f"input is not a regular file: {path}"
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(fd)
        stable = (
            before.st_dev == after.st_dev
            and before.st_ino == after.st_ino
            and before.st_size == after.st_size == total
            and before.st_mtime_ns == after.st_mtime_ns
        )
        if not stable:
            return None, f"input changed while hashing: {path}"
        return FileObservation(
            path=path,
            sha256=digest.hexdigest(),
            bytes=total,
            st_dev=int(after.st_dev),
            st_ino=int(after.st_ino),
        ), None
    except OSError as exc:
        return None, f"cannot hash input {path}: {exc}"
    finally:
        os.close(fd)


def _load_json(raw: Path, root: Path) -> tuple[dict[str, Any] | None, FileObservation | None, str | None]:
    observed, error = _observe_regular_file(str(raw), root)
    if error or observed is None:
        return None, None, error
    if observed.bytes > MAX_JSON_BYTES:
        return None, observed, f"JSON input exceeds {MAX_JSON_BYTES} bytes: {observed.path}"
    try:
        value = json.loads(observed.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, observed, f"invalid JSON {observed.path}: {exc}"
    if not isinstance(value, dict):
        return None, observed, f"JSON root must be an object: {observed.path}"
    return value, observed, None


def _expected_bindings(program: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    model = program.get("model") if isinstance(program.get("model"), dict) else {}
    runtime = (
        program.get("runtime_identity")
        if isinstance(program.get("runtime_identity"), dict)
        else {}
    )
    expected = {
        "program.model.parent_revision_sha256": model.get("parent_revision_sha256"),
        "program.model.config_sha256": model.get("config_sha256"),
        "program.model.tokenizer_sha256": model.get("tokenizer_sha256"),
        "program.runtime_identity.backend_kernel_sha256": runtime.get("backend_kernel_sha256"),
        "program.runtime_identity.drafter_artifact_sha256": runtime.get("drafter_artifact_sha256"),
        "program.runtime_identity.verifier_path_sha256": runtime.get("verifier_path_sha256"),
        "program.runtime_identity.kv_policy_sha256": runtime.get("kv_policy_sha256"),
        "program.runtime_identity.cache_namespace_sha256": runtime.get("cache_namespace_sha256"),
        "program.runtime_identity.adaptive_policy_sha256": runtime.get("adaptive_policy_sha256"),
        "cell.calibration_sha256": cell.get("calibration_sha256"),
        "cell.selection_sha256": cell.get("selection_sha256"),
        "cell.final_eval_sha256": cell.get("final_eval_sha256"),
        "cell.worker_source_sha256": cell.get("worker_source_sha256"),
    }
    for node in program.get("operators", []):
        if isinstance(node, dict) and isinstance(node.get("id"), str):
            executor = node.get("executor") if isinstance(node.get("executor"), dict) else {}
            expected[f"program.operator:{node['id']}.executor.source_sha256"] = executor.get(
                "source_sha256"
            )
    return expected


def _validate_executors(
    program: dict[str, Any],
    cell: dict[str, Any],
    adapters: Mapping[str, AdapterSpec],
    root: Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    errors: list[str] = []
    adapter_paths: dict[str, Path] = {}
    backend = cell.get("backend")
    nodes = program.get("operators")
    if not isinstance(nodes, list) or not nodes:
        return adapter_paths, _gate("executors", ["program has no operators"])
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"operators[{index}] is not an object")
            continue
        node_id = node.get("id")
        prefix = str(node_id) if isinstance(node_id, str) else f"operators[{index}]"
        executor = node.get("executor")
        if not isinstance(executor, dict):
            errors.append(f"{prefix}: executor missing")
            continue
        if set(executor) != EXECUTOR_KEYS:
            errors.append(f"{prefix}: executor keys must be exactly {sorted(EXECUTOR_KEYS)}")
        if executor.get("wired") is not True:
            errors.append(f"{prefix}: executor is not wired")
        adapter_id = executor.get("adapter_id")
        if not isinstance(adapter_id, str) or not adapter_id:
            errors.append(f"{prefix}: adapter_id missing")
            continue
        if executor.get("argv") != [ADAPTER_ARGV_MARKER, adapter_id]:
            errors.append(f"{prefix}: arbitrary argv rejected; only canonical adapter identity is allowed")
        spec = adapters.get(adapter_id)
        if spec is None:
            errors.append(f"{prefix}: adapter_id {adapter_id!r} is not allow-listed")
            continue
        if not abi.is_sha256(spec.source_sha256):
            errors.append(f"{prefix}: registered adapter source hash is invalid")
        if executor.get("source_sha256") != spec.source_sha256:
            errors.append(f"{prefix}: executor source hash differs from adapter registration")
        if node.get("kind") not in spec.operator_kinds:
            errors.append(f"{prefix}: adapter is not scoped for operator kind {node.get('kind')!r}")
        if backend not in spec.backends:
            errors.append(f"{prefix}: adapter is not scoped for backend {backend!r}")
        support = node.get("backend_support")
        if not isinstance(support, dict) or support.get(backend) not in {"measured", "prototype"}:
            errors.append(f"{prefix}: program does not declare executable backend support")
        if node.get("implementation_status") not in {"measured", "prototype"}:
            errors.append(f"{prefix}: implementation_status is not executable")
        source_path, source_error = _confined_path(spec.source_path, root)
        if source_error or source_path is None:
            errors.append(f"{prefix}: registered adapter source path invalid: {source_error}")
        else:
            adapter_paths[f"program.operator:{prefix}.executor.source_sha256"] = source_path
    return adapter_paths, _gate(
        "executors",
        errors,
        registered_adapter_count=len(adapters),
        operator_count=len(nodes),
    )


def _validate_inputs(
    program: dict[str, Any],
    cell: dict[str, Any],
    contract: dict[str, Any],
    adapter_paths: Mapping[str, Path],
    root: Path,
    worker_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    rows = contract.get("input_files")
    if not isinstance(rows, list) or not rows:
        return _gate("input_files", ["execution contract input_files must be non-empty"])
    expected = _expected_bindings(program, cell)
    seen: set[str] = set()
    cache: dict[Path, FileObservation] = {}
    for index, row in enumerate(rows):
        prefix = f"input_files[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if set(row) != INPUT_FILE_KEYS:
            errors.append(f"{prefix} keys must be exactly {sorted(INPUT_FILE_KEYS)}")
        binding = row.get("binding")
        if not isinstance(binding, str) or binding not in expected:
            errors.append(f"{prefix} has unknown binding {binding!r}")
            continue
        if binding in seen:
            errors.append(f"duplicate input binding {binding}")
            continue
        seen.add(binding)
        expected_sha = expected[binding]
        if not abi.is_sha256(expected_sha):
            errors.append(f"{binding}: bound ABI hash is invalid")
        if row.get("sha256") != expected_sha:
            errors.append(f"{binding}: declared input hash differs from bound ABI hash")
        declared_bytes = row.get("bytes")
        if not _is_int(declared_bytes, nonnegative=True):
            errors.append(f"{binding}: bytes must be a nonnegative integer")
        path, path_error = _confined_path(row.get("path"), root)
        if path_error or path is None:
            errors.append(f"{binding}: {path_error}")
            continue
        if binding == "cell.worker_source_sha256" and path != worker_path.resolve():
            errors.append("cell.worker_source_sha256 must bind this exact worker source")
        adapter_path = adapter_paths.get(binding)
        if adapter_path is not None and path != adapter_path:
            errors.append(f"{binding}: input path differs from registered adapter source")
        observed = cache.get(path)
        if observed is None:
            observed, observe_error = _observe_regular_file(str(path), root)
            if observe_error or observed is None:
                errors.append(f"{binding}: {observe_error}")
                continue
            cache[path] = observed
        if row.get("sha256") != observed.sha256:
            errors.append(f"{binding}: actual SHA-256 mismatch")
        if declared_bytes != observed.bytes:
            errors.append(f"{binding}: actual byte count mismatch")
    missing = sorted(set(expected) - seen)
    extra = sorted(seen - set(expected))
    if missing:
        errors.append("missing input bindings: " + ", ".join(missing))
    if extra:
        errors.append("unexpected input bindings: " + ", ".join(extra))
    dataset_hashes = {
        cell.get("calibration_sha256"), cell.get("selection_sha256"), cell.get("final_eval_sha256")
    }
    if len(dataset_hashes) != 3:
        errors.append("calibration, selection, and final-eval files must be hash-distinct")
    return _gate(
        "input_files",
        errors,
        required_binding_count=len(expected),
        observed_binding_count=len(seen),
        unique_file_count=len(cache),
    )


def _validate_execution_contract(cell: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    errors: list[str] = []
    contract = cell.get("execution_contract")
    if not isinstance(contract, dict):
        return {}, _gate("execution_contract", ["cell execution_contract missing"])
    if set(contract) != EXECUTION_CONTRACT_KEYS:
        errors.append(f"execution_contract keys must be exactly {sorted(EXECUTION_CONTRACT_KEYS)}")
    if contract.get("schema") != EXECUTION_CONTRACT_SCHEMA:
        errors.append(f"execution_contract schema must be {EXECUTION_CONTRACT_SCHEMA}")
    if contract.get("single_cell") is not True:
        errors.append("execution_contract single_cell must be true")
    if contract.get("worker_capability") != "preflight_only_no_execution":
        errors.append("worker_capability must be preflight_only_no_execution")
    reserve = contract.get("disk_reserve_bytes")
    checkpoint_reserve = contract.get("checkpoint_reserve_bytes")
    if not _is_int(reserve, nonnegative=True) or reserve < MIN_DISK_RESERVE_BYTES:
        errors.append(f"disk_reserve_bytes must be an integer >= {MIN_DISK_RESERVE_BYTES}")
    if not _is_int(checkpoint_reserve, nonnegative=True):
        errors.append("checkpoint_reserve_bytes must be a nonnegative integer")
    lease = contract.get("heavy_lease")
    if not isinstance(lease, dict) or set(lease) != HEAVY_LEASE_KEYS:
        errors.append(f"heavy_lease keys must be exactly {sorted(HEAVY_LEASE_KEYS)}")
    return contract, _gate("execution_contract", errors)


def _validate_checkpoint_contract(cell: dict[str, Any]) -> dict[str, Any]:
    actual = cell.get("checkpoint_contract")
    errors = [] if actual == EXACT_CHECKPOINT_CONTRACT else [
        "checkpoint_contract does not exactly match the worker v2 contract"
    ]
    return _gate(
        "checkpoint_contract",
        errors,
        required_state=list(CHECKPOINT_REQUIRED_STATE),
    )


def _validate_heavy_lease(
    contract: dict[str, Any], root: Path, env: Mapping[str, str]
) -> dict[str, Any]:
    errors: list[str] = []
    lease = contract.get("heavy_lease")
    if not isinstance(lease, dict):
        return _gate("heavy_lease", ["heavy_lease declaration missing"])
    if lease.get("fd_env") != HEAVY_LEASE_FD_ENV:
        errors.append(f"heavy lease fd_env must be {HEAVY_LEASE_FD_ENV}")
    try:
        fd = int(env.get(HEAVY_LEASE_FD_ENV, ""))
    except (TypeError, ValueError):
        fd = -1
    try:
        inherited = os.fstat(fd)
    except OSError:
        inherited = None
        errors.append("inherited heavy lease file descriptor is absent or closed")
    lease_path, path_error = _confined_path(lease.get("path"), root)
    if path_error or lease_path is None:
        errors.append(f"heavy lease path invalid: {path_error}")
        lease_stat = None
    else:
        try:
            lease_stat = os.stat(lease_path, follow_symlinks=False)
            if not stat.S_ISREG(lease_stat.st_mode):
                errors.append("heavy lease path is not a regular file")
        except OSError as exc:
            lease_stat = None
            errors.append(f"heavy lease path cannot be stat'ed: {exc}")
    for field in ("st_dev", "st_ino"):
        if not _is_int(lease.get(field), nonnegative=True):
            errors.append(f"heavy lease {field} must be a nonnegative integer")
    if inherited is not None and lease_stat is not None:
        identity = (int(inherited.st_dev), int(inherited.st_ino))
        declared = (lease.get("st_dev"), lease.get("st_ino"))
        path_identity = (int(lease_stat.st_dev), int(lease_stat.st_ino))
        if identity != declared or identity != path_identity:
            errors.append("inherited heavy lease identity does not match cell and lock path")
        if not stat.S_ISREG(inherited.st_mode):
            errors.append("inherited heavy lease descriptor is not a regular file")
        # Prove the inherited open-file description already owns the lock.  A
        # fresh open must conflict; the inherited descriptor itself must not.
        fresh_fd = -1
        fresh_acquired = False
        try:
            fresh_fd = os.open(lease_path, os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
            try:
                fcntl.flock(fresh_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fresh_acquired = True
            except BlockingIOError:
                fresh_acquired = False
            if fresh_acquired:
                fcntl.flock(fresh_fd, fcntl.LOCK_UN)
                errors.append("heavy lease file is not already locked by the inherited lease")
            else:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    errors.append("heavy lease is held by a different open-file description")
        except OSError as exc:
            errors.append(f"heavy lease ownership probe failed: {exc}")
        finally:
            if fresh_fd >= 0:
                os.close(fresh_fd)
    return _gate(
        "heavy_lease",
        errors,
        fd=fd if fd >= 0 else None,
        st_dev=int(inherited.st_dev) if inherited is not None else None,
        st_ino=int(inherited.st_ino) if inherited is not None else None,
    )


def _sysctl_raw(name: str) -> bytes:
    libc = ctypes.CDLL(None, use_errno=True)
    fn = libc.sysctlbyname
    fn.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    fn.restype = ctypes.c_int
    size = ctypes.c_size_t(0)
    encoded = name.encode("ascii")
    if fn(encoded, None, ctypes.byref(size), None, 0) != 0 or size.value <= 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"sysctl size probe failed for {name}")
    buffer = ctypes.create_string_buffer(size.value)
    if fn(encoded, buffer, ctypes.byref(size), None, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"sysctl value probe failed for {name}")
    return bytes(buffer.raw[: size.value])


def _sysctl_uint(name: str) -> int:
    raw = _sysctl_raw(name)
    if len(raw) not in {1, 2, 4, 8}:
        raise ValueError(f"unexpected integer sysctl width for {name}: {len(raw)}")
    return int.from_bytes(raw, byteorder=sys.byteorder, signed=False)


def _power_source() -> str:
    iokit = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
    cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    iokit.IOPSCopyPowerSourcesInfo.argtypes = []
    iokit.IOPSCopyPowerSourcesInfo.restype = ctypes.c_void_p
    iokit.IOPSGetProvidingPowerSourceType.argtypes = [ctypes.c_void_p]
    iokit.IOPSGetProvidingPowerSourceType.restype = ctypes.c_void_p
    cf.CFStringGetCString.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32
    ]
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    cf.CFRelease.restype = None
    blob = iokit.IOPSCopyPowerSourcesInfo()
    if not blob:
        raise RuntimeError("IOKit power-source snapshot unavailable")
    try:
        value = iokit.IOPSGetProvidingPowerSourceType(blob)
        if not value:
            raise RuntimeError("IOKit providing power source unavailable")
        buffer = ctypes.create_string_buffer(128)
        # kCFStringEncodingUTF8
        if not cf.CFStringGetCString(value, buffer, len(buffer), 0x08000100):
            raise RuntimeError("cannot decode IOKit power source")
        return buffer.value.decode("utf-8")
    finally:
        cf.CFRelease(blob)


def _thermal_state() -> int:
    # NSProcessInfo.thermalState: 0 nominal, 1 fair, 2 serious, 3 critical.
    ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
    if not address:
        raise RuntimeError("objc_msgSend unavailable")
    send_object = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    )(address)
    send_integer = ctypes.CFUNCTYPE(
        ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p
    )(address)
    cls = objc.objc_getClass(b"NSProcessInfo")
    process_sel = objc.sel_registerName(b"processInfo")
    thermal_sel = objc.sel_registerName(b"thermalState")
    if not cls or not process_sel or not thermal_sel:
        raise RuntimeError("NSProcessInfo thermal selectors unavailable")
    process = send_object(cls, process_sel)
    if not process:
        raise RuntimeError("NSProcessInfo unavailable")
    state_value = int(send_integer(process, thermal_sel))
    if state_value not in {0, 1, 2, 3}:
        raise RuntimeError(f"unknown NSProcessInfo thermal state {state_value}")
    return state_value


def _resource_snapshot(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    pressure = swap_used = physical = power = thermal = None
    if sys.platform != "darwin":
        errors.append("resource probes support Darwin only")
    else:
        try:
            pressure = _sysctl_uint("kern.memorystatus_vm_pressure_level")
        except Exception as exc:  # fail closed; probe details are evidence
            errors.append(f"memory pressure probe failed: {exc}")
        try:
            raw_swap = _sysctl_raw("vm.swapusage")
            if len(raw_swap) < struct.calcsize("@QQQII"):
                raise ValueError(f"unexpected vm.swapusage width {len(raw_swap)}")
            _total, _available, swap_used, _pagesize, _encrypted = struct.unpack_from(
                "@QQQII", raw_swap
            )
        except Exception as exc:
            errors.append(f"swap probe failed: {exc}")
            swap_used = None
        try:
            physical = _sysctl_uint("hw.memsize")
        except Exception as exc:
            errors.append(f"physical-memory probe failed: {exc}")
        try:
            power = _power_source()
        except Exception as exc:
            errors.append(f"power-source probe failed: {exc}")
        try:
            thermal = _thermal_state()
        except Exception as exc:
            errors.append(f"thermal-state probe failed: {exc}")
    try:
        usage = shutil.disk_usage(root)
        disk_free = int(usage.free)
        disk_total = int(usage.total)
    except OSError as exc:
        errors.append(f"disk probe failed: {exc}")
        disk_free = disk_total = None
    return {
        "probe_ok": not errors,
        "errors": errors,
        "pressure_level": pressure,
        "swap_used_bytes": int(swap_used) if isinstance(swap_used, int) else None,
        "physical_memory_bytes": int(physical) if isinstance(physical, int) else None,
        "power_source": power,
        "thermal_state": thermal,
        "disk_free_bytes": disk_free,
        "disk_total_bytes": disk_total,
    }


def _validate_resources(
    program: dict[str, Any],
    cell: dict[str, Any],
    contract: dict[str, Any],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    if snapshot.get("probe_ok") is not True:
        errors.extend(str(value) for value in snapshot.get("errors", ["resource probes unavailable"]))
    if snapshot.get("pressure_level") != 1:
        errors.append(f"memory pressure must be normal/1, observed {snapshot.get('pressure_level')!r}")
    if snapshot.get("swap_used_bytes") != 0:
        errors.append(f"swap must be exactly zero bytes, observed {snapshot.get('swap_used_bytes')!r}")
    if snapshot.get("power_source") != "AC Power":
        errors.append(f"AC Power must be confirmed, observed {snapshot.get('power_source')!r}")
    if snapshot.get("thermal_state") != 0:
        errors.append(f"thermal state must be nominal/0, observed {snapshot.get('thermal_state')!r}")

    estimate = cell.get("resource_estimate")
    required_estimates = {
        "peak_memory_bytes",
        "resident_bytes",
        "scratch_bytes",
        "output_bytes",
        "source_read_bytes",
        "estimated_seconds",
    }
    if not isinstance(estimate, dict):
        estimate = {}
        errors.append("cell resource_estimate missing")
    missing = sorted(required_estimates - set(estimate))
    if missing:
        errors.append("resource_estimate missing exact fields: " + ", ".join(missing))
    for field in required_estimates:
        value = estimate.get(field)
        positive = field in {"peak_memory_bytes", "resident_bytes", "source_read_bytes"}
        if not _is_int(value, positive=positive, nonnegative=not positive):
            errors.append(f"resource_estimate.{field} must be an exact integer")

    target = program.get("target") if isinstance(program.get("target"), dict) else {}
    process_ceiling = target.get("process_peak_bytes_ceiling")
    resident_ceiling = target.get("resident_bytes_ceiling")
    if not _is_int(process_ceiling, positive=True):
        errors.append("program target.process_peak_bytes_ceiling must be a positive integer")
    elif _is_int(estimate.get("peak_memory_bytes"), positive=True) and estimate[
        "peak_memory_bytes"
    ] > process_ceiling:
        errors.append("cell peak-memory estimate exceeds program process ceiling")
    if not _is_int(resident_ceiling, positive=True):
        errors.append("program target.resident_bytes_ceiling must be a positive integer")
    elif _is_int(estimate.get("resident_bytes"), positive=True) and estimate[
        "resident_bytes"
    ] > resident_ceiling:
        errors.append("cell resident estimate exceeds program resident ceiling")
    physical = snapshot.get("physical_memory_bytes")
    if not _is_int(physical, positive=True):
        errors.append("physical-memory measurement unavailable")
    elif _is_int(estimate.get("peak_memory_bytes"), positive=True) and estimate[
        "peak_memory_bytes"
    ] >= physical:
        errors.append("cell peak-memory estimate does not leave any OS headroom")

    reserve = contract.get("disk_reserve_bytes")
    checkpoint_reserve = contract.get("checkpoint_reserve_bytes")
    scratch = estimate.get("scratch_bytes")
    output = estimate.get("output_bytes")
    if all(_is_int(value, nonnegative=True) for value in (reserve, checkpoint_reserve, scratch, output)):
        required_disk = reserve + checkpoint_reserve + scratch + output
    else:
        required_disk = None
    disk_free = snapshot.get("disk_free_bytes")
    if not _is_int(disk_free, nonnegative=True):
        errors.append("disk-free measurement unavailable")
    elif required_disk is None:
        errors.append("required disk cannot be computed exactly")
    elif disk_free < required_disk:
        errors.append(f"disk free {disk_free} < exact requirement {required_disk}")
    return _gate(
        "resources",
        errors,
        pressure_level=snapshot.get("pressure_level"),
        swap_used_bytes=snapshot.get("swap_used_bytes"),
        power_source=snapshot.get("power_source"),
        thermal_state=snapshot.get("thermal_state"),
        disk_free_bytes=disk_free,
        required_disk_bytes=required_disk,
        physical_memory_bytes=physical,
    )


def _preflight_documents(
    program: dict[str, Any],
    cell: dict[str, Any],
    *,
    root: Path,
    worker_path: Path,
    adapters: Mapping[str, AdapterSpec],
    snapshot: dict[str, Any],
    env: Mapping[str, str],
) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []
    gates.append(_gate("abi.program", abi.validate_program(program, allow_planned=False)))
    gates.append(_gate("abi.cell", abi.validate_cell(cell, program)))
    contract, contract_gate = _validate_execution_contract(cell)
    gates.append(contract_gate)
    gates.append(_validate_checkpoint_contract(cell))
    adapter_paths, executor_gate = _validate_executors(program, cell, adapters, root)
    gates.append(executor_gate)
    gates.append(_validate_inputs(program, cell, contract, adapter_paths, root, worker_path))
    gates.append(_validate_heavy_lease(contract, root, env))
    gates.append(_validate_resources(program, cell, contract, snapshot))
    return {
        "schema": PREFLIGHT_SCHEMA,
        "mode": "preflight_only_no_execution",
        "ok": all(gate.get("ok") is True for gate in gates),
        "program_sha256": program.get("program_sha256"),
        "cell_sha256": cell.get("cell_sha256"),
        "adapter_allowlist": sorted(adapters),
        "gates": gates,
        "execution_available": False,
    }


def preflight_paths(
    program_path: Path,
    cell_path: Path,
    *,
    root: Path = ROOT,
    worker_path: Path = Path(__file__).resolve(),
    adapters: Mapping[str, AdapterSpec] = ADAPTER_REGISTRY,
    snapshot: dict[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    program, program_file, program_error = _load_json(program_path, root)
    cell, cell_file, cell_error = _load_json(cell_path, root)
    if program_error or cell_error or program is None or cell is None:
        errors = [value for value in (program_error, cell_error) if value]
        return {
            "schema": PREFLIGHT_SCHEMA,
            "mode": "preflight_only_no_execution",
            "ok": False,
            "adapter_allowlist": sorted(adapters),
            "gates": [_gate("documents", errors)],
            "program_file_sha256": program_file.sha256 if program_file else None,
            "cell_file_sha256": cell_file.sha256 if cell_file else None,
            "execution_available": False,
        }
    report = _preflight_documents(
        program,
        cell,
        root=root,
        worker_path=worker_path,
        adapters=adapters,
        snapshot=snapshot if snapshot is not None else _resource_snapshot(root),
        env=env if env is not None else os.environ,
    )
    report["program_file_sha256"] = program_file.sha256 if program_file else None
    report["cell_file_sha256"] = cell_file.sha256 if cell_file else None
    return report


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stamp_cell(cell: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cell)
    payload = dict(out)
    payload.pop("cell_sha256", None)
    payload.pop("generated_at", None)
    out["cell_sha256"] = abi.hash_value(payload)
    return out


def selftest() -> int:
    assert ADAPTER_REGISTRY == {}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        files: dict[str, Path] = {}
        contents = {
            "parent": b"parent revision manifest\n",
            "config": b"model config\n",
            "tokenizer": b"tokenizer manifest\n",
            "backend_kernel": b"backend kernel manifest\n",
            "drafter_artifact": b"drafter artifact manifest\n",
            "verifier_path": b"verifier path manifest\n",
            "kv_policy": b"KV policy manifest\n",
            "cache_namespace": b"cache namespace manifest\n",
            "adaptive_policy": b"adaptive policy manifest\n",
            "calibration": b"calibration rows\n",
            "selection": b"selection rows\n",
            "final_eval": b"final evaluation rows\n",
            "adapter": b"reviewed in-process adapter identity only\n",
            "worker": Path(__file__).read_bytes(),
        }
        for label, payload in contents.items():
            path = root / f"{label}.bin"
            path.write_bytes(payload)
            files[label] = path
        hashes = {label: hashlib.sha256(payload).hexdigest() for label, payload in contents.items()}
        sizes = {label: len(payload) for label, payload in contents.items()}

        node = {
            "id": "n0-selftest",
            "kind": "base_codec",
            "phase": "offline",
            "mechanism": "selftest",
            "mechanism_version": "selftest.v1",
            "implementation_status": "prototype",
            "depends_on": [],
            "parameters": {},
            "backend_support": {
                "apple_cpu": "prototype",
                "metal": "unsupported",
                "cuda": "unsupported",
                "distributed": "unsupported",
                "future_specialized": "unsupported",
            },
            "cost_contract": {
                "actual_bytes_authoritative": True,
                "dynamic_required_metrics": [],
            },
            "executor": {
                "wired": True,
                "adapter_id": "selftest.adapter",
                "source_sha256": hashes["adapter"],
                "argv": [ADAPTER_ARGV_MARKER, "selftest.adapter"],
            },
        }
        program = abi.make_planned_program(
            label="selftest-1B",
            params_b=1.0,
            active_b=1.0,
            physical_bpw=0.5,
            operators=[node],
        )
        program["mode"] = "executable"
        program["model"].update({
            "parent_revision_sha256": hashes["parent"],
            "config_sha256": hashes["config"],
            "tokenizer_sha256": hashes["tokenizer"],
        })
        program["runtime_identity"].update({
            "backend_kernel_sha256": hashes["backend_kernel"],
            "drafter_artifact_sha256": hashes["drafter_artifact"],
            "verifier_path_sha256": hashes["verifier_path"],
            "kv_policy_sha256": hashes["kv_policy"],
            "cache_namespace_sha256": hashes["cache_namespace"],
            "adaptive_policy_sha256": hashes["adaptive_policy"],
        })
        program = abi.stamp_program(program)

        lease_path = root / "heavy.lock"
        lease_handle = open(lease_path, "a+b")
        fcntl.flock(lease_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease_stat = os.fstat(lease_handle.fileno())
        contract = {
            "schema": EXECUTION_CONTRACT_SCHEMA,
            "single_cell": True,
            "worker_capability": "preflight_only_no_execution",
            "heavy_lease": {
                "fd_env": HEAVY_LEASE_FD_ENV,
                "path": lease_path.name,
                "st_dev": int(lease_stat.st_dev),
                "st_ino": int(lease_stat.st_ino),
            },
            "disk_reserve_bytes": MIN_DISK_RESERVE_BYTES,
            "checkpoint_reserve_bytes": 1_000_000_000,
            "input_files": [],
        }
        cell = {
            "schema": abi.CELL_SCHEMA,
            "program_sha256": program["program_sha256"],
            "backend": "apple_cpu",
            "proof_state": "planned",
            "fidelity": "F0",
            "seed": 17,
            "calibration_sha256": hashes["calibration"],
            "selection_sha256": hashes["selection"],
            "final_eval_sha256": hashes["final_eval"],
            "worker_source_sha256": hashes["worker"],
            "resource_estimate": {
                "peak_memory_bytes": 2_000_000_000,
                "resident_bytes": 1_000_000_000,
                "scratch_bytes": 1_000_000_000,
                "output_bytes": 1_000_000_000,
                "source_read_bytes": 1_000_000,
                "estimated_seconds": 60,
            },
            "checkpoint_contract": copy.deepcopy(EXACT_CHECKPOINT_CONTRACT),
            "execution_contract": contract,
        }
        bindings = {
            "program.model.parent_revision_sha256": "parent",
            "program.model.config_sha256": "config",
            "program.model.tokenizer_sha256": "tokenizer",
            "program.runtime_identity.backend_kernel_sha256": "backend_kernel",
            "program.runtime_identity.drafter_artifact_sha256": "drafter_artifact",
            "program.runtime_identity.verifier_path_sha256": "verifier_path",
            "program.runtime_identity.kv_policy_sha256": "kv_policy",
            "program.runtime_identity.cache_namespace_sha256": "cache_namespace",
            "program.runtime_identity.adaptive_policy_sha256": "adaptive_policy",
            "cell.calibration_sha256": "calibration",
            "cell.selection_sha256": "selection",
            "cell.final_eval_sha256": "final_eval",
            "cell.worker_source_sha256": "worker",
            "program.operator:n0-selftest.executor.source_sha256": "adapter",
        }
        for binding, label in bindings.items():
            contract["input_files"].append({
                "binding": binding,
                "path": files[label].name,
                "sha256": hashes[label],
                "bytes": sizes[label],
            })
        cell = _stamp_cell(cell)
        program_path = root / "program.json"
        cell_path = root / "cell.json"
        _write_json(program_path, program)
        _write_json(cell_path, cell)
        registry = {
            "selftest.adapter": AdapterSpec(
                source_path=files["adapter"].name,
                source_sha256=hashes["adapter"],
                operator_kinds=frozenset({"base_codec"}),
                backends=frozenset({"apple_cpu"}),
            )
        }
        green = {
            "probe_ok": True,
            "errors": [],
            "pressure_level": 1,
            "swap_used_bytes": 0,
            "physical_memory_bytes": 96 * 1024 ** 3,
            "power_source": "AC Power",
            "thermal_state": 0,
            "disk_free_bytes": 500_000_000_000,
            "disk_total_bytes": 1_000_000_000_000,
        }
        env = {HEAVY_LEASE_FD_ENV: str(lease_handle.fileno())}
        try:
            good = preflight_paths(
                program_path,
                cell_path,
                root=root,
                worker_path=files["worker"],
                adapters=registry,
                snapshot=green,
                env=env,
            )
            assert good["ok"], good
            assert good["execution_available"] is False

            empty_allowlist = preflight_paths(
                program_path,
                cell_path,
                root=root,
                worker_path=files["worker"],
                adapters=ADAPTER_REGISTRY,
                snapshot=green,
                env=env,
            )
            assert not empty_allowlist["ok"]
            assert any(
                "not allow-listed" in error
                for gate in empty_allowlist["gates"] for error in gate["errors"]
            )

            planned = copy.deepcopy(program)
            planned["mode"] = "planned"
            planned["operators"][0]["executor"] = {
                "wired": False, "argv": [], "source_sha256": None
            }
            planned = abi.stamp_program(planned)
            _write_json(program_path, planned)
            planned_report = preflight_paths(
                program_path,
                cell_path,
                root=root,
                worker_path=files["worker"],
                adapters=registry,
                snapshot=green,
                env=env,
            )
            assert not planned_report["ok"]
            assert any(
                "planned program is not executable" in error
                for gate in planned_report["gates"] for error in gate["errors"]
            )
            _write_json(program_path, program)

            red_swap = dict(green)
            red_swap["swap_used_bytes"] = 1
            assert not preflight_paths(
                program_path,
                cell_path,
                root=root,
                worker_path=files["worker"],
                adapters=registry,
                snapshot=red_swap,
                env=env,
            )["ok"]
            assert not preflight_paths(
                program_path,
                cell_path,
                root=root,
                worker_path=files["worker"],
                adapters=registry,
                snapshot=green,
                env={},
            )["ok"]

            files["calibration"].write_bytes(b"tampered\n")
            assert not preflight_paths(
                program_path,
                cell_path,
                root=root,
                worker_path=files["worker"],
                adapters=registry,
                snapshot=green,
                env=env,
            )["ok"]
            files["calibration"].write_bytes(contents["calibration"])

            bad_checkpoint_cell = copy.deepcopy(cell)
            bad_checkpoint_cell["checkpoint_contract"]["fsync_file"] = False
            bad_checkpoint_cell = _stamp_cell(bad_checkpoint_cell)
            _write_json(cell_path, bad_checkpoint_cell)
            checkpoint_report = preflight_paths(
                program_path,
                cell_path,
                root=root,
                worker_path=files["worker"],
                adapters=registry,
                snapshot=green,
                env=env,
            )
            assert not checkpoint_report["ok"]
            assert any(
                "does not exactly match" in error
                for gate in checkpoint_report["gates"] for error in gate["errors"]
            )
        finally:
            fcntl.flock(lease_handle.fileno(), fcntl.LOCK_UN)
            lease_handle.close()
    print("doctor_frontier_worker.py selftest OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight", help="validate one program/cell without executing it")
    preflight.add_argument("--program", required=True, type=Path)
    preflight.add_argument("--cell", required=True, type=Path)
    sub.add_parser("selftest", help="run deterministic fail-closed contract tests")
    args = parser.parse_args()
    if args.command == "selftest":
        return selftest()
    report = preflight_paths(args.program, args.cell)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
