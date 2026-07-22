#!/usr/bin/env python3.12
"""Fail-closed live supervisor for the sealed Kimi-K2.6 Phase-1 plan.

Importing this module and the ``preflight``/``status`` commands are read-only:
they start no process and perform no network operation.  Only ``run`` may start
the pinned Hugging Face downloader.  Source/cache objects are never moved or
removed by this supervisor; a failed run deliberately leaves the dedicated HF
cache resumable.

Power assertion is intentionally outside the trusted argv.  If desired, wrap
the supervisor itself, for example::

    caffeinate -dimsu -- python3.12 tools/condense/kimi_k26_download_supervisor.py run ...

The child argv remains the exact sealed Phase-1 venv ``hf`` command.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import dataclasses
import errno
import fcntl
import hashlib
import json
import os
import re
import signal
import stat
import struct
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

try:  # package import under ``python -m`` / repository tests
    from tools.condense import kimi_k26_release_cycle as phase1
except ModuleNotFoundError:  # direct ``python tools/condense/...py`` execution
    import kimi_k26_release_cycle as phase1  # type: ignore[no-redef]


# The accepted historical allocation is the exact complete source view plus
# its filesystem metadata.  Runtime is the measured retained runtime budget.
PROJECTED_SOURCE_LOGICAL_BYTES = phase1.KIMI_TOTAL_BYTES
KIMI_MANIFEST_SHA256_OBJECTS = 67
PROJECTED_SOURCE_ALLOCATED_BYTES = 595_205_144_576
PROJECTED_RUNTIME_ALLOCATED_BYTES = 2_310_770_688
PROJECTED_HEADROOM_BYTES = 25_322_093_168
PROJECTED_CAPACITY_BYTES = (
    PROJECTED_SOURCE_ALLOCATED_BYTES
    + PROJECTED_RUNTIME_ALLOCATED_BYTES
    + PROJECTED_HEADROOM_BYTES
)
PRESTART_FREE_DISK_BYTES = 622_838_008_432
SESSION_ALLOCATION_CAP_BYTES = (
    PROJECTED_SOURCE_ALLOCATED_BYTES + PROJECTED_RUNTIME_ALLOCATED_BYTES
)
RUNTIME_FREE_DISK_FLOOR_BYTES = PROJECTED_HEADROOM_BYTES

MONITOR_INTERVAL_SECONDS = 0.25
DEFAULT_MONITOR_INTERVAL_SECONDS = 0.20
NETWORK_SAMPLE_INTERVAL_SECONDS = 1.0
RAMP_WARMUP_SECONDS = 60.0
RAMP_MEASUREMENT_SECONDS = 120.0
RAMP_TARGET_BYTES_PER_SECOND = 750_000_000
RAMP_MIN_COUNTER_SAMPLES = 16
TERMINATION_GRACE_SECONDS = 5.0

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_INVOCATION_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_INTERFACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,31}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_PROCESS_UNIQUE_INCOMPLETE = re.compile(
    r"(?P<manifest_sha256>[0-9a-f]{64})\."
    r"(?P<process_nonce>[0-9a-f]{8})\.incomplete\Z"
)
_GENESIS = "0" * 64
_JOURNAL_NAME = "kimi-k26-download-supervisor.journal.jsonl"
_LEASE_NAME = ".kimi-k26-download-supervisor.lease"
_MAX_JOURNAL_BYTES = 128 * 1024 * 1024
_CLEANUP_AUDIT_EVENT = "STALE_INCOMPLETE_AUDITED"
_CLEANUP_STARTED_EVENT = "STALE_INCOMPLETE_CLEANUP_STARTED"
_CLEANUP_UNLINK_EVENT = "STALE_INCOMPLETE_UNLINK_COMMITTED"
_CLEANUP_PARTIAL_EVENT = "STALE_INCOMPLETE_CLEANUP_PARTIAL_FAILURE"
_CLEANUP_COMPLETED_EVENT = "STALE_INCOMPLETE_CLEANUP_COMPLETED"
_CLEANUP_RECEIPT_SCHEMA = "hawking.kimi_k26.stale_download_cleanup.receipt.v1"


class DownloadSupervisorError(RuntimeError):
    """Raised before live transfer when an authority or safety gate fails."""


class Clock(Protocol):
    def utc_now(self) -> str: ...

    def monotonic_ns(self) -> int: ...

    def sleep(self, seconds: float) -> None: ...


class ResourceSampler(Protocol):
    def sample(self, layout: phase1.SessionLayout) -> "ResourceSnapshot": ...


class NetworkProbe(Protocol):
    def capture_active_default(self) -> dict[str, Any]: ...

    def received_bytes(self, interface: str) -> int: ...


class ProcessAuditor(Protocol):
    def audit(
        self, layout: phase1.SessionLayout, plan: dict[str, Any]
    ) -> dict[str, Any]: ...


class ChildProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


@dataclass(frozen=True)
class ResourceSnapshot:
    free_disk_bytes: int
    session_allocated_bytes: int


@dataclass(frozen=True)
class SupervisorPolicy:
    monitor_interval_seconds: float = DEFAULT_MONITOR_INTERVAL_SECONDS
    network_sample_interval_seconds: float = NETWORK_SAMPLE_INTERVAL_SECONDS
    ramp_warmup_seconds: float = RAMP_WARMUP_SECONDS
    ramp_measurement_seconds: float = RAMP_MEASUREMENT_SECONDS
    ramp_target_bytes_per_second: int = RAMP_TARGET_BYTES_PER_SECOND
    ramp_min_counter_samples: int = RAMP_MIN_COUNTER_SAMPLES
    termination_grace_seconds: float = TERMINATION_GRACE_SECONDS
    permit_phase1_authorized_ramp: bool = True

    def validate(self) -> None:
        if not 0 < self.monitor_interval_seconds <= MONITOR_INTERVAL_SECONDS:
            _fail("resource monitor interval must be in (0, 0.25] seconds")
        if self.network_sample_interval_seconds < self.monitor_interval_seconds:
            _fail("network sample interval cannot be shorter than monitor interval")
        if self.ramp_warmup_seconds < 0 or self.ramp_measurement_seconds <= 0:
            _fail("ramp timing is invalid")
        if self.ramp_target_bytes_per_second <= 0:
            _fail("ramp throughput target must be positive")
        if self.ramp_min_counter_samples < 2:
            _fail("ramp requires at least two measured counter samples")
        if self.termination_grace_seconds <= 0:
            _fail("termination grace must be positive")


@dataclass(frozen=True)
class Phase1Hooks:
    build_plan: Callable[..., dict[str, Any]]
    verify_plan: Callable[..., dict[str, Any]]
    verify_runtime: Callable[[dict[str, Any]], dict[str, Any]]

    @classmethod
    def live(cls) -> "Phase1Hooks":
        runtime_verifier = getattr(phase1, "verify_transfer_runtime_binding", None)
        if not callable(runtime_verifier):
            _fail("Phase 1 has no pinned transfer-runtime verifier")
        return cls(
            build_plan=phase1.build_download_plan,
            verify_plan=phase1.verify_download_plan,
            verify_runtime=runtime_verifier,
        )


@dataclass(frozen=True)
class TransferProfile:
    name: str
    workers: int
    argv: tuple[str, ...]
    environment: dict[str, str]

    @property
    def argv_sha256(self) -> str:
        return _json_sha256(list(self.argv))

    @property
    def environment_sha256(self) -> str:
        return _json_sha256(self.environment)


class SystemClock:
    def utc_now(self) -> str:
        seconds = time.time()
        whole = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds))
        micros = int((seconds - int(seconds)) * 1_000_000)
        return f"{whole}.{micros:06d}Z"

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def _fail(message: str) -> None:
    raise DownloadSupervisorError(message)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(phase1.canonical_json(value)).hexdigest()


def _safe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:2_000]


def _require_platform_guards() -> None:
    if not _NOFOLLOW or not _DIRECTORY:
        _fail("O_NOFOLLOW and O_DIRECTORY are required")


def _layout_tmp(layout: phase1.SessionLayout) -> Path:
    value = getattr(layout, "tmp", layout.build / "tmp")
    return Path(value)


def _layout_hf_home(layout: phase1.SessionLayout) -> Path:
    value = getattr(layout, "hf_home", layout.build / "hf-home")
    return Path(value)


def _validate_execution_directories(layout: phase1.SessionLayout) -> list[dict[str, Any]]:
    expected = {
        "TMPDIR": _layout_tmp(layout),
        "HF_HOME": _layout_hf_home(layout),
        "HF_HUB_CACHE": layout.hub,
        "HF_XET_CACHE": layout.xet,
    }
    rows: list[dict[str, Any]] = []
    for label, path in expected.items():
        try:
            row = phase1._private_directory_metadata(path)  # noqa: SLF001
        except (FileNotFoundError, OSError, phase1.ReleaseCycleError) as exc:
            raise DownloadSupervisorError(
                f"required existing private {label} directory is unavailable: {exc}"
            ) from exc
        if row["path"] != row["realpath"]:
            _fail(f"{label} resolves through a symlink")
        rows.append({"binding": label, **row})
    return rows


def _validate_profile(
    profile: TransferProfile,
    layout: phase1.SessionLayout,
    *,
    expected_workers: int,
) -> None:
    if profile.workers != expected_workers:
        _fail(f"{profile.name} worker binding changed")
    argv = list(profile.argv)
    if not argv or argv[0] != os.fspath(phase1.HF_CLI):
        _fail(f"{profile.name} does not use the pinned Phase-1 hf executable")
    if any(not isinstance(item, str) or not item or "\x00" in item for item in argv):
        _fail(f"{profile.name} argv is not an exact string vector")
    if "--token" in argv or any(item.startswith("--token=") for item in argv):
        _fail("token-bearing downloader argv is forbidden")
    if argv.count("--max-workers") != 1:
        _fail(f"{profile.name} has no unique --max-workers binding")
    worker_index = argv.index("--max-workers")
    if worker_index + 1 >= len(argv) or argv[worker_index + 1] != str(expected_workers):
        _fail(f"{profile.name} command worker count changed")
    environment = profile.environment
    required_keys = {
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_XET_CACHE",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
        "HF_HUB_DISABLE_TELEMETRY",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "HF_HUB_OFFLINE",
        "HF_XET_CHUNK_CACHE_SIZE_BYTES",
        "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS",
        "HF_XET_HIGH_PERFORMANCE",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "PYTHONPYCACHEPREFIX",
        "PYTHONSAFEPATH",
        "TEMP",
        "TMP",
        "TMPDIR",
    }
    if set(environment) != required_keys:
        _fail(f"{profile.name} environment is not the exact isolated key set")
    if any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or "\x00" in key
        or "\x00" in value
        for key, value in environment.items()
    ):
        _fail(f"{profile.name} environment contains a non-string or NUL")
    exact_paths = {
        "HF_HOME": os.fspath(_layout_hf_home(layout)),
        "HF_HUB_CACHE": os.fspath(layout.hub),
        "HF_XET_CACHE": os.fspath(layout.xet),
        "PYTHONPYCACHEPREFIX": os.fspath(_layout_tmp(layout) / "pycache"),
        "TEMP": os.fspath(_layout_tmp(layout)),
        "TMP": os.fspath(_layout_tmp(layout)),
        "TMPDIR": os.fspath(_layout_tmp(layout)),
    }
    for key, expected in exact_paths.items():
        if environment.get(key) != expected:
            _fail(f"{profile.name} {key} changed or could fall back")
    exact_controls = {
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "HF_HUB_OFFLINE": "0",
        "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
        "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": str(expected_workers),
        "HF_XET_HIGH_PERFORMANCE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
    }
    for key, expected in exact_controls.items():
        if environment.get(key) != expected:
            _fail(f"{profile.name} control {key} changed")
    tokenish = [
        key
        for key in environment
        if "TOKEN" in key.upper() and key != "HF_HUB_DISABLE_IMPLICIT_TOKEN"
    ]
    if tokenish:
        _fail(f"token-bearing environment keys are forbidden: {tokenish}")


def _mapping_profile(value: Mapping[str, Any], *, name: str, workers: int) -> TransferProfile:
    argv = value.get("command_argv", value.get("argv"))
    environment = value.get("environment")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        _fail(f"Phase-1 profile {name} has no exact argv")
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in environment.items()
    ):
        _fail(f"Phase-1 profile {name} has no exact environment")
    return TransferProfile(name, workers, tuple(argv), dict(environment))


def _profiles_from_plan(
    plan: dict[str, Any], layout: phase1.SessionLayout
) -> tuple[TransferProfile, TransferProfile | None]:
    primary = _mapping_profile(plan, name="primary-8", workers=8)
    _validate_profile(primary, layout, expected_workers=8)
    ramp: TransferProfile | None = None
    profiles = plan.get("restart_profiles")
    candidates: list[Mapping[str, Any]] = []
    if isinstance(profiles, Mapping):
        candidates.extend(
            item for item in profiles.values() if isinstance(item, Mapping)
        )
    elif isinstance(profiles, list):
        candidates.extend(item for item in profiles if isinstance(item, Mapping))
    for candidate in candidates:
        declared = candidate.get(
            "maximum_file_download_workers",
            candidate.get("workers", candidate.get("max_workers")),
        )
        argv = candidate.get("command_argv", candidate.get("argv"))
        if declared == 16 or (
            isinstance(argv, list)
            and "--max-workers" in argv
            and argv[argv.index("--max-workers") + 1 : argv.index("--max-workers") + 2]
            == ["16"]
        ):
            if ramp is not None:
                _fail("Phase 1 supplied more than one 16-worker restart profile")
            ramp = _mapping_profile(candidate, name="conditional-ramp-16", workers=16)
    if ramp is not None:
        _validate_profile(ramp, layout, expected_workers=16)
        for key in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "TMPDIR"):
            if ramp.environment[key] != primary.environment[key]:
                _fail("16-worker restart does not preserve the exact resumable cache")
        if ramp.argv[1:-1] != primary.argv[1:-1]:
            # The only argv value allowed to change is the value following the
            # already-pinned --max-workers flag.
            left = list(primary.argv)
            right = list(ramp.argv)
            left[left.index("--max-workers") + 1] = "16"
            if left != right:
                _fail("16-worker restart changes more than worker concurrency")
    return primary, ramp


def _validated_plan(
    layout: phase1.SessionLayout,
    *,
    supplied_plan: dict[str, Any] | None,
    hooks: Phase1Hooks,
    manifest_path: Path,
    mop_root: Path,
    shared_xet: Path,
) -> dict[str, Any]:
    kwargs = {
        "manifest_path": manifest_path,
        "mop_root": mop_root,
        "shared_xet": shared_xet,
    }
    expected = hooks.build_plan(layout, **kwargs)
    hooks.verify_plan(expected, layout, **kwargs)
    plan = expected if supplied_plan is None else supplied_plan
    hooks.verify_plan(plan, layout, **kwargs)
    if phase1.canonical_json(plan) != phase1.canonical_json(expected):
        _fail("supplied plan differs from the exact deterministic Phase-1 plan")
    if plan.get("environment_mode") != "REPLACE_NOT_MERGE":
        _fail("Phase-1 environment replacement authority changed")
    runtime = plan.get("transfer_runtime")
    if not isinstance(runtime, dict):
        _fail("Phase-1 plan has no pinned transfer-runtime binding")
    hooks.verify_runtime(runtime)
    if plan.get("transfer_runtime_seal_sha256") != runtime.get("seal_sha256"):
        _fail("Phase-1 plan/runtime seal binding changed")
    _validate_execution_directories(layout)
    _profiles_from_plan(plan, layout)
    return plan


def _verified_manifest_sha256_rows(
    manifest_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return only exact manifest SHA-256 identities used by Xet temp names."""
    verification = phase1.verify_manifest(manifest_path)
    raw = phase1._read_regular_bytes(  # noqa: SLF001
        phase1._require_absolute_clean(  # noqa: SLF001
            manifest_path, label="official manifest path"
        ),
        label="official manifest for incomplete-name audit",
        maximum_bytes=1_000_000,
        expected_uid=os.getuid(),
    )
    manifest = phase1._manifest_from_bytes(  # noqa: SLF001
        raw, label="official manifest for incomplete-name audit"
    )
    rows: dict[str, dict[str, Any]] = {}
    for item in manifest["files"]:
        digest = item["sha256"]
        if digest is None:
            continue
        if digest in rows:
            _fail("official manifest repeats a SHA-256 content identity")
        rows[digest] = {
            "path": item["path"],
            "expected_logical_bytes": item["size"],
        }
    if len(rows) != KIMI_MANIFEST_SHA256_OBJECTS:
        _fail("official manifest SHA-256 identity count changed")
    return rows, verification


def _incomplete_file_facts(
    name: str,
    metadata: os.stat_result,
    *,
    manifest_row: Mapping[str, Any],
    manifest_sha256: str,
    process_nonce: str,
) -> dict[str, Any]:
    if not stat.S_ISREG(metadata.st_mode):
        _fail(f"nonresumable incomplete entry is not regular: {name}")
    if metadata.st_uid != os.getuid():
        _fail(f"nonresumable incomplete entry has wrong owner: {name}")
    if metadata.st_nlink != 1:
        _fail(f"nonresumable incomplete entry has unsafe hard-link count: {name}")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        _fail(f"nonresumable incomplete entry mode is not 0600: {name}")
    expected_size = manifest_row.get("expected_logical_bytes")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int):
        _fail("manifest incomplete-file size binding is malformed")
    if metadata.st_size < 0 or metadata.st_size > expected_size:
        _fail(f"nonresumable incomplete entry exceeds its manifest object: {name}")
    return {
        "name": name,
        "manifest_sha256": manifest_sha256,
        "manifest_path": manifest_row["path"],
        "manifest_logical_bytes": expected_size,
        "process_nonce": process_nonce,
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "logical_bytes": int(metadata.st_size),
        "blocks": int(metadata.st_blocks),
        "allocated_bytes": int(metadata.st_blocks) * 512,
        "mtime_ns": int(metadata.st_mtime_ns),
        "ctime_ns": int(metadata.st_ctime_ns),
        "uid": int(metadata.st_uid),
        "gid": int(metadata.st_gid),
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "hard_links": int(metadata.st_nlink),
    }


def _scan_nonresumable_incomplete_files(
    layout: phase1.SessionLayout,
    *,
    manifest_path: Path = phase1.OFFICIAL_MANIFEST,
) -> dict[str, Any]:
    """Descriptor-scan exact HF 1.24 process-unique Xet temp objects.

    The scanner never uses a glob and never follows a symlink.  Any leaf that
    merely ends in ``.incomplete`` but is not the exact
    ``<manifest-sha256>.<8-lowerhex>.incomplete`` shape is a blocker rather
    than something this instrument silently ignores.
    """
    manifest_rows, manifest_verification = _verified_manifest_sha256_rows(
        manifest_path
    )
    try:
        descriptor = phase1._open_absolute_directory(layout.blobs)  # noqa: SLF001
    except FileNotFoundError:
        files: list[dict[str, Any]] = []
        root_present = False
        root_identity = None
    else:
        root_present = True
        try:
            root_metadata = os.fstat(descriptor)
            if root_metadata.st_uid != os.getuid():
                _fail("dedicated blob directory has the wrong owner")
            root_identity = {
                "device": int(root_metadata.st_dev),
                "inode": int(root_metadata.st_ino),
                "uid": int(root_metadata.st_uid),
                "mode": f"{stat.S_IMODE(root_metadata.st_mode):04o}",
            }
            files = []
            for name in sorted(os.listdir(descriptor)):
                if not name.endswith(".incomplete"):
                    continue
                match = _PROCESS_UNIQUE_INCOMPLETE.fullmatch(name)
                if match is None:
                    _fail(
                        "unknown .incomplete leaf is not an exact HF 1.24 "
                        f"manifest-SHA/process-nonce name: {name}"
                    )
                digest = match.group("manifest_sha256")
                manifest_row = manifest_rows.get(digest)
                if manifest_row is None:
                    _fail(
                        "process-unique .incomplete leaf is not bound to the "
                        f"official Kimi manifest: {name}"
                    )
                named_before = os.stat(
                    name, dir_fd=descriptor, follow_symlinks=False
                )
                facts = _incomplete_file_facts(
                    name,
                    named_before,
                    manifest_row=manifest_row,
                    manifest_sha256=digest,
                    process_nonce=match.group("process_nonce"),
                )
                named_after = os.stat(
                    name, dir_fd=descriptor, follow_symlinks=False
                )
                if not phase1._identity_equal(  # noqa: SLF001
                    named_before, named_after
                ) or named_before.st_mtime_ns != named_after.st_mtime_ns \
                        or named_before.st_blocks != named_after.st_blocks \
                        or named_before.st_ctime_ns != named_after.st_ctime_ns:
                    _fail(f"nonresumable incomplete entry changed during scan: {name}")
                files.append(facts)
        finally:
            os.close(descriptor)
    return phase1.seal_document(
        {
            "schema": (
                "hawking.kimi_k26.download_supervisor."
                "nonresumable_incomplete_inventory.v1"
            ),
            "status": "PASS_EXACT_INVENTORY",
            "session": os.fspath(layout.session),
            "blobs_root": os.fspath(layout.blobs),
            "blobs_root_present": root_present,
            "blobs_root_identity": root_identity,
            "manifest_verification_seal_sha256": manifest_verification[
                "seal_sha256"
            ],
            "manifest_seal_sha256": manifest_verification[
                "manifest_seal_sha256"
            ],
            "filename_contract": (
                "LOWERCASE_MANIFEST_SHA256_DOT_8_LOWERHEX_DOT_INCOMPLETE"
            ),
            "file_count": len(files),
            "logical_bytes": sum(item["logical_bytes"] for item in files),
            "allocated_bytes": sum(item["allocated_bytes"] for item in files),
            "files": files,
            "glob_used": False,
            "network_accessed": False,
            "filesystem_written": False,
        }
    )


def _assert_no_nonresumable_incomplete(inventory: dict[str, Any]) -> None:
    phase1.verify_sealed_document(
        inventory, label="nonresumable incomplete inventory"
    )
    count = inventory.get("file_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        _fail("nonresumable incomplete inventory count is malformed")
    if count:
        _fail(
            f"{count} process-unique .incomplete files are nonresumable; "
            "run the sealed two-phase stale cleanup before launching"
        )


def _allocated_bytes_fd(descriptor: int, seen: set[tuple[int, int]]) -> int:
    metadata = os.fstat(descriptor)
    identity = (int(metadata.st_dev), int(metadata.st_ino))
    total = 0
    if identity not in seen:
        seen.add(identity)
        total += int(metadata.st_blocks) * 512
    try:
        names = os.listdir(descriptor)
    except OSError as exc:
        raise DownloadSupervisorError(f"cannot enumerate session allocation: {exc}") from exc
    for name in names:
        try:
            item = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise DownloadSupervisorError(
                "session allocation changed during safety sampling"
            ) from exc
        identity = (int(item.st_dev), int(item.st_ino))
        kind = stat.S_IFMT(item.st_mode)
        if kind == stat.S_IFDIR:
            try:
                child = os.open(
                    name,
                    os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise DownloadSupervisorError(
                    f"cannot open session directory during safety sampling: {exc}"
                ) from exc
            try:
                total += _allocated_bytes_fd(child, seen)
            finally:
                os.close(child)
        elif kind in {stat.S_IFREG, stat.S_IFLNK}:
            if identity not in seen:
                seen.add(identity)
                total += int(item.st_blocks) * 512
        else:
            _fail("non-file object appeared inside the dedicated session")
    return total


class SystemResourceSampler:
    def sample(self, layout: phase1.SessionLayout) -> ResourceSnapshot:
        _require_platform_guards()
        descriptor = phase1._open_absolute_directory(layout.session)  # noqa: SLF001
        try:
            filesystem = os.fstatvfs(descriptor)
            free = int(filesystem.f_bavail) * int(filesystem.f_frsize)
            allocated = _allocated_bytes_fd(descriptor, set())
        finally:
            os.close(descriptor)
        return ResourceSnapshot(free, allocated)


def _checked_readonly_command(argv: Sequence[str]) -> str:
    if not argv or not Path(argv[0]).is_absolute():
        _fail("network evidence helper executable must be absolute")
    try:
        completed = subprocess.run(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"LC_ALL": "C"},
            shell=False,
            close_fds=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DownloadSupervisorError(
            f"network evidence helper failed: {_safe_error(exc)}"
        ) from exc
    if completed.returncode != 0:
        _fail(f"network evidence helper exited {completed.returncode}")
    if len(completed.stdout) > 128 * 1024:
        _fail("network evidence helper output is oversized")
    try:
        return completed.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DownloadSupervisorError("network evidence helper output is not UTF-8") from exc


class SystemNetworkProbe:
    route_executable = Path("/sbin/route")
    ifconfig_executable = Path("/sbin/ifconfig")
    netstat_executable = Path("/usr/sbin/netstat")

    def _binding(self, path: Path) -> dict[str, Any]:
        return {
            "path": os.fspath(path),
            **phase1._hash_regular(  # noqa: SLF001
                path, label=f"network helper {path.name}", expected_uid=0
            ),
        }

    def capture_active_default(self) -> dict[str, Any]:
        route_text = _checked_readonly_command(
            [os.fspath(self.route_executable), "-n", "get", "default"]
        )
        fields: dict[str, str] = {}
        for line in route_text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
        interface = fields.get("interface", "")
        if _INTERFACE.fullmatch(interface) is None:
            _fail("active default route has no safe interface binding")
        ifconfig_text = _checked_readonly_command(
            [os.fspath(self.ifconfig_executable), interface]
        )
        media = ""
        active = ""
        mtu: int | None = None
        first_line = ifconfig_text.splitlines()[0] if ifconfig_text.splitlines() else ""
        match = re.search(r"\bmtu\s+(\d+)\b", first_line)
        if match:
            mtu = int(match.group(1))
        for line in ifconfig_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("media:"):
                media = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("status:"):
                active = stripped.split(":", 1)[1].strip().lower()
        if active != "active" or not media:
            _fail("default-route interface is not active with recorded media")
        return phase1.seal_document(
            {
                "schema": "hawking.kimi_k26.download_supervisor.network_path.v1",
                "status": "ACTIVE_DEFAULT_ROUTE_CAPTURED",
                "interface": interface,
                "gateway": fields.get("gateway"),
                "route_flags": fields.get("flags"),
                "media": media,
                "mtu": mtu,
                "planned_10g_media_observed": "10g" in media.lower(),
                "route_command": [os.fspath(self.route_executable), "-n", "get", "default"],
                "media_command": [os.fspath(self.ifconfig_executable), interface],
                "counter_command": [os.fspath(self.netstat_executable), "-bI", interface],
                "helper_bindings": [
                    self._binding(self.route_executable),
                    self._binding(self.ifconfig_executable),
                    self._binding(self.netstat_executable),
                ],
            }
        )

    def received_bytes(self, interface: str) -> int:
        if _INTERFACE.fullmatch(interface) is None:
            _fail("network counter interface is invalid")
        text = _checked_readonly_command(
            [os.fspath(self.netstat_executable), "-bI", interface]
        )
        lines = [line.split() for line in text.splitlines() if line.strip()]
        if len(lines) < 2 or "Ibytes" not in lines[0]:
            _fail("network counter output has no Ibytes column")
        index = lines[0].index("Ibytes")
        for row in lines[1:]:
            if row and row[0] == interface and len(row) > index:
                try:
                    value = int(row[index])
                except ValueError as exc:
                    raise DownloadSupervisorError("network Ibytes is not an integer") from exc
                if value < 0:
                    _fail("network Ibytes is negative")
                return value
        _fail("network counter output has no bound interface row")


class DarwinExactProcessAuditor:
    """Inspect structured Darwin procargs for an existing cache writer.

    This deliberately does not parse the display string emitted by ``ps``.
    PIDs and executable identities come from libproc; argv and environment are
    parsed from the kernel's length-delimited KERN_PROCARGS2 buffer.  The audit
    is conservative for current-UID processes because the 0700 session cannot
    be entered by another UID.
    """

    _PROC_ALL_PIDS = 1
    _PROC_PIDTBSDINFO = 3
    _CTL_KERN = 1
    _KERN_PROCARGS2 = 49
    _MAX_PROCARGS_BYTES = 4 * 1024 * 1024

    def __init__(self) -> None:
        if sys.platform != "darwin":
            _fail("native structured process audit requires Darwin")
        try:
            self.libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            self.libc = ctypes.CDLL(None, use_errno=True)
        except OSError as exc:
            raise DownloadSupervisorError(
                f"cannot load native process-audit libraries: {exc}"
            ) from exc
        self.libproc.proc_listpids.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.libproc.proc_listpids.restype = ctypes.c_int
        self.libproc.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.libproc.proc_pidinfo.restype = ctypes.c_int
        self.libproc.proc_pidpath.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self.libproc.proc_pidpath.restype = ctypes.c_int
        self.libc.sysctl.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.libc.sysctl.restype = ctypes.c_int

    def _pids(self) -> list[int]:
        needed = self.libproc.proc_listpids(self._PROC_ALL_PIDS, 0, None, 0)
        if needed <= 0:
            _fail("native process enumeration failed")
        slots = needed // ctypes.sizeof(ctypes.c_int) + 1024
        values = (ctypes.c_int * slots)()
        received = self.libproc.proc_listpids(
            self._PROC_ALL_PIDS, 0, values, ctypes.sizeof(values)
        )
        if received < 0:
            _fail("native process enumeration returned an error")
        return sorted(
            {
                int(pid)
                for pid in values[: received // ctypes.sizeof(ctypes.c_int)]
                if int(pid) > 1
            }
        )

    def _uid(self, pid: int) -> int | None:
        buffer = ctypes.create_string_buffer(256)
        received = self.libproc.proc_pidinfo(
            pid, self._PROC_PIDTBSDINFO, 0, buffer, len(buffer)
        )
        if received < 24:
            return None
        return int(struct.unpack_from("I", buffer.raw, 20)[0])

    def _pidpath(self, pid: int) -> str | None:
        buffer = ctypes.create_string_buffer(4096)
        received = self.libproc.proc_pidpath(pid, buffer, len(buffer))
        if received <= 0:
            return None
        return os.fsdecode(buffer.raw[:received].split(b"\x00", 1)[0])

    def _procargs(self, pid: int) -> tuple[list[str], dict[str, str]] | None:
        mib = (ctypes.c_int * 3)(self._CTL_KERN, self._KERN_PROCARGS2, pid)
        size = ctypes.c_size_t(0)
        if self.libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0:
            return None
        if size.value < 5 or size.value > self._MAX_PROCARGS_BYTES:
            _fail(f"PID {pid} has an unsafe structured procargs size")
        buffer = ctypes.create_string_buffer(size.value)
        if self.libc.sysctl(
            mib, 3, buffer, ctypes.byref(size), None, 0
        ) != 0:
            return None
        raw = buffer.raw[: size.value]
        if len(raw) < 4:
            return None
        argc = int(struct.unpack_from("i", raw, 0)[0])
        if argc < 0 or argc > 65_536:
            _fail(f"PID {pid} has an unsafe argc")
        cursor = 4
        executable_end = raw.find(b"\x00", cursor)
        if executable_end < 0:
            return None
        cursor = executable_end + 1
        while cursor < len(raw) and raw[cursor] == 0:
            cursor += 1
        argv: list[str] = []
        for _ in range(argc):
            end = raw.find(b"\x00", cursor)
            if end < 0:
                return None
            argv.append(os.fsdecode(raw[cursor:end]))
            cursor = end + 1
        while cursor < len(raw) and raw[cursor] == 0:
            cursor += 1
        environment: dict[str, str] = {}
        while cursor < len(raw):
            end = raw.find(b"\x00", cursor)
            if end < 0:
                break
            item = os.fsdecode(raw[cursor:end])
            if not item:
                break
            key, separator, value = item.partition("=")
            if separator and key not in environment:
                environment[key] = value
            cursor = end + 1
        return argv, environment

    @staticmethod
    def _value_after(argv: Sequence[str], option: str) -> str | None:
        positions = [index for index, value in enumerate(argv) if value == option]
        if len(positions) != 1 or positions[0] + 1 >= len(argv):
            return None
        return argv[positions[0] + 1]

    def _uses_exact_cache(
        self,
        argv: Sequence[str],
        environment: Mapping[str, str],
        layout: phase1.SessionLayout,
    ) -> bool:
        if "download" not in argv or phase1.KIMI_REPO not in argv:
            return False
        cache_argument = self._value_after(argv, "--cache-dir")
        exact_argument = cache_argument in {
            os.fspath(layout.session),
            os.fspath(layout.hub),
            os.fspath(layout.xet),
        }
        exact_environment = any(
            environment.get(key) == expected
            for key, expected in {
                "HF_HOME": os.fspath(layout.hf_home),
                "HF_HUB_CACHE": os.fspath(layout.hub),
                "HF_XET_CACHE": os.fspath(layout.xet),
                "TMPDIR": os.fspath(layout.tmp),
            }.items()
        )
        return exact_argument or exact_environment

    def audit(
        self, layout: phase1.SessionLayout, plan: dict[str, Any]
    ) -> dict[str, Any]:
        current_uid = os.getuid()
        inspected = 0
        vanished = 0
        conflicts: list[dict[str, Any]] = []
        uninspectable: list[int] = []
        for pid in self._pids():
            if pid == os.getpid():
                continue
            uid = self._uid(pid)
            if uid is None:
                vanished += 1
                continue
            if uid != current_uid:
                continue
            executable_before = self._pidpath(pid)
            process = self._procargs(pid)
            if process is None:
                if self._uid(pid) == current_uid:
                    uninspectable.append(pid)
                else:
                    vanished += 1
                continue
            argv, environment = process
            executable_after = self._pidpath(pid)
            if executable_before is None or executable_after is None:
                vanished += 1
                continue
            if executable_before != executable_after:
                _fail(f"PID {pid} changed identity during native process audit")
            inspected += 1
            if self._uses_exact_cache(argv, environment, layout):
                conflicts.append(
                    {
                        "pid": pid,
                        "executable": executable_after,
                        "argv_sha256": _json_sha256(argv),
                        "matching_identity": (
                            "KIMI_DOWNLOAD_AND_EXACT_SESSION_OR_CACHE_BINDING"
                        ),
                    }
                )
        if uninspectable:
            _fail(
                "cannot attest current-UID process identities before launch: "
                + ",".join(str(pid) for pid in uninspectable[:16])
            )
        if conflicts:
            _fail(
                "existing manually launched Kimi downloader uses the exact session/cache: "
                + ",".join(str(item["pid"]) for item in conflicts)
            )
        runtime = plan.get("transfer_runtime", {})
        interpreter = (
            runtime.get("resolved_interpreter", {}).get("path")
            if isinstance(runtime, dict)
            else None
        )
        return phase1.seal_document(
            {
                "schema": "hawking.kimi_k26.download_supervisor.process_audit.v1",
                "status": (
                    "PASS_SNAPSHOT_NO_EXISTING_EXACT_SESSION_CACHE_DOWNLOADER_"
                    "BEST_EFFORT_WITH_RACE"
                ),
                "method": "DARWIN_LIBPROC_PLUS_STRUCTURED_KERN_PROCARGS2",
                "current_uid_processes_inspected": inspected,
                "processes_vanished_during_audit": vanished,
                "conflict_count": 0,
                "pinned_downloader_interpreter": interpreter,
                "string_display_command_matching_used": False,
                "residual_race": (
                    "AN_UNCOOPERATIVE_PROCESS_CAN_START_AFTER_THIS_SNAPSHOT;"
                    "THE_PRIVATE_0700_SESSION_AND_INHERITED_COOPERATIVE_LEASE_"
                    "LIMIT_BUT_CANNOT_ATOMICALLY_EXCLUDE_THAT_MANUAL_ACTION"
                ),
            }
        )


def _ensure_private_file_metadata(descriptor: int, *, label: str) -> os.stat_result:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} is not a regular file")
    if metadata.st_uid != os.getuid():
        _fail(f"{label} is not owned by the current uid")
    if metadata.st_nlink != 1:
        _fail(f"{label} has unsafe hard-link count")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        _fail(f"{label} mode is not 0600")
    return metadata


def _open_evidence_leaf(
    layout: phase1.SessionLayout,
    name: str,
    *,
    flags: int,
    exclusive: bool,
) -> int:
    if not name or "/" in name or "\x00" in name or name in {".", ".."}:
        _fail("evidence filename is not a safe leaf")
    evidence = phase1._private_directory_metadata(layout.evidence)  # noqa: SLF001
    if evidence["path"] != evidence["realpath"]:
        _fail("evidence directory resolves through a symlink")
    directory = phase1._open_absolute_directory(layout.evidence)  # noqa: SLF001
    try:
        open_flags = flags | _NOFOLLOW | _CLOEXEC
        if exclusive:
            open_flags |= os.O_CREAT | os.O_EXCL
        descriptor = os.open(name, open_flags, 0o600, dir_fd=directory)
        try:
            _ensure_private_file_metadata(descriptor, label=f"evidence/{name}")
            named = os.stat(name, dir_fd=directory, follow_symlinks=False)
            opened = os.fstat(descriptor)
            if not phase1._identity_equal(named, opened):  # noqa: SLF001
                _fail(f"evidence/{name} changed while opening")
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor
    finally:
        os.close(directory)


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            _fail("short evidence write")
        view = view[written:]


def _write_new_document(
    layout: phase1.SessionLayout, name: str, value: dict[str, Any]
) -> Path:
    sealed = phase1.seal_document(value)
    raw = phase1.canonical_json(sealed) + b"\n"
    descriptor = _open_evidence_leaf(
        layout, name, flags=os.O_WRONLY, exclusive=True
    )
    try:
        _write_all(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = phase1._open_absolute_directory(layout.evidence)  # noqa: SLF001
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return layout.evidence / name


def _read_evidence_leaf(
    layout: phase1.SessionLayout, name: str, *, maximum_bytes: int
) -> bytes | None:
    directory = phase1._open_absolute_directory(layout.evidence)  # noqa: SLF001
    try:
        try:
            named = os.stat(name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(named.st_mode):
            _fail(f"evidence/{name} is not a no-follow regular file")
        descriptor = os.open(
            name, os.O_RDONLY | _NOFOLLOW | _CLOEXEC, dir_fd=directory
        )
        try:
            opened = _ensure_private_file_metadata(
                descriptor, label=f"evidence/{name}"
            )
            if not phase1._identity_equal(named, opened):  # noqa: SLF001
                _fail(f"evidence/{name} changed while opening")
            if opened.st_size > maximum_bytes:
                _fail(f"evidence/{name} exceeds the read bound")
            chunks: list[bytes] = []
            remaining = int(opened.st_size)
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    _fail(f"evidence/{name} truncated while reading")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                _fail(f"evidence/{name} grew while reading")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def _strict_json_line(raw: bytes, *, label: str) -> dict[str, Any]:
    return phase1.strict_json_bytes(raw, label=label)


def _verify_journal_bytes(raw: bytes) -> list[dict[str, Any]]:
    if not raw:
        return []
    if not raw.endswith(b"\n"):
        _fail("journal has an incomplete tail")
    entries: list[dict[str, Any]] = []
    previous = _GENESIS
    for index, line in enumerate(raw.splitlines()):
        try:
            entry = _strict_json_line(line, label=f"journal line {index + 1}")
            phase1.verify_sealed_document(entry, label=f"journal line {index + 1}")
        except (phase1.ReleaseCycleError, ValueError, TypeError) as exc:
            raise DownloadSupervisorError(
                f"journal line {index + 1} failed strict sealed verification: {exc}"
            ) from exc
        if entry.get("schema") != "hawking.kimi_k26.download_supervisor.journal_entry.v1":
            _fail("journal schema changed")
        if entry.get("sequence") != index + 1:
            _fail("journal sequence is not contiguous")
        if entry.get("previous_entry_seal_sha256") != previous:
            _fail("journal hash chain is broken")
        previous = entry["seal_sha256"]
        entries.append(entry)
    return entries


class JournalWriter:
    def __init__(self, layout: phase1.SessionLayout) -> None:
        self.layout = layout
        self.descriptor = _open_evidence_leaf(
            layout,
            _JOURNAL_NAME,
            flags=os.O_RDWR | os.O_APPEND | os.O_CREAT,
            exclusive=False,
        )
        fcntl.flock(self.descriptor, fcntl.LOCK_EX)
        size = os.fstat(self.descriptor).st_size
        if size > _MAX_JOURNAL_BYTES:
            _fail("journal exceeds the safety read bound")
        raw = os.pread(self.descriptor, size, 0)
        if len(raw) != size:
            _fail("journal changed while opening")
        self.entries = _verify_journal_bytes(raw)
        self.sequence = len(self.entries)
        self.head = self.entries[-1]["seal_sha256"] if self.entries else _GENESIS
        self.byte_length = int(size)

    def append(
        self,
        *,
        event: str,
        invocation_id: str,
        timestamp_utc: str,
        monotonic_ns: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        entry = phase1.seal_document(
            {
                "schema": "hawking.kimi_k26.download_supervisor.journal_entry.v1",
                "sequence": self.sequence + 1,
                "previous_entry_seal_sha256": self.head,
                "event": event,
                "invocation_id": invocation_id,
                "timestamp_utc": timestamp_utc,
                "monotonic_ns": monotonic_ns,
                "payload": payload,
            }
        )
        raw = phase1.canonical_json(entry) + b"\n"
        current_size = int(os.fstat(self.descriptor).st_size)
        if current_size != self.byte_length:
            _fail("journal byte length changed outside the locked writer")
        if current_size + len(raw) > _MAX_JOURNAL_BYTES:
            _fail("journal append would cross the replay safety bound")
        _write_all(self.descriptor, raw)
        os.fsync(self.descriptor)
        self.byte_length += len(raw)
        self.sequence += 1
        self.head = entry["seal_sha256"]
        self.entries.append(entry)
        return entry

    def close(self) -> None:
        if self.descriptor >= 0:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> "JournalWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@contextlib.contextmanager
def _exclusive_lease(layout: phase1.SessionLayout) -> Iterator[int]:
    descriptor = _open_evidence_leaf(
        layout,
        _LEASE_NAME,
        flags=os.O_RDWR | os.O_CREAT,
        exclusive=False,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DownloadSupervisorError(
                "another Kimi download supervisor holds the exclusive lease"
            ) from exc
        yield descriptor
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _pid_appears_live(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _assert_no_unfinished_live_child(entries: Sequence[dict[str, Any]]) -> None:
    active: dict[int, str] = {}
    for entry in entries:
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            continue
        pid = payload.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int):
            continue
        if entry.get("event") == "CHILD_STARTED":
            active[pid] = str(entry.get("invocation_id"))
        elif entry.get("event") == "CHILD_EXITED":
            active.pop(pid, None)
    for pid, invocation in active.items():
        if _pid_appears_live(pid):
            _fail(
                f"unfinished invocation {invocation} still has live PID {pid}; "
                "concurrent resume is forbidden"
            )


def _read_finished_status(
    layout: phase1.SessionLayout, entry: dict[str, Any]
) -> dict[str, Any]:
    payload = entry.get("payload")
    if not isinstance(payload, dict):
        _fail("finished invocation journal payload is malformed")
    invocation = entry.get("invocation_id")
    if not isinstance(invocation, str) or _INVOCATION_ID.fullmatch(invocation) is None:
        _fail("finished invocation id is malformed")
    raw_path = payload.get("status_path")
    if not isinstance(raw_path, str):
        _fail("finished invocation has no status path")
    path = Path(raw_path)
    if path.parent != layout.evidence or path.name != raw_path.rsplit("/", 1)[-1]:
        _fail("finished invocation status path escapes exact evidence directory")
    raw = _read_evidence_leaf(layout, path.name, maximum_bytes=2_000_000)
    if raw is None:
        _fail("finished invocation status document is absent")
    status_value = phase1.strict_json_bytes(raw, label="finished invocation status")
    phase1.verify_sealed_document(status_value, label="finished invocation status")
    if status_value.get("schema") != "hawking.kimi_k26.download_supervisor.status.v1":
        _fail("finished invocation status schema changed")
    if status_value.get("invocation_id") != invocation:
        _fail("finished invocation status id disagrees with journal")
    if status_value.get("seal_sha256") != payload.get("status_seal_sha256"):
        _fail("finished invocation status seal disagrees with journal")
    if status_value.get("status") != payload.get("status"):
        _fail("finished invocation status outcome disagrees with journal")
    if status_value.get("exit_code") != payload.get("exit_code"):
        _fail("finished invocation exit code disagrees with journal")
    return status_value


def _latest_finished_context(
    layout: phase1.SessionLayout, entries: Sequence[dict[str, Any]]
) -> dict[str, Any] | None:
    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        if entry.get("event") != "INVOCATION_FINISHED":
            continue
        return {
            "index": index,
            "entry": entry,
            "status": _read_finished_status(layout, entry),
        }
    return None


def _measured_ramp_authority(
    entries: Sequence[dict[str, Any]], *, before_index: int
) -> dict[str, Any] | None:
    """Recover the latest exact 8-worker measurement and serial-stop proof."""
    for evaluation_index in range(before_index, -1, -1):
        evaluation = entries[evaluation_index]
        if evaluation.get("event") != "RAMP_16_EVALUATED":
            continue
        payload = evaluation.get("payload")
        if not isinstance(payload, dict) or payload.get("below_target") is not True \
                or payload.get("phase1_16_profile_available") is not True:
            continue
        measured = payload.get("measured_network_bytes_per_second")
        target = payload.get("target_network_bytes_per_second")
        if isinstance(measured, bool) or not isinstance(measured, (int, float)) \
                or isinstance(target, bool) or not isinstance(target, (int, float)) \
                or measured < 0 or target <= 0 or measured >= target:
            _fail("sealed ramp decision has inconsistent below-target evidence")
        invocation = evaluation.get("invocation_id")
        pid = payload.get("pid")
        if not isinstance(invocation, str) or not isinstance(pid, int) or pid <= 1:
            _fail("sealed ramp decision lacks an exact invocation/PID")

        started: dict[str, Any] | None = None
        for candidate in reversed(entries[:evaluation_index]):
            candidate_payload = candidate.get("payload")
            if candidate.get("event") == "CHILD_STARTED" \
                    and candidate.get("invocation_id") == invocation \
                    and isinstance(candidate_payload, dict) \
                    and candidate_payload.get("pid") == pid \
                    and candidate_payload.get("workers") == 8 \
                    and candidate_payload.get("profile") == "primary-8":
                started = candidate
                break
        if started is None:
            _fail("sealed ramp decision has no matching primary child start")

        stop_index: int | None = None
        stop: dict[str, Any] | None = None
        for index in range(evaluation_index + 1, before_index + 1):
            candidate = entries[index]
            candidate_payload = candidate.get("payload")
            if candidate.get("event") == "RAMP_8_PROCESS_STOPPED" \
                    and candidate.get("invocation_id") == invocation \
                    and isinstance(candidate_payload, dict) \
                    and candidate_payload.get("pid") == pid:
                stop_index = index
                stop = candidate
                break
        if stop is None or stop_index is None:
            continue
        stop_payload = stop["payload"]
        if stop_payload.get("prior_pid_fully_exited") is not True \
                or stop_payload.get("same_resumable_cache") is not True:
            _fail("sealed ramp stop does not prove serial same-cache restart")

        exited: dict[str, Any] | None = None
        for candidate in entries[stop_index + 1 : before_index + 1]:
            candidate_payload = candidate.get("payload")
            if candidate.get("event") == "CHILD_EXITED" \
                    and candidate.get("invocation_id") == invocation \
                    and isinstance(candidate_payload, dict) \
                    and candidate_payload.get("pid") == pid \
                    and candidate_payload.get("workers") == 8 \
                    and candidate_payload.get("profile") == "primary-8":
                exited = candidate
                break
        if exited is None:
            continue
        return phase1.seal_document(
            {
                "schema": "hawking.kimi_k26.download_supervisor.ramp_resume_authority.v1",
                "status": "PASS_PRIOR_MEASURED_RAMP_AND_SERIAL_PRIMARY_EXIT",
                "measurement_invocation_id": invocation,
                "primary_pid": pid,
                "ramp_evaluation_entry_seal_sha256": evaluation["seal_sha256"],
                "primary_stop_entry_seal_sha256": stop["seal_sha256"],
                "primary_exit_entry_seal_sha256": exited["seal_sha256"],
                "measured_network_bytes_per_second": measured,
                "target_network_bytes_per_second": target,
                "below_target": True,
                "phase1_16_profile_available": True,
                "prior_pid_fully_exited": True,
                "same_resumable_cache": True,
            }
        )
    return None


def _verified_cleanup_receipt_for_resume(
    layout: phase1.SessionLayout,
    entries: Sequence[dict[str, Any]],
    *,
    prior: dict[str, Any],
    incomplete_inventory: dict[str, Any],
) -> dict[str, Any] | None:
    prior_entry = prior["entry"]
    prior_status = prior["status"]
    prior_id = prior_entry["invocation_id"]
    after_prior = entries[prior["index"] + 1 :]
    audits = [
        entry
        for entry in after_prior
        if entry.get("event") == _CLEANUP_AUDIT_EVENT
        and isinstance(entry.get("payload"), dict)
        and entry["payload"].get("prior_invocation_id") == prior_id
    ]
    incomplete_audit_present = any(
        isinstance(entry["payload"].get("file_count"), int)
        and not isinstance(entry["payload"].get("file_count"), bool)
        and entry["payload"]["file_count"] > 0
        for entry in audits
    )
    receipt_required = (
        prior_status.get("status")
        == "RESOURCE_GUARD_TERMINATED_RESUMABLE_CACHE_PRESERVED"
        or incomplete_audit_present
    )
    if not receipt_required:
        return None
    if not audits:
        _fail("resource-guarded direct-16 resume requires a cleanup audit/receipt")
    audit = audits[-1]
    completions = [
        entry
        for entry in after_prior
        if entry.get("event") == _CLEANUP_COMPLETED_EVENT
        and isinstance(entry.get("payload"), dict)
        and entry["payload"].get("prior_invocation_id") == prior_id
        and entry["payload"].get("audit_event_seal_sha256")
        == audit["seal_sha256"]
    ]
    if not completions:
        _fail("direct-16 resume requires the exact completed cleanup receipt")
    completed = completions[-1]
    payload = completed["payload"]
    audit_payload = audit["payload"]
    if completed.get("invocation_id") != audit.get("invocation_id"):
        _fail("cleanup completion id disagrees with its audit")
    for key in (
        "removed_inventory_seal_sha256",
        "removed_file_count",
        "removed_logical_bytes",
        "removed_allocated_bytes",
    ):
        audit_key = {
            "removed_file_count": "file_count",
            "removed_logical_bytes": "logical_bytes",
            "removed_allocated_bytes": "allocated_bytes",
        }.get(key, key)
        if payload.get(key) != audit_payload.get(audit_key):
            _fail(f"cleanup completion {key} disagrees with the audited inventory")
    audit_index = entries.index(audit)
    completed_index = entries.index(completed)
    if completed_index <= audit_index:
        _fail("cleanup completion precedes its audit")
    progress = [
        entry
        for entry in entries[audit_index + 1 : completed_index]
        if entry.get("event") == _CLEANUP_UNLINK_EVENT
        and entry.get("invocation_id") == audit.get("invocation_id")
        and isinstance(entry.get("payload"), dict)
        and entry["payload"].get("audit_event_seal_sha256")
        == audit["seal_sha256"]
    ]
    audited_count = audit_payload.get("file_count")
    if isinstance(audited_count, bool) or not isinstance(audited_count, int) \
            or audited_count < 0 or len(progress) != audited_count:
        _fail("cleanup progress does not commit every audited inventory row")
    for row_index, entry in enumerate(progress):
        progress_payload = entry["payload"]
        if progress_payload.get("row_index") != row_index \
                or progress_payload.get("removed_inventory_seal_sha256") != (
                    audit_payload.get("removed_inventory_seal_sha256")
                ):
            _fail("cleanup progress is not the exact audited inventory prefix")
    progress_seals = [entry["seal_sha256"] for entry in progress]
    progress_chain = hashlib.sha256(
        phase1.canonical_json({"entry_seal_sha256s": progress_seals})
    ).hexdigest()
    receipt_name = payload.get("receipt_name")
    if not isinstance(receipt_name, str) or "/" in receipt_name \
            or receipt_name in {"", ".", ".."}:
        _fail("cleanup completion event has an unsafe receipt name")
    raw = _read_evidence_leaf(layout, receipt_name, maximum_bytes=2_000_000)
    if raw is None:
        _fail("cleanup receipt referenced by journal is absent")
    receipt = phase1.strict_json_bytes(raw, label="stale incomplete cleanup receipt")
    phase1.verify_sealed_document(receipt, label="stale incomplete cleanup receipt")
    exact = {
        "schema": _CLEANUP_RECEIPT_SCHEMA,
        "status": "PASS_EXACT_STALE_INCOMPLETE_CLEANUP",
        "session": os.fspath(layout.session),
        "prior_invocation_id": prior_id,
        "prior_invocation_status": prior_status["status"],
        "prior_invocation_journal_head_sha256": prior_entry["seal_sha256"],
        "prior_status_seal_sha256": prior_status["seal_sha256"],
        "manifest_verification_seal_sha256": incomplete_inventory[
            "manifest_verification_seal_sha256"
        ],
        "manifest_seal_sha256": incomplete_inventory["manifest_seal_sha256"],
        "audit_document_seal_sha256": audit_payload.get("audit_seal_sha256"),
        "audit_event_seal_sha256": audit["seal_sha256"],
        "removed_inventory_seal_sha256": payload.get(
            "removed_inventory_seal_sha256"
        ),
        "removed_file_count": payload.get("removed_file_count"),
        "removed_logical_bytes": payload.get("removed_logical_bytes"),
        "removed_allocated_bytes": payload.get("removed_allocated_bytes"),
        "post_cleanup_inventory_seal_sha256": incomplete_inventory[
            "seal_sha256"
        ],
        "post_cleanup_incomplete_count": 0,
        "progress_entry_seal_sha256s": progress_seals,
        "progress_chain_sha256": progress_chain,
        "all_original_inventory_rows_committed": True,
        "supervisor_journal_head_before_final_receipt": entries[
            completed_index - 1
        ]["seal_sha256"],
    }
    for key, expected in exact.items():
        if receipt.get(key) != expected:
            _fail(f"cleanup receipt {key} is not bound to the direct-16 resume")
    if receipt.get("seal_sha256") != payload.get("receipt_seal_sha256"):
        _fail("cleanup receipt seal disagrees with completion journal event")
    if payload.get("progress_chain_sha256") != progress_chain \
            or payload.get("all_original_inventory_rows_committed") is not True:
        _fail("cleanup completion does not bind the committed unlink chain")
    if audited_count and receipt.get("blob_directory_fsynced") is not True:
        _fail("cleanup receipt does not prove the blob directory was fsynced")
    if incomplete_inventory.get("file_count") != 0:
        _fail("cleanup receipt cannot authorize resume while incompletes remain")
    return receipt


def _select_initial_profile(
    layout: phase1.SessionLayout,
    entries: Sequence[dict[str, Any]],
    *,
    primary: TransferProfile,
    ramp: TransferProfile | None,
    incomplete_inventory: dict[str, Any],
) -> tuple[TransferProfile, dict[str, Any]]:
    prior = _latest_finished_context(layout, entries)
    if prior is None:
        return primary, phase1.seal_document(
            {
                "schema": "hawking.kimi_k26.download_supervisor.initial_profile.v1",
                "status": "PRISTINE_OR_NO_FINISHED_RAMP_AUTHORITY_START_PRIMARY_8",
                "workers": 8,
                "prior_invocation_id": None,
                "ramp_resume_authority": None,
                "cleanup_receipt_seal_sha256": None,
            }
        )
    ramp_authority = _measured_ramp_authority(
        entries, before_index=prior["index"]
    )
    if ramp_authority is None:
        return primary, phase1.seal_document(
            {
                "schema": "hawking.kimi_k26.download_supervisor.initial_profile.v1",
                "status": "NO_VERIFIED_PRIOR_RAMP_AUTHORITY_START_PRIMARY_8",
                "workers": 8,
                "prior_invocation_id": prior["entry"]["invocation_id"],
                "ramp_resume_authority": None,
                "cleanup_receipt_seal_sha256": None,
            }
        )
    if ramp is None:
        _fail("prior measured ramp exists but the sealed Phase-1 16 profile is absent")
    cleanup_receipt = _verified_cleanup_receipt_for_resume(
        layout,
        entries,
        prior=prior,
        incomplete_inventory=incomplete_inventory,
    )
    return ramp, phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.download_supervisor.initial_profile.v1",
            "status": "RESUME_DIRECTLY_AT_PHASE1_SEALED_16",
            "workers": 16,
            "prior_invocation_id": prior["entry"]["invocation_id"],
            "prior_invocation_status": prior["status"]["status"],
            "prior_status_seal_sha256": prior["status"]["seal_sha256"],
            "ramp_resume_authority": ramp_authority,
            "cleanup_receipt_seal_sha256": None
            if cleanup_receipt is None
            else cleanup_receipt["seal_sha256"],
        }
    )


def _capacity_document(snapshot: ResourceSnapshot) -> dict[str, Any]:
    if isinstance(snapshot.free_disk_bytes, bool) or snapshot.free_disk_bytes < 0:
        _fail("free-disk sampler returned an invalid value")
    if isinstance(snapshot.session_allocated_bytes, bool) or snapshot.session_allocated_bytes < 0:
        _fail("allocation sampler returned an invalid value")
    remaining_session_budget = max(
        0, SESSION_ALLOCATION_CAP_BYTES - snapshot.session_allocated_bytes
    )
    remaining_projected_need = remaining_session_budget + PROJECTED_HEADROOM_BYTES
    pristine = snapshot.session_allocated_bytes == 0
    runtime_floor_pass = snapshot.free_disk_bytes >= RUNTIME_FREE_DISK_FLOOR_BYTES
    return {
        "prestart_free_disk_bytes": snapshot.free_disk_bytes,
        "required_prestart_free_disk_bytes": PRESTART_FREE_DISK_BYTES,
        "prestart_floor_pass": snapshot.free_disk_bytes >= PRESTART_FREE_DISK_BYTES,
        "prestart_floor_is_launch_gate": pristine,
        "capacity_mode": (
            "FIRST_PRISTINE_ATTEMPT_EXACT_THRESHOLD"
            if pristine
            else "RESUME_CREDITS_EXISTING_CAPPED_SESSION_ALLOCATION"
        ),
        "existing_session_allocated_bytes": snapshot.session_allocated_bytes,
        "projected_source_logical_bytes": PROJECTED_SOURCE_LOGICAL_BYTES,
        "projected_source_allocated_bytes": PROJECTED_SOURCE_ALLOCATED_BYTES,
        "projected_runtime_allocated_bytes": PROJECTED_RUNTIME_ALLOCATED_BYTES,
        "projected_headroom_bytes": PROJECTED_HEADROOM_BYTES,
        "projected_capacity_bytes": PROJECTED_CAPACITY_BYTES,
        "projected_capacity_exact_sum_pass": PROJECTED_CAPACITY_BYTES
        == PRESTART_FREE_DISK_BYTES,
        "remaining_projected_need_bytes": remaining_projected_need,
        "required_free_disk_for_this_attempt_bytes": max(
            RUNTIME_FREE_DISK_FLOOR_BYTES, remaining_projected_need
        ),
        "remaining_projected_capacity_pass": snapshot.free_disk_bytes
        >= remaining_projected_need,
        "runtime_free_disk_floor_bytes": RUNTIME_FREE_DISK_FLOOR_BYTES,
        "runtime_free_disk_floor_pass": runtime_floor_pass,
        "session_allocation_cap_bytes": SESSION_ALLOCATION_CAP_BYTES,
        "session_allocation_cap_pass": snapshot.session_allocated_bytes
        <= SESSION_ALLOCATION_CAP_BYTES,
    }


def preflight(
    layout: phase1.SessionLayout,
    *,
    supplied_plan: dict[str, Any] | None = None,
    hooks: Phase1Hooks | None = None,
    sampler: ResourceSampler | None = None,
    manifest_path: Path = phase1.OFFICIAL_MANIFEST,
    mop_root: Path = phase1.MOP_ROOT,
    shared_xet: Path = phase1.SHARED_HF_XET_ROOT,
) -> dict[str, Any]:
    """Perform a read-only, process-free, network-free executable preflight."""
    selected_hooks = hooks or Phase1Hooks.live()
    selected_sampler = sampler or SystemResourceSampler()
    plan = _validated_plan(
        layout,
        supplied_plan=supplied_plan,
        hooks=selected_hooks,
        manifest_path=manifest_path,
        mop_root=mop_root,
        shared_xet=shared_xet,
    )
    snapshot = selected_sampler.sample(layout)
    capacity = _capacity_document(snapshot)
    if not capacity["session_allocation_cap_pass"]:
        _fail("existing resumable session already exceeds its allocation cap")
    if not capacity["runtime_free_disk_floor_pass"]:
        _fail("free disk is below the invariant runtime headroom floor")
    if not capacity["remaining_projected_capacity_pass"]:
        _fail("free disk cannot hold the remaining exact source/runtime/headroom")
    incomplete_inventory = _scan_nonresumable_incomplete_files(
        layout, manifest_path=manifest_path
    )
    _assert_no_nonresumable_incomplete(incomplete_inventory)
    primary, ramp = _profiles_from_plan(plan, layout)
    return phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.download_supervisor.preflight.v1",
            "status": "PASS_READY_NO_LIVE_ACTION",
            "network_accessed": False,
            "process_started": False,
            "filesystem_written": False,
            "phase1_plan_seal_sha256": plan["seal_sha256"],
            "transfer_runtime_seal_sha256": plan["transfer_runtime"]["seal_sha256"],
            "primary_profile": {
                "name": primary.name,
                "workers": primary.workers,
                "argv_sha256": primary.argv_sha256,
                "environment_sha256": primary.environment_sha256,
            },
            "conditional_ramp_profile": None
            if ramp is None
            else {
                "name": ramp.name,
                "workers": ramp.workers,
                "argv_sha256": ramp.argv_sha256,
                "environment_sha256": ramp.environment_sha256,
            },
            "ramp_authority": "PHASE1_16_PROFILE_REQUIRED"
            if ramp is None
            else "PHASE1_SEALED_16_PROFILE_AVAILABLE_CONDITIONAL_RESTART_ONLY",
            "capacity": capacity,
            "resumable_cache_policy": (
                "PRESERVE_CACHE_NEVER_CONCURRENT_COOPERATING_SUPERVISORS"
            ),
            "nonresumable_incomplete_inventory": {
                "file_count": incomplete_inventory["file_count"],
                "inventory_seal_sha256": incomplete_inventory["seal_sha256"],
                "filename_contract": incomplete_inventory["filename_contract"],
            },
            "manual_process_exclusion": {
                "preflight_status": "DEFERRED_TO_IMMEDIATE_PRE_POPEN_RUN_GATE",
                "scope": "BEST_EFFORT_SNAPSHOT_WITH_POST_SNAPSHOT_RACE",
                "production_method": (
                    "DARWIN_LIBPROC_PLUS_STRUCTURED_KERN_PROCARGS2"
                ),
                "weak_display_command_matching_forbidden": True,
                "residual_race": (
                    "UNCOOPERATIVE_MANUAL_LAUNCH_AFTER_THE_NATIVE_SNAPSHOT_"
                    "CANNOT_BE_ATOMICALLY_EXCLUDED"
                ),
            },
            "source_mutation_capability": "ABSENT",
        }
    )


def _new_invocation_id(clock: Clock) -> str:
    return f"run-{os.getpid()}-{clock.monotonic_ns()}"


def _sample_payload(
    *,
    sequence: int,
    snapshot: ResourceSnapshot,
    network_received_bytes: int | None,
    network_interval_bytes_per_second: float | None,
    network_sustained_bytes_per_second: float | None,
    allocation_growth_bytes_per_second: float | None,
    sample_interval_seconds: float | None,
) -> dict[str, Any]:
    return {
        "sample_sequence": sequence,
        "free_disk_bytes": snapshot.free_disk_bytes,
        "session_allocated_bytes": snapshot.session_allocated_bytes,
        "network_received_bytes": network_received_bytes,
        "network_interval_bytes_per_second": network_interval_bytes_per_second,
        "network_sustained_bytes_per_second": network_sustained_bytes_per_second,
        "network_measurement_method": "ACTIVE_DEFAULT_INTERFACE_IBYTES",
        "allocation_growth_bytes_per_second": allocation_growth_bytes_per_second,
        "resource_sample_interval_seconds": sample_interval_seconds,
        "resource_sample_interval_ceiling_seconds": MONITOR_INTERVAL_SECONDS,
        "runtime_free_disk_floor_bytes": RUNTIME_FREE_DISK_FLOOR_BYTES,
        "session_allocation_cap_bytes": SESSION_ALLOCATION_CAP_BYTES,
    }


def _resource_violation(snapshot: ResourceSnapshot) -> str | None:
    if snapshot.free_disk_bytes < RUNTIME_FREE_DISK_FLOOR_BYTES:
        return "RUNTIME_FREE_DISK_FLOOR_VIOLATED"
    if snapshot.session_allocated_bytes > SESSION_ALLOCATION_CAP_BYTES:
        return "SESSION_ALLOCATION_CAP_VIOLATED"
    if not _capacity_document(snapshot)["remaining_projected_capacity_pass"]:
        return "REMAINING_PROJECTED_CAPACITY_VIOLATED"
    return None


def _terminate_then_kill(
    process: ChildProcess, *, grace_seconds: float
) -> tuple[int, list[str]]:
    actions: list[str] = []

    def signal_child(sig: int, fallback: Callable[[], None], label: str) -> None:
        # Real children are session leaders because Popen uses
        # start_new_session=True.  Signal that exact process group so a helper
        # descendant cannot survive the supervised CLI.  Protocol fakes use
        # their explicit methods and can never target a host PID by accident.
        popen_type = subprocess.Popen
        is_real_popen = isinstance(popen_type, type) and isinstance(process, popen_type)
        if is_real_popen:
            try:
                process_group = os.getpgid(process.pid)
            except ProcessLookupError:
                actions.append(f"{label}_PROCESS_ALREADY_GONE")
                return
            if process_group == process.pid:
                try:
                    os.killpg(process_group, sig)
                except ProcessLookupError:
                    actions.append(f"{label}_PROCESS_GROUP_ALREADY_GONE")
                    return
                actions.append(f"{label}_PROCESS_GROUP_SENT")
                return
            actions.append(f"{label}_GROUP_IDENTITY_MISMATCH_FALLBACK_CHILD_ONLY")
        fallback()
        actions.append(f"{label}_CHILD_SENT")

    signal_child(signal.SIGTERM, process.terminate, "TERMINATE")
    try:
        return process.wait(timeout=grace_seconds), actions
    except subprocess.TimeoutExpired:
        actions.append("TERMINATION_GRACE_EXPIRED")
        signal_child(signal.SIGKILL, process.kill, "KILL")
        return process.wait(timeout=grace_seconds), actions


def _launch_child(
    *,
    profile: TransferProfile,
    plan: dict[str, Any],
    layout: phase1.SessionLayout,
    hooks: Phase1Hooks,
    popen_factory: Callable[..., ChildProcess],
    stdout_descriptor: int,
    stderr_descriptor: int,
    lease_descriptor: int,
    process_auditor: ProcessAuditor,
    manifest_path: Path,
) -> tuple[ChildProcess, dict[str, Any]]:
    # This verifier is intentionally adjacent to Popen.  It re-hashes/rebinds
    # the exact Phase-1 CLI, shebang chain, interpreter, and distributions after
    # all earlier checks and before any network-capable process exists.
    _validate_execution_directories(layout)
    _validate_profile(profile, layout, expected_workers=profile.workers)
    process_audit = process_auditor.audit(layout, plan)
    phase1.verify_sealed_document(process_audit, label="pre-exec process audit")
    if process_audit.get("status") != (
        "PASS_SNAPSHOT_NO_EXISTING_EXACT_SESSION_CACHE_DOWNLOADER_"
        "BEST_EFFORT_WITH_RACE"
    ):
        _fail("pre-exec process audit did not grant launch")
    incomplete_inventory = _scan_nonresumable_incomplete_files(
        layout, manifest_path=manifest_path
    )
    _assert_no_nonresumable_incomplete(incomplete_inventory)
    hooks.verify_runtime(plan["transfer_runtime"])
    process = popen_factory(
        list(profile.argv),
        stdin=subprocess.DEVNULL,
        stdout=stdout_descriptor,
        stderr=stderr_descriptor,
        env=dict(profile.environment),
        shell=False,
        close_fds=True,
        pass_fds=(lease_descriptor,),
        start_new_session=True,
        cwd=os.fspath(layout.session),
    )
    return process, process_audit


@dataclass
class _Metrics:
    samples: int = 0
    min_free: int | None = None
    max_allocation: int = 0
    first_network_counter: int | None = None
    first_network_ns: int | None = None
    last_network_counter: int | None = None
    last_network_ns: int | None = None
    counter_samples: int = 0
    peak_interval_network_bps: float | None = None
    first_allocation: int | None = None
    first_allocation_ns: int | None = None
    last_snapshot: ResourceSnapshot | None = None
    network_samples: list[tuple[int, int]] = dataclasses.field(default_factory=list)

    def record_resource(self, snapshot: ResourceSnapshot, now_ns: int) -> float | None:
        self.samples += 1
        self.min_free = (
            snapshot.free_disk_bytes
            if self.min_free is None
            else min(self.min_free, snapshot.free_disk_bytes)
        )
        self.max_allocation = max(self.max_allocation, snapshot.session_allocated_bytes)
        if self.first_allocation is None:
            self.first_allocation = snapshot.session_allocated_bytes
            self.first_allocation_ns = now_ns
        self.last_snapshot = snapshot
        if self.first_allocation_ns is None or now_ns <= self.first_allocation_ns:
            return None
        return max(0, snapshot.session_allocated_bytes - int(self.first_allocation or 0)) / (
            (now_ns - self.first_allocation_ns) / 1_000_000_000
        )

    def record_network(self, counter: int, now_ns: int) -> tuple[float | None, float | None]:
        interval: float | None = None
        if self.last_network_counter is not None and self.last_network_ns is not None:
            if counter < self.last_network_counter:
                _fail("active-interface byte counter moved backwards")
            delta_ns = now_ns - self.last_network_ns
            if delta_ns > 0:
                interval = (counter - self.last_network_counter) / (
                    delta_ns / 1_000_000_000
                )
                self.peak_interval_network_bps = (
                    interval
                    if self.peak_interval_network_bps is None
                    else max(self.peak_interval_network_bps, interval)
                )
        if self.first_network_counter is None:
            self.first_network_counter = counter
            self.first_network_ns = now_ns
        self.last_network_counter = counter
        self.last_network_ns = now_ns
        self.counter_samples += 1
        self.network_samples.append((now_ns, counter))
        sustained: float | None = None
        if self.first_network_ns is not None and now_ns > self.first_network_ns:
            sustained = (counter - int(self.first_network_counter or 0)) / (
                (now_ns - self.first_network_ns) / 1_000_000_000
            )
        return interval, sustained

    def summary(self) -> dict[str, Any]:
        elapsed = None
        transferred = None
        sustained = None
        if (
            self.first_network_counter is not None
            and self.last_network_counter is not None
            and self.first_network_ns is not None
            and self.last_network_ns is not None
            and self.last_network_ns > self.first_network_ns
        ):
            transferred = self.last_network_counter - self.first_network_counter
            elapsed = (self.last_network_ns - self.first_network_ns) / 1_000_000_000
            sustained = transferred / elapsed
        return {
            "sample_count": self.samples,
            "network_counter_sample_count": self.counter_samples,
            "minimum_free_disk_bytes": self.min_free,
            "maximum_session_allocated_bytes": self.max_allocation,
            "network_received_bytes_estimate": transferred,
            "network_measurement_elapsed_seconds": elapsed,
            "network_sustained_bytes_per_second": sustained,
            "network_peak_interval_bytes_per_second": self.peak_interval_network_bps,
            "network_estimate_scope": (
                "ACTIVE_DEFAULT_INTERFACE_TOTAL_TRAFFIC_DURING_INVOCATION"
            ),
        }


def _ramp_decision(
    metrics: _Metrics,
    *,
    primary_started_ns: int,
    now_ns: int,
    policy: SupervisorPolicy,
) -> tuple[bool, dict[str, Any] | None]:
    warmup_end_ns = primary_started_ns + int(
        policy.ramp_warmup_seconds * 1_000_000_000
    )
    eligible = [
        (sample_ns, counter)
        for sample_ns, counter in metrics.network_samples
        if sample_ns >= warmup_end_ns
    ]
    if not eligible:
        return False, None
    baseline_ns, baseline_counter = eligible[0]
    required_end_ns = baseline_ns + int(
        policy.ramp_measurement_seconds * 1_000_000_000
    )
    if now_ns < required_end_ns:
        return False, None
    completed = [item for item in eligible if item[0] >= required_end_ns]
    if not completed:
        return False, None
    end_ns, end_counter = completed[0]
    window_samples = [
        item for item in eligible if baseline_ns <= item[0] <= end_ns
    ]
    if len(window_samples) < policy.ramp_min_counter_samples:
        return False, None
    if end_counter < baseline_counter or end_ns <= baseline_ns:
        _fail("post-warmup ramp counter window is invalid")
    measured_seconds = (end_ns - baseline_ns) / 1_000_000_000
    measured_bytes = end_counter - baseline_counter
    measured = measured_bytes / measured_seconds
    evidence = {
        "warmup_seconds": policy.ramp_warmup_seconds,
        "configured_measurement_window_seconds": policy.ramp_measurement_seconds,
        "actual_post_warmup_measurement_seconds": measured_seconds,
        "post_warmup_counter_samples": len(window_samples),
        "post_warmup_received_bytes": measured_bytes,
        "measurement_baseline_monotonic_ns": baseline_ns,
        "measurement_end_monotonic_ns": end_ns,
        "measured_network_bytes_per_second": measured,
        "target_network_bytes_per_second": policy.ramp_target_bytes_per_second,
        "below_target": measured < policy.ramp_target_bytes_per_second,
    }
    return bool(evidence["below_target"]), evidence


def run(
    layout: phase1.SessionLayout,
    *,
    invocation_id: str | None = None,
    supplied_plan: dict[str, Any] | None = None,
    hooks: Phase1Hooks | None = None,
    sampler: ResourceSampler | None = None,
    network_probe: NetworkProbe | None = None,
    process_auditor: ProcessAuditor | None = None,
    clock: Clock | None = None,
    popen_factory: Callable[..., ChildProcess] = subprocess.Popen,
    policy: SupervisorPolicy = SupervisorPolicy(),
    manifest_path: Path = phase1.OFFICIAL_MANIFEST,
    mop_root: Path = phase1.MOP_ROOT,
    shared_xet: Path = phase1.SHARED_HF_XET_ROOT,
) -> dict[str, Any]:
    """Execute the exact sealed plan under lease, monitoring, and evidence."""
    policy.validate()
    selected_hooks = hooks or Phase1Hooks.live()
    selected_sampler = sampler or SystemResourceSampler()
    selected_network = network_probe or SystemNetworkProbe()
    selected_process_auditor = process_auditor or DarwinExactProcessAuditor()
    selected_clock = clock or SystemClock()
    selected_id = invocation_id or _new_invocation_id(selected_clock)
    if _INVOCATION_ID.fullmatch(selected_id) is None:
        _fail("invocation id must match [a-z0-9][a-z0-9._-]{0,63}")

    # First gate is wholly read-only and cannot leave an authority artifact.
    preflight_record = preflight(
        layout,
        supplied_plan=supplied_plan,
        hooks=selected_hooks,
        sampler=selected_sampler,
        manifest_path=manifest_path,
        mop_root=mop_root,
        shared_xet=shared_xet,
    )

    previous_umask = os.umask(0o077)
    try:
        with _exclusive_lease(layout) as lease_descriptor, JournalWriter(layout) as journal:
            _assert_no_unfinished_live_child(journal.entries)
            # Rebuild under the lease to close preflight/execute races.
            plan = _validated_plan(
                layout,
                supplied_plan=supplied_plan,
                hooks=selected_hooks,
                manifest_path=manifest_path,
                mop_root=mop_root,
                shared_xet=shared_xet,
            )
            primary, ramp = _profiles_from_plan(plan, layout)
            guarded = selected_sampler.sample(layout)
            capacity = _capacity_document(guarded)
            if not all(
                capacity[key]
                for key in (
                    "remaining_projected_capacity_pass",
                    "runtime_free_disk_floor_pass",
                    "session_allocation_cap_pass",
                )
            ):
                _fail("capacity changed after lease acquisition")

            incomplete_inventory = _scan_nonresumable_incomplete_files(
                layout, manifest_path=manifest_path
            )
            _assert_no_nonresumable_incomplete(incomplete_inventory)
            initial_profile, initial_profile_authority = _select_initial_profile(
                layout,
                journal.entries,
                primary=primary,
                ramp=ramp,
                incomplete_inventory=incomplete_inventory,
            )

            network_path = selected_network.capture_active_default()
            phase1.verify_sealed_document(network_path, label="active network path")
            interface = network_path.get("interface")
            if not isinstance(interface, str) or _INTERFACE.fullmatch(interface) is None:
                _fail("network path did not bind an active default interface")
            initial_counter = selected_network.received_bytes(interface)
            prepared_utc = selected_clock.utc_now()
            prepared_ns = selected_clock.monotonic_ns()
            intent_name = f"invocation.{selected_id}.json"
            status_name = f"status.{selected_id}.json"
            _write_new_document(
                layout,
                intent_name,
                {
                    "schema": "hawking.kimi_k26.download_supervisor.invocation.v1",
                    "status": "PREPARED_BEFORE_NETWORK_CHILD",
                    "invocation_id": selected_id,
                    "pid": None,
                    "prepared_at_utc": prepared_utc,
                    "prepared_monotonic_ns": prepared_ns,
                    "phase1_plan_seal_sha256": plan["seal_sha256"],
                    "transfer_runtime": plan["transfer_runtime"],
                    "preflight_seal_sha256": preflight_record["seal_sha256"],
                    "network_path": network_path,
                    "initial_network_received_bytes": initial_counter,
                    "primary_profile": dataclasses.asdict(primary),
                    "conditional_ramp_profile": None
                    if ramp is None
                    else dataclasses.asdict(ramp),
                    "initial_profile": dataclasses.asdict(initial_profile),
                    "initial_profile_authority": initial_profile_authority,
                    "nonresumable_incomplete_inventory_seal_sha256": (
                        incomplete_inventory["seal_sha256"]
                    ),
                    "capacity": capacity,
                    "umask": "0077",
                    "stdin": "DEVNULL",
                    "shell": False,
                    "environment_mode": "REPLACE_NOT_MERGE",
                    "caffeinate": "OPTIONAL_EXTERNAL_WRAPPER_NOT_IN_CHILD_ARGV",
                    "source_mutation_capability": "ABSENT",
                },
            )
            journal.append(
                event="INVOCATION_PREPARED",
                invocation_id=selected_id,
                timestamp_utc=prepared_utc,
                monotonic_ns=prepared_ns,
                payload={
                    "intent_path": os.fspath(layout.evidence / intent_name),
                    "phase1_plan_seal_sha256": plan["seal_sha256"],
                    "network_path_seal_sha256": network_path["seal_sha256"],
                    "initial_workers": initial_profile.workers,
                    "initial_profile_authority_seal_sha256": (
                        initial_profile_authority["seal_sha256"]
                    ),
                },
            )

            metrics = _Metrics()
            profile = initial_profile
            profile_index = 0
            resumed_directly_at_16 = profile.workers == 16
            ramp_evaluated = resumed_directly_at_16
            ramp_performed = False
            child_exit_code: int | None = None
            final_state = "FAILED_BEFORE_CHILD_EXIT"
            violation: str | None = None
            child_pid: int | None = None
            started_at_utc: str | None = None
            exited_at_utc: str | None = None
            last_network_sample_ns = prepared_ns
            metrics.record_network(initial_counter, prepared_ns)

            while True:
                log_suffix = "primary-8" if profile.workers == 8 else "ramp-16"
                stdout_name = f"stdout.{selected_id}.{log_suffix}.log"
                stderr_name = f"stderr.{selected_id}.{log_suffix}.log"
                stdout_descriptor = _open_evidence_leaf(
                    layout, stdout_name, flags=os.O_WRONLY, exclusive=True
                )
                try:
                    stderr_descriptor = _open_evidence_leaf(
                        layout, stderr_name, flags=os.O_WRONLY, exclusive=True
                    )
                except BaseException:
                    os.close(stdout_descriptor)
                    raise
                process: ChildProcess | None = None
                try:
                    # Exact deterministic plan is rebuilt yet again at the
                    # final execution boundary, then its runtime is reverified
                    # inside _launch_child immediately adjacent to Popen.
                    immediate_plan = _validated_plan(
                        layout,
                        supplied_plan=plan,
                        hooks=selected_hooks,
                        manifest_path=manifest_path,
                        mop_root=mop_root,
                        shared_xet=shared_xet,
                    )
                    immediate_primary, immediate_ramp = _profiles_from_plan(
                        immediate_plan, layout
                    )
                    immediate_profile = (
                        immediate_primary if profile.workers == 8 else immediate_ramp
                    )
                    if immediate_profile is None:
                        _fail("Phase-1 16-worker profile disappeared before restart")
                    if immediate_profile != profile:
                        _fail("selected transfer profile changed before execution")
                    process, process_audit = _launch_child(
                        profile=profile,
                        plan=immediate_plan,
                        layout=layout,
                        hooks=selected_hooks,
                        popen_factory=popen_factory,
                        stdout_descriptor=stdout_descriptor,
                        stderr_descriptor=stderr_descriptor,
                        lease_descriptor=lease_descriptor,
                        process_auditor=selected_process_auditor,
                        manifest_path=manifest_path,
                    )
                    child_pid = int(process.pid)
                    if child_pid <= 1:
                        _fail("downloader returned an unsafe PID")
                    started_at_utc = selected_clock.utc_now()
                    profile_started_ns = selected_clock.monotonic_ns()
                    next_sample_ns = profile_started_ns
                    previous_resource_sample_ns: int | None = None
                    started_name = f"started.{selected_id}.{log_suffix}.json"
                    _write_new_document(
                        layout,
                        started_name,
                        {
                            "schema": "hawking.kimi_k26.download_supervisor.started.v1",
                            "status": "NETWORK_CHILD_STARTED",
                            "invocation_id": selected_id,
                            "profile_index": profile_index,
                            "profile": profile.name,
                            "workers": profile.workers,
                            "pid": child_pid,
                            "argv": list(profile.argv),
                            "argv_sha256": profile.argv_sha256,
                            "environment": profile.environment,
                            "environment_sha256": profile.environment_sha256,
                            "phase1_plan_seal_sha256": immediate_plan["seal_sha256"],
                            "transfer_runtime_seal_sha256": immediate_plan[
                                "transfer_runtime"
                            ]["seal_sha256"],
                            "started_at_utc": started_at_utc,
                            "started_monotonic_ns": profile_started_ns,
                            "pre_exec_process_audit": process_audit,
                            "stdout_path": os.fspath(layout.evidence / stdout_name),
                            "stderr_path": os.fspath(layout.evidence / stderr_name),
                        },
                    )
                    journal.append(
                        event="CHILD_STARTED",
                        invocation_id=selected_id,
                        timestamp_utc=started_at_utc,
                        monotonic_ns=profile_started_ns,
                        payload={
                            "pid": child_pid,
                            "profile": profile.name,
                            "workers": profile.workers,
                            "argv_sha256": profile.argv_sha256,
                            "environment_sha256": profile.environment_sha256,
                            "process_audit_seal_sha256": process_audit[
                                "seal_sha256"
                            ],
                            "started_path": os.fspath(layout.evidence / started_name),
                        },
                    )

                    while True:
                        now_ns = selected_clock.monotonic_ns()
                        resource_sample_interval = (
                            None
                            if previous_resource_sample_ns is None
                            else (now_ns - previous_resource_sample_ns) / 1_000_000_000
                        )
                        cadence_violation = (
                            "MONITOR_CADENCE_CEILING_VIOLATED"
                            if resource_sample_interval is not None
                            and resource_sample_interval > MONITOR_INTERVAL_SECONDS
                            else None
                        )
                        previous_resource_sample_ns = now_ns
                        snapshot = selected_sampler.sample(layout)
                        allocation_bps = metrics.record_resource(snapshot, now_ns)
                        network_counter: int | None = None
                        interval_bps: float | None = None
                        sustained_bps = metrics.summary()[
                            "network_sustained_bytes_per_second"
                        ]
                        if (
                            now_ns - last_network_sample_ns
                            >= int(policy.network_sample_interval_seconds * 1_000_000_000)
                        ):
                            network_counter = selected_network.received_bytes(interface)
                            interval_bps, sustained_bps = metrics.record_network(
                                network_counter, now_ns
                            )
                            last_network_sample_ns = now_ns
                        journal.append(
                            event="RESOURCE_SAMPLE",
                            invocation_id=selected_id,
                            timestamp_utc=selected_clock.utc_now(),
                            monotonic_ns=now_ns,
                            payload={
                                "pid": child_pid,
                                "profile": profile.name,
                                **_sample_payload(
                                    sequence=metrics.samples,
                                    snapshot=snapshot,
                                    network_received_bytes=network_counter,
                                    network_interval_bytes_per_second=interval_bps,
                                    network_sustained_bytes_per_second=sustained_bps,
                                    allocation_growth_bytes_per_second=allocation_bps,
                                    sample_interval_seconds=resource_sample_interval,
                                ),
                            },
                        )
                        code = process.poll()
                        if code is not None:
                            child_exit_code = int(code)
                            break
                        violation = cadence_violation or _resource_violation(snapshot)
                        if violation is not None:
                            journal.append(
                                event="RESOURCE_GUARD_TRIGGERED",
                                invocation_id=selected_id,
                                timestamp_utc=selected_clock.utc_now(),
                                monotonic_ns=selected_clock.monotonic_ns(),
                                payload={
                                    "pid": child_pid,
                                    "profile": profile.name,
                                    "violation": violation,
                                    "free_disk_bytes": snapshot.free_disk_bytes,
                                    "session_allocated_bytes": snapshot.session_allocated_bytes,
                                },
                            )
                            child_exit_code, actions = _terminate_then_kill(
                                process, grace_seconds=policy.termination_grace_seconds
                            )
                            journal.append(
                                event="RESOURCE_GUARD_PROCESS_STOPPED",
                                invocation_id=selected_id,
                                timestamp_utc=selected_clock.utc_now(),
                                monotonic_ns=selected_clock.monotonic_ns(),
                                payload={
                                    "pid": child_pid,
                                    "exit_code": child_exit_code,
                                    "actions": actions,
                                    "resumable_cache_preserved": True,
                                },
                            )
                            break
                        if (
                            profile.workers == 8
                            and not ramp_evaluated
                            and policy.permit_phase1_authorized_ramp
                        ):
                            should_ramp, evidence = _ramp_decision(
                                metrics,
                                primary_started_ns=profile_started_ns,
                                now_ns=now_ns,
                                policy=policy,
                            )
                            if evidence is not None:
                                ramp_evaluated = True
                                journal.append(
                                    event="RAMP_16_EVALUATED",
                                    invocation_id=selected_id,
                                    timestamp_utc=selected_clock.utc_now(),
                                    monotonic_ns=now_ns,
                                    payload={
                                        "pid": child_pid,
                                        **evidence,
                                        "phase1_16_profile_available": ramp is not None,
                                    },
                                )
                            if should_ramp and ramp is not None:
                                child_exit_code, actions = _terminate_then_kill(
                                    process,
                                    grace_seconds=policy.termination_grace_seconds,
                                )
                                journal.append(
                                    event="RAMP_8_PROCESS_STOPPED",
                                    invocation_id=selected_id,
                                    timestamp_utc=selected_clock.utc_now(),
                                    monotonic_ns=selected_clock.monotonic_ns(),
                                    payload={
                                        "pid": child_pid,
                                        "exit_code": child_exit_code,
                                        "actions": actions,
                                        "prior_pid_fully_exited": process.poll() is not None,
                                        "same_resumable_cache": True,
                                    },
                                )
                                if process.poll() is None:
                                    _fail("8-worker PID did not fully exit before ramp")
                                ramp_performed = True
                                break
                        next_sample_ns += int(
                            policy.monitor_interval_seconds * 1_000_000_000
                        )
                        remaining_ns = next_sample_ns - selected_clock.monotonic_ns()
                        if remaining_ns > 0:
                            selected_clock.sleep(remaining_ns / 1_000_000_000)
                except BaseException as original_fault:
                    # Once Popen has returned, no supervisor-internal failure is
                    # allowed to strand a network-capable child.  The inherited
                    # lease descriptor also remains locked by the child until
                    # exec exit, closing the small Popen/start-record crash gap.
                    fault_actions: list[str] = []
                    fault_exit_code: int | None = None
                    cleanup_notes: list[str] = []
                    if process is not None:
                        try:
                            observed = process.poll()
                            if observed is None:
                                fault_exit_code, fault_actions = _terminate_then_kill(
                                    process,
                                    grace_seconds=policy.termination_grace_seconds,
                                )
                            else:
                                fault_exit_code = int(observed)
                                fault_actions = ["CHILD_ALREADY_EXITED_AT_FAULT"]
                        except BaseException as cleanup_fault:
                            cleanup_notes.append(
                                f"child stop failed: {_safe_error(cleanup_fault)}"
                            )
                    fault_utc = selected_clock.utc_now()
                    fault_ns = selected_clock.monotonic_ns()
                    try:
                        journal.append(
                            event="SUPERVISOR_FAULT_CHILD_STOPPED",
                            invocation_id=selected_id,
                            timestamp_utc=fault_utc,
                            monotonic_ns=fault_ns,
                            payload={
                                "pid": child_pid,
                                "profile": profile.name,
                                "fault": _safe_error(original_fault),
                                "exit_code": fault_exit_code,
                                "stop_actions": fault_actions,
                                "resumable_cache_preserved": True,
                            },
                        )
                        if child_pid is not None and fault_exit_code is not None:
                            journal.append(
                                event="CHILD_EXITED",
                                invocation_id=selected_id,
                                timestamp_utc=fault_utc,
                                monotonic_ns=fault_ns,
                                payload={
                                    "pid": child_pid,
                                    "profile": profile.name,
                                    "workers": profile.workers,
                                    "exit_code": fault_exit_code,
                                    "exit_reason": "SUPERVISOR_FAULT_FAIL_CLOSED_STOP",
                                },
                            )
                    except BaseException as evidence_fault:
                        cleanup_notes.append(
                            f"fault journal evidence failed: {_safe_error(evidence_fault)}"
                        )
                    try:
                        _write_new_document(
                            layout,
                            status_name,
                            {
                                "schema": "hawking.kimi_k26.download_supervisor.status.v1",
                                "status": "SUPERVISOR_FAULT_CHILD_STOPPED_FAIL_CLOSED",
                                "invocation_id": selected_id,
                                "pid": child_pid,
                                "argv_sha256": profile.argv_sha256,
                                "environment_sha256": profile.environment_sha256,
                                "phase1_plan_seal_sha256": plan["seal_sha256"],
                                "transfer_runtime_seal_sha256": plan[
                                    "transfer_runtime"
                                ]["seal_sha256"],
                                "prepared_at_utc": prepared_utc,
                                "started_at_utc": started_at_utc,
                                "fault_at_utc": fault_utc,
                                "exit_code": fault_exit_code,
                                "fault": _safe_error(original_fault),
                                "stop_actions": fault_actions,
                                "cleanup_notes": cleanup_notes,
                                "resumable_cache_preserved": True,
                                "source_mutation_capability": "ABSENT",
                            },
                        )
                    except BaseException as status_fault:
                        cleanup_notes.append(
                            f"fault status evidence failed: {_safe_error(status_fault)}"
                        )
                    for note in cleanup_notes:
                        with contextlib.suppress(AttributeError):
                            original_fault.add_note(note)
                    raise
                finally:
                    with contextlib.suppress(OSError):
                        os.close(stdout_descriptor)
                    with contextlib.suppress(OSError):
                        os.close(stderr_descriptor)

                if process is None or child_pid is None or child_exit_code is None:
                    _fail("downloader lifecycle ended without a bound child exit")
                exited_at_utc = selected_clock.utc_now()
                journal.append(
                    event="CHILD_EXITED",
                    invocation_id=selected_id,
                    timestamp_utc=exited_at_utc,
                    monotonic_ns=selected_clock.monotonic_ns(),
                    payload={
                        "pid": child_pid,
                        "profile": profile.name,
                        "workers": profile.workers,
                        "exit_code": child_exit_code,
                    },
                )
                if violation is not None:
                    final_state = "RESOURCE_GUARD_TERMINATED_RESUMABLE_CACHE_PRESERVED"
                    break
                if ramp_performed and profile.workers == 8:
                    if ramp is None:
                        _fail("internal ramp authority mismatch")
                    profile = ramp
                    profile_index += 1
                    child_exit_code = None
                    continue
                final_state = (
                    "DOWNLOAD_COMMAND_EXITED_ZERO_SOURCE_VERIFICATION_REQUIRED"
                    if child_exit_code == 0
                    else "CHILD_EXIT_NONZERO_RESUMABLE_CACHE_PRESERVED"
                )
                break

            final_network: dict[str, Any]
            try:
                final_network = selected_network.capture_active_default()
                phase1.verify_sealed_document(final_network, label="final network path")
            except BaseException as exc:  # evidence the fault; source remains untouched
                final_network = {
                    "status": "FINAL_NETWORK_PATH_CAPTURE_FAILED",
                    "error": _safe_error(exc),
                }
            finished_utc = selected_clock.utc_now()
            finished_ns = selected_clock.monotonic_ns()
            status_value = phase1.seal_document(
                {
                    "schema": "hawking.kimi_k26.download_supervisor.status.v1",
                    "status": final_state,
                    "invocation_id": selected_id,
                    "pid": child_pid,
                    "argv_sha256": profile.argv_sha256,
                    "environment_sha256": profile.environment_sha256,
                    "phase1_plan_seal_sha256": plan["seal_sha256"],
                    "transfer_runtime_seal_sha256": plan["transfer_runtime"][
                        "seal_sha256"
                    ],
                    "prepared_at_utc": prepared_utc,
                    "started_at_utc": started_at_utc,
                    "exited_at_utc": exited_at_utc,
                    "finished_at_utc": finished_utc,
                    "finished_monotonic_ns": finished_ns,
                    "exit_code": child_exit_code,
                    "resource_violation": violation,
                    "primary_workers": 8,
                    "initial_workers": initial_profile.workers,
                    "final_workers": profile.workers,
                    "resumed_directly_at_16": resumed_directly_at_16,
                    "initial_profile_authority": initial_profile_authority,
                    "ramp_evaluated": ramp_evaluated,
                    "ramp_performed": ramp_performed,
                    "metrics": metrics.summary(),
                    "initial_network_path": network_path,
                    "final_network_path": final_network,
                    "stdout_path": os.fspath(
                        layout.evidence
                        / f"stdout.{selected_id}.{'primary-8' if profile.workers == 8 else 'ramp-16'}.log"
                    ),
                    "stderr_path": os.fspath(
                        layout.evidence
                        / f"stderr.{selected_id}.{'primary-8' if profile.workers == 8 else 'ramp-16'}.log"
                    ),
                    "journal_path": os.fspath(layout.evidence / _JOURNAL_NAME),
                    "journal_head_before_status": journal.head,
                    "resumable_cache_preserved": True,
                    "cooperating_supervisor_max_concurrency": 1,
                    "manual_process_exclusion_scope": (
                        "BEST_EFFORT_PRE_EXEC_SNAPSHOT_WITH_POST_SNAPSHOT_RACE"
                    ),
                    "source_mutation_capability": "ABSENT",
                }
            )
            _write_new_document(layout, status_name, status_value)
            journal.append(
                event="INVOCATION_FINISHED",
                invocation_id=selected_id,
                timestamp_utc=finished_utc,
                monotonic_ns=finished_ns,
                payload={
                    "pid": child_pid,
                    "status": final_state,
                    "exit_code": child_exit_code,
                    "status_path": os.fspath(layout.evidence / status_name),
                    "status_seal_sha256": status_value["seal_sha256"],
                    "resumable_cache_preserved": True,
                },
            )
            return status_value
    finally:
        os.umask(previous_umask)


def status(layout: phase1.SessionLayout) -> dict[str, Any]:
    """Read and verify durable state without starting a process or network I/O."""
    phase1._private_directory_metadata(layout.evidence)  # noqa: SLF001
    raw = _read_evidence_leaf(
        layout, _JOURNAL_NAME, maximum_bytes=_MAX_JOURNAL_BYTES
    )
    entries = [] if raw is None else _verify_journal_bytes(raw)
    latest_finished = next(
        (
            entry
            for entry in reversed(entries)
            if entry.get("event") == "INVOCATION_FINISHED"
        ),
        None,
    )
    active: dict[int, str] = {}
    for entry in entries:
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            continue
        pid = payload.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int):
            continue
        if entry.get("event") == "CHILD_STARTED":
            active[pid] = str(entry.get("invocation_id"))
        elif entry.get("event") == "CHILD_EXITED":
            active.pop(pid, None)
    return phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.download_supervisor.read_status.v1",
            "status": "IDLE_NO_INVOCATIONS" if not entries else "DURABLE_STATE_VERIFIED",
            "network_accessed": False,
            "process_started": False,
            "filesystem_written": False,
            "journal_path": os.fspath(layout.evidence / _JOURNAL_NAME),
            "journal_entries": len(entries),
            "journal_head_sha256": entries[-1]["seal_sha256"] if entries else _GENESIS,
            "latest_finished": latest_finished,
            "unfinished_children": [
                {"pid": pid, "invocation_id": invocation}
                for pid, invocation in sorted(active.items())
            ],
            "resumable_cache_policy": (
                "PRESERVE_CACHE_NEVER_CONCURRENT_COOPERATING_SUPERVISORS"
            ),
            "source_mutation_capability": "ABSENT",
        }
    )


def _print_json(value: dict[str, Any]) -> None:
    sys.stdout.buffer.write(phase1.canonical_json(value) + b"\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supervise the exact sealed Kimi-K2.6 HF/Xet transfer.",
        epilog=(
            "Optional power assertion must wrap this supervisor externally: "
            "caffeinate -dimsu -- python3.12 ... run. It is never inserted "
            "into the pinned downloader argv."
        ),
    )
    parser.add_argument(
        "command", choices=("preflight", "run", "status")
    )
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--invocation-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        layout = phase1.layout_for(args.session, parent=phase1.SESSION_PARENT)
        if args.command == "preflight":
            value = preflight(layout)
        elif args.command == "status":
            value = status(layout)
        else:
            value = run(
                layout,
                invocation_id=args.invocation_id,
            )
        _print_json(value)
        return 0
    except (DownloadSupervisorError, phase1.ReleaseCycleError, OSError) as exc:
        _print_json(
            phase1.seal_document(
                {
                    "schema": "hawking.kimi_k26.download_supervisor.error.v1",
                    "status": "FAIL_CLOSED",
                    "error": _safe_error(exc),
                    "network_action_started": False
                    if args.command != "run"
                    else "UNKNOWN_CHECK_DURABLE_JOURNAL",
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
