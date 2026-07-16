#!/usr/bin/env python3
"""Owner-free, lease-held launcher for the default-off Metal RHT probe."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any

CRATE = Path(__file__).resolve().parents[1]
ROOT = CRATE.parents[1]
ADMITTED_ROOT = (ROOT / "build" / "native-execution").resolve()
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))

import doctor_v5_stacked_admission as stacked  # noqa: E402
import doctor_v5_local_observer as local_observer  # noqa: E402
import ram_scheduler  # noqa: E402

SCHEMA = "hawking.strand.native-probe-admission.v1"
GENERATOR_SCHEMA = "hawking.strand.native-probe-launcher.v1"
HEAVY_LOCK = ROOT / "reports" / "cron" / "studio_heavy.lock"
GENERATOR_SOURCE = Path(__file__).resolve()
OWNER_PATTERN_SOURCE = (CONDENSE / "doctor_v5_local_observer.py").resolve()


class ProbeAdmissionError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def confined(path: Path, *, label: str, require_file: bool = False) -> Path:
    if path.is_symlink():
        raise ProbeAdmissionError(f"{label} must not be a symlink")
    resolved = path.resolve()
    try:
        resolved.relative_to(ADMITTED_ROOT)
    except ValueError as error:
        raise ProbeAdmissionError(f"{label} must be below {ADMITTED_ROOT}") from error
    if require_file and (
        not resolved.is_file() or not stat.S_ISREG(resolved.stat().st_mode)
    ):
        raise ProbeAdmissionError(f"{label} must be an existing regular file")
    return resolved


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path = confined(path, label="admission receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ProbeAdmissionError("refusing to replace an admission receipt")
    temporary = path.with_name(f".{path.name}.worker-{os.getpid()}-{time.time_ns()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        payload = (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while sealing admission receipt")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        sync_directory(path.parent)
        os.link(temporary, path, follow_symlinks=False)
        sync_directory(path.parent)
        temporary.unlink()
        sync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def active_heavy_owners_fail_closed() -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ProbeAdmissionError(f"cannot take heavy-owner process snapshot: {error}") from error
    owners: list[dict[str, Any]] = []
    own_pid = os.getpid()
    for line in completed.stdout.splitlines():
        pid_text, separator, command = line.strip().partition(" ")
        if not separator:
            continue
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid != own_pid and any(
            pattern.search(command)
            for pattern in local_observer.HEAVY_COMMAND_PATTERNS
        ):
            owners.append({"pid": pid, "command": command})
    return owners


def lease_is_already_owned(descriptor: int, path: Path) -> bool:
    owned = os.fstat(descriptor)
    named = path.stat()
    if not stat.S_ISREG(owned.st_mode) or (owned.st_dev, owned.st_ino) != (
        named.st_dev,
        named.st_ino,
    ):
        return False
    fresh = os.open(path, os.O_RDWR)
    try:
        try:
            fcntl.flock(fresh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(fresh, fcntl.LOCK_UN)
            return False
    finally:
        os.close(fresh)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def launch(args: argparse.Namespace) -> int:
    staging_root = confined(args.staging_root, label="staging root")
    staging_root.mkdir(parents=True, exist_ok=True)
    probe = confined(args.probe, label="probe binary", require_file=True)
    if not os.access(probe, os.X_OK):
        raise ProbeAdmissionError("probe binary is not executable")
    receipt = confined(args.receipt, label="probe receipt")
    admission = confined(args.admission_receipt, label="admission receipt")
    for path in (receipt, admission):
        try:
            path.relative_to(staging_root)
        except ValueError as error:
            raise ProbeAdmissionError("probe receipts must be inside the staging root") from error
        if path.exists() or path.is_symlink():
            raise ProbeAdmissionError(f"refusing to replace staged receipt: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    HEAVY_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lease = HEAVY_LOCK.open("a+")
    try:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ProbeAdmissionError("shared heavy lease is held") from error
        if not lease_is_already_owned(lease.fileno(), HEAVY_LOCK):
            raise ProbeAdmissionError("heavy lease ownership could not be proven")
        owners = active_heavy_owners_fail_closed()
        if owners:
            raise ProbeAdmissionError("heavy owners remain after exclusive lease acquisition")
        snapshot = ram_scheduler.resource_snapshot(ROOT)
        thermal = stacked._thermal_probe()
        health = stacked.resource_health(snapshot, thermal)
        if health.get("ok") is not True:
            raise ProbeAdmissionError(
                "resource admission is not green: " + "; ".join(health.get("blockers", []))
            )
        now = time.time_ns()
        lease_stat = os.fstat(lease.fileno())
        document: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "admitted",
            "generator_schema": GENERATOR_SCHEMA,
            "generator_path": str(GENERATOR_SOURCE),
            "generator_sha256": sha256_file(GENERATOR_SOURCE),
            "owner_pattern_source_path": str(OWNER_PATTERN_SOURCE),
            "owner_pattern_source_sha256": sha256_file(OWNER_PATTERN_SOURCE),
            "generated_unix_ns": now,
            "expires_unix_ns": now + 30_000_000_000,
            "staging_root": str(staging_root),
            "probe": str(probe),
            "probe_sha256": sha256_file(probe),
            "active_heavy_owner_count": 0,
            "owners_rechecked_under_lease": True,
            "lease_path": str(HEAVY_LOCK.resolve()),
            "lease_fd": lease.fileno(),
            "lease_device": lease_stat.st_dev,
            "lease_inode": lease_stat.st_ino,
            "resource_health": health,
            "resource_health_ok": True,
            "runtime_activation": False,
        }
        document["document_sha256"] = canonical_sha256(document)
        atomic_json_exclusive(admission, document)
        admission_sha256 = sha256_file(admission)
        command = [
            str(probe),
            "--dispatch",
            "--staging-root",
            str(staging_root),
            "--receipt",
            str(receipt),
            "--admission-receipt",
            str(admission),
            "--lease-fd",
            str(lease.fileno()),
        ]
        environment = dict(os.environ)
        environment["HAWKING_NATIVE_PROBE_ADMITTED"] = "1"
        environment["HAWKING_NATIVE_PROBE_ADMISSION_SHA256"] = admission_sha256
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            pass_fds=(lease.fileno(),),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise ProbeAdmissionError(
                f"physical probe failed ({completed.returncode}): {completed.stderr[-2000:]}"
            )
        if not receipt.is_file():
            raise ProbeAdmissionError("physical probe returned without its staged receipt")
        print(completed.stdout, end="")
        return 0
    finally:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
        finally:
            lease.close()


def selftest() -> int:
    inside = confined(ADMITTED_ROOT / "probe-selftest" / "receipt.json", label="inside")
    assert inside.is_relative_to(ADMITTED_ROOT)
    try:
        confined(Path("/tmp/native-probe-escape.json"), label="escape")
    except ProbeAdmissionError:
        pass
    else:
        raise AssertionError("path escape was not rejected")
    print("native_probe.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--staging-root", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--admission-receipt", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    try:
        if args.selftest:
            return selftest()
        if any(
            value is None
            for value in (
                args.staging_root,
                args.probe,
                args.receipt,
                args.admission_receipt,
            )
        ):
            raise ProbeAdmissionError("dispatch requires all staging/probe/receipt paths")
        return launch(args)
    except (OSError, subprocess.SubprocessError, ProbeAdmissionError) as error:
        print(json.dumps({"status": "refused", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
