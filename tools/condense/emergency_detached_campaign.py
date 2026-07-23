#!/usr/bin/env python3.12
"""Crash-safe detached handoff from Kimi recovery into the GLM live boundary.

This controller is deliberately small in authority.  It may run the committed
Kimi recovery verifier and the committed Phase-2 release CLI, followed by the
supported GLM offline-plan verifier and (only when all normal controller and
Telegram authority files already exist) the production live-Xet driver.  It has
no deletion implementation of its own.

``launchd`` invokes ``tick`` through ``caffeinate``.  Each tick takes an anchored
singleton flock, reconciles any child left by a prior crash, and advances the
durable hash-chained state machine.  Missing Telegram/adapter/controller
authority produces a durable ``BLOCKED_NEEDS_AUTHORITY`` terminal state; this
program never manufactures or bypasses that authority.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import plistlib
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


SCHEMA = "hawking.emergency_detached_campaign.state.v1"
EVENT_SCHEMA = "hawking.emergency_detached_campaign.event.v1"
LABEL = "com.hawking.emergency.detached.campaign"
PYTHON = Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12")
EXECUTOR_ROOT = Path("/Users/scammermike/Downloads/hawking-emergency-executor")
REPO_ROOT = EXECUTOR_ROOT
CONTROLLER_ROOT = EXECUTOR_ROOT
GLM_PYTHON = Path("/Users/scammermike/Downloads/hawking/.venv/glm52/bin/python")
EXPECTED_EXECUTOR_BRANCH = "campaign/glm52-bf16-xet-gravity"
OFFICIAL_CAMPAIGN_REF = "origin/campaign/glm52-bf16-xet-gravity"
OFFICIAL_CAMPAIGN_REMOTE_REF = "refs/heads/campaign/glm52-bf16-xet-gravity"
OFFICIAL_ORIGIN_URL = "git@github.com:joshuahickscorp/hawking.git"
EXPECTED_OPERATION_FILE_SHA256 = {
    "GLM52_XET_AUTOTUNE_PLAN.json": "0c48a53f6c1f0b9cf338ef223ceaf4fe0d92cb908615c3fe68479c4a88dec6ce",
    "reports/condense/kimi_k26/KIMI_K26_OFFICIAL_MANIFEST.json": "a4584e22df830b040d87e3ce1b3d17fe9e13221bc3849c8f791073d9fa8c07fe",
    "tools/condense/glm52_xet_autotune.py": "6ba224ddef71ac2b3b5d968b4bd92707a5e0d9db37b3eb060365e92d7597548f",
    "tools/condense/glm52_xet_live_driver.py": "f9ad68b36c28f6f4b6b7d55a2bf0b824c7a91aac88ef7c7a030eb9337224af0d",
    "tools/condense/kimi_k26_phase2_recovery.py": "9aee86831902e655620ef7205c89ca945ffebdebf84cd512e1bcf3e25f51d477",
    "tools/condense/kimi_k26_phase2_release.py": "9770f52c92e210db9204c47f2ac4e9b1f499d8b10002b9131ba20de6217185d1",
    "tools/condense/kimi_k26_release_cycle.py": "109d29d1acf18c1a20cc5156284f1c5a7566e295531f0324c46e055be3f5bdf7",
    "tools/glm52_gravity.py": "181ea54be84a313261916979eca2d36a6598d8092762ed2ba8ed827142f068eb",
}
SESSION = Path(
    "/Users/scammermike/Library/Application Support/Hawking/"
    "KimiK26ReleaseCycle/kimi-stop2-20260721"
)
STATE_ROOT = Path(
    "/Users/scammermike/Library/Application Support/Hawking/"
    "EmergencyDetachedCampaign-v2"
)
# Tests and the install subcommand validate the plist adjacent to the invoked
# controller source.  The plist invocation itself is deliberately frozen to a
# separate, clean controller worktree.
SOURCE_ROOT = Path(__file__).resolve().parents[2]
PLIST_SOURCE = SOURCE_ROOT / "deploy/launchd/com.hawking.emergency.detached.campaign.plist"
PLIST_DESTINATION = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"

RECOVERY_RECEIPT_NAME = "KIMI_K26_PHASE2_RECOVERY.json"
RELEASE_BUNDLE_NAME = "KIMI_K26_PHASE2_RELEASE_BUNDLE.json"
RELEASE_CONFIRMATION_NAME = "KIMI_K26_PHASE2_RELEASE_CONFIRMATION.json"
RELEASE_RECEIPT_NAME = "KIMI_K26_PHASE2_RELEASE_RECEIPT.json"
GLM_PLAN_RESULT_NAME = "GLM52_XET_AUTOTUNE_PLAN_VERIFY.json"
GLM_LIVE_RESULT_NAME = "GLM52_XET_AUTOTUNE_LIVE_RESULT.json"

MINIMUM_AVAILABLE_RAM_BYTES = 16 * 1024**3
MAXIMUM_SWAP_USED_BYTES = 8 * 1024**3
MAXIMUM_SWAP_GROWTH_BYTES = 0
KIMI_EMERGENCY_DISK_FLOOR_BYTES = 5 * 1024**3
KIMI_OPERATIONAL_DISK_FLOOR_BYTES = 25_322_093_168
GLM_REQUIRED_DISK_FLOOR_BYTES = 416_036_394_619
RESOURCE_BACKOFF_SECONDS = 60
CHILD_POLL_SECONDS = 1.0
CHILD_TERM_GRACE_SECONDS = 30.0
MAX_ACTION_ATTEMPTS = 3
MAX_CAPTURE_BYTES = 8 * 1024 * 1024
ZERO_HASH = "0" * 64
AUTHORIZATION_SCHEMA = "hawking.emergency_detached_campaign.kimi_release_authorization.v1"

PHASE_WAIT_RECOVERY = "WAIT_KIMI_PHASE2_RECOVERY"
PHASE_VERIFY_RECOVERY = "VERIFY_KIMI_PHASE2_RECOVERY"
PHASE_RELEASE_AUDIT = "KIMI_RELEASE_AUDIT"
PHASE_RELEASE_CONFIRM = "KIMI_RELEASE_CONFIRM"
PHASE_RELEASE_EXECUTE = "KIMI_RELEASE_EXECUTE"
PHASE_RELEASE_VERIFY = "KIMI_RELEASE_VERIFY"
PHASE_GLM_PLAN_VERIFY = "GLM_PLAN_VERIFY"
PHASE_GLM_LIVE_GATE = "GLM_LIVE_GATE"
PHASE_BLOCKED_AUTHORITY = "BLOCKED_NEEDS_AUTHORITY"
PHASE_BLOCKED = "BLOCKED_FAIL_CLOSED"

ACTION_RECOVERY_GENERATE = "RECOVERY_GENERATE"
ACTION_RECOVERY_VERIFY = "RECOVERY_VERIFY"
ACTION_RELEASE_AUDIT = "RELEASE_AUDIT"
ACTION_RELEASE_CONFIRM = "RELEASE_CONFIRM"
ACTION_RELEASE_EXECUTE = "RELEASE_EXECUTE"
ACTION_RELEASE_VERIFY = "RELEASE_VERIFY"
ACTION_GLM_PLAN_VERIFY = "GLM_PLAN_VERIFY"
ACTION_GLM_STATUS = "GLM_STATUS"
ACTION_GLM_LIVE = "GLM_LIVE"

RECOVERY_CAPSULE_STATUS_TO_MODE = {
    "PASS_EXACT_RECOVERED_CAPSULE": "LEGACY_EXACT_BYTES",
    "PASS_DETERMINISTIC_SEMANTIC_REPLACEMENT_CAPSULE": "DETERMINISTIC_SEMANTIC_REPLACEMENT",
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_SWAP_USED = re.compile(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)([BKMGTPE])\b", re.I)
_VM_PAGE = re.compile(r"page size of\s+([0-9]+)\s+bytes", re.I)
_BOOT_SECONDS = re.compile(r"\bsec\s*=\s*([0-9]+)\b")
_JOURNAL_TEMP = re.compile(
    r"^\.[0-9]{8}\.json\.[1-9][0-9]*\.[1-9][0-9]*\.tmp$"
)


class HandoffError(RuntimeError):
    """A fail-closed handoff invariant failed."""


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HandoffError(f"value is not canonical JSON: {exc}") from exc


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in dict(value).items() if key != "seal_sha256"}
    return {**unsigned, "seal_sha256": hashlib.sha256(canonical(unsigned)).hexdigest()}


def verify_seal(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    document = dict(value)
    recorded = document.get("seal_sha256")
    if not isinstance(recorded, str) or not _HEX64.fullmatch(recorded):
        raise HandoffError(f"{label} has no valid seal")
    if seal(document)["seal_sha256"] != recorded:
        raise HandoffError(f"{label} seal mismatch")
    return document


def _strict_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise HandoffError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                HandoffError(f"{label} contains non-finite value {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HandoffError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise HandoffError(f"{label} root is not an object")
    return value


def _ensure_private_directory(path: Path, *, create: bool = True) -> None:
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise HandoffError(f"private path is not an anchored directory: {path}")
    if metadata.st_uid != os.getuid():
        raise HandoffError(f"private directory owner differs: {path}")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        os.chmod(path, 0o700)
        metadata = path.lstat()
        if stat.S_IMODE(metadata.st_mode) != 0o700:
            raise HandoffError(f"cannot make directory private: {path}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_private_bytes(path: Path, raw: bytes) -> None:
    _ensure_private_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    # A failed replace intentionally leaves a private, uniquely named forensic
    # temporary behind.  The controller owns no general unlink capability.
    os.replace(temporary, path)
    os.chmod(path, 0o600)
    _fsync_directory(path.parent)


def _atomic_private_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_private_bytes(path, canonical(dict(value)) + b"\n")


def _read_private_json(path: Path, *, label: str, sealed: bool = False) -> dict[str, Any]:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) \
            or metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600 \
            or metadata.st_nlink != 1:
        raise HandoffError(f"{label} is not a private regular file")
    if metadata.st_size > MAX_CAPTURE_BYTES:
        raise HandoffError(f"{label} exceeds the bounded JSON size")
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise HandoffError(f"{label} has a torn final write")
    value = _strict_json_bytes(raw[:-1], label=label)
    return verify_seal(value, label=label) if sealed else value


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class DurableStore:
    """Private hash-chained immutable journal plus rebuildable state snapshot."""

    def __init__(self, root: Path = STATE_ROOT) -> None:
        self.root = Path(root)
        self.journal = self.root / "journal"
        self.outputs = self.root / "outputs"
        self.logs = self.root / "logs"
        self.state_path = self.root / "state.json"
        self.authorization_path = self.root / "kimi-release-authorization.json"
        self.lock_path = self.root / ".controller.lease"

    def prepare(self) -> None:
        old_umask = os.umask(0o077)
        try:
            for path in (self.root, self.journal, self.outputs, self.logs):
                _ensure_private_directory(path)
        finally:
            os.umask(old_umask)

    @contextmanager
    def lease(self, *, blocking: bool = False) -> Iterator[None]:
        self.prepare()
        descriptor = os.open(
            self.lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
            operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(descriptor, operation)
            except BlockingIOError as exc:
                raise HandoffError("another detached handoff controller owns the lease") from exc
            opened = os.fstat(descriptor)
            named = self.lock_path.lstat()
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 \
                    or stat.S_ISLNK(named.st_mode) or not stat.S_ISREG(named.st_mode) \
                    or named.st_nlink != 1 \
                    or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
                raise HandoffError("controller lease descriptor/name identity is unsafe")
            owner = seal({
                "schema": "hawking.emergency_detached_campaign.lease_owner.v1",
                "pid": os.getpid(),
                "started_at": utc_now(),
            })
            raw = canonical(owner) + b"\n"
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, raw)
            os.fsync(descriptor)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _records(self) -> list[Path]:
        records = sorted(self.journal.glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].json"))
        unknown: list[str] = []
        for path in self.journal.iterdir():
            if path in records:
                continue
            if _JOURNAL_TEMP.fullmatch(path.name):
                metadata = path.lstat()
                if stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode) \
                        and metadata.st_uid == os.getuid() and metadata.st_nlink == 1 \
                        and stat.S_IMODE(metadata.st_mode) == 0o600 \
                        and metadata.st_size <= MAX_CAPTURE_BYTES:
                    # A crash before os.replace may leave exactly this private,
                    # controller-created name.  It is never journal authority.
                    continue
            unknown.append(path.name)
        unknown.sort()
        if unknown:
            raise HandoffError(f"journal contains unexpected entries: {unknown}")
        return records

    def history(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        previous = ZERO_HASH
        for index, path in enumerate(self._records()):
            if path.name != f"{index:08d}.json":
                raise HandoffError("journal sequence has a gap")
            value = _read_private_json(path, label=f"journal record {index}", sealed=True)
            if value.get("schema") != EVENT_SCHEMA or value.get("sequence") != index \
                    or value.get("previous_sha256") != previous \
                    or not isinstance(value.get("state"), dict):
                raise HandoffError("journal chain fields changed")
            state = verify_seal(value["state"], label=f"journal state {index}")
            if state.get("schema") != SCHEMA or state.get("generation") != index:
                raise HandoffError("journal state generation changed")
            result.append(value)
            previous = value["seal_sha256"]
        return result

    def load(self) -> dict[str, Any] | None:
        history = self.history()
        if not history:
            if self.state_path.exists():
                raise HandoffError("state snapshot exists without journal authority")
            return None
        state = history[-1]["state"]
        try:
            snapshot = _read_private_json(
                self.state_path, label="state snapshot", sealed=True
            ) if self.state_path.exists() else None
        except HandoffError:
            # The immutable journal is the authority.  A snapshot can be torn if
            # the process dies between journal fsync and atomic snapshot replace.
            snapshot = None
        if snapshot != state:
            _atomic_private_json(self.state_path, state)
        return dict(state)

    def commit(self, state: Mapping[str, Any], event: str, detail: Mapping[str, Any]) -> dict[str, Any]:
        history = self.history()
        sequence = len(history)
        previous = history[-1]["seal_sha256"] if history else ZERO_HASH
        next_state = seal({
            **{key: value for key, value in dict(state).items() if key != "seal_sha256"},
            "schema": SCHEMA,
            "generation": sequence,
            "updated_at": utc_now(),
        })
        record = seal({
            "schema": EVENT_SCHEMA,
            "sequence": sequence,
            "previous_sha256": previous,
            "recorded_at": utc_now(),
            "event": event,
            "detail": dict(detail),
            "state": next_state,
        })
        destination = self.journal / f"{sequence:08d}.json"
        if destination.exists():
            raise HandoffError("journal append destination unexpectedly exists")
        _atomic_private_json(destination, record)
        _atomic_private_json(self.state_path, next_state)
        self.log(event, detail)
        return next_state

    def log(self, event: str, detail: Mapping[str, Any]) -> None:
        path = self.logs / "controller.jsonl"
        line = canonical({"at": utc_now(), "event": event, "detail": dict(detail)}) + b"\n"
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
            view = memoryview(line)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def parse_vm_stat(text: str) -> tuple[int, int]:
    page = _VM_PAGE.search(text)
    if page is None:
        raise HandoffError("vm_stat omitted page size")
    values: dict[str, int] = {}
    for line in text.splitlines()[1:]:
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        number = raw.strip().rstrip(".")
        if number.isdigit():
            values[name.strip()] = int(number)
    required = ("Pages free", "Pages inactive", "Pages speculative", "Swapouts")
    missing = [name for name in required if name not in values]
    if missing:
        raise HandoffError(f"vm_stat omitted fields: {missing}")
    available = sum(values[name] for name in required[:3]) * int(page.group(1))
    return available, values["Swapouts"]


def parse_swap_used(text: str) -> int:
    match = _SWAP_USED.search(text)
    if match is None:
        raise HandoffError("vm.swapusage omitted used bytes")
    multipliers = {
        "B": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
        "P": 1024**5,
        "E": 1024**6,
    }
    try:
        value = Decimal(match.group(1)) * multipliers[match.group(2).upper()]
    except (InvalidOperation, KeyError) as exc:
        raise HandoffError("vm.swapusage used bytes are malformed") from exc
    if value < 0:
        raise HandoffError("vm.swapusage used bytes are negative")
    # macOS formats vm.swapusage to a bounded decimal precision.  Values such
    # as 325.76M therefore need not expand to an integral binary byte count.
    # Round upward so the memory guard never understates the reported usage.
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def parse_boot_epoch(text: str) -> int:
    match = _BOOT_SECONDS.search(text)
    if match is None:
        raise HandoffError("kern.boottime omitted boot epoch")
    value = int(match.group(1))
    if value <= 0:
        raise HandoffError("kern.boottime is not positive")
    return value


def _command_text(argv: Sequence[str]) -> str:
    result = subprocess.run(
        list(argv), check=False, capture_output=True, text=True, timeout=15,
        env={**os.environ, "LC_ALL": "C"},
    )
    if result.returncode != 0:
        raise HandoffError(f"resource command failed: {argv[0]} rc={result.returncode}")
    return result.stdout


def sample_resources(root: Path) -> dict[str, int]:
    available, swapouts = parse_vm_stat(_command_text(("/usr/bin/vm_stat",)))
    swap_used = parse_swap_used(_command_text(("/usr/sbin/sysctl", "-n", "vm.swapusage")))
    boot_epoch = parse_boot_epoch(
        _command_text(("/usr/sbin/sysctl", "-n", "kern.boottime"))
    )
    try:
        free_disk = shutil.disk_usage(root).free
    except OSError as exc:
        raise HandoffError(f"cannot sample filesystem free bytes: {exc}") from exc
    return {
        "available_ram_bytes": available,
        "swap_used_bytes": swap_used,
        "swapouts": swapouts,
        "free_disk_bytes": free_disk,
        "boot_epoch_seconds": boot_epoch,
    }


def resource_failures(
    sample: Mapping[str, int], *, baseline: Mapping[str, int], disk_floor: int
) -> list[str]:
    failures: list[str] = []
    if int(sample.get("boot_epoch_seconds", -1)) \
            != int(baseline.get("boot_epoch_seconds", -2)):
        failures.append("BOOT_ID_CHANGED")
    if int(sample["available_ram_bytes"]) < MINIMUM_AVAILABLE_RAM_BYTES:
        failures.append("AVAILABLE_RAM_BELOW_16_GIB")
    if int(sample["swap_used_bytes"]) > MAXIMUM_SWAP_USED_BYTES:
        failures.append("SWAP_USED_ABOVE_8_GIB")
    if int(sample["swap_used_bytes"]) - int(baseline["swap_used_bytes"]) \
            > MAXIMUM_SWAP_GROWTH_BYTES:
        failures.append("SWAP_USED_GREW")
    if int(sample["swapouts"]) > int(baseline["swapouts"]):
        failures.append("SWAPOUT_COUNTER_GREW")
    if int(sample["free_disk_bytes"]) < disk_floor:
        failures.append("FREE_DISK_BELOW_PHASE_FLOOR")
    return failures


_RESOURCE_KEYS = (
    "available_ram_bytes",
    "swap_used_bytes",
    "swapouts",
    "free_disk_bytes",
    "boot_epoch_seconds",
)


def validate_resource_sample(sample: Mapping[str, Any], *, label: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for key in _RESOURCE_KEYS:
        value = sample.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise HandoffError(f"{label} has invalid {key}")
        result[key] = value
    if result["boot_epoch_seconds"] <= 0:
        raise HandoffError(f"{label} has invalid boot identity")
    return result


def _ps_rows() -> list[tuple[int, str]]:
    output = _command_text(("/bin/ps", "-ww", "-axo", "pid=,command="))
    rows: list[tuple[int, str]] = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        rows.append((int(fields[0]), fields[1]))
    return rows


def recovery_generate_pids(rows: Sequence[tuple[int, str]], session: Path = SESSION) -> list[int]:
    result: list[int] = []
    session_text = os.fspath(session)
    for pid, command in rows:
        try:
            words = shlex.split(command)
        except ValueError:
            continue
        script_indexes = [
            index for index, word in enumerate(words)
            if Path(word).name == "kimi_k26_phase2_recovery.py"
        ]
        if not script_indexes:
            continue
        index = script_indexes[-1]
        tail = words[index + 1:]
        if "generate" not in tail or "--session" not in tail:
            continue
        try:
            candidate = tail[tail.index("--session") + 1]
        except IndexError:
            continue
        if candidate == session_text:
            result.append(pid)
    return sorted(set(result))


def _process_identity(pid: int) -> dict[str, Any] | None:
    """Return a stable-enough identity for one exact PID, never a name match."""
    if not isinstance(pid, int) or pid <= 0:
        return None
    result = subprocess.run(
        (
            "/bin/ps", "-ww", "-p", str(pid), "-o", "pid=", "-o", "pgid=",
            "-o", "lstart=", "-o", "command=",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "LC_ALL": "C"},
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise HandoffError("ps returned ambiguous exact-PID identity")
    fields = lines[0].strip().split(maxsplit=7)
    if len(fields) != 8 or not fields[0].isdigit() or not fields[1].isdigit():
        raise HandoffError("ps returned malformed exact-PID identity")
    observed_pid = int(fields[0])
    if observed_pid != pid:
        raise HandoffError("ps exact-PID identity changed")
    command = fields[7]
    return {
        "pid": observed_pid,
        "pgid": int(fields[1]),
        "process_started": " ".join(fields[2:7]),
        "process_command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
    }


def _same_process(child: Mapping[str, Any], observed: Mapping[str, Any] | None) -> bool:
    if observed is None:
        return False
    keys = ("pid", "pgid", "process_started", "process_command_sha256")
    return all(child.get(key) == observed.get(key) for key in keys)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("/usr/bin/git", "-C", os.fspath(root), *arguments),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "LC_ALL": "C", "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode != 0:
        raise HandoffError(f"git {arguments[0]} failed for {root}")
    return result.stdout.strip()


def _sha256_regular_file(root: Path, relative: str) -> str:
    path = root / relative
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) \
            or metadata.st_nlink != 1 or path.resolve(strict=True) != path:
        raise HandoffError(f"operation file is not an anchored regular file: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_release_executor(
    root: Path = EXECUTOR_ROOT, *, require_remote: bool = True
) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    if resolved != root:
        raise HandoffError("release executor path changed through a symlink")
    git_directory = root / ".git"
    git_metadata = git_directory.lstat()
    if stat.S_ISLNK(git_metadata.st_mode) or not stat.S_ISDIR(git_metadata.st_mode):
        raise HandoffError("emergency executor is not a standalone clone")
    if not EXPECTED_OPERATION_FILE_SHA256 or any(
        not _HEX64.fullmatch(value)
        for value in EXPECTED_OPERATION_FILE_SHA256.values()
    ):
        raise HandoffError("operation-file hash pins have not been finalized")
    head = _git(root, "rev-parse", "HEAD")
    upstream = _git(root, "rev-parse", "@{u}")
    upstream_name = _git(
        root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"
    )
    origin_url = _git(root, "remote", "get-url", "origin")
    branch = _git(root, "branch", "--show-current")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    required = tuple(EXPECTED_OPERATION_FILE_SHA256)
    tracked = set(_git(root, "ls-files", "--error-unmatch", *required).splitlines())
    remote_head = upstream
    if require_remote:
        remote = _git(
            root, "ls-remote", "--exit-code", "origin", OFFICIAL_CAMPAIGN_REMOTE_REF
        )
        remote_lines = [line.split() for line in remote.splitlines() if line.strip()]
        if len(remote_lines) != 1 or len(remote_lines[0]) != 2 \
                or remote_lines[0][1] != OFFICIAL_CAMPAIGN_REMOTE_REF \
                or not _HEX40.fullmatch(remote_lines[0][0]):
            raise HandoffError("official remote returned an ambiguous campaign HEAD")
        remote_head = remote_lines[0][0]
    if branch != EXPECTED_EXECUTOR_BRANCH \
            or upstream_name != OFFICIAL_CAMPAIGN_REF \
            or origin_url != OFFICIAL_ORIGIN_URL \
            or head != upstream or head != remote_head \
            or status or tracked != set(required):
        raise HandoffError("emergency executor is not clean at exact official remote HEAD")
    observed_hashes = {
        relative: _sha256_regular_file(root, relative) for relative in required
    }
    if observed_hashes != EXPECTED_OPERATION_FILE_SHA256:
        raise HandoffError("emergency executor operation-file hash pin mismatch")
    return {
        "root": os.fspath(root),
        "head": head,
        "upstream": upstream,
        "remote_head": remote_head,
        "remote_verified_live": require_remote,
        "official_ref": OFFICIAL_CAMPAIGN_REF,
        "origin_url": origin_url,
        "branch": branch,
        "operation_file_sha256": observed_hashes,
    }


def _initial_state(
    args: argparse.Namespace,
    *,
    executor_head: str,
    resource_baseline: Mapping[str, int],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "generation": 0,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "phase": PHASE_WAIT_RECOVERY,
        "session": os.fspath(args.session.resolve(strict=True)),
        "executor_root": os.fspath(args.executor_root.resolve(strict=True)),
        "executor_head": executor_head,
        "repo_root": os.fspath(args.repo_root.resolve(strict=True)),
        "kimi_release_authorization": dict(authorization),
        "resource_baseline": dict(resource_baseline),
        "glm_config": os.fspath(args.glm_config.resolve(strict=False)),
        "glm_authority": os.fspath(args.glm_authority.resolve(strict=False)),
        "glm_scratch_root": os.fspath(args.glm_scratch_root.resolve(strict=False)),
        "child": None,
        "attempts": {},
        "not_before_epoch": 0,
        "last_resource_sample": dict(resource_baseline),
        "block": None,
        "artifacts": {},
    }


def _validate_state(value: Mapping[str, Any]) -> dict[str, Any]:
    state = verify_seal(value, label="controller state")
    if state.get("schema") != SCHEMA \
            or state.get("session") != os.fspath(SESSION) \
            or state.get("executor_root") != os.fspath(EXECUTOR_ROOT) \
            or state.get("repo_root") != os.fspath(REPO_ROOT):
        raise HandoffError("controller state identity changed")
    executor_head = state.get("executor_head")
    if not isinstance(executor_head, str) or not _HEX40.fullmatch(executor_head):
        raise HandoffError("controller state has no valid bootstrap executor HEAD")
    authorization = state.get("kimi_release_authorization")
    if not isinstance(authorization, dict):
        raise HandoffError("controller state omitted explicit Kimi release authorization")
    _validate_authorization(authorization, executor_head=executor_head)
    baseline = state.get("resource_baseline")
    if not isinstance(baseline, dict):
        raise HandoffError("controller state omitted campaign resource baseline")
    validate_resource_sample(baseline, label="campaign resource baseline")
    return state


def _validate_authorization(
    value: Mapping[str, Any], *, executor_head: str
) -> dict[str, Any]:
    if not _HEX40.fullmatch(executor_head):
        raise HandoffError("authorization executor HEAD is malformed")
    authorization = verify_seal(value, label="durable Kimi release authorization")
    expected = {
        "schema": AUTHORIZATION_SCHEMA,
        "explicit_cli_flag": True,
        "session": os.fspath(SESSION),
        "executor_root": os.fspath(EXECUTOR_ROOT),
        "executor_head": executor_head,
        "scope": "EXACT_KIMI_PHASE2_SOURCE_RELEASE_ONLY",
        "authorized_by_uid": os.getuid(),
    }
    for key, item in expected.items():
        if authorization.get(key) != item:
            raise HandoffError(f"durable Kimi release authorization changed {key}")
    if not isinstance(authorization.get("authorized_at"), str):
        raise HandoffError("durable Kimi release authorization omitted timestamp")
    return authorization


def _durable_authorization(
    store: DurableStore, *, executor_head: str
) -> dict[str, Any]:
    if store.authorization_path.exists():
        return _validate_authorization(
            _read_private_json(
                store.authorization_path,
                label="durable Kimi release authorization",
                sealed=True,
            ),
            executor_head=executor_head,
        )
    authorization = seal({
        "schema": AUTHORIZATION_SCHEMA,
        "explicit_cli_flag": True,
        "session": os.fspath(SESSION),
        "executor_root": os.fspath(EXECUTOR_ROOT),
        "executor_head": executor_head,
        "scope": "EXACT_KIMI_PHASE2_SOURCE_RELEASE_ONLY",
        "authorized_by_uid": os.getuid(),
        "authorized_at": utc_now(),
    })
    _atomic_private_json(store.authorization_path, authorization)
    return _validate_authorization(authorization, executor_head=executor_head)


def _artifact_path(store: DurableStore, name: str) -> Path:
    if Path(name).name != name:
        raise HandoffError("artifact name is not a leaf")
    return store.outputs / name


def _persist_document(store: DurableStore, name: str, document: Mapping[str, Any]) -> Path:
    path = _artifact_path(store, name)
    raw = canonical(dict(document)) + b"\n"
    if path.exists():
        existing = _read_private_json(path, label=name)
        if canonical(existing) != canonical(dict(document)):
            raise HandoffError(f"refusing to replace differing durable artifact {name}")
        return path
    _atomic_private_bytes(path, raw)
    return path


def _read_child_document(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) \
            or metadata.st_uid != os.getuid() or metadata.st_size > MAX_CAPTURE_BYTES:
        raise HandoffError("child output is not a bounded private regular file")
    raw = path.read_bytes().strip()
    return _strict_json_bytes(raw, label=f"child output {path.name}")


def _safe_stderr(path: Path) -> str:
    try:
        raw = path.read_bytes()[-4000:]
    except OSError:
        return "stderr unavailable"
    return raw.decode("utf-8", errors="replace").replace("\x00", "")[-2000:]


class Controller:
    def __init__(
        self,
        store: DurableStore,
        *,
        resource_sampler: Callable[[Path], dict[str, int]] = sample_resources,
        process_rows: Callable[[], list[tuple[int, str]]] = _ps_rows,
        process_identity: Callable[[int], dict[str, Any] | None] = _process_identity,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        sleeper: Callable[[float], None] = time.sleep,
        killpg: Callable[[int, int], None] = os.killpg,
    ) -> None:
        self.store = store
        self.resource_sampler = resource_sampler
        self.process_rows = process_rows
        self.process_identity = process_identity
        self.popen = popen
        self.sleeper = sleeper
        self.killpg = killpg
        self.state: dict[str, Any] = {}

    def commit(self, event: str, detail: Mapping[str, Any], **changes: Any) -> None:
        body = {key: value for key, value in self.state.items() if key != "seal_sha256"}
        body.update(changes)
        self.state = self.store.commit(body, event, detail)

    @property
    def session(self) -> Path:
        return Path(self.state["session"])

    @property
    def executor(self) -> Path:
        return Path(self.state["executor_root"])

    @property
    def repo(self) -> Path:
        return Path(self.state["repo_root"])

    def _disk_floor(self, action: str) -> int:
        if action in {ACTION_GLM_PLAN_VERIFY, ACTION_GLM_STATUS, ACTION_GLM_LIVE}:
            return GLM_REQUIRED_DISK_FLOOR_BYTES
        if action in {ACTION_RELEASE_EXECUTE, ACTION_RELEASE_VERIFY}:
            return KIMI_EMERGENCY_DISK_FLOOR_BYTES
        return KIMI_OPERATIONAL_DISK_FLOOR_BYTES

    def _command(self, action: str) -> tuple[list[str], Path, dict[str, str]]:
        python = os.fspath(PYTHON)
        glm_python = os.fspath(GLM_PYTHON)
        recovery = self.executor / "tools/condense/kimi_k26_phase2_recovery.py"
        release = self.executor / "tools/condense/kimi_k26_phase2_release.py"
        bundle = _artifact_path(self.store, RELEASE_BUNDLE_NAME)
        confirmation = _artifact_path(self.store, RELEASE_CONFIRMATION_NAME)
        receipt = _artifact_path(self.store, RELEASE_RECEIPT_NAME)
        environment = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "LC_ALL": "C",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        }
        if action == ACTION_RECOVERY_GENERATE:
            return [python, os.fspath(recovery), "generate", "--session", os.fspath(self.session)], self.executor, environment
        if action == ACTION_RECOVERY_VERIFY:
            return [python, os.fspath(recovery), "verify", "--session", os.fspath(self.session)], self.executor, environment
        if action == ACTION_RELEASE_AUDIT:
            return [python, os.fspath(release), "audit", "--session", os.fspath(self.session)], self.executor, environment
        if action == ACTION_RELEASE_CONFIRM:
            return [python, os.fspath(release), "confirm-token", "--bundle", os.fspath(bundle)], self.executor, environment
        if action == ACTION_RELEASE_EXECUTE:
            token = _read_private_json(confirmation, label="release confirmation", sealed=True).get("confirmation_token")
            if not isinstance(token, str) or not token.startswith("CONFIRM-KIMI-K26-PHASE2-"):
                raise HandoffError("release confirmation token is absent or malformed")
            return [
                python, os.fspath(release), "execute", "--session", os.fspath(self.session),
                "--bundle", os.fspath(bundle), "--confirm", token,
            ], self.executor, environment
        if action == ACTION_RELEASE_VERIFY:
            return [
                python, os.fspath(release), "verify-receipt", "--bundle", os.fspath(bundle),
                "--receipt", os.fspath(receipt),
            ], self.executor, environment
        if action == ACTION_GLM_PLAN_VERIFY:
            return [
                glm_python, os.fspath(self.repo / "tools/condense/glm52_xet_autotune.py"),
                "verify", "--offline", "--root", os.fspath(self.repo),
                "--plan", os.fspath(self.repo / "GLM52_XET_AUTOTUNE_PLAN.json"),
            ], self.repo, environment
        if action == ACTION_GLM_STATUS:
            return [
                glm_python, os.fspath(self.repo / "tools/glm52_gravity.py"),
                "--config", self.state["glm_config"], "status",
            ], self.repo, environment
        if action == ACTION_GLM_LIVE:
            environment["HAWKING_GLM52_XET_EXECUTE"] = "1"
            return [
                glm_python, os.fspath(self.repo / "tools/condense/glm52_xet_live_driver.py"),
                "run", "--config", self.state["glm_config"],
                "--plan", os.fspath(self.repo / "GLM52_XET_AUTOTUNE_PLAN.json"),
                "--authority", self.state["glm_authority"],
                "--scratch-root", self.state["glm_scratch_root"], "--execute-live",
            ], self.repo, environment
        raise HandoffError(f"unsupported child action: {action}")

    def _resource_backoff(self, action: str, sample: Mapping[str, int], failures: Sequence[str]) -> None:
        self.commit(
            "RESOURCE_BACKOFF",
            {"action": action, "failures": list(failures), "sample": dict(sample)},
            child=None,
            not_before_epoch=int(time.time()) + RESOURCE_BACKOFF_SECONDS,
            last_resource_sample=dict(sample),
        )

    def _campaign_baseline(self) -> dict[str, int]:
        baseline = self.state.get("resource_baseline")
        if not isinstance(baseline, dict):
            raise HandoffError("campaign resource baseline is absent")
        return validate_resource_sample(baseline, label="campaign resource baseline")

    def _sample(self) -> dict[str, int]:
        return validate_resource_sample(
            self.resource_sampler(self.session), label="live resource sample"
        )

    def _establish_new_boot_baseline(self, sample: Mapping[str, int], *, reason: str) -> None:
        validated = validate_resource_sample(sample, label="new-boot resource baseline")
        old = self._campaign_baseline()
        if validated["boot_epoch_seconds"] == old["boot_epoch_seconds"]:
            raise HandoffError("refusing to reset resource baseline within one boot")
        self.commit(
            "RESOURCE_BASELINE_ESTABLISHED_AFTER_BOOT",
            {
                "reason": reason,
                "old_boot_epoch_seconds": old["boot_epoch_seconds"],
                "new_boot_epoch_seconds": validated["boot_epoch_seconds"],
                "baseline": validated,
            },
            resource_baseline=validated,
            last_resource_sample=validated,
        )

    def _terminate_exact_child(self, child: Mapping[str, Any], *, reason: str) -> bool:
        """Signal only the persisted PID/PGID identity and wait for it to leave."""
        pid = child.get("pid")
        pgid = child.get("pgid")
        if not isinstance(pid, int) or pid <= 0 or not isinstance(pgid, int) or pgid <= 0:
            return True
        try:
            observed = self.process_identity(pid)
        except Exception as exc:  # Keep authority; a later tick can retry safely.
            self.store.log(
                "CHILD_IDENTITY_SAMPLE_FAILED",
                {"action": child.get("action"), "pid": pid, "error": str(exc)},
            )
            return False
        if observed is None or not _same_process(child, observed):
            return True
        try:
            self.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        except OSError as exc:
            self.store.log(
                "CHILD_GROUP_SIGNAL_FAILED",
                {"action": child.get("action"), "pid": pid, "reason": reason, "error": str(exc)},
            )
            return False
        deadline = time.monotonic() + CHILD_TERM_GRACE_SECONDS
        while time.monotonic() < deadline:
            self.sleeper(0.25)
            try:
                current = self.process_identity(pid)
            except Exception:
                continue
            if not _same_process(child, current):
                return True
        try:
            current = self.process_identity(pid)
        except Exception:
            return False
        if not _same_process(child, current):
            return True
        try:
            self.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        for _ in range(40):
            self.sleeper(0.25)
            try:
                if not _same_process(child, self.process_identity(pid)):
                    return True
            except Exception:
                continue
        return False

    def _clear_dead_child(
        self, child: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any] | None]:
        action = str(child["action"])
        try:
            sample = self._sample()
        except Exception as exc:
            # Retain the child record and retry: an unavailable sampler must not
            # cause a detached child/output to escape the watchdog.
            self.store.log(
                "DEAD_CHILD_FINAL_RESOURCE_SAMPLE_FAILED",
                {"action": action, "pid": child.get("pid"), "error": str(exc)},
            )
            return "WAIT", None
        baseline = self._campaign_baseline()
        if sample["boot_epoch_seconds"] != baseline["boot_epoch_seconds"]:
            self._establish_new_boot_baseline(sample, reason="DEAD_CHILD_FROM_PRIOR_BOOT")
            self.commit(
                "CHILD_LOST_ACROSS_BOOT",
                {"action": action, "pid": child.get("pid")},
                child=None,
            )
            return "RETRY", None
        failures = resource_failures(sample, baseline=baseline, disk_floor=self._disk_floor(action))
        if failures:
            self._resource_backoff(action, sample, failures)
            return "RETRY", None
        output = Path(child["stdout_path"])
        if output.exists() and output.stat().st_size:
            try:
                document = _read_child_document(output)
            except HandoffError:
                document = None
            if document is not None:
                self.commit(
                    "ORPHAN_CHILD_OUTPUT_RECONCILED",
                    {"action": child["action"], "pid": child.get("pid")},
                    child=None,
                    last_resource_sample=sample,
                )
                return "DONE", document
        attempts = dict(self.state.get("attempts", {}))
        attempts[action] = int(attempts.get(action, 0)) + 1
        self.commit(
            "DEAD_CHILD_RETRY",
            {"action": action, "pid": child.get("pid"), "attempt": attempts[action]},
            child=None,
            attempts=attempts,
            not_before_epoch=int(time.time()) + RESOURCE_BACKOFF_SECONDS,
        )
        if attempts[action] >= MAX_ACTION_ATTEMPTS:
            self._block("CHILD_DIED_WITHOUT_RECONCILABLE_OUTPUT", action=action)
        return "RETRY", None

    def _recover_spawn_gap_child(
        self, child: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Recover the one crash window between Popen and PID journal commit."""
        action = str(child.get("action"))
        try:
            argv, _cwd, _environment = self._command(action)
        except (HandoffError, OSError):
            return None
        expected_command_sha256 = hashlib.sha256(
            b"\0".join(item.encode() for item in argv)
        ).hexdigest()
        if child.get("command_sha256") != expected_command_sha256 or len(argv) < 2:
            raise HandoffError("spawn-gap child command authority changed")
        expected_tail = argv[1:]
        candidates: list[int] = []
        for pid, command in self.process_rows():
            try:
                words = shlex.split(command)
            except ValueError:
                continue
            if len(words) >= len(expected_tail) and words[-len(expected_tail):] == expected_tail:
                candidates.append(pid)
        candidates = sorted(set(candidates))
        if not candidates:
            return None
        if len(candidates) != 1:
            self.store.log(
                "SPAWN_GAP_CHILD_IDENTITY_AMBIGUOUS",
                {"action": action, "candidate_pids": candidates},
            )
            raise HandoffError("spawn-gap child identity is ambiguous")
        observed = self.process_identity(candidates[0])
        if observed is None or observed.get("pgid") != observed.get("pid"):
            raise HandoffError("spawn-gap child lacks its exact private process group")
        recovered = {**dict(child), **observed}
        self.commit(
            "CHILD_IDENTITY_RECOVERED_AFTER_CONTROLLER_CRASH",
            {"action": action, "pid": observed["pid"], "pgid": observed["pgid"]},
            child=recovered,
        )
        return recovered

    def _reconcile_child(self) -> tuple[str, dict[str, Any] | None]:
        child = self.state.get("child")
        if not isinstance(child, dict):
            return "NONE", None
        action = str(child.get("action"))
        pid = child.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            try:
                recovered = self._recover_spawn_gap_child(child)
            except Exception as exc:
                self.store.log(
                    "SPAWN_GAP_CHILD_RECOVERY_FAILED",
                    {"action": action, "error": str(exc)},
                )
                return "WAIT", None
            if recovered is None:
                return self._clear_dead_child(child)
            child = recovered
            pid = child["pid"]
        try:
            observed = self.process_identity(pid)
        except Exception as exc:
            self.store.log(
                "ACTIVE_CHILD_IDENTITY_SAMPLE_FAILED",
                {"action": action, "pid": pid, "error": str(exc)},
            )
            return "WAIT", None
        if _same_process(child, observed):
            try:
                sample = self._sample()
                failures = resource_failures(
                    sample,
                    baseline=self._campaign_baseline(),
                    disk_floor=self._disk_floor(action),
                )
            except Exception as exc:
                stopped = self._terminate_exact_child(child, reason="RESOURCE_SAMPLER_FAILED")
                self.store.log(
                    "ACTIVE_CHILD_RESOURCE_SAMPLE_FAILED",
                    {"action": action, "pid": pid, "stopped": stopped, "error": str(exc)},
                )
                return "WAIT", None
            if failures:
                stopped = self._terminate_exact_child(child, reason="RESOURCE_POLICY_VIOLATION")
                if stopped:
                    self._resource_backoff(action, sample, failures)
                else:
                    self.store.log(
                        "ACTIVE_CHILD_RESOURCE_STOP_PENDING",
                        {"action": action, "pid": pid, "failures": failures},
                    )
                return "WAIT", None
            self.store.log(
                "WAITING_FOR_EXACT_EXISTING_CHILD",
                {"action": action, "pid": pid, "pgid": child.get("pgid"), "sample": sample},
            )
            return "WAIT", None
        if observed is not None:
            self.store.log(
                "STORED_CHILD_PID_REUSED_NOT_SIGNALLED",
                {"action": action, "pid": pid, "observed": observed},
            )
        return self._clear_dead_child(child)

    def run_child(self, action: str) -> tuple[str, dict[str, Any] | None]:
        reconciled, document = self._reconcile_child()
        if reconciled != "NONE":
            return reconciled, document
        if int(self.state.get("not_before_epoch", 0)) > int(time.time()):
            return "WAIT", None
        baseline = self._campaign_baseline()
        sample = self._sample()
        if sample["boot_epoch_seconds"] != baseline["boot_epoch_seconds"]:
            self._establish_new_boot_baseline(sample, reason="PRE_ACTION_BOOT_CHANGE")
            return "WAIT", None
        failures = resource_failures(
            sample, baseline=baseline, disk_floor=self._disk_floor(action)
        )
        if failures:
            self._resource_backoff(action, sample, failures)
            return "WAIT", None
        argv, cwd, environment = self._command(action)
        action_id = f"{self.state['generation'] + 1:08d}-{action.lower()}"
        stdout_path = self.store.outputs / f"{action_id}.stdout.json"
        stderr_path = self.store.logs / f"{action_id}.stderr.log"
        command_sha256 = hashlib.sha256(b"\0".join(item.encode() for item in argv)).hexdigest()
        child = {
            "action": action,
            "action_id": action_id,
            "command_sha256": command_sha256,
            "pid": None,
            "stdout_path": os.fspath(stdout_path),
            "stderr_path": os.fspath(stderr_path),
            "start_resource_sample": sample,
            "started_at": utc_now(),
        }
        self.commit("CHILD_INTENT", {"action": action, "action_id": action_id}, child=child)
        stdout_fd = os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        stderr_fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            process = self.popen(
                argv,
                cwd=os.fspath(cwd),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_fd,
                stderr=stderr_fd,
                start_new_session=True,
                close_fds=True,
            )
        except BaseException:
            os.close(stdout_fd)
            os.close(stderr_fd)
            raise
        os.close(stdout_fd)
        os.close(stderr_fd)
        observed: dict[str, Any] | None = None
        for _ in range(20):
            if process.poll() is not None:
                break
            try:
                observed = self.process_identity(int(process.pid))
            except Exception:
                observed = None
            if observed is not None:
                break
            self.sleeper(0.01)
        if process.poll() is None and (
            observed is None or observed.get("pid") != int(process.pid)
            or observed.get("pgid") != int(process.pid)
        ):
            process.terminate()
            try:
                process.wait(timeout=CHILD_TERM_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            self.commit(
                "CHILD_IDENTITY_CAPTURE_FAILED",
                {"action": action, "pid": process.pid, "observed": observed},
                child=None,
                not_before_epoch=int(time.time()) + RESOURCE_BACKOFF_SECONDS,
            )
            return "WAIT", None
        child = {
            **child,
            "pid": int(process.pid),
            **({} if observed is None else observed),
        }
        self.commit(
            "CHILD_STARTED",
            {"action": action, "pid": process.pid, "pgid": child.get("pgid")},
            child=child,
        )
        resource_violation: tuple[dict[str, int], list[str]] | None = None
        while process.poll() is None:
            self.sleeper(CHILD_POLL_SECONDS)
            try:
                sample = self._sample()
            except Exception as exc:
                stopped = self._terminate_exact_child(child, reason="RESOURCE_SAMPLER_FAILED")
                self.store.log(
                    "CHILD_RESOURCE_SAMPLE_FAILED",
                    {"action": action, "pid": process.pid, "stopped": stopped, "error": str(exc)},
                )
                if stopped:
                    process.wait()
                    return "WAIT", None
                continue
            failures = resource_failures(
                sample, baseline=baseline, disk_floor=self._disk_floor(action)
            )
            if failures:
                resource_violation = (sample, failures)
                if self._terminate_exact_child(child, reason="RESOURCE_POLICY_VIOLATION"):
                    process.wait()
                    break
                self.store.log(
                    "CHILD_RESOURCE_STOP_PENDING",
                    {"action": action, "pid": process.pid, "failures": failures},
                )
                resource_violation = None
        returncode = int(process.wait())
        if resource_violation is not None:
            sample, failures = resource_violation
            self._resource_backoff(action, sample, failures)
            return "WAIT", None
        try:
            document = _read_child_document(stdout_path)
        except HandoffError:
            document = None
        try:
            final_sample = self._sample()
        except Exception as exc:
            self.store.log(
                "CHILD_FINAL_RESOURCE_SAMPLE_FAILED",
                {"action": action, "pid": process.pid, "error": str(exc)},
            )
            return "WAIT", None
        final_failures = resource_failures(
            final_sample, baseline=baseline, disk_floor=self._disk_floor(action)
        )
        if final_failures:
            self._resource_backoff(action, final_sample, final_failures)
            return "WAIT", None
        self.commit(
            "CHILD_EXITED",
            {"action": action, "pid": process.pid, "returncode": returncode},
            child=None,
            last_resource_sample=final_sample,
        )
        if document is not None:
            return ("DONE" if returncode == 0 else "FAILED"), document
        if returncode != 0:
            self._block(
                "CHILD_COMMAND_FAILED",
                authority=action in {ACTION_GLM_STATUS, ACTION_GLM_LIVE},
                action=action,
                returncode=returncode,
                stderr=_safe_stderr(stderr_path),
            )
            return "BLOCKED", None
        self._block("CHILD_RETURNED_NO_JSON", action=action)
        return "BLOCKED", None

    def _block(self, reason: str, *, authority: bool = False, **detail: Any) -> None:
        phase = PHASE_BLOCKED_AUTHORITY if authority else PHASE_BLOCKED
        block = {"reason": reason, **detail, "blocked_at": utc_now()}
        self.commit("BLOCKED", block, phase=phase, block=block, child=None)

    def _expect(
        self,
        action: str,
        document: Mapping[str, Any],
        *,
        status: str | Sequence[str],
        sealed: bool,
    ) -> dict[str, Any]:
        value = verify_seal(document, label=action) if sealed else dict(document)
        accepted = (status,) if isinstance(status, str) else tuple(status)
        if not accepted or any(not isinstance(item, str) for item in accepted):
            raise HandoffError("accepted child statuses are malformed")
        if value.get("status") not in accepted:
            self._block(
                "CHILD_STATUS_REFUSED",
                authority=action in {ACTION_GLM_STATUS, ACTION_GLM_LIVE},
                action=action,
                expected_statuses=list(accepted),
                observed_status=value.get("status"),
            )
            raise HandoffError(f"{action} did not return an accepted status")
        return value

    def _run_expected(
        self, action: str, *, status: str | Sequence[str], sealed: bool = True
    ) -> dict[str, Any] | None:
        outcome, document = self.run_child(action)
        if outcome in {"WAIT", "RETRY", "BLOCKED"}:
            return None
        if document is None:
            return None
        if outcome == "FAILED":
            observed = document.get("status")
            if action == ACTION_RELEASE_EXECUTE \
                    and observed == "PARTIAL_FAILURE_EXACT_PHASE2_SOURCE_RELEASE":
                _persist_document(self.store, RELEASE_RECEIPT_NAME, document)
                self._block("KIMI_RELEASE_PARTIAL_FAILURE_RECEIPT_SEALED")
                return None
            self._block(
                "CHILD_RETURNED_FAILURE",
                authority=action in {ACTION_GLM_STATUS, ACTION_GLM_LIVE},
                action=action,
                observed_status=observed,
            )
            return None
        return self._expect(action, document, status=status, sealed=sealed)

    def _verify_executor_or_block(self) -> bool:
        try:
            # Bootstrap already proved exact pushed-remote identity.  Kimi
            # recovery/release is thereafter bound to that immutable HEAD and
            # exact file hashes, so a transient WAN outage must not strand it.
            receipt = verify_release_executor(self.executor, require_remote=False)
        except (HandoffError, OSError, subprocess.SubprocessError) as exc:
            self._block("RELEASE_EXECUTOR_PROVENANCE_FAILED", error=str(exc))
            return False
        if receipt.get("head") != self.state.get("executor_head"):
            self._block(
                "RELEASE_EXECUTOR_HEAD_CHANGED_AFTER_BOOTSTRAP",
                expected=self.state.get("executor_head"),
                observed=receipt.get("head"),
            )
            return False
        if self.state.get("executor_provenance") != receipt:
            self.commit("EXECUTOR_PROVENANCE_VERIFIED", receipt, executor_provenance=receipt)
        return True

    def _glm_provenance(self) -> tuple[bool, str]:
        required = (
            "tools/condense/glm52_xet_autotune.py",
            "tools/condense/glm52_xet_live_driver.py",
            "tools/glm52_gravity.py",
            "GLM52_XET_AUTOTUNE_PLAN.json",
        )
        try:
            if self.repo.resolve(strict=True) != self.repo:
                return False, "GLM_REPOSITORY_PATH_IS_NOT_ANCHORED"
            if not GLM_PYTHON.exists() or not os.access(GLM_PYTHON, os.X_OK):
                return False, "GLM_PINNED_VENV_PYTHON_MISSING"
            provenance = verify_release_executor(self.repo)
            head = str(provenance["head"])
            tracked = set(_git(self.repo, "ls-files", "--error-unmatch", *required).splitlines())
        except (HandoffError, OSError, subprocess.SubprocessError):
            return False, "GLM_PROVENANCE_COMMAND_FAILED"
        if head != self.state.get("executor_head"):
            return False, "GLM_HEAD_CHANGED_AFTER_CONTROLLER_BOOTSTRAP"
        if tracked != set(required):
            return False, "GLM_SUPPORTED_LIVE_DRIVER_NOT_COMMITTED"
        return True, head

    def advance_once(self) -> bool:
        phase = self.state["phase"]
        if phase in {PHASE_BLOCKED, PHASE_BLOCKED_AUTHORITY}:
            return False
        if phase == PHASE_WAIT_RECOVERY:
            receipt = self.session / "evidence" / RECOVERY_RECEIPT_NAME
            live = recovery_generate_pids(self.process_rows(), self.session)
            if live:
                try:
                    sample = self._sample()
                except Exception as exc:
                    self.store.log(
                        "EXISTING_KIMI_RECOVERY_RESOURCE_SAMPLE_FAILED",
                        {"pids": live, "error": str(exc)},
                    )
                    return False
                baseline = self._campaign_baseline()
                if sample["boot_epoch_seconds"] != baseline["boot_epoch_seconds"]:
                    self._establish_new_boot_baseline(
                        sample, reason="EXISTING_RECOVERY_ON_NEW_BOOT"
                    )
                    baseline = self._campaign_baseline()
                failures = resource_failures(
                    sample,
                    baseline=baseline,
                    disk_floor=KIMI_OPERATIONAL_DISK_FLOOR_BYTES,
                )
                self.store.log(
                    "WAITING_FOR_EXISTING_KIMI_RECOVERY",
                    {
                        "pids": live,
                        "sample": sample,
                        "failures": failures,
                        "receipt_exists": receipt.exists(),
                    },
                )
                return False
            if not self._verify_executor_or_block():
                return False
            if not receipt.exists():
                value = self._run_expected(
                    ACTION_RECOVERY_GENERATE,
                    status=tuple(RECOVERY_CAPSULE_STATUS_TO_MODE),
                )
                if value is None:
                    return False
                recovery_mode = RECOVERY_CAPSULE_STATUS_TO_MODE[value["status"]]
            else:
                recovery_mode = self.state.get("recovery_capsule_mode")
            self.commit(
                "KIMI_RECOVERY_READY",
                {"receipt": os.fspath(receipt), "recovery_capsule_mode": recovery_mode},
                phase=PHASE_VERIFY_RECOVERY,
                recovery_capsule_mode=recovery_mode,
            )
            return True
        if phase == PHASE_VERIFY_RECOVERY:
            value = self._run_expected(
                ACTION_RECOVERY_VERIFY,
                status=tuple(RECOVERY_CAPSULE_STATUS_TO_MODE),
            )
            if value is None:
                return False
            _persist_document(self.store, "KIMI_K26_PHASE2_RECOVERY_VERIFY.json", value)
            self.commit(
                "KIMI_RECOVERY_VERIFIED",
                {
                    "seal_sha256": value["seal_sha256"],
                    "recovery_capsule_mode": RECOVERY_CAPSULE_STATUS_TO_MODE[value["status"]],
                },
                phase=PHASE_RELEASE_AUDIT,
                recovery_capsule_mode=RECOVERY_CAPSULE_STATUS_TO_MODE[value["status"]],
            )
            return True
        if phase == PHASE_RELEASE_AUDIT:
            if not self._verify_executor_or_block():
                return False
            value = self._run_expected(
                ACTION_RELEASE_AUDIT, status="PASS_CONFIRMATION_REQUIRED"
            )
            if value is None:
                return False
            path = _persist_document(self.store, RELEASE_BUNDLE_NAME, value)
            self.commit(
                "KIMI_RELEASE_AUDIT_PASSED",
                {"bundle": os.fspath(path), "seal_sha256": value["seal_sha256"]},
                phase=PHASE_RELEASE_CONFIRM,
            )
            return True
        if phase == PHASE_RELEASE_CONFIRM:
            value = self._run_expected(
                ACTION_RELEASE_CONFIRM,
                status="EXPLICIT_CONFIRMATION_TOKEN_DERIVED",
            )
            if value is None:
                return False
            path = _persist_document(self.store, RELEASE_CONFIRMATION_NAME, value)
            self.commit(
                "KIMI_RELEASE_CONFIRMATION_DERIVED",
                {"confirmation": os.fspath(path), "seal_sha256": value["seal_sha256"]},
                phase=PHASE_RELEASE_EXECUTE,
            )
            return True
        if phase == PHASE_RELEASE_EXECUTE:
            if not self._verify_executor_or_block():
                return False
            value = self._run_expected(
                ACTION_RELEASE_EXECUTE, status="PASS_EXACT_PHASE2_SOURCE_RELEASE"
            )
            if value is None:
                return False
            if value.get("terminal_status") != "SUCCESS" \
                    or value.get("mop_touched") is not False \
                    or value.get("shared_xet_touched") is not False \
                    or value.get("capsule_retained") is not True:
                self._block("KIMI_RELEASE_RECEIPT_SAFETY_FIELDS_FAILED")
                return False
            path = _persist_document(self.store, RELEASE_RECEIPT_NAME, value)
            self.commit(
                "KIMI_RELEASE_EXECUTED",
                {"receipt": os.fspath(path), "seal_sha256": value["seal_sha256"]},
                phase=PHASE_RELEASE_VERIFY,
            )
            return True
        if phase == PHASE_RELEASE_VERIFY:
            value = self._run_expected(
                ACTION_RELEASE_VERIFY, status="PASS_EXACT_PHASE2_SOURCE_RELEASE"
            )
            if value is None:
                return False
            self.commit(
                "KIMI_RELEASE_VERIFIED",
                {"seal_sha256": value["seal_sha256"]},
                phase=PHASE_GLM_PLAN_VERIFY,
            )
            return True
        if phase == PHASE_GLM_PLAN_VERIFY:
            # The offline planner is a supported no-body path.  Its own verifier
            # fails closed if any bound input or seal differs.
            script = self.repo / "tools/condense/glm52_xet_autotune.py"
            plan = self.repo / "GLM52_XET_AUTOTUNE_PLAN.json"
            if not script.exists() or not plan.exists():
                self._block("GLM_OFFLINE_PLAN_PATH_MISSING", authority=True)
                return False
            value = self._run_expected(ACTION_GLM_PLAN_VERIFY, status="PASS", sealed=False)
            if value is None:
                return False
            if value.get("model_body_bytes_read") != 0 or value.get("network_access") is not False:
                self._block("GLM_OFFLINE_PLAN_CLAIM_CHANGED")
                return False
            path = _persist_document(self.store, GLM_PLAN_RESULT_NAME, value)
            self.commit(
                "GLM_OFFLINE_PLAN_VERIFIED",
                {"result": os.fspath(path), "plan_seal_sha256": value.get("seal_sha256")},
                phase=PHASE_GLM_LIVE_GATE,
            )
            return True
        if phase == PHASE_GLM_LIVE_GATE:
            clean, evidence = self._glm_provenance()
            if not clean:
                self._block(evidence, authority=True)
                return False
            missing = [
                label for label, path in (
                    ("GLM_CONTROLLER_CONFIG", Path(self.state["glm_config"])),
                    ("GLM_TELEGRAM_TRANSITION_AUTHORITY", Path(self.state["glm_authority"])),
                    ("GLM_ARTIFACT_SCRATCH_ROOT", Path(self.state["glm_scratch_root"])),
                ) if not path.exists()
            ]
            if missing:
                self._block(
                    "GLM_LIVE_AUTHORITY_INPUTS_MISSING",
                    authority=True,
                    missing=missing,
                    pushed_head=evidence,
                )
                return False
            status_value = self._run_expected(ACTION_GLM_STATUS, status="GREEN", sealed=False)
            if status_value is None:
                if self.state["phase"] not in {PHASE_BLOCKED, PHASE_BLOCKED_AUTHORITY}:
                    self._block("GLM_CONTROLLER_OR_TELEGRAM_NOT_GREEN", authority=True)
                return False
            live_value = self._run_expected(
                ACTION_GLM_LIVE,
                status="PASS_ALL_12_TRIALS_AND_TWO_FULL_HASH_VALIDATIONS",
                sealed=False,
            )
            if live_value is None:
                if self.state["phase"] not in {PHASE_BLOCKED, PHASE_BLOCKED_AUTHORITY}:
                    self._block("GLM_LIVE_DRIVER_REFUSED_AUTHORITY", authority=True)
                return False
            path = _persist_document(self.store, GLM_LIVE_RESULT_NAME, live_value)
            self._block(
                "GLM_LIVE_AUTOTUNE_COMPLETE_NEXT_SUPPORTED_STAGE_REQUIRES_REVIEW",
                authority=True,
                result=os.fspath(path),
            )
            return False
        self._block("UNKNOWN_DURABLE_PHASE", observed_phase=phase)
        return False

    def tick(self, *, maximum_transitions: int = 16) -> dict[str, Any]:
        loaded = self.store.load()
        if loaded is None:
            raise HandoffError("controller is not bootstrapped")
        self.state = _validate_state(loaded)
        for _ in range(maximum_transitions):
            if not self.advance_once():
                break
        return self.status()

    def status(self) -> dict[str, Any]:
        if not self.store.root.exists():
            return {"status": "NOT_BOOTSTRAPPED", "state_root": os.fspath(self.store.root)}
        self.store.prepare()
        history = self.store.history()
        state = self.state or self.store.load()
        if state is None:
            return {"status": "NOT_BOOTSTRAPPED", "state_root": os.fspath(self.store.root)}
        state = _validate_state(state)
        child = state.get("child")
        return {
            "status": state["phase"],
            "state_root": os.fspath(self.store.root),
            "session": state["session"],
            "generation": state["generation"],
            "journal_records": len(history),
            "journal_head_sha256": history[-1]["seal_sha256"],
            "child": None if child is None else {
                "action": child.get("action"), "pid": child.get("pid"),
                "started_at": child.get("started_at"),
            },
            "not_before_epoch": state.get("not_before_epoch", 0),
            "last_resource_sample": state.get("last_resource_sample"),
            "block": state.get("block"),
        }


def _bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    if args.authorize_exact_kimi_release is not True:
        raise HandoffError(
            "bootstrap requires explicit --authorize-exact-kimi-release"
        )
    if args.session.resolve(strict=True) != SESSION \
            or args.executor_root.resolve(strict=True) != EXECUTOR_ROOT \
            or args.repo_root.resolve(strict=True) != REPO_ROOT:
        raise HandoffError(
            "bootstrap session/operation roots differ from the frozen emergency handoff"
        )
    provenance = verify_release_executor(args.executor_root)
    executor_head = str(provenance["head"])
    store = DurableStore(args.state_root)
    with store.lease(blocking=True):
        authorization = _durable_authorization(store, executor_head=executor_head)
        existing = store.load()
        if existing is None:
            baseline = validate_resource_sample(
                sample_resources(args.session), label="bootstrap resource baseline"
            )
            state = _initial_state(
                args,
                executor_head=executor_head,
                resource_baseline=baseline,
                authorization=authorization,
            )
            store.commit(state, "BOOTSTRAPPED", {
                "session": state["session"],
                "executor_head": state["executor_head"],
                "authorization_seal_sha256": authorization["seal_sha256"],
                "resource_baseline": baseline,
            })
        else:
            state = _validate_state(existing)
            if state["executor_head"] != executor_head:
                raise HandoffError("emergency executor HEAD changed after bootstrap")
            if state["kimi_release_authorization"] != authorization:
                raise HandoffError("state and durable Kimi release authorization differ")
    return Controller(store).status()


def _validate_plist(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = plistlib.load(handle)
    expected = [
        "/usr/bin/caffeinate", "-dimsu", "--", os.fspath(PYTHON),
        os.fspath(CONTROLLER_ROOT / "tools/condense/emergency_detached_campaign.py"),
        "tick", "--state-root", os.fspath(STATE_ROOT),
    ]
    if value.get("Label") != LABEL or value.get("ProgramArguments") != expected \
            or value.get("WorkingDirectory") != os.fspath(CONTROLLER_ROOT) \
            or value.get("RunAtLoad") is not True or value.get("StartInterval") != 30:
        raise HandoffError("launchd plist does not match the frozen detached invocation")
    return value


def _install(args: argparse.Namespace) -> dict[str, Any]:
    _validate_plist(args.plist)
    destination = args.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = args.plist.read_bytes()
    _atomic_private_bytes(destination, raw)
    return {
        "status": "INSTALLED_NOT_BOOTSTRAPPED_INTO_LAUNCHD",
        "plist": os.fspath(destination),
        "next_command": f"launchctl bootstrap gui/{os.getuid()} {shlex.quote(os.fspath(destination))}",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bootstrap = commands.add_parser("bootstrap", help="initialize private durable state only")
    bootstrap.add_argument("--state-root", type=Path, default=STATE_ROOT)
    bootstrap.add_argument("--session", type=Path, default=SESSION)
    bootstrap.add_argument("--executor-root", type=Path, default=EXECUTOR_ROOT)
    bootstrap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    bootstrap.add_argument(
        "--authorize-exact-kimi-release",
        action="store_true",
        help="durably authorize only the exact verified Kimi Phase-2 source release",
    )
    bootstrap.add_argument(
        "--glm-config", type=Path, default=REPO_ROOT / "GLM52_CONTROLLER_CONFIG.json"
    )
    bootstrap.add_argument(
        "--glm-authority", type=Path,
        default=REPO_ROOT / "GLM52_XET_EXECUTION_AUTHORITY.json",
    )
    bootstrap.add_argument(
        "--glm-scratch-root", type=Path,
        default=Path(
            "/Users/scammermike/Library/Application Support/Hawking/GLM52Gravity/artifacts"
        ),
    )
    tick = commands.add_parser("tick", help="advance or reconcile the detached state machine")
    tick.add_argument("--state-root", type=Path, default=STATE_ROOT)
    status = commands.add_parser("status", help="read verified durable status")
    status.add_argument("--state-root", type=Path, default=STATE_ROOT)
    install = commands.add_parser("install", help="install plist without invoking launchctl")
    install.add_argument("--plist", type=Path, default=PLIST_SOURCE)
    install.add_argument("--destination", type=Path, default=PLIST_DESTINATION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "bootstrap":
            result = _bootstrap(args)
        elif args.command == "install":
            result = _install(args)
        else:
            store = DurableStore(args.state_root)
            if args.command == "status":
                result = Controller(store).status()
            else:
                with store.lease():
                    result = Controller(store).tick()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") not in {PHASE_BLOCKED, PHASE_BLOCKED_AUTHORITY} else 3
    except HandoffError as exc:
        print(json.dumps({"status": "REFUSED", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
