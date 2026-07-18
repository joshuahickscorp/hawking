#!/usr/bin/env python3.12
"""Singleton lease, owned process groups, and launchd install for the detached watcher.

Master goal 6.4. The successor control plane must never run two controllers at once, must
never kill a process it does not provably own, and must adopt a running heavy experiment
only after an exact identity match (command + source + environment + PID ancestry +
start-time + campaign identity). This module is that ownership spine.

It provides:
  - SingletonLease: one controller lease via fcntl.flock(LOCK_EX | LOCK_NB). A second
    acquire on the same path is refused as already-running; the lease releases on close.
  - OwnedProcessGroup: an explicitly owned process-group descriptor for one heavy
    experiment (own pgid via os.setsid at launch time in production; described here).
  - can_adopt(observed, expected): fail-closed adoption gate. Adoption requires an exact
    match on command, source_sha256, campaign_generation, pid_ancestry, and start_time.
    A pid alone NEVER authorizes adoption.
  - sample_resources(): read-only RSS/pressure/swap/thermal/disk/cpu sampling. It never
    signals, adopts, or mutates anything.
  - launchd_plist / install_watcher / uninstall_watcher: a durable LaunchAgent plist
    generator and install/uninstall that touch ~/Library/LaunchAgents only on explicit
    --go (selftest writes to an injected tempdir, never the real LaunchAgents).
  - caffeinate_scope: caffeination only while owned heavy work is active; safe drain.

Everything is additive, default-off, and non-interfering. This module never writes under
reports/condense/doctor_v5_ultra (the campaign namespace); successor state lives under
reports/condense/event_horizon_successor/. It never kills unknown processes, never adopts
a live pid without full identity proof, and its selftest uses only synthetic data.
"""
from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import json
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, IO, Iterator

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import (  # noqa: E402
    EcoError, seal_field, sealed, hash_value, now_iso, atomic_write_json,
    read_json_safe, is_sha256, repo_root,
)

WATCHDOG_SCHEMA = "hawking.successor.watchdog.v1"
LEASE_SCHEMA = "hawking.successor.watchdog_lease.v1"
IDENTITY_SCHEMA = "hawking.successor.heavy_identity.v1"
SAMPLE_SCHEMA = "hawking.successor.resource_sample.v1"

# LaunchAgent label for the low-overhead successor watcher. Distinct from the campaign's
# com.hawking.doctorv5ultra.* labels so the two never collide in launchd's namespace.
WATCHER_LABEL = "com.hawking.successor.watchdog"

# The successor's own report/state namespace. Deliberately separate from the campaign's
# reports/condense/doctor_v5_ultra so nothing here can touch campaign-owned state.
def successor_state_root() -> Path:
    return repo_root() / "reports" / "condense" / "event_horizon_successor"


class WatchdogError(EcoError):
    """Fail-closed error in the successor watchdog / ownership spine."""


# ── configuration ─────────────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class Config:
    """Immutable watchdog configuration."""

    label: str = WATCHER_LABEL
    start_interval_seconds: int = 30
    # Low-overhead sampling: read-only stats only, no heavy compute.
    process_type: str = "Background"
    # Aggregate hard stop mirrored from the campaign's operating envelope (advisory here;
    # this module never enforces it against the live campaign).
    aggregate_hard_stop_bytes: int = 78_000_000_000
    # A heavy experiment must be freshly owned; adoption of anything older than this many
    # seconds of divergence in start_time is refused (exact match required, this is a guard).
    start_time_exact: bool = True


def default_config() -> Config:
    return Config()


# ── singleton controller lease (fcntl.flock, non-blocking) ─────────────────────────────
class SingletonLease:
    """One exclusive advisory lease on a lock file via fcntl.flock(LOCK_EX | LOCK_NB).

    A second acquire on the same path (this process or another) is refused with an
    "already-running" WatchdogError. The lease writes an informational owner record into
    the lock file (pid + start marker) and releases the flock on close(). The file is not
    truncated on release so a stale record is visible for diagnostics, but the flock is
    the sole source of truth for liveness.
    """

    def __init__(self, path: str | os.PathLike[str], *, owner: str = "successor-controller"):
        self.path = Path(path)
        self.owner = owner
        self._handle: IO[str] | None = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> "SingletonLease":
        if self._handle is not None:
            raise WatchdogError("lease already held by this handle")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise WatchdogError(
                f"already-running: singleton controller lease is held: {self.path}"
            ) from exc
        # record a diagnostic owner stamp (never authoritative; the flock is).
        try:
            handle.seek(0)
            handle.truncate()
            stamp = {
                "schema": LEASE_SCHEMA,
                "owner": self.owner,
                "pid": os.getpid(),
                "acquired_at": now_iso(),
            }
            handle.write(json.dumps(stamp, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        except OSError:
            # a failed diagnostic write must not defeat the (already-held) lease.
            pass
        self._handle = handle
        return self

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "SingletonLease":
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.close()


def acquire(path: str | os.PathLike[str], *, owner: str = "successor-controller") -> SingletonLease:
    """Acquire the singleton controller lease, or raise already-running WatchdogError."""
    return SingletonLease(path, owner=owner).acquire()


# ── owned process group descriptor ─────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class OwnedProcessGroup:
    """One explicitly owned process group for a single heavy experiment.

    In production the controller launches the heavy child with start_new_session=True so
    the child becomes its own session/process-group leader (pgid == child pid); the
    controller records that pgid here. Ownership means: we launched it, we know its pgid,
    and only processes in this pgid are eligible for our own drain/stop signals. Unknown
    pids are never touched.
    """

    pgid: int
    launcher_pid: int
    identity_sha256: str
    launched_at: str

    def owns(self, pgid: int) -> bool:
        return isinstance(pgid, int) and not isinstance(pgid, bool) and pgid == self.pgid


# ── heavy-experiment identity + adoption gate ──────────────────────────────────────────
_IDENTITY_KEYS = (
    "command", "source_sha256", "campaign_generation", "pid_ancestry", "start_time",
)


def heavy_identity(*, command: list[str], source_sha256: str, campaign_generation: str,
                   pid_ancestry: list[int], start_time: str) -> dict[str, Any]:
    """Build a canonical, self-sealed heavy-experiment identity descriptor.

    This is the full evidence tuple that adoption compares. It intentionally excludes the
    raw pid: a pid is where you look, never why you adopt.
    """
    if not isinstance(command, list) or not command or not all(isinstance(c, str) for c in command):
        raise WatchdogError("command must be a non-empty list of strings")
    if not is_sha256(source_sha256):
        raise WatchdogError("source_sha256 must be a 64-hex sha256")
    if not isinstance(campaign_generation, str) or not campaign_generation:
        raise WatchdogError("campaign_generation must be a non-empty string")
    if not isinstance(pid_ancestry, list) or not pid_ancestry \
            or not all(isinstance(p, int) and not isinstance(p, bool) for p in pid_ancestry):
        raise WatchdogError("pid_ancestry must be a non-empty list of ints")
    if not isinstance(start_time, str) or not start_time:
        raise WatchdogError("start_time must be a non-empty string")
    ident = {
        "schema": IDENTITY_SCHEMA,
        "command": list(command),
        "source_sha256": source_sha256,
        "campaign_generation": campaign_generation,
        "pid_ancestry": list(pid_ancestry),
        "start_time": start_time,
    }
    return seal_field(ident, "identity_sha256")


def can_adopt(observed: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Fail-closed adoption gate: True only on an EXACT identity match, never on pid alone.

    Requires exact equality on command, source_sha256, campaign_generation, pid_ancestry,
    and start_time. Any missing field, type error, seal mismatch, or single divergence
    returns False. There is deliberately no path that adopts on pid or process liveness.
    """
    if not isinstance(observed, dict) or not isinstance(expected, dict):
        return False
    # Both descriptors must be well-formed and self-sealed.
    for side in (observed, expected):
        if side.get("schema") != IDENTITY_SCHEMA:
            return False
        if not sealed(side, "identity_sha256"):
            return False
    # Every identity field must be present and exactly equal.
    for key in _IDENTITY_KEYS:
        if key not in observed or key not in expected:
            return False
        if observed[key] != expected[key]:
            return False
    # Redundant belt-and-braces: the sealed identity hashes must also match, which folds
    # in every field at once (so no unlisted field can silently diverge).
    if observed.get("identity_sha256") != expected.get("identity_sha256"):
        return False
    return True


# ── read-only resource sampling (never signals anything) ───────────────────────────────
def _sysctl_int(name: str) -> int | None:
    try:
        out = subprocess.run(["/usr/sbin/sysctl", "-n", name], capture_output=True,
                             text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    raw = out.stdout.strip()
    try:
        return int(raw)
    except ValueError:
        return None


def _swap_usage() -> dict[str, Any]:
    """Best-effort swap accounting via `sysctl vm.swapusage`. Read-only."""
    try:
        out = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                             capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return {"available": False}
    if out.returncode != 0:
        return {"available": False}
    # example: "total = 2048.00M  used = 512.00M  free = 1536.00M  (encrypted)"
    fields: dict[str, float] = {}
    for token in ("total", "used", "free"):
        marker = f"{token} = "
        idx = out.stdout.find(marker)
        if idx < 0:
            continue
        tail = out.stdout[idx + len(marker):].strip().split()
        if not tail:
            continue
        value = tail[0]
        try:
            if value.endswith("M"):
                fields[token] = float(value[:-1]) * 1024 * 1024
            elif value.endswith("G"):
                fields[token] = float(value[:-1]) * 1024 * 1024 * 1024
            elif value.endswith("K"):
                fields[token] = float(value[:-1]) * 1024
            else:
                fields[token] = float(value)
        except ValueError:
            continue
    if not fields:
        return {"available": False}
    return {"available": True,
            "total_bytes": int(fields.get("total", 0.0)),
            "used_bytes": int(fields.get("used", 0.0)),
            "free_bytes": int(fields.get("free", 0.0))}


def _thermal_pressure() -> dict[str, Any]:
    """Best-effort thermal / power hint. Read-only; absence is not an error."""
    level = _sysctl_int("machdep.xcpm.cpu_thermal_level")
    result: dict[str, Any] = {"available": level is not None}
    if level is not None:
        result["cpu_thermal_level"] = level
    return result


def sample_resources(*, path_for_disk: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Read-only snapshot of free disk, swap, cpu count, thermal, and self RSS.

    NEVER signals, adopts, or mutates anything. Every probe is best-effort: a probe that
    is unavailable on this host is reported as such rather than raising.
    """
    disk_target = Path(path_for_disk) if path_for_disk is not None else repo_root()
    try:
        usage = shutil.disk_usage(str(disk_target))
        free_disk_gb = round(usage.free / 1_000_000_000, 3)
        total_disk_gb = round(usage.total / 1_000_000_000, 3)
    except OSError:
        free_disk_gb = -1.0
        total_disk_gb = -1.0
    self_rss_bytes = _self_rss_bytes()
    return {
        "schema": SAMPLE_SCHEMA,
        "sampled_at": now_iso(),
        "free_disk_gb": free_disk_gb,
        "total_disk_gb": total_disk_gb,
        "disk_path": str(disk_target),
        "cpu_count": os.cpu_count() or 0,
        "physical_memory_bytes": _sysctl_int("hw.memsize"),
        "swap": _swap_usage(),
        "thermal": _thermal_pressure(),
        "self_rss_bytes": self_rss_bytes,
        "read_only": True,
    }


def _self_rss_bytes() -> int | None:
    """Best-effort resident-set size of THIS process via `ps`. Read-only, no signal."""
    try:
        out = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
                             capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    raw = out.stdout.strip()
    try:
        return int(raw) * 1024  # ps reports RSS in KiB
    except ValueError:
        return None


def heartbeat(path: str | os.PathLike[str], payload: dict[str, Any] | None = None) -> Path:
    """Atomically write a self-sealed heartbeat marker for the watcher. Read-only elsewhere."""
    doc = {
        "schema": WATCHDOG_SCHEMA,
        "beat_at": now_iso(),
        "pid": os.getpid(),
        "resources": sample_resources(),
        **(payload or {}),
    }
    doc = seal_field(doc, "heartbeat_sha256")
    return atomic_write_json(path, doc)


# ── launchd plist generation + install / uninstall ─────────────────────────────────────
def launchd_plist(label: str, program_args: list[str], start_interval: int, *,
                  working_directory: str | None = None,
                  stdout_path: str | None = None, stderr_path: str | None = None,
                  process_type: str = "Background", run_at_load: bool = True) -> str:
    """Return the LaunchAgent plist XML for the low-overhead successor watcher.

    Uses StartInterval (periodic sampling) rather than a resident daemon, matching the
    campaign's autoresume LaunchAgent idiom. ProcessType defaults to Background so the
    watcher stays low-priority and never competes with owned heavy work.
    """
    if not isinstance(label, str) or not label:
        raise WatchdogError("plist label must be a non-empty string")
    if not isinstance(program_args, list) or not program_args \
            or not all(isinstance(a, str) for a in program_args):
        raise WatchdogError("program_args must be a non-empty list of strings")
    if not isinstance(start_interval, int) or isinstance(start_interval, bool) or start_interval <= 0:
        raise WatchdogError("start_interval must be a positive int (seconds)")
    document: dict[str, Any] = {
        "Label": label,
        "ProgramArguments": list(program_args),
        "RunAtLoad": run_at_load,
        "StartInterval": start_interval,
        "ProcessType": process_type,
    }
    if working_directory is not None:
        document["WorkingDirectory"] = working_directory
    if stdout_path is not None:
        document["StandardOutPath"] = stdout_path
    if stderr_path is not None:
        document["StandardErrorPath"] = stderr_path
    return plistlib.dumps(document, fmt=plistlib.FMT_XML, sort_keys=True).decode("utf-8")


def _launch_agents_dir(base: str | os.PathLike[str] | None) -> Path:
    """Resolve the LaunchAgents directory. `base` is injectable so selftest never touches
    the real ~/Library/LaunchAgents."""
    if base is not None:
        return Path(base)
    return Path.home() / "Library" / "LaunchAgents"


def _atomic_bytes(path: Path, raw: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def install_watcher(label: str, program_args: list[str], start_interval: int, *,
                    go: bool = False, launch_agents_dir: str | os.PathLike[str] | None = None,
                    load: bool = False, **plist_kwargs: Any) -> dict[str, Any]:
    """Write the watcher LaunchAgent plist. Writes to disk ONLY on explicit go=True.

    launch_agents_dir is injectable; selftest passes a tempdir so the real
    ~/Library/LaunchAgents is never modified. `load` gates the (real) launchctl bootstrap
    behind go as well, and is left False in selftest.
    """
    xml = launchd_plist(label, program_args, start_interval, **plist_kwargs)
    target = _launch_agents_dir(launch_agents_dir) / f"{label}.plist"
    if not go:
        return {"go": False, "would_write": str(target), "installed": False,
                "plist_preview_sha256": hash_value(xml)}
    _atomic_bytes(target, xml.encode("utf-8"))
    result: dict[str, Any] = {"go": True, "installed": True, "path": str(target)}
    if load:
        domain = f"gui/{os.getuid()}"
        boot = subprocess.run(["/bin/launchctl", "bootstrap", domain, str(target)],
                             capture_output=True, text=True, check=False)
        result["loaded"] = boot.returncode == 0
        result["launchctl_returncode"] = boot.returncode
    return result


def uninstall_watcher(label: str, *, go: bool = False,
                      launch_agents_dir: str | os.PathLike[str] | None = None,
                      unload: bool = False) -> dict[str, Any]:
    """Remove the watcher LaunchAgent plist. Removes from disk ONLY on explicit go=True.

    This never touches campaign-owned LaunchAgents: it acts solely on the file named
    `<label>.plist` in the (injectable) LaunchAgents directory.
    """
    target = _launch_agents_dir(launch_agents_dir) / f"{label}.plist"
    if not go:
        return {"go": False, "would_remove": str(target), "removed": False,
                "present": target.exists()}
    result: dict[str, Any] = {"go": True}
    if unload:
        domain = f"gui/{os.getuid()}"
        subprocess.run(["/bin/launchctl", "bootout", domain, str(target)],
                       capture_output=True, text=True, check=False)
        result["unloaded"] = True
    existed = target.exists()
    if existed and not target.is_symlink():
        target.unlink()
    result["removed"] = existed
    result["path"] = str(target)
    return result


# ── caffeination scoped to owned heavy work ────────────────────────────────────────────
@contextlib.contextmanager
def caffeinate_scope(owned: OwnedProcessGroup | None, *, go: bool = False) -> Iterator[dict[str, Any]]:
    """Hold macOS caffeinate ONLY while an owned heavy experiment is active.

    Caffeination is refused when there is no owned process group (nothing heavy to keep
    awake for). The subprocess is spawned only on go=True (selftest leaves go=False and
    asserts the guard shape). On exit the caffeinate process is terminated (safe drain);
    if it was never started, exit is a no-op.
    """
    if owned is None:
        # No owned heavy work: do not caffeinate.
        yield {"caffeinated": False, "reason": "no owned heavy work"}
        return
    proc: subprocess.Popen[bytes] | None = None
    if go:
        try:
            proc = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(owned.pgid)])
        except OSError:
            proc = None
    try:
        yield {"caffeinated": proc is not None, "pgid": owned.pgid,
               "go": go}
    finally:
        if proc is not None:
            with contextlib.suppress(OSError):
                proc.terminate()
            with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                proc.wait(timeout=5)


def drain(owned: OwnedProcessGroup | None, *, go: bool = False) -> dict[str, Any]:
    """Safe drain of an owned heavy process group. NEVER signals an unknown pgid.

    Refuses to act unless there is a recorded OwnedProcessGroup whose pgid we launched.
    On go=True it would send SIGTERM to the owned pgid only (os.killpg). selftest leaves
    go=False so no signal is ever delivered; the design is exercised, the effect is inert.
    """
    if owned is None:
        return {"drained": False, "reason": "nothing owned to drain", "signalled": False}
    if not isinstance(owned.pgid, int) or isinstance(owned.pgid, bool) or owned.pgid <= 1:
        # pgid 0/1 or non-int would broadcast; refuse fail-closed.
        raise WatchdogError(f"refusing to drain unsafe pgid: {owned.pgid!r}")
    if not go:
        return {"drained": False, "would_signal_pgid": owned.pgid, "signal": "SIGTERM",
                "signalled": False, "note": "go=False; no signal delivered"}
    import signal
    os.killpg(owned.pgid, signal.SIGTERM)
    return {"drained": True, "signalled": True, "pgid": owned.pgid, "signal": "SIGTERM"}


# ── offline selftest ───────────────────────────────────────────────────────────────────
def selftest() -> dict[str, Any]:
    import tempfile

    results: dict[str, Any] = {"ok": True}
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        # 1) singleton lease: first acquire holds; second acquire is refused already-running.
        lock = root / "controller.lock"
        first = acquire(lock)
        second_refused = False
        try:
            SingletonLease(lock).acquire()
        except WatchdogError as exc:
            second_refused = "already-running" in str(exc)
        if not second_refused:
            raise WatchdogError("second acquire was not refused as already-running")
        # release; a fresh acquire then succeeds.
        first.close()
        reacquired = acquire(lock)
        reacquired.close()
        results["singleton_lease"] = {"second_refused": True, "reacquire_ok": True}

        # 2) can_adopt: exact match True; any mismatch and pid-only False.
        base = dict(command=["python3.12", "campaign.py", "start"],
                    source_sha256="a" * 64, campaign_generation="gen-7",
                    pid_ancestry=[1, 4020, 4090], start_time="2026-07-17T00:00:00+00:00")
        expected = heavy_identity(**base)
        observed_match = heavy_identity(**base)
        if not can_adopt(observed_match, expected):
            raise WatchdogError("exact identity match was not adopted")

        mismatches = {}
        # command differs
        m_cmd = heavy_identity(**{**base, "command": ["python3.12", "campaign.py", "resume"]})
        mismatches["command"] = can_adopt(m_cmd, expected)
        # source_sha256 differs
        m_src = heavy_identity(**{**base, "source_sha256": "b" * 64})
        mismatches["source_sha256"] = can_adopt(m_src, expected)
        # campaign_generation differs
        m_gen = heavy_identity(**{**base, "campaign_generation": "gen-8"})
        mismatches["campaign_generation"] = can_adopt(m_gen, expected)
        # pid_ancestry differs
        m_anc = heavy_identity(**{**base, "pid_ancestry": [1, 4020, 9999]})
        mismatches["pid_ancestry"] = can_adopt(m_anc, expected)
        # start_time differs
        m_st = heavy_identity(**{**base, "start_time": "2026-07-17T00:00:01+00:00"})
        mismatches["start_time"] = can_adopt(m_st, expected)
        # pid-only lookalike: same pid_ancestry tail but everything else wrong.
        m_pidonly = heavy_identity(command=["evil"], source_sha256="c" * 64,
                                   campaign_generation="rogue",
                                   pid_ancestry=base["pid_ancestry"],
                                   start_time="1970-01-01T00:00:00+00:00")
        mismatches["pid_only"] = can_adopt(m_pidonly, expected)
        # tampered seal: flip a field after sealing.
        tampered = dict(observed_match)
        tampered["campaign_generation"] = "gen-999"
        mismatches["tampered_seal"] = can_adopt(tampered, expected)
        # not an identity object at all.
        mismatches["empty"] = can_adopt({}, expected)

        if any(mismatches.values()):
            raise WatchdogError(f"a mismatch was wrongly adopted: {mismatches}")
        results["can_adopt"] = {"exact_match_adopted": True,
                                "all_mismatches_refused": True,
                                "mismatch_cases": sorted(mismatches)}

        # 3) sample_resources: read-only dict with free_disk_gb.
        sample = sample_resources(path_for_disk=str(root))
        if "free_disk_gb" not in sample or not sample.get("read_only"):
            raise WatchdogError("sample_resources missing free_disk_gb / read_only")
        if not isinstance(sample.get("cpu_count"), int):
            raise WatchdogError("sample_resources cpu_count not an int")
        results["sample_resources"] = {"free_disk_gb": sample["free_disk_gb"],
                                       "cpu_count": sample["cpu_count"],
                                       "keys": sorted(sample)}

        # 4) launchd_plist: contains label + program args + start interval.
        args = [sys.executable, str(root / "succ_watch_entry.py"), "sample"]
        xml = launchd_plist(WATCHER_LABEL, args, 30, working_directory=str(root),
                            stdout_path=str(root / "watch.log"))
        if WATCHER_LABEL not in xml or args[0] not in xml or "StartInterval" not in xml:
            raise WatchdogError("launchd_plist missing label/args/interval")
        parsed = plistlib.loads(xml.encode("utf-8"))
        if parsed.get("Label") != WATCHER_LABEL or parsed.get("ProgramArguments") != args \
                or parsed.get("StartInterval") != 30:
            raise WatchdogError("launchd_plist round-trip mismatch")

        # 5) install/uninstall write ONLY to the injected tempdir, and only on go=True.
        agents = root / "FakeLaunchAgents"
        dry = install_watcher(WATCHER_LABEL, args, 30, go=False, launch_agents_dir=str(agents))
        if dry["installed"] or agents.exists():
            raise WatchdogError("dry-run install wrote to disk")
        wet = install_watcher(WATCHER_LABEL, args, 30, go=True, launch_agents_dir=str(agents),
                              working_directory=str(root))
        target = agents / f"{WATCHER_LABEL}.plist"
        if not wet["installed"] or not target.exists():
            raise WatchdogError("go install did not write the plist")
        # confirm we never touched the real LaunchAgents path.
        if str(Path.home() / "Library" / "LaunchAgents") in str(target):
            raise WatchdogError("install targeted the real LaunchAgents dir")
        rm_dry = uninstall_watcher(WATCHER_LABEL, go=False, launch_agents_dir=str(agents))
        if rm_dry["removed"] or not target.exists():
            raise WatchdogError("dry-run uninstall removed the plist")
        rm = uninstall_watcher(WATCHER_LABEL, go=True, launch_agents_dir=str(agents))
        if not rm["removed"] or target.exists():
            raise WatchdogError("go uninstall did not remove the plist")
        results["launchd"] = {"plist_ok": True, "install_go_only": True,
                              "uninstall_go_only": True, "used_tempdir": str(agents)}

        # 6) ownership: owned pgid, caffeinate + drain refuse to signal without go.
        ident = heavy_identity(**base)
        owned = OwnedProcessGroup(pgid=424242, launcher_pid=os.getpid(),
                                  identity_sha256=ident["identity_sha256"],
                                  launched_at=now_iso())
        if not owned.owns(424242) or owned.owns(1):
            raise WatchdogError("OwnedProcessGroup.owns is wrong")
        with caffeinate_scope(None) as c_none:
            if c_none["caffeinated"]:
                raise WatchdogError("caffeinated with no owned work")
        with caffeinate_scope(owned, go=False) as c_owned:
            if c_owned["caffeinated"]:
                raise WatchdogError("caffeinate spawned without go")
        drain_none = drain(None)
        drain_dry = drain(owned, go=False)
        if drain_none["signalled"] or drain_dry["signalled"]:
            raise WatchdogError("drain signalled without go / owner")
        # unsafe pgid refuses fail-closed.
        unsafe_refused = False
        try:
            drain(OwnedProcessGroup(pgid=1, launcher_pid=os.getpid(),
                                    identity_sha256=ident["identity_sha256"],
                                    launched_at=now_iso()), go=True)
        except WatchdogError:
            unsafe_refused = True
        if not unsafe_refused:
            raise WatchdogError("drain did not refuse unsafe pgid")
        results["ownership"] = {"owns_ok": True, "caffeinate_guarded": True,
                                "drain_go_gated": True, "unsafe_pgid_refused": True}

        # 7) heartbeat writes a sealed marker under an injected dir (never campaign ns).
        hb_path = root / "successor" / "heartbeat.json"
        heartbeat(hb_path, {"state": "MONITOR"})
        hb = read_json_safe(hb_path)
        if not sealed(hb, "heartbeat_sha256") or hb.get("schema") != WATCHDOG_SCHEMA:
            raise WatchdogError("heartbeat not sealed / wrong schema")
        results["heartbeat"] = {"sealed": True, "schema": hb["schema"]}

    results["non_interference"] = {
        "successor_state_root": str(successor_state_root()),
        "never_campaign_namespace": "doctor_v5_ultra" not in str(successor_state_root()),
    }
    return results


if __name__ == "__main__":
    print(json.dumps(selftest(), indent=2, sort_keys=True))
