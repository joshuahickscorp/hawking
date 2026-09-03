"""Resource-budget supervisor for resumable DeepSeek-V4 ``build-full``.

This is a launchable control-plane, not a downloader.  Each candidate is
started through the existing::

    tools/condense/deepseek_v4_gravity.py build-full ...

CLI so the sealed stream journal remains the resumption authority.  The
supervisor never opens Hub/Xet sessions, never rewrites source-range limits,
and never treats shell ``ulimit`` as proof of an RSS cap.

Policy surface (small, injectable for tests):

* **RSS budget** — hard 5 GiB process RSS for the child; breach fails closed
  (terminate child, stop ramp) before hoping the OOM killer intervenes.
* **Disk floor** — caller-provided hard floor only.  ``25 GiB`` is a common
  *invocation* example; it is not a default replacement for the caller's
  value (the gravity engine still enforces its own non-negotiable lower bound).
* **Swap** — no growth above the baseline captured at ramp start.
* **Worker ramp** — explicit bounded sequence (default ``4, 8, 12, 16``),
  never above the gravity code ceiling of ``32``.  A stable measurement
  window is sampled at a bounded interval; the ramp stops on budget breach,
  swap growth, error exit, or no material throughput gain.
* **Utilization** — observed CPU percent is recorded as-is.  This module
  does not promise or attempt “zero CPU idle”.

Dry-run validates the plan and emits a full receipt without spawning a
child process.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


# ---------------------------------------------------------------------------
# Constants — keep aligned with lab.operators.deepseek_v4_gravity
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_FULL_CLI = REPO_ROOT / "tools" / "condense" / "deepseek_v4_gravity.py"

# Match deepseek_v4_gravity.MAX_PARALLEL_WORKERS without importing that module
# (it pulls numpy and the full stream engine).
MAX_PARALLEL_WORKERS = 32
DEFAULT_WORKER_RAMP: tuple[int, ...] = (4, 8, 12, 16)

# Strict child RSS budget.  Enforcement is by sampling + terminate, not ulimit.
RSS_BUDGET_BYTES = 5 * 1024**3

# Gravity still refuses floors below 15 GiB; we surface the same floor as a
# documentation default for callers who do not override.  25 GiB is only an
# invocation example in the CLI help, never a hard-coded replacement.
DEFAULT_PROTECTED_FLOOR_BYTES = 15 * 1024**3
EXAMPLE_INVOCATION_FLOOR_BYTES = 25 * 1024**3

DEFAULT_SAMPLE_INTERVAL_SECONDS = 2.0
DEFAULT_MEASURE_WINDOW_SECONDS = 30.0
DEFAULT_MIN_THROUGHPUT_GAIN = 0.05  # 5% material gain required to continue ramp
DEFAULT_TERMINATE_GRACE_SECONDS = 5.0
MAX_SAMPLE_INTERVAL_SECONDS = 30.0
MAX_MEASURE_WINDOW_SECONDS = 3600.0
MIN_SAMPLE_INTERVAL_SECONDS = 0.05

RECEIPT_SCHEMA = "hawking.gravity.deepseek_v4.build_full_resource_supervisor.v1"
SAMPLE_SCHEMA = "hawking.gravity.deepseek_v4.build_full_resource_sample.v1"

_SWAP_USED_RE = re.compile(
    r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)([KMGT])\b", re.IGNORECASE
)
_BYTE_UNITS = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


class ResourceSupervisorError(RuntimeError):
    """Configuration or hard policy failure in the resource supervisor."""


# ---------------------------------------------------------------------------
# Injectable surfaces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceSample:
    """One bounded observation of the child / host during a candidate trial."""

    monotonic_s: float
    process_rss_bytes: int
    cpu_percent: float
    free_disk_bytes: int
    swap_used_bytes: int
    transfer_bytes: int
    pid: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SAMPLE_SCHEMA,
            "monotonic_s": self.monotonic_s,
            "process_rss_bytes": self.process_rss_bytes,
            "cpu_percent": self.cpu_percent,
            "free_disk_bytes": self.free_disk_bytes,
            "swap_used_bytes": self.swap_used_bytes,
            "transfer_bytes": self.transfer_bytes,
            "pid": self.pid,
        }


class ProcessHandle(Protocol):
    """Minimal child process surface used by the supervisor loop."""

    @property
    def pid(self) -> int | None: ...

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class ResourceProvider(Protocol):
    """Host/process stat provider.  Tests inject a fake implementation."""

    def sample(self, pid: int | None) -> ResourceSample: ...


ChildLauncher = Callable[[Sequence[str], Mapping[str, str] | None], ProcessHandle]
Sleeper = Callable[[float], None]
Clock = Callable[[], float]


@dataclass
class SupervisorPolicy:
    """Fail-closed budgets applied to every sample in a ramp."""

    rss_budget_bytes: int = RSS_BUDGET_BYTES
    protected_floor_bytes: int = DEFAULT_PROTECTED_FLOOR_BYTES
    allow_swap_growth: bool = False
    min_throughput_gain: float = DEFAULT_MIN_THROUGHPUT_GAIN

    def __post_init__(self) -> None:
        if (
            isinstance(self.rss_budget_bytes, bool)
            or not isinstance(self.rss_budget_bytes, int)
            or self.rss_budget_bytes <= 0
        ):
            raise ResourceSupervisorError("rss_budget_bytes must be a positive integer")
        if (
            isinstance(self.protected_floor_bytes, bool)
            or not isinstance(self.protected_floor_bytes, int)
            or self.protected_floor_bytes <= 0
        ):
            raise ResourceSupervisorError(
                "protected_floor_bytes must be a positive integer"
            )
        if (
            isinstance(self.min_throughput_gain, bool)
            or not isinstance(self.min_throughput_gain, (int, float))
            or self.min_throughput_gain < 0
        ):
            raise ResourceSupervisorError(
                "min_throughput_gain must be a non-negative number"
            )
        object.__setattr__(
            self, "min_throughput_gain", float(self.min_throughput_gain)
        )


@dataclass
class SupervisorConfig:
    """Caller-facing plan for one supervised ``build-full`` ramp."""

    artifact_dir: Path
    workspace_root: Path
    xet_root: Path
    protected_floor_bytes: int = DEFAULT_PROTECTED_FLOOR_BYTES
    range_bytes: int | None = None
    worker_ramp: tuple[int, ...] = DEFAULT_WORKER_RAMP
    rss_budget_bytes: int = RSS_BUDGET_BYTES
    sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS
    measure_window_seconds: float = DEFAULT_MEASURE_WINDOW_SECONDS
    min_throughput_gain: float = DEFAULT_MIN_THROUGHPUT_GAIN
    python_executable: str = field(default_factory=lambda: sys.executable)
    gravity_cli: Path = field(default_factory=lambda: BUILD_FULL_CLI)
    dry_run: bool = False
    terminate_grace_seconds: float = DEFAULT_TERMINATE_GRACE_SECONDS

    def __post_init__(self) -> None:
        self.artifact_dir = _absolute(self.artifact_dir, "artifact_dir")
        self.workspace_root = _absolute(self.workspace_root, "workspace_root")
        self.xet_root = _absolute(self.xet_root, "xet_root")
        self.gravity_cli = _absolute(self.gravity_cli, "gravity_cli")
        self.worker_ramp = normalize_worker_ramp(self.worker_ramp)
        if (
            isinstance(self.sample_interval_seconds, bool)
            or not isinstance(self.sample_interval_seconds, (int, float))
            or not (
                MIN_SAMPLE_INTERVAL_SECONDS
                <= float(self.sample_interval_seconds)
                <= MAX_SAMPLE_INTERVAL_SECONDS
            )
        ):
            raise ResourceSupervisorError(
                "sample_interval_seconds must be in "
                f"[{MIN_SAMPLE_INTERVAL_SECONDS}, {MAX_SAMPLE_INTERVAL_SECONDS}]"
            )
        self.sample_interval_seconds = float(self.sample_interval_seconds)
        if (
            isinstance(self.measure_window_seconds, bool)
            or not isinstance(self.measure_window_seconds, (int, float))
            or not (0.0 < float(self.measure_window_seconds) <= MAX_MEASURE_WINDOW_SECONDS)
        ):
            raise ResourceSupervisorError(
                "measure_window_seconds must be in "
                f"(0, {MAX_MEASURE_WINDOW_SECONDS}]"
            )
        self.measure_window_seconds = float(self.measure_window_seconds)
        if self.range_bytes is not None:
            if (
                isinstance(self.range_bytes, bool)
                or not isinstance(self.range_bytes, int)
                or self.range_bytes <= 0
            ):
                raise ResourceSupervisorError(
                    "range_bytes must be a positive integer when provided"
                )
        if (
            isinstance(self.terminate_grace_seconds, bool)
            or not isinstance(self.terminate_grace_seconds, (int, float))
            or float(self.terminate_grace_seconds) < 0
        ):
            raise ResourceSupervisorError(
                "terminate_grace_seconds must be a non-negative number"
            )
        self.terminate_grace_seconds = float(self.terminate_grace_seconds)
        # Policy validation lives on SupervisorPolicy.
        self.policy()

    def policy(self) -> SupervisorPolicy:
        return SupervisorPolicy(
            rss_budget_bytes=self.rss_budget_bytes,
            protected_floor_bytes=self.protected_floor_bytes,
            allow_swap_growth=False,
            min_throughput_gain=self.min_throughput_gain,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _absolute(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ResourceSupervisorError(f"{label} must be an absolute path")
    return Path(os.path.abspath(os.fspath(path)))


def normalize_worker_ramp(values: Sequence[int] | str) -> tuple[int, ...]:
    """Parse and validate an explicit worker ramp against the 32-worker ceiling."""
    if isinstance(values, str):
        parts = [part.strip() for part in values.split(",") if part.strip()]
        if not parts:
            raise ResourceSupervisorError("worker_ramp must not be empty")
        try:
            raw: Sequence[object] = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise ResourceSupervisorError(
                "worker_ramp must be a comma-separated list of integers"
            ) from exc
    else:
        raw = tuple(values)
    if not raw:
        raise ResourceSupervisorError("worker_ramp must not be empty")
    parsed: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ResourceSupervisorError(
                "every worker_ramp entry must be an integer >= 1"
            )
        if value > MAX_PARALLEL_WORKERS:
            raise ResourceSupervisorError(
                f"worker_ramp must not exceed the gravity ceiling of "
                f"{MAX_PARALLEL_WORKERS} workers"
            )
        parsed.append(value)
    if len(set(parsed)) != len(parsed):
        raise ResourceSupervisorError("worker_ramp must not contain duplicates")
    if parsed != sorted(parsed):
        raise ResourceSupervisorError("worker_ramp must be strictly ascending")
    return tuple(parsed)


def build_full_argv(
    config: SupervisorConfig,
    *,
    parallel_workers: int,
) -> list[str]:
    """Construct the exact ``build-full`` argv for one candidate.

    Source-range limits are only forwarded when the caller set them; this
    supervisor never invents a different range size.
    """
    if parallel_workers < 1 or parallel_workers > MAX_PARALLEL_WORKERS:
        raise ResourceSupervisorError(
            f"parallel_workers must be between 1 and {MAX_PARALLEL_WORKERS}"
        )
    argv = [
        str(config.python_executable),
        str(config.gravity_cli),
        "build-full",
        "--artifact-dir",
        str(config.artifact_dir),
        "--workspace-root",
        str(config.workspace_root),
        "--xet-root",
        str(config.xet_root),
        "--protected-floor-bytes",
        str(config.protected_floor_bytes),
        "--parallel-workers",
        str(parallel_workers),
    ]
    if config.range_bytes is not None:
        argv.extend(["--range-bytes", str(config.range_bytes)])
    return argv


def parse_swapusage(text: str) -> int:
    """Parse Darwin ``sysctl vm.swapusage`` used= field into bytes."""
    match = _SWAP_USED_RE.search(text)
    if match is None:
        raise ResourceSupervisorError("cannot parse swap usage text")
    return int(float(match.group(1)) * _BYTE_UNITS[match.group(2).upper()])


def parse_ps_cpu_rss(text: str) -> tuple[float, int]:
    """Parse ``ps -o %cpu=,rss=`` output into (cpu_percent, rss_bytes)."""
    fields = text.split()
    if len(fields) != 2:
        raise ResourceSupervisorError("cannot parse ps CPU/RSS sample")
    try:
        cpu = float(fields[0])
        rss_kib = int(fields[1])
    except ValueError as exc:
        raise ResourceSupervisorError("ps CPU/RSS sample is nonnumeric") from exc
    if cpu < 0 or rss_kib < 0:
        raise ResourceSupervisorError("ps CPU/RSS sample has negative values")
    return cpu, rss_kib * 1024


def directory_transfer_bytes(root: Path) -> int:
    """Sum regular-file sizes under *root* as a transfer proxy.

    Used as the default transfer meter: growth in the content-addressed
    artifact directory reflects completed range writes without opening a
    network client.
    """
    if not root.exists():
        return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            path = Path(dirpath) / name
            try:
                st = path.lstat()
            except OSError:
                continue
            if stat_is_regular(st.st_mode):
                total += int(st.st_size)
    return total


def stat_is_regular(mode: int) -> bool:
    import stat as statmod

    return statmod.S_ISREG(mode)


def evaluate_sample(
    sample: ResourceSample,
    *,
    baseline_swap_used_bytes: int,
    policy: SupervisorPolicy,
) -> list[str]:
    """Return ordered breach reasons; empty means the sample is admissible."""
    reasons: list[str] = []
    if sample.process_rss_bytes > policy.rss_budget_bytes:
        reasons.append("RSS_BUDGET_BREACH")
    if sample.free_disk_bytes < policy.protected_floor_bytes:
        reasons.append("DISK_FLOOR_BREACH")
    if (
        not policy.allow_swap_growth
        and sample.swap_used_bytes > baseline_swap_used_bytes
    ):
        reasons.append("SWAP_GROWTH")
    return reasons


def material_throughput_gain(
    previous_bps: float | None,
    current_bps: float,
    *,
    min_gain: float,
) -> bool:
    """True when *current_bps* is a material improvement over *previous_bps*."""
    if previous_bps is None:
        return True
    if previous_bps <= 0:
        return current_bps > 0
    return current_bps >= previous_bps * (1.0 + min_gain)


# ---------------------------------------------------------------------------
# Default providers
# ---------------------------------------------------------------------------


class HostResourceProvider:
    """Portable host sampler; process RSS/CPU via ``ps`` when a pid is live.

    Enforcement of the RSS budget is done by comparing sampled
    ``process_rss_bytes`` to the policy budget and terminating the child.
    This deliberately does **not** use shell ``ulimit`` as proof.
    """

    def __init__(
        self,
        *,
        workspace_root: Path,
        artifact_dir: Path,
        command_runner: Callable[[Sequence[str]], str] | None = None,
        clock: Clock = time.monotonic,
        transfer_bytes_fn: Callable[[], int] | None = None,
    ) -> None:
        self.workspace_root = workspace_root
        self.artifact_dir = artifact_dir
        self.command_runner = command_runner or _run_text
        self.clock = clock
        self.transfer_bytes_fn = transfer_bytes_fn or (
            lambda: directory_transfer_bytes(artifact_dir)
        )

    def sample(self, pid: int | None) -> ResourceSample:
        try:
            free_disk = int(shutil.disk_usage(self.workspace_root).free)
        except OSError as exc:
            raise ResourceSupervisorError(
                f"cannot sample free disk for {self.workspace_root}: {exc}"
            ) from exc
        swap_used = self._swap_used_bytes()
        if pid is None:
            cpu = 0.0
            rss = 0
        else:
            cpu, rss = self._cpu_rss(pid)
        return ResourceSample(
            monotonic_s=float(self.clock()),
            process_rss_bytes=rss,
            cpu_percent=cpu,
            free_disk_bytes=free_disk,
            swap_used_bytes=swap_used,
            transfer_bytes=int(self.transfer_bytes_fn()),
            pid=pid,
        )

    def _swap_used_bytes(self) -> int:
        # Prefer Darwin sysctl; fall back to /proc/meminfo on Linux; else 0
        # with an explicit unavailable path only when both fail is unacceptable
        # for swap-growth policy — surface 0 only when the platform has no
        # swap counter we can read (tests inject a fake).
        if sys.platform == "darwin":
            text = self.command_runner(("/usr/sbin/sysctl", "-n", "vm.swapusage"))
            return parse_swapusage(text)
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            total = 0
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith(("SwapTotal:", "SwapFree:")):
                    parts = line.split()
                    if len(parts) >= 2:
                        kib = int(parts[1])
                        if line.startswith("SwapTotal:"):
                            total += kib
                        else:
                            total -= kib
            if total < 0:
                total = 0
            return total * 1024
        return 0

    def _cpu_rss(self, pid: int) -> tuple[float, int]:
        text = self.command_runner(("/bin/ps", "-o", "%cpu=,rss=", "-p", str(pid)))
        return parse_ps_cpu_rss(text)


def _run_text(argv: Sequence[str], *, timeout: float = 5.0) -> str:
    try:
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ResourceSupervisorError(
            f"resource command failed to start: {argv[0]}: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise ResourceSupervisorError(
            f"resource command failed ({completed.returncode}): {argv[0]}: {detail}"
        )
    return completed.stdout


class SubprocessHandle:
    """Thin wrapper so tests can implement ``ProcessHandle`` without subprocess."""

    def __init__(self, process: subprocess.Popen[Any]) -> None:
        self._process = process

    @property
    def pid(self) -> int | None:
        return self._process.pid

    def poll(self) -> int | None:
        return self._process.poll()

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        self._process.kill()

    def wait(self, timeout: float | None = None) -> int:
        return int(self._process.wait(timeout=timeout))


def default_launcher(
    argv: Sequence[str], env: Mapping[str, str] | None = None
) -> ProcessHandle:
    """Spawn a child without shelling out; inherits a filtered environment."""
    child_env = os.environ.copy()
    if env:
        child_env.update(dict(env))
    try:
        process = subprocess.Popen(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=child_env,
        )
    except OSError as exc:
        raise ResourceSupervisorError(f"failed to launch child: {exc}") from exc
    return SubprocessHandle(process)


# ---------------------------------------------------------------------------
# Supervisor loop
# ---------------------------------------------------------------------------


@dataclass
class CandidateResult:
    workers: int
    status: str
    reasons: list[str]
    samples: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    throughput_bytes_per_second: float | None
    observed_cpu_percent_mean: float | None
    peak_rss_bytes: int | None
    exit_code: int | None
    argv: list[str]
    duration_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "workers": self.workers,
            "status": self.status,
            "reasons": list(self.reasons),
            "samples": list(self.samples),
            "actions": list(self.actions),
            "throughput_bytes_per_second": self.throughput_bytes_per_second,
            "observed_cpu_percent_mean": self.observed_cpu_percent_mean,
            "peak_rss_bytes": self.peak_rss_bytes,
            "exit_code": self.exit_code,
            "argv": list(self.argv),
            "duration_seconds": self.duration_seconds,
        }


def stop_child(
    handle: ProcessHandle,
    *,
    grace_seconds: float,
    sleeper: Sleeper = time.sleep,
    actions: list[dict[str, Any]] | None = None,
    reason: str = "MEASURE_WINDOW_COMPLETE",
) -> int | None:
    """Fail-closed child stop: SIGTERM, wait grace, then SIGKILL if needed."""
    log = actions if actions is not None else []
    if handle.poll() is not None:
        code = handle.poll()
        log.append(
            {
                "action": "CHILD_ALREADY_EXITED",
                "reason": reason,
                "exit_code": code,
                "at": _utc_now(),
            }
        )
        return code
    log.append(
        {
            "action": "CHILD_TERMINATE",
            "reason": reason,
            "pid": handle.pid,
            "at": _utc_now(),
        }
    )
    try:
        handle.terminate()
    except OSError:
        pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        code = handle.poll()
        if code is not None:
            log.append(
                {
                    "action": "CHILD_EXITED_AFTER_TERMINATE",
                    "exit_code": code,
                    "at": _utc_now(),
                }
            )
            return code
        sleeper(min(0.05, max(0.0, deadline - time.monotonic())))
    log.append(
        {
            "action": "CHILD_KILL",
            "reason": "TERMINATE_GRACE_EXCEEDED",
            "pid": handle.pid,
            "at": _utc_now(),
        }
    )
    try:
        handle.kill()
    except OSError:
        pass
    try:
        code = handle.wait(timeout=max(grace_seconds, 1.0))
    except Exception:
        code = handle.poll()
    log.append(
        {
            "action": "CHILD_EXITED_AFTER_KILL",
            "exit_code": code,
            "at": _utc_now(),
        }
    )
    return code


def run_candidate(
    config: SupervisorConfig,
    *,
    workers: int,
    provider: ResourceProvider,
    launcher: ChildLauncher,
    baseline_swap_used_bytes: int,
    sleeper: Sleeper = time.sleep,
    clock: Clock = time.monotonic,
) -> CandidateResult:
    """Launch one ``build-full`` candidate, sample a stable window, stop cleanly."""
    policy = config.policy()
    argv = build_full_argv(config, parallel_workers=workers)
    actions: list[dict[str, Any]] = [
        {
            "action": "LAUNCH_CANDIDATE",
            "workers": workers,
            "argv": list(argv),
            "at": _utc_now(),
            "reason": "RAMP_STEP",
        }
    ]
    samples: list[dict[str, Any]] = []
    start = float(clock())
    handle = launcher(argv, None)
    actions.append(
        {
            "action": "CHILD_STARTED",
            "pid": handle.pid,
            "workers": workers,
            "at": _utc_now(),
        }
    )

    peak_rss = 0
    cpu_values: list[float] = []
    first_transfer: int | None = None
    first_t: float | None = None
    last_transfer: int | None = None
    last_t: float | None = None
    stop_reasons: list[str] = []
    status = "MEASURED"
    exit_code: int | None = None

    try:
        while True:
            now = float(clock())
            elapsed = now - start
            pid = handle.pid
            poll = handle.poll()
            # A child can exit between the previous sample and this iteration.
            # Do not ask macOS ``ps`` for a dead PID: it returns exit 1 and used
            # to turn a normal child-exit observation into a supervisor error.
            sample = provider.sample(pid if poll is None else None)
            samples.append(sample.as_dict())
            peak_rss = max(peak_rss, int(sample.process_rss_bytes))
            cpu_values.append(float(sample.cpu_percent))
            if first_transfer is None:
                first_transfer = int(sample.transfer_bytes)
                first_t = sample.monotonic_s
            last_transfer = int(sample.transfer_bytes)
            last_t = sample.monotonic_s

            breaches = evaluate_sample(
                sample,
                baseline_swap_used_bytes=baseline_swap_used_bytes,
                policy=policy,
            )
            if breaches:
                stop_reasons = breaches
                status = "BUDGET_BREACH"
                actions.append(
                    {
                        "action": "FAIL_CLOSED",
                        "reasons": list(breaches),
                        "workers": workers,
                        "at": _utc_now(),
                    }
                )
                exit_code = stop_child(
                    handle,
                    grace_seconds=config.terminate_grace_seconds,
                    sleeper=sleeper,
                    actions=actions,
                    reason=",".join(breaches),
                )
                break

            if poll is not None:
                exit_code = poll
                if poll == 0:
                    status = "CHILD_SUCCESS"
                    stop_reasons = ["CHILD_EXITED_ZERO"]
                else:
                    status = "ERROR_EXIT"
                    stop_reasons = ["ERROR_EXIT"]
                actions.append(
                    {
                        "action": "CHILD_EXIT_OBSERVED",
                        "exit_code": poll,
                        "workers": workers,
                        "at": _utc_now(),
                        "reason": stop_reasons[0],
                    }
                )
                break

            if elapsed >= config.measure_window_seconds:
                status = "MEASURED"
                stop_reasons = ["MEASURE_WINDOW_COMPLETE"]
                exit_code = stop_child(
                    handle,
                    grace_seconds=config.terminate_grace_seconds,
                    sleeper=sleeper,
                    actions=actions,
                    reason="MEASURE_WINDOW_COMPLETE",
                )
                break

            sleeper(config.sample_interval_seconds)
    except Exception as exc:
        status = "SUPERVISOR_ERROR"
        stop_reasons = [f"SUPERVISOR_ERROR:{type(exc).__name__}"]
        actions.append(
            {
                "action": "SUPERVISOR_ERROR",
                "error": str(exc),
                "at": _utc_now(),
            }
        )
        try:
            exit_code = stop_child(
                handle,
                grace_seconds=config.terminate_grace_seconds,
                sleeper=sleeper,
                actions=actions,
                reason="SUPERVISOR_ERROR",
            )
        except Exception:
            exit_code = handle.poll()
        raise

    duration = max(0.0, float(clock()) - start)
    throughput: float | None = None
    if (
        first_transfer is not None
        and last_transfer is not None
        and first_t is not None
        and last_t is not None
        and last_t > first_t
    ):
        throughput = max(0.0, (last_transfer - first_transfer) / (last_t - first_t))
    elif duration > 0 and first_transfer is not None and last_transfer is not None:
        throughput = max(0.0, (last_transfer - first_transfer) / duration)

    cpu_mean = (
        sum(cpu_values) / len(cpu_values) if cpu_values else None
    )
    return CandidateResult(
        workers=workers,
        status=status,
        reasons=stop_reasons,
        samples=samples,
        actions=actions,
        throughput_bytes_per_second=throughput,
        observed_cpu_percent_mean=cpu_mean,
        peak_rss_bytes=peak_rss if samples else None,
        exit_code=exit_code,
        argv=argv,
        duration_seconds=duration,
    )


def plan_dry_run(config: SupervisorConfig) -> dict[str, Any]:
    """Validate the ramp and emit a receipt body without launching children."""
    actions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for workers in config.worker_ramp:
        argv = build_full_argv(config, parallel_workers=workers)
        action = {
            "action": "PLAN_CANDIDATE",
            "workers": workers,
            "argv": argv,
            "reason": "DRY_RUN",
            "at": _utc_now(),
        }
        actions.append(action)
        candidates.append(
            {
                "workers": workers,
                "status": "PLANNED",
                "reasons": ["DRY_RUN"],
                "samples": [],
                "actions": [action],
                "throughput_bytes_per_second": None,
                "observed_cpu_percent_mean": None,
                "peak_rss_bytes": None,
                "exit_code": None,
                "argv": argv,
                "duration_seconds": 0.0,
            }
        )
    recommended = int(config.worker_ramp[0])
    return _receipt_body(
        config,
        status="DRY_RUN",
        recommended_workers=recommended,
        stop_reason="DRY_RUN_NO_CHILD",
        candidates=candidates,
        actions=actions,
        baseline_swap_used_bytes=None,
    )


def run_ramp(
    config: SupervisorConfig,
    *,
    provider: ResourceProvider | None = None,
    launcher: ChildLauncher | None = None,
    sleeper: Sleeper = time.sleep,
    clock: Clock = time.monotonic,
) -> dict[str, Any]:
    """Execute the bounded worker ramp and return a JSON-serializable receipt."""
    if config.dry_run:
        return plan_dry_run(config)

    active_provider = provider or HostResourceProvider(
        workspace_root=config.workspace_root,
        artifact_dir=config.artifact_dir,
    )
    active_launcher = launcher or default_launcher

    # Pre-flight sample with no child: capture swap baseline and disk floor.
    preflight = active_provider.sample(None)
    preflight_reasons = evaluate_sample(
        preflight,
        baseline_swap_used_bytes=preflight.swap_used_bytes,
        policy=config.policy(),
    )
    # RSS is zero without a child; only disk floor matters pre-flight.
    preflight_reasons = [r for r in preflight_reasons if r != "RSS_BUDGET_BREACH"]
    actions: list[dict[str, Any]] = [
        {
            "action": "PREFLIGHT_SAMPLE",
            "sample": preflight.as_dict(),
            "at": _utc_now(),
        }
    ]
    if preflight_reasons:
        actions.append(
            {
                "action": "PREFLIGHT_FAIL_CLOSED",
                "reasons": preflight_reasons,
                "at": _utc_now(),
            }
        )
        return _receipt_body(
            config,
            status="PREFLIGHT_FAILED",
            recommended_workers=None,
            stop_reason=",".join(preflight_reasons),
            candidates=[],
            actions=actions,
            baseline_swap_used_bytes=preflight.swap_used_bytes,
        )

    baseline_swap = int(preflight.swap_used_bytes)
    candidates: list[dict[str, Any]] = []
    recommended: int | None = None
    previous_throughput: float | None = None
    previous_workers: int | None = None
    stop_reason = "RAMP_COMPLETE"
    status = "COMPLETED"

    for workers in config.worker_ramp:
        result = run_candidate(
            config,
            workers=workers,
            provider=active_provider,
            launcher=active_launcher,
            baseline_swap_used_bytes=baseline_swap,
            sleeper=sleeper,
            clock=clock,
        )
        candidates.append(result.as_dict())
        actions.extend(result.actions)

        if result.status == "BUDGET_BREACH":
            status = "STOPPED"
            stop_reason = ",".join(result.reasons) or "BUDGET_BREACH"
            recommended = previous_workers
            actions.append(
                {
                    "action": "STOP_RAMP",
                    "reason": stop_reason,
                    "recommended_workers": recommended,
                    "at": _utc_now(),
                }
            )
            break

        if result.status == "ERROR_EXIT":
            status = "STOPPED"
            stop_reason = "ERROR_EXIT"
            recommended = previous_workers
            actions.append(
                {
                    "action": "STOP_RAMP",
                    "reason": stop_reason,
                    "exit_code": result.exit_code,
                    "recommended_workers": recommended,
                    "at": _utc_now(),
                }
            )
            break

        current_bps = result.throughput_bytes_per_second
        if current_bps is None:
            # No transfer signal yet; still accept the candidate as the best
            # known-safe worker count if it stayed under budget.
            recommended = workers
            previous_workers = workers
            previous_throughput = current_bps
            continue

        if not material_throughput_gain(
            previous_throughput,
            current_bps,
            min_gain=config.min_throughput_gain,
        ):
            status = "STOPPED"
            stop_reason = "NO_MATERIAL_THROUGHPUT_GAIN"
            recommended = previous_workers if previous_workers is not None else workers
            actions.append(
                {
                    "action": "STOP_RAMP",
                    "reason": stop_reason,
                    "previous_throughput_bytes_per_second": previous_throughput,
                    "current_throughput_bytes_per_second": current_bps,
                    "min_throughput_gain": config.min_throughput_gain,
                    "recommended_workers": recommended,
                    "at": _utc_now(),
                }
            )
            break

        recommended = workers
        previous_workers = workers
        previous_throughput = current_bps
        actions.append(
            {
                "action": "ACCEPT_CANDIDATE",
                "workers": workers,
                "throughput_bytes_per_second": current_bps,
                "observed_cpu_percent_mean": result.observed_cpu_percent_mean,
                "at": _utc_now(),
            }
        )
    else:
        # Exhausted ramp without a stop condition.
        if recommended is None and candidates:
            recommended = int(candidates[-1]["workers"])
        stop_reason = "RAMP_COMPLETE"
        status = "COMPLETED"
        actions.append(
            {
                "action": "RAMP_COMPLETE",
                "recommended_workers": recommended,
                "at": _utc_now(),
            }
        )

    return _receipt_body(
        config,
        status=status,
        recommended_workers=recommended,
        stop_reason=stop_reason,
        candidates=candidates,
        actions=actions,
        baseline_swap_used_bytes=baseline_swap,
    )


def _receipt_body(
    config: SupervisorConfig,
    *,
    status: str,
    recommended_workers: int | None,
    stop_reason: str,
    candidates: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    baseline_swap_used_bytes: int | None,
) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "created_at": _utc_now(),
        "recommended_workers": recommended_workers,
        "stop_reason": stop_reason,
        "policy": {
            "rss_budget_bytes": config.rss_budget_bytes,
            "protected_floor_bytes": config.protected_floor_bytes,
            "allow_swap_growth": False,
            "min_throughput_gain": config.min_throughput_gain,
            "max_parallel_workers_ceiling": MAX_PARALLEL_WORKERS,
            "worker_ramp": list(config.worker_ramp),
            "sample_interval_seconds": config.sample_interval_seconds,
            "measure_window_seconds": config.measure_window_seconds,
            "rss_enforcement": "sample_and_terminate_not_ulimit",
            "zero_cpu_idle_promised": False,
            "invocation_floor_example_bytes": EXAMPLE_INVOCATION_FLOOR_BYTES,
            "invocation_floor_example_note": (
                "25 GiB is an invocation policy example only; the caller "
                "protected_floor_bytes value is the hard floor used here."
            ),
        },
        "target": {
            "artifact_dir": str(config.artifact_dir),
            "workspace_root": str(config.workspace_root),
            "xet_root": str(config.xet_root),
            "range_bytes": config.range_bytes,
            "gravity_cli": str(config.gravity_cli),
            "python_executable": config.python_executable,
            "launch_path": "tools/condense/deepseek_v4_gravity.py build-full",
            "downloads_model_objects_directly": False,
            "changes_source_range_limits": False,
        },
        "baseline_swap_used_bytes": baseline_swap_used_bytes,
        "candidates": candidates,
        "actions": actions,
        "notes": [
            "Observed CPU utilization is reported per candidate; this supervisor "
            "does not promise or attempt zero CPU idle.",
            "Each candidate is launched through the existing build-full CLI so "
            "the sealed journal permits resumption after a measured stop.",
        ],
    }


def write_receipt(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    """Atomically write the supervisor receipt as pretty JSON."""
    target = _absolute(path, "receipt_out")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(encoded, encoding="utf-8")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resource-budget supervisor for DeepSeek-V4 build-full worker ramps. "
            "Enforces a 5 GiB child RSS budget by sampling (not ulimit), preserves "
            "the caller disk floor, requires no swap growth, and recommends a "
            "worker count from an explicit bounded ramp."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Absolute Gravity artifact directory (journal resumes here).",
    )
    parser.add_argument(
        "--workspace-root",
        required=True,
        help="Absolute workspace root used for the disk-floor sample.",
    )
    parser.add_argument(
        "--xet-root",
        required=True,
        help="Absolute empty Xet retention root passed through to build-full.",
    )
    parser.add_argument(
        "--protected-floor-bytes",
        type=int,
        default=DEFAULT_PROTECTED_FLOOR_BYTES,
        help=(
            "Caller hard disk floor in bytes (default 15 GiB). "
            f"Example invocation policy only: {EXAMPLE_INVOCATION_FLOOR_BYTES} "
            "(25 GiB); that example is not substituted for this flag."
        ),
    )
    parser.add_argument(
        "--range-bytes",
        type=int,
        default=None,
        help=(
            "Optional pass-through of build-full --range-bytes. "
            "Omitted means the gravity CLI default; this supervisor never "
            "changes source-range limits on its own."
        ),
    )
    parser.add_argument(
        "--worker-ramp",
        default=",".join(str(value) for value in DEFAULT_WORKER_RAMP),
        help=(
            f"Comma-separated ascending worker counts "
            f"(default {','.join(str(v) for v in DEFAULT_WORKER_RAMP)}; "
            f"each entry must be 1..{MAX_PARALLEL_WORKERS})."
        ),
    )
    parser.add_argument(
        "--rss-budget-bytes",
        type=int,
        default=RSS_BUDGET_BYTES,
        help=f"Child process RSS budget in bytes (default {RSS_BUDGET_BYTES} = 5 GiB).",
    )
    parser.add_argument(
        "--sample-interval-seconds",
        type=float,
        default=DEFAULT_SAMPLE_INTERVAL_SECONDS,
        help="Bounded sampling interval for CPU/RSS/disk/transfer (default 2s).",
    )
    parser.add_argument(
        "--measure-window-seconds",
        type=float,
        default=DEFAULT_MEASURE_WINDOW_SECONDS,
        help="Stable measurement window per candidate before stop (default 30s).",
    )
    parser.add_argument(
        "--min-throughput-gain",
        type=float,
        default=DEFAULT_MIN_THROUGHPUT_GAIN,
        help=(
            "Minimum fractional throughput gain required to accept the next "
            "ramp step (default 0.05 = 5%%)."
        ),
    )
    parser.add_argument(
        "--receipt-out",
        required=True,
        help="Absolute path for the JSON supervisor receipt.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch build-full (default: current).",
    )
    parser.add_argument(
        "--gravity-cli",
        default=str(BUILD_FULL_CLI),
        help="Absolute path to deepseek_v4_gravity.py (default: repo tools path).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the ramp and write a receipt without spawning a child.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = SupervisorConfig(
            artifact_dir=args.artifact_dir,
            workspace_root=args.workspace_root,
            xet_root=args.xet_root,
            protected_floor_bytes=args.protected_floor_bytes,
            range_bytes=args.range_bytes,
            worker_ramp=normalize_worker_ramp(args.worker_ramp),
            rss_budget_bytes=args.rss_budget_bytes,
            sample_interval_seconds=args.sample_interval_seconds,
            measure_window_seconds=args.measure_window_seconds,
            min_throughput_gain=args.min_throughput_gain,
            python_executable=args.python,
            gravity_cli=args.gravity_cli,
            dry_run=bool(args.dry_run),
        )
        receipt = run_ramp(config)
        path = write_receipt(args.receipt_out, receipt)
    except ResourceSupervisorError as exc:
        print(f"resource-supervisor error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema": RECEIPT_SCHEMA,
                "status": receipt["status"],
                "recommended_workers": receipt["recommended_workers"],
                "stop_reason": receipt["stop_reason"],
                "receipt_out": str(path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if receipt["status"] in {"PREFLIGHT_FAILED"}:
        return 1
    if receipt["status"] == "STOPPED" and receipt["recommended_workers"] is None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
