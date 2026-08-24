"""Resource classes, concurrency limits, and the crash-safe mutation lock.

GPU_DECODE is bounded by ACTIVE_DECODE_LIMIT, which is a different number
from how many runtimes are resident. An idle CPU class is not a reason to
invent work.

Backend health, retryability classification, and the circuit breaker live
here too. They are deliberately not folded into ``can_admit``: that function
is resource-class occupancy. A scheduler consults health separately (see
``BackendHealth.allows_new_assignments``).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

from .persist import atomic_write_text as _atomic_write_text


class ResourceClass(str, Enum):
    GPU_DECODE = "GPU_DECODE"
    GPU_EXCLUSIVE = "GPU_EXCLUSIVE"
    GPU_DIRTY_OK = "GPU_DIRTY_OK"
    CPU_HEAVY = "CPU_HEAVY"
    COMPILE = "COMPILE"
    TEST = "TEST"
    TEST_AUTHORING = "TEST_AUTHORING"
    STATIC_ANALYSIS = "STATIC_ANALYSIS"
    MEMORY_HEAVY = "MEMORY_HEAVY"
    IO_HEAVY = "IO_HEAVY"
    TOOL_WAIT = "TOOL_WAIT"
    LIGHT_CONTROL = "LIGHT_CONTROL"
    MUTATION = "MUTATION"
    GROK = "GROK"


MUTATION_LOCK_FILENAME = "mutation.lock"
DEFAULT_DECODE_LIMIT = 1
TOOL_AND_CONTROL_LIMIT = 128
MEMORY_HEAVY_LIMIT = 2


def _cpu_count() -> int:
    return os.cpu_count() or 1


def _default_repo_root() -> Path:
    from .paths import find_repo_root

    return find_repo_root(Path(__file__))


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def resolve_active_decode_limit(
    repo_root: Optional[Union[str, Path]] = None,
) -> Tuple[int, str]:
    """Return (limit, source). Thin adapter over ``resolve_runtime_limits``.

    The previous body re-read the same genome files without freshness or
    machine-identity checks, so the scheduler could admit a STALE prior
    that RuntimePool had already refused. One authority: ``hcli.machine``.

    Verified caller: ``ResourceLimits.resolve`` (and tests that import this
    name). Do not grow a second file-walker here.
    """
    from .machine import resolve_runtime_limits

    kwargs: Dict[str, Any] = {}
    if repo_root is not None:
        kwargs["repo_root"] = repo_root
        kwargs["start_dir"] = repo_root
    resolved = resolve_runtime_limits(**kwargs)
    return resolved.active_decode_limit, resolved.active_source


@dataclass
class ResourceLimits:
    gpu_decode: int
    gpu_decode_source: str
    gpu_exclusive: int = 1
    mutation: int = 1
    cpu_heavy: int = 1
    compile: int = 1
    test: int = 1
    test_authoring: int = 1
    static_analysis: int = 1
    memory_heavy: int = MEMORY_HEAVY_LIMIT
    io_heavy: int = 8
    tool_wait: int = TOOL_AND_CONTROL_LIMIT
    light_control: int = TOOL_AND_CONTROL_LIMIT
    grok: int = 2
    grok_source: str = "fallback"

    @classmethod
    def resolve(
        cls, repo_root: Optional[Union[str, Path]] = None
    ) -> "ResourceLimits":
        decode, source = resolve_active_decode_limit(repo_root=repo_root)
        ncpu = _cpu_count()
        grok_n, grok_src = 2, "fallback"
        try:
            from .max_policy import resolve_grok_admitted

            grok_n, grok_src = resolve_grok_admitted(repo_root)
        except Exception:
            pass
        return cls(
            gpu_decode=decode,
            gpu_decode_source=source,
            gpu_exclusive=1,
            mutation=1,
            cpu_heavy=ncpu,
            compile=ncpu,
            test=ncpu,
            test_authoring=ncpu,
            static_analysis=ncpu,
            memory_heavy=MEMORY_HEAVY_LIMIT,
            io_heavy=max(8, ncpu),
            tool_wait=TOOL_AND_CONTROL_LIMIT,
            light_control=TOOL_AND_CONTROL_LIMIT,
            grok=grok_n,
            grok_source=grok_src,
        )

    def limit_for(self, resource_class: str) -> int:
        rc = normalize_resource_class(resource_class)
        mapping = {
            ResourceClass.GPU_DECODE.value: self.gpu_decode,
            ResourceClass.GPU_DIRTY_OK.value: self.gpu_decode,
            ResourceClass.GPU_EXCLUSIVE.value: self.gpu_exclusive,
            ResourceClass.MUTATION.value: self.mutation,
            ResourceClass.CPU_HEAVY.value: self.cpu_heavy,
            ResourceClass.COMPILE.value: self.compile,
            ResourceClass.TEST.value: self.test,
            ResourceClass.TEST_AUTHORING.value: self.test_authoring,
            ResourceClass.STATIC_ANALYSIS.value: self.static_analysis,
            ResourceClass.MEMORY_HEAVY.value: self.memory_heavy,
            ResourceClass.IO_HEAVY.value: self.io_heavy,
            ResourceClass.TOOL_WAIT.value: self.tool_wait,
            ResourceClass.LIGHT_CONTROL.value: self.light_control,
            ResourceClass.GROK.value: self.grok,
        }
        return mapping.get(rc, self.light_control)


def normalize_resource_class(value: Any) -> str:
    if value is None:
        return ResourceClass.LIGHT_CONTROL.value
    if isinstance(value, ResourceClass):
        return value.value
    text = str(value).strip()
    try:
        return ResourceClass(text).value
    except ValueError:
        return ResourceClass.LIGHT_CONTROL.value


def occupancy_of(units: Iterable[Any]) -> Counter:
    counts: Counter = Counter()
    for wu in units:
        if getattr(wu, "status", None) == "running":
            counts[normalize_resource_class(getattr(wu, "resource_class", None))] += 1
    return counts


def can_admit(resource_class: str, occupied: Counter, limits: ResourceLimits) -> bool:
    """True iff ``resource_class`` has a free slot under ``limits``.

    GPU_EXCLUSIVE takes the whole GPU: while one runs, GPU_DECODE is refused,
    and GPU_EXCLUSIVE will not start while any GPU_DECODE is running.
    MUTATION is hard-capped at 1.
    """
    rc = normalize_resource_class(resource_class)
    decode_busy = (
        occupied[ResourceClass.GPU_DECODE.value]
        + occupied[ResourceClass.GPU_DIRTY_OK.value]
    )
    if rc in (
        ResourceClass.GPU_DECODE.value,
        ResourceClass.GPU_DIRTY_OK.value,
    ):
        if occupied[ResourceClass.GPU_EXCLUSIVE.value] > 0:
            return False
        return decode_busy < limits.gpu_decode
    if rc == ResourceClass.GPU_EXCLUSIVE.value:
        if occupied[ResourceClass.GPU_EXCLUSIVE.value] > 0:
            return False
        if decode_busy > 0:
            return False
        return True
    return occupied[rc] < limits.limit_for(rc)


def next_class_slot(
    resource_class: str,
    used_slots: Dict[str, set],
    limits: ResourceLimits,
) -> Optional[int]:
    rc = normalize_resource_class(resource_class)
    limit = limits.limit_for(rc)
    taken = used_slots.get(rc, set())
    for idx in range(limit):
        if idx not in taken:
            return idx
    return None


def pid_is_alive(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _start_token_libproc(pid: int) -> Optional[str]:
    """Darwin: process start time via libproc (does not spawn ``ps``)."""
    try:
        import ctypes
        import ctypes.util
    except ImportError:
        return None
    try:
        libname = ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib"
        lib = ctypes.CDLL(libname)
    except OSError:
        return None

    class proc_bsdinfo(ctypes.Structure):
        _fields_ = [
            ("pbi_flags", ctypes.c_uint32),
            ("pbi_status", ctypes.c_uint32),
            ("pbi_xstatus", ctypes.c_uint32),
            ("pbi_pid", ctypes.c_uint32),
            ("pbi_ppid", ctypes.c_uint32),
            ("pbi_uid", ctypes.c_uint32),
            ("pbi_gid", ctypes.c_uint32),
            ("pbi_ruid", ctypes.c_uint32),
            ("pbi_rgid", ctypes.c_uint32),
            ("pbi_svuid", ctypes.c_uint32),
            ("pbi_svgid", ctypes.c_uint32),
            ("reserved2", ctypes.c_uint32),
            ("pbi_comm", ctypes.c_char * 16),
            ("pbi_name", ctypes.c_char * 32),
            ("pbi_nfiles", ctypes.c_uint32),
            ("pbi_pgid", ctypes.c_uint32),
            ("pbi_pjobc", ctypes.c_uint32),
            ("e_tdev", ctypes.c_uint32),
            ("e_tpgid", ctypes.c_uint32),
            ("pbi_nice", ctypes.c_int32),
            ("pbi_start_tvsec", ctypes.c_uint64),
            ("pbi_start_tvusec", ctypes.c_uint64),
        ]

    PROC_PIDTBSDINFO = 3
    info = proc_bsdinfo()
    try:
        lib.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.proc_pidinfo.restype = ctypes.c_int
        got = lib.proc_pidinfo(
            int(pid),
            PROC_PIDTBSDINFO,
            0,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
    except Exception:
        return None
    if got <= 0:
        return None
    return f"{int(info.pbi_start_tvsec)}.{int(info.pbi_start_tvusec):06d}"


def _start_token_procfs(pid: int) -> Optional[str]:
    """Linux: starttime field of /proc/<pid>/stat."""
    path = Path(f"/proc/{int(pid)}/stat")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    close = text.rfind(")")
    if close == -1:
        return None
    fields = text[close + 1 :].split()
    # Field 22 in the full stat record is starttime; after comm that is index 19.
    if len(fields) < 20:
        return None
    return fields[19]


def _start_token_ps(pid: int) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    token = (proc.stdout or "").strip()
    if proc.returncode != 0 or not token:
        return None
    return token


def process_start_token(pid: int) -> Optional[str]:
    """Stable identifier of a specific process incarnation (pid + start time).

    A recycled pid has a different start token, so matching pid AND token is
    required before treating a lock holder as still alive.

    Prefers libproc / procfs over spawning ``ps``, which is frequently
    blocked in sandboxed agent runs.
    """
    if not isinstance(pid, int) or pid <= 0:
        return None
    for reader in (_start_token_libproc, _start_token_procfs, _start_token_ps):
        token = reader(pid)
        if token:
            return token
    return None


_EMPTY_LOCK_GRACE_S = 5.0


class MutationLock:
    """Exclusive, crash-safe single-writer lock for MUTATION work units.

    The lock record stores holder pid and process start time. A later
    scheduler may break the lock only when the holder is provably gone:
    the pid is dead, or the pid is alive but its start token does not
    match (recycled pid).
    """

    def __init__(self, workspace: Optional[Union[str, Path]] = None) -> None:
        self.workspace = Path(workspace) if workspace is not None else None
        self.path = (
            self.workspace / ".hcli" / MUTATION_LOCK_FILENAME
            if self.workspace is not None
            else None
        )
        self._memory: Optional[Dict[str, Any]] = None

    def read(self) -> Optional[Dict[str, Any]]:
        if self.path is None:
            return dict(self._memory) if self._memory else None
        if not self.path.is_file():
            return None
        data = _load_json(self.path)
        return data

    def write(self, record: Dict[str, Any]) -> None:
        if self.path is None:
            self._memory = dict(record)
            return
        _atomic_write_text(self.path, json.dumps(record, indent=2, sort_keys=True))

    def clear(self) -> None:
        if self.path is None:
            self._memory = None
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def holder_is_live(self, record: Optional[Dict[str, Any]] = None) -> bool:
        rec = record if record is not None else self.read()
        if not rec:
            return False
        try:
            pid = int(rec.get("pid"))
        except (TypeError, ValueError):
            return False
        if not pid_is_alive(pid):
            return False
        recorded = rec.get("start_time")
        if not recorded:
            # Pid is alive and we have no start token to refute it.
            return True
        live = process_start_token(pid)
        if live is None:
            # Could not read start time; do not break a lock that might be live.
            return True
        return str(recorded) == str(live)

    def try_break_stale(self) -> bool:
        """Break a lock whose holder is gone. True iff the lock is now free.

        An unreadable or empty lock file is treated as stale: O_EXCL create
        can leave a zero-byte file if the holder crashed between open and
        write, and that file must not block every later acquire.
        """
        if self.path is None:
            rec = self.read()
            if rec is None:
                return True
            if self.holder_is_live(rec):
                return False
            self.clear()
            return True
        if not self.path.is_file():
            return True
        rec = self.read()
        if rec is None:
            # Unreadable or empty. acquire() publishes a complete file via
            # os.link, so this cannot come from a live acquirer mid-write.
            # It is either corruption or a pre-upgrade lock file; break it only
            # once it is demonstrably not being written right now.
            try:
                age = time.time() - self.path.stat().st_mtime
            except OSError:
                return False
            if age < _EMPTY_LOCK_GRACE_S:
                return False
            self.clear()
            return True
        if self.holder_is_live(rec):
            return False
        self.clear()
        return True

    def acquire(self, unit_id: str) -> bool:
        """Take the lock with a real exclusive create, not check-then-replace.

        The previous implementation wrote the lock record via ``os.replace``,
        which two processes could both complete. Measured: 80/80 undelayed
        process pairs both acquired. ``O_CREAT|O_EXCL`` makes the filesystem
        the mutex: exactly one open succeeds.

        A live holder — including this process, if a lock was planted for
        it — is exclusive. Re-acquire would let a later scheduler steal a
        still-held lock just because it shares a pid.
        """
        record = {
            "pid": os.getpid(),
            "start_time": process_start_token(os.getpid()),
            "acquired_at": time.time(),
            "unit_id": unit_id,
        }
        if self.path is None:
            if not self.try_break_stale():
                return False
            self.write(record)
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
        return self._link_exclusive(payload)

    def _link_exclusive(self, payload: str) -> bool:
        """Publish a COMPLETE lock file atomically, or lose the race.

        O_EXCL alone is not enough here. `os.open(O_CREAT|O_EXCL)` succeeds and
        THEN the payload is written, so a rival that arrives inside that window
        sees a zero-byte lock file. Since an empty file is indistinguishable
        from a holder that crashed between open and write, stale-breaking
        clears it and both processes end up holding the lock. Measured on this
        box: 130 of 200 undelayed process pairs both acquired.

        `os.link` closes the window: the temp file already contains the whole
        record, and the link either publishes it atomically or fails with
        FileExistsError. The lock file is therefore never observable in a
        partial state, so "empty means stale" is no longer needed and no longer
        reintroduces the race.
        """
        assert self.path is not None
        tmp = self.path.with_name(
            f"{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            for attempt in (0, 1):
                try:
                    os.link(tmp, self.path)
                    return True
                except FileExistsError:
                    # Someone holds it. Break it only if the holder is provably
                    # gone, and only try once more.
                    if attempt == 1 or not self.try_break_stale():
                        return False
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
        return False

    def release(self, unit_id: Optional[str] = None) -> None:
        rec = self.read()
        if rec is None:
            return
        if unit_id is not None and rec.get("unit_id") not in (None, unit_id):
            return
        if rec.get("pid") not in (None, os.getpid()):
            return
        self.clear()


# ---------------------------------------------------------------------------
# Backend health, retryability, circuit breaker
# ---------------------------------------------------------------------------
#
# Backends a mixed scheduler can route to. Names match preferred_backend
# on WorkUnit (landed after this snapshot; default "cpu" in goal.py).
KNOWN_BACKENDS: Tuple[str, ...] = ("qwen", "grok", "cpu")

HEALTH_FILENAME = "backend_health.json"
HEALTH_VERSION = 1

STATE_HEALTHY = "healthy"
STATE_DEGRADED = "degraded"
STATE_CIRCUIT_OPEN = "circuit_open"

# After this many *consecutive backend* failures the breaker opens.
CIRCUIT_FAILURE_THRESHOLD = 3
# New assignments are refused only while this window after the last
# failure has not elapsed. The breaker then reopens (degraded, not
# permanently open).
CIRCUIT_COOLING_SECONDS = 30.0
# A persisted record older than this is a prior observation, not present
# truth. ``allows_new_assignments`` will not refuse work on a stale open.
HEALTH_STALE_AFTER_SECONDS = 3600.0

TRANSIENT_BACKEND = "TRANSIENT_BACKEND"
VERIFIER_FAILURE = "VERIFIER_FAILURE"
DETERMINISTIC_IMPLEMENTATION = "DETERMINISTIC_IMPLEMENTATION"
INVALID_OUTPUT = "INVALID_OUTPUT"
RATE_LIMIT = "RATE_LIMIT"
UNAVAILABLE_DEPENDENCY = "UNAVAILABLE_DEPENDENCY"
IMPOSSIBLE_CONTRACT = "IMPOSSIBLE_CONTRACT"

FAILURE_KINDS: Tuple[str, ...] = (
    TRANSIENT_BACKEND,
    VERIFIER_FAILURE,
    DETERMINISTIC_IMPLEMENTATION,
    INVALID_OUTPUT,
    RATE_LIMIT,
    UNAVAILABLE_DEPENDENCY,
    IMPOSSIBLE_CONTRACT,
)

# Audit classification: these names, when observed, are not worth retrying.
# The unit's retry budget must not be charged (see counts_toward_retry_budget).
# VACUOUS_COMMAND / EMPTY_COMMAND / ContextPreflightError are matched by
# name; the modules that raise them (executors.py, context_budget.py, the
# post-HEAD verifier_pipeline.command_is_admissible) are not in this
# snapshot and are not reimplemented here.
NON_RETRYABLE = {
    "GrokNotAvailable",
    "GrokContractError",
    "VACUOUS_COMMAND",
    "EMPTY_COMMAND",
    "NO_OP_MUTATION",
    "ContextPreflightError",
}

_NON_RETRYABLE_KIND = {
    "GrokNotAvailable": UNAVAILABLE_DEPENDENCY,
    "GrokContractError": IMPOSSIBLE_CONTRACT,
    "VACUOUS_COMMAND": IMPOSSIBLE_CONTRACT,
    "EMPTY_COMMAND": IMPOSSIBLE_CONTRACT,
    "NO_OP_MUTATION": DETERMINISTIC_IMPLEMENTATION,
    "ContextPreflightError": IMPOSSIBLE_CONTRACT,
}

# Failures that describe the backend itself. Verifier / contract /
# no-op-mutation failures do not trip the breaker.
_CIRCUIT_KINDS = frozenset(
    {
        TRANSIENT_BACKEND,
        RATE_LIMIT,
        INVALID_OUTPUT,
        UNAVAILABLE_DEPENDENCY,
    }
)

_HTTP_CODE_RE = re.compile(r"\bHTTP\s+(\d{3})\b", re.IGNORECASE)
_RATE_LIMIT_RE = re.compile(r"\brate[\s_-]?limit", re.IGNORECASE)


@dataclass(frozen=True)
class FailureClassification:
    """Result of classify_failure.

    ``kind`` is one of FAILURE_KINDS. ``retryable`` is False exactly when
    an observed token is in NON_RETRYABLE (or the kind is one that only
    those tokens produce). ``observed`` is the token that decided it,
    or empty if the fallback path was used.
    """

    kind: str
    retryable: bool
    observed: str = ""


def normalize_backend(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text not in KNOWN_BACKENDS:
        raise ValueError(
            f"unknown backend {value!r}; expected one of {KNOWN_BACKENDS}"
        )
    return text


def _exception_name(exc: BaseException) -> str:
    return type(exc).__name__


def _tokens_from_context(context: Any) -> Tuple[list, list]:
    """Return (named tokens, blob strings) extracted from a failure context.

    Named tokens are exception class names, ``reason``/``error`` fields, and
    members of NON_RETRYABLE / FAILURE_KINDS. Blobs are free-text messages
    searched for HTTP codes and phrases.
    """
    names: list = []
    blobs: list = []

    def add_name(value: Any) -> None:
        if value is None:
            return
        text = str(value).strip()
        if text:
            names.append(text)

    def add_blob(value: Any) -> None:
        if value is None:
            return
        text = str(value)
        if text:
            blobs.append(text)

    if context is None:
        return names, blobs

    if isinstance(context, BaseException):
        add_name(_exception_name(context))
        add_blob(str(context))
        add_name(getattr(context, "reason", None))
        code = getattr(context, "code", None)
        if isinstance(code, int) and 100 <= code <= 599:
            add_blob(f"HTTP {code}")
        return names, blobs

    if isinstance(context, str):
        stripped = context.strip()
        if stripped in NON_RETRYABLE or stripped in FAILURE_KINDS:
            add_name(stripped)
        add_blob(context)
        return names, blobs

    if not isinstance(context, dict):
        add_blob(context)
        return names, blobs

    for key in ("reason", "error", "exception", "type", "name", "kind", "code"):
        val = context.get(key)
        if val is None:
            continue
        if isinstance(val, BaseException):
            nested_names, nested_blobs = _tokens_from_context(val)
            names.extend(nested_names)
            blobs.extend(nested_blobs)
            continue
        if isinstance(val, type) and issubclass(val, BaseException):
            add_name(val.__name__)
            continue
        text = str(val).strip()
        if not text:
            continue
        add_name(text)
        add_blob(text)

    for key in ("message", "detail", "stderr", "stdout", "output"):
        add_blob(context.get(key))

    status = context.get("http_status")
    if status is None:
        status = context.get("status_code")
    if isinstance(status, int) and 100 <= status <= 599:
        add_blob(f"HTTP {status}")
        add_name(f"HTTP {status}")

    return names, blobs


def _http_codes(blobs: Iterable[str]) -> list:
    codes = []
    for blob in blobs:
        for match in _HTTP_CODE_RE.finditer(blob):
            try:
                codes.append(int(match.group(1)))
            except ValueError:
                continue
    return codes


def classify_failure(context: Any = None) -> FailureClassification:
    """Classify an observed failure into one of FAILURE_KINDS.

    Input is whatever the code can actually put in a WorkUnit
    ``failure_context`` today: a dict (``reason`` / ``error`` / HTTP
    status / message), an exception instance, or a string. This module
    does not import grok_bridge, engine, executors, or context_budget —
    classification is by observed name and message, so names landed
    after this snapshot still classify when they appear in context.

    Currently unreachable kinds: none. RATE_LIMIT has no dedicated
    exception class; it is the HTTP 429 branch of the llama-server
    HTTPError path (backends.py LlamaServerBackend.complete,
    engine.py Engine._call_model). VACUOUS_COMMAND, EMPTY_COMMAND and
    ContextPreflightError are not instantiable in this snapshot but
    are classified when those names appear.
    """
    names, blobs = _tokens_from_context(context)
    named_set = set(names)
    joined = "\n".join(blobs)
    joined_upper = joined.upper()

    for token in names:
        if token in _NON_RETRYABLE_KIND:
            kind = _NON_RETRYABLE_KIND[token]
            return FailureClassification(
                kind=kind, retryable=False, observed=token
            )
        # "NO_OP_MUTATION: identical bytes" etc.
        for marker, kind in _NON_RETRYABLE_KIND.items():
            if token == marker or token.startswith(marker + ":"):
                return FailureClassification(
                    kind=kind, retryable=False, observed=marker
                )

    for blob in blobs:
        for marker, kind in _NON_RETRYABLE_KIND.items():
            if marker in blob:
                return FailureClassification(
                    kind=kind, retryable=False, observed=marker
                )

    codes = _http_codes(blobs)
    if 429 in codes or _RATE_LIMIT_RE.search(joined):
        observed = "HTTP 429" if 429 in codes else "rate_limit"
        return FailureClassification(
            kind=RATE_LIMIT, retryable=True, observed=observed
        )

    if (
        "INVALID JSON" in joined_upper
        or "JSONDECODEERROR" in named_set
        or "JSONDecodeError" in named_set
        or "INVALID_OUTPUT" in named_set
    ):
        observed = "invalid JSON"
        if "INVALID_OUTPUT" in named_set:
            observed = "INVALID_OUTPUT"
        return FailureClassification(
            kind=INVALID_OUTPUT, retryable=True, observed=observed
        )

    if named_set & {
        "TEST_FAILED",
        "TEST_ERROR",
        "NO_EVIDENCE",
        "PlanError",
        "VERIFIER_FAILURE",
    } or any(
        token.startswith("TEST_ERROR:") for token in names
    ):
        observed = next(
            (
                t
                for t in names
                if t in {
                    "TEST_FAILED",
                    "TEST_ERROR",
                    "NO_EVIDENCE",
                    "PlanError",
                    "VERIFIER_FAILURE",
                }
                or t.startswith("TEST_ERROR:")
            ),
            "TEST_FAILED",
        )
        return FailureClassification(
            kind=VERIFIER_FAILURE, retryable=True, observed=observed
        )

    if (
        "PYTEST_UNAVAILABLE" in named_set
        or "GrokNotAvailable" in named_set
        or "UNAVAILABLE_DEPENDENCY" in named_set
    ):
        observed = "PYTEST_UNAVAILABLE" if "PYTEST_UNAVAILABLE" in named_set else (
            "GrokNotAvailable" if "GrokNotAvailable" in named_set else "UNAVAILABLE_DEPENDENCY"
        )
        retryable = observed not in NON_RETRYABLE
        return FailureClassification(
            kind=UNAVAILABLE_DEPENDENCY, retryable=retryable, observed=observed
        )

    server_codes = [c for c in codes if 500 <= c <= 599]
    transient_markers = (
        "GrokRunError",
        "TimeoutExpired",
        "TimeoutError",
        "timed out",
        "request failed",
        "not ready",
        "ConnectionRefusedError",
        "URLError",
        "TRANSIENT_BACKEND",
    )
    if server_codes or any(
        marker in named_set or marker.lower() in joined.lower()
        for marker in transient_markers
    ):
        if server_codes:
            observed = f"HTTP {server_codes[0]}"
        else:
            observed = next(
                (
                    m
                    for m in transient_markers
                    if m in named_set or m.lower() in joined.lower()
                ),
                TRANSIENT_BACKEND,
            )
        return FailureClassification(
            kind=TRANSIENT_BACKEND, retryable=True, observed=str(observed)
        )

    # Honest fallback: nothing matched. Treat as a retryable blip rather
    # than inventing a kind the caller did not observe.
    return FailureClassification(
        kind=TRANSIENT_BACKEND, retryable=True, observed=""
    )


def counts_toward_retry_budget(context: Any = None) -> bool:
    """True iff this failure should consume a WorkUnit retry attempt.

    Non-retryable names (NON_RETRYABLE) return False. The scheduler
    should consult this in ``Scheduler.fail`` / the repair emitter
    *before* incrementing ``WorkUnit.attempts`` or emitting a repair
    that would burn DEFAULT_RETRY_BUDGET. This module does not touch
    workunit.py; attempts live there.
    """
    return classify_failure(context).retryable


def _empty_backend_record() -> Dict[str, Any]:
    return {
        "consecutive_failures": 0,
        "last_failure_time": None,
        "last_success_time": None,
    }


def _coerce_record(raw: Any) -> Dict[str, Any]:
    rec = _empty_backend_record()
    if not isinstance(raw, dict):
        return rec
    try:
        rec["consecutive_failures"] = max(0, int(raw.get("consecutive_failures") or 0))
    except (TypeError, ValueError):
        rec["consecutive_failures"] = 0
    for key in ("last_failure_time", "last_success_time"):
        val = raw.get(key)
        if val is None or val == "":
            rec[key] = None
            continue
        try:
            rec[key] = float(val)
        except (TypeError, ValueError):
            rec[key] = None
    return rec


class BackendHealth:
    """Durable per-backend health with a bounded circuit breaker.

    Persists under ``<workspace>/.hcli/backend_health.json`` via the same
    temp+fsync+replace path as the mutation lock record. Survives process
    restart. Derived state is computed at read time from the observations
    and the clock; a persisted ``state`` field is never authority.

    Clock injection: pass ``clock=`` a zero-arg callable returning epoch
    seconds. Tests advance that clock; production uses ``time.time``.

    Scheduler consult site (do not wire this here)
    ----------------------------------------------
    ``allows_new_assignments(backend)`` is the accessor. Call it from
    ``assign_ready`` in workunit.py, after ``can_admit`` succeeds for the
    resource class and before ``transition_status(..., "running")``, using
    the unit's ``preferred_backend`` (default ``"cpu"``). If it returns
    False, skip the unit (leave it ready) and try the next — the same
    skip pattern ``can_admit`` already uses when a class is full.

    Do not fold this into ``can_admit``. Occupancy of GPU_DECODE is not
    "grok is down". Mixing them would make a GPU slot look full because
    a different backend's breaker is open.

    The mixed-backend dispatcher (mission / scheduler, reserved) is the
    other consult site: before handing a unit to the qwen, grok, or cpu
    executor, call the same accessor. Healthy backends keep running.
    """

    def __init__(
        self,
        workspace: Optional[Union[str, Path]] = None,
        *,
        clock: Optional[Callable[[], float]] = None,
        failure_threshold: int = CIRCUIT_FAILURE_THRESHOLD,
        cooling_seconds: float = CIRCUIT_COOLING_SECONDS,
        stale_after_seconds: float = HEALTH_STALE_AFTER_SECONDS,
    ) -> None:
        self.workspace = Path(workspace) if workspace is not None else None
        self.path = (
            self.workspace / ".hcli" / HEALTH_FILENAME
            if self.workspace is not None
            else None
        )
        self._clock = clock if clock is not None else time.time
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooling_seconds = max(0.0, float(cooling_seconds))
        self.stale_after_seconds = max(0.0, float(stale_after_seconds))
        self._memory: Dict[str, Any] = {
            "version": HEALTH_VERSION,
            "updated_at": None,
            "backends": {name: _empty_backend_record() for name in KNOWN_BACKENDS},
        }
        if self.path is not None:
            loaded = self._read_disk()
            if loaded is not None:
                self._memory = loaded

    def _now(self, now: Optional[float] = None) -> float:
        if now is not None:
            return float(now)
        return float(self._clock())

    def _read_disk(self) -> Optional[Dict[str, Any]]:
        if self.path is None or not self.path.is_file():
            return None
        data = _load_json(self.path)
        if not data:
            return None
        return self._normalize_document(data)

    def _normalize_document(self, data: Dict[str, Any]) -> Dict[str, Any]:
        backends_raw = data.get("backends")
        if not isinstance(backends_raw, dict):
            backends_raw = {}
        backends = {}
        for name in KNOWN_BACKENDS:
            backends[name] = _coerce_record(backends_raw.get(name))
        updated_at = data.get("updated_at")
        try:
            updated_at_f = float(updated_at) if updated_at is not None else None
        except (TypeError, ValueError):
            updated_at_f = None
        return {
            "version": HEALTH_VERSION,
            "updated_at": updated_at_f,
            "backends": backends,
        }

    def _persist(self) -> None:
        document = {
            "version": HEALTH_VERSION,
            "updated_at": self._memory.get("updated_at"),
            "backends": {
                name: dict(self._memory["backends"][name])
                for name in KNOWN_BACKENDS
            },
        }
        if self.path is None:
            return
        _atomic_write_text(
            self.path, json.dumps(document, indent=2, sort_keys=True) + "\n"
        )

    def _record(self, backend: str) -> Dict[str, Any]:
        name = normalize_backend(backend)
        rec = self._memory["backends"].get(name)
        if rec is None:
            rec = _empty_backend_record()
            self._memory["backends"][name] = rec
        return rec

    def _derive_state(
        self, rec: Dict[str, Any], now: float
    ) -> str:
        nfail = int(rec.get("consecutive_failures") or 0)
        last_fail = rec.get("last_failure_time")
        if nfail >= self.failure_threshold and last_fail is not None:
            elapsed = now - float(last_fail)
            if elapsed < self.cooling_seconds:
                return STATE_CIRCUIT_OPEN
            return STATE_DEGRADED
        if nfail > 0:
            return STATE_DEGRADED
        return STATE_HEALTHY

    def _is_stale(self, now: float) -> bool:
        updated_at = self._memory.get("updated_at")
        if updated_at is None:
            return False
        return (now - float(updated_at)) >= self.stale_after_seconds

    def snapshot(
        self, backend: str, *, now: Optional[float] = None
    ) -> Dict[str, Any]:
        """Present-tense view of one backend. ``stale`` means the on-disk
        record is a prior, not present truth.
        """
        name = normalize_backend(backend)
        rec = dict(self._record(name))
        ts = self._now(now)
        state = self._derive_state(rec, ts)
        stale = self._is_stale(ts)
        allows = (state != STATE_CIRCUIT_OPEN) or stale
        return {
            "backend": name,
            "consecutive_failures": int(rec["consecutive_failures"]),
            "last_failure_time": rec["last_failure_time"],
            "last_success_time": rec["last_success_time"],
            "state": state,
            "updated_at": self._memory.get("updated_at"),
            "stale": stale,
            "allows_new": allows,
        }

    def snapshot_all(self, *, now: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        ts = self._now(now)
        return {name: self.snapshot(name, now=ts) for name in KNOWN_BACKENDS}

    def state(self, backend: str, *, now: Optional[float] = None) -> str:
        return self.snapshot(backend, now=now)["state"]

    def allows_new_assignments(
        self, backend: str, *, now: Optional[float] = None
    ) -> bool:
        """True iff NEW units may be routed to this backend.

        False only while the breaker is open *and* the record is fresh.
        A stale open circuit is a prior, not present truth: new work is
        allowed (the cooling window has typically already elapsed; even
        if it has not, we will not refuse work on an hour-old file).

        Where to call this: ``assign_ready`` (workunit.py) after
        ``can_admit``, keyed by ``WorkUnit.preferred_backend``. Not
        inside ``can_admit``.
        """
        return bool(self.snapshot(backend, now=now)["allows_new"])

    def record_success(
        self, backend: str, *, now: Optional[float] = None
    ) -> Dict[str, Any]:
        name = normalize_backend(backend)
        ts = self._now(now)
        rec = self._record(name)
        rec["consecutive_failures"] = 0
        rec["last_success_time"] = ts
        self._memory["updated_at"] = ts
        self._persist()
        return self.snapshot(name, now=ts)

    def record_failure(
        self,
        backend: str,
        context: Any = None,
        *,
        now: Optional[float] = None,
    ) -> FailureClassification:
        """Record a failure against ``backend``. Returns the classification.

        Consecutive-failure count (the circuit) increments only for
        backend-health kinds. Non-retryable unit failures such as
        NO_OP_MUTATION / GrokContractError do not trip the breaker;
        GrokNotAvailable does (the grok backend is the unavailable
        dependency). Retry budget is a separate question:
        ``classification.retryable`` / ``counts_toward_retry_budget``.
        """
        name = normalize_backend(backend)
        clf = classify_failure(context)
        if clf.kind not in _CIRCUIT_KINDS:
            return clf
        ts = self._now(now)
        rec = self._record(name)
        rec["consecutive_failures"] = int(rec["consecutive_failures"]) + 1
        rec["last_failure_time"] = ts
        self._memory["updated_at"] = ts
        self._persist()
        return clf

    def reload(self) -> None:
        """Re-read disk. Used by a second process after another wrote."""
        if self.path is None:
            return
        loaded = self._read_disk()
        if loaded is not None:
            self._memory = loaded
        else:
            self._memory = {
                "version": HEALTH_VERSION,
                "updated_at": None,
                "backends": {
                    name: _empty_backend_record() for name in KNOWN_BACKENDS
                },
            }
