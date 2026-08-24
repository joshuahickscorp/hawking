"""GrokBridge — HCLI's thin wrapper around the `grok-run` binary.

This is a module, not a slash-command handler. Wiring `/grok` into
``commands.py::CommandHandler`` later is a single ``_cmd_grok`` method
that calls this class; that method must not know ``grok-run``'s CLI
shape.

Public surface
--------------
Exceptions:
  ``GrokNotAvailable``   ``grok-run`` is not on PATH (``shutil.which``).
  ``GrokContractError``  caller-supplied contract is missing/empty or
                         lacks WRITE/VERIFY sections, or ``grok-run``'s
                         linter rejected it (``NO_VERIFICATION``,
                         ``UNBOUNDED_WRITE``, ``NO_ACCEPTANCE``, …).
  ``GrokRunError``       ``grok-run`` ran and failed for some other reason.

Functions:
  ``parse_grok_status(text) -> dict``
      Pure parser for ``grok-run status --id`` human text. Returns
      ``{"state": "running"|"done"|"failed"|"unknown", "exit_code": int|None}``.
      Garbage input returns ``state="unknown"`` and does not raise.
  ``find_grok_run() -> str``
      PATH lookup. Raises ``GrokNotAvailable``. Never invents a binary.
  ``validate_contract_text(text)``
      Raises ``GrokContractError`` if the caller did not supply a contract
      with WRITE and VERIFY sections. This module never drafts one.

``GrokRunHandle``
  ``task_id``, ``command_run`` (exact grok-run argv), ``started_at``,
  plus ``dry_run``, ``task_dir``, ``receipt_path``, ``stdout``, ``stderr``,
  ``resolved_command`` (inner ``grok`` command when ``GROK_DRYRUN=1``).

``GrokBridge(workspace, receipts_dir=None)``
  ``delegate(task, contract_text, *, profile="power", background=True,
             no_worktree=True, mutation_lock=None, dry_run=None)``
  ``audit(task, contract_text, *, background=True, dry_run=None)``
  ``consult(prompt, *, background=True, dry_run=None)``
  ``status(task_id) -> dict``     normalized via ``parse_grok_status``
  ``wait(task_id, timeout=3600.0) -> dict``
  ``report(task_id) -> str``      contents of ``grok-report.md``
  ``cancel(task_id) -> dict``     TERM/KILL the launch pid; cleanup does not
  ``cleanup(task_id) -> dict``    grok-run cleanup (worktree only; does not kill)

Follow-up: ``CommandHandler._cmd_grok``
--------------------------------------
A later lane should add ``_cmd_grok(self, arg: str) -> str`` that:

1. Builds ``GrokBridge(Path(self.controller.workspace_root))``.
2. Parses ``arg`` as one of
   ``delegate|audit|consult|status|wait|report|cleanup`` plus operands.
3. For ``delegate`` / ``audit``, **the caller supplies ``contract_text``**.
   This module will not write a prose contract on the controller's behalf.
4. For ``delegate``, **must** pass ``mutation_lock=`` wrapping
   ``hcli.resources.MutationLock`` acquire/release (see below). Omitting
   it is not legal inside a live mission.
5. Returns a human string (task id, state, report excerpt). Structured
   data lives on the handle and at
   ``<workspace>/.hcli/grok/<task_id>.json``.
6. ``dry_run=True`` or env ``GROK_DRYRUN=1`` previews the exact argv at
   zero cost. ``handle.command_run`` is the grok-run argv; grok-run's own
   dry-run stdout is the *inner* ``grok`` binary command (it does not echo
   ``--task`` / ``--profile`` / ``--background`` / ``--no-worktree``).

Mutation serialization (B3) — do not miss this
----------------------------------------------
Directive law: there MUST NOT be two independent writers mutating the
repository. ``delegate()`` is MUTATION-class work (see
``hcli/resources.py``, ``ResourceClass.MUTATION``, cap 1,
crash-safe ``MutationLock``).

This module does **not** schedule and does **not** import ``resources``
or ``scheduler``. The caller (mission loop / future ``_cmd_grok``) MUST
pass ``mutation_lock``: a zero-arg callable returning an
``AbstractContextManager`` that serializes against that lock.

Example (future wiring)::

    from contextlib import contextmanager
    from hcli.resources import MutationLock

    lock = MutationLock(workspace_root)

    @contextmanager
    def mutation_lock():
        if not lock.acquire(unit_id):
            raise RuntimeError("MUTATION lock held")
        try:
            yield
        finally:
            lock.release(unit_id)

    bridge.delegate(task, contract_text, mutation_lock=mutation_lock)

If ``mutation_lock`` is omitted, ``delegate()`` still launches (useful
while nothing calls it from a live mission) but logs a WARNING that
**no mutation-serialization was applied**. A live mission must not
rely on that path. ``audit()`` and ``consult()`` are read-only and do
not take a mutation lock.

Receipts (B2)
-------------
Every ``delegate`` / ``audit`` / ``consult`` writes
``<workspace>/.hcli/grok/<task_id>.json`` (directory created if needed)
with task id, exact argv, timestamps, dry-run flag, and — once resolved
— final status and a path to ``grok-report.md``. A Grok delegation from
inside HCLI is as auditable as one of HCLI's own mutations.

GROK_DRYRUN
-----------
``grok-run`` prints the resolved inner command and exits without
spending a Grok session when ``GROK_DRYRUN=1``. Pass ``dry_run=True``
on a launch method, or set that env var. The bridge never invents a
task id: it parses grok-run's ``task dir:`` line (dry-run) or the
printed id (live).
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .persist import atomic_write_json, atomic_write_text

LOG = logging.getLogger("hcli.grok_bridge")

NO_MUTATION_LOCK_WARNING = (
    "GrokBridge.delegate: no mutation_lock provided; "
    "no mutation-serialization was applied. "
    "delegate() is MUTATION-class work (HCLI resources.MUTATION, cap 1). "
    "A live mission MUST pass mutation_lock= a callable returning a "
    "context manager that acquires that lock; omitting it is only safe "
    "outside a live mission."
)

DEFAULT_TASKS_ROOT = Path.home() / ".claude-grok" / "tasks"
DEFAULT_GROK_RUN_HINT = Path.home() / ".claude-grok" / "bin" / "grok-run"

_WRITE_HEADING = re.compile(
    r"^\s{0,3}#{0,3}\s*(?:WRITE|EDIT|OUTPUT)\b",
    re.IGNORECASE | re.MULTILINE,
)
_VERIFY_HEADING = re.compile(
    r"^\s{0,3}#{0,3}\s*VERIFY\b",
    re.IGNORECASE | re.MULTILINE,
)
_STATUS_LINE = re.compile(
    r"status:\s*(?P<state>\S+)\s*\(exit\s+(?P<exit>[^)]+)\)",
    re.IGNORECASE,
)
_TASK_DIR_LINE = re.compile(r"^task dir:\s+(\S+)\s*$", re.MULTILINE)
_TASK_ID_LINE = re.compile(r"^[A-Za-z0-9._-]+-\d{8}-\d{6}$")
_LINT_BLOCK = re.compile(
    r"contract rejected before launch:\s*(.*?)(?:\nFix the contract|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_LINT_CODE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\s*:")
_DRY_RUN_BODY = re.compile(
    r"DRY RUN\s+[—\-]+\s+would execute:\n(.*?)(?:\ntask dir:|\Z)",
    re.DOTALL,
)

MutationLockFactory = Callable[[], AbstractContextManager[Any]]


class GrokNotAvailable(RuntimeError):
    """``grok-run`` is not on PATH. Never swallowed, never faked with a task id."""


class GrokContractError(ValueError):
    """Caller-supplied contract is missing, malformed, or lint-rejected."""

    def __init__(self, message: str, *, codes: Optional[Sequence[str]] = None) -> None:
        super().__init__(message)
        self.codes = list(codes or [])


class GrokRunError(RuntimeError):
    """``grok-run`` ran and failed. ``argv`` is the exact command attempted."""

    def __init__(
        self,
        message: str,
        *,
        argv: Optional[Sequence[str]] = None,
        returncode: Optional[int] = None,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.argv = list(argv or [])
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@dataclass
class GrokRunHandle:
    """Result of a grok-run launch. ``command_run`` is the exact argv for audit."""

    task_id: str
    command_run: List[str]
    started_at: str
    dry_run: bool = False
    task_dir: Optional[str] = None
    receipt_path: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    resolved_command: str = ""
    mode: str = ""
    launch_pid: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def find_grok_run() -> str:
    """Return the ``grok-run`` executable. PATH is authority; never invent one."""
    override = os.environ.get("GROK_RUN")
    if override:
        path = Path(override)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        raise GrokNotAvailable(
            f"GROK_RUN={override} is set but is not an executable file"
        )
    found = shutil.which("grok-run")
    if found:
        return found
    # PATH is authority, but the canonical install location is not a guess --
    # it is the exact path this module already tells you to install to, and it
    # is checked for executability before use. Requiring the caller to also
    # export PATH made HCLI refuse to drive Grok on a machine where the binary
    # was sitting right there, which is a PATH dependency in production, not a
    # safety property. GROK_RUN still overrides, and a missing binary still
    # raises rather than inventing a task id.
    if DEFAULT_GROK_RUN_HINT.is_file() and os.access(DEFAULT_GROK_RUN_HINT, os.X_OK):
        return str(DEFAULT_GROK_RUN_HINT)
    raise GrokNotAvailable(
        "grok-run is not on PATH (shutil.which returned None) and is not at "
        f"{DEFAULT_GROK_RUN_HINT}. Install it there, or set GROK_RUN. "
        "Refusing to invent a task id or pretend a Grok session ran."
    )


def validate_contract_text(
    contract_text: Optional[str],
    *,
    require_write: bool = True,
    require_verify: bool = True,
) -> str:
    """Require a caller-supplied contract with the shape grok-run's linter wants.

    ``grok-run`` rejects contracts lacking verification or naming no write
    scope (``NO_VERIFICATION``, ``UNBOUNDED_WRITE``, ``NO_ACCEPTANCE`` in
    ``~/.claude-grok/v2/contract.mjs``). This preflight catches the empty
    and section-less cases *before* any subprocess is spawned.
    """
    if contract_text is None or not str(contract_text).strip():
        raise GrokContractError(
            "contract_text is required and must be supplied by the caller; "
            "GrokBridge will not invent a contract"
        )
    text = str(contract_text)
    missing: List[str] = []
    if require_write and not _WRITE_HEADING.search(text):
        missing.append("WRITE")
    if require_verify and not _VERIFY_HEADING.search(text):
        missing.append("VERIFY")
    if missing:
        raise GrokContractError(
            "contract is missing required section(s): "
            + ", ".join(missing)
            + ". grok-run's linter requires WRITE and VERIFY "
            "(see ~/.claude-grok/v2/contract.mjs: NO_VERIFICATION, "
            "UNBOUNDED_WRITE). Pass a real contract; this module will not "
            "draft one."
        )
    return text


def parse_grok_status(text: Optional[str]) -> Dict[str, Any]:
    """Parse ``grok-run status --id`` human text into a structured dict.

    Known shapes (from the real binary)::

        status: running (exit -)
        status: done (exit 0)

    ``grok-run`` writes ``done`` to the status file even on a nonzero
    Grok exit; a nonzero exit_code is normalized to ``state="failed"``.
    Garbage input returns ``{"state": "unknown", "exit_code": None}``
    and does not raise.
    """
    unknown: Dict[str, Any] = {"state": "unknown", "exit_code": None}
    if not text or not isinstance(text, str):
        return dict(unknown)
    matches = list(_STATUS_LINE.finditer(text))
    if not matches:
        return dict(unknown)
    m = matches[-1]
    raw_state = (m.group("state") or "").strip().lower()
    raw_exit = (m.group("exit") or "").strip()
    exit_code: Optional[int] = None
    if raw_exit not in ("-", "", "?", "none"):
        try:
            exit_code = int(raw_exit)
        except ValueError:
            exit_code = None
    if raw_state == "running":
        state = "running"
    elif raw_state in ("failed", "error"):
        state = "failed"
    elif raw_state == "done":
        if exit_code is None or exit_code == 0:
            state = "done"
        else:
            state = "failed"
    elif raw_state == "unknown":
        state = "unknown"
    else:
        state = "unknown"
    return {"state": state, "exit_code": exit_code}


def grok_succeeded(status: Optional[Dict[str, Any]] = None) -> bool:
    """True only when the task is terminal-good AND exit_code is 0.

    grok-run writes ``status=done`` even on a nonzero exit (the on-disk
    shape of ``consult-20260822-223811``: status file ``done``, exit_code
    file ``1``). A caller that reads only ``state`` would treat that as
    success. This field is the explicit normalisation; do not recompute
    it at every call site.
    """
    if not isinstance(status, dict):
        return False
    state = str(status.get("state") or "").strip().lower()
    exit_code = status.get("exit_code")
    try:
        exit_int = int(exit_code) if exit_code is not None else None
    except (TypeError, ValueError):
        exit_int = None
    return state == "done" and exit_int == 0


def extract_task_id(stdout: str, stderr: str = "") -> str:
    """Pull grok-run's task id out of its output. Never synthesizes one."""
    blob = stdout or ""
    m = _TASK_DIR_LINE.search(blob)
    if m:
        return Path(m.group(1)).name
    lines = [ln.strip() for ln in blob.splitlines() if ln.strip()]
    candidates = [ln for ln in lines if _TASK_ID_LINE.fullmatch(ln)]
    if candidates:
        return candidates[-1]
    raise GrokRunError(
        "grok-run produced no task id (refusing to invent one). "
        f"stdout={stdout!r} stderr={stderr!r}",
        stdout=stdout,
        stderr=stderr,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_SLUG_COUNTER = itertools.count()
_SLUG_LOCK = threading.Lock()


def unique_task_slug(prefix: str = "consult") -> str:
    """A task slug that cannot collide inside a single second.

    grok-run derives its task id by appending a one-second-resolution timestamp
    to this slug. Two dispatches in the same second therefore landed on the SAME
    task directory: in the recorded mixed-max run, WorkUnits `grok1` and `grok2`
    both resolved to `consult-20260822-224557`, so two units were accepted off
    one Grok execution and the second unit's prompt was never sent at all.

    grok-run itself is out of this repository's mutation scope, so uniqueness
    has to come from the caller. pid plus a monotonic counter plus microseconds
    is unique across threads within a process, and across processes on this box.
    """
    with _SLUG_LOCK:
        n = next(_SLUG_COUNTER)
    micros = int(time.time() * 1_000_000) % 1_000_000
    return f"{prefix}-{os.getpid():d}x{n:d}x{micros:06d}"


def process_alive(pid: Optional[int]) -> bool:
    """True only when the pid names a live process. False on any doubt."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, OverflowError, ValueError, TypeError):
        return False
    return True


_TERMINAL_STATES = frozenset({"done", "failed", "cancelled", "stale-running"})


def _reap_if_child(pid: int) -> None:
    """Reap a direct child so a zombie is not reported as alive."""
    try:
        os.waitpid(int(pid), os.WNOHANG)
    except (OSError, OverflowError, ValueError, TypeError, ChildProcessError):
        return


_PROC_LIB: Any = None
_PROC_LIB_READY = False
PROC_PGRP_ONLY = 2
PROC_PPID_ONLY = 6


def _proc_lib() -> Any:
    """Darwin libproc. ``ps`` is frequently blocked in sandboxed agent runs."""
    global _PROC_LIB, _PROC_LIB_READY
    if _PROC_LIB_READY:
        return _PROC_LIB
    _PROC_LIB_READY = True
    try:
        import ctypes
        import ctypes.util

        lib = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
        lib.proc_listpids.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.proc_listpids.restype = ctypes.c_int
        lib.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.proc_pidinfo.restype = ctypes.c_int
        _PROC_LIB = lib
        return lib
    except Exception:
        _PROC_LIB = None
        return None


def _proc_listpids(kind: int, info: int) -> List[int]:
    lib = _proc_lib()
    if lib is None:
        return []
    import ctypes

    bufsize = 4096 * 4
    buf = (ctypes.c_int * (bufsize // 4))()
    try:
        n = lib.proc_listpids(int(kind), int(info), buf, bufsize)
    except Exception:
        return []
    if n <= 0:
        return []
    count = n // ctypes.sizeof(ctypes.c_int)
    return [int(buf[i]) for i in range(count) if buf[i] > 1]


def _proc_pgid(pid: int) -> Optional[int]:
    lib = _proc_lib()
    if lib is None:
        return None
    import ctypes

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

    info = proc_bsdinfo()
    try:
        got = lib.proc_pidinfo(int(pid), 3, 0, ctypes.byref(info), ctypes.sizeof(info))
    except Exception:
        return None
    if got <= 0:
        return None
    pgid = int(info.pbi_pgid)
    return pgid if pgid > 1 else None


def _process_tree(pid: int) -> List[int]:
    """``pid`` plus descendants and process-group members.

    grok-run's background wrapper is ``( trap '' HUP INT; execute_task ) &``.
    Killing only the wrapper pid reparents the grok child; the expensive
    process keeps running. Walk the tree (libproc; ``ps`` if allowed).
    """
    try:
        target = int(pid)
    except (TypeError, ValueError):
        return []
    if target <= 1:
        return []
    found = {target}
    pgid = _proc_pgid(target)
    if pgid:
        found.update(_proc_listpids(PROC_PGRP_ONLY, pgid))
    stack = [target]
    seen_walk = set()
    while stack:
        cur = stack.pop()
        if cur in seen_walk:
            continue
        seen_walk.add(cur)
        for child in _proc_listpids(PROC_PPID_ONLY, cur):
            if child not in found:
                found.add(child)
                stack.append(child)
    if len(found) == 1:
        # libproc unavailable or empty: try ps, then the pid alone.
        children: Dict[int, List[int]] = {}
        try:
            out = subprocess.check_output(
                ["ps", "-A", "-o", "pid=", "-o", "ppid="],
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return [target]
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                cpid, ppid = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            children.setdefault(ppid, []).append(cpid)
        acc: List[int] = []
        stack = [target]
        seen = set()
        while stack:
            cur = stack.pop()
            if cur in seen or cur <= 1:
                continue
            seen.add(cur)
            acc.append(cur)
            stack.extend(children.get(cur, ()))
        return acc
    return [p for p in found if p > 1]


def _terminate_pids(pids: Sequence[int], *, grace: float = 2.0) -> None:
    """TERM then KILL each pid, process-group first. Never signals this process."""
    protected = {0, 1, os.getpid(), os.getppid()}
    ordered = [int(p) for p in pids if p and int(p) not in protected]
    if not ordered:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in ordered:
            if pid in protected:
                continue
            try:
                os.killpg(pid, sig)
            except OSError:
                try:
                    os.kill(pid, sig)
                except OSError:
                    pass
            _reap_if_child(pid)
        deadline = time.monotonic() + (grace if sig == signal.SIGTERM else 1.0)
        while time.monotonic() < deadline:
            if not any(process_alive(pid) for pid in ordered):
                return
            time.sleep(0.05)
        for pid in ordered:
            _reap_if_child(pid)


_BG_PID_RE = re.compile(r"started in background \(pid\s+(\d+)\)")


def extract_background_pid(text: str) -> Optional[int]:
    """Recover the launched wrapper pid from grok-run's own stdout.

    This is the only per-task liveness signal available: nothing in the process
    table carries the task id, so `ps` cannot answer "is this task still alive".
    Without it a status file left at `running` by a dead process is
    indistinguishable from real work, and the scheduler holds the slot for the
    full wait timeout.
    """
    m = _BG_PID_RE.search(text or "")
    return int(m.group(1)) if m else None


def _as_mutation_lock(
    mutation_lock: Optional[Union[MutationLockFactory, AbstractContextManager[Any]]],
) -> AbstractContextManager[Any]:
    if mutation_lock is None:
        return nullcontext()
    # Classes (e.g. nullcontext) have __enter__ on the type but are factories.
    if (
        not isinstance(mutation_lock, type)
        and hasattr(mutation_lock, "__enter__")
        and hasattr(mutation_lock, "__exit__")
    ):
        return mutation_lock  # type: ignore[return-value]
    if callable(mutation_lock):
        return mutation_lock()
    raise TypeError(
        "mutation_lock must be a callable returning a context manager, "
        f"or a context manager instance; got {type(mutation_lock)!r}"
    )


def _want_dry_run(explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return bool(explicit)
    return str(os.environ.get("GROK_DRYRUN", "0")).strip() == "1"


def _resolved_command(stdout: str) -> str:
    m = _DRY_RUN_BODY.search(stdout or "")
    if not m:
        return ""
    return m.group(1).strip()


def _task_dir_from_stdout(stdout: str) -> Optional[str]:
    m = _TASK_DIR_LINE.search(stdout or "")
    if not m:
        return None
    return m.group(1)


def _lint_codes(blob: str) -> List[str]:
    return _LINT_CODE.findall(blob or "")


def _read_retries(task_dir: Optional[Union[str, Path]]) -> Dict[str, Any]:
    if not task_dir:
        return {"retries": None, "reason": "telemetry.json absent"}
    path = Path(task_dir) / "telemetry.json"
    if not path.is_file():
        return {"retries": None, "reason": "telemetry.json absent"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"retries": None, "reason": "telemetry.json unreadable"}
    if not isinstance(data, dict):
        return {"retries": None, "reason": "telemetry.json is not an object"}
    if "retries" not in data:
        return {"retries": None, "reason": "retries key absent from telemetry.json"}
    return {"retries": data["retries"], "reason": None}


def _read_throttle(task_dir: Optional[Union[str, Path]]) -> Dict[str, Any]:
    if not task_dir:
        return {"value": None, "reason": "not present in telemetry.json"}
    path = Path(task_dir) / "telemetry.json"
    if not path.is_file():
        return {"value": None, "reason": "not present in telemetry.json"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"value": None, "reason": "not present in telemetry.json"}
    if not isinstance(data, dict):
        return {"value": None, "reason": "not present in telemetry.json"}
    for key in ("throttle", "throttle_evidence", "throttled", "rate_limit"):
        if key in data:
            return {"value": {key: data[key]}, "reason": None}
    return {"value": None, "reason": "not present in telemetry.json"}


def _read_failure_evidence(task_dir: Optional[Union[str, Path]]) -> Dict[str, Any]:
    if not task_dir:
        return {"value": None, "reason": "grok-stderr.log absent or empty"}
    path = Path(task_dir) / "grok-stderr.log"
    if not path.is_file():
        return {"value": None, "reason": "grok-stderr.log absent or empty"}
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return {"value": None, "reason": "grok-stderr.log unreadable"}
    if not text:
        return {"value": None, "reason": "grok-stderr.log absent or empty"}
    return {"value": {"source": "grok-stderr.log", "text": text[-4000:]}, "reason": None}


class GrokBridge:
    """Thin, honest wrapper around ``grok-run`` for HCLI."""

    def __init__(
        self,
        workspace: Union[str, Path],
        receipts_dir: Optional[Path] = None,
    ) -> None:
        self.workspace = Path(workspace)
        if receipts_dir is None:
            self.receipts_dir = self.workspace / ".hcli" / "grok"
        else:
            self.receipts_dir = Path(receipts_dir)

    def delegate(
        self,
        task: str,
        contract_text: str,
        *,
        profile: str = "power",
        background: bool = True,
        no_worktree: bool = True,
        mutation_lock: Optional[Union[MutationLockFactory, AbstractContextManager[Any]]] = None,
        dry_run: Optional[bool] = None,
    ) -> GrokRunHandle:
        """Launch a mutating grok-run delegate.

        ``mutation_lock`` serializes this call against HCLI's MUTATION
        resource class. If omitted, a warning is logged and the launch
        still proceeds — that path is not legal inside a live mission.
        """
        validate_contract_text(contract_text)
        self._require_task(task)
        serialized = mutation_lock is not None
        if mutation_lock is None:
            LOG.warning(NO_MUTATION_LOCK_WARNING)
        locker: AbstractContextManager[Any] = _as_mutation_lock(mutation_lock)
        with locker:
            return self._launch(
                mode="delegate",
                task=str(task).strip(),
                contract_text=str(contract_text),
                profile=profile or "power",
                background=background,
                no_worktree=no_worktree,
                dry_run=_want_dry_run(dry_run),
                mutation_serialized=serialized,
            )

    def audit(
        self,
        task: str,
        contract_text: str,
        *,
        background: bool = True,
        dry_run: Optional[bool] = None,
    ) -> GrokRunHandle:
        """Launch a read-only grok-run audit. No mutation lock."""
        validate_contract_text(contract_text)
        self._require_task(task)
        return self._launch(
            mode="audit",
            task=str(task).strip(),
            contract_text=str(contract_text),
            background=background,
            dry_run=_want_dry_run(dry_run),
            mutation_serialized=False,
        )

    def consult(
        self,
        prompt: str,
        *,
        background: bool = True,
        dry_run: Optional[bool] = None,
    ) -> GrokRunHandle:
        """Launch a read-only grok-run consult from a prompt. No mutation lock."""
        if prompt is None or not str(prompt).strip():
            raise GrokContractError(
                "consult requires a prompt; GrokBridge will not invent one"
            )
        return self._launch(
            mode="consult",
            task=unique_task_slug("consult"),
            prompt=str(prompt),
            background=background,
            dry_run=_want_dry_run(dry_run),
            mutation_serialized=False,
        )

    def status(self, task_id: str) -> Dict[str, Any]:
        """Run ``grok-run status --id`` and return the normalized dict."""
        self._require_task_id(task_id)
        argv = [find_grok_run(), "status", "--id", str(task_id)]
        result = self._run(argv, dry_run=False, check=False)
        parsed = dict(
            parse_grok_status((result.stdout or "") + "\n" + (result.stderr or ""))
        )
        parsed["successful"] = grok_succeeded(parsed)
        parsed["grok_state"] = parsed.get("state")
        report_path = self._report_path(task_id)
        parsed["task_id"] = str(task_id)
        parsed["raw_returncode"] = result.returncode

        receipt = self._read_receipt(str(task_id)) or {}
        cancelled = bool(receipt.get("cancelled")) or (
            isinstance(receipt.get("status"), dict)
            and str(receipt["status"].get("state") or "").strip().lower()
            == "cancelled"
        )
        # grok-run's status file does not know about cancel. Never let a
        # still-`running` file un-cancel a task this bridge already killed.
        if not cancelled:
            receipt_status = {
                "state": parsed.get("state"),
                "exit_code": parsed.get("exit_code"),
                "successful": parsed["successful"],
            }
            self._update_receipt(
                str(task_id),
                status=receipt_status,
                grok_state=parsed.get("state"),
                report_path=str(report_path) if report_path is not None else None,
            )
            receipt = self._read_receipt(str(task_id)) or receipt

        # `grok-run status` reads a status FILE and has no pid, so a task whose
        # process died mid-run reports `running` forever and the scheduler holds
        # its slot for the whole wait timeout. Judge by the process, not the file.
        pid = receipt.get("launch_pid")
        state = str(parsed.get("state") or "")
        if state == "running" and pid is not None:
            alive = process_alive(pid)
            parsed["launch_pid"] = pid
            parsed["process_alive"] = alive
            if not alive and not cancelled:
                # Do not silently rewrite it to `failed`: the distinction
                # between "this task failed" and "nobody is running this any
                # more" is exactly what a caller needs to decide retry vs adopt.
                parsed["state"] = "stale-running"
                parsed["stale_reason"] = (
                    f"status file says running but launch pid {pid} is gone"
                )
                self._update_receipt(str(task_id), extra={"stale_running": True})
        elif pid is not None:
            parsed["launch_pid"] = pid
            parsed["process_alive"] = process_alive(pid)
        if cancelled:
            parsed["state"] = "cancelled"
            parsed["successful"] = False
            parsed["grok_state"] = "cancelled"
            if pid is not None:
                parsed["launch_pid"] = pid
                parsed["process_alive"] = process_alive(pid)
        return parsed

    def poll(self, task_id: str) -> Dict[str, Any]:
        """Non-blocking status. Scheduler-friendly; does not wait."""
        return self.status(task_id)

    def wait(
        self,
        task_id: str,
        timeout: float = 3600.0,
        poll_interval: float = 0.5,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Poll ``status`` until terminal or ``timeout`` seconds elapse.

        Ownership of the deadline is here. This method does **not** invoke
        ``grok-run wait`` (whose outer Python subprocess timeout used to
        kill a 3600s wait at 120s). Each poll is a short ``status`` call.

        ``cancelled`` and ``stale-running`` are terminal: a cancelled mission
        must not sit in this loop while the Grok process it started keeps
        running, and a dead launch pid is not "still going".
        """
        self._require_task_id(task_id)
        seconds = max(0.0, float(timeout))
        deadline = time.monotonic() + seconds
        interval = max(0.05, float(poll_interval))
        last: Dict[str, Any] = {"state": "unknown", "exit_code": None, "task_id": str(task_id)}
        while True:
            if callable(is_cancelled) and is_cancelled():
                return self.cancel(task_id)
            last = self.status(task_id)
            state = str(last.get("state") or "")
            if state in _TERMINAL_STATES:
                report_path = self._report_path(task_id)
                last["report_path"] = str(report_path) if report_path is not None else None
                return last
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GrokRunError(
                    f"grok-run wait timed out after {int(seconds)}s for {task_id}",
                    argv=[
                        find_grok_run(),
                        "status",
                        "--id",
                        str(task_id),
                    ],
                )
            time.sleep(min(interval, remaining))
            interval = min(interval * 1.5, 5.0)

    def compact_report(self, task_id: str) -> Dict[str, Any]:
        """Compile a compact structured result. Raw traces stay on disk."""
        from .report_compiler import compile_backend_report

        self._require_task_id(task_id)
        raw_path = self._report_path(task_id)
        raw_text = ""
        try:
            raw_text = self.report(task_id)
        except GrokRunError:
            if raw_path is not None and raw_path.is_file():
                raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
        receipt = self._read_receipt(task_id) or {}
        compact = compile_backend_report(
            backend="grok",
            task_id=str(task_id),
            raw_text=raw_text,
            raw_report_path=str(raw_path) if raw_path is not None else None,
            extra=receipt,
        )
        self.receipts_dir.mkdir(parents=True, exist_ok=True)
        dest = self.receipts_dir / f"{task_id}.compact.json"
        atomic_write_json(dest, compact)
        compact["compact_path"] = str(dest)
        self._update_receipt(str(task_id), extra={"compact_path": str(dest)})
        return compact

    def report(self, task_id: str) -> str:
        """Return the contents of ``grok-report.md``. Does not fabricate one."""
        self._require_task_id(task_id)
        path = self._report_path(task_id)
        if path is None or not path.is_file():
            raise GrokRunError(
                f"no grok-report.md for task {task_id} "
                "(the task has not produced a report; refusing to invent one)"
            )
        return path.read_text(encoding="utf-8")

    def cleanup(self, task_id: str) -> Dict[str, Any]:
        """Run ``grok-run cleanup --id``. Artifacts stay on disk; grok-run says so.

        This does **not** kill the launch pid. grok-run's cleanup only
        considers worktrees. Use ``cancel`` to stop a running task.
        """
        self._require_task_id(task_id)
        argv = [find_grok_run(), "cleanup", "--id", str(task_id)]
        result = self._run(argv, dry_run=False, check=False)
        out = {
            "task_id": str(task_id),
            "ok": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }
        self._update_receipt(str(task_id), extra={"cleanup": out})
        return out

    def launch_pid_for(self, task_id: str) -> Optional[int]:
        """The wrapper pid grok-run printed at launch, if we observed one."""
        receipt = self._read_receipt(str(task_id)) or {}
        pid = receipt.get("launch_pid")
        try:
            value = int(pid) if pid is not None else None
        except (TypeError, ValueError):
            return None
        return value if value and value > 0 else None

    def cancel(self, task_id: str, *, grace: float = 2.0) -> Dict[str, Any]:
        """Stop a Grok task this bridge started.

        grok-run has no cancel command, and ``cleanup`` does not kill. The
        launch pid is recorded on the receipt from grok-run's
        ``started in background (pid N)`` line; that is the only mapping
        from task id to process. The background wrapper ignores HUP/INT
        (``trap '' HUP INT``), so SIGTERM/SIGKILL on the tree is the
        signal that actually reaches it.

        Putting the pid only on ``mission.child_pids`` is not enough:
        executors never register it, so ``Mission._stop_children`` never
        sees a Grok pid. Cancel lives here, where the pid is known.
        """
        self._require_task_id(task_id)
        pid = self.launch_pid_for(task_id)
        alive_before = process_alive(pid)
        tree = _process_tree(pid) if pid else []
        _terminate_pids(tree, grace=grace)
        if pid:
            _reap_if_child(pid)
        alive_after = process_alive(pid)
        self._update_receipt(
            str(task_id),
            status={
                "state": "cancelled",
                "exit_code": None,
                "successful": False,
            },
            grok_state="cancelled",
            extra={"cancelled": True, "launch_pid": pid},
        )
        return {
            "task_id": str(task_id),
            "launch_pid": pid,
            "alive_before": alive_before,
            "process_alive": alive_after,
            "killed": bool(alive_before and not alive_after),
            "state": "cancelled",
            "pids": tree,
        }

    def receipt_path(self, task_id: str) -> Path:
        return self.receipts_dir / f"{task_id}.json"

    def _require_task(self, task: str) -> None:
        if task is None or not str(task).strip():
            raise GrokContractError("task slug is required")

    def _require_task_id(self, task_id: str) -> None:
        if task_id is None or not str(task_id).strip():
            raise GrokRunError("task_id is required")

    def _launch(
        self,
        *,
        mode: str,
        task: str,
        contract_text: Optional[str] = None,
        prompt: Optional[str] = None,
        profile: Optional[str] = None,
        background: bool = True,
        no_worktree: bool = False,
        dry_run: bool = False,
        mutation_serialized: bool = False,
    ) -> GrokRunHandle:
        grok_run = find_grok_run()
        started_at = _now_iso()
        contract_path: Optional[Path] = None
        argv: List[str] = [grok_run, mode]
        if mode in ("delegate", "audit"):
            assert contract_text is not None
            contract_path = self._write_contract_file(task, contract_text)
            argv.extend(["--task", task, "--contract", str(contract_path)])
        elif mode == "consult":
            argv.extend(["--prompt", str(prompt)])
        if mode == "delegate" and profile:
            argv.extend(["--profile", str(profile)])
        if mode == "delegate" and no_worktree:
            argv.append("--no-worktree")
        if background:
            argv.append("--background")
        argv.extend(["--repo", str(self.workspace)])

        result = self._run(argv, dry_run=dry_run, check=False)
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        combined = stderr + "\n" + stdout
        if result.returncode != 0:
            self._raise_from_failure(argv, result, combined)

        try:
            task_id = extract_task_id(stdout, stderr)
        except GrokRunError as exc:
            exc.argv = list(argv)
            exc.returncode = result.returncode
            raise

        task_dir = _task_dir_from_stdout(stdout)
        if task_dir is None:
            task_dir = str(DEFAULT_TASKS_ROOT / task_id)
        resolved = _resolved_command(stdout)
        handle = GrokRunHandle(
            task_id=task_id,
            command_run=list(argv),
            started_at=started_at,
            dry_run=dry_run,
            task_dir=task_dir,
            stdout=stdout,
            stderr=stderr,
            resolved_command=resolved,
            mode=mode,
        )
        receipt = self._write_receipt(
            handle,
            mode=mode,
            mutation_serialized=mutation_serialized,
            contract_path=str(contract_path) if contract_path else None,
            prompt=prompt,
            returncode=result.returncode,
        )
        handle.receipt_path = str(receipt)
        # Record the launched wrapper pid while we still have it. grok-run
        # prints it once, on this stdout, and nothing else ever links a task id
        # to a process. Without it, liveness is unknowable.
        pid = extract_background_pid(combined)
        if pid is not None:
            handle.launch_pid = pid
            self._update_receipt(handle.task_id, extra={"launch_pid": pid})
        return handle

    def _run(
        self,
        argv: Sequence[str],
        *,
        dry_run: bool,
        check: bool = True,
        timeout: float = 120.0,
    ) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        if dry_run:
            env["GROK_DRYRUN"] = "1"
        else:
            env.pop("GROK_DRYRUN", None)
        try:
            # grok-run dry-run quotes the executor --rules string with bash %q;
            # em-dashes in that string are not always valid UTF-8 on stdout.
            result = subprocess.run(
                list(argv),
                cwd=str(self.workspace),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise GrokNotAvailable(
                f"grok-run executable was not found when launching {argv[0]!r}: {exc}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise GrokRunError(
                f"grok-run timed out after {timeout}s: {argv}",
                argv=argv,
                stdout=exc.stdout.decode("utf-8", "replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or ""),
                stderr=exc.stderr.decode("utf-8", "replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or ""),
            ) from exc
        if check and result.returncode != 0:
            self._raise_from_failure(argv, result, (result.stderr or "") + "\n" + (result.stdout or ""))
        return result

    def _raise_from_failure(
        self,
        argv: Sequence[str],
        result: subprocess.CompletedProcess,
        combined: str,
    ) -> None:
        if "contract rejected" in combined:
            block = _LINT_BLOCK.search(combined)
            detail = block.group(1).strip() if block else combined.strip()
            codes = _lint_codes(detail)
            raise GrokContractError(
                "grok-run rejected the contract before launch: " + detail,
                codes=codes,
            )
        raise GrokRunError(
            f"grok-run failed (exit {result.returncode}): "
            + (combined.strip() or "no output"),
            argv=argv,
            returncode=result.returncode,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

    def _write_contract_file(self, task: str, contract_text: str) -> Path:
        dest_dir = self.receipts_dir / "contracts"
        dest_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        path = dest_dir / f"{task}-{stamp}.md"
        atomic_write_text(path, contract_text)
        return path

    def _write_receipt(
        self,
        handle: GrokRunHandle,
        *,
        mode: str,
        mutation_serialized: bool,
        contract_path: Optional[str],
        prompt: Optional[str],
        returncode: int,
    ) -> Path:
        self.receipts_dir.mkdir(parents=True, exist_ok=True)
        path = self.receipt_path(handle.task_id)
        report = self._report_path_for_dir(handle.task_dir)
        status: Optional[Dict[str, Any]]
        if handle.dry_run:
            status = None
        else:
            status = {"state": "running", "exit_code": None}
        executable = handle.command_run[0] if handle.command_run else None
        receipt = {
            "task_id": handle.task_id,
            "mode": mode,
            "command_run": list(handle.command_run),
            "executable": executable,
            "dry_run": handle.dry_run,
            "resolved_command": handle.resolved_command,
            "timestamps": {
                "started_at": handle.started_at,
                "finished_at": _now_iso() if handle.dry_run else None,
            },
            "status": status,
            "grok_state": None if handle.dry_run or status is None else status.get("state"),
            "report_path": str(report) if report is not None else None,
            "compact_path": None,
            "task_dir": handle.task_dir,
            "mutation_serialized": mutation_serialized,
            "contract_path": contract_path,
            "prompt": prompt,
            "workspace": str(self.workspace),
            "returncode": returncode,
            "launch_pid": getattr(handle, "launch_pid", None),
            "retries": None,
            "retries_reason": "telemetry.json absent",
            "verifier_command": None,
            "verifier_command_reason": (
                "not observed: GrokBridge does not execute the WorkUnit verifier"
            ),
            "verifier_outcome": None,
            "verifier_outcome_reason": (
                "not observed: GrokBridge does not execute the WorkUnit verifier"
            ),
            "throttle_evidence": None,
            "throttle_evidence_reason": "not present in telemetry.json",
            "failure_evidence": None,
            "failure_evidence_reason": "grok-stderr.log absent or empty",
        }
        self._fill_observed_receipt(receipt)
        atomic_write_json(path, receipt)
        return path

    def _read_receipt(self, task_id: str) -> Optional[Dict[str, Any]]:
        path = self.receipt_path(task_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _update_receipt(self, task_id: str, **fields: Any) -> None:
        current = self._read_receipt(task_id)
        if current is None:
            return
        extra = fields.pop("extra", None)
        current.update(fields)
        if extra:
            current.update(extra)
        ts = current.get("timestamps")
        if not isinstance(ts, dict):
            ts = {}
            current["timestamps"] = ts
        if fields.get("status") is not None:
            state = None
            status = fields.get("status")
            if isinstance(status, dict):
                state = status.get("state")
                if "successful" not in status:
                    status = dict(status)
                    status["successful"] = grok_succeeded(status)
                    current["status"] = status
            if state in {"done", "failed", "stale-running", "cancelled"}:
                ts["finished_at"] = _now_iso()
            if state is not None:
                current["grok_state"] = state
        self._fill_observed_receipt(current)
        path = self.receipt_path(task_id)
        atomic_write_json(path, current)

    def _fill_observed_receipt(self, receipt: Dict[str, Any]) -> None:
        """Copy observable fields. Null plus a reason when a value cannot be seen."""
        command_run = receipt.get("command_run")
        if isinstance(command_run, list) and command_run:
            receipt["executable"] = command_run[0]
        elif receipt.get("executable") is None:
            receipt["executable"] = None
            receipt.setdefault(
                "executable_reason",
                "command_run is empty; executable not observed",
            )

        status = receipt.get("status")
        if isinstance(status, dict):
            if "successful" not in status:
                status = dict(status)
                status["successful"] = grok_succeeded(status)
                receipt["status"] = status
            if status.get("state") is not None:
                receipt["grok_state"] = status.get("state")

        task_id = str(receipt.get("task_id") or "")
        task_dir = receipt.get("task_dir")
        retries_info = _read_retries(task_dir)
        receipt["retries"] = retries_info["retries"]
        if retries_info["retries"] is None:
            receipt["retries_reason"] = retries_info["reason"]
        else:
            receipt.pop("retries_reason", None)

        compact = receipt.get("compact_path")
        if not compact and task_id:
            candidate = self.receipts_dir / f"{task_id}.compact.json"
            if candidate.is_file():
                compact = str(candidate)
        receipt["compact_path"] = compact if compact else None

        if receipt.get("verifier_command"):
            receipt.pop("verifier_command_reason", None)
        else:
            receipt["verifier_command"] = None
            receipt.setdefault(
                "verifier_command_reason",
                "not observed: GrokBridge does not execute the WorkUnit verifier",
            )
        if receipt.get("verifier_outcome") is not None:
            receipt.pop("verifier_outcome_reason", None)
        else:
            receipt["verifier_outcome"] = None
            receipt.setdefault(
                "verifier_outcome_reason",
                "not observed: GrokBridge does not execute the WorkUnit verifier",
            )

        throttle = _read_throttle(task_dir)
        receipt["throttle_evidence"] = throttle["value"]
        if throttle["value"] is None:
            receipt["throttle_evidence_reason"] = throttle["reason"]
        else:
            receipt.pop("throttle_evidence_reason", None)

        failure = _read_failure_evidence(task_dir)
        receipt["failure_evidence"] = failure["value"]
        if failure["value"] is None:
            receipt["failure_evidence_reason"] = failure["reason"]
        else:
            receipt.pop("failure_evidence_reason", None)

        if receipt.get("launch_pid") is None:
            receipt["launch_pid"] = None
            receipt.setdefault(
                "launch_pid_reason",
                "not observed at launch",
            )
        else:
            receipt.pop("launch_pid_reason", None)

    def _report_path(self, task_id: str) -> Optional[Path]:
        receipt = self._read_receipt(task_id)
        if receipt:
            recorded = receipt.get("report_path")
            if recorded:
                p = Path(str(recorded))
                if p.is_file():
                    return p
            task_dir = receipt.get("task_dir")
            found = self._report_path_for_dir(task_dir)
            if found is not None:
                return found
        return self._report_path_for_dir(DEFAULT_TASKS_ROOT / str(task_id))

    @staticmethod
    def _report_path_for_dir(task_dir: Optional[Union[str, Path]]) -> Optional[Path]:
        if not task_dir:
            return None
        directory = Path(task_dir)
        path = directory / "grok-report.md"
        if path.is_file() or directory.is_dir():
            return path
        return None
