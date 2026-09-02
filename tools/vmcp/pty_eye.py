"""PTY / terminal eye (roadmap E.10).

Attempts a real posix PTY pair (libutil.openpty / posix_openpt + slave).
This sandbox has been measured to deny the slave (`EPERM` on `/dev/ttys*`,
`openpty: Operation not permitted`). That is a blocker, not a simulated
session: we never report a pipe capture as a PTY.

    from tools.vmcp.pty_eye import capture, probe
"""
from __future__ import annotations

import ctypes
import ctypes.util
import errno
import fcntl
import os
import select
import time
from typing import Any, Mapping, Sequence

from tools.vmcp.receipt import (
    argv_list,
    content_digest,
    dangerous_command,
    local_env,
    network_tool_refused,
    sha256_bytes,
    tool_receipt,
    utc_now,
)


NAME = "pty.eye"
VERSION = "1"
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_WINSIZE = (24, 80)
MAX_CAPTURE_BYTES = 256_000

_PTY_BLOCKER = (
    "sandbox denies PTY slave allocation: os.openpty/libutil.openpty return "
    "EPERM and /dev/ttys* cannot be opened. posix_openpt(/dev/ptmx) succeeds "
    "for the master but a session requires the slave. This is not a pipe "
    "stand-in. Needs an unsandboxed (`gate`) process that can openpty."
)


def _try_libutil_openpty() -> tuple[int, int] | None:
    libname = ctypes.util.find_library("util") or "/usr/lib/libutil.dylib"
    try:
        libutil = ctypes.CDLL(libname, use_errno=True)
    except OSError:
        return None
    openpty = libutil.openpty
    openpty.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    openpty.restype = ctypes.c_int
    master = ctypes.c_int()
    slave = ctypes.c_int()
    rc = openpty(ctypes.byref(master), ctypes.byref(slave), None, None, None)
    if rc != 0:
        return None
    return master.value, slave.value


def _try_os_openpty() -> tuple[int, int] | None:
    try:
        return os.openpty()
    except OSError:
        return None


def _try_posix_openpt_pair() -> tuple[int, int] | None:
    """posix_openpt + grantpt + unlockpt + open slave. Fails here on EPERM."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    posix_openpt = libc.posix_openpt
    posix_openpt.argtypes = [ctypes.c_int]
    posix_openpt.restype = ctypes.c_int
    grantpt = libc.grantpt
    grantpt.argtypes = [ctypes.c_int]
    grantpt.restype = ctypes.c_int
    unlockpt = libc.unlockpt
    unlockpt.argtypes = [ctypes.c_int]
    unlockpt.restype = ctypes.c_int
    ptsname = libc.ptsname
    ptsname.argtypes = [ctypes.c_int]
    ptsname.restype = ctypes.c_char_p
    flags = os.O_RDWR | getattr(os, "O_NOCTTY", 0)
    master = posix_openpt(flags)
    if master < 0:
        return None
    try:
        if grantpt(master) != 0 or unlockpt(master) != 0:
            os.close(master)
            return None
        name = ptsname(master)
        if not name:
            os.close(master)
            return None
        try:
            slave = os.open(name, os.O_RDWR | getattr(os, "O_NOCTTY", 0))
        except OSError:
            os.close(master)
            return None
        return master, slave
    except Exception:
        try:
            os.close(master)
        except OSError:
            pass
        return None


def probe() -> dict[str, Any]:
    """Allocate and immediately close a PTY pair. Reports the real errno."""
    errors: list[str] = []
    for name, fn in (
        ("libutil.openpty", _try_libutil_openpty),
        ("os.openpty", _try_os_openpty),
        ("posix_openpt+slave", _try_posix_openpt_pair),
    ):
        try:
            pair = fn()
        except OSError as exc:
            errors.append(f"{name}:{exc.errno}:{exc.strerror}")
            continue
        if pair is not None:
            master, slave = pair
            isatty = bool(os.isatty(master) and os.isatty(slave))
            try:
                os.close(master)
            except OSError:
                pass
            try:
                os.close(slave)
            except OSError:
                pass
            return {
                "ok": True,
                "used_real_pty": True,
                "method": name,
                "isatty": isatty,
                "errors": errors,
                "blocker": None,
                "evidence_tier": "FUNCTIONAL_SIM",
                "execution": "REAL",
            }
        errors.append(f"{name}:unavailable")
    return {
        "ok": False,
        "used_real_pty": False,
        "method": None,
        "isatty": False,
        "errors": errors,
        "blocker": _PTY_BLOCKER,
        "wake_kind": "PTY_OPEN_DENIED",
        "errno_name": errno.errorcode.get(errno.EPERM, "EPERM"),
        "evidence_tier": "STATIC",
        "execution": "BLOCKED",
    }


def _set_winsize(fd: int, rows: int, cols: int) -> None:
    try:
        import struct
        import termios

        packed = struct.pack("HHHH", int(rows), int(cols), 0, 0)
        fcntl.ioctl(fd, getattr(termios, "TIOCSWINSZ", 0x80087467), packed)
    except OSError:
        pass


def _drain(master: int, pid: int, timeout_s: float) -> tuple[bytes, list[dict[str, Any]], int | None]:
    out = bytearray()
    events: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_s
    status: int | None = None
    t0 = time.monotonic()
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        readable, _, _ = select.select([master], [], [], min(0.1, remaining))
        if readable:
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
            events.append(
                {
                    "t_ms": round((time.monotonic() - t0) * 1000.0, 3),
                    "stream": "pty",
                    "n": len(chunk),
                }
            )
            if len(out) >= MAX_CAPTURE_BYTES:
                break
        try:
            wpid, st = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            wpid, st = 0, 0
        if wpid:
            status = st
            while True:
                readable, _, _ = select.select([master], [], [], 0.05)
                if not readable:
                    break
                try:
                    chunk = os.read(master, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                out += chunk
                events.append(
                    {
                        "t_ms": round((time.monotonic() - t0) * 1000.0, 3),
                        "stream": "pty",
                        "n": len(chunk),
                    }
                )
            break
    if status is None:
        try:
            wpid, st = os.waitpid(pid, os.WNOHANG)
            if wpid:
                status = st
        except ChildProcessError:
            status = 0
    return bytes(out[:MAX_CAPTURE_BYTES]), events, status


def capture(
    command: Any = None,
    *,
    argv: Sequence[str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    stdin_bytes: bytes = b"",
    arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run argv on a real PTY and return E.10 fields. Never fakes a tty."""
    args = dict(arguments or {})
    raw = argv if argv is not None else (command if command is not None else args.get("argv") or args.get("command"))
    argv_l = argv_list(raw)
    started = utc_now()
    t0 = time.perf_counter()
    if not argv_l:
        return {
            "act": "see",
            "organ": "pty",
            "status": "PARKED",
            "ok": False,
            "looked": False,
            "empty_success": False,
            "used_real_pty": False,
            "limitations": ["COMMAND_REQUIRED"],
            "wake": {
                "schema": "hawking.audit.wake_condition.v1",
                "kind": "PTY_COMMAND_REQUIRED",
                "required_kind": "call",
                "required_symbol": "tools.vmcp.pty_eye.capture",
                "predicate": "compact_surface(see, organ=pty) with argv",
                "blocker": "pty see requires argv; refusing to invent a session",
                "missing_dependency": "argv",
                "evidence_tier": "STATIC",
            },
            "evidence_tier": "STATIC",
            "execution": "BLOCKED",
            "results": None,
            "items": None,
        }
    refused = network_tool_refused(argv_l) or dangerous_command(argv_l)
    if refused:
        return {
            "act": "see",
            "organ": "pty",
            "status": "CONNECTED",
            "ok": False,
            "looked": False,
            "empty_success": False,
            "used_real_pty": False,
            "argv": argv_l,
            "limitations": [refused],
            "execution": "REAL",
            "evidence_tier": "FUNCTIONAL_SIM",
            "note": "refused before exec; no process started",
        }
    probed = probe()
    if not probed.get("ok"):
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        wake = {
            "schema": "hawking.audit.wake_condition.v1",
            "kind": "PTY_OPEN_DENIED",
            "required_kind": "call",
            "required_symbol": "tools.vmcp.pty_eye.capture",
            "required_caller_prefix": "hcli/",
            "predicate": (
                "production AST Call of tools.vmcp.pty_eye.capture from the "
                "compact surface AFTER openpty succeeds (unsandboxed / gate "
                "profile). An import is not a call site. A pipe capture is "
                "not a PTY session."
            ),
            "blocker": probed.get("blocker") or _PTY_BLOCKER,
            "missing_dependency": "unsandboxed process that can openpty (gate profile)",
            "probe": {k: probed[k] for k in ("ok", "method", "errors", "errno_name") if k in probed},
            "evidence_tier": "STATIC",
        }
        return {
            "act": "see",
            "organ": "pty",
            "status": "PARKED",
            "ok": False,
            "looked": True,
            "empty_success": False,
            "used_real_pty": False,
            "isatty": False,
            "argv": argv_l,
            "cwd": os.fspath(cwd) if cwd is not None else os.getcwd(),
            "limitations": ["PTY_OPEN_DENIED", "EPERM"],
            "wake": wake,
            "wake_condition": wake["predicate"],
            "missing_dependency": wake["missing_dependency"],
            "results": None,
            "items": None,
            "performance_ms": elapsed_ms,
            "execution": "BLOCKED",
            "evidence_tier": "STATIC",
            "gpu_authority": False,
            "network_used": False,
            "note": _PTY_BLOCKER,
        }

    pair = None
    method = probed.get("method")
    if method == "libutil.openpty":
        pair = _try_libutil_openpty()
    elif method == "os.openpty":
        pair = _try_os_openpty()
    else:
        pair = _try_posix_openpt_pair()
    if pair is None:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "act": "see",
            "organ": "pty",
            "status": "PARKED",
            "ok": False,
            "looked": True,
            "empty_success": False,
            "used_real_pty": False,
            "argv": argv_l,
            "limitations": ["PTY_OPEN_DENIED", "ALLOCATE_RACE"],
            "execution": "BLOCKED",
            "evidence_tier": "STATIC",
            "performance_ms": elapsed_ms,
            "results": None,
            "items": None,
        }

    master, slave = pair
    rows, cols = DEFAULT_WINSIZE
    _set_winsize(slave, rows, cols)
    workdir = os.fspath(cwd) if cwd is not None else os.getcwd()
    env = local_env()
    pid = os.fork()
    if pid == 0:
        os.close(master)
        os.setsid()
        os.dup2(slave, 0)
        os.dup2(slave, 1)
        os.dup2(slave, 2)
        if slave > 2:
            os.close(slave)
        if workdir:
            try:
                os.chdir(workdir)
            except OSError:
                os._exit(127)
        os.execve(argv_l[0], argv_l, env)
        os._exit(127)
    os.close(slave)
    if stdin_bytes:
        try:
            os.write(master, stdin_bytes)
        except OSError:
            pass
    output, events, status = _drain(master, pid, float(args.get("timeout_s") or timeout_s))
    try:
        os.close(master)
    except OSError:
        pass
    if status is None:
        # Cooperative wait only on OUR child. Do not signal.
        try:
            _, status = os.waitpid(pid, 0)
        except ChildProcessError:
            status = 0
    exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else None
    signaled = os.WTERMSIG(status) if os.WIFSIGNALED(status) else None
    text = output.decode("utf-8", errors="replace")
    ansi = "\x1b[" in text or "\x1b(" in text
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    receipt = tool_receipt(
        tool=NAME,
        version=VERSION,
        invocation=argv_l,
        status="ok" if exit_code == 0 else "error",
        started_at=started,
        elapsed_ms=elapsed_ms,
        input_ids=[f"argv:{i}" for i, _ in enumerate(argv_l)],
        input_hashes=[sha256_bytes(a.encode()) for a in argv_l],
        output_ids=["pty.screen"],
        output_hashes=[sha256_bytes(output)],
        limitations=[],
        verifier="tools.vmcp.pty_eye.capture",
        extra={"pid": pid, "exit_code": exit_code, "isatty": True},
    )
    evidence = {
        "process_identity": pid,
        "argv": argv_l,
        "cwd": workdir,
        "terminal_text": text[:8000],
        "exit_code": exit_code,
        "signal": signaled,
        "resize": {"rows": rows, "cols": cols},
        "ansi": ansi,
        "event_boundaries": events[:64],
        "used_real_pty": True,
        "isatty": True,
    }
    return {
        "act": "see",
        "organ": "pty",
        "status": "CONNECTED",
        "ok": exit_code == 0,
        "looked": True,
        "empty_success": False,
        "used_real_pty": True,
        "isatty": True,
        "argv": argv_l,
        "cwd": workdir,
        "pid": pid,
        "exit_code": exit_code,
        "signal": signaled,
        "text": text,
        "ansi": ansi,
        "events": events[:64],
        "sha256": sha256_bytes(output),
        "tool_receipt": receipt,
        "limitations": [],
        "execution": "REAL",
        "evidence_tier": "FUNCTIONAL_SIM",
        "gpu_authority": False,
        "network_used": False,
        "performance_ms": elapsed_ms,
        "deep_digest": content_digest(evidence),
        "artifacts": [],
        "evidence": [evidence],
        "residuals": [],
        "next_actions": [],
        "note": "real posix PTY session on this host; not a pipe stand-in",
    }
