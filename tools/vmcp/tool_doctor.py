"""Tool doctor (roadmap E.3 / E.4) hosted on Hawking, not visionmcp.

Profiles a real local invocation into a ToolReceipt: availability, version,
permissions, health, known limits, fallback/refusal. Network tools and
dangerous argv are refused before exec. subprocess never uses shell=True.

    from tools.vmcp.tool_doctor import profile, report
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.vmcp.receipt import (
    argv_list,
    basename_of,
    content_digest,
    dangerous_command,
    local_env,
    network_tool_refused,
    sha256_bytes,
    tool_receipt,
    utc_now,
)


NAME = "tool.doctor"
VERSION = "1"
DEFAULT_TIMEOUT_S = 8.0
MAX_CAPTURE = 64_000

# E.3 required classes. CONNECTED means this Hawking sidecar hosts a real
# local implementation. PARKED means the foreign visionmcp extra / hardware.
E3_CLASSES: tuple[tuple[str, str, str], ...] = (
    ("file classifier", "CONNECTED", "tools.vmcp.file_eye.observe"),
    ("hashing", "CONNECTED", "hashlib"),
    ("archive/compression", "CONNECTED", "zipfile/tarfile/gzip"),
    ("browser/CDP", "PARKED", "visionmcp web extra + host Chrome"),
    ("HTML/DOM capture", "PARKED", "visionmcp web extra"),
    ("CSS parser", "PARKED", "visionmcp web extra"),
    ("source-map parser", "PARKED", "visionmcp opener"),
    ("visual diff", "PARKED", "visionmcp compiler residual"),
    ("image handling", "CONNECTED", "PNG/JPEG/GIF magic in tools.vmcp.file_eye (no PIL)"),
    ("OBJ/GLTF parser", "PARKED", "visionmcp 3d extra"),
    ("spatial validator", "PARKED", "visionmcp 3d extra + Blender"),
    ("independent renderer/viewer", "PARKED", "Blender CLI"),
    ("PTY capture", "PARKED", "openpty EPERM in this sandbox; tools.vmcp.pty_eye"),
    ("process inspection", "CONNECTED", "tools.vmcp.tool_doctor.profile"),
    ("profiling hooks", "CONNECTED", "subprocess + resource of local argv"),
)


def _resolve(argv: Sequence[str]) -> tuple[list[str], str | None]:
    if not argv:
        return [], "COMMAND_REQUIRED"
    first = argv[0]
    if os.path.sep in first or first.startswith("."):
        path = Path(first)
        if not path.exists():
            return list(argv), "TOOL_ABSENT"
        resolved = str(path.resolve())
        return [resolved, *argv[1:]], None
    found = shutil.which(first)
    if not found:
        return list(argv), "TOOL_ABSENT"
    return [found, *argv[1:]], None


def _version_probe(executable: str, timeout_s: float = 2.0) -> str | None:
    env = local_env()
    for flag in ("--version", "-version", "-V"):
        try:
            proc = subprocess.run(
                [executable, flag],
                capture_output=True,
                timeout=timeout_s,
                env=env,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        text = (proc.stdout or b"") + (proc.stderr or b"")
        line = text.decode("utf-8", errors="replace").strip().splitlines()
        if line:
            return line[0][:200]
    return None


def _file_perms(path: str) -> dict[str, Any]:
    try:
        st = os.stat(path)
    except OSError as exc:
        return {"error": str(exc)}
    mode = st.st_mode
    return {
        "mode": oct(mode),
        "uid": st.st_uid,
        "gid": st.st_gid,
        "executable": bool(mode & stat.S_IXUSR),
        "readable": bool(mode & stat.S_IRUSR),
        "size": st.st_size,
    }


def profile(
    command: Any = None,
    *,
    argv: Sequence[str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    arguments: Mapping[str, Any] | None = None,
    stdin_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Run argv locally and emit an E.4 ToolReceipt. Real process, no network."""
    args = dict(arguments or {})
    raw = argv if argv is not None else (command if command is not None else args.get("argv") or args.get("command") or args.get("tool"))
    argv_l = argv_list(raw)
    started = utc_now()
    t0 = time.perf_counter()
    if not argv_l:
        return {
            "act": "check",
            "organ": "tool_doctor",
            "status": "CONNECTED",
            "ok": False,
            "looked": False,
            "empty_success": False,
            "limitations": ["COMMAND_REQUIRED"],
            "execution": "REAL",
            "evidence_tier": "FUNCTIONAL_SIM",
            "note": "tool doctor requires argv; refusing to invent an invocation",
        }
    refused = network_tool_refused(argv_l) or dangerous_command(argv_l)
    if refused:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        receipt = tool_receipt(
            tool=basename_of(argv_l) or NAME,
            invocation=argv_l,
            status="refused",
            started_at=started,
            elapsed_ms=elapsed_ms,
            limitations=[refused],
            verifier="tools.vmcp.tool_doctor.profile",
        )
        return {
            "act": "check",
            "organ": "tool_doctor",
            "status": "CONNECTED",
            "ok": False,
            "looked": False,
            "empty_success": False,
            "argv": argv_l,
            "available": False,
            "limitations": [refused],
            "tool_receipt": receipt,
            "execution": "REAL",
            "evidence_tier": "FUNCTIONAL_SIM",
            "gpu_authority": False,
            "network_used": False,
            "note": "refused before exec",
        }
    resolved, absence = _resolve(argv_l)
    if absence:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        receipt = tool_receipt(
            tool=basename_of(argv_l) or NAME,
            invocation=argv_l,
            status="absent",
            started_at=started,
            elapsed_ms=elapsed_ms,
            limitations=[absence],
            verifier="tools.vmcp.tool_doctor.profile",
        )
        return {
            "act": "check",
            "organ": "tool_doctor",
            "status": "CONNECTED",
            "ok": False,
            "looked": True,
            "empty_success": False,
            "argv": argv_l,
            "available": False,
            "limitations": [absence],
            "tool_receipt": receipt,
            "execution": "REAL",
            "evidence_tier": "FUNCTIONAL_SIM",
            "note": "availability probe: tool is not on PATH / not on disk",
        }
    executable = resolved[0]
    perms = _file_perms(executable)
    version = _version_probe(executable)
    env = local_env()
    workdir = os.fspath(cwd) if cwd is not None else None
    try:
        proc = subprocess.run(
            resolved,
            capture_output=True,
            timeout=float(args.get("timeout_s") or timeout_s),
            cwd=workdir,
            env=env,
            check=False,
            input=stdin_bytes,
        )
        timed_out = False
        exception = None
    except subprocess.TimeoutExpired as exc:
        proc = None
        timed_out = True
        exception = type(exc).__name__
    except OSError as exc:
        proc = None
        timed_out = False
        exception = f"{type(exc).__name__}:{exc.errno}"
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    limitations: list[str] = []
    stdout = b""
    stderr = b""
    exit_code: int | None = None
    if timed_out:
        limitations.append("TIMEOUT")
        status = "timeout"
        ok = False
    elif proc is None:
        limitations.append(f"EXEC_FAILED:{exception}")
        status = "error"
        ok = False
    else:
        stdout = proc.stdout[:MAX_CAPTURE] if proc.stdout else b""
        stderr = proc.stderr[:MAX_CAPTURE] if proc.stderr else b""
        if proc.stdout and len(proc.stdout) > MAX_CAPTURE:
            limitations.append("STDOUT_TRUNCATED")
        if proc.stderr and len(proc.stderr) > MAX_CAPTURE:
            limitations.append("STDERR_TRUNCATED")
        exit_code = int(proc.returncode)
        status = "ok" if exit_code == 0 else "error"
        ok = exit_code == 0
    out_hash = sha256_bytes(stdout)
    err_hash = sha256_bytes(stderr)
    receipt = tool_receipt(
        tool=basename_of(resolved) or NAME,
        version=version,
        invocation=resolved,
        status=status,
        started_at=started,
        elapsed_ms=elapsed_ms,
        input_ids=[f"argv:{i}" for i, _ in enumerate(resolved)],
        input_hashes=[sha256_bytes(a.encode()) for a in resolved],
        output_ids=["stdout", "stderr"],
        output_hashes=[out_hash, err_hash],
        limitations=limitations,
        verifier="tools.vmcp.tool_doctor.profile",
        extra={
            "exit_code": exit_code,
            "cwd": workdir,
            "permissions": perms,
        },
    )
    evidence = {
        "available": True,
        "version": version,
        "permissions": perms,
        "health": status,
        "known_limits": limitations,
        "fallback": None if ok else "receipt-recorded-failure",
        "exit_code": exit_code,
        "stdout_sha256": out_hash,
        "stderr_sha256": err_hash,
        "stdout_text": stdout.decode("utf-8", errors="replace")[:4000],
        "stderr_text": stderr.decode("utf-8", errors="replace")[:4000],
    }
    return {
        "act": "check",
        "organ": "tool_doctor",
        "status": "CONNECTED",
        "ok": ok,
        "looked": True,
        "empty_success": False,
        "argv": resolved,
        "available": True,
        "version": version,
        "permissions": perms,
        "exit_code": exit_code,
        "stdout": evidence["stdout_text"],
        "stderr": evidence["stderr_text"],
        "stdout_sha256": out_hash,
        "stderr_sha256": err_hash,
        "limitations": limitations,
        "tool_receipt": receipt,
        "execution": "REAL",
        "evidence_tier": "FUNCTIONAL_SIM",
        "gpu_authority": False,
        "network_used": False,
        "performance_ms": elapsed_ms,
        "deep_digest": content_digest(evidence),
        "artifacts": [],
        "evidence": [evidence],
        "residuals": limitations,
        "next_actions": [] if ok else ["inspect tool_receipt.limitations"],
        "note": (
            "real local subprocess on this host; elapsed_ms is wall clock of "
            "the child, not a GPU measurement and not HARDWARE_MEASURED"
        ),
    }


def report(*, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """E.3 class table for this host. Does not import visionmcp."""
    del arguments
    from tools.vmcp.pty_eye import probe as pty_probe

    probed = pty_probe()
    classes = []
    for name, disposition, hosted_by in E3_CLASSES:
        row = {
            "class": name,
            "disposition": disposition,
            "hosted_by": hosted_by,
            "empty_success": False,
        }
        if name == "PTY capture":
            if probed.get("ok"):
                row["disposition"] = "CONNECTED"
                row["hosted_by"] = "tools.vmcp.pty_eye.capture"
            else:
                row["disposition"] = "PARKED"
                row["blocker"] = probed.get("blocker")
        classes.append(row)
    python = {
        "executable": sys.executable,
        "version": sys.version.split()[0],
        "available": True,
    }
    echo = shutil.which("echo") or "/bin/echo"
    git = shutil.which("git")
    return {
        "act": "check",
        "organ": "tool_doctor",
        "status": "CONNECTED",
        "ok": True,
        "looked": True,
        "empty_success": False,
        "scope": "hawking-local",
        "profile": "core-equivalent (no visionmcp)",
        "classes": classes,
        "python": python,
        "echo": echo,
        "git": git,
        "pty_probe": {
            "ok": bool(probed.get("ok")),
            "blocker": probed.get("blocker"),
            "errors": probed.get("errors"),
        },
        "network_used": False,
        "gpu_authority": False,
        "execution": "REAL",
        "evidence_tier": "FUNCTIONAL_SIM",
        "n_connected": sum(1 for c in classes if c["disposition"] == "CONNECTED"),
        "n_parked": sum(1 for c in classes if c["disposition"] == "PARKED"),
        "note": (
            "Hawking-hosted doctor of local classes. visionmcp system.doctor "
            "is a different symbol in a foreign package and is not called."
        ),
    }
