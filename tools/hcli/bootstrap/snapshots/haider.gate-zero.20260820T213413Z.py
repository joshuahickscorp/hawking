#!/usr/bin/env python3
"""haider — HCLI-v0 bootstrap. Gate Zero: haider 1.

Supervises one llama-server, drives the P0 tool-bridge observation loop,
performs a scoped edit, runs deterministic validation, and emits a
machine-generated receipt.

Usage:
    python tools/haider/haider.py 1
    python tools/haider/haider.py 1 --model /path/to/model.gguf --debug
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import P0 components from the same directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p0_tool_bridge as p0

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HAIDER_DIR = Path(".haider")
RECEIPTS_DIR = HAIDER_DIR / "receipts"
LOGS_DIR = HAIDER_DIR / "logs"
DEFAULT_HOST = "127.0.0.1"
READY_TIMEOUT_S = 60.0
SHUTDOWN_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# Port allocation
# ---------------------------------------------------------------------------


def allocate_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((DEFAULT_HOST, 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# llama-server supervision
# ---------------------------------------------------------------------------


def start_llama_server(
    model: str,
    port: int,
    host: str,
    ctx_size: int = 4096,
) -> subprocess.Popen:
    """Spawn llama-server and return the Popen handle."""
    cmd = [
        "llama-server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--ctx-size", str(ctx_size),
        "--no-webui",
    ]
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"llama_{port}.log"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._haider_log_file = log_file  # type: ignore[attr-defined]
    return proc


def wait_for_ready(port: int, host: str, timeout: float = READY_TIMEOUT_S) -> bool:
    """Poll /health until the server responds or timeout."""
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(0.25)
    return False


def stop_llama_server(proc: subprocess.Popen) -> None:
    """Gracefully terminate llama-server."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=2)
    except (ProcessLookupError, OSError):
        pass
    finally:
        log_file = getattr(proc, "_haider_log_file", None)
        if log_file:
            log_file.close()


# ---------------------------------------------------------------------------
# Scoped edit validation (pure, deterministic, testable)
# ---------------------------------------------------------------------------


def validate_scoped_edit(
    guard: p0.RepositoryGuard,
    path: str,
    old_text: str,
    new_text: str,
) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """Validate a scoped edit against the working tree.

    Returns (ok, error_message, full_path, original_content).
    On success, full_path and original_content are set.
    """
    if not path or not isinstance(path, str) or not path.strip():
        return False, "path is empty", None, None
    if not old_text or not isinstance(old_text, str):
        return False, "old_text is empty", None, None
    if not new_text or not isinstance(new_text, str):
        return False, "new_text is empty", None, None
    if old_text == new_text:
        return False, "no-op edit: old_text equals new_text", None, None

    try:
        full_path = guard.resolve(path)
    except p0.ToolError as e:
        return False, f"path rejected: {e}", None, None

    if not os.path.isfile(full_path):
        return False, f"path is not a regular file: {path}", None, None

    try:
        original = Path(full_path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return False, f"cannot read file: {e}", None, None

    if old_text not in original:
        return False, "old_text not found in file", None, None

    match_count = original.count(old_text)
    if match_count > 1:
        return False, f"old_text matches {match_count} locations (prefer exactly 1)", None, None

    return True, "", full_path, original


# ---------------------------------------------------------------------------
# Scoped edit
# ---------------------------------------------------------------------------


def _extract_edit_json(content: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON edit object from model output."""
    content = content.strip()
    try:
        edit = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return None
        try:
            edit = json.loads(m.group())
        except json.JSONDecodeError:
            return None
    if not isinstance(edit, dict):
        return None
    return edit


def apply_scoped_edit(
    client: p0.ModelClient,
    guard: p0.RepositoryGuard,
    executor: p0.ToolExecutor,
    obs_result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Ask the model for a scoped edit grounded in observed evidence.

    Uses up to 2 attempts. On the second attempt, the actual file content
    is provided so the model can produce an exact old_text.
    """
    obs_final = obs_result.get("final", "")
    obs_stats = obs_result.get("stats", {})

    system_prompt = (
        "You are HAIDER. You have already inspected the repository and gathered evidence.\n"
        "Based on that evidence, make exactly one minimal, safe, reversible edit "
        "to a single file in this repository.\n"
        "The edit must be small (1-5 lines changed).\n"
        "Respond with ONLY a JSON object: "
        '{"path": "relative/path", "old_text": "exact text to replace", '
        '"new_text": "replacement text"}\n'
        "old_text must be an exact substring of the current file content.\n"
        "old_text must appear exactly once in the file.\n"
        "No prose. No markdown. No code fences."
    )

    user_prompt = (
        f"OBSERVATION EVIDENCE (from your inspection):\n"
        f"{obs_final}\n\n"
        f"Stats: {json.dumps(obs_stats)}\n\n"
        "Propose one small, safe edit. Return ONLY the JSON edit object."
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(2):
        try:
            content, _ = client.chat(messages)
        except p0.ModelError:
            return None

        edit = _extract_edit_json(content)
        if edit is None:
            if attempt == 0:
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": "Invalid response. Return ONLY the JSON edit object."})
                continue
            return None

        path = edit.get("path", "")
        old_text = edit.get("old_text", "")
        new_text = edit.get("new_text", "")

        ok, err, full_path, original = validate_scoped_edit(guard, path, old_text, new_text)
        if ok:
            updated = original.replace(old_text, new_text, 1)
            old_sha = hashlib.sha256(original.encode()).hexdigest()
            new_sha = hashlib.sha256(updated.encode()).hexdigest()
            Path(full_path).write_text(updated, encoding="utf-8")
            diff_lines = abs(len(updated.splitlines()) - len(original.splitlines()))
            return {
                "path": path,
                "old_sha256": old_sha,
                "new_sha256": new_sha,
                "diff_lines": diff_lines,
            }

        # Validation failed. On first attempt, read the file and retry with content.
        if attempt == 0:
            messages.append({"role": "assistant", "content": content})
            # Try to read the file to give the model exact content.
            file_content = ""
            if path:
                try:
                    fp = guard.resolve(path)
                    if os.path.isfile(fp):
                        file_content = Path(fp).read_text(encoding="utf-8", errors="ignore")[:4000]
                except p0.ToolError:
                    pass
            retry_msg = (
                f"Edit rejected: {err}\n"
            )
            if file_content:
                retry_msg += (
                    f"\nActual content of {path} (first 4000 chars):\n"
                    f"{file_content}\n\n"
                    "Propose the edit again with an exact old_text from the content above. "
                    "Return ONLY the JSON edit object."
                )
            else:
                retry_msg += (
                    "\nChoose a different file or fix old_text. "
                    "Return ONLY the JSON edit object."
                )
            messages.append({"role": "user", "content": retry_msg})
            continue

        return None

    return None


# ---------------------------------------------------------------------------
# Deterministic validation
# ---------------------------------------------------------------------------


def run_validation(guard: p0.RepositoryGuard) -> Dict[str, Any]:
    """Run the P0 deterministic test suite."""
    test_file = Path("tools/haider/test_p0_tool_bridge.py")
    if not test_file.exists():
        return {
            "command": "none",
            "exit_code": 1,
            "duration_ms": 0,
            "stdout": "",
            "stderr": "test file not found",
        }

    cmd = [sys.executable, str(test_file)]
    started = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=guard.root,
        )
        duration_ms = int((time.monotonic() - started) * 1000)
        return {
            "command": " ".join(cmd),
            "exit_code": result.returncode,
            "duration_ms": duration_ms,
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-2000:],
        }
    except subprocess.TimeoutExpired:
        duration_ms = int((time.monotonic() - started) * 1000)
        return {
            "command": " ".join(cmd),
            "exit_code": 124,
            "duration_ms": duration_ms,
            "stdout": "",
            "stderr": "timeout",
        }


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def write_receipt(receipt: Dict[str, Any]) -> Path:
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = RECEIPTS_DIR / f"gate_zero_{ts}.json"
    path.write_text(json.dumps(receipt, indent=2))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(prog="haider", description="HCLI-v0 bootstrap")
    parser.add_argument(
        "n",
        type=int,
        nargs="?",
        default=1,
        help="Number of independent runtimes (Gate Zero: 1)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("HAIDER_MODEL_PATH", "model.gguf"),
        help="Path to model artifact",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host")
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=4096,
        help="Context size for llama-server",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="Max observation turns for the P0 session",
    )
    parser.add_argument("--debug", action="store_true", help="Debug output")
    args = parser.parse_args()

    if args.n != 1:
        print(f"haider: Gate Zero supports N=1 only (got {args.n})", file=sys.stderr)
        return 2

    HAIDER_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Allocate port.
    port = allocate_port()
    print(f"[haider] port={port}")

    # 2. Start llama-server.
    proc = start_llama_server(args.model, port, args.host, args.ctx_size)
    print(f"[haider] llama-server pid={proc.pid}")

    try:
        # 3. Wait for readiness.
        if not wait_for_ready(port, args.host):
            print("[haider] FATAL: server not ready in time", file=sys.stderr)
            return 1
        print("[haider] server ready")

        # 4. Create P0 components.
        guard = p0.RepositoryGuard.detect(os.getcwd())
        executor = p0.ToolExecutor(guard)
        model_timeout = float(os.environ.get("HAIDER_MODEL_TIMEOUT", "1800"))

        client = p0.ModelClient(
            api_base=f"http://{args.host}:{port}/v1",
            model="local",
            api_key="sk-local",
            timeout=model_timeout,
            max_tokens=2048,
            debug=args.debug,
        )

        print(f"[haider] model timeout={model_timeout:.0f}s")

        # 5. Observation phase: P0 Session model->tool->observation loop.
        print("[haider] observation phase...")
        session = p0.Session(
            "parent",
            (
                "Inspect the current repository state. Check git status, list "
                "key files, and identify one small safe edit you could make. "
                "Use read-only tools. When done, respond with a final JSON "
                "object summarizing your findings."
            ),
            client,
            executor,
            guard,
            max_turns=args.max_turns,
            debug=args.debug,
            emit=lambda msg: print(f"  {msg}"),
        )
        obs_result = session.run()
        print(
            f"[haider] observation done: ok={obs_result['ok']} "
            f"turns={obs_result['stats']['model_turns']} "
            f"tools={obs_result['stats']['tool_calls']} "
            f"elapsed={obs_result['elapsed_s']}s"
        )

        # 6. Edit phase (grounded in observation evidence).
        print("[haider] edit phase...")
        edit = apply_scoped_edit(client, guard, executor, obs_result)
        if edit is None:
            print("[haider] FATAL: no valid edit produced", file=sys.stderr)
            return 1
        print(f"[haider] edit: {edit['path']} ({edit['diff_lines']} lines changed)")

        # 7. Validation phase.
        print("[haider] validation phase...")
        validation = run_validation(guard)
        print(
            f"[haider] validation: exit={validation['exit_code']} "
            f"({validation['duration_ms']}ms)"
        )

        # 8. Receipt.
        status = "PASS" if validation["exit_code"] == 0 else "FAIL"
        receipt = {
            "gate": "zero",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runtime_pid": proc.pid,
            "port": port,
            "model": args.model,
            "observation": {
                "ok": obs_result["ok"],
                "model_turns": obs_result["stats"]["model_turns"],
                "tool_calls": obs_result["stats"]["tool_calls"],
                "protocol_errors": obs_result["stats"]["protocol_errors"],
                "elapsed_s": obs_result["elapsed_s"],
            },
            "edit": edit,
            "validation": validation,
            "usage": client.usage,
            "status": status,
        }
        receipt_path = write_receipt(receipt)
        print(f"[haider] receipt: {receipt_path}")
        print(f"[haider] status: {status}")

        return 0 if status == "PASS" else 1

    finally:
        # 9. Clean shutdown.
        stop_llama_server(proc)
        print("[haider] llama-server stopped")


if __name__ == "__main__":
    sys.exit(main())
