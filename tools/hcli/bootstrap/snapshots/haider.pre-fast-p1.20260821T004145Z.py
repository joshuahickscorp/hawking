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
import uuid
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
MISSIONS_DIR = HAIDER_DIR / "missions"
DEFAULT_HOST = "127.0.0.1"
READY_TIMEOUT_S = float(os.environ.get("HAIDER_READY_TIMEOUT", "180"))
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
# Durable mission state
# ---------------------------------------------------------------------------


def create_mission_state(
    task: str,
    source: str,
    max_cycles: int = 8,
) -> Dict[str, Any]:
    if max_cycles < 1:
        raise ValueError("max_cycles must be >= 1")

    now = datetime.now(timezone.utc).isoformat()

    return {
        "mission_id": uuid.uuid4().hex,
        "task": task,
        "source": source,
        "status": "running",
        "created_at": now,
        "updated_at": now,
        "cycle": 0,
        "max_cycles": int(max_cycles),
    }


def write_mission_state(state: Dict[str, Any]) -> Path:
    MISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    mission_id = str(state["mission_id"])
    path = MISSIONS_DIR / f"{mission_id}.json"
    temp = MISSIONS_DIR / f".{mission_id}.{os.getpid()}.tmp"

    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    payload = json.dumps(
        state,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )

    temp.write_text(payload, encoding="utf-8")
    os.replace(temp, path)

    return path


def load_mission_state(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mission_candidate_paths(
    mission: str,
    guard: p0.RepositoryGuard,
) -> List[str]:
    """Extract explicitly named repository files from a mission.

    This is deterministic retrieval, not model reasoning.
    """

    candidates: List[str] = []

    # Captures repo-looking paths that include a filename extension.
    pattern = re.compile(
        r"(?<![A-Za-z0-9_.-])"
        r"((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+)"
    )

    for raw in pattern.findall(mission or ""):
        raw = raw.rstrip(".,:;)`]}'\"")

        if raw.startswith(".git/") or "/.git/" in raw:
            continue

        try:
            full = guard.resolve(raw)
        except p0.ToolError:
            continue

        if not os.path.isfile(full):
            continue

        rel = guard.relative(full)

        if rel not in candidates:
            candidates.append(rel)

    return candidates


def build_fast_mission_evidence(
    mission: str,
    guard: p0.RepositoryGuard,
    max_files: int = 6,
    max_chars_per_file: int = 16000,
    max_total_chars: int = 48000,
) -> Optional[Dict[str, Any]]:
    """Build deterministic evidence when a mission names concrete files.

    Avoids spending model turns rediscovering paths already supplied by
    the operator/mission.
    """

    paths = _mission_candidate_paths(mission, guard)[:max_files]

    if not paths:
        return None

    chunks: List[str] = []
    used = 0

    for rel in paths:
        try:
            full = guard.resolve(rel)
            text = Path(full).read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        remaining = max_total_chars - used
        if remaining <= 0:
            break

        excerpt = text[: min(max_chars_per_file, remaining)]

        chunks.append(
            f"===== FILE: {rel} =====\\n"
            f"{excerpt}\\n"
            f"===== END FILE: {rel} ====="
        )

        used += len(excerpt)

    if not chunks:
        return None

    return {
        "role": "parent",
        "final": (
            "FAST DETERMINISTIC MISSION EVIDENCE\\n\\n"
            + "\\n\\n".join(chunks)
        ),
        "ok": True,
        "stats": {
            "model_turns": 0,
            "tool_calls": 0,
            "protocol_errors": 0,
            "fast_evidence_files": len(chunks),
        },
        "elapsed_s": 0.0,
    }



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

    normalized_path = path.replace("\\", "/")
    path_parts = [
        part for part in normalized_path.split("/")
        if part not in ("", ".")
    ]

    if ".git" in path_parts:
        return (
            False,
            "path rejected: mutation of .git/** is forbidden",
            None,
            None,
        )

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
# Mutation operations are validated mechanically before application.
# Mutation v2 — compact deterministic edit operations
# ---------------------------------------------------------------------------


def _reject_mutation_path(
    guard: p0.RepositoryGuard,
    path: str,
    *,
    require_existing: bool,
) -> Tuple[bool, str, Optional[str]]:
    if not isinstance(path, str) or not path.strip():
        return False, "path is empty", None

    normalized = path.replace("\\", "/")
    parts = [
        part
        for part in normalized.split("/")
        if part not in ("", ".")
    ]

    if ".git" in parts:
        return False, "mutation of .git/** is forbidden", None

    try:
        full = guard.resolve(path)
    except p0.ToolError as e:
        return False, f"path rejected: {e}", None

    if require_existing and not os.path.isfile(full):
        return False, f"path is not a regular file: {path}", None

    return True, "", full


def validate_mutation_operation(
    guard: p0.RepositoryGuard,
    operation: Dict[str, Any],
) -> Tuple[bool, str]:
    if not isinstance(operation, dict):
        return False, "operation must be an object"

    op = operation.get("op")
    path = operation.get("path", "")

    if op == "replace":
        ok, err, full = _reject_mutation_path(
            guard,
            path,
            require_existing=True,
        )
        if not ok:
            return False, err

        old = operation.get("old_text")
        new = operation.get("new_text")

        if not isinstance(old, str) or not old:
            return False, "replace old_text is empty"

        if not isinstance(new, str):
            return False, "replace new_text must be a string"

        if old == new:
            return False, "replace is a no-op"

        text = Path(full).read_text(
            encoding="utf-8",
            errors="ignore",
        )

        count = text.count(old)

        if count == 0:
            return False, "replace old_text not found"

        if count != 1:
            return False, f"replace old_text matches {count} locations"

        return True, ""

    if op in ("insert_before", "insert_after"):
        ok, err, full = _reject_mutation_path(
            guard,
            path,
            require_existing=True,
        )
        if not ok:
            return False, err

        anchor = operation.get("anchor")
        text_to_insert = operation.get("text")

        if not isinstance(anchor, str) or not anchor:
            return False, "insert anchor is empty"

        if not isinstance(text_to_insert, str) or not text_to_insert:
            return False, "insert text is empty"

        text = Path(full).read_text(
            encoding="utf-8",
            errors="ignore",
        )

        count = text.count(anchor)

        if count == 0:
            return False, "insert anchor not found"

        if count != 1:
            return False, f"insert anchor matches {count} locations"

        return True, ""

    if op == "create":
        ok, err, full = _reject_mutation_path(
            guard,
            path,
            require_existing=False,
        )
        if not ok:
            return False, err

        if os.path.exists(full):
            return False, "create target already exists"

        content = operation.get("content")

        if not isinstance(content, str) or not content:
            return False, "create content is empty"

        return True, ""

    return False, f"unsupported mutation op: {op!r}"


def apply_mutation_operations(
    guard: p0.RepositoryGuard,
    operations: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Validate ALL operations before writing, then apply atomically-ish.

    Existing files are snapshotted in memory. If any write fails, restore
    previous content and remove newly-created files.
    """

    if not isinstance(operations, list) or not operations:
        return None

    if len(operations) > 8:
        return None

    for operation in operations:
        ok, _err = validate_mutation_operation(
            guard,
            operation,
        )

        if not ok:
            return None

    originals: Dict[str, Optional[str]] = {}
    changed: List[Dict[str, Any]] = []

    try:
        for operation in operations:
            op = operation["op"]
            path = operation["path"]

            require_existing = op != "create"

            ok, err, full = _reject_mutation_path(
                guard,
                path,
                require_existing=require_existing,
            )

            if not ok or full is None:
                raise RuntimeError(err)

            if full not in originals:
                if os.path.isfile(full):
                    originals[full] = Path(full).read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )
                else:
                    originals[full] = None

            if op == "replace":
                current = Path(full).read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

                updated = current.replace(
                    operation["old_text"],
                    operation["new_text"],
                    1,
                )

            elif op == "insert_before":
                current = Path(full).read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

                updated = current.replace(
                    operation["anchor"],
                    operation["text"] + operation["anchor"],
                    1,
                )

            elif op == "insert_after":
                current = Path(full).read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

                updated = current.replace(
                    operation["anchor"],
                    operation["anchor"] + operation["text"],
                    1,
                )

            elif op == "create":
                current = ""
                updated = operation["content"]

                Path(full).parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

            else:
                raise RuntimeError(
                    f"unsupported operation: {op}"
                )

            old_sha = hashlib.sha256(
                current.encode()
            ).hexdigest()

            new_sha = hashlib.sha256(
                updated.encode()
            ).hexdigest()

            Path(full).write_text(
                updated,
                encoding="utf-8",
            )

            changed.append(
                {
                    "op": op,
                    "path": guard.relative(full),
                    "old_sha256": old_sha,
                    "new_sha256": new_sha,
                }
            )

        return {
            "operations": changed,
            "paths": [item["path"] for item in changed],
            "operation_count": len(changed),
        }

    except Exception:
        for full, original in originals.items():
            try:
                if original is None:
                    if os.path.exists(full):
                        os.unlink(full)
                else:
                    Path(full).write_text(
                        original,
                        encoding="utf-8",
                    )
            except Exception:
                pass

        return None


def _extract_mutation_json(
    content: str,
) -> Optional[List[Dict[str, Any]]]:
    content = content.strip()

    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(
            r"\{.*\}",
            content,
            re.DOTALL,
        )

        if not match:
            return None

        try:
            value = json.loads(match.group())
        except json.JSONDecodeError:
            return None

    if not isinstance(value, dict):
        return None

    operations = value.get("operations")

    if not isinstance(operations, list):
        return None

    return operations



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
    mission_text: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Generate and apply compact mission-grounded Mutation-v2 operations."""

    obs_final = obs_result.get("final", "")
    obs_stats = obs_result.get("stats", {})

    mission = (
        mission_text.strip()
        if isinstance(mission_text, str)
        and mission_text.strip()
        else (
            "Make one small safe reversible repository improvement "
            "grounded in the supplied evidence."
        )
    )

    system_prompt = """You are HAIDER's mutation synthesizer.

Advance the explicit mission using ONLY the supplied repository evidence.

Do not perform unrelated cleanup.

Never modify .git/**.

Return ONLY JSON in this exact outer form:

{
  "operations": [
    ...
  ]
}

Allowed operations:

1. Replace an exact unique substring:

{
  "op": "replace",
  "path": "relative/file.py",
  "old_text": "exact unique substring",
  "new_text": "replacement"
}

2. Insert code immediately AFTER an exact unique anchor:

{
  "op": "insert_after",
  "path": "relative/file.py",
  "anchor": "exact unique anchor",
  "text": "text to insert"
}

3. Insert code immediately BEFORE an exact unique anchor:

{
  "op": "insert_before",
  "path": "relative/file.py",
  "anchor": "exact unique anchor",
  "text": "text to insert"
}

4. Create a new repository file:

{
  "op": "create",
  "path": "relative/new_file.py",
  "content": "complete file contents"
}

Rules:

- Maximum 8 operations.
- Prefer compact anchors over copying giant files.
- Every anchor/old_text must come from evidence.
- Do not invent existing file content.
- Operations should form one coherent implementation transaction.
- It is valid to modify source AND create/update its focused tests.
- Do not output prose, Markdown, or code fences.
"""

    user_prompt = (
        "EXPLICIT MISSION:\n"
        f"{mission}\n\n"
        "DETERMINISTIC REPOSITORY EVIDENCE:\n"
        f"{obs_final}\n\n"
        f"EVIDENCE STATS: {json.dumps(obs_stats)}\n\n"
        "Synthesize the smallest coherent set of mutation operations "
        "that materially advances the mission."
    )

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    synthesis_budget = int(
        os.environ.get("HAIDER_SYNTHESIS_TOKENS", "1536")
    )

    original_budget = client.max_tokens

    for attempt in range(2):
        try:
            client.max_tokens = synthesis_budget
            content, _ = client.chat(messages)
        except p0.ModelError:
            return None
        finally:
            client.max_tokens = original_budget

        # Persist raw synthesis output before attempting to parse it.
        # Invalid model output is evidence, not garbage.
        synthesis_dir = HAIDER_DIR / "synthesis"
        synthesis_dir.mkdir(parents=True, exist_ok=True)

        synthesis_path = synthesis_dir / (
            f"mutation_{int(time.time())}_attempt_{attempt + 1}.txt"
        )

        synthesis_path.write_text(
            content,
            encoding="utf-8",
        )

        print(
            f"[haider] synthesis: {synthesis_path} "
            f"chars={len(content)}"
        )

        operations = _extract_mutation_json(
            content
        )

        if operations is None:
            print(
                "[haider] mutation parse failed; "
                f"raw response retained at {synthesis_path}"
            )

        if operations is not None:
            result = apply_mutation_operations(
                guard,
                operations,
            )

            if result is not None:
                return result

        if attempt == 0:
            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The mutation was syntactically invalid or failed "
                        "deterministic validation. Return corrected JSON only. "
                        "Use short exact anchors from the supplied evidence."
                    ),
                }
            )

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
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Mission text for this HAIDER run.",
    )
    parser.add_argument(
        "--task-file",
        type=str,
        default=None,
        help="Read the HAIDER mission from a UTF-8 text or Markdown file.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=8,
        help="Maximum mission cycles for persistent self-host execution.",
    )

    args = parser.parse_args()

    if args.task and args.task_file:
        parser.error("--task and --task-file are mutually exclusive")

    if args.max_cycles < 1:
        parser.error("--max-cycles must be >= 1")

    mission_text = None
    mission_source = "gate-zero-default"

    if args.task_file:
        task_path = Path(args.task_file).expanduser()

        if not task_path.is_absolute():
            task_path = Path.cwd() / task_path

        task_path = task_path.resolve()

        if not task_path.is_file():
            parser.error(f"task file not found: {task_path}")

        mission_text = task_path.read_text(encoding="utf-8").strip()

        if not mission_text:
            parser.error(f"task file is empty: {task_path}")

        mission_source = str(task_path)

    elif args.task:
        mission_text = args.task.strip()

        if not mission_text:
            parser.error("--task cannot be empty")

        mission_source = "inline"


    if args.n != 1:
        print(f"haider: Gate Zero supports N=1 only (got {args.n})", file=sys.stderr)
        return 2

    HAIDER_DIR.mkdir(parents=True, exist_ok=True)

    mission_state = None
    mission_state_path = None

    if mission_text is not None:
        mission_state = create_mission_state(
            mission_text,
            mission_source,
            args.max_cycles,
        )
        mission_state_path = write_mission_state(mission_state)
        print(f"[haider] mission: {mission_state_path}")

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
            max_tokens=int(
                os.environ.get("HAIDER_MAX_OUTPUT_TOKENS", "2048")
            ),
            debug=args.debug,
        )

        print(f"[haider] model timeout={model_timeout:.0f}s")

        # 5. Evidence acquisition.
        #
        # Explicit missions that name concrete repository files take the
        # deterministic fast path and avoid expensive model rediscovery.
        fast_evidence = None

        if mission_text is not None:
            fast_evidence = build_fast_mission_evidence(
                mission_text,
                guard,
            )

        if fast_evidence is not None:
            print(
                "[haider] fast evidence: "
                f"{fast_evidence['stats']['fast_evidence_files']} files"
            )
            obs_result = fast_evidence
        else:
            print("[haider] observation phase...")
            session = p0.Session(
                "parent",
                (
                    mission_text
                    if mission_text is not None
                    else (
                        "Inspect the current repository state. Check git status, "
                        "list key files, and identify one small safe edit you "
                        "could make. Use read-only tools. When done, respond with "
                        "a final JSON object summarizing your findings."
                    )
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
        edit = apply_scoped_edit(
            client,
            guard,
            executor,
            obs_result,
            mission_text=mission_text,
        )
        if edit is None:
            print("[haider] FATAL: no valid edit produced", file=sys.stderr)
            return 1
        print(
            f"[haider] mutation: {edit['operation_count']} operations "
            f"across {len(edit['paths'])} paths"
        )

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
