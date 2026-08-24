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
import concurrent.futures
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

MAX_MUTATION_OPERATIONS = 20

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
# WorkUnit representation
# ---------------------------------------------------------------------------

WORKUNIT_STATUSES = ("pending", "ready", "running", "completed", "failed")


class WorkUnit:
    """Small deterministic work unit with mechanically controlled status."""

    def __init__(
        self,
        id: str,
        role: str,
        description: str,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        self.id = id
        self.role = role
        self.description = description
        self.dependencies = list(dependencies or [])
        self.status = "pending"
        self.assigned_runtime: Optional[int] = None
        self.attempts = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "description": self.description,
            "dependencies": self.dependencies,
            "status": self.status,
            "assigned_runtime": self.assigned_runtime,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkUnit":
        wu = cls(
            id=data["id"],
            role=data["role"],
            description=data["description"],
            dependencies=data.get("dependencies", []),
        )
        wu.status = data.get("status", "pending")
        wu.assigned_runtime = data.get("assigned_runtime")
        wu.attempts = data.get("attempts", 0)
        return wu


def _transition_workunit_status(
    wu: WorkUnit,
    new_status: str,
) -> bool:
    """Mechanically controlled status transition.

    Valid transitions:
        pending -> ready
        ready -> running
        running -> completed
        running -> failed
        failed -> ready  (retry)
    """
    valid = {
        "pending": {"ready"},
        "ready": {"running"},
        "running": {"completed", "failed"},
        "failed": {"ready"},
        "completed": set(),
    }
    if new_status in valid.get(wu.status, set()):
        wu.status = new_status
        return True
    return False


def _workunit_is_ready(
    wu: WorkUnit,
    all_units: Dict[str, WorkUnit],
) -> bool:
    """A work unit is ready when pending or failed-within-budget and all deps are completed."""
    if wu.status not in ("pending", "failed"):
        return False
    if wu.status == "failed" and wu.attempts >= DEFAULT_RETRY_BUDGET:
        return False
    for dep_id in wu.dependencies:
        dep = all_units.get(dep_id)
        if dep is None or dep.status != "completed":
            return False
    return True


def _identify_ready_workunits(
    units: Dict[str, WorkUnit],
) -> List[WorkUnit]:
    """Identify work units that can transition to ready."""
    ready: List[WorkUnit] = []
    for wu in units.values():
        if _workunit_is_ready(wu, units):
            _transition_workunit_status(wu, "ready")
            ready.append(wu)
    return ready


def _assign_ready_workunits(
    ready_units: List[WorkUnit],
    runtime_count: int,
) -> List[Tuple[WorkUnit, int]]:
    """Assign ready independent work to runtime indices.

    Prevents duplicate simultaneous assignment by tracking assigned runtimes.
    Returns list of (workunit, runtime_index) pairs.
    """
    assignments: List[Tuple[WorkUnit, int]] = []
    used_runtimes: set = set()

    for wu in ready_units:
        if wu.status != "ready":
            continue
        # Find a free runtime index
        assigned = False
        for idx in range(runtime_count):
            if idx not in used_runtimes:
                _transition_workunit_status(wu, "running")
                wu.assigned_runtime = idx
                wu.attempts += 1
                used_runtimes.add(idx)
                assignments.append((wu, idx))
                assigned = True
                break
        if not assigned:
            break

    return assignments


def _mark_workunit_completed(wu: WorkUnit) -> None:
    _transition_workunit_status(wu, "completed")
    wu.assigned_runtime = None


def _mark_workunit_failed(wu: WorkUnit) -> None:
    _transition_workunit_status(wu, "failed")
    wu.assigned_runtime = None


def parse_haider_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Compatibility boundary: HAIDER uses HCLI's canonical product grammar.

    HCLI owns the public argument contract.  HAIDER retains ``args.n`` as a
    temporary compatibility alias for bootstrap callers that have not yet
    migrated to ``runtime_count``.
    """
    from hcli.cli import parse_haider_args as parse_hcli_args

    args = parse_hcli_args(argv)

    # Temporary HAIDER compatibility alias.
    args.n = args.runtime_count

    return args

DEFAULT_RETRY_BUDGET = 3


def _workunit_can_retry(wu: WorkUnit, budget: int = DEFAULT_RETRY_BUDGET) -> bool:
    """Check if a failed work unit can be retried within budget."""
    return wu.status == "failed" and wu.attempts < budget


def _retry_failed_workunits(units: Dict[str, WorkUnit], budget: int = DEFAULT_RETRY_BUDGET) -> List[WorkUnit]:
    """Transition failed work units back to ready if retry budget remains."""
    retried: List[WorkUnit] = []
    for wu in units.values():
        if _workunit_can_retry(wu, budget):
            if _transition_workunit_status(wu, "ready"):
                retried.append(wu)
    return retried


def _deterministic_evidence(
    guard: "p0.RepositoryGuard",
    mission_text: str,
) -> Dict[str, Any]:
    """Build bounded workspace evidence without any model call.

    RepositoryGuard is the scope authority.  Do not duplicate its root state;
    resolve "." through the guard to obtain the current workspace.
    """
    workspace = Path(guard.resolve("."))

    explicit_pattern = re.compile(
        r"(?<![A-Za-z0-9_.-])"
        r"((?:[A-Za-z0-9_.-]+/)*"
        r"[A-Za-z0-9_.-]+\."
        r"(?:py|pyi|js|jsx|ts|tsx|md|json|yaml|yml|toml|rs|go|txt|cfg|ini))"
    )

    named_files: List[str] = []

    for raw in explicit_pattern.findall(mission_text or ""):
        raw = raw.rstrip(".,:;)`]}'\"")

        if raw.startswith(".git/") or "/.git/" in raw:
            continue

        try:
            full = Path(guard.resolve(raw))
        except Exception:
            continue

        if not full.is_file():
            continue

        try:
            rel = guard.relative(str(full))
        except Exception:
            rel = str(full.relative_to(workspace))

        if rel not in named_files:
            named_files.append(rel)

    tracked: List[str] = []

    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            tracked = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip()
            ]
    except (OSError, subprocess.TimeoutExpired):
        pass

    if not tracked:
        ignored = {
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            "build",
            "dist",
            "target",
            "coverage",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
        }

        for path in workspace.rglob("*"):
            if not path.is_file():
                continue

            try:
                rel_path = path.relative_to(workspace)
            except ValueError:
                continue

            if any(part in ignored for part in rel_path.parts):
                continue

            tracked.append(str(rel_path))

    # Stable de-duplication.
    tracked = list(dict.fromkeys(tracked))

    mission_lower = (mission_text or "").lower()

    tokens = {
        token
        for token in re.findall(r"[a-zA-Z0-9_]+", mission_lower)
        if len(token) >= 3
    }

    def relevance(path: str) -> tuple:
        lower = path.lower()

        score = 0

        if path in named_files:
            score += 1000

        if "test" in mission_lower and "test" in lower:
            score += 100

        for token in tokens:
            if token in lower:
                score += 10

        # Prefer source / configuration / goal surfaces over arbitrary bulk.
        if lower.endswith(
            (
                ".py",
                ".js",
                ".ts",
                ".tsx",
                ".rs",
                ".go",
                ".md",
                ".toml",
                ".json",
                ".yaml",
                ".yml",
            )
        ):
            score += 1

        return (-score, len(path), path)

    universe = list(dict.fromkeys(named_files + tracked))

    ranked = sorted(universe, key=relevance)

    # Enough evidence for useful deterministic narrowing without flooding the
    # worker context.
    selected = ranked[:24]

    files: List[Dict[str, Any]] = []

    for rel in selected:
        try:
            full = Path(guard.resolve(rel))
        except Exception:
            continue

        if not full.is_file():
            continue

        try:
            content = full.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            continue

        # Evidence is a bounded cache, not authority.
        if len(content) > 12000:
            content = content[:12000] + "\n...[truncated]"

        files.append(
            {
                "path": rel,
                "content": content,
            }
        )

    final_chunks = []

    for item in files:
        final_chunks.append(
            f"--- {item['path']} ---\n{item['content']}"
        )

    stats = {
        "model_turns": 0,
        "fast_evidence_files": len(files),
    }

    return {
        "ok": True,
        "files": files,
        "named_files": named_files,
        "tracked_files": tracked[:500],
        "fast_evidence_files": len(files),
        "model_turns": 0,
        "stats": stats,
        "final": "\n\n".join(final_chunks),
    }


def _build_scheduler_plan(
    units: Dict[str, WorkUnit],
    runtime_count: int,
) -> List[Tuple[WorkUnit, int]]:
    """Build one deterministic scheduling wave.

    Pending work with satisfied dependencies is promoted to ready.

    Already-ready work is dispatchable.

    Failed work is deliberately *not* auto-promoted here.  Retry policy owns
    the explicit failed -> ready transition, which prevents an ordinary
    scheduler tick from silently consuming another retry attempt.
    """
    ready: List[WorkUnit] = []

    for wu in units.values():
        if wu.status == "ready":
            deps_complete = all(
                dep_id in units
                and units[dep_id].status == "completed"
                for dep_id in wu.dependencies
            )

            if deps_complete:
                ready.append(wu)

            continue

        if wu.status != "pending":
            continue

        deps_complete = all(
            dep_id in units
            and units[dep_id].status == "completed"
            for dep_id in wu.dependencies
        )

        if not deps_complete:
            continue

        if _transition_workunit_status(wu, "ready"):
            ready.append(wu)

    return _assign_ready_workunits(
        ready,
        runtime_count,
    )


def _build_workunit_receipt(
    wu: WorkUnit,
    runtime_index: int,
    pid: Optional[int],
    port: Optional[int],
    ctx_size: int,
    mutation: Optional[Dict[str, Any]],
    validation: Optional[Dict[str, Any]],
    rolled_back: bool,
) -> Dict[str, Any]:
    """Construct a deterministic mission receipt for a completed work unit."""
    return {
        "workunit_id": wu.id,
        "role": wu.role,
        "description": wu.description,
        "status": wu.status,
        "attempts": wu.attempts,
        "runtime_index": runtime_index,
        "pid": pid,
        "port": port,
        "ctx_size": ctx_size,
        "mutation": mutation,
        "validation": validation,
        "rolled_back": rolled_back,
        "timestamp": _fast_now(),
    }


def _validate_insert_anchor(
    guard: "p0.RepositoryGuard",
    path: str,
    anchor: str,
    op: str,
) -> Optional[str]:
    """Validate insert_before/insert_after anchor safety.

    Returns None if valid, or an error message string if invalid.
    Checks:
      - anchor is non-empty and unique in the file
      - anchor does not end with an opening paren (partial def/class)
      - anchor is not a partial line (must be a complete line or block boundary)
      - anchor is not a partial Python declaration (def/class/return/if/elif/else/for/while/with/try/except/finally/lambda/async def)
    """
    if not anchor or not anchor.strip():
        return "anchor is empty"
    stripped = anchor.strip()
    if stripped.endswith("("):
        return f"anchor ends with '(': partial declaration not allowed for {op}"
    # Reject partial Python declarations: lines that start a compound statement
    # but are not complete (e.g. "def foo(" or "if x:")
    partial_decl_re = re.compile(
        r"^(async\s+def|def|class|return|if|elif|else|for|while|with|try|except|finally|lambda)\b"
    )
    if partial_decl_re.match(stripped):
        # A complete declaration line must end with ':' or be a full expression
        # For def/class: must end with ':' (complete signature)
        # For if/elif/for/while/with/try/except/finally: must end with ':'
        # For return/lambda: must be a complete expression (no trailing operator)
        if re.match(r"^(async\s+def|def|class|if|elif|for|while|with|try|except|finally)\b", stripped):
            if not stripped.endswith(":"):
                return f"anchor is a partial declaration (missing ':'): {op}"
        elif re.match(r"^(return|lambda)\b", stripped):
            # return/lambda must not end with an operator or be just the keyword
            if stripped in ("return", "lambda") or stripped.rstrip().endswith(("+", "-", "*", "/", "%", "&", "|", "^", "<", ">", "=", ",")):
                return f"anchor is a partial expression: {op}"
    try:
        full = guard.resolve(path)
        content = Path(full).read_text(encoding="utf-8")
    except Exception as e:
        return f"cannot read file for anchor validation: {e}"
    lines = content.splitlines()
    matches = [i for i, ln in enumerate(lines) if ln.strip() == stripped]
    if len(matches) == 0:
        return f"anchor not found in {path}"
    if len(matches) > 1:
        return f"anchor is ambiguous ({len(matches)} matches) in {path}"
    # Check that the anchor line is a complete statement boundary
    idx = matches[0]
    stripped = lines[idx].strip()
    # Reject if line ends with a colon followed by nothing (incomplete block header is ok for insert_before)
    # Reject if line ends with an operator or comma (incomplete expression)
    if stripped and stripped[-1] in ("+", "-", "*", "/", "=", ",", "&", "|", "<", ">", "!", "~"):
        return f"anchor ends with operator '{stripped[-1]}': incomplete expression"
    return None



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

    if len(operations) > MAX_MUTATION_OPERATIONS:
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



# ===========================================================================
# P1 FAST ENGINE
#
# Bootstrap implementation:
#
# - independent llama-server processes for haider N
# - direct/no-reasoning builder runtimes
# - parallel role synthesis
# - deterministic multi-cycle mission state transitions
# - transaction rollback if validation fails
#
# This is intentionally inside HAIDER itself so the system can begin using
# parallel self-hosting before the later native HCLI rewrite.
# ===========================================================================


def _fast_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fast_state_transition(
    state: Dict[str, Any],
    *,
    mutation_ok: bool,
    validation_ok: bool,
    complete: bool = False,
    failure_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Pure deterministic mission-state transition."""

    result = dict(state)

    # Mission text is immutable authority.
    original_task = state.get("task")

    if not mutation_ok:
        result["status"] = "failed"
        result["stop_reason"] = failure_reason or "mutation_failed"
        result["updated_at"] = _fast_now()
        result["task"] = original_task
        return result

    if not validation_ok:
        result["status"] = "failed"
        result["stop_reason"] = failure_reason or "validation_failed"
        result["updated_at"] = _fast_now()
        result["task"] = original_task
        return result

    result["cycle"] = int(result.get("cycle", 0)) + 1

    if complete:
        result["status"] = "complete"
        result["stop_reason"] = "complete"
    elif result["cycle"] >= int(result.get("max_cycles", 1)):
        result["status"] = "max_cycles"
        result["stop_reason"] = "max_cycles"
    else:
        result["status"] = "running"
        result.pop("stop_reason", None)

    result["updated_at"] = _fast_now()
    result["task"] = original_task
    return result


def _compact_mission_evidence(
    mission: str,
    guard: p0.RepositoryGuard,
    max_files: int = 6,
    max_chars_per_file: int = 16000,
    max_total_chars: int = 32000,
) -> Optional[Dict[str, Any]]:
    """Deterministic evidence packet optimized for model context.

    Large files contribute both head and tail so HAIDER sees imports/helpers
    and main/runtime wiring without spending inference turns navigating.
    """

    paths = _mission_candidate_paths(
        mission,
        guard,
    )[:max_files]

    if not paths:
        return None

    chunks: List[str] = []
    used = 0

    for rel in paths:
        if used >= max_total_chars:
            break

        try:
            full = guard.resolve(rel)
            text = Path(full).read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        remaining = max_total_chars - used
        allowance = min(
            max_chars_per_file,
            remaining,
        )

        if len(text) <= allowance:
            excerpt = text
        else:
            half = max(1, allowance // 2)

            excerpt = (
                text[:half]
                + "\n\n"
                + "===== MIDDLE OMITTED DETERMINISTICALLY ====="
                + "\n\n"
                + text[-half:]
            )

        chunks.append(
            f"===== FILE: {rel} =====\n"
            f"{excerpt}\n"
            f"===== END FILE: {rel} ====="
        )

        used += len(excerpt)

    if not chunks:
        return None

    return {
        "role": "parent",
        "final": (
            "DETERMINISTIC FAST REPOSITORY EVIDENCE\n\n"
            + "\n\n".join(chunks)
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


def _llama_reasoning_flags_disabled() -> List[str]:
    """Builder runtimes must emit artifacts, not burn their budget thinking."""

    return [
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--reasoning-format",
        "none",
    ]


def _start_fast_runtime_server(
    model: str,
    port: int,
    host: str,
    ctx_size: int,
    index: int,
) -> subprocess.Popen:
    """Start one physically independent llama-server process."""

    cmd = [
        "llama-server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--ctx-size",
        str(ctx_size),
        "--parallel",
        "1",
        "--no-webui",
    ]

    # Installed llama-server used by this project already supports these.
    # If a future build does not, retry once without them.
    cmd += _llama_reasoning_flags_disabled()

    LOGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        LOGS_DIR
        / f"runtime_{index}_{port}.log"
    )

    log_file = open(
        log_path,
        "w",
    )

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    proc._haider_log_file = log_file  # type: ignore[attr-defined]
    proc._haider_log_path = str(log_path)  # type: ignore[attr-defined]

    return proc


class FastRuntimePool:
    """N physically independent llama-server runtimes."""

    def __init__(
        self,
        n: int,
        model: str,
        host: str,
        ctx_size: int,
        debug: bool = False,
    ):
        if n < 1:
            raise ValueError("runtime count must be >= 1")

        self.n = n
        self.model = model
        self.host = host
        self.ctx_size = ctx_size
        self.debug = debug
        self.slots: List[Dict[str, Any]] = []

    def start(self) -> None:
        # Launch all processes first so model loading overlaps.
        for index in range(self.n):
            port = allocate_port()

            proc = _start_fast_runtime_server(
                self.model,
                port,
                self.host,
                self.ctx_size,
                index,
            )

            self.slots.append(
                {
                    "index": index,
                    "port": port,
                    "proc": proc,
                    "client": None,
                }
            )

            print(
                f"[haider] runtime[{index}] "
                f"pid={proc.pid} port={port}"
            )

        # All servers are now loading concurrently.
        timeout = float(
            os.environ.get(
                "HAIDER_READY_TIMEOUT",
                "240",
            )
        )

        for slot in self.slots:
            proc = slot["proc"]

            if not wait_for_ready(
                slot["port"],
                self.host,
                timeout=timeout,
            ):
                log_path = getattr(
                    proc,
                    "_haider_log_path",
                    "",
                )

                tail = ""

                if log_path and os.path.isfile(log_path):
                    try:
                        tail = (
                            Path(log_path)
                            .read_text(
                                encoding="utf-8",
                                errors="ignore",
                            )[-3000:]
                        )
                    except Exception:
                        pass

                raise RuntimeError(
                    f"runtime {slot['index']} "
                    f"failed readiness\n{tail}"
                )

            client = p0.ModelClient(
                api_base=(
                    f"http://{self.host}:"
                    f"{slot['port']}/v1"
                ),
                model="local",
                api_key="sk-local",
                timeout=float(
                    os.environ.get(
                        "HAIDER_MODEL_TIMEOUT",
                        "1800",
                    )
                ),
                # Direct builder output.
                max_tokens=int(
                    os.environ.get(
                        "HAIDER_BUILDER_TOKENS",
                        "1200",
                    )
                ),
                debug=self.debug,
            )

            slot["client"] = client

            print(
                f"[haider] runtime[{slot['index']}] ready"
            )

        self.write_descriptor("running")

    def write_descriptor(
        self,
        status: str,
    ) -> None:
        HAIDER_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        descriptor = {
            "status": status,
            "timestamp": _fast_now(),
            "runtime_count": len(self.slots),
            "runtimes": [
                {
                    "index": slot["index"],
                    "pid": slot["proc"].pid,
                    "port": slot["port"],
                    "ctx_size": self.ctx_size,
                }
                for slot in self.slots
            ],
        }

        (
            HAIDER_DIR
            / "runtime_pool.json"
        ).write_text(
            json.dumps(
                descriptor,
                indent=2,
            ),
            encoding="utf-8",
        )

    def stop(self) -> None:
        for slot in reversed(self.slots):
            try:
                stop_llama_server(
                    slot["proc"]
                )
            except Exception:
                pass

        self.write_descriptor("stopped")

        print("[haider] runtime pool stopped")

    @property
    def clients(self) -> List[p0.ModelClient]:
        return [
            slot["client"]
            for slot in self.slots
            if slot.get("client") is not None
        ]


def _candidate_prompt(
    mission: str,
    evidence: Dict[str, Any],
    role: str,
) -> List[Dict[str, str]]:
    role_instruction = {
        "core": (
            "ROLE: CORE BUILDER. Prioritize the primary implementation "
            "required by the mission."
        ),
        "tests": (
            "ROLE: TEST BUILDER. Prefer focused deterministic tests or "
            "test-supporting implementation that advances the mission."
        ),
        "adversary": (
            "ROLE: ADVERSARIAL BUILDER. Prefer the smallest correctness, "
            "safety, rollback, lifecycle, or invariant improvement required "
            "by the mission."
        ),
        "alternate": (
            "ROLE: ALTERNATE BUILDER. Produce an independent coherent "
            "implementation approach."
        ),
    }.get(
        role,
        "ROLE: BUILDER.",
    )

    system = """You are a HAIDER direct mutation builder.

Do not think aloud.
Do not explain.
Do not emit prose.

Return ONLY valid JSON:

{"operations":[ ... ]}

Allowed operations:

{"op":"replace","path":"...","old_text":"...","new_text":"..."}
{"op":"insert_before","path":"...","anchor":"...","text":"..."}
{"op":"insert_after","path":"...","anchor":"...","text":"..."}
{"op":"create","path":"...","content":"..."}

Rules:

- never mutate .git/**
- use short exact unique anchors from evidence
- one coherent transaction
- maximum 20 operations
- related multi-file changes are allowed
- no markdown fences
- no commentary
"""

    user = (
        f"{role_instruction}\n\n"
        "MISSION:\n"
        f"{mission}\n\n"
        "REPOSITORY EVIDENCE:\n"
        f"{evidence.get('final', '')}\n\n"
        "Emit the mutation JSON now."
    )

    return [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": user,
        },
    ]


def _structured_builder_request(
    host: str,
    port: int,
    messages: List[Dict[str, str]],
    *,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any], str, str]:
    """Direct llama.cpp structured builder call.

    Builder workers are artifact emitters:
      - thinking disabled
      - JSON-object constrained
      - low temperature
      - bounded output

    Returns:
        content, usage, finish_reason, reasoning_content
    """

    payload = {
        "model": "local",
        "messages": messages,
        "temperature": 0.05,
        "max_tokens": int(max_tokens),

        # llama.cpp OpenAI-compatible structured output.
        "response_format": {
            "type": "json_object",
        },

        # Disable thinking both through server startup flags and
        # request-level template controls.
        "chat_template_kwargs": {
            "enable_thinking": False,
        },

        # Supported llama-server request-level reasoning control.
        "reasoning_effort": "none",
    }

    data = json.dumps(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(
        (
            f"http://{host}:{port}"
            "/v1/chat/completions"
        ),
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-local",
        },
        method="POST",
    )

    timeout = float(
        os.environ.get(
            "HAIDER_MODEL_TIMEOUT",
            "1800",
        )
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            obj = json.loads(
                response.read().decode(
                    "utf-8"
                )
            )
    except Exception as exc:
        raise p0.ModelError(
            f"structured builder request failed: {exc}"
        ) from exc

    try:
        choice = obj["choices"][0]
        message = choice["message"]

        content = (
            message.get("content")
            or ""
        )

        reasoning_content = (
            message.get(
                "reasoning_content"
            )
            or ""
        )

        finish_reason = (
            choice.get(
                "finish_reason"
            )
            or ""
        )

    except Exception as exc:
        raise p0.ModelError(
            f"invalid structured response: {exc}"
        ) from exc

    return (
        content,
        obj.get("usage") or {},
        finish_reason,
        reasoning_content,
    )


def _compact_builder_prompt(
    mission: str,
    evidence: Dict[str, Any],
    role: str,
) -> List[Dict[str, str]]:
    role_instruction = {
        "core": (
            "You are CORE. Implement ONE highest-leverage source-code "
            "change that advances the mission."
        ),
        "tests": (
            "You are TEST. Implement ONE focused deterministic test "
            "change that proves an important mission invariant."
        ),
        "adversary": (
            "You are ADVERSARY. Implement ONE small correctness, safety, "
            "lifecycle, scheduler, or rollback improvement required by "
            "the mission."
        ),
        "alternate": (
            "You are ALTERNATE. Implement ONE independent high-leverage "
            "mission improvement."
        ),
    }.get(
        role,
        "Implement ONE mission improvement.",
    )

    system = """You are a direct HAIDER mutation emitter.

DO NOT THINK ALOUD.
DO NOT EXPLAIN.
DO NOT WRITE MARKDOWN.
DO NOT WRITE A PLAN.

Your entire response is ONE JSON object.

Exact outer form:

{"operations":[OPERATION, ...]}

IMPORTANT:
- Return one COHERENT transaction.
- Use as many operations as the bounded feature slice actually requires.
- Maximum 20 operations.
- Prefer several precise edits over one giant file replacement.
- Never mutate .git/**.
- Prefer insert_before or insert_after with a SHORT unique anchor.
- Do not repeat large existing source blocks.
- New source code should normally be <= 120 lines.
- New test code should normally be <= 160 lines.
- If creating a test file, create exactly one focused file.
- Return immediately after the JSON closes.

Allowed operations:

{"op":"replace","path":"relative/file.py","old_text":"short exact unique text","new_text":"replacement"}

{"op":"insert_before","path":"relative/file.py","anchor":"short exact unique anchor","text":"new code"}

{"op":"insert_after","path":"relative/file.py","anchor":"short exact unique anchor","text":"new code"}

{"op":"create","path":"relative/new_file.py","content":"complete new file"}
"""

    evidence_text = (
        evidence.get(
            "final",
            "",
        )
    )

    user = (
        f"{role_instruction}\n\n"
        "MISSION:\n"
        f"{mission}\n\n"
        "REPOSITORY EVIDENCE:\n"
        f"{evidence_text}\n\n"
        "Emit ONE compact Mutation-v2 operation now."
    )

    return [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": user,
        },
    ]


def _generate_runtime_candidate(
    slot: Dict[str, Any],
    host: str,
    mission: str,
    evidence: Dict[str, Any],
    role: str,
    runtime_index: int,
) -> Dict[str, Any]:
    """Generate one structured compact builder candidate."""

    started = time.monotonic()

    messages = _compact_builder_prompt(
        mission,
        evidence,
        role,
    )

    initial_budget = int(
        os.environ.get(
            "HAIDER_BUILDER_TOKENS",
            "2200",
        )
    )

    repair_budget = int(
        os.environ.get(
            "HAIDER_BUILDER_REPAIR_TOKENS",
            "3000",
        )
    )

    synth_dir = (
        HAIDER_DIR
        / "synthesis"
    )

    synth_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    attempts: List[Dict[str, Any]] = []

    for attempt in range(2):
        budget = (
            initial_budget
            if attempt == 0
            else repair_budget
        )

        try:
            (
                content,
                usage,
                finish_reason,
                reasoning_content,
            ) = _structured_builder_request(
                host,
                slot["port"],
                messages,
                max_tokens=budget,
            )

        except Exception as exc:
            return {
                "ok": False,
                "role": role,
                "runtime": runtime_index,
                "error": str(exc),
                "attempts": attempts,
                "elapsed_s": round(
                    time.monotonic()
                    - started,
                    3,
                ),
            }

        synth_path = synth_dir / (
            f"structured_{int(time.time())}_"
            f"r{runtime_index}_{role}_"
            f"a{attempt + 1}.json"
        )

        synth_path.write_text(
            content,
            encoding="utf-8",
        )

        operations = (
            _extract_mutation_json(
                content
            )
        )

        parsed = (
            isinstance(
                operations,
                list,
            )
            and 1 <= len(operations) <= MAX_MUTATION_OPERATIONS
        )

        attempt_record = {
            "attempt": attempt + 1,
            "budget": budget,
            "chars": len(content),
            "reasoning_chars": len(
                reasoning_content
            ),
            "finish_reason": finish_reason,
            "parsed": parsed,
            "usage": usage,
            "synthesis_path": str(
                synth_path
            ),
        }

        attempts.append(
            attempt_record
        )

        print(
            "[haider] builder "
            f"r={runtime_index} "
            f"role={role} "
            f"attempt={attempt + 1} "
            f"finish={finish_reason!r} "
            f"chars={len(content)} "
            f"reasoning={len(reasoning_content)} "
            f"parsed={parsed}"
        )

        if parsed:
            return {
                "ok": True,
                "role": role,
                "runtime": runtime_index,
                "operations": operations,
                "chars": len(content),
                "usage": usage,
                "finish_reason": finish_reason,
                "reasoning_chars": len(
                    reasoning_content
                ),
                "synthesis_path": str(
                    synth_path
                ),
                "attempts": attempts,
                "elapsed_s": round(
                    time.monotonic()
                    - started,
                    3,
                ),
            }

        # One automatic repair. Keep it compact instead of merely asking
        # for the same response with a larger budget.
        if attempt == 0:
            messages = list(messages)

            messages.append(
                {
                    "role": "assistant",
                    "content": content[-5000:],
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous JSON was incomplete or invalid. "
                        "Start over. Return a smaller coherent transaction with no more than 12 operations. "
                        "Use a short anchor. Do not explain. "
                        "Close the JSON object early."
                    ),
                }
            )

    return {
        "ok": False,
        "role": role,
        "runtime": runtime_index,
        "operations": None,
        "chars": (
            attempts[-1]["chars"]
            if attempts
            else 0
        ),
        "attempts": attempts,
        "elapsed_s": round(
            time.monotonic()
            - started,
            3,
        ),
    }



def _candidate_operations_valid(
    guard: p0.RepositoryGuard,
    operations: Any,
) -> bool:
    if not isinstance(operations, list):
        print("[haider] candidate reject: operations is not a list")
        return False

    if not operations:
        print("[haider] candidate reject: empty operation list")
        return False

    if len(operations) > MAX_MUTATION_OPERATIONS:
        print(
            "[haider] candidate reject: "
            f"operation_count={len(operations)} "
            f"limit={MAX_MUTATION_OPERATIONS}"
        )
        return False

    for index, operation in enumerate(operations):
        ok, reason = validate_mutation_operation(
            guard,
            operation,
        )

        if not ok:
            path = (
                operation.get("path", "")
                if isinstance(operation, dict)
                else ""
            )

            print(
                "[haider] candidate reject: "
                f"operation={index} "
                f"path={path!r} "
                f"reason={reason!r}"
            )

            return False

    return True



def _parallel_generate(
    runtime_pool: FastRuntimePool,
    mission: str,
    evidence: Dict[str, Any],
    guard: p0.RepositoryGuard,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run physically independent builders concurrently."""

    slots = list(
        runtime_pool.slots
    )

    roles = [
        "core",
        "tests",
        "adversary",
    ]

    while len(roles) < len(slots):
        roles.append(
            "alternate"
        )

    roles = roles[:len(slots)]

    results: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(slots)
    ) as pool:
        futures = []

        for index, (
            slot,
            role,
        ) in enumerate(
            zip(
                slots,
                roles,
            )
        ):
            futures.append(
                pool.submit(
                    _generate_runtime_candidate,
                    slot,
                    runtime_pool.host,
                    mission,
                    evidence,
                    role,
                    index,
                )
            )

        for future in concurrent.futures.as_completed(
            futures
        ):
            result = future.result()

            results.append(
                result
            )

            print(
                "[haider] candidate "
                f"runtime={result.get('runtime')} "
                f"role={result.get('role')} "
                f"ok={result.get('ok')} "
                f"chars={result.get('chars', 0)} "
                f"elapsed={result.get('elapsed_s')}s"
            )

    priority = {
        "core": 0,
        "tests": 1,
        "adversary": 2,
        "alternate": 3,
    }

    valid: List[Dict[str, Any]] = []

    for result in results:
        operations = result.get(
            "operations"
        )

        if not result.get("ok"):
            continue

        if not _candidate_operations_valid(
            guard,
            operations,
        ):
            continue

        valid.append(
            result
        )

    valid.sort(
        key=lambda item: (
            priority.get(
                item.get("role"),
                9,
            ),
            item.get(
                "runtime",
                99,
            ),
        )
    )

    merged: List[Dict[str, Any]] = []
    occupied_paths = set()

    for result in valid:
        operations = result["operations"]

        candidate_paths = {
            str(operation.get("path", ""))
            for operation in operations
        }

        # Do not merge two independently generated candidates that both
        # modify the same file.  But a SINGLE candidate is explicitly
        # allowed to contain many operations across many related files.
        if occupied_paths.intersection(candidate_paths):
            continue

        if len(merged) + len(operations) > MAX_MUTATION_OPERATIONS:
            continue

        merged.extend(operations)
        occupied_paths.update(candidate_paths)

        if len(merged) >= MAX_MUTATION_OPERATIONS:
            break

    return merged, results


def _snapshot_operations(
    guard: p0.RepositoryGuard,
    operations: List[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    snapshot: Dict[str, Optional[str]] = {}

    for operation in operations:
        path = operation.get(
            "path",
            "",
        )

        try:
            full = guard.resolve(path)
        except p0.ToolError:
            continue

        if full in snapshot:
            continue

        if os.path.isfile(full):
            snapshot[full] = Path(
                full
            ).read_text(
                encoding="utf-8",
                errors="ignore",
            )
        else:
            snapshot[full] = None

    return snapshot


def _restore_snapshot(
    snapshot: Dict[str, Optional[str]],
) -> None:
    for full, original in snapshot.items():
        try:
            if original is None:
                if os.path.exists(full):
                    os.unlink(full)
            else:
                Path(full).parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                Path(full).write_text(
                    original,
                    encoding="utf-8",
                )
        except Exception:
            pass


def _run_fast_haider_validation(
    guard: p0.RepositoryGuard,
) -> Dict[str, Any]:
    """Run all focused HAIDER tests.

    They are sub-second, so inference—not tests—is the optimization target.
    """

    tests = [
        "tools/haider/test_haider_mutation_v2.py",
        "tools/haider/test_haider_mission_state.py",
        "tools/haider/test_haider_mission_engine.py",
        "tools/haider/test_haider_fast_engine.py",
        "tools/haider/test_haider_edit.py",
        "tools/haider/test_haider_mission_ingress.py",
    ]

    tests = [
        test
        for test in tests
        if (
            Path(
                guard.root,
                test,
            ).is_file()
        )
    ]

    started = time.monotonic()

    stdout_parts: List[str] = []
    stderr_parts: List[str] = []

    for test in tests:
        proc = subprocess.run(
            [
                sys.executable,
                test,
            ],
            cwd=guard.root,
            capture_output=True,
            text=True,
            timeout=120,
        )

        stdout_parts.append(
            proc.stdout
        )

        stderr_parts.append(
            proc.stderr
        )

        if proc.returncode != 0:
            return {
                "command": (
                    "focused-haider-tests"
                ),
                "exit_code": proc.returncode,
                "duration_ms": int(
                    (
                        time.monotonic()
                        - started
                    )
                    * 1000
                ),
                "stdout": "".join(
                    stdout_parts
                )[-5000:],
                "stderr": "".join(
                    stderr_parts
                )[-5000:],
                "failed_test": test,
            }

    return {
        "command": (
            "focused-haider-tests"
        ),
        "exit_code": 0,
        "duration_ms": int(
            (
                time.monotonic()
                - started
            )
            * 1000
        ),
        "stdout": "".join(
            stdout_parts
        )[-5000:],
        "stderr": "".join(
            stderr_parts
        )[-5000:],
        "tests": tests,
    }


def _append_cycle_evidence(
    state: Dict[str, Any],
    cycle_record: Dict[str, Any],
) -> None:
    history = state.setdefault(
        "history",
        []
    )

    history.append(
        cycle_record
    )


def _resolve_fast_task(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> Tuple[Optional[str], str]:
    if args.task and args.task_file:
        parser.error(
            "--task and --task-file are mutually exclusive"
        )

    if args.task_file:
        path = Path(
            args.task_file
        ).expanduser()

        if not path.is_absolute():
            path = (
                Path.cwd()
                / path
            )

        path = path.resolve()

        if not path.is_file():
            parser.error(
                f"task file not found: {path}"
            )

        text = path.read_text(
            encoding="utf-8",
        ).strip()

        if not text:
            parser.error(
                f"task file is empty: {path}"
            )

        return text, str(path)

    if args.task:
        text = args.task.strip()

        if not text:
            parser.error(
                "--task cannot be empty"
            )

        return text, "inline"

    return None, "gate-zero-default"


def main_fast() -> int:
    # HAIDER itself never wants an interactive pager.
    os.environ["GIT_PAGER"] = "cat"
    os.environ["PAGER"] = "cat"

    parser = argparse.ArgumentParser(
        prog="haider",
        description=(
            "HCLI-v0 accelerated bootstrap"
        ),
    )

    parser.add_argument(
        "n",
        type=int,
        nargs="?",
        default=1,
        help=(
            "Number of physically independent "
            "llama-server runtimes"
        ),
    )

    parser.add_argument(
        "--model",
        default=os.environ.get(
            "HAIDER_MODEL_PATH",
            "model.gguf",
        ),
    )

    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
    )

    parser.add_argument(
        "--ctx-size",
        type=int,
        default=16384,
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-cycles",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--task",
        default=None,
    )

    parser.add_argument(
        "--task-file",
        default=None,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    parser.add_argument(
        "--pool-smoke",
        action="store_true",
        help=(
            "Start N independent runtimes, verify "
            "readiness, print pool, then exit"
        ),
    )

    args = parser.parse_args()

    if args.n < 1:
        parser.error("n must be >= 1")

    if args.max_cycles < 1:
        parser.error(
            "--max-cycles must be >= 1"
        )

    mission_text, mission_source = (
        _resolve_fast_task(
            parser,
            args,
        )
    )

    guard = p0.RepositoryGuard.detect(
        os.getcwd()
    )

    executor = p0.ToolExecutor(
        guard
    )

    runtime_pool = FastRuntimePool(
        args.n,
        args.model,
        args.host,
        args.ctx_size,
        debug=args.debug,
    )

    mission_state = None
    mission_path = None

    if mission_text is not None:
        mission_state = create_mission_state(
            mission_text,
            mission_source,
            args.max_cycles,
        )

        mission_state.setdefault(
            "history",
            []
        )

        mission_path = write_mission_state(
            mission_state
        )

        print(
            f"[haider] mission={mission_path}"
        )

    try:
        print(
            f"[haider] starting RuntimePool N={args.n}"
        )

        runtime_pool.start()

        print(
            f"[haider] RuntimePool READY N={len(runtime_pool.clients)}"
        )

        for slot in runtime_pool.slots:
            print(
                f"[haider] RUNTIME "
                f"index={slot['index']} "
                f"pid={slot['proc'].pid} "
                f"port={slot['port']}"
            )

        if args.pool_smoke:
            print(
                "[haider] POOL_SMOKE PASS"
            )

            return 0

        if mission_text is None:
            print(
                "[haider] no explicit mission; "
                "RuntimePool bootstrap verified"
            )

            return 0

        # --------------------------------------------------------------
        # Bounded persistent mission loop.
        # --------------------------------------------------------------

        while (
            int(
                mission_state.get(
                    "cycle",
                    0,
                )
            )
            < int(
                mission_state.get(
                    "max_cycles",
                    1,
                )
            )
        ):
            cycle_index = int(
                mission_state.get(
                    "cycle",
                    0,
                )
            )

            print(
                f"[haider] ===== CYCLE {cycle_index + 1}/"
                f"{mission_state['max_cycles']} ====="
            )

            evidence = _compact_mission_evidence(
                mission_text,
                guard,
            )

            if evidence is None:
                # Fallback only when mission did not name concrete files.
                print(
                    "[haider] no named-file fast evidence; "
                    "using one bounded observation session"
                )

                client = runtime_pool.clients[
                    cycle_index
                    % len(
                        runtime_pool.clients
                    )
                ]

                session = p0.Session(
                    "parent",
                    mission_text,
                    client,
                    executor,
                    guard,
                    max_turns=args.max_turns,
                    debug=args.debug,
                    emit=lambda msg: print(
                        f"  {msg}"
                    ),
                )

                evidence = session.run()

                if not evidence.get(
                    "ok"
                ):
                    mission_state = (
                        _fast_state_transition(
                            mission_state,
                            mutation_ok=False,
                            validation_ok=False,
                            failure_reason=(
                                "evidence_failed"
                            ),
                        )
                    )

                    write_mission_state(
                        mission_state
                    )

                    print(
                        "[haider] mission failed: "
                        "evidence acquisition"
                    )

                    return 1

            else:
                print(
                    "[haider] fast evidence "
                    f"files="
                    f"{evidence['stats']['fast_evidence_files']} "
                    "model_turns=0"
                )

            operations, candidates = (
                _parallel_generate(
                    runtime_pool,
                    mission_text,
                    evidence,
                    guard,
                )
            )

            if not operations:
                mission_state = (
                    _fast_state_transition(
                        mission_state,
                        mutation_ok=False,
                        validation_ok=False,
                        failure_reason=(
                            "no_valid_parallel_candidate"
                        ),
                    )
                )

                _append_cycle_evidence(
                    mission_state,
                    {
                        "cycle": (
                            cycle_index
                            + 1
                        ),
                        "status": (
                            "mutation_failed"
                        ),
                        "candidates": candidates,
                    },
                )

                write_mission_state(
                    mission_state
                )

                print(
                    "[haider] FATAL: "
                    "no valid parallel mutation"
                )

                return 1

            print(
                f"[haider] merged "
                f"{len(operations)} operations"
            )

            snapshot = _snapshot_operations(
                guard,
                operations,
            )

            mutation_started = (
                time.monotonic()
            )

            mutation = (
                apply_mutation_operations(
                    guard,
                    operations,
                )
            )

            mutation_elapsed = round(
                time.monotonic()
                - mutation_started,
                3,
            )

            if mutation is None:
                _restore_snapshot(
                    snapshot
                )

                mission_state = (
                    _fast_state_transition(
                        mission_state,
                        mutation_ok=False,
                        validation_ok=False,
                        failure_reason=(
                            "mutation_apply_failed"
                        ),
                    )
                )

                write_mission_state(
                    mission_state
                )

                return 1

            print(
                "[haider] mutation "
                f"operations="
                f"{mutation['operation_count']} "
                f"paths="
                f"{mutation['paths']}"
            )

            validation = (
                _run_fast_haider_validation(
                    guard
                )
            )

            validation_ok = (
                validation["exit_code"]
                == 0
            )

            cycle_record = {
                "cycle": (
                    cycle_index
                    + 1
                ),
                "timestamp": _fast_now(),
                "mutation": mutation,
                "mutation_elapsed_s": (
                    mutation_elapsed
                ),
                "validation": validation,
                "candidate_count": len(
                    candidates
                ),
                "runtimes": [
                    {
                        "runtime": c.get(
                            "runtime"
                        ),
                        "role": c.get(
                            "role"
                        ),
                        "ok": c.get(
                            "ok"
                        ),
                        "chars": c.get(
                            "chars",
                            0,
                        ),
                        "elapsed_s": c.get(
                            "elapsed_s"
                        ),
                    }
                    for c in candidates
                ],
            }

            if not validation_ok:
                print(
                    "[haider] validation FAIL; "
                    "rolling back transaction"
                )

                _restore_snapshot(
                    snapshot
                )

                cycle_record[
                    "rolled_back"
                ] = True

                _append_cycle_evidence(
                    mission_state,
                    cycle_record,
                )

                mission_state = (
                    _fast_state_transition(
                        mission_state,
                        mutation_ok=True,
                        validation_ok=False,
                        failure_reason=(
                            "validation_failed_rolled_back"
                        ),
                    )
                )

                write_mission_state(
                    mission_state
                )

                # P0: Do not terminate mission on single validation failure.
                # Continue to next cycle if budget remains.
                continue

            cycle_record[
                "rolled_back"
            ] = False

            _append_cycle_evidence(
                mission_state,
                cycle_record,
            )

            mission_state = (
                _fast_state_transition(
                    mission_state,
                    mutation_ok=True,
                    validation_ok=True,
                )
            )

            write_mission_state(
                mission_state
            )

            print(
                "[haider] validated cycle "
                f"{mission_state['cycle']}"
            )

        print(
            "[haider] bounded mission run complete "
            f"status={mission_state['status']} "
            f"cycles={mission_state['cycle']}"
        )

        # max_cycles means the requested bounded work budget completed.
        # It is not falsely relabeled mission-complete.
        return 0

    finally:
        runtime_pool.stop()


if __name__ == "__main__":
    sys.exit(main_fast())
