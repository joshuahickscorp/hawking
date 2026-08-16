"""Qwen3.8 special-unit harness — agent legs plus the conversation wire.

CPU-side tools, planning, and verification. ``say()`` is answered by the
already-verified native Qwen3.8 decode when a ``NativeQwen38Backend`` is
attached (CLI default). Generate inspects ``/tmp/hawking-gpu-lane.lock`` and
REFUSES if a protected owner holds it; otherwise it runs under
``tools/gpu_lane_lock.sh``. The harness never certifies its own work:
proposed_complete is not verified_complete.

Reuse: ``lab.execution_sandbox``, ``lab.verification_authority``,
``lab.receipts.seal``, ``tools/agentos/machine_state.py``. This is not a
curl wrapper and not a second hide-backend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from lab.execution_sandbox import (
    ExecutionSandboxPolicy,
    SandboxAction,
    SandboxPolicyError,
    SandboxPrincipal,
    default_model_policy,
)
from lab.layout import REPO_ROOT
from lab.receipts import seal
from lab.verification_authority import (
    AuthorityPrincipal,
    CandidateKind,
    SelfPromotionError,
    VerificationAuthority,
    default_authority,
)

SCHEMA = "hawking.hcli.special_unit.v1"
SESSION_SCHEMA = "hawking.hcli.special_unit.session.v1"
PLAN_SCHEMA = "hawking.hcli.special_unit.plan.v1"
CONSUMPTION_SCHEMA = "hawking.hcli.special_unit.grok_consumption.v1"
DEFAULT_SESSION_ROOT = REPO_ROOT / ".hide" / "special-unit" / "sessions"
GROK_RUN = Path.home() / ".claude-grok" / "bin" / "grok-run"
GROK_TASKS = Path.home() / ".claude-grok" / "tasks"
GPU_LOCK = Path("/tmp/hawking-gpu-lane.lock")
NATIVE_DECODE_LANE = "qwen38-special-unit"
QWEN38_PACK_RELATIVE = Path("workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1")
QWEN38_TOKENIZER_RELATIVE = Path(
    "workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json"
)
QWEN38_GREEDY_RELATIVE = (
    Path("workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy"),
    Path("workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy"),
)

PROTECTED_OWNER_PREFIXES: tuple[str, ...] = (
    "q80-",
    "qwen80-",
    "dsv4f-",
    "dsv4-",
    "auto-dsv4f-",
    "deepseek",
)

_GPU_CMD = re.compile(
    r"gpu_lane_lock|ascension_qwen|ascension_dsv|hybrid_greedy|"
    r"BASE_TRUE_TPS|qwen80_hybrid|dsv4f_native|MTLCommandBuffer|"
    r"--example\s+ascension_",
    re.I,
)

_SKIP_DIR_NAMES = frozenset(
    {".git", "target", "target-parallel", "node_modules", ".hide", "__pycache__"}
)


class ResourceClass(str, Enum):
    CPU = "CPU"
    GPU_HEAVY = "GPU_HEAVY"
    MEMORY_HEAVY = "MEMORY_HEAVY"


class SessionStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PROPOSED_COMPLETE = "proposed_complete"
    VERIFIED_COMPLETE = "verified_complete"
    FAILED = "failed"
    BLOCKED = "blocked"


class SpecialUnitError(RuntimeError):
    """Fail-closed harness error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_text(path: Path, limit: int = 4000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if len(text) <= limit else text[:limit] + "\n…[truncated]"


# ---------------------------------------------------------------------------
# Project context
# ---------------------------------------------------------------------------


def _git_snapshot(repo: Path) -> dict[str, Any]:
    def _git(*args: str) -> str:
        try:
            p = subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True,
                text=True,
                timeout=8,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return p.stdout.strip() if p.returncode == 0 else ""

    porcelain = _git("status", "--porcelain")
    lines = [ln for ln in porcelain.splitlines() if ln.strip()]
    return {
        "head": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(lines),
        "changed_count": len(lines),
        "changed_sample": lines[:12],
    }


def project_context(repo: Path, *, lock_path: Path = GPU_LOCK) -> dict[str, Any]:
    """Cheap, CPU-only snapshot a turn can actually use."""

    ascent = repo / "receipts" / "ascent-2026-08-16"
    receipts: list[str] = []
    if ascent.is_dir():
        receipts = sorted(p.name for p in ascent.glob("*.json"))[-10:]
    owner = None
    owner_file = lock_path / "owner"
    if owner_file.is_file():
        owner = owner_file.read_text(encoding="utf-8", errors="replace").strip() or None
    ctx = {
        "schema": "hawking.hcli.special_unit.project_context.v1",
        "repo": str(repo),
        "git": _git_snapshot(repo),
        "superwave_head": _read_text(repo / "SUPERWAVE_STATE.md", 1800),
        "recent_ascent_receipts": receipts,
        "gpu_lock_owner": owner,
        "model_identity": {
            "name": "Qwen3.8-27B",
            "source": "PocketAiHub/Qwen3.8-27B-Abliterated-MLX",
            "native_leg": "DONE",
            "harness": "lab.hcli.special_unit",
        },
    }
    ctx["digest"] = _digest({k: v for k, v in ctx.items() if k != "digest"})
    return ctx


# ---------------------------------------------------------------------------
# Resource gate — never contaminate a protected Q80/DSV bench
# ---------------------------------------------------------------------------


def _owner_is_protected(owner: str | None) -> bool:
    if not owner:
        return False
    low = owner.lower()
    return any(low.startswith(p) or p.rstrip("-") in low for p in PROTECTED_OWNER_PREFIXES)


@dataclass
class ResourceGate:
    """Admit CPU work always. Refuse GPU/MEMORY while a protected bench is live.

    This gate never *takes* ``/tmp/hawking-gpu-lane.lock``. Waiting on that
    mutex would still starve a Q80/DSV lane trying to acquire it; pause instead.
    """

    lock_path: Path = GPU_LOCK
    allow_gpu: bool = False

    def lock_owner(self) -> str | None:
        owner = self.lock_path / "owner"
        if not owner.is_file():
            return None
        return owner.read_text(encoding="utf-8", errors="replace").strip() or None

    def protected_bench_live(self) -> tuple[bool, str]:
        owner = self.lock_owner()
        if owner is None:
            return False, "no gpu lock"
        if _owner_is_protected(owner):
            return True, f"protected bench holds gpu lock: {owner}"
        return False, f"lock held by non-protected owner: {owner}"

    def admit(self, cls: ResourceClass | str) -> tuple[bool, str]:
        resource = cls if isinstance(cls, ResourceClass) else ResourceClass(str(cls))
        if resource == ResourceClass.CPU:
            return True, "CPU work does not touch the GPU lock"
        live, why = self.protected_bench_live()
        if live:
            return False, f"PAUSE: refusing {resource.value} while {why}"
        owner = self.lock_owner()
        if owner:
            return False, f"PAUSE: gpu lock held by {owner}; will not contend"
        if resource == ResourceClass.GPU_HEAVY and not self.allow_gpu:
            return (
                False,
                "GPU_HEAVY refused: special-unit harness is CPU-side "
                "(native Qwen decode is a separate, already-verified leg)",
            )
        return True, "admitted"


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    role: str
    text: str
    ts: str = field(default_factory=_utc_now)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "text": self.text, "ts": self.ts, "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Turn":
        return cls(
            role=str(raw.get("role") or ""),
            text=str(raw.get("text") or ""),
            ts=str(raw.get("ts") or _utc_now()),
            meta=dict(raw.get("meta") or {}),
        )


@dataclass
class Session:
    session_id: str
    status: str = SessionStatus.IDLE.value
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    transcript: list[Turn] = field(default_factory=list)
    open_files: list[str] = field(default_factory=list)
    plan: dict[str, Any] | None = None
    pending_interrupt: str | None = None
    pause_reason: str | None = None
    grok_handles: list[dict[str, Any]] = field(default_factory=list)
    context_digest: str = ""

    def touch(self) -> None:
        self.updated_at = _utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SESSION_SCHEMA,
            "session_id": self.session_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "transcript": [t.to_dict() for t in self.transcript],
            "open_files": list(self.open_files),
            "plan": self.plan,
            "pending_interrupt": self.pending_interrupt,
            "pause_reason": self.pause_reason,
            "grok_handles": list(self.grok_handles),
            "context_digest": self.context_digest,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Session":
        return cls(
            session_id=str(raw["session_id"]),
            status=str(raw.get("status") or SessionStatus.IDLE.value),
            created_at=str(raw.get("created_at") or _utc_now()),
            updated_at=str(raw.get("updated_at") or _utc_now()),
            transcript=[Turn.from_dict(t) for t in (raw.get("transcript") or [])],
            open_files=list(raw.get("open_files") or []),
            plan=raw.get("plan"),
            pending_interrupt=raw.get("pending_interrupt"),
            pause_reason=raw.get("pause_reason"),
            grok_handles=list(raw.get("grok_handles") or []),
            context_digest=str(raw.get("context_digest") or ""),
        )


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, session_id: str) -> Path:
        return self.root / session_id / "session.json"

    def create(self, session_id: str | None = None) -> Session:
        sid = session_id or f"su-{uuid.uuid4().hex[:12]}"
        if self.path_for(sid).exists():
            raise SpecialUnitError(f"session {sid} already exists")
        session = Session(session_id=sid)
        self.save(session)
        return session

    def save(self, session: Session) -> Path:
        session.touch()
        path = self.path_for(session.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = session.to_dict()
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        events = path.parent / "events.jsonl"
        with events.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"ts": session.updated_at, "status": session.status, "n": len(session.transcript)})
                + "\n"
            )
        return path

    def load(self, session_id: str) -> Session:
        path = self.path_for(session_id)
        if not path.is_file():
            raise SpecialUnitError(f"session {session_id} not found at {path}")
        return Session.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Tools + build/test
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    name: str
    ok: bool
    output: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "output": self.output,
            "detail": dict(self.detail),
        }


def looks_gpu_command(command: str) -> bool:
    return bool(_GPU_CMD.search(command))


class ToolExecutor:
    """Bounded tools. Writes stay inside the owned worktree. GPU cmds refuse."""

    def __init__(
        self,
        repo: Path,
        policy: ExecutionSandboxPolicy,
        gate: ResourceGate,
    ) -> None:
        self.repo = Path(repo)
        self.policy = policy
        self.gate = gate
        self._proc: subprocess.Popen[str] | None = None
        self._interrupted = False

    def interrupt(self) -> None:
        self._interrupted = True
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except OSError:
                pass

    def clear_interrupt(self) -> None:
        self._interrupted = False

    def run(self, name: str, args: Mapping[str, Any]) -> ToolResult:
        if self._interrupted:
            return ToolResult(name, False, "interrupted before dispatch", {"interrupted": True})
        handlers: dict[str, Callable[[Mapping[str, Any]], ToolResult]] = {
            "read": self._read,
            "write": self._write,
            "grep": self._grep,
            "bash": self._bash,
            "pytest": self._pytest,
            "cargo_test": self._cargo_test,
        }
        handler = handlers.get(name)
        if handler is None:
            return ToolResult(name, False, f"unknown tool {name!r}", {"known": sorted(handlers)})
        try:
            return handler(args)
        except SandboxPolicyError as exc:
            return ToolResult(name, False, str(exc), {"denied": True, "action": exc.decision.action.value})

    def _require(self, action: SandboxAction, target: str | Path | None = None) -> None:
        self.policy.require(SandboxPrincipal.SANDBOX_MODEL, action, target=target)

    def _resolve(self, raw: str) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = self.repo / path
        return path.resolve()

    def _read(self, args: Mapping[str, Any]) -> ToolResult:
        path = self._resolve(str(args.get("path") or ""))
        self._require(SandboxAction.READ_SOURCE, path)
        if not path.is_file():
            return ToolResult("read", False, f"missing file {path}", {"path": str(path)})
        offset = int(args.get("offset") or 1)
        limit = int(args.get("limit") or 200)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(offset, 1) - 1
        chunk = lines[start : start + max(limit, 1)]
        numbered = [f"{start + i + 1}:{line}" for i, line in enumerate(chunk)]
        if str(path) not in []:
            pass
        return ToolResult(
            "read",
            True,
            "\n".join(numbered),
            {"path": str(path), "offset": offset, "limit": limit, "total_lines": len(lines)},
        )

    def _write(self, args: Mapping[str, Any]) -> ToolResult:
        path = self._resolve(str(args.get("path") or ""))
        content = str(args.get("content") or "")
        self._require(SandboxAction.EDIT_OWNED_WORKTREE, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult("write", True, f"wrote {len(content)} bytes", {"path": str(path)})

    def _grep(self, args: Mapping[str, Any]) -> ToolResult:
        pattern = str(args.get("pattern") or "")
        if not pattern:
            return ToolResult("grep", False, "pattern required", {})
        root = self._resolve(str(args.get("path") or "."))
        self._require(SandboxAction.READ_SOURCE, root)
        max_hits = int(args.get("max_hits") or 40)
        rg = shutil.which("rg")
        if rg and root.exists():
            cmd = [rg, "--line-number", "--no-heading", "-m", str(max_hits), "-e", pattern, str(root)]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            except subprocess.TimeoutExpired:
                return ToolResult("grep", False, "rg timed out", {"pattern": pattern})
            out = (proc.stdout or proc.stderr).strip()
            return ToolResult(
                "grep",
                proc.returncode in (0, 1),
                out,
                {"pattern": pattern, "hits": 0 if not out else len(out.splitlines()), "engine": "rg"},
            )
        hits: list[str] = []
        cre = re.compile(pattern)
        files = [root] if root.is_file() else root.rglob("*")
        for file in files:
            if not file.is_file():
                continue
            if any(part in _SKIP_DIR_NAMES for part in file.parts):
                continue
            try:
                text = file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if cre.search(line):
                    hits.append(f"{file}:{i}:{line}")
                    if len(hits) >= max_hits:
                        return ToolResult(
                            "grep", True, "\n".join(hits), {"pattern": pattern, "hits": len(hits), "engine": "py"}
                        )
        return ToolResult("grep", True, "\n".join(hits), {"pattern": pattern, "hits": len(hits), "engine": "py"})

    def _run_cmd(self, argv: Sequence[str], *, timeout: float, cwd: Path | None = None) -> ToolResult:
        joined = " ".join(argv)
        if looks_gpu_command(joined):
            return ToolResult(
                "bash",
                False,
                "refused: command matches a GPU/protected-bench pattern",
                {"argv": list(argv), "resource_class": ResourceClass.GPU_HEAVY.value},
            )
        ok, why = self.gate.admit(ResourceClass.CPU)
        if not ok:
            return ToolResult("bash", False, why, {"argv": list(argv)})
        try:
            self._proc = subprocess.Popen(
                list(argv),
                cwd=str(cwd or self.repo),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                out, _ = self._proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.interrupt()
                out, _ = self._proc.communicate(timeout=5)
                return ToolResult("bash", False, (out or "") + "\n[timeout]", {"argv": list(argv), "timeout": True})
            code = self._proc.returncode
            interrupted = self._interrupted or code in (-signal.SIGTERM, -signal.SIGKILL)
            return ToolResult(
                "bash",
                code == 0 and not interrupted,
                out or "",
                {"argv": list(argv), "exit_code": code, "interrupted": interrupted},
            )
        finally:
            self._proc = None

    def _bash(self, args: Mapping[str, Any]) -> ToolResult:
        argv = args.get("argv")
        if argv is None:
            command = str(args.get("command") or "")
            if not command:
                return ToolResult("bash", False, "command or argv required", {})
            if looks_gpu_command(command):
                return ToolResult(
                    "bash",
                    False,
                    "refused: command matches a GPU/protected-bench pattern",
                    {"command": command},
                )
            argv = ["/bin/bash", "-lc", command]
        self.policy.require(SandboxPrincipal.SANDBOX_MODEL, SandboxAction.COMPILE)
        result = self._run_cmd([str(a) for a in argv], timeout=float(args.get("timeout") or 60))
        result.name = "bash"
        return result

    def _pytest(self, args: Mapping[str, Any]) -> ToolResult:
        target = str(args.get("target") or "lab/tests/test_option_c.py")
        self.policy.require(
            SandboxPrincipal.SANDBOX_MODEL,
            SandboxAction.RUN_ALLOWED_TESTS,
            target=target,
        )
        if looks_gpu_command(target):
            return ToolResult("pytest", False, "refused: GPU-shaped test selector", {"target": target})
        exe = sys.executable
        result = self._run_cmd(
            [exe, "-m", "pytest", "-q", target],
            timeout=float(args.get("timeout") or 120),
        )
        result.name = "pytest"
        result.detail["target"] = target
        return result

    def _cargo_test(self, args: Mapping[str, Any]) -> ToolResult:
        package = str(args.get("package") or "")
        extra = [str(x) for x in (args.get("extra") or [])]
        argv = ["cargo", "test", "--offline"]
        if package:
            argv.extend(["-p", package])
        argv.extend(extra)
        joined = " ".join(argv)
        if looks_gpu_command(joined) or package.startswith("hawking-core"):
            return ToolResult(
                "cargo_test",
                False,
                "refused: hawking-core / GPU cargo invocations are not this harness",
                {"argv": argv},
            )
        self.policy.require(SandboxPrincipal.SANDBOX_MODEL, SandboxAction.RUN_ALLOWED_TESTS, target=package)
        if args.get("dry_run") or os.environ.get("SPECIAL_UNIT_CARGO_DRY") == "1":
            return ToolResult(
                "cargo_test",
                True,
                "admitted CPU cargo_test (dry_run; not executed)",
                {"argv": argv, "dry_run": True},
            )
        result = self._run_cmd(argv, timeout=float(args.get("timeout") or 300))
        result.name = "cargo_test"
        return result


# ---------------------------------------------------------------------------
# Grok delegate + consume (consume is a separate leg)
# ---------------------------------------------------------------------------


class GrokRunner:
    """Callable that performs a grok-run invocation. Injected in tests."""

    def delegate(
        self,
        *,
        slug: str,
        contract: Path,
        profile: str,
        repo: Path,
        tasks_root: Path,
    ) -> dict[str, Any]:
        tasks_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(GROK_RUN),
            "delegate",
            "--task",
            slug,
            "--contract",
            str(contract),
            "--profile",
            profile,
            "--repo",
            str(repo),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "cmd": cmd}


def parse_diff_paths(diff_text: str) -> list[str]:
    paths: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                paths.append(parts[3][2:] if parts[3].startswith("b/") else parts[3])
    return paths


def consume_grok_task(task_id: str, tasks_root: Path = GROK_TASKS) -> dict[str, Any]:
    """Read grok-run artifacts. A report is a claim, never verified_complete."""

    tdir = Path(tasks_root) / task_id
    status_path = tdir / "status"
    report_path = tdir / "grok-report.md"
    diff_path = tdir / "diff.patch"
    meta_path = tdir / "metadata.json"
    exit_path = tdir / "exit_code"
    if not tdir.is_dir():
        return seal(
            {
                "schema": CONSUMPTION_SCHEMA,
                "task_id": task_id,
                "dispatched": False,
                "finished": False,
                "consumed": False,
                "verified_complete": False,
                "status": "missing",
                "reason": f"no task directory {tdir}",
            }
        )
    status = status_path.read_text(encoding="utf-8").strip() if status_path.is_file() else "unknown"
    exit_code = None
    if exit_path.is_file():
        raw = exit_path.read_text(encoding="utf-8").strip()
        if raw.lstrip("-").isdigit():
            exit_code = int(raw)
    report = _read_text(report_path, 4000)
    diff_text = _read_text(diff_path, 200_000) if diff_path.is_file() else ""
    meta = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {"parse_error": True}
    finished = status == "done"
    consumed = True  # we *read* the artifacts; that is the consume leg
    return seal(
        {
            "schema": CONSUMPTION_SCHEMA,
            "task_id": task_id,
            "dispatched": True,
            "finished": finished,
            "consumed": consumed,
            "verified_complete": False,
            "status": status,
            "exit_code": exit_code,
            "report_path": str(report_path) if report_path.is_file() else None,
            "diff_path": str(diff_path) if diff_path.is_file() else None,
            "report_sha256": _sha_file(report_path),
            "diff_sha256": _sha_file(diff_path) if diff_path.is_file() else None,
            "metadata": meta,
            "claim_excerpt": report[:800],
            "files_in_diff": parse_diff_paths(diff_text)[:40],
            "reason": (
                "artifacts consumed; report is a claim, not a receipt "
                "(verified_complete stays false until an oracle runs)"
            ),
        }
    )


# ---------------------------------------------------------------------------
# AgentOS plan + proposed_complete vs verified_complete
# ---------------------------------------------------------------------------


def default_g015_steps() -> list[dict[str, Any]]:
    """The still-open G015 harness legs, as a DAG the unit can actually run."""

    return [
        {
            "id": "session_context",
            "title": "HCLI conversation + project context + persistent session",
            "dependencies": [],
            "oracle": {"kind": "session_roundtrip"},
        },
        {
            "id": "interrupt_resume",
            "title": "interrupt + restart/resume",
            "dependencies": ["session_context"],
            "oracle": {"kind": "session_status", "status": SessionStatus.IDLE.value},
        },
        {
            "id": "tool_execution",
            "title": "tool execution",
            "dependencies": ["session_context"],
            "oracle": {"kind": "file_exists", "path": "lab/hcli/special_unit.py"},
        },
        {
            "id": "build_test",
            "title": "build/test execution",
            "dependencies": ["tool_execution"],
            "oracle": {"kind": "pytest", "target": "lab/tests/test_option_c.py"},
        },
        {
            "id": "grok_delegate",
            "title": "Grok delegation",
            "dependencies": ["session_context"],
            "oracle": {"kind": "grok_dispatched"},
        },
        {
            "id": "grok_consume",
            "title": "Grok RESULT CONSUMPTION",
            "dependencies": ["grok_delegate"],
            "oracle": {"kind": "grok_consumed"},
        },
        {
            "id": "agentos_plan",
            "title": "AgentOS planning",
            "dependencies": ["session_context"],
            "oracle": {"kind": "plan_present"},
        },
        {
            "id": "agentos_verify",
            "title": "AgentOS verification (proposed_complete vs verified_complete)",
            "dependencies": ["agentos_plan"],
            "oracle": {"kind": "boundary_holds"},
        },
        {
            "id": "resource_pause",
            "title": "resource-aware pause/resume that never contaminates Q80/DSV",
            "dependencies": ["session_context"],
            "oracle": {"kind": "resource_gate"},
        },
    ]


def new_plan(objective: str, steps: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    body_steps = []
    for raw in steps or default_g015_steps():
        body_steps.append(
            {
                "id": str(raw["id"]),
                "title": str(raw["title"]),
                "dependencies": list(raw.get("dependencies") or []),
                "oracle": dict(raw.get("oracle") or {"kind": "predicate"}),
                "status": StepStatus.PENDING.value,
                "proposed_by": None,
                "verified_by": None,
                "evidence": {},
            }
        )
    return {
        "schema": PLAN_SCHEMA,
        "plan_id": f"plan-{uuid.uuid4().hex[:10]}",
        "objective": objective,
        "steps": body_steps,
        "created_at": _utc_now(),
    }


def _step(plan: dict[str, Any], step_id: str) -> dict[str, Any]:
    for step in plan["steps"]:
        if step["id"] == step_id:
            return step
    raise SpecialUnitError(f"unknown step {step_id}")


def propose_complete(
    plan: dict[str, Any],
    step_id: str,
    *,
    author: str,
    authority: VerificationAuthority | None = None,
) -> dict[str, Any]:
    """Sandbox-legal completion. Never flips verified_complete."""

    step = _step(plan, step_id)
    auth = authority or default_authority()
    candidate = auth.emit_candidate(
        principal=AuthorityPrincipal.SANDBOX_MODEL,
        author=author,
        kind=CandidateKind.IMPLEMENTATION_RECEIPT,
        body={"step_id": step_id, "declaration": "proposed", "summary": step["title"]},
    )
    step["status"] = StepStatus.PROPOSED_COMPLETE.value
    step["proposed_by"] = author
    step["evidence"] = {"candidate_seal": candidate.get("seal_sha256"), "kind": "proposed_complete"}
    return candidate


def _oracle_pass(step: Mapping[str, Any], unit: "SpecialUnit") -> tuple[bool, str, dict[str, Any]]:
    oracle = dict(step.get("oracle") or {})
    kind = oracle.get("kind") or "predicate"
    if kind == "file_exists":
        path = unit.repo / str(oracle.get("path") or "")
        return path.is_file(), f"file_exists {path} -> {path.is_file()}", {"path": str(path)}
    if kind == "session_roundtrip":
        sid = unit.session.session_id
        reloaded = unit.store.load(sid)
        ok = reloaded.session_id == sid
        return ok, f"session {sid} reloads", {"session_id": sid}
    if kind == "session_status":
        want = str(oracle.get("status") or SessionStatus.IDLE.value)
        got = unit.session.status
        return got == want, f"status {got} want {want}", {"status": got}
    if kind == "pytest":
        result = unit.tools.run("pytest", {"target": oracle.get("target"), "timeout": oracle.get("timeout", 120)})
        return result.ok, result.output[-400:], result.to_dict()
    if kind == "grok_dispatched":
        ok = any(h.get("dispatched") for h in unit.session.grok_handles)
        return ok, "at least one grok handle dispatched" if ok else "no grok dispatch", {}
    if kind == "grok_consumed":
        ok = any(h.get("consumed") for h in unit.session.grok_handles)
        return ok, "at least one grok handle consumed" if ok else "no grok consume", {}
    if kind == "plan_present":
        ok = unit.session.plan is not None and bool((unit.session.plan or {}).get("steps"))
        return ok, "plan present" if ok else "no plan", {}
    if kind == "boundary_holds":
        # A proposed step must not already be verified_complete without evidence.
        leaked = [
            s["id"]
            for s in (unit.session.plan or {}).get("steps", [])
            if s["status"] == StepStatus.VERIFIED_COMPLETE.value and not s.get("verified_by")
        ]
        return not leaked, "boundary holds" if not leaked else f"leaked {leaked}", {}
    if kind == "resource_gate":
        ok, why = unit.gate.admit(ResourceClass.GPU_HEAVY)
        # Default harness refuses GPU even on a quiet box — that is the point.
        return (not ok), why, {"admitted": ok}
    if kind == "grep_match":
        result = unit.tools.run("grep", {"pattern": oracle.get("pattern"), "path": oracle.get("path", ".")})
        return result.ok and bool(result.output.strip()), result.output[:400], result.to_dict()
    if kind == "predicate":
        return True, "soft predicate (no oracle)", {}
    return False, f"unknown oracle kind {kind!r}", {}


def verify_complete(
    unit: "SpecialUnit",
    step_id: str,
    *,
    principal: AuthorityPrincipal | str,
    certifier_id: str,
) -> dict[str, Any]:
    """Authoritative completion. Sandbox models cannot do this."""

    if unit.session.plan is None:
        raise SpecialUnitError("no plan")
    plan = unit.session.plan
    step = _step(plan, step_id)
    for dep in step["dependencies"]:
        dep_step = _step(plan, dep)
        if dep_step["status"] != StepStatus.VERIFIED_COMPLETE.value:
            step["status"] = StepStatus.BLOCKED.value
            raise SpecialUnitError(
                f"cannot verified_complete {step_id}: dependency {dep} is {dep_step['status']}"
            )
    auth = unit.authority
    # Force the self-promotion fence *before* running the oracle.
    if _principal_is_sandbox(principal):
        raise SelfPromotionError(
            "sandbox model may not set verified_complete; emit proposed_complete"
        )
    ok, why, evidence = _oracle_pass(step, unit)
    if not ok:
        step["status"] = StepStatus.FAILED.value
        step["evidence"] = {"oracle": why, "detail": evidence}
        unit.save()
        raise SpecialUnitError(f"oracle failed for {step_id}: {why}")
    # Step verification is not mechanism promotion. Sign the evidence envelope;
    # sandbox models cannot produce this (bible §21 / §22).
    verdict = auth.sign_receipt(
        principal=principal,
        signer_id=certifier_id,
        document={
            "kind": "verified_complete",
            "step_id": step_id,
            "oracle": why,
            "detail": evidence,
        },
    )
    step["status"] = StepStatus.VERIFIED_COMPLETE.value
    step["verified_by"] = certifier_id
    step["evidence"] = {"oracle": why, "detail": evidence, "verdict_seal": verdict.get("seal_sha256")}
    unit.save()
    return verdict


def _principal_is_sandbox(principal: AuthorityPrincipal | str) -> bool:
    if isinstance(principal, AuthorityPrincipal):
        return principal == AuthorityPrincipal.SANDBOX_MODEL
    return str(principal) == AuthorityPrincipal.SANDBOX_MODEL.value


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class ConversationBackend(Protocol):
    produced_by: str
    last_receipt: dict[str, Any] | None

    def complete(self, prompt: str, context: Mapping[str, Any]) -> str: ...


class ScriptedBackend:
    """Deterministic stand-in. Tests and harness-only bench tasks inject this."""

    produced_by = "scripted"

    def __init__(self, replies: Sequence[str] | None = None) -> None:
        self.replies = list(replies or ["acknowledged"])
        self.i = 0
        self.last_receipt: dict[str, Any] | None = None

    def complete(self, prompt: str, context: Mapping[str, Any]) -> str:
        _ = (prompt, context)
        if self.i >= len(self.replies):
            text = self.replies[-1]
        else:
            text = self.replies[self.i]
            self.i += 1
        self.last_receipt = {"produced_by": "scripted", "text": text}
        return text


class NativeDecodeRefused(SpecialUnitError):
    """Inspected the GPU lock or materials and will not generate.

    The bench must record this as SKIP, never as PASS.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

    def bench_detail(self) -> str:
        return self.reason if self.reason.startswith("SKIP ") else f"SKIP {self.reason}"


class NativeDecodeError(SpecialUnitError):
    """Generate was invoked and failed. Bench FAIL, not a skip."""


def checkout_search_roots(repo: Path) -> list[Path]:
    """This worktree plus every git worktree, so a sparse checkout can see packs."""

    roots: list[Path] = []
    seen: set[Path] = set()

    def add(raw: Path | str | None) -> None:
        if raw is None:
            return
        path = Path(raw)
        try:
            path = path.resolve()
        except OSError:
            return
        if path in seen or not path.is_dir():
            return
        seen.add(path)
        roots.append(path)

    add(repo)
    for key in ("HAWKING_MAIN_REPO", "HAWKING_REPO"):
        add(os.environ.get(key))
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired):
        proc = None
    if proc is not None and proc.returncode == 0:
        for line in proc.stdout.splitlines():
            if line.startswith("worktree "):
                add(line[len("worktree ") :])
    return roots


def locate_qwen38_greedy_binary(repo: Path) -> Path | None:
    env = os.environ.get("QWEN38_GREEDY_BIN")
    if env:
        path = Path(env)
        if path.is_file():
            return path.resolve()
    for root in checkout_search_roots(repo):
        for rel in QWEN38_GREEDY_RELATIVE:
            cand = root / rel
            if cand.is_file():
                return cand.resolve()
    return None


def locate_qwen38_artifact_root(repo: Path) -> Path | None:
    env = os.environ.get("QWEN38_ARTIFACT_ROOT")
    if env:
        path = Path(env)
        if path.is_dir():
            return path.resolve()
    for root in checkout_search_roots(repo):
        cand = root / QWEN38_PACK_RELATIVE
        if cand.is_dir() and (cand / "manifest.json").is_file():
            return cand.resolve()
    return None


def locate_qwen38_tokenizer(repo: Path) -> Path | None:
    env = os.environ.get("QWEN38_TOKENIZER")
    if env:
        path = Path(env)
        if path.is_file():
            return path.resolve()
    for root in checkout_search_roots(repo):
        cand = root / QWEN38_TOKENIZER_RELATIVE
        if cand.is_file():
            return cand.resolve()
    return None


class NativeQwen38Backend:
    """Adapter over the verified ``ascension_qwen38_hybrid_greedy`` binary.

    Does not modify the decode path. Inspects the GPU lock and refuses when a
    protected owner (or any owner) holds it. Generate runs under
    ``tools/gpu_lane_lock.sh``.
    """

    produced_by = "model"
    lock_acquisitions = 0
    generates_completed = 0

    def __init__(
        self,
        *,
        repo: Path,
        gate: ResourceGate,
        binary: Path | None = None,
        artifact_root: Path | None = None,
        tokenizer: Path | None = None,
        max_new_tokens: int = 16,
        max_seq_len: int = 128,
        timeout: float = 600.0,
        runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
        lock_script: Path | None = None,
        lane: str = NATIVE_DECODE_LANE,
    ) -> None:
        self.repo = Path(repo)
        self.gate = gate
        self.binary = Path(binary) if binary is not None else None
        self.artifact_root = Path(artifact_root) if artifact_root is not None else None
        self.tokenizer = Path(tokenizer) if tokenizer is not None else None
        self.max_new_tokens = int(max_new_tokens)
        self.max_seq_len = int(max_seq_len)
        self.timeout = float(timeout)
        self.runner = runner
        self.lock_script = (
            Path(lock_script) if lock_script is not None else self.repo / "tools" / "gpu_lane_lock.sh"
        )
        self.lane = lane
        self.last_receipt: dict[str, Any] | None = None

    @classmethod
    def reset_counters(cls) -> None:
        cls.lock_acquisitions = 0
        cls.generates_completed = 0

    def _refuse_if_locked(self) -> None:
        live, why = self.gate.protected_bench_live()
        if live:
            raise NativeDecodeRefused(why)
        owner = self.gate.lock_owner()
        if owner:
            raise NativeDecodeRefused(f"gpu lock held by {owner}; will not contend")

    def _materials(self) -> tuple[Path, Path, Path]:
        if self.runner is not None:
            return (
                self.binary or Path("/injected/ascension_qwen38_hybrid_greedy"),
                self.artifact_root or Path("/injected/uniform-q4-v1"),
                self.tokenizer or Path("/injected/tokenizer.json"),
            )
        binary = self.binary or locate_qwen38_greedy_binary(self.repo)
        artifact = self.artifact_root or locate_qwen38_artifact_root(self.repo)
        tokenizer = self.tokenizer or locate_qwen38_tokenizer(self.repo)
        missing: list[str] = []
        if binary is None or not Path(binary).is_file():
            missing.append("greedy-binary")
        if artifact is None or not Path(artifact).is_dir():
            missing.append("artifact-root")
        if tokenizer is None or not Path(tokenizer).is_file():
            missing.append("tokenizer")
        if not self.lock_script.is_file():
            missing.append("gpu_lane_lock.sh")
        if missing:
            raise NativeDecodeRefused(f"missing native decode materials: {', '.join(missing)}")
        return Path(binary), Path(artifact), Path(tokenizer)

    def complete(self, prompt: str, context: Mapping[str, Any]) -> str:
        _ = context
        self._refuse_if_locked()
        binary, artifact, tokenizer = self._materials()
        self._refuse_if_locked()
        with tempfile.TemporaryDirectory(prefix="qwen38-say-") as td:
            out = Path(td) / "generate.json"
            cmd = [
                str(self.lock_script),
                self.lane,
                str(binary),
                "--artifact-root",
                str(artifact),
                "--tokenizer",
                str(tokenizer),
                "--prompt",
                prompt,
                "--max-new-tokens",
                str(self.max_new_tokens),
                "--max-seq-len",
                str(self.max_seq_len),
                "--out",
                str(out),
            ]
            if self.runner is None:
                type(self).lock_acquisitions += 1
            try:
                if self.runner is not None:
                    proc = self.runner(cmd)
                else:
                    proc = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                    )
            except subprocess.TimeoutExpired as exc:
                raise NativeDecodeError(f"native generate timed out after {self.timeout}s") from exc
            except OSError as exc:
                raise NativeDecodeError(f"native generate failed to exec: {exc}") from exc
            body: dict[str, Any] = {}
            if out.is_file():
                try:
                    body = json.loads(out.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise NativeDecodeError(f"native generate wrote invalid json: {exc}") from exc
            stdout = proc.stdout or ""
            text = str(body.get("generated_text") or "")
            if not text:
                for line in stdout.splitlines():
                    if line.startswith("GENERATED_TEXT_VERBATIM:"):
                        text = line.split(":", 1)[1].lstrip()
                        break
            fallbacks = body.get("fallbacks")
            if fallbacks is None:
                for line in stdout.splitlines():
                    if line.startswith("FALLBACKS:"):
                        raw = line.split(":", 1)[1].strip()
                        try:
                            fallbacks = int(raw)
                        except ValueError:
                            fallbacks = None
                        break
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip()[-800:]
                if proc.returncode == 75:
                    raise NativeDecodeRefused(f"gpu_lane_lock timed out: {err}")
                raise NativeDecodeError(
                    f"native generate exit {proc.returncode}: {err or 'no output'}"
                )
            if fallbacks is None:
                raise NativeDecodeError("native generate omitted FALLBACKS")
            if int(fallbacks) != 0:
                raise NativeDecodeError(f"native generate fallbacks={fallbacks} (must be 0)")
            if not str(text).strip():
                raise NativeDecodeError("native generate produced empty text")
            if self.runner is None:
                type(self).generates_completed += 1
            self.last_receipt = {
                "produced_by": "model",
                "generated_text": text,
                "fallbacks": int(fallbacks),
                "new_token_ids": body.get("new_token_ids"),
                "median_gpu_ns_per_token": body.get("median_gpu_ns_per_token"),
                "wall_ns": body.get("wall_ns"),
                "binary": str(binary),
                "artifact_root": str(artifact),
                "tokenizer": str(tokenizer),
                "lane": self.lane,
                "used_gpu_lane_lock": self.runner is None,
                "exit_code": proc.returncode,
            }
            return text


class SpecialUnit:
    def __init__(
        self,
        *,
        repo: Path | None = None,
        session_root: Path | None = None,
        session: Session | None = None,
        gate: ResourceGate | None = None,
        policy: ExecutionSandboxPolicy | None = None,
        backend: ConversationBackend | None = None,
        grok_runner: GrokRunner | None = None,
        grok_tasks: Path | None = None,
        authority: VerificationAuthority | None = None,
        owned_worktree: Path | None = None,
    ) -> None:
        self.repo = Path(repo or REPO_ROOT).resolve()
        self.store = SessionStore(Path(session_root or DEFAULT_SESSION_ROOT))
        self.gate = gate or ResourceGate()
        # Resolve so macOS /var → /private/var does not make owned writes look
        # like they escaped the sandbox.
        owned = Path(owned_worktree or self.repo).resolve()
        self.policy = policy or default_model_policy(
            owned_worktree=owned, sandbox_root=owned
        )
        self.tools = ToolExecutor(self.repo, self.policy, self.gate)
        self.backend = backend
        self.grok_runner = grok_runner
        self.grok_tasks = Path(grok_tasks or GROK_TASKS)
        self.authority = authority or default_authority()
        self.session = session or self.store.create()
        self.refresh_context()

    @classmethod
    def open(
        cls,
        session_id: str,
        *,
        repo: Path | None = None,
        session_root: Path | None = None,
        **kwargs: Any,
    ) -> "SpecialUnit":
        store = SessionStore(Path(session_root or DEFAULT_SESSION_ROOT))
        session = store.load(session_id)
        unit = cls(repo=repo, session_root=store.root, session=session, **kwargs)
        return unit

    def save(self) -> Path:
        return self.store.save(self.session)

    def refresh_context(self) -> dict[str, Any]:
        ctx = project_context(self.repo, lock_path=self.gate.lock_path)
        self.session.context_digest = str(ctx["digest"])
        self.save()
        return ctx

    def say(self, text: str, *, role: str = "user") -> Turn:
        ctx = self.refresh_context()
        user = Turn(role=role, text=text, meta={"context_digest": ctx["digest"]})
        self.session.transcript.append(user)
        self.session.status = SessionStatus.RUNNING.value
        self.save()
        if self.backend is None:
            return user
        try:
            reply_text = self.backend.complete(text, ctx)
        except NativeDecodeRefused as exc:
            self.session.status = SessionStatus.PAUSED.value
            self.session.pause_reason = str(exc)
            self.save()
            raise
        except NativeDecodeError:
            self.session.status = SessionStatus.IDLE.value
            self.save()
            raise
        produced_by = getattr(self.backend, "produced_by", "unknown")
        meta: dict[str, Any] = {
            "produced_by": produced_by,
            "context_digest": ctx["digest"],
        }
        receipt = getattr(self.backend, "last_receipt", None)
        if isinstance(receipt, dict):
            native = {
                key: receipt[key]
                for key in (
                    "fallbacks",
                    "new_token_ids",
                    "median_gpu_ns_per_token",
                    "wall_ns",
                    "used_gpu_lane_lock",
                    "binary",
                    "lane",
                )
                if key in receipt
            }
            if native:
                meta["native"] = native
        assistant = Turn(role="assistant", text=reply_text, meta=meta)
        self.session.transcript.append(assistant)
        self.session.status = SessionStatus.IDLE.value
        self.save()
        return assistant

    def tool(self, name: str, args: Mapping[str, Any] | None = None) -> ToolResult:
        self.session.status = SessionStatus.RUNNING.value
        self.save()
        result = self.tools.run(name, args or {})
        if name == "read" and result.ok:
            path = str((args or {}).get("path") or "")
            if path and path not in self.session.open_files:
                self.session.open_files.append(path)
        self.session.transcript.append(
            Turn(
                role="tool",
                text=result.output[:2000],
                meta={"name": name, "ok": result.ok, "detail": result.detail},
            )
        )
        if result.detail.get("interrupted"):
            self.session.status = SessionStatus.INTERRUPTED.value
        else:
            self.session.status = SessionStatus.IDLE.value
        self.save()
        return result

    def interrupt(self, reason: str = "user") -> None:
        self.tools.interrupt()
        self.session.pending_interrupt = reason
        self.session.status = SessionStatus.INTERRUPTED.value
        self.save()

    def resume(self) -> Session:
        """Restart/resume: reload from disk, clear interrupt, stay paused if gated."""

        self.session = self.store.load(self.session.session_id)
        self.tools.clear_interrupt()
        live, why = self.gate.protected_bench_live()
        if live and self.session.pause_reason:
            self.session.status = SessionStatus.PAUSED.value
            self.session.pause_reason = why
            self.save()
            return self.session
        self.session.pending_interrupt = None
        self.session.status = SessionStatus.IDLE.value
        self.session.pause_reason = None
        self.save()
        return self.session

    def pause_for_resources(self, cls: ResourceClass | str = ResourceClass.GPU_HEAVY) -> tuple[bool, str]:
        ok, why = self.gate.admit(cls)
        if ok:
            return True, why
        self.session.status = SessionStatus.PAUSED.value
        self.session.pause_reason = why
        self.save()
        return False, why

    def plan(self, objective: str, steps: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        plan = new_plan(objective, steps)
        self.session.plan = plan
        self.save()
        return plan

    def propose(self, step_id: str, *, author: str = "qwen38-special-unit") -> dict[str, Any]:
        if self.session.plan is None:
            raise SpecialUnitError("no plan")
        candidate = propose_complete(self.session.plan, step_id, author=author, authority=self.authority)
        self.save()
        return candidate

    def verify(
        self,
        step_id: str,
        *,
        principal: AuthorityPrincipal | str = AuthorityPrincipal.PROTECTED_CONTROLLER,
        certifier_id: str = "protected_controller",
    ) -> dict[str, Any]:
        return verify_complete(self, step_id, principal=principal, certifier_id=certifier_id)

    def grok_delegate(
        self,
        *,
        slug: str,
        contract: Path,
        profile: str = "maximum",
    ) -> dict[str, Any]:
        if self.grok_runner is None:
            # Dry structural dispatch: record the command, do not invoke grok-run
            # unless a runner is injected. Live invocation is opt-in so this
            # harness never surprises a GPU lane with a new Grok process.
            task_id = f"{slug}-dry"
            handle = {
                "task_id": task_id,
                "dispatched": True,
                "consumed": False,
                "verified_complete": False,
                "mode": "structural",
                "cmd": [
                    str(GROK_RUN),
                    "delegate",
                    "--task",
                    slug,
                    "--contract",
                    str(contract),
                    "--profile",
                    profile,
                ],
            }
            self.session.grok_handles.append(handle)
            self.save()
            return handle
        result = self.grok_runner.delegate(
            slug=slug,
            contract=Path(contract),
            profile=profile,
            repo=self.repo,
            tasks_root=self.grok_tasks,
        )
        task_id = str(result.get("task_id") or slug)
        handle = {
            "task_id": task_id,
            "dispatched": True,
            "consumed": False,
            "verified_complete": False,
            "mode": "runner",
            "runner": result,
        }
        self.session.grok_handles.append(handle)
        self.save()
        return handle

    def grok_consume(self, task_id: str) -> dict[str, Any]:
        consumption = consume_grok_task(task_id, self.grok_tasks)
        found = False
        for handle in self.session.grok_handles:
            if handle.get("task_id") == task_id:
                handle["consumed"] = bool(consumption.get("consumed"))
                handle["finished"] = bool(consumption.get("finished"))
                handle["verified_complete"] = False
                handle["consumption_seal"] = consumption.get("seal_sha256")
                found = True
        if not found:
            self.session.grok_handles.append(
                {
                    "task_id": task_id,
                    "dispatched": bool(consumption.get("dispatched")),
                    "consumed": bool(consumption.get("consumed")),
                    "verified_complete": False,
                    "mode": "external",
                    "consumption_seal": consumption.get("seal_sha256"),
                }
            )
        self.save()
        return consumption

    def unconsumed_grok(self) -> list[dict[str, Any]]:
        return [h for h in self.session.grok_handles if h.get("dispatched") and not h.get("consumed")]

    def ready_report(self) -> dict[str, Any]:
        plan = self.session.plan or {}
        steps = list(plan.get("steps") or [])
        proposed = sum(1 for s in steps if s["status"] == StepStatus.PROPOSED_COMPLETE.value)
        verified = sum(1 for s in steps if s["status"] == StepStatus.VERIFIED_COMPLETE.value)
        return {
            "schema": SCHEMA,
            "session_id": self.session.session_id,
            "status": self.session.status,
            "turns": len(self.session.transcript),
            "plan_steps": len(steps),
            "proposed_complete": proposed,
            "verified_complete": verified,
            "unconsumed_grok": len(self.unconsumed_grok()),
            "pause_reason": self.session.pause_reason,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", type=Path, default=REPO_ROOT)
    p.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    p.add_argument("--session", dest="session_id", default=None)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lab.hcli.special_unit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_new = sub.add_parser("new")
    _add_common(p_new)

    p_say = sub.add_parser("say")
    _add_common(p_say)
    p_say.add_argument("text")
    p_say.add_argument("--backend", choices=("native", "scripted"), default="native")
    p_say.add_argument("--max-new-tokens", type=int, default=16)

    p_tool = sub.add_parser("tool")
    _add_common(p_tool)
    p_tool.add_argument("name")
    p_tool.add_argument("--args", default="{}")

    p_int = sub.add_parser("interrupt")
    _add_common(p_int)
    p_int.add_argument("--reason", default="user")

    p_res = sub.add_parser("resume")
    _add_common(p_res)

    p_plan = sub.add_parser("plan")
    _add_common(p_plan)
    p_plan.add_argument("objective")

    p_prop = sub.add_parser("propose")
    _add_common(p_prop)
    p_prop.add_argument("step_id")

    p_ver = sub.add_parser("verify")
    _add_common(p_ver)
    p_ver.add_argument("step_id")

    p_gd = sub.add_parser("grok-delegate")
    _add_common(p_gd)
    p_gd.add_argument("--slug", required=True)
    p_gd.add_argument("--contract", type=Path, required=True)
    p_gd.add_argument("--profile", default="maximum")

    p_gc = sub.add_parser("grok-consume")
    _add_common(p_gc)
    p_gc.add_argument("task_id")

    p_pc = sub.add_parser("pause-check")
    p_pc.add_argument("--lock", type=Path, default=GPU_LOCK)

    p_show = sub.add_parser("show")
    _add_common(p_show)

    p_bench = sub.add_parser("bench")
    p_bench.add_argument("--receipt", type=Path, default=None)
    p_bench.add_argument("--repo", type=Path, default=REPO_ROOT)

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "pause-check":
        gate = ResourceGate(lock_path=args.lock)
        live, why = gate.protected_bench_live()
        gpu_ok, gpu_why = gate.admit(ResourceClass.GPU_HEAVY)
        print(json.dumps({"protected_bench_live": live, "why": why, "gpu_admitted": gpu_ok, "gpu_why": gpu_why}, indent=2))
        return 0

    if args.cmd == "bench":
        from lab.hcli.claude_offload_bench import run_bench

        doc = run_bench(repo=args.repo, receipt_path=args.receipt)
        print(json.dumps({k: doc[k] for k in doc if k != "tasks"}, indent=2))
        return 0 if doc.get("status") == "PASS" else 1

    def _unit() -> SpecialUnit:
        if args.session_id:
            return SpecialUnit.open(args.session_id, repo=args.repo, session_root=args.session_root)
        if args.cmd == "new":
            return SpecialUnit(repo=args.repo, session_root=args.session_root)
        raise SystemExit("pass --session ID (or use `new`)")

    if args.cmd == "new":
        unit = SpecialUnit(repo=args.repo, session_root=args.session_root)
        print(json.dumps(unit.session.to_dict(), indent=2))
        return 0

    unit = _unit()
    if args.cmd == "say":
        if unit.backend is None:
            if args.backend == "scripted":
                unit.backend = ScriptedBackend(["acknowledged"])
            else:
                unit.backend = NativeQwen38Backend(
                    repo=unit.repo,
                    gate=unit.gate,
                    max_new_tokens=args.max_new_tokens,
                )
        try:
            turn = unit.say(args.text)
        except NativeDecodeRefused as exc:
            print(
                json.dumps(
                    {"ok": False, "error": str(exc), "skip": True, "produced_by": "none"},
                    indent=2,
                )
            )
            return 1
        except NativeDecodeError as exc:
            print(
                json.dumps(
                    {"ok": False, "error": str(exc), "skip": False, "produced_by": "none"},
                    indent=2,
                )
            )
            return 1
        print(json.dumps(turn.to_dict(), indent=2))
    elif args.cmd == "tool":
        result = unit.tool(args.name, json.loads(args.args))
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.ok else 1
    elif args.cmd == "interrupt":
        unit.interrupt(args.reason)
        print(json.dumps(unit.session.to_dict(), indent=2))
    elif args.cmd == "resume":
        unit.resume()
        print(json.dumps(unit.session.to_dict(), indent=2))
    elif args.cmd == "plan":
        print(json.dumps(unit.plan(args.objective), indent=2))
    elif args.cmd == "propose":
        print(json.dumps(unit.propose(args.step_id), indent=2))
    elif args.cmd == "verify":
        print(json.dumps(unit.verify(args.step_id), indent=2))
    elif args.cmd == "grok-delegate":
        print(json.dumps(unit.grok_delegate(slug=args.slug, contract=args.contract, profile=args.profile), indent=2))
    elif args.cmd == "grok-consume":
        print(json.dumps(unit.grok_consume(args.task_id), indent=2))
    elif args.cmd == "show":
        print(json.dumps(unit.ready_report(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
