"""Durable high-level goals waiting behind the active goal.

Steering changes the current mission. A banked goal is different: it is a
future objective, kept on disk until the current objective reaches a terminal
successful state. The bank is workspace-scoped rather than session-scoped so
closing ``hcli`` or starting a resident does not make the queue disappear.

The file is deliberately small and boring JSON. The lock serializes multiple
HCLI front ends, while atomic_write_json makes a crash leave the previous
queue generation intact. A claimed item records its owner PID; a later
controller returns it to ``queued`` only when that owner is gone.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .persist import atomic_write_json
from .resources import process_start_token

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None  # type: ignore[assignment]


GOAL_BANK_SCHEMA = "hcli.goal_bank.v1"
GOAL_BANK_FILENAME = "goal-bank.json"
GOAL_BANK_LOCK_FILENAME = "goal-bank.lock"
MAX_BANKED_GOALS = 128
MAX_TERMINAL_RECORDS = 32
MAX_GOAL_CHARS = 12000
MAX_DISPLAY_GOAL_CHARS = 640
VALID_MODES = frozenset({"auto", "mission"})
VALID_STATUSES = frozenset({"queued", "running", "completed", "failed", "cancelled"})


class GoalBankError(RuntimeError):
    """The durable goal bank exists but cannot be trusted or updated."""


def _clip(value: Any, limit: int = 600) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _pid_alive(value: Any) -> bool:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return False
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (OSError, OverflowError, ValueError):
        return False
    return True


def _owner_alive(item: Dict[str, Any]) -> bool:
    """Match a claimed owner by PID and, when available, start identity."""
    try:
        pid = int(item.get("owner_pid"))
    except (TypeError, ValueError):
        return False
    if not _pid_alive(pid):
        return False
    expected = item.get("owner_start_token")
    if not expected:
        # Older queue records only had a PID; keep them recoverable.
        return True
    observed = process_start_token(pid)
    # A missing probe is not proof of death. The supervisor may be running in
    # a restricted environment where ps/libproc is unavailable.
    return observed is None or str(observed) == str(expected)


class GoalBank:
    """A FIFO, workspace-scoped queue of future goals."""

    def __init__(self, workspace: str | os.PathLike[str]):
        self.workspace = Path(workspace).expanduser().resolve()
        self.directory = self.workspace / ".hcli"
        self.path = self.directory / GOAL_BANK_FILENAME
        self.lock_path = self.directory / GOAL_BANK_LOCK_FILENAME
        self.directory.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        """Serialize read-modify-write operations across local HCLI clients."""
        handle = self.lock_path.open("a+")
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            handle.close()

    @staticmethod
    def _empty() -> Dict[str, Any]:
        return {
            "schema": GOAL_BANK_SCHEMA,
            "generation": 0,
            "updated_at": None,
            "next_seq": 1,
            "goals": [],
        }

    def _read_unlocked(self) -> Dict[str, Any]:
        if not self.path.is_file():
            return self._empty()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise GoalBankError(f"goal bank is unreadable: {self.path}") from exc
        if not isinstance(raw, dict):
            raise GoalBankError(f"goal bank is not an object: {self.path}")
        schema = raw.get("schema")
        if schema not in (None, GOAL_BANK_SCHEMA):
            raise GoalBankError(f"unsupported goal bank schema: {schema!r}")
        goals = raw.get("goals", [])
        if not isinstance(goals, list):
            raise GoalBankError("goal bank goals must be a list")
        checked: List[Dict[str, Any]] = []
        for item in goals:
            if not isinstance(item, dict) or not item.get("id") or not item.get("goal"):
                raise GoalBankError("goal bank contains a malformed goal record")
            status = str(item.get("status") or "queued")
            if status not in VALID_STATUSES:
                raise GoalBankError(f"goal bank contains invalid status: {status!r}")
            mode = str(item.get("mode") or "auto")
            if mode not in VALID_MODES:
                raise GoalBankError(f"goal bank contains invalid mode: {mode!r}")
            checked.append(dict(item, status=status, mode=mode))
        try:
            next_seq = max(1, int(raw.get("next_seq") or 1))
        except (TypeError, ValueError) as exc:
            raise GoalBankError("goal bank next_seq is not an integer") from exc
        raw["schema"] = GOAL_BANK_SCHEMA
        raw["goals"] = checked
        raw["next_seq"] = next_seq
        return raw

    @staticmethod
    def _sort_key(item: Dict[str, Any]) -> tuple:
        try:
            seq = int(item.get("seq") or 0)
        except (TypeError, ValueError):
            seq = 0
        return seq, str(item.get("created_at") or "")

    @staticmethod
    def _terminal_key(item: Dict[str, Any]) -> tuple:
        return float(item.get("finished_at") or item.get("created_at") or 0.0), GoalBank._sort_key(item)

    def _prune_unlocked(self, data: Dict[str, Any]) -> None:
        goals = list(data.get("goals") or [])
        if len(goals) <= MAX_BANKED_GOALS:
            return
        terminal = sorted(
            (item for item in goals if item.get("status") in {"completed", "failed", "cancelled"}),
            key=self._terminal_key,
        )
        while len(goals) > MAX_BANKED_GOALS and terminal:
            old = terminal.pop(0)
            try:
                goals.remove(old)
            except ValueError:
                pass
        data["goals"] = goals

    def _write_unlocked(self, data: Dict[str, Any]) -> None:
        self._prune_unlocked(data)
        try:
            generation = int(data.get("generation") or 0) + 1
        except (TypeError, ValueError):
            generation = 1
        data["schema"] = GOAL_BANK_SCHEMA
        data["generation"] = generation
        data["updated_at"] = time.time()
        atomic_write_json(self.path, data)

    @staticmethod
    def _public(item: Dict[str, Any], *, display_limit: int = MAX_DISPLAY_GOAL_CHARS) -> Dict[str, Any]:
        result = {
            key: item.get(key)
            for key in (
                "id",
                "seq",
                "goal",
                "mode",
                "status",
                "created_at",
                "started_at",
                "finished_at",
                "attempts",
                "owner_pid",
                "last_error",
            )
            if key in item
        }
        if "goal" in result:
            result["goal"] = _clip(result["goal"], display_limit)
        if isinstance(item.get("result"), dict):
            result["result"] = dict(item["result"])
        return result

    def recover_inflight(self) -> int:
        """Return goals from dead HCLI owners to the FIFO queue."""
        recovered = 0
        with self._lock():
            data = self._read_unlocked()
            for item in data["goals"]:
                if item.get("status") != "running" or _owner_alive(item):
                    continue
                item["status"] = "queued"
                item["recovered_at"] = time.time()
                item.pop("owner_pid", None)
                item.pop("owner_start_token", None)
                item.pop("started_at", None)
                item["last_error"] = "previous HCLI owner ended; goal returned to the bank"
                recovered += 1
            if recovered:
                self._write_unlocked(data)
        return recovered

    def add(self, goal: str, *, mode: str = "auto") -> Dict[str, Any]:
        text = str(goal or "").strip()
        if not text:
            raise ValueError("banked goal is required")
        if len(text) > MAX_GOAL_CHARS:
            raise ValueError(f"banked goal exceeds {MAX_GOAL_CHARS} characters")
        mode = str(mode or "auto").strip().lower()
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
        with self._lock():
            data = self._read_unlocked()
            try:
                seq = max(1, int(data.get("next_seq") or 1))
            except (TypeError, ValueError):
                seq = 1
            item = {
                "id": f"bank-{uuid.uuid4().hex[:12]}",
                "seq": seq,
                "goal": text,
                "mode": mode,
                "status": "queued",
                "created_at": time.time(),
                "attempts": 0,
            }
            data["next_seq"] = seq + 1
            data["goals"].append(item)
            self._write_unlocked(data)
            return dict(item)

    def claim_next(self) -> Optional[Dict[str, Any]]:
        """Claim one goal, recording the current PID before execution starts."""
        with self._lock():
            data = self._read_unlocked()
            queued = [item for item in data["goals"] if item.get("status") == "queued"]
            if not queued:
                return None
            item = min(queued, key=self._sort_key)
            item["status"] = "running"
            item["owner_pid"] = os.getpid()
            item["owner_start_token"] = process_start_token(os.getpid())
            item["started_at"] = time.time()
            item["attempts"] = int(item.get("attempts") or 0) + 1
            item.pop("last_error", None)
            self._write_unlocked(data)
            return dict(item)

    @staticmethod
    def _result_summary(result: Any) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return {"status": _clip(result, 120)}
        summary: Dict[str, Any] = {}
        for key in ("status", "state", "verdict", "reason", "mission_id", "goal_id"):
            value = result.get(key)
            if value is not None and value != "":
                summary[key] = _clip(value, 320)
        receipt = result.get("receipt")
        if isinstance(receipt, dict):
            for key in ("path", "receipt_path", "goal_id"):
                if receipt.get(key):
                    summary[f"receipt_{key}"] = _clip(receipt[key], 320)
        elif receipt:
            summary["receipt"] = _clip(receipt, 320)
        return summary

    def finish(self, goal_id: str, result: Any = None, *, status: Optional[str] = None, error: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Mark a claimed goal terminal and retain a compact outcome."""
        wanted = str(status or (result.get("status") if isinstance(result, dict) else "") or "failed").lower()
        if wanted not in VALID_STATUSES - {"queued", "running"}:
            wanted = "failed"
        with self._lock():
            data = self._read_unlocked()
            found = next((item for item in data["goals"] if item.get("id") == goal_id), None)
            if found is None:
                return None
            found["status"] = wanted
            found["finished_at"] = time.time()
            found.pop("owner_pid", None)
            found.pop("owner_start_token", None)
            found.pop("started_at", None)
            summary = self._result_summary(result)
            if error:
                summary["error"] = _clip(error, 500)
                found["last_error"] = _clip(error, 500)
            if summary:
                found["result"] = summary
            self._write_unlocked(data)
            return dict(found)

    def drop(self, selector: str) -> Optional[Dict[str, Any]]:
        """Drop one queued goal by exact id or one-based queue position."""
        token = str(selector or "").strip()
        if not token:
            return None
        with self._lock():
            data = self._read_unlocked()
            queued = sorted(
                (item for item in data["goals"] if item.get("status") == "queued"),
                key=self._sort_key,
            )
            target = None
            if token.isdigit():
                index = int(token) - 1
                if 0 <= index < len(queued):
                    target = queued[index]
            else:
                target = next((item for item in queued if item.get("id") == token), None)
            if target is None:
                return None
            data["goals"].remove(target)
            self._write_unlocked(data)
            return dict(target)

    def clear(self) -> int:
        """Remove waiting goals while preserving running and terminal history."""
        with self._lock():
            data = self._read_unlocked()
            before = len(data["goals"])
            data["goals"] = [item for item in data["goals"] if item.get("status") != "queued"]
            removed = before - len(data["goals"])
            if removed:
                self._write_unlocked(data)
            return removed

    def snapshot(
        self,
        *,
        queued_limit: int = 16,
        recent_limit: int = 6,
        display_limit: int = MAX_DISPLAY_GOAL_CHARS,
    ) -> Dict[str, Any]:
        """Return bounded display/context state, never the whole archive."""
        with self._lock():
            data = self._read_unlocked()
        queued = sorted(
            (item for item in data["goals"] if item.get("status") == "queued"),
            key=self._sort_key,
        )
        running = sorted(
            (item for item in data["goals"] if item.get("status") == "running"),
            key=self._sort_key,
        )
        recent = sorted(
            (item for item in data["goals"] if item.get("status") in {"completed", "failed", "cancelled"}),
            key=self._terminal_key,
            reverse=True,
        )
        return {
            "schema": GOAL_BANK_SCHEMA,
            "available": True,
            "path": str(self.path),
            "generation": int(data.get("generation") or 0),
            "queued_count": len(queued),
            "running_count": len(running),
            "queued": [self._public(item, display_limit=display_limit) for item in queued[: max(0, queued_limit)]],
            "running": [self._public(item, display_limit=display_limit) for item in running[: max(0, recent_limit)]],
            "recent": [self._public(item, display_limit=display_limit) for item in recent[: max(0, recent_limit)]],
            "next": self._public(queued[0], display_limit=display_limit) if queued else None,
        }


__all__ = [
    "GOAL_BANK_SCHEMA",
    "GoalBank",
    "GoalBankError",
    "MAX_GOAL_CHARS",
]
