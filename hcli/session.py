from __future__ import annotations

import gzip
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .persist import atomic_write_bytes, atomic_write_json


SESSION_MESSAGE_LIMIT = 64
SESSION_MESSAGE_CHARS = 8000
CONTEXT_MEMORY_SCHEMA = "hcli.context.memory.v1"
CONTEXT_MEMORY_CHARS = 16000


class Session:
    def __init__(self, session_id: Optional[str] = None, goal: str = "", runtime_count: int = 1, model: Optional[str] = None):
        self.id = session_id or str(uuid.uuid4())
        self.goal = goal
        self.runtime_count = runtime_count
        self.model = model
        self.messages: List[Dict[str, Any]] = []
        self.steering: List[str] = []
        self.mission_id: Optional[str] = None
        self.created_at = datetime.now(timezone.utc).isoformat()
        # ``messages`` is the hot tail.  ``memory`` is a bounded semantic
        # checkpoint made by Controller.compact_context; it is deliberately
        # structured so compaction remembers facts and state rather than a
        # lossy middle slice of prose.
        self.memory: Dict[str, Any] = {}
        self.compaction_count: int = 0
        self.compacted_at: Optional[str] = None
        self.next_message_seq: int = 1
        self.history_archived_through: int = 0

    def append_message(
        self,
        role: str,
        content: Any,
        *,
        kind: str = "conversation",
    ) -> None:
        """Keep a bounded hot tail for the next semantic checkpoint."""
        text = str(content or "")
        if len(text) > SESSION_MESSAGE_CHARS:
            text = text[: SESSION_MESSAGE_CHARS - 1].rstrip() + "…"
        self.messages.append(
            {
                "seq": self.next_message_seq,
                "role": str(role or "user"),
                "content": text,
                "kind": str(kind or "conversation"),
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.next_message_seq += 1
        if len(self.messages) > SESSION_MESSAGE_LIMIT:
            self.messages = self.messages[-SESSION_MESSAGE_LIMIT :]

    def set_memory(self, memory: Optional[Dict[str, Any]]) -> None:
        """Install a JSON-safe semantic checkpoint, with a hard size guard."""
        value = dict(memory or {})
        value.setdefault("schema", CONTEXT_MEMORY_SCHEMA)
        # The checkpoint is prompt material on the next turn.  Refuse to let
        # an unexpectedly verbose steering/goal entry turn the memory lane
        # into the same unbounded transcript it exists to replace.
        try:
            encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            value = {
                "schema": CONTEXT_MEMORY_SCHEMA,
                "memory_error": "checkpoint was not JSON-serializable",
            }
        else:
            if len(encoded) > CONTEXT_MEMORY_CHARS:
                def shrink(item: Any, depth: int = 0) -> Any:
                    if isinstance(item, str):
                        return item[:400] + ("…" if len(item) > 400 else "")
                    if depth >= 3:
                        return "[omitted]"
                    if isinstance(item, list):
                        return [shrink(child, depth + 1) for child in item[:6]]
                    if isinstance(item, dict):
                        keys = list(item)[:16]
                        return {str(key): shrink(item[key], depth + 1) for key in keys}
                    return item

                value = {
                    str(key): shrink(item)
                    for key, item in value.items()
                }
                value["truncated"] = True
                encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
                if len(encoded) > CONTEXT_MEMORY_CHARS:
                    # Preserve the invariants that matter for safe resumption;
                    # an oversized arbitrary payload is never allowed to be
                    # sliced into invalid JSON on the next model call.
                    core = {
                        key: value.get(key)
                        for key in (
                            "active_goal",
                            "mission",
                            "ledger",
                            "steering",
                            "staging",
                            "prior_knowledge",
                            "goal_bank",
                            "receipts",
                        )
                        if value.get(key) not in (None, {}, [], "")
                    }
                    value = {
                        "schema": CONTEXT_MEMORY_SCHEMA,
                        "generation": value.get("generation"),
                        "compacted_at": value.get("compacted_at"),
                        "truncated": True,
                        **{
                            str(key): shrink(item)
                            for key, item in core.items()
                        },
                    }
        self.memory = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "runtime_count": self.runtime_count,
            "model": self.model,
            "messages": self.messages,
            "steering": self.steering,
            "mission_id": self.mission_id,
            "created_at": self.created_at,
            "memory": self.memory,
            "compaction_count": self.compaction_count,
            "compacted_at": self.compacted_at,
            "next_message_seq": self.next_message_seq,
            "history_archived_through": self.history_archived_through,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        s = cls(session_id=data.get("id"), goal=data.get("goal", ""),
                runtime_count=data.get("runtime_count", 1), model=data.get("model"))
        s.messages = data.get("messages", [])
        s.steering = data.get("steering", [])
        s.mission_id = data.get("mission_id")
        s.created_at = data.get("created_at", s.created_at)
        memory = data.get("memory")
        s.memory = dict(memory) if isinstance(memory, dict) else {}
        try:
            s.compaction_count = max(0, int(data.get("compaction_count") or 0))
        except (TypeError, ValueError):
            s.compaction_count = 0
        compacted_at = data.get("compacted_at")
        s.compacted_at = str(compacted_at) if compacted_at else None
        try:
            highest = max(
                int(item.get("seq") or 0)
                for item in s.messages
                if isinstance(item, dict)
            )
        except (TypeError, ValueError):
            highest = 0
        try:
            s.next_message_seq = max(
                highest + 1,
                int(data.get("next_message_seq") or 1),
            )
        except (TypeError, ValueError):
            s.next_message_seq = highest + 1
        try:
            s.history_archived_through = max(
                0, int(data.get("history_archived_through") or 0)
            )
        except (TypeError, ValueError):
            s.history_archived_through = 0
        return s


class SessionStore:
    def __init__(self, workspace: str):
        self.dir = os.path.join(workspace, ".hcli", "sessions")
        os.makedirs(self.dir, exist_ok=True)

    def save(self, session: Session):
        path = os.path.join(self.dir, f"{session.id}.json")
        atomic_write_json(path, session.to_dict())

    def archive_messages(self, session: Session) -> Dict[str, Any]:
        """Append uncached hot messages to one crash-safe gzip history file.

        The archive is cold storage, not prompt material. Sequence numbers make
        repeated compactions idempotent even though the hot tail is retained
        after each checkpoint. If the archive write fails, the caller can leave
        the hot transcript intact and report the storage error.
        """
        pending = [
            item
            for item in session.messages
            if isinstance(item, dict)
            and int(item.get("seq") or 0) > session.history_archived_through
        ]
        if not pending:
            return {
                "path": str(self.history_path(session.id)),
                "records": 0,
                "compressed_bytes": self.history_path(session.id).stat().st_size
                if self.history_path(session.id).is_file()
                else 0,
            }

        path = self.history_path(session.id)
        raw = b""
        if path.is_file():
            try:
                raw = gzip.decompress(path.read_bytes())
            except (OSError, EOFError, gzip.BadGzipFile) as exc:
                raise RuntimeError(f"history archive is unreadable: {path}") from exc
        additions = b"".join(
            (json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            for item in pending
        )
        atomic_write_bytes(path, gzip.compress(raw + additions, compresslevel=6, mtime=0))
        session.history_archived_through = max(
            session.history_archived_through,
            max(int(item.get("seq") or 0) for item in pending),
        )
        return {
            "path": str(path),
            "records": len(pending),
            "compressed_bytes": path.stat().st_size,
        }

    def history_path(self, session_id: str) -> Path:
        return Path(self.dir) / f"{session_id}.history.jsonl.gz"

    def load_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Read the optional cold archive for operator/debugging use."""
        path = self.history_path(session_id)
        if not os.path.isfile(path):
            return []
        with open(path, "rb") as handle:
            raw = gzip.decompress(handle.read()).decode("utf-8")
        rows = []
        for line in raw.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def load(self, session_id: str) -> Optional[Session]:
        path = os.path.join(self.dir, f"{session_id}.json")
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return Session.from_dict(json.load(f))
