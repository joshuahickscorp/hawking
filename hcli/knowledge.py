"""Durable, bounded prior knowledge for long-lived HCLI workspaces.

The transcript is a poor long-term memory: it is noisy, expensive to replay,
and can contain claims that were never verified.  This store keeps a small
semantic index on the workspace and appends the same bounded records to a
gzip archive on the SSD.  The index is what enters a prompt; the archive is
for recovery and audit, not an excuse to stuff the whole past into context.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

from .persist import atomic_write_json

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None  # type: ignore[assignment]


KNOWLEDGE_SCHEMA = "hcli.workspace_knowledge.v1"
KNOWLEDGE_FILENAME = "knowledge.json"
KNOWLEDGE_ARCHIVE_FILENAME = "knowledge.jsonl.gz"
KNOWLEDGE_LOCK_FILENAME = "knowledge.lock"
MAX_INDEX_RECORDS = 24
MAX_RECORD_CHARS = 5000
MAX_SNAPSHOT_CHARS = 8000
MAX_ARCHIVE_SCAN_BYTES = 16 * 1024 * 1024
_FOCUS_TERM_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*", re.IGNORECASE)
_FOCUS_STOPWORDS = frozenset(
    {
        "a", "an", "and", "answer", "as", "at", "be", "by", "do", "for",
        "from", "how", "i", "in", "is", "it", "me", "of", "on", "or", "please",
        "the", "this", "to", "what", "with", "you",
    }
)

# Control-plane tokens that, when pasted into a worker prompt as prior
# knowledge, steer some residents into a reasoning-only reply. Neutralise
# them with readable markers so the record stays diagnostic. Marker strings
# are themselves in the match list so a second pass is a no-op.
_SANITISED_MARKERS = (
    "[chat-template-kwargs]",
    "[hidden-thought-field]",
    "[thought-chain-field]",
    "[reasoning-field]",
    "[/reasoning-tag]",
    "[reasoning-tag]",
    "[thinking-flag]",
    "[/thinking-tag]",
    "[thinking-tag]",
    "[special-token]",
    "[/thought-tag]",
    "[thought-tag]",
    "[/think-tag]",
    "[think-tag]",
    "[reasoning]",
)
_MARKER_ALT = "|".join(
    re.escape(marker) for marker in sorted(_SANITISED_MARKERS, key=len, reverse=True)
)
_CONTROL_TOKEN_RE = re.compile(
    rf"(?P<marker>{_MARKER_ALT})"
    rf"|(?P<special><\|[^|]{{0,80}}\|>)"
    rf"|(?P<tag></?(?:[a-z0-9_-]*think[a-z0-9_-]*|[a-z0-9_-]*reasoning[a-z0-9_-]*|thought)"
    rf"(?:\s[^>]*)?>)"
    rf"|(?P<reasoning_content>reasoning[_]?content)"
    rf"|(?P<hidden_reasoning>hidden[_]?reasoning)"
    rf"|(?P<chain_of_thought>chain[_-]?of[_-]?thought)"
    rf"|(?P<enable_thinking>enable[_]?thinking)"
    rf"|(?P<chat_template_kwargs>chat[_]?template[_]?kwargs)"
    rf"|(?P<reasoning>\breasoning\b)",
    re.IGNORECASE,
)
_GROUP_MARKERS = {
    "reasoning_content": "[reasoning-field]",
    "hidden_reasoning": "[hidden-thought-field]",
    "chain_of_thought": "[thought-chain-field]",
    "enable_thinking": "[thinking-flag]",
    "chat_template_kwargs": "[chat-template-kwargs]",
    "reasoning": "[reasoning]",
    "special": "[special-token]",
}


def _marker_for_tag(raw: str) -> str:
    close = raw.startswith("</")
    body = raw[2:] if close else raw[1:]
    name = body.rstrip(">").split(None, 1)[0].lower() if body else ""
    if "reason" in name:
        family = "reasoning-tag"
    elif "thought" in name:
        family = "thought-tag"
    else:
        family = "think-tag"
    return f"[{'/' if close else ''}{family}]"


def _replace_control_token(match: re.Match[str]) -> str:
    if match.group("marker"):
        return match.group("marker")
    if match.group("tag"):
        return _marker_for_tag(match.group("tag"))
    for name, marker in _GROUP_MARKERS.items():
        if match.group(name):
            return marker
    return match.group(0)


def _sanitise_text(text: str) -> str:
    return _CONTROL_TOKEN_RE.sub(_replace_control_token, text)


def _sanitise_value(value: Any, *, depth: int) -> Any:
    if depth > 20:
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _sanitise_text(value)
    if isinstance(value, bytes):
        decoded = value.decode("utf-8", errors="replace")
        sanitised = _sanitise_text(decoded)
        if sanitised == decoded:
            return value
        return sanitised.encode("utf-8", errors="replace")
    if isinstance(value, (bytearray, memoryview)):
        return _sanitise_value(bytes(value), depth=depth)
    if isinstance(value, Mapping):
        out: Dict[Any, Any] = {}
        for key, child in value.items():
            new_key = _sanitise_text(key) if isinstance(key, str) else key
            out[new_key] = _sanitise_value(child, depth=depth + 1)
        return out
    if isinstance(value, list):
        return [_sanitise_value(child, depth=depth + 1) for child in value]
    if isinstance(value, tuple):
        return tuple(_sanitise_value(child, depth=depth + 1) for child in value)
    return value


def sanitise_knowledge(value: Any) -> Any:
    """Replace model-steering control tokens with readable markers.

    Applied on the way into the store and on the way out of snapshot/recall
    so already-poisoned on-disk records cannot steer the next prompt.
    Idempotent, total over odd types, and never raises.
    """
    try:
        return _sanitise_value(value, depth=0)
    except Exception:
        return value


class KnowledgeError(RuntimeError):
    """The workspace knowledge index exists but cannot be trusted."""


def _clip(value: Any, limit: int = 600) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _bound(value: Any, *, depth: int = 0) -> Any:
    """Make arbitrary semantic payloads small and JSON-safe."""
    if isinstance(value, str):
        return _clip(value, 900)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if depth >= 4:
        return _clip(value, 240)
    if isinstance(value, Mapping):
        return {
            str(key): _bound(child, depth=depth + 1)
            for key, child in list(value.items())[:24]
        }
    if isinstance(value, (list, tuple)):
        return [_bound(child, depth=depth + 1) for child in list(value)[:16]]
    return _clip(value, 600)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _focus_terms(value: Any) -> tuple[str, ...]:
    """Extract stable query terms without importing a search dependency."""
    terms = []
    for raw in _FOCUS_TERM_RE.findall(str(value or "").lower()):
        if raw in _FOCUS_STOPWORDS or len(raw) < 3:
            continue
        if raw not in terms:
            terms.append(raw)
    return tuple(terms[:32])


class KnowledgeStore:
    """Workspace-scoped semantic memory with a bounded hot index."""

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        *,
        archive_root: Optional[str | os.PathLike[str]] = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.directory = self.workspace / ".hcli"
        self.path = self.directory / KNOWLEDGE_FILENAME
        self.lock_path = self.directory / KNOWLEDGE_LOCK_FILENAME
        self.directory.mkdir(parents=True, exist_ok=True)
        configured = archive_root or os.environ.get("HCLI_CONTEXT_ARCHIVE_ROOT")
        if configured:
            namespace = hashlib.sha256(str(self.workspace).encode("utf-8")).hexdigest()[:20]
            try:
                candidate = Path(str(configured)).expanduser().resolve() / namespace
                candidate.mkdir(parents=True, exist_ok=True)
            except (OSError, TypeError, ValueError):
                candidate = self.directory
            self.archive_path = candidate / KNOWLEDGE_ARCHIVE_FILENAME
        else:
            self.archive_path = self.directory / KNOWLEDGE_ARCHIVE_FILENAME

    @contextmanager
    def _lock(self) -> Iterator[None]:
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
            "schema": KNOWLEDGE_SCHEMA,
            "generation": 0,
            "updated_at": None,
            "records": [],
        }

    def _read_unlocked(self) -> Dict[str, Any]:
        if not self.path.is_file():
            return self._empty()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise KnowledgeError(f"knowledge index is unreadable: {self.path}") from exc
        if not isinstance(raw, dict):
            raise KnowledgeError(f"knowledge index is not an object: {self.path}")
        schema = raw.get("schema")
        if schema not in (None, KNOWLEDGE_SCHEMA):
            raise KnowledgeError(f"unsupported knowledge schema: {schema!r}")
        records = raw.get("records", [])
        if not isinstance(records, list):
            raise KnowledgeError("knowledge records must be a list")
        checked = [item for item in records if isinstance(item, dict) and item.get("id")]
        raw["schema"] = KNOWLEDGE_SCHEMA
        raw["records"] = checked
        return raw

    @staticmethod
    def _rank(record: Mapping[str, Any]) -> tuple:
        try:
            priority = int(record.get("priority") or 0)
        except (TypeError, ValueError):
            priority = 0
        try:
            at = float(record.get("at") or 0.0)
        except (TypeError, ValueError):
            at = 0.0
        return priority, at

    @classmethod
    def _ordered_records(
        cls,
        records: List[Mapping[str, Any]],
        terms: tuple[str, ...],
    ) -> List[Mapping[str, Any]]:
        rows = list(records)
        if terms:
            def focused_rank(record: Mapping[str, Any]) -> tuple:
                haystack = _canonical(
                    {
                        "kind": record.get("kind"),
                        "source": record.get("source"),
                        "data": record.get("data"),
                    }
                ).lower()
                matches = sum(1 for term in terms if term in haystack)
                try:
                    priority = int(record.get("priority") or 0)
                except (TypeError, ValueError):
                    priority = 0
                verified = 1 if record.get("verified") else 0
                try:
                    at = float(record.get("at") or 0.0)
                except (TypeError, ValueError):
                    at = 0.0
                # Priority/verification remain meaningful even when a query
                # is novel; relevance wins once it is materially present.
                return matches, verified, priority, at

            rows.sort(key=focused_rank, reverse=True)
        else:
            rows.sort(key=cls._rank, reverse=True)
        return rows

    def _write_index_unlocked(self, data: Dict[str, Any]) -> None:
        records = list(data.get("records") or [])
        if len(records) > MAX_INDEX_RECORDS:
            # Keep explicit constraints/corrections and the newest evidence;
            # ordinary conversational notes are the first to age out.
            records = sorted(records, key=self._rank, reverse=True)[:MAX_INDEX_RECORDS]
            records.sort(key=lambda item: float(item.get("at") or 0.0), reverse=True)
        data["schema"] = KNOWLEDGE_SCHEMA
        data["records"] = records
        try:
            generation = int(data.get("generation") or 0) + 1
        except (TypeError, ValueError):
            generation = 1
        data["generation"] = generation
        data["updated_at"] = time.time()
        atomic_write_json(self.path, data)

    def _append_archive_unlocked(self, record: Mapping[str, Any]) -> None:
        line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        # A gzip file may contain concatenated members. Appending one member
        # keeps the steady-state cost O(record), instead of decompressing and
        # recompressing the complete overnight archive for every turn. The
        # bounded JSON index is the authority if a power loss leaves only a
        # partial cold member at the tail.
        with self.archive_path.open("ab") as handle:
            handle.write(gzip.compress(line, compresslevel=6, mtime=0))
            handle.flush()
            os.fsync(handle.fileno())

    def record(
        self,
        kind: str,
        payload: Any,
        *,
        source: str = "hcli",
        priority: int = 50,
        verified: bool = False,
    ) -> Dict[str, Any]:
        """Record one bounded semantic fact, deduplicating identical facts."""
        kind_value = sanitise_knowledge(_clip(kind, 80))
        source_value = sanitise_knowledge(_clip(source, 120))
        data_value = sanitise_knowledge(_bound(payload))
        fingerprint = hashlib.sha256(
            _canonical({"kind": kind_value, "data": data_value}).encode("utf-8")
        ).hexdigest()[:20]
        now = time.time()
        record = {
            "id": f"knowledge-{uuid.uuid4().hex[:12]}",
            "at": now,
            "kind": kind_value,
            "source": source_value,
            "priority": max(0, min(100, int(priority))),
            "verified": bool(verified),
            "fingerprint": fingerprint,
            "data": data_value,
        }
        if len(_canonical(record)) > MAX_RECORD_CHARS:
            record["data"] = _bound(data_value, depth=1)
        with self._lock():
            index = self._read_unlocked()
            previous = next(
                (item for item in index["records"] if item.get("fingerprint") == fingerprint),
                None,
            )
            if previous is not None:
                record["id"] = previous.get("id") or record["id"]
            index["records"] = [
                item for item in index["records"] if item.get("fingerprint") != fingerprint
            ]
            index["records"].append(record)
            self._write_index_unlocked(index)
            # The index remains authoritative if a cold archive write is
            # unavailable. A future call can still retry archival without
            # losing the fact from the prompt path.
            try:
                self._append_archive_unlocked(record)
            except (OSError, KnowledgeError):
                pass
        return dict(record)

    def record_note(self, text: str, *, kind: str = "knowledge", source: str = "operator") -> Dict[str, Any]:
        priorities = {"constraint": 100, "correction": 90, "knowledge": 75}
        return self.record(
            f"operator_{kind}",
            {"text": _clip(text, 1200)},
            source=source,
            priority=priorities.get(kind, 70),
            verified=False,
        )

    def record_checkpoint(self, memory: Mapping[str, Any], *, source: str = "session_checkpoint") -> Dict[str, Any]:
        """Persist the high-value part of a semantic context checkpoint."""
        payload = {
            key: memory.get(key)
            for key in (
                "active_goal",
                "mission",
                "ledger",
                "steering",
                "staging",
                "goal_bank",
                "receipts",
            )
            if memory.get(key) not in (None, {}, [], "")
        }
        verified = bool(
            isinstance(payload.get("mission"), Mapping)
            and str(payload["mission"].get("phase") or "") == "completed"
        )
        return self.record(
            "semantic_checkpoint",
            payload,
            source=source,
            priority=85 if verified else 60,
            verified=verified,
        )

    def record_result(
        self,
        goal: str,
        result: Any,
        *,
        source: str = "hcli_result",
    ) -> Optional[Dict[str, Any]]:
        """Keep a compact result claim without treating model prose as proof."""
        if not isinstance(result, Mapping):
            return None
        status = _clip(result.get("status"), 80).lower()
        if not status:
            return None
        payload = {
            "goal": _clip(goal, 700),
            "status": status,
            "claim": _clip(result.get("claim") or result.get("content"), 900),
            "verdict": _clip(result.get("verdict"), 160),
            "reason": _clip(result.get("reason") or result.get("error"), 500),
            "next_action": _clip(result.get("next_action"), 500),
            "receipt": _clip(result.get("receipt"), 320),
        }
        payload = {key: value for key, value in payload.items() if value}
        return self.record(
            "result_claim",
            payload,
            source=source,
            priority=80 if status == "completed" else 65,
            verified=False,
        )

    def snapshot(
        self,
        *,
        limit: int = 8,
        max_chars: int = MAX_SNAPSHOT_CHARS,
        focus: str = "",
    ) -> Dict[str, Any]:
        """Return bounded prior knowledge, ranked for the current question.

        The index remains small and local.  With a focus, relevance is a
        deterministic tie-breaker ahead of recency while verified/high
        priority constraints retain their gravity.  The cold gzip archive is
        deliberately not read here; it is recovery material, not prompt bulk.
        """
        with self._lock():
            index = self._read_unlocked()
        terms = _focus_terms(focus)
        records = self._ordered_records(list(index.get("records") or []), terms)
        chosen: List[Dict[str, Any]] = []
        for item in records[: max(0, int(limit))]:
            candidate = {
                key: item.get(key)
                for key in ("id", "at", "kind", "source", "priority", "verified", "data")
                if key in item
            }
            if len(_canonical({"records": chosen + [candidate]})) > max_chars:
                # Keep the record identity and its first-order claim while
                # dropping deep staging/receipt detail from lower-priority rows.
                data = candidate.get("data")
                if isinstance(data, Mapping):
                    compact = {
                        key: _bound(data[key], depth=2)
                        for key in ("active_goal", "mission", "ledger", "steering", "staging", "goal_bank")
                        if key in data
                    }
                    candidate["data"] = compact
                if len(_canonical({"records": chosen + [candidate]})) > max_chars:
                    continue
            chosen.append(candidate)
        return sanitise_knowledge({
            "schema": KNOWLEDGE_SCHEMA,
            "available": True,
            "path": str(self.path),
            "generation": int(index.get("generation") or 0),
            "retrieval": {
                "mode": "focus_ranked" if terms else "priority_recent",
                "focus": _clip(focus, 320) if focus else None,
                "terms": list(terms),
            },
            "records": chosen,
            "archive": {
                "path": str(self.archive_path),
                "compression": "gzip",
                "cold": True,
            },
        })

    def recall(
        self,
        focus: str,
        *,
        limit: int = 8,
        max_chars: int = MAX_SNAPSHOT_CHARS,
    ) -> Dict[str, Any]:
        """Recall older bounded facts from the cold archive on demand.

        Normal turns use only the hot index.  This explicit path scans at most
        a fixed amount of decompressed archive data, merges non-duplicate
        records with the hot index, and returns the same prompt-safe shape.
        It is a retrieval valve for multi-day work, not an automatic archive
        replay loop.
        """
        try:
            wanted = max(1, min(32, int(limit)))
        except (TypeError, ValueError):
            wanted = 8
        try:
            char_limit = max(512, min(32_000, int(max_chars)))
        except (TypeError, ValueError):
            char_limit = MAX_SNAPSHOT_CHARS

        with self._lock():
            index = self._read_unlocked()
        terms = _focus_terms(focus)
        hot = list(index.get("records") or [])
        known = {
            str(item.get("fingerprint"))
            for item in hot
            if isinstance(item, Mapping) and item.get("fingerprint")
        }
        cold: List[Mapping[str, Any]] = []
        scanned_records = 0
        scanned_bytes = 0
        truncated = False
        archive_error: Optional[str] = None
        if self.archive_path.is_file():
            try:
                with gzip.open(self.archive_path, "rt", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        encoded = len(line.encode("utf-8"))
                        if scanned_bytes + encoded > MAX_ARCHIVE_SCAN_BYTES:
                            truncated = True
                            break
                        scanned_bytes += encoded
                        scanned_records += 1
                        try:
                            item = json.loads(line)
                        except (TypeError, ValueError):
                            continue
                        if not isinstance(item, Mapping) or not item.get("id"):
                            continue
                        fingerprint = str(item.get("fingerprint") or "")
                        if fingerprint and fingerprint in known:
                            continue
                        cold.append(item)
            except (OSError, EOFError, gzip.BadGzipFile) as exc:
                archive_error = f"{type(exc).__name__}: {exc}"

        rows = self._ordered_records(hot + cold, terms)
        chosen: List[Dict[str, Any]] = []
        for item in rows:
            candidate = {
                key: item.get(key)
                for key in ("id", "at", "kind", "source", "priority", "verified", "data")
                if key in item
            }
            if len(_canonical({"records": chosen + [candidate]})) > char_limit:
                data = candidate.get("data")
                if isinstance(data, Mapping):
                    candidate["data"] = {
                        key: _bound(data[key], depth=2)
                        for key in ("active_goal", "mission", "ledger", "steering", "staging", "goal_bank")
                        if key in data
                    }
                if len(_canonical({"records": chosen + [candidate]})) > char_limit:
                    continue
            chosen.append(candidate)
            if len(chosen) >= wanted:
                break

        retrieval: Dict[str, Any] = {
            "mode": "cold_recall",
            "focus": _clip(focus, 320),
            "terms": list(terms),
            "cold_scanned_records": scanned_records,
            "cold_scanned_bytes": scanned_bytes,
            "cold_truncated": truncated,
        }
        if archive_error:
            retrieval["cold_error"] = archive_error
        return sanitise_knowledge({
            "schema": KNOWLEDGE_SCHEMA,
            "available": True,
            "path": str(self.path),
            "generation": int(index.get("generation") or 0),
            "retrieval": retrieval,
            "records": chosen,
            "archive": {
                "path": str(self.archive_path),
                "compression": "gzip",
                "cold": True,
            },
        })


__all__ = [
    "KNOWLEDGE_ARCHIVE_FILENAME",
    "KNOWLEDGE_SCHEMA",
    "KnowledgeError",
    "KnowledgeStore",
    "MAX_ARCHIVE_SCAN_BYTES",
    "sanitise_knowledge",
]
