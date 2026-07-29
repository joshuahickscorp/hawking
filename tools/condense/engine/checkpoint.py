#!/usr/bin/env python3.12
"""Durable checkpoint and hash-chained event log — one substrate for all campaigns.

Crash/resume contract:
* events are append-only, hash-chained JSONL
* checkpoints are atomic sealed snapshots pointing at the log tail
* resume loads the latest checkpoint and refuses split tails
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


CHECKPOINT_SCHEMA = "hawking.condense.controller_checkpoint.v1"
EVENT_SCHEMA = "hawking.condense.controller_event.v1"
GENESIS_HASH = "0" * 64


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{os.urandom(4).hex()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


@dataclass
class HashChainLog:
    """Append-only hash-chained JSONL event log."""

    path: Path
    _head: str = GENESIS_HASH
    _count: int = 0
    _loaded: bool = False

    def load(self) -> None:
        self._head = GENESIS_HASH
        self._count = 0
        if not self.path.is_file():
            self._loaded = True
            return
        prev = GENESIS_HASH
        count = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"corrupt event log {self.path}:{line_no}: {exc}"
                    ) from exc
                if not isinstance(event, dict):
                    raise RuntimeError(
                        f"corrupt event log {self.path}:{line_no}: not an object"
                    )
                if event.get("prev_sha256") != prev:
                    raise RuntimeError(
                        f"event log chain break at {self.path}:{line_no}"
                    )
                body = {k: v for k, v in event.items() if k != "event_sha256"}
                expected = _sha256(body)
                if event.get("event_sha256") != expected:
                    raise RuntimeError(
                        f"event seal mismatch at {self.path}:{line_no}"
                    )
                prev = expected
                count += 1
        self._head = prev
        self._count = count
        self._loaded = True

    @property
    def head(self) -> str:
        if not self._loaded:
            self.load()
        return self._head

    @property
    def count(self) -> int:
        if not self._loaded:
            self.load()
        return self._count

    def append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        if not self._loaded:
            self.load()
        body = {
            "schema": EVENT_SCHEMA,
            "prev_sha256": self._head,
            "at": _utc_now(),
            **dict(event),
        }
        sealed = {**body, "event_sha256": _sha256(body)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sealed, sort_keys=True, ensure_ascii=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._head = sealed["event_sha256"]
        self._count += 1
        return sealed


@dataclass
class CheckpointStore:
    """Atomic sealed checkpoints anchored to a hash-chained event log."""

    root: Path
    campaign_id: str
    checkpoint_name: str = "checkpoint.json"
    event_log_name: str = "events.jsonl"
    _log: HashChainLog | None = field(default=None, init=False, repr=False)

    @property
    def checkpoint_path(self) -> Path:
        return self.root / self.checkpoint_name

    @property
    def log(self) -> HashChainLog:
        if self._log is None:
            self._log = HashChainLog(self.root / self.event_log_name)
        return self._log

    def record(self, event: str, detail: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self.log.append(
            {
                "campaign_id": self.campaign_id,
                "event": event,
                "detail": dict(detail or {}),
            }
        )

    def save(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Write an atomic sealed checkpoint for *state*."""
        unsigned = {
            "schema": CHECKPOINT_SCHEMA,
            "campaign_id": self.campaign_id,
            "at": _utc_now(),
            "event_head": self.log.head,
            "event_count": self.log.count,
            "state": dict(state),
        }
        sealed = {**unsigned, "seal_sha256": _sha256(unsigned)}
        _atomic_text(
            self.checkpoint_path,
            json.dumps(sealed, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )
        return sealed

    def load(self) -> dict[str, Any] | None:
        path = self.checkpoint_path
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read checkpoint {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise RuntimeError(f"checkpoint root is not an object: {path}")
        recorded = raw.get("seal_sha256")
        unsigned = {k: v for k, v in raw.items() if k != "seal_sha256"}
        expected = _sha256(unsigned)
        if recorded != expected:
            raise RuntimeError(
                f"checkpoint seal mismatch: recorded={recorded!r} expected={expected}"
            )
        if raw.get("campaign_id") != self.campaign_id:
            raise RuntimeError(
                f"checkpoint campaign_id mismatch: {raw.get('campaign_id')!r} "
                f"!= {self.campaign_id!r}"
            )
        # Refuse split-brain: checkpoint head must match or be a prefix of live log.
        self.log.load()
        ck_head = raw.get("event_head")
        ck_count = int(raw.get("event_count") or 0)
        if ck_count > self.log.count:
            raise RuntimeError(
                f"checkpoint event_count {ck_count} ahead of log {self.log.count} "
                f"(split-brain / truncated log)"
            )
        if ck_count == self.log.count and ck_head != self.log.head:
            raise RuntimeError(
                "checkpoint event_head does not match log head (split-brain)"
            )
        return raw

    def resume_state(self) -> dict[str, Any]:
        """Load checkpoint state for resume, or empty initial state."""
        loaded = self.load()
        if loaded is None:
            return {
                "campaign_id": self.campaign_id,
                "phase": "idle",
                "completed_steps": [],
                "claims": [],
                "fault_reason": None,
            }
        state = loaded.get("state") or {}
        if not isinstance(state, dict):
            raise RuntimeError("checkpoint state is not an object")
        return dict(state)
