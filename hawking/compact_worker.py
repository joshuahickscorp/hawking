"""Compact, reference-first worker packets and source anchors.

Models contribute semantic intent.  Hawking owns source identity, authority,
and the exact bytes handed to the canonical mutation engine.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .persist import atomic_write_json


COMPACT_WORKER_SCHEMA = "hawking.compact_worker.v1"
ANCHOR_SCHEMA = "hawking.source_anchor.v1"
COMPACT_STATUSES = frozenset({"READ", "PLAN", "MUTATE", "REVIEW", "DONE", "BLOCKED"})


class CompactWorkerError(ValueError):
    """A compact packet or anchor cannot be admitted safely."""


class StaleAnchor(CompactWorkerError):
    code = "STALE_ANCHOR"


class NoOpMutation(CompactWorkerError):
    code = "NO_OP"


class InvalidLease(CompactWorkerError):
    code = "LEASE_INVALID"


@dataclass(frozen=True)
class CompactPacket:
    status: str
    refs: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()
    next: str = ""
    uncertainty: str = ""
    evidence: tuple[str, ...] = ()
    need: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"schema": COMPACT_WORKER_SCHEMA, "s": self.status}
        for key, data in (("refs", self.refs), ("facts", self.facts), ("evidence", self.evidence), ("need", self.need)):
            if data:
                value[key] = list(data)
        for key, data in (("next", self.next), ("uncertainty", self.uncertainty)):
            if data:
                value[key] = data
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompactPacket":
        status = str(value.get("s") or value.get("status") or "").strip().upper()
        if status not in COMPACT_STATUSES:
            raise CompactWorkerError("INVALID_COMPACT_STATUS")

        def strings(key: str) -> tuple[str, ...]:
            raw = value.get(key) or ()
            return (str(raw),) if isinstance(raw, str) else tuple(str(item) for item in raw if str(item).strip())

        return cls(status, strings("refs"), strings("facts"), str(value.get("next") or ""), str(value.get("uncertainty") or ""), strings("evidence"), strings("need"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _anchor_id(path: str, digest: str, start: int, end: int) -> str:
    return "SRC-" + _sha256(f"{path}:{digest}:{start}:{end}".encode())[:12].upper()


@dataclass(frozen=True)
class SourceAnchor:
    anchor: str
    path: str
    file_digest: str
    anchor_digest: str
    start_line: int
    end_line: int
    source: str
    workspace: str
    revision: str = ""
    symbol: str = ""
    schema: str = ANCHOR_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "anchor": self.anchor, "path": self.path,
            "file_digest": self.file_digest, "anchor_digest": self.anchor_digest,
            "start_line": self.start_line, "end_line": self.end_line,
            "workspace": self.workspace, "revision": self.revision, "symbol": self.symbol,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceAnchor":
        required = ("anchor", "path", "file_digest", "anchor_digest", "workspace")
        if any(not value.get(key) for key in required):
            raise CompactWorkerError("INVALID_SOURCE_ANCHOR")
        return cls(
            anchor=str(value["anchor"]), path=str(value["path"]),
            file_digest=str(value["file_digest"]), anchor_digest=str(value["anchor_digest"]),
            start_line=int(value.get("start_line") or 1), end_line=int(value.get("end_line") or 1),
            source=str(value.get("source") or ""), workspace=str(value["workspace"]),
            revision=str(value.get("revision") or ""), symbol=str(value.get("symbol") or ""),
        )


def create_source_anchor(root: str | Path, path: str | Path, *, start_line: int = 1,
                         end_line: int | None = None, symbol: str = "", revision: str = "") -> SourceAnchor:
    workspace = Path(root).expanduser().resolve()
    target = Path(path)
    if not target.is_absolute():
        target = workspace / target
    target = target.resolve()
    target.relative_to(workspace)
    raw = target.read_bytes()
    text = raw.decode("utf-8")
    lines = text.splitlines(keepends=True)
    first = max(1, int(start_line))
    last = len(lines) if end_line is None else min(len(lines), int(end_line))
    if first > last + 1:
        raise CompactWorkerError("INVALID_ANCHOR_RANGE")
    selected = "".join(lines[first - 1:last]) if lines else ""
    digest = _sha256(raw)
    anchor = SourceAnchor(
        anchor=_anchor_id(str(target.relative_to(workspace)), digest, first, last),
        path=str(target.relative_to(workspace)), file_digest=digest,
        anchor_digest=_sha256(selected.encode("utf-8")), start_line=first,
        end_line=last, source=selected, workspace=str(workspace), revision=revision, symbol=symbol,
    )
    store = workspace / ".hawking" / "anchors"
    atomic_write_json(store / f"{anchor.anchor}.json", anchor.to_dict() | {"source": selected})
    return anchor


def load_source_anchor(root: str | Path, anchor_id: str) -> SourceAnchor:
    path = Path(root).expanduser().resolve() / ".hawking" / "anchors" / f"{anchor_id}.json"
    try:
        return SourceAnchor.from_mapping(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise CompactWorkerError("MISSING_SOURCE_ANCHOR") from exc


def _current_anchor_source(anchor: SourceAnchor) -> tuple[Path, str, str]:
    workspace = Path(anchor.workspace).resolve()
    target = (workspace / anchor.path).resolve()
    target.relative_to(workspace)
    raw = target.read_bytes()
    current_file_digest = _sha256(raw)
    if current_file_digest != anchor.file_digest:
        raise StaleAnchor("source file digest changed")
    text = raw.decode("utf-8")
    lines = text.splitlines(keepends=True)
    current = "".join(lines[anchor.start_line - 1:anchor.end_line]) if lines else ""
    if _sha256(current.encode("utf-8")) != anchor.anchor_digest:
        raise StaleAnchor("source anchor digest changed")
    return target, current, current_file_digest


def compile_mutation_intent(intent: Mapping[str, Any], *, root: str | Path,
                            goal_id: str = "", workunit_id: str = "",
                            authority: Mapping[str, Any] | None = None):
    """Compile compact intent into the existing canonical MutationProposal."""
    from .mutation import MutationProposal

    if str(intent.get("s") or intent.get("status") or "").upper() != "MUTATE":
        raise CompactWorkerError("MUTATION_STATUS_REQUIRED")
    if any(key in intent for key in ("old_text", "old", "expected_base_hashes", "canonical_root", "mutation_lease")):
        raise CompactWorkerError("COMPACT_PROVIDER_STATE_FORBIDDEN")
    authority = authority or {}
    if authority and ("repo.edit" not in {str(item) for item in authority.get("capabilities") or ()} or not authority.get("mutation_lease")):
        raise InvalidLease("repo.edit lease required")
    anchor = load_source_anchor(root, str(intent.get("anchor") or ""))
    target, old_text, file_digest = _current_anchor_source(anchor)
    body = intent.get("body")
    if not isinstance(body, str):
        raise CompactWorkerError("MUTATION_BODY_REQUIRED")
    if body == old_text:
        raise NoOpMutation("replacement equals anchored source")
    op = str(intent.get("op") or "replace").lower()
    if op not in {"replace", "replace_range"}:
        raise CompactWorkerError("UNSUPPORTED_ANCHOR_OPERATION")
    tests = intent.get("tests") or ()
    tests = (tests,) if isinstance(tests, str) else tuple(str(item) for item in tests if str(item).strip())
    return MutationProposal(
        objective=str(intent.get("objective") or "compact anchored mutation"),
        operations=({"op": "replace", "path": str(target.relative_to(Path(anchor.workspace).resolve())), "old_text": old_text, "new_text": body},),
        expected_base_hashes={anchor.path: file_digest}, required_tests=tests,
        canonical_root=str(Path(anchor.workspace).resolve()), goal_id=goal_id,
        workunit_id=workunit_id,
        completion_contract={"compact_worker_schema": COMPACT_WORKER_SCHEMA, "anchor": anchor.anchor},
    )


def token_usage(raw: Mapping[str, Any] | None, *, stage: str, model: str = "") -> dict[str, Any]:
    """Normalize provider usage for the acceptance funnel without inventing cost."""
    usage = dict(raw or {})
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total = int(usage.get("total_tokens") or prompt + completion)
    result = {"stage": stage, "model": model, "input_tokens": prompt,
              "cached_input_tokens": int(usage.get("cached_input_tokens") or 0),
              "output_tokens": completion, "total_tokens": total}
    if usage.get("cost_usd") is not None:
        result["cost_usd"] = float(usage["cost_usd"])
    return result


def record_funnel_event(root: str | Path, *, workunit_id: str, model: str,
                        stage: str, usage: Mapping[str, Any] | None = None,
                        outcome: str = "") -> dict[str, Any]:
    """Append bounded stage accounting to the existing Hawking telemetry tree."""
    workspace = Path(root).expanduser().resolve()
    path = workspace / ".hawking" / "telemetry" / "acceptance-funnel.json"
    try:
        current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError, json.JSONDecodeError):
        current = {}
    if not isinstance(current, dict):
        current = {}
    current.setdefault("schema", "hawking.acceptance_funnel.v1")
    events = current.setdefault("events", [])
    if not isinstance(events, list):
        events = []
        current["events"] = events
    event = token_usage(usage, stage=stage, model=model)
    event.update({"workunit_id": str(workunit_id), "outcome": str(outcome or "")})
    events.append(event)
    # Keep telemetry bounded; durable WorkUnit receipts remain the detailed log.
    current["events"] = events[-2000:]
    atomic_write_json(path, current)
    return event
