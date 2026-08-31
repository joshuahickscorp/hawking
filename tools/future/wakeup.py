"""RECEIPT_WAKEUP — completion wakes the graph, not the launcher.

A receipt appearing (or changing by content hash) at an expected path is the
completion event. The verifier, the dependency graph and the named frontier
are callables registered by name, so the launching context is irrelevant.
A restarted daemon can pick up work it never started because the evidence
and the expectation live on disk.

The model-facing API is register-expectation and dispatch-on-event. There is
no is-it-done-yet call a generation loop can sit in. Disk state is authority.

    python3 tools/future/wakeup.py --selftest
    python3 tools/future/wakeup.py --build
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO


import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from hcli.persist import atomic_write_json
from tools.future._common import git
from tools.future.repro_science import seal_is_valid as _repro_seal_is_valid
from tools.future.workunit_species import emit_hcli_workunit, validate_emitted_unit


RECEIPT = "RECEIPT_WAKEUP.json"
SCHEMA = "hawking.future.wakeup.v1"
LEDGER_SCHEMA = "hawking.future.wakeup.ledger.v1"
FIXTURE_SCHEMA = "hawking.future.wakeup.fixture.v1"
RECORDED_BY = "tools/future/wakeup.py"
VERSION = 1

# Three consumers dispatch always wakes. Names, not a rotting integer bound.
CONSUMER_KINDS: tuple[str, ...] = ("verifier", "graph", "frontier")
DEFAULT_FRONTIER = "future.wakeup"

WAITING = "WAITING"
SLEEPING = "SLEEPING"
COMPLETED = "COMPLETED"
MISSING_PAST_DEADLINE = "MISSING_PAST_DEADLINE"
PARTIAL_TRUNCATED = "PARTIAL_TRUNCATED"
SEAL_MISMATCH = "SEAL_MISMATCH"

TERMINAL_FAIL = frozenset({MISSING_PAST_DEADLINE, PARTIAL_TRUNCATED, SEAL_MISMATCH})
HARVEST_STATES = frozenset({COMPLETED, MISSING_PAST_DEADLINE, PARTIAL_TRUNCATED, SEAL_MISMATCH})

GRAPH_UNBLOCKED = "UNBLOCKED"
GRAPH_BLOCKED = "BLOCKED"

POLL_ALIASES = frozenset(
    {"is_done", "is_complete", "poll", "wait", "wait_for", "done_yet", "is_it_done_yet"}
)

ERAS = (
    "I Genesis of the Laboratory",
    "II Compounding Civilization",
    "III Autonomous Science Civilization",
    "IV Synthetic Machine Civilization",
    "V Released Hawking Civilization",
)
ODYSSEYS = (
    "I WHAT IS TRUE?",
    "II WHAT DID HAWKING ALREADY LEARN?",
    "III WHERE IS HAWKING WRONG?",
)

RECOVERY_PROBES: tuple[tuple[str, str], ...] = (
    (
        "tools/future/codex_ingest.py",
        "landed content-hashed cursor over a receipt directory; closest existing mechanism; extend the idea, do not fork the classifier",
    ),
    (
        "tools/future/propagate.py",
        "idempotent routing of deltas into named consumers; dispatch ledger is the same idea at completion grain",
    ),
    (
        "hcli/events.py",
        "in-memory EventBus; launching-context bound — recovered as the anti-pattern. Completion must not live here.",
    ),
    (
        "hcli/ledger.py",
        "obligation ledger: VERIFIED only via run_verify. Read-only. Wakeup does not mark HCLI obligations.",
    ),
    (
        "hcli/persist.py",
        "atomic_write_json is the crash-safe writer; the wakeup ledger goes through it.",
    ),
    (
        "tools/future/repro_science.py",
        "corrupt_receipt / partial_result fail-closed faults; seal_is_valid reused.",
    ),
    (
        "tools/future/freshness.py",
        "byte sha vs meaning; wakeup change detector is bytes, like ingest, never mtime.",
    ),
    (
        "tools/future/workunit_species.py",
        "emit_hcli_workunit into the recovered HCLI field set; SLEEPING maps to blocked + wakeup_state.",
    ),
    (
        "receipts/future/FUTURE_SUBSTRATE_HANDOFF.json",
        "sidecar inventory; wakeup was not in the 49-system set — this lane adds the resident-facing completion bus",
    ),
    (
        "receipts/future/CLAUDE_GLOBAL_FRONTIER.json",
        "live frontier the named consumer feeds; this module does not rewrite it",
    ),
    (
        "receipts/future/CODEX_INGEST_STATE.json",
        "durable sha256 cursor; shape reference for the wakeup ledger",
    ),
    (
        "receipts/future/PROPAGATION_STATE.json",
        "applied_keys ledger; same-event-once is the same invariant at a different grain",
    ),
)

Consumer = Callable[["WakeEvent"], None]


class FailClosed(Exception):
    """Refuse, and say why. Never a default, never a silent unblock."""

    def __init__(self, fault: str, reason: str) -> None:
        self.fault = fault
        self.reason = reason
        super().__init__(f"FAIL_CLOSED [{fault}]: {reason}")


# ---------------------------------------------------------------------------
# Bytes, seals, event identity. Pure functions of content, never of mtime.
# ---------------------------------------------------------------------------


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(blob).hexdigest()


def seal_document(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a content hash over everything except the hash itself (same as _common.seal)."""
    out = dict(doc)
    body = {k: v for k, v in out.items() if k != "seal_sha256"}
    out["seal_sha256"] = _canonical_hash(body)
    return out


def seal_is_valid(doc: Mapping[str, Any]) -> bool:
    """True iff seal_sha256 matches the canonical body. Missing/wrong is False."""
    if not isinstance(doc, dict):
        return False
    # Reuse the landed repro_science checker so a seal-algorithm drift is a
    # single-module bug, not two. Local seal_document matches _common.seal.
    try:
        return bool(_repro_seal_is_valid(dict(doc)))
    except (TypeError, ValueError):
        got = doc.get("seal_sha256")
        if not isinstance(got, str) or len(got) != 64:
            return False
        body = {k: v for k, v in doc.items() if k != "seal_sha256"}
        return _canonical_hash(body) == got


def _read_readonly(path: Path) -> bytes:
    """Open with O_RDONLY. Never create, never truncate. Matches codex_ingest."""
    fd = os.open(path, os.O_RDONLY)
    with os.fdopen(fd, "rb") as fh:
        return fh.read()


def write_sealed(path: str | Path, doc: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically write a sealed JSON receipt to an arbitrary path.

    Completion writers must replace, not append: a torn file is PARTIAL_TRUNCATED.
    """
    sealed = seal_document(doc)
    atomic_write_json(path, sealed)
    return sealed


def event_id_for(
    *,
    unit_id: str,
    path: str,
    state: str,
    content_sha256: str | None,
) -> str:
    return _canonical_hash(
        {
            "unit_id": str(unit_id),
            "path": str(path),
            "state": str(state),
            "content_sha256": str(content_sha256 or ""),
        }
    )


def classify_receipt_bytes(
    raw: bytes,
    *,
    required_schema: str | None = None,
) -> tuple[str, str]:
    """Pure function of file bytes. Same bytes ⇒ same (state, reason).

    Distinct terminals:
      PARTIAL_TRUNCATED  empty, undecodable, invalid JSON, non-object, complete:false
      SEAL_MISMATCH      parseable object whose seal is missing or does not match
      COMPLETED          sealed object that is not an explicit partial
    """
    if not raw or not raw.strip():
        return PARTIAL_TRUNCATED, "empty or whitespace-only file"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return PARTIAL_TRUNCATED, "not utf-8"
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return PARTIAL_TRUNCATED, f"truncated or invalid JSON: {exc.msg}"
    if not isinstance(obj, dict):
        return PARTIAL_TRUNCATED, "receipt is not a JSON object"
    if obj.get("complete") is False:
        return PARTIAL_TRUNCATED, "complete is false"
    if required_schema is not None and obj.get("schema") != required_schema:
        return PARTIAL_TRUNCATED, (
            f"schema {obj.get('schema')!r} does not match required {required_schema!r}"
        )
    if not seal_is_valid(obj):
        return SEAL_MISMATCH, "seal does not match canonical body (missing or wrong)"
    return COMPLETED, "sealed receipt"


# ---------------------------------------------------------------------------
# Event / result records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WakeEvent:
    event_id: str
    unit_id: str
    path: str
    state: str
    content_sha256: str | None
    verifier: str
    graph: str
    frontier: str
    dependents: tuple[str, ...]
    reason: str
    required_schema: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "unit_id": self.unit_id,
            "path": self.path,
            "state": self.state,
            "content_sha256": self.content_sha256,
            "verifier": self.verifier,
            "graph": self.graph,
            "frontier": self.frontier,
            "dependents": list(self.dependents),
            "reason": self.reason,
            "required_schema": self.required_schema,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "WakeEvent":
        deps = tuple(str(d) for d in (row.get("dependents") or ()))
        return cls(
            event_id=str(row["event_id"]),
            unit_id=str(row["unit_id"]),
            path=str(row["path"]),
            state=str(row["state"]),
            content_sha256=row.get("content_sha256"),
            verifier=str(row["verifier"]),
            graph=str(row["graph"]),
            frontier=str(row["frontier"]),
            dependents=deps,
            reason=str(row.get("reason") or ""),
            required_schema=row.get("required_schema"),
        )


@dataclass
class DispatchResult:
    event_id: str
    unit_id: str
    state: str
    duplicate: bool
    consumers_invoked: tuple[str, ...] = ()
    consumers_missing: tuple[str, ...] = ()
    unblocked: tuple[str, ...] = ()
    refused_unblocked: tuple[str, ...] = ()
    consumer_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "unit_id": self.unit_id,
            "state": self.state,
            "duplicate": self.duplicate,
            "consumers_invoked": list(self.consumers_invoked),
            "consumers_missing": list(self.consumers_missing),
            "unblocked": list(self.unblocked),
            "refused_unblocked": list(self.refused_unblocked),
            "consumer_errors": list(self.consumer_errors),
        }


# ---------------------------------------------------------------------------
# Durable watcher. Ledger on disk; callables in the resident that is awake.
# ---------------------------------------------------------------------------


def _empty_graph_row(*, blocked_by: Iterable[str] = (), state: str = GRAPH_BLOCKED) -> dict[str, Any]:
    by = sorted({str(x) for x in blocked_by if str(x)})
    return {
        "blocked_by": by,
        "state": GRAPH_UNBLOCKED if not by and state == GRAPH_UNBLOCKED else state,
        "reason": None,
    }


class Watcher:
    """Disk-backed expectation ledger plus named-consumer dispatch.

    Reconstructing a Watcher from the same ledger path shares no in-memory
    state with the process that registered the expectation. Consumers are
    rebound by name in the process that is awake.
    """

    def __init__(self, ledger_path: str | Path, *, root: str | Path | None = None) -> None:
        self.ledger_path = Path(ledger_path)
        self.root = Path(root) if root is not None else self.ledger_path.parent
        self._consumers: dict[str, Consumer] = {}
        self._expectations: dict[str, dict[str, Any]] = {}
        self._graph: dict[str, dict[str, Any]] = {}
        self._dispatched: dict[str, dict[str, Any]] = {}
        self._cursor: dict[str, dict[str, Any]] = {}
        if self.ledger_path.is_file():
            self._load()

    def register_consumer(self, name: str, fn: Consumer) -> None:
        if not name or not str(name).strip():
            raise ValueError("consumer name is required")
        if not callable(fn):
            raise TypeError("consumer must be callable")
        self._consumers[str(name)] = fn

    def register_expectation(
        self,
        *,
        unit_id: str,
        path: str | Path,
        dependents: Sequence[str] = (),
        verifier: str | None = None,
        graph: str | None = None,
        frontier: str | None = None,
        deadline_tick: int | None = None,
        required_schema: str | None = None,
        sleeping: bool = False,
    ) -> str:
        uid = str(unit_id).strip()
        if not uid:
            raise ValueError("unit_id is required")
        resolved = self._resolve(path)
        deps = tuple(sorted({str(d) for d in dependents if str(d).strip()}))
        rec = {
            "unit_id": uid,
            "path": str(resolved),
            "deadline_tick": None if deadline_tick is None else int(deadline_tick),
            "dependents": list(deps),
            "verifier": str(verifier or f"{uid}.verifier"),
            "graph": str(graph or f"{uid}.graph"),
            "frontier": str(frontier or DEFAULT_FRONTIER),
            "required_schema": required_schema,
            "sleeping": bool(sleeping),
            "state": SLEEPING if sleeping else WAITING,
            "content_sha256": None,
            "last_event_id": None,
            "reason": None,
        }
        existing = self._expectations.get(uid)
        if existing is not None:
            # Re-register is idempotent on identity; do not clobber a harvest.
            rec["state"] = existing.get("state") or rec["state"]
            rec["content_sha256"] = existing.get("content_sha256")
            rec["last_event_id"] = existing.get("last_event_id")
            rec["reason"] = existing.get("reason")
        self._expectations[uid] = rec
        self._touch_graph_for_register(uid, deps, sleeping=bool(sleeping))
        self._save()
        return uid

    def harvest(self, *, now_tick: int = 0) -> list[WakeEvent]:
        """Read expected paths. Return not-yet-dispatched completion/fail events.

        This is the supervisor/daemon entry, not a model poll. A generation
        loop has nothing to sit in: it registered and moved on.
        """
        tick = int(now_tick)
        events: list[WakeEvent] = []
        for uid in sorted(self._expectations):
            exp = self._expectations[uid]
            event = self._inspect(exp, tick)
            if event is None:
                continue
            exp["state"] = event.state
            exp["content_sha256"] = event.content_sha256
            exp["last_event_id"] = event.event_id
            exp["reason"] = event.reason
            self._cursor[event.path] = {
                "sha256": event.content_sha256,
                "state": event.state,
            }
            if event.event_id not in self._dispatched:
                events.append(event)
        self._save()
        return events

    def notify(self, path: str | Path, *, now_tick: int = 0) -> list[WakeEvent]:
        """FS-event entry: harvest only expectations whose path matches."""
        target = str(self._resolve(path))
        tick = int(now_tick)
        events: list[WakeEvent] = []
        for uid in sorted(self._expectations):
            exp = self._expectations[uid]
            if str(exp.get("path")) != target:
                continue
            event = self._inspect(exp, tick)
            if event is None:
                continue
            exp["state"] = event.state
            exp["content_sha256"] = event.content_sha256
            exp["last_event_id"] = event.event_id
            exp["reason"] = event.reason
            self._cursor[event.path] = {
                "sha256": event.content_sha256,
                "state": event.state,
            }
            if event.event_id not in self._dispatched:
                events.append(event)
        self._save()
        return events

    def dispatch(self, event: WakeEvent | Mapping[str, Any]) -> DispatchResult:
        """Wake verifier, graph, frontier. Same event_id is a no-op.

        COMPLETED is the only state that unblocks dependents. The three fail
        states notify consumers and refuse the unblock. A COMPLETED event
        whose bytes are not on disk is a synthetic result and is refused.
        """
        ev = event if isinstance(event, WakeEvent) else WakeEvent.from_dict(event)
        if ev.event_id in self._dispatched:
            prev = self._dispatched[ev.event_id]
            return DispatchResult(
                event_id=ev.event_id,
                unit_id=ev.unit_id,
                state=str(prev.get("state") or ev.state),
                duplicate=True,
            )
        if ev.state == COMPLETED:
            self._assert_disk_confirms(ev)

        # Record first so a crashing consumer cannot double-fire on retry.
        # Consumers must be idempotent; this module guarantees at-most-once.
        self._dispatched[ev.event_id] = {
            "event_id": ev.event_id,
            "unit_id": ev.unit_id,
            "state": ev.state,
            "content_sha256": ev.content_sha256,
        }
        exp = self._expectations.get(ev.unit_id)
        if exp is not None:
            exp["state"] = ev.state
            exp["content_sha256"] = ev.content_sha256
            exp["last_event_id"] = ev.event_id
            exp["reason"] = ev.reason

        unblocked, refused = self._apply_graph(ev)
        invoked: list[str] = []
        missing: list[str] = []
        errors: list[str] = []
        for kind in CONSUMER_KINDS:
            name = getattr(ev, kind)
            fn = self._consumers.get(name)
            if fn is None:
                missing.append(kind)
                continue
            try:
                fn(ev)
            except Exception as exc:  # noqa: BLE001 — record, do not unblock-on-raise
                errors.append(f"{kind}:{type(exc).__name__}:{exc}")
                continue
            invoked.append(kind)

        self._dispatched[ev.event_id]["consumers_invoked"] = list(invoked)
        self._dispatched[ev.event_id]["consumers_missing"] = list(missing)
        self._save()
        return DispatchResult(
            event_id=ev.event_id,
            unit_id=ev.unit_id,
            state=ev.state,
            duplicate=False,
            consumers_invoked=tuple(invoked),
            consumers_missing=tuple(missing),
            unblocked=tuple(unblocked),
            refused_unblocked=tuple(refused),
            consumer_errors=tuple(errors),
        )

    def record(self, unit_id: str) -> dict[str, Any]:
        """Ledger inspection. Not a poll: does not harvest, does not wait."""
        rec = self._expectations.get(str(unit_id))
        if rec is None:
            return {"unit_id": str(unit_id), "state": None, "present_in_ledger": False}
        out = dict(rec)
        out["present_in_ledger"] = True
        return out

    def graph_record(self, unit_id: str) -> dict[str, Any]:
        rec = self._graph.get(str(unit_id))
        if rec is None:
            return {
                "unit_id": str(unit_id),
                "blocked_by": [],
                "state": None,
                "reason": None,
                "present_in_ledger": False,
            }
        return {
            "unit_id": str(unit_id),
            "blocked_by": list(rec.get("blocked_by") or []),
            "state": rec.get("state"),
            "reason": rec.get("reason"),
            "present_in_ledger": True,
        }

    def dispatched_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._dispatched))

    def __getattr__(self, name: str) -> Any:
        if name in POLL_ALIASES:
            raise AttributeError(
                f"{name!r} is not part of the wakeup API; "
                "register an expectation and dispatch on the disk event. "
                "The model does not poll."
            )
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    # --- internals ---------------------------------------------------------

    def _resolve(self, path: str | Path) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        return p

    def _touch_graph_for_register(self, uid: str, deps: Sequence[str], *, sleeping: bool) -> None:
        self_row = self._graph.get(uid) or _empty_graph_row(state=SLEEPING if sleeping else WAITING)
        if self_row.get("state") in {None, WAITING, SLEEPING, GRAPH_BLOCKED}:
            if self_row.get("state") not in HARVEST_STATES and self_row.get("state") != GRAPH_UNBLOCKED:
                self_row["state"] = SLEEPING if sleeping else WAITING
        self._graph[uid] = self_row
        for dep in deps:
            row = self._graph.get(dep) or _empty_graph_row(blocked_by=(uid,), state=GRAPH_BLOCKED)
            blocked = list(row.get("blocked_by") or [])
            if uid not in blocked:
                blocked.append(uid)
            row["blocked_by"] = sorted(blocked)
            if row.get("state") == GRAPH_UNBLOCKED and row["blocked_by"]:
                row["state"] = GRAPH_BLOCKED
            if not row.get("state"):
                row["state"] = GRAPH_BLOCKED
            self._graph[dep] = row

    def _inspect(self, exp: dict[str, Any], now_tick: int) -> WakeEvent | None:
        path = Path(exp["path"])
        present = False
        try:
            st = path.lstat()
            if stat.S_ISREG(st.st_mode):
                present = True
        except OSError:
            present = False

        if not present:
            deadline = exp.get("deadline_tick")
            if deadline is not None and int(now_tick) > int(deadline):
                event = self._make_event(exp, MISSING_PAST_DEADLINE, None, "receipt missing past deadline")
                if event.event_id == exp.get("last_event_id") and event.event_id in self._dispatched:
                    return None
                return event
            return None

        try:
            raw = _read_readonly(path)
        except OSError:
            return None
        sha = _sha256_bytes(raw)
        if sha == exp.get("content_sha256") and exp.get("last_event_id"):
            return None
        state, reason = classify_receipt_bytes(
            raw, required_schema=exp.get("required_schema")
        )
        return self._make_event(exp, state, sha, reason)

    def _make_event(
        self,
        exp: Mapping[str, Any],
        state: str,
        sha: str | None,
        reason: str,
    ) -> WakeEvent:
        path = str(exp["path"])
        uid = str(exp["unit_id"])
        eid = event_id_for(unit_id=uid, path=path, state=state, content_sha256=sha)
        deps = tuple(str(d) for d in (exp.get("dependents") or ()))
        return WakeEvent(
            event_id=eid,
            unit_id=uid,
            path=path,
            state=state,
            content_sha256=sha,
            verifier=str(exp["verifier"]),
            graph=str(exp["graph"]),
            frontier=str(exp["frontier"]),
            dependents=deps,
            reason=reason,
            required_schema=exp.get("required_schema"),
        )

    def _assert_disk_confirms(self, ev: WakeEvent) -> None:
        path = Path(ev.path)
        try:
            raw = _read_readonly(path)
        except OSError as exc:
            raise FailClosed(
                "synthetic_completion",
                f"COMPLETED event for {ev.unit_id!r} has no readable file at {ev.path}: {exc}",
            ) from exc
        sha = _sha256_bytes(raw)
        if ev.content_sha256 and sha != ev.content_sha256:
            raise FailClosed(
                "synthetic_completion",
                f"COMPLETED event sha {ev.content_sha256} does not match disk {sha}",
            )
        state, reason = classify_receipt_bytes(raw, required_schema=ev.required_schema)
        if state != COMPLETED:
            raise FailClosed(
                "synthetic_completion",
                f"disk classifies as {state} ({reason}); refusing to forge COMPLETED",
            )

    def _apply_graph(self, ev: WakeEvent) -> tuple[list[str], list[str]]:
        unblocked: list[str] = []
        refused: list[str] = []
        self_row = self._graph.get(ev.unit_id) or _empty_graph_row(state=ev.state)
        self_row["state"] = ev.state
        self_row["reason"] = ev.reason
        self._graph[ev.unit_id] = self_row
        if ev.state == COMPLETED:
            for dep in ev.dependents:
                row = self._graph.get(dep) or _empty_graph_row(blocked_by=(ev.unit_id,), state=GRAPH_BLOCKED)
                blocked = [x for x in (row.get("blocked_by") or []) if x != ev.unit_id]
                row["blocked_by"] = sorted(blocked)
                if not blocked:
                    if row.get("state") != GRAPH_UNBLOCKED:
                        unblocked.append(dep)
                    row["state"] = GRAPH_UNBLOCKED
                    row["reason"] = None
                else:
                    row["state"] = GRAPH_BLOCKED
                    row["reason"] = row.get("reason")
                    refused.append(dep)
                self._graph[dep] = row
        else:
            for dep in ev.dependents:
                row = self._graph.get(dep) or _empty_graph_row(blocked_by=(ev.unit_id,), state=GRAPH_BLOCKED)
                blocked = list(row.get("blocked_by") or [])
                if ev.unit_id not in blocked:
                    blocked.append(ev.unit_id)
                row["blocked_by"] = sorted(blocked)
                row["state"] = GRAPH_BLOCKED
                row["reason"] = f"upstream:{ev.unit_id}:{ev.state}"
                self._graph[dep] = row
                refused.append(dep)
        unblocked.sort()
        refused.sort()
        return unblocked, refused

    def _ledger_body(self) -> dict[str, Any]:
        return {
            "schema": LEDGER_SCHEMA,
            "version": VERSION,
            "expectations": {k: self._expectations[k] for k in sorted(self._expectations)},
            "graph": {k: self._graph[k] for k in sorted(self._graph)},
            "dispatched": {k: self._dispatched[k] for k in sorted(self._dispatched)},
            "cursor": {k: self._cursor[k] for k in sorted(self._cursor)},
        }

    def _save(self) -> None:
        body = self._ledger_body()
        body["checksum_sha256"] = _canonical_hash(body)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.ledger_path, body)

    def _load(self) -> None:
        try:
            doc = load_json(self.ledger_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FailClosed(
                "corrupt_ledger",
                f"wakeup ledger at {self.ledger_path} is truncated or unreadable: {exc}",
            ) from exc
        if not isinstance(doc, dict):
            raise FailClosed("corrupt_ledger", "wakeup ledger is not a JSON object")
        got = doc.get("checksum_sha256")
        body = {k: v for k, v in doc.items() if k != "checksum_sha256"}
        expected = _canonical_hash(body)
        if not isinstance(got, str) or got != expected:
            raise FailClosed(
                "corrupt_ledger",
                "wakeup ledger checksum does not match; refusing to reset the dispatch set",
            )
        if doc.get("schema") != LEDGER_SCHEMA:
            raise FailClosed(
                "corrupt_ledger",
                f"wakeup ledger schema {doc.get('schema')!r} is not {LEDGER_SCHEMA}",
            )
        self._expectations = {
            str(k): dict(v) for k, v in (doc.get("expectations") or {}).items() if isinstance(v, dict)
        }
        self._graph = {
            str(k): dict(v) for k, v in (doc.get("graph") or {}).items() if isinstance(v, dict)
        }
        self._dispatched = {
            str(k): dict(v) for k, v in (doc.get("dispatched") or {}).items() if isinstance(v, dict)
        }
        self._cursor = {
            str(k): dict(v) for k, v in (doc.get("cursor") or {}).items() if isinstance(v, dict)
        }


def dispatch(event: WakeEvent | Mapping[str, Any], watcher: Watcher) -> DispatchResult:
    """Module-level dispatch. The watcher is explicit: no hidden launching context."""
    return watcher.dispatch(event)


def run_once(watcher: Watcher, *, now_tick: int = 0) -> list[DispatchResult]:
    """Supervisor step: harvest disk events, dispatch each once."""
    events = watcher.harvest(now_tick=now_tick)
    return [watcher.dispatch(ev) for ev in events]


def __getattr__(name: str) -> Any:
    if name in POLL_ALIASES:
        raise AttributeError(
            f"{name!r} is not part of the wakeup API; "
            "register an expectation and dispatch on the disk event. "
            "The model does not poll."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# WorkUnit emission — HCLI field set, SLEEPING is blocked not a fake result.
# ---------------------------------------------------------------------------


def emit_wakeup_workunits() -> list[dict[str, Any]]:
    """Units the resident can schedule. Counts derived from the tuple below."""
    specs = (
        {
            "id": "future.wakeup.dispatch-on-receipt",
            "role": "science",
            "description": (
                "Watch sealed receipts at expected paths and dispatch verifier, "
                "dependency graph and named frontier. Completion evidence is on "
                "disk; a supervisor restart harvests work it did not start."
            ),
            "status": "pending",
            "classification": "STATIC_ONLY",
            "resource_class": "STATIC_ANALYSIS",
            "effect_class": "READ_ONLY",
            "wakeup_state": WAITING,
            "blocked_reason": None,
            "sleeping": False,
        },
        {
            "id": "future.wakeup.sleeping-until-qualification",
            "role": "science",
            "description": (
                "SLEEPING WorkUnit for blocked physical work. Wakes when a sealed "
                "qualification receipt appears on disk. Never a synthetic GPU or "
                "Metal result; UNKNOWN stays UNKNOWN until hardware qualifies."
            ),
            "status": "blocked",
            "classification": "BLOCKED",
            "resource_class": "STATIC_ANALYSIS",
            "effect_class": "READ_ONLY",
            "wakeup_state": SLEEPING,
            "blocked_reason": (
                "SLEEPING: awaiting a sealed qualification receipt on disk. "
                "MetalContext / xcrun / protected bench lock / HEAVY quiescence "
                "are Codex physical blockers; this sidecar will not invent them."
            ),
            "sleeping": True,
        },
    )
    units: list[dict[str, Any]] = []
    for spec in specs:
        extras = {
            "species": "independent_reproduction",
            "evidence_parents": [
                "tools/future/codex_ingest.py",
                "tools/future/propagate.py",
                "tools/future/repro_science.py",
                "hcli/persist.py",
            ],
            "output_receipt_path": f"receipts/future/{RECEIPT}",
            "claim_boundary": (
                "Static sidecar artifact. No hardware measurement. "
                "Cannot promote, weaken a verifier, or forge a completion."
            ),
            "requires_quiescence": False,
            "candidate_status": "STATIC_ONLY",
            "wakeup_state": spec["wakeup_state"],
            "blocked_reason": spec["blocked_reason"],
            "frontier": DEFAULT_FRONTIER,
        }
        row = emit_hcli_workunit(
            id=spec["id"],
            role=spec["role"],
            description=spec["description"],
            dependencies=[],
            resource_class=spec["resource_class"],
            verifier="future.wakeup.seal_and_dispatch",
            provider="future.wakeup",
            effect_class=spec["effect_class"],
            status=spec["status"],
            classification=spec["classification"],
            extras=extras,
        )
        validate_emitted_unit(row)
        units.append(row)
    units.sort(key=lambda r: str(r["id"]))
    return units


# ---------------------------------------------------------------------------
# Recovery, proofs, receipt.
# ---------------------------------------------------------------------------


def _probe_path(rel: str) -> dict[str, Any]:
    p = REPO / rel
    in_git = bool(git("ls-tree", "--name-only", "HEAD", rel).strip())
    return {"path": rel, "on_disk": p.is_file() or p.is_dir(), "in_git": in_git}


def recovered_implementation() -> dict[str, Any]:
    rows = []
    for path, note in RECOVERY_PROBES:
        row = _probe_path(path)
        row["note"] = note
        rows.append(row)
    present = [r["path"] for r in rows if r["on_disk"] or r["in_git"]]
    unresolved = [r["path"] for r in rows if not r["on_disk"] and not r["in_git"]]
    return {
        "already_existed": rows,
        "present": present,
        "unresolved_in_this_checkout": unresolved,
        "adequate_existing_watcher": None,
        "why_not_redundant": (
            "codex_ingest.py hashes a directory and emits LAW/SCAR deltas; it does "
            "not register a per-unit expectation, wake a verifier/graph/frontier by "
            "name, fail-close on truncated/seal-mismatch/deadline, or survive a "
            "supervisor restart as a completion bus. hcli.events.EventBus is "
            "in-memory and dies with the launching context. propagate.py is the "
            "idempotent consumer router this module mirrors at completion grain."
        ),
        "extended": [
            "content-hashed cursor (codex_ingest) — appearance OR sha change is the event",
            "idempotent applied-key ledger (propagate) — same event_id dispatches once",
            "corrupt_receipt / partial_result (repro_science) — distinct terminals, no silent pass",
            "atomic_write_json (hcli.persist) — torn ledger must not reset the dispatch set",
            "O_RDONLY open (codex_ingest) — the watcher never creates the evidence it waits on",
        ],
        "not_used_as_completion_path": "hcli.events.EventBus",
    }


def gaps_closed() -> list[str]:
    return [
        "receipt appearance or content-hash change at an expected path is the completion event",
        "dispatch(event) wakes named verifier, dependency graph and named frontier callables",
        "cross-process: a child writes the receipt; a freshly-constructed watcher dispatches",
        "MISSING_PAST_DEADLINE, PARTIAL_TRUNCATED and SEAL_MISMATCH are distinct terminals that never unblock dependents",
        "same completion event_id dispatches once; replay is a no-op",
        "no is-it-done-yet / poll / wait API; model-facing surface is register-expectation and dispatch-on-event",
        "SLEEPING units wait on disk evidence of qualification; synthetic COMPLETED without matching bytes is refused",
        "corrupt wakeup ledger fails closed instead of resetting the dispatch set",
    ]


def negative_findings(recovered: Mapping[str, Any]) -> list[str]:
    findings = [
        f"unresolved: {p}" for p in recovered.get("unresolved_in_this_checkout") or []
    ]
    findings.extend(
        [
            "hcli.events.EventBus cannot be the completion path: it is in-memory and bound to the launching process",
            "Codex physical blockers (no Metal GPU, no Metal compiler, unproven bench locks, HEAVY quiescence, Flash NX SCAFFOLD_ONLY, teacher capture 0/256) stay SLEEPING; wakeup will not mint a synthetic qualification receipt",
            "this sidecar produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE; every emission is STATIC_ONLY / bench UNKNOWN",
            "named consumers workgraph.py / frontiers.py / detached.py are this-wave siblings and are not imported; Watcher is the local interface until they land",
        ]
    )
    return findings


def resident_callable(work_units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "can_hcli_invoke": True,
        "entry_point": "python3 tools/future/wakeup.py --selftest",
        "callables": {
            "register_expectation": "tools.future.wakeup.Watcher.register_expectation",
            "harvest": "tools.future.wakeup.Watcher.harvest",
            "dispatch": "tools.future.wakeup.dispatch",
            "run_once": "tools.future.wakeup.run_once",
        },
        "workunit_emitted": [u.get("id") for u in work_units],
        "workunits": list(work_units),
        "receipt": f"receipts/future/{RECEIPT}",
        "frontier_fed": DEFAULT_FRONTIER,
        "fail_closed": {
            MISSING_PAST_DEADLINE: (
                "expected path absent and now_tick > deadline_tick; dependents stay BLOCKED"
            ),
            PARTIAL_TRUNCATED: (
                "empty, undecodable, invalid JSON, non-object, or complete:false; dependents stay BLOCKED"
            ),
            SEAL_MISMATCH: (
                "parseable object whose seal_sha256 is missing or does not match canonical body; dependents stay BLOCKED"
            ),
            "synthetic_completion": (
                "dispatch of COMPLETED without matching sealed bytes on disk is refused"
            ),
            "corrupt_ledger": (
                "truncated or checksum-mismatched wakeup ledger refuses to load rather than replay every event"
            ),
        },
        "how_the_resident_uses_it": (
            "HCLI registers an expectation naming the output receipt, the verifier, "
            "the graph consumer and the frontier. The worker writes the receipt via "
            "atomic replace. A supervisor — possibly a different process after a "
            "restart — harvests the path and dispatch() wakes the three consumers. "
            "The model never asks whether it is done."
        ),
    }


def _require(cond: Any, fault: str, reason: str) -> None:
    if not cond:
        raise FailClosed(fault, reason)


def _capture() -> tuple[dict[str, list[str]], dict[str, Consumer]]:
    hits: dict[str, list[str]] = {k: [] for k in CONSUMER_KINDS}

    def _mk(kind: str) -> Consumer:
        def _fn(event: WakeEvent) -> None:
            hits[kind].append(event.state)

        return _fn

    consumers = {
        "v": _mk("verifier"),
        "g": _mk("graph"),
        "f": _mk("frontier"),
    }
    return hits, consumers


def _bind(watcher: Watcher) -> dict[str, list[str]]:
    hits, consumers = _capture()
    for name, fn in consumers.items():
        watcher.register_consumer(name, fn)
    return hits


def _spawn(src: str, *, timeout: int = 20) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + pp if pp else "")
    return subprocess.run(
        [sys.executable, "-c", src],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _fixture(unit: str, *, complete: bool = True) -> dict[str, Any]:
    return {
        "schema": FIXTURE_SCHEMA,
        "complete": complete,
        "unit": unit,
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
    }


def run_proofs(tmp: Path | None = None) -> dict[str, Any]:
    """Execute the fail-closed and cross-process proofs. Raise if a guard is silent."""
    if tmp is not None:
        root = Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        return _run_proofs_in(root)
    with tempfile.TemporaryDirectory(prefix="wakeup-proof-") as td:
        return _run_proofs_in(Path(td))


def _run_proofs_in(root: Path) -> dict[str, Any]:
    proofs: dict[str, Any] = {}

    # --- valid completion unblocks and wakes every consumer kind -----------
    d_ok = root / "ok"
    d_ok.mkdir(exist_ok=True)
    w = Watcher(d_ok / "ledger.json")
    hits = _bind(w)
    w.register_expectation(
        unit_id="ok.u",
        path=d_ok / "done.json",
        dependents=["ok.child"],
        verifier="v",
        graph="g",
        frontier="f",
    )
    write_sealed(d_ok / "done.json", _fixture("ok.u"))
    results = run_once(w)
    _require(len(results) == 1, "selftest", "valid receipt did not harvest one event")
    _require(results[0].state == COMPLETED, "selftest", f"expected COMPLETED, got {results[0].state}")
    _require(not results[0].duplicate, "selftest", "first dispatch marked duplicate")
    _require(
        set(results[0].consumers_invoked) == set(CONSUMER_KINDS),
        "selftest",
        f"consumers invoked {results[0].consumers_invoked} != {CONSUMER_KINDS}",
    )
    child = w.graph_record("ok.child")
    _require(child["state"] == GRAPH_UNBLOCKED, "selftest", f"child not unblocked: {child}")
    _require(
        all(hits[k] == [COMPLETED] for k in CONSUMER_KINDS),
        "selftest",
        f"consumer hits {hits}",
    )
    proofs["valid_unblocks_and_wakes_consumers"] = True

    # --- truncated: distinct terminal, dependents stay blocked --------------
    d_tr = root / "trunc"
    d_tr.mkdir(exist_ok=True)
    w = Watcher(d_tr / "ledger.json")
    hits = _bind(w)
    w.register_expectation(
        unit_id="trunc.u",
        path=d_tr / "done.json",
        dependents=["trunc.child"],
        verifier="v",
        graph="g",
        frontier="f",
    )
    (d_tr / "done.json").write_bytes(b'{"schema": "hawking.future.wakeup.fixture.v1", "complete":')
    results = run_once(w)
    _require(len(results) == 1, "selftest", "truncated receipt did not harvest")
    _require(
        results[0].state == PARTIAL_TRUNCATED,
        "selftest",
        f"truncated classified {results[0].state}, not {PARTIAL_TRUNCATED}",
    )
    child = w.graph_record("trunc.child")
    _require(child["state"] != GRAPH_UNBLOCKED, "selftest", "truncated receipt unblocked a dependent")
    _require(child["state"] == GRAPH_BLOCKED, "selftest", f"truncated child state {child}")
    _require("trunc.u" in child["blocked_by"], "selftest", "parent dropped from blocked_by on truncated")
    proofs["truncated_does_not_unblock"] = True
    proofs["truncated_terminal"] = PARTIAL_TRUNCATED

    # --- seal mismatch: distinct terminal, dependents stay blocked ----------
    d_sm = root / "seal"
    d_sm.mkdir(exist_ok=True)
    w = Watcher(d_sm / "ledger.json")
    hits = _bind(w)
    w.register_expectation(
        unit_id="seal.u",
        path=d_sm / "done.json",
        dependents=["seal.child"],
        verifier="v",
        graph="g",
        frontier="f",
    )
    body = _fixture("seal.u")
    body["seal_sha256"] = "0" * 64
    atomic_write_json(d_sm / "done.json", body)
    results = run_once(w)
    _require(len(results) == 1, "selftest", "seal-mismatch receipt did not harvest")
    _require(
        results[0].state == SEAL_MISMATCH,
        "selftest",
        f"seal mismatch classified {results[0].state}, not {SEAL_MISMATCH}",
    )
    child = w.graph_record("seal.child")
    _require(child["state"] == GRAPH_BLOCKED, "selftest", "seal-mismatch unblocked a dependent")
    proofs["seal_mismatch_does_not_unblock"] = True
    proofs["seal_mismatch_terminal"] = SEAL_MISMATCH

    # --- dispatch twice performs the downstream action once -----------------
    d_id = root / "idemp"
    d_id.mkdir(exist_ok=True)
    w = Watcher(d_id / "ledger.json")
    hits = _bind(w)
    w.register_expectation(
        unit_id="idemp.u",
        path=d_id / "done.json",
        dependents=["idemp.child"],
        verifier="v",
        graph="g",
        frontier="f",
    )
    write_sealed(d_id / "done.json", _fixture("idemp.u"))
    events = w.harvest()
    _require(len(events) == 1, "selftest", "idempotence harvest did not yield one event")
    first = w.dispatch(events[0])
    second = w.dispatch(events[0])
    _require(not first.duplicate, "selftest", "first dispatch was duplicate")
    _require(second.duplicate, "selftest", "second dispatch was not duplicate")
    _require(
        all(hits[k] == [COMPLETED] for k in CONSUMER_KINDS),
        "selftest",
        f"idempotence consumer hits {hits} (downstream must run once)",
    )
    replay = run_once(w)
    _require(replay == [], "selftest", f"replay harvest dispatched again: {replay}")
    proofs["dispatch_twice_is_once"] = True
    proofs["replay_is_noop"] = True

    # --- missing past deadline ---------------------------------------------
    d_ms = root / "miss"
    d_ms.mkdir(exist_ok=True)
    w = Watcher(d_ms / "ledger.json")
    hits = _bind(w)
    w.register_expectation(
        unit_id="miss.u",
        path=d_ms / "never.json",
        dependents=["miss.child"],
        verifier="v",
        graph="g",
        frontier="f",
        deadline_tick=0,
    )
    results = run_once(w, now_tick=1)
    _require(len(results) == 1, "selftest", "missing-past-deadline did not harvest")
    _require(
        results[0].state == MISSING_PAST_DEADLINE,
        "selftest",
        f"missing classified {results[0].state}",
    )
    child = w.graph_record("miss.child")
    _require(child["state"] == GRAPH_BLOCKED, "selftest", "missing-past-deadline unblocked a dependent")
    proofs["missing_past_deadline"] = True
    proofs["missing_terminal"] = MISSING_PAST_DEADLINE

    # --- SLEEPING without a receipt is not a synthetic result ---------------
    d_sl = root / "sleep"
    d_sl.mkdir(exist_ok=True)
    w = Watcher(d_sl / "ledger.json")
    w.register_expectation(
        unit_id="sleep.u",
        path=d_sl / "qual.json",
        dependents=["sleep.child"],
        verifier="v",
        graph="g",
        frontier="f",
        sleeping=True,
    )
    results = run_once(w, now_tick=10**9)
    _require(results == [], "selftest", "SLEEPING unit dispatched without a receipt")
    rec = w.record("sleep.u")
    _require(rec["state"] == SLEEPING, "selftest", f"SLEEPING unit moved to {rec['state']}")
    child = w.graph_record("sleep.child")
    _require(child["state"] == GRAPH_BLOCKED, "selftest", "SLEEPING unit unblocked a dependent")
    forged = WakeEvent(
        event_id=event_id_for(
            unit_id="sleep.u",
            path=str(d_sl / "qual.json"),
            state=COMPLETED,
            content_sha256="ab" * 32,
        ),
        unit_id="sleep.u",
        path=str(d_sl / "qual.json"),
        state=COMPLETED,
        content_sha256="ab" * 32,
        verifier="v",
        graph="g",
        frontier="f",
        dependents=("sleep.child",),
        reason="forged",
    )
    refused = False
    try:
        w.dispatch(forged)
    except FailClosed as exc:
        refused = exc.fault == "synthetic_completion"
        _require(refused, "selftest", f"forged COMPLETED failed closed as {exc.fault}")
    _require(refused, "selftest", "forged COMPLETED was accepted")
    child = w.graph_record("sleep.child")
    _require(child["state"] == GRAPH_BLOCKED, "selftest", "forged COMPLETED unblocked a dependent")
    proofs["sleeping_without_receipt_is_not_synthetic"] = True
    proofs["synthetic_completion_refused"] = True

    # --- mtime alone is not a completion -----------------------------------
    d_mt = root / "mtime"
    d_mt.mkdir(exist_ok=True)
    w = Watcher(d_mt / "ledger.json")
    w.register_expectation(
        unit_id="mtime.u",
        path=d_mt / "done.json",
        verifier="v",
        graph="g",
        frontier="f",
    )
    write_sealed(d_mt / "done.json", _fixture("mtime.u"))
    first = run_once(w)
    _require(len(first) == 1 and first[0].state == COMPLETED, "selftest", "mtime setup failed")
    os.utime(d_mt / "done.json", (0, 0))
    again = run_once(w)
    _require(again == [], "selftest", "mtime-only change re-dispatched")
    proofs["mtime_is_not_a_change"] = True

    # --- corrupt ledger fails closed ----------------------------------------
    d_lg = root / "ledger-corrupt"
    d_lg.mkdir(exist_ok=True)
    w = Watcher(d_lg / "ledger.json")
    w.register_expectation(unit_id="lg.u", path=d_lg / "x.json", verifier="v", graph="g", frontier="f")
    (d_lg / "ledger.json").write_bytes(b'{"schema": "hawking.future.wakeup.ledger.v1"')
    closed = False
    try:
        Watcher(d_lg / "ledger.json")
    except FailClosed as exc:
        closed = exc.fault == "corrupt_ledger"
    _require(closed, "selftest", "truncated ledger did not fail closed")
    proofs["corrupt_ledger_fail_closed"] = True

    # --- cross-process: registrar dies, writer is a child, detector is fresh
    d_xp = root / "xproc"
    d_xp.mkdir(exist_ok=True)
    ledger = d_xp / "ledger.json"
    receipt = d_xp / "done.json"
    register_src = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "from pathlib import Path\n"
        "from tools.future.wakeup import Watcher\n"
        f"w = Watcher(Path({str(ledger)!r}))\n"
        "w.register_expectation(\n"
        "    unit_id='cross.u',\n"
        f"    path=Path({str(receipt)!r}),\n"
        "    dependents=['cross.child'],\n"
        "    verifier='v', graph='g', frontier='f',\n"
        ")\n"
        "print('registered')\n"
    )
    write_src = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "from pathlib import Path\n"
        "from tools.future.wakeup import write_sealed\n"
        f"write_sealed(Path({str(receipt)!r}), "
        "{'schema': 'hawking.future.wakeup.fixture.v1', 'complete': True, "
        "'unit': 'cross.u', 'evidence_class': 'STATIC_ONLY', "
        "'bench_state': 'UNKNOWN', 'gpu_authority': False})\n"
        "print('wrote')\n"
    )
    r1 = _spawn(register_src)
    _require(r1.returncode == 0, "cross_process", f"registrar child failed: {r1.stderr}")
    r2 = _spawn(write_src)
    _require(r2.returncode == 0, "cross_process", f"writer child failed: {r2.stderr}")
    detector = Watcher(ledger)
    hits = _bind(detector)
    results = run_once(detector)
    _require(len(results) == 1, "cross_process", f"fresh watcher harvested {len(results)} events")
    _require(results[0].state == COMPLETED, "cross_process", f"cross-process state {results[0].state}")
    _require(not results[0].duplicate, "cross_process", "cross-process dispatch was duplicate")
    child = detector.graph_record("cross.child")
    _require(child["state"] == GRAPH_UNBLOCKED, "cross_process", f"cross-process child {child}")
    _require(
        all(hits[k] == [COMPLETED] for k in CONSUMER_KINDS),
        "cross_process",
        f"cross-process consumers {hits}",
    )
    proofs["cross_process"] = {
        "registrar_pid_separate": True,
        "writer_pid_separate": True,
        "fresh_watcher_dispatched": True,
        "child_unblocked": True,
        "consumers_fired": sorted(CONSUMER_KINDS),
        "registrar_stdout": r1.stdout.strip(),
        "writer_stdout": r2.stdout.strip(),
    }

    # Terminals are distinct.
    terminals = {
        proofs["truncated_terminal"],
        proofs["seal_mismatch_terminal"],
        proofs["missing_terminal"],
    }
    _require(len(terminals) == 3, "selftest", f"fail-closed terminals were not distinct: {terminals}")
    proofs["fail_closed_terminals_distinct"] = True
    proofs["n_proofs"] = len(proofs)
    return proofs


def build(*, tmp: Path | None = None) -> Path:
    recovered = recovered_implementation()
    proofs = run_proofs(tmp)
    units = emit_wakeup_workunits()
    callable_block = resident_callable(units)
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "When work finishes, the verifier, the dependency graph and the "
            "relevant frontier wake — not necessarily the model context that "
            "launched it. A receipt on disk is the completion event."
        ),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "watch_receipts_not_processes": True,
        "dispatch_wakes": list(CONSUMER_KINDS),
        "named_frontier": DEFAULT_FRONTIER,
        "fail_closed_terminals": sorted(TERMINAL_FAIL),
        "idempotent": True,
        "no_model_poll": True,
        "poll_aliases_refused": sorted(POLL_ALIASES),
        "vocabulary": {
            "eras": list(ERAS),
            "odysseys": list(ODYSSEYS),
            "no_era_vi": True,
            "no_odyssey_iv": True,
            "fpga_is": "part of Accelerator / Physical Compiler / Fusion, not its own civilization",
            "evidence_classes_we_do_not_emit": ["DIAGNOSTIC_RELATIVE", "PROTECTED_ABSOLUTE"],
            "evidence_class_we_emit": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
        },
        "api": {
            "register_expectation": "persist a path-keyed wait; launching context may then exit",
            "register_consumer": "bind a named callable in the process that is awake",
            "harvest": "supervisor reads expected paths; not a model poll",
            "dispatch": "wake verifier, graph, frontier; at most once per event_id",
            "run_once": "harvest then dispatch",
            "refused": sorted(POLL_ALIASES),
        },
        "proofs": proofs,
        "workunits": units,
        "workunit_ids": [u["id"] for u in units],
        "recovered_implementation": recovered,
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(recovered),
        "resident_callable": callable_block,
        "integration_points": {
            "workgraph.py": "swap Watcher._apply_graph for the landed dependency graph",
            "frontiers.py": "swap the named frontier consumer for the landed frontier refill",
            "detached.py": "supervisor that harvests after a detached worker exits",
            "evidence_dag.py": "completion events are DAG edges once that module lands",
            "resident_api.py": "resident discovers this via --selftest / run_once",
            "qualification_pipeline.py": "SLEEPING units wake on its sealed receipt, never on a synthetic",
        },
        "claim_boundary": (
            "Static sidecar artifact. No hardware measurement. "
            "Neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE. "
            "A missing Metal GPU is SLEEPING work, not a forged COMPLETED."
        ),
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--selftest", action="store_true", help="run proofs and seal RECEIPT_WAKEUP.json")
    ap.add_argument("--build", action="store_true", help="alias for --selftest")
    ap.parse_args(argv)
    out = selftest()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
