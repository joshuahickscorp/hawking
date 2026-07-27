"""The Ledger: append-only, correction by supersession, never by edit.

`odyssey/ledger/LEDGER_CONTRACT.json` states the law -- "what is not recorded did not
happen; what is recorded cannot be revised" -- and a correction policy: a wrong row is
superseded by a new row that names it; rows are never edited or removed.

This makes that enforceable rather than declared.  The store exposes no update and no
delete.  `supersede()` writes a NEW row pointing at the old one, so the mistake and its
correction are both permanently visible.  That is the point: a research system that can
quietly rewrite its own history cannot be audited, and a system that cannot be audited
cannot be trusted about its own failures.

Staged in the Hawking repository; migrates to the Ramanujan repository at
HAWKING_EVOLUTION_COMPLETE per RAMANUJAN_HANDOFF_CONTRACT.json.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# From LEDGER_CONTRACT.json. A row whose kind is not here is rejected: an open
# vocabulary would let any subsystem invent a category to slip past review.
EVENT_KINDS = frozenset(
    {
        "claim",
        "objection",
        "formalization",
        "proof_attempt",
        "verifier_event",
        "tribunal_decision",
        "literature_query",
        "budget_grant",
        "sovereignty_event",
        "sandbox_event",
        "checkpoint",
        "rollback",
        "supersession",
    }
)


class LedgerViolation(RuntimeError):
    """Raised rather than returned, so a caller cannot ignore it by accident."""


@dataclass
class Row:
    seq: int
    kind: str
    payload: dict
    actor: str
    at: str
    prev_hash: str
    row_hash: str
    supersedes: int | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "seq": self.seq,
                "kind": self.kind,
                "payload": self.payload,
                "actor": self.actor,
                "at": self.at,
                "prev_hash": self.prev_hash,
                "row_hash": self.row_hash,
                "supersedes": self.supersedes,
            },
            sort_keys=True,
        )


def _hash_row(seq: int, kind: str, payload: dict, actor: str, at: str, prev: str) -> str:
    body = json.dumps(
        {"seq": seq, "kind": kind, "payload": payload, "actor": actor, "at": at, "prev_hash": prev},
        sort_keys=True,
    )
    return hashlib.sha256(body.encode()).hexdigest()


@dataclass
class Ledger:
    """A hash-chained append-only log.

    The chain is what makes tampering detectable rather than merely forbidden. Editing a
    row in place breaks every subsequent `prev_hash`, and `verify_chain()` says exactly
    where.
    """

    path: Path
    clock: Any = None  # callable returning an ISO timestamp; injected so tests are deterministic
    _rows: list[Row] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.exists():
            self._rows = [self._row_from(json.loads(l)) for l in self.path.read_text().splitlines() if l.strip()]

    @staticmethod
    def _row_from(d: dict) -> Row:
        return Row(
            seq=d["seq"],
            kind=d["kind"],
            payload=d["payload"],
            actor=d["actor"],
            at=d["at"],
            prev_hash=d["prev_hash"],
            row_hash=d["row_hash"],
            supersedes=d.get("supersedes"),
        )

    def _now(self) -> str:
        if self.clock is not None:
            return self.clock()
        import time

        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def append(self, kind: str, payload: dict, actor: str, supersedes: int | None = None) -> Row:
        if kind not in EVENT_KINDS:
            raise LedgerViolation(
                f"unknown event kind {kind!r}. The vocabulary is closed on purpose: an open one "
                f"lets a subsystem invent a category to avoid review. Known: {sorted(EVENT_KINDS)}"
            )
        seq = len(self._rows)
        prev = self._rows[-1].row_hash if self._rows else "0" * 64
        at = self._now()
        rh = _hash_row(seq, kind, payload, actor, at, prev)
        row = Row(seq, kind, payload, actor, at, prev, rh, supersedes)
        self._rows.append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(row.to_json() + "\n")
        return row

    def supersede(self, seq: int, reason: str, actor: str, corrected: dict) -> Row:
        """Correct an earlier row by writing a new one that names it.

        There is deliberately no `edit` and no `delete`. Both rows stay readable forever.
        """
        if not 0 <= seq < len(self._rows):
            raise LedgerViolation(f"cannot supersede seq {seq}: no such row")
        return self.append(
            "supersession",
            {"supersedes_seq": seq, "reason": reason, "corrected": corrected},
            actor,
            supersedes=seq,
        )

    def superseded_seqs(self) -> set[int]:
        return {r.supersedes for r in self._rows if r.supersedes is not None}

    def live_rows(self) -> Iterator[Row]:
        """Rows not superseded by a later one. Superseded rows still exist and are readable."""
        dead = self.superseded_seqs()
        return (r for r in self._rows if r.seq not in dead)

    def rows(self) -> list[Row]:
        return list(self._rows)

    def verify_chain(self) -> tuple[bool, str]:
        """Recompute every hash. Returns (ok, message naming the first break)."""
        prev = "0" * 64
        for r in self._rows:
            if r.prev_hash != prev:
                return False, f"row {r.seq}: prev_hash mismatch (chain broken before this row)"
            expect = _hash_row(r.seq, r.kind, r.payload, r.actor, r.at, r.prev_hash)
            if expect != r.row_hash:
                return False, f"row {r.seq}: contents were edited in place"
            prev = r.row_hash
        return True, f"chain intact over {len(self._rows)} rows"
