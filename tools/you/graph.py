#!/usr/bin/env python3.12
"""The YOU personal context graph: local-first, scoped, inspectable, deletable.

This is the schema every other YOU subsystem binds to, so its invariants matter more than
its breadth.  Three are enforced here rather than described:

  1. Every record carries a SCOPE, and connector-scoped content never becomes global
     without an explicit recorded promotion.  A personal assistant that quietly promotes
     what it read in your email to a global fact about you is the failure mode.

  2. `forget` is real deletion for user-scoped data, not a tombstone.  "No hidden permanent
     memory" is only true if forgetting actually removes.

  3. Every record is reachable by `inspect`.  A record that exists and cannot be listed is
     hidden memory whatever it is called.

Local-first: plain SQLite, no service, no network.  Runs under LIGHT_ONLY.

    python3.12 -m tools.you.graph --demo
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

ENTITIES = (
    "person", "organization", "project", "goal", "commitment", "event", "document",
    "message", "file", "place", "preference", "decision", "task", "idea", "artifact",
    "account",
)

RELATIONS = (
    "belongs_to", "depends_on", "mentions", "created_by", "scheduled_for", "supersedes",
    "contradicts", "supports", "derived_from", "shared_with", "requires_action",
)

# Orthogonal to the six memory classes: a record has exactly one class and one scope.
SCOPES = (
    "global", "workspace", "project", "conversation", "connector", "person",
    "private_vault", "ephemeral",
)

SENSITIVITY = ("public", "personal", "sensitive", "secret")


class GraphViolation(RuntimeError):
    pass


@dataclass(frozen=True)
class Provenance:
    """Where a record came from. Non-optional: a record without provenance cannot be
    audited, corrected or trusted, and every control below depends on knowing the source."""

    source: str          # "user" | "connector:<id>" | "research" | "inference" | "import"
    captured_at: str
    confidence: float    # 0..1
    supersedes: str | None = None

    def validate(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise GraphViolation("confidence must be a probability")
        if not self.source:
            raise GraphViolation("provenance requires a source; a record with none cannot be audited")


@dataclass
class Record:
    kind: str
    body: dict
    scope: str
    sensitivity: str = "personal"
    retention_days: int | None = None
    pinned: bool = False
    prov: Provenance | None = None

    @property
    def rid(self) -> str:
        blob = json.dumps({"kind": self.kind, "body": self.body, "scope": self.scope}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
  rid TEXT PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL, scope TEXT NOT NULL,
  sensitivity TEXT NOT NULL, retention_days INTEGER, pinned INTEGER NOT NULL DEFAULT 0,
  source TEXT NOT NULL, captured_at TEXT NOT NULL, confidence REAL NOT NULL,
  supersedes TEXT, expired INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS relations (
  src TEXT NOT NULL, rel TEXT NOT NULL, dst TEXT NOT NULL,
  PRIMARY KEY (src, rel, dst)
);
CREATE TABLE IF NOT EXISTS promotions (
  rid TEXT NOT NULL, from_scope TEXT NOT NULL, to_scope TEXT NOT NULL,
  at TEXT NOT NULL, approved_by TEXT NOT NULL
);
"""


@dataclass
class PersonalGraph:
    path: Path
    clock: object = None
    _db: sqlite3.Connection = field(init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.executescript(SCHEMA)
        self._db.commit()

    def _now(self) -> str:
        if self.clock:
            return self.clock()  # type: ignore[operator]
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # -- write ------------------------------------------------------------
    def add(self, r: Record) -> str:
        if r.kind not in ENTITIES:
            raise GraphViolation(f"unknown entity kind {r.kind!r}; the vocabulary is closed")
        if r.scope not in SCOPES:
            raise GraphViolation(f"unknown scope {r.scope!r}")
        if r.sensitivity not in SENSITIVITY:
            raise GraphViolation(f"unknown sensitivity {r.sensitivity!r}")
        if r.prov is None:
            raise GraphViolation("every record requires provenance")
        r.prov.validate()
        self._db.execute(
            "INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,0)",
            (r.rid, r.kind, json.dumps(r.body, sort_keys=True), r.scope, r.sensitivity,
             r.retention_days, int(r.pinned), r.prov.source, r.prov.captured_at,
             r.prov.confidence, r.prov.supersedes),
        )
        self._db.commit()
        return r.rid

    def relate(self, src: str, rel: str, dst: str) -> None:
        if rel not in RELATIONS:
            raise GraphViolation(f"unknown relation {rel!r}")
        self._db.execute("INSERT OR IGNORE INTO relations VALUES (?,?,?)", (src, rel, dst))
        self._db.commit()

    # -- the user controls -------------------------------------------------
    def inspect(self, scope: str | None = None) -> list[dict]:
        """Everything, or everything in one scope. No record is unreachable from here."""
        q = "SELECT rid,kind,body,scope,sensitivity,retention_days,pinned,source,captured_at,confidence,expired FROM records"
        rows = self._db.execute(q + (" WHERE scope=?" if scope else ""),
                                (scope,) if scope else ()).fetchall()
        cols = ["rid", "kind", "body", "scope", "sensitivity", "retention_days", "pinned",
                "source", "captured_at", "confidence", "expired"]
        return [dict(zip(cols, r)) for r in rows]

    def correct(self, rid: str, new_body: dict, by: str) -> str:
        """Correction is supersession: the new record names the old, both remain visible
        until the old is forgotten. Editing in place would erase the fact of the mistake."""
        old = self._db.execute("SELECT kind,scope,sensitivity FROM records WHERE rid=?", (rid,)).fetchone()
        if not old:
            raise GraphViolation(f"no record {rid}")
        kind, scope, sens = old
        r = Record(kind=kind, body=new_body, scope=scope, sensitivity=sens,
                   prov=Provenance(source=by, captured_at=self._now(), confidence=1.0, supersedes=rid))
        return self.add(r)

    def pin(self, rid: str, pinned: bool = True) -> None:
        """Pinned records are exempt from expiry, never from forget."""
        self._db.execute("UPDATE records SET pinned=? WHERE rid=?", (int(pinned), rid))
        self._db.commit()

    def promote_scope(self, rid: str, to_scope: str, approved_by: str) -> None:
        """The explicit, recorded transition. Connector content cannot reach global any
        other way, and the approval is stored so it can be audited later."""
        row = self._db.execute("SELECT scope FROM records WHERE rid=?", (rid,)).fetchone()
        if not row:
            raise GraphViolation(f"no record {rid}")
        frm = row[0]
        if to_scope not in SCOPES:
            raise GraphViolation(f"unknown scope {to_scope!r}")
        if not approved_by:
            raise GraphViolation("scope promotion requires an explicit approver")
        self._db.execute("UPDATE records SET scope=? WHERE rid=?", (to_scope, rid))
        self._db.execute("INSERT INTO promotions VALUES (?,?,?,?,?)",
                         (rid, frm, to_scope, self._now(), approved_by))
        self._db.commit()

    def expire_due(self, now_days: float) -> int:
        """Retention expiry. Pinned records survive; everything else with an elapsed
        retention is marked expired and leaves the working set."""
        n = 0
        for row in self._db.execute(
            "SELECT rid,retention_days,pinned,captured_at FROM records WHERE expired=0"
        ).fetchall():
            rid, ret, pinned, _cap = row
            if ret is not None and not pinned and now_days >= ret:
                self._db.execute("UPDATE records SET expired=1 WHERE rid=?", (rid,))
                n += 1
        self._db.commit()
        return n

    def forget(self, rid: str) -> bool:
        """Real deletion. Not a tombstone, not an expired flag.

        'No hidden permanent memory' is only true if forgetting removes. Relations
        referencing the record go with it, so a forgotten person does not survive as a
        dangling edge.
        """
        cur = self._db.execute("DELETE FROM records WHERE rid=?", (rid,))
        self._db.execute("DELETE FROM relations WHERE src=? OR dst=?", (rid, rid))
        self._db.commit()
        return cur.rowcount > 0

    def forget_scope(self, scope: str) -> int:
        rids = [r[0] for r in self._db.execute("SELECT rid FROM records WHERE scope=?", (scope,)).fetchall()]
        for rid in rids:
            self.forget(rid)
        return len(rids)

    def export(self) -> dict:
        """Portable, complete, and readable without this tool. The user owns the graph."""
        return {
            "schema": "hide.you.personal_graph.export.v1",
            "at": self._now(),
            "records": self.inspect(),
            "relations": [dict(zip(("src", "rel", "dst"), r))
                          for r in self._db.execute("SELECT src,rel,dst FROM relations").fetchall()],
            "promotions": [dict(zip(("rid", "from_scope", "to_scope", "at", "approved_by"), r))
                           for r in self._db.execute("SELECT * FROM promotions").fetchall()],
        }

    def active(self) -> list[dict]:
        return [r for r in self.inspect() if not r["expired"]]


def _demo() -> int:
    import tempfile
    g = PersonalGraph(Path(tempfile.mkdtemp()) / "you.db")
    p = Provenance(source="connector:gmail", captured_at="2026-07-27T00:00:00Z", confidence=0.8)
    rid = g.add(Record("person", {"name": "A. Colleague"}, scope="connector", prov=p))
    print("added connector-scoped record:", rid)
    print("in global scope:", len(g.inspect("global")), "-- connector content is not global")
    g.promote_scope(rid, "global", approved_by="user")
    print("after explicit promotion, global:", len(g.inspect("global")))
    print("forgot:", g.forget(rid), "| remaining:", len(g.inspect()))
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo() if "--demo" in sys.argv else 0)
