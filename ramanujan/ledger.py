"""The research Ledger: append-only, hash-chained, corrected only by supersession.

`odyssey/ledger/LEDGER_CONTRACT.json` states the law this module enforces:

    what is not recorded did not happen; what is recorded cannot be revised

So the only write operation is `append`. There is deliberately no update and no
delete. A wrong row is corrected by appending a new row that *names* it, which
leaves both visible forever: the mistake and the correction. A ledger that lets
you quietly fix yesterday is a ledger that cannot be used as evidence.

The chaining discipline mirrors `HashChainLog` in the GLM controller, which is
proven in use: every row carries its sequence, the hash of the row before it,
and its own chain hash, so a deletion or an edit anywhere breaks verification at
that point rather than passing silently. It is reimplemented rather than
imported because that class lives inside a 3000-line GLM-specific module, and
Ramanujan should not depend on the GLM controller to record a claim.

Verification is whole-chain and refuses partial credit: `verify` walks every row
from genesis and raises on the first break.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64

# From the contract. A kind outside this set is refused rather than recorded,
# because an unrecognised kind is how a ledger stops being a ledger.
KINDS = frozenset({
    "claim", "objection", "formalization", "proof_attempt", "verifier_event",
    "tribunal_decision", "literature_query", "budget_grant", "sovereignty_event",
    "sandbox_event", "checkpoint", "rollback",
})
REQUIRED = ("id", "at", "kind", "role", "parents", "payload_sha256")
_ROW_KEYS = frozenset({*REQUIRED, "seq", "prev_hash", "chain_sha256", "supersedes"})


class LedgerError(RuntimeError):
    """A write that would break the law, or a chain that already is broken."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Row:
    seq: int
    id: str
    at: str
    kind: str
    role: str
    parents: list[str]
    payload_sha256: str
    prev_hash: str
    chain_sha256: str
    supersedes: str | None = None


class Ledger:
    """Append-only JSONL. One file, one chain, no in-place edits."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # -- reading -------------------------------------------------------------

    def rows(self) -> list[dict[str, Any]]:
        """Every row, verified. Raises on the first break rather than skipping it."""
        if not self.path.exists():
            return []
        raw = self.path.read_text()
        if raw and not raw.endswith("\n"):
            raise LedgerError(f"torn tail in {self.path}: last row is unterminated")
        out: list[dict[str, Any]] = []
        prev = GENESIS
        for seq, line in enumerate(l for l in raw.splitlines() if l.strip()):
            row = json.loads(line)
            if set(row) - _ROW_KEYS or not set(REQUIRED) <= set(row):
                raise LedgerError(f"row {seq} has unexpected or missing fields")
            if row["seq"] != seq:
                raise LedgerError(f"sequence break at row {seq}: recorded {row['seq']}")
            if row["prev_hash"] != prev:
                raise LedgerError(f"chain break at row {seq}: a row was edited or removed")
            if row["chain_sha256"] != self._chain_hash(row):
                raise LedgerError(f"row {seq} hash does not match its content")
            out.append(row)
            prev = row["chain_sha256"]
        return out

    @staticmethod
    def _chain_hash(row: dict[str, Any]) -> str:
        body = {k: row[k] for k in sorted(_ROW_KEYS & set(row)) if k != "chain_sha256"}
        return _sha(_canonical(body))

    def verify(self) -> int:
        """Return the row count, or raise. Cheap enough to call in a gate."""
        return len(self.rows())

    # -- writing -------------------------------------------------------------

    def append(
        self,
        *,
        kind: str,
        role: str,
        payload: Any,
        id: str,
        parents: list[str] | None = None,
        supersedes: str | None = None,
    ) -> Row:
        if kind not in KINDS:
            raise LedgerError(f"unknown event kind {kind!r}; the contract names {sorted(KINDS)}")
        existing = self.rows()
        if any(r["id"] == id for r in existing):
            raise LedgerError(f"id {id!r} is already recorded; correct it by superseding, not by reusing the id")
        if supersedes is not None and not any(r["id"] == supersedes for r in existing):
            raise LedgerError(f"cannot supersede {supersedes!r}: no such row")
        row: dict[str, Any] = {
            "seq": len(existing),
            "id": id,
            "at": _now(),
            "kind": kind,
            "role": role,
            "parents": list(parents or []),
            "payload_sha256": _sha(_canonical(payload)),
            "prev_hash": existing[-1]["chain_sha256"] if existing else GENESIS,
        }
        if supersedes is not None:
            row["supersedes"] = supersedes
        row["chain_sha256"] = self._chain_hash(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        return Row(**row) if "supersedes" in row else Row(**row, supersedes=None)

    # -- reading the corrected view -----------------------------------------

    def current(self) -> list[dict[str, Any]]:
        """Rows that nothing later supersedes. The superseded ones stay in `rows`."""
        rows = self.rows()
        dead = {r["supersedes"] for r in rows if r.get("supersedes")}
        return [r for r in rows if r["id"] not in dead]


def demo() -> None:
    """Self-check: the law holds against the three ways it could be broken."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        led = Ledger(Path(d) / "L.jsonl")
        led.append(kind="claim", role="researcher", payload={"x": 1}, id="c1")
        led.append(kind="objection", role="critic", payload={"why": "n=1"}, id="o1", parents=["c1"])
        assert led.verify() == 2

        # A wrong row is corrected by supersession, and both stay visible.
        led.append(kind="claim", role="researcher", payload={"x": 2}, id="c2", supersedes="c1")
        assert led.verify() == 3
        assert {r["id"] for r in led.current()} == {"o1", "c2"}, "superseded row still current"
        assert len(led.rows()) == 3, "supersession must not remove history"

        # An unknown kind is refused.
        try:
            led.append(kind="vibes", role="researcher", payload={}, id="x1")
            raise AssertionError("unknown kind was accepted")
        except LedgerError:
            pass

        # Reusing an id is refused.
        try:
            led.append(kind="claim", role="researcher", payload={}, id="c1")
            raise AssertionError("duplicate id was accepted")
        except LedgerError:
            pass

        # Editing a row in place breaks verification.
        lines = led.path.read_text().splitlines()
        row = json.loads(lines[0]); row["role"] = "tamperer"
        lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
        led.path.write_text("\n".join(lines) + "\n")
        try:
            led.verify()
            raise AssertionError("an edited row verified")
        except LedgerError:
            pass
    print("ok: append-only, supersede-not-edit, unknown kind refused, tamper detected")


if __name__ == "__main__":
    demo()
