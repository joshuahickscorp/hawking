#!/usr/bin/env python3
"""Ascension Knowledge Plane store: append-only genomes + negative science.

The contract at
  workspace/campaign/records/ascension-sandbox/lifecycle/ASCENSION_V3_KNOWLEDGE_PLANE_CONTRACT.json
specifies the record shapes but its own status is CONTROLLER_CONFIGURATION_ONLY, and its
own claim_boundary says `configuration_is_not_knowledge_plane_evidence`. This module is
the store that makes it evidence.

Field names are read FROM the contract rather than duplicated here, so a contract change
cannot silently diverge from what is written.

Two rules from the contract are enforced mechanically, not by convention:
  negative_science_is_not_deleted_or_silently_ignored
      -> the store is append-only; there is no delete path.
  negative_science_is_retrieved_before_repeat_experiment
      -> `recall` exists so a lane can check before spending machine time.

Usage:
    knowledge_plane.py append <kind> <record.json>
    knowledge_plane.py recall <term> [<term> ...]
    knowledge_plane.py index          # rebuild the sqlite mechanism index
    knowledge_plane.py verify         # every record validates against the contract
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SANDBOX = REPO / "workspace/campaign/records/ascension-sandbox"
CONTRACT = SANDBOX / "lifecycle/ASCENSION_V3_KNOWLEDGE_PLANE_CONTRACT.json"
STORE = SANDBOX / "knowledge-plane"
INDEX = STORE / "ASCENSION_MECHANISM_INDEX.sqlite"

# contract output name -> file basename
KINDS = {
    "kernel": "ASCENSION_KERNEL_GENOME",
    "representation": "ASCENSION_REPRESENTATION_GENOME",
    "scheduler": "ASCENSION_SCHEDULER_GENOME",
    "negative": "ASCENSION_NEGATIVE_SCIENCE",
    "transfer": "ASCENSION_TRANSFER_MATRIX",
}


def contract() -> dict:
    return json.loads(CONTRACT.read_text())


def fields_for(kind: str) -> list[str]:
    name = KINDS[kind]
    spec = contract()["outputs"][name]
    return list(spec["fields"])


def path_for(kind: str) -> Path:
    return STORE / f"{KINDS[kind]}.jsonl"


def _digest(record: dict) -> str:
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def append(kind: str, record: dict) -> str:
    """Append one record. Unknown or missing fields are refused, not coerced."""
    allowed = set(fields_for(kind))
    got = set(record)
    if got - allowed:
        raise SystemExit(f"{kind}: unknown fields {sorted(got - allowed)}")
    if allowed - got:
        raise SystemExit(f"{kind}: missing fields {sorted(allowed - got)}")
    STORE.mkdir(parents=True, exist_ok=True)
    digest = _digest(record)
    line = json.dumps({**record, "record_sha256": digest}, sort_keys=True)
    target = path_for(kind)
    # Idempotent: the same record appended twice stays one row.
    if target.exists() and digest in target.read_text():
        return digest
    with target.open("a") as handle:
        handle.write(line + "\n")
    return digest


def read(kind: str) -> list[dict]:
    target = path_for(kind)
    if not target.exists():
        return []
    return [json.loads(line) for line in target.read_text().splitlines() if line.strip()]


def recall(terms: list[str]) -> list[tuple[str, dict]]:
    """Retrieve prior science matching any term. Call this BEFORE spending machine time."""
    hits = []
    lowered = [t.lower() for t in terms]
    for kind in KINDS:
        for record in read(kind):
            blob = json.dumps(record).lower()
            if any(t in blob for t in lowered):
                hits.append((kind, record))
    return hits


def build_index() -> int:
    """sqlite mechanism index; the contract requires its hash be bound by the transfer matrix."""
    STORE.mkdir(parents=True, exist_ok=True)
    if INDEX.exists():
        INDEX.unlink()
    connection = sqlite3.connect(INDEX)
    connection.execute(
        "CREATE TABLE mechanism (kind TEXT, key TEXT, record_sha256 TEXT, body TEXT)"
    )
    total = 0
    for kind in KINDS:
        for record in read(kind):
            key = (
                record.get("operator")
                or record.get("tensor_or_organ")
                or record.get("task_class")
                or record.get("mechanism")
                or "?"
            )
            connection.execute(
                "INSERT INTO mechanism VALUES (?,?,?,?)",
                (kind, key, record.get("record_sha256", ""), json.dumps(record, sort_keys=True)),
            )
            total += 1
    connection.commit()
    connection.close()
    return total


def verify() -> int:
    """Every stored record must still validate against the contract's current fields."""
    problems = 0
    for kind in KINDS:
        allowed = set(fields_for(kind)) | {"record_sha256"}
        for record in read(kind):
            stored = record.get("record_sha256")
            body = {k: v for k, v in record.items() if k != "record_sha256"}
            if set(record) - allowed:
                print(f"FAIL {kind}: unknown fields {sorted(set(record) - allowed)}")
                problems += 1
            if stored != _digest(body):
                print(f"FAIL {kind}: record_sha256 does not match body for {body.get('mechanism') or body.get('operator')}")
                problems += 1
    return problems


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    command = argv[1]
    if command == "append":
        kind, source = argv[2], Path(argv[3])
        payload = json.loads(source.read_text())
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            print(append(kind, record))
        return 0
    if command == "recall":
        hits = recall(argv[2:])
        for kind, record in hits:
            key = record.get("mechanism") or record.get("operator") or record.get("tensor_or_organ")
            print(f"[{kind}] {key}: {record.get('measured_outcome') or record.get('kernel_grammar') or ''}")
            if record.get("reopen_condition"):
                print(f"    reopen: {record['reopen_condition']}")
        print(f"{len(hits)} record(s)")
        return 0
    if command == "index":
        print(f"indexed {build_index()} records into {INDEX}")
        return 0
    if command == "verify":
        problems = verify()
        print("OK" if problems == 0 else f"{problems} problem(s)")
        return 1 if problems else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
