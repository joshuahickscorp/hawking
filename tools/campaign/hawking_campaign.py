#!/usr/bin/env python3
"""Restart-safe controller + status ledger for the pre-refactor completion campaign.

One append-only JSONL ledger is the record; HAWKING_CAMPAIGN_STATUS.json is a
derived view rebuilt from it on every write.  Nothing here holds state in memory
across runs, so a crash costs at most the row that was being written.

    python3 tools/campaign/hawking_campaign.py gate <id> <state> [note]
    python3 tools/campaign/hawking_campaign.py spend <cad> [note]
    python3 tools/campaign/hawking_campaign.py job <name> <pid> <logpath>
    python3 tools/campaign/hawking_campaign.py status
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "HAWKING_CAMPAIGN_LEDGER.jsonl"
STATUS = ROOT / "HAWKING_CAMPAIGN_STATUS.json"

# Terminal gates, verbatim from HAWKING_BUDGET_CONSTRAINED_100_PERCENT_CAMPAIGN.md §5.
GATES = {
    "T01": "complete GLM General .gravity artifact exists",
    "T02": "lowest honest GLM frontier is sealed",
    "T03": "GLM runs end to end in HIDE",
    "T04": "Prometheus Claim A is sealed",
    "T05": "training package is complete",
    "T06": "Prometheus model trained, imported, validated, frozen",
    "T07": "Forge/sovereignty/continuity are sealed",
    "T08": "direct Metal complete-token runtime works",
    "T09": "base true TPS and prefill are measured",
    "T10": "acceleration gauntlet is terminal",
    "T11": "accelerated accepted TPS is measured",
    "T12": "General and specialist HIDE tests are green",
    "T13": "source lifecycle and rollback are green",
    "T14": "all campaign work is committed and pushed",
}
STATES = {"OPEN", "IN_PROGRESS", "BLOCKED", "GREEN", "NEGATIVE_SEALED"}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append(row: dict) -> None:
    row["at"] = _now()
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def _rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError) as exc:
        return isinstance(exc, PermissionError)
    return True


def _derive() -> dict:
    rows = _rows()
    gates = {gid: {"desc": desc, "state": "OPEN", "note": "", "at": None}
             for gid, desc in GATES.items()}
    spend, jobs = 0.0, {}
    for row in rows:
        kind = row.get("kind")
        if kind == "gate" and row["gate"] in gates:
            gates[row["gate"]].update(state=row["state"], note=row.get("note", ""),
                                      at=row["at"])
        elif kind == "spend":
            spend += float(row["cad"])
        elif kind == "job":
            jobs[row["name"]] = {"pid": row["pid"], "log": row["log"], "at": row["at"]}
    for job in jobs.values():
        job["alive"] = _alive(int(job["pid"]))
    green = sum(1 for g in gates.values() if g["state"] in ("GREEN", "NEGATIVE_SEALED"))
    return {
        "schema": "hawking.campaign.status.v1",
        "updated_at": _now(),
        "gates_total": len(GATES),
        "gates_closed": green,
        "completion_fraction": round(green / len(GATES), 4),
        "spend_cad": round(spend, 2),
        "gates": gates,
        "detached_jobs": jobs,
        "endpoint": ("HAWKING_PRE_REFACTOR_COMPLETE" if green == len(GATES)
                     else "IN_PROGRESS"),
    }


def _seal() -> dict:
    status = _derive()
    STATUS.write_text(json.dumps(status, indent=1, sort_keys=True) + "\n")
    return status


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "status"
    if cmd == "gate":
        gate, state = argv[2], argv[3].upper()
        if gate not in GATES:
            raise SystemExit(f"unknown gate {gate}; known: {sorted(GATES)}")
        if state not in STATES:
            raise SystemExit(f"unknown state {state}; known: {sorted(STATES)}")
        _append({"kind": "gate", "gate": gate, "state": state,
                 "note": " ".join(argv[4:])})
    elif cmd == "spend":
        _append({"kind": "spend", "cad": float(argv[2]), "note": " ".join(argv[3:])})
    elif cmd == "job":
        _append({"kind": "job", "name": argv[2], "pid": int(argv[3]), "log": argv[4]})
    elif cmd != "status":
        raise SystemExit(__doc__)
    print(json.dumps(_seal(), indent=1, sort_keys=True))
    return 0


def selftest() -> None:
    """Round-trip the ledger in a temp root so the derive path has a real check."""
    import tempfile
    global LEDGER, STATUS
    with tempfile.TemporaryDirectory() as tmp:
        LEDGER, STATUS = Path(tmp) / "l.jsonl", Path(tmp) / "s.json"
        _append({"kind": "gate", "gate": "T01", "state": "GREEN", "note": "x"})
        _append({"kind": "spend", "cad": 12.5, "note": ""})
        _append({"kind": "gate", "gate": "T01", "state": "BLOCKED", "note": "y"})
        _append({"kind": "job", "name": "fetch", "pid": os.getpid(), "log": "/dev/null"})
        s = _seal()
        assert s["gates"]["T01"]["state"] == "BLOCKED", "last write must win"
        assert s["gates_closed"] == 0 and s["spend_cad"] == 12.5, s
        assert s["detached_jobs"]["fetch"]["alive"] is True
        assert json.loads(STATUS.read_text())["endpoint"] == "IN_PROGRESS"
        print("selftest PASS")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        raise SystemExit(main(sys.argv))
