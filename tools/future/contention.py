"""The contention recorder (G014).

G014 requires "a recorded decision EACH TIME lanes are paused/reaped for the
resident, with the guard reading that forced it". Four such decisions exist on
disk and every one of them was typed by hand: nothing in the repository writes
CONTENTION_DECISIONS.jsonl, and autonomy_ratio.py only reads it. "Each time" is
therefore enforced by memory, which is not enforcement -- a pause that nobody
remembered to write down leaves no trace, and the record cannot be distinguished
from a complete one.

So the decision is taken HERE, against a live guard reading, and recording is
not a separate step a caller can forget:

    decide("superwave-2", expected_gb=8.0, lanes=20)

returns ALLOW or PAUSE and has already written the entry, with the exact
snapshot that forced it. A decision cannot be recorded without a guard reading;
`record` refuses one that carries no measured state.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "receipts" / "future" / "CONTENTION_DECISIONS.jsonl"

sys.path.insert(0, str(REPO / "tools" / "future"))
import campaign_memory_guard as guard  # noqa: E402

# The authoritative experiment is the autonomy round. Everything else yields to
# it -- that is the whole content of the obligation, so it is a constant here
# rather than a caller's argument.
AUTHORITATIVE = "the HCLI autonomy round (the resident)"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record(action: str, subject: str, why: str, snapshot: Any) -> Dict[str, Any]:
    """Append one decision. Refuses without a real guard reading.

    A decision with no measurement behind it is prose, and prose is what this
    log already had four of.
    """
    if snapshot is None:
        raise ValueError("refusing to record a contention decision with no guard reading")
    g = snapshot.as_dict() if hasattr(snapshot, "as_dict") else dict(snapshot)
    if "state" not in g:
        raise ValueError("refusing to record a guard reading with no measured state")
    entry = {"ts": _now(), "action": action, "subject": subject, "why": why, "guard": g}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def decide(subject: str, expected_gb: float = 0.0, lanes: int = 0,
           _snapshot: Optional[Any] = None) -> Dict[str, Any]:
    """Take AND record the contention decision in one call.

    `_snapshot` exists so the PAUSE path can be exercised against an injected
    guard state. Without it there is no way to test the branch that matters:
    the host is healthy almost all of the time, so a test that waits for real
    pressure is a test that never runs.
    """
    snap = _snapshot if _snapshot is not None else guard.sample(expected_gb=expected_gb)
    state = getattr(snap, "state", None) or dict(snap).get("state")

    if state == "OK":
        action = "allowed"
        why = (f"guard {state} with {getattr(snap, 'free_gb', '?')} GB free and "
               f"{getattr(snap, 'headroom_gb', '?')} GB headroom against {expected_gb} GB "
               f"expected, so {lanes} lane(s) run CONCURRENTLY with {AUTHORITATIVE}. "
               f"The Odyssey does not wait idle on the streak.")
    else:
        action = "paused"
        why = (f"guard {state}: {', '.join(getattr(snap, 'reasons', ()) or ()) or 'no reason given'}. "
               f"{lanes} lane(s) YIELD to {AUTHORITATIVE}. The authoritative experiment wins; "
               f"lowest-information lanes shed first.")
    entry = record(action, subject, why, snap)
    entry["allow"] = action == "allowed"
    return entry


def _selftest() -> list:
    """Both branches, and the refusal. The PAUSE branch is the one that matters."""
    fails = []
    before = LOG.read_text().count("\n") if LOG.exists() else 0

    ok = guard.Snapshot(free_gb=60.0, compressor_gb=0.0, swapfiles=0, wired_gb=5.0,
                        state="OK", reasons=(), headroom_gb=52.0)
    d = decide("SELFTEST-allow", expected_gb=8.0, lanes=20, _snapshot=ok)
    if not d["allow"] or d["action"] != "allowed":
        fails.append("healthy guard did not produce an allow decision")

    stop = guard.Snapshot(free_gb=3.0, compressor_gb=12.0, swapfiles=9, wired_gb=70.0,
                          state="STOP", reasons=("free 3.0 GB below floor",
                                                 "swap 28.4 GB near the 30 GB ceiling"),
                          headroom_gb=-5.0)
    d2 = decide("SELFTEST-pause", expected_gb=8.0, lanes=20, _snapshot=stop)
    if d2["allow"] or d2["action"] != "paused":
        fails.append("STOP guard did not produce a pause decision -- the branch G014 is about")
    if "STOP" not in d2["why"] or "swap 28.4" not in d2["why"]:
        fails.append("pause decision does not carry the guard reading that forced it")
    if d2["guard"]["state"] != "STOP":
        fails.append("recorded snapshot does not match the state that forced the decision")

    try:
        record("paused", "SELFTEST-no-guard", "because I said so", None)
        fails.append("recorded a decision with NO guard reading -- prose passed as evidence")
    except ValueError:
        pass

    after = LOG.read_text().count("\n")
    if after - before != 2:
        fails.append(f"expected exactly 2 new entries, got {after - before}")

    # Leave the log as it was found: a selftest must not pollute the record it
    # is testing. Rewrite without the SELFTEST lines.
    kept = [l for l in LOG.read_text().splitlines(keepends=True)
            if '"SELFTEST-' not in l]
    LOG.write_text("".join(kept))
    if LOG.read_text().count("\n") != before:
        fails.append("selftest did not restore the decision log to its prior length")
    return fails


if __name__ == "__main__":
    fails = _selftest()
    for f in fails:
        print("SELFTEST FAIL:", f)
    print(f"selftest: {'FAILED' if fails else 'PASS'} — allow, pause, and the no-guard refusal")
    raise SystemExit(1 if fails else 0)
