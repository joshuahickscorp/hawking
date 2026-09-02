#!/usr/bin/env python3
"""G011 producer: did Odyssey streams I/II/III overlap, driven by HCLI?

Run it:  python3 tools/sovereign/g011_streaming.py

That argv shape is deliberate. ``shell.exec`` in hcli/tool_registry.py accepts
``python3 <path>`` (no ``-c``, no ``-m`` outside pytest/unittest/compileall) at
mutation class ``reversible_runtime``, which the resident's default registry
grants -- so the resident can run this producer itself today, with no new tool
and no change to the permission set.

WHAT THE THREE STREAMS ARE (established from source, not invented here).
hcli/odyssey.py is HCLI's own Odyssey campaign ledger, written only by that
module into workspace/campaign/odyssey/HCLI_LEDGER.json:

    I    laws               record_law()               discovery
    II   transfer_probes    create_transfer_probe()    does a law transfer?
    III  adversarial_probes create_adversarial_probe() does a law survive attack?

Its docstring states the rule this gate measures: II and III require only that
a law has been *claimed*, "so I/II/III can overlap the moment a law is claimed,
per the campaign's no-phase-barrier rule". A stream's wall span is therefore
[first recorded_at, last recorded_at] over its own entries -- real event times
the resident wrote as it worked, never a timer this producer started.

WHAT MAKES ``hcli_owned`` TRUE, AND WHY IT CANNOT BE ASSERTED.
Two independent facts, both checked at run time, both false for a human shell:

  1. This process shares the live resident worker's session id. The worker is
     spawned with ``start_new_session=True`` (hcli/agentos/resident.py), so it
     leads its own session and only its descendants share it. A terminal gets
     a different sid. The worker pid is additionally matched against its
     recorded start token so a recycled pid does not pass.
  2. Every stream entry carries ``hcli_owned: true``, stamped by
     hcli/odyssey.py at the moment the entry was appended. A ledger typed in
     by hand carries no stamp and fails here.

REFUSAL IS THE DEFAULT. If any required fact is missing this writes NOTHING,
prints the reasons, and exits 1. A red gate is an open question; a fabricated
receipt is a false answer, which is worse.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LEDGER = REPO / "workspace" / "campaign" / "odyssey" / "HCLI_LEDGER.json"
ODYSSEY_STATE = REPO / "workspace" / "campaign" / "odyssey" / "ODYSSEY_STATE.json"
RESIDENT_STATE = REPO / ".hcli" / "resident" / "state.json"
RECEIPT = REPO / "receipts" / "sovereign" / "G011_odyssey_streaming.json"

PRODUCED_BY = "tools/sovereign/g011_streaming.py"
COMMAND = "python3 tools/sovereign/g011_streaming.py"

# stream -> the ledger list that stream writes into
STREAMS = {"I": "laws", "II": "transfer_probes", "III": "adversarial_probes"}


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _epoch(iso: object) -> float | None:
    """ISO-8601 recorded_at -> epoch seconds. hcli/odyssey.py writes UTC."""
    if not isinstance(iso, str) or not iso.strip():
        return None
    try:
        stamp = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def running_under_resident_worker() -> tuple[bool, dict]:
    """True only inside the live resident worker's session.

    Not a claim about intent: a shell run lands in the terminal's session and
    returns False here, which is exactly what stops this producer from ever
    minting ``hcli_owned: true`` on my behalf.
    """
    detail: dict = {"basis": "session id of the live resident worker (start_new_session=True)"}
    state = _load_json(RESIDENT_STATE)
    if not isinstance(state, dict):
        return False, {**detail, "reason": f"{RESIDENT_STATE} is absent or unreadable"}
    pid = state.get("worker_pid")
    detail["worker_pid"] = pid
    if not isinstance(pid, int) or pid <= 0:
        return False, {**detail, "reason": "resident records no live worker pid"}
    try:
        from hcli.resources import process_start_token
    except ImportError as exc:  # pragma: no cover - repo layout is fixed
        return False, {**detail, "reason": f"cannot import hcli.resources: {exc}"}
    observed = process_start_token(pid)
    if observed != state.get("worker_start_token"):
        return False, {**detail, "reason": "worker start token does not match; pid was recycled or the worker is gone"}
    try:
        mine, theirs = os.getsid(0), os.getsid(pid)
    except (OSError, ProcessLookupError) as exc:
        return False, {**detail, "reason": f"session id unavailable: {exc}"}
    detail.update(producer_sid=mine, worker_sid=theirs, producer_pid=os.getpid())
    if mine != theirs:
        return False, {**detail, "reason": "producer runs outside the resident worker's session (a human shell)"}
    return True, detail


def stream_spans(ledger: dict) -> tuple[dict, list[str]]:
    """Wall span per stream from the entries' own recorded_at timestamps."""
    spans: dict[str, dict] = {}
    problems: list[str] = []
    for name, key in STREAMS.items():
        entries = ledger.get(key)
        if not isinstance(entries, list) or not entries:
            problems.append(f"stream {name} ({key}) has no entries; it never ran")
            continue
        stamps = [t for t in (_epoch(e.get("recorded_at")) for e in entries if isinstance(e, dict)) if t]
        if not stamps:
            problems.append(f"stream {name} ({key}) has entries but no usable recorded_at")
            continue
        unstamped = [e for e in entries if isinstance(e, dict) and e.get("hcli_owned") is not True]
        if unstamped:
            problems.append(
                f"stream {name} ({key}) has {len(unstamped)}/{len(entries)} entries not stamped "
                "hcli_owned by hcli/odyssey.py; they were not written by the resident"
            )
        spans[name] = {
            "ledger_key": key,
            "entry_count": len(entries),
            "start_s": min(stamps),
            "end_s": max(stamps),
            "span_s": max(stamps) - min(stamps),
        }
    return spans, problems


def overlapping_pairs(spans: dict) -> list[list[str]]:
    names = [n for n in STREAMS if n in spans]
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y = spans[a], spans[b]
            if x["start_s"] < y["end_s"] and y["start_s"] < x["end_s"]:
                pairs.append([a, b])
    return pairs


def assess(ledger, odyssey_state, owned: bool, owner_detail: dict,
           ledger_sha256: str = "") -> tuple[dict | None, list[str]]:
    """Build the receipt, or refuse with reasons. Never both."""
    reasons: list[str] = []
    if not isinstance(ledger, dict):
        return None, [f"{LEDGER} is absent or unreadable; HCLI has recorded no Odyssey science"]

    spans, problems = stream_spans(ledger)
    reasons += problems
    pairs = overlapping_pairs(spans)
    if len(spans) == 3 and not pairs:
        reasons.append("no two stream spans overlap in wall time; that is a global barrier")

    laws = ledger.get("laws") if isinstance(ledger.get("laws"), list) else []
    scars = ledger.get("scars") if isinstance(ledger.get("scars"), list) else []
    if len(laws) + len(scars) == 0:
        reasons.append("no laws and no scars; WorkUnit bookkeeping is not scientific movement")

    patients = (odyssey_state or {}).get("patients") or []
    incomplete = sorted(p.get("oxx") for p in patients if isinstance(p, dict) and not p.get("on_disk") and p.get("oxx"))
    if not incomplete:
        reasons.append(
            "every specimen is complete, so 'science did not stall on a download' is untestable here; "
            "refusing to record a vacuous false"
        )

    if not owned:
        reasons.append("hcli_owned cannot be earned: " + str(owner_detail.get("reason") or "producer is not the resident"))

    if reasons:
        return None, reasons

    return {
        "schema": "hawking.sovereign.g011_odyssey_streaming.v1",
        "gate": "G011",
        "produced_by": PRODUCED_BY,
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "command": COMMAND,
        "status": "completed",
        "streams": spans,
        "overlapping_pairs": pairs,
        "blocked_on_incomplete_specimen": False,
        "hcli_owned": True,
        "laws_or_scars_added": len(laws) + len(scars),
        "measurement_basis": {
            "stream_spans": "MEASURED: min/max of each stream's own recorded_at timestamps, "
                            "written by hcli/odyssey.py at the moment each entry was appended",
            "hcli_owned": "MEASURED: this process shares the live resident worker's session id, "
                          "and every stream entry carries the hcli_owned stamp written at append time",
            "blocked_on_incomplete_specimen": "DERIVED, with a stated limit: specimens listed in "
                                              "incomplete_specimens are not on disk NOW and the streams "
                                              "recorded work before now. ODYSSEY_STATE.json keeps no "
                                              "history of on_disk, so this cannot be proven per-entry.",
            "estimated_fields": [],
        },
        "evidence": {
            "ledger_path": str(LEDGER.relative_to(REPO)),
            "ledger_sha256": ledger_sha256,
            "odyssey_state_path": str(ODYSSEY_STATE.relative_to(REPO)),
            "law_count": len(laws),
            "scar_count": len(scars),
            "incomplete_specimens": incomplete,
            "specimen_count": len(patients),
            "ownership": owner_detail,
        },
    }, []


def main() -> int:
    owned, owner_detail = running_under_resident_worker()
    try:
        digest = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    except OSError:
        digest = ""
    doc, reasons = assess(_load_json(LEDGER), _load_json(ODYSSEY_STATE), owned, owner_detail, digest)
    if doc is None:
        print("G011 REFUSED - no receipt written. A red gate is an open question:")
        for reason in reasons:
            print(f"  - {reason}")
        return 1
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"G011 receipt written: {RECEIPT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
