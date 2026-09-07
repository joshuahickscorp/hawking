"""Per-specimen Odyssey state for all 56 bodies: what is measured, what is owed.

G034 requires every specimen to end with GPU, CPU, TPS, EBPW, NR-candidate and
NX-disposition evidence, OR AN EXPLICIT RECORDED REASON IT COULD NOT. That "or"
is the whole design. A campaign over 56 heterogeneous bodies will not measure six
axes on all of them -- 7 have no safetensors, 3 ship pre-quantized, 35 have no
expert organ -- and the failure mode to avoid is a specimen quietly carrying a
blank where a reason belongs, which reads as "not done yet" forever.

So an axis is in exactly one of three states and never a fourth:

    MEASURED   a value, with the receipt that produced it
    REFUSED    a named mechanism saying why this axis cannot exist for this body
    OWED       nothing yet; real outstanding work

`progress()` counts REFUSED as resolved, because a recorded impossibility is a
finding. It counts a MEASURED value with no receipt as OWED, because an
unattributable number is not evidence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

AXES = ("anatomy", "ebpw", "gpu", "cpu", "tps", "nr_candidate", "nx_disposition")
MEASURED, REFUSED, OWED = "MEASURED", "REFUSED", "OWED"


class LedgerError(RuntimeError):
    pass


def _blank(slug: str, gib: float, family: str, klass: str) -> dict:
    return {"slug": slug, "gib": gib, "family": family, "class": klass,
            "axes": {a: {"state": OWED, "value": None, "reason": None, "receipt": None}
                     for a in AXES}}


def measured(rec: dict, axis: str, value: Any, receipt: str) -> None:
    """Record a value. A value without a receipt is refused at the door."""
    if axis not in AXES:
        raise LedgerError(f"{axis} is not an Odyssey axis; expected one of {AXES}")
    if not receipt:
        raise LedgerError(
            f"{rec['slug']}/{axis}: a measurement without a receipt is not evidence")
    # ...and a receipt that does not EXIST is not evidence either. This checked only for
    # emptiness, so any non-empty string passed: recording
    # "receipts/future/THIS_FILE_DOES_NOT_EXIST.json" was accepted as MEASURED, verified
    # live. Nothing downstream would have caught it -- tps_contract, the tool that would
    # notice an unreadable physical receipt, has no caller outside its own test.
    if not Path(receipt).exists():
        raise LedgerError(
            f"{rec['slug']}/{axis}: receipt {receipt!r} does not exist. A path that names "
            f"nothing is not evidence; write the receipt first, then record the cell.")
    rec["axes"][axis] = {"state": MEASURED, "value": value, "reason": None,
                         "receipt": receipt}


def refused(rec: dict, axis: str, reason: str) -> None:
    """Record why this axis cannot exist here. The reason must name a mechanism."""
    if axis not in AXES:
        raise LedgerError(f"{axis} is not an Odyssey axis; expected one of {AXES}")
    if not reason or len(reason) < 20:
        raise LedgerError(
            f"{rec['slug']}/{axis}: refusal needs a mechanism, not {reason!r}. "
            f"'unsupported' or 'n/a' is how an unmeasured axis disguises itself as a finding.")
    rec["axes"][axis] = {"state": REFUSED, "value": None, "reason": reason, "receipt": None}


def progress(ledger: dict) -> dict:
    rows = ledger["specimens"]
    per_axis = {a: {MEASURED: 0, REFUSED: 0, OWED: 0} for a in AXES}
    complete = 0
    for r in rows:
        for a in AXES:
            per_axis[a][r["axes"][a]["state"]] += 1
        if all(r["axes"][a]["state"] in (MEASURED, REFUSED) for a in AXES):
            complete += 1
    total = len(rows) * len(AXES)
    resolved = sum(per_axis[a][MEASURED] + per_axis[a][REFUSED] for a in AXES)
    return {"specimens": len(rows), "specimens_complete": complete,
            "axes_total": total, "axes_resolved": resolved,
            "axes_owed": total - resolved,
            "pct": round(100 * resolved / max(total, 1), 1),
            "per_axis": per_axis}


def build(census_path: str) -> dict:
    """Seed the ledger from the reachability census, carrying its refusals across."""
    c = json.load(open(census_path))
    rows = []
    for s in c["rows"]:
        r = _blank(s["slug"], s.get("gib", 0.0), s.get("family", "?"), s["klass"])
        if "complete_ebpw" in s:
            measured(r, "ebpw", s["complete_ebpw"], census_path)
        elif s.get("accounting_error"):
            refused(r, "ebpw", f"accounting refused rather than guess a denominator: "
                               f"{s['accounting_error']}")
        elif s["klass"] == "NO-SAFETENSORS":
            refused(r, "ebpw", f"no safetensors to account: {s.get('blocked_by', '')}")

        if s["klass"] == "NO-SAFETENSORS":
            refused(r, "anatomy", f"no safetensors under the snapshot: "
                                  f"{s.get('blocked_by', 'weights are in another format')}")
        elif s["klass"] == "MOE-PREQUANTIZED":
            refused(r, "anatomy", f"expert organ is pre-quantized on disk: "
                                  f"{s.get('blocked_by', '')}. A spectrum over quantization "
                                  f"codes measures the codebook, not the organism.")
        rows.append(r)
    return {"schema": "odyssey-ledger-1", "source_census": census_path, "specimens": rows}


def _selfcheck() -> None:
    led = {"schema": "odyssey-ledger-1", "specimens":
           [_blank("a--b", 1.0, "fam", "DENSE")]}
    r = led["specimens"][0]
    p = progress(led)
    assert p["axes_owed"] == len(AXES) and p["specimens_complete"] == 0, p

    # A value with no receipt is not evidence.
    try:
        measured(r, "ebpw", 16.0, "")
        raise AssertionError("accepted a measurement with no receipt")
    except LedgerError as e:
        assert "not evidence" in str(e)

    # A refusal must name a mechanism; "n/a" is how a blank disguises itself.
    for junk in ("n/a", "unsupported", ""):
        try:
            refused(r, "gpu", junk)
            raise AssertionError(f"accepted {junk!r} as a refusal reason")
        except LedgerError as e:
            assert "mechanism" in str(e)

    measured(r, "ebpw", 16.0, "receipts/x.json")
    assert progress(led)["axes_resolved"] == 1

    # A recorded impossibility counts as resolved -- that is the point of REFUSED.
    for a in AXES:
        if a != "ebpw":
            refused(r, a, "this body has no transformer decoder, so the axis is undefined")
    p = progress(led)
    assert p["specimens_complete"] == 1 and p["axes_owed"] == 0, p

    # An unknown axis is a typo, not a new axis.
    try:
        measured(r, "tokens_per_joule", 1.0, "receipts/x.json")
        raise AssertionError("accepted an unknown axis")
    except LedgerError as e:
        assert "not an Odyssey axis" in str(e)
    print("selfcheck OK")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        led = build(sys.argv[1])
        p = progress(led)
        print(json.dumps(p, indent=1))
        if len(sys.argv) > 2:
            json.dump(led, open(sys.argv[2], "w"), indent=1)
            print(f"wrote {sys.argv[2]}")
