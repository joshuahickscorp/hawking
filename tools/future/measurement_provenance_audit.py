"""A hardware number with no provenance cannot be read, only believed.

This machine has a RECORDED contamination hazard: an interactive session inflates
decode throughput several-fold, which is why paired A/B and kernel-count ratios
exist. So a receipt that says `accepted_tps: 23.63` and nothing about when it was
taken, under what load, or whether the GPU lane lock was held is not a
measurement anyone can interpret -- it is a number.

`tools/future/_common.py` already provides both halves of the rule:

    write_receipt()          REFUSES hardware fields outright, for the static
                             sidecar lane that must never carry one
    measurement_provenance() the shape a real hardware number needs -- lock_held,
                             loadavg, lane, measured_at

The gap this audits is that the second one is used only in tools/future/, the
lane FORBIDDEN from having hardware numbers, and not once in tools/headless/,
the lane where the GPU numbers actually live.

It counts rather than fixes. Retrofitting provenance onto an existing number
would be inventing the very thing that is missing: nobody can now say whether a
figure recorded months ago held the lane. The deficit is made visible, countable
and non-growing instead, and the fix is cheap for whoever re-runs a measurement.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from tools.future._common import HARDWARE_FIELDS

REPO = Path(__file__).resolve().parents[2]
RECEIPT_ROOTS = ("receipts/headless", "receipts/future", "receipts/acceptance")

#: Any of these anywhere in a receipt means someone recorded the conditions.
PROVENANCE_MARKERS = (
    "measurement_provenance", "lock_held", "gpu_authority",
    "lane_lock", "measured_at", "loadavg",
)


def hardware_numbers(doc: Any, path: str = "") -> Iterator[tuple[str, float]]:
    """Every nonzero hardware-named number in a document, with its path.

    The field list is imported from _common rather than retyped: two lists of
    what counts as a hardware number drift, and then a guard and an audit
    disagree about what they are protecting.
    """
    if isinstance(doc, dict):
        for key, value in doc.items():
            if key in HARDWARE_FIELDS and isinstance(value, (int, float)) and value:
                yield f"{path}.{key}", float(value)
            yield from hardware_numbers(value, f"{path}.{key}")
    elif isinstance(doc, list):
        for i, value in enumerate(doc):
            yield from hardware_numbers(value, f"{path}[{i}]")


def audit() -> dict[str, Any]:
    rows = []
    scanned = carrying = provenanced = bare = 0
    for root in RECEIPT_ROOTS:
        base = REPO / root
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*.json")):
            scanned += 1
            try:
                doc = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            hits = list(hardware_numbers(doc))
            if not hits:
                continue
            carrying += 1
            blob = json.dumps(doc)
            has = [m for m in PROVENANCE_MARKERS if m in blob]
            if has:
                provenanced += 1
            else:
                bare += 1
                rows.append({
                    "receipt": f"{root}/{path.name}",
                    "hardware_fields": len(hits),
                    "example": {"field": hits[0][0], "value": hits[0][1]},
                })
    return {
        "schema": "hawking.future.measurement_provenance_audit.v1",
        "evidence_tier": "STATIC",
        "hardware_field_names": sorted(HARDWARE_FIELDS),
        "roots": list(RECEIPT_ROOTS),
        "receipts_scanned": scanned,
        "receipts_carrying_a_hardware_number": carrying,
        "with_provenance": provenanced,
        "without_provenance": bare,
        "bare": rows,
        "why_it_is_not_retrofitted": (
            "Provenance cannot be added after the fact. Nobody can now say whether a "
            "figure recorded months ago held the GPU lane, and writing lock_held:true "
            "to make an audit pass would fabricate exactly the fact that is missing. "
            "The deficit is counted and pinned so it can only shrink; it shrinks when "
            "a measurement is RE-RUN through measurement_provenance()."
        ),
        "claim_boundary": (
            "STATIC scan of committed receipts. It measures nothing itself and "
            "changes no recorded number."
        ),
    }


def build() -> Path:
    from tools.future._common import write_receipt
    return write_receipt("MEASUREMENT_PROVENANCE_AUDIT.json", audit(),
                         recorded_by="tools/future/measurement_provenance_audit.py")


def main() -> int:
    doc = audit()
    print(f"scanned {doc['receipts_scanned']} receipts across {len(doc['roots'])} roots")
    print(f"carrying a hardware number: {doc['receipts_carrying_a_hardware_number']}")
    print(f"  with provenance:    {doc['with_provenance']}")
    print(f"  WITHOUT provenance: {doc['without_provenance']}")
    for row in doc["bare"][:8]:
        print(f"    {row['receipt']:56s} {row['hardware_fields']:3d} fields")
    print(f"wrote {build()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
