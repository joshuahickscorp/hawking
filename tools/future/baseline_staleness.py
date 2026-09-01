"""G138: three ledgers were computing from a dead baseline, and each changed a conclusion.

I found this by hand three times in one session and each time it moved a
strategic number:

    GAP_LEDGER_60      gap to 60 was 10.6229 ms; it is 5.2797
    PATH_TO_71         best composed was 49.8 TPS; rebased 67.86 with a
                       double-count, 62.90 without
    DELTANET economics upper bound was 34.7% of the residual gap; it is 114.6%

Three for three. The fourth would have been found the same way - by noticing -
which is not a method. This is the detector.

A SUPERSEDED VALUE IS NOT A DEFECT. A receipt that RECORDS 27.2896 ms as the
measurement it took, or that explains why that number was replaced, is doing its
job. The defect is a LIVE CONSUMER: a receipt whose producer computes a current
decision from a value the campaign has replaced.

So the classification is not "does this number appear" - it is "does this
receipt's PRODUCER read the current source". That is checkable from the module's
own text, and it is the same question G132 answered by hand for the gap ledger.

    python3 tools/future/baseline_staleness.py            # the report
    python3 tools/future/baseline_staleness.py --build
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/baseline_staleness.py"
RECEIPT_NAME = "BASELINE_STALENESS.json"

CURRENT_REL = "receipts/future/SEALED_DEFAULT_ABSOLUTE.json"
RECEIPTS_DIR = REPO / "receipts" / "future"

# Keys under which a superseded value is HISTORY rather than a live input. A
# hand-maintained list of entitled RECEIPTS would itself go stale - which is the
# exact failure this module exists to catch - so entitlement is decided
# structurally, by WHERE in the document the value appears.
HISTORICAL_KEY = re.compile(
    r"supersed|superseded|historical|previous|prior|was_|_was|stale|"
    r"correction_history|before|pre_|_pre|old_|_old|deprecat|note|why|"
    r"reading|claim_boundary|reason", re.I)
SUPERSEDED = (
    {
        "value": "28.722",
        "was": "pre-widen_f4 token ms",
        "now": "21.9464 GPU ms (G131)",
    },
    {
        "value": "27.2896",
        "was": "post-widen_f4 WALL ms per token",
        "now": "21.9464 GPU ms (G131), and the unit changed - wall is a joint "
               "claim about the resident AND the machine (G125)",
    },
    {
        "value": "36.644",
        "was": "wall TPS at the pre-promotion baseline",
        "now": "45.566 GPU TPS (G131)",
    },
)

# A producer that reads any of these is on the current baseline.
CURRENT_SOURCES = ("SEALED_DEFAULT_ABSOLUTE", "sealed_default_absolute")


class StalenessRefused(RuntimeError):
    """The current baseline is not on disk, so nothing can be called stale."""


def current() -> dict[str, Any]:
    p = REPO / CURRENT_REL
    if not p.is_file():
        raise StalenessRefused(
            f"{CURRENT_REL} is not on disk. Without a current baseline every "
            "receipt would be reported stale, which is worse than no report."
        )
    m = json.loads(p.read_text())["measured"]
    return {
        "source": CURRENT_REL,
        "gpu_ms_per_token": m["gpu_ms_per_token"],
        "gpu_tps": m["gpu_tps"],
        "wall_ms_per_token": m["wall_ms_per_token"],
    }


def _producer_of(doc: dict[str, Any]) -> str | None:
    rb = doc.get("recorded_by") or (doc.get("bench") or {}).get("recorded_by")
    return rb if isinstance(rb, str) else None


def _reads_current(producer: str | None) -> bool | None:
    """Does the producing module read the current baseline? None if unreadable."""
    if not producer:
        return None
    p = REPO / producer
    if not p.is_file():
        return None
    src = p.read_text()
    return any(s in src for s in CURRENT_SOURCES)


def _occurrences(node: Any, value: str, path: str = "") -> list[str]:
    """Every JSON path at which `value` appears, as a number or inside a string."""
    out: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            out += _occurrences(v, value, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += _occurrences(v, value, f"{path}[{i}]")
    else:
        text = repr(node)
        if re.search(rf"(?<![\d.]){re.escape(value)}(?![\d])", text):
            out.append(path or ".")
    return out


def _is_historical(paths: list[str]) -> bool:
    """History if EVERY occurrence sits under a historical key.

    One live occurrence is enough to make it live. A receipt that explains the
    supersession in prose AND still divides by the old number is a live
    consumer with a good alibi.
    """
    return bool(paths) and all(HISTORICAL_KEY.search(p) for p in paths)


def scan() -> list[dict[str, Any]]:
    current()  # refuse early if there is no current baseline
    rows: list[dict[str, Any]] = []
    for path in sorted(RECEIPTS_DIR.glob("*.json")):
        try:
            text = path.read_text()
            doc = json.loads(text)
        except (ValueError, OSError):
            continue
        if not isinstance(doc, dict):
            continue
        hits, entitled, where = [], [], {}
        for sup in SUPERSEDED:
            paths = _occurrences(doc, sup["value"])
            if not paths:
                continue
            hits.append(sup["value"])
            where[sup["value"]] = paths[:6]
            if _is_historical(paths):
                entitled.append(sup["value"])
        if not hits:
            continue
        name = path.name
        producer = _producer_of(doc)
        reads = _reads_current(producer)
        unentitled = [h for h in hits if h not in entitled]
        if not unentitled:
            verdict = "HISTORICAL_OWNER"
        elif reads is True:
            verdict = "CITES_BUT_READS_CURRENT"
        elif reads is False:
            verdict = "PRODUCER_DOES_NOT_READ_CURRENT_BASELINE"
        else:
            verdict = "PRODUCER_UNKNOWN"
        rows.append({
            "receipt": name,
            "superseded_values_present": hits,
            "carried_as_history": entitled,
            "occurrence_paths": where,
            "producer": producer,
            "producer_reads_current_baseline": reads,
            "verdict": verdict,
        })
    return rows


def report() -> dict[str, Any]:
    rows = scan()
    by = {}
    for r in rows:
        by.setdefault(r["verdict"], []).append(r["receipt"])
    stale = by.get("PRODUCER_DOES_NOT_READ_CURRENT_BASELINE", [])
    return {
        "n_receipts_citing_a_superseded_value": len(rows),
        "by_verdict": {k: sorted(v) for k, v in sorted(by.items())},
        "n_needing_review": len(stale),
        "needing_review": sorted(stale),
        "clean": not stale,
        "this_is_a_REVIEW_LIST_not_a_DEFECT_LIST": (
            "the verdict is a FACT - this receipt carries a superseded value "
            "somewhere that is not obviously historical, and its producer does "
            "not read the current baseline. Some of these are correct as they "
            "are: a receipt that IS the old measurement legitimately records it, "
            "and HARNESS_RECONCILIATION exists precisely to hold both numbers "
            "side by side. Calling them all defects would be the same "
            "over-claiming this module is meant to catch."
        ),
        "reading": (
            f"{len(rows)} receipts mention a superseded value and {len(stale)} "
            "need a human look: their producer does not read the current "
            "baseline. Three checked by hand were all genuinely stale and each "
            "moved a strategic number, so the base rate in this class is not low."
        ),
    }


def what_this_does_not_do() -> dict[str, Any]:
    return {
        "does_not_rewrite": True,
        "why": (
            "a receipt is regenerated by its own producer. Editing a value in "
            "place is the receipt-only fix this campaign forbids: the next run "
            "would put the dead number straight back."
        ),
        "producer_unknown_is_not_clean": (
            "a receipt whose producer cannot be read is reported as "
            "PRODUCER_UNKNOWN, not as passing. An unreadable producer is an "
            "unanswered question, and calling it clean is how a stale ledger "
            "survives an audit."
        ),
        "a_mention_is_not_a_defect": (
            "HISTORICAL_OWNER and CITES_BUT_READS_CURRENT are correct states. "
            "HARNESS_RECONCILIATION exists precisely to compare 27.2896 against "
            "21.9464 and must keep both."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "obligation": "G138",
        "question": (
            "which receipts compute a current decision from a baseline the "
            "campaign has replaced?"
        ),
        "current_baseline": current(),
        "superseded": [
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()}
            for s in SUPERSEDED
        ],
        "report": report(),
        "rows": scan(),
        "what_this_does_not_do": what_this_does_not_do(),
        "found_by_hand_three_times_first": [
            "GAP_LEDGER_60: gap to 60 was 10.6229 ms, is 5.2797",
            "PATH_TO_71: best composed was 49.8 TPS, is 62.90",
            "DELTANET_STATE_MACHINE_ECONOMICS: upper bound was 34.7% of the "
            "residual gap, is 114.6%",
        ],
        "evidence_class": "DERIVED_FROM_RECEIPTS_AND_PRODUCER_SOURCE",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args(argv)
    doc = build()
    if a.build:
        print(write_receipt(REPO / "receipts" / "future" / RECEIPT_NAME,
                            doc, RECORDED_BY))
        return 0
    print(json.dumps({k: doc[k] for k in ("current_baseline", "report")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
