"""Stamp S032 §3 machine state onto every performance receipt that lacks one.

Idempotent, and it must be, because this repo has a THIRD WRITER: a launchd job
(tools/odyssey_driver.sh) regenerates receipts every five minutes and writes them
without a bench block. A one-time backfill cannot hold against a live writer --
eighteen receipts had already lost their stamp forty minutes after the first pass,
and the corpus test is what noticed.

The state is UNKNOWN unless the receipt carries quiescence evidence of its own, in
which case it is DERIVED from that. Never QUIESCED by default: S032 §3 is verbatim
"If quiescence is unknown: BENCH_STATE = UNKNOWN, not quiet."

    python3 tools/accelerator/stamp_bench_state.py            # stamp
    python3 tools/accelerator/stamp_bench_state.py --check    # exit 1 if any lack one
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import receipt as R

RH = R.REPO / "receipts" / "headless"
MACHINE = ("Apple M3 Ultra, 60 GPU cores, 96 GiB "
           "(receipts/headless/MACHINE_GENOME.json)")
# A raw data dump is not a receipt and never made a claim.
RECEIPT_MARKERS = {"schema", "receipt", "identities", "result", "date"}


def file_indent(text: str) -> int:
    """Preserve the file's own indentation.

    Reformatting 297 receipts to a different indent produced an 863,000-line diff
    in which the single added field was invisible.
    """
    for line in text.split("\n")[1:]:
        stripped = line.lstrip(" ")
        if stripped and stripped != "}":
            return len(line) - len(stripped)
    return 1


def embedded_samples(node, found=None) -> list:
    """Quiescence samples a receipt already carries, wherever it put them."""
    found = [] if found is None else found
    if isinstance(node, dict):
        if "contenders" in node and ("quiet" in node or "n_contenders" in node):
            found.append(node)
        for v in node.values():
            embedded_samples(v, found)
    elif isinstance(node, list):
        for v in node:
            embedded_samples(v, found)
    return found


def performance_receipts():
    for f in sorted(RH.glob("*.json")):
        try:
            raw = f.read_text()
            d = json.loads(raw)
        except Exception:
            continue
        if not isinstance(d, dict) or not (RECEIPT_MARKERS & set(d)):
            continue
        if R._timing_keys(d, "root"):
            yield f, raw, d


def derive(d: dict) -> tuple[str, dict | None]:
    samples = embedded_samples(d)
    if not samples:
        return "UNKNOWN", None
    quiet = [s.get("quiet") for s in samples]
    if any(q is None for q in quiet):
        state = "UNKNOWN"
    elif all(q is True for q in quiet):
        state = "QUIESCED"
    else:
        state = "CONTENDED"
    return state, max(samples, key=lambda s: s.get("max_rss_gib") or 0.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report and exit 1 instead of writing")
    a = ap.parse_args(argv)

    missing, stamped = [], {"UNKNOWN": 0, "QUIESCED": 0, "CONTENDED": 0}
    for f, raw, d in performance_receipts():
        if isinstance(d.get("bench"), dict):
            continue
        missing.append(f.name)
        if a.check:
            continue
        state, sample = derive(d)
        d["bench"] = {
            "state": state,
            "recorded_at": "2026-08-26T00:00:00Z",
            "recorded_by": "S032 §3 backfill, tools/accelerator/stamp_bench_state.py",
            "machine": MACHINE,
            "quiescence": sample,
            "rule": "S032 §3 -- if quiescence is unknown the state is UNKNOWN, not quiet",
            "provenance": ("DERIVED from a quiescence sample this receipt already carried"
                           if sample else
                           "this receipt recorded no quiescence. UNKNOWN is what it was "
                           "measured under, not a claim that the machine was busy"),
        }
        f.write_text(json.dumps(d, indent=file_indent(raw)))
        stamped[state] += 1

    if a.check:
        if missing:
            print(f"{len(missing)} performance receipts carry no bench block: "
                  f"{missing[:8]}")
            print("fix: python3 tools/accelerator/stamp_bench_state.py")
            return 1
        print("every performance receipt states its machine")
        return 0
    print(f"stamped {stamped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
