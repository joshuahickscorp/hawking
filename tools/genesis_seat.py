#!/usr/bin/env python3
"""Seat the current best verified ancestor as GENESIS, and report the lineage.

The lineage machinery (lab/lineage/) was built and tested but never turned on, so
nothing was actually seated - Genesis existed as a tournament result and a set of
receipts rather than as a running lineage. This seats it.

Generation 0 is Qwen3.8 uniform-q4-v1 at 4.2527 BPW / 35,227,918 ns. Nothing has
beaten it: the fusion child regressed 10.68 ms, the attention codec produced a
cosine-only candidate with the complete token unmoved, cross-token reuse was
refuted, and shared sessions save memory without amortizing DRAM. That is what
makes it CURRENT - not that it is good, but that it is the best VERIFIED ancestor.

    genesis_seat.py seat     # install generation 0 as CURRENT + LAST_KNOWN_GOOD
    genesis_seat.py status   # print the three slots
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lab.lineage.identity import make_qwen38_genesis  # noqa: E402
from lab.lineage.state import LineageState  # noqa: E402

STATE_FILE = REPO / "receipts" / "ascent-2026-08-16" / "GENESIS_LINEAGE_CURRENT.json"


def seat() -> int:
    lineage = LineageState()
    g0 = make_qwen38_genesis()
    lineage.install(g0)
    # LAST_KNOWN_GOOD must never be empty: a failed successor launch has to have
    # somewhere to roll back to, and "there must always be a valid Genesis" is the
    # one invariant that cannot be recovered after the fact.
    lineage.snapshot_current_as_lkg()

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(lineage.to_dict(), indent=2, default=str))
    print(f"SEATED generation {g0.generation}: {g0.instance_id}")
    print(f"  BPW              {g0.representation_bpw}")
    print(f"  complete token   {g0.complete_token_ns:,} ns")
    print(f"  artifact sha     {g0.artifact_sha[:16]}")
    print(f"  valid instances  {lineage.valid_count()}")
    print(f"  state            {STATE_FILE.relative_to(REPO)}")
    return 0


def status() -> int:
    if not STATE_FILE.is_file():
        print("no lineage seated - run `genesis_seat.py seat`", file=sys.stderr)
        return 2
    d = json.loads(STATE_FILE.read_text())
    slots = d.get("slots", {})
    for slot in ("CURRENT", "CANDIDATE", "LAST_KNOWN_GOOD"):
        v = slots.get(slot)
        if not v:
            print(f"{slot:16} (empty)")
            continue
        print(f"{slot:16} gen {v.get('generation')} {v.get('instance_id')} "
              f"{v.get('representation_bpw')} BPW  {v.get('complete_token_ns'):,} ns  "
              f"{v.get('tps'):.2f} TPS")
    print(f"{'valid':16} {d.get('valid_count')}   zero_valid={d.get('zero_valid_genesis')}")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    sys.exit(seat() if mode == "seat" else status())
