#!/usr/bin/env python3
"""Fill the box to a target fraction with parent + children, and always keep a
generation slot free.

Every constant below is MEASURED on this box on 2026-08-16, not modelled:

  resident weight body        15,120,416,768 B   RSS after load, uniform-q4-v1
  marginal attached session      173,703,168 B   identical on each of 3 deltas, seq=128
  session workspace @seq2048     427,000,000 B   workspace formula 175,361,796 B + KV
  separate PROCESS child       8,770,000,000 B   4 children, sum RSS 35.09 GB

Two facts shape the whole design and both were measured, not assumed:

  1. Separate processes do NOT share artifact pages. 8.77 GB against an 8.5 GB
     artifact means each process holds a private copy. Never scale that design.
  2. Extra attached sessions do NOT add throughput. 4-session aggregate was
     9.427 tok/s against 26.653 for one. Concurrent decode ceiling is 1.

So sessions are RESEARCH SLOTS - many cheap resident candidates, stepped one at a
time - and not a TPS lever. Scaling to 90% buys breadth of search, not speed.

The generation reserve is not optional. A promoted child has to be able to launch
while the parent is still seated, or the lineage cannot reproduce without a window
where zero valid Genesis exists. That reserve is subtracted BEFORE children.

    genesis_capacity.py plan [--fill 0.90] [--seq 2048]
    genesis_capacity.py admit <n> [--seq 2048]     # exit 0 admit, 1 REFUSE
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

GIB = 1024 ** 3

# --- measured, 2026-08-16, receipts/ascent-2026-08-16/QWEN38_SHARED_SESSIONS.json
BODY_BYTES = 15_120_416_768
SESSION_BYTES = {128: 173_703_168, 2048: 427_000_000}
PROCESS_CHILD_BYTES = 8_770_000_000
# A promoted successor needs a whole body of its own: a new generation is a new
# artifact, so it cannot attach to the parent's weight buffers.
GENERATION_RESERVE_BYTES = BODY_BYTES
# macOS does not fail loudly when it swaps; it just makes every later number a lie.
NO_SWAP_FLOOR_BYTES = 4 * GIB


def mem() -> dict:
    total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                               capture_output=True, text=True).stdout.strip())
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = 16384
    vals = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip().rstrip(".")
        if v.isdigit():
            vals[k.strip()] = int(v) * page
    # Inactive and purgeable pages are reclaimable under pressure, so "free" alone
    # badly understates what is actually available and would refuse work that fits.
    available = (vals.get("Pages free", 0) + vals.get("Pages inactive", 0)
                 + vals.get("Pages purgeable", 0))
    return {"total": total, "available": available,
            "wired": vals.get("Pages wired down", 0),
            "used": total - available}


def plan(fill: float, seq: int) -> dict:
    m = mem()
    per_session = SESSION_BYTES.get(seq, SESSION_BYTES[2048])
    target_used = int(m["total"] * fill)
    # Everything already resident counts against the target - Grok lanes included.
    budget = target_used - m["used"]
    budget -= GENERATION_RESERVE_BYTES          # reserve BEFORE children
    hard_cap = m["available"] - NO_SWAP_FLOOR_BYTES - GENERATION_RESERVE_BYTES
    budget = min(budget, hard_cap)

    body_needed = BODY_BYTES
    sessions = 0
    if budget > body_needed:
        sessions = max(0, int((budget - body_needed) // per_session))
    elif budget > 0:
        body_needed = 0                          # cannot even seat a body

    return {
        "total_gib": round(m["total"] / GIB, 2),
        "used_gib": round(m["used"] / GIB, 2),
        "available_gib": round(m["available"] / GIB, 2),
        "target_fill": fill,
        "target_used_gib": round(target_used / GIB, 2),
        "generation_reserve_gib": round(GENERATION_RESERVE_BYTES / GIB, 2),
        "no_swap_floor_gib": round(NO_SWAP_FLOOR_BYTES / GIB, 2),
        "seq_len": seq,
        "per_session_mib": round(per_session / (1024 ** 2), 1),
        "parent_body_gib": round(body_needed / GIB, 2),
        "admissible_sessions": sessions,
        "projected_used_gib": round(
            (m["used"] + body_needed + sessions * per_session
             + GENERATION_RESERVE_BYTES) / GIB, 2),
        "equivalent_processes_if_not_shared": round(
            (body_needed + sessions * per_session) / PROCESS_CHILD_BYTES, 1),
        "throughput_note": "concurrent decode ceiling is 1 (measured); sessions are "
                           "research slots, not a TPS lever",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "admit"])
    ap.add_argument("n", nargs="?", type=int)
    ap.add_argument("--fill", type=float, default=0.90)
    ap.add_argument("--seq", type=int, default=2048)
    a = ap.parse_args()

    p = plan(a.fill, a.seq)
    if a.mode == "plan":
        print(json.dumps(p, indent=2))
        return 0

    if a.n is None:
        print("admit needs a count", file=sys.stderr)
        return 2
    if a.n > p["admissible_sessions"]:
        print(f"REFUSE {a.n}: admissible is {p['admissible_sessions']} at seq={a.seq} "
              f"(available {p['available_gib']} GiB, generation reserve "
              f"{p['generation_reserve_gib']} GiB, no-swap floor {p['no_swap_floor_gib']} GiB)",
              file=sys.stderr)
        return 1
    print(f"ADMIT {a.n} (of {p['admissible_sessions']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
