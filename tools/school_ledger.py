#!/usr/bin/env python3
"""Teacher-dependence ledger. The quantity this phase exists to reverse.

Reads what is on disk -- the school log and git history -- and reports who is
actually writing HCLI's improvements. Nothing here is estimated.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / ".hcli" / "school" / "log.jsonl"


def rows() -> list:
    if not LOG.is_file():
        return []
    out = []
    for line in LOG.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def main() -> int:
    data = rows()
    real = [r for r in data if r.get("phase") not in (None, "driver_error", "paused_low_disk")]
    landed = [r for r in real if r.get("landed")]
    authored = [r for r in real if r.get("kind") == "mutation"]
    rejected = [r for r in authored if not r.get("landed")]

    claude = subprocess.run(
        ["git", "log", "--oneline", "--since=24 hours ago"],
        cwd=REPO, capture_output=True, text=True,
    ).stdout.strip().splitlines()

    print("HCLI_CODE_MUTATIONS              ", len(authored))
    print("HCLI_ACCEPTED_MUTATIONS          ", len(landed))
    print("HCLI_REJECTED_MUTATIONS          ", len(rejected))
    print("CLAUDE_CODE_MUTATIONS (24h)      ", len(claude))
    ratio = (len(claude) / len(landed)) if landed else None
    print("CLAUDE_INTERVENTIONS_PER_ACCEPTED", f"{ratio:.1f}" if ratio else "undefined (0 accepted)")
    print()
    print("cycles run:", len(real), "| landed:", len(landed))
    for r in real[-5:]:
        v = r.get("verdict") or {}
        print(
            f"  L{r.get('level')}.{r.get('attempt')} {str(r.get('phase')):11} "
            f"kind={str(r.get('kind')):9} landed={r.get('landed')} "
            f"calls={r.get('model_calls')} rounds={v.get('mean_rounds_per_goal')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
