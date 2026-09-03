#!/usr/bin/env python3
"""The autonomy-gate record for one HCLI attempt, from its receipt and events.

Gate 1 and Gate 2 are not passed by a mutation appearing. They are passed by a
mutation HCLI authored, verified deterministically, repaired if it was wrong,
and had promoted by protected acceptance -- with Claude writing no part of the
patch. This prints the fields that distinguish that from a lucky completion, and
refuses to report a pass it cannot evidence.
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
RECEIPTS = REPO / ".hcli" / "receipts"
EVENTS = REPO / ".hcli" / "mission" / "events.jsonl"


def newest_receipt() -> dict:
    files = sorted(RECEIPTS.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return {}
    return json.loads(files[-1].read_text())


def events() -> list:
    if not EVENTS.is_file():
        return []
    out = []
    for line in EVENTS.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def main() -> int:
    r = newest_receipt()
    if not r:
        print("no receipt")
        return 1
    # An error receipt files them under model_calls, a completed one under
    # calls. Reading only one reported "0 resident calls" for an attempt that
    # had made seven.
    calls = r.get("calls") or r.get("model_calls") or []
    ev = events()

    tool_calls = sum(1 for e in ev if e.get("type") == "tool_invoked")
    tool_fail = sum(
        1 for e in ev
        if e.get("type") == "tool_invoked" and not (e.get("data") or {}).get("ok")
    )
    retries = sum(1 for e in ev if e.get("type") == "contract_retry")
    reduced = sum(1 for e in ev if e.get("type") == "context_reduced")

    ops = r.get("operations") or []
    validation = r.get("validation") or {}
    landed = (
        r.get("kind") == "mutation"
        and r.get("status") == "completed"
        and not r.get("rolled_back")
        and bool(validation.get("ok"))
    )

    prompt_tokens = sum(c.get("prompt_tokens", 0) for c in calls)
    reused = sum(c.get("prefix_reused_tokens", 0) for c in calls)
    stepped = sum(c.get("prefill_tokens_stepped", 0) for c in calls)
    wall = sum(c.get("wall_s", 0.0) for c in calls)

    print(f"receipt              {r.get('goal_id') or r.get('id') or '?'}")
    print(f"kind / status        {r.get('kind')} / {r.get('status')}")
    print(f"resident calls       {len(calls)}")
    print(f"prompt tokens        {prompt_tokens:,}")
    print(f"reused prefix tokens {reused:,}"
          f"   ({reused / max(prompt_tokens, 1):.0%} of prompt)")
    print(f"prefill stepped      {stepped:,}")
    print(f"tool calls           {tool_calls}  ({tool_fail} failed)")
    print(f"contract retries     {retries}")
    print(f"context reductions   {reduced}")
    print(f"wall time            {wall:.0f}s")
    print(f"mutations proposed   {len(ops)}")
    print(f"verifier result      {validation.get('ok')}  "
          f"{str(validation.get('reason') or validation.get('detail') or '')[:80]}")
    if r.get("error"):
        print(f"error                {str(r['error'])[:150]}")
    print()
    print(f"LANDED               {landed}")
    if not landed:
        print("  a mutation is landed only when it is kind=mutation, completed,")
        print("  not rolled back, AND its validation passed. Anything else is an")
        print("  attempt, however far it got.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
