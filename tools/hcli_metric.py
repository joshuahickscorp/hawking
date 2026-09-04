#!/usr/bin/env python3
"""The primary HCLI metric: verified useful work per unattended wall hour.

Defect count is the wrong objective. It rewards finding many small things and
says nothing about whether the machine produced work. What matters is how much
VERIFIED work HCLI completes per hour that nobody was watching, and what each
accepted WorkUnit physically cost.

A WorkUnit counts as verified-useful only when it changed real source and the
verifier accepted it on evidence that could have failed:

    kind == mutation, status == completed, not rolled_back,
    validation.ok, and red_before_green is not False.

red_before_green matters as much as the rest. The harness computes it correctly
and records it as advisory, so a mutation whose tests were already green is
otherwise indistinguishable from one that made them pass.
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
RECEIPTS = REPO / ".hcli" / "receipts"


def load() -> list:
    out = []
    for path in sorted(RECEIPTS.glob("*.json"), key=lambda p: p.stat().st_mtime):
        try:
            d = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        d["_mtime"] = path.stat().st_mtime
        d["_name"] = path.name[:8]
        out.append(d)
    return out


def accepted(r: dict) -> bool:
    v = r.get("validation") or {}
    return bool(
        r.get("kind") == "mutation"
        and r.get("status") == "completed"
        and not r.get("rolled_back")
        and v.get("ok")
        and v.get("red_before_green") is not False
    )


def calls(r: dict) -> list:
    return r.get("calls") or r.get("model_calls") or []


def main() -> int:
    # A WINDOW, stated. The receipts directory spans weeks of unrelated
    # experimentation, and dividing 2 accepted units by 322 lifetime hours
    # produces a number that is arithmetically true and says nothing about the
    # machine's current rate. Default to the last day; pass a number of hours
    # to widen it, or 0 for everything on disk.
    hours = 24.0
    if len(sys.argv) > 1:
        try:
            hours = float(sys.argv[1])
        except ValueError:
            print("usage: hcli_metric.py [window-hours, 0 for all]")
            return 2

    receipts = load()
    if not receipts:
        print("no receipts")
        return 1
    if hours > 0:
        newest = receipts[-1]["_mtime"]
        receipts = [r for r in receipts if newest - r["_mtime"] <= hours * 3600]
        print(f"window                      last {hours:.0f}h "
              f"({len(receipts)} receipts)")

    good = [r for r in receipts if accepted(r)]
    span_h = (receipts[-1]["_mtime"] - receipts[0]["_mtime"]) / 3600.0

    print(f"receipts                    {len(receipts)}")
    print(f"VERIFIED_USEFUL_WORKUNITS   {len(good)}")
    print(f"unattended wall hours       {span_h:.1f}")
    if span_h > 0:
        print(f"PRIMARY METRIC              {len(good) / span_h:.3f} verified units / hour")
    print()

    if not good:
        print("no accepted WorkUnit yet; nothing to break down")
        return 0

    total_calls = sum(len(calls(r)) for r in receipts)
    failed_calls = total_calls - sum(len(calls(r)) for r in good)

    print(f"{'unit':<9}{'calls':>6}{'prompt':>9}{'gen':>7}{'reuse':>7}"
          f"{'prefill':>9}{'decode':>8}{'wall':>7}")
    for r in good:
        cs = calls(r)
        prompt = sum(c.get("prompt_tokens", 0) for c in cs)
        gen = sum(c.get("completion_tokens", 0) for c in cs)
        reuse = sum(c.get("prefix_reused_tokens", 0) for c in cs)
        wall = sum(c.get("wall_s", 0.0) for c in cs)
        prefill = sum(
            (c.get("prefill_profile") or {}).get("totals", {}).get("wall_ns", 0)
            for c in cs
        ) / 1e9
        print(f"{r['_name']:<9}{len(cs):>6}{prompt:>9,}{gen:>7,}"
              f"{reuse / max(prompt, 1):>6.0%}{prefill:>9.0f}s"
              f"{max(wall - prefill, 0):>7.0f}s{wall:>7.0f}s")

    print()
    print(f"resident calls / accepted   {total_calls / len(good):.1f}")
    print(f"failed calls / accepted     {failed_calls / len(good):.1f}")
    print()
    print("tool wall, verifier wall, human interventions and external-model")
    print("escalations are not yet instrumented per WorkUnit. They are reported")
    print("as unknown rather than zero: a zero here would read as 'no human")
    print("touched it', which for this run would be false.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
