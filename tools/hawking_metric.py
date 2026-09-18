#!/usr/bin/env python3
"""The primary Hawking metric: verified useful work per unattended wall hour.

Defect count is the wrong objective. It rewards finding many small things and
says nothing about whether the machine produced work. What matters is how much
VERIFIED work Hawking completes per hour that nobody was watching, and what each
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
RECEIPTS = REPO / ".hawking" / "receipts"
LEGACY_RECEIPTS = REPO / ".hcli-legacy" / "receipts"
# Tracked rather than under the active state directory: an autonomy record
# that cannot be audited by anyone but the process that wrote it is not a
# record. Existing HCLI-era rows remain read-only migration evidence.
LEDGER = REPO / "receipts" / "hawking-v1" / "interventions.jsonl"
LEGACY_LEDGER = REPO / "receipts" / "hcli-v1" / "interventions.jsonl"

#: Named regimes. A clean production window must not be compared to the bootstrap
#: window unlabelled: during BOOTSTRAP the harness itself was being repaired
#: between almost every call, so its failed-call rate measures the harness, not
#: the resident. Epoch seconds, start-inclusive.
REGIMES = [
    ("BOOTSTRAP", 0, 1788500000),
    ("POST_BOOTSTRAP_RUNTIME", 1788500000, 1 << 62),
]


def regime_of(ts: float) -> str:
    for name, lo, hi in REGIMES:
        if lo <= ts < hi:
            return name
    return "UNLABELLED"


def interventions() -> list:
    ledger = LEDGER if LEDGER.is_file() else LEGACY_LEDGER
    if not ledger.is_file():
        return []
    out = []
    for line in ledger.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def load() -> list:
    out = []
    for receipt_dir in (RECEIPTS, LEGACY_RECEIPTS):
        for path in receipt_dir.glob("*.json"):
            try:
                stat = path.stat()
                payload = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            payload["_mtime"] = stat.st_mtime
            payload["_name"] = path.name[:8]
            out.append(payload)
    return sorted(out, key=lambda payload: payload["_mtime"])


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


def _finite_nonnegative(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0 or number != number or number == float("inf"):
        return None
    return number


def prefill_seconds(model_calls: list[dict]) -> float | None:
    """Return measured prefill time, or ``None`` if any call lacks it."""
    if not model_calls:
        return None
    total_ns = 0.0
    for model_call in model_calls:
        if not isinstance(model_call, dict):
            return None
        profile = model_call.get("prefill_profile")
        totals = profile.get("totals") if isinstance(profile, dict) else None
        wall_ns = totals.get("wall_ns") if isinstance(totals, dict) else None
        measured = _finite_nonnegative(wall_ns)
        if measured is None:
            return None
        total_ns += measured
    return total_ns / 1e9


def decode_seconds(model_calls: list[dict]) -> float | None:
    """Return measured wall minus measured prefill, never a guessed residual."""
    prefill = prefill_seconds(model_calls)
    if prefill is None:
        return None
    wall = 0.0
    for model_call in model_calls:
        if not isinstance(model_call, dict):
            return None
        measured = _finite_nonnegative(model_call.get("wall_s"))
        if measured is None:
            return None
        wall += measured
    return max(wall - prefill, 0.0)


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
            print("usage: hawking_metric.py [window-hours, 0 for all]")
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
          f"{'prefill':>18}{'decode':>18}{'wall':>7}")
    for r in good:
        cs = calls(r)
        prompt = sum(c.get("prompt_tokens", 0) for c in cs)
        gen = sum(c.get("completion_tokens", 0) for c in cs)
        reuse = sum(c.get("prefix_reused_tokens", 0) for c in cs)
        wall = sum(c.get("wall_s", 0.0) for c in cs)
        prefill = prefill_seconds(cs)
        decode = decode_seconds(cs)
        prefill_label = f"{prefill:.0f}s" if prefill is not None else "NOT_INSTRUMENTED"
        decode_label = f"{decode:.0f}s" if decode is not None else "NOT_INSTRUMENTED"
        print(f"{r['_name']:<9}{len(cs):>6}{prompt:>9,}{gen:>7,}"
              f"{reuse / max(prompt, 1):>6.0%}{prefill_label:>18}"
              f"{decode_label:>18}{wall:>7.0f}s")

    print()
    print(f"resident calls / accepted   {total_calls / len(good):.1f}")
    print(f"failed calls / accepted     {failed_calls / len(good):.1f}")
    print()
    # ---------------------------------------------------------------- regimes
    print()
    print("REGIMES")
    for name, lo, hi in REGIMES:
        units = [r for r in good if lo <= r["_mtime"] < hi]
        allr = [r for r in receipts if lo <= r["_mtime"] < hi]
        if not allr:
            continue
        span = (max(r["_mtime"] for r in allr) - min(r["_mtime"] for r in allr)) / 3600.0
        rate = len(units) / span if span > 0 else float("nan")
        # A rate from a couple of units over a fraction of an hour is a number
        # that will be quoted and should not be. Say so beside it rather than
        # letting it travel alone.
        weak = " [SMALL SAMPLE, not a throughput claim]" if (
            len(units) < 5 or span < 2.0
        ) else ""
        print(f"  {name:<24} {len(units)} units over {span:5.1f}h  = {rate:.3f}/h{weak}")

    # ------------------------------------------------------- human dependence
    rows = interventions()
    print()
    print("HUMAN DEPENDENCE")
    if not rows:
        print("  ledger absent. NOT reported as zero: an unrecorded human action")
        print("  is exactly the kind that inflates an autonomy number.")
        return 0
    kinds = {}
    for r in rows:
        k = r.get("kind", "?")
        kinds.setdefault(k, [0, 0])
        kinds[k][0] += 1
        kinds[k][1] += 1 if r.get("causal") else 0
    for k in sorted(kinds):
        n, c = kinds[k]
        print(f"  {k:<26} {n:>3}  causal {c}")
    causal = sum(c for _, c in kinds.values())
    print()
    print(f"  total human actions        {len(rows)}")
    print(f"  CAUSAL human interventions {causal}")
    if causal:
        print(f"  ACCEPTED WORK / CAUSAL     {len(good) / causal:.2f}")
    print()
    print("  A human watching is not a human fixing. Writing the goal is not")
    print("  causal -- the goal is the task. Naming the exact anchor IS causal,")
    print("  and both Gate 1 and Gate 2 goals did that, so they are counted.")
    print()
    print("  tool wall and verifier wall remain uninstrumented per WorkUnit and")
    print("  are reported as unknown rather than zero.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
