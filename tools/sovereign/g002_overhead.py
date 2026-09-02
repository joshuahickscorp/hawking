#!/usr/bin/env python3
"""G002 producer: paired direct-vs-HCLI rate + per-stage time attribution.

Receipt: receipts/sovereign/G002_hcli_overhead.json

This producer REFUSES to emit a partial receipt. G002 gates on twelve measured
stages and a paired rate comparison; anything less is an open question, and a
receipt that converts an open question into a false answer is worse than a red
gate. Every exit path that cannot measure honestly writes nothing and returns
non-zero with the precise reason.

Two modes:

  --attribute   Durable-evidence attribution. Needs no model body. Reads the
                receipts HCLI already wrote and reports exactly what they can
                and cannot support. Safe to run beside a live resident.

  --paired      The full measurement. Needs the ONLY model body on this box to
                be free, because `direct_tok_s` and `hcli_tok_s` must be taken
                against the same weights. Refuses while a resident is live.

Why `--paired` cannot run today: the body is a pipe-attached subprocess of the
HCLI worker (stdin/stdout JSONL, no listening socket). Nothing outside that
worker can address it, so a "direct" arm would require launching a SECOND
~11 GB body. That is a real block, not a missing feature.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECEIPT = REPO / "receipts/sovereign/G002_hcli_overhead.json"
HCLI_RECEIPTS = REPO / ".hcli/receipts"
PRODUCED_BY = "tools/sovereign/g002_overhead.py"
SCHEMA = "hawking.sovereign.g002_hcli_overhead.v1"

# Verbatim from hcli/test_hcli_overhead.py STAGES. Do not guess these.
STAGES = (
    "context_construction_ns", "retrieval_ns", "provider_prepare_ns",
    "serialization_ns", "transport_ns", "native_prefill_ns", "native_decode_ns",
    "parse_ns", "tool_dispatch_ns", "verifier_ns", "evidence_ingest_ns",
    "schedule_ns",
)

# Stages no durable receipt on disk can supply. The connector computes prefill/
# decode/gpu_ns per call (hcli/hawking_native.py metric_keys) but model_calls[]
# persists only wall_s/prompt_tokens/completion_tokens, so they are discarded.
NEEDS_LIVE_CALL = ("transport_ns", "native_prefill_ns", "native_decode_ns")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resident_is_live() -> tuple[bool, str]:
    """A live resident owns the only body; a second one must not be opened."""
    try:
        out = subprocess.run(
            ["pgrep", "-lf", "ascension_qwen38_resident"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return True, f"could not determine resident state ({exc}); refusing to assume it is free"
    if out:
        return True, f"a model body is already resident:\n{out}"
    return False, "no ascension_qwen38_resident process found"


def _load_calls() -> tuple[list[dict], dict]:
    """Successful, deduplicated model calls from durable HCLI receipts.

    Failed calls carry completion_tokens=None; coercing that to 0 invents a
    pure-prefill sample that never happened. They are excluded, not zeroed.
    """
    calls: list[dict] = []
    seen: set[tuple] = set()
    stats = {"receipts": 0, "failed_or_null": 0, "duplicates": 0}
    for path in sorted(HCLI_RECEIPTS.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entries = doc.get("model_calls")
        if not isinstance(entries, list) or not entries:
            continue
        stats["receipts"] += 1
        for call in entries:
            if not isinstance(call, dict):
                continue
            comp, wall = call.get("completion_tokens"), call.get("wall_s")
            if comp is None or wall is None:
                stats["failed_or_null"] += 1
                continue
            key = (call.get("prefix_key"), wall, call.get("prompt_tokens"), comp)
            if key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(key)
            calls.append({
                "prompt_tokens": int(call.get("prompt_tokens") or 0),
                "completion_tokens": int(comp),
                "wall_s": float(wall),
            })
    return calls, stats


def attribute() -> dict:
    """What the durable receipts genuinely support, and what they do not."""
    calls, stats = _load_calls()
    if not calls:
        raise SystemExit("no successful model calls in .hcli/receipts; nothing to attribute")

    wall = sum(c["wall_s"] for c in calls)
    comp = sum(c["completion_tokens"] for c in calls)
    prompt = sum(c["prompt_tokens"] for c in calls)
    pure_prefill = [c for c in calls if c["completion_tokens"] == 0]

    # Per-unit: how much of the unit's wall clock sits inside a model call.
    shares = []
    for path in sorted(HCLI_RECEIPTS.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        ts, entries = doc.get("timestamps") or {}, doc.get("model_calls")
        if not isinstance(entries, list) or not entries or not ts.get("started_at"):
            continue
        try:
            span = (datetime.fromisoformat(ts["finished_at"])
                    - datetime.fromisoformat(ts["started_at"])).total_seconds()
        except (ValueError, KeyError, TypeError):
            continue
        inside = sum(float(c.get("wall_s") or 0) for c in entries if isinstance(c, dict))
        if span > 0:
            shares.append(inside / span)
    shares.sort()

    return {
        "calls_used": len(calls),
        "excluded_failed_or_null": stats["failed_or_null"],
        "excluded_duplicates": stats["duplicates"],
        "total_call_wall_s": round(wall, 3),
        "total_completion_tokens": comp,
        "total_prompt_tokens": prompt,
        "aggregate_completion_tok_s": round(comp / wall, 4) if wall else None,
        "pure_prefill_samples": len(pure_prefill),
        "model_call_share_of_unit_wall": {
            "n_units": len(shares),
            "min": round(shares[0], 4) if shares else None,
            "median": round(shares[len(shares) // 2], 4) if shares else None,
            "max": round(shares[-1], 4) if shares else None,
        },
        "attributable_stages": [],
        "unattributable_stages": list(STAGES),
        "why_unattributable": (
            "model_calls[] persists only wall_s/prompt_tokens/completion_tokens. "
            "The connector's prefill/decode/gpu_ns envelope is computed per call "
            "and discarded. With zero pure-prefill samples, prefill and decode "
            "are not separable: a least-squares split of wall on (prompt, completion) "
            "returns a negative fixed term, which is unphysical."
        ),
    }


def paired() -> dict:
    """The measurement that discharges G002. Requires an unowned body."""
    live, detail = _resident_is_live()
    if live:
        raise SystemExit(
            "REFUSED: --paired needs the only model body on this box to be free.\n"
            f"{detail}\n"
            "direct_tok_s and hcli_tok_s must hit the SAME weights to be paired. "
            "The resident's body is a pipe-attached subprocess with no listening "
            "socket, so a direct arm would mean launching a second ~11 GB body. "
            "Stop the resident first, then re-run. This producer will not "
            "estimate the direct arm."
        )
    raise SystemExit(
        "NOT IMPLEMENTED: the direct arm has never been runnable on this box.\n"
        "Implement it here against hcli.hawking_native with the resident stopped, "
        "reading native_prefill_ns/native_decode_ns from the connector's metrics "
        "envelope. Until then G002 stays red, which is the honest state."
    )


def _write(doc: dict) -> None:
    """Emit only a receipt that would honestly pass G002."""
    stages = doc.get("stages") or {}
    missing = [s for s in STAGES if s not in stages]
    if missing:
        raise SystemExit(f"REFUSED to write {RECEIPT}: unmeasured stages {missing}")
    for field in ("direct_tok_s", "hcli_tok_s"):
        value = doc.get(field)
        if not isinstance(value, (int, float)) or value == 0:
            raise SystemExit(f"REFUSED to write {RECEIPT}: {field} is not a measurement")
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {RECEIPT}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--attribute", action="store_true", help="durable-evidence attribution; no body needed")
    g.add_argument("--paired", action="store_true", help="full paired measurement; needs a free body")
    args = ap.parse_args()

    if args.attribute:
        report = attribute()
        report["produced_by"] = PRODUCED_BY
        report["produced_at"] = _now()
        report["command"] = f"python3 {PRODUCED_BY} --attribute"
        report["schema"] = SCHEMA + ".attribution"
        print(json.dumps(report, indent=2, sort_keys=True))
        print(
            "\nNo receipt written. This attribution covers 0 of 12 stages and "
            "neither paired rate; G002 stays red.",
            file=sys.stderr,
        )
        return 0

    _write(paired())
    return 0


if __name__ == "__main__":
    sys.exit(main())
