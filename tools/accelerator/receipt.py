"""Accelerator receipt schema. FRONT A (G043, steer S015 §79).

The steer's rule is blunt: NO RESULT WITHOUT PHYSICAL IDENTITY. Every Accelerator
receipt carries eight identities. Where one genuinely does not apply -- there is no
TransportIdentity for a single-device Metal run -- it is recorded ABSENT with a
reason, never omitted and never invented. That is the same discipline the kernel
library already uses, and it is what keeps a missing field from reading as a
covered one.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "hawking.accelerator.receipt.v1"

IDENTITIES = ("experiment", "machine", "device", "model",
              "representation", "kernel", "runtime", "transport")

# The steer's canonical experiment classes. A receipt may not invent a class.
EXPERIMENT_CLASSES = {
    "ACCEL-KERNEL", "ACCEL-FUSION", "ACCEL-LAYOUT", "ACCEL-MEMORY",
    "ACCEL-DISPATCH", "ACCEL-SCHEDULING", "ACCEL-REPRESENTATION", "ACCEL-STATE",
    "ACCEL-DEVICE", "ACCEL-C2M", "ACCEL-EGB", "ACCEL-HUMF", "ACCEL-SUSTAINED",
}

BENCH_STATES = ("QUIESCED", "CONTENDED", "UNKNOWN")

# S032 §3: quiescence is a BENCHMARK INPUT, not a footnote. Any receipt that
# quotes a duration, a rate or a ratio-of-durations is a performance receipt and
# must carry the machine state it was measured under.
#
# The rule that forces the issue is the steer's own: "If quiescence is unknown:
# BENCH_STATE = UNKNOWN, not quiet." A receipt with no bench block at all reads
# as quiet to every downstream reader, which is exactly the claim it never made.
_TIMING_SUFFIXES = ("_ns", "_us", "_ms", "_s", "_sec", "_seconds", "_tps",
                    "_gbps", "_gib_s", "_hz", "_pct_faster", "_speedup")
_TIMING_NAMES = {"tps", "speedup", "latency", "wall", "throughput", "gbps",
                 "ns_per_token", "us_per_dispatch", "median_s", "p50", "p95"}


def _timing_keys(node, path="result") -> list[str]:
    """Every key in the result tree that quotes time, rate, or a speed ratio."""
    found = []
    if isinstance(node, dict):
        for k, v in node.items():
            lk = str(k).lower()
            if lk in _TIMING_NAMES or lk.endswith(_TIMING_SUFFIXES):
                found.append(f"{path}.{k}")
            found += _timing_keys(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            found += _timing_keys(v, f"{path}[{i}]")
    return found


def _check_bench(bench, result) -> None:
    timing = _timing_keys(result)
    if bench is None:
        if timing:
            raise ValueError(
                f"this receipt quotes timing at {timing[:5]} but carries no bench "
                f"block. S032 §3: a performance receipt records the machine state "
                f"it was measured under, and an absent state reads as QUIESCED to "
                f"every downstream reader. Pass bench=bench.bench_block(...) or, if "
                f"the state genuinely is not known, bench_state UNKNOWN.")
        return
    state = bench.get("state")
    if state not in BENCH_STATES:
        raise ValueError(f"bench state {state!r} is not one of {BENCH_STATES}")
    for k in ("recorded_at", "machine"):
        if not bench.get(k):
            raise ValueError(f"bench block is missing {k!r}; a state without a "
                             f"timestamp and a machine identity is not a measurement")
    # QUIESCED is the only state that is a CLAIM. It has to be earned.
    if state == "QUIESCED":
        q = bench.get("quiescence")
        if not isinstance(q, dict) or q.get("quiet") is not True:
            raise ValueError(
                "bench state QUIESCED without an enumerating quiescence sample "
                "reporting quiet=True. Unknown is UNKNOWN, never quiet.")
        if q.get("n_contenders"):
            raise ValueError(
                f"bench state QUIESCED while {q['n_contenders']} contenders are "
                f"recorded: {[c.get('comm') for c in q.get('contenders') or []][:4]}")
        # ...AND THE SUMMARY MUST AGREE WITH THE SAMPLES IT SUMMARISES. bench_block
        # derives `quiescence` as the WORST sample, so the two agree by construction
        # -- but a hand-built bench dict can carry a quiet summary over noisy
        # samples, and this program has already shipped one harness that wrote
        # "bench_state": "QUIESCED" as a LITERAL while every one of its six
        # quiescence probes read quiet=False. That harness never reached this
        # function; the hole it revealed is that the summary was trusted alone.
        for side, sample in (bench.get("samples") or {}).items():
            if isinstance(sample, dict) and sample.get("quiet") is not True:
                raise ValueError(
                    f"bench state QUIESCED but the {side!r} sample reports "
                    f"quiet={sample.get('quiet')!r} with "
                    f"{sample.get('n_contenders')} contenders "
                    f"{[c.get('comm') for c in sample.get('contenders') or []][:3]}. "
                    f"A summary that disagrees with its own samples is an ASSERTED "
                    f"state, and S032 §3 requires a derived one.")


# Steer §80. Never promote too early.
KNOWLEDGE_LEVELS = ("INSTANCE", "MODEL_FAMILY", "ARCHITECTURE", "REPRESENTATION",
                    "SOC_FAMILY", "DEVICE_CLASS", "APPLE_GENERAL", "EGB_TOPOLOGY",
                    "GENERAL_PHYSICAL")


def absent(reason: str) -> dict[str, str]:
    return {"status": "ABSENT", "reason": reason}


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return None


def build(*, experiment_class: str, knowledge_level: str, identities: dict[str, Any],
          result: dict[str, Any], claim_boundary: str, passed: bool,
          bench: dict[str, Any] | None = None) -> dict[str, Any]:
    if experiment_class not in EXPERIMENT_CLASSES:
        raise ValueError(f"{experiment_class!r} is not a canonical class; "
                         f"known: {sorted(EXPERIMENT_CLASSES)}")
    if knowledge_level not in KNOWLEDGE_LEVELS:
        raise ValueError(f"{knowledge_level!r} is not a knowledge level")
    missing = [k for k in IDENTITIES if k not in identities]
    if missing:
        raise ValueError(f"receipt is missing identities {missing}; record them "
                         f"ABSENT with a reason rather than omitting them")
    for k in IDENTITIES:
        v = identities[k]
        if isinstance(v, dict) and v.get("status") == "ABSENT" and not v.get("reason"):
            raise ValueError(f"identity {k!r} is ABSENT without a reason")
    _check_bench(bench, result)
    return {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_class": experiment_class,
        "knowledge_level": knowledge_level,
        "git_head": git_head(),
        "identities": {k: identities[k] for k in IDENTITIES},
        "result": result,
        "claim_boundary": claim_boundary,
        "pass": bool(passed),
        # Present as an explicit null when the receipt makes no timing claim, so a
        # reader can tell "not a performance receipt" from "state not recorded".
        "bench": bench,
    }


def write(receipt: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=1))
    return path
