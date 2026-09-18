#!/usr/bin/env python3
"""Measure the raw KIMI_BASE MLX generation path with explicit timing boundaries.

This is a physical baseline, not a capability or promotion result.  The MLX
``generate_step`` API exposes the first yielded token after prefill, so the
first-yield boundary necessarily includes that token's decode work.  We report
that boundary separately from steady decode instead of calling it pure
prefill.  The existing live derivative server is not touched.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tools" / "future") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools" / "future"))

import gravity_outlier_eval as G  # noqa: E402

SCHEMA = "hawking.future.kimi_raw_tps_baseline.v1"
CONTEXT_REPEATS = (0, 64, 256)
REPS = 2
MAX_TOKENS = 32
EXPECTED_GB = 32.0


def _prompt(repeats: int) -> str:
    return (
        "You are participating in a fixed physical throughput benchmark. "
        + ("Context token. " * repeats)
        + " Reply with one short sentence describing the benchmark."
    )


def _snapshot(expected_gb: float = 0.0) -> dict[str, Any]:
    from campaign_memory_guard import sample

    return sample(expected_gb=expected_gb).as_dict()


def _measure(model, tok, prompt_text: str, max_tokens: int) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    prompt = mx.array(tok.encode(prompt_text))
    started = time.perf_counter()
    generator = generate_step(prompt, model, max_tokens=max_tokens)
    tokens: list[int] = []
    first_yield_s: float | None = None
    for token, _logprobs in generator:
        if first_yield_s is None:
            first_yield_s = time.perf_counter() - started
        tokens.append(int(token))

    total_s = time.perf_counter() - started
    n = len(tokens)
    # The first-yield interval is the only boundary the public generator gives
    # us for prefill.  It includes first-token decode; keep it named honestly.
    first_s = first_yield_s or total_s
    steady_s = max(0.0, total_s - first_s)
    steady_n = max(0, n - 1)
    return {
        "prompt_tokens": int(prompt.size),
        "completion_tokens": n,
        "first_yield_ms": round(first_s * 1000.0, 3),
        "total_generation_ms": round(total_s * 1000.0, 3),
        "steady_decode_ms": round(steady_s * 1000.0, 3),
        "prefill_plus_first_decode_tps": round(prompt.size / first_s, 3) if first_s else None,
        "steady_decode_tps": round(steady_n / steady_s, 3) if steady_s and steady_n else None,
        "completion_end_to_end_tps": round(n / total_s, 3) if total_s else None,
        "token_count_contract": "all generated tokens yielded before EOS or max_tokens",
    }


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return statistics.median(values) if values else None


def run(reps: int, max_tokens: int, output: Path) -> dict[str, Any]:
    from campaign_memory_guard import require_ok, resource_cost, watch

    started = time.perf_counter()
    entry = require_ok("raw KIMI_BASE TPS baseline", expected_gb=EXPECTED_GB)
    load_started = time.perf_counter()
    model, tok = G._load()
    load_s = time.perf_counter() - load_started
    loaded = _snapshot(EXPECTED_GB)

    # A short warmup is excluded from all reported rows.  It makes the later
    # rows a steady resident measurement without conflating first-use setup.
    warmup_started = time.perf_counter()
    warmup = _measure(model, tok, _prompt(0), min(4, max_tokens))
    warmup_s = time.perf_counter() - warmup_started
    warm = _snapshot(EXPECTED_GB)

    contexts: list[dict[str, Any]] = []
    with (
        resource_cost(interval_s=2.0, label="raw KIMI_BASE timing") as resource,
        watch(interval_s=2.0) as live_watch,
    ):
        for context_repeats in CONTEXT_REPEATS:
            rows: list[dict[str, Any]] = []
            for rep in range(reps):
                row = _measure(model, tok, _prompt(context_repeats), max_tokens)
                row.update({"rep": rep, "context_repeats": context_repeats})
                rows.append(row)
            contexts.append({
                "context_repeats": context_repeats,
                "rows": rows,
                "summary": {
                    "rep_count": len(rows),
                    "prompt_tokens_median": _median(rows, "prompt_tokens"),
                    "completion_tokens_median": _median(rows, "completion_tokens"),
                    "first_yield_ms_median": _median(rows, "first_yield_ms"),
                    "total_generation_ms_median": _median(rows, "total_generation_ms"),
                    "steady_decode_ms_median": _median(rows, "steady_decode_ms"),
                    "prefill_plus_first_decode_tps_median": _median(rows, "prefill_plus_first_decode_tps"),
                    "steady_decode_tps_median": _median(rows, "steady_decode_tps"),
                    "completion_end_to_end_tps_median": _median(rows, "completion_end_to_end_tps"),
                },
            })
    end = _snapshot()
    return {
        "schema": SCHEMA,
        "status": "RAW_KIMI_BASE_TIMING_COMPLETE_NOT_QUALIFIED",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "model_snapshot": G.SNAP,
        "measurement_contract": {
            "contexts": list(CONTEXT_REPEATS),
            "reps_per_context": reps,
            "warmup_excluded": True,
            "max_tokens": max_tokens,
            "sampling": "greedy generate_step default",
            "concurrency": 1,
            "api": "mlx_lm.generate_step",
            "prefill_decode_boundary": (
                "first_yield_ms includes prompt processing plus the first decode step; "
                "steady_decode_ms excludes that first-yield interval"
            ),
        },
        "timing": {"load_s": round(load_s, 3), "warmup_s": round(warmup_s, 3), "warmup": warmup,
                   "total_s": round(time.perf_counter() - started, 3)},
        "resource": {"entry": entry.as_dict(), "loaded": loaded, "warm": warm,
                      "measurement_end": end, "bracket": resource,
                      "watch_worst_state": live_watch.worst,
                      "watch_samples": live_watch.samples},
        "contexts": contexts,
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
        "claim_boundary": (
            "Raw KIMI_BASE physical generation timing only. The cached local snapshot was used; "
            "this is not a CorpDrive cold-load measurement, not a direct Noetic decoder, not a "
            "capability-preservation receipt, not a qualified TPS milestone, and not promotion "
            "or Odyssey admission evidence."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.reps <= 0 or args.max_tokens <= 0:
        raise SystemExit("reps and max-tokens must be positive")
    out = run(args.reps, args.max_tokens, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, args.output)
    print(json.dumps({"output": str(args.output), "status": out["status"],
                      "load_s": out["timing"]["load_s"],
                      "contexts": [c["summary"] for c in out["contexts"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
