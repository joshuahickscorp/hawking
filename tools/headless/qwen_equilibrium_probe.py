#!/usr/bin/env python3
"""Measure the local Qwen decode equilibrium against a live llama-server.

Aggregate decode throughput as a function of concurrent active decodes.
Concurrency is only a lever if aggregate tokens/second rises with it; on this
box prior receipts put the ceiling near 1.2x a single decoder, and this probe
exists to re-measure that against the runtime actually serving HCLI rather than
to restate it.

Contamination control: rungs are run in **alternating paired reps**, not one
block per rung. A single run of each rung is page-cache and thermal confounded;
interleaving them makes the drift common-mode. The spread across reps is
reported, because a tight spread is itself evidence.

Read-only with respect to the repository. It sends inference requests to an
already-running server and writes one receipt.

    python3 tools/headless/qwen_equilibrium_probe.py --rungs 1,2 --reps 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

# Fixed, deterministic, and long enough that decode dominates prefill.
DECODE_PROMPT = (
    "Count upward from one, writing each number as an English word on its own "
    "line, starting at one and continuing without stopping or commenting."
)


def _post(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def runtime_identity(port: int) -> Dict[str, Any]:
    """Exact identity of the runtime under test. A number without this is noise."""
    url = f"http://127.0.0.1:{port}/props"
    with urllib.request.urlopen(url, timeout=10) as resp:
        props = json.loads(resp.read().decode("utf-8"))
    gen = props.get("default_generation_settings") or {}
    return {
        "port": port,
        "model_path": props.get("model_path"),
        "total_slots": props.get("total_slots"),
        "per_slot_n_ctx": gen.get("n_ctx"),
        "build": props.get("build_info"),
    }


def one_decode(port: int, max_tokens: int, timeout: float) -> Dict[str, Any]:
    started = time.perf_counter()
    body = _post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        {
            "model": "local",
            "messages": [{"role": "user", "content": DECODE_PROMPT}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout,
    )
    wall = time.perf_counter() - started
    usage = body.get("usage") or {}
    return {
        "wall_s": wall,
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "finish_reason": (body.get("choices") or [{}])[0].get("finish_reason"),
    }


def run_rung(port: int, concurrency: int, max_tokens: int, timeout: float) -> Dict[str, Any]:
    """Fire `concurrency` decodes at once; measure the aggregate, not the mean."""
    results: List[Optional[Dict[str, Any]]] = [None] * concurrency
    errors: List[str] = []

    def worker(i: int) -> None:
        try:
            results[i] = one_decode(port, max_tokens, timeout)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall_start

    got = [r for r in results if r]
    total_completion = sum(r["completion_tokens"] for r in got)
    return {
        "concurrency": concurrency,
        "wall_s": round(wall, 4),
        "completed": len(got),
        "errors": errors,
        "total_completion_tokens": total_completion,
        "aggregate_tps": round(total_completion / wall, 4) if wall > 0 else 0.0,
        "per_stream_tps": [
            round(r["completion_tokens"] / r["wall_s"], 4) for r in got if r["wall_s"] > 0
        ],
        "finish_reasons": sorted({str(r["finish_reason"]) for r in got}),
    }


def spawn_pool(slots: int):
    """Bring up HCLI's own runtime so the measurement is of the real thing.

    Slot topology means one llama-server process with `slots` sequences, so the
    weights are paid for once regardless of the rung.
    """
    from hcli.models import ModelRegistry
    from hcli.runtime import RuntimePool

    found = ModelRegistry().discover()
    if not found:
        raise RuntimeError("no model discovered")
    first = found[0]
    model = getattr(first, "path", None) or (
        first.get("path") if isinstance(first, dict) else first
    )
    os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = str(slots)
    pool = RuntimePool(str(model), requested_n=slots, workspace=str(REPO_ROOT))
    pool.start()
    live = [r for r in pool.runtimes if getattr(r, "active", False)]
    if not live:
        pool.stop()
        raise RuntimeError(f"pool admitted nothing: {pool.refusal_reason}")
    return pool, live, str(model)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=0,
                    help="attach to an existing server; 0 spawns HCLI's own pool")
    ap.add_argument("--slots", type=int, default=2,
                    help="sequences to allocate when spawning")
    ap.add_argument("--rungs", default="1,2", help="comma-separated concurrency rungs")
    ap.add_argument("--reps", type=int, default=3, help="alternating paired repetitions")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default="receipts/headless/QWEN_MAX_EQUILIBRIUM.json")
    args = ap.parse_args(argv)

    rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    pool = None
    if args.port:
        port = args.port
        model = None
    else:
        pool, live, model = spawn_pool(max(args.slots, max(rungs)))
        port = live[0].port
        print(f"spawned pool: port={port} slots={len(live)} pid={live[0].pid}", flush=True)
    args.port = port
    identity = runtime_identity(port)
    identity["spawned_by_hcli"] = pool is not None
    identity["model_path"] = identity.get("model_path") or model

    # One discarded warm-up so the first measured rung is not paying for a cold
    # slot; the rung order is then interleaved across reps.
    one_decode(args.port, 32, args.timeout)

    samples: List[Dict[str, Any]] = []
    for rep in range(args.reps):
        for c in rungs:
            rec = run_rung(args.port, c, args.max_tokens, args.timeout)
            rec["rep"] = rep
            samples.append(rec)
            print(
                f"rep={rep} c={c} agg_tps={rec['aggregate_tps']:.3f} "
                f"wall={rec['wall_s']:.2f}s completed={rec['completed']} "
                f"errors={len(rec['errors'])}",
                flush=True,
            )

    by_rung: Dict[int, Dict[str, Any]] = {}
    for c in rungs:
        vals = [s["aggregate_tps"] for s in samples if s["concurrency"] == c and not s["errors"]]
        if not vals:
            by_rung[c] = {"reps": 0, "note": "no clean samples"}
            continue
        by_rung[c] = {
            "reps": len(vals),
            "median_aggregate_tps": round(statistics.median(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "spread_pct": round((max(vals) - min(vals)) / statistics.median(vals) * 100, 3),
        }

    base = by_rung.get(rungs[0], {}).get("median_aggregate_tps")
    for c in rungs:
        med = by_rung[c].get("median_aggregate_tps")
        if base and med:
            by_rung[c]["scaling_vs_rung1"] = round(med / base, 4)

    # The equilibrium is the highest rung that actually improved on its
    # predecessor by more than the observed measurement spread.
    equilibrium = rungs[0]
    for prev, cur in zip(rungs, rungs[1:]):
        p = by_rung.get(prev, {}).get("median_aggregate_tps")
        q = by_rung.get(cur, {}).get("median_aggregate_tps")
        spread = max(
            by_rung.get(prev, {}).get("spread_pct", 0.0),
            by_rung.get(cur, {}).get("spread_pct", 0.0),
        )
        if p and q and (q - p) / p * 100 > spread:
            equilibrium = cur
        else:
            break

    receipt = {
        "gate": "QWEN_MAX_EQUILIBRIUM",
        "probe": "qwen_equilibrium_probe",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_identity": identity,
        "protocol": {
            "rungs": rungs,
            "reps": args.reps,
            "max_tokens": args.max_tokens,
            "ordering": "alternating paired reps (interleaved, not blocked)",
            "warmup": "one discarded 32-token decode",
            "prompt": DECODE_PROMPT,
        },
        "samples": samples,
        "by_rung": by_rung,
        "measured_equilibrium_active_decodes": equilibrium,
        "equilibrium_rule": (
            "highest rung whose median aggregate tok/s beats its predecessor by more "
            "than the observed measurement spread"
        ),
    }

    if pool is not None:
        pool.stop()

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"by_rung": by_rung, "equilibrium": equilibrium}, indent=2))
    print(f"receipt: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
