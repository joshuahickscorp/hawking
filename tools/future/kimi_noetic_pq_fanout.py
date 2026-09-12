#!/usr/bin/env python3
"""Fan out measured direct PQ references for one KIMI organ tensor.

This is an organ-level discriminator.  Each candidate is fit and executed
independently against the same immutable tensor and input vector.  The fanout
records wall time (fit plus reference execution), direct matvec time, dense
oracle time, complete accounting, and output error.  A candidate only becomes
interesting when it is both physically executable and accurate enough for the
next capability experiment; sub-1 accounting alone is not a promotion claim.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.future.kimi_noetic_pq_reference import (  # noqa: E402
    DEFAULT_SPEC,
    DEFAULT_TENSOR,
    run,
)

SCHEMA = "hawking.future.kimi_noetic_pq_fanout.v1"
METHOD_SOFT_BUDGET_MS = 20000.0
VARIANTS = (
    {"name": "pq_d16_e16_s2", "d": 16, "entries": 16, "steps": 2},
    {"name": "pq_d16_e32_s2", "d": 16, "entries": 32, "steps": 2},
    {"name": "pq_d32_e16_s2", "d": 32, "entries": 16, "steps": 2},
    {"name": "pq_d32_e32_s2", "d": 32, "entries": 32, "steps": 2},
    {"name": "pq_d64_e16_s2", "d": 64, "entries": 16, "steps": 2},
    {"name": "pq_d64_e32_s2", "d": 64, "entries": 32, "steps": 2},
)


def _one(spec: Path, tensor: str, variant: dict[str, int | str], reps: int) -> dict[str, Any]:
    started = time.perf_counter()
    result = run(
        spec,
        tensor,
        d=int(variant["d"]),
        steps=int(variant["steps"]),
        reps=reps,
        entries=int(variant["entries"]),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result["fanout"] = {
        "variant": variant,
        "elapsed_ms": elapsed_ms,
        "soft_budget_ms": METHOD_SOFT_BUDGET_MS,
        "soft_budget_status": "WITHIN" if elapsed_ms <= METHOD_SOFT_BUDGET_MS else "OVER",
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--tensor", default=DEFAULT_TENSOR)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--workers", type=int, default=len(VARIANTS))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.reps <= 0 or args.workers <= 0:
        raise SystemExit("reps and workers must be positive")

    started = time.perf_counter()
    workers = min(args.workers, len(VARIANTS))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, args.spec, args.tensor, variant, args.reps) for variant in VARIANTS]
        results = [future.result() for future in futures]
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    def summary(result: dict[str, Any]) -> dict[str, Any]:
        accounting = result["representation"]["complete_accounting"]
        return {
            "variant": result["fanout"]["variant"],
            "complete_ebpw": accounting["complete_ebpw"],
            "relative_l2": result["error"]["relative_l2"],
            "cosine": result["error"]["cosine"],
            "direct_median_ms": result["timing"]["direct_median_ms"],
            "dense_oracle_median_ms": result["timing"]["dense_oracle_median_ms"],
            "elapsed_ms": result["fanout"]["elapsed_ms"],
            "direct_verified": result["direct_execution"]["verified"],
        }

    out: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "DIRECT_REFERENCE_FANOUT_COMPLETE",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "specimen": str(args.spec),
        "tensor": args.tensor,
        "variants": [result["fanout"]["variant"] for result in results],
        "results": results,
        "fanout_timing": {
            "workers": workers,
            "variant_count": len(results),
            "wall_elapsed_ms": elapsed_ms,
            "sum_variant_elapsed_ms": sum(result["fanout"]["elapsed_ms"] for result in results),
            "execution": "CPU direct-reference fanout; not model TPS",
            "method_soft_budget_ms": METHOD_SOFT_BUDGET_MS,
        },
        "ranked_summary": sorted(
            (summary(result) for result in results),
            key=lambda row: (row["complete_ebpw"], row["relative_l2"]),
        ),
        "claim_boundary": (
            "One-tensor CPU reference fanout only. It validates a family of no-full-dense "
            "direct-consumer mechanisms and their accounting/error/timing tradeoffs. It is "
            "not a fitted shared-organ representation, model capability result, end-to-end "
            "TPS result, production kernel qualification, or KIMI promotion."
        ),
        "next_discriminator": (
            "Use the Pareto survivors to fit genuinely shared codebooks across the selected "
            "MoE organ, add output-aware protected residuals, and measure capability before "
            "body-wide compilation."
        ),
    }
    out["generated_at"] = datetime.now(timezone.utc).isoformat()
    out["python"] = platform.python_version()
    out["script"] = str(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": out["status"],
        "wall_elapsed_ms": elapsed_ms,
        "ranked_summary": out["ranked_summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
