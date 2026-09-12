#!/usr/bin/env python3
"""Derive an auditable KIMI token-budget receipt from canonical TPS timing.

This is a bookkeeping layer over a completed canonical benchmark.  It does
not pretend that a wall-clock timer can split GPU compute, routing, sampling,
and recurrent state movement without counters.  Observable boundaries are
reported; unobservable terms remain explicitly ``NOT_SEPARABLE``.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "receipts/future/KIMI_CANONICAL_TPS_PARETO_RAW_VS_INT8_ROWRELATIVE_20260910.json"
SCHEMA = "hawking.future.kimi_tps_budget_decomposition.v1"
TPS_LADDER = (100.0, 150.0, 250.0, 350.0, 500.0)


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _arm_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    summary = row.get("summary") or {}
    completion = _num(summary.get("completion_tokens_median"))
    steady_ms = _num(summary.get("steady_decode_ms_median"))
    total_ms = _num(summary.get("total_generation_ms_median"))
    prompt_tokens = _num(summary.get("prompt_tokens_median"))
    prefill_ms = _num(summary.get("prefill_ms_median"))
    steady_tokens = max(0.0, completion - 1.0) if completion is not None else None
    steady_ms_per_token = (
        steady_ms / steady_tokens if steady_ms is not None and steady_tokens else None
    )
    e2e_ms_per_token = total_ms / completion if total_ms is not None and completion else None
    prefill_ms_per_prompt_token = (
        prefill_ms / prompt_tokens if prefill_ms is not None and prompt_tokens else None
    )
    return {
        "context_repeats": row.get("context_repeats"),
        "prompt_tokens_median": prompt_tokens,
        "completion_tokens_median": completion,
        "steady_decode_tps_median": _num(summary.get("steady_decode_tps_median")),
        "completion_end_to_end_tps_median": _num(summary.get("completion_end_to_end_tps_median")),
        "prefill_forward_tps_median": _num(summary.get("prefill_forward_tps_median")),
        "first_yield_ms_median": _num(summary.get("first_yield_ms_median")),
        "steady_decode_ms_median": steady_ms,
        "total_generation_ms_median": total_ms,
        "prefill_ms_median": prefill_ms,
        "observable_terms_ms_per_token": {
            "steady_decode": steady_ms_per_token,
            "completion_end_to_end": e2e_ms_per_token,
            "prefill_forward_per_prompt_token": prefill_ms_per_prompt_token,
        },
        "unseparated_terms": [
            "model_compute_gpu",
            "routing_and_expert_gather",
            "representation_decode",
            "attention_or_recurrent_state",
            "sampling",
            "dispatch_and_synchronization",
        ],
    }


def _context_map(arm: Mapping[str, Any]) -> dict[Any, dict[str, Any]]:
    return {row["context_repeats"]: _arm_summary(row) for row in arm.get("contexts", [])}


def _ladder(tps: float | None) -> list[dict[str, Any]]:
    return [
        {
            "target_tps": target,
            "reached": bool(tps is not None and tps >= target),
            "required_speedup_from_current": (target / tps if tps and tps > 0 else None),
            "token_budget_ms": 1000.0 / target,
        }
        for target in TPS_LADDER
    ]


def build(source: Mapping[str, Any], source_path: str) -> dict[str, Any]:
    timing = source.get("timing") or {}
    raw = timing.get("baseline") or {}
    patched = timing.get("patched") or {}
    raw_rows = _context_map(raw)
    patched_rows = _context_map(patched)
    paired: list[dict[str, Any]] = []
    for context in sorted(set(raw_rows) & set(patched_rows)):
        r = raw_rows[context]
        p = patched_rows[context]
        r_ms = r["observable_terms_ms_per_token"]["steady_decode"]
        p_ms = p["observable_terms_ms_per_token"]["steady_decode"]
        paired.append({
            "context_repeats": context,
            "raw": r,
            "patched": p,
            "direct_organ_steady_penalty": {
                "ms_per_steady_token": p_ms - r_ms if p_ms is not None and r_ms is not None else None,
                "relative_wall_penalty": (p_ms / r_ms - 1.0) if p_ms is not None and r_ms else None,
                "interpretation": "paired wall-clock delta; not assigned to one internal term",
            },
        })
    raw_tps = [r.get("steady_decode_tps_median") for r in raw_rows.values()]
    raw_tps = [x for x in raw_tps if x is not None]
    canonical_raw = sum(raw_tps) / len(raw_tps) if raw_tps else None
    active = ((source.get("representation") or {}).get("active_factor_bytes_per_token") or
              (source.get("representation") or {}).get("active_bytes", {}).get("active_factor_bytes_per_token_total"))
    if active is not None:
        active = int(active)
    out = {
        "schema": SCHEMA,
        "status": "DERIVED_FROM_CANONICAL_TIMING",
        "source_receipt": source_path,
        "source_status": source.get("status"),
        "source_model": source.get("source_model"),
        "measurement_contract": source.get("measurement_contract"),
        "canonical_reference": {
            "raw_steady_decode_tps_mean_across_context_medians": canonical_raw,
            "raw_steady_decode_tps_context_medians": raw_tps,
            "raw_tps_ladder": _ladder(canonical_raw),
            "claim": "canonical raw KIMI_BASE timing only; not a qualified capability or promotion result",
        },
        "paired_contexts": paired,
        "representation_accounting": {
            "active_factor_bytes_per_token_top_k2": active,
            "active_bytes_scope": "selected gate/up/down factor rows only; the rest of the model remains dense",
            "effective_active_factor_gib_per_steady_token": active / (1 << 30) if active is not None else None,
            "note": "This is not whole-model bytes/token and cannot be converted into whole-model bandwidth without full graph coverage.",
        },
        "latency_budget": {
            "token_budget_targets_ms": {str(int(target)): 1000.0 / target for target in TPS_LADDER},
            "observable": [
                "first_yield_ms",
                "steady_decode_ms",
                "completion_end_to_end_ms",
                "prefill_forward_ms",
            ],
            "not_separable_with_current_receipt": [
                "GPU model compute",
                "routing/expert gather",
                "representation decode",
                "attention/MLA/DeltaNet state",
                "sampling",
                "dispatch/synchronization",
            ],
            "next_measurement": "matched full-model native counters or instrumented custom kernels, with no microkernel substitution",
        },
        "claim_boundary": (
            "Derived bookkeeping over the canonical paired benchmark. It reports the measured raw "
            "steady decode record and paired direct-organ wall-clock penalty, but does not infer an "
            "internal bottleneck, whole-model active bandwidth, capability preservation, full-model "
            "complete EBPW, resident adoption, Odyssey admission, or promotion."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__)),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    source = json.loads(args.source.read_text())
    out = build(source, str(args.source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.{__import__('os').getpid()}.tmp")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "status": out["status"],
        "raw_tps": out["canonical_reference"],
        "contexts": len(out["paired_contexts"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
