#!/usr/bin/env python3
"""Canonical paired TPS benchmark for raw and direct-organ KIMI paths.

Prefill is measured as a separate full-forward boundary.  Generation timing
uses ``generate_step`` and reports first-yield-plus-first-decode separately
from steady decode, so no microkernel number is presented as end-to-end TPS.
The candidate replaces only layer-10 routed gate/up/down organs; all other
KIMI_BASE weights remain dense and the receipt says so explicitly.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools" / "future") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools" / "future"))

from tools.future import gravity_outlier_eval as G  # noqa: E402
from tools.future.kimi_nova_routed_kernel import (  # noqa: E402
    ExpertLowRankRoutedLinear,
    Int8ExpertLowRankRoutedLinear,
)

SCHEMA = "hawking.future.kimi_canonical_tps_benchmark.v1"
CONTEXT_REPEATS = (0, 64, 256)
REPS = 2
MAX_TOKENS = 64
EXPECTED_GB = 32.0


def _prompt(repeats: int) -> str:
    return (
        "You are participating in a fixed physical throughput benchmark. "
        + ("Context token. " * repeats)
        + " Reply with one short sentence describing the benchmark."
    )


def _select_under_one(receipt_path: Path):
    receipt = json.loads(receipt_path.read_text())
    eligible = [row for row in receipt["results"]
                if row["representation"]["complete_accounting"]["complete_ebpw"] <= 1.0]
    if not eligible:
        raise ValueError(f"no under-1 candidate in {receipt_path}")
    best = min(eligible, key=lambda row: row["heldout"]["mean_relative_l2"])
    artifact = Path(best["factor_artifact"]["path"])
    if not artifact.is_absolute():
        artifact = ROOT / artifact
    if receipt["schema"].startswith("hawking.future.kimi_int8_expert_lowrank_fanout"):
        consumer = Int8ExpertLowRankRoutedLinear.from_int8_artifact(
            artifact, int(receipt["expert_count"])
        )
        runtime = "row-wise INT8 factors + FP32 scales + two-stage gather_mm"
    else:
        consumer = ExpertLowRankRoutedLinear.from_bf16_artifact(
            artifact, int(receipt["expert_count"])
        )
        runtime = "BF16 independent factors + two-stage gather_mm"
    return receipt, best, artifact, consumer, runtime


def _prefill(model, tok, prompt_text: str, mx) -> dict[str, Any]:
    ids = mx.array([tok.encode(prompt_text)])
    started = time.perf_counter()
    logits = model(ids)
    mx.eval(logits)
    elapsed = time.perf_counter() - started
    tokens = int(ids.shape[-1])
    return {
        "prompt_tokens": tokens,
        "prefill_ms": elapsed * 1000.0,
        "prefill_forward_tps": tokens / elapsed if elapsed else None,
        "boundary": "full model forward without KV-cache reuse; separate from generation first-yield",
    }


def _generation(model, tok, prompt_text: str, max_tokens: int, mx) -> dict[str, Any]:
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
    first_s = first_yield_s or total_s
    steady_s = max(0.0, total_s - first_s)
    n = len(tokens)
    steady_n = max(0, n - 1)
    return {
        "prompt_tokens": int(prompt.size),
        "completion_tokens": n,
        "first_yield_ms": first_s * 1000.0,
        "total_generation_ms": total_s * 1000.0,
        "steady_decode_ms": steady_s * 1000.0,
        "prefill_plus_first_decode_tps": prompt.size / first_s if first_s else None,
        "steady_decode_tps": steady_n / steady_s if steady_s and steady_n else None,
        "completion_end_to_end_tps": n / total_s if total_s else None,
        "token_count_contract": "all generated tokens yielded before EOS or max_tokens",
    }


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return statistics.median(values) if values else None


def _run_state(model, tok, state: str, max_tokens: int, reps: int, mx) -> dict[str, Any]:
    warmup_prompt = _prompt(0)
    warmup_started = time.perf_counter()
    warmup_prefill = _prefill(model, tok, warmup_prompt, mx)
    warmup_generation = _generation(model, tok, warmup_prompt, min(4, max_tokens), mx)
    warmup_s = time.perf_counter() - warmup_started
    contexts = []
    for repeats in CONTEXT_REPEATS:
        rows = []
        for rep in range(reps):
            prompt_text = _prompt(repeats)
            prefill = _prefill(model, tok, prompt_text, mx)
            generation = _generation(model, tok, prompt_text, max_tokens, mx)
            rows.append({"rep": rep, "context_repeats": repeats,
                         "prefill": prefill, "generation": generation})
        contexts.append({
            "context_repeats": repeats,
            "rows": rows,
            "summary": {
                "rep_count": len(rows),
                "prompt_tokens_median": _median([r["generation"] for r in rows], "prompt_tokens"),
                "completion_tokens_median": _median([r["generation"] for r in rows], "completion_tokens"),
                "prefill_ms_median": _median([r["prefill"] for r in rows], "prefill_ms"),
                "prefill_forward_tps_median": _median([r["prefill"] for r in rows], "prefill_forward_tps"),
                "first_yield_ms_median": _median([r["generation"] for r in rows], "first_yield_ms"),
                "steady_decode_ms_median": _median([r["generation"] for r in rows], "steady_decode_ms"),
                "prefill_plus_first_decode_tps_median": _median([r["generation"] for r in rows], "prefill_plus_first_decode_tps"),
                "steady_decode_tps_median": _median([r["generation"] for r in rows], "steady_decode_tps"),
                "completion_end_to_end_tps_median": _median([r["generation"] for r in rows], "completion_end_to_end_tps"),
            },
        })
    return {"state": state, "warmup_s": warmup_s, "warmup_prefill": warmup_prefill,
            "warmup_generation": warmup_generation, "contexts": contexts}


def _active_bytes_per_token(selected: dict[str, Any], top_k: int = 2) -> dict[str, Any]:
    rows = []
    total_active = 0
    for family, info in selected.items():
        rep = info["best"]["representation"]
        accounting = rep["complete_accounting"]
        expert_count = int(info["expert_count"])
        persistent = int(accounting["executable_bytes"])
        per_expert = persistent // expert_count
        active = per_expert * top_k
        rows.append({"family": family, "expert_count": expert_count,
                     "persistent_executable_bytes": persistent,
                     "active_factor_bytes_per_selected_expert": per_expert,
                     "top_k": top_k, "active_factor_bytes_per_token": active})
        total_active += active
    return {"top_k": top_k, "by_projection": rows,
            "active_factor_bytes_per_token_total": total_active,
            "accounting_note": "metadata/runtime support are persistent broadcast bytes; active value charges selected expert factor rows only"}


def run(args) -> dict[str, Any]:
    import mlx.core as mx
    from tools.future.campaign_memory_guard import require_ok, resource_cost, sample, watch

    entry = require_ok("canonical KIMI TPS benchmark", expected_gb=EXPECTED_GB).as_dict()
    load_started = time.perf_counter()
    model, tok = G._load()
    load_s = time.perf_counter() - load_started
    loaded = sample(expected_gb=EXPECTED_GB).as_dict()
    selected: dict[str, Any] = {}
    consumers: dict[str, Any] = {}
    for family, path in {"gate_proj": args.gate_receipt,
                         "up_proj": args.up_receipt,
                         "down_proj": args.down_receipt}.items():
        receipt, best, artifact, consumer, runtime = _select_under_one(path)
        selected[family] = {"receipt": str(path), "best": best,
                            "expert_count": int(receipt["expert_count"]),
                            "artifact": {"path": str(artifact),
                                         "sha256": best["factor_artifact"]["sha256"]},
                            "runtime": runtime}
        consumers[family] = consumer

    # Baseline and candidate share one loaded KIMI_BASE graph so the paired
    # comparison does not mix model-load or machine-state effects.
    with (resource_cost(interval_s=2.0, label="canonical KIMI TPS benchmark") as resource,
          watch(interval_s=2.0) as live_watch):
        baseline = _run_state(model, tok, "raw_KIMI_BASE", args.max_tokens, args.reps, mx)
        switch = model.layers[10].mlp.switch_mlp
        switch.gate_proj = consumers["gate_proj"]
        switch.up_proj = consumers["up_proj"]
        switch.down_proj = consumers["down_proj"]
        patched = _run_state(model, tok, "layer10_three_projection_direct_organs", args.max_tokens, args.reps, mx)
    end = sample().as_dict()
    capability_receipt = json.loads(args.capability_receipt.read_text()) if args.capability_receipt else None
    out = {
        "schema": SCHEMA,
        "status": "CANONICAL_TPS_BENCHMARK_COMPLETE",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "model_snapshot": G.SNAP,
        "lineage": {"source": "KIMI_BASE", "organ_scope": "layer10 routed gate/up/down",
                     "candidate_receipts": {k: str(v["receipt"]) for k, v in selected.items()},
                     "capability_receipt": str(args.capability_receipt) if args.capability_receipt else None},
        "representation": {
            "scope": "three-projection organ integration only; unreplaced model remains dense",
            "selected": selected,
            "active_bytes": _active_bytes_per_token(selected, top_k=2),
            "direct_execution": True,
            "dense_parent_materialized_by_candidate_consumers": False,
            "full_model_direct_execution": False,
        },
        "measurement_contract": {
            "contexts": list(CONTEXT_REPEATS), "reps_per_context": args.reps,
            "max_tokens": args.max_tokens, "warmup_excluded": True,
            "batch": 1, "concurrency": 1, "sampling": "generate_step default greedy",
            "cpu_gpu_assignment": str(mx.default_device()),
            "prefill": "full model forward on prompt without KV-cache reuse; reported separately",
            "generation": "mlx_lm.generate_step; first_yield includes first decode, steady excludes it",
            "microkernel_claim": False,
        },
        "timing": {"model_load_s": load_s, "baseline": baseline, "patched": patched},
        "resource": {"entry": entry, "loaded": loaded, "measurement_end": end,
                      "bracket": resource, "watch_worst_state": live_watch.worst,
                      "watch_samples": live_watch.samples},
        "capability": {
            "status": "PATCHED_FAILS_CAPABILITY_GATE",
            "source_receipt": str(args.capability_receipt) if args.capability_receipt else None,
            "patched_median_fourgram_repeat_fraction": (
                capability_receipt["capability_gate"]["patched"]["median_fourgram_repeat_fraction"]
                if capability_receipt else None
            ),
            "gate_limit": G.R4_MAX,
        },
        "claim_boundary": (
            "Canonical paired physical timing for raw KIMI_BASE and a three-organ direct consumer. "
            "The 0.9618 complete-EBPW figure is the down-organ scope, not a full-model body rate. "
            "Other model weights remain dense; this does not establish a full reduced Star, "
            "capability preservation, resident adoption, NX, or promotion."
        ),
        "teardown": {"planned": "synchronize, release references, clear MLX cache",
                     "status": "CLEANUP_ATTEMPTED",
                     "process_exit_mode": "controlled_os_exit_0" if args.controlled_exit else "normal_interpreter_exit"},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
        "script": str(Path(__file__)),
    }
    mx.synchronize()
    del consumers, switch, model, tok
    mx.clear_cache()
    out["teardown"]["status"] = "SYNCHRONIZED_RELEASED_CACHE_CLEARED"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, args.output)
    print(json.dumps({"output": str(args.output), "status": out["status"],
                      "load_s": load_s,
                      "baseline": [c["summary"] for c in baseline["contexts"]],
                      "patched": [c["summary"] for c in patched["contexts"]],
                      "active_bytes": out["representation"]["active_bytes"],
                      "watch_worst_state": live_watch.worst,
                      "teardown": out["teardown"]}, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate-receipt", type=Path, required=True)
    ap.add_argument("--up-receipt", type=Path, required=True)
    ap.add_argument("--down-receipt", type=Path, required=True)
    ap.add_argument("--capability-receipt", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--controlled-exit", action="store_true",
                    help="use os._exit(0) after receipt commit to bypass native MLX interpreter teardown")
    args = ap.parse_args()
    if args.reps <= 0 or args.max_tokens <= 0:
        raise SystemExit("reps and max-tokens must be positive")
    run(args)
    if args.controlled_exit:
        sys.stdout.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
