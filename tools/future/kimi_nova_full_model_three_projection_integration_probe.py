#!/usr/bin/env python3
"""Probe a three-projection Nova replacement inside the real KIMI graph."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools" / "future") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools" / "future"))

from tools.future import gravity_outlier_eval as G  # noqa: E402
from tools.future.kimi_nova_full_model_organ_integration_probe import (  # noqa: E402
    _generate,
    _nll,
)
from tools.future.kimi_nova_routed_kernel import (  # noqa: E402
    ExpertLowRankRoutedLinear,
    Int8ExpertLowRankRoutedLinear,
)

SCHEMA = "hawking.future.kimi_nova_full_model_three_projection_integration_probe.v1"
TARGET_LAYER = 10


def _best_consumer(receipt_path: Path, requested_rank: int | None = None):
    receipt = json.loads(receipt_path.read_text())
    eligible = [row for row in receipt["results"]
                if row["representation"]["complete_accounting"]["complete_ebpw"] <= 1.0]
    if requested_rank is not None:
        eligible = [row for row in eligible if int(row["variant"]["rank"]) == requested_rank]
    if not eligible:
        suffix = f" at rank {requested_rank}" if requested_rank is not None else ""
        raise ValueError(f"no under-1 candidate in {receipt_path}{suffix}")
    best = min(eligible, key=lambda row: row["heldout"]["mean_relative_l2"])
    artifact = Path(best["factor_artifact"]["path"])
    if not artifact.is_absolute():
        artifact = ROOT / artifact
    if receipt["schema"].startswith("hawking.future.kimi_int8_expert_lowrank_fanout"):
        consumer = Int8ExpertLowRankRoutedLinear.from_int8_artifact(
            artifact, int(receipt["expert_count"])
        )
        consumer_runtime = "row-wise INT8 factors + FP32 scales"
    else:
        consumer = ExpertLowRankRoutedLinear.from_bf16_artifact(
            artifact, int(receipt["expert_count"])
        )
        consumer_runtime = "BF16 factors"
    return receipt, best, artifact, consumer, consumer_runtime, eligible


def probe(receipt_paths: dict[str, Path], output: Path, max_tokens: int,
          controlled_exit: bool = False,
          rank_overrides: dict[str, int | None] | None = None) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import generate
    from tools.future.campaign_memory_guard import require_ok, sample, watch

    entry = require_ok("KIMI Nova three-projection integration probe", expected_gb=32.0).as_dict()
    load_started = time.perf_counter()
    model, tok = G._load()
    load_s = time.perf_counter() - load_started
    loaded = sample(expected_gb=32.0).as_dict()

    selected: dict[str, Any] = {}
    consumers = {}
    for family, receipt_path in receipt_paths.items():
        requested_rank = (rank_overrides or {}).get(family)
        receipt, best, artifact, consumer, consumer_runtime, eligible = _best_consumer(
            receipt_path, requested_rank=requested_rank
        )
        selected[family] = {
            "receipt": str(receipt_path),
            "variant": best["variant"],
            "artifact": best["factor_artifact"],
            "heldout": best["heldout"],
            "artifact_path": str(artifact),
            "consumer_runtime": consumer_runtime,
            "selection_constraint": "complete_ebpw <= 1.0",
            "requested_rank": requested_rank,
            "eligible_variant_count": len(eligible),
        }
        consumers[family] = consumer

    switch = model.layers[TARGET_LAYER].mlp.switch_mlp
    target_names = {}
    for family in ("gate_proj", "up_proj", "down_proj"):
        target = getattr(switch, family)
        target_names[family] = next(
            (name for name, module in model.named_modules() if module is target),
            f"language_model.model.layers.{TARGET_LAYER}.mlp.switch_mlp.{family}",
        )

    texts = [G.NLL_TEXT, " ".join(G.PROMPTS)]
    baseline_nll = []
    patched_nll = []
    with watch(interval_s=2.0) as live_watch:
        for text in texts:
            baseline_nll.append(_nll(model, tok, text, mx))
        baseline_generation = _generate(model, tok, G.PROMPTS, max_tokens, generate)

        switch.gate_proj = consumers["gate_proj"]
        switch.up_proj = consumers["up_proj"]
        switch.down_proj = consumers["down_proj"]
        for text in texts:
            patched_nll.append(_nll(model, tok, text, mx))
        patched_generation = _generate(model, tok, G.PROMPTS, max_tokens, generate)

    def summarize(values: list[tuple[float, float, float]]) -> dict[str, Any]:
        return {
            "rows": [{"nll": nll, "ppl": ppl, "elapsed_s": elapsed}
                     for nll, ppl, elapsed in values],
            "mean_nll": sum(row[0] for row in values) / len(values),
            "mean_ppl": sum(row[1] for row in values) / len(values),
            "mean_elapsed_s": sum(row[2] for row in values) / len(values),
        }

    baseline_summary = summarize(baseline_nll)
    patched_summary = summarize(patched_nll)
    patched_pass = bool(
        patched_nll[0][1] <= G.PPL_MAX
        and patched_generation["median_fourgram_repeat_fraction"] <= G.R4_MAX
    )
    capability_gate = {
        "reference_r4_max": G.R4_MAX,
        "reference_ppl_max": G.PPL_MAX,
        "baseline": {
            "ppl": baseline_nll[0][1],
            "median_fourgram_repeat_fraction": baseline_generation[
                "median_fourgram_repeat_fraction"
            ],
            "pass": bool(
                baseline_nll[0][1] <= G.PPL_MAX
                and baseline_generation["median_fourgram_repeat_fraction"] <= G.R4_MAX
            ),
        },
        "patched": {
            "ppl": patched_nll[0][1],
            "median_fourgram_repeat_fraction": patched_generation[
                "median_fourgram_repeat_fraction"
            ],
            "pass": bool(
                patched_nll[0][1] <= G.PPL_MAX
                and patched_generation["median_fourgram_repeat_fraction"] <= G.R4_MAX
            ),
        },
        "verdict": "PATCHED_PASSES_CAPABILITY_GATE"
        if patched_pass else "PATCHED_FAILS_CAPABILITY_GATE",
    }
    out = {
        "schema": SCHEMA,
        "status": "NOVA_THREE_PROJECTION_FULL_GRAPH_PROBE_COMPLETE",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "selected_projections": selected,
        "target": {"layer": TARGET_LAYER, "modules": target_names,
                   "replacement": "selected direct independent expert factors"},
        "model_scope": {
            "replaced_organs": 3,
            "replaced_tensor_families": "layer10 routed gate_proj + up_proj + down_proj",
            "other_weights": "unchanged KIMI_BASE dense/MLX weights",
            "full_model_direct_execution": False,
        },
        "guard": {"entry": entry, "loaded": loaded,
                  "watch_worst_state": live_watch.worst,
                  "watch_samples": live_watch.samples,
                  "end": sample().as_dict()},
        "load": {"elapsed_s": load_s, "runtime": "MLX"},
        "workload": {"nll_text_count": len(texts), "generation_prompts": len(G.PROMPTS),
                      "max_tokens": max_tokens, "sampling": "mlx_lm.generate default"},
        "baseline": {"nll": baseline_summary, "generation": baseline_generation},
        "patched": {"nll": patched_summary, "generation": patched_generation},
        "capability_gate": capability_gate,
        "deltas": {
            "mean_nll_patched_minus_baseline": patched_summary["mean_nll"] - baseline_summary["mean_nll"],
            "mean_ppl_ratio": patched_summary["mean_ppl"] / max(baseline_summary["mean_ppl"], 1e-12),
            "generation_median_s_delta": patched_generation["median_s"] - baseline_generation["median_s"],
            "generation_repeat_delta": patched_generation["median_fourgram_repeat_fraction"] - baseline_generation["median_fourgram_repeat_fraction"],
            "generation_unique_fraction_delta": patched_generation["median_unique_token_fraction"] - baseline_generation["median_unique_token_fraction"],
        },
        "claim_boundary": (
            "Paired three-projection replacement in the real KIMI_BASE MLX graph. The dense model "
            "remains present for all unreplaced organs, so this is an integration discriminator, "
            "not a complete reduced representation, full-model direct execution, capability "
            "qualification, TPS result, NX artifact, or promotion evidence."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "script": str(Path(__file__)),
        "teardown": {
            "planned": "mx.synchronize then release model/consumer references and clear MLX cache",
            "status": "CLEANUP_ATTEMPTED",
            "process_exit_mode": "controlled_os_exit_0" if controlled_exit else "normal_interpreter_exit",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    mx.synchronize()
    del consumers, switch, model, tok
    mx.clear_cache()
    out["teardown"]["status"] = "SYNCHRONIZED_RELEASED_CACHE_CLEARED"
    output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "status": out["status"],
                      "deltas": out["deltas"],
                      "baseline_nll": baseline_summary, "patched_nll": patched_summary,
                      "watch_worst_state": live_watch.worst,
                      "teardown": out["teardown"]}, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate-receipt", type=Path, required=True)
    ap.add_argument("--up-receipt", type=Path, required=True)
    ap.add_argument("--down-receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--down-rank", type=int, default=None,
                    help="select an exact down-projection rank from the supplied receipt")
    ap.add_argument("--controlled-exit", action="store_true",
                    help="after receipt commit and MLX cleanup, use os._exit(0) to bypass a native interpreter teardown crash")
    args = ap.parse_args()
    if args.max_tokens <= 0:
        raise SystemExit("max-tokens must be positive")
    probe({"gate_proj": args.gate_receipt, "up_proj": args.up_receipt,
           "down_proj": args.down_receipt}, args.output, args.max_tokens,
          controlled_exit=args.controlled_exit,
          rank_overrides={"down_proj": args.down_rank})
    if args.controlled_exit:
        sys.stdout.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
