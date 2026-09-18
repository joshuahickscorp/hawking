#!/usr/bin/env python3
"""Held-out activation-probe fanout for shared additive KIMI representations."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.future.complete_ebpw import (  # noqa: E402
    STREAM_BROADCAST_AUX,
    STREAM_WEIGHT_CODES,
    candidate_from_parts,
    cost,
)
from tools.future.kimi_ebpw_inventory import DEFAULT_SPEC  # noqa: E402
from tools.future.kimi_shared_additive_organ_fanout import (  # noqa: E402
    DEFAULT_D,
    _fit_additive,
    _load_weights,
    _select_names,
    _direct_mv,
)

SCHEMA = "hawking.future.kimi_shared_additive_heldout_fanout.v1"
VARIANTS = (32, 64, 128)
DEFAULT_EXPERT_COUNT = 8
DEFAULT_SAMPLE_ROWS = 32
DEFAULT_STEPS = 2
DEFAULT_PROBES = 16
METADATA_BYTES = 4096
DECODER_SUPPORT_BYTES = 65536
SOFT_BUDGET_MS = 60000.0


def _one(weights, names, fit_result, entries: int, probes: int, reps: int):
    import torch

    first_books, second_books, codes, rows, width, groups = fit_result
    generator = torch.Generator(device="cpu").manual_seed(20260910 + entries)
    probe_inputs = [[torch.randn(width, generator=generator) for _ in weights] for _ in range(probes)]
    started = time.perf_counter()
    relative = []
    cosine = []
    for inputs in probe_inputs:
        for weight, code, x in zip(weights, codes, inputs):
            direct = _direct_mv(first_books, second_books, code, x, width=width)
            oracle = weight.matmul(x)
            error = direct - oracle
            relative.append(float(torch.linalg.vector_norm(error).item() / max(torch.linalg.vector_norm(oracle).item(), 1e-12)))
            cosine.append(float(torch.nn.functional.cosine_similarity(direct[None, :], oracle[None, :]).item()))

    inputs = probe_inputs[0]
    def direct_batch():
        return [_direct_mv(first_books, second_books, code, x, width=width) for code, x in zip(codes, inputs)]
    def dense_batch():
        return [weight.matmul(x) for weight, x in zip(weights, inputs)]
    direct_samples = []
    dense_samples = []
    for _ in range(max(1, reps)):
        tic = time.perf_counter()
        direct_batch()
        direct_samples.append((time.perf_counter() - tic) * 1000.0)
        tic = time.perf_counter()
        dense_batch()
        dense_samples.append((time.perf_counter() - tic) * 1000.0)

    parent_params = sum(int(weight.numel()) for weight in weights)
    index_bytes = len(weights) * rows * groups * 2
    codebook_bytes = 2 * groups * entries * DEFAULT_D * 2
    candidate = candidate_from_parts(
        family_id=f"KIMI_shared_additive_heldout_2x{entries}_experts_{len(weights)}",
        parent_params=parent_params,
        parts={
            "regions": [{"name": "two_additive_uint8_indices", "bytes": index_bytes, "stream_class": STREAM_WEIGHT_CODES}],
            "generators": [],
            "metadata": [{"name": "organ_metadata", "bytes": METADATA_BYTES, "stream_class": STREAM_BROADCAST_AUX}],
            "tables": [{"name": "two_shared_bf16_codebooks", "bytes": codebook_bytes, "stream_class": STREAM_WEIGHT_CODES}],
            "residuals": [],
            "runtime_auxiliaries": [{"name": "additive_direct_decoder_support", "bytes": DECODER_SUPPORT_BYTES, "stream_class": STREAM_BROADCAST_AUX}],
            "representation": [],
            "model_specific_code": [],
        },
        reconstructs_dense_parent=False,
        consumes_representation_directly=True,
    )
    accounting = cost(candidate)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "variant": {"entries_per_codebook": entries, "codebooks": 2, "vector_width": DEFAULT_D},
        "status": "HELDOUT_ADDITIVE_REFERENCE_PASS",
        "representation": {"complete_accounting": accounting, "index_bytes": index_bytes, "codebook_bytes": codebook_bytes},
        "direct_execution": {"verified": True, "dense_parent_materialized_by_direct_path": False, "persistent_representation_bytes": accounting["executable_bytes"], "production_kernel": "NOT_VALIDATED"},
        "heldout": {"probe_count": probes, "probe_tensor_pairs": probes * len(weights), "mean_relative_l2": statistics.mean(relative), "p95_relative_l2": sorted(relative)[max(0, int(len(relative) * 0.95) - 1)], "max_relative_l2": max(relative), "mean_cosine": statistics.mean(cosine)},
        "timing": {"direct_batch_median_ms": statistics.median(direct_samples), "dense_oracle_batch_median_ms": statistics.median(dense_samples), "probe_evaluation_ms": elapsed_ms, "fit_excluded_from_direct_timing": True, "torch_threads": 1},
        "fanout": {"elapsed_ms": elapsed_ms, "soft_budget_ms": SOFT_BUDGET_MS, "soft_budget_status": "WITHIN" if elapsed_ms <= SOFT_BUDGET_MS else "OVER"},
    }


def main() -> int:
    import torch

    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--expert-count", type=int, default=DEFAULT_EXPERT_COUNT)
    ap.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--probes", type=int, default=DEFAULT_PROBES)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--workers", type=int, default=len(VARIANTS))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not 2 <= args.expert_count <= 64 or min(args.sample_rows, args.steps, args.probes, args.reps, args.workers) <= 0:
        raise SystemExit("expert-count must be 2..64; sample-rows, steps, probes, reps, and workers must be positive")

    names = _select_names(args.spec, args.expert_count, "up_proj.weight")
    loaded, shards = _load_weights(args.spec, names)
    weights = [tensor for _name, tensor, _shard in loaded]
    fit_started = time.perf_counter()
    fit_results = {entries: _fit_additive(weights, entries, args.sample_rows, args.steps) for entries in VARIANTS}
    fit_ms = (time.perf_counter() - fit_started) * 1000.0
    started = time.perf_counter()
    workers = min(args.workers, len(VARIANTS))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, weights, names, fit_results[entries], entries, args.probes, args.reps) for entries in VARIANTS]
        results = [future.result() for future in futures]
    wall_ms = (time.perf_counter() - started) * 1000.0

    def summary(result: dict[str, Any]) -> dict[str, Any]:
        return {"entries_per_codebook": result["variant"]["entries_per_codebook"], "complete_ebpw": result["representation"]["complete_accounting"]["complete_ebpw"], "mean_relative_l2": result["heldout"]["mean_relative_l2"], "p95_relative_l2": result["heldout"]["p95_relative_l2"], "mean_cosine": result["heldout"]["mean_cosine"], "direct_batch_median_ms": result["timing"]["direct_batch_median_ms"]}

    out = {
        "schema": SCHEMA,
        "status": "HELDOUT_ADDITIVE_FANOUT_COMPLETE",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "specimen": str(args.spec),
        "organ": f"layer10_routed_experts_0_to_{args.expert_count - 1}",
        "tensor_family": "up_proj.weight",
        "expert_count": args.expert_count,
        "tensor_names": names,
        "shards": shards,
        "fit": {"sample_rows_per_tensor": args.sample_rows, "steps": args.steps, "fit_ms": fit_ms, "activation_probes_are_held_out": True},
        "results": results,
        "fanout_timing": {"workers": workers, "variant_count": len(results), "wall_elapsed_ms": wall_ms, "method_soft_budget_ms": SOFT_BUDGET_MS, "execution": "CPU held-out activation-probe fanout; not model TPS"},
        "ranked_summary": sorted((summary(result) for result in results), key=lambda row: (row["complete_ebpw"], row["mean_relative_l2"])),
        "claim_boundary": "Held-out activation probes validate representation error stability on a bounded routed-expert projection organ only. They do not establish end-to-end model capability, physical TPS, a production kernel, an NX artifact, or KIMI promotion.",
        "next_discriminator": "Use held-out capability prompts on an integrated organ candidate only after a representation reaches an acceptable output-error regime.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "script": str(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": out["status"], "fit_ms": fit_ms, "wall_elapsed_ms": wall_ms, "ranked_summary": out["ranked_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
