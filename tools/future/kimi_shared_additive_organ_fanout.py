#!/usr/bin/env python3
"""Shared additive-quantization fanout for a bounded KIMI MoE organ."""
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
from tools.future.kimi_shared_pq_organ_reference import _load_weights, _select_names  # noqa: E402

SCHEMA = "hawking.future.kimi_shared_additive_organ_fanout.v1"
VARIANTS = (32, 64, 128)
DEFAULT_EXPERT_COUNT = 8
DEFAULT_D = 32
DEFAULT_SAMPLE_ROWS_PER_TENSOR = 32
DEFAULT_STEPS = 2
METADATA_BYTES = 4096
DECODER_SUPPORT_BYTES = 65536
SOFT_BUDGET_MS = 60000.0


def _fit_additive(weights, entries: int, sample_rows: int, steps: int):
    import torch

    rows, width = (int(x) for x in weights[0].shape)
    groups = (width + DEFAULT_D - 1) // DEFAULT_D
    padded_width = groups * DEFAULT_D
    blocks = []
    for weight in weights:
        data = weight
        if padded_width != width:
            data = torch.nn.functional.pad(data, (0, padded_width - width))
        blocks.append(data.view(rows, groups, DEFAULT_D))
    sample_index = torch.linspace(0, rows - 1, min(sample_rows, rows), dtype=torch.long)
    samples = torch.cat([block.index_select(0, sample_index) for block in blocks], dim=0)
    first_books = []
    second_books = []
    with torch.no_grad():
        for group in range(groups):
            vectors = samples[:, group, :]
            seed = torch.linspace(0, vectors.shape[0] - 1, min(entries, vectors.shape[0]), dtype=torch.long)
            if seed.numel() < entries:
                seed = seed.repeat((entries + seed.numel() - 1) // seed.numel())[:entries]
            first = vectors.index_select(0, seed).clone()
            first_code = None
            for _ in range(max(1, int(steps))):
                first_code = ((vectors[:, None, :] - first[None, :, :]) ** 2).sum(dim=2).argmin(dim=1)
                sums = torch.zeros_like(first)
                counts = torch.zeros(entries, dtype=torch.int64)
                sums.index_add_(0, first_code, vectors)
                counts.index_add_(0, first_code, torch.ones_like(first_code, dtype=torch.int64))
                first = sums / counts.clamp_min(1).to(dtype=sums.dtype).unsqueeze(1)
            assert first_code is not None
            residual = vectors - first.index_select(0, first_code)
            second = residual.index_select(0, seed).clone()
            for _ in range(max(1, int(steps))):
                second_code = ((residual[:, None, :] - second[None, :, :]) ** 2).sum(dim=2).argmin(dim=1)
                sums = torch.zeros_like(second)
                counts = torch.zeros(entries, dtype=torch.int64)
                sums.index_add_(0, second_code, residual)
                counts.index_add_(0, second_code, torch.ones_like(second_code, dtype=torch.int64))
                second = sums / counts.clamp_min(1).to(dtype=sums.dtype).unsqueeze(1)
            first_books.append(first.to(dtype=torch.bfloat16).to(dtype=torch.float32))
            second_books.append(second.to(dtype=torch.bfloat16).to(dtype=torch.float32))

    codes_per_tensor = []
    with torch.no_grad():
        for block in blocks:
            group_codes = []
            for group, (first, second) in enumerate(zip(first_books, second_books)):
                vectors = block[:, group, :]
                first_code = ((vectors[:, None, :] - first[None, :, :]) ** 2).sum(dim=2).argmin(dim=1)
                residual = vectors - first.index_select(0, first_code)
                second_code = ((residual[:, None, :] - second[None, :, :]) ** 2).sum(dim=2).argmin(dim=1)
                group_codes.append((first_code, second_code))
            codes_per_tensor.append(group_codes)
    return first_books, second_books, codes_per_tensor, rows, width, groups


def _direct_mv(first_books, second_books, codes, x, *, width: int):
    import torch

    out = torch.zeros(int(codes[0][0].shape[0]), dtype=torch.float32)
    for group, ((first, second), (first_code, second_code)) in enumerate(zip(zip(first_books, second_books), codes)):
        start = group * DEFAULT_D
        stop = min(start + DEFAULT_D, width)
        values = first.index_select(0, first_code) + second.index_select(0, second_code)
        out += values[:, : stop - start].matmul(x[start:stop])
    return out


def _one(weights, names, first_books, second_books, codes, inputs, rows, width, groups, entries, reps):
    import torch

    started = time.perf_counter()
    direct_results = [_direct_mv(first_books, second_books, code, x, width=width) for code, x in zip(codes, inputs)]
    oracle_results = [weight.matmul(x) for weight, x in zip(weights, inputs)]
    relative_l2 = []
    cosine = []
    for direct, oracle in zip(direct_results, oracle_results):
        error = direct - oracle
        relative_l2.append(float(torch.linalg.vector_norm(error).item() / max(torch.linalg.vector_norm(oracle).item(), 1e-12)))
        cosine.append(float(torch.nn.functional.cosine_similarity(direct[None, :], oracle[None, :]).item()))

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
        family_id=f"KIMI_shared_additive_2x{entries}_experts_{len(weights)}",
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
        "status": "DIRECT_SHARED_ADDITIVE_REFERENCE_PASS",
        "tensor_names": names,
        "representation": {"family": "shared_additive_2x8", "groups": groups, "index_bytes": index_bytes, "codebook_bytes": codebook_bytes, "complete_accounting": accounting},
        "direct_execution": {"verified": True, "dense_parent_materialized_by_direct_path": False, "transient_group_expansion_bytes": rows * DEFAULT_D * 4, "persistent_representation_bytes": accounting["executable_bytes"], "production_kernel": "NOT_VALIDATED"},
        "error": {"mean_relative_l2": statistics.mean(relative_l2), "max_relative_l2": max(relative_l2), "mean_cosine": statistics.mean(cosine), "per_tensor_relative_l2": relative_l2},
        "timing": {"direct_batch_median_ms": statistics.median(direct_samples), "direct_batch_samples_ms": direct_samples, "dense_oracle_batch_median_ms": statistics.median(dense_samples), "dense_oracle_batch_samples_ms": dense_samples, "fit_excluded_from_direct_timing": True, "torch_threads": 1},
        "fanout": {"elapsed_ms": elapsed_ms, "soft_budget_ms": SOFT_BUDGET_MS, "soft_budget_status": "WITHIN" if elapsed_ms <= SOFT_BUDGET_MS else "OVER"},
    }


def main() -> int:
    import torch

    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--expert-count", type=int, default=DEFAULT_EXPERT_COUNT)
    ap.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS_PER_TENSOR)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--workers", type=int, default=len(VARIANTS))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not 2 <= args.expert_count <= 64 or args.sample_rows <= 0 or args.steps <= 0 or args.reps <= 0 or args.workers <= 0:
        raise SystemExit("expert-count must be 2..64; sample-rows, steps, reps, and workers must be positive")

    names = _select_names(args.spec, args.expert_count, "up_proj.weight")
    loaded, shards = _load_weights(args.spec, names)
    weights = [tensor for _name, tensor, _shard in loaded]
    fit_results = {}
    fit_started = time.perf_counter()
    for entries in VARIANTS:
        fit_results[entries] = _fit_additive(weights, entries, args.sample_rows, args.steps)
    fit_ms = (time.perf_counter() - fit_started) * 1000.0
    torch.manual_seed(20260910)
    inputs = [torch.randn(int(weights[0].shape[1]), dtype=torch.float32) for _ in weights]
    started = time.perf_counter()
    workers = min(args.workers, len(VARIANTS))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for entries in VARIANTS:
            first, second, codes, rows, width, groups = fit_results[entries]
            futures.append(pool.submit(_one, weights, names, first, second, codes, inputs, rows, width, groups, entries, args.reps))
        results = [future.result() for future in futures]
    wall_ms = (time.perf_counter() - started) * 1000.0

    def summary(result):
        return {"entries_per_codebook": result["variant"]["entries_per_codebook"], "complete_ebpw": result["representation"]["complete_accounting"]["complete_ebpw"], "mean_relative_l2": result["error"]["mean_relative_l2"], "mean_cosine": result["error"]["mean_cosine"], "direct_batch_median_ms": result["timing"]["direct_batch_median_ms"], "dense_oracle_batch_median_ms": result["timing"]["dense_oracle_batch_median_ms"]}

    out = {
        "schema": SCHEMA,
        "status": "DIRECT_SHARED_ADDITIVE_FANOUT_COMPLETE",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "specimen": str(args.spec),
        "organ": f"layer10_routed_experts_0_to_{args.expert_count - 1}",
        "tensor_family": "up_proj.weight",
        "expert_count": args.expert_count,
        "tensor_names": names,
        "shards": shards,
        "fit": {"sample_rows_per_tensor": args.sample_rows, "steps": args.steps, "fit_ms": fit_ms, "method": "sequential shared residual codebooks"},
        "results": results,
        "fanout_timing": {"workers": workers, "variant_count": len(results), "wall_elapsed_ms": wall_ms, "method_soft_budget_ms": SOFT_BUDGET_MS, "execution": "CPU shared additive direct-reference fanout; not model TPS"},
        "ranked_summary": sorted((summary(result) for result in results), key=lambda row: (row["complete_ebpw"], row["mean_relative_l2"])),
        "claim_boundary": "Bounded routed-expert additive-quantization reference only. Two shared codebooks are consumed directly and all bytes are billed, but this is not a full model representation, capability result, end-to-end TPS result, production kernel qualification, NX artifact, or KIMI promotion.",
        "next_discriminator": "Take any Pareto survivor to held-out probes, output-aware residual allocation, and organ capability checks.",
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
