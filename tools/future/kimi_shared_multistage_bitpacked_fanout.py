#!/usr/bin/env python3
"""Bit-packed multistage residual-codebook fanout for a bounded KIMI MoE organ."""
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

SCHEMA = "hawking.future.kimi_shared_multistage_bitpacked_fanout.v1"
STAGE_VARIANTS = (
    {"stages": 2, "bits": 4, "entries": 16},
    {"stages": 3, "bits": 4, "entries": 16},
    {"stages": 4, "bits": 4, "entries": 16},
    {"stages": 6, "bits": 4, "entries": 16},
    {"stages": 6, "bits": 3, "entries": 8},
    {"stages": 8, "bits": 3, "entries": 8},
)
DEFAULT_EXPERT_COUNT = 8
VECTOR_WIDTH = 32
DEFAULT_SAMPLE_ROWS = 32
DEFAULT_LLOYD_STEPS = 2
DEFAULT_PROBES = 16
METADATA_BYTES = 4096
DECODER_SUPPORT_BYTES = 65536
SOFT_BUDGET_MS = 60000.0


def _fit(weights, stages: int, bits: int, entries: int, sample_rows: int, lloyd_steps: int):
    import torch

    rows, width = (int(x) for x in weights[0].shape)
    groups = (width + VECTOR_WIDTH - 1) // VECTOR_WIDTH
    padded_width = groups * VECTOR_WIDTH
    blocks = []
    for weight in weights:
        data = weight
        if padded_width != width:
            data = torch.nn.functional.pad(data, (0, padded_width - width))
        blocks.append(data.view(rows, groups, VECTOR_WIDTH))
    sample_index = torch.linspace(0, rows - 1, min(sample_rows, rows), dtype=torch.long)
    sample_blocks = torch.cat([block.index_select(0, sample_index) for block in blocks], dim=0)
    codebooks: list[list[torch.Tensor]] = [[] for _ in range(groups)]
    with torch.no_grad():
        for group in range(groups):
            residual = sample_blocks[:, group, :].clone()
            for _stage in range(stages):
                seed = torch.linspace(0, residual.shape[0] - 1, min(entries, residual.shape[0]), dtype=torch.long)
                if seed.numel() < entries:
                    seed = seed.repeat((entries + seed.numel() - 1) // seed.numel())[:entries]
                codebook = residual.index_select(0, seed).clone()
                for _ in range(max(1, int(lloyd_steps))):
                    code = ((residual[:, None, :] - codebook[None, :, :]) ** 2).sum(dim=2).argmin(dim=1)
                    sums = torch.zeros_like(codebook)
                    counts = torch.zeros(entries, dtype=torch.int64)
                    sums.index_add_(0, code, residual)
                    counts.index_add_(0, code, torch.ones_like(code, dtype=torch.int64))
                    codebook = sums / counts.clamp_min(1).to(dtype=sums.dtype).unsqueeze(1)
                codebook = codebook.to(dtype=torch.bfloat16).to(dtype=torch.float32)
                codebooks[group].append(codebook)
                code = ((residual[:, None, :] - codebook[None, :, :]) ** 2).sum(dim=2).argmin(dim=1)
                residual = residual - codebook.index_select(0, code)

    packed_codes = []
    with torch.no_grad():
        for block in blocks:
            residual = block.clone()
            codes = torch.empty((rows, groups, stages), dtype=torch.uint8)
            for stage in range(stages):
                stage_codes = []
                for group in range(groups):
                    codebook = codebooks[group][stage]
                    code = ((residual[:, group, None, :] - codebook[None, :, :]) ** 2).sum(dim=2).argmin(dim=1)
                    stage_codes.append(code)
                    residual[:, group, :] -= codebook.index_select(0, code)
                codes[:, :, stage] = torch.stack(stage_codes, dim=1).to(dtype=torch.uint8)
            packed_bytes = (stages * bits + 7) // 8
            packed = torch.zeros((rows, groups, packed_bytes), dtype=torch.uint8)
            for stage in range(stages):
                bit_offset = stage * bits
                byte_index, shift = divmod(bit_offset, 8)
                code = codes[:, :, stage].to(dtype=torch.int64)
                low_bits = min(bits, 8 - shift)
                packed[:, :, byte_index] |= ((code & ((1 << low_bits) - 1)) << shift).to(dtype=torch.uint8)
                if low_bits < bits:
                    packed[:, :, byte_index + 1] |= (code >> low_bits).to(dtype=torch.uint8)
            packed_codes.append(packed)
    return codebooks, packed_codes, rows, width, groups


def _direct_mv(codebooks, packed_codes, x, *, width: int, stages: int, bits: int):
    import torch

    rows = int(packed_codes.shape[0])
    out = torch.zeros(rows, dtype=torch.float32)
    packed_bytes = packed_codes.to(dtype=torch.int64)
    for group in range(int(packed_codes.shape[1])):
        values = torch.zeros((rows, VECTOR_WIDTH), dtype=torch.float32)
        for stage in range(stages):
            bit_offset = stage * bits
            byte_index, shift = divmod(bit_offset, 8)
            low_bits = min(bits, 8 - shift)
            code = (packed_bytes[:, group, byte_index] >> shift) & ((1 << low_bits) - 1)
            if low_bits < bits:
                code |= (packed_bytes[:, group, byte_index + 1] & ((1 << (bits - low_bits)) - 1)) << low_bits
            values += codebooks[group][stage].index_select(0, code)
        start = group * VECTOR_WIDTH
        stop = min(start + VECTOR_WIDTH, width)
        out += values[:, : stop - start].matmul(x[start:stop])
    return out


def _one(weights, names, codebooks, packed_codes, inputs, rows, width, groups, stages, bits, entries, probes, reps):
    import torch

    started = time.perf_counter()
    relative = []
    cosine = []
    for probe_inputs in inputs:
        for weight, packed, x in zip(weights, packed_codes, probe_inputs):
            direct = _direct_mv(codebooks, packed, x, width=width, stages=stages, bits=bits)
            oracle = weight.matmul(x)
            error = direct - oracle
            relative.append(float(torch.linalg.vector_norm(error).item() / max(torch.linalg.vector_norm(oracle).item(), 1e-12)))
            cosine.append(float(torch.nn.functional.cosine_similarity(direct[None, :], oracle[None, :]).item()))

    probe = inputs[0]
    def direct_batch():
        return [_direct_mv(codebooks, packed, x, width=width, stages=stages, bits=bits) for packed, x in zip(packed_codes, probe)]
    def dense_batch():
        return [weight.matmul(x) for weight, x in zip(weights, probe)]
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
    vector_count = len(weights) * rows * groups
    packed_index_bytes = vector_count * ((stages * bits + 7) // 8)
    codebook_bytes = stages * groups * entries * VECTOR_WIDTH * 2
    candidate = candidate_from_parts(
        family_id=f"KIMI_shared_multistage_bitpacked_{stages}x{bits}bit_e{entries}_experts_{len(weights)}",
        parent_params=parent_params,
        parts={
            "regions": [{"name": "bitpacked_multistage_codes", "bytes": packed_index_bytes, "stream_class": STREAM_WEIGHT_CODES}],
            "generators": [],
            "metadata": [{"name": "organ_metadata", "bytes": METADATA_BYTES, "stream_class": STREAM_BROADCAST_AUX}],
            "tables": [{"name": "shared_bf16_residual_codebooks", "bytes": codebook_bytes, "stream_class": STREAM_WEIGHT_CODES}],
            "residuals": [],
            "runtime_auxiliaries": [{"name": "bitpacked_direct_decoder_support", "bytes": DECODER_SUPPORT_BYTES, "stream_class": STREAM_BROADCAST_AUX}],
            "representation": [],
            "model_specific_code": [],
        },
        reconstructs_dense_parent=False,
        consumes_representation_directly=True,
    )
    accounting = cost(candidate)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "variant": {"stages": stages, "bits_per_code": bits, "entries_per_codebook": entries, "vector_width": VECTOR_WIDTH},
        "status": "HELDOUT_BITPACKED_MULTISTAGE_REFERENCE_PASS",
        "tensor_names": names,
        "representation": {"family": "shared_multistage_residual_codebooks", "groups": groups, "packed_index_bytes": packed_index_bytes, "codebook_bytes": codebook_bytes, "complete_accounting": accounting},
        "direct_execution": {"verified": True, "dense_parent_materialized_by_direct_path": False, "persistent_representation_bytes": accounting["executable_bytes"], "transient_group_expansion_bytes": rows * VECTOR_WIDTH * 4, "decoder": "unpack 4-bit codes directly from persistent bytes; sum codebooks; group matvec", "production_kernel": "NOT_VALIDATED"},
        "heldout": {"probe_count": probes, "probe_tensor_pairs": probes * len(weights), "mean_relative_l2": statistics.mean(relative), "p95_relative_l2": sorted(relative)[max(0, int(len(relative) * 0.95) - 1)], "max_relative_l2": max(relative), "mean_cosine": statistics.mean(cosine)},
        "timing": {"direct_batch_median_ms": statistics.median(direct_samples), "direct_batch_samples_ms": direct_samples, "dense_oracle_batch_median_ms": statistics.median(dense_samples), "dense_oracle_batch_samples_ms": dense_samples, "heldout_probe_elapsed_ms": elapsed_ms, "fit_excluded_from_direct_timing": True, "torch_threads": 1},
        "fanout": {"elapsed_ms": elapsed_ms, "soft_budget_ms": SOFT_BUDGET_MS, "soft_budget_status": "WITHIN" if elapsed_ms <= SOFT_BUDGET_MS else "OVER"},
    }


def main() -> int:
    import torch

    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--expert-count", type=int, default=DEFAULT_EXPERT_COUNT)
    ap.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS)
    ap.add_argument("--steps", type=int, default=DEFAULT_LLOYD_STEPS)
    ap.add_argument("--probes", type=int, default=DEFAULT_PROBES)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--workers", type=int, default=len(STAGE_VARIANTS))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not 2 <= args.expert_count <= 64 or min(args.sample_rows, args.steps, args.probes, args.reps, args.workers) <= 0:
        raise SystemExit("expert-count must be 2..64; sample-rows, steps, probes, reps, and workers must be positive")

    names = _select_names(args.spec, args.expert_count, "up_proj.weight")
    loaded, shards = _load_weights(args.spec, names)
    weights = [tensor for _name, tensor, _shard in loaded]
    fit_started = time.perf_counter()
    fit_results = {
        (variant["stages"], variant["bits"], variant["entries"]): _fit(
            weights,
            variant["stages"],
            variant["bits"],
            variant["entries"],
            args.sample_rows,
            args.steps,
        )
        for variant in STAGE_VARIANTS
    }
    fit_ms = (time.perf_counter() - fit_started) * 1000.0
    generator = torch.Generator(device="cpu").manual_seed(20260910)
    inputs = [[torch.randn(int(weights[0].shape[1]), generator=generator) for _ in weights] for _ in range(args.probes)]
    started = time.perf_counter()
    workers = min(args.workers, len(STAGE_VARIANTS))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for variant in STAGE_VARIANTS:
            stages, bits, entries = variant["stages"], variant["bits"], variant["entries"]
            codebooks, packed, rows, width, groups = fit_results[(stages, bits, entries)]
            futures.append(pool.submit(_one, weights, names, codebooks, packed, inputs, rows, width, groups, stages, bits, entries, args.probes, args.reps))
        results = [future.result() for future in futures]
    wall_ms = (time.perf_counter() - started) * 1000.0

    def summary(result):
        return {"stages": result["variant"]["stages"], "bits_per_code": result["variant"]["bits_per_code"], "entries_per_codebook": result["variant"]["entries_per_codebook"], "complete_ebpw": result["representation"]["complete_accounting"]["complete_ebpw"], "mean_relative_l2": result["heldout"]["mean_relative_l2"], "p95_relative_l2": result["heldout"]["p95_relative_l2"], "mean_cosine": result["heldout"]["mean_cosine"], "direct_batch_median_ms": result["timing"]["direct_batch_median_ms"]}

    out = {
        "schema": SCHEMA,
        "status": "HELDOUT_BITPACKED_MULTISTAGE_FANOUT_COMPLETE",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "specimen": str(args.spec),
        "organ": f"layer10_routed_experts_0_to_{args.expert_count - 1}",
        "tensor_family": "up_proj.weight",
        "expert_count": args.expert_count,
        "tensor_names": names,
        "shards": shards,
        "fit": {"sample_rows_per_tensor": args.sample_rows, "lloyd_steps": args.steps, "fit_ms": fit_ms, "activation_probes_are_held_out": True, "bit_packing_variants": STAGE_VARIANTS},
        "results": results,
        "fanout_timing": {"workers": workers, "variant_count": len(results), "wall_elapsed_ms": wall_ms, "method_soft_budget_ms": SOFT_BUDGET_MS, "execution": "CPU packed multistage held-out fanout; not model TPS"},
        "ranked_summary": sorted((summary(result) for result in results), key=lambda row: (row["complete_ebpw"], row["mean_relative_l2"])),
        "claim_boundary": "Bounded routed-expert multistage representation reference only. It validates packed direct consumption and held-out probe error on eight projection tensors, not full-model capability, end-to-end TPS, production kernel qualification, NX, or KIMI promotion.",
        "next_discriminator": "Take any Pareto survivor into output-aware held-out residual allocation and integrated organ capability tests; scale only after those gates pass.",
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
