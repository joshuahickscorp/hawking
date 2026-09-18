#!/usr/bin/env python3
"""Bounded shared-codebook PQ reference for a KIMI MoE organ.

The reference loads a selected family of routed-expert projection tensors,
fits one codebook per input group across that family, stores one code per
weight vector, and executes each matrix-vector product directly from the
shared representation. It is deliberately bounded and organ-level: it does
not build a model artifact or make a capability/TPS claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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

SCHEMA = "hawking.future.kimi_shared_pq_organ_reference.v1"
DEFAULT_TENSOR_SUFFIX = "up_proj.weight"
DEFAULT_EXPERT_COUNT = 8
DEFAULT_SAMPLE_ROWS_PER_TENSOR = 32
METADATA_BYTES = 4096
DECODER_SUPPORT_BYTES = 65536
METHOD_SOFT_BUDGET_MS = 60000.0
VARIANTS = (
    {"name": "shared_pq_d16_e16_s2", "d": 16, "entries": 16, "steps": 2},
    {"name": "shared_pq_d32_e16_s2", "d": 32, "entries": 16, "steps": 2},
    {"name": "shared_pq_d32_e32_s2", "d": 32, "entries": 32, "steps": 2},
    {"name": "shared_pq_d64_e16_s2", "d": 64, "entries": 16, "steps": 2},
    {"name": "shared_pq_d64_e32_s2", "d": 64, "entries": 32, "steps": 2},
    {"name": "shared_pq_d32_e64_s2", "d": 32, "entries": 64, "steps": 2},
    {"name": "shared_pq_d32_e128_s2", "d": 32, "entries": 128, "steps": 2},
    {"name": "shared_pq_d32_e256_s2", "d": 32, "entries": 256, "steps": 2},
    {"name": "shared_pq_d64_e128_s2", "d": 64, "entries": 128, "steps": 2},
    {"name": "shared_scaled_pq_d32_e16_s2", "d": 32, "entries": 16, "steps": 2, "scale_mode": "vector_rms"},
    {"name": "shared_scaled_pq_d32_e32_s2", "d": 32, "entries": 32, "steps": 2, "scale_mode": "vector_rms"},
    {"name": "shared_scaled_pq_d32_e64_s2", "d": 32, "entries": 64, "steps": 2, "scale_mode": "vector_rms"},
    {"name": "shared_scaled_pq_d64_e16_s2", "d": 64, "entries": 16, "steps": 2, "scale_mode": "vector_rms"},
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_weights(spec: Path, tensor_names: list[str]):
    import torch

    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("run with /tmp/hawking-kimi-runtime/bin/python") from exc
    index = json.loads((spec / "model.safetensors.index.json").read_text())
    loaded = []
    shards: dict[str, dict[str, Any]] = {}
    for name in tensor_names:
        shard_name = str(index["weight_map"][name])
        shard = spec / shard_name
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name).to(dtype=torch.float32).contiguous()
        loaded.append((name, tensor, shard))
        shards[str(shard)] = {"path": str(shard), "sha256": _sha256(shard)}
    return loaded, list(shards.values())


def _select_names(spec: Path, expert_count: int, suffix: str) -> list[str]:
    index = json.loads((spec / "model.safetensors.index.json").read_text())
    prefix = "language_model.model.layers.10.mlp.experts."
    names = []
    for expert in range(expert_count):
        name = f"{prefix}{expert}.{suffix}"
        if name not in index["weight_map"]:
            raise RuntimeError(f"missing selected tensor: {name}")
        names.append(name)
    return names


def _fit_shared(weights, d: int, entries: int, steps: int, sample_rows: int, scale_mode: str = "none"):
    import torch

    rows, width = (int(x) for x in weights[0].shape)
    groups = math.ceil(width / d)
    padded_width = groups * d
    blocks = []
    scales = []
    for weight in weights:
        data = weight
        if padded_width != width:
            data = torch.nn.functional.pad(data, (0, padded_width - width))
        block = data.view(rows, groups, d)
        if scale_mode == "vector_rms":
            scale = torch.sqrt(torch.mean(block * block, dim=2)).clamp_min(1e-8)
            block = block / scale.unsqueeze(2)
            scales.append(scale)
        elif scale_mode == "none":
            scales.append(None)
        else:
            raise ValueError(f"unknown scale mode: {scale_mode}")
        blocks.append(block)

    samples = []
    sample_index = torch.linspace(0, rows - 1, min(sample_rows, rows), dtype=torch.long)
    for block in blocks:
        samples.append(block.index_select(0, sample_index))
    sample_blocks = torch.cat(samples, dim=0)

    codebooks = []
    with torch.no_grad():
        for group in range(groups):
            vectors = sample_blocks[:, group, :]
            seed = torch.linspace(0, vectors.shape[0] - 1, min(entries, vectors.shape[0]), dtype=torch.long)
            if seed.numel() < entries:
                seed = seed.repeat((entries + seed.numel() - 1) // seed.numel())[:entries]
            codebook = vectors.index_select(0, seed).clone()
            for _ in range(max(1, int(steps))):
                distance = ((vectors[:, None, :] - codebook[None, :, :]) ** 2).sum(dim=2)
                code = distance.argmin(dim=1)
                sums = torch.zeros_like(codebook)
                counts = torch.zeros(codebook.shape[0], dtype=torch.int64)
                sums.index_add_(0, code, vectors)
                counts.index_add_(0, code, torch.ones_like(code, dtype=torch.int64))
                codebook = sums / counts.clamp_min(1).to(dtype=sums.dtype).unsqueeze(1)
            codebooks.append(codebook.to(dtype=torch.bfloat16).to(dtype=torch.float32))

    codes_per_tensor = []
    with torch.no_grad():
        for block in blocks:
            groups_codes = []
            for group, codebook in enumerate(codebooks):
                vectors = block[:, group, :]
                groups_codes.append(
                    ((vectors[:, None, :] - codebook[None, :, :]) ** 2)
                    .sum(dim=2)
                    .argmin(dim=1)
                )
            codes_per_tensor.append(groups_codes)
    return codebooks, codes_per_tensor, scales, rows, width, groups


def _direct_mv(codebooks, codes, x, *, width: int, d: int, scales=None):
    import torch

    rows = int(codes[0].shape[0])
    out = torch.zeros(rows, dtype=torch.float32)
    for group, (codebook, code) in enumerate(zip(codebooks, codes)):
        start = group * d
        stop = min(start + d, width)
        values = codebook.index_select(0, code)
        if scales is not None:
            values = values * scales[:, group].unsqueeze(1)
        out += values[:, : stop - start].matmul(x[start:stop])
    return out


def _median_ms(fn, reps: int):
    samples = []
    result = None
    for _ in range(max(1, reps)):
        started = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples), samples, result


def _one(spec: Path, names: list[str], variant: dict[str, int | str], reps: int, sample_rows: int) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    load_started = time.perf_counter()
    loaded, shards = _load_weights(spec, names)
    load_ms = (time.perf_counter() - load_started) * 1000.0
    weights = [tensor for _name, tensor, _shard in loaded]
    fit_started = time.perf_counter()
    scale_mode = str(variant.get("scale_mode", "none"))
    codebooks, codes, scales, rows, width, groups = _fit_shared(
        weights,
        d=int(variant["d"]),
        entries=int(variant["entries"]),
        steps=int(variant["steps"]),
        sample_rows=sample_rows,
        scale_mode=scale_mode,
    )
    fit_ms = (time.perf_counter() - fit_started) * 1000.0
    torch.manual_seed(20260910)
    inputs = [torch.randn(width, dtype=torch.float32) for _ in weights]
    direct_fns = [
        lambda cb=codebooks, c=code, x=x, s=scale: _direct_mv(
            cb, c, x, width=width, d=int(variant["d"]), scales=s
        )
        for code, x, scale in zip(codes, inputs, scales)
    ]
    dense_fns = [lambda w=w, x=x: w.matmul(x) for w, x in zip(weights, inputs)]
    direct_results = [fn() for fn in direct_fns]
    oracle_results = [fn() for fn in dense_fns]
    direct_started = time.perf_counter()
    direct_samples = []
    for _ in range(max(1, reps)):
        tic = time.perf_counter()
        direct_results = [fn() for fn in direct_fns]
        direct_samples.append((time.perf_counter() - tic) * 1000.0)
    direct_ms = statistics.median(direct_samples)
    dense_started = time.perf_counter()
    dense_samples = []
    for _ in range(max(1, reps)):
        tic = time.perf_counter()
        oracle_results = [fn() for fn in dense_fns]
        dense_samples.append((time.perf_counter() - tic) * 1000.0)
    dense_ms = statistics.median(dense_samples)
    direct_elapsed_ms = (time.perf_counter() - direct_started) * 1000.0
    dense_elapsed_ms = (time.perf_counter() - dense_started) * 1000.0

    errors = [direct - oracle for direct, oracle in zip(direct_results, oracle_results)]
    relative_l2 = [
        float(torch.linalg.vector_norm(error).item() / max(torch.linalg.vector_norm(oracle).item(), 1e-12))
        for error, oracle in zip(errors, oracle_results)
    ]
    cosine = [
        float(torch.nn.functional.cosine_similarity(direct[None, :], oracle[None, :]).item())
        for direct, oracle in zip(direct_results, oracle_results)
    ]
    parent_params = sum(int(weight.numel()) for weight in weights)
    index_bytes = len(weights) * rows * groups
    scale_bytes = len(weights) * rows * groups * 2 if scale_mode == "vector_rms" else 0
    codebook_bytes = groups * int(variant["entries"]) * int(variant["d"]) * 2
    candidate = candidate_from_parts(
        family_id=f"KIMI_{variant['name']}_layer10_experts_{len(weights)}_{DEFAULT_TENSOR_SUFFIX}",
        parent_params=parent_params,
        parts={
            "regions": [
                {"name": "shared_pq_indices", "bytes": index_bytes, "stream_class": STREAM_WEIGHT_CODES},
                *([{"name": "per_vector_bf16_rms_scales", "bytes": scale_bytes, "stream_class": STREAM_WEIGHT_CODES}] if scale_bytes else []),
            ],
            "generators": [],
            "metadata": [{"name": "organ_metadata", "bytes": METADATA_BYTES, "stream_class": STREAM_BROADCAST_AUX}],
            "tables": [{"name": "shared_bf16_codebooks", "bytes": codebook_bytes, "stream_class": STREAM_WEIGHT_CODES}],
            "residuals": [],
            "runtime_auxiliaries": [{"name": "organ_direct_decoder_support", "bytes": DECODER_SUPPORT_BYTES, "stream_class": STREAM_BROADCAST_AUX}],
            "representation": [],
            "model_specific_code": [],
        },
        reconstructs_dense_parent=False,
        consumes_representation_directly=True,
    )
    accounting = cost(candidate)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "variant": variant,
        "status": "DIRECT_SHARED_ORGAN_REFERENCE_PASS",
        "tensor_names": names,
        "shards": shards,
        "shape": [rows, width],
        "representation": {
            "family": "shared_codebook_rowwise_pq",
            "groups": groups,
            "codebook_entries": int(variant["entries"]),
            "codebook_dtype": "bf16",
            "fit_steps": int(variant["steps"]),
            "sample_rows_per_tensor": sample_rows,
            "index_bytes": index_bytes,
            "scale_bytes": scale_bytes,
            "scale_mode": scale_mode,
            "codebook_bytes": codebook_bytes,
            "complete_accounting": accounting,
        },
        "direct_execution": {
            "verified": True,
            "dense_parent_materialized_by_direct_path": False,
            "transient_group_expansion_bytes": rows * int(variant["d"]) * 4,
            "persistent_representation_bytes": accounting["executable_bytes"],
            "direct_path": "shared codebook lookup plus group matvec",
            "production_kernel": "NOT_VALIDATED",
        },
        "error": {
            "per_tensor_relative_l2": relative_l2,
            "mean_relative_l2": statistics.mean(relative_l2),
            "max_relative_l2": max(relative_l2),
            "per_tensor_cosine": cosine,
            "mean_cosine": statistics.mean(cosine),
        },
        "timing": {
            "load_ms": load_ms,
            "fit_ms": fit_ms,
            "encode_ms": max(0.0, elapsed_ms - load_ms - fit_ms - direct_elapsed_ms - dense_elapsed_ms),
            "direct_batch_median_ms": direct_ms,
            "direct_batch_samples_ms": direct_samples,
            "dense_oracle_batch_median_ms": dense_ms,
            "dense_oracle_batch_samples_ms": dense_samples,
            "torch_threads": 1,
            "fit_excluded_from_direct_timing": True,
        },
        "fanout": {
            "elapsed_ms": elapsed_ms,
            "soft_budget_ms": METHOD_SOFT_BUDGET_MS,
            "soft_budget_status": "WITHIN" if elapsed_ms <= METHOD_SOFT_BUDGET_MS else "OVER",
        },
    }


def main() -> int:
    global torch
    import torch

    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--expert-count", type=int, default=DEFAULT_EXPERT_COUNT)
    ap.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS_PER_TENSOR)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--workers", type=int, default=len(VARIANTS))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not 2 <= args.expert_count <= 64 or args.sample_rows <= 0 or args.reps <= 0 or args.workers <= 0:
        raise SystemExit("expert-count must be 2..64; sample-rows, reps, and workers must be positive")

    names = _select_names(args.spec, args.expert_count, DEFAULT_TENSOR_SUFFIX)
    started = time.perf_counter()
    workers = min(args.workers, len(VARIANTS))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, args.spec, names, variant, args.reps, args.sample_rows) for variant in VARIANTS]
        results = [future.result() for future in futures]
    wall_ms = (time.perf_counter() - started) * 1000.0

    def summary(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "variant": result["variant"],
            "complete_ebpw": result["representation"]["complete_accounting"]["complete_ebpw"],
            "mean_relative_l2": result["error"]["mean_relative_l2"],
            "mean_cosine": result["error"]["mean_cosine"],
            "direct_batch_median_ms": result["timing"]["direct_batch_median_ms"],
            "dense_oracle_batch_median_ms": result["timing"]["dense_oracle_batch_median_ms"],
            "elapsed_ms": result["fanout"]["elapsed_ms"],
        }

    out = {
        "schema": SCHEMA,
        "status": "DIRECT_SHARED_ORGAN_FANOUT_COMPLETE",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "specimen": str(args.spec),
        "organ": "layer10_routed_experts_0_to_" + str(args.expert_count - 1),
        "tensor_family": DEFAULT_TENSOR_SUFFIX,
        "expert_count": args.expert_count,
        "variants": [result["variant"] for result in results],
        "results": results,
        "fanout_timing": {
            "workers": workers,
            "variant_count": len(results),
            "wall_elapsed_ms": wall_ms,
            "sum_variant_elapsed_ms": sum(result["fanout"]["elapsed_ms"] for result in results),
            "method_soft_budget_ms": METHOD_SOFT_BUDGET_MS,
            "execution": "CPU shared-organ direct-reference fanout; not model TPS",
        },
        "ranked_summary": sorted(
            (summary(result) for result in results),
            key=lambda row: (row["complete_ebpw"], row["mean_relative_l2"]),
        ),
        "claim_boundary": (
            "Bounded routed-expert projection-organ reference only. Codebooks are shared "
            "across the selected tensors and the direct path avoids dense-parent materialization, "
            "but this is not a full layer/model representation, capability result, end-to-end "
            "TPS result, production kernel qualification, or KIMI promotion."
        ),
        "next_discriminator": (
            "Take Pareto survivors into output-aware protected-residual allocation and test "
            "whether residual bytes repair enough organ error to justify the complete accounting."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "script": str(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": out["status"],
        "organ": out["organ"],
        "wall_elapsed_ms": wall_ms,
        "ranked_summary": out["ranked_summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
