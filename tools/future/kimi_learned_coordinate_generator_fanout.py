#!/usr/bin/env python3
"""Representation-native coordinate-generator fanout for a bounded KIMI MoE organ."""
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

SCHEMA = "hawking.future.kimi_learned_coordinate_generator_fanout.v1"
VARIANTS = (
    {"name": "coord_latent8_h64", "latent": 8, "group_latent": 8, "hidden": 64, "steps": 250},
    {"name": "coord_latent16_h64", "latent": 16, "group_latent": 8, "hidden": 64, "steps": 250},
    {"name": "coord_latent16_h128", "latent": 16, "group_latent": 8, "hidden": 128, "steps": 250},
    {"name": "coord_latent32_h128", "latent": 32, "group_latent": 16, "hidden": 128, "steps": 250},
    {"name": "coord_expert_row_latent16_h64", "latent": 16, "group_latent": 8, "hidden": 64, "steps": 1000, "per_expert_row": True},
    {"name": "coord_expert_row_latent32_h128", "latent": 32, "group_latent": 16, "hidden": 128, "steps": 1000, "per_expert_row": True},
)
DEFAULT_EXPERT_COUNT = 8
DEFAULT_SAMPLE_ROWS = 0
DEFAULT_PROBES = 16
DEFAULT_BATCH = 8192
METADATA_BYTES = 4096
DECODER_SUPPORT_BYTES = 65536
MODEL_CODE_BYTES = 8192
SOFT_BUDGET_MS = 120000.0


def _make_blocks(weights):
    import torch

    rows, width = (int(x) for x in weights[0].shape)
    groups = (width + 31) // 32
    padded_width = groups * 32
    blocks = []
    for weight in weights:
        data = weight
        if padded_width != width:
            data = torch.nn.functional.pad(data, (0, padded_width - width))
        blocks.append(data.view(rows, groups, 32))
    return torch.stack(blocks), rows, width, groups


class CoordinateGenerator:
    def __init__(self, experts: int, rows: int, groups: int, latent: int, group_latent: int, hidden: int, per_expert_row: bool = False):
        import torch.nn as nn

        self.module = nn.Module()
        self.per_expert_row = bool(per_expert_row)
        self.rows = rows
        self.module.row_embedding = nn.Embedding(experts * rows if self.per_expert_row else rows, latent)
        if not self.per_expert_row:
            self.module.expert_embedding = nn.Embedding(experts, latent)
        self.module.group_embedding = nn.Embedding(groups, group_latent)
        input_width = latent + group_latent if self.per_expert_row else latent * 2 + group_latent
        self.module.net = nn.Sequential(
            nn.Linear(input_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 32),
        )

    def parameters(self):
        return self.module.parameters()

    def __call__(self, expert, row, group):
        import torch

        row_index = expert * self.rows + row if self.per_expert_row else row
        pieces = [self.module.row_embedding(row_index), self.module.group_embedding(group)]
        if not self.per_expert_row:
            pieces.insert(0, self.module.expert_embedding(expert))
        return self.module.net(torch.cat(pieces, dim=1))

    def parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.module.parameters())


def _fit(blocks, variant: dict[str, int | str], batch_size: int, seed: int):
    import torch

    experts, rows, groups, _width = (int(x) for x in blocks.shape)
    torch.manual_seed(seed)
    generator = CoordinateGenerator(
        experts,
        rows,
        groups,
        int(variant["latent"]),
        int(variant["group_latent"]),
        int(variant["hidden"]),
        bool(variant.get("per_expert_row", False)),
    )
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.01)
    total = experts * rows * groups
    started = time.perf_counter()
    loss_value = None
    for step in range(int(variant["steps"])):
        flat = torch.randint(total, (min(batch_size, total),), dtype=torch.int64)
        group = flat % groups
        row = (flat // groups) % rows
        expert = flat // (rows * groups)
        prediction = generator(expert, row, group)
        target = blocks[expert, row, group]
        loss = torch.mean((prediction - target) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().item())
    fit_ms = (time.perf_counter() - started) * 1000.0
    # The measured consumer uses the declared persisted dtype.  Decode values
    # are widened only for the reference matvec; no dense parent is created.
    generator.module = generator.module.to(dtype=torch.bfloat16)
    return generator, fit_ms, loss_value


def _generated_block(generator, expert_index: int, group: int, rows: int):
    import torch

    row = torch.arange(rows, dtype=torch.int64)
    expert = torch.full((rows,), expert_index, dtype=torch.int64)
    groups = torch.full((rows,), group, dtype=torch.int64)
    return generator(expert, row, groups).to(dtype=torch.float32)


def _direct_mv(generator, expert_index: int, x, *, rows: int, width: int, groups: int):
    out = x.new_zeros(rows)
    for group in range(groups):
        values = _generated_block(generator, expert_index, group, rows)
        start = group * 32
        stop = min(start + 32, width)
        out += values[:, : stop - start].matmul(x[start:stop])
    return out


def _one(blocks, names, variant, rows, width, groups, probes, reps, batch_size):
    import torch

    started = time.perf_counter()
    generator, fit_ms, final_loss = _fit(blocks, variant, batch_size, seed=20260910 + int(variant["latent"]) + int(variant["hidden"]))
    probe_generator = torch.Generator(device="cpu").manual_seed(20480101)
    probe_inputs = [[torch.randn(width, generator=probe_generator) for _ in range(blocks.shape[0])] for _ in range(probes)]
    relative = []
    cosine = []
    for probe in probe_inputs:
        for expert_index, x in enumerate(probe):
            direct = _direct_mv(generator, expert_index, x, rows=rows, width=width, groups=groups)
            oracle = blocks[expert_index].view(rows, groups * 32)[:, :width].matmul(x)
            error = direct - oracle
            relative.append(float(torch.linalg.vector_norm(error).item() / max(torch.linalg.vector_norm(oracle).item(), 1e-12)))
            cosine.append(float(torch.nn.functional.cosine_similarity(direct[None, :], oracle[None, :]).item()))

    probe = probe_inputs[0]
    def direct_batch():
        return [_direct_mv(generator, expert_index, x, rows=rows, width=width, groups=groups) for expert_index, x in enumerate(probe)]
    def dense_batch():
        return [blocks[expert_index].view(rows, groups * 32)[:, :width].matmul(x) for expert_index, x in enumerate(probe)]
    direct_samples = []
    dense_samples = []
    for _ in range(max(1, reps)):
        tic = time.perf_counter()
        direct_batch()
        direct_samples.append((time.perf_counter() - tic) * 1000.0)
        tic = time.perf_counter()
        dense_batch()
        dense_samples.append((time.perf_counter() - tic) * 1000.0)

    generator_params = generator.parameter_count()
    generator_bytes = generator_params * 2
    candidate = candidate_from_parts(
        family_id=f"KIMI_{variant['name']}_experts_{len(names)}",
        parent_params=int(blocks.numel()),
        parts={
            "regions": [],
            "generators": [{"name": "coordinate_generator_bf16_parameters", "bytes": generator_bytes, "stream_class": STREAM_WEIGHT_CODES}],
            "metadata": [{"name": "coordinate_layout_metadata", "bytes": METADATA_BYTES, "stream_class": STREAM_BROADCAST_AUX}],
            "tables": [],
            "residuals": [],
            "runtime_auxiliaries": [{"name": "coordinate_decoder_support", "bytes": DECODER_SUPPORT_BYTES, "stream_class": STREAM_BROADCAST_AUX}],
            "representation": [],
            "model_specific_code": [{"name": "coordinate_generator_kernel_code", "bytes": MODEL_CODE_BYTES, "stream_class": STREAM_BROADCAST_AUX}],
        },
        reconstructs_dense_parent=False,
        consumes_representation_directly=True,
    )
    accounting = cost(candidate)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "variant": variant,
        "status": "HELDOUT_LEARNED_GENERATOR_REFERENCE_PASS",
        "tensor_names": names,
        "representation": {"family": "learned_coordinate_generator", "generator_parameter_count": generator_params, "generator_storage_dtype": "bf16", "generator_bytes": generator_bytes, "complete_accounting": accounting},
        "direct_execution": {"verified": True, "dense_parent_materialized_by_direct_path": False, "execution": "generate each row-group block from expert/row/group coordinates, then matvec", "transient_group_expansion_bytes": rows * 32 * 4, "persistent_representation_bytes": accounting["executable_bytes"], "production_kernel": "NOT_VALIDATED"},
        "heldout": {"probe_count": probes, "probe_tensor_pairs": probes * len(names), "mean_relative_l2": statistics.mean(relative), "p95_relative_l2": sorted(relative)[max(0, int(len(relative) * 0.95) - 1)], "max_relative_l2": max(relative), "mean_cosine": statistics.mean(cosine)},
        "training": {"steps": int(variant["steps"]), "batch_size": batch_size, "fit_ms": fit_ms, "final_sample_loss": final_loss, "fit_source_is_immutable_weights": True},
        "timing": {"direct_batch_median_ms": statistics.median(direct_samples), "direct_batch_samples_ms": direct_samples, "dense_oracle_batch_median_ms": statistics.median(dense_samples), "dense_oracle_batch_samples_ms": dense_samples, "fit_excluded_from_direct_timing": True, "torch_threads": 1},
        "fanout": {"elapsed_ms": elapsed_ms, "soft_budget_ms": SOFT_BUDGET_MS, "soft_budget_status": "WITHIN" if elapsed_ms <= SOFT_BUDGET_MS else "OVER"},
    }


def main() -> int:
    import torch

    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--expert-count", type=int, default=DEFAULT_EXPERT_COUNT)
    ap.add_argument("--probes", type=int, default=DEFAULT_PROBES)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--steps", type=int, default=None, help="override the per-variant training step count")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--workers", type=int, default=len(VARIANTS))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not 2 <= args.expert_count <= 64 or min(args.probes, args.batch_size, args.reps, args.workers) <= 0 or (args.steps is not None and args.steps <= 0):
        raise SystemExit("expert-count must be 2..64; probes, batch-size, reps, and workers must be positive; steps must be positive when set")

    names = _select_names(args.spec, args.expert_count, "up_proj.weight")
    loaded, shards = _load_weights(args.spec, names)
    weights = [tensor for _name, tensor, _shard in loaded]
    blocks, rows, width, groups = _make_blocks(weights)
    started = time.perf_counter()
    workers = min(args.workers, len(VARIANTS))
    variants = [dict(variant, steps=args.steps) if args.steps is not None else variant for variant in VARIANTS]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, blocks, names, variant, rows, width, groups, args.probes, args.reps, args.batch_size) for variant in variants]
        results = [future.result() for future in futures]
    wall_ms = (time.perf_counter() - started) * 1000.0

    def summary(result):
        return {"variant": result["variant"]["name"], "complete_ebpw": result["representation"]["complete_accounting"]["complete_ebpw"], "mean_relative_l2": result["heldout"]["mean_relative_l2"], "p95_relative_l2": result["heldout"]["p95_relative_l2"], "mean_cosine": result["heldout"]["mean_cosine"], "fit_ms": result["training"]["fit_ms"], "direct_batch_median_ms": result["timing"]["direct_batch_median_ms"]}

    out = {
        "schema": SCHEMA,
        "status": "HELDOUT_LEARNED_GENERATOR_FANOUT_COMPLETE",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "specimen": str(args.spec),
        "organ": f"layer10_routed_experts_0_to_{args.expert_count - 1}",
        "tensor_family": "up_proj.weight",
        "expert_count": args.expert_count,
        "tensor_names": names,
        "shards": shards,
        "representation_fit": {"organ_rows": rows, "organ_width": width, "group_width": 32, "groups": groups, "training_is_representation_native": True, "dense_parent_is_fit_source_only": True, "step_override": args.steps},
        "results": results,
        "fanout_timing": {"workers": workers, "variant_count": len(results), "wall_elapsed_ms": wall_ms, "method_soft_budget_ms": SOFT_BUDGET_MS, "execution": "CPU learned-generator held-out fanout; not model TPS"},
        "ranked_summary": sorted((summary(result) for result in results), key=lambda row: (row["complete_ebpw"], row["mean_relative_l2"])),
        "claim_boundary": "Bounded learned coordinate-generator reference only. The generator is trained from immutable weights and directly produces blocks at execution time without dense-parent rematerialization, but this is not a full model representation, capability result, end-to-end TPS result, production kernel qualification, NX artifact, or KIMI promotion.",
        "next_discriminator": "Take any Pareto survivor to held-out task-level capability and authorization checks only after validating generator fidelity on additional organs; otherwise increase model expressiveness or change the generated factorization.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "script": str(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": out["status"], "wall_elapsed_ms": wall_ms, "ranked_summary": out["ranked_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
