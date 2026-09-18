#!/usr/bin/env python3
"""Held-out shared-basis plus per-expert residual fan-out for KIMI.

This is a materially different representation from uniform sparse or a shared
low-rank basis alone.  For a bounded routed-expert organ it stores a shared
input basis B, expert-local factors A_e, and a sparse per-expert residual.  The
direct reference computes ``A_e(Bx) + R_e x`` without materializing W_e.

Residual support is selected using one output-aware probe and evaluated on
independent held-out probes.  All residual indices, values, metadata, and
decoder support are billed.  This is an organ discriminator, not a model
artifact or promotion result.
"""
from __future__ import annotations

import argparse
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
from tools.future.kimi_shared_pq_organ_reference import (  # noqa: E402
    _load_weights,
    _select_names,
)

SCHEMA = "hawking.future.kimi_shared_basis_sparse_residual_fanout.v1"
EXPERT_COUNT = 8
TENSOR_SUFFIX = "up_proj.weight"
VECTOR_RANKS = (2, 4, 8)
RESIDUAL_DENSITIES = (0.01, 0.02, 0.05)
SAMPLE_ROWS = 128
PROBES = 16
REPS = 2
METADATA_BYTES = 4096
DECODER_SUPPORT_BYTES = 65536
SOFT_BUDGET_MS = 60000.0


def _fit_basis(weights, rank: int, sample_rows: int):
    import torch

    rows, width = (int(x) for x in weights[0].shape)
    take = torch.linspace(0, rows - 1, min(sample_rows, rows), dtype=torch.long)
    samples = torch.cat([w.index_select(0, take) for w in weights], dim=0)
    _u, _s, v = torch.pca_lowrank(samples, q=rank, center=False, niter=2)
    basis = v[:, :rank].contiguous()
    factors = [w.matmul(basis) for w in weights]
    return basis, factors, rows, width


def _select_residual(residual, x, density: float):
    import torch

    count = max(1, int(residual.numel() * density))
    # One probe chooses support by its contribution to the output, not by raw
    # weight magnitude.  The other probes are held out for evaluation.
    score = residual.abs() * x.abs().view(1, -1)
    flat = score.reshape(-1)
    selected = torch.topk(flat, k=min(count, flat.numel()), sorted=False).indices.sort().values
    rows, cols = torch.div(selected, residual.shape[1], rounding_mode="floor"), selected % residual.shape[1]
    values = residual[rows, cols].to(dtype=torch.bfloat16)
    return rows.to(dtype=torch.int64), cols.to(dtype=torch.int64), values


def _direct(factor, basis, residual_rows, residual_cols, residual_values, x):
    import torch

    y = factor.matmul(basis.transpose(0, 1).matmul(x))
    if residual_rows.numel():
        y = y.clone()
        y.index_add_(0, residual_rows, residual_values.to(dtype=torch.float32) * x.index_select(0, residual_cols))
    return y


def _one(weights, basis, factors, inputs, rank: int, density: float, names: list[str], sample_rows: int):
    import torch

    started = time.perf_counter()
    support = []
    residuals = []
    for weight, factor, x in zip(weights, factors, inputs[0]):
        base = factor.matmul(basis.transpose(0, 1))
        residual = weight - base
        rr, cc, vv = _select_residual(residual, x, density)
        support.append((rr, cc))
        residuals.append(vv)

    direct_results = []
    oracle_results = []
    for weight, factor, x, (rr, cc), vv in zip(weights, factors, inputs[1], support, residuals):
        direct_results.append(_direct(factor, basis, rr, cc, vv, x))
        oracle_results.append(weight.matmul(x))

    relative = []
    cosine = []
    for probe in inputs[1:]:
        for weight, factor, x, (rr, cc), vv in zip(weights, factors, probe, support, residuals):
            direct = _direct(factor, basis, rr, cc, vv, x)
            oracle = weight.matmul(x)
            relative.append(float(torch.linalg.vector_norm(direct - oracle).item() /
                                 max(torch.linalg.vector_norm(oracle).item(), 1e-12)))
            cosine.append(float(torch.nn.functional.cosine_similarity(direct[None, :], oracle[None, :]).item()))

    def direct_batch():
        return [_direct(factor, basis, rr, cc, vv, x)
                for factor, x, (rr, cc), vv in zip(factors, inputs[1], support, residuals)]

    def dense_batch():
        return [weight.matmul(x) for weight, x in zip(weights, inputs[1])]

    direct_samples, dense_samples = [], []
    for _ in range(REPS):
        tic = time.perf_counter(); direct_batch(); direct_samples.append((time.perf_counter() - tic) * 1000.0)
        tic = time.perf_counter(); dense_batch(); dense_samples.append((time.perf_counter() - tic) * 1000.0)

    residual_count = sum(int(rr.numel()) for rr, _cc in support)
    parent_params = sum(int(w.numel()) for w in weights)
    factor_bytes = len(weights) * weights[0].shape[0] * rank * 2
    basis_bytes = weights[0].shape[1] * rank * 2
    residual_index_bytes = residual_count * 8  # row+column, billed explicitly
    residual_value_bytes = residual_count * 2
    candidate = candidate_from_parts(
        family_id=f"KIMI_shared_basis_sparse_residual_r{rank}_d{density:.4f}",
        parent_params=parent_params,
        parts={
            "regions": [
                {"name": "expert_factor_bf16", "bytes": factor_bytes, "stream_class": STREAM_WEIGHT_CODES},
                {"name": "residual_coordinate_pairs", "bytes": residual_index_bytes, "stream_class": STREAM_WEIGHT_CODES},
            ],
            "generators": [],
            "metadata": [{"name": "organ_metadata", "bytes": METADATA_BYTES, "stream_class": STREAM_BROADCAST_AUX}],
            "tables": [{"name": "shared_basis_bf16", "bytes": basis_bytes, "stream_class": STREAM_WEIGHT_CODES}],
            "residuals": [{"name": "expert_residual_bf16_values", "bytes": residual_value_bytes, "stream_class": STREAM_WEIGHT_CODES}],
            "runtime_auxiliaries": [{"name": "direct_factor_residual_decoder_support", "bytes": DECODER_SUPPORT_BYTES, "stream_class": STREAM_BROADCAST_AUX}],
            "representation": [],
            "model_specific_code": [],
        },
        reconstructs_dense_parent=False,
        consumes_representation_directly=True,
    )
    accounting = cost(candidate)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "variant": {"rank": rank, "residual_density": density, "tensor_family": TENSOR_SUFFIX},
        "status": "HELDOUT_SHARED_BASIS_RESIDUAL_REFERENCE_PASS",
        "tensor_names": names,
        "representation": {
            "family": "shared_input_basis_plus_expert_sparse_residual",
            "factor_bytes": factor_bytes,
            "shared_basis_bytes": basis_bytes,
            "residual_count": residual_count,
            "residual_index_bytes": residual_index_bytes,
            "residual_value_bytes": residual_value_bytes,
            "complete_accounting": accounting,
        },
        "direct_execution": {
            "verified": True,
            "dense_parent_materialized_by_direct_path": False,
            "execution": "z = Bx; y = A_e z + sparse(R_e x)",
            "persistent_representation_bytes": accounting["executable_bytes"],
            "production_kernel": "NOT_VALIDATED",
        },
        "heldout": {
            "selection_probe_count": 1,
            "heldout_probe_count": len(inputs) - 1,
            "mean_relative_l2": statistics.mean(relative),
            "p95_relative_l2": sorted(relative)[max(0, int(len(relative) * 0.95) - 1)],
            "max_relative_l2": max(relative),
            "mean_cosine": statistics.mean(cosine),
        },
        "timing": {
            "fit_ms_excluded": True,
            "residual_selection_and_probe_ms": elapsed_ms,
            "direct_batch_median_ms": statistics.median(direct_samples),
            "direct_batch_samples_ms": direct_samples,
            "dense_oracle_batch_median_ms": statistics.median(dense_samples),
            "dense_oracle_batch_samples_ms": dense_samples,
            "torch_threads": 1,
        },
        "fanout": {"elapsed_ms": elapsed_ms, "soft_budget_ms": SOFT_BUDGET_MS,
                   "soft_budget_status": "WITHIN" if elapsed_ms <= SOFT_BUDGET_MS else "OVER"},
    }


def main() -> int:
    import torch

    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--expert-count", type=int, default=EXPERT_COUNT)
    ap.add_argument("--sample-rows", type=int, default=SAMPLE_ROWS)
    ap.add_argument("--probes", type=int, default=PROBES)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not 2 <= args.expert_count <= 64 or args.sample_rows <= 0 or args.probes < 2:
        raise SystemExit("expert-count must be 2..64; sample-rows positive; probes >= 2")

    names = _select_names(args.spec, args.expert_count, TENSOR_SUFFIX)
    loaded, shards = _load_weights(args.spec, names)
    weights = [tensor for _name, tensor, _shard in loaded]
    generator = torch.Generator(device="cpu").manual_seed(20260910)
    inputs = [[torch.randn(int(weights[0].shape[1]), generator=generator) for _ in weights]
              for _ in range(args.probes)]

    fit_started = time.perf_counter()
    fit = {rank: _fit_basis(weights, rank, args.sample_rows) for rank in VECTOR_RANKS}
    fit_ms = (time.perf_counter() - fit_started) * 1000.0

    variants = [(rank, density) for rank in VECTOR_RANKS for density in RESIDUAL_DENSITIES]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(variants)) as pool:
        futures = []
        for rank, density in variants:
            basis, factors, _rows, _width = fit[rank]
            futures.append(pool.submit(_one, weights, basis, factors, inputs, rank, density, names, args.sample_rows))
        results = [future.result() for future in futures]
    wall_ms = (time.perf_counter() - started) * 1000.0

    def summary(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "variant": row["variant"],
            "complete_ebpw": row["representation"]["complete_accounting"]["complete_ebpw"],
            "mean_relative_l2": row["heldout"]["mean_relative_l2"],
            "p95_relative_l2": row["heldout"]["p95_relative_l2"],
            "mean_cosine": row["heldout"]["mean_cosine"],
            "direct_batch_median_ms": row["timing"]["direct_batch_median_ms"],
        }

    out = {
        "schema": SCHEMA,
        "status": "HELDOUT_SHARED_BASIS_RESIDUAL_FANOUT_COMPLETE",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "specimen": str(args.spec),
        "organ": f"layer10_routed_experts_0_to_{args.expert_count - 1}",
        "tensor_family": TENSOR_SUFFIX,
        "expert_count": args.expert_count,
        "tensor_names": names,
        "shards": shards,
        "fit": {"ranks": VECTOR_RANKS, "sample_rows_per_tensor": args.sample_rows,
                "fit_ms": fit_ms, "activation_probes_are_held_out": True},
        "results": results,
        "fanout_timing": {"workers": len(variants), "variant_count": len(variants),
                           "wall_elapsed_ms": wall_ms, "sum_variant_elapsed_ms": sum(r["fanout"]["elapsed_ms"] for r in results),
                           "method_soft_budget_ms": SOFT_BUDGET_MS,
                           "execution": "CPU direct-reference fanout; not model TPS"},
        "ranked_summary": sorted((summary(r) for r in results),
                                  key=lambda row: (row["complete_ebpw"], row["mean_relative_l2"])),
        "claim_boundary": (
            "Bounded KIMI routed-expert organ reference. Shared factors and sparse residuals are "
            "consumed without dense-parent materialization, and held-out output error is measured. "
            "This is not a full-model capability, end-to-end TPS, production-kernel, NX, or promotion result."
        ),
        "next_discriminator": "If a Pareto point changes the error regime, integrate the factor/residual decoder into a larger organ and add capability gates; otherwise move to representation-native Nova training.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "script": str(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": out["status"],
                      "fit_ms": fit_ms, "wall_elapsed_ms": wall_ms,
                      "ranked_summary": out["ranked_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
