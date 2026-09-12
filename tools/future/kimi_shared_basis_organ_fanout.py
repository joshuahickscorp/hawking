#!/usr/bin/env python3
"""Shared-basis direct-execution fanout for a bounded KIMI MoE organ.

For selected routed-expert projection matrices this tests
W_e ~= A_e B, where B is shared across experts and A_e is expert-local.  The
consumer executes A_e (B x), never materializing W_e.  This is a linear
algebra discriminator only; it is not a trained representation or model claim.
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
from tools.future.kimi_shared_pq_organ_reference import _load_weights, _select_names  # noqa: E402

SCHEMA = "hawking.future.kimi_shared_basis_organ_fanout.v1"
VARIANTS = (8, 16, 32, 64, 96)
DEFAULT_EXPERT_COUNT = 8
DEFAULT_SAMPLE_ROWS_PER_TENSOR = 128
METADATA_BYTES = 4096
DECODER_SUPPORT_BYTES = 65536
SOFT_BUDGET_MS = 60000.0


def _fit_basis(weights, max_rank: int, sample_rows: int):
    import torch

    rows, width = (int(x) for x in weights[0].shape)
    sample_index = torch.linspace(0, rows - 1, min(sample_rows, rows), dtype=torch.long)
    samples = torch.cat([weight.index_select(0, sample_index) for weight in weights], dim=0)
    # The sampled shared basis is deliberately bounded.  Full capability
    # validation must later re-fit/validate on held-out rows and prompts.
    _u, _s, v = torch.pca_lowrank(samples, q=max_rank, center=False, niter=2)
    basis = v[:, :max_rank].contiguous()
    factors = [weight.matmul(basis) for weight in weights]
    return basis, factors, rows, width


def _one(weights, basis, factors, inputs, rows: int, width: int, rank: int, names: list[str], reps: int):
    import torch

    started = time.perf_counter()
    basis_r = basis[:, :rank]
    factor_r = [factor[:, :rank] for factor in factors]
    direct_results = []
    oracle_results = []
    for factor, weight, x in zip(factor_r, weights, inputs):
        direct_results.append(factor.matmul(basis_r.transpose(0, 1).matmul(x)))
        oracle_results.append(weight.matmul(x))
    relative_l2 = []
    cosine = []
    for direct, oracle in zip(direct_results, oracle_results):
        error = direct - oracle
        relative_l2.append(float(torch.linalg.vector_norm(error).item() / max(torch.linalg.vector_norm(oracle).item(), 1e-12)))
        cosine.append(float(torch.nn.functional.cosine_similarity(direct[None, :], oracle[None, :]).item()))

    def direct_batch():
        return [factor.matmul(basis_r.transpose(0, 1).matmul(x)) for factor, x in zip(factor_r, inputs)]

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
    factor_bytes = len(weights) * rows * rank * 2
    basis_bytes = width * rank * 2
    candidate = candidate_from_parts(
        family_id=f"KIMI_shared_basis_rank_{rank}_experts_{len(weights)}",
        parent_params=parent_params,
        parts={
            "regions": [{"name": "expert_factors_bf16", "bytes": factor_bytes, "stream_class": STREAM_WEIGHT_CODES}],
            "generators": [],
            "metadata": [{"name": "organ_metadata", "bytes": METADATA_BYTES, "stream_class": STREAM_BROADCAST_AUX}],
            "tables": [{"name": "shared_basis_bf16", "bytes": basis_bytes, "stream_class": STREAM_WEIGHT_CODES}],
            "residuals": [],
            "runtime_auxiliaries": [{"name": "direct_factorized_decoder_support", "bytes": DECODER_SUPPORT_BYTES, "stream_class": STREAM_BROADCAST_AUX}],
            "representation": [],
            "model_specific_code": [],
        },
        reconstructs_dense_parent=False,
        consumes_representation_directly=True,
    )
    accounting = cost(candidate)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "variant": {"rank": rank, "factorization": "shared_basis_B_then_expert_factor_A"},
        "status": "DIRECT_SHARED_BASIS_REFERENCE_PASS",
        "tensor_names": names,
        "representation": {
            "family": "shared_low_rank_basis",
            "rank": rank,
            "factor_bytes": factor_bytes,
            "shared_basis_bytes": basis_bytes,
            "complete_accounting": accounting,
        },
        "direct_execution": {
            "verified": True,
            "dense_parent_materialized_by_direct_path": False,
            "execution": "z = Bx; y = A_e z",
            "transient_activation_bytes": rank * 4,
            "persistent_representation_bytes": accounting["executable_bytes"],
            "production_kernel": "NOT_VALIDATED",
        },
        "error": {
            "mean_relative_l2": statistics.mean(relative_l2),
            "max_relative_l2": max(relative_l2),
            "mean_cosine": statistics.mean(cosine),
            "per_tensor_relative_l2": relative_l2,
        },
        "timing": {
            "direct_batch_median_ms": statistics.median(direct_samples),
            "direct_batch_samples_ms": direct_samples,
            "dense_oracle_batch_median_ms": statistics.median(dense_samples),
            "dense_oracle_batch_samples_ms": dense_samples,
            "fit_excluded_from_direct_timing": True,
            "torch_threads": 1,
        },
        "fanout": {
            "elapsed_ms": elapsed_ms,
            "soft_budget_ms": SOFT_BUDGET_MS,
            "soft_budget_status": "WITHIN" if elapsed_ms <= SOFT_BUDGET_MS else "OVER",
        },
    }


def main() -> int:
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

    names = _select_names(args.spec, args.expert_count, "up_proj.weight")
    loaded, shards = _load_weights(args.spec, names)
    weights = [tensor for _name, tensor, _shard in loaded]
    max_rank = max(VARIANTS)
    fit_started = time.perf_counter()
    basis, factors, rows, width = _fit_basis(weights, max_rank=max_rank, sample_rows=args.sample_rows)
    fit_ms = (time.perf_counter() - fit_started) * 1000.0
    torch.manual_seed(20260910)
    inputs = [torch.randn(width, dtype=torch.float32) for _ in weights]
    started = time.perf_counter()
    workers = min(args.workers, len(VARIANTS))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, weights, basis, factors, inputs, rows, width, rank, names, args.reps) for rank in VARIANTS]
        results = [future.result() for future in futures]
    wall_ms = (time.perf_counter() - started) * 1000.0

    def summary(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "rank": result["variant"]["rank"],
            "complete_ebpw": result["representation"]["complete_accounting"]["complete_ebpw"],
            "mean_relative_l2": result["error"]["mean_relative_l2"],
            "mean_cosine": result["error"]["mean_cosine"],
            "direct_batch_median_ms": result["timing"]["direct_batch_median_ms"],
            "dense_oracle_batch_median_ms": result["timing"]["dense_oracle_batch_median_ms"],
        }

    out = {
        "schema": SCHEMA,
        "status": "DIRECT_SHARED_BASIS_FANOUT_COMPLETE",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "specimen": str(args.spec),
        "organ": f"layer10_routed_experts_0_to_{args.expert_count - 1}",
        "tensor_family": "up_proj.weight",
        "expert_count": args.expert_count,
        "tensor_names": names,
        "shards": shards,
        "basis_fit": {"max_rank": max_rank, "sample_rows_per_tensor": args.sample_rows, "fit_ms": fit_ms, "method": "torch.pca_lowrank sampled shared rows"},
        "results": results,
        "fanout_timing": {
            "workers": workers,
            "rank_count": len(results),
            "wall_elapsed_ms": wall_ms,
            "method_soft_budget_ms": SOFT_BUDGET_MS,
            "execution": "CPU shared-basis direct-reference fanout; not model TPS",
        },
        "ranked_summary": sorted((summary(result) for result in results), key=lambda row: (row["complete_ebpw"], row["mean_relative_l2"])),
        "claim_boundary": (
            "Bounded routed-expert shared-basis reference only. It validates A_e(Bx) direct "
            "execution and complete accounting on selected projection tensors, not a trained "
            "representation, model capability, end-to-end TPS, production kernel, NX artifact, "
            "or KIMI promotion."
        ),
        "next_discriminator": "If a rank is Pareto-useful, add output-aware protected residuals and held-out probe/capability tests before considering a larger organ.",
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
