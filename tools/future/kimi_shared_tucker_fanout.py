#!/usr/bin/env python3
"""Held-out shared two-sided Tucker/CORE fan-out for a KIMI MoE organ.

This is a materially different discriminator from the one-sided shared input
basis.  The representation is ``W_e ~= U C_e V^T``: U and V are shared across
experts while C_e is expert-local.  The direct reference consumes the factors
as ``U(C_e(V^T x))`` and never materializes W_e on the execution path.

This is an organ-level representation experiment.  It is not a full-model
capability, TPS, production-kernel, NX, or promotion result.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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

SCHEMA = "hawking.future.kimi_shared_tucker_fanout.v1"
EXPERT_COUNT = 8
TENSOR_SUFFIX = "up_proj.weight"
ROW_RANKS = (8, 16, 32, 48)
COL_RANKS = (8, 16, 32)
SAMPLE_ROWS = 128
PROBES = 16
REPS = 3
METADATA_BYTES = 4096
DECODER_SUPPORT_BYTES = 65536
SOFT_BUDGET_MS = 60000.0


def _fit_tucker(weights, row_rank: int, col_rank: int, sample_rows: int):
    import torch

    rows, width = (int(x) for x in weights[0].shape)
    take = torch.linspace(0, rows - 1, min(sample_rows, rows), dtype=torch.long)
    sampled = torch.cat([w.index_select(0, take).float() for w in weights], dim=0)
    # Shared input basis first.  Projecting the full selected organ into V makes
    # the output basis see the same input geometry that the direct consumer will
    # execute, rather than fitting two unrelated marginal decompositions.
    _u, _s, v = torch.pca_lowrank(sampled, q=col_rank, center=False, niter=2)
    v = v[:, :col_rank].contiguous()
    projected = [w.float().matmul(v) for w in weights]
    # U is shared in the output/row axis, so the expert projections are joined
    # by columns.  Concatenating by rows would produce a basis over an artificial
    # 8*rows axis and could not be applied to a real output vector.
    projected_joined = torch.cat(projected, dim=1)
    u, _s, _v = torch.pca_lowrank(projected_joined, q=row_rank, center=False, niter=2)
    u = u[:, :row_rank].contiguous()
    cores = [u.transpose(0, 1).matmul(p).contiguous() for p in projected]
    return u, v, cores, rows, width


def _direct(u, v, core, x):
    return u.matmul(core.matmul(v.transpose(0, 1).matmul(x)))


def _one(weights, inputs_by_expert, names: list[str], row_rank: int, col_rank: int, sample_rows: int):
    import torch

    started = time.perf_counter()
    fit_started = time.perf_counter()
    u, v, cores, rows, width = _fit_tucker(weights, row_rank, col_rank, sample_rows)
    fit_ms = (time.perf_counter() - fit_started) * 1000.0

    relative: list[float] = []
    cosine: list[float] = []
    with torch.no_grad():
        for weight, core, expert_inputs in zip(weights, cores, inputs_by_expert):
            for x in expert_inputs:
                direct = _direct(u, v, core, x)
                oracle = weight.float().matmul(x)
                relative.append(float(torch.linalg.vector_norm(direct - oracle).item() /
                                     max(torch.linalg.vector_norm(oracle).item(), 1e-12)))
                cosine.append(float(torch.nn.functional.cosine_similarity(
                    direct[None, :], oracle[None, :]).item()))

        def direct_batch():
            return [_direct(u, v, core, expert_inputs[0])
                    for core, expert_inputs in zip(cores, inputs_by_expert)]

        def dense_batch():
            return [weight.float().matmul(expert_inputs[0])
                    for weight, expert_inputs in zip(weights, inputs_by_expert)]

        direct_samples: list[float] = []
        dense_samples: list[float] = []
        for _ in range(REPS):
            tic = time.perf_counter(); direct_batch()
            direct_samples.append((time.perf_counter() - tic) * 1000.0)
            tic = time.perf_counter(); dense_batch()
            dense_samples.append((time.perf_counter() - tic) * 1000.0)

    core_bytes = len(weights) * row_rank * col_rank * 2
    u_bytes = rows * row_rank * 2
    v_bytes = width * col_rank * 2
    parent_params = sum(int(w.numel()) for w in weights)
    candidate = candidate_from_parts(
        family_id=f"KIMI_shared_tucker_u{row_rank}_v{col_rank}",
        parent_params=parent_params,
        parts={
            "regions": [{"name": "expert_core_bf16", "bytes": core_bytes,
                          "stream_class": STREAM_WEIGHT_CODES}],
            "generators": [],
            "metadata": [{"name": "organ_metadata", "bytes": METADATA_BYTES,
                          "stream_class": STREAM_BROADCAST_AUX}],
            "tables": [
                {"name": "shared_output_basis_bf16", "bytes": u_bytes,
                 "stream_class": STREAM_WEIGHT_CODES},
                {"name": "shared_input_basis_bf16", "bytes": v_bytes,
                 "stream_class": STREAM_WEIGHT_CODES},
            ],
            "residuals": [],
            "runtime_auxiliaries": [{"name": "direct_tucker_decoder_support",
                                     "bytes": DECODER_SUPPORT_BYTES,
                                     "stream_class": STREAM_BROADCAST_AUX}],
            "representation": [],
            "model_specific_code": [],
        },
        reconstructs_dense_parent=False,
        consumes_representation_directly=True,
    )
    accounting = cost(candidate)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "variant": {"row_rank": row_rank, "col_rank": col_rank,
                    "tensor_family": TENSOR_SUFFIX},
        "status": "HELDOUT_SHARED_TUCKER_REFERENCE_PASS",
        "tensor_names": names,
        "representation": {
            "family": "shared_output_basis_plus_expert_core_plus_shared_input_basis",
            "row_rank": row_rank, "col_rank": col_rank,
            "shared_output_basis_bytes": u_bytes,
            "shared_input_basis_bytes": v_bytes,
            "expert_core_bytes": core_bytes,
            "complete_accounting": accounting,
        },
        "direct_execution": {
            "verified": True,
            "dense_parent_materialized_by_direct_path": False,
            "execution": "z = V^T x; y = U(C_e z)",
            "persistent_representation_bytes": accounting["executable_bytes"],
            "production_kernel": "NOT_VALIDATED",
        },
        "heldout": {
            "probe_count_by_expert": [len(x) for x in inputs_by_expert],
            "total_probe_count": sum(len(x) for x in inputs_by_expert),
            "mean_relative_l2": statistics.mean(relative),
            "p95_relative_l2": sorted(relative)[max(0, int(len(relative) * 0.95) - 1)],
            "max_relative_l2": max(relative),
            "mean_cosine": statistics.mean(cosine),
        },
        "timing": {
            "fit_ms": fit_ms,
            "direct_batch_median_ms": statistics.median(direct_samples),
            "direct_batch_samples_ms": direct_samples,
            "dense_oracle_batch_median_ms": statistics.median(dense_samples),
            "dense_oracle_batch_samples_ms": dense_samples,
            "torch_threads": 1,
            "elapsed_ms": elapsed_ms,
        },
        "fanout": {"elapsed_ms": elapsed_ms, "soft_budget_ms": SOFT_BUDGET_MS,
                   "soft_budget_status": "WITHIN" if elapsed_ms <= SOFT_BUDGET_MS else "OVER"},
    }


def main() -> int:
    import torch
    from tools.future.campaign_memory_guard import require_ok, sample

    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--expert-count", type=int, default=EXPERT_COUNT)
    ap.add_argument("--sample-rows", type=int, default=SAMPLE_ROWS)
    ap.add_argument("--probes", type=int, default=PROBES)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--activation-corpus", type=Path,
                    help="optional .npz from kimi_organ_activation_capture.py")
    args = ap.parse_args()
    if not 2 <= args.expert_count <= 64 or args.sample_rows <= 0 or args.probes < 2:
        raise SystemExit("expert-count must be 2..64; sample-rows positive; probes >= 2")

    entry = require_ok("shared Tucker organ fanout", expected_gb=2.0).as_dict()
    names = _select_names(args.spec, args.expert_count, TENSOR_SUFFIX)
    loaded, shards = _load_weights(args.spec, names)
    weights = [tensor for _name, tensor, _shard in loaded]
    loaded_state = sample(expected_gb=2.0).as_dict()
    input_source: dict[str, Any]
    if args.activation_corpus:
        import numpy as np

        with np.load(args.activation_corpus, allow_pickle=False) as corpus:
            inputs_by_expert = []
            corpus_rows: dict[str, int] = {}
            for expert in range(args.expert_count):
                key = f"expert_{expert}"
                if key not in corpus.files:
                    raise RuntimeError(f"activation corpus lacks required {key}")
                arr = np.asarray(corpus[key], dtype=np.float32)
                expected_width = int(weights[0].shape[1])
                if arr.ndim != 2 or arr.shape[1] != expected_width:
                    raise RuntimeError(
                        f"activation width mismatch for {key}: {arr.shape}; "
                        f"expected (*, {expected_width})")
                # The factor fit uses only weights.  These real routed inputs
                # therefore remain held out from the factor fit while retaining
                # each expert's actual input distribution.
                choose = np.linspace(0, arr.shape[0] - 1,
                                     min(args.probes, arr.shape[0]), dtype=np.int64)
                inputs_by_expert.append([torch.from_numpy(arr[i].copy()) for i in choose])
                corpus_rows[str(expert)] = int(arr.shape[0])
        input_source = {"kind": "REAL_FORWARD_CAPTURED",
                        "path": str(args.activation_corpus),
                        "rows_by_selected_expert": corpus_rows,
                        "probes_per_expert_limit": args.probes}
    else:
        generator = torch.Generator(device="cpu").manual_seed(20260910)
        inputs_by_expert = [[torch.randn(int(weights[0].shape[1]), generator=generator)
                             for _ in range(args.probes)] for _ in weights]
        input_source = {"kind": "GAUSSIAN_CONTROL", "seed": 20260910,
                        "probes_per_expert": args.probes}

    variants = [(row_rank, col_rank) for row_rank in ROW_RANKS for col_rank in COL_RANKS]
    started = time.perf_counter()
    # Static CPU fan-out is independent, but limit workers so the measurement
    # reflects the repository lane rather than an uncontrolled BLAS storm.
    with ThreadPoolExecutor(max_workers=min(4, len(variants))) as pool:
        futures = [pool.submit(_one, weights, inputs_by_expert, names, row_rank, col_rank, args.sample_rows)
                   for row_rank, col_rank in variants]
        results = [future.result() for future in futures]
    wall_ms = (time.perf_counter() - started) * 1000.0
    ended = sample().as_dict()

    def summary(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "variant": row["variant"],
            "complete_ebpw": row["representation"]["complete_accounting"]["complete_ebpw"],
            "mean_relative_l2": row["heldout"]["mean_relative_l2"],
            "p95_relative_l2": row["heldout"]["p95_relative_l2"],
            "direct_batch_median_ms": row["timing"]["direct_batch_median_ms"],
            "fit_ms": row["timing"]["fit_ms"],
        }

    out = {
        "schema": SCHEMA,
        "status": "HELDOUT_SHARED_TUCKER_FANOUT_COMPLETE",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "specimen": str(args.spec),
        "organ": f"layer10_routed_experts_0_to_{args.expert_count - 1}",
        "tensor_family": TENSOR_SUFFIX,
        "expert_count": args.expert_count,
        "tensor_names": names,
        "shards": shards,
        "guard": {"entry": entry, "loaded": loaded_state, "end": ended},
        "fit": {"row_ranks": ROW_RANKS, "col_ranks": COL_RANKS,
                "sample_rows_per_tensor": args.sample_rows,
                "probes_are_held_out_from_factor_fit": True},
        "input_source": input_source,
        "results": results,
        "fanout_timing": {"workers": min(4, len(variants)),
                           "variant_count": len(variants),
                           "wall_elapsed_ms": wall_ms,
                           "sum_variant_elapsed_ms": sum(r["fanout"]["elapsed_ms"] for r in results),
                           "method_soft_budget_ms": SOFT_BUDGET_MS,
                           "execution": "CPU direct-reference fanout; not model TPS"},
        "ranked_summary": sorted((summary(r) for r in results),
                                  key=lambda row: (row["complete_ebpw"], row["mean_relative_l2"])),
        "claim_boundary": (
            "Bounded KIMI routed-expert organ reference. Shared U/V bases and expert cores are "
            "consumed without dense-parent materialization, and held-out output error is measured. "
            "This is not a full-model capability, end-to-end TPS, production-kernel, NX, or promotion result."
        ),
        "next_discriminator": "If the two-sided factorization changes the error regime, add output-aware residual allocation and organ capability; otherwise use the result to bound this factorization before representation-native Nova.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "script": str(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, args.output)
    print(json.dumps({"output": str(args.output), "status": out["status"],
                      "wall_elapsed_ms": wall_ms,
                      "ranked_summary": out["ranked_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
