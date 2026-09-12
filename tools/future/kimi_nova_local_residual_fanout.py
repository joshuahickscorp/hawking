#!/usr/bin/env python3
"""Nova fan-out with a shared core plus learned expert-local low-rank residual.

The deployed operator is ``U(C_e(V^T x)) + A_e(B_e^T x)``.  The dense source
weights provide teacher outputs during fitting and held-out scoring only; the
consumer uses the factors directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools" / "future") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools" / "future"))

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
from tools.future.kimi_representation_native_nova_fanout import (  # noqa: E402
    _split_corpus,
)
from tools.future.kimi_shared_tucker_fanout import _fit_tucker  # noqa: E402

SCHEMA = "hawking.future.kimi_nova_local_residual_fanout.v1"
EXPERT_COUNT = 8
TENSOR_SUFFIX = "up_proj.weight"
# The first six arms are the established organ frontier.  The second group
# tests whether the held-out error is simply representation capacity limited;
# every arm is still charged with complete executable bytes and remains below
# the 1.0 EBPW gate for this MoE organ.
VARIANTS = (
    (16, 16, 4), (32, 16, 4), (32, 16, 8),
    (32, 32, 4), (32, 32, 8), (48, 32, 8),
    (64, 32, 16), (64, 48, 16), (64, 64, 16),
    (96, 64, 16), (96, 96, 16), (96, 96, 32),
)
SAMPLE_ROWS = 128
TRAIN_STEPS = 300
LEARNING_RATE = 0.02
REPS = 3
METADATA_BYTES = 4096
DECODER_SUPPORT_BYTES = 65536
SOFT_BUDGET_MS = 120000.0


def _write_factor_artifact(path: Path, u, v, cores, a, b) -> dict[str, Any]:
    """Persist the logical factor payload as BF16 bit patterns.

    The compressed NPZ is a reproducibility artifact, not the final NX
    container.  Logical deployed bytes remain the separately audited EBPW
    accounting; zip compression is never used to lower that accounting.
    """
    import torch

    def bits(tensor):
        return tensor.detach().cpu().to(torch.bfloat16).view(torch.uint16).numpy()

    arrays = {"u": bits(u), "v": bits(v)}
    arrays.update({f"core_{i}": bits(tensor) for i, tensor in enumerate(cores)})
    arrays.update({f"residual_a_{i}": bits(tensor) for i, tensor in enumerate(a)})
    arrays.update({f"residual_b_{i}": bits(tensor) for i, tensor in enumerate(b)})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(),
            "container": "compressed_npz_bf16_bit_patterns",
            "logical_factor_arrays": sorted(arrays),
            "compressed_file_bytes": path.stat().st_size}


def _direct_residual(u, v, core, a, b, x):
    return u.matmul(core.matmul(v.transpose(0, 1).matmul(x))) + a.matmul(b.transpose(0, 1).matmul(x))


def _one(weights, train, heldout, names: list[str], row_rank: int, col_rank: int,
        residual_rank: int, sample_rows: int, steps: int, learning_rate: float,
        artifact_dir: Path):
    import torch

    started = time.perf_counter()
    u0, v0, cores0, rows, width = _fit_tucker(weights, row_rank, col_rank, sample_rows)
    u = torch.nn.Parameter(u0.float())
    v = torch.nn.Parameter(v0.float())
    cores = [torch.nn.Parameter(core.float()) for core in cores0]
    generator = torch.Generator(device="cpu").manual_seed(
        20260910 + row_rank * 100 + col_rank * 10 + residual_rank)
    a = [torch.nn.Parameter(torch.randn(rows, residual_rank, generator=generator) * 0.01)
         for _ in weights]
    b = [torch.nn.Parameter(torch.randn(width, residual_rank, generator=generator) * 0.01)
         for _ in weights]
    optimizer = torch.optim.Adam([u, v, *cores, *a, *b], lr=learning_rate)
    fit_started = time.perf_counter()
    losses: list[float] = []
    with torch.enable_grad():
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), dtype=torch.float32)
            for weight, core, aa, bb, xs in zip(weights, cores, a, b, train):
                x = torch.from_numpy(xs)
                target = x.matmul(weight.float().transpose(0, 1))
                predicted = x.matmul(v).matmul(core.transpose(0, 1)).matmul(u.transpose(0, 1))
                predicted = predicted + x.matmul(bb).matmul(aa.transpose(0, 1))
                loss = loss + (((predicted - target) ** 2).mean()
                               / target.square().mean().clamp_min(1e-12))
            loss = loss / len(weights)
            loss.backward()
            optimizer.step()
            if step in (0, steps - 1):
                losses.append(float(loss.detach().item()))
    fit_ms = (time.perf_counter() - fit_started) * 1000.0

    relative: list[float] = []
    cosine: list[float] = []
    with torch.no_grad():
        for weight, core, aa, bb, xs in zip(weights, cores, a, b, heldout):
            for row in xs:
                x = torch.from_numpy(row)
                direct = _direct_residual(u, v, core, aa, bb, x)
                oracle = weight.float().matmul(x)
                relative.append(float(torch.linalg.vector_norm(direct - oracle).item()
                                     / max(torch.linalg.vector_norm(oracle).item(), 1e-12)))
                cosine.append(float(torch.nn.functional.cosine_similarity(
                    direct[None, :], oracle[None, :]).item()))

        def direct_batch():
            return [_direct_residual(u, v, core, aa, bb, torch.from_numpy(xs[0]))
                    for core, aa, bb, xs in zip(cores, a, b, heldout)]

        def dense_batch():
            return [weight.float().matmul(torch.from_numpy(xs[0]))
                    for weight, xs in zip(weights, heldout)]

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
    residual_bytes = len(weights) * (rows + width) * residual_rank * 2
    parent_params = sum(int(w.numel()) for w in weights)
    candidate = candidate_from_parts(
        family_id=f"KIMI_nova_shared_core_local_residual_u{row_rank}_v{col_rank}_r{residual_rank}",
        parent_params=parent_params,
        parts={
            "regions": [
                {"name": "expert_core_bf16", "bytes": core_bytes,
                 "stream_class": STREAM_WEIGHT_CODES},
                {"name": "expert_local_residual_factors_bf16", "bytes": residual_bytes,
                 "stream_class": STREAM_WEIGHT_CODES},
            ],
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
            "runtime_auxiliaries": [{"name": "direct_local_residual_decoder_support",
                                     "bytes": DECODER_SUPPORT_BYTES,
                                     "stream_class": STREAM_BROADCAST_AUX}],
            "representation": [],
            "model_specific_code": [],
        },
        reconstructs_dense_parent=False,
        consumes_representation_directly=True,
    )
    accounting = cost(candidate)
    artifact = _write_factor_artifact(
        artifact_dir / f"nova_local_residual_u{row_rank}_v{col_rank}_r{residual_rank}.npz",
        u, v, cores, a, b)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "variant": {"row_rank": row_rank, "col_rank": col_rank,
                    "residual_rank": residual_rank, "tensor_family": TENSOR_SUFFIX},
        "status": "REPRESENTATION_NATIVE_NOVA_LOCAL_RESIDUAL_HELDOUT_PASS",
        "tensor_names": names,
        "training": {"steps": steps, "learning_rate": learning_rate,
                      "initial_final_normalized_loss": losses,
                      "optimizer_state_deployed": False},
        "representation": {
            "family": "learned_shared_tucker_plus_expert_local_low_rank_residual",
            "row_rank": row_rank, "col_rank": col_rank,
            "residual_rank": residual_rank,
            "shared_output_basis_bytes": u_bytes,
            "shared_input_basis_bytes": v_bytes,
            "expert_core_bytes": core_bytes,
            "expert_local_residual_factor_bytes": residual_bytes,
            "complete_accounting": accounting,
        },
        "factor_artifact": artifact,
        "direct_execution": {
            "verified": True,
            "dense_parent_materialized_by_direct_path": False,
            "execution": "z = V^T x; y = U(C_e z) + A_e(B_e^T x)",
            "persistent_representation_bytes": accounting["executable_bytes"],
            "production_kernel": "NOT_VALIDATED",
        },
        "heldout": {
            "heldout_rows_by_expert": [int(xs.shape[0]) for xs in heldout],
            "total_heldout_rows": len(relative),
            "mean_relative_l2": statistics.mean(relative),
            "p95_relative_l2": sorted(relative)[max(0, int(len(relative) * 0.95) - 1)],
            "max_relative_l2": max(relative),
            "mean_cosine": statistics.mean(cosine),
        },
        "timing": {
            "training_ms": fit_ms,
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
    torch.set_num_interop_threads(1)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--activation-corpus", type=Path, required=True)
    ap.add_argument("--expert-count", type=int, default=EXPERT_COUNT)
    ap.add_argument("--sample-rows", type=int, default=SAMPLE_ROWS)
    ap.add_argument("--steps", type=int, default=TRAIN_STEPS)
    ap.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    ap.add_argument("--artifact-dir", type=Path,
                    default=ROOT / "workspace/campaign/odyssey/representations/kimi_nova_local_residual")
    ap.add_argument("--variant-limit", type=int, default=None,
                    help="run only the first N registered arms; useful for receipt-preserving replays")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not 2 <= args.expert_count <= 64 or args.sample_rows <= 0 or args.steps <= 0:
        raise SystemExit("expert-count 2..64; sample-rows and steps positive")
    if args.learning_rate <= 0:
        raise SystemExit("learning-rate must be positive")
    if args.variant_limit is not None and not 1 <= args.variant_limit <= len(VARIANTS):
        raise SystemExit(f"variant-limit must be 1..{len(VARIANTS)}")

    entry = require_ok("Nova local residual organ fanout", expected_gb=2.0).as_dict()
    names = _select_names(args.spec, args.expert_count, TENSOR_SUFFIX)
    loaded, shards = _load_weights(args.spec, names)
    weights = [tensor for _name, tensor, _shard in loaded]
    loaded_state = sample(expected_gb=2.0).as_dict()
    width = int(weights[0].shape[1])
    train, heldout, counts = _split_corpus(args.activation_corpus, args.expert_count, width)
    variants = VARIANTS if args.variant_limit is None else VARIANTS[:args.variant_limit]

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_one, weights, train, heldout, names, row_rank, col_rank,
                               residual_rank, args.sample_rows, args.steps, args.learning_rate,
                               args.artifact_dir)
                   for row_rank, col_rank, residual_rank in variants]
        results = [future.result() for future in futures]
    wall_ms = (time.perf_counter() - started) * 1000.0
    ended = sample().as_dict()

    def summary(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "variant": row["variant"],
            "complete_ebpw": row["representation"]["complete_accounting"]["complete_ebpw"],
            "mean_relative_l2": row["heldout"]["mean_relative_l2"],
            "p95_relative_l2": row["heldout"]["p95_relative_l2"],
            "mean_cosine": row["heldout"]["mean_cosine"],
            "training_ms": row["timing"]["training_ms"],
        }

    out = {
        "schema": SCHEMA,
        "status": "REPRESENTATION_NATIVE_NOVA_LOCAL_RESIDUAL_FANOUT_COMPLETE",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "specimen": str(args.spec),
        "organ": f"layer10_routed_experts_0_to_{args.expert_count - 1}",
        "tensor_family": TENSOR_SUFFIX,
        "expert_count": args.expert_count,
        "tensor_names": names,
        "shards": shards,
        "activation_corpus": {"path": str(args.activation_corpus),
                              "train_and_heldout_rows": counts,
                              "split": "interleaved even rows train, odd rows held-out"},
        "guard": {"entry": entry, "loaded": loaded_state, "end": ended},
        "fit": {"variants": variants, "sample_rows_per_tensor": args.sample_rows,
                "steps": args.steps, "learning_rate": args.learning_rate,
                "teacher_targets": "W_e x computed from immutable source weights"},
        "results": results,
        "fanout_timing": {"workers": 2, "variant_count": len(VARIANTS),
                           "wall_elapsed_ms": wall_ms,
                           "sum_variant_elapsed_ms": sum(r["fanout"]["elapsed_ms"] for r in results),
                           "method_soft_budget_ms": SOFT_BUDGET_MS,
                           "execution": "CPU representation-native training plus direct-reference fanout"},
        "ranked_summary": sorted((summary(r) for r in results),
                                  key=lambda row: (row["complete_ebpw"], row["mean_relative_l2"])),
        "claim_boundary": (
            "Bounded representation-native Nova organ experiment. Dense source weights are teacher "
            "targets only; deployed shared and expert-local factors are consumed without dense-parent "
            "materialization and held-out real routed inputs are measured. This is not full-model "
            "capability, end-to-end TPS, a production kernel, NX, or promotion evidence."
        ),
        "next_discriminator": "If the local residual materially improves held-out error, add native-kernel and organ capability checks; otherwise change topology rather than increasing sparse metadata.",
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
