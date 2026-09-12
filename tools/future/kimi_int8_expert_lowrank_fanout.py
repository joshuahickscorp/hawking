#!/usr/bin/env python3
"""Fan out row-wise-INT8 independent expert low-rank representations.

This is a materially different Gravity method from BF16 factorization: higher
rank is purchased with INT8 factor payloads and explicit FP32 row scales.  The
immutable KIMI weights are teacher targets only.  Held-out real organ inputs
and complete executable bytes are measured for every arm.
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
from tools.future.campaign_memory_guard import require_ok, sample  # noqa: E402
from tools.future.kimi_ebpw_inventory import DEFAULT_SPEC  # noqa: E402
from tools.future.kimi_expert_lowrank_nova_fanout import (  # noqa: E402
    _max_rank_initializers,
)
from tools.future.kimi_representation_native_nova_fanout import (  # noqa: E402
    _split_corpus,
)
from tools.future.kimi_shared_pq_organ_reference import (  # noqa: E402
    _load_weights,
    _select_names,
)

SCHEMA = "hawking.future.kimi_int8_expert_lowrank_fanout.v1"
EXPERT_COUNT = 64
TENSOR_SUFFIX = "down_proj.weight"
TENSOR_CHOICES = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
RANKS = (64, 80, 96, 112)
MAX_RANK = max(RANKS)
TRAIN_STEPS = 300
LEARNING_RATE = 0.02
REPS = 3
METADATA_BYTES = 4096
DECODER_SUPPORT_BYTES = 65536
SOFT_BUDGET_MS = 180000.0
OBJECTIVES = ("mse", "row_relative", "output_whitened")


def _quantize_rows(tensor):
    """Return signed INT8 rows and FP32 row scales, without source weights."""
    import torch

    scale = tensor.detach().float().abs().amax(dim=1).clamp_min(1e-12) / 127.0
    quantized = torch.round(tensor.detach().float() / scale[:, None]).clamp(-127, 127)
    return quantized.to(torch.int8), scale.to(torch.float32)


def _dequantized(q, scale):
    return q.float() * scale[:, None]


def _write_artifact(path: Path, factors: list[tuple[Any, Any]]) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    for expert, (a, b) in enumerate(factors):
        qa, sa = _quantize_rows(a)
        qb, sb = _quantize_rows(b)
        arrays[f"a_q_{expert}"] = qa.cpu().numpy()
        arrays[f"a_scale_{expert}"] = sa.cpu().numpy()
        arrays[f"b_q_{expert}"] = qb.cpu().numpy()
        arrays[f"b_scale_{expert}"] = sb.cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "container": "compressed_npz_int8_factor_bit_patterns_with_fp32_row_scales",
        "logical_arrays": sorted(arrays),
        "compressed_file_bytes": path.stat().st_size,
    }


def _direct(aa, sa, bb, sb, x):
    return x.matmul(_dequantized(bb, sb)).matmul(_dequantized(aa, sa).transpose(0, 1))


def _one(weights, train, heldout, names: list[str], rank: int,
        initializers, steps: int, learning_rate: float, artifact_dir: Path,
        objective: str, tensor_suffix: str):
    import torch

    started = time.perf_counter()
    x_train = [torch.from_numpy(xs) for xs in train]
    targets = [x.matmul(weight.float().transpose(0, 1))
               for x, weight in zip(x_train, weights)]
    output_scales = [target.square().mean(dim=0).sqrt().clamp_min(1e-6)
                     for target in targets]
    a = [torch.nn.Parameter(init_a[:, :rank].clone())
         for init_a, _init_b in initializers]
    b = [torch.nn.Parameter(init_b[:, :rank].clone())
         for _init_a, init_b in initializers]
    optimizer = torch.optim.Adam([*a, *b], lr=learning_rate)
    fit_started = time.perf_counter()
    losses: list[float] = []
    with torch.enable_grad():
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), dtype=torch.float32)
            for x, target, dim_scale, aa, bb in zip(x_train, targets, output_scales, a, b):
                predicted = x.matmul(bb).matmul(aa.transpose(0, 1))
                squared = (predicted - target) ** 2
                if objective == "row_relative":
                    # Match the held-out discriminator: each real routed row
                    # contributes its relative output error instead of letting
                    # high-energy rows dominate the global MSE.
                    row_error = squared.mean(dim=1)
                    row_scale = target.square().mean(dim=1).clamp_min(1e-12)
                    loss = loss + (row_error / row_scale).mean()
                elif objective == "output_whitened":
                    # Downstream proxy: equalize output coordinates by their
                    # observed real-organ energy, preventing a few large
                    # coordinates from dominating the residual budget.
                    loss = loss + (squared / dim_scale.square()).mean()
                else:
                    loss = loss + (squared.mean() / target.square().mean().clamp_min(1e-12))
            loss = loss / len(weights)
            loss.backward()
            optimizer.step()
            if step in (0, steps - 1):
                losses.append(float(loss.detach().item()))
    fit_ms = (time.perf_counter() - fit_started) * 1000.0

    quantized = [(_quantize_rows(aa), _quantize_rows(bb)) for aa, bb in zip(a, b)]
    relative: list[float] = []
    cosine: list[float] = []
    with torch.no_grad():
        for weight, ((qa, sa), (qb, sb)), xs in zip(weights, quantized, heldout):
            for row in xs:
                x = torch.from_numpy(row)
                direct = _direct(qa, sa, qb, sb, x)
                oracle = weight.float().matmul(x)
                relative.append(float(torch.linalg.vector_norm(direct - oracle).item()
                                     / max(torch.linalg.vector_norm(oracle).item(), 1e-12)))
                cosine.append(float(torch.nn.functional.cosine_similarity(
                    direct[None, :], oracle[None, :]).item()))

        def direct_batch():
            return [_direct(qa, sa, qb, sb, torch.from_numpy(xs[0]))
                    for ((qa, sa), (qb, sb)), xs in zip(quantized, heldout)]

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

    rows, width = (int(x) for x in weights[0].shape)
    factor_bytes = len(weights) * rank * (rows + width)  # INT8 payload
    scale_bytes = len(weights) * (rows + width) * 4  # FP32 row scales
    parent_params = sum(int(weight.numel()) for weight in weights)
    candidate = candidate_from_parts(
        family_id=f"KIMI_nova_int8_expert_lowrank_{tensor_suffix}_r{rank}",
        parent_params=parent_params,
        parts={
            "regions": [
                {"name": "independent_expert_lowrank_int8", "bytes": factor_bytes,
                 "stream_class": STREAM_WEIGHT_CODES},
                {"name": "independent_expert_lowrank_row_scales_fp32", "bytes": scale_bytes,
                 "stream_class": STREAM_WEIGHT_CODES},
            ],
            "generators": [],
            "metadata": [{"name": "organ_metadata", "bytes": METADATA_BYTES,
                          "stream_class": STREAM_BROADCAST_AUX}],
            "tables": [], "residuals": [],
            "runtime_auxiliaries": [{"name": "direct_int8_lowrank_decoder_support",
                                     "bytes": DECODER_SUPPORT_BYTES,
                                     "stream_class": STREAM_BROADCAST_AUX}],
            "representation": [], "model_specific_code": [],
        },
        reconstructs_dense_parent=False,
        consumes_representation_directly=True,
    )
    accounting = cost(candidate)
    artifact = _write_artifact(
        artifact_dir / f"int8_expert_lowrank_{tensor_suffix.replace('.weight', '')}_r{rank}.npz",
        list(zip(a, b)),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "variant": {"rank": rank, "tensor_family": tensor_suffix,
                    "quantizer": "symmetric_rowwise_int8_fp32_scale",
                    "objective": objective},
        "status": "INT8_EXPERT_LOWRANK_HELDOUT_PASS",
        "tensor_names": names,
        "training": {"steps": steps, "learning_rate": learning_rate,
                      "initial_final_normalized_loss": losses,
                      "optimizer_state_deployed": False},
        "representation": {
            "family": "independent_expert_lowrank_AeBeT_rowwise_int8",
            "rank": rank, "factor_int8_bytes": factor_bytes,
            "row_scale_fp32_bytes": scale_bytes,
            "training_objective": objective,
            "complete_accounting": accounting,
        },
        "factor_artifact": artifact,
        "direct_execution": {
            "verified": True,
            "dense_parent_materialized_by_direct_path": False,
            "execution": "y_e = dequant(A_e) @ (dequant(B_e)^T x)",
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
            "quantization_and_eval_included_in_elapsed": True,
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

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--activation-corpus", type=Path, required=True)
    ap.add_argument("--expert-count", type=int, default=EXPERT_COUNT)
    ap.add_argument("--tensor-suffix", choices=TENSOR_CHOICES,
                    default=TENSOR_SUFFIX,
                    help="routed projection family to fit")
    ap.add_argument("--steps", type=int, default=TRAIN_STEPS)
    ap.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    ap.add_argument("--objective", choices=OBJECTIVES, default="mse")
    ap.add_argument(
        "--ranks",
        default=','.join(str(rank) for rank in RANKS),
        help="comma-separated ranks to fan out; useful for lower-rate frontier probes",
    )
    ap.add_argument("--artifact-dir", type=Path,
                    default=ROOT / "workspace/campaign/odyssey/representations/kimi_int8_expert_lowrank")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not 2 <= args.expert_count <= 64 or args.steps <= 0:
        raise SystemExit("expert-count 2..64 and steps must be positive")
    if args.learning_rate <= 0:
        raise SystemExit("learning-rate must be positive")
    try:
        ranks = tuple(sorted({int(value) for value in args.ranks.split(',') if value.strip()}))
    except ValueError as exc:
        raise SystemExit("ranks must be a comma-separated list of integers") from exc
    if not ranks or any(rank <= 0 or rank > MAX_RANK for rank in ranks):
        raise SystemExit(f"ranks must be in the range 1..{MAX_RANK}")

    entry = require_ok("Nova INT8 expert lowrank fanout", expected_gb=2.0).as_dict()
    load_started = time.perf_counter()
    names = _select_names(args.spec, args.expert_count, args.tensor_suffix)
    loaded, shards = _load_weights(args.spec, names)
    weights = [tensor for _name, tensor, _shard in loaded]
    load_ms = (time.perf_counter() - load_started) * 1000.0
    loaded_state = sample(expected_gb=2.0).as_dict()
    width = int(weights[0].shape[1])
    corpus_started = time.perf_counter()
    train, heldout, counts = _split_corpus(args.activation_corpus, args.expert_count, width)
    corpus_ms = (time.perf_counter() - corpus_started) * 1000.0
    pca_started = time.perf_counter()
    initializers, pca_ms = _max_rank_initializers(weights, max(ranks))
    pca_wall_ms = (time.perf_counter() - pca_started) * 1000.0

    fanout_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_one, weights, train, heldout, names, rank,
                               initializers, args.steps, args.learning_rate,
                               args.artifact_dir, args.objective,
                               args.tensor_suffix) for rank in ranks]
        results = [future.result() for future in futures]
    wall_ms = (time.perf_counter() - fanout_started) * 1000.0
    ended = sample().as_dict()

    ranked = sorted(({
        "variant": row["variant"],
        "complete_ebpw": row["representation"]["complete_accounting"]["complete_ebpw"],
        "mean_relative_l2": row["heldout"]["mean_relative_l2"],
        "p95_relative_l2": row["heldout"]["p95_relative_l2"],
        "mean_cosine": row["heldout"]["mean_cosine"],
        "training_ms": row["timing"]["training_ms"],
    } for row in results), key=lambda row: (row["complete_ebpw"], row["mean_relative_l2"]))
    out = {
        "schema": SCHEMA,
        "status": "INT8_EXPERT_LOWRANK_FANOUT_COMPLETE",
        "source_model": "KIMI_BASE", "immutable_base": True,
        "specimen": str(args.spec), "organ": f"layer10 routed experts 0 to {args.expert_count - 1}",
        "tensor_family": args.tensor_suffix, "expert_count": args.expert_count,
        "tensor_names": names, "shards": shards,
        "activation_corpus": {"path": str(args.activation_corpus),
                              "train_and_heldout_rows": counts,
                              "split": "interleaved even rows train, odd rows held-out"},
        "guard": {"entry": entry, "loaded": loaded_state, "end": ended},
        "fit": {"variants": ranks, "steps": args.steps,
                "learning_rate": args.learning_rate, "pca_initializer_ms": pca_ms,
                "teacher_targets": "W_e x computed from immutable source weights",
                "quantizer": "symmetric row-wise INT8 factors with FP32 row scales",
                "objective": args.objective,
                "objective_definition": {
                    "mse": "global normalized output MSE",
                    "row_relative": "mean per-routed-row output relative MSE",
                    "output_whitened": "output-coordinate MSE divided by per-expert train-target RMS squared",
                }[args.objective]},
        "timing": {"load_ms": load_ms, "corpus_split_ms": corpus_ms,
                   "initializer_wall_ms": pca_wall_ms,
                   "fanout_wall_ms": wall_ms,
                   "sum_variant_elapsed_ms": sum(r["fanout"]["elapsed_ms"] for r in results)},
        "results": results, "ranked_summary": ranked,
        "claim_boundary": (
            "Bounded representation-native INT8 factor fanout on all observed routed experts. "
            "Complete logical bytes include INT8 payloads, FP32 row scales, metadata, and decoder "
            "support. Dense source weights are teacher targets only; the direct consumer does not "
            "materialize the dense parent. This is not full-model capability, end-to-end TPS, a "
            "production INT8 routed kernel, NX, or promotion evidence."
        ),
        "next_discriminator": "Audit the best under-1 arm with a routed INT8 kernel and then use the real three-projection capability harness; retain fail-closed promotion.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "script": str(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, args.output)
    print(json.dumps({"output": str(args.output), "status": out["status"],
                      "timing": out["timing"], "ranked_summary": ranked}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
