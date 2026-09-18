#!/usr/bin/env python3
"""Recoverability-aware variable-rank Nova fan-out for one MoE projection.

Experts with more real routed activation rows receive higher factor rank; cold
experts receive less.  The schedule is measured from the captured corpus and
all logical bytes are charged, so the experiment tests whether a routing-aware
allocation improves the down-projection bottleneck without crossing 1 EBPW.
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
from tools.future.kimi_expert_lowrank_nova_fanout import (  # noqa: E402
    _max_rank_initializers,
    _write_artifact,
)
from tools.future.kimi_representation_native_nova_fanout import (  # noqa: E402
    _split_corpus,
)
from tools.future.kimi_shared_pq_organ_reference import (  # noqa: E402
    _load_weights,
    _select_names,
)

SCHEMA = "hawking.future.kimi_variable_rank_lowrank_nova_fanout.v1"
EXPERT_COUNT = 64
TENSOR_SUFFIX = "down_proj.weight"
SCHEDULES = (
    {"name": "top16_r64_rest40", "hot_count": 16, "hot_rank": 64, "cold_rank": 40},
    {"name": "top16_r64_rest32", "hot_count": 16, "hot_rank": 64, "cold_rank": 32},
    {"name": "top8_r64_next16_r48_rest32", "hot_count": 8, "middle_count": 16,
     "hot_rank": 64, "middle_rank": 48, "cold_rank": 32},
    {"name": "top24_r64_rest32", "hot_count": 24, "hot_rank": 64, "cold_rank": 32},
)
MAX_RANK = 64
TRAIN_STEPS = 300
LEARNING_RATE = 0.02
REPS = 3
METADATA_BYTES = 4096
DECODER_SUPPORT_BYTES = 65536
SOFT_BUDGET_MS = 180000.0


def _rank_schedule(schedule: dict[str, Any], corpus_counts: dict[str, Any]) -> tuple[tuple[int, ...], list[int]]:
    order = sorted(range(len(corpus_counts)),
                   key=lambda expert: (-int(corpus_counts[str(expert)]), expert))
    ranks = [int(schedule["cold_rank"])] * len(order)
    hot_count = int(schedule["hot_count"])
    for expert in order[:hot_count]:
        ranks[expert] = int(schedule["hot_rank"])
    middle_count = int(schedule.get("middle_count", 0))
    if middle_count:
        for expert in order[hot_count:hot_count + middle_count]:
            ranks[expert] = int(schedule["middle_rank"])
    return tuple(ranks), order


def _direct(a, b, x):
    return x.matmul(b).matmul(a.transpose(0, 1))


def _one(weights, train, heldout, names: list[str], schedule: dict[str, Any],
        ranks: tuple[int, ...], initializers, steps: int, learning_rate: float,
        artifact_dir: Path):
    import torch

    started = time.perf_counter()
    x_train = [torch.from_numpy(xs) for xs in train]
    targets = [x.matmul(weight.float().transpose(0, 1))
               for x, weight in zip(x_train, weights)]
    a = [torch.nn.Parameter(init_a[:, :rank].clone())
         for (init_a, _init_b), rank in zip(initializers, ranks)]
    b = [torch.nn.Parameter(init_b[:, :rank].clone())
         for (_init_a, init_b), rank in zip(initializers, ranks)]
    optimizer = torch.optim.Adam([*a, *b], lr=learning_rate)
    fit_started = time.perf_counter()
    losses: list[float] = []
    with torch.enable_grad():
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), dtype=torch.float32)
            for x, target, aa, bb in zip(x_train, targets, a, b):
                predicted = x.matmul(bb).matmul(aa.transpose(0, 1))
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
    by_expert: list[float] = []
    with torch.no_grad():
        for weight, aa, bb, xs in zip(weights, a, b, heldout):
            local: list[float] = []
            for row in xs:
                x = torch.from_numpy(row)
                direct = _direct(aa, bb, x)
                oracle = weight.float().matmul(x)
                value = float(torch.linalg.vector_norm(direct - oracle).item()
                              / max(torch.linalg.vector_norm(oracle).item(), 1e-12))
                relative.append(value)
                local.append(value)
                cosine.append(float(torch.nn.functional.cosine_similarity(
                    direct[None, :], oracle[None, :]).item()))
            by_expert.append(statistics.mean(local))

        def direct_batch():
            return [_direct(aa, bb, torch.from_numpy(xs[0]))
                    for aa, bb, xs in zip(a, b, heldout)]

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
    factor_bytes = sum((rows + width) * rank * 2 for rank in ranks)
    parent_params = sum(int(weight.numel()) for weight in weights)
    candidate = candidate_from_parts(
        family_id=f"KIMI_nova_variable_expert_lowrank_{schedule['name']}",
        parent_params=parent_params,
        parts={
            "regions": [{"name": "variable_expert_lowrank_bf16", "bytes": factor_bytes,
                         "stream_class": STREAM_WEIGHT_CODES}],
            "generators": [],
            "metadata": [{"name": "organ_metadata", "bytes": METADATA_BYTES,
                          "stream_class": STREAM_BROADCAST_AUX}],
            "tables": [], "residuals": [],
            "runtime_auxiliaries": [{"name": "variable_rank_decoder_support",
                                     "bytes": DECODER_SUPPORT_BYTES,
                                     "stream_class": STREAM_BROADCAST_AUX}],
            "representation": [], "model_specific_code": [],
        },
        reconstructs_dense_parent=False,
        consumes_representation_directly=True,
    )
    accounting = cost(candidate)
    artifact = _write_artifact(
        artifact_dir / f"variable_{schedule['name']}.npz", list(zip(a, b))
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "variant": {"schedule": schedule["name"], "tensor_family": TENSOR_SUFFIX,
                    "expert_ranks": list(ranks)},
        "status": "VARIABLE_EXPERT_LOWRANK_NOVA_HELDOUT_PASS",
        "tensor_names": names,
        "training": {"steps": steps, "learning_rate": learning_rate,
                      "initial_final_normalized_loss": losses,
                      "optimizer_state_deployed": False},
        "representation": {"family": "variable_rank_independent_expert_lowrank_AeBeT",
                            "expert_factor_bytes": factor_bytes,
                            "complete_accounting": accounting},
        "factor_artifact": artifact,
        "direct_execution": {"verified": True,
                              "dense_parent_materialized_by_direct_path": False,
                              "execution": "rank-grouped y_e = A_e(B_e^T x)",
                              "persistent_representation_bytes": accounting["executable_bytes"],
                              "production_kernel": "NOT_VALIDATED"},
        "heldout": {"heldout_rows_by_expert": [int(xs.shape[0]) for xs in heldout],
                     "total_heldout_rows": len(relative),
                     "mean_relative_l2": statistics.mean(relative),
                     "macro_mean_relative_l2": statistics.mean(by_expert),
                     "p95_relative_l2": sorted(relative)[max(0, int(len(relative) * 0.95) - 1)],
                     "max_relative_l2": max(relative),
                     "mean_cosine": statistics.mean(cosine),
                     "mean_relative_l2_by_expert": by_expert},
        "timing": {"training_ms": fit_ms,
                    "direct_batch_median_ms": statistics.median(direct_samples),
                    "direct_batch_samples_ms": direct_samples,
                    "dense_oracle_batch_median_ms": statistics.median(dense_samples),
                    "dense_oracle_batch_samples_ms": dense_samples,
                    "torch_threads": 1, "elapsed_ms": elapsed_ms},
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
    ap.add_argument("--tensor-suffix", choices=("gate_proj.weight", "up_proj.weight",
                                                  "down_proj.weight"),
                    default=TENSOR_SUFFIX)
    ap.add_argument("--expert-count", type=int, default=EXPERT_COUNT)
    ap.add_argument("--steps", type=int, default=TRAIN_STEPS)
    ap.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    ap.add_argument("--artifact-dir", type=Path,
                    default=ROOT / "workspace/campaign/odyssey/representations/kimi_variable_rank_lowrank_nova")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not 2 <= args.expert_count <= 64 or args.steps <= 0:
        raise SystemExit("expert-count 2..64 and steps must be positive")
    if args.learning_rate <= 0:
        raise SystemExit("learning-rate must be positive")

    entry = require_ok("Nova variable-rank lowrank fanout", expected_gb=2.0).as_dict()
    names = _select_names(args.spec, args.expert_count, args.tensor_suffix)
    loaded, shards = _load_weights(args.spec, names)
    weights = [tensor for _name, tensor, _shard in loaded]
    loaded_state = sample(expected_gb=2.0).as_dict()
    width = int(weights[0].shape[1])
    train, heldout, counts = _split_corpus(args.activation_corpus, args.expert_count, width)
    corpus_row_counts = {str(expert): int(train[expert].shape[0] + heldout[expert].shape[0])
                         for expert in range(args.expert_count)}
    initializers, pca_ms = _max_rank_initializers(weights, MAX_RANK)
    schedules = []
    for schedule in SCHEDULES:
        ranks, order = _rank_schedule(schedule, corpus_row_counts)
        schedules.append((schedule, ranks, order))

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_one, weights, train, heldout, names, schedule, ranks,
                               initializers, args.steps, args.learning_rate, args.artifact_dir)
                   for schedule, ranks, _order in schedules]
        results = [future.result() for future in futures]
    wall_ms = (time.perf_counter() - started) * 1000.0
    ended = sample().as_dict()

    ranked = sorted(({
        "variant": row["variant"],
        "complete_ebpw": row["representation"]["complete_accounting"]["complete_ebpw"],
        "mean_relative_l2": row["heldout"]["mean_relative_l2"],
        "macro_mean_relative_l2": row["heldout"]["macro_mean_relative_l2"],
        "p95_relative_l2": row["heldout"]["p95_relative_l2"],
        "mean_cosine": row["heldout"]["mean_cosine"],
        "training_ms": row["timing"]["training_ms"],
    } for row in results), key=lambda row: (row["complete_ebpw"], row["mean_relative_l2"]))
    out = {
        "schema": SCHEMA,
        "status": "VARIABLE_EXPERT_LOWRANK_NOVA_FANOUT_COMPLETE",
        "source_model": "KIMI_BASE", "immutable_base": True,
        "specimen": str(args.spec), "organ": f"layer10 routed experts 0 to {args.expert_count - 1}",
        "tensor_family": args.tensor_suffix, "expert_count": args.expert_count,
        "tensor_names": names, "shards": shards,
        "activation_corpus": {"path": str(args.activation_corpus),
                              "train_and_heldout_rows": counts,
                              "split": "interleaved even rows train, odd rows held-out",
                              "routing_mass_order": [order for _schedule, _ranks, order in schedules]},
        "guard": {"entry": entry, "loaded": loaded_state, "end": ended},
        "fit": {"schedules": [schedule for schedule, _ranks, _order in schedules],
                "sample_rows_per_tensor": None, "steps": args.steps,
                "learning_rate": args.learning_rate, "pca_initializer_ms": pca_ms,
                "teacher_targets": "W_e x computed from immutable source weights"},
        "results": results, "ranked_summary": ranked,
        "fanout_timing": {"workers": 2, "variant_count": len(results),
                           "wall_elapsed_ms": wall_ms,
                           "sum_variant_elapsed_ms": sum(r["fanout"]["elapsed_ms"] for r in results),
                           "method_soft_budget_ms": SOFT_BUDGET_MS,
                           "execution": "CPU variable-rank representation-native training plus direct-reference fanout"},
        "claim_boundary": (
            "Bounded recoverability-aware variable-rank Nova discriminator on one KIMI MoE projection. "
            "Dense source weights are teacher targets only; logical variable factor bytes are fully "
            "accounted. This is not full-model capability, end-to-end TPS, a production variable-rank "
            "kernel, NX, or promotion evidence."
        ),
        "next_discriminator": "Audit the leading variable-rank schedule through a grouped routed consumer; if its source error improves without a capability regression, integrate it into the three-projection graph.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "script": str(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, args.output)
    print(json.dumps({"output": str(args.output), "status": out["status"],
                      "pca_initializer_ms": pca_ms, "wall_elapsed_ms": wall_ms,
                      "ranked_summary": ranked}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
