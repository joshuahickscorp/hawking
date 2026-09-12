#!/usr/bin/env python3
"""Audit a persisted row-wise-INT8 factor organ through MLX routing."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools" / "future") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools" / "future"))

from tools.future.kimi_nova_routed_kernel import (  # noqa: E402
    Int8ExpertLowRankRoutedLinear,
)
from tools.future.kimi_representation_native_nova_fanout import (  # noqa: E402
    _split_corpus,
)
from tools.future.kimi_shared_pq_organ_reference import (  # noqa: E402
    _load_weights,
    _select_names,
)

SCHEMA = "hawking.future.kimi_int8_expert_lowrank_routed_audit.v1"


def _metrics(actual, expected) -> dict[str, Any]:
    import mlx.core as mx

    mx.eval(actual, expected)
    a = np.asarray(actual.astype(mx.float32)).reshape(-1, actual.shape[-1])
    e = np.asarray(expected.astype(mx.float32)).reshape(-1, expected.shape[-1])
    relative = np.linalg.norm(a - e, axis=1) / np.maximum(np.linalg.norm(e, axis=1), 1e-12)
    cosine = np.sum(a * e, axis=1) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(e, axis=1), 1e-12
    )
    return {
        "rows": int(a.shape[0]),
        "max_abs_error": float(np.max(np.abs(a - e))),
        "mean_abs_error": float(np.mean(np.abs(a - e))),
        "mean_relative_l2": float(np.mean(relative)),
        "p95_relative_l2": float(np.percentile(relative, 95)),
        "max_relative_l2": float(np.max(relative)),
        "mean_cosine": float(np.mean(cosine)),
        "exact_shape": list(actual.shape) == list(expected.shape),
        "direct_shape": list(actual.shape),
        "oracle_shape": list(expected.shape),
    }


def _timed(fn, reps: int) -> list[float]:
    values = []
    for _ in range(reps):
        started = time.perf_counter()
        fn()
        values.append((time.perf_counter() - started) * 1000.0)
    return values


def audit(receipt_path: Path, corpus_path: Path, output: Path, reps: int,
          requested_rank: int | None = None) -> dict[str, Any]:
    import mlx.core as mx

    receipt = json.loads(receipt_path.read_text())
    eligible = [row for row in receipt["results"]
                if row["representation"]["complete_accounting"]["complete_ebpw"] <= 1.0]
    if requested_rank is not None:
        eligible = [row for row in eligible if int(row["variant"]["rank"]) == requested_rank]
    if not eligible:
        suffix = f" at rank {requested_rank}" if requested_rank is not None else ""
        raise ValueError(f"receipt contains no under-1 complete-EBPW candidate{suffix}")
    best = min(eligible, key=lambda row: row["heldout"]["mean_relative_l2"])
    artifact = Path(best["factor_artifact"]["path"])
    if not artifact.is_absolute():
        artifact = ROOT / artifact
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    expert_count = int(receipt["expert_count"])
    names = _select_names(Path(receipt["specimen"]), expert_count, receipt["tensor_family"])
    loaded, shards = _load_weights(Path(receipt["specimen"]), names)
    source = mx.stack([mx.array(tensor.numpy()).astype(mx.bfloat16)
                       for _name, tensor, _shard in loaded], axis=0)
    width = int(source.shape[-1])
    output_dim = int(source.shape[-2])
    _train, heldout, counts = _split_corpus(corpus_path, expert_count, width)
    del _train
    consumer = Int8ExpertLowRankRoutedLinear.from_int8_artifact(artifact, expert_count)
    factor_a, factor_b = consumer.dequantized_factors()
    factor_dense = mx.einsum("eor,eir->eoi", factor_a, factor_b)
    mx.eval(factor_dense, source)

    sorted_rows = np.concatenate(heldout, axis=0).astype(np.float32, copy=False)
    sorted_ids = np.concatenate([np.full(rows.shape[0], expert, dtype=np.int32)
                                 for expert, rows in enumerate(heldout)], axis=0)
    x_sorted = mx.array(sorted_rows).astype(mx.bfloat16)[:, None, :]
    ids_sorted = mx.array(sorted_ids, dtype=mx.int32)
    direct_sorted = consumer(x_sorted, ids_sorted, sorted_indices=True)
    factor_oracle_sorted = mx.gather_mm(
        x_sorted, factor_dense.swapaxes(-1, -2), rhs_indices=ids_sorted, sorted_indices=True)
    source_oracle_sorted = mx.gather_mm(
        x_sorted, source.swapaxes(-1, -2), rhs_indices=ids_sorted, sorted_indices=True)
    sorted_kernel = _metrics(direct_sorted, factor_oracle_sorted)
    sorted_source = _metrics(direct_sorted, source_oracle_sorted)
    sorted_direct_t = _timed(lambda: consumer(x_sorted, ids_sorted, sorted_indices=True), reps)
    sorted_factor_t = _timed(lambda: mx.eval(mx.gather_mm(
        x_sorted, factor_dense.swapaxes(-1, -2), rhs_indices=ids_sorted,
        sorted_indices=True)), reps)

    pair_rows = min(16, int(heldout[0].shape[0]))
    if pair_rows < 2:
        raise RuntimeError("expert 0 needs at least two held-out rows")
    pair_x = mx.array(heldout[0][:pair_rows]).astype(mx.bfloat16).reshape(1, pair_rows, 1, width)
    pair_ids = mx.array(np.stack([
        np.zeros(pair_rows, dtype=np.int32), np.ones(pair_rows, dtype=np.int32)
    ], axis=-1)[None, ...], dtype=mx.int32)
    direct_pair = consumer(pair_x, pair_ids, sorted_indices=False)
    pair_flat_x = mx.repeat(pair_x.reshape(-1, 1, width), 2, axis=0)
    pair_flat_ids = pair_ids.reshape(-1)
    factor_pair = mx.gather_mm(
        pair_flat_x, factor_dense.swapaxes(-1, -2), rhs_indices=pair_flat_ids
    ).reshape(1, pair_rows, 2, 1, output_dim)
    source_pair = mx.gather_mm(
        pair_flat_x, source.swapaxes(-1, -2), rhs_indices=pair_flat_ids
    ).reshape(1, pair_rows, 2, 1, output_dim)
    pair_kernel = _metrics(direct_pair, factor_pair)
    pair_source = _metrics(direct_pair, source_pair)
    pair_direct_t = _timed(lambda: consumer(pair_x, pair_ids, sorted_indices=False), reps)
    pair_factor_t = _timed(lambda: mx.eval(mx.gather_mm(
        pair_flat_x, factor_dense.swapaxes(-1, -2), rhs_indices=pair_flat_ids
    ).reshape(1, pair_rows, 2, 1, output_dim)), reps)
    out = {
        "schema": SCHEMA,
        "status": "INT8_EXPERT_LOWRANK_ROUTED_AUDIT_PASS"
        if sorted_kernel["exact_shape"] and pair_kernel["exact_shape"]
        and sorted_kernel["max_abs_error"] <= 0.03 and pair_kernel["max_abs_error"] <= 0.03
        else "INT8_EXPERT_LOWRANK_ROUTED_AUDIT_FAIL",
        "source_model": "KIMI_BASE", "immutable_base": True,
        "training_receipt": str(receipt_path), "source_specimen": receipt["specimen"],
        "tensor_names": names, "shards": shards, "selected_variant": best["variant"],
        "selection": {"constraint": "complete_ebpw <= 1.0"
                       + (f" and rank == {requested_rank}" if requested_rank is not None else ""),
                       "eligible_variants": [row["variant"] for row in eligible]},
        "factor_artifact": best["factor_artifact"],
        "execution": {"runtime": "MLX", "dtype": "INT8 factors + FP32 scales, BF16 inputs",
                      "operator": "y_e = dequant(A_e) @ (dequant(B_e)^T x)",
                      "kernel": "two-stage gather_mm; B row scales precondition selected inputs and A row scales postcondition outputs",
                      "dense_parent_materialized": False,
                      "dense_parent_role": "source_and_dequantized_factor_oracles_only",
                      "production_kernel": "organ-level direct consumer; not full-model integrated",
                      "routing_contract": "SwitchGLU/SwitchLinear M=1 sorted and unsorted layouts"},
        "activation_corpus": {"path": str(corpus_path), "rows": counts,
                              "sorted_rows": int(sorted_rows.shape[0]), "topk_pair_rows": pair_rows},
        "sorted_route": {"input_shape": list(x_sorted.shape), "indices_shape": list(ids_sorted.shape),
                         "kernel_equivalence": sorted_kernel, "source_approximation": sorted_source,
                         "timing_ms": {"direct_median": statistics.median(sorted_direct_t),
                                       "factor_oracle_median": statistics.median(sorted_factor_t),
                                       "direct_samples": sorted_direct_t,
                                       "factor_oracle_samples": sorted_factor_t}},
        "unsorted_topk_route": {"input_shape": list(pair_x.shape), "indices_shape": list(pair_ids.shape),
                                "kernel_equivalence": pair_kernel, "source_approximation": pair_source,
                                "timing_ms": {"direct_median": statistics.median(pair_direct_t),
                                              "factor_oracle_median": statistics.median(pair_factor_t),
                                              "direct_samples": pair_direct_t,
                                              "factor_oracle_samples": pair_factor_t}},
        "claim_boundary": "Organ-level INT8 direct-routing evidence only; not full-model capability, TPS, NX, or promotion evidence.",
        "generated_at": datetime.now(timezone.utc).isoformat(), "script": str(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "status": out["status"],
                      "selected_variant": out["selected_variant"],
                      "sorted": out["sorted_route"], "unsorted_topk": out["unsorted_topk_route"]}, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rank", type=int, default=None,
                    help="audit one exact rank instead of selecting the best under-1 arm")
    args = ap.parse_args()
    if args.reps < 2:
        raise SystemExit("reps must be at least 2")
    out = audit(args.receipt, args.corpus, args.output, args.reps, args.rank)
    return 0 if out["status"].endswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
