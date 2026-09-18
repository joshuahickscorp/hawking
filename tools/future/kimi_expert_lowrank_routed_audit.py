#!/usr/bin/env python3
"""Audit independent expert low-rank factors through MLX routed execution."""
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

from tools.future.kimi_expert_lowrank_nova_fanout import _split_corpus  # noqa: E402
from tools.future.kimi_nova_routed_kernel import ExpertLowRankRoutedLinear  # noqa: E402
from tools.future.kimi_nova_routed_kernel_audit import _metrics, _timed  # noqa: E402
from tools.future.kimi_shared_pq_organ_reference import (  # noqa: E402
    _load_weights,
    _select_names,
)

SCHEMA = "hawking.future.kimi_expert_lowrank_routed_audit.v1"
DEFAULT_REPS = 5


def audit(receipt_path: Path, corpus_path: Path, output: Path, reps: int) -> dict[str, Any]:
    import mlx.core as mx

    receipt = json.loads(receipt_path.read_text())
    best = min(receipt["results"], key=lambda row: row["heldout"]["mean_relative_l2"])
    artifact = Path(best["factor_artifact"]["path"])
    if not artifact.is_absolute():
        artifact = ROOT / artifact
    expert_count = int(receipt["expert_count"])
    names = _select_names(Path(receipt["specimen"]), expert_count, receipt["tensor_family"])
    loaded, shards = _load_weights(Path(receipt["specimen"]), names)
    source = mx.stack(
        [mx.array(tensor.numpy()).astype(mx.bfloat16) for _name, tensor, _shard in loaded],
        axis=0,
    )
    width = int(source.shape[-1])
    output_dim = int(source.shape[-2])
    _train, heldout, counts = _split_corpus(corpus_path, expert_count, width)
    del _train
    consumer = ExpertLowRankRoutedLinear.from_bf16_artifact(artifact, expert_count)
    factor_dense = mx.einsum("eor,eir->eoi", consumer.factor_a, consumer.factor_b)
    mx.eval(source, factor_dense)

    sorted_rows = np.concatenate(heldout, axis=0).astype(np.float32, copy=False)
    sorted_ids = np.concatenate([
        np.full(rows.shape[0], expert, dtype=np.int32)
        for expert, rows in enumerate(heldout)
    ], axis=0)
    x_sorted = mx.array(sorted_rows).astype(mx.bfloat16)[:, None, :]
    ids_sorted = mx.array(sorted_ids, dtype=mx.int32)
    direct_sorted = consumer(x_sorted, ids_sorted, sorted_indices=True)
    factor_sorted = mx.gather_mm(
        x_sorted, factor_dense.swapaxes(-1, -2), rhs_indices=ids_sorted, sorted_indices=True
    )
    source_sorted = mx.gather_mm(
        x_sorted, source.swapaxes(-1, -2), rhs_indices=ids_sorted, sorted_indices=True
    )
    sorted_kernel = _metrics(direct_sorted, factor_sorted)
    sorted_source = _metrics(direct_sorted, source_sorted)
    direct_sorted_timing = _timed(
        lambda: consumer(x_sorted, ids_sorted, sorted_indices=True), reps
    )
    factor_sorted_timing = _timed(
        lambda: mx.eval(mx.gather_mm(
            x_sorted, factor_dense.swapaxes(-1, -2), rhs_indices=ids_sorted,
            sorted_indices=True
        )), reps
    )
    source_sorted_timing = _timed(
        lambda: mx.eval(mx.gather_mm(
            x_sorted, source.swapaxes(-1, -2), rhs_indices=ids_sorted,
            sorted_indices=True
        )), reps
    )

    pair_rows = min(16, int(heldout[0].shape[0]))
    pair_x = mx.array(heldout[0][:pair_rows]).astype(mx.bfloat16).reshape(
        1, pair_rows, 1, width
    )
    pair_ids = mx.array(np.stack([
        np.zeros(pair_rows, dtype=np.int32), np.ones(pair_rows, dtype=np.int32)
    ], axis=-1)[None, ...], dtype=mx.int32)
    pair_base = pair_x.reshape(-1, 1, width)
    pair_flat_x = mx.repeat(pair_base, 2, axis=0)
    pair_flat_ids = pair_ids.reshape(-1)
    direct_pair = consumer(pair_x, pair_ids)
    factor_pair = mx.gather_mm(
        pair_flat_x, factor_dense.swapaxes(-1, -2), rhs_indices=pair_flat_ids
    ).reshape(1, pair_rows, 2, 1, output_dim)
    source_pair = mx.gather_mm(
        pair_flat_x, source.swapaxes(-1, -2), rhs_indices=pair_flat_ids
    ).reshape(1, pair_rows, 2, 1, output_dim)
    pair_kernel = _metrics(direct_pair, factor_pair)
    pair_source = _metrics(direct_pair, source_pair)
    direct_pair_timing = _timed(lambda: consumer(pair_x, pair_ids), reps)
    factor_pair_timing = _timed(
        lambda: mx.eval(mx.gather_mm(
            pair_flat_x, factor_dense.swapaxes(-1, -2), rhs_indices=pair_flat_ids
        ).reshape(1, pair_rows, 2, 1, output_dim)), reps
    )
    source_pair_timing = _timed(
        lambda: mx.eval(mx.gather_mm(
            pair_flat_x, source.swapaxes(-1, -2), rhs_indices=pair_flat_ids
        ).reshape(1, pair_rows, 2, 1, output_dim)), reps
    )

    kernel_pass = (
        sorted_kernel["exact_shape"] and pair_kernel["exact_shape"]
        and sorted_kernel["max_abs_error"] <= 0.02
        and pair_kernel["max_abs_error"] <= 0.02
    )
    out = {
        "schema": SCHEMA,
        "status": "EXPERT_LOWRANK_ROUTED_AUDIT_PASS" if kernel_pass
        else "EXPERT_LOWRANK_ROUTED_AUDIT_FAIL",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "training_receipt": str(receipt_path),
        "source_specimen": receipt["specimen"],
        "tensor_names": names,
        "shards": shards,
        "selected_variant": best["variant"],
        "factor_artifact": best["factor_artifact"],
        "execution": {
            "runtime": "MLX",
            "dtype": "BF16 factors, BF16 inputs, BF16 routed oracle",
            "operator": "A_e(B_e^T x)",
            "dense_parent_materialized": False,
            "dense_parent_role": "source_and_factor_oracles_only",
            "routing_contract": "SwitchGLU/SwitchLinear gather_mm-compatible M=1 layouts",
            "production_kernel": "validated organ-level direct consumer; not integrated into KIMI model",
        },
        "activation_corpus": {
            "path": str(corpus_path), "rows": counts,
            "sorted_rows": int(sorted_rows.shape[0]), "topk_pair_rows": pair_rows,
        },
        "sorted_route": {
            "input_shape": list(x_sorted.shape), "indices_shape": list(ids_sorted.shape),
            "kernel_equivalence": sorted_kernel,
            "source_approximation": sorted_source,
            "timing_ms": {
                "direct_median": statistics.median(direct_sorted_timing),
                "factor_oracle_median": statistics.median(factor_sorted_timing),
                "source_oracle_median": statistics.median(source_sorted_timing),
                "direct_samples": direct_sorted_timing,
                "factor_oracle_samples": factor_sorted_timing,
                "source_oracle_samples": source_sorted_timing,
            },
        },
        "unsorted_topk_route": {
            "input_shape": list(pair_x.shape), "indices_shape": list(pair_ids.shape),
            "kernel_equivalence": pair_kernel,
            "source_approximation": pair_source,
            "timing_ms": {
                "direct_median": statistics.median(direct_pair_timing),
                "factor_oracle_median": statistics.median(factor_pair_timing),
                "source_oracle_median": statistics.median(source_pair_timing),
                "direct_samples": direct_pair_timing,
                "factor_oracle_samples": factor_pair_timing,
                "source_oracle_samples": source_pair_timing,
            },
        },
        "claim_boundary": (
            "Independent MLX replay of a saved BF16 per-expert low-rank artifact through "
            "both SwitchGLU routed layouts. Kernel equivalence and source approximation "
            "are separate measurements. This is organ-level evidence only, not full-model "
            "capability, end-to-end TPS, NX, or promotion evidence."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "status": out["status"],
                      "selected_variant": out["selected_variant"],
                      "sorted": out["sorted_route"],
                      "unsorted_topk": out["unsorted_topk_route"]}, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    args = ap.parse_args()
    out = audit(args.receipt, args.corpus, args.output, args.reps)
    return 0 if out["status"].endswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
