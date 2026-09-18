#!/usr/bin/env python3
"""Audit a persisted Nova organ through MLX's routed MoE contract.

The direct path consumes only the saved shared factors, expert cores, and
local residual factors.  KIMI_BASE expert weights are loaded solely for the
independent ``gather_mm`` oracle; they are never used by the direct consumer.
This is an organ-level execution audit, not a full-model or promotion claim.
"""
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
    NovaLocalResidualRoutedLinear,
    dense_oracle_from_factors,
)
from tools.future.kimi_representation_native_nova_fanout import (  # noqa: E402
    _split_corpus,
)
from tools.future.kimi_shared_pq_organ_reference import (  # noqa: E402
    _load_weights,
    _select_names,
)

SCHEMA = "hawking.future.kimi_nova_routed_kernel_audit.v1"
DEFAULT_REPS = 5


def _metrics(actual, expected) -> dict[str, Any]:
    import mlx.core as mx

    mx.eval(actual, expected)
    a = np.asarray(actual.astype(mx.float32)).reshape(-1, actual.shape[-1])
    e = np.asarray(expected.astype(mx.float32)).reshape(-1, expected.shape[-1])
    relative = np.linalg.norm(a - e, axis=1) / np.maximum(
        np.linalg.norm(e, axis=1), 1e-12
    )
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
    samples: list[float] = []
    for _ in range(reps):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def audit(receipt_path: Path, corpus_path: Path, output: Path, reps: int) -> dict[str, Any]:
    import mlx.core as mx

    if reps < 2:
        raise ValueError("reps must be at least 2")
    receipt = json.loads(receipt_path.read_text())
    best = min(receipt["results"], key=lambda row: row["heldout"]["mean_relative_l2"])
    artifact = Path(best["factor_artifact"]["path"])
    if not artifact.is_absolute():
        artifact = ROOT / artifact
    if not artifact.exists():
        raise FileNotFoundError(artifact)

    expert_count = int(receipt["expert_count"])
    names = _select_names(
        Path(receipt["specimen"]), expert_count, receipt["tensor_family"]
    )
    loaded, shards = _load_weights(Path(receipt["specimen"]), names)
    source = mx.stack(
        [mx.array(tensor.numpy()).astype(mx.bfloat16) for _name, tensor, _shard in loaded],
        axis=0,
    )
    width = int(source.shape[-1])
    output_dim = int(source.shape[-2])
    _train, heldout, counts = _split_corpus(corpus_path, expert_count, width)
    del _train

    consumer = NovaLocalResidualRoutedLinear.from_bf16_artifact(artifact, expert_count)
    # This reconstruction is an oracle-only control.  The direct consumer
    # above never forms it; keeping it separate makes approximation error
    # distinguishable from routed-kernel arithmetic/shape error.
    factor_dense = dense_oracle_from_factors(
        consumer.u, consumer.v, consumer.cores,
        consumer.residual_a, consumer.residual_b,
    )
    mx.eval(factor_dense, source)

    # Sorted routed layout: one activation row per selected expert.  The
    # corpus is grouped by expert, which is the same ordering produced by the
    # production SwitchGLU _gather_sort path before gather_mm.
    sorted_rows = np.concatenate(heldout, axis=0).astype(np.float32, copy=False)
    sorted_ids = np.concatenate(
        [np.full(rows.shape[0], expert, dtype=np.int32)
         for expert, rows in enumerate(heldout)],
        axis=0,
    )
    x_sorted = mx.array(sorted_rows).astype(mx.bfloat16)[:, None, :]
    ids_sorted = mx.array(sorted_ids, dtype=mx.int32)
    direct_sorted = consumer(x_sorted, ids_sorted, sorted_indices=True)
    factor_oracle_sorted = mx.gather_mm(
        x_sorted, factor_dense.swapaxes(-1, -2), rhs_indices=ids_sorted, sorted_indices=True
    )
    source_oracle_sorted = mx.gather_mm(
        x_sorted, source.swapaxes(-1, -2), rhs_indices=ids_sorted, sorted_indices=True
    )
    sorted_kernel_metrics = _metrics(direct_sorted, factor_oracle_sorted)
    sorted_source_metrics = _metrics(direct_sorted, source_oracle_sorted)
    mx.eval(direct_sorted, factor_oracle_sorted, source_oracle_sorted)
    direct_sorted_timing = _timed(
        lambda: consumer(x_sorted, ids_sorted, sorted_indices=True), reps
    )
    factor_oracle_sorted_timing = _timed(
        lambda: mx.eval(mx.gather_mm(
            x_sorted, factor_dense.swapaxes(-1, -2), rhs_indices=ids_sorted,
            sorted_indices=True
        )),
        reps,
    )
    source_oracle_sorted_timing = _timed(
        lambda: mx.eval(mx.gather_mm(
            x_sorted, source.swapaxes(-1, -2), rhs_indices=ids_sorted,
            sorted_indices=True
        )),
        reps,
    )

    # Unsorted top-k layout: each token row is intentionally routed to two
    # experts.  This exercises the shape/replication boundary used by
    # SwitchGLU before its optional sorting pass.
    pair_rows = min(16, int(heldout[0].shape[0]))
    if pair_rows < 2:
        raise RuntimeError("expert 0 needs at least two held-out rows for top-k audit")
    pair_x = mx.array(heldout[0][:pair_rows]).astype(mx.bfloat16).reshape(
        1, pair_rows, 1, width
    )
    pair_ids_np = np.stack(
        [np.zeros(pair_rows, dtype=np.int32),
         np.ones(pair_rows, dtype=np.int32)], axis=-1
    )[None, ...]
    pair_ids = mx.array(pair_ids_np, dtype=mx.int32)
    direct_pair = consumer(pair_x, pair_ids, sorted_indices=False)
    pair_base = pair_x.reshape(-1, 1, width)
    pair_flat_x = mx.repeat(pair_base, 2, axis=0)
    pair_flat_ids = pair_ids.reshape(-1)
    factor_oracle_pair = mx.gather_mm(
        pair_flat_x, factor_dense.swapaxes(-1, -2), rhs_indices=pair_flat_ids
    ).reshape(1, pair_rows, 2, 1, output_dim)
    source_oracle_pair = mx.gather_mm(
        pair_flat_x, source.swapaxes(-1, -2), rhs_indices=pair_flat_ids
    ).reshape(1, pair_rows, 2, 1, output_dim)
    pair_kernel_metrics = _metrics(direct_pair, factor_oracle_pair)
    pair_source_metrics = _metrics(direct_pair, source_oracle_pair)
    mx.eval(direct_pair, factor_oracle_pair, source_oracle_pair)
    direct_pair_timing = _timed(
        lambda: consumer(pair_x, pair_ids, sorted_indices=False), reps
    )
    factor_oracle_pair_timing = _timed(
        lambda: mx.eval(mx.gather_mm(
            pair_flat_x, factor_dense.swapaxes(-1, -2), rhs_indices=pair_flat_ids
        ).reshape(1, pair_rows, 2, 1, output_dim)),
        reps,
    )
    source_oracle_pair_timing = _timed(
        lambda: mx.eval(mx.gather_mm(
            pair_flat_x, source.swapaxes(-1, -2), rhs_indices=pair_flat_ids
        ).reshape(1, pair_rows, 2, 1, output_dim)),
        reps,
    )

    out = {
        "schema": SCHEMA,
        "status": "NOVA_MLX_ROUTED_KERNEL_AUDIT_PASS"
        if sorted_kernel_metrics["exact_shape"]
        and pair_kernel_metrics["exact_shape"]
        and sorted_kernel_metrics["max_abs_error"] <= 0.02
        and pair_kernel_metrics["max_abs_error"] <= 0.02
        else "NOVA_MLX_ROUTED_KERNEL_AUDIT_FAIL",
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
            "operator": "U(C_e(V^T x)) + A_e(B_e^T x)",
            "dense_parent_materialized": False,
            "dense_parent_role": "source_and_factor_oracles_only",
            "production_kernel": "validated organ-level direct consumer; not integrated into KIMI model",
            "routing_contract": "SwitchGLU/SwitchLinear gather_mm-compatible M=1 layouts",
        },
        "activation_corpus": {
            "path": str(corpus_path),
            "rows": counts,
            "sorted_rows": int(sorted_rows.shape[0]),
            "topk_pair_rows": pair_rows,
        },
        "sorted_route": {
            "input_shape": list(x_sorted.shape),
            "indices_shape": list(ids_sorted.shape),
            "kernel_equivalence": sorted_kernel_metrics,
            "source_approximation": sorted_source_metrics,
            "timing_ms": {
                "direct_median": statistics.median(direct_sorted_timing),
                "factor_oracle_median": statistics.median(factor_oracle_sorted_timing),
                "source_oracle_median": statistics.median(source_oracle_sorted_timing),
                "direct_samples": direct_sorted_timing,
                "factor_oracle_samples": factor_oracle_sorted_timing,
                "source_oracle_samples": source_oracle_sorted_timing,
            },
        },
        "unsorted_topk_route": {
            "input_shape": list(pair_x.shape),
            "indices_shape": list(pair_ids.shape),
            "kernel_equivalence": pair_kernel_metrics,
            "source_approximation": pair_source_metrics,
            "timing_ms": {
                "direct_median": statistics.median(direct_pair_timing),
                "factor_oracle_median": statistics.median(factor_oracle_pair_timing),
                "source_oracle_median": statistics.median(source_oracle_pair_timing),
                "direct_samples": direct_pair_timing,
                "factor_oracle_samples": factor_oracle_pair_timing,
                "source_oracle_samples": source_oracle_pair_timing,
            },
        },
        "claim_boundary": (
            "Independent MLX replay of a saved BF16 Nova factor artifact through both "
            "SwitchGLU routed tensor layouts against an actual KIMI_BASE gather_mm oracle. "
            "This is organ-level direct-execution evidence only; it is not full-model "
            "capability, end-to-end TPS, a sparse full-model decoder, NX, or promotion evidence."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "status": out["status"],
        "selected_variant": out["selected_variant"],
        "sorted": out["sorted_route"],
        "unsorted_topk": out["unsorted_topk_route"],
    }, indent=2))
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
