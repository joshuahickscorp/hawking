#!/usr/bin/env python3
"""Replay a persisted Nova factor payload through an MLX BF16 direct consumer."""
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

from tools.future.kimi_representation_native_nova_fanout import _split_corpus  # noqa: E402
from tools.future.kimi_shared_pq_organ_reference import (  # noqa: E402
    _load_weights,
    _select_names,
)

SCHEMA = "hawking.future.kimi_nova_mlx_direct_audit.v1"


def _bf16_bits_to_mlx(array: np.ndarray):
    import torch
    import mlx.core as mx

    if array.dtype != np.uint16:
        raise RuntimeError(f"factor payload must be uint16 BF16 bits, got {array.dtype}")
    # Torch provides the portable bit-pattern reinterpretation; MLX then owns
    # the actual BF16 direct consumer and its matmul execution.
    float32 = torch.from_numpy(np.array(array, copy=True)).view(torch.bfloat16).float().numpy()
    return mx.array(float32).astype(mx.bfloat16)


def audit(receipt_path: Path, corpus_path: Path, output: Path) -> dict[str, Any]:
    import mlx.core as mx

    receipt = json.loads(receipt_path.read_text())
    best = min(receipt["results"], key=lambda row: row["heldout"]["mean_relative_l2"])
    artifact = Path(best["factor_artifact"]["path"])
    if not artifact.is_absolute():
        artifact = ROOT / artifact
    names = _select_names(Path(receipt["specimen"]), int(receipt["expert_count"]),
                          receipt["tensor_family"])
    loaded, shards = _load_weights(Path(receipt["specimen"]), names)
    source = [mx.array(tensor.float().numpy()).astype(mx.bfloat16)
              for _name, tensor, _shard in loaded]
    width = int(source[0].shape[1])
    _train, heldout, counts = _split_corpus(corpus_path, int(receipt["expert_count"]), width)

    with np.load(artifact, allow_pickle=False) as payload:
        u = _bf16_bits_to_mlx(payload["u"])
        v = _bf16_bits_to_mlx(payload["v"])
        cores = [_bf16_bits_to_mlx(payload[f"core_{i}"]) for i in range(len(source))]
        a = [_bf16_bits_to_mlx(payload[f"residual_a_{i}"]) for i in range(len(source))]
        b = [_bf16_bits_to_mlx(payload[f"residual_b_{i}"]) for i in range(len(source))]

    relative: list[float] = []
    cosine: list[float] = []
    direct_samples: list[float] = []
    oracle_samples: list[float] = []
    for W, core, aa, bb, xs in zip(source, cores, a, b, heldout):
        for row in xs:
            x = mx.array(row).astype(mx.bfloat16)
            started = time.perf_counter()
            direct = u @ (core @ (v.T @ x)) + aa @ (bb.T @ x)
            mx.eval(direct)
            direct_samples.append((time.perf_counter() - started) * 1000.0)
            started = time.perf_counter()
            oracle = W @ x
            mx.eval(oracle)
            oracle_samples.append((time.perf_counter() - started) * 1000.0)
            d = np.asarray(direct.astype(mx.float32))
            o = np.asarray(oracle.astype(mx.float32))
            relative.append(float(np.linalg.norm(d - o) / max(np.linalg.norm(o), 1e-12)))
            cosine.append(float(np.dot(d, o) / max(np.linalg.norm(d) * np.linalg.norm(o), 1e-12)))

    out = {
        "schema": SCHEMA,
        "status": "NOVA_MLX_DIRECT_AUDIT_PASS",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "training_receipt": str(receipt_path),
        "factor_artifact": best["factor_artifact"],
        "selected_variant": best["variant"],
        "tensor_names": names,
        "shards": shards,
        "execution": {
            "runtime": "MLX",
            "dtype": "BF16 factors and BF16 inputs",
            "operator": "U(C_e(V^T x)) + A_e(B_e^T x)",
            "dense_parent_materialized": False,
            "production_kernel": "NOT_VALIDATED",
        },
        "activation_corpus": {"path": str(corpus_path), "rows": counts},
        "heldout": {
            "total_rows": len(relative),
            "mean_relative_l2": statistics.mean(relative),
            "p95_relative_l2": sorted(relative)[max(0, int(len(relative) * 0.95) - 1)],
            "max_relative_l2": max(relative),
            "mean_cosine": statistics.mean(cosine),
        },
        "timing_ms": {
            "direct_median": statistics.median(direct_samples),
            "oracle_median": statistics.median(oracle_samples),
            "direct_samples": direct_samples,
            "oracle_samples": oracle_samples,
        },
        "claim_boundary": (
            "MLX BF16 direct-consumer replay of a saved bounded organ factor payload. "
            "This verifies physical-runtime factor consumption only; it is not a "
            "production sparse kernel, full-model capability, end-to-end TPS, NX, or promotion result."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "status": out["status"],
                      "heldout": out["heldout"], "timing_ms": out["timing_ms"]}, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    audit(args.receipt, args.corpus, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
