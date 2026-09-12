#!/usr/bin/env python3
"""Output-aware protected-row fanout for a shared KIMI PQ organ reference.

This keeps one shared PQ representation and adds a sparse set of exact BF16
rows selected by their observed output error for a fixed probe vector.  The
selection is intentionally a discriminator, not a general capability claim:
it measures whether a small protected-island budget can repair a cheap MoE
representation without hiding dense-parent bytes.
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
    _fit_shared,
    _load_weights,
    _select_names,
    _direct_mv,
)

SCHEMA = "hawking.future.kimi_shared_pq_residual_fanout.v1"
DEFAULT_EXPERT_COUNT = 8
DEFAULT_SAMPLE_ROWS_PER_TENSOR = 32
BASE_D = 32
BASE_ENTRIES = 16
BASE_STEPS = 2
METADATA_BYTES = 4096
DECODER_SUPPORT_BYTES = 65536
SOFT_BUDGET_MS = 30000.0
FRACTIONS = (0.0, 0.01, 0.02, 0.05, 0.10)


def _run_fraction(
    weights,
    names: list[str],
    codebooks,
    codes,
    inputs,
    rows: int,
    width: int,
    groups: int,
    fraction: float,
) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    direct_results = []
    oracle_results = []
    selected_rows: list[torch.Tensor] = []
    residual_values: list[torch.Tensor] = []
    base_errors: list[torch.Tensor] = []
    for weight, code, x in zip(weights, codes, inputs):
        direct = _direct_mv(codebooks, code, x, width=width, d=BASE_D)
        oracle = weight.matmul(x)
        error = direct - oracle
        base_errors.append(error)
        count = int(math.ceil(rows * fraction))
        if count:
            selected = torch.topk(error.abs(), k=count, largest=True, sorted=False).indices.sort().values
            selected_rows.append(selected)
            residual_values.append(weight.index_select(0, selected).to(dtype=torch.bfloat16))
            direct = direct.clone()
            direct[selected] = residual_values[-1].to(dtype=torch.float32).matmul(x)
        else:
            selected_rows.append(torch.empty(0, dtype=torch.int64))
            residual_values.append(torch.empty((0, width), dtype=torch.bfloat16))
        direct_results.append(direct)
        oracle_results.append(oracle)

    residual_row_count = sum(int(x.numel()) for x in selected_rows)
    # A 32-bit row locator is intentionally billed even though the current
    # bounded tensors would fit a smaller index: future organs may not.
    residual_index_bytes = residual_row_count * 4
    residual_value_bytes = residual_row_count * width * 2
    index_bytes = len(weights) * rows * groups
    codebook_bytes = groups * BASE_ENTRIES * BASE_D * 2
    candidate = candidate_from_parts(
        family_id=f"KIMI_shared_pq_d32_e16_residual_{fraction:.4f}",
        parent_params=sum(int(weight.numel()) for weight in weights),
        parts={
            "regions": [
                {"name": "shared_pq_indices", "bytes": index_bytes, "stream_class": STREAM_WEIGHT_CODES},
                {"name": "protected_row_ids", "bytes": residual_index_bytes, "stream_class": STREAM_WEIGHT_CODES},
            ],
            "generators": [],
            "metadata": [{"name": "organ_metadata", "bytes": METADATA_BYTES, "stream_class": STREAM_BROADCAST_AUX}],
            "tables": [{"name": "shared_bf16_codebooks", "bytes": codebook_bytes, "stream_class": STREAM_WEIGHT_CODES}],
            "residuals": [{"name": "protected_bf16_rows", "bytes": residual_value_bytes, "stream_class": STREAM_WEIGHT_CODES}],
            "runtime_auxiliaries": [{"name": "organ_direct_decoder_support", "bytes": DECODER_SUPPORT_BYTES, "stream_class": STREAM_BROADCAST_AUX}],
            "representation": [],
            "model_specific_code": [],
        },
        reconstructs_dense_parent=False,
        consumes_representation_directly=True,
    )
    accounting = cost(candidate)
    relative_l2 = []
    cosine = []
    for direct, oracle in zip(direct_results, oracle_results):
        error = direct - oracle
        relative_l2.append(float(torch.linalg.vector_norm(error).item() / max(torch.linalg.vector_norm(oracle).item(), 1e-12)))
        cosine.append(float(torch.nn.functional.cosine_similarity(direct[None, :], oracle[None, :]).item()))

    # Time the actual direct representation, including sparse protected-row
    # repair, separately from the dense oracle.  Fit is outside this timing.
    def direct_batch():
        out = []
        for weight, code, x, selected, residual in zip(weights, codes, inputs, selected_rows, residual_values):
            result = _direct_mv(codebooks, code, x, width=width, d=BASE_D)
            if selected.numel():
                result = result.clone()
                result[selected] = residual.to(dtype=torch.float32).matmul(x)
            out.append(result)
        return out

    def dense_batch():
        return [weight.matmul(x) for weight, x in zip(weights, inputs)]

    direct_samples = []
    dense_samples = []
    for _ in range(2):
        tic = time.perf_counter()
        direct_batch()
        direct_samples.append((time.perf_counter() - tic) * 1000.0)
        tic = time.perf_counter()
        dense_batch()
        dense_samples.append((time.perf_counter() - tic) * 1000.0)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "fraction": fraction,
        "status": "PROTECTED_RESIDUAL_REFERENCE_PASS",
        "representation": {
            "base_family": "shared_codebook_rowwise_pq_d32_e16",
            "protected_selection": "top output-error rows for one fixed probe vector",
            "protected_fraction_of_rows_per_tensor": fraction,
            "protected_row_count": residual_row_count,
            "protected_row_id_bytes": residual_index_bytes,
            "protected_bf16_value_bytes": residual_value_bytes,
            "complete_accounting": accounting,
        },
        "direct_execution": {
            "verified": True,
            "dense_parent_materialized_by_direct_path": False,
            "transient_group_expansion_bytes": rows * BASE_D * 4,
            "persistent_representation_bytes": accounting["executable_bytes"],
            "production_kernel": "NOT_VALIDATED",
        },
        "error": {
            "base_mean_relative_l2": statistics.mean(
                float(torch.linalg.vector_norm(error).item() / max(torch.linalg.vector_norm(oracle).item(), 1e-12))
                for error, oracle in zip(base_errors, oracle_results)
            ),
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
            "selection_and_probe_elapsed_ms": elapsed_ms,
            "fit_excluded_from_direct_timing": True,
            "torch_threads": 1,
        },
        "fanout": {
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
    ap.add_argument("--workers", type=int, default=len(FRACTIONS))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not 2 <= args.expert_count <= 64 or args.sample_rows <= 0 or args.workers <= 0:
        raise SystemExit("expert-count must be 2..64; sample-rows and workers must be positive")

    names = _select_names(args.spec, args.expert_count, "up_proj.weight")
    loaded, shards = _load_weights(args.spec, names)
    weights = [tensor for _name, tensor, _shard in loaded]
    fit_started = time.perf_counter()
    codebooks, codes, _scales, rows, width, groups = _fit_shared(
        weights, d=BASE_D, entries=BASE_ENTRIES, steps=BASE_STEPS, sample_rows=args.sample_rows, scale_mode="none"
    )
    fit_ms = (time.perf_counter() - fit_started) * 1000.0
    torch.manual_seed(20260910)
    inputs = [torch.randn(width, dtype=torch.float32) for _ in weights]
    started = time.perf_counter()
    workers = min(args.workers, len(FRACTIONS))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_fraction, weights, names, codebooks, codes, inputs, rows, width, groups, fraction) for fraction in FRACTIONS]
        results = [future.result() for future in futures]
    wall_ms = (time.perf_counter() - started) * 1000.0

    def summary(result: dict[str, Any]) -> dict[str, Any]:
        accounting = result["representation"]["complete_accounting"]
        return {
            "fraction": result["fraction"],
            "complete_ebpw": accounting["complete_ebpw"],
            "mean_relative_l2": result["error"]["mean_relative_l2"],
            "mean_cosine": result["error"]["mean_cosine"],
            "direct_batch_median_ms": result["timing"]["direct_batch_median_ms"],
            "dense_oracle_batch_median_ms": result["timing"]["dense_oracle_batch_median_ms"],
        }

    out = {
        "schema": SCHEMA,
        "status": "PROTECTED_RESIDUAL_FANOUT_COMPLETE",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "specimen": str(args.spec),
        "organ": f"layer10_routed_experts_0_to_{args.expert_count - 1}",
        "tensor_family": "up_proj.weight",
        "expert_count": args.expert_count,
        "tensor_names": names,
        "shards": shards,
        "base_fit": {"d": BASE_D, "entries": BASE_ENTRIES, "steps": BASE_STEPS, "sample_rows_per_tensor": args.sample_rows, "fit_ms": fit_ms},
        "results": results,
        "fanout_timing": {
            "workers": workers,
            "fraction_count": len(results),
            "wall_elapsed_ms": wall_ms,
            "method_soft_budget_ms": SOFT_BUDGET_MS,
            "execution": "CPU protected-residual fanout; not model TPS",
        },
        "ranked_summary": sorted((summary(result) for result in results), key=lambda row: (row["complete_ebpw"], row["mean_relative_l2"])),
        "claim_boundary": (
            "Bounded output-aware protected-row reference only. Residual selection is tied to one "
            "probe vector, so it is a discriminator rather than a general capability result. "
            "No dense parent is materialized by the direct consumer; no full-model, TPS, NX, "
            "production-kernel, or KIMI-promotion claim is made."
        ),
        "next_discriminator": "Repeat selection across held-out probes, replace probe-specific rows with a recoverability/output-aware allocator, and then test organ capability.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "script": str(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": out["status"], "wall_elapsed_ms": wall_ms, "ranked_summary": out["ranked_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
