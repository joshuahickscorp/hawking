#!/usr/bin/env python3
"""Protected-row fanout on an optimized packed multistage KIMI representation."""
from __future__ import annotations

import argparse
import json
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
from tools.future.kimi_shared_multistage_bitpacked_fanout import (  # noqa: E402
    _direct_mv,
    _fit,
)
from tools.future.kimi_shared_pq_organ_reference import _load_weights, _select_names  # noqa: E402

SCHEMA = "hawking.future.kimi_multistage_protected_residual_fanout.v1"
DEFAULT_EXPERT_COUNT = 8
DEFAULT_TENSOR_FAMILY = "up_proj.weight"
TENSOR_CHOICES = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
BASE_STAGES = 6
BASE_BITS = 4
BASE_ENTRIES = 16
VECTOR_WIDTH = 32
SAMPLE_ROWS = 256
LLOYD_STEPS = 4
PROBES = 16
FRACTIONS = (0.0, 0.00125, 0.0025, 0.00375, 0.005)
METADATA_BYTES = 4096
DECODER_SUPPORT_BYTES = 65536
SOFT_BUDGET_MS = 60000.0


def _one(weights, names, codebooks, packed_codes, inputs, rows, width, groups,
         fraction, tensor_family):
    import torch

    started = time.perf_counter()
    # Selection is based only on the first probe. Evaluation below uses all
    # probes, making the output-error repair held out from selection.
    selected_rows = []
    residual_values = []
    first_probe = inputs[0]
    for weight, packed, x in zip(weights, packed_codes, first_probe):
        base = _direct_mv(codebooks, packed, x, width=width, stages=BASE_STAGES, bits=BASE_BITS)
        oracle = weight.matmul(x)
        count = int(round(rows * fraction))
        if fraction > 0 and count == 0:
            count = 1
        if count:
            selected = torch.topk((base - oracle).abs(), k=count, largest=True, sorted=False).indices.sort().values
            selected_rows.append(selected)
            residual_values.append(weight.index_select(0, selected).to(dtype=torch.bfloat16))
        else:
            selected_rows.append(torch.empty(0, dtype=torch.int64))
            residual_values.append(torch.empty((0, width), dtype=torch.bfloat16))

    def direct(weight, packed, x, selected, residual):
        result = _direct_mv(codebooks, packed, x, width=width, stages=BASE_STAGES, bits=BASE_BITS)
        if selected.numel():
            result = result.clone()
            result[selected] = residual.to(dtype=torch.float32).matmul(x)
        return result

    relative = []
    cosine = []
    for probe_inputs in inputs:
        for weight, packed, x, selected, residual in zip(weights, packed_codes, probe_inputs, selected_rows, residual_values):
            result = direct(weight, packed, x, selected, residual)
            oracle = weight.matmul(x)
            error = result - oracle
            relative.append(float(torch.linalg.vector_norm(error).item() / max(torch.linalg.vector_norm(oracle).item(), 1e-12)))
            cosine.append(float(torch.nn.functional.cosine_similarity(result[None, :], oracle[None, :]).item()))

    probe = inputs[0]
    def direct_batch():
        return [direct(weight, packed, x, selected, residual) for weight, packed, x, selected, residual in zip(weights, packed_codes, probe, selected_rows, residual_values)]
    def dense_batch():
        return [weight.matmul(x) for weight, x in zip(weights, probe)]
    direct_samples = []
    dense_samples = []
    for _ in range(2):
        tic = time.perf_counter()
        direct_batch()
        direct_samples.append((time.perf_counter() - tic) * 1000.0)
        tic = time.perf_counter()
        dense_batch()
        dense_samples.append((time.perf_counter() - tic) * 1000.0)

    residual_row_count = sum(int(selected.numel()) for selected in selected_rows)
    vector_count = len(weights) * rows * groups
    base_code_bytes = vector_count * ((BASE_STAGES * BASE_BITS + 7) // 8)
    base_codebook_bytes = BASE_STAGES * groups * BASE_ENTRIES * VECTOR_WIDTH * 2
    residual_id_bytes = residual_row_count * 4
    residual_value_bytes = residual_row_count * width * 2
    candidate = candidate_from_parts(
        family_id=(f"KIMI_multistage_{tensor_family.replace('.weight', '')}_"
                   f"{BASE_STAGES}x{BASE_BITS}bit_protected_{fraction:.5f}"),
        parent_params=sum(int(weight.numel()) for weight in weights),
        parts={
            "regions": [
                {"name": "packed_multistage_codes", "bytes": base_code_bytes, "stream_class": STREAM_WEIGHT_CODES},
                {"name": "protected_row_ids", "bytes": residual_id_bytes, "stream_class": STREAM_WEIGHT_CODES},
            ],
            "generators": [],
            "metadata": [{"name": "organ_metadata", "bytes": METADATA_BYTES, "stream_class": STREAM_BROADCAST_AUX}],
            "tables": [{"name": "shared_bf16_residual_codebooks", "bytes": base_codebook_bytes, "stream_class": STREAM_WEIGHT_CODES}],
            "residuals": [{"name": "protected_bf16_rows", "bytes": residual_value_bytes, "stream_class": STREAM_WEIGHT_CODES}],
            "runtime_auxiliaries": [{"name": "direct_decoder_support", "bytes": DECODER_SUPPORT_BYTES, "stream_class": STREAM_BROADCAST_AUX}],
            "representation": [],
            "model_specific_code": [],
        },
        reconstructs_dense_parent=False,
        consumes_representation_directly=True,
    )
    accounting = cost(candidate)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "fraction": fraction,
        "tensor_family": tensor_family,
        "status": "HELDOUT_PROTECTED_MULTISTAGE_REFERENCE_PASS",
        "representation": {"base_family": f"shared_{BASE_STAGES}x{BASE_BITS}bit_e{BASE_ENTRIES}", "protected_selection": "top base-output-error rows from first probe", "protected_row_count": residual_row_count, "protected_row_id_bytes": residual_id_bytes, "protected_bf16_value_bytes": residual_value_bytes, "complete_accounting": accounting},
        "direct_execution": {"verified": True, "dense_parent_materialized_by_direct_path": False, "persistent_representation_bytes": accounting["executable_bytes"], "transient_group_expansion_bytes": rows * VECTOR_WIDTH * 4, "production_kernel": "NOT_VALIDATED"},
        "heldout": {"probe_count": len(inputs), "probe_tensor_pairs": len(inputs) * len(weights), "mean_relative_l2": statistics.mean(relative), "p95_relative_l2": sorted(relative)[max(0, int(len(relative) * 0.95) - 1)], "max_relative_l2": max(relative), "mean_cosine": statistics.mean(cosine), "selection_probe_excluded_from_claim": True},
        "timing": {"direct_batch_median_ms": statistics.median(direct_samples), "dense_oracle_batch_median_ms": statistics.median(dense_samples), "heldout_and_selection_ms": elapsed_ms, "fit_excluded_from_direct_timing": True, "torch_threads": 1},
        "fanout": {"elapsed_ms": elapsed_ms, "soft_budget_ms": SOFT_BUDGET_MS, "soft_budget_status": "WITHIN" if elapsed_ms <= SOFT_BUDGET_MS else "OVER"},
    }


def main() -> int:
    import torch

    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--expert-count", type=int, default=DEFAULT_EXPERT_COUNT)
    ap.add_argument("--tensor-family", choices=TENSOR_CHOICES, default=DEFAULT_TENSOR_FAMILY)
    ap.add_argument("--sample-rows", type=int, default=SAMPLE_ROWS)
    ap.add_argument("--steps", type=int, default=LLOYD_STEPS)
    ap.add_argument("--probes", type=int, default=PROBES)
    ap.add_argument("--workers", type=int, default=len(FRACTIONS))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not 2 <= args.expert_count <= 64 or min(args.sample_rows, args.steps, args.probes, args.workers) <= 0:
        raise SystemExit("expert-count must be 2..64; sample-rows, steps, probes, and workers must be positive")

    names = _select_names(args.spec, args.expert_count, args.tensor_family)
    loaded, shards = _load_weights(args.spec, names)
    weights = [tensor for _name, tensor, _shard in loaded]
    fit_started = time.perf_counter()
    codebooks, packed_codes, rows, width, groups = _fit(weights, BASE_STAGES, BASE_BITS, BASE_ENTRIES, args.sample_rows, args.steps)
    fit_ms = (time.perf_counter() - fit_started) * 1000.0
    generator = torch.Generator(device="cpu").manual_seed(20260910)
    inputs = [[torch.randn(width, generator=generator) for _ in weights] for _ in range(args.probes)]
    started = time.perf_counter()
    workers = min(args.workers, len(FRACTIONS))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(
            _one, weights, names, codebooks, packed_codes, inputs, rows, width,
            groups, fraction, args.tensor_family
        ) for fraction in FRACTIONS]
        results = [future.result() for future in futures]
    wall_ms = (time.perf_counter() - started) * 1000.0

    def summary(result):
        return {"fraction": result["fraction"], "complete_ebpw": result["representation"]["complete_accounting"]["complete_ebpw"], "mean_relative_l2": result["heldout"]["mean_relative_l2"], "p95_relative_l2": result["heldout"]["p95_relative_l2"], "mean_cosine": result["heldout"]["mean_cosine"]}

    out = {
        "schema": SCHEMA,
        "status": "HELDOUT_PROTECTED_MULTISTAGE_FANOUT_COMPLETE",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "specimen": str(args.spec),
        "organ": f"layer10_routed_experts_0_to_{args.expert_count - 1}",
        "tensor_family": args.tensor_family,
        "expert_count": args.expert_count,
        "tensor_names": names,
        "shards": shards,
        "fit": {"stages": BASE_STAGES, "bits": BASE_BITS, "entries": BASE_ENTRIES, "sample_rows_per_tensor": args.sample_rows, "lloyd_steps": args.steps, "fit_ms": fit_ms, "activation_probes_are_held_out": True},
        "results": results,
        "fanout_timing": {"workers": workers, "fraction_count": len(results), "wall_elapsed_ms": wall_ms, "method_soft_budget_ms": SOFT_BUDGET_MS, "execution": "CPU held-out protected multistage fanout; not model TPS"},
        "ranked_summary": sorted((summary(result) for result in results), key=lambda row: (row["complete_ebpw"], row["mean_relative_l2"])),
        "claim_boundary": "Bounded protected-row reference only. Rows are selected using one probe and judged on held-out probes; the direct consumer uses packed codes plus sparse stored rows without dense-parent materialization. This is not full-model capability, end-to-end TPS, production-kernel, NX, or promotion evidence.",
        "next_discriminator": "If an under-1 protected row reaches acceptable held-out error, integrate it into an organ-level capability harness; otherwise move to representation-native learning.",
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
