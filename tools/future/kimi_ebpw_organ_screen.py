#!/usr/bin/env python3
"""Cheap KIMI organ-level complete-EBPW screen.

This is a Gravity.Reduce discriminator, not a compressor.  It reads only
safetensors headers, selects a few representative KIMI organs, and bills
source bytes, packed indices, codebooks, shape metadata, and a fixed scoped
decoder-support charge through the canonical complete-EBPW accountant.

The PQ rows are hypotheses with explicit assumptions.  They do not load
weights, fit codebooks, measure capability, or prove direct execution.  A
surviving row earns a later organ load, distortion/capability test, and direct
kernel test; it does not earn a KIMI body claim.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.future.complete_ebpw import (  # noqa: E402
    STREAM_BROADCAST_AUX,
    STREAM_WEIGHT_CODES,
    candidate_from_parts,
    cost,
)
from tools.future.kimi_ebpw_inventory import DEFAULT_SPEC, InventoryRefused  # noqa: E402

SCHEMA = "hawking.future.kimi_ebpw_organ_screen.v1"
SCOPED_METADATA_BYTES = 4096
SCOPED_DECODER_SUPPORT_BYTES = 65536
METHODS = (
    "source_bf16",
    "packed_int4",
    "binary_latent",
    "pq_d8_shared",
    "pq_d16_shared",
    "pq_d32_shared",
    "pq_d64_shared",
    "additive_2x8_shared",
    "generated_seeded_1mb",
)
METHOD_SOFT_BUDGET_MS = 2000.0


def _headers(spec: Path) -> list[tuple[str, list[int], int]]:
    index_path = spec / "model.safetensors.index.json"
    if not index_path.is_file():
        raise InventoryRefused(f"missing KIMI index: {index_path}")
    index = json.loads(index_path.read_text())
    names = set(str(x) for x in (index.get("weight_map") or {}))
    rows: list[tuple[str, list[int], int]] = []
    seen: set[str] = set()
    for shard in sorted(set(str(x) for x in index["weight_map"].values())):
        path = spec / shard
        with path.open("rb") as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                raise InventoryRefused(f"missing header length: {path}")
            n = struct.unpack("<Q", raw)[0]
            if n <= 0 or n > 128 * 1024 * 1024:
                raise InventoryRefused(f"unreasonable header length {n}: {path}")
            header = json.loads(fh.read(n))
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            shape = tensor.get("shape")
            offsets = tensor.get("data_offsets")
            if not isinstance(shape, list) or not all(isinstance(x, int) and x >= 0 for x in shape):
                raise InventoryRefused(f"invalid shape for {path}:{name}")
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise InventoryRefused(f"invalid offsets for {path}:{name}")
            payload = int(offsets[1]) - int(offsets[0])
            if payload < 0:
                raise InventoryRefused(f"negative payload for {path}:{name}")
            rows.append((str(name), [int(x) for x in shape], payload))
            seen.add(str(name))
    if seen != names:
        raise InventoryRefused(
            f"index/header mismatch: missing={sorted(names - seen)[:4]} "
            f"extra={sorted(seen - names)[:4]}"
        )
    return rows


def _organ_predicates() -> dict[str, Callable[[str], bool]]:
    return {
        "layer10_routed_expert23": lambda n: ".layers.10.mlp.experts.23." in n,
        "layer10_shared_experts": lambda n: ".layers.10.mlp.shared_experts." in n,
        "layer10_attention": lambda n: ".layers.10.self_attn." in n,
    }


def _tensor_vectors(shape: list[int], d: int) -> tuple[int, int]:
    """Return (vectors, padded vector groups) for a row-wise PQ hypothesis."""
    if not shape:
        raise InventoryRefused("scalar tensor cannot receive row-wise PQ")
    width = int(shape[-1])
    groups = math.ceil(width / d)
    rows = math.prod(shape[:-1]) if len(shape) > 1 else 1
    return rows * groups, groups


def _parts_for_family(
    rows: list[tuple[str, list[int], int]],
    family: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    assumptions: dict[str, Any] = {
        "scoped_metadata_bytes": SCOPED_METADATA_BYTES,
        "scoped_decoder_support_bytes": SCOPED_DECODER_SUPPORT_BYTES,
    }
    if family == "source_bf16":
        regions = [
            {"name": "source_bf16_payload", "bytes": sum(r[2] for r in rows),
             "stream_class": STREAM_WEIGHT_CODES}
        ]
    elif family == "packed_int4":
        regions = [
            {"name": "packed_int4_indices", "elements": sum(math.prod(r[1]) for r in rows),
             "bitwidth": 4, "stream_class": STREAM_WEIGHT_CODES}
        ]
        assumptions["quantizer"] = "uniform packed 4-bit control only; no capability claim"
    elif family.startswith("pq_d") and family.endswith("_shared"):
        d = int(family.removeprefix("pq_d").removesuffix("_shared"))
        index_bytes = 0
        codebook_groups: dict[int, int] = {}
        for name, shape, _payload in rows:
            vectors, groups = _tensor_vectors(shape, d)
            index_bytes += vectors  # one 8-bit code per vector
            codebook_groups[groups] = codebook_groups.get(groups, 0) + 1
        regions = [{
            "name": f"pq_d{d}_indices",
            "bytes": index_bytes,
            "stream_class": STREAM_WEIGHT_CODES,
        }]
        # One BF16 codebook per input-width/group class, shared by all tensors
        # in this organ.  This is the hypothesis being screened, not a result.
        for groups, n_tensors in sorted(codebook_groups.items()):
            codebook_bytes = groups * 256 * d * 2
            tables.append({
                "name": f"pq_d{d}_bf16_codebook_groups_{groups}",
                "bytes": codebook_bytes,
                "stream_class": STREAM_WEIGHT_CODES,
            })
        assumptions.update({
            "vector_width": d,
            "index_bits_per_vector": 8,
            "codebook_entries_per_group": 256,
            "codebook_dtype": "bf16",
            "codebook_sharing": "one table per distinct row input-width class",
            "codebook_group_classes": {str(k): v for k, v in sorted(codebook_groups.items())},
            "direct_kernel": "NOT_VALIDATED",
        })
    elif family == "binary_latent":
        regions = [{
            "name": "binary_latent_sign_codes",
            "elements": sum(math.prod(r[1]) for r in rows),
            "bitwidth": 1,
            "stream_class": STREAM_WEIGHT_CODES,
        }]
        tables = [{
            "name": "binary_per_tensor_scale_offset_f32",
            "bytes": 8 * len(rows),
            "stream_class": STREAM_BROADCAST_AUX,
        }]
        assumptions.update({
            "codes": "one sign bit per source element",
            "scale_offset_bytes_per_tensor": 8,
            "direct_kernel": "NOT_VALIDATED",
        })
    elif family == "additive_2x8_shared":
        d = 32
        index_bytes = 0
        codebook_groups: dict[int, int] = {}
        for _name, shape, _payload in rows:
            vectors, groups = _tensor_vectors(shape, d)
            index_bytes += vectors * 2  # two 8-bit additive codewords
            codebook_groups[groups] = codebook_groups.get(groups, 0) + 1
        regions = [{
            "name": "additive_2x8_indices",
            "bytes": index_bytes,
            "stream_class": STREAM_WEIGHT_CODES,
        }]
        for groups, _n_tensors in sorted(codebook_groups.items()):
            tables.append({
                "name": f"additive_2x8_bf16_codebooks_groups_{groups}",
                "bytes": groups * 2 * 256 * d * 2,
                "stream_class": STREAM_WEIGHT_CODES,
            })
        assumptions.update({
            "vector_width": d,
            "index_bits_per_vector": 16,
            "codebooks": 2,
            "codebook_entries": 256,
            "codebook_dtype": "bf16",
            "codebook_sharing": "one pair per distinct row input-width class",
            "direct_kernel": "NOT_VALIDATED",
        })
    elif family == "generated_seeded_1mb":
        regions = [{
            "name": "generator_latent_codes",
            "bytes": 256 * len(rows),
            "stream_class": STREAM_BROADCAST_AUX,
        }]
        assumptions.update({
            "generator_support_bytes": 1024 * 1024,
            "latent_bytes_per_tensor": 256,
            "generator": "hypothetical; no generator implementation or fit exists",
            "direct_kernel": "NOT_VALIDATED",
        })
        parts_generator = {
            "name": "hypothetical_shared_generator_support",
            "bytes": 1024 * 1024,
            "stream_class": STREAM_BROADCAST_AUX,
        }
    else:
        raise ValueError(f"unknown family: {family}")

    if family == "generated_seeded_1mb":
        generators = [parts_generator]
    else:
        generators = []

    parts = {
        "regions": regions,
        "generators": generators,
        "metadata": [{
            "name": "scoped_shape_and_dtype_metadata",
            "bytes": SCOPED_METADATA_BYTES,
            "stream_class": STREAM_BROADCAST_AUX,
        }],
        "tables": tables,
        "residuals": [],
        "runtime_auxiliaries": [{
            "name": "scoped_decoder_support_reserve",
            "bytes": SCOPED_DECODER_SUPPORT_BYTES,
            "stream_class": STREAM_BROADCAST_AUX,
        }],
        "representation": [],
        "model_specific_code": [],
    }
    return parts, assumptions


def _screen_family(
    organ: str,
    rows: list[tuple[str, list[int], int]],
    family: str,
) -> tuple[str, dict[str, Any], float]:
    started = time.perf_counter()
    parts, assumptions = _parts_for_family(rows, family)
    parent_params = sum(math.prod(shape) for _, shape, _ in rows)
    candidate = candidate_from_parts(
        family_id=f"KIMI_{organ}_{family}",
        parent_params=parent_params,
        parts=parts,
        reconstructs_dense_parent=False,
        consumes_representation_directly=True,
    )
    billed = cost(candidate)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return family, {
        "complete_accounting": billed,
        "assumptions": assumptions,
        "direct_execution_status": "NOT_VALIDATED",
        "capability_status": "NOT_MEASURED",
        "timing": {
            "accounting_elapsed_ms": round(elapsed_ms, 6),
            "soft_budget_ms": METHOD_SOFT_BUDGET_MS,
            "soft_budget_status": "WITHIN" if elapsed_ms <= METHOD_SOFT_BUDGET_MS else "OVER",
        },
    }, elapsed_ms


def _screen_one(
    organ: str,
    rows: list[tuple[str, list[int], int]],
    *,
    workers: int | None = None,
) -> dict[str, Any]:
    parent_params = sum(math.prod(shape) for _, shape, _ in rows)
    source_payload = sum(payload for _, _, payload in rows)
    n_workers = max(1, min(int(workers or (os.cpu_count() or 1)), len(METHODS)))
    started = time.perf_counter()
    families: dict[str, dict[str, Any]] = {}
    elapsed_sum = 0.0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        jobs = [pool.submit(_screen_family, organ, rows, family) for family in METHODS]
        for job in jobs:
            family, result, elapsed_ms = job.result()
            families[family] = result
            elapsed_sum += elapsed_ms
    elapsed_wall_ms = (time.perf_counter() - started) * 1000.0
    return {
        "organ": organ,
        "tensor_count": len(rows),
        "tensor_names": [name for name, _, _ in rows],
        "parent_params": parent_params,
        "source_bf16_payload_bytes": source_payload,
        "families": families,
        "fanout_timing": {
            "workers": n_workers,
            "method_count": len(METHODS),
            "sum_method_elapsed_ms": round(elapsed_sum, 6),
            "wall_elapsed_ms": round(elapsed_wall_ms, 6),
            "execution": "CPU header/accounting only; not model TPS",
        },
        "cheap_discriminator": {
            "best_accounted_family": min(
                families,
                key=lambda f: families[f]["complete_accounting"]["complete_ebpw"],
            ),
            "source_to_candidate_payload_only": "not a capability or physical result",
            "next_test": "load the selected organ, fit measured codebooks, then test output distortion/capability and a direct decoder",
        },
    }


def screen(spec: Path) -> dict[str, Any]:
    rows = _headers(spec)
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "ACCOUNTING_SCREEN_ONLY",
        "source_model": "KIMI_BASE",
        "specimen": str(spec),
        "immutable_base": True,
        "weight_load": False,
        "organ_selection": "layer 10 routed expert 23, shared experts, and attention",
        "families": list(METHODS),
        "organs": {},
        "claim_boundary": (
            "Header-only scoped accounting. PQ codebooks are hypothetical shared BF16 tables; "
            "no weights were loaded, no codebook was fitted, no direct kernel was validated, "
            "and no capability, full-model EBPW, TPS, promotion, or NX claim is made."
        ),
    }
    for organ, predicate in _organ_predicates().items():
        selected = [(n, shape, payload) for n, shape, payload in rows if predicate(n)]
        if not selected:
            raise InventoryRefused(f"selected organ has no tensors: {organ}")
        out["organs"][organ] = _screen_one(organ, selected)
    out["fanout_standard"] = {
        "worker_policy": "min(cpu_count, method_count)",
        "method_soft_budget_ms": METHOD_SOFT_BUDGET_MS,
        "stop_rule": "time budget flags a static method; it does not authorize skipping accounting or capability gates",
        "optimization_target": "reduce search wall and summed accounting time without changing candidate equations",
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = screen(args.spec)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "organs": {
            name: {
                "params": row["parent_params"],
                "best_accounted_family": row["cheap_discriminator"]["best_accounted_family"],
                "best_complete_ebpw": row["families"][row["cheap_discriminator"]["best_accounted_family"]]["complete_accounting"]["complete_ebpw"],
            }
            for name, row in result["organs"].items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
