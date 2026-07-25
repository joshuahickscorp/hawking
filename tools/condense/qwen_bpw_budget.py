#!/usr/bin/env python3.12
"""Byte-exact whole-model BPW plan for the Qwen3-235B Gravity frontier.

The 120B result established a *construction* floor near 0.77 whole-artifact BPW, but did not
preserve capability there. This plan transfers the useful result instead of repeating uniform
sub-bit RVQ: keep routers and norms native, give the heavy-tailed expert down projection protected
islands, spend fewer bits on the more robust expert gate/up pair, and separately budget every large
non-expert matrix. The plan is metadata-only and makes no quality claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import qwen3_moe_adapter as A

SCHEMA = "hawking.qwen3_235b.gravity_whole_bpw_plan.v1"
TARGET_WHOLE_BPW = 0.77
METADATA_BITS_PER_TENSOR = 64 * 8       # same fixed charge as gravity_forge.ByteLedger
CONTAINER_RESERVE_BYTES = 64 * 1024 * 1024

PQ_DENSE = {"family": "product_quant", "dim": 32, "subspaces": 8, "k": 16}
PQ_EXPERT_IN = {"family": "product_quant", "dim": 32, "subspaces": 4, "k": 8}
PQ_EXPERT_DOWN = {"family": "pq_protected_islands", "dim": 16, "subspaces": 4,
                  "k": 16, "budget_frac": 0.03, "strategy": "residual_energy"}
KEEP_NATIVE = {"family": "kept_original", "bpw": 16.0}

_DENSE_ORGANS = {
    A.ORGAN_EMBED, A.ORGAN_LM_HEAD, A.ORGAN_Q, A.ORGAN_K, A.ORGAN_V, A.ORGAN_O,
}


def _ceil_byte(bits: int) -> int:
    return math.ceil(bits / 8) * 8


def _pq_bits(shape: tuple[int, ...], *, dim: int, subspaces: int, k: int,
             budget_frac: float = 0.0) -> int:
    if len(shape) != 2:
        raise ValueError(f"PQ requires a matrix, got {shape}")
    rows, cols = shape
    if cols % dim or dim % subspaces:
        raise ValueError(f"invalid PQ geometry D={dim} S={subspaces} for shape {shape}")
    n_weights = rows * cols
    n_vectors = n_weights // dim
    index_bits = n_vectors * subspaces * math.ceil(math.log2(k))
    # S codebooks x K rows x (D/S) fp16 scalars = K*D fp16 scalars.
    codebook_bits = k * dim * 16
    bits = index_bits + codebook_bits + METADATA_BITS_PER_TENSOR
    if budget_frac:
        n_islands = max(1, min(rows, int(round(rows * budget_frac))))
        row_index_bits = math.ceil(math.log2(max(2, rows)))
        bits += n_islands * (cols * 16 + row_index_bits)
    return _ceil_byte(bits)


def _native_bits(shape: tuple[int, ...]) -> int:
    return _ceil_byte(math.prod(shape) * 16 + METADATA_BITS_PER_TENSOR)


def _spec_for(organ: str) -> dict[str, Any]:
    if organ in (A.ORGAN_EXP_GATE, A.ORGAN_EXP_UP):
        return PQ_EXPERT_IN
    if organ == A.ORGAN_EXP_DOWN:
        return PQ_EXPERT_DOWN
    if organ in _DENSE_ORGANS:
        return PQ_DENSE
    return KEEP_NATIVE


def _bits_for(shape: tuple[int, ...], spec: dict[str, Any]) -> int:
    if spec["family"] == "kept_original":
        return _native_bits(shape)
    return _pq_bits(shape, dim=int(spec["dim"]), subspaces=int(spec["subspaces"]),
                    k=int(spec["k"]), budget_frac=float(spec.get("budget_frac", 0.0)))


def build_plan(config: dict[str, Any], index: dict[str, Any], *,
               target_bpw: float = TARGET_WHOLE_BPW,
               container_reserve_bytes: int = CONTAINER_RESERVE_BYTES) -> dict[str, Any]:
    inv = A.build_inventory(config, index)
    ver = A.verify_index_names(inv.geometry, index)
    if not inv.cross_check_ok() or not ver.ok:
        raise ValueError("Qwen config/index identity failed before BPW planning")

    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"tensor_count": 0, "n_weights": 0, "payload_bits": 0}
    )
    payload_bits = 0
    for tensor in inv.tensors:
        spec = _spec_for(tensor.organ_class)
        bits = _bits_for(tensor.shape, spec)
        payload_bits += bits
        row = groups[tensor.organ_class]
        row["tensor_count"] += 1
        row["n_weights"] += tensor.param_count
        row["payload_bits"] += bits
        row["spec"] = dict(spec)

    reserve_bits = int(container_reserve_bytes) * 8
    total_bits = payload_bits + reserve_bits
    target_bits = math.floor(inv.grand_params * float(target_bpw))
    group_rows = {}
    for organ, row in sorted(groups.items()):
        group_rows[organ] = {
            **row,
            "realized_bpw": round(row["payload_bits"] / row["n_weights"], 9),
            "physical_bytes": math.ceil(row["payload_bits"] / 8),
        }

    plan = {
        "schema": SCHEMA,
        "parent": {
            "repo": "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "revision": A.IMMUTABLE_REVISION,
            "parameters": inv.grand_params,
            "source_payload_bytes": inv.grand_bytes,
            "source_shards": len(inv.shard_files),
            "config_index_identity_ok": True,
        },
        "target": {
            "whole_artifact_bpw_ceiling": float(target_bpw),
            "provenance": "GPT-OSS-120B byte-complete construction floor (~0.77 BPW)",
            "important_boundary": "the 120B rate passed but capability failed; this is a Qwen byte "
                                  "envelope to test, not a capability prediction",
        },
        "allocation": group_rows,
        "accounting": {
            "packed_tensor_payload_bits": payload_bits,
            "container_metadata_reserve_bytes": int(container_reserve_bytes),
            "total_artifact_bits": total_bits,
            "total_artifact_bytes": math.ceil(total_bits / 8),
            "projected_whole_artifact_bpw": round(total_bits / inv.grand_params, 9),
            "target_total_bits": target_bits,
            "margin_bits": target_bits - total_bits,
            "margin_bytes": math.floor((target_bits - total_bits) / 8),
            "target_met": total_bits <= target_bits,
        },
        "quality_strategy": {
            "routers_and_norms": "kept BF16 to stabilize routing and residual scale",
            "expert_gate_up": "Q2-driven input reallocation: four subspaces after gate/up dominated error",
            "expert_down": "reduced-cardinality PQ plus 3% protected rows; Q2 down-only error was lower",
            "attention_embed_head": "separately billed 1-BPW PQ lane; never hidden as native overhead",
            "forbidden_shortcut": "expert-only BPW must not be reported as whole-model BPW",
        },
        "claim": "BYTE PLAN ONLY; capability requires real parent-vs-packed forward gates",
    }
    plan["sha256"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return plan


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", type=Path, default=A.DEFAULT_META)
    ap.add_argument("--target-bpw", type=float, default=TARGET_WHOLE_BPW)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    plan = build_plan(A.load_config(args.meta_dir), A.load_index(args.meta_dir),
                      target_bpw=args.target_bpw)
    rendered = json.dumps(plan, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0 if plan["accounting"]["target_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
