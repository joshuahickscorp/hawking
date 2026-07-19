#!/usr/bin/env python3.12
"""Efficient FULL-SCOPE expert packing for the Second Light actual run.

The naive path (load_expert per expert) re-reads the whole ~1 GiB MXFP4 blocks tensor once PER
expert, so a full 128-expert row would read ~128 GiB. That makes a genuine full-scope run
impractical and tempts a bounded pilot (which Section 3 forbids calling the actual run). This module
reads each (block, kind) blocks+scales tensor ONCE, and packs a shared_expert_grammar over ALL 128
experts with a sampled codebook fit + per-expert streaming assignment, so peak memory stays ~1.5 GiB
and one full expert row completes in minutes.

Byte accounting is EXACT and identical in spirit to gravity_forge.pack_shared_grammar: the shared
codebook is billed once, every expert's indices are billed, metadata is billed. The codebook is FIT
on a deterministic sample (standard minibatch k-means) but ASSIGNMENT is exact over every vector of
every expert, so no weight is uncounted and the result is deterministic for a fixed seed.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any, Iterator

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import doctor_v5_gptoss_mxfp4 as mxfp4          # noqa: E402
import gravity_forge as gf                      # noqa: E402
from gptoss_moe_runtime import _bf16_bits_to_f32  # noqa: E402

_OUT_ROWS = {"mlp1_weight": 5760, "mlp2_weight": 2880}


def dequant_all_experts(reader: Any, block: int, kind: str) -> Iterator[tuple[int, np.ndarray]]:
    """Read the (block, kind) blocks+scales tensors ONCE, yield (expert, w[out,in] fp32) for all.

    kind is 'mlp1_weight' or 'mlp2_weight'. This is the single-read primitive: ~1 GiB read for
    mlp1 (vs ~128 GiB with per-expert re-reads)."""
    out_rows = _OUT_ROWS[kind]
    blocks = reader.u8(f"block.{block}.mlp.{kind}.blocks")     # [128, out, 90, 16]
    scales = reader.u8(f"block.{block}.mlp.{kind}.scales")     # [128, out, 90]
    n_experts = blocks.shape[0]
    for e in range(n_experts):
        bits = mxfp4.decode_mxfp4_groups_bf16(blocks[e], scales[e])
        w = _bf16_bits_to_f32(np.asarray(bits)).reshape(out_rows, -1)
        yield e, np.ascontiguousarray(w, dtype=np.float32)


def _fit_shared_codebooks(sample_vecs, *, k: int, stages: int, seed: int):
    """Additive multi-stage k-means on a pooled SAMPLE (deterministic). Returns list of centroids."""
    torch = gf._torch()
    dev = gf._device()
    res = torch.from_numpy(np.ascontiguousarray(sample_vecs, dtype=np.float32)).to(dev)
    cbs = []
    for m in range(stages):
        cb = gf._kmeans(res, k, iters=12, seed=seed + m)
        res = res - cb[gf._assign(res, cb)]
        cbs.append(cb)
    return cbs


def pack_layer_grammar_full(reader: Any, block: int, kind: str, *, dim: int, k: int, stages: int,
                            sample_vectors: int = 2_000_000, sample_experts: int = 16,
                            seed: int = 0) -> dict[str, Any]:
    """Full-scope shared_expert_grammar over ALL experts of one (block, kind), memory-bounded.

    Returns an exact byte accounting + per-expert weight-space rel error. The codebook is fit on a
    deterministic sample drawn from `sample_experts` strided experts; assignment is exact per expert.
    """
    torch = gf._torch()
    dev = gf._device()
    out_rows = _OUT_ROWS[kind]

    # ---- pass 1: fit shared codebooks on a strided, seeded sample (bounded memory) --------------
    blocks = reader.u8(f"block.{block}.mlp.{kind}.blocks")
    scales = reader.u8(f"block.{block}.mlp.{kind}.scales")
    n_experts = int(blocks.shape[0])
    stride = max(1, n_experts // max(1, sample_experts))
    sample_ids = list(range(0, n_experts, stride))[:sample_experts]
    pooled = []
    for e in sample_ids:
        bits = mxfp4.decode_mxfp4_groups_bf16(blocks[e], scales[e])
        w = _bf16_bits_to_f32(np.asarray(bits)).reshape(out_rows, -1)
        pooled.append(w.reshape(-1, dim))
    pool = np.concatenate(pooled, axis=0)
    rng = np.random.default_rng(seed)
    if pool.shape[0] > sample_vectors:
        sel = rng.choice(pool.shape[0], size=sample_vectors, replace=False)
        pool = pool[sel]
    cbs = _fit_shared_codebooks(pool, k=k, stages=stages, seed=seed)
    del pool, pooled

    # ---- pass 2: assign every expert exactly, accumulate bits + error (streaming) ---------------
    idx_bits_each = max(1, math.ceil(math.log2(max(2, k))))
    total_index_bits = 0
    total_weights = 0
    rel_errors = []
    for e in range(n_experts):
        bits = mxfp4.decode_mxfp4_groups_bf16(blocks[e], scales[e])
        w = _bf16_bits_to_f32(np.asarray(bits)).reshape(out_rows, -1)
        v = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).to(dev).reshape(-1, dim)
        recon = torch.zeros_like(v)
        residual = v.clone()
        n_vec = int(v.shape[0])
        for cb in cbs:
            idx = gf._assign(residual, cb)
            recon = recon + cb[idx]
            residual = residual - cb[idx]
            total_index_bits += n_vec * idx_bits_each
        rel = float(torch.linalg.norm(v - recon) / (torch.linalg.norm(v) + 1e-9))
        rel_errors.append(rel)
        total_weights += w.size
        del v, recon, residual

    codebook_bits = stages * (k * dim) * 16
    metadata_bits = 64 * 8
    total_bits = total_index_bits + codebook_bits + metadata_bits
    physical_bytes = math.ceil(total_bits / 8)
    whole_bpw = total_bits / max(1, total_weights)
    return {
        "family": "shared_expert_grammar_streaming",
        "block": block, "kind": kind, "n_experts": n_experts,
        "dim": dim, "k": k, "stages": stages,
        "n_weights": total_weights,
        "index_bits": total_index_bits, "codebook_bits": codebook_bits,
        "metadata_bits": metadata_bits, "physical_bits": total_bits,
        "physical_bytes": physical_bytes, "whole_artifact_bpw": whole_bpw,
        "mean_rel_error": float(np.mean(rel_errors)),
        "max_rel_error": float(np.max(rel_errors)),
        "min_rel_error": float(np.min(rel_errors)),
        "sample_experts": sample_ids, "sample_vectors_cap": sample_vectors,
        "full_scope": True, "bounded_experts": False,
    }


def selftest(block: int = 0) -> dict[str, Any]:
    """Prove the full-scope packer on real layer-0 mlp2 (smaller): all 128 experts, one read."""
    import time
    from gptoss_moe_runtime import ProvenanceReader
    reader = ProvenanceReader()
    t0 = time.time()
    r = pack_layer_grammar_full(reader, block, "mlp2_weight", dim=16, k=64, stages=2,
                                sample_vectors=1_000_000, sample_experts=16, seed=block)
    r["seconds"] = round(time.time() - t0, 1)
    return r


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2, sort_keys=True))
