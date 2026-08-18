#!/usr/bin/env python3
"""Parse the completed sweep log, run leftover CPU experiments, write JSON."""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp")
from g1_vq_sweep import (
    ACT,
    BF16,
    SEED,
    assign_batch,
    binary_g,
    cosine,
    hadamard_lattice,
    kmeans_shared,
    lattice_group,
    load_bf16_matrix,
    load_hidden,
    load_index,
    nearest_e8,
    pq_shared,
    rel_l2,
    reshape_sub,
    rss_gb,
    score,
    snr_db,
    mse,
    uniform_qn,
    viterbi_tcq_sample,
    weight_stats,
)

LOG = Path("/tmp/g1_vq_sweep.log")
OUT = Path("/tmp/g1_vq_results.json")

ROW = re.compile(
    r"^(?P<tag>\S+)\s+bpw=\s*(?P<bpw>[-0-9.]+)\s+w_cos=(?P<wcos>[-0-9.]+)\s+"
    r"w_rel=(?P<wrel>[-0-9.]+)\s+out_cos=\s*(?P<ocos>\S+)\s+(?P<clr>PASS|fail|-)\s+"
    r"reg=(?P<reg>[01])\s+tab=\s*(?P<tab>\d+)\s+rss=(?P<rss>[-0-9.]+)GB"
)
HEAD = re.compile(r"^===== (?P<name>\S+) grid=(?P<grid>\S+) =====")
LOAD = re.compile(r"^\s+loaded \((?P<r>\d+), (?P<c>\d+)\)")


def parse_log() -> dict:
    tensors = []
    cur = None
    for line in LOG.read_text().splitlines():
        m = HEAD.match(line)
        if m:
            if cur:
                tensors.append(cur)
            cur = {
                "tensor": m.group("name"),
                "grid": m.group("grid"),
                "variants": [],
            }
            continue
        m = LOAD.match(line)
        if m and cur is not None:
            cur["shape"] = [int(m.group("r")), int(m.group("c"))]
            continue
        m = ROW.match(line)
        if m and cur is not None:
            ocos = m.group("ocos")
            oc = None if ocos in {"n/a", "-"} else float(ocos)
            cur["variants"].append(
                {
                    "tag": m.group("tag"),
                    "bpw": float(m.group("bpw")),
                    "weight_cosine": float(m.group("wcos")),
                    "weight_rel_l2": float(m.group("wrel")),
                    "output_cosine": oc,
                    "clears_0p990_printed": m.group("clr"),
                    "register_only": bool(int(m.group("reg"))),
                    "table_bytes": int(m.group("tab")),
                    "rss_gb": float(m.group("rss")),
                }
            )
    if cur:
        tensors.append(cur)
    return tensors


def e8_radius1(W: np.ndarray, group: int = 64) -> tuple[np.ndarray, dict]:
    """E8 with coords clipped to [-1,1]. b=1 in the broken run used radius=0.

    Honest bits: 8 coords x log2(3) levels, minus 1 parity, plus 1 coset,
    plus 16-bit scale / 64.
    """
    d = 8
    flat = W.reshape(-1)
    V = flat.reshape(-1, d).copy()
    vecs_per_group = group // d
    nvec = V.shape[0]
    ng = math.ceil(nvec / vecs_per_group)
    pad = ng * vecs_per_group - nvec
    if pad:
        V = np.pad(V, ((0, pad), (0, 0)))
    G = V.reshape(ng, vecs_per_group, d)
    gflat = G.reshape(ng, -1)
    scales = (np.max(np.abs(gflat), axis=1) / 1.0).astype(np.float16).astype(np.float32)
    den = np.where(scales > 0, scales, 1.0)
    Y = (G / den[:, None, None]).reshape(-1, d)
    Q = np.clip(nearest_e8(Y), -1.0, 1.0)
    recon = (Q.reshape(ng, vecs_per_group, d) * scales[:, None, None]).reshape(-1, d)[:nvec]
    What = recon.reshape(W.shape)
    bits_per_vec = 8 * math.log2(3) - 1 + 1  # ~12.68
    index_bits = (W.size // d) * bits_per_vec
    scale_bits = math.ceil(W.size / group) * 16
    meta = {
        "family": "lattice_E8_radius1_g64",
        "tag": "lattice_E8_radius1_FIXED",
        "bpw": (index_bits + scale_bits) / W.size,
        "index_bits": index_bits,
        "scale_bits": scale_bits,
        "register_only": True,
        "table_bytes": 0,
        "note": "replacement for broken E8_b1 (radius=0). Coords in [-1,0,1], packed as log2(3)/coord.",
    }
    return What.astype(np.float32), meta


def main() -> int:
    t0 = time.perf_counter()
    tensors = parse_log()
    print(f"parsed {len(tensors)} tensors, variants={sum(len(t['variants']) for t in tensors)}", flush=True)

    idx = load_index()
    wm = idx["weight_map"]

    # leftover: cross-layer L0 -> L32 PQ d=8 K=256
    print("CROSS-LAYER", flush=True)
    W0 = load_bf16_matrix(
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight", wm
    )
    W32 = load_bf16_matrix(
        "language_model.model.layers.32.linear_attn.in_proj_qkv.weight", wm
    )
    X32 = load_hidden(32)
    Vtr = reshape_sub(W0, 8).reshape(-1, 8)
    C = kmeans_shared(Vtr, 256, seed=SEED + 9000)
    Vte = reshape_sub(W32, 8).reshape(-1, 8)
    labels = assign_batch(Vte, C)
    What = C[labels].reshape(W32.shape)
    bits_idx = 8
    S = W32.shape[1] // 8
    index_bits = W32.shape[0] * S * bits_idx
    codebook_bits = 256 * 8 * 16
    metrics = score(W32, What, X32)
    cross = {
        "family": "PQ_shared_crosslayer_d8_K256",
        "tag": "cross_L0toL32_pq_d8_K256",
        "bpw": (index_bits + codebook_bits) / W32.size,
        "bpw_if_codebook_shared_across_48": (index_bits + codebook_bits / 48.0) / W32.size,
        "index_bits": index_bits,
        "codebook_bits_fp16": codebook_bits,
        "register_only": False,
        "table_bytes": codebook_bits // 8,
        **metrics,
    }
    print(
        f"  cross out_cos={cross['output_cosine']:.5f} w_cos={cross['weight_cosine']:.5f} "
        f"bpw={cross['bpw']:.4f} shared48={cross['bpw_if_codebook_shared_across_48']:.4f}",
        flush=True,
    )
    del What, Vtr, Vte

    # same-layer control: L32 trained on itself (already in log as pq_shared_8_256)
    # Viterbi sample
    print("VITERBI", flush=True)
    sample = W0.reshape(-1)[: 256 * 256]
    vits = []
    for k, L in [(1, 5), (2, 6), (3, 7)]:
        t1 = time.perf_counter()
        r = viterbi_tcq_sample(sample, k, L, block=256)
        r["wall_s"] = time.perf_counter() - t1
        vits.append(r)
        print(
            f"  {r['family']} bpw={r['bpw']:.4f} w_cos={r['weight_cosine']:.5f} "
            f"w_rel={r['weight_rel_l2']:.4f} {r['wall_s']:.2f}s",
            flush=True,
        )

    # E8 radius-1 fix on L0 qkv + L3 q
    print("E8 radius-1 fix", flush=True)
    e8fix = []
    for name, layer, functional in [
        ("language_model.model.layers.0.linear_attn.in_proj_qkv.weight", 0, True),
        ("language_model.model.layers.3.self_attn.q_proj.weight", 3, True),
    ]:
        W = load_bf16_matrix(name, wm)
        X = load_hidden(layer) if functional else None
        What, meta = e8_radius1(W)
        metrics = score(W, What, X)
        rec = {**meta, **metrics, "tensor": name}
        e8fix.append(rec)
        print(
            f"  {name.split('.')[-2]} bpw={rec['bpw']:.4f} w_cos={rec['weight_cosine']:.5f} "
            f"out_cos={rec.get('output_cosine')}",
            flush=True,
        )
        del W, What

    # L63 Q3 min-row (probe accepted flattened 0.99086)
    print("L63 Q3 min-row recompute", flush=True)
    W = load_bf16_matrix(
        "language_model.model.layers.63.self_attn.q_proj.weight", wm
    )
    X = load_hidden(63)
    What, meta = uniform_qn(W, 3, 64)
    metrics = score(W, What, X)
    print(
        f"  L63 q Q3 flattened={metrics['output_cosine']:.5f} "
        f"mean_row={metrics['output_cosine_mean_row']:.5f} "
        f"min_row={metrics['output_cosine_min_row']:.5f}",
        flush=True,
    )
    l63q3 = {**meta, **metrics, "tensor": "L63.q_proj", "tag": "uniform_3_64_L63_detail"}

    results = {
        "schema": "hawking.superwave.g1.vector_quantization.v1",
        "date": "2026-08-17",
        "host_note": "CPU only. No GPU, no pack, no inference.",
        "source_weights": str(BF16),
        "source_activations": str(ACT),
        "quality_bar_output_cosine": 0.990,
        "bar_source": "receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_VERDICT.json",
        "sweep_log": str(LOG),
        "sweep_wall_s_from_tee": 671.39,
        "peak_rss_gb_from_log": 5.758,
        "tensors": tensors,
        "cross_layer": cross,
        "viterbi_sample": vits,
        "e8_radius1_fix": e8fix,
        "l63_q3_detail": l63q3,
        "finish_wall_s": time.perf_counter() - t0,
        "finish_rss_gb": rss_gb(),
    }
    OUT.write_text(json.dumps(results, indent=2))
    print(f"WROTE {OUT} {OUT.stat().st_size} bytes rss={rss_gb():.3f}GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
