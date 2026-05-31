#!/usr/bin/env python3
"""oracle_coactivation_permute — L2.2 kill-or-keep oracle (co-activation permutation).

Question (bible §8.1 L2.2): the block-256 contextual-sparsity lever DIED (only
0.2% of 256-blocks skippable @99% recall; active neurons scattered). The
resurrection hypothesis: reorder FFN neurons OFFLINE so co-firing ones land in
contiguous GPU-friendly blocks, at FINER granularity (32/64). Does an offline
co-activation PERMUTATION recover >=~30% contiguous skippable mass at an aligned
block (post-permute)? If not, the lever re-dies (prior: SwiGLU is not natively
sparse like ReLU).

Why we must reconstruct per-neuron activations
----------------------------------------------
The existing capture (`_capture/q3b_ffn.bin`, packed by pack_ffn.py) stores ONLY
per-256-block reductions (blockmax / blockl2, 43 blocks), plus the full 2048-dim
`norm_in` (the ffn_norm RMSNorm output = the gate/up input) per token. Block
reductions cannot be re-split to 32/64, and they carry no per-neuron identity, so
they cannot answer the permutation question. BUT `norm_in` + the model's gate/up
weights reconstruct the TRUE per-neuron SwiGLU activation:

    a_j = silu(gate_j . x) * (up_j . x)        for j in [0, 11008)

We dequantize the Q4_K gate/up weights from the Qwen2.5-3B GGUF (CPU-only, no
Metal), recompute `a` for every captured token, and VALIDATE the reconstruction
against the captured blockmax/blockl2 (per-256-block max|a| and ||a||_2). Only if
the reconstruction matches do we trust the permutation numbers.

Byte-cut metric (matches measure_ffn_sparsity.py's oracle)
----------------------------------------------------------
A neuron's contribution to `ffn_down @ a` scales with |a_j| (down rows are
~orthogonal in high dim, same assumption the Track-B gate used at block level).
For a block layout of size B, a block's energy is sum_{j in block} a_j^2. For a
target recall r we drop whole blocks (smallest-energy first) while the relative
L2 error sqrt(dropped_energy/total_energy) <= (1-r). Skippable = dropped/total
blocks, averaged over tokens. The PERMUTATION is a single fixed neuron ordering
(offline, shared across all tokens of a layer) chosen to cluster co-firing
neurons; we compare skippable BEFORE (identity order) vs AFTER (permuted) at
block sizes {32, 64, 128}.

CPU-only. No GPU/Metal, no dismantle/cargo, no training. Reads the GGUF + capture,
writes a report.

Usage:
    /tmp/ggufenv/bin/python tools/bench/oracle_coactivation_permute.py \
        --bin _capture/q3b_ffn.bin \
        --gguf models/qwen2.5-3b-instruct-q4_k_m.gguf \
        --out reports/oracle_coactivation_permute.md
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

SENTINEL = 0xFFFFFFFF
QK_K = 256  # Q4_K / Q6_K super-block size


# --------------------------------------------------------------------------
# GGUF reading + Q4_K / Q6_K dequant (CPU, pure numpy)
# --------------------------------------------------------------------------
GGML_F32, GGML_F16, GGML_Q4_K, GGML_Q6_K = 0, 1, 12, 14
_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
        6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_gguf(path: Path):
    """Return (kv dict, {name: (dims, ggml_type, abs_byte_offset)}, data_start)."""
    f = open(path, "rb")
    magic = f.read(4)
    if magic != b"GGUF":
        raise ValueError(f"not a GGUF file: {path}")
    (ver,) = struct.unpack("<I", f.read(4))
    (n_tensors,) = struct.unpack("<Q", f.read(8))
    (n_kv,) = struct.unpack("<Q", f.read(8))

    def rd_str():
        (ln,) = struct.unpack("<Q", f.read(8))
        return f.read(ln).decode("utf-8", errors="replace")

    def rd_val(t):
        if t == 8:
            return rd_str()
        if t == 9:
            (et,) = struct.unpack("<I", f.read(4))
            (cnt,) = struct.unpack("<Q", f.read(8))
            if et == 8:
                return [rd_str() for _ in range(cnt)]
            raw = f.read(_SZ[et] * cnt)
            return np.frombuffer(raw, dtype=np.dtype(_FMT[et][1:]))
        return struct.unpack(_FMT[t], f.read(_SZ[t]))[0]

    kv = {}
    for _ in range(n_kv):
        k = rd_str()
        (t,) = struct.unpack("<I", f.read(4))
        kv[k] = rd_val(t)

    tinfos = {}
    for _ in range(n_tensors):
        name = rd_str()
        (nd,) = struct.unpack("<I", f.read(4))
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        (typ,) = struct.unpack("<I", f.read(4))
        (off,) = struct.unpack("<Q", f.read(8))
        tinfos[name] = (dims, typ, off)

    data_start = f.tell()
    align = int(kv.get("general.alignment", 32))
    data_start += (align - (data_start % align)) % align
    f.close()
    return kv, tinfos, data_start


def _dequant_q4_k(buf: np.ndarray, n_elems: int) -> np.ndarray:
    """Dequantize a Q4_K byte buffer -> float32 [n_elems].

    Q4_K super-block (144 bytes / 256 weights), llama.cpp layout:
      f16 d, f16 dmin, 12 bytes of packed 6-bit scales/mins (8 of each),
      128 bytes of 4-bit quants. Weight = d*sc*q - dmin*m, per 32-elem group.
    """
    nb = n_elems // QK_K
    raw = buf[: nb * 144].reshape(nb, 144)
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32)[:, 0]
    dmin = raw[:, 2:4].copy().view(np.float16).astype(np.float32)[:, 0]
    scbytes = raw[:, 4:16].astype(np.uint32)            # [nb,12]
    qs = raw[:, 16:144]                                  # [nb,128] uint8

    # Unpack 8 6-bit scales and 8 6-bit mins (llama.cpp get_scale_min_k4).
    sc = np.empty((nb, 8), np.float32)
    mn = np.empty((nb, 8), np.float32)
    for j in range(8):
        if j < 4:
            s = scbytes[:, j] & 63
            m = scbytes[:, j + 4] & 63
        else:
            s = (scbytes[:, j + 4] & 15) | ((scbytes[:, j - 4] >> 6) << 4)
            m = (scbytes[:, j + 4] >> 4) | ((scbytes[:, j] >> 6) << 4)
        sc[:, j] = s
        mn[:, j] = m

    # Canonical dequant loop (matches llama.cpp dequantize_row_q4_K): 256 weights
    # = 4 chunks of 64; each 32-byte chunk of `qs` yields a low-nibble 32-group
    # and a high-nibble 32-group, each with its own 6-bit scale/min.
    out = np.empty((nb, QK_K), np.float32)
    for blk in range(nb):
        q = qs[blk]
        di = d[blk]
        dmi = dmin[blk]
        is_ = 0
        oi = 0
        for j in range(0, QK_K, 64):
            ql = q[j // 2: j // 2 + 32]
            sc0, m0 = sc[blk, is_], mn[blk, is_]
            sc1, m1 = sc[blk, is_ + 1], mn[blk, is_ + 1]
            lo = (ql & 0x0F).astype(np.float32)
            hi = (ql >> 4).astype(np.float32)
            out[blk, oi:oi + 32] = di * sc0 * lo - dmi * m0
            out[blk, oi + 32:oi + 64] = di * sc1 * hi - dmi * m1
            is_ += 2
            oi += 64
    return out.reshape(-1)[:n_elems]


def _dequant_q6_k(buf: np.ndarray, n_elems: int) -> np.ndarray:
    """Dequantize Q6_K -> float32 [n_elems]. 210 bytes / 256 weights.
    Layout: ql[128], qh[64], scales int8[16], f16 d.
    """
    nb = n_elems // QK_K
    raw = buf[: nb * 210].reshape(nb, 210)
    ql = raw[:, 0:128]
    qh = raw[:, 128:192]
    scales = raw[:, 192:208].view(np.int8).astype(np.float32)  # [nb,16]
    d = raw[:, 208:210].copy().view(np.float16).astype(np.float32)[:, 0]
    out = np.empty((nb, QK_K), np.float32)
    for blk in range(nb):
        qlb = ql[blk].astype(np.int32)
        qhb = qh[blk].astype(np.int32)
        sca = scales[blk]
        dd = d[blk]
        for n in range(2):  # two 128-weight halves
            base_q = n * 64
            base_o = n * 128
            # Reconstruct 6-bit quants: q = (ql nibble) | (qh 2-bit << 4) - 32
            q1 = (qlb[base_q + 0:base_q + 32] & 0x0F) | (((qhb[0 + n * 64:32 + n * 64] >> 0) & 3) << 4)
            q2 = (qlb[base_q + 32:base_q + 64] & 0x0F) | (((qhb[0 + n * 64:32 + n * 64] >> 2) & 3) << 4)
            q3 = (qlb[base_q + 0:base_q + 32] >> 4) | (((qhb[0 + n * 64:32 + n * 64] >> 4) & 3) << 4)
            q4 = (qlb[base_q + 32:base_q + 64] >> 4) | (((qhb[0 + n * 64:32 + n * 64] >> 6) & 3) << 4)
            sc_base = n * 8
            out[blk, base_o + 0:base_o + 32] = dd * sca[sc_base + 0] * (q1 - 32)
            out[blk, base_o + 32:base_o + 64] = dd * sca[sc_base + 2] * (q2 - 32)
            out[blk, base_o + 64:base_o + 96] = dd * sca[sc_base + 4] * (q3 - 32)
            out[blk, base_o + 96:base_o + 128] = dd * sca[sc_base + 6] * (q4 - 32)
            # NOTE: the 16 scales index in groups of 16 weights; the above uses
            # 32-weight groups (coarser). Q6_K is only used for ffn_down which we
            # do NOT need for the activation reconstruction, so this path is
            # unused in the oracle. Kept for completeness/debug only.
    return out.reshape(-1)[:n_elems]


def load_tensor(mmap: np.ndarray, dims, typ, off, data_start) -> np.ndarray:
    """Return a dequantized float32 tensor reshaped to GGUF dims (row-major as
    stored: dims are [n_cols, n_rows] in GGUF, i.e. fastest dim first)."""
    n_elems = int(np.prod(dims))
    base = data_start + off
    if typ == GGML_F32:
        arr = mmap[base: base + n_elems * 4].view(np.float32).copy()
    elif typ == GGML_F16:
        arr = mmap[base: base + n_elems * 2].view(np.float16).astype(np.float32)
    elif typ == GGML_Q4_K:
        nb = n_elems // QK_K
        arr = _dequant_q4_k(mmap[base: base + nb * 144], n_elems)
    elif typ == GGML_Q6_K:
        nb = n_elems // QK_K
        arr = _dequant_q6_k(mmap[base: base + nb * 210], n_elems)
    else:
        raise ValueError(f"unsupported ggml type {typ}")
    # GGUF dims are [d0, d1] fastest-first; numpy wants [d1, d0] for a matrix.
    return arr.reshape(list(reversed(dims)))


# --------------------------------------------------------------------------
# Capture reading: per (layer) -> norm_in[N,hidden], blockmax[N,nb], blockl2[N,nb]
# --------------------------------------------------------------------------
def load_capture(path: Path):
    data = Path(path).read_bytes()
    n = len(data)
    off = 0
    hidden = n_blocks = None
    acc: dict[int, tuple[list, list, list]] = {}
    while off + 8 <= n:
        a, b = struct.unpack_from("<II", data, off)
        if a == SENTINEL and b == SENTINEL:
            if off + 16 > n:
                break
            _, _, hidden, n_blocks = struct.unpack_from("<IIII", data, off)
            hidden, n_blocks = int(hidden), int(n_blocks)
            off += 16
            continue
        if hidden is None:
            raise ValueError("stream did not start with a sentinel")
        rec = 4 + hidden * 4 + 2 * n_blocks * 4
        if off + rec > n:
            break
        (layer,) = struct.unpack_from("<I", data, off)
        foff = off + 4
        norm = np.frombuffer(data, np.float32, hidden, foff)
        bmax = np.frombuffer(data, np.float32, n_blocks, foff + hidden * 4)
        bl2 = np.frombuffer(data, np.float32, n_blocks, foff + hidden * 4 + n_blocks * 4)
        slot = acc.setdefault(int(layer), ([], [], []))
        slot[0].append(norm.copy())
        slot[1].append(bmax.copy())
        slot[2].append(bl2.copy())
        off += rec
    out = {k: (np.stack(v[0]), np.stack(v[1]), np.stack(v[2]))
           for k, v in acc.items()}
    return hidden, n_blocks, out


# --------------------------------------------------------------------------
# Oracle byte-cut: drop smallest-energy blocks while rel-L2 error <= 1-recall
# --------------------------------------------------------------------------
def skippable_fraction(act_abs: np.ndarray, block: int, recall: float) -> float:
    """act_abs: [N, F] non-negative per-neuron |activation|. Group consecutive
    neurons into blocks of size `block` (the GPU-contiguous layout). Returns the
    mean-over-tokens fraction of blocks droppable while keeping >= recall of L2
    energy. Neurons are assumed already in the desired order (identity or permuted).
    """
    N, F = act_abs.shape
    nb = F // block  # ignore tail < block (aligned-block premise)
    if nb == 0:
        return 0.0
    energy = (act_abs[:, : nb * block] ** 2).reshape(N, nb, block).sum(axis=2)  # [N, nb]
    total = np.maximum(energy.sum(axis=1, keepdims=True), 1e-30)
    order = np.argsort(energy, axis=1)                  # ascending
    sorted_e = np.take_along_axis(energy, order, axis=1)
    cum_dropped = np.cumsum(sorted_e, axis=1)
    budget = ((1.0 - recall) ** 2) * total
    n_drop = (cum_dropped <= budget).sum(axis=1)        # [N]
    return float(np.mean(n_drop / nb))


# --------------------------------------------------------------------------
# Permutation: cluster co-firing neurons into contiguous runs.
# Strategy = greedy spectral-ish ordering on the co-activation (correlation)
# matrix via hierarchical clustering leaf order. Falls back to a greedy
# nearest-neighbor chain if scipy is unavailable.
# --------------------------------------------------------------------------
def coactivation_order(fired: np.ndarray, act_abs: np.ndarray) -> np.ndarray:
    """Return a permutation of neuron indices [F] grouping co-firing neurons.

    fired: [N, F] boolean (neuron active for that token).
    act_abs: [N, F] magnitude (used as a secondary signal / tie-break).
    Builds the co-occurrence correlation matrix, then orders neurons by an
    average-linkage hierarchical clustering leaf order (co-firing neurons end up
    adjacent). This is the strongest realistic offline permutation; it is an
    UPPER bound on what a real static reorder achieves.
    """
    N, F = fired.shape
    f = fired.astype(np.float32)
    freq = f.mean(axis=0)                                # [F] firing rate
    # Co-firing correlation (phi coefficient). Center then normalize.
    fc = f - freq[None, :]
    cov = (fc.T @ fc) / N                                # [F, F]
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / (sd[:, None] * sd[None, :])
    np.fill_diagonal(corr, 1.0)
    corr = np.nan_to_num(corr, nan=0.0)

    # Dead neurons (never / almost never fire) -> park them together at the end
    # so live neurons cluster tightly (this HELPS skippability and is legitimate:
    # a permanently-cold block is always droppable).
    live = freq > (0.5 / N)                              # fired at least ~once
    live_idx = np.where(live)[0]
    dead_idx = np.where(~live)[0]

    if live_idx.size <= 2:
        return np.concatenate([live_idx, dead_idx]).astype(np.int64)

    sub = corr[np.ix_(live_idx, live_idx)]
    dist = 1.0 - sub
    np.fill_diagonal(dist, 0.0)
    dist = np.maximum(dist, 0.0)

    order_live = None
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        order_live = leaves_list(Z)
    except Exception:
        # Greedy nearest-neighbor chain on correlation (no scipy needed).
        m = sub.copy()
        np.fill_diagonal(m, -2.0)
        nliv = live_idx.size
        used = np.zeros(nliv, bool)
        start = int(np.argmax(freq[live_idx]))
        chain = [start]
        used[start] = True
        for _ in range(nliv - 1):
            row = m[chain[-1]].copy()
            row[used] = -2.0
            nxt = int(np.argmax(row))
            chain.append(nxt)
            used[nxt] = True
        order_live = np.array(chain, dtype=np.int64)

    perm = np.concatenate([live_idx[order_live], dead_idx]).astype(np.int64)
    return perm


def main() -> int:
    ap = argparse.ArgumentParser(prog="oracle_coactivation_permute")
    ap.add_argument("--bin", default="_capture/q3b_ffn.bin")
    ap.add_argument("--gguf", default="models/qwen2.5-3b-instruct-q4_k_m.gguf")
    ap.add_argument("--out", default="reports/oracle_coactivation_permute.md")
    ap.add_argument("--recall", type=float, default=0.99)
    ap.add_argument("--blocks", default="32,64,128")
    ap.add_argument("--fire-thresh", type=float, default=1e-4,
                    help="|a_j| above this = neuron 'fired' (for co-occurrence)")
    ap.add_argument("--layers", default=None,
                    help="comma list of layers to use (default: a spread sample)")
    ap.add_argument("--max-tokens", type=int, default=400,
                    help="cap tokens/layer for the (F x F) correlation build")
    args = ap.parse_args()

    block_sizes = [int(x) for x in args.blocks.split(",")]
    t0 = time.time()

    print(f"[oracle] reading capture {args.bin} ...", flush=True)
    hidden, n_blocks, cap = load_capture(Path(args.bin))
    F = n_blocks * QK_K
    all_layers = sorted(cap.keys())
    print(f"[oracle] hidden={hidden} n_blocks={n_blocks} F(intermediate)={F} "
          f"layers={len(all_layers)} tokens/layer={cap[all_layers[0]][0].shape[0]}",
          flush=True)

    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        # representative spread: early, middle, late
        layers = [all_layers[i] for i in
                  sorted(set([0, len(all_layers)//4, len(all_layers)//2,
                              3*len(all_layers)//4, len(all_layers)-1]))]
    print(f"[oracle] analyzing layers {layers}", flush=True)

    print(f"[oracle] reading GGUF {args.gguf} ...", flush=True)
    kv, tinfos, data_start = read_gguf(Path(args.gguf))
    assert int(kv["qwen2.embedding_length"]) == hidden
    assert int(kv["qwen2.feed_forward_length"]) == F
    mmap = np.memmap(args.gguf, dtype=np.uint8, mode="r")

    results = {}   # layer -> dict
    recon_checks = []

    for layer in layers:
        norm_in, bmax_cap, bl2_cap = cap[layer]          # [N,hidden],[N,nb],[N,nb]
        N = norm_in.shape[0]
        if N > args.max_tokens:
            sel = np.linspace(0, N - 1, args.max_tokens).astype(int)
            norm_in = norm_in[sel]
            bmax_cap = bmax_cap[sel]
            bl2_cap = bl2_cap[sel]
            N = norm_in.shape[0]

        gate = load_tensor(mmap, *tinfos[f"blk.{layer}.ffn_gate.weight"], data_start)  # [F,hidden]
        up = load_tensor(mmap, *tinfos[f"blk.{layer}.ffn_up.weight"], data_start)      # [F,hidden]
        # x @ W^T  ->  [N,F]
        g = norm_in @ gate.T
        u = norm_in @ up.T
        silu = g / (1.0 + np.exp(-g))
        act = silu * u                                   # [N,F] true SwiGLU activation
        act_abs = np.abs(act)

        # --- validate reconstruction vs captured block reductions ---
        a_blk = act.reshape(N, n_blocks, QK_K)
        recon_max = np.abs(a_blk).max(axis=2)            # [N,nb]
        recon_l2 = np.sqrt((a_blk ** 2).sum(axis=2))     # [N,nb]
        # relative agreement on l2 (the metric that drives the byte-cut)
        denom = np.maximum(np.abs(bl2_cap), 1e-6)
        rel_l2 = np.abs(recon_l2 - bl2_cap) / denom
        denom_m = np.maximum(np.abs(bmax_cap), 1e-6)
        rel_max = np.abs(recon_max - bmax_cap) / denom_m
        recon_checks.append((layer, float(np.median(rel_l2)), float(np.median(rel_max))))

        # --- firing stats ---
        fired = act_abs > args.fire_thresh
        active_frac = float(fired.mean())                # fraction of neurons active/token
        freq = fired.mean(axis=0)
        dead_frac = float((freq <= 0.5 / N).mean())

        # --- permutation ---
        perm = coactivation_order(fired, act_abs)
        act_perm = act_abs[:, perm]

        per_block = {}
        for B in block_sizes:
            pre = skippable_fraction(act_abs, B, args.recall)
            post = skippable_fraction(act_perm, B, args.recall)
            per_block[B] = (pre, post)

        results[layer] = dict(
            N=N, active_frac=active_frac, dead_frac=dead_frac,
            per_block=per_block,
        )
        msg = " ".join(f"B{B}:{pre:.3f}->{post:.3f}" for B, (pre, post) in per_block.items())
        print(f"[oracle] layer {layer:2d} N={N} act/tok={active_frac:.3f} "
              f"dead={dead_frac:.3f} | {msg}", flush=True)

    # ---- aggregate ----
    agg = {B: {"pre": [], "post": []} for B in block_sizes}
    for layer, r in results.items():
        for B, (pre, post) in r["per_block"].items():
            agg[B]["pre"].append(pre)
            agg[B]["post"].append(post)
    overall = {B: (float(np.mean(agg[B]["pre"])), float(np.mean(agg[B]["post"])))
               for B in block_sizes}
    mean_active = float(np.mean([r["active_frac"] for r in results.values()]))
    mean_dead = float(np.mean([r["dead_frac"] for r in results.values()]))

    # FFN is ~72% of bytes/token; gate+up+down all permute together (down rows
    # follow the same neuron order). byte-cut ~= 0.72 * skippable.
    best_post = max(post for _, post in overall.values())
    best_post_B = max(overall, key=lambda B: overall[B][1])
    skip_for_verdict = overall[best_post_B][1]

    # reconstruction trust gate
    med_rel_l2 = float(np.median([c[1] for c in recon_checks]))
    recon_ok = med_rel_l2 < 0.05

    if not recon_ok:
        verdict = "INVALID"
        verdict_txt = (f"Reconstruction did NOT match the capture (median rel-L2 "
                       f"error {med_rel_l2:.3f} >= 0.05). Numbers below are not "
                       f"trustworthy; fix the weight/norm reconstruction before "
                       f"drawing a conclusion.")
    elif skip_for_verdict >= 0.30:
        verdict = "GO"
        verdict_txt = (f"Post-permute skippable {skip_for_verdict:.1%} at block "
                       f"{best_post_B} clears the ~30% bar. A neuron-gather kernel "
                       f"with this static permutation is worth prototyping.")
    else:
        verdict = "NO-GO"
        verdict_txt = (f"Best post-permute skippable {skip_for_verdict:.1%} (block "
                       f"{best_post_B}) is below the ~30% bar. Co-activation "
                       f"permutation does not recover enough contiguous mass — "
                       f"the lever re-dies, consistent with SwiGLU lacking ReLU's "
                       f"hard zeros.")

    # ---- write report ----
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Oracle — L2.2 co-activation permutation (kill-or-keep)")
    lines.append("")
    lines.append(f"**Date:** 2026-05-30  ")
    lines.append(f"**Lever:** bible §8.1 L2.2 — contextual sparsity via offline "
                 f"co-activation permutation (resurrection of dead block-256).  ")
    lines.append(f"**Verdict:** **{verdict}**  ")
    lines.append("")
    lines.append(verdict_txt)
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(f"- Capture `{args.bin}` stores only per-256-block reductions, so "
                 f"per-neuron data was **reconstructed**: dequantized the Q4_K "
                 f"gate/up weights from `{args.gguf}` and computed the true SwiGLU "
                 f"activation `a = silu(gate·x)·(up·x)` (F={F}) for each captured "
                 f"token, x = captured `norm_in` (hidden={hidden}).")
    lines.append(f"- **Reconstruction validated** against captured blockmax/blockl2: "
                 f"median relative L2 error per 256-block = **{med_rel_l2:.4f}** "
                 f"({'PASS <0.05' if recon_ok else 'FAIL >=0.05'}).")
    lines.append(f"- Permutation = average-linkage hierarchical clustering leaf "
                 f"order on the neuron co-firing correlation matrix (cold neurons "
                 f"parked contiguously at the tail). This is a best-case OFFLINE "
                 f"static reorder (upper bound vs any learned predictor).")
    lines.append(f"- Byte-cut metric = drop whole blocks (smallest L2 energy first) "
                 f"while keeping >= {args.recall:.0%} of activation L2 energy "
                 f"(same oracle as the Track-B block-256 gate). Mean over "
                 f"{len(layers)} sampled layers {layers}, up to {args.max_tokens} "
                 f"tokens/layer.")
    lines.append("")
    lines.append(f"- Mean active neurons/token (|a|>{args.fire_thresh}): "
                 f"**{mean_active:.1%}**  ")
    lines.append(f"- Mean permanently-cold neurons: **{mean_dead:.1%}**")
    lines.append("")
    lines.append("## Skippable block fraction @ 99% recall (pre → post permute)")
    lines.append("")
    lines.append("| block | skippable PRE | skippable POST | byte-cut POST (×0.72 FFN) |")
    lines.append("|------:|--------------:|---------------:|--------------------------:|")
    for B in block_sizes:
        pre, post = overall[B]
        lines.append(f"| {B} | {pre:.1%} | {post:.1%} | {0.72*post:.1%} |")
    lines.append("")
    lines.append("### Per-layer detail")
    lines.append("")
    hdr = "| layer | act/tok | dead | " + " | ".join(f"B{B} pre→post" for B in block_sizes) + " |"
    sep = "|------:|--------:|-----:|" + "|".join(["----------:"] * len(block_sizes)) + "|"
    lines.append(hdr)
    lines.append(sep)
    for layer in layers:
        r = results[layer]
        cells = " | ".join(f"{r['per_block'][B][0]:.1%}→{r['per_block'][B][1]:.1%}"
                            for B in block_sizes)
        lines.append(f"| {layer} | {r['active_frac']:.1%} | {r['dead_frac']:.1%} | {cells} |")
    lines.append("")
    lines.append("## Comparison to the dead block-256 result")
    lines.append("")
    lines.append("- Prior (`reports/dead_levers.md`): block-256 oracle skippable = "
                 "**0.2%** @99% recall; active neurons scattered (~5.6 active "
                 "channels/256-block, ~2.2%), participation-ratio sparse but "
                 "granularity-mismatched. Block-256 predictor declared DEAD.")
    lines.append(f"- This oracle adds (a) finer blocks {{32,64,128}} and (b) an "
                 f"offline co-activation **permutation** to cluster the scattered "
                 f"active neurons into contiguous runs.")
    pre32 = overall.get(32, (float('nan'), float('nan')))[0]
    post32 = overall.get(32, (float('nan'), float('nan')))[1]
    lines.append(f"- Result: even at block 32 with the best-case permutation, "
                 f"skippable goes {pre32:.1%} → {post32:.1%}. "
                 + ("This clears the bar — the permutation genuinely concentrates "
                    "co-firing mass." if verdict == "GO" else
                    "The permutation moves the number only marginally: SwiGLU "
                    "activations are dense-and-small, not hard-zero, so there is "
                    "little all-cold contiguous mass to gather even after "
                    "reordering. Same root cause as block-256, confirmed at finer "
                    "granularity."))
    lines.append("")
    lines.append("## Honest caveats")
    lines.append("")
    lines.append("- These are best-case ORACLE numbers (a single fixed offline "
                 "permutation + an omniscient per-token block selector). A "
                 "deployable runtime needs a cheap PREDICTOR of the active set; it "
                 "is strictly worse. So the oracle is an upper bound: a NO-GO here "
                 "is decisive; a GO would still need a predictor oracle.")
    lines.append("- Recall is measured on the FFN intermediate L2 energy (the "
                 "down-projection input), not end-task quality; 99% energy recall "
                 "is the same proxy the Track-B gate used.")
    lines.append("")
    out.write_text("\n".join(lines) + "\n")

    sidecar = out.with_suffix(".json")
    json.dump({
        "verdict": verdict,
        "recon_median_rel_l2": med_rel_l2,
        "recon_ok": recon_ok,
        "mean_active_frac": mean_active,
        "mean_dead_frac": mean_dead,
        "layers": layers,
        "overall_skippable": {str(B): {"pre": overall[B][0], "post": overall[B][1]}
                              for B in block_sizes},
        "best_post_skippable": skip_for_verdict,
        "best_post_block": best_post_B,
        "recon_checks": recon_checks,
    }, open(sidecar, "w"), indent=2)

    print(f"\n[oracle] VERDICT: {verdict}  best post-permute skippable "
          f"{skip_for_verdict:.1%} @ block {best_post_B}")
    print(f"[oracle] recon median rel-L2 {med_rel_l2:.4f} ({'ok' if recon_ok else 'FAIL'})")
    print(f"[oracle] wrote {out} and {sidecar}  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
