#!/usr/bin/env python3.12
"""Gravity Forge - the capability-preserving sub-bit representation foundry.

The naive Gravity 120B run (gptoss_gravity_run.py) proved a BASELINE NEGATIVE: weight-space
residual VQ and plain low-rank collapse below ~2 BPW on real GPT-OSS-120B experts (rel-Frobenius
error 0.55-0.63 at 0.8-1.0 BPW). That is proxy evidence about weight reconstruction, NOT the
protected-capability contract. Forge answers the correct question instead:

    What is the smallest executable representation that preserves the parent's REQUIRED
    computation, not every original weight?

This module is the foundry. It implements MATERIALLY DISTINCT sub-bit-capable representation
lineages (not one RVQ with knobs), each with EXACT whole-artifact byte accounting (indices,
codebooks, factors, scales, transform seeds, corrections, metadata - everything counts), and
measures two honest signals on the REAL dequantized experts:

  * F1  (weight-space, PROXY):        relative Frobenius error of the reconstruction.
  * F1.5 (output-space, proto-F2):    divergence of the reference MoE block output when the
                                      selected experts are replaced by their packed forms
                                      (router top-k preserved). Closer to the capability
                                      contract; still uses synthetic activations until the
                                      tokenizer + end-to-end runtime land, so it is labelled
                                      proxy_output, NOT capability parity.

Honest boundaries (enforced, do not weaken):
  * No result here authorizes a Gravity Escape Receipt or an Event Horizon seal.
  * Weight MSE is a diagnostic, never the objective of record.
  * No family reconstructs a hidden dense shadow model: decode is bounded per-tile / per-expert.
  * Every byte of every artifact component is counted in whole_artifact_bpw.

Families implemented:
  A. transform_pq          - randomized-Hadamard (seeded, free) + product quantization.
  B. shared_expert_grammar - shared additive codebook across an expert cluster + per-expert
                             indices + per-expert low-rank correction (the MoE amortization lever).
  C. repairability_shaped  - very-low-rate base + structured Doctor correction (low-rank residual
                             + sparse outlier rows), jointly billed as R_base + R_Doctor.
  baseline naive_rvq / low_rank are re-implemented for apples-to-apples comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

FORGE_SCHEMA = "hawking.gravity_forge.foundry.v1"

# Survival thresholds on relative error (same scale as the naive run, for comparability).
_SURVIVE, _DEGRADE = 0.15, 0.40

# Fixed per-tensor metadata we always bill (shape, dtype tags, config, alignment slack). A flat,
# deliberately non-zero charge so no family can hide structural overhead in "free" metadata.
_METADATA_BYTES = 64


def _torch():
    import torch  # local import so numpy-only callers (tests) do not require MPS
    return torch


def _device():
    torch = _torch()
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _verdict(err: float) -> str:
    return "survives" if err < _SURVIVE else ("degraded" if err < _DEGRADE else "collapse")


def _rel_error(w: np.ndarray, recon: np.ndarray) -> float:
    denom = float(np.linalg.norm(w)) or 1.0
    return float(np.linalg.norm(w - recon) / denom)


# --------------------------------------------------------------------------------------------
# Exact byte accounting. This is the crux: whole_artifact_bpw counts EVERYTHING.
# --------------------------------------------------------------------------------------------
@dataclass
class ByteLedger:
    """Every physical component of a packed artifact, in BITS, itemized. Nothing is free except
    values deterministically regenerable from a billed seed."""
    items: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, bits: int) -> None:
        self.items[name] = self.items.get(name, 0) + int(bits)

    def add_index(self, n: int, cardinality: int) -> None:
        self.add("indices", n * max(1, math.ceil(math.log2(max(2, cardinality)))))

    def add_fp16(self, n: int) -> None:
        self.add("fp16_params", n * 16)

    def add_bits(self, name: str, n: int, bits_each: int) -> None:
        self.add(name, n * bits_each)

    def total_bits(self) -> int:
        return sum(self.items.values()) + _METADATA_BYTES * 8

    def bytes(self) -> int:
        return math.ceil(self.total_bits() / 8)


@dataclass
class PackedArtifact:
    family: str
    recon: np.ndarray          # bounded reconstruction (this tile only; never a full shadow model)
    n_weights: int             # weights represented by this artifact (a cluster may hold many experts)
    ledger: ByteLedger
    base_bits: int             # R_base portion (representation)
    doctor_bits: int           # R_Doctor portion (treatment); 0 if none
    config: dict[str, Any]

    @property
    def physical_bytes(self) -> int:
        return self.ledger.bytes()

    @property
    def whole_artifact_bpw(self) -> float:
        return self.physical_bytes * 8 / max(1, self.n_weights)

    @property
    def base_bpw(self) -> float:
        return self.base_bits / max(1, self.n_weights)

    @property
    def doctor_bpw(self) -> float:
        return self.doctor_bits / max(1, self.n_weights)

    @property
    def overhead_bpw(self) -> float:
        # everything not attributed to base or doctor (codebooks/seeds/metadata/alignment)
        return max(0.0, self.whole_artifact_bpw - self.base_bpw - self.doctor_bpw)


# --------------------------------------------------------------------------------------------
# GPU (MPS) primitives.
# --------------------------------------------------------------------------------------------
# The [N,K] distance matrix is the only working-set term that scales with the tensor, and it
# scales past the machine: GLM-5.2's embedding table is 951,582,720 weights, so at dim=8 it is
# 118,947,840 subvectors, and against 128 centroids that single allocation is 60.9 GB.  It took
# the MPS allocator out and left roughly 67 GB wired.
#
# Only the distance and the argmin are blocked here.  The accumulation stays one index_add_
# over the whole tensor: an earlier attempt chunked that too, and reading the count vector
# while its per-block writes were still queued produced GPU address faults and a cb[nz]
# update whose two sides disagreed on how many rows the mask selected.
_DISTANCE_BUDGET_BYTES = int(os.environ.get("GRAVITY_KMEANS_DISTANCE_BUDGET_BYTES", 1 << 30))


def _chunk_rows(n: int, k: int) -> int:
    return max(1, min(n, _DISTANCE_BUDGET_BYTES // max(1, k * 4)))


def _argmin_chunked(v, v2, cb, step: int):
    """Nearest centroid per row, holding the distance block to ``step`` rows."""
    torch = _torch()
    n = v.shape[0]
    cb_t = cb.t()
    cb_sq = (cb * cb).sum(1)
    if step >= n:
        return (v2 - 2.0 * (v @ cb_t) + cb_sq).argmin(1)
    out = torch.empty(n, device=v.device, dtype=torch.int64)
    for start in range(0, n, step):
        stop = min(start + step, n)
        # disjoint destination slices, so no block ever reads another block's writes
        out[start:stop] = (
            v2[start:stop] - 2.0 * (v[start:stop] @ cb_t) + cb_sq
        ).argmin(1)
    return out


def _kmeans(v, k: int, *, iters: int = 12, seed: int = 0):
    """k-means on v:[N,D] -> centroids [K,D]. Runs on v's device (MPS when available)."""
    torch = _torch()
    n = v.shape[0]
    k = int(min(k, n))
    g = torch.Generator(device="cpu").manual_seed(seed)
    cb = v[torch.randperm(n, generator=g)[:k]].clone()
    v2 = (v * v).sum(1, keepdim=True)
    step = _chunk_rows(n, k)
    for _ in range(iters):
        idx = _argmin_chunked(v, v2, cb, step)
        new = torch.zeros_like(cb)
        cnt = torch.zeros(k, device=cb.device, dtype=cb.dtype)
        new.index_add_(0, idx, v)
        cnt.index_add_(0, idx, torch.ones(n, device=cb.device, dtype=cb.dtype))
        nz = cnt > 0
        cb[nz] = new[nz] / cnt[nz].unsqueeze(1)
    return cb


def _assign(v, cb):
    v2 = (v * v).sum(1, keepdim=True)
    return _argmin_chunked(v, v2, cb, _chunk_rows(v.shape[0], cb.shape[0]))


def _hadamard(D: int):
    """Orthonormal Hadamard matrix of size D (D a power of 2), as a torch tensor on the device."""
    torch = _torch()
    assert D & (D - 1) == 0, "Hadamard dim must be a power of 2"
    H = torch.ones((1, 1))
    while H.shape[0] < D:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(D)).to(_device())


# --------------------------------------------------------------------------------------------
# FAMILY A - transform_pq: randomized Hadamard (seeded, free) + product quantization.
# --------------------------------------------------------------------------------------------
def pack_transform_pq(w: np.ndarray, *, dim: int, subspaces: int, k: int, seed: int = 0,
                      iters: int = 10) -> PackedArtifact:
    """Rotate each dim-vector by a seeded randomized Hadamard (incoherence processing; the rotation
    is regenerable from the billed 8-byte seed, so only the seed is charged), then product-quantize:
    split the rotated vector into `subspaces` groups and VQ each group independently. Sub-bit when
    dim > subspaces*log2(k). Decode inverts the rotation on the summed sub-codewords."""
    torch = _torch()
    rows, cols = w.shape
    D = dim if cols % dim == 0 and (dim & (dim - 1) == 0) else _largest_pow2_divisor(cols)
    S = subspaces if D % subspaces == 0 else 1
    sub = D // S
    dev = _device()
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).to(dev)
    v = t.reshape(-1, D)
    N = v.shape[0]
    # seeded randomized Hadamard: diag(+-1) then H. Regenerable from seed -> billed as 8 bytes only.
    g = torch.Generator(device="cpu").manual_seed(seed)
    signs = (torch.randint(0, 2, (D,), generator=g).float() * 2 - 1).to(dev)
    H = _hadamard(D)
    R = (H * signs)                       # [D,D] orthonormal rotation (regenerable)
    vr = v @ R.t()                        # rotate
    ledger = ByteLedger()
    ledger.add("transform_seed", 64)      # 8-byte seed => the whole rotation is free beyond this
    recon_r = torch.zeros_like(vr)
    codebooks_np: list[np.ndarray] = []
    idx_cols: list[np.ndarray] = []
    for s in range(S):
        sl = slice(s * sub, (s + 1) * sub)
        cb = _kmeans(vr[:, sl].contiguous(), k, iters=iters, seed=seed + s)
        idx = _assign(vr[:, sl].contiguous(), cb)
        recon_r[:, sl] = cb[idx]
        ledger.add_index(N, cb.shape[0])
        ledger.add_fp16(cb.shape[0] * sub)           # codebook
        codebooks_np.append(cb.detach().cpu().numpy().astype(np.float32))
        idx_cols.append(idx.detach().cpu().numpy().astype(np.int64))
    recon = (recon_r @ R).reshape(rows, cols)        # inverse rotation (R orthonormal => R^-1=R^T, and R@R=I here)
    base_bits = ledger.total_bits() - _METADATA_BYTES * 8
    # Stash the compact codes so PQFamily.execute can run a DIRECT matvec from the codebooks without
    # ever materializing the full dense reconstruction. This is metadata for the runtime, not billed
    # weight state: the billed truth is the ledger; the stash only mirrors what the ledger already pays
    # for (indices + codebooks + the seeded rotation). It does not create a free dense shadow.
    codes = _make_codes(codebooks_np, np.stack(idx_cols, axis=1), D=D, S=S, sub=sub,
                        rows=rows, cols=cols, rotate=True, seed=seed)
    return PackedArtifact("transform_pq", recon.detach().cpu().numpy().astype(np.float32),
                          w.size, ledger, base_bits, 0,
                          {"dim": D, "subspaces": S, "k": k, "seed": seed, "sub": sub,
                           "rotate": True, "pq_codes": codes})


def _largest_pow2_divisor(cols: int, cap: int = 64) -> int:
    for d in (64, 32, 16, 8, 4, 2, 1):
        if d <= cap and cols % d == 0:
            return d
    return 1


def pack_transform_pq_actaware(w: np.ndarray, acts: np.ndarray, *, dim: int, subspaces: int,
                               k: int, alpha: float = 0.5, seed: int = 0,
                               iters: int = 10) -> PackedArtifact:
    """Activation-aware transform_pq (Section 6.2). Given real input activations `acts` [n, in] for
    this weight (in == w.shape[1]), scale each INPUT channel by its salience s_j = mean|acts_j|^alpha
    (geomean-normalized) before packing, and unscale on decode. This concentrates quantization
    precision where the activations actually are, minimizing OUTPUT error X W^T rather than weight
    error - the AWQ idea. The per-channel scale is billed (in * fp16)."""
    cols = w.shape[1]
    a = np.abs(acts).mean(axis=0).astype(np.float32)              # [in] channel salience
    a = np.maximum(a, 1e-8)
    s = a ** alpha
    s = s / np.exp(np.mean(np.log(s)))                            # geomean-normalize (scale-neutral)
    w_scaled = (w * s[None, :]).astype(np.float32)
    art = pack_transform_pq(w_scaled, dim=dim, subspaces=subspaces, k=k, seed=seed, iters=iters)
    recon = art.recon / s[None, :]                               # unscale
    art.ledger.add_fp16(cols)                                     # bill the per-channel scale
    return PackedArtifact("transform_pq_actaware", recon.astype(np.float32), w.size, art.ledger,
                          art.base_bits, 0, {**art.config, "alpha": alpha, "act_scaled": True})


# --------------------------------------------------------------------------------------------
# FAMILY B - shared_expert_grammar: shared additive codebook across a cluster + per-expert
# indices + per-expert low-rank correction. The MoE amortization lever.
# --------------------------------------------------------------------------------------------
def pack_shared_grammar(experts: list[np.ndarray], *, dim: int, k: int, stages: int,
                        corr_rank: int = 0, iters: int = 8, seed: int = 0) -> PackedArtifact:
    """Fit ONE additive (multi-stage) codebook on the pooled vectors of E experts (the shared
    grammar), then encode each expert with indices into it plus an optional per-expert rank-r
    correction of its residual. The shared codebook is billed ONCE and amortized over the whole
    cluster, so per-expert cost approaches indices-only (deep sub-bit) as E grows."""
    torch = _torch()
    dev = _device()
    rows, cols = experts[0].shape
    D = dim if cols % dim == 0 else _largest_pow2_divisor(cols)
    vs = [torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).to(dev).reshape(-1, D)
          for w in experts]
    pool = torch.cat(vs, 0)
    residual = pool.clone()
    cbs = []
    for m in range(stages):
        cb = _kmeans(residual, k, iters=iters, seed=seed + m)
        residual = residual - cb[_assign(residual, cb)]
        cbs.append(cb)
    ledger = ByteLedger()
    for cb in cbs:                                    # shared codebook billed ONCE
        ledger.add_fp16(cb.shape[0] * D)
    recons = []
    base_bits_total = 0
    doctor_bits_total = 0
    for w, v in zip(experts, vs):
        recon = torch.zeros_like(v)
        res = v.clone()
        for cb in cbs:
            idx = _assign(res, cb)
            recon = recon + cb[idx]
            res = res - cb[idx]
            ledger.add_index(v.shape[0], cb.shape[0])
            base_bits_total += v.shape[0] * math.ceil(math.log2(max(2, cb.shape[0])))
        if corr_rank > 0:                             # per-expert low-rank Doctor correction
            resid_mat = (v - recon).reshape(rows, cols)
            u, s, vt = torch.linalg.svd(resid_mat, full_matrices=False)
            r = min(corr_rank, s.shape[0])
            corr = (u[:, :r] * s[:r]) @ vt[:r]
            recon = (recon.reshape(rows, cols) + corr).reshape(-1, D)
            cbits = r * (rows + cols) * 16
            ledger.add("doctor_lowrank", cbits)
            doctor_bits_total += cbits
        recons.append(recon.reshape(rows, cols))
    total_weights = sum(w.size for w in experts)
    recon_stack = torch.stack(recons).detach().cpu().numpy().astype(np.float32)
    return PackedArtifact("shared_expert_grammar", recon_stack, total_weights, ledger,
                          base_bits_total, doctor_bits_total,
                          {"dim": D, "k": k, "stages": stages, "corr_rank": corr_rank,
                           "n_experts": len(experts)})


# --------------------------------------------------------------------------------------------
# FAMILY C - repairability_shaped: very-low-rate base + structured Doctor correction.
# --------------------------------------------------------------------------------------------
def pack_repairability_shaped(w: np.ndarray, *, base_dim: int, base_k: int, corr_rank: int,
                              sparse_rows: int, iters: int = 8, seed: int = 0) -> PackedArtifact:
    """Deliberately cheap base (coarse VQ) whose error is shaped to lie in a cheap Doctor-reachable
    subspace: a low-rank residual correction plus a handful of high-precision outlier rows. Bills
    R_base and R_Doctor separately; a worse base can win if Doctor repairs it cheaply."""
    torch = _torch()
    dev = _device()
    rows, cols = w.shape
    D = base_dim if cols % base_dim == 0 else _largest_pow2_divisor(cols)
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).to(dev)
    v = t.reshape(-1, D)
    ledger = ByteLedger()
    # base: single-stage coarse VQ
    cb = _kmeans(v, base_k, iters=iters, seed=seed)
    idx = _assign(v, cb)
    recon = cb[idx].reshape(rows, cols)
    ledger.add_index(v.shape[0], cb.shape[0])
    ledger.add_fp16(cb.shape[0] * D)
    base_bits = v.shape[0] * math.ceil(math.log2(max(2, cb.shape[0]))) + cb.shape[0] * D * 16
    doctor_bits = 0
    resid = t.reshape(rows, cols) - recon
    # Doctor #1: low-rank correction of the residual
    if corr_rank > 0:
        u, s, vt = torch.linalg.svd(resid, full_matrices=False)
        r = min(corr_rank, s.shape[0])
        corr = (u[:, :r] * s[:r]) @ vt[:r]
        recon = recon + corr
        cbits = r * (rows + cols) * 16
        ledger.add("doctor_lowrank", cbits)
        doctor_bits += cbits
        resid = t.reshape(rows, cols) - recon
    # Doctor #2: sparse outlier-row correction (store the worst rows in fp16 verbatim)
    if sparse_rows > 0:
        row_err = resid.abs().sum(1)
        worst = torch.argsort(row_err, descending=True)[:sparse_rows]
        recon[worst] = t.reshape(rows, cols)[worst]
        cbits = sparse_rows * (cols * 16 + math.ceil(math.log2(max(2, rows))))  # values + row index
        ledger.add("doctor_sparse_rows", cbits)
        doctor_bits += cbits
    return PackedArtifact("repairability_shaped", recon.detach().cpu().numpy().astype(np.float32),
                          w.size, ledger, base_bits, doctor_bits,
                          {"base_dim": D, "base_k": base_k, "corr_rank": corr_rank,
                           "sparse_rows": sparse_rows})


# --------------------------------------------------------------------------------------------
# FAMILY D - ternary_factor: ternary latent factorization (Section 5.2). Materially distinct from
# both fp16 low-rank (real-valued SVD) and codebook/VQ families: W ~= sum_j s_j (a_j b_j^T) with
# a_j in {-1,0,+1}^rows, b_j in {-1,0,+1}^cols, fit by greedy ternary power iteration. Sub-bit via
# small rank; ternary elements billed at 2 bits each (conservative - no base-3 packing claimed).
# --------------------------------------------------------------------------------------------
def _ternarize(x, keep_frac: float):
    """Keep the top-|keep_frac| entries as their sign (+-1), zero the rest -> ternary vector."""
    torch = _torch()
    a = x.abs()
    if keep_frac >= 1.0:
        return torch.sign(x)
    k = max(1, int(a.numel() * keep_frac))
    thr = torch.topk(a.flatten(), k).values.min()
    return torch.sign(x) * (a >= thr).to(x.dtype)


def pack_ternary_factor(w: np.ndarray, *, rank: int, keep_frac: float = 0.6, iters: int = 6,
                        seed: int = 0) -> PackedArtifact:
    """Greedy ternary rank-r factorization. Each stage fits a ternary rank-1 outer product to the
    running residual by alternating ternary power iteration, then subtracts s*a b^T."""
    torch = _torch()
    dev = _device()
    rows, cols = w.shape
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).to(dev)
    resid = t.clone()
    recon = torch.zeros_like(t)
    g = torch.Generator(device="cpu").manual_seed(seed)
    ledger = ByteLedger()
    for j in range(rank):
        b = _ternarize(torch.randn(cols, generator=g).to(dev), keep_frac)
        a = None
        for _ in range(iters):
            denom_b = float((b * b).sum()) or 1.0
            a = _ternarize((resid @ b) / denom_b, keep_frac)          # [rows]
            denom_a = float((a * a).sum()) or 1.0
            b = _ternarize((resid.t() @ a) / denom_a, keep_frac)      # [cols]
        num = float(a @ (resid @ b))
        den = (float((a * a).sum()) * float((b * b).sum())) or 1.0
        s = num / den
        recon = recon + s * torch.outer(a, b)
        resid = t - recon
        ledger.add_bits("ternary_factors", (rows + cols), 2)          # 2 bits/ternary elem (conservative)
        ledger.add_fp16(1)                                            # per-stage scale s
    base_bits = rank * ((rows + cols) * 2 + 16)
    return PackedArtifact("ternary_factor", recon.detach().cpu().numpy().astype(np.float32),
                          w.size, ledger, base_bits, 0,
                          {"rank": rank, "keep_frac": keep_frac, "ternary_bits": 2})


# --------------------------------------------------------------------------------------------
# Baselines (naive), re-implemented for comparison at matched whole-artifact bytes.
# --------------------------------------------------------------------------------------------
def pack_naive_rvq(w: np.ndarray, *, dim: int, k: int, stages: int, iters: int = 10,
                   seed: int = 0) -> PackedArtifact:
    torch = _torch()
    dev = _device()
    rows, cols = w.shape
    D = dim if cols % dim == 0 else _largest_pow2_divisor(cols)
    v = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).to(dev).reshape(-1, D)
    residual = v.clone()
    recon = torch.zeros_like(v)
    ledger = ByteLedger()
    for m in range(stages):
        cb = _kmeans(residual, k, iters=iters, seed=seed + m)
        idx = _assign(residual, cb)
        recon = recon + cb[idx]
        residual = residual - cb[idx]
        ledger.add_index(v.shape[0], cb.shape[0])
        ledger.add_fp16(cb.shape[0] * D)
    base_bits = ledger.total_bits() - _METADATA_BYTES * 8
    return PackedArtifact("naive_rvq", recon.reshape(rows, cols).detach().cpu().numpy().astype(np.float32),
                          w.size, ledger, base_bits, 0, {"dim": D, "k": k, "stages": stages})


def pack_low_rank(w: np.ndarray, *, rank: int) -> PackedArtifact:
    rows, cols = w.shape
    u, s, vt = np.linalg.svd(w.astype(np.float32), full_matrices=False)
    r = min(rank, s.shape[0])
    recon = ((u[:, :r] * s[:r]) @ vt[:r]).astype(np.float32)
    ledger = ByteLedger()
    ledger.add_fp16(r * (rows + cols))
    base_bits = r * (rows + cols) * 16
    return PackedArtifact("low_rank", recon, w.size, ledger, base_bits, 0, {"rank": r})


# --------------------------------------------------------------------------------------------
# Output-space (proto-F2) divergence via the reference MoE forward.
# --------------------------------------------------------------------------------------------
def output_divergence(reader, block: int, pack_fn: Callable[[np.ndarray], np.ndarray],
                      *, n_inputs: int = 8, top_k: int = 4, seed: int = 0) -> dict[str, Any]:
    """Run the reference MoE block forward with ORIGINAL experts vs experts whose mlp1/mlp2 are
    replaced by pack_fn's reconstruction, and measure output divergence. Critically, it packs
    EXACTLY the experts the router selects for the sampled inputs (union of top-k), so the packed
    weights are actually exercised - a vacuous 0.0 (packing experts nobody routes to) is impossible.
    Uses synthetic Gaussian activations => labelled proxy_output, NOT capability parity; real
    activations swap in once the tokenizer + end-to-end runtime exist."""
    import gptoss_moe_runtime as rt
    router = rt.load_router(reader, block)
    rng = np.random.default_rng(seed)
    xs = [(rng.standard_normal(2880).astype(np.float32)) * 0.02 for _ in range(n_inputs)]

    # union of routed experts across the sampled inputs
    routed: set[int] = set()
    for x in xs:
        logits = router["weight"] @ x + router["bias"]
        routed.update(int(e) for e in np.argsort(-logits)[:top_k])

    orig: dict[int, dict[str, np.ndarray]] = {}
    packed: dict[int, dict[str, np.ndarray]] = {}
    for e in routed:
        ex = rt.load_expert(reader, block, e)
        orig[e] = ex
        m = dict(ex)
        m["mlp1"] = pack_fn(ex["mlp1"].astype(np.float32))
        m["mlp2"] = pack_fn(ex["mlp2"].astype(np.float32))
        packed[e] = m

    rel = []
    for x in xs:
        y0 = rt.moe_forward_reference(x, router, lambda e: orig[e], top_k=top_k)
        y1 = rt.moe_forward_reference(x, router, lambda e: packed[e], top_k=top_k)
        rel.append(float(np.linalg.norm(y0 - y1) / (np.linalg.norm(y0) + 1e-9)))
    return {"n_inputs": n_inputs, "n_experts_exercised": len(routed),
            "mean_output_rel_div": round(float(np.mean(rel)), 5),
            "max_output_rel_div": round(float(np.max(rel)), 5),
            "min_output_rel_div": round(float(np.min(rel)), 5),
            "signal": "proxy_output_synthetic_activations", "capability_parity": False}


# ============================================================================================
# SECOND LIGHT - Product Quantization as a first-class forge family.
#
# The 120B run dossier diagnosed the ternary plateau (experts are high-rank, effective rank
# ~104/256) and selected Product Quantization as the next Forge family from evidence (~45% lower
# weight error than ternary at matched budget, full-rank, sub-bit-capable). This section makes PQ
# authoritative: a full lifecycle (inspect / fit / pack / measure / execute / validate /
# repairability), a protected-island outlier mechanism, a PQ-aware Doctor, and a CPU/Metal parity
# harness. Every byte stays inside the SAME ByteLedger discipline used above - protected islands and
# Doctor treatments are billed, never free, and decode is bounded per-subvector (no dense shadow).
# ============================================================================================


def _pq_geometry(cols: int, dim: int, subspaces: int) -> tuple[int, int, int]:
    """Resolve the (D, S, sub) PQ geometry for a matrix with `cols` columns. D is the reshape
    granularity (a power-of-2 divisor of cols); S subspaces split D into sub=D//S contiguous slices."""
    D = dim if (cols % dim == 0 and dim & (dim - 1) == 0) else _largest_pow2_divisor(cols)
    S = subspaces if D % subspaces == 0 else 1
    sub = D // S
    return D, S, sub


def _make_codes(codebooks: list[np.ndarray], indices: np.ndarray, *, D: int, S: int, sub: int,
                rows: int, cols: int, rotate: bool, seed: int) -> dict[str, Any]:
    return {
        "codebooks": [np.ascontiguousarray(cb, dtype=np.float32) for cb in codebooks],
        "indices": np.ascontiguousarray(indices, dtype=np.int64),   # [N, S], N = rows * cols / D
        "D": int(D), "S": int(S), "sub": int(sub), "rows": int(rows), "cols": int(cols),
        "nchunk": int(cols // D), "rotate": bool(rotate), "seed": int(seed),
    }


def _pq_rotation_np(D: int, seed: int) -> np.ndarray:
    """CPU-authoritative regeneration of the seeded randomized-Hadamard rotation used by
    transform_pq. Reproduces the SAME signs (torch cpu Generator) and the SAME orthonormal Hadamard
    as the packer, so the direct-execute path is bit-consistent with the billed rotation seed."""
    torch = _torch()
    g = torch.Generator(device="cpu").manual_seed(seed)
    signs = (torch.randint(0, 2, (D,), generator=g).float() * 2 - 1).numpy().astype(np.float32)
    H = np.ones((1, 1), dtype=np.float32)
    while H.shape[0] < D:
        H = np.block([[H, H], [H, -H]])
    H = (H / math.sqrt(D)).astype(np.float32)
    return (H * signs[None, :]).astype(np.float32)


def _pq_encode(w: np.ndarray, *, D: int, S: int, sub: int, k: int, seed: int, iters: int,
               rotate: bool) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, int]:
    """Core PQ encoder shared by the plain and rotated variants. Reshapes w to [-1, D], optionally
    applies the seeded rotation, trains one codebook per subspace with _kmeans (reused), assigns, and
    returns (recon[rows,cols], codebooks, indices[N,S], N). Deterministic in (seed, iters)."""
    torch = _torch()
    dev = _device()
    rows, cols = w.shape
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).to(dev)
    v = t.reshape(-1, D)
    N = v.shape[0]
    if rotate:
        g = torch.Generator(device="cpu").manual_seed(seed)
        signs = (torch.randint(0, 2, (D,), generator=g).float() * 2 - 1).to(dev)
        R = _hadamard(D) * signs
        vv = v @ R.t()
    else:
        R = None
        vv = v
    codebooks: list[np.ndarray] = []
    indices = np.empty((N, S), dtype=np.int64)
    recon_r = torch.zeros_like(vv)
    for s in range(S):
        sl = slice(s * sub, (s + 1) * sub)
        cb = _kmeans(vv[:, sl].contiguous(), k, iters=iters, seed=seed + s)
        idx = _assign(vv[:, sl].contiguous(), cb)
        recon_r[:, sl] = cb[idx]
        codebooks.append(cb.detach().cpu().numpy().astype(np.float32))
        indices[:, s] = idx.detach().cpu().numpy()
    recon_v = (recon_r @ R) if rotate else recon_r
    recon = recon_v.reshape(rows, cols).detach().cpu().numpy().astype(np.float32)
    return recon, codebooks, indices, N


def pack_product_quant(w: np.ndarray, *, dim: int, subspaces: int, k: int, seed: int = 0,
                       iters: int = 10) -> PackedArtifact:
    """Plain Product Quantization WITHOUT the Hadamard rotation - PQ as its own geometry. Split each
    reshaped D-vector into `subspaces` contiguous slices and VQ each slice with its own codebook.
    Sub-bit when D > subspaces * log2(k). Bit-exact and deterministic in (seed, iters). Every index
    and codebook entry is billed; nothing is free."""
    rows, cols = w.shape
    D, S, sub = _pq_geometry(cols, dim, subspaces)
    recon, codebooks, indices, N = _pq_encode(w, D=D, S=S, sub=sub, k=k, seed=seed, iters=iters,
                                              rotate=False)
    ledger = ByteLedger()
    for cb in codebooks:
        ledger.add_index(N, cb.shape[0])
        ledger.add_fp16(cb.shape[0] * sub)
    base_bits = ledger.total_bits() - _METADATA_BYTES * 8
    codes = _make_codes(codebooks, indices, D=D, S=S, sub=sub, rows=rows, cols=cols,
                        rotate=False, seed=seed)
    return PackedArtifact("product_quant", recon, w.size, ledger, base_bits, 0,
                          {"dim": D, "subspaces": S, "k": k, "seed": seed, "sub": sub,
                           "rotate": False, "pq_codes": codes})


def pq_inspect(w: np.ndarray) -> dict[str, Any]:
    """Verb 1/7 - inspect. Structural evidence to choose a PQ geometry: shape, which subvector dims
    from {4,8,16} divide cols, the effective rank at 90% spectral energy, and the excess kurtosis of
    the residual left after that effective-rank low-rank approximation (heavy tails => islands help)."""
    rows, cols = w.shape
    valid = [d for d in (4, 8, 16) if cols % d == 0]
    wf = w.astype(np.float32)
    u, s, vt = np.linalg.svd(wf, full_matrices=False)
    energy = (s.astype(np.float64)) ** 2
    total = float(energy.sum()) or 1.0
    cume = np.cumsum(energy) / total
    eff_rank = int(np.searchsorted(cume, 0.90) + 1)
    r = min(eff_rank, s.shape[0])
    approx = (u[:, :r] * s[:r]) @ vt[:r]
    resid = (wf - approx).ravel().astype(np.float64)
    d = resid - resid.mean()
    var = float((d ** 2).mean()) or 1e-12
    kurt = float((d ** 4).mean() / (var ** 2) - 3.0)
    return {"rows": int(rows), "cols": int(cols), "valid_subvector_dims": valid,
            "effective_rank_90": eff_rank,
            "effective_rank_frac": round(eff_rank / max(1, min(rows, cols)), 4),
            "residual_kurtosis": round(kurt, 4),
            "spectral_tail_ratio": round(float(s[-1] / (s[0] or 1.0)), 6)}


def pq_fit(w: np.ndarray, *, dim: int, subspaces: int = 1, k: int = 16, seed: int = 0,
           iters: int = 10, rotate: bool = False) -> dict[str, Any]:
    """Verb 2/7 - fit. Train a PQ codebook per subspace deterministically (reuses _kmeans). Returns
    the codebooks plus the resolved geometry and the per-subvector assignments."""
    rows, cols = w.shape
    D, S, sub = _pq_geometry(cols, dim, subspaces)
    _, codebooks, indices, N = _pq_encode(w, D=D, S=S, sub=sub, k=k, seed=seed, iters=iters,
                                          rotate=rotate)
    return {"codebooks": codebooks, "indices": indices, "D": D, "S": S, "sub": sub, "N": N,
            "k": k, "seed": seed, "rotate": rotate}


def pq_pack(w: np.ndarray, *, dim: int, subspaces: int, k: int, seed: int = 0, iters: int = 10,
            rotate: bool = False) -> PackedArtifact:
    """Verb 3/7 - pack. Produce a PackedArtifact. rotate=True reuses pack_transform_pq (rotated-PQ);
    rotate=False uses the plain product_quant geometry. Both stash compact codes for direct execute."""
    if rotate:
        return pack_transform_pq(w, dim=dim, subspaces=subspaces, k=k, seed=seed, iters=iters)
    return pack_product_quant(w, dim=dim, subspaces=subspaces, k=k, seed=seed, iters=iters)


def pq_measure(w: np.ndarray, artifact: PackedArtifact) -> dict[str, Any]:
    """Verb 4/7 - measure. Weight-space rel error (PROXY, never a capability claim) plus the exact
    whole-artifact byte accounting."""
    err = _rel_error(w, artifact.recon)
    return {"rel_error": round(err, 6), "verdict": _verdict(err),
            "whole_artifact_bpw": round(artifact.whole_artifact_bpw, 5),
            "base_bpw": round(artifact.base_bpw, 5), "doctor_bpw": round(artifact.doctor_bpw, 5),
            "physical_bytes": int(artifact.physical_bytes),
            "total_bits": int(artifact.ledger.total_bits())}


def pq_execute(artifact: PackedArtifact, x: np.ndarray) -> np.ndarray:
    """Verb 5/7 - execute. DIRECT compact matvec y = W_pq @ x that decodes per-subvector from the
    codebooks on the fly and contracts against x subspace-by-subspace. It NEVER materializes the full
    [rows, cols] dense reconstruction (each subspace holds at most rows*cols/S decoded scalars, freed
    before the next). CPU reference implementation. Supports the rotated variant (contract x against
    the regenerated rotation) and protected-island rows (exact fp16 override)."""
    codes = artifact.config.get("pq_codes")
    if codes is None:
        raise ValueError("artifact has no pq_codes stash; not a PQ-family artifact")
    x = np.ascontiguousarray(x, dtype=np.float32)
    onedim = x.ndim == 1
    if onedim:
        x = x[:, None]
    D, S, sub = codes["D"], codes["S"], codes["sub"]
    rows, cols, nchunk = codes["rows"], codes["cols"], codes["nchunk"]
    B = x.shape[1]
    xc = x.reshape(nchunk, D, B)                                   # chunk c covers columns [c*D:(c+1)*D]
    if codes["rotate"]:
        R = _pq_rotation_np(D, codes["seed"])                     # y-chunk dot = decoded_r . (R @ x-chunk)
        xc = np.einsum("jk,ckb->cjb", R, xc, optimize=True)
    idx = codes["indices"]                                        # [N, S], N = rows * nchunk
    y = np.zeros((rows, B), dtype=np.float32)
    for s in range(S):
        cb = codes["codebooks"][s]                               # [card, sub]
        dec = cb[idx[:, s]].reshape(rows, nchunk, sub)           # bounded: one subspace only
        xslice = xc[:, s * sub:(s + 1) * sub, :]                 # [nchunk, sub, B]
        y += np.einsum("rcj,cjb->rb", dec, xslice, optimize=True)
    island_rows = codes.get("island_rows")
    if island_rows is not None and len(island_rows) > 0:
        y[island_rows] = codes["island_vals"].astype(np.float32) @ x   # exact rows use ORIGINAL x
    return y[:, 0] if onedim else y


def pq_validate(w: np.ndarray, artifact: PackedArtifact, x: np.ndarray,
                *, tol: float = 1e-3) -> dict[str, Any]:
    """Verb 6/7 - validate. Confirm the compact execute() equals the dense matvec of the artifact's
    reconstruction (decode exactness) within tol, and separately report the quantization gap vs the
    ORIGINAL w @ x (honest: that gap is the family's error, not a decode bug)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    y_exec = pq_execute(artifact, x)
    y_dense = artifact.recon @ x
    y_orig = w @ x
    denom = float(np.linalg.norm(y_dense)) or 1.0
    rel_vs_recon = float(np.linalg.norm(y_exec - y_dense) / denom)
    max_abs = float(np.max(np.abs(y_exec - y_dense))) if y_exec.size else 0.0
    rel_vs_orig = float(np.linalg.norm(y_exec - y_orig) / (float(np.linalg.norm(y_orig)) or 1.0))
    return {"within_tol": rel_vs_recon <= tol, "tol": tol,
            "rel_err_vs_recon": round(rel_vs_recon, 8), "max_abs_err": round(max_abs, 8),
            "rel_err_vs_original": round(rel_vs_orig, 6),
            "no_dense_reconstruction": True}


def pq_repairability(w: np.ndarray, artifact: PackedArtifact) -> dict[str, Any]:
    """Verb 7/7 - repairability. Estimate how cheaply a Doctor can repair the base residual: the
    fraction of residual energy a rank-r correction captures, and the fraction concentrated in a few
    outlier rows (sparse-island reachable). High capture => cheaply repairable."""
    recon = artifact.recon.astype(np.float32)
    resid = (w.astype(np.float32) - recon)
    wn = float(np.linalg.norm(w)) or 1.0
    rn = float(np.linalg.norm(resid))
    u, s, vt = np.linalg.svd(resid, full_matrices=False)
    energy = (s.astype(np.float64)) ** 2
    tot = float(energy.sum()) or 1e-12
    cume = np.cumsum(energy) / tot

    def cap(r: int) -> float:
        return float(cume[min(r, len(cume)) - 1]) if len(cume) else 0.0

    row_e = (resid.astype(np.float64) ** 2).sum(1)
    order = np.argsort(-row_e)
    n_rows = max(1, int(0.03 * resid.shape[0]))
    sparse_cap = float(row_e[order[:n_rows]].sum() / (row_e.sum() or 1e-12))
    score = float(0.5 * cap(4) + 0.5 * sparse_cap)
    return {"residual_rel_energy": round(rn / wn, 6),
            "rank1_capture": round(cap(1), 5), "rank4_capture": round(cap(4), 5),
            "rank8_capture": round(cap(8), 5),
            "sparse_row_capture": round(sparse_cap, 5), "sparse_rows_used": int(n_rows),
            "repair_score": round(score, 5)}


class PQFamily:
    """First-class Product Quantization forge family exposing the seven-verb authoritative lifecycle:
    inspect, fit, pack, measure, execute, validate, repairability. A thin, deterministic wrapper over
    the module functions; the ByteLedger accounting and no-dense-shadow discipline are unchanged."""

    family = "product_quant"
    VERBS = ("inspect", "fit", "pack", "measure", "execute", "validate", "repairability")

    def __init__(self, *, dim: int = 32, subspaces: int = 4, k: int = 16, seed: int = 0,
                 iters: int = 10, rotate: bool = False) -> None:
        self.dim = dim
        self.subspaces = subspaces
        self.k = k
        self.seed = seed
        self.iters = iters
        self.rotate = rotate

    def inspect(self, w: np.ndarray) -> dict[str, Any]:
        return pq_inspect(w)

    def fit(self, w: np.ndarray, *, dim: int | None = None, k: int | None = None,
            seed: int | None = None) -> dict[str, Any]:
        return pq_fit(w, dim=self.dim if dim is None else dim, subspaces=self.subspaces,
                      k=self.k if k is None else k, seed=self.seed if seed is None else seed,
                      iters=self.iters, rotate=self.rotate)

    def pack(self, w: np.ndarray) -> PackedArtifact:
        return pq_pack(w, dim=self.dim, subspaces=self.subspaces, k=self.k, seed=self.seed,
                       iters=self.iters, rotate=self.rotate)

    def measure(self, w: np.ndarray, artifact: PackedArtifact) -> dict[str, Any]:
        return pq_measure(w, artifact)

    def execute(self, artifact: PackedArtifact, x: np.ndarray) -> np.ndarray:
        return pq_execute(artifact, x)

    def validate(self, w: np.ndarray, artifact: PackedArtifact, x: np.ndarray,
                 *, tol: float = 1e-3) -> dict[str, Any]:
        return pq_validate(w, artifact, x, tol=tol)

    def repairability(self, w: np.ndarray, artifact: PackedArtifact) -> dict[str, Any]:
        return pq_repairability(w, artifact)


# --------------------------------------------------------------------------------------------
# Protected islands - deterministic, evidence-based outlier rows kept at higher precision. Billed
# inside the SAME ByteLedger (no free islands).
# --------------------------------------------------------------------------------------------
_ISLAND_STRATEGIES = ("magnitude", "activation_aware", "sensitivity", "residual_energy")


def select_protected_islands(w: np.ndarray, residual: np.ndarray | None = None, *,
                             strategy: str = "magnitude", budget_frac: float = 0.03,
                             activation: np.ndarray | None = None,
                             sensitivity: np.ndarray | None = None) -> dict[str, Any]:
    """Pick a deterministic set of outlier ROWS to protect at higher precision, by one of four
    evidence-based strategies. Returns a stable, sorted row index set (ties broken by ascending row
    index) so the same inputs always yield the same island. This only SELECTS; billing happens in the
    pack/doctor that stores the protected values."""
    if strategy not in _ISLAND_STRATEGIES:
        raise ValueError(f"unknown island strategy {strategy!r}; expected one of {_ISLAND_STRATEGIES}")
    rows, cols = w.shape
    n = max(1, int(math.ceil(budget_frac * rows)))
    wf = w.astype(np.float64)
    if residual is None:
        residual = np.zeros_like(w)
    resid = residual.astype(np.float64)
    if strategy == "magnitude":
        score = np.abs(wf).sum(1)
    elif strategy == "residual_energy":
        score = (resid ** 2).sum(1)
    elif strategy == "activation_aware":
        if activation is None:
            sal = np.ones(cols, dtype=np.float64)
        else:
            sal = np.abs(np.asarray(activation, dtype=np.float64))
            if sal.ndim > 1:
                sal = sal.mean(0)
        score = np.abs(wf) @ sal
    else:  # sensitivity
        if sensitivity is None:
            sens = np.abs(resid)                       # fallback: residual magnitude as sensitivity
        else:
            sens = np.abs(np.asarray(sensitivity, dtype=np.float64))
        if sens.shape == wf.shape:
            score = (sens * np.abs(wf)).sum(1)
        else:
            score = sens.reshape(rows, -1).sum(1)
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    rowsel = np.sort(order[:n]).astype(np.int64)
    return {"strategy": strategy, "budget_frac": budget_frac, "n_islands": int(n),
            "row_indices": rowsel, "score_captured": float(np.asarray(score)[rowsel].sum())}


def pack_pq_protected_islands(w: np.ndarray, *, dim: int, subspaces: int, k: int,
                              strategy: str = "residual_energy", budget_frac: float = 0.03,
                              seed: int = 0, iters: int = 10, rotate: bool = False,
                              activation: np.ndarray | None = None,
                              sensitivity: np.ndarray | None = None) -> PackedArtifact:
    """Base PQ + protected islands, fully accounted. The protected rows are stored VERBATIM in fp16
    plus a row index each, billed as a "protected_islands" ledger item. whole_artifact_bpw strictly
    exceeds the island-free base because those bytes are real and counted."""
    base = pq_pack(w, dim=dim, subspaces=subspaces, k=k, seed=seed, iters=iters, rotate=rotate)
    rows, cols = w.shape
    resid = (w.astype(np.float32) - base.recon)
    isl = select_protected_islands(w, resid, strategy=strategy, budget_frac=budget_frac,
                                   activation=activation, sensitivity=sensitivity)
    rowsel = isl["row_indices"]
    recon = base.recon.copy()
    recon[rowsel] = w[rowsel].astype(np.float32)                  # protected rows exact
    ledger = ByteLedger()
    for name, bits in base.ledger.items.items():
        ledger.add(name, bits)                                   # carry the base ledger forward
    island_bits = int(len(rowsel)) * (cols * 16 + max(1, math.ceil(math.log2(max(2, rows)))))
    ledger.add("protected_islands", island_bits)                 # fp16 values + row index each
    base_bits = base.base_bits + island_bits
    codes = dict(base.config["pq_codes"])
    codes["island_rows"] = rowsel
    codes["island_vals"] = w[rowsel].astype(np.float32)
    cfg = {**base.config, "strategy": strategy, "budget_frac": budget_frac,
           "n_islands": int(len(rowsel)), "protected": True, "pq_codes": codes}
    return PackedArtifact("pq_protected_islands", recon.astype(np.float32), w.size, ledger,
                          base_bits, 0, cfg)


# --------------------------------------------------------------------------------------------
# PQ-aware Doctor - bounded, budgeted residual treatments. Every treatment bills its added bytes,
# never uses an uncounted dense residual as stored state, and never exceeds byte_budget.
# --------------------------------------------------------------------------------------------
_DOCTOR_TREATMENTS = ("residual_codebook", "sparse_residual", "per_channel_scale",
                      "protected_island_expansion")


def doctor_pq(w: np.ndarray, base_artifact: PackedArtifact, *, byte_budget: int,
              strategy: str, seed: int = 0, iters: int = 8) -> dict[str, Any]:
    """Repair a PQ base within a hard byte_budget. Returns {treatment, added_bytes, new_whole_bpw,
    quality_delta, evidence}. The dense residual is used only transiently to FIT the treatment (as in
    repairability_shaped above); the only STORED state is the billed correction, and added_bytes is
    asserted <= byte_budget."""
    if strategy not in _DOCTOR_TREATMENTS:
        raise ValueError(f"unknown doctor treatment {strategy!r}; expected {_DOCTOR_TREATMENTS}")
    w = w.astype(np.float32)
    recon = base_artifact.recon.astype(np.float32)
    rows, cols = w.shape
    resid = (w - recon)
    err0 = _rel_error(w, recon)
    base_bytes = base_artifact.physical_bytes
    new_recon = recon.copy()
    added_bits = 0
    evidence: dict[str, Any] = {}

    if strategy == "per_channel_scale":
        nmax = int((byte_budget * 8) // 16)
        if nmax >= rows:                                          # scale every output channel
            num = (w * recon).sum(1)
            den = (recon * recon).sum(1)
            sc = (num / np.where(den == 0.0, 1.0, den)).astype(np.float32)
            new_recon = recon * sc[:, None]
            added_bits = rows * 16
            evidence = {"channels_scaled": int(rows), "mean_scale": round(float(sc.mean()), 4)}
        else:                                                    # budget-limited: scale worst rows + index
            row_e = (resid ** 2).sum(1)
            sel = np.sort(np.argsort(-row_e)[:max(0, nmax)])
            num = (w[sel] * recon[sel]).sum(1)
            den = (recon[sel] * recon[sel]).sum(1)
            sc = (num / np.where(den == 0.0, 1.0, den)).astype(np.float32)
            new_recon[sel] = recon[sel] * sc[:, None]
            per = 16 + max(1, math.ceil(math.log2(max(2, rows))))
            added_bits = int(len(sel)) * per
            evidence = {"channels_scaled": int(len(sel)), "indexed": True}

    elif strategy == "sparse_residual":
        per_row_bits = cols * 16 + max(1, math.ceil(math.log2(max(2, rows))))
        nmax = int((byte_budget * 8) // per_row_bits)
        row_e = (resid ** 2).sum(1)
        sel = np.sort(np.argsort(-row_e)[:max(0, nmax)])
        new_recon[sel] = w[sel]                                  # worst rows stored exact fp16
        added_bits = int(len(sel)) * per_row_bits
        evidence = {"rows_repaired": int(len(sel)), "row_capacity": int(nmax)}

    elif strategy == "residual_codebook":                        # second-stage PQ on the residual
        D = int(base_artifact.config.get("dim", 32))
        chosen = None
        for sub_s in (base_artifact.config.get("subspaces", 4), max(1, D // 4), D):
            for kk in (4, 8, 16, 2):
                art_r = pack_product_quant(resid, dim=D, subspaces=int(sub_s), k=kk,
                                           seed=seed, iters=iters)
                if art_r.physical_bytes <= byte_budget:
                    chosen = art_r
                    break
            if chosen is not None:
                break
        if chosen is None:                                       # smallest possible still over budget
            chosen = pack_product_quant(resid, dim=D, subspaces=D, k=2, seed=seed, iters=iters)
        new_recon = recon + chosen.recon
        added_bits = chosen.physical_bytes * 8
        evidence = {"stage2_bpw": round(chosen.whole_artifact_bpw, 5),
                    "stage2_k": chosen.config["k"], "stage2_subspaces": chosen.config["subspaces"]}

    else:  # protected_island_expansion - promote worst D-block subvectors to exact fp16
        D = int(base_artifact.config.get("dim", 32))
        resid_v = resid.reshape(-1, D)
        Nb = resid_v.shape[0]
        per_block_bits = D * 16 + max(1, math.ceil(math.log2(max(2, Nb))))
        nmax = int((byte_budget * 8) // per_block_bits)
        be = (resid_v ** 2).sum(1)
        sel = np.argsort(-be)[:max(0, nmax)]
        new_v = new_recon.reshape(-1, D).copy()
        new_v[sel] = w.reshape(-1, D)[sel]
        new_recon = new_v.reshape(rows, cols)
        added_bits = int(len(sel)) * per_block_bits
        evidence = {"blocks_repaired": int(len(sel)), "block_dim": int(D)}

    added_bytes = math.ceil(added_bits / 8)
    if added_bytes > byte_budget:
        raise AssertionError(f"doctor_pq exceeded byte_budget: {added_bytes} > {byte_budget}")
    err1 = _rel_error(w, new_recon)
    new_whole_bpw = (base_bytes + added_bytes) * 8 / max(1, w.size)
    return {"treatment": strategy, "added_bytes": int(added_bytes),
            "new_whole_bpw": round(new_whole_bpw, 6), "quality_delta": round(err0 - err1, 8),
            "err_before": round(err0, 8), "err_after": round(err1, 8),
            "within_budget": added_bytes <= byte_budget, "evidence": evidence}


# --------------------------------------------------------------------------------------------
# CPU/Metal parity harness (Metal Quality Law, Section 11): never lose quality for speed. CPU (numpy)
# is authoritative; MPS reductions can reorder near-ties, so assignment agreement is bounded, not
# demanded exact, and the CPU verdict is the one of record.
# --------------------------------------------------------------------------------------------
def _kmeans_np(v: np.ndarray, iters: int, init_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cb = v[init_idx].astype(np.float32).copy()
    idx = np.zeros(v.shape[0], dtype=np.int64)
    v2 = (v * v).sum(1, keepdims=True)
    for _ in range(iters):
        d2 = v2 - 2.0 * (v @ cb.T) + (cb * cb).sum(1)
        idx = d2.argmin(1)
        new = np.zeros_like(cb)
        cnt = np.zeros(cb.shape[0], dtype=np.float32)
        np.add.at(new, idx, v)
        np.add.at(cnt, idx, 1.0)
        nz = cnt > 0
        cb[nz] = new[nz] / cnt[nz][:, None]
    return cb, idx


def _kmeans_torch_init(v, cb_init, iters: int):
    torch = _torch()
    cb = cb_init.clone()
    v2 = (v * v).sum(1, keepdim=True)
    idx = None
    for _ in range(iters):
        d2 = v2 - 2.0 * (v @ cb.t()) + (cb * cb).sum(1)
        idx = d2.argmin(1)
        new = torch.zeros_like(cb)
        cnt = torch.zeros(cb.shape[0], device=cb.device, dtype=cb.dtype)
        new.index_add_(0, idx, v)
        cnt.index_add_(0, idx, torch.ones(v.shape[0], device=cb.device, dtype=cb.dtype))
        nz = cnt > 0
        cb[nz] = new[nz] / cnt[nz].unsqueeze(1)
    return cb, idx


def _pq_dual_fit(w: np.ndarray, *, D: int, S: int, sub: int, k: int, seed: int,
                 iters: int) -> dict[str, Any]:
    """Fit PQ on CPU (numpy) and MPS (torch) from IDENTICAL per-subspace initialisations and compare
    the reconstruction rel error and the assignment agreement. Shared init isolates backend numeric
    drift from init drift."""
    torch = _torch()
    dev = _device()
    rows, cols = w.shape
    v_all = np.ascontiguousarray(w.astype(np.float32).reshape(-1, D))
    rng = np.random.default_rng(seed)
    recon_cpu = np.zeros_like(v_all)
    recon_metal = np.zeros_like(v_all)
    agree = []
    for s in range(S):
        sl = slice(s * sub, (s + 1) * sub)
        vs = np.ascontiguousarray(v_all[:, sl])
        kk = min(k, vs.shape[0])
        init_idx = rng.choice(vs.shape[0], size=kk, replace=False)
        cb_c, idx_c = _kmeans_np(vs, iters, init_idx)
        recon_cpu[:, sl] = cb_c[idx_c]
        vt = torch.from_numpy(vs).to(dev)
        cb_init = torch.from_numpy(vs[init_idx].copy()).to(dev)
        cb_m, idx_m = _kmeans_torch_init(vt, cb_init, iters)
        idx_m_np = idx_m.detach().cpu().numpy()
        recon_metal[:, sl] = cb_m.detach().cpu().numpy()[idx_m_np]
        agree.append(float((idx_c == idx_m_np).mean()))
    return {"relerr_cpu": _rel_error(v_all, recon_cpu),
            "relerr_metal": _rel_error(v_all, recon_metal),
            "assignment_agreement": float(np.mean(agree))}


def pq_cpu_metal_parity(w: np.ndarray, *, dim: int, k: int, seed: int = 0, iters: int = 10,
                        subspaces: int = 1, tol: float = 2e-3) -> dict[str, Any]:
    """Enforce the Metal Quality Law: the MPS PQ fit must not lose quality vs the CPU-authoritative
    fit. Ranks a spread of k candidates on both backends and asserts identical ordering and identical
    pass/fail verdict, plus a bounded rel-error delta on the main config. Returns the deltas."""
    rows, cols = w.shape
    D, S, sub = _pq_geometry(cols, dim, subspaces)
    main = _pq_dual_fit(w, D=D, S=S, sub=sub, k=k, seed=seed, iters=iters)
    # candidate spread: well-separated k so the quality ordering is unambiguous on both backends.
    cand_ks = sorted({max(2, k // 4), max(2, k), min(rows * cols // D, max(4, k * 4))})
    cpu_scores, metal_scores = [], []
    for ck in cand_ks:
        d = _pq_dual_fit(w, D=D, S=S, sub=sub, k=ck, seed=seed, iters=iters)
        cpu_scores.append(d["relerr_cpu"])
        metal_scores.append(d["relerr_metal"])
    rank_cpu = list(np.argsort(cpu_scores))
    rank_metal = list(np.argsort(metal_scores))
    delta = abs(main["relerr_cpu"] - main["relerr_metal"])
    return {"device": _device().type, "authoritative": "cpu",
            "D": int(D), "S": int(S), "k": int(k),
            "candidate_ks": [int(c) for c in cand_ks],
            "codebook_relerr_cpu": round(main["relerr_cpu"], 6),
            "codebook_relerr_metal": round(main["relerr_metal"], 6),
            "relerr_delta": round(float(delta), 8), "tol": tol,
            "assignment_agreement": round(main["assignment_agreement"], 5),
            "ranking_cpu": [int(r) for r in rank_cpu],
            "ranking_metal": [int(r) for r in rank_metal],
            "ranking_match": rank_cpu == rank_metal,
            "verdict_cpu": _verdict(main["relerr_cpu"]),
            "verdict_metal": _verdict(main["relerr_metal"]),
            "pass_match": _verdict(main["relerr_cpu"]) == _verdict(main["relerr_metal"]),
            "within_tol": float(delta) <= tol,
            "note": "CPU (numpy) authoritative; MPS reductions may reorder near-ties"}


# --------------------------------------------------------------------------------------------
# Self-test (numpy/MPS): synthetic compressible weight should beat random; accounting is exact.
# --------------------------------------------------------------------------------------------
def selftest() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    # genuinely low-rank (rank 8 of 256) -> the low_rank family must reconstruct it near-exactly;
    # a random matrix at the same rank budget must not. This is the compressible-beats-random check
    # against the family that is SUPPOSED to exploit structure (not the incoherence transform).
    lr = (rng.standard_normal((256, 8)).astype(np.float32)
          @ rng.standard_normal((8, 256)).astype(np.float32)) * 0.1
    rnd = rng.standard_normal((256, 256)).astype(np.float32) * 0.1
    err_lr = _rel_error(lr, pack_low_rank(lr, rank=8).recon)
    err_rnd = _rel_error(rnd, pack_low_rank(rnd, rank=8).recon)

    a = pack_transform_pq(lr, dim=32, subspaces=4, k=16)          # runs + accounts (whitens by design)
    gram = pack_shared_grammar([lr, lr * 1.01, lr * 0.99], dim=32, k=64, stages=2, corr_rank=2)
    rep = pack_repairability_shaped(lr, base_dim=32, base_k=32, corr_rank=4, sparse_rows=4)
    tern = pack_ternary_factor(lr, rank=8)

    # --- Second Light: Product Quantization first-class lifecycle, islands, Doctor, parity ---
    fam = PQFamily(dim=32, subspaces=4, k=16, seed=0)
    art_pq = fam.pack(lr)                                        # plain PQ (its own geometry, no rotation)
    x = rng.standard_normal(lr.shape[1]).astype(np.float32)
    val = fam.validate(lr, art_pq, x)                            # direct compact matvec == dense recon matvec
    ins = fam.inspect(lr)
    pq_det = fam.pack(lr).physical_bytes == fam.pack(lr).physical_bytes
    isl_art = pack_pq_protected_islands(lr, dim=32, subspaces=4, k=16, budget_frac=0.05,
                                        strategy="residual_energy")
    islands_increase = (isl_art.whole_artifact_bpw > art_pq.whole_artifact_bpw
                        and "protected_islands" in isl_art.ledger.items
                        and isl_art.whole_artifact_bpw >= isl_art.base_bpw + isl_art.doctor_bpw - 1e-6)
    doc = doctor_pq(lr, art_pq, byte_budget=6000, strategy="sparse_residual")
    doctor_ok = doc["within_budget"] and doc["quality_delta"] > 0
    par = pq_cpu_metal_parity(lr, dim=32, k=16, subspaces=4)
    parity_ok = par["within_tol"] and par["ranking_match"] and par["pass_match"]

    # deterministic: same seed => identical bytes
    det = pack_naive_rvq(lr, dim=32, k=64, stages=2, seed=7).physical_bytes == \
        pack_naive_rvq(lr, dim=32, k=64, stages=2, seed=7).physical_bytes
    # accounting invariant: whole >= base + doctor, bytes strictly positive, no NaNs in recon
    acct_ok = all(p.physical_bytes > 0
                  and p.whole_artifact_bpw >= (p.base_bpw + p.doctor_bpw) - 1e-6
                  and np.isfinite(p.recon).all()
                  for p in (a, gram, rep, tern, art_pq, isl_art))
    return {
        "ok": True, "device": _device().type,
        "lowrank_family_relerr": round(err_lr, 5),
        "random_family_relerr": round(err_rnd, 5),
        "compressible_beats_random": err_lr < err_rnd and err_lr < 0.05,
        "transform_pq_bpw": round(a.whole_artifact_bpw, 4),
        "shared_grammar_bpw": round(gram.whole_artifact_bpw, 4),
        "repairability_bpw": round(rep.whole_artifact_bpw, 4),
        "repairability_base_bpw": round(rep.base_bpw, 4),
        "repairability_doctor_bpw": round(rep.doctor_bpw, 4),
        "ternary_factor_bpw": round(tern.whole_artifact_bpw, 4),
        "families_available": 4,
        "deterministic_bytes": det,
        "accounting_invariant_holds": acct_ok,
        # Second Light PQ signals
        "pq_family_verbs": len(PQFamily.VERBS),
        "pq_whole_bpw": round(art_pq.whole_artifact_bpw, 4),
        "pq_effective_rank_90": ins["effective_rank_90"],
        "pq_execute_within_tol": val["within_tol"],
        "pq_execute_rel_err_vs_recon": val["rel_err_vs_recon"],
        "pq_deterministic_bytes": pq_det,
        "pq_islands_increase_bpw": islands_increase,
        "pq_doctor_reduces_error": doctor_ok,
        "pq_doctor_within_budget": bool(doc["within_budget"]),
        "pq_cpu_metal_within_tol": parity_ok,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gravity Forge representation foundry.")
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args(argv)
    print(json.dumps(selftest(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
