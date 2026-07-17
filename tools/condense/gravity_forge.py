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
def _kmeans(v, k: int, *, iters: int = 12, seed: int = 0):
    """k-means on v:[N,D] -> centroids [K,D]. Runs on v's device (MPS when available)."""
    torch = _torch()
    n = v.shape[0]
    k = int(min(k, n))
    g = torch.Generator(device="cpu").manual_seed(seed)
    cb = v[torch.randperm(n, generator=g)[:k]].clone()
    v2 = (v * v).sum(1, keepdim=True)
    for _ in range(iters):
        d2 = v2 - 2.0 * (v @ cb.t()) + (cb * cb).sum(1)
        idx = d2.argmin(1)
        new = torch.zeros_like(cb)
        cnt = torch.zeros(k, device=cb.device, dtype=cb.dtype)
        new.index_add_(0, idx, v)
        cnt.index_add_(0, idx, torch.ones(n, device=cb.device, dtype=cb.dtype))
        nz = cnt > 0
        cb[nz] = new[nz] / cnt[nz].unsqueeze(1)
    return cb


def _assign(v, cb):
    torch = _torch()
    d2 = (v * v).sum(1, keepdim=True) - 2.0 * (v @ cb.t()) + (cb * cb).sum(1)
    return d2.argmin(1)


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
    for s in range(S):
        sl = slice(s * sub, (s + 1) * sub)
        cb = _kmeans(vr[:, sl].contiguous(), k, iters=iters, seed=seed + s)
        idx = _assign(vr[:, sl].contiguous(), cb)
        recon_r[:, sl] = cb[idx]
        ledger.add_index(N, cb.shape[0])
        ledger.add_fp16(cb.shape[0] * sub)           # codebook
    recon = (recon_r @ R).reshape(rows, cols)        # inverse rotation (R orthonormal => R^-1=R^T, and R@R=I here)
    base_bits = ledger.total_bits() - _METADATA_BYTES * 8
    return PackedArtifact("transform_pq", recon.detach().cpu().numpy().astype(np.float32),
                          w.size, ledger, base_bits, 0,
                          {"dim": D, "subspaces": S, "k": k, "seed": seed})


def _largest_pow2_divisor(cols: int, cap: int = 64) -> int:
    for d in (64, 32, 16, 8, 4, 2, 1):
        if d <= cap and cols % d == 0:
            return d
    return 1


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

    # deterministic: same seed => identical bytes
    det = pack_naive_rvq(lr, dim=32, k=64, stages=2, seed=7).physical_bytes == \
        pack_naive_rvq(lr, dim=32, k=64, stages=2, seed=7).physical_bytes
    # accounting invariant: whole >= base + doctor, bytes strictly positive, no NaNs in recon
    acct_ok = all(p.physical_bytes > 0
                  and p.whole_artifact_bpw >= (p.base_bpw + p.doctor_bpw) - 1e-6
                  and np.isfinite(p.recon).all()
                  for p in (a, gram, rep))
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
        "deterministic_bytes": det,
        "accounting_invariant_holds": acct_ok,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gravity Forge representation foundry.")
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args(argv)
    print(json.dumps(selftest(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
