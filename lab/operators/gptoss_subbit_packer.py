"""Recomposed science module gptoss_subbit_packer (C-SCI-R1)."""
from __future__ import annotations
import sys as _sys_a1
from pathlib import Path as _Path_a1
import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
_A1_HERE = _Path_a1(__file__).resolve().parent
_A1_CONDENSE = _A1_HERE.parent if _A1_HERE.name == 'archive' else _A1_HERE
_A1_REPO = _A1_CONDENSE.parents[1]
if str(_A1_CONDENSE) not in _sys_a1.path:
    _sys_a1.path.insert(0, str(_A1_CONDENSE))
_HERE = str(_A1_CONDENSE)
PACKER_SCHEMA = 'hawking.gravity.subbit_packer.v1'

def _device() -> torch.device:
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def gpu_kmeans(v: torch.Tensor, k: int, *, iters: int=12, seed: int=0) -> torch.Tensor:
    """k-means on the GPU. v: [N, D] -> codebook [K, D]. Distances via the ||a-b||^2 identity
    (matmul-heavy, so it saturates the Metal cores)."""
    n = v.shape[0]
    g = torch.Generator(device='cpu').manual_seed(seed)
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

def _assign(v: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
    d2 = (v * v).sum(1, keepdim=True) - 2.0 * (v @ cb.t()) + (cb * cb).sum(1)
    return d2.argmin(1)

@dataclass
class RvqCode:
    codebooks: torch.Tensor
    indices: torch.Tensor
    dim: int
    rows: int
    cols: int
    bpw: float

def rvq_encode(w: np.ndarray, *, dim: int, k: int, stages: int, iters: int=12, device: torch.device | None=None) -> RvqCode:
    """Residual vector quantization on the GPU. bpw = stages*log2(k)/dim."""
    device = device or _device()
    rows, cols = w.shape
    d = dim if cols % dim == 0 else 1
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).to(device)
    v = t.reshape(-1, d)
    residual = v.clone()
    codebooks, indices = ([], [])
    for _ in range(stages):
        cb = gpu_kmeans(residual, k, iters=iters)
        idx = _assign(residual, cb)
        residual = residual - cb[idx]
        codebooks.append(cb)
        indices.append(idx)
    bpw = stages * math.log2(k) / d
    return RvqCode(torch.stack(codebooks), torch.stack(indices, 1), d, rows, cols, bpw)

def rvq_decode(code: RvqCode) -> torch.Tensor:
    """Sum the M codebook rows per vector -> reconstructed weight [rows, cols]."""
    recon = torch.zeros(code.indices.shape[0], code.dim, device=code.codebooks.device)
    for m in range(code.codebooks.shape[0]):
        recon += code.codebooks[m][code.indices[:, m]]
    return recon.reshape(code.rows, code.cols)

def pack_bytes(code: RvqCode) -> int:
    """Deployable artifact size: packed indices (ceil(log2 K) bits each) + fp16 codebooks."""
    m, k, d = code.codebooks.shape
    n = code.indices.shape[0]
    index_bits = n * m * math.ceil(math.log2(k))
    codebook_bits = m * k * d * 16
    return math.ceil(index_bits / 8) + math.ceil(codebook_bits / 8)

def pick_config(cols: int, target_bpw: float, *, k: int=256, max_dim: int=64) -> tuple[int, int, int]:
    """Choose (dim, k, stages) so stages*log2(k)/dim ~= target_bpw, dim | cols. Prefer >=2 RVQ
    stages when the rate allows (more expressive than single-stage at equal bpw)."""
    bits_per_vec = math.log2(k)
    stages = max(1, min(4, round(target_bpw * max_dim / bits_per_vec)))
    dim = max(1, round(stages * bits_per_vec / target_bpw))
    dim = max((d for d in range(1, min(cols, max_dim) + 1) if cols % d == 0 and d <= dim), default=1)
    stages = max(1, round(target_bpw * dim / bits_per_vec))
    return (dim, k, stages)

def pack_weight(w: np.ndarray, target_bpw: float, *, k: int=256, iters: int=12) -> dict[str, Any]:
    """Pack a real weight at a target sub-bit rate on the GPU and measure the roundtrip."""
    device = _device()
    dim, k, stages = pick_config(w.shape[1], target_bpw, k=k)
    t0 = time.time()
    code = rvq_encode(w, dim=dim, k=k, stages=stages, iters=iters, device=device)
    recon = rvq_decode(code)
    w_t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).to(device)
    rel_err = float((torch.linalg.norm(w_t - recon) / (torch.linalg.norm(w_t) + 1e-12)).cpu())
    if device.type == 'mps':
        torch.mps.synchronize()
    packed = pack_bytes(code)
    whole_bpw = packed * 8 / w.size
    return {'target_bpw': target_bpw, 'achieved_bpw': round(code.bpw, 4), 'whole_artifact_bpw': round(whole_bpw, 4), 'rel_error': round(rel_err, 5), 'config': {'dim': dim, 'k': k, 'stages': stages}, 'packed_bytes': packed, 'device': device.type, 'seconds': round(time.time() - t0, 2), 'verdict': 'survives' if rel_err < 0.15 else 'degraded' if rel_err < 0.4 else 'collapse'}

def expert_genome_probe(weights: list[np.ndarray], *, dim: int, k: int, stages: int, iters: int=12) -> dict[str, Any]:
    """Aggressive sub-bit test: fit ONE shared RVQ codebook across N experts (the 'expert
    genome'), then encode each expert with only indices into it. The codebook amortizes across
    all N experts, so the whole-artifact bpw approaches the index-only rate stages*log2(k)/dim
    (deep sub-bit). Measures whether the shared genome reconstructs every expert -> whether the
    120B experts are redundant enough for sub-bit. Runs on the GPU."""
    device = _device()
    rows, cols = weights[0].shape
    d = dim if cols % dim == 0 else 1
    vs = [torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).to(device).reshape(-1, d) for w in weights]
    pool = torch.cat(vs, 0)
    residual = pool.clone()
    codebooks = []
    for _ in range(stages):
        cb = gpu_kmeans(residual, k, iters=iters)
        residual = residual - cb[_assign(residual, cb)]
        codebooks.append(cb)
    cbs = torch.stack(codebooks)
    errs = []
    n_index_bits = 0
    for v in vs:
        recon = torch.zeros_like(v)
        res = v.clone()
        for m in range(stages):
            idx = _assign(res, cbs[m])
            recon += cbs[m][idx]
            res = res - cbs[m][idx]
            n_index_bits += v.shape[0] * math.ceil(math.log2(k))
        errs.append(float((torch.linalg.norm(v - recon) / (torch.linalg.norm(v) + 1e-12)).cpu()))
    if device.type == 'mps':
        torch.mps.synchronize()
    total_weights = sum((w.size for w in weights))
    codebook_bits = stages * k * d * 16
    whole_bpw = (n_index_bits + codebook_bits) / total_weights
    mean_err = float(np.mean(errs))
    return {'n_experts': len(weights), 'shared_codebook': True, 'index_only_bpw': round(stages * math.log2(k) / d, 4), 'whole_artifact_bpw': round(whole_bpw, 4), 'mean_rel_error': round(mean_err, 5), 'per_expert_rel_error': [round(e, 4) for e in errs], 'config': {'dim': d, 'k': k, 'stages': stages}, 'device': device.type, 'verdict': 'survives' if mean_err < 0.15 else 'degraded' if mean_err < 0.4 else 'collapse'}
