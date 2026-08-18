#!/usr/bin/env python3
"""Qwen3.8 sub-1-bit region sweep — CPU, real BF16, real 256-token capture.

Apply one cheap representation to one region, leave everything else exact,
score layer-output and residual-stream error, rank bits-saved / final error.

No GPU. No generate. No pack. No resident touch. Peak RSS target < 15 GB.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import struct
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import numpy as np
import torch

torch.set_num_threads(8)
torch.set_num_interop_threads(1)

MODEL_DIR = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
)
CAPTURE_DIR = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
)

N_LAYERS = 64
N_TOKENS = 256
HIDDEN = 5120
INTERMEDIATE = 17408
VOCAB = 248320
FIT_N = 192
HOLD_N = 64
G_BIN = 128
G_UNI = 64
EPS = 1e-12

N_SRC = 26_895_998_464
G0_BPW = 4.252735126866492
G0_BYTES = 14_297_694_680
Q4_BODY = 4.25  # 4 + 16/64
SMALL_ELEMS = 2_645_504
SMALL_BYTES_F32 = 10_584_840

DN_Q = 16 * 128
DN_K = 16 * 128
DN_V = 48 * 128
GQA_H = 24
GQA_KV = 4
GQA_D = 256
GQA_ROT = 64
ROPE_THETA = 10_000_000.0
RMS_EPS = 1e-6


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 3)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} rss={rss_gb():.2f}GiB] {msg}", flush=True)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.reshape(-1).to(torch.float64)
    y = b.reshape(-1).to(torch.float64)
    num = float(torch.dot(x, y))
    den = float(torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y))
    if den <= EPS:
        return 1.0 if abs(num) <= EPS else 0.0
    return num / den


def rel_l2(ref: torch.Tensor, hat: torch.Tensor) -> float:
    r = ref.reshape(-1).to(torch.float64)
    h = hat.reshape(-1).to(torch.float64)
    n = float(torch.linalg.vector_norm(r))
    if n <= EPS:
        return 0.0
    return float(torch.linalg.vector_norm(r - h) / n)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def mean_row_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.to(torch.float64), dim=-1).mean())


# ---------------------------------------------------------------------------
# safetensors
# ---------------------------------------------------------------------------

_HEADER_CACHE: dict[Path, tuple[int, dict[str, Any]]] = {}
_WEIGHT_MAP: dict[str, str] | None = None


def load_weight_map() -> dict[str, str]:
    global _WEIGHT_MAP
    if _WEIGHT_MAP is None:
        idx = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
        _WEIGHT_MAP = dict(idx["weight_map"])
    return _WEIGHT_MAP


def _shard_header(shard: Path) -> tuple[int, dict[str, Any]]:
    if shard not in _HEADER_CACHE:
        with shard.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        _HEADER_CACHE[shard] = (n, hdr)
    return _HEADER_CACHE[shard]


def tensor_info(name: str) -> tuple[Path, int, dict[str, Any]]:
    shard = MODEL_DIR / load_weight_map()[name]
    hn, hdr = _shard_header(shard)
    return shard, hn, hdr[name]


def load_tensor(name: str) -> torch.Tensor:
    shard, hn, info = tensor_info(name)
    dtype = info.get("dtype", "BF16")
    shape = tuple(int(x) for x in info["shape"])
    lo, hi = info["data_offsets"]
    with shard.open("rb") as fh:
        fh.seek(8 + hn + lo)
        raw = fh.read(hi - lo)
    if dtype not in ("BF16", "BFLOAT16"):
        raise RuntimeError(f"{name} dtype {dtype}")
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape).copy()
    return torch.from_numpy(np.ascontiguousarray(f32))


def load_rows(name: str, rows: np.ndarray) -> torch.Tensor:
    """Load selected rows of a BF16 [R, C] tensor. Groups do not cross rows
    on this model (in_dim % 128 == 0), so row-local codecs match flatten."""
    shard, hn, info = tensor_info(name)
    shape = tuple(int(x) for x in info["shape"])
    rows_n, cols = shape
    lo, _ = info["data_offsets"]
    rows = np.asarray(rows, dtype=np.int64)
    out = np.empty((rows.size, cols), dtype=np.float32)
    with shard.open("rb") as fh:
        base = 8 + hn + lo
        for i, r in enumerate(rows):
            if r < 0 or r >= rows_n:
                raise RuntimeError(f"row {r} out of {rows_n} for {name}")
            fh.seek(base + int(r) * cols * 2)
            raw = fh.read(cols * 2)
            u16 = np.frombuffer(raw, dtype=np.uint16)
            out[i] = (u16.astype(np.uint32) << 16).view(np.float32)
    return torch.from_numpy(np.ascontiguousarray(out))


def iter_row_tiles(name: str, tile: int = 4096):
    shard, hn, info = tensor_info(name)
    shape = tuple(int(x) for x in info["shape"])
    rows_n, cols = shape
    lo, _ = info["data_offsets"]
    with shard.open("rb") as fh:
        base = 8 + hn + lo
        for r0 in range(0, rows_n, tile):
            r1 = min(rows_n, r0 + tile)
            fh.seek(base + r0 * cols * 2)
            raw = fh.read((r1 - r0) * cols * 2)
            u16 = np.frombuffer(raw, dtype=np.uint16)
            f32 = (u16.astype(np.uint32) << 16).view(np.float32).reshape(r1 - r0, cols).copy()
            yield r0, r1, torch.from_numpy(np.ascontiguousarray(f32))


def load_hidden(layer: int) -> torch.Tensor:
    path = CAPTURE_DIR / "hidden" / f"L{layer:02d}.f32"
    x = np.fromfile(path, dtype=np.float32)
    if x.size != N_TOKENS * HIDDEN:
        raise RuntimeError(f"{path} size {x.size}")
    return torch.from_numpy(np.ascontiguousarray(x.reshape(N_TOKENS, HIDDEN)))


# ---------------------------------------------------------------------------
# codecs
# ---------------------------------------------------------------------------

def _pad_groups(flat: torch.Tensor, gsz: int) -> tuple[torch.Tensor, int]:
    n = int(flat.numel())
    groups = math.ceil(n / gsz)
    pad = groups * gsz - n
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    return flat.reshape(groups, gsz), n


def binary_recon(W: torch.Tensor) -> tuple[torch.Tensor, int]:
    flat = W.reshape(-1).to(torch.float32)
    g, n = _pad_groups(flat, G_BIN)
    scales = g.abs().mean(dim=1).to(torch.float16).to(torch.float32)
    signs = torch.where(g >= 0, 1.0, -1.0)
    recon = (signs * scales[:, None]).reshape(-1)[:n].reshape(W.shape)
    nbytes = 261 + n // 8 + (n // G_BIN) * 2
    return recon, nbytes


def ternary_recon(W: torch.Tensor, thr_mul: float = 0.7) -> tuple[torch.Tensor, int]:
    flat = W.reshape(-1).to(torch.float32)
    g, n = _pad_groups(flat, G_BIN)
    base = g.abs().mean(dim=1)
    thr = (base * thr_mul).to(torch.float16).to(torch.float32)
    active = g.abs() >= thr[:, None]
    selected = torch.where(active, g.abs(), torch.zeros_like(g))
    count = active.sum(dim=1).clamp(min=1).to(torch.float32)
    scales = (selected.sum(dim=1) / count).to(torch.float16).to(torch.float32)
    recon = torch.where(active, torch.where(g >= 0, 1.0, -1.0) * scales[:, None], torch.zeros_like(g))
    recon = recon.reshape(-1)[:n].reshape(W.shape)
    groups = g.shape[0]
    nbytes = 32 + groups * 2 + groups * 2 + (groups * G_BIN * 2 + 7) // 8
    return recon, nbytes


def uniform_recon(W: torch.Tensor, bits: int = 4, gsz: int = G_UNI) -> tuple[torch.Tensor, int]:
    flat = W.reshape(-1).to(torch.float32)
    g, n = _pad_groups(flat, gsz)
    bound = (1 << (bits - 1)) - 1
    scales = (g.abs().amax(dim=1) / max(bound, 1)).to(torch.float16).to(torch.float32)
    den = torch.where(scales > 0, scales, torch.ones_like(scales))
    q = torch.round(g / den[:, None]).clamp(-bound, bound)
    recon = (q * scales[:, None]).reshape(-1)[:n].reshape(W.shape)
    groups = g.shape[0]
    header = 32 + 8  # rank-2-ish
    nbytes = header + groups * 2 + (n * bits + 7) // 8
    return recon, nbytes


def zps_recon(W: torch.Tensor, m: int = 8, gsz: int = G_BIN) -> tuple[torch.Tensor, int]:
    """Structured zero-plus-sign: keep 1 of every m consecutive weights,
    one fp16 scale per gsz, signs of survivors. Groups do not cross rows."""
    if gsz % m != 0:
        raise ValueError("gsz must divide by m")
    flat = W.reshape(-1).to(torch.float32)
    g, n = _pad_groups(flat, gsz)
    ng = g.shape[0]
    blocks = g.reshape(ng, gsz // m, m)
    idx = blocks.abs().argmax(dim=-1, keepdim=True)
    kept = torch.take_along_dim(blocks, idx, dim=-1).squeeze(-1)
    scales = kept.abs().mean(dim=1).to(torch.float16).to(torch.float32)
    signs = torch.where(kept >= 0, 1.0, -1.0)
    recon_blocks = torch.zeros_like(blocks)
    recon_blocks.scatter_(-1, idx, (signs * scales[:, None])[..., None])
    recon = recon_blocks.reshape(-1)[:n].reshape(W.shape)
    pos_bits = int(math.ceil(math.log2(m)))
    code_bits_per = pos_bits + 1
    n_slots = (n + m - 1) // m
    nbytes = 32 + (n_slots * code_bits_per + 7) // 8 + (n // gsz) * 2
    return recon, nbytes


def template_fit(W: torch.Tensor, k: int = 16, n_sample: int = 512, iters: int = 6, seed: int = 0) -> torch.Tensor:
    rows, dim = W.shape
    k_use = min(k, rows)
    rng = np.random.default_rng(seed)
    if rows > n_sample:
        pick = rng.choice(rows, size=n_sample, replace=False)
        sample = W[torch.from_numpy(pick.astype(np.int64))]
    else:
        sample = W
    # init: random sample rows, normalized
    init_idx = rng.choice(sample.shape[0], size=k_use, replace=False)
    cents = sample[torch.from_numpy(init_idx.astype(np.int64))].clone()
    # scale-invariant k-means via cosine
    for _ in range(iters):
        cn = cents / cents.norm(dim=1, keepdim=True).clamp_min(1e-8)
        sn = sample / sample.norm(dim=1, keepdim=True).clamp_min(1e-8)
        assign = (sn @ cn.T).argmax(dim=1)
        for j in range(k_use):
            mask = assign == j
            if int(mask.sum()) == 0:
                # reinit on a random sample row
                cents[j] = sample[int(rng.integers(0, sample.shape[0]))]
            else:
                # mean of unit rows, then unnormalize later via scale
                cents[j] = sn[mask].mean(dim=0)
        nz = cents.norm(dim=1, keepdim=True).clamp_min(1e-8)
        cents = cents / nz
    return cents


def template_recon(W: torch.Tensor, cents: torch.Tensor) -> tuple[torch.Tensor, int, dict[str, Any]]:
    # row ≈ scale * template; scale = (row·c)/(c·c); c already unit so scale = row·c
    cn = cents / cents.norm(dim=1, keepdim=True).clamp_min(1e-8)
    # assign by cosine = row_unit · c
    Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)
    scores = Wn @ cn.T
    assign = scores.argmax(dim=1)
    chosen = cn[assign]
    # signed scale in original units
    scale = (W * chosen).sum(dim=1)
    recon = chosen * scale[:, None]
    rows, dim = W.shape
    k = int(cents.shape[0])
    assign_bits = max(1, int(math.ceil(math.log2(max(k, 2)))))
    nbytes = 32 + k * dim * 2 + rows * 2 + (rows * assign_bits + 7) // 8
    return recon, nbytes, {"k": k, "assign_bits": assign_bits}


def template_bytes_amortized(out_rows: int, in_dim: int, k: int, n_share: int) -> int:
    assign_bits = max(1, int(math.ceil(math.log2(max(k, 2)))))
    codebook = (k * in_dim * 2 + n_share - 1) // n_share
    return 32 + codebook + out_rows * 2 + (out_rows * assign_bits + 7) // 8


# ---------------------------------------------------------------------------
# mixers
# ---------------------------------------------------------------------------

def rmsnorm_last(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # x: [..., D], w: [D]
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + RMS_EPS)
    return x / rms * w


def build_rope(max_pos: int, rot: int = GQA_ROT, theta: float = ROPE_THETA) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float32) / rot))
    pos = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(pos, inv)  # [P, rot/2]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    # x: [T, H, D], rope first rot dims as pairs
    rot = cos.shape[-1] * 2
    x1 = x[..., :rot]
    rest = x[..., rot:]
    c = cos[positions][:, None, :].to(x.dtype)
    s = sin[positions][:, None, :].to(x.dtype)
    even = x1[..., 0::2]
    odd = x1[..., 1::2]
    rot_even = even * c - odd * s
    rot_odd = even * s + odd * c
    out = torch.empty_like(x1)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return torch.cat([out, rest], dim=-1)


def gqa_mixer(
    q_full: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    seqs: list[tuple[int, int]],
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    T = q_full.shape[0]
    q = q_full[:, : GQA_H * GQA_D].reshape(T, GQA_H, GQA_D)
    gate = q_full[:, GQA_H * GQA_D :].reshape(T, GQA_H, GQA_D)
    k = k.reshape(T, GQA_KV, GQA_D)
    v = v.reshape(T, GQA_KV, GQA_D)
    q = rmsnorm_last(q, q_norm)
    k = rmsnorm_last(k, k_norm)
    pos = torch.empty(T, dtype=torch.long)
    for s0, s1 in seqs:
        pos[s0:s1] = torch.arange(s1 - s0)
    q = apply_rope(q, cos, sin, pos)
    k = apply_rope(k, cos, sin, pos)
    rep = GQA_H // GQA_KV
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    out = torch.empty(T, GQA_H, GQA_D, dtype=q.dtype)
    scale = GQA_D ** -0.5
    for s0, s1 in seqs:
        qq = q[s0:s1]
        kk = k[s0:s1]
        vv = v[s0:s1]
        L = s1 - s0
        att = torch.einsum("thd,shd->hts", qq, kk) * scale
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        att = att.masked_fill(mask[None, :, :], float("-inf"))
        att = torch.softmax(att, dim=-1)
        ctx = torch.einsum("hts,shd->thd", att, vv)
        out[s0:s1] = ctx * torch.sigmoid(gate[s0:s1])
    return out.reshape(T, GQA_H * GQA_D)


def dn_linear_mixer(qkv: torch.Tensor, z: torch.Tensor, seqs: list[tuple[int, int]]) -> torch.Tensor:
    """Causal linear attention over (q,k,v) then * silu(z).

    PROXY: not full gated-delta (no conv1d, A_log decay, beta, dt_bias).
    Uses Q, K and V so in_proj_qkv error is not V-only-blind.
    """
    T = qkv.shape[0]
    q = qkv[:, :DN_Q].reshape(T, 16, 128)
    k = qkv[:, DN_Q : DN_Q + DN_K].reshape(T, 16, 128)
    v = qkv[:, DN_Q + DN_K :].reshape(T, 16, 384)
    rec = torch.empty(T, 16, 384, dtype=qkv.dtype)
    for s0, s1 in seqs:
        qq = q[s0:s1]
        kk = k[s0:s1]
        vv = v[s0:s1]
        kv = kk.unsqueeze(-1) * vv.unsqueeze(-2)  # L,16,128,384
        S = torch.cumsum(kv, dim=0)
        rec[s0:s1] = (qq.unsqueeze(-2) @ S).squeeze(-2)
    return rec.reshape(T, DN_V) * silu(z)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def slice_hold(x: torch.Tensor) -> torch.Tensor:
    return x[FIT_N:]


def score_pair(
    y_ref: torch.Tensor,
    y_hat: torch.Tensor,
    h: torch.Tensor,
    remain_amp: float,
    w: torch.Tensor | None = None,
    w_hat: torch.Tensor | None = None,
) -> dict[str, float]:
    y_r = slice_hold(y_ref)
    y_h = slice_hold(y_hat)
    h_h = slice_hold(h)
    d = y_r - y_h
    y_norm = float(torch.linalg.vector_norm(y_r.to(torch.float64)))
    h_norm = float(torch.linalg.vector_norm(h_h.to(torch.float64)))
    d_norm = float(torch.linalg.vector_norm(d.to(torch.float64)))
    rel_y = d_norm / y_norm if y_norm > EPS else 0.0
    rel_h = d_norm / h_norm if h_norm > EPS else 0.0
    final = rel_h * remain_amp
    out = {
        "hold_cosine": cosine(y_r, y_h),
        "hold_rel_l2_y": rel_y,
        "hold_rel_l2_h": rel_h,
        "hold_d_norm": d_norm,
        "hold_y_norm": y_norm,
        "hold_h_norm": h_norm,
        "remain_amp": remain_amp,
        "final_stream_rel": final,
        "all_rel_l2_y": rel_l2(y_ref, y_hat),
        "all_cosine": cosine(y_ref, y_hat),
    }
    if w is not None and w_hat is not None:
        out["weight_cosine"] = cosine(w, w_hat)
        out["weight_rel_l2"] = rel_l2(w, w_hat)
    return out


def bits_saved(elements: int, codec_bpw: float, baseline: float = Q4_BODY) -> float:
    return float(elements) * (baseline - codec_bpw)


# ---------------------------------------------------------------------------
# capture / amp
# ---------------------------------------------------------------------------

def load_capture_meta() -> dict[str, Any]:
    return json.loads((CAPTURE_DIR / "capture-result.json").read_text())


def token_ids_and_seqs(meta: dict[str, Any]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    ids: list[int] = []
    seqs: list[tuple[int, int]] = []
    off = 0
    for p in meta["prompts"]:
        n = int(p["n_tokens"])
        ids.extend(p["ids"])
        seqs.append((off, off + n))
        off += n
    if len(ids) != N_TOKENS:
        raise RuntimeError(f"token count {len(ids)} != {N_TOKENS}")
    return np.asarray(ids, dtype=np.int64), seqs


def measure_amp() -> dict[str, Any]:
    norms = []
    rms = []
    for i in range(N_LAYERS):
        h = load_hidden(i).numpy()
        nt = np.linalg.norm(h, axis=1)
        norms.append(nt)
        rms.append(float(np.sqrt(np.mean(h.astype(np.float64) ** 2))))
    nt = np.stack(norms)
    A = [float(np.mean(nt[l + 1] / np.maximum(nt[l], 1e-12))) for l in range(N_LAYERS - 1)]
    remain = [float(np.mean(nt[N_LAYERS - 1] / np.maximum(nt[l], 1e-12))) for l in range(N_LAYERS)]
    return {
        "rms": rms,
        "token_norm_mean": [float(nt[l].mean()) for l in range(N_LAYERS)],
        "A_token_mean": A,
        "remain_token_mean": remain,
        "product_A": float(np.prod(A)),
        "H63_over_H0": float(np.mean(nt[63] / np.maximum(nt[0], 1e-12))),
    }


# ---------------------------------------------------------------------------
# layer sweep
# ---------------------------------------------------------------------------

CodecFn = Any


def codec_table(W: torch.Tensor, donors: dict[str, torch.Tensor], class_key: str, n_share: int) -> list[tuple[str, torch.Tensor, int, str]]:
    """Return list of (name, recon, nbytes, family)."""
    out: list[tuple[str, torch.Tensor, int, str]] = []
    r, b = binary_recon(W)
    out.append(("binary_g128", r, b, "binary"))
    r, b = ternary_recon(W)
    out.append(("ternary_t0.7_g128", r, b, "ternary"))
    r, b = zps_recon(W, m=8)
    out.append(("zps_1of8_g128", r, b, "zero_plus_sign"))
    r, b = zps_recon(W, m=4)
    out.append(("zps_1of4_g128", r, b, "zero_plus_sign"))
    cents = template_fit(W, k=16)
    r, b, _ = template_recon(W, cents)
    out.append(("tmpl_k16", r, b, "shared_template"))
    donor = donors.get(class_key)
    if donor is not None:
        r, b, meta = template_recon(W, donor)
        b_am = template_bytes_amortized(W.shape[0], W.shape[1], int(donor.shape[0]), n_share)
        out.append(("tmpl_l0_shared", r, b_am, "shared_template"))
    r, b = uniform_recon(W, bits=4)
    out.append(("q4_g64", r, b, "incumbent"))
    return out


def record_base(
    layer: int,
    mixer: str,
    cls: str,
    name: str,
    codec: str,
    family: str,
    elements: int,
    nbytes: int,
    residual_kind: str,
    site: str,
) -> dict[str, Any]:
    bpw = 8.0 * nbytes / elements if elements else 0.0
    return {
        "layer": layer,
        "mixer": mixer,
        "class": cls,
        "name": name,
        "codec": codec,
        "family": family,
        "elements": elements,
        "payload_bytes": nbytes,
        "physical_bpw": bpw,
        "bits_saved_vs_q4body": bits_saved(elements, bpw),
        "residual_kind": residual_kind,
        "site": site,
    }


def finish_row(row: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    row.update(metrics)
    err = max(float(row.get("final_stream_rel", 1.0)), 1e-15)
    row["bits_per_final_err"] = float(row["bits_saved_vs_q4body"]) / err
    err_y = max(float(row.get("hold_rel_l2_y", 1.0)), 1e-15)
    row["bits_per_rel_y"] = float(row["bits_saved_vs_q4body"]) / err_y
    return row


def sweep_mlp(
    layer: int,
    H: torch.Tensor,
    remain: float,
    donors: dict[str, torch.Tensor],
    n_share_mlp: int,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    pref = f"language_model.model.layers.{layer}.mlp."
    Wg = load_tensor(pref + "gate_proj.weight")
    Wu = load_tensor(pref + "up_proj.weight")
    Wd = load_tensor(pref + "down_proj.weight")
    g = H @ Wg.T
    u = H @ Wu.T
    s = silu(g) * u
    y = s @ Wd.T
    new_donors: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []

    def run_one(cls: str, W: torch.Tensor, apply_hat) -> None:
        key = f"mlp.{cls}"
        if key not in donors:
            new_donors[key] = template_fit(W, k=16)
        packed = codec_table(W, {**donors, **new_donors}, key, n_share_mlp)
        name = pref + f"{cls}_proj.weight"
        for cname, what, nbytes, fam in packed:
            y_hat, y_layer_ref, y_layer_hat = apply_hat(what)
            rec = record_base(
                layer, "mlp", cls, name, cname, fam, int(W.numel()), nbytes,
                residual_kind="mlp_residual_exact",
                site="post_input_norm_hidden",
            )
            m = score_pair(y, y_hat, H, remain, W, what)
            m["layer_out_hold_cosine"] = cosine(slice_hold(y_layer_ref), slice_hold(y_layer_hat))
            m["layer_out_hold_rel_l2"] = rel_l2(slice_hold(y_layer_ref), slice_hold(y_layer_hat))
            rows.append(finish_row(rec, m))
            del what, y_hat
        del packed

    def apply_gate(what):
        g_hat = H @ what.T
        s_hat = silu(g_hat) * u
        return s_hat @ Wd.T, g, g_hat

    def apply_up(what):
        u_hat = H @ what.T
        s_hat = silu(g) * u_hat
        return s_hat @ Wd.T, u, u_hat

    def apply_down(what):
        y_hat = s @ what.T
        return y_hat, y, y_hat

    run_one("gate", Wg, apply_gate)
    run_one("up", Wu, apply_up)
    run_one("down", Wd, apply_down)
    del Wg, Wu, Wd, g, u, s, y
    return rows, new_donors


def sweep_gqa(
    layer: int,
    H: torch.Tensor,
    remain: float,
    donors: dict[str, torch.Tensor],
    seqs: list[tuple[int, int]],
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_share: int,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    pref = f"language_model.model.layers.{layer}.self_attn."
    Wq = load_tensor(pref + "q_proj.weight")
    Wk = load_tensor(pref + "k_proj.weight")
    Wv = load_tensor(pref + "v_proj.weight")
    Wo = load_tensor(pref + "o_proj.weight")
    qn = load_tensor(pref + "q_norm.weight").reshape(-1)
    kn = load_tensor(pref + "k_norm.weight").reshape(-1)
    q = H @ Wq.T
    k = H @ Wk.T
    v = H @ Wv.T
    mix = gqa_mixer(q, k, v, qn, kn, seqs, cos, sin)
    y = mix @ Wo.T
    new_donors: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []

    specs = [
        ("q", Wq, pref + "q_proj.weight",
         lambda what: (lambda qh: (gqa_mixer(qh, k, v, qn, kn, seqs, cos, sin) @ Wo.T, q, qh))(H @ what.T)),
        ("k", Wk, pref + "k_proj.weight",
         lambda what: (lambda kh: (gqa_mixer(q, kh, v, qn, kn, seqs, cos, sin) @ Wo.T, k, kh))(H @ what.T)),
        ("v", Wv, pref + "v_proj.weight",
         lambda what: (lambda vh: (gqa_mixer(q, k, vh, qn, kn, seqs, cos, sin) @ Wo.T, v, vh))(H @ what.T)),
        ("o", Wo, pref + "o_proj.weight",
         lambda what: (lambda yh: (yh, y, yh))(mix @ what.T)),
    ]
    for cls, W, name, apply in specs:
        key = f"gqa.{cls}"
        if key not in donors:
            new_donors[key] = template_fit(W, k=16)
        packed = codec_table(W, {**donors, **new_donors}, key, n_share)
        for cname, what, nbytes, fam in packed:
            y_hat, y_layer_ref, y_layer_hat = apply(what)
            rec = record_base(
                layer, "gqa", cls, name, cname, fam, int(W.numel()), nbytes,
                residual_kind="gqa_residual_exact",
                site="post_input_norm_hidden" if cls != "o" else "gqa_mixer_x_computed",
            )
            m = score_pair(y, y_hat, H, remain, W, what)
            m["layer_out_hold_cosine"] = cosine(slice_hold(y_layer_ref), slice_hold(y_layer_hat))
            m["layer_out_hold_rel_l2"] = rel_l2(slice_hold(y_layer_ref), slice_hold(y_layer_hat))
            rows.append(finish_row(rec, m))
            del what, y_hat
        del packed
    del Wq, Wk, Wv, Wo, q, k, v, mix, y
    return rows, new_donors


def sweep_dn(
    layer: int,
    H: torch.Tensor,
    remain: float,
    donors: dict[str, torch.Tensor],
    seqs: list[tuple[int, int]],
    n_share: int,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    pref = f"language_model.model.layers.{layer}.linear_attn."
    Wqkv = load_tensor(pref + "in_proj_qkv.weight")
    Wz = load_tensor(pref + "in_proj_z.weight")
    Wa = load_tensor(pref + "in_proj_a.weight")
    Wb = load_tensor(pref + "in_proj_b.weight")
    Wo = load_tensor(pref + "out_proj.weight")
    qkv = H @ Wqkv.T
    z = H @ Wz.T
    a = H @ Wa.T
    b = H @ Wb.T
    mix = dn_linear_mixer(qkv, z, seqs)
    y = mix @ Wo.T
    new_donors: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []

    def do(cls, W, name, apply, kind, site):
        key = f"dn.{cls}"
        if key not in donors:
            new_donors[key] = template_fit(W, k=min(16, W.shape[0]))
        packed = codec_table(W, {**donors, **new_donors}, key, n_share)
        for cname, what, nbytes, fam in packed:
            y_hat, y_layer_ref, y_layer_hat = apply(what)
            rec = record_base(
                layer, "deltanet", cls, name, cname, fam, int(W.numel()), nbytes,
                residual_kind=kind, site=site,
            )
            # residual target is y for writers/proxied; for a/b use their own layer out vs H
            if cls in ("a", "b"):
                m = score_pair(y_layer_ref, y_layer_hat, H, remain, W, what)
            else:
                m = score_pair(y, y_hat, H, remain, W, what)
            m["layer_out_hold_cosine"] = cosine(slice_hold(y_layer_ref), slice_hold(y_layer_hat))
            m["layer_out_hold_rel_l2"] = rel_l2(slice_hold(y_layer_ref), slice_hold(y_layer_hat))
            rows.append(finish_row(rec, m))
            del what, y_hat
        del packed

    do("qkv", Wqkv, pref + "in_proj_qkv.weight",
       lambda what: (lambda qh: (dn_linear_mixer(qh, z, seqs) @ Wo.T, qkv, qh))(H @ what.T),
       "dn_linear_attn_proxy", "post_input_norm_hidden")
    do("z", Wz, pref + "in_proj_z.weight",
       lambda what: (lambda zh: (dn_linear_mixer(qkv, zh, seqs) @ Wo.T, z, zh))(H @ what.T),
       "dn_linear_attn_proxy", "post_input_norm_hidden")
    do("a", Wa, pref + "in_proj_a.weight",
       lambda what: (lambda ah: (ah, a, ah))(H @ what.T),
       "unmeasured_modulator_layer_out", "post_input_norm_hidden")
    do("b", Wb, pref + "in_proj_b.weight",
       lambda what: (lambda bh: (bh, b, bh))(H @ what.T),
       "unmeasured_modulator_layer_out", "post_input_norm_hidden")
    do("out", Wo, pref + "out_proj.weight",
       lambda what: (lambda yh: (yh, y, yh))(mix @ what.T),
       "dn_linear_attn_proxy", "dn_mixer_x_linear_proxy")
    del Wqkv, Wz, Wa, Wb, Wo, qkv, z, a, b, mix, y
    return rows, new_donors


def sweep_embed(
    ids: np.ndarray,
    H0: torch.Tensor,
    remain0: float,
    donors: dict[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    name = "language_model.model.embed_tokens.weight"
    used = np.unique(ids)
    W_used = load_rows(name, used)
    # map token id -> row in W_used
    id_to = {int(t): i for i, t in enumerate(used)}
    gather = torch.tensor([id_to[int(t)] for t in ids], dtype=torch.long)
    E = W_used[gather]  # [256, 5120] exact embed rows
    new_donors: dict[str, torch.Tensor] = {}
    key = "embed"
    # fit templates on a broader sample of embed rows
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(VOCAB, size=2048, replace=False)
    W_sample = load_rows(name, sample_idx)
    cents = template_fit(W_sample, k=16)
    new_donors[key] = cents
    del W_sample

    rows: list[dict[str, Any]] = []
    # encode on used rows only — valid because groups stay inside rows
    packed = []
    packed.append(("binary_g128", *binary_recon(W_used), "binary"))
    packed.append(("ternary_t0.7_g128", *ternary_recon(W_used), "ternary"))
    packed.append(("zps_1of8_g128", *zps_recon(W_used, 8), "zero_plus_sign"))
    packed.append(("zps_1of4_g128", *zps_recon(W_used, 4), "zero_plus_sign"))
    r, b, _ = template_recon(W_used, cents)
    packed.append(("tmpl_k16", r, b, "shared_template"))
    r, b = uniform_recon(W_used, 4)
    packed.append(("q4_g64", r, b, "incumbent"))
    elements = VOCAB * HIDDEN
    for cname, what, nbytes_used, fam in packed:
        # scale payload from used rows to full table (row-independent codec)
        nbytes = int(round(nbytes_used * (elements / max(W_used.numel(), 1))))
        if cname == "tmpl_k16":
            nbytes = 32 + 16 * HIDDEN * 2 + VOCAB * 2 + (VOCAB * 4 + 7) // 8
        E_hat = what[gather]
        rec = record_base( -1, "table", "embed", name, cname, fam, elements, nbytes,
                          residual_kind="embed_lookup_exact_used_rows",
                          site="token_ids_from_capture")
        m = score_pair(E, E_hat, H0, remain0, W_used, what)
        m["layer_out_hold_cosine"] = m["hold_cosine"]
        m["layer_out_hold_rel_l2"] = m["hold_rel_l2_y"]
        m["used_rows"] = int(used.size)
        rows.append(finish_row(rec, m))
    return rows, new_donors


def sweep_lm_head(
    H63: torch.Tensor,
    donors: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    name = "language_model.lm_head.weight"
    # tiled Y_ref and Y_hat; do not materialise 248320×5120 f32
    tiles: list[tuple[int, int, torch.Tensor]] = []
    y_ref_parts = []
    for r0, r1, Wt in iter_row_tiles(name, tile=4096):
        tiles.append((r0, r1, Wt))
        y_ref_parts.append(H63 @ Wt.T)
    y_ref = torch.cat(y_ref_parts, dim=1)
    del y_ref_parts

    # donor: embed centroids if present, else fit on first 2048 rows
    embed_cents = donors.get("embed")
    first_rows = torch.cat([t[2][: min(256, t[2].shape[0])] for t in tiles[:8]], dim=0)
    local_cents = template_fit(first_rows, k=16)
    del first_rows

    codec_names = [
        ("binary_g128", "binary", lambda W: binary_recon(W)),
        ("ternary_t0.7_g128", "ternary", lambda W: ternary_recon(W)),
        ("zps_1of8_g128", "zero_plus_sign", lambda W: zps_recon(W, 8)),
        ("zps_1of4_g128", "zero_plus_sign", lambda W: zps_recon(W, 4)),
        ("tmpl_k16", "shared_template", lambda W: template_recon(W, local_cents)[:2]),
        ("q4_g64", "incumbent", lambda W: uniform_recon(W, 4)),
    ]
    if embed_cents is not None:
        codec_names.append(
            ("tmpl_embed_shared", "shared_template", lambda W: template_recon(W, embed_cents)[:2])
        )

    rows: list[dict[str, Any]] = []
    elements = VOCAB * HIDDEN
    for cname, fam, fn in codec_names:
        parts = []
        nbytes = 0
        wcos_num = 0.0
        wcos_den_a = 0.0
        wcos_den_b = 0.0
        werr = 0.0
        wref = 0.0
        for r0, r1, Wt in tiles:
            what, nb = fn(Wt)
            parts.append(H63 @ what.T)
            nbytes += nb
            # accumulate weight metrics in float64 chunks
            a = Wt.reshape(-1).to(torch.float64)
            b = what.reshape(-1).to(torch.float64)
            wcos_num += float(torch.dot(a, b))
            wcos_den_a += float(torch.dot(a, a))
            wcos_den_b += float(torch.dot(b, b))
            werr += float(torch.dot(a - b, a - b))
            wref += float(torch.dot(a, a))
            del what
        y_hat = torch.cat(parts, dim=1)
        rec = record_base(
            64, "table", "lm_head", name, cname, fam, elements, nbytes,
            residual_kind="lm_head_logits_not_residual",
            site="L63_post_input_norm_NOT_confirmed_final_norm",
        )
        m = score_pair(y_ref, y_hat, H63, 1.0)
        den = math.sqrt(wcos_den_a * wcos_den_b)
        m["weight_cosine"] = (wcos_num / den) if den > EPS else 0.0
        m["weight_rel_l2"] = math.sqrt(werr) / math.sqrt(wref) if wref > EPS else 0.0
        m["layer_out_hold_cosine"] = m["hold_cosine"]
        m["layer_out_hold_rel_l2"] = m["hold_rel_l2_y"]
        # top-1 flips on hold
        ref_top = slice_hold(y_ref).argmax(dim=1)
        hat_top = slice_hold(y_hat).argmax(dim=1)
        m["hold_top1_agree"] = float((ref_top == hat_top).to(torch.float64).mean())
        rows.append(finish_row(rec, m))
        del y_hat, parts
    del tiles, y_ref
    return rows


def sweep_small_spot(H0: torch.Tensor, remain0: float) -> list[dict[str, Any]]:
    """Spot-check RMSNorm / conv1d: they are not collapse candidates."""
    rows = []
    checks = [
        (0, "input_layernorm", f"language_model.model.layers.0.input_layernorm.weight", H0),
        (31, "input_layernorm", f"language_model.model.layers.31.input_layernorm.weight", None),
        (63, "input_layernorm", f"language_model.model.layers.63.input_layernorm.weight", None),
        (0, "conv1d", "language_model.model.layers.0.linear_attn.conv1d.weight", H0),
    ]
    for layer, cls, name, H in checks:
        W = load_tensor(name)
        if H is None:
            H = load_hidden(layer)
        # treat as scale on H if 1-d matching hidden, else weight-space only
        for cname, fn, fam in (
            ("binary_g128", binary_recon, "binary"),
            ("zps_1of8_g128", lambda w: zps_recon(w, 8), "zero_plus_sign"),
            ("q4_g64", lambda w: uniform_recon(w, 4), "incumbent"),
        ):
            what, nb = fn(W)
            rec = record_base(layer, "small", cls, name, cname, fam, int(W.numel()), nb,
                              residual_kind="spot_weight_or_scale", site="spot")
            if W.ndim == 1 and W.numel() == HIDDEN:
                y_ref = H * W
                y_hat = H * what
                m = score_pair(y_ref, y_hat, H, remain0 if layer == 0 else 1.0, W, what)
            else:
                m = {
                    "hold_cosine": cosine(W, what),
                    "hold_rel_l2_y": rel_l2(W, what),
                    "hold_rel_l2_h": rel_l2(W, what),
                    "final_stream_rel": rel_l2(W, what),
                    "remain_amp": 1.0,
                    "weight_cosine": cosine(W, what),
                    "weight_rel_l2": rel_l2(W, what),
                    "layer_out_hold_cosine": cosine(W, what),
                    "layer_out_hold_rel_l2": rel_l2(W, what),
                    "all_rel_l2_y": rel_l2(W, what),
                    "all_cosine": cosine(W, what),
                    "hold_d_norm": 0.0,
                    "hold_y_norm": 0.0,
                    "hold_h_norm": 0.0,
                }
            rec["spot"] = True
            rows.append(finish_row(rec, m))
        del W
    return rows


def is_gqa(layer: int) -> bool:
    return (layer + 1) % 4 == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="all", help="all or comma list")
    ap.add_argument("--out", default="/tmp/g1_sub1bit_regions.json")
    ap.add_argument("--jsonl", default="/tmp/g1_sub1bit_regions.jsonl")
    ap.add_argument("--skip-tables", action="store_true")
    ap.add_argument("--skip-attn", action="store_true")
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--skip-small", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    meta = load_capture_meta()
    ids, seqs = token_ids_and_seqs(meta)
    amp = measure_amp()
    remain = amp["remain_token_mean"]
    log(f"capture sha256_self={meta['sha256_self']} status={meta['status']}")
    log(f"amp remain[0]={remain[0]:.6f} remain[31]={remain[31]:.6f} product_A={amp['product_A']:.6f}")

    if args.layers == "all":
        layers = list(range(N_LAYERS))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    jsonl_path = Path(args.jsonl)
    if jsonl_path.exists():
        jsonl_path.unlink()

    donors: dict[str, torch.Tensor] = {}
    all_rows: list[dict[str, Any]] = []
    cos = sin = None

    def emit(rows: list[dict[str, Any]]) -> None:
        all_rows.extend(rows)
        with jsonl_path.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    # calibration: L0 gate binary hold cosine vs 0.861852
    if (not args.skip_mlp) and (0 in layers):
        H = load_hidden(0)
        Wg = load_tensor("language_model.model.layers.0.mlp.gate_proj.weight")
        what, _ = binary_recon(Wg)
        y_ref = H[FIT_N:] @ Wg.T
        y_hat = H[FIT_N:] @ what.T
        cal = cosine(y_ref, y_hat)
        log(f"CALIBRATION L0 gate binary hold cosine={cal:.12f} target=0.861852194430")
        del Wg, what, y_ref, y_hat, H

    if not args.skip_small:
        emit(sweep_small_spot(load_hidden(0), remain[0]))
        log(f"small spot n={len(all_rows)}")

    n_mlp_share = sum(1 for L in layers)
    n_dn_share = sum(1 for L in layers if not is_gqa(L))
    n_gqa_share = sum(1 for L in layers if is_gqa(L))

    for L in layers:
        H = load_hidden(L)
        tL = time.perf_counter()
        if not args.skip_mlp:
            rows, nd = sweep_mlp(L, H, remain[L], donors, max(n_mlp_share, 1))
            donors.update(nd)
            emit(rows)
        if not args.skip_attn:
            if is_gqa(L):
                if cos is None:
                    cos, sin = build_rope(max(s1 - s0 for s0, s1 in seqs) + 8)
                rows, nd = sweep_gqa(L, H, remain[L], donors, seqs, cos, sin, max(n_gqa_share, 1))
                donors.update(nd)
                emit(rows)
            else:
                rows, nd = sweep_dn(L, H, remain[L], donors, seqs, max(n_dn_share, 1))
                donors.update(nd)
                emit(rows)
        del H
        log(f"layer {L:02d} done +{time.perf_counter()-tL:.1f}s rows={len(all_rows)}")

    if not args.skip_tables:
        H0 = load_hidden(0)
        rows, nd = sweep_embed(ids, H0, remain[0], donors)
        donors.update(nd)
        emit(rows)
        del H0
        log(f"embed done rows={len(all_rows)}")
        H63 = load_hidden(63)
        emit(sweep_lm_head(H63, donors))
        del H63
        log(f"lm_head done rows={len(all_rows)}")

    wall = time.perf_counter() - t0
    payload = {
        "schema": "hawking.g1.sub1bit_regions.v1",
        "wall_s": wall,
        "rss_max_gb": rss_gb(),
        "capture_sha256_self": meta["sha256_self"],
        "capture_status": meta["status"],
        "n_tokens": N_TOKENS,
        "fit_n": FIT_N,
        "hold_n": HOLD_N,
        "layers": layers,
        "n_rows": len(all_rows),
        "amp": amp,
        "g0_bpw": G0_BPW,
        "g0_bytes": G0_BYTES,
        "n_src": N_SRC,
        "calibration_note": "L0 gate binary hold cosine printed to stdout; expect 0.861852194430",
        "rows": all_rows,
    }
    Path(args.out).write_text(json.dumps(payload))
    log(f"wrote {args.out} rows={len(all_rows)} wall={wall:.1f}s rss={rss_gb():.2f}GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
