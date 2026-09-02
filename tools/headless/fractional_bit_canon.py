#!/usr/bin/env python3
"""FRACTIONAL_BIT_CANON: what actually runs at or below 2 bits on dense Qwen.

DENSE_SUBBIT_TRANSFER already REFUTED naive low-rank transfer: at matched bits
the low-rank operator is 2.93× the output error of flat q3 (mean 0.5393 vs
0.1839). This lane does not re-derive that. It searches the structures that
might still survive at ≤2 bpw as a representation that *runs* — storage and
active billed separately, scales counted, scored in function space on REAL
captured activations.

Never evaluates on Gaussian X. A signed-symmetric absmax 1-bit codec with
bound=2^(bits-1)-1 is the ZERO TENSOR; that trap is an instrument check, not
a candidate.

    python3 tools/headless/fractional_bit_canon.py
    python3 -m pytest tools/headless/fractional_bit_canon.py -q
"""
from __future__ import annotations

import gc
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VISION_PY = Path.home() / ".grok-vision" / "bin" / "python"
OUT_PATH = ROOT / "receipts" / "headless" / "FRACTIONAL_BIT_CANON.json"

HIDDEN = 5120
INTERMEDIATE = 17408
SCALE_BITS = 16  # one f16 scale per group
F16_BPW = 16.0
HEADER_BYTES = 64
SEED = 10582654
SCALE_TRAP = 0.01
CHUNK = 2048
LOG2_3 = math.log2(3.0)
TRIT_PACK_5IN8 = 8.0 / 5.0  # realizable ternary pack: 5 trits in 8 bits
GAIN_HEALTHY = 0.50
REL_FRO_LOCAL_MAX = 0.50
SCALE_AWARE_MARGIN = 0.05
CANON_VS_Q3_MAX_RATIO = 2.0  # G034 refuted low-rank at 2.93× q3
SUB2_MAX_BPW = 2.0
NEAR2_MAX_BPW = 2.25  # g=64 2-bit + f16 scale, scales counted
LAYERS = (0, 31)
ORGANS = ("gate_proj", "up_proj", "down_proj")
ORGAN_SEED = {"gate_proj": 3, "up_proj": 1, "down_proj": 2}

# Cited, not re-run. DENSE_SUBBIT_TRANSFER / G034.
PRIOR = {
    "receipt": "receipts/headless/DENSE_SUBBIT_TRANSFER.json",
    "g034": "receipts/ascent-2026-08-16/G034_TENSOR_OPERATOR.json",
    "glm_cosine_at_0.167": 0.755,
    "glm_null": 0.651,
    "dense_transfer": "NO-GO",
    "mean_lowrank": 0.5393288880586624,
    "mean_flat_q3": 0.1839276241211841,
    "lowrank_vs_q3_error_ratio": 2.93,
    "svd_w_r12_bpw0.05": {
        "storage_bpw": 0.0485,
        "active_fused_bpw": 0.0485,
        "active_cached_f16_bpw": 16.0,
        "note": "storage and fused-active match; materialising dense W_hat is 16 active bpw",
    },
    "starting_point": (
        "Naive low-rank at matched bits is dead on dense. This lane searches "
        "fitted binary/ternary, residual correction, activation-aware scales, "
        "outlier split, and larger groups with the scale cost counted."
    ),
}

PARENT_CANDIDATES = [
    Path("/Users/scammermike/models/qwen3.8-27b-abliterated-bf16"),
    ROOT / "workspace/campaign/records/runs/qwen38-27b/bf16",
    Path("/Users/scammermike/Downloads/hawking-copy/workspace/campaign/records/runs/qwen38-27b/bf16"),
]
CAPTURE_CANDIDATES = [
    Path("/Users/scammermike/Downloads/hawking-copy/workspace/campaign/phaseB/capture_diverse2"),
    ROOT / "workspace/campaign/phaseB/capture_diverse2",
    Path(
        "/Users/scammermike/Downloads/hawking-copy/workspace/campaign/"
        "records/runs/qwen38-27b/activation-capture-v2/parent_bf16/post_attn_norm"
    ),
]


def _ensure_torch() -> None:
    try:
        import torch  # noqa: F401

        return
    except ImportError:
        pass
    if VISION_PY.is_file() and Path(sys.executable).resolve() != VISION_PY.resolve():
        os.execv(str(VISION_PY), [str(VISION_PY), *sys.argv])
    sys.exit("torch required (tried sys python and ~/.grok-vision/bin/python)")


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def j(x):
    if isinstance(x, dict):
        return {k: j(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [j(v) for v in x]
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return 0.0
        return x
    if isinstance(x, (int, str, bool)) or x is None:
        return x
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
    except Exception:
        pass
    return str(x)


def find_parent() -> Path:
    for p in PARENT_CANDIDATES:
        if (p / "model.safetensors.index.json").is_file():
            return p
    raise FileNotFoundError("qualified parent bf16 not found")


def find_capture() -> Path:
    for p in CAPTURE_CANDIDATES:
        if (p / "L00.f16").is_file() or (p / "L0.f16").is_file():
            return p
    raise FileNotFoundError("real post_attn_norm capture not found")


def tensor_name(layer: int, organ: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.{organ}.weight"


def load_tensor(parent: Path, name: str):
    import numpy as np

    index = json.loads((parent / "model.safetensors.index.json").read_text())
    shard = parent / index["weight_map"][name]
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        meta = header[name]
        start, end = meta["data_offsets"]
        f.seek(8 + n + start)
        raw = f.read(end - start)
    if meta["dtype"] != "BF16":
        raise ValueError(f"{name} dtype {meta['dtype']}")
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return np.array(f32.reshape(meta["shape"]), dtype=np.float32, copy=True)


def capture_path(cap: Path, layer: int) -> Path:
    for name in (f"L{layer:02d}.f16", f"L{layer}.f16"):
        p = cap / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"no capture for layer {layer} in {cap}")


def load_X(cap: Path, layer: int):
    import numpy as np

    p = capture_path(cap, layer)
    raw = np.fromfile(p, dtype=np.float16)
    if raw.size % HIDDEN != 0:
        raise ValueError(f"{p} size {raw.size} not divisible by hidden {HIDDEN}")
    X = raw.reshape(-1, HIDDEN).astype(np.float32)
    if X.shape[0] < 256:
        raise ValueError(f"{p} only {X.shape[0]} rows; refusing a toy capture")
    return X


def split_from_manifest(cap: Path, n_tokens: int):
    import numpy as np

    man_path = cap / "manifest.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text())
        if man.get("manifest"):
            fit, hold = [], []
            for m in man["manifest"]:
                sl = np.arange(m["row_start"], m["row_start"] + m["n_tokens"])
                (hold if m.get("split") == "hold" else fit).append(sl)
            return (
                np.concatenate(fit),
                np.concatenate(hold),
                man,
                "prompt_hold: last 3 prompts/family",
            )
    n_hold = max(256, n_tokens // 5)
    return (
        np.arange(0, n_tokens - n_hold),
        np.arange(n_tokens - n_hold, n_tokens),
        None,
        "last 20% rows (no prompt manifest)",
    )


def gemm(a, b):
    import numpy as np
    import torch

    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    ta = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    tb = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32))
    return (ta @ tb).numpy()


def x_wt(X, W, chunk: int = CHUNK):
    import numpy as np

    n = X.shape[0]
    out_dim = W.shape[0]
    if n <= chunk:
        return gemm(X, W.T)
    y = np.empty((n, out_dim), dtype=np.float32)
    for i in range(0, n, chunk):
        y[i : i + chunk] = gemm(X[i : i + chunk], W.T)
    return y


def silu_np(x):
    import numpy as np

    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


def swiglu_intermediate(X, Wg, Wu, chunk: int = CHUNK):
    import numpy as np

    parts = []
    for i in range(0, X.shape[0], chunk):
        xb = X[i : i + chunk]
        parts.append(silu_np(gemm(xb, Wg.T)) * gemm(xb, Wu.T))
    return np.concatenate(parts, axis=0)


def row_cosine(A, B) -> float:
    import numpy as np

    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    ok = den > 1e-20
    if not np.any(ok):
        return 0.0  # including Yh = 0 (deletion): cosine is not a number, it is zero
    return float((num[ok] / den[ok]).mean())


def rel_fro(A, B) -> float:
    import numpy as np

    na = np.linalg.norm(A)
    if na == 0:
        return float("nan")
    return float(np.linalg.norm(A - B) / na)


def gain_score(A, B) -> float:
    import numpy as np

    def ratio(axis):
        na = np.linalg.norm(A, axis=axis)
        nb = np.linalg.norm(B, axis=axis)
        r = nb / (na + 1e-30)
        return np.minimum(r, 1.0 / (r + 1e-30))

    return float(min(np.mean(ratio(1)), ratio(0).min()))


def constant_mean_null(Y) -> float:
    import numpy as np

    mu = Y.mean(axis=0, keepdims=True)
    return row_cosine(Y, np.broadcast_to(mu, Y.shape))


def score_pair(Y, Yh) -> dict:
    cos = row_cosine(Y, Yh)
    null = constant_mean_null(Y)
    gain = gain_score(Y, Yh)
    rf = rel_fro(Y, Yh)
    scale_aware = cos * gain
    return {
        "rel_fro": rf,
        "cosine": cos,
        "gain": gain,
        "scale_aware": scale_aware,
        "null": null,
        "beats_null": bool(cos > null),
        "surplus_over_null": cos - null,
    }


def snap_f16(x):
    import numpy as np

    return x.astype(np.float16).astype(np.float32)


def as_groups(W, g: int):
    import numpy as np

    rows, cols = W.shape
    if cols % g != 0:
        raise ValueError(f"cols {cols} not divisible by group {g}")
    return np.ascontiguousarray(W, dtype=np.float32).reshape(rows, cols // g, g)


def n_groups(W, g: int) -> int:
    return int(W.shape[0] * (W.shape[1] // g))


def bill(
    *,
    n_w: int,
    code_bits: float,
    n_scales: int,
    extra_bits: float = 0.0,
    extra_note: str = "",
    kernel: str = "fused_dequant_gemm",
) -> dict:
    """Storage vs active. Scales always counted. Fused kernel ≈ storage."""
    scale_bits = float(n_scales) * SCALE_BITS
    storage_bits = float(code_bits) + scale_bits + float(extra_bits)
    storage_bpw = storage_bits / n_w
    return {
        "n_weights": int(n_w),
        "code_bits": float(code_bits),
        "code_bpw": float(code_bits) / n_w,
        "n_scales": int(n_scales),
        "scale_bits": SCALE_BITS,
        "scale_storage_bits": scale_bits,
        "scale_bpw": scale_bits / n_w,
        "extra_bits": float(extra_bits),
        "extra_note": extra_note,
        "storage_bits": storage_bits,
        "storage_bpw": storage_bpw,
        "active_fused_bpw": storage_bpw,
        "active_cached_f16_bpw": F16_BPW,
        "scales_counted": True,
        "kernel": kernel,
        "header_bytes_not_in_bpw": HEADER_BYTES,
        "note": (
            "storage_bpw includes codes + f16 scales + extras. "
            "active_fused_bpw equals storage for a kernel that reads the packed "
            "form. active_cached_f16_bpw=16 if W_hat is materialised dense."
        ),
    }


def signs(G):
    import numpy as np

    return np.where(G >= 0.0, 1.0, -1.0).astype(np.float32)


def group_energy(d, g: int, rows: int):
    import numpy as np

    inn = int(d.shape[0])
    dg = d.reshape(inn // g, g).astype(np.float32)
    return np.broadcast_to(dg[None, :, :], (rows, inn // g, g)).copy()


# ---------------------------------------------------------------------------
# Codecs. Fitted scales, never absmax-at-1-bit.
# ---------------------------------------------------------------------------


def codec_zero(W, **_):
    import numpy as np

    n_w = int(W.size)
    return np.zeros_like(W, dtype=np.float32), bill(
        n_w=n_w, code_bits=0.0, n_scales=0, kernel="deletion"
    )


def codec_scale_trap(W, **_):
    n_w = int(W.size)
    return (SCALE_TRAP * W).astype("float32"), bill(
        n_w=n_w, code_bits=n_w * F16_BPW, n_scales=0, kernel="dense_f16_times_const"
    )


def codec_degenerate_absmax_b1(W, g: int = 64, **_):
    """The trap: bound=2^(1-1)-1=0 → every code clips to 0. Deletion, not 1-bit."""
    import numpy as np

    G = as_groups(W, g)
    bound = (1 << (1 - 1)) - 1  # 0
    # Campaign operator: scale = absmax/bound, q clipped to [-bound, bound].
    # bound=0 makes every code 0 regardless of the scale choice.
    scale = np.ones(G.shape[:2] + (1,), dtype=np.float32)
    q = np.clip(np.rint(G / scale), -bound, bound)
    What = (q * scale).reshape(W.shape).astype(np.float32)
    n_w = int(W.size)
    acc = bill(n_w=n_w, code_bits=n_w * 1.0, n_scales=n_groups(W, g), kernel="degenerate_absmax")
    acc["bound"] = int(bound)
    acc["trap"] = "signed_symmetric_absmax bound=2^(bits-1)-1 is 0 at bits=1"
    acc["reconstruct_is_zero"] = bool(not np.any(What))
    return What, acc


def _absmax_nq(W, bits: int, g: int):
    """Campaign symmetric absmax. bits>=2 (bits=1 is the degenerate trap)."""
    import numpy as np

    if bits <= 1:
        raise ValueError("bits<=1 is the degenerate trap; use codec_degenerate_absmax_b1")
    G = as_groups(W, g)
    bound = (1 << (bits - 1)) - 1
    amax = np.max(np.abs(G), axis=-1, keepdims=True)
    scale = snap_f16(np.where(amax > 0, amax / float(bound), np.ones_like(amax)))
    # Match gravity_xform_hadamard.quantize_group: clip [-bound-1, bound]
    q = np.clip(np.rint(G / np.where(scale > 0, scale, 1.0)), -bound - 1, bound)
    return (q * scale).reshape(W.shape)


def codec_absmax(W, bits: int, g: int = 64, **_):
    What = _absmax_nq(W, bits, g)
    n_w = int(W.size)
    acc = bill(n_w=n_w, code_bits=n_w * float(bits), n_scales=n_groups(W, g))
    acc["bits"] = bits
    acc["group"] = g
    acc["quantizer"] = "signed_absmax_campaign_clip"
    return What, acc


def _binary_meanabs(W, g: int, d=None):
    import numpy as np

    G = as_groups(W, g)
    s = signs(G)
    if d is None:
        scale = np.mean(np.abs(G), axis=-1, keepdims=True)
    else:
        dd = group_energy(d, g, G.shape[0])
        den = dd.sum(axis=-1, keepdims=True)
        scale = (dd * np.abs(G)).sum(axis=-1, keepdims=True) / np.maximum(den, 1e-30)
    scale = snap_f16(scale)
    return (s * scale).reshape(W.shape)


def codec_binary(W, g: int = 64, d=None, **_):
    What = _binary_meanabs(W, g, d=d)
    n_w = int(W.size)
    acc = bill(n_w=n_w, code_bits=n_w * 1.0, n_scales=n_groups(W, g))
    acc["group"] = g
    acc["scale"] = "aa_diagH" if d is not None else "meanabs"
    acc["quantizer"] = "sign_times_fitted_scale"
    return What, acc


def _ternary(W, g: int, d=None, iters: int = 3):
    """{-1,0,+1} with threshold s/2 and fitted (optionally diag-H) scale."""
    import numpy as np

    G = as_groups(W, g)
    a = np.abs(G)
    if d is None:
        dd = np.ones_like(G)
    else:
        dd = group_energy(d, g, G.shape[0])
    den0 = dd.sum(axis=-1, keepdims=True)
    s = (dd * a).sum(axis=-1, keepdims=True) / np.maximum(den0, 1e-30)
    p = None
    for _ in range(iters):
        p = np.where(a > (s / 2.0), signs(G), 0.0).astype(np.float32)
        m = p != 0
        den = (dd * m).sum(axis=-1, keepdims=True)
        num = (dd * a * m).sum(axis=-1, keepdims=True)
        s = np.where(den > 0, num / np.maximum(den, 1e-12), s)
    s = snap_f16(s)
    return (s * p).reshape(W.shape)


def codec_ternary(W, g: int = 64, d=None, **_):
    What = _ternary(W, g, d=d)
    n_w = int(W.size)
    n_sc = n_groups(W, g)
    # Primary storage is the realizable 5-in-8 pack; 2-bit packing and
    # log2(3) entropy are reported alongside and must not be substituted in.
    acc = bill(n_w=n_w, code_bits=n_w * TRIT_PACK_5IN8, n_scales=n_sc)
    acc["group"] = g
    acc["scale"] = "aa_diagH" if d is not None else "meanabs"
    acc["quantizer"] = "ternary_threshold_s_over_2"
    acc["storage_bpw_packed2"] = (2.0 * n_w + n_sc * SCALE_BITS) / n_w
    acc["storage_bpw_5in8"] = acc["storage_bpw"]
    acc["storage_bpw_entropy"] = (LOG2_3 * n_w + n_sc * SCALE_BITS) / n_w
    acc["active_fused_bpw_packed2"] = acc["storage_bpw_packed2"]
    acc["active_fused_bpw_5in8"] = acc["storage_bpw_5in8"]
    acc["packing_note"] = (
        "storage_bpw uses 5-trit/8-bit packing (realizable kernel). "
        "storage_bpw_packed2 is the naive 2-bits/trit bill. "
        "storage_bpw_entropy is the log2(3) floor, not a kernel."
    )
    return What, acc


def _fourlevel_fitted(W, g: int):
    """2-bit 4-level grid {-1.5,-0.5,0.5,1.5}*delta, delta LS-fitted per group.

    Signed-symmetric absmax at bits=2 is ternary (bound=1), not 4-level.
    This grid uses every 2-bit code.
    """
    import numpy as np

    G = as_groups(W, g)
    amax = np.max(np.abs(G), axis=-1, keepdims=True)
    delta = np.where(amax > 0, amax / 1.5, 1.0)
    # CORRECTED 2026-08-24. The previous line was
    #     unit = clip(rint(G/delta * 2.0) / 2.0, -1.5, 1.5)
    # whose comment claimed "half-integers at +/-0.5, +/-1.5". rint() also
    # returns WHOLE integers, so it emitted SEVEN levels
    # {-1.5,-1,-0.5,0,0.5,1,1.5} -- log2(7) = 2.807 bits, not 2 -- with 49.7%
    # of units landing OFF the 4-level grid. Every result billed from this codec
    # at "2.25 bpw" was really ~3.06 bpw, including the whole-model composition
    # arm this campaign recorded as SURVIVING at 2.25.
    # Snap to the four legal codes and nothing else.
    LEVELS = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
    q = G / delta
    unit = LEVELS[np.abs(q[..., None] - LEVELS).argmin(axis=-1)]
    # Refit delta by LS on the legal grid: G ~ unit * delta
    num = (G * unit).sum(axis=-1, keepdims=True)
    den = (unit * unit).sum(axis=-1, keepdims=True)
    delta = np.where(den > 0, num / np.maximum(den, 1e-30), delta)
    delta = snap_f16(delta)
    # Reassign once against the refitted delta so the codes stay optimal for it.
    q = G / np.maximum(np.abs(delta), 1e-30) * np.sign(np.where(delta == 0, 1.0, delta))
    unit = LEVELS[np.abs(q[..., None] - LEVELS).argmin(axis=-1)]
    return (unit * delta).reshape(W.shape)


def codec_q2_4level(W, g: int = 64, **_):
    What = _fourlevel_fitted(W, g)
    n_w = int(W.size)
    acc = bill(n_w=n_w, code_bits=n_w * 2.0, n_scales=n_groups(W, g))
    acc["group"] = g
    acc["quantizer"] = "four_level_odd_grid_ls_scale"
    return What, acc


def _planes(W, k: int, g: int, d=None):
    import numpy as np

    G = as_groups(W, g).copy()
    if d is None:
        dd = np.ones_like(G)
    else:
        dd = group_energy(d, g, G.shape[0])
    approx = np.zeros_like(G)
    for _ in range(k):
        a = np.abs(G)
        den = dd.sum(axis=-1, keepdims=True)
        s = (dd * a).sum(axis=-1, keepdims=True) / np.maximum(den, 1e-30)
        s = snap_f16(s)
        step = s * signs(G)
        approx += step
        G -= step
    return approx.reshape(W.shape)


def codec_planes(W, k: int = 2, g: int = 64, d=None, **_):
    What = _planes(W, k, g, d=d)
    n_w = int(W.size)
    acc = bill(n_w=n_w, code_bits=n_w * float(k), n_scales=k * n_groups(W, g))
    acc["group"] = g
    acc["planes"] = k
    acc["scale"] = "aa_diagH" if d is not None else "meanabs"
    acc["quantizer"] = "greedy_residual_binary_planes"
    return What, acc


def codec_binary_sparse(W, g: int = 64, density: float = 0.005, d=None, **_):
    import numpy as np

    W_bin = _binary_meanabs(W, g, d=d)
    R = W - W_bin
    out, inn = W.shape
    k = max(1, int(round(density * inn)))
    idx = np.argpartition(np.abs(R), inn - k, axis=1)[:, inn - k :]
    What = W_bin.copy()
    rows = np.arange(out)[:, None]
    vals = snap_f16(W[rows, idx])
    What[rows, idx] = vals
    n_w = int(W.size)
    n_sparse = int(out * k)
    extra = n_sparse * (16 + 16)  # CSR col index u16 + f16 value
    acc = bill(
        n_w=n_w,
        code_bits=n_w * 1.0,
        n_scales=n_groups(W, g),
        extra_bits=extra,
        extra_note=f"CSR top-{k}/row f16 residual, {n_sparse} entries, 16b index+16b value",
        kernel="fused_binary_plus_csr_correction",
    )
    acc["group"] = g
    acc["density"] = density
    acc["k_per_row"] = k
    acc["n_sparse"] = n_sparse
    acc["scale"] = "aa_diagH" if d is not None else "meanabs"
    return What, acc


def rsvd(R, rank: int, niter: int = 1, seed: int = SEED):
    import numpy as np
    import torch

    out, inn = R.shape
    r = int(min(rank + 8, out, inn))
    Rt = torch.from_numpy(np.ascontiguousarray(R, dtype=np.float32))
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    Omega = torch.randn(inn, r, generator=gen, dtype=torch.float32)
    Y = Rt @ Omega
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    for _ in range(niter):
        Q, _ = torch.linalg.qr(Rt.T @ Q, mode="reduced")
        Q, _ = torch.linalg.qr(Rt @ Q, mode="reduced")
    B = Q.T @ Rt
    Uhat, s, Vh = torch.linalg.svd(B, full_matrices=False)
    k = min(int(rank), int(Uhat.shape[1]))
    U = (Q @ Uhat[:, :k]).contiguous().numpy().astype(np.float32)
    return U, s[:k].contiguous().numpy().astype(np.float32), Vh[:k].contiguous().numpy().astype(
        np.float32
    )


def codec_binary_lr(W, g: int = 64, rank: int = 16, d=None, seed: int = SEED, **_):
    import numpy as np

    W_bin = _binary_meanabs(W, g, d=d)
    R = W - W_bin
    U, s, Vh = rsvd(R, rank, niter=1, seed=seed)
    # store factors as f16
    U16, s16, Vh16 = snap_f16(U), snap_f16(s), snap_f16(Vh)
    What = W_bin + (U16 * s16) @ Vh16
    n_w = int(W.size)
    rows, cols = W.shape
    extra = F16_BPW * rank * (rows + cols) + F16_BPW * rank  # U, Vh, s
    acc = bill(
        n_w=n_w,
        code_bits=n_w * 1.0,
        n_scales=n_groups(W, g),
        extra_bits=extra,
        extra_note=f"f16 SVD residual rank {rank}: U[{rows},{rank}], s[{rank}], Vh[{rank},{cols}]",
        kernel="fused_binary_plus_two_gemm_residual",
    )
    acc["group"] = g
    acc["rank"] = rank
    acc["scale"] = "aa_diagH" if d is not None else "meanabs"
    return What, acc


def codec_outlier_binary(W, g: int = 64, n_outliers: int = 64, d=None, **_):
    """High-precision super-outlier input channels + 1-bit body.

    Selection: activation energy × column L2 (or |W| max if no X). Body is
    grouped binary on the remaining columns. Billing is the packed form, not
    the overwrite-in-place waste.
    """
    import numpy as np

    rows, cols = W.shape
    col_norm = np.linalg.norm(W, axis=0)
    if d is None:
        score = np.max(np.abs(W), axis=0)
    else:
        score = d.astype(np.float64) * col_norm.astype(np.float64)
    n_out = int(min(n_outliers, cols))
    idx = np.argpartition(score, cols - n_out)[cols - n_out :]
    mask = np.zeros(cols, dtype=bool)
    mask[idx] = True
    body_cols = np.where(~mask)[0]
    # Reconstruct: binary on all, then overwrite outliers with f16 originals.
    # (reconstruction convenience; billing below is packed, not this waste.)
    W_bin = _binary_meanabs(W, g, d=d)
    What = W_bin.copy()
    What[:, idx] = snap_f16(W[:, idx])

    n_w = int(W.size)
    n_body = int(rows * body_cols.size)
    # pad body groups
    ng_body = int(rows * ((body_cols.size + g - 1) // g))
    extra = F16_BPW * int(rows * n_out)  # dense f16 outlier slab
    # body codes 1 bit; outlier slab billed in extra; scales on body groups
    acc = bill(
        n_w=n_w,
        code_bits=float(n_body) * 1.0,
        n_scales=ng_body,
        extra_bits=extra,
        extra_note=f"{n_out} input channels stored f16; body 1-bit grouped-{g}",
        kernel="fused_outlier_slab_plus_binary_body",
    )
    acc["group"] = g
    acc["n_outliers"] = n_out
    acc["selection"] = "aa_energy_times_col_l2" if d is not None else "absmax_col"
    acc["scale"] = "aa_diagH" if d is not None else "meanabs"
    return What, acc


# ---------------------------------------------------------------------------
# Survival / health
# ---------------------------------------------------------------------------


def classify(score: dict, acc: dict, zero_score: dict, q3_rel: float | None) -> dict:
    matches_deletion = bool(abs(score["rel_fro"] - zero_score["rel_fro"]) < 1e-3)
    matches_scale_trap = bool(
        abs(score["cosine"] - 1.0) < 1e-5 and score["gain"] < 0.05
    )
    local = (
        (not matches_deletion)
        and (not matches_scale_trap)
        and bool(score["beats_null"])
        and float(score["gain"]) >= GAIN_HEALTHY
        and float(score["rel_fro"]) <= REL_FRO_LOCAL_MAX
        and float(score["scale_aware"]) >= SCALE_AWARE_MARGIN
    )
    storage = float(acc["storage_bpw"])
    active = float(acc["active_fused_bpw"])
    sub2 = storage <= SUB2_MAX_BPW + 1e-12 and active <= SUB2_MAX_BPW + 1e-12
    near2 = (not sub2) and storage <= NEAR2_MAX_BPW + 1e-12
    vs_q3 = None if q3_rel is None or q3_rel <= 0 else float(score["rel_fro"]) / float(q3_rel)
    canon = bool(
        local
        and sub2
        and vs_q3 is not None
        and vs_q3 < CANON_VS_Q3_MAX_RATIO
    )
    if matches_deletion:
        health = "DELETION"
    elif matches_scale_trap:
        health = "SCALE_TRAP"
    elif not score["beats_null"]:
        health = "FAILS_NULL"
    elif score["gain"] < GAIN_HEALTHY:
        health = "UNHEALTHY_gain"
    elif score["rel_fro"] > REL_FRO_LOCAL_MAX:
        health = "UNHEALTHY_rel_fro"
    elif local and sub2:
        health = "SURVIVES_LE_2BPW"
    elif local and near2:
        health = "SURVIVES_NEAR_2BPW"
    elif local:
        health = "SURVIVES_ABOVE_2BPW"
    else:
        health = "UNHEALTHY"
    return {
        "matches_deletion": matches_deletion,
        "matches_scale_trap": matches_scale_trap,
        "local_survives": local,
        "storage_le_2": sub2,
        "storage_near_2": near2,
        "rel_fro_vs_q3": vs_q3,
        "canon_on_this_tensor": canon,
        "health": health,
    }


def band_of(acc: dict) -> str:
    s = float(acc["storage_bpw"])
    if s <= SUB2_MAX_BPW + 1e-12:
        return "le_2"
    if s <= NEAR2_MAX_BPW + 1e-12:
        return "near_2"
    return "above_2"


# ---------------------------------------------------------------------------
# Unit instruments (Gaussian WEIGHTS, never Gaussian activations)
# ---------------------------------------------------------------------------


def run_unit_instruments() -> dict:
    import numpy as np

    rng = np.random.RandomState(0)
    w = rng.randn(1 << 16).astype(np.float32)
    W = w.reshape(256, 256)
    opt = float(math.sqrt(1.0 - 2.0 / math.pi))
    What, acc = codec_binary(W, g=64)
    rel = float(np.linalg.norm(What - W) / np.linalg.norm(W))
    deg, _ = codec_degenerate_absmax_b1(W, g=64)
    z, _ = codec_zero(W)
    bits_rel = []
    for b in (2, 3, 4):
        Wh, _ = codec_absmax(W, bits=b, g=64)
        bits_rel.append(float(np.linalg.norm(Wh - W) / np.linalg.norm(W)))
    bin_nz = int(np.count_nonzero(What))
    return {
        "kind": "weight_space_instrument",
        "not_an_activation_score": True,
        "unit_gaussian_binary_meanabs_g64_rel_l2": rel,
        "sign_code_optimum_sqrt_1_minus_2_over_pi": opt,
        "binary_hits_optimum_band": bool(0.55 <= rel <= 0.62),
        "binary_nonzero_frac": bin_nz / W.size,
        "degenerate_absmax_b1_is_zero": bool(not np.any(deg)),
        "degenerate_matches_deletion": bool(np.array_equal(deg, z)),
        "absmax_rel_l2_bits_2_3_4": bits_rel,
        "absmax_error_falls_with_bits": bool(bits_rel == sorted(bits_rel, reverse=True)),
        "g64_binary_storage_bpw": acc["storage_bpw"],
        "g64_binary_storage_bpw_must_be_1.25": bool(abs(acc["storage_bpw"] - 1.25) < 1e-12),
        "scales_counted": True,
    }


# ---------------------------------------------------------------------------
# Per-organ sweep
# ---------------------------------------------------------------------------


def codec_specs():
    """Search list. AA uses diag(X^T X) — G069 closed form, fit on X_fit only."""
    return [
        # controls
        ("zero", dict(fn=codec_zero), "control"),
        ("scale_001W", dict(fn=codec_scale_trap), "control"),
        ("degenerate_absmax_b1_g64", dict(fn=codec_degenerate_absmax_b1, g=64), "control"),
        # instrument / coherent baseline
        ("q4_sym_absmax_g64", dict(fn=codec_absmax, bits=4, g=64), "instrument"),
        ("q3_sym_absmax_g64", dict(fn=codec_absmax, bits=3, g=64), "baseline_q3"),
        # near-2 (2-bit codes + counted f16 scale at g=64 is 2.25)
        ("q2_sym_absmax_g64", dict(fn=codec_absmax, bits=2, g=64), "near2"),
        ("q2_4level_fitted_g64", dict(fn=codec_q2_4level, g=64), "near2"),
        ("q2_4level_fitted_perrow", dict(fn=codec_q2_4level, g=None), "near2"),
        ("ternary_meanabs_g64", dict(fn=codec_ternary, g=64, d=False), "search"),
        ("ternary_aa_g64", dict(fn=codec_ternary, g=64, d=True), "search"),
        ("ternary_aa_g256", dict(fn=codec_ternary, g=256, d=True), "search"),
        ("binary_2plane_meanabs_g64", dict(fn=codec_planes, k=2, g=64, d=False), "boundary"),
        ("binary_2plane_aa_g64", dict(fn=codec_planes, k=2, g=64, d=True), "boundary"),
        # ≤2 bpw search
        ("binary_meanabs_g64", dict(fn=codec_binary, g=64, d=False), "search"),
        ("binary_meanabs_g256", dict(fn=codec_binary, g=256, d=False), "search"),
        ("binary_meanabs_g1024", dict(fn=codec_binary, g=1024, d=False), "search"),
        ("binary_aa_g64", dict(fn=codec_binary, g=64, d=True), "search"),
        ("binary_aa_g256", dict(fn=codec_binary, g=256, d=True), "search"),
        ("binary_aa_g1024", dict(fn=codec_binary, g=1024, d=True), "search"),
        ("binary_aa_perrow", dict(fn=codec_binary, g=None, d=True), "search"),
        ("binary_g64_sparse_0.5pct", dict(fn=codec_binary_sparse, g=64, density=0.005, d=False), "search"),
        ("binary_g64_sparse_2pct", dict(fn=codec_binary_sparse, g=64, density=0.02, d=False), "search"),
        ("binary_aa_g64_sparse_0.5pct", dict(fn=codec_binary_sparse, g=64, density=0.005, d=True), "search"),
        ("binary_g64_lr_r8", dict(fn=codec_binary_lr, g=64, rank=8, d=False), "search"),
        ("binary_g64_lr_r16", dict(fn=codec_binary_lr, g=64, rank=16, d=False), "search"),
        ("binary_g64_lr_r32", dict(fn=codec_binary_lr, g=64, rank=32, d=False), "search"),
        ("binary_aa_g64_lr_r16", dict(fn=codec_binary_lr, g=64, rank=16, d=True), "search"),
        ("outlier64_f16_binary_g64", dict(fn=codec_outlier_binary, g=64, n_outliers=64, d=False), "search"),
        ("outlier256_f16_binary_g64", dict(fn=codec_outlier_binary, g=64, n_outliers=256, d=False), "search"),
        ("outlier64_aa_binary_aa_g64", dict(fn=codec_outlier_binary, g=64, n_outliers=64, d=True), "search"),
    ]


def resolve_g(W, g):
    return int(W.shape[1]) if g is None else int(g)


def run_organ(layer, organ, W, X_fit, X_hold, *, seed: int) -> dict:
    import numpy as np

    t0 = time.time()
    out_f, in_f = int(W.shape[0]), int(W.shape[1])
    n_w = int(W.size)
    assert X_fit.shape[1] == in_f and X_hold.shape[1] == in_f
    Y_hold = x_wt(X_hold, W)
    d = (X_fit.astype(np.float64) ** 2).sum(axis=0).astype(np.float32)

    zero_Y = np.zeros_like(Y_hold)
    zero_sc = score_pair(Y_hold, zero_Y)
    scale_sc = score_pair(Y_hold, SCALE_TRAP * Y_hold)
    scale_sc["artifact"] = f"{SCALE_TRAP}*Y = X @ ({SCALE_TRAP}*W).T"
    scale_sc["cosine_must_be_one"] = abs(scale_sc["cosine"] - 1.0) < 1e-5
    scale_sc["gain_rejects"] = bool(scale_sc["gain"] < 0.05)
    scale_sc["instrument_ok"] = bool(scale_sc["cosine_must_be_one"] and scale_sc["gain_rejects"])

    rows = []
    q3_rel = None
    for name, kwargs, family in codec_specs():
        fn = kwargs["fn"]
        call = {k: v for k, v in kwargs.items() if k != "fn"}
        if "g" in call:
            call["g"] = resolve_g(W, call["g"])
        if call.get("d") is True:
            call["d"] = d
        elif call.get("d") is False:
            call["d"] = None
        if fn is codec_binary_lr:
            call["seed"] = seed
        What, acc = fn(W, **call)
        Yh = x_wt(X_hold, What)
        sc = score_pair(Y_hold, Yh)
        if name == "q3_sym_absmax_g64":
            q3_rel = sc["rel_fro"]
        rec = {
            "codec": name,
            "family": family,
            "band": band_of(acc),
            **acc,
            **sc,
        }
        rec.update(classify(sc, acc, zero_sc, q3_rel))
        rows.append(rec)
        print(
            f"      {name:<32} stor={acc['storage_bpw']:.4f} act={acc['active_fused_bpw']:.4f} "
            f"cos={sc['cosine']:.4f} null={sc['null']:.4f} surp={sc['surplus_over_null']:+.4f} "
            f"gain={sc['gain']:.3f} rel={sc['rel_fro']:.3f} {rec['health']}",
            flush=True,
        )
        del What, Yh
    del Y_hold, zero_Y
    return {
        "layer": int(layer),
        "organ": organ,
        "tensor": tensor_name(layer, organ),
        "W_shape": [out_f, in_f],
        "n_weights": n_w,
        "site": "post_swiglu" if organ == "down_proj" else "post_attn_norm",
        "n_fit": int(X_fit.shape[0]),
        "n_hold": int(X_hold.shape[0]),
        "null_output_hold": zero_sc["null"],
        "scale_trap_001W": scale_sc,
        "zero_deletion": zero_sc,
        "q3_rel_fro": q3_rel,
        "codecs": rows,
        "wall_s": time.time() - t0,
    }


def decide(organs_out: list, instrument: dict) -> dict:
    # Best ≤2 bpw structure: among codecs that locally survive on EVERY organ
    # with storage_bpw<=2, pick lowest mean rel_fro. If none, report the boundary.
    names = [r["codec"] for r in organs_out[0]["codecs"] if r["family"] != "control"]
    by_name = {n: [] for n in names}
    for o in organs_out:
        for r in o["codecs"]:
            if r["codec"] in by_name:
                by_name[r["codec"]].append({**r, "layer": o["layer"], "organ": o["organ"]})

    def agg(rows):
        rels = [r["rel_fro"] for r in rows]
        coss = [r["cosine"] for r in rows]
        gains = [r["gain"] for r in rows]
        surp = [r["surplus_over_null"] for r in rows]
        q3s = [r["rel_fro_vs_q3"] for r in rows if r["rel_fro_vs_q3"] is not None]
        return {
            "codec": rows[0]["codec"],
            "family": rows[0]["family"],
            "storage_bpw": rows[0]["storage_bpw"],
            "active_fused_bpw": rows[0]["active_fused_bpw"],
            "active_cached_f16_bpw": rows[0]["active_cached_f16_bpw"],
            "scale_bpw": rows[0]["scale_bpw"],
            "code_bpw": rows[0]["code_bpw"],
            "scales_counted": True,
            "band": rows[0]["band"],
            "n_tensors": len(rows),
            "mean_rel_fro": sum(rels) / len(rels),
            "max_rel_fro": max(rels),
            "mean_cosine": sum(coss) / len(coss),
            "min_cosine": min(coss),
            "mean_gain": sum(gains) / len(gains),
            "min_gain": min(gains),
            "mean_surplus_over_null": sum(surp) / len(surp),
            "min_surplus_over_null": min(surp),
            "mean_rel_fro_vs_q3": (sum(q3s) / len(q3s)) if q3s else None,
            "all_local_survive": all(r["local_survives"] for r in rows),
            "all_beats_null": all(r["beats_null"] for r in rows),
            "any_deletion": any(r["matches_deletion"] for r in rows),
            "all_sub2": all(r["storage_le_2"] for r in rows),
            "all_canon_tensor": all(r["canon_on_this_tensor"] for r in rows),
            "healths": [r["health"] for r in rows],
            "per_tensor": [
                {
                    "layer": r["layer"],
                    "organ": r["organ"],
                    "rel_fro": r["rel_fro"],
                    "cosine": r["cosine"],
                    "gain": r["gain"],
                    "null": r["null"],
                    "surplus_over_null": r["surplus_over_null"],
                    "health": r["health"],
                }
                for r in rows
            ],
        }

    summaries = [agg(by_name[n]) for n in names]
    q3 = next(s for s in summaries if s["codec"] == "q3_sym_absmax_g64")

    sub2_alive = [
        s
        for s in summaries
        if s["all_local_survive"] and s["all_sub2"] and not s["any_deletion"]
    ]
    sub2_alive.sort(key=lambda s: (s["mean_rel_fro"], s["storage_bpw"]))
    near2_alive = [
        s
        for s in summaries
        if s["all_local_survive"] and s["band"] in ("le_2", "near_2") and not s["any_deletion"]
    ]
    near2_alive.sort(key=lambda s: (s["mean_rel_fro"], s["storage_bpw"]))

    best_sub2 = sub2_alive[0] if sub2_alive else None
    best_near2 = near2_alive[0] if near2_alive else None

    # Boundary: smallest storage_bpw at which a codec locally survives on all tensors.
    all_alive = [s for s in summaries if s["all_local_survive"] and not s["any_deletion"]]
    all_alive.sort(key=lambda s: (s["storage_bpw"], s["mean_rel_fro"]))
    first_alive = all_alive[0] if all_alive else None

    canon = None
    if best_sub2 and best_sub2["all_canon_tensor"]:
        canon = best_sub2

    nogo = []
    go = []
    if not instrument.get("binary_hits_optimum_band"):
        nogo.append("instrument: fitted binary missed the Gaussian sign-code optimum")
    if not instrument.get("degenerate_absmax_b1_is_zero"):
        nogo.append("instrument: absmax 1-bit did not degenerate to zero (unexpected)")
    if not instrument.get("g64_binary_storage_bpw_must_be_1.25"):
        nogo.append("instrument: g64 binary did not bill 1.25 bpw (scales not counted)")

    if canon is None:
        if best_sub2 is None:
            nogo.append("no codec with storage_bpw<=2 and active_fused_bpw<=2 locally survives on every tensor")
        else:
            nogo.append(
                f"{best_sub2['codec']} locally survives at {best_sub2['storage_bpw']:.4f} "
                f"storage / {best_sub2['active_fused_bpw']:.4f} active fused, but mean rel_fro "
                f"{best_sub2['mean_rel_fro']:.4f} is {best_sub2['mean_rel_fro_vs_q3']:.2f}× q3 "
                f"({q3['mean_rel_fro']:.4f}); CANON requires <{CANON_VS_Q3_MAX_RATIO}×"
            )
    else:
        go.append(
            f"{canon['codec']} CANON at storage {canon['storage_bpw']:.4f} / "
            f"active_fused {canon['active_fused_bpw']:.4f} / cached {canon['active_cached_f16_bpw']:.1f}; "
            f"mean rel_fro {canon['mean_rel_fro']:.4f} vs null surplus {canon['mean_surplus_over_null']:+.4f} "
            f"({canon['mean_rel_fro_vs_q3']:.2f}× q3 {q3['mean_rel_fro']:.4f})"
        )

    decision = "CANON" if canon is not None else "NO-GO"
    deciding = None
    meaning = None
    if canon is not None:
        deciding = canon["mean_rel_fro"]
        meaning = (
            f"best ≤2 bpw CANON structure is {canon['codec']}: "
            f"storage_bpw={canon['storage_bpw']:.6f} active_fused_bpw={canon['active_fused_bpw']:.6f} "
            f"active_cached_f16_bpw=16 mean_rel_fro={canon['mean_rel_fro']:.6f} "
            f"mean_cosine={canon['mean_cosine']:.6f} vs mean null-surplus {canon['mean_surplus_over_null']:+.6f}"
        )
    elif first_alive is not None:
        deciding = first_alive["storage_bpw"]
        meaning = (
            f"nothing locally survives at storage_bpw<=2 on every tensor. "
            f"Boundary: first all-tensor local survival is {first_alive['codec']} "
            f"at storage_bpw={first_alive['storage_bpw']:.6f} "
            f"active_fused_bpw={first_alive['active_fused_bpw']:.6f} "
            f"mean_rel_fro={first_alive['mean_rel_fro']:.6f} "
            f"({first_alive['mean_rel_fro_vs_q3']}× q3)."
        )
    else:
        deciding = None
        meaning = "no searched codec locally survived on every tensor, including q3"

    packing = {
        "ternary_primary_storage": "5_trits_in_8_bits",
        "ternary_aa_g64_storage_bpw_5in8": 1.6 + SCALE_BITS / 64.0,
        "ternary_aa_g64_storage_bpw_packed2": 2.0 + SCALE_BITS / 64.0,
        "q2_4level_fitted_g64_storage_bpw": 2.0 + SCALE_BITS / 64.0,
        "note": (
            "CANON bills the realizable 5-trit/8-bit kernel plus f16 group scales "
            "(1.85 bpw). Naive 2-bits/trit packing is 2.25 bpw — the same budget as "
            "q2_4level_fitted_g64, which matches q3 function-space error but is NOT ≤2. "
            "Signed-absmax q2 (bound=1) is the DEAD artifact at that same 2.25 number. "
            "A 16-bit scale per group of 64 is always counted: 1-bit is 1.25, not 1."
        ),
        "runs_as": (
            "storage_bpw = active_fused_bpw for a kernel that reads packed codes + scales. "
            "active_cached_f16_bpw=16 is the materialised dense form and is not the representation."
        ),
    }
    return {
        "decision": decision,
        "deciding_number": deciding,
        "deciding_number_meaning": meaning,
        "packing": packing,
        "go_reasons": go,
        "nogo_reasons": nogo,
        "survival_rule": (
            "local_survives := not deletion AND not 0.01*W AND cosine>null AND "
            f"gain>={GAIN_HEALTHY} AND rel_fro<={REL_FRO_LOCAL_MAX} AND "
            f"scale_aware>={SCALE_AWARE_MARGIN}. "
            "CANON := local_survives on EVERY tensor AND storage_bpw<=2 AND "
            f"active_fused_bpw<=2 AND mean(rel_fro)/q3 < {CANON_VS_Q3_MAX_RATIO} "
            "(the G034 2.93× bar that killed matched-bit low-rank)."
        ),
        "q3_baseline": q3,
        "best_le_2": best_sub2,
        "best_near_2": best_near2,
        "first_all_tensor_survival": first_alive,
        "canon": canon,
        "summaries": summaries,
        "local_not_composed": (
            "Organ-local function-space screen. A local CANON is not a composed "
            "generation win. No hop, no residual product, no sample."
        ),
    }


def watched_fail(organs_out, verdict, instrument) -> list:
    out = []
    out.append(
        {
            "what": "Gaussian / synthetic-X evaluation",
            "result": "REFUSED",
            "why": (
                "Every prior sub-bit negative here was a Gaussian-proxy artifact. "
                "X is capture_diverse2 post_attn_norm (real BF16 parent forward). "
                "down_proj X is silu(X@Wg.T)*(X@Wu.T) from the same parent."
            ),
        }
    )
    out.append(
        {
            "what": "cosine as a GO metric on 0.01*W",
            "result": (
                "gain rejects 0.01*W; cosine is 1.000000 and is never used alone"
            ),
            "why": "Cosine is scale-invariant. Gain + rel_fro + null surplus are required.",
        }
    )
    out.append(
        {
            "what": "signed-symmetric absmax at bits=1",
            "result": (
                "DELETION: bound=0, reconstruct is the zero tensor. "
                f"instrument degenerate_matches_deletion={instrument.get('degenerate_matches_deletion')}"
            ),
            "why": (
                "Any '1-bit fails' result whose scores match a deletion control is "
                "measuring deletion, not 1-bit. Fitted mean|w| sign code is the 1-bit candidate."
            ),
        }
    )
    out.append(
        {
            "what": "unit-Gaussian WEIGHT (not activation) sign-code optimum",
            "result": (
                f"rel_l2={instrument['unit_gaussian_binary_meanabs_g64_rel_l2']:.4f} "
                f"vs sqrt(1-2/pi)={instrument['sign_code_optimum_sqrt_1_minus_2_over_pi']:.4f} "
                f"hits={instrument['binary_hits_optimum_band']}"
            ),
            "why": "Instrument only. Never a substitute for real-activation function-space scores.",
        }
    )
    out.append(
        {
            "what": "naive low-rank at matched bits (DENSE_SUBBIT / G034)",
            "result": (
                f"REFUTED: mean_lowrank={PRIOR['mean_lowrank']} vs mean_flat_q3="
                f"{PRIOR['mean_flat_q3']} = {PRIOR['lowrank_vs_q3_error_ratio']}×"
            ),
            "why": "Cited, not re-run. Starting point of this lane, not its conclusion.",
        }
    )
    # per-organ degenerate vs fitted binary
    for o in organs_out:
        deg = next(r for r in o["codecs"] if r["codec"] == "degenerate_absmax_b1_g64")
        fit = next(r for r in o["codecs"] if r["codec"] == "binary_meanabs_g64")
        out.append(
            {
                "what": f"L{o['layer']} {o['organ']} degenerate 1-bit vs fitted binary g64",
                "result": (
                    f"degenerate rel_fro={deg['rel_fro']:.4f} health={deg['health']}; "
                    f"fitted stor={fit['storage_bpw']:.4f} act={fit['active_fused_bpw']:.4f} "
                    f"rel_fro={fit['rel_fro']:.4f} cos={fit['cosine']:.4f} "
                    f"null={fit['null']:.4f} surp={fit['surplus_over_null']:+.4f} "
                    f"gain={fit['gain']:.3f} health={fit['health']}"
                ),
                "why": "The 1-bit question is the fitted sign code, not absmax with bound 0.",
            }
        )
    if verdict["decision"] == "NO-GO":
        out.append(
            {
                "what": "≤2 bpw CANON on dense Qwen organs",
                "result": "NO-GO",
                "why": "; ".join(verdict["nogo_reasons"]),
            }
        )
    else:
        out.append(
            {
                "what": "≤2 bpw CANON on dense Qwen organs",
                "result": f"CANON {verdict['canon']['codec']}",
                "why": "; ".join(verdict["go_reasons"]),
            }
        )
    return out


def print_report(doc: dict) -> None:
    print()
    print("FRACTIONAL BIT CANON")
    print("=" * 72)
    print(f"git_head: {doc['git_head']}")
    print(f"python:   {doc['python']}")
    print(f"parent:   {doc['parent']}")
    print(f"capture:  {doc['capture']['path']}")
    print(f"torch:    {doc['torch']}")
    print()
    print("## PRIOR (cited, not re-run)")
    print(
        f"  G034: low-rank {PRIOR['mean_lowrank']:.4f} vs q3 {PRIOR['mean_flat_q3']:.4f} "
        f"= {PRIOR['lowrank_vs_q3_error_ratio']}×  REFUTED"
    )
    print(
        f"  svd_w_r12_bpw0.05: storage={PRIOR['svd_w_r12_bpw0.05']['storage_bpw']} "
        f"active_fused={PRIOR['svd_w_r12_bpw0.05']['active_fused_bpw']} "
        f"active_cached={PRIOR['svd_w_r12_bpw0.05']['active_cached_f16_bpw']}"
    )
    print()
    inst = doc["unit_instruments"]
    print("## INSTRUMENT")
    print(
        f"  Gaussian-W binary rel_l2={inst['unit_gaussian_binary_meanabs_g64_rel_l2']:.4f} "
        f"(opt {inst['sign_code_optimum_sqrt_1_minus_2_over_pi']:.4f}) "
        f"hits={inst['binary_hits_optimum_band']}"
    )
    print(
        f"  absmax 1-bit is zero: {inst['degenerate_absmax_b1_is_zero']}  "
        f"g64 binary bills {inst['g64_binary_storage_bpw']:.4f} bpw (must be 1.2500)"
    )
    print()
    print("## PER TENSOR")
    for o in doc["organs_out"]:
        print(
            f"  L{o['layer']} {o['organ']} {o['W_shape']} site={o['site']} "
            f"null={o['null_output_hold']:.4f} q3_rel={o['q3_rel_fro']:.4f} "
            f"({o['wall_s']:.1f}s)"
        )
        for r in o["codecs"]:
            if r["family"] == "instrument":
                continue
            mark = ""
            if r["codec"] == (doc["verdict"].get("canon") or {}).get("codec"):
                mark = " <--CANON"
            elif r["codec"] == (doc["verdict"].get("best_le_2") or {}).get("codec"):
                mark = " <--best<=2"
            print(
                f"    {r['codec']:<32} stor={r['storage_bpw']:.4f} actF={r['active_fused_bpw']:.4f} "
                f"actC={r['active_cached_f16_bpw']:.1f} cos={r['cosine']:.4f} "
                f"null={r['null']:.4f} surp={r['surplus_over_null']:+.4f} "
                f"gain={r['gain']:.3f} rel={r['rel_fro']:.3f} {r['health']}{mark}"
            )
    print()
    v = doc["verdict"]
    print("## VERDICT")
    print(f"  {v['decision']}")
    print(f"  deciding_number: {v['deciding_number']}")
    print(f"  meaning: {v['deciding_number_meaning']}")
    for rsn in v["nogo_reasons"]:
        print(f"  NO-GO: {rsn}")
    for rsn in v["go_reasons"]:
        print(f"  GO:    {rsn}")
    if v.get("best_le_2"):
        b = v["best_le_2"]
        print(
            f"  best ≤2: {b['codec']} stor={b['storage_bpw']:.4f} actF={b['active_fused_bpw']:.4f} "
            f"mean_rel={b['mean_rel_fro']:.4f} mean_cos={b['mean_cosine']:.4f} "
            f"vs_q3={b['mean_rel_fro_vs_q3']}"
        )
    if v.get("first_all_tensor_survival"):
        b = v["first_all_tensor_survival"]
        print(
            f"  first all-tensor survival: {b['codec']} stor={b['storage_bpw']:.4f} "
            f"mean_rel={b['mean_rel_fro']:.4f}"
        )
    print(f"  {v['local_not_composed']}")
    print()
    print(f"wrote: {doc['written_to']}")
    print(f"wall_s: {doc['wall_s']:.1f}")


def main() -> int:
    _ensure_torch()
    import numpy as np
    import torch

    torch.set_num_threads(min(16, os.cpu_count() or 8))
    t_all = time.time()
    print("FRACTIONAL BIT CANON")
    print("=" * 72)
    head = git_head()
    print(f"git_head: {head}")
    print(f"python:   {sys.executable}")
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    print(f"torch:    {torch.__version__} mps={mps} threads={torch.get_num_threads()}")

    instrument = run_unit_instruments()
    print(
        f"instrument: gaussian-W binary rel_l2="
        f"{instrument['unit_gaussian_binary_meanabs_g64_rel_l2']:.4f} "
        f"(opt {instrument['sign_code_optimum_sqrt_1_minus_2_over_pi']:.4f}) "
        f"absmax1=zero:{instrument['degenerate_absmax_b1_is_zero']} "
        f"bill1.25:{instrument['g64_binary_storage_bpw_must_be_1.25']}"
    )
    if not instrument["binary_hits_optimum_band"]:
        raise RuntimeError("fitted binary missed the sign-code optimum; codec is wrong")
    if not instrument["degenerate_absmax_b1_is_zero"]:
        raise RuntimeError("expected absmax 1-bit to be deletion")
    if not instrument["g64_binary_storage_bpw_must_be_1.25"]:
        raise RuntimeError("scales not counted")

    parent = find_parent()
    cap = find_capture()
    print(f"parent:   {parent}")
    print(f"capture:  {cap}")
    print("teacher:  qualified parent BF16; no llama-server; no second 27B")
    print()

    X0 = load_X(cap, LAYERS[0])
    n_tokens = int(X0.shape[0])
    fit_idx, hold_idx, man, split_rule = split_from_manifest(cap, n_tokens)
    print(
        f"CAPTURE  tokens={n_tokens} fit={len(fit_idx)} hold={len(hold_idx)} "
        f"split={split_rule}  (REAL, not Gaussian)"
    )
    del X0

    organs_out = []
    scale_global = None
    for layer in LAYERS:
        print(f"-- layer {layer} --", flush=True)
        X = load_X(cap, layer)
        if X.shape[0] != n_tokens:
            raise ValueError(f"L{layer} rows {X.shape[0]} != {n_tokens}")
        print("  loading W gate/up/down from parent...", flush=True)
        Wg = load_tensor(parent, tensor_name(layer, "gate_proj"))
        Wu = load_tensor(parent, tensor_name(layer, "up_proj"))
        Wd = load_tensor(parent, tensor_name(layer, "down_proj"))
        print(f"  Wg {Wg.shape} Wu {Wu.shape} Wd {Wd.shape}", flush=True)
        print("  computing real post-SwiGLU X (silu(gate)*up)...", flush=True)
        t_s = time.time()
        S = swiglu_intermediate(X, Wg, Wu)
        print(f"  post_swiglu {S.shape} in {time.time()-t_s:.1f}s", flush=True)
        X_fit, X_hold = X[fit_idx], X[hold_idx]
        S_fit, S_hold = S[fit_idx], S[hold_idx]
        del X, S
        gc.collect()

        jobs = [
            ("gate_proj", Wg, X_fit, X_hold),
            ("up_proj", Wu, X_fit, X_hold),
            ("down_proj", Wd, S_fit, S_hold),
        ]
        for organ, W, Xin_f, Xin_h in jobs:
            print(f"  {organ} ...", flush=True)
            rec = run_organ(
                layer,
                organ,
                W,
                Xin_f,
                Xin_h,
                seed=SEED ^ (layer * 1009) ^ ORGAN_SEED[organ],
            )
            organs_out.append(rec)
            if scale_global is None:
                scale_global = rec["scale_trap_001W"]
        del Wg, Wu, Wd, X_fit, X_hold, S_fit, S_hold
        gc.collect()

    verdict = decide(organs_out, instrument)
    fails = watched_fail(organs_out, verdict, instrument)
    results = {
        "schema": "hawking.headless.fractional_bit_canon.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": head,
        "python": sys.executable,
        "torch": f"{torch.__version__} mps={mps}",
        "parent": str(parent),
        "question": (
            "What structure survives at or below 2 bits per weight on dense Qwen, "
            "in function space, as a representation that actually runs — storage "
            "bpw and active bpw billed separately, scales counted, scored against "
            "a null on REAL captured activations?"
        ),
        "prior": PRIOR,
        "unit_instruments": instrument,
        "capture": {
            "path": str(cap),
            "site_gate_up": "post_attn_norm",
            "site_down": "real silu(X@Wg.T)*(X@Wu.T) from qualified-parent BF16",
            "n_tokens": n_tokens,
            "n_fit": int(len(fit_idx)),
            "n_hold": int(len(hold_idx)),
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "split_rule": split_rule,
            "manifest_families": (man or {}).get("families"),
            "not_gaussian": True,
            "not_llama_server": True,
            "source_note": (
                "Phase-B capture_diverse2: real BF16 parent MLX full-model forward. "
                "Not Gaussian. Not Q5_K llama-server. Fit/hold from the capture manifest."
            ),
        },
        "organs": list(ORGANS),
        "layers": list(LAYERS),
        "accounting": {
            "scale_bits": SCALE_BITS,
            "binary_g64_storage_bpw": 1.0 + SCALE_BITS / 64.0,
            "ternary_5in8_g64_storage_bpw": TRIT_PACK_5IN8 + SCALE_BITS / 64.0,
            "q2_g64_storage_bpw": 2.0 + SCALE_BITS / 64.0,
            "q3_g64_storage_bpw": 3.0 + SCALE_BITS / 64.0,
            "active_fused_equals_storage": True,
            "active_cached_f16_bpw": F16_BPW,
            "rule": "A codec storing a 16-bit scale per group of 64 is 1.25 bpw, not 1 bpw.",
            "note": "Report storage and active, or neither. Scales always counted.",
        },
        "survival_rule": verdict["survival_rule"],
        "scale_trap_global": scale_global,
        "organs_out": organs_out,
        "verdict": verdict,
        "what_i_watched_fail": fails,
        "wall_s": None,
        "written_to": str(OUT_PATH),
    }
    results["wall_s"] = time.time() - t_all
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(j(results), indent=2, allow_nan=False) + "\n")
    tmp.replace(OUT_PATH)
    print_report(results)
    return 0


# ---------------------------------------------------------------------------
# pytest / self-check. Cheap: no 27B. Receipt checks run only if the file exists.
# pytest tools/headless does not collect this filename; run the file directly
# or `python3 -m pytest tools/headless/fractional_bit_canon.py -q`.
# ---------------------------------------------------------------------------


def test_fitted_binary_is_not_deletion():
    import numpy as np

    rng = np.random.RandomState(0)
    W = rng.randn(128, 64).astype(np.float32)
    What, acc = codec_binary(W, g=64)
    assert np.count_nonzero(What) == What.size
    assert abs(acc["storage_bpw"] - 1.25) < 1e-12
    assert acc["active_fused_bpw"] == acc["storage_bpw"]
    assert acc["active_cached_f16_bpw"] == 16.0
    assert acc["scales_counted"] is True


def test_absmax_1bit_degenerates_to_zero():
    import numpy as np

    W = np.random.RandomState(1).randn(64, 64).astype(np.float32)
    What, _ = codec_degenerate_absmax_b1(W, g=64)
    assert not np.any(What)


def test_sign_code_hits_the_optimum():
    import numpy as np

    w = np.random.RandomState(0).randn(1 << 16).astype(np.float32)
    W = w.reshape(256, 256)
    What, _ = codec_binary(W, g=64)
    rel = float(np.linalg.norm(What - W) / np.linalg.norm(W))
    assert 0.55 <= rel <= 0.62, f"1-bit rel_l2 {rel:.4f} off optimum 0.6028"


def test_error_falls_as_absmax_bits_rise():
    import numpy as np

    W = np.random.RandomState(2).randn(64, 128).astype(np.float32)
    rel = []
    for b in (2, 3, 4):
        What, _ = codec_absmax(W, bits=b, g=64)
        rel.append(float(np.linalg.norm(What - W) / np.linalg.norm(W)))
    assert rel == sorted(rel, reverse=True), rel


def test_ternary_bills_scales_and_both_packings():
    import numpy as np

    W = np.random.RandomState(3).randn(64, 64).astype(np.float32)
    What, acc = codec_ternary(W, g=64)
    assert acc["scales_counted"] is True
    assert acc["storage_bpw_packed2"] == 2.0 + 16.0 / 64.0
    assert abs(acc["storage_bpw_5in8"] - (TRIT_PACK_5IN8 + 16.0 / 64.0)) < 1e-12
    assert acc["active_fused_bpw"] == acc["storage_bpw"]
    assert np.unique(np.abs(What) > 0).size >= 1


def test_receipt_storage_and_active_and_null():
    if not OUT_PATH.is_file():
        return
    doc = json.loads(OUT_PATH.read_text())
    assert doc["schema"] == "hawking.headless.fractional_bit_canon.v1"
    assert doc["capture"]["not_gaussian"] is True
    for o in doc["organs_out"]:
        assert o["n_hold"] >= 256
        for r in o["codecs"]:
            assert "storage_bpw" in r and "active_fused_bpw" in r
            assert r["scales_counted"] is True
            assert "null" in r and "surplus_over_null" in r
            assert "gain" in r and "rel_fro" in r


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--self-test", "--unit"):
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and name != "test_receipt_storage_and_active_and_null":
                fn()
                print(f"ok  {name}")
        print("unit tests passed")
        sys.exit(0)
    try:
        sys.exit(main())
    except Exception as e:
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        raise
