#!/usr/bin/env python3
"""G-XFORM Walsh-Hadamard / butterfly measurement. CPU only. No GPU.

Writes /tmp/g1_xform_hadamard.json. Does not touch the repo.
"""
from __future__ import annotations
import gc, json, os, struct, sys, time, traceback
import numpy as np

try:
    import resource
    def rss_gb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
except Exception:
    def rss_gb():
        return float("nan")

sys.path.insert(0, "/Users/scammermike/.claude-grok/worktrees/201-xform-hadamard-20260817-181007/tools")
from gravity_doctor_gate import axes, gate, AXIS_MARGIN  # noqa: E402

BF16 = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
CAPTURE = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
N_LANG = 26_895_998_464
OUT_JSON = "/tmp/g1_xform_hadamard.json"
GROUP = 128
HEAD_DIM = 256
ROTARY = 64
N_HEADS = 24
N_KV = 4
THETA = 10_000_000.0
EPS = 1e-6

_IDX = None
_X_HIDDEN = {}
_PEAK = 0.0


def bump():
    global _PEAK
    r = rss_gb()
    if r == r and r > _PEAK:
        _PEAK = r
    return r


def idx():
    global _IDX
    if _IDX is None:
        _IDX = json.load(open(os.path.join(BF16, "model.safetensors.index.json")))
    return _IDX


def load_tensor(name):
    shard = idx()["weight_map"][name]
    with open(os.path.join(BF16, shard), "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
        base = 8 + hlen
        meta = hdr[name]
        s, e = meta["data_offsets"]
        f.seek(base + s)
        raw = f.read(e - s)
    dt = meta["dtype"]
    if dt == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        arr = u16.view(np.float32)
    elif dt == "F32":
        arr = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(dt)
    return arr.reshape(meta["shape"]).astype(np.float32, copy=True)


def load_X_hidden(layer):
    if layer not in _X_HIDDEN:
        res = json.load(open(os.path.join(CAPTURE, "capture-result.json")))
        per = res["per_layer"][str(layer)]
        X = np.fromfile(per["path"], dtype=np.float32).reshape(per["n_rows"], res["hidden"])
        _X_HIDDEN[layer] = np.ascontiguousarray(X)
    return _X_HIDDEN[layer]


def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -60, 60)))


def c_uniform(W, bits=4, group=GROUP):
    lim = (1 << (bits - 1)) - 1
    out, d = W.shape
    ng = d // group
    if ng == 0:
        return W.copy()
    Wg = W[:, : ng * group].reshape(out, ng, group)
    amax = np.max(np.abs(Wg), axis=2, keepdims=True) + np.float32(1e-30)
    step = amax / np.float32(lim)
    Q = np.clip(np.round(Wg / step), -lim, lim) * step
    Wh = W.copy()
    Wh[:, : ng * group] = Q.reshape(out, ng * group)
    return Wh


def fwht_last(x, stages=None):
    """Normalized FWHT on last axis. stages=None => full log2(n). Involutive if full."""
    x = np.array(x, dtype=np.float32, copy=True)
    n = x.shape[-1]
    if n == 0 or (n & (n - 1)):
        raise ValueError(f"fwht length {n} not power of 2")
    max_stages = int(np.log2(n))
    smax = max_stages if stages is None else stages
    if smax < 0 or smax > max_stages:
        raise ValueError(f"stages {smax} vs {max_stages}")
    h = 1
    for _ in range(smax):
        y = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a = y[..., 0, :].copy()
        b = y[..., 1, :].copy()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        h *= 2
    x *= np.float32(2.0 ** (-0.5 * smax))
    return x


def bfly_last(x, stages, inverse=False):
    """First `stages` FWHT butterflies. Inverse applies those stages in reverse."""
    x = np.array(x, dtype=np.float32, copy=True)
    n = x.shape[-1]
    hs = [1 << i for i in range(stages)]
    if inverse:
        hs = hs[::-1]
    for h in hs:
        y = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a = y[..., 0, :].copy()
        b = y[..., 1, :].copy()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
    x *= np.float32(2.0 ** (-0.5 * stages))
    return x


def block_apply(W, block, fn, axis=1):
    W = np.ascontiguousarray(W, dtype=np.float32)
    n = W.shape[axis]
    if n % block != 0:
        raise ValueError(f"axis{axis}={n} not divisible by {block}")
    if axis == 1:
        X = W.reshape(W.shape[0], n // block, block)
        Y = fn(X)
        return Y.reshape(W.shape)
    if axis == 0:
        X = W.reshape(n // block, block, W.shape[1])
        X = np.ascontiguousarray(np.moveaxis(X, 1, -1))
        Y = fn(X)
        Y = np.moveaxis(Y, -1, 1)
        return np.ascontiguousarray(Y.reshape(W.shape))
    raise ValueError(axis)


def signs_for(n, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)


def apply_named(W, name):
    """Return (Wt, meta). Inverse is apply_named_inv."""
    if name == "id":
        return W.copy(), {}
    if name.startswith("wh_"):
        b = int(name.split("_")[1])
        return block_apply(W, b, fwht_last, axis=1), {"block": b, "axis": 1}
    if name.startswith("out_wh_"):
        b = int(name.split("_")[2])
        return block_apply(W, b, fwht_last, axis=0), {"block": b, "axis": 0}
    if name.startswith("rht_"):
        b = int(name.split("_")[1])
        s = signs_for(W.shape[1], seed=0xC0FFEE + b)
        Wt = W * s
        Wt = block_apply(Wt, b, fwht_last, axis=1)
        return Wt, {"block": b, "axis": 1, "signs": s}
    if name.startswith("out_rht_"):
        b = int(name.split("_")[2])
        s = signs_for(W.shape[0], seed=0xBEEF + b)
        Wt = W * s[:, None]
        Wt = block_apply(Wt, b, fwht_last, axis=0)
        return Wt, {"block": b, "axis": 0, "signs": s}
    if name.startswith("bfly1_"):
        b = int(name.split("_")[1])
        return block_apply(W, b, lambda x: bfly_last(x, 1, False), axis=1), {"block": b, "stages": 1}
    if name == "kron_h1024_i5":
        if W.shape[1] != 5120:
            raise ValueError("kron only on d_in=5120")
        X = W.reshape(W.shape[0], 1024, 5)
        X = np.ascontiguousarray(np.moveaxis(X, 1, -1))
        Y = fwht_last(X)
        Y = np.moveaxis(Y, -1, 1)
        return Y.reshape(W.shape), {}
    raise ValueError(name)


def apply_named_inv(Wt, name, meta):
    if name == "id":
        return Wt.copy()
    if name.startswith("wh_"):
        return block_apply(Wt, meta["block"], fwht_last, axis=1)
    if name.startswith("out_wh_"):
        return block_apply(Wt, meta["block"], fwht_last, axis=0)
    if name.startswith("rht_"):
        X = block_apply(Wt, meta["block"], fwht_last, axis=1)
        return X * meta["signs"]
    if name.startswith("out_rht_"):
        X = block_apply(Wt, meta["block"], fwht_last, axis=0)
        return X * meta["signs"][:, None]
    if name.startswith("bfly1_"):
        return block_apply(Wt, meta["block"], lambda x: bfly_last(x, 1, True), axis=1)
    if name == "kron_h1024_i5":
        X = Wt.reshape(Wt.shape[0], 1024, 5)
        X = np.ascontiguousarray(np.moveaxis(X, 1, -1))
        Y = fwht_last(X)
        Y = np.moveaxis(Y, -1, 1)
        return Y.reshape(Wt.shape)
    raise ValueError(name)


def excess_kurtosis_1d(v):
    v = np.asarray(v, dtype=np.float64)
    m = v.mean()
    xc = v - m
    m2 = np.mean(xc * xc)
    m4 = np.mean(xc ** 4)
    return float(m4 / (m2 * m2 + 1e-30) - 3.0)


def group_stats(W, group=GROUP):
    out, d = W.shape
    ng = d // group
    X = W[:, : ng * group].reshape(out, ng, group).astype(np.float64, copy=False)
    mean = X.mean(axis=2, keepdims=True)
    xc = X - mean
    m2 = (xc * xc).mean(axis=2)
    m4 = (xc ** 4).mean(axis=2)
    kurt = m4 / (m2 * m2 + 1e-30) - 3.0
    ax = np.abs(X)
    amax = ax.max(axis=2)
    amed = np.median(ax, axis=2)
    arms = np.sqrt((X * X).mean(axis=2))
    dr = amax / (amed + 1e-30)
    crest = amax / (arms + 1e-30)
    return {
        "n_groups": int(out * ng),
        "kurt_mean": float(kurt.mean()),
        "kurt_p50": float(np.median(kurt)),
        "kurt_p99": float(np.quantile(kurt, 0.99)),
        "kurt_max": float(kurt.max()),
        "dr_mean": float(dr.mean()),
        "dr_p50": float(np.median(dr)),
        "dr_p99": float(np.quantile(dr, 0.99)),
        "dr_max": float(dr.max()),
        "crest_mean": float(crest.mean()),
        "crest_p99": float(np.quantile(crest, 0.99)),
        "crest_max": float(crest.max()),
    }


def row_energy_stats(W, special_row=None):
    rms = np.sqrt((W.astype(np.float64) ** 2).mean(axis=1))
    out = {
        "row_rms_kurtosis": excess_kurtosis_1d(rms),
        "row_rms_max": float(rms.max()),
        "row_rms_median": float(np.median(rms)),
        "row_rms_max_over_med": float(rms.max() / (np.median(rms) + 1e-30)),
        "row_rms_argmax": int(np.argmax(rms)),
    }
    if special_row is not None and 0 <= special_row < W.shape[0]:
        dropped = np.delete(rms, special_row)
        out["special_row"] = int(special_row)
        out["special_row_rms"] = float(rms[special_row])
        out["special_row_xmed"] = float(rms[special_row] / (np.median(rms) + 1e-30))
        out["row_rms_kurtosis_without_special"] = excess_kurtosis_1d(dropped)
        out["f_frac_special"] = float((rms[special_row] ** 2) / (np.sum(rms ** 2) + 1e-30))
    return out


def gemm_x_wt(X, W):
    """(tok, in) @ (out, in).T -> (tok, out)."""
    try:
        import torch
        with torch.no_grad():
            return torch.mm(
                torch.from_numpy(np.ascontiguousarray(X)),
                torch.from_numpy(np.ascontiguousarray(W)).t(),
            ).numpy()
    except Exception:
        return X @ W.T


# monkeypatch doctor_gate to use BLAS gemm
import gravity_doctor_gate as gdg

def _rowcos(A, B):
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-30
    return float(np.mean(num / den))

def observed_score_fast(W, Wh, X):
    return _rowcos(gemm_x_wt(X, W), gemm_x_wt(X, Wh))

def probed_score_fast(W, Wh, d_in, n=256, seed=0):
    P = gdg._probe(d_in, n, seed)
    return _rowcos(gemm_x_wt(P, W), gemm_x_wt(P, Wh))

def axes_fast(W, Wh, X, seed=0):
    P = gdg._probe(W.shape[1], seed=seed)
    Yw = gemm_x_wt(X, W)
    Yh = gemm_x_wt(X, Wh)
    Pw = gemm_x_wt(P, W)
    Ph = gemm_x_wt(P, Wh)
    return {
        "observed": _rowcos(Yw, Yh),
        "probed": _rowcos(Pw, Ph),
        "worst_unit": min(gdg._worst_unit(Yw, Yh), gdg._worst_unit(Pw, Ph)),
    }

def gate_fast(W, Wh, X, ref=None, seed=0):
    a = axes_fast(W, Wh, X, seed=seed)
    if ref is None:
        g = min(a.values())
        return {**a, "gate": g, "healthy": g >= gdg.PASS_THRESHOLD, "mode": "absolute"}
    deficits = {k: a[k] - (ref[k] - AXIS_MARGIN[k]) for k in a}
    worst = min(deficits, key=deficits.get)
    return {**a, "deficit": {k: float(v) for k, v in deficits.items()},
            "gate": float(deficits[worst]), "worst_axis": worst,
            "healthy": bool(deficits[worst] >= 0.0), "mode": "relative"}


def tname(layer, suffix):
    return f"language_model.model.layers.{layer}.{suffix}"


def mixer_kind(layer):
    return "gqa" if (layer + 1) % 4 == 0 else "deltanet"


def make_X(layer, suffix, W):
    """Activation for this GEMV. Returns (X, site_label)."""
    din = W.shape[1]
    Xh = load_X_hidden(layer)
    if suffix.endswith("mlp.down_proj.weight"):
        Wg = load_tensor(tname(layer, "mlp.gate_proj.weight"))
        Wu = load_tensor(tname(layer, "mlp.up_proj.weight"))
        X = silu(gemm_x_wt(Xh, Wg)) * gemm_x_wt(Xh, Wu)
        del Wg, Wu
        return np.ascontiguousarray(X), "swiglu_reconstructed"
    if suffix.endswith("linear_attn.out_proj.weight"):
        Wqkv = load_tensor(tname(layer, "linear_attn.in_proj_qkv.weight"))
        Wz = load_tensor(tname(layer, "linear_attn.in_proj_z.weight"))
        qkv = gemm_x_wt(Xh, Wqkv)
        z = gemm_x_wt(Xh, Wz)
        v = qkv[:, 4096:4096 + 6144]
        X = v * silu(z)
        del Wqkv, Wz, qkv, z, v
        return np.ascontiguousarray(X), "dn_mixer_proxy_v_silu_z"
    if suffix.endswith("self_attn.o_proj.weight"):
        Wq = load_tensor(tname(layer, "self_attn.q_proj.weight"))
        Wv = load_tensor(tname(layer, "self_attn.v_proj.weight"))
        q = gemm_x_wt(Xh, Wq).reshape(-1, N_HEADS, 512)
        v = gemm_x_wt(Xh, Wv).reshape(-1, N_KV, HEAD_DIM)
        gate = q[:, :, HEAD_DIM:]
        vrep = np.repeat(v, N_HEADS // N_KV, axis=1)
        sig = 1.0 / (1.0 + np.exp(-np.clip(gate, -60, 60)))
        X = (vrep * sig).reshape(Xh.shape[0], N_HEADS * HEAD_DIM)
        del Wq, Wv, q, v, gate, vrep, sig
        return np.ascontiguousarray(X), "gqa_mixer_proxy_vrep_sigmoid_qgate"
    if din == Xh.shape[1]:
        return Xh, "captured_post_norm_hidden"
    rng = np.random.default_rng(0)
    P = rng.standard_normal((256, din)).astype(np.float32)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return P, "isotropic_standin"


# ---------------- RoPE ----------------

def rotate_half_rope(x, positions, q_norm=None):
    """x: [T, H, 256]  q_norm: [256] or None. Matches qwen38_gqa_qk_norm_rope_cache_f32."""
    if q_norm is None:
        nrm = x
    else:
        rms = np.sqrt((x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True) + EPS)
        nrm = (x / rms.astype(np.float32)) * (1.0 + q_norm.astype(np.float32))
    out = nrm.copy()
    half = ROTARY // 2
    x1 = nrm[..., :half]
    x2 = nrm[..., half:ROTARY]
    freq = np.arange(half, dtype=np.float64)
    inv = THETA ** (-2.0 * freq / ROTARY)
    angle = positions.astype(np.float64)[:, None, None] * inv[None, None, :]
    c = np.cos(angle).astype(np.float32)
    s = np.sin(angle).astype(np.float32)
    out[..., :half] = x1 * c - x2 * s
    out[..., half:ROTARY] = x2 * c + x1 * s
    return out


def attn_scores(q, k):
    k_exp = np.repeat(k, N_HEADS // N_KV, axis=1)
    scale = np.float32(HEAD_DIM ** -0.5)
    return np.einsum("thd,shd->hts", q, k_exp, optimize=True) * scale


def score_cmp(S0, S1):
    a = S0.reshape(-1).astype(np.float64)
    b = S1.reshape(-1).astype(np.float64)
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    d = a - b
    return {
        "score_cosine": num / den,
        "max_abs_delta": float(np.max(np.abs(d))),
        "rms_delta": float(np.sqrt(np.mean(d * d))),
        "mean_abs_delta": float(np.mean(np.abs(d))),
    }


def apply_head_xform(x, kind, rng):
    """x [T,H,256] -> transformed in feature dim."""
    T, H, D = x.shape
    if kind == "id":
        return x.copy()
    if kind == "wh256":
        return fwht_last(x)
    if kind == "wh64_rotary":
        y = x.copy()
        y[..., :64] = fwht_last(y[..., :64])
        return y
    if kind == "wh64_nonrotary":
        y = x.copy()
        for s0 in (64, 128, 192):
            y[..., s0:s0 + 64] = fwht_last(y[..., s0:s0 + 64])
        return y
    if kind == "wh128_nonrotary_mid":
        y = x.copy()
        y[..., 64:192] = fwht_last(y[..., 64:192])
        return y
    if kind == "pair_wh2":
        y = x.copy()
        # H2 on (i, i+32) for i in 0..31
        a = y[..., :32]
        b = y[..., 32:64]
        s = np.float32(2.0 ** -0.5)
        y[..., :32] = (a + b) * s
        y[..., 32:64] = (a - b) * s
        return y
    if kind == "pair_rot45":
        y = x.copy()
        a = y[..., :32]
        b = y[..., 32:64]
        s = np.float32(2.0 ** -0.5)
        y[..., :32] = (a - b) * s
        y[..., 32:64] = (a + b) * s
        return y
    if kind == "signs_iid":
        sg = rng.choice(np.array([-1.0, 1.0], np.float32), size=D)
        return x * sg
    if kind == "signs_pair_const":
        s32 = rng.choice(np.array([-1.0, 1.0], np.float32), size=32)
        sg = np.ones(D, dtype=np.float32)
        sg[:32] = s32
        sg[32:64] = s32
        sg[64:] = rng.choice(np.array([-1.0, 1.0], np.float32), size=D - 64)
        return x * sg
    if kind == "signs_pair_flip":
        # opposite signs on RoPE partners — should fail commutation
        s32 = rng.choice(np.array([-1.0, 1.0], np.float32), size=32)
        sg = np.ones(D, dtype=np.float32)
        sg[:32] = s32
        sg[32:64] = -s32
        return x * sg
    if kind == "generic_orth256":
        A = rng.standard_normal((D, D)).astype(np.float32)
        Qm, R = np.linalg.qr(A)
        Qm = Qm * np.sign(np.diag(R))
        return np.einsum("thd,ed->the", x, Qm, optimize=True)
    if kind == "wh_across_heads":
        # mix all heads: treat as T x (H*D)
        flat = x.reshape(T, H * D)
        # 24*256 = 6144, 6144/1024=6
        y = block_apply(flat, 256, fwht_last, axis=1)
        return y.reshape(T, H, D)
    if kind == "mix_q_and_gate":
        # only valid when caller passes 512-wide
        return fwht_last(x)
    raise ValueError(kind)


def rope_battery(layer):
    Xh = load_X_hidden(layer)
    Wq = load_tensor(tname(layer, "self_attn.q_proj.weight"))
    Wk = load_tensor(tname(layer, "self_attn.k_proj.weight"))
    qn = load_tensor(tname(layer, "self_attn.q_norm.weight"))
    kn = load_tensor(tname(layer, "self_attn.k_norm.weight"))
    q_all = gemm_x_wt(Xh, Wq).reshape(-1, N_HEADS, 512)
    k = gemm_x_wt(Xh, Wk).reshape(-1, N_KV, HEAD_DIM)
    q = q_all[:, :, :HEAD_DIM]
    q_gate = q_all[:, :, HEAD_DIM:]
    T = q.shape[0]
    pos = np.arange(T, dtype=np.int32)
    q0 = rotate_half_rope(q, pos, qn)
    k0 = rotate_half_rope(k, pos, kn)
    S0 = attn_scores(q0, k0)

    kinds = [
        "id",
        "wh256",
        "wh64_rotary",
        "wh64_nonrotary",
        "wh128_nonrotary_mid",
        "pair_wh2",
        "pair_rot45",
        "signs_iid",
        "signs_pair_const",
        "signs_pair_flip",
        "generic_orth256",
        "wh_across_heads",
    ]
    rows = []
    for kind in kinds:
        rng = np.random.default_rng(12345)
        qt = apply_head_xform(q, kind, rng)
        rng = np.random.default_rng(12345)  # same H on k
        # k has 4 heads; apply_head_xform works
        if kind == "wh_across_heads":
            # flatten k as T x (4*256)=1024, block 256
            kflat = k.reshape(T, N_KV * HEAD_DIM)
            kt = block_apply(kflat, 256, fwht_last, axis=1).reshape(T, N_KV, HEAD_DIM)
        else:
            kt = apply_head_xform(k, kind, rng)
        q1 = rotate_half_rope(qt, pos, qn)
        k1 = rotate_half_rope(kt, pos, kn)
        S1 = attn_scores(q1, k1)
        cmp_ = score_cmp(S0, S1)
        # also: apply H AFTER rope (should preserve if H orthogonal and same on q,k)
        rng = np.random.default_rng(12345)
        q_post = apply_head_xform(q0, kind, rng)
        rng = np.random.default_rng(12345)
        if kind == "wh_across_heads":
            kflat = k0.reshape(T, N_KV * HEAD_DIM)
            k_post = block_apply(kflat, 256, fwht_last, axis=1).reshape(T, N_KV, HEAD_DIM)
        else:
            k_post = apply_head_xform(k0, kind, rng)
        Sp = attn_scores(q_post, k_post)
        cmp_post = score_cmp(S0, Sp)
        # q_norm commutation: ||H x|| vs ||x|| (RMS)
        rms0 = np.sqrt((q.astype(np.float64) ** 2).mean(-1))
        rms1 = np.sqrt((qt.astype(np.float64) ** 2).mean(-1))
        rows.append({
            "kind": kind,
            "pre_rope": cmp_,
            "post_rope": cmp_post,
            "rms_rel_max": float(np.max(np.abs(rms1 - rms0) / (rms0 + 1e-30))),
            "pre_breaks": bool(cmp_["max_abs_delta"] > 1e-4),
            "post_preserves": bool(cmp_post["max_abs_delta"] < 1e-4),
        })
        print(f"  ROPE L{layer} {kind:<22} pre_cos={cmp_['score_cosine']:.8f} "
              f"pre_maxd={cmp_['max_abs_delta']:.3e} post_maxd={cmp_post['max_abs_delta']:.3e} "
              f"{'BREAKS' if cmp_['max_abs_delta']>1e-4 else 'safe'}", flush=True)

    # mix query half with gate half (512-wide WH per head) — q only
    q512 = q_all.copy()
    q512_t = fwht_last(q512)  # last axis 512
    q_mixed = q512_t[:, :, :HEAD_DIM]
    q1 = rotate_half_rope(q_mixed, pos, qn)
    S1 = attn_scores(q1, k0)
    cmp_mix = score_cmp(S0, S1)
    rows.append({
        "kind": "wh512_mix_query_and_gate",
        "pre_rope": cmp_mix,
        "post_rope": None,
        "rms_rel_max": None,
        "pre_breaks": True,
        "post_preserves": None,
        "note": "mixes RoPE query half with attn-output-gate half of q_proj",
    })
    print(f"  ROPE L{layer} wh512_mix_query_and_gate pre_cos={cmp_mix['score_cosine']:.8f} "
          f"pre_maxd={cmp_mix['max_abs_delta']:.3e} BREAKS", flush=True)

    # input-side only: q,k unchanged
    rows.append({
        "kind": "input_side_only",
        "pre_rope": {"score_cosine": 1.0, "max_abs_delta": 0.0, "rms_delta": 0.0, "mean_abs_delta": 0.0},
        "post_rope": None,
        "pre_breaks": False,
        "post_preserves": True,
        "note": "W' = W H_in leaves q,k in original basis",
    })

    del Wq, Wk, qn, kn, q_all, k, q, q_gate, q0, k0, S0
    return rows


def deltanet_l2_battery(layer):
    """Does output-side WH preserve per-head Q/K L2 (used after conv)?"""
    Xh = load_X_hidden(layer)
    W = load_tensor(tname(layer, "linear_attn.in_proj_qkv.weight"))
    y = gemm_x_wt(Xh, W)  # (T, 10240)
    # layout: Q 2048 = 16*128, K 2048, V 6144
    def head_l2(vec, n_heads, dim, off):
        sl = vec[:, off:off + n_heads * dim].reshape(-1, n_heads, dim)
        return np.sqrt((sl.astype(np.float64) ** 2).sum(-1))  # (T, heads)

    q0 = head_l2(y, 16, 128, 0)
    k0 = head_l2(y, 16, 128, 2048)
    results = []
    for name in ("out_wh_128", "out_wh_256", "out_wh_1024"):
        Wt, meta = apply_named(W, name)
        yt = gemm_x_wt(Xh, Wt)
        q1 = head_l2(yt, 16, 128, 0)
        k1 = head_l2(yt, 16, 128, 2048)
        rec = {
            "xform": name,
            "q_l2_rel_max": float(np.max(np.abs(q1 - q0) / (q0 + 1e-30))),
            "k_l2_rel_max": float(np.max(np.abs(k1 - k0) / (k0 + 1e-30))),
            "q_l2_cosine": float(np.mean(
                np.sum(q0 * q1, 1) / (np.linalg.norm(q0, 1) * np.linalg.norm(q1, 1) + 1e-30))),
        }
        rec["preserves_head_l2"] = rec["q_l2_rel_max"] < 1e-4 and rec["k_l2_rel_max"] < 1e-4
        results.append(rec)
        print(f"  DN-L2 L{layer} {name} q_rel={rec['q_l2_rel_max']:.3e} "
              f"k_rel={rec['k_l2_rel_max']:.3e} "
              f"{'preserves' if rec['preserves_head_l2'] else 'BREAKS'}", flush=True)
        del Wt, yt
    del W, y
    return results


# ---------------- sweep ----------------

SITES = [
    (0, "mlp.gate_proj.weight", "gate"),
    (0, "mlp.down_proj.weight", "down"),
    (0, "linear_attn.in_proj_qkv.weight", "in_qkv"),
    (0, "linear_attn.out_proj.weight", "lin_o"),
    (3, "mlp.gate_proj.weight", "gate"),
    (3, "self_attn.q_proj.weight", "q"),
    (3, "self_attn.k_proj.weight", "k"),
    (3, "self_attn.v_proj.weight", "v"),
    (3, "self_attn.o_proj.weight", "o"),
    (15, "mlp.gate_proj.weight", "gate"),
    (15, "mlp.down_proj.weight", "down"),
    (31, "mlp.gate_proj.weight", "gate"),
    (31, "mlp.down_proj.weight", "down"),
    (31, "self_attn.q_proj.weight", "q"),
    (31, "self_attn.k_proj.weight", "k"),
    (31, "self_attn.v_proj.weight", "v"),
    (31, "self_attn.o_proj.weight", "o"),
    (32, "linear_attn.in_proj_qkv.weight", "in_qkv"),
    (32, "linear_attn.out_proj.weight", "lin_o"),
    (47, "self_attn.q_proj.weight", "q"),
    (63, "mlp.gate_proj.weight", "gate"),
    (63, "mlp.down_proj.weight", "down"),
    (63, "self_attn.q_proj.weight", "q"),
    (63, "self_attn.o_proj.weight", "o"),
]

KURT_XFORMS = [
    "id", "wh_32", "wh_64", "wh_128", "wh_256", "wh_512", "wh_1024",
    "rht_128", "rht_256", "bfly1_128", "bfly1_256",
    "out_wh_128", "out_wh_256", "out_wh_1024", "out_rht_256",
    "kron_h1024_i5",
]
GATE_XFORMS = ["id", "wh_128", "wh_256", "wh_1024", "rht_256", "bfly1_128", "out_wh_256"]
BITS = [2, 3, 4]


def dump(obj):
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, OUT_JSON)


def self_check():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 256)).astype(np.float32)
    y = fwht_last(fwht_last(x))
    inv_err = float(np.max(np.abs(y - x)))
    # orthogonality: row dots
    H = fwht_last(np.eye(128, dtype=np.float32))
    gram = H @ H.T
    orth_err = float(np.max(np.abs(gram - np.eye(128))))
    # bfly1 involution
    z = rng.standard_normal((3, 128)).astype(np.float32)
    z2 = bfly_last(bfly_last(z, 1, False), 1, True)
    bfly_err = float(np.max(np.abs(z2 - z)))
    # c_uniform vs loop on tiny
    W = rng.standard_normal((7, 256)).astype(np.float32)
    # compare to doctor
    ref = gdg.c_uniform(W, 4, 128)
    mine = c_uniform(W, 4, 128)
    q_err = float(np.max(np.abs(ref - mine)))
    # named inv
    Wt, meta = apply_named(W, "rht_128")
    back = apply_named_inv(Wt, "rht_128", meta)
    rht_err = float(np.max(np.abs(back - W)))
    Wt, meta = apply_named(W, "wh_128")
    back = apply_named_inv(Wt, "wh_128", meta)
    wh_err = float(np.max(np.abs(back - W)))
    out = {
        "fwht_involution_maxabs": inv_err,
        "fwht_orth_maxabs": orth_err,
        "bfly1_involution_maxabs": bfly_err,
        "c_uniform_vs_doctor_maxabs": q_err,
        "rht_roundtrip_maxabs": rht_err,
        "wh_roundtrip_maxabs": wh_err,
        "ok": max(inv_err, orth_err, bfly_err, q_err, rht_err, wh_err) < 2e-5,
    }
    print("SELF", json.dumps(out), flush=True)
    return out


def process_site(layer, suffix, cls):
    name = tname(layer, suffix)
    t0 = time.perf_counter()
    W = load_tensor(name)
    X, x_site = make_X(layer, suffix, W)
    bump()
    print(f"SITE L{layer} {cls} {tuple(W.shape)} X={tuple(X.shape)} {x_site} rss={rss_gb():.3f}",
          flush=True)
    special = 3994 if (cls in ("lin_o", "o", "down") and W.shape[0] == 5120) else None
    base_g = group_stats(W)
    base_r = row_energy_stats(W, special)
    # Q4 reference
    Wq4 = c_uniform(W, 4, GROUP)
    ref = axes_fast(W, Wq4, X)
    ref = {k: float(ref[k]) for k in ref}
    print(f"  Q4 ref obs={ref['observed']:.6f} prb={ref['probed']:.6f} wu={ref['worst_unit']:.6f}",
          flush=True)

    kurt_rows = {}
    for xf in KURT_XFORMS:
        try:
            Wt, meta = apply_named(W, xf)
        except ValueError:
            continue
        st = group_stats(Wt)
        st.update({k: v for k, v in row_energy_stats(Wt, special).items()})
        # don't keep signs in kurt dump
        kurt_rows[xf] = st
        del Wt, meta
    print(f"  kurt id={base_g['kurt_mean']:.4f}  "
          + " ".join(
              f"{k}={kurt_rows[k]['kurt_mean']:.3f}"
              for k in ("wh_128", "wh_256", "rht_256", "out_wh_256")
              if k in kurt_rows
          ),
          flush=True)

    gate_rows = {}
    for xf in GATE_XFORMS:
        try:
            Wt, meta = apply_named(W, xf)
        except ValueError:
            continue
        gate_rows[xf] = {}
        for bits in BITS:
            Q = c_uniform(Wt, bits, GROUP)
            Wh = apply_named_inv(Q, xf, meta)
            g = gate_fast(W, Wh, X, ref=ref)
            rec = {
                "observed": float(g["observed"]),
                "probed": float(g["probed"]),
                "worst_unit": float(g["worst_unit"]),
                "gate": float(g["gate"]),
                "worst_axis": g["worst_axis"],
                "healthy": bool(g["healthy"]),
                "deficit": {k: float(v) for k, v in g["deficit"].items()},
            }
            gate_rows[xf][str(bits)] = rec
            print(f"  GATE {xf:12s} Q{bits} obs={rec['observed']:.6f} prb={rec['probed']:.6f} "
                  f"wu={rec['worst_unit']:.6f} m={rec['gate']:+.5f} "
                  f"{'HEALTHY' if rec['healthy'] else 'UNHEALTHY'} ({rec['worst_axis']})",
                  flush=True)
            del Q, Wh
        del Wt, meta
        gc.collect()

    rec = {
        "tensor": name,
        "layer": layer,
        "class": cls,
        "mixer": mixer_kind(layer),
        "shape": [int(W.shape[0]), int(W.shape[1])],
        "x_site": x_site,
        "x_shape": [int(X.shape[0]), int(X.shape[1])],
        "q4_ref": ref,
        "id_group": base_g,
        "id_row": base_r,
        "kurtosis": kurt_rows,
        "gate": gate_rows,
        "wall_s": time.perf_counter() - t0,
        "rss_gb": rss_gb(),
    }
    del W, Wq4, X
    gc.collect()
    bump()
    return rec


def main():
    t_all = time.perf_counter()
    out = {
        "schema": "hawking.gravity1.xform_hadamard.v1",
        "bf16": BF16,
        "capture": CAPTURE,
        "N": N_LANG,
        "group": GROUP,
        "axis_margin": AXIS_MARGIN,
        "self_check": self_check(),
        "sites": [],
        "rope": {},
        "deltanet_l2": {},
        "peak_rss_gb": 0.0,
    }
    dump(out)
    if not out["self_check"]["ok"]:
        print("SELF-CHECK FAILED", out["self_check"], flush=True)

    # geometry facts (no tensor load)
    out["geometry"] = {
        "hidden": 5120,
        "hidden_sylvester": False,
        "hidden_factors": "5 * 1024",
        "longest_walsh_on_hidden": 1024,
        "head_dim": 256,
        "rotary_dim": 64,
        "partial_rotary_factor": 0.25,
        "rope_style": "rotate_half first 64 of 256, peer=dim±32, theta=1e7",
        "q_proj_rows": 12288,
        "q_proj_layout": "24 heads * (256 query + 256 attn_output_gate)",
        "attn_output_gate": "sigmoid (shipping metal) / config output_gate_type=swish",
        "gqa": "24:4",
        "full_attention_interval": 4,
        "dn_qkv_rows": 10240,
        "dn_qkv_layout": "Q 16*128 + K 16*128 + V 48*128",
        "metal_rht": "strand_rht_forward_cols 256-wide, in_features % 256 == 0",
    }

    for layer, suffix, cls in SITES:
        try:
            rec = process_site(layer, suffix, cls)
            out["sites"].append(rec)
        except Exception as e:
            traceback.print_exc()
            out["sites"].append({
                "tensor": tname(layer, suffix),
                "layer": layer,
                "class": cls,
                "error": repr(e),
            })
        out["peak_rss_gb"] = _PEAK
        out["wall_s"] = time.perf_counter() - t_all
        dump(out)

    for layer in (3, 31, 63):
        print(f"ROPE battery L{layer}", flush=True)
        try:
            out["rope"][str(layer)] = rope_battery(layer)
        except Exception as e:
            traceback.print_exc()
            out["rope"][str(layer)] = {"error": repr(e)}
        out["peak_rss_gb"] = _PEAK
        dump(out)
        gc.collect()

    for layer in (0, 32):
        print(f"DN L2 battery L{layer}", flush=True)
        try:
            out["deltanet_l2"][str(layer)] = deltanet_l2_battery(layer)
        except Exception as e:
            traceback.print_exc()
            out["deltanet_l2"][str(layer)] = {"error": repr(e)}
        out["peak_rss_gb"] = _PEAK
        dump(out)
        gc.collect()

    out["peak_rss_gb"] = _PEAK
    out["wall_s"] = time.perf_counter() - t_all
    dump(out)
    print(f"DONE wall={out['wall_s']:.1f}s peak_rss={_PEAK:.3f}GB -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
