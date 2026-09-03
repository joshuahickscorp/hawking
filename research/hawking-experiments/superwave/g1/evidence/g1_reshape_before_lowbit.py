#!/usr/bin/env python3
"""Reshape-before-lowbit measurement. CPU numpy only. No GPU.

Scores binary / ternary / q2 with the doctor adequacy gate on real Qwen3.8
tensors, before and after reconditioning that preserves the function
exactly or nearly exactly.
"""
from __future__ import annotations

import json, os, resource, struct, sys, time
from collections import OrderedDict

import numpy as np

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "Users")  # unused; we inject via sys.path below

HERE = "/Users/scammermike/.claude-grok/worktrees/209-reshape-before-lowbit-20260817-181049"
sys.path.insert(0, os.path.join(HERE, "tools"))

from gravity_doctor_gate import (  # noqa: E402
    AXIS_MARGIN, _probe, _rowcos, _worst_unit, c_uniform, load_tensor,
)

BF16 = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
CAP1 = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
CAP2 = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
N_SOURCE = 26_895_998_464
GROUP = 128
OUT_JSON = "/tmp/g1_reshape_before_lowbit.json"
SITE_W = {
    "post_swiglu": 17408,
    "post_attn_norm": 5120,
    "post_input_norm": 5120,
    "mixer_x": 6144,
}


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def now():
    return time.perf_counter()


def f16(x):
    return np.asarray(x).astype(np.float16).astype(np.float32)


def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def load_W(name):
    return load_tensor(name, root=BF16)


def load_X_v1(layer):
    res = json.load(open(os.path.join(CAP1, "capture-result.json")))
    per = res["per_layer"][str(layer)]
    X = np.fromfile(per["path"], dtype=np.float32).reshape(per["n_rows"], res["hidden"])
    return X


_CAP2 = None


def cap2():
    global _CAP2
    if _CAP2 is None:
        _CAP2 = json.load(open(os.path.join(CAP2, "capture-result.json")))
    return _CAP2


def swiglu_split():
    """Official prompt-level fit/hold row indices on the full 23216 post_swiglu stream."""
    d = cap2()
    fit, hold, row = [], [], 0
    for pr in d["prompts"]:
        idx = range(row, row + pr["n_tokens"])
        (fit if pr["split"] == "fit" else hold).extend(idx)
        row += pr["n_tokens"]
    return np.asarray(fit, np.int64), np.asarray(hold, np.int64)


def load_site_rows(site, layer, idx):
    width = SITE_W[site]
    path = os.path.join(CAP2, site, f"L{layer:02d}.f16")
    nbytes = os.path.getsize(path)
    n = nbytes // (2 * width)
    raw = np.fromfile(path, dtype=np.float16).reshape(n, width)
    idx = np.asarray(idx)
    idx = idx[idx < n]
    out = raw[idx].astype(np.float32)
    del raw
    return out


def take_prefix(idx, n):
    return np.asarray(idx)[:n]


def group_view(W, g=GROUP):
    o, d = W.shape
    if d % g != 0:
        raise ValueError(f"in_dim {d} not divisible by {g}")
    return W.reshape(o, d // g, g)


def codec_binary(W, g=GROUP):
    G = group_view(W, g)
    scales = f16(np.abs(G).mean(axis=-1, dtype=np.float64))
    signs = np.where(G >= 0.0, 1.0, -1.0)
    return (signs * scales[..., None]).reshape(W.shape)


def codec_ternary(W, tmul=0.7, g=GROUP):
    G = group_view(W, g)
    base = np.abs(G).mean(axis=-1, dtype=np.float64).astype(np.float32)
    thr = f16(base * tmul)
    active = np.abs(G) >= thr[..., None]
    sel = np.where(active, np.abs(G), 0.0)
    cnt = np.maximum(active.sum(axis=-1), 1)
    scales = f16(sel.sum(axis=-1) / cnt)
    codes = np.where(active, np.where(G >= 0.0, 1.0, -1.0), 0.0)
    return (codes * scales[..., None]).reshape(W.shape)


def codec_uniform(W, bits, g=GROUP):
    """Vectorized twin of gravity_doctor_gate.c_uniform (f32 scale, +1e-30 amax)."""
    G = group_view(W, g)
    lim = (1 << (bits - 1)) - 1
    amax = np.abs(G).max(axis=-1) + 1e-30
    step = amax / lim
    q = np.clip(np.round(G / step[..., None]), -lim, lim)
    return (q * step[..., None]).reshape(W.shape)


def codec_ternary_bestt(W, tmuls=(0.35, 0.50, 0.70, 0.90, 1.15), g=GROUP):
    """Per-group threshold multiplier chosen by weight-space MSE. Pack-time, no X."""
    G = group_view(W, g)
    base = np.abs(G).mean(axis=-1, dtype=np.float64).astype(np.float32)
    best_err = None
    best = None
    for tm in tmuls:
        thr = f16(base * tm)
        active = np.abs(G) >= thr[..., None]
        sel = np.where(active, np.abs(G), 0.0)
        cnt = np.maximum(active.sum(axis=-1), 1)
        scales = f16(sel.sum(axis=-1) / cnt)
        codes = np.where(active, np.where(G >= 0.0, 1.0, -1.0), 0.0)
        Wq = codes * scales[..., None]
        err = ((G - Wq) ** 2).sum(axis=-1)
        if best is None:
            best_err, best = err, Wq
        else:
            take = err < best_err
            best_err = np.where(take, err, best_err)
            best = np.where(take[..., None], Wq, best)
    return best.reshape(W.shape)


def codec_ternary_anneal(W, tmul=0.7, g=GROUP, taus=(1.0, 0.45, 0.18, 0.07, 0.025, 0.008)):
    """Soft-to-hard ternary: 3-level softmax on squared distance, anneal tau, LS scale."""
    G = group_view(W, g).astype(np.float32)
    base = np.abs(G).mean(axis=-1, dtype=np.float64).astype(np.float32)
    s = np.maximum(base, 1e-12)[..., None]
    for tau in taus:
        d0 = G * G
        dp = (G - s) ** 2
        dn = (G + s) ** 2
        m = np.minimum(d0, np.minimum(dp, dn))
        inv = 1.0 / max(tau, 1e-8)
        e0 = np.exp(-(d0 - m) * inv)
        ep = np.exp(-(dp - m) * inv)
        en = np.exp(-(dn - m) * inv)
        z = e0 + ep + en + 1e-30
        unit = (ep - en) / z
        num = (G * unit).sum(axis=-1, keepdims=True)
        den = (unit * unit).sum(axis=-1, keepdims=True) + 1e-30
        s = np.maximum(np.abs(num / den), 1e-12)
    # hard snap at annealed scale, threshold from original mean-abs (HGRAVT01 geometry)
    thr = f16(base * tmul)[..., None]
    active = np.abs(G) >= thr
    unit_h = np.where(active, np.where(G >= 0.0, 1.0, -1.0), 0.0)
    num = (G * unit_h).sum(axis=-1, keepdims=True)
    den = (unit_h * unit_h).sum(axis=-1, keepdims=True) + 1e-30
    s_h = f16(np.abs(num / den).squeeze(-1))
    return (unit_h * s_h[..., None]).reshape(W.shape)


def actls_scales(W, X, codes, g=GROUP):
    """Per-row-group least-squares scale against X. codes same shape as W, in {-1,0,1} or {±1}."""
    o, d = W.shape
    ng = d // g
    G = W.reshape(o, ng, g)
    C = codes.reshape(o, ng, g)
    scales = np.empty((o, ng), dtype=np.float32)
    for j in range(ng):
        Xg = X[:, j * g:(j + 1) * g]
        Yt = Xg @ G[:, j, :].T
        Yc = Xg @ C[:, j, :].T
        num = (Yt * Yc).sum(axis=0)
        den = (Yc * Yc).sum(axis=0) + 1e-30
        scales[:, j] = (num / den).astype(np.float32)
    return f16(scales)


def apply_actls(W, X, kind, tmul=0.7, g=GROUP):
    G = group_view(W, g)
    if kind == "binary":
        C = np.where(G >= 0.0, 1.0, -1.0)
    elif kind == "ternary":
        base = np.abs(G).mean(axis=-1, dtype=np.float64).astype(np.float32)
        thr = f16(base * tmul)
        active = np.abs(G) >= thr[..., None]
        C = np.where(active, np.where(G >= 0.0, 1.0, -1.0), 0.0)
    else:
        raise ValueError(kind)
    s = actls_scales(W, X, C.reshape(W.shape), g=g)
    return (C * s[..., None]).reshape(W.shape)


def col_scales(W, X=None, alpha=0.0, clip=8.0):
    """Per-input-channel scale. alpha=0: 1/w_rms (pure col-eq). 0.5 SmoothQuant. 1.0 AWQ."""
    w_rms = np.sqrt(np.mean(W.astype(np.float64) ** 2, axis=0)) + 1e-12
    if X is None or alpha <= 0.0:
        s = 1.0 / w_rms
    else:
        x_rms = np.sqrt(np.mean(X.astype(np.float64) ** 2, axis=0)) + 1e-12
        s = (x_rms ** alpha) / (w_rms ** (1.0 - alpha))
    s = s / np.exp(np.mean(np.log(s)))
    s = np.clip(s, 1.0 / clip, clip).astype(np.float32)
    return s


def fold_col(W, s):
    return (W / s[None, :]).astype(np.float32)


def unfold_col(Wq, s):
    return (Wq * s[None, :]).astype(np.float32)


class Prep:
    __slots__ = ("X", "P", "Yx", "Yp")

    def __init__(self, W, X, seed=0):
        self.X = X
        self.P = _probe(W.shape[1], 256, seed)
        self.Yx = X @ W.T
        self.Yp = self.P @ W.T


def score(prep: Prep, Wh):
    Yxh = prep.X @ Wh.T
    Yph = prep.P @ Wh.T
    a = {
        "observed": _rowcos(prep.Yx, Yxh),
        "probed": _rowcos(prep.Yp, Yph),
        "worst_unit": min(_worst_unit(prep.Yx, Yxh), _worst_unit(prep.Yp, Yph)),
    }
    return a


def vs_ref(a, ref):
    deficits = {k: a[k] - (ref[k] - AXIS_MARGIN[k]) for k in a}
    worst = min(deficits, key=deficits.get)
    return {
        **a,
        "deficit": deficits,
        "gate": deficits[worst],
        "worst_axis": worst,
        "healthy": deficits[worst] >= 0.0,
    }


def bytes_of(kind, elements, g=GROUP):
    groups = elements // g
    if kind == "binary":
        return (elements + 7) // 8 + groups * 2 + 40
    if kind == "ternary":
        return (elements * 2 + 7) // 8 + groups * 2 + groups * 2 + 40  # codes + scale + threshold
    if kind.startswith("q"):
        bits = int(kind[1:])
        return (elements * bits + 7) // 8 + groups * 2 + 40
    raise ValueError(kind)


def rec(a):
    out = {}
    for k, v in a.items():
        if k in ("meta",):
            continue
        if k == "worst_axis":
            out[k] = v
        elif isinstance(v, dict):
            out[k] = {kk: float(vv) for kk, vv in v.items()}
        elif isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        else:
            out[k] = float(v)
    return out


# --------------------------------------------------------------------------- tensors
JOBS = [
    # (layer, cls, site, tag)
    (0,  "mlp.down_proj",              "post_swiglu",      "down"),
    (31, "mlp.down_proj",              "post_swiglu",      "down"),
    (58, "mlp.down_proj",              "post_swiglu",      "down"),
    (62, "mlp.down_proj",              "post_swiglu",      "down"),
    (63, "mlp.down_proj",              "post_swiglu",      "down"),
    (0,  "mlp.gate_proj",              "post_attn_norm",   "gate"),
    (31, "mlp.gate_proj",              "post_attn_norm",   "gate"),
    (63, "mlp.gate_proj",              "post_attn_norm",   "gate"),
    (31, "self_attn.q_proj",           "post_input_norm",  "q"),
    (0,  "linear_attn.out_proj",       "mixer_x",          "out"),
]


def tname(layer, cls):
    return f"language_model.model.layers.{layer}.{cls}.weight"


def site_xy(site, layer, n_fit, n_hold):
    if site == "post_swiglu":
        fit_i, hold_i = swiglu_split()
        Xf = load_site_rows(site, layer, take_prefix(fit_i, n_fit))
        Xh = load_site_rows(site, layer, take_prefix(hold_i, n_hold))
        return Xf, Xh, {"fit_pool": int(fit_i.size), "hold_pool": int(hold_i.size),
                        "n_fit": int(Xf.shape[0]), "n_hold": int(Xh.shape[0]),
                        "split": "v2_prompt_official"}
    path = os.path.join(CAP2, site, f"L{layer:02d}.f16")
    width = SITE_W[site]
    n = os.path.getsize(path) // (2 * width)
    # stored prefix: last 25% hold, earlier rows fit. Deterministic, no shuffle.
    n_h = min(n_hold, max(1, n // 4))
    n_f = min(n_fit, n - n_h)
    raw = np.fromfile(path, dtype=np.float16).reshape(n, width)
    Xf = raw[:n_f].astype(np.float32)
    Xh = raw[n - n_h:].astype(np.float32)
    del raw
    return Xf, Xh, {"fit_pool": int(n - n_h), "hold_pool": int(n_h),
                    "n_fit": int(Xf.shape[0]), "n_hold": int(Xh.shape[0]),
                    "split": "v2_stored_prefix_tail25"}


def reproduce_v1_cites():
    """Reproduce mlp-floor L58 binary / L62 q2 hold-output-cosine (last 64 of v1 256)."""
    out = {}
    for layer, cls, codec_name in (
        (58, "mlp.down_proj", "binary"),
        (62, "mlp.down_proj", "q2_g64"),
        (0, "mlp.down_proj", "binary"),
    ):
        t0 = now()
        Xh = load_X_v1(layer)
        Wg = load_W(tname(layer, "mlp.gate_proj"))
        Wu = load_W(tname(layer, "mlp.up_proj"))
        inter = silu(Xh @ Wg.T) * (Xh @ Wu.T)
        del Wg, Wu
        Wd = load_W(tname(layer, cls))
        inter_hold = inter[192:]
        if codec_name == "binary":
            Wq = codec_binary(Wd)
        else:
            Wq = codec_uniform(Wd, 2, g=64)
        y = inter_hold @ Wd.T
        yq = inter_hold @ Wq.T
        cos = _rowcos(y, yq)
        out[f"L{layer}_{codec_name}"] = {
            "hold_output_cosine": float(cos),
            "n_hold": 64,
            "wall_s": float(now() - t0),
            "protocol": "v1_hidden + silu(X@Wg.T)*(X@Wu.T), last 64 of 256",
        }
        print(f"REPRO L{layer} {codec_name} hold_cos={cos:.12f} wall={now()-t0:.2f}s rss={rss_gb():.3f}",
              flush=True)
        del Xh, inter, inter_hold, Wd, Wq, y, yq
    return out


def identity_recond_err(W, X, s):
    y0 = X @ W.T
    y1 = (X * s[None, :]) @ (W / s[None, :]).T
    return {
        "max_abs": float(np.max(np.abs(y0 - y1))),
        "rel_rms": float(np.linalg.norm(y0 - y1) / (np.linalg.norm(y0) + 1e-30)),
    }


def run_tensor(layer, cls, site, tag, n_fit=256, n_hold=256, extra=False):
    name = tname(layer, cls)
    t_load = now()
    W = load_W(name)
    Xf, Xh, split = site_xy(site, layer, n_fit, n_hold)
    load_s = now() - t_load
    assert Xh.shape[1] == W.shape[1], (Xh.shape, W.shape, name)
    prep = Prep(W, Xh)
    ref = vs_ref(score(prep, codec_uniform(W, 4, GROUP)), {"observed": 1, "probed": 1, "worst_unit": 1})
    # real ref = Q4 axes themselves
    q4 = score(prep, codec_uniform(W, 4, GROUP))
    ref = q4

    rows = OrderedDict()
    rows["q4_g128"] = vs_ref(q4, ref)

    def add(key, Wh, wall, meta=None):
        a = vs_ref(score(prep, Wh), ref)
        a["wall_s"] = float(wall)
        if meta:
            a["meta"] = meta
        rows[key] = a
        print(f"  {key:<28} obs={a['observed']:.6f} prb={a['probed']:.6f} "
              f"wu={a['worst_unit']:.6f} gate={a['gate']:+.6f} "
              f"{'HEALTHY' if a['healthy'] else 'UNHEALTHY'} {wall:.2f}s",
              flush=True)

    # RAW
    for key, fn in (
        ("binary_raw", lambda: codec_binary(W)),
        ("ternary_raw", lambda: codec_ternary(W)),
        ("q2_raw", lambda: codec_uniform(W, 2, GROUP)),
    ):
        t0 = now(); Wh = fn(); add(key, Wh, now() - t0)

    # column-scale families (exact fold-back)
    for rname, alpha in (("col_eq", 0.0), ("sq05", 0.5), ("awq1", 1.0)):
        if rname == "awq1" and tag not in ("down", "out"):
            continue
        t0 = now()
        s = col_scales(W, Xf, alpha=alpha)
        iderr = identity_recond_err(W, Xh, s)
        Wn = fold_col(W, s)
        for cname, cfn in (
            ("binary", lambda M: codec_binary(M)),
            ("ternary", lambda M: codec_ternary(M)),
            ("q2", lambda M: codec_uniform(M, 2, GROUP)),
        ):
            t1 = now()
            Wq = unfold_col(cfn(Wn), s)
            add(f"{cname}_{rname}", Wq, now() - t1,
                meta={"s_min": float(s.min()), "s_max": float(s.max()),
                      "s_p99": float(np.quantile(s, 0.99)),
                      "id_rel_rms": iderr["rel_rms"], "id_max_abs": iderr["max_abs"],
                      "alpha": alpha})
        del Wn

    # activation-aware group scale (learned modulation, no W reshape)
    for cname, kind in (("binary", "binary"), ("ternary", "ternary")):
        t0 = now()
        Wh = apply_actls(W, Xf, kind)
        add(f"{cname}_actls", Wh, now() - t0, meta={"n_fit": int(Xf.shape[0])})

    # q2 act-aware: per-group MSE scale grid around absmax (8 multipliers)
    t0 = now()
    Wh = q2_act_mse(W, Xf)
    add("q2_actmse", Wh, now() - t0, meta={"n_fit": int(Xf.shape[0])})

    # annealed / best-t ternary (weight space) + optional actls
    if tag in ("down", "gate", "out") or extra:
        t0 = now(); Wh = codec_ternary_anneal(W); add("ternary_anneal", Wh, now() - t0)
        t0 = now(); Wh = codec_ternary_bestt(W); add("ternary_bestt", Wh, now() - t0)
        t0 = now()
        # anneal then replace scale with act-LS, keep annealed codes
        Ga = group_view(codec_ternary_anneal(W))
        codes = np.sign(Ga)
        codes[Ga == 0] = 0
        s = actls_scales(W, Xf, codes.reshape(W.shape))
        Wh = (codes * s[..., None]).reshape(W.shape)
        add("ternary_anneal_actls", Wh, now() - t0, meta={"n_fit": int(Xf.shape[0])})

        # col_eq then ternary_actls (combo)
        t0 = now()
        s_col = col_scales(W, Xf, alpha=0.0)
        Wn = fold_col(W, s_col)
        Xn = Xf * s_col[None, :]
        Wq = unfold_col(apply_actls(Wn, Xn, "ternary"), s_col)
        add("ternary_col_eq_actls", Wq, now() - t0)
        Wq = unfold_col(apply_actls(Wn, Xn, "binary"), s_col)
        add("binary_col_eq_actls", Wq, now() - t0)
        # sq05 + actls
        s_sq = col_scales(W, Xf, alpha=0.5)
        Wn = fold_col(W, s_sq)
        Xn = Xf * s_sq[None, :]
        Wq = unfold_col(apply_actls(Wn, Xn, "ternary"), s_sq)
        add("ternary_sq05_actls", Wq, now() - t0)
        Wq = unfold_col(apply_actls(Wn, Xn, "binary"), s_sq)
        add("binary_sq05_actls", Wq, now() - t0)

    # gamma / up absorption witness (cheap stats)
    absorb = {}
    if tag == "gate":
        gamma = load_W(f"language_model.model.layers.{layer}.post_attention_layernorm.weight")
        s = col_scales(W, Xf, alpha=0.0)
        gn = gamma * s
        absorb = {
            "kind": "post_attention_layernorm.weight * s",
            "gamma_min": float(gamma.min()), "gamma_max": float(gamma.max()),
            "gamma_new_min": float(gn.min()), "gamma_new_max": float(gn.max()),
            "sidecar_bytes": 0,
            "note": "overwrites existing RMSNorm gamma; function exact",
        }
    if tag == "down":
        s = col_scales(W, Xf, alpha=0.0)
        absorb = {
            "kind": "up_proj.row_j *= s_j  (SwiGLU channel j)",
            "s_min": float(s.min()), "s_max": float(s.max()),
            "sidecar_if_not_absorbed_bytes": int(s.size * 2),
            "sidecar_if_not_absorbed_bpw": float(8 * s.size * 2 / N_SOURCE),
        }

    rec_rows = {}
    for k, v in rows.items():
        rec_rows[k] = rec(v)
        if "meta" in v:
            rec_rows[k]["meta"] = v["meta"]

    out = {
        "tensor": name, "layer": layer, "cls": cls, "tag": tag, "site": site,
        "shape": [int(W.shape[0]), int(W.shape[1])],
        "elements": int(W.size),
        "split": split,
        "load_s": float(load_s),
        "rss_gb": float(rss_gb()),
        "q4_ref": rec(rows["q4_g128"]),
        "rows": rec_rows,
        "absorb": absorb,
        "bpw": {
            "binary": float(8 * bytes_of("binary", W.size) / N_SOURCE),
            "ternary": float(8 * bytes_of("ternary", W.size) / N_SOURCE),
            "q2": float(8 * bytes_of("q2", W.size) / N_SOURCE),
            "q4": float(8 * bytes_of("q4", W.size) / N_SOURCE),
            "note": "per-tensor contribution to complete BPW; not a program BPW",
        },
    }
    del W, Xf, Xh, prep
    return out


def q2_act_mse(W, X, g=GROUP, alphas=(0.55, 0.65, 0.75, 0.82, 0.88, 0.92, 0.96, 1.00)):
    """Per-group scale = α * amax / qmax, α picked to min ||Xg (w-wq)||^2 on fit X."""
    o, d = W.shape
    ng = d // g
    G = W.reshape(o, ng, g)
    amax = np.abs(G).max(axis=-1) + 1e-30
    lim = 1.0  # q2: qmax = 1
    best_err = None
    best = None
    for a in alphas:
        step = (amax * a) / lim
        q = np.clip(np.round(G / step[..., None]), -1, 1)
        Wq = q * step[..., None]
        # activation-weighted error per group, summed over tokens via Gram diag proxy:
        # e = w-wq; err = sum_t (Xg_t · e)^2 = e^T (Xg^T Xg) e. Compute exactly per group.
        err = np.zeros((o, ng), dtype=np.float64)
        E = (G - Wq).astype(np.float32)
        for j in range(ng):
            Xg = X[:, j * g:(j + 1) * g]
            ye = Xg @ E[:, j, :].T
            err[:, j] = (ye * ye).sum(axis=0)
        if best is None:
            best_err, best = err, Wq
        else:
            take = err < best_err
            best_err = np.where(take, err, best_err)
            best = np.where(take[..., None], Wq, best)
    return best.reshape(W.shape)


def calib_curve(layer=58, ns=(64, 128, 256, 512, 1024)):
    """Wall time + hold quality vs n_fit on L58 down. Fit from official v2 fit pool."""
    name = tname(layer, "mlp.down_proj")
    W = load_W(name)
    fit_i, hold_i = swiglu_split()
    Xh = load_site_rows("post_swiglu", layer, take_prefix(hold_i, 256))
    prep = Prep(W, Xh)
    q4 = score(prep, codec_uniform(W, 4, GROUP))
    raw_b = vs_ref(score(prep, codec_binary(W)), q4)
    raw_t = vs_ref(score(prep, codec_ternary(W)), q4)
    out = {"raw_binary": rec(raw_b), "raw_ternary": rec(raw_t), "q4": rec(q4), "points": []}
    for n in ns:
        Xf = load_site_rows("post_swiglu", layer, take_prefix(fit_i, n))
        t0 = now()
        Wb = apply_actls(W, Xf, "binary")
        tb = now() - t0
        t0 = now()
        Wt = apply_actls(W, Xf, "ternary")
        tt = now() - t0
        t0 = now()
        Wa = codec_ternary_anneal(W)
        ta = now() - t0
        sb = vs_ref(score(prep, Wb), q4)
        st = vs_ref(score(prep, Wt), q4)
        sa = vs_ref(score(prep, Wa), q4)
        pt = {
            "n_fit": int(Xf.shape[0]),
            "binary_actls": {**rec(sb), "fit_s": float(tb)},
            "ternary_actls": {**rec(st), "fit_s": float(tt)},
            "ternary_anneal": {**rec(sa), "fit_s": float(ta)},
        }
        out["points"].append(pt)
        print(f"CALIB n={Xf.shape[0]} bin_obs={sb['observed']:.6f} "
              f"ter_obs={st['observed']:.6f} ann_obs={sa['observed']:.6f} "
              f"fit_s={tb:.2f}/{tt:.2f}/{ta:.2f}", flush=True)
        del Xf, Wb, Wt, Wa
    del W, Xh, prep
    return out


def up_side_effect(layer=58):
    """Does absorbing down col-eq scales into up rows hurt up's own low-bit gate?"""
    Wd = load_W(tname(layer, "mlp.down_proj"))
    Wu = load_W(tname(layer, "mlp.up_proj"))
    Xf, Xh, split = site_xy("post_attn_norm", layer, 256, 256)
    # down scales from post_swiglu
    Xfd, Xhd, _ = site_xy("post_swiglu", layer, 256, 256)
    s = col_scales(Wd, Xfd, alpha=0.0)          # [17408]
    s5 = col_scales(Wd, Xfd, alpha=0.5)
    out = {"split": split, "up_shape": [int(Wu.shape[0]), int(Wu.shape[1])]}
    for label, up in (
        ("up_raw", Wu),
        ("up_after_down_col_eq", (Wu * s[:, None]).astype(np.float32)),
        ("up_after_down_sq05", (Wu * s5[:, None]).astype(np.float32)),
    ):
        prep = Prep(up, Xh)
        q4 = score(prep, codec_uniform(up, 4, GROUP))
        row = {"weight_cosine_vs_raw_up": float(_rowcos(Wu, up)) if up is not Wu else 1.0}
        for cname, fn in (
            ("binary", lambda M: codec_binary(M)),
            ("ternary", lambda M: codec_ternary(M)),
            ("q4", lambda M: codec_uniform(M, 4, GROUP)),
        ):
            row[cname] = rec(vs_ref(score(prep, fn(up)), q4))
        out[label] = row
        print(f"UP {label} bin_obs={row['binary']['observed']:.6f} "
              f"ter_obs={row['ternary']['observed']:.6f} "
              f"wcos={row['weight_cosine_vs_raw_up']:.6f}", flush=True)
        del prep
    del Wd, Wu, Xf, Xh, Xfd, Xhd, prep
    return out


def hold512_check(layer, codec_key="binary_raw"):
    """Stability of observed on 512 hold tokens for a decisive down."""
    W = load_W(tname(layer, "mlp.down_proj"))
    _, hold_i = swiglu_split()
    Xh = load_site_rows("post_swiglu", layer, take_prefix(hold_i, 512))
    prep = Prep(W, Xh)
    q4 = score(prep, codec_uniform(W, 4, GROUP))
    raw = vs_ref(score(prep, codec_binary(W) if "binary" in codec_key else codec_ternary(W)), q4)
    # best combo we will have: col_eq + actls
    Xf, _, _ = site_xy("post_swiglu", layer, 256, 8)  # fit only
    s = col_scales(W, Xf, alpha=0.0)
    Wn = fold_col(W, s)
    Xn = Xf * s[None, :]
    kind = "binary" if "binary" in codec_key else "ternary"
    Wq = unfold_col(apply_actls(Wn, Xn, kind), s)
    recnd = vs_ref(score(prep, Wq), q4)
    del W, Xh, Xf, prep
    return {"n_hold": 512, "raw": rec(raw), "col_eq_actls": rec(recnd)}


def main():
    t_all = now()
    result = {
        "schema": "hawking.gravity1.reshape_before_lowbit.v1",
        "host": "Hawking Apple M3 Ultra, CPU numpy, no GPU",
        "N": N_SOURCE,
        "group": GROUP,
        "axis_margin": AXIS_MARGIN,
        "source": BF16,
        "capture_v1": CAP1,
        "capture_v2": CAP2,
        "capture_v2_sha256_self": cap2().get("sha256_self"),
        "started_unix": time.time(),
    }
    print("=== codec twin vs c_uniform ===", flush=True)
    rng = np.random.default_rng(0)
    Wt = rng.standard_normal((32, 256)).astype(np.float32)
    dmax = float(np.max(np.abs(codec_uniform(Wt, 4) - c_uniform(Wt, 4, 128))))
    result["c_uniform_twin_maxabs"] = dmax
    print(f"c_uniform twin maxabs={dmax:.3e}", flush=True)
    if dmax > 1e-6:
        raise SystemExit(f"codec_uniform drifted from c_uniform: {dmax}")

    print("=== doctor demo ===", flush=True)
    import gravity_doctor_gate as gdg
    # keep demo off real tensors
    try:
        gdg.demo()
        result["doctor_demo"] = "PASS"
    except Exception as e:
        result["doctor_demo"] = f"FAIL {e}"
        raise

    print("=== v1 cite reproduction ===", flush=True)
    result["v1_repro"] = reproduce_v1_cites()

    print("=== per-tensor gate ===", flush=True)
    tensors = []
    for layer, cls, site, tag in JOBS:
        print(f"-- L{layer} {cls} site={site} rss={rss_gb():.3f} --", flush=True)
        tensors.append(run_tensor(layer, cls, site, tag))
    result["tensors"] = tensors

    print("=== calib curve L58 down ===", flush=True)
    result["calib_L58_down"] = calib_curve(58)

    print("=== up side-effect L58 ===", flush=True)
    result["up_side_effect_L58"] = up_side_effect(58)

    print("=== hold512 L58/L62 ===", flush=True)
    result["hold512"] = {
        "L58_binary": hold512_check(58, "binary"),
        "L62_ternary": hold512_check(62, "ternary"),
    }

    result["wall_s"] = float(now() - t_all)
    result["rss_max_gb"] = float(rss_gb())
    json.dump(result, open(OUT_JSON, "w"), indent=2)
    print(f"WROTE {OUT_JSON} wall={result['wall_s']:.1f}s rss_max={result['rss_max_gb']:.3f} GB",
          flush=True)


if __name__ == "__main__":
    main()
