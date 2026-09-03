#!/usr/bin/env python3
"""G-PLANES: greedy ternary/binary residual planes vs adequacy gate.

Fits P1..Pk against activation-conditioned (diagonal-Gram) error, scores
each rung with tools/gravity_doctor_gate.axes / gate, and compares to a
flat uniform code of the same complete width.

Writes /tmp/g1_planes_ternary.json incrementally. CPU/numpy only.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import resource
import struct
import sys
import time
import traceback
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# worktree tools/ lives relative to cwd when launched from the repo
REPO = os.environ.get("G1_REPO", os.getcwd())
TOOLS = os.path.join(REPO, "tools")

BF16 = os.environ.get(
    "G1_BF16",
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16",
)
CAPTURE = os.environ.get(
    "G1_CAPTURE",
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1",
)
OUT_JSON = os.environ.get("G1_PLANES_JSON", "/tmp/g1_planes_ternary.json")

N_SOURCE = 26_895_998_464
HEADER = 40
GROUP = 128
LOG2_3 = math.log2(3.0)
TRIT_PACK_BPW = 8.0 / 5.0  # 5 trits / byte = 1.6
MULTS = (0.40, 0.55, 0.70, 0.85, 1.00)
SPARSE_KEEPS = (0.05, 0.01, 0.002)
ROOF_GB_S = 639.2522341137478  # RECEIPT sealed addressing

spec = importlib.util.spec_from_file_location(
    "gravity_doctor_gate", os.path.join(TOOLS, "gravity_doctor_gate.py")
)
gdg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gdg)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def load_tensor(name, root=BF16):
    return gdg.load_tensor(name, root=root)


def load_X_hidden(layer, capture=CAPTURE):
    return gdg.load_X(layer, capture=capture)


def load_index(root=BF16):
    return json.load(open(os.path.join(root, "model.safetensors.index.json")))


# ---------------------------------------------------------------- packing / cost


def n_groups(out, inn, group=GROUP):
    return out * ((inn + group - 1) // group)


def bytes_dense_ternary(out, inn, group=GROUP, header=HEADER):
    n = out * inn
    code = (n + 4) // 5  # 5 trits / byte
    return int(code + n_groups(out, inn, group) * 2 + header)


def bytes_dense_ternary_2bit(out, inn, group=GROUP, header=HEADER):
    n = out * inn
    code = (n * 2 + 7) // 8
    return int(code + n_groups(out, inn, group) * 2 + header)


def bytes_binary(out, inn, group=GROUP, header=HEADER):
    n = out * inn
    code = (n + 7) // 8
    return int(code + n_groups(out, inn, group) * 2 + header)


def bytes_uniform(out, inn, bits, group=GROUP, header=HEADER):
    n = out * inn
    code = (n * bits + 7) // 8
    return int(code + n_groups(out, inn, group) * 2 + header)


def bytes_sparse_plane(out, inn, nnz, n_live_groups, group=GROUP, header=HEADER):
    """Cheapest of bitmask+trits, CSR, COO, live-group skip. Includes f16 scales."""
    n = out * inn
    ng = n_groups(out, inn, group)
    trit_nnz = (int(nnz) + 4) // 5
    row_scales = out * 2
    live_scales = int(n_live_groups) * 2

    bitmask = (n + 7) // 8
    opt_mask = bitmask + trit_nnz + row_scales + header

    # CSR: i32 rowptr + u16 col (inn<=17408) + packed trits + per-row f16
    csr = (out + 1) * 4 + int(nnz) * 2 + trit_nnz + row_scales + header

    # COO: u32 linear index + f16 value (signed magnitude, no trit pack)
    coo = int(nnz) * 6 + header

    # live-group: bitmap of groups + packed trits only on live groups + live scales
    gsz = group
    live_code = (int(n_live_groups) * gsz + 4) // 5
    live = (ng + 7) // 8 + live_code + live_scales + header

    cands = {
        "bitmask_trits": opt_mask,
        "csr": csr,
        "coo": coo,
        "live_group": live,
    }
    winner = min(cands, key=cands.get)
    return int(cands[winner]), winner, cands


def bpw_of(nbytes):
    return 8.0 * nbytes / N_SOURCE


def tensor_bpw(nbytes, elements):
    return 8.0 * nbytes / elements if elements else float("nan")


# ---------------------------------------------------------------- plane fit


def _pack(W, group):
    out, inn = W.shape
    pad = (group - (inn % group)) % group
    if pad:
        Wp = np.pad(W, ((0, 0), (0, pad)))
    else:
        Wp = W
    return Wp.reshape(out, -1, group), pad


def fit_ternary_plane(R, a, group=GROUP, mults=MULTS):
    """Per-group MSE-optimal ternary. a is (inn,) column weights (rms)."""
    out, inn = R.shape
    R3, pad = _pack(R, group)
    a_pad = np.pad(a.astype(np.float32), (0, pad)) if pad else a.astype(np.float32)
    aw = (a_pad.reshape(-1, group) ** 2).astype(np.float32)  # (ng, g)
    amax = np.max(np.abs(R3), axis=-1).astype(np.float32)  # (out, ng)

    best_err = None
    best_T = None
    best_s = None
    best_m = None

    for m in mults:
        s0 = (m * amax).astype(np.float32)
        s_safe = np.maximum(s0, 1e-30)
        T = np.clip(np.rint(R3 / s_safe[..., None]), -1.0, 1.0).astype(np.float32)
        num = np.sum(aw * R3 * T, axis=-1)
        den = np.sum(aw * (T * T), axis=-1)
        s = (num / np.maximum(den, 1e-30)).astype(np.float32)
        recon = s[..., None] * T
        err = np.sum(aw * (R3 - recon) ** 2, axis=-1)
        if best_err is None:
            best_err = err
            best_T = T
            best_s = s
            best_m = np.full(err.shape, m, dtype=np.float32)
        else:
            take = err < best_err
            best_err = np.where(take, err, best_err)
            best_T = np.where(take[..., None], T, best_T)
            best_s = np.where(take, s, best_s)
            best_m = np.where(take, m, best_m)

    recon = (best_s[..., None] * best_T).reshape(out, -1)[:, :inn]
    T_full = best_T.reshape(out, -1)[:, :inn]
    nnz = int(np.count_nonzero(T_full))
    n_live = int(np.count_nonzero(np.any(best_T != 0, axis=-1)))
    return {
        "recon": recon.astype(np.float32, copy=False),
        "nnz": nnz,
        "nnz_frac": nnz / float(out * inn),
        "n_live_groups": n_live,
        "n_groups": int(best_s.size),
        "live_group_frac": n_live / float(best_s.size),
        "scale_abs_mean": float(np.mean(np.abs(best_s))),
        "scale_abs_p95": float(np.quantile(np.abs(best_s), 0.95)),
        "mult_mean": float(np.mean(best_m)),
        "mult_frac_1": float(np.mean(best_m == 1.0)),
        "kind": "ternary_g128",
    }


def fit_binary_plane(R, a, group=GROUP):
    out, inn = R.shape
    R3, pad = _pack(R, group)
    a_pad = np.pad(a.astype(np.float32), (0, pad)) if pad else a.astype(np.float32)
    aw = (a_pad.reshape(-1, group) ** 2).astype(np.float32)
    T = np.sign(R3).astype(np.float32)
    T[T == 0] = 1.0
    num = np.sum(aw * R3 * T, axis=-1)
    den = np.sum(aw, axis=-1)
    s = (num / np.maximum(den, 1e-30)).astype(np.float32)
    recon = (s[..., None] * T).reshape(out, -1)[:, :inn]
    return {
        "recon": recon.astype(np.float32, copy=False),
        "nnz": int(out * inn),
        "nnz_frac": 1.0,
        "n_live_groups": int(s.size),
        "n_groups": int(s.size),
        "live_group_frac": 1.0,
        "scale_abs_mean": float(np.mean(np.abs(s))),
        "scale_abs_p95": float(np.quantile(np.abs(s), 0.95)),
        "kind": "binary_g128",
    }


def fit_sparse_ternary(R, a, keep, group=GROUP, mults=MULTS):
    """Keep the top-`keep` fraction of |R|*a per row, then ternary-RTN."""
    score = np.abs(R) * a.reshape(1, -1)
    # per-row quantile; keep at least 1
    q = np.quantile(score, 1.0 - keep, axis=1, keepdims=True)
    mask = score >= q
    # zero the rest for the fitter
    Rz = R * mask
    pl = fit_ternary_plane(Rz, a, group=group, mults=mults)
    # force recon on unmasked to 0 (fitter already sees zeros)
    pl["keep"] = float(keep)
    pl["kind"] = f"sparse_keep{keep:g}"
    return pl


def residual_concentration(R, a):
    """Lorenz of activation-weighted residual energy. MEASURED."""
    e = (R.astype(np.float64) * a.reshape(1, -1)) ** 2
    flat = e.ravel()
    tot = float(flat.sum())
    n = flat.size
    if tot <= 0:
        return {"energy": 0.0, "frac_for": {0.5: 0.0, 0.9: 0.0, 0.99: 0.0}}
    # partial sort: we need the top until 99%
    # full argsort of 89e6 is OK (~1-2s) but we can use partition ladder
    order = np.argsort(flat)[::-1]
    c = np.cumsum(flat[order])
    c /= c[-1]
    out = {"energy": tot, "n": int(n)}
    fr = {}
    for p in (0.5, 0.8, 0.9, 0.95, 0.99):
        k = int(np.searchsorted(c, p) + 1)
        fr[p] = k / n
    out["frac_for"] = fr
    # kurtosis of residual (weight space)
    r = R.ravel().astype(np.float64)
    r = r - r.mean()
    m2 = float((r * r).mean())
    m4 = float((r * r * r * r).mean())
    out["excess_kurtosis"] = (m4 / (m2 * m2) - 3.0) if m2 > 0 else 0.0
    return out


# ---------------------------------------------------------------- scoring


def _rowcos(A, B):
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-30
    return float(np.mean(num / den))


def _worst_unit(A, B):
    num = (A * B).sum(0)
    na, nb = np.linalg.norm(A, axis=0), np.linalg.norm(B, axis=0)
    live = na > 1e-20
    cos = np.zeros_like(num)
    denom = na * nb + 1e-30
    cos[live] = num[live] / denom[live]
    return float(cos[live].min()) if live.any() else 1.0


def axes_pre(Yw, Yp, Yhw, Yhp):
    return {
        "observed": _rowcos(Yw, Yhw),
        "probed": _rowcos(Yp, Yhp),
        "worst_unit": min(_worst_unit(Yw, Yhw), _worst_unit(Yp, Yhp)),
    }


def gate_from(a, ref):
    deficits = {k: a[k] - (ref[k] - gdg.AXIS_MARGIN[k]) for k in a}
    worst = min(deficits, key=deficits.get)
    return {
        **a,
        "deficit": deficits,
        "gate": deficits[worst],
        "worst_axis": worst,
        "healthy": deficits[worst] >= 0.0,
        "mode": "relative",
    }


def matvec(X, W):
    return X @ W.T


# ---------------------------------------------------------------- sites


def col_rms(X):
    return np.sqrt(np.mean(X.astype(np.float64) ** 2, axis=0)).astype(np.float32)


def make_site(layer, cls, W, Xh, cache):
    """Return (X_fit_site, a, site_label, extra_loaded).

    X_fit_site has shape (n, d_in) matching W.shape[1].
    a is column rms from the FIT split (even rows).
    """
    din = W.shape[1]
    key = (layer, cls)

    if din == 5120 and Xh is not None:
        X = Xh
        site = "post_norm_hidden_UNCONFIRMED_inproj"
        a = col_rms(X[0::2])
        return X, a, site

    if cls.endswith("mlp.down_proj") or cls == "mlp.down_proj":
        # reconstruct SwiGLU intermediate
        ck = ("down_X", layer)
        if ck not in cache:
            Wg = load_tensor(f"language_model.model.layers.{layer}.mlp.gate_proj.weight")
            Yg = silu(Xh @ Wg.T)
            del Wg
            Wu = load_tensor(f"language_model.model.layers.{layer}.mlp.up_proj.weight")
            Yu = Xh @ Wu.T
            del Wu
            cache[ck] = (Yg * Yu).astype(np.float32)
        X = cache[ck]
        a = col_rms(X[0::2])
        return X, a, "reconstructed_swiglu"

    if cls.endswith("linear_attn.out_proj") or cls == "linear_attn.out_proj":
        ck = ("dn_out_X", layer)
        if ck not in cache:
            Wqkv = load_tensor(
                f"language_model.model.layers.{layer}.linear_attn.in_proj_qkv.weight"
            )
            Y = Xh @ Wqkv.T
            del Wqkv
            V = Y[:, 4096:]
            del Y
            Wz = load_tensor(
                f"language_model.model.layers.{layer}.linear_attn.in_proj_z.weight"
            )
            Z = silu(Xh @ Wz.T)
            del Wz
            cache[ck] = (V * Z).astype(np.float32)
        X = cache[ck]
        a = col_rms(X[0::2])
        return X, a, "dn_v_silu_z_proxy_NOT_recurrent"

    if cls.endswith("self_attn.o_proj") or cls == "self_attn.o_proj":
        ck = ("gqa_o_X", layer)
        if ck not in cache:
            Wq = load_tensor(
                f"language_model.model.layers.{layer}.self_attn.q_proj.weight"
            )
            Y = Xh @ Wq.T
            del Wq
            Q, G = Y[:, :6144], Y[:, 6144:]
            cache[ck] = (Q * sigmoid(G)).astype(np.float32)
        X = cache[ck]
        a = col_rms(X[0::2])
        return X, a, "gqa_q_sigmoid_gate_proxy_NOT_attn_mix"

    # fallback: isotropic
    rng = np.random.default_rng(0)
    X = rng.standard_normal((256, din)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    a = np.ones((din,), dtype=np.float32)
    return X, a, "isotropic_fallback"


# ---------------------------------------------------------------- one tensor


def plane_stats_only(pl):
    d = {k: v for k, v in pl.items() if k != "recon"}
    return d


def encode_rungs(out, inn, planes, sparse_after1=None):
    """Complete-tensor BPW at each rung, dense and sparse-aware."""
    rows = []
    acc_dense = 0
    for i, pl in enumerate(planes, 1):
        if pl["kind"].startswith("binary"):
            b = bytes_binary(out, inn)
        else:
            b = bytes_dense_ternary(out, inn)
        acc_dense += b
        rec = {
            "k": i,
            "dense_bytes": acc_dense,
            "dense_tensor_bpw": tensor_bpw(acc_dense, out * inn),
            "this_nnz_frac": pl["nnz_frac"],
            "this_live_group_frac": pl["live_group_frac"],
        }
        # sparse encoding: plane 1 always dense (almost never sparse enough),
        # later planes take min(dense, sparse)
        acc_sp = 0
        enc = []
        for j, q in enumerate(planes[:i]):
            if j == 0 or q["nnz_frac"] > 0.35:
                if q["kind"].startswith("binary"):
                    bj = bytes_binary(out, inn)
                    enc.append("binary_dense")
                else:
                    bj = bytes_dense_ternary(out, inn)
                    enc.append("ternary_dense")
            else:
                bj, win, _ = bytes_sparse_plane(
                    out, inn, q["nnz"], q["n_live_groups"]
                )
                enc.append(win)
            acc_sp += bj
        rec["sparseaware_bytes"] = acc_sp
        rec["sparseaware_tensor_bpw"] = tensor_bpw(acc_sp, out * inn)
        rec["encodings"] = enc
        rows.append(rec)
    if sparse_after1 is not None:
        p1 = planes[0]
        b1 = (
            bytes_binary(out, inn)
            if p1["kind"].startswith("binary")
            else bytes_dense_ternary(out, inn)
        )
        for name, pl in sparse_after1.items():
            bs, win, detail = bytes_sparse_plane(
                out, inn, pl["nnz"], pl["n_live_groups"]
            )
            rows.append(
                {
                    "k": f"1+{name}",
                    "dense_bytes": b1 + bytes_dense_ternary(out, inn),
                    "dense_tensor_bpw": tensor_bpw(
                        b1 + bytes_dense_ternary(out, inn), out * inn
                    ),
                    "sparseaware_bytes": b1 + bs,
                    "sparseaware_tensor_bpw": tensor_bpw(b1 + bs, out * inn),
                    "this_nnz_frac": pl["nnz_frac"],
                    "this_live_group_frac": pl["live_group_frac"],
                    "encodings": ["p1_dense", win],
                    "sparse_detail": detail,
                }
            )
    return rows


def c_uniform_fast(W, bits, group=GROUP):
    """Vectorized sibling of gdg.c_uniform (same math, no Python group loop)."""
    if bits <= 1:
        raise ValueError("c_uniform bits<=1 is degenerate (lim=0)")
    lim = (1 << (bits - 1)) - 1
    out, inn = W.shape
    W3, pad = _pack(W.astype(np.float32, copy=False), group)
    amax = np.max(np.abs(W3), axis=-1, keepdims=True) + 1e-30
    step = amax / lim
    Q = np.clip(np.rint(W3 / step), -lim, lim) * step
    return Q.reshape(out, -1)[:, :inn].astype(np.float32, copy=False)


def run_tensor(name, layer, cls, cache, probe_seed=0):
    t0 = time.time()
    W = load_tensor(name)
    out, inn = int(W.shape[0]), int(W.shape[1])
    Xh = load_X_hidden(layer) if layer is not None else None
    X, a, site = make_site(layer, cls, W, Xh, cache)
    if X.shape[1] != inn:
        raise RuntimeError(f"site X {X.shape} vs W {W.shape} on {name}")

    hold_idx = np.arange(X.shape[0])[1::2]
    X_all = X

    P = gdg._probe(inn, n=256, seed=probe_seed)
    Yw = matvec(X_all, W)
    Yp = matvec(P, W)
    Yw_hold = Yw[hold_idx]

    # Q4 reference
    Wq4 = c_uniform_fast(W, 4, GROUP)
    ref = axes_pre(Yw, Yp, matvec(X_all, Wq4), matvec(P, Wq4))
    del Wq4

    def score_Wh(Wh):
        Yh = matvec(X_all, Wh)
        Ph = matvec(P, Wh)
        ax = axes_pre(Yw, Yp, Yh, Ph)
        g = gate_from(ax, ref)
        hold = _rowcos(Yw_hold, Yh[hold_idx])
        return g, hold

    # flats
    flats = OrderedDict()
    flats["binary_g128"] = None  # filled below via plane
    for bits, tag in ((2, "q2_ternary_absmax_g128"), (3, "q3_g128"), (4, "q4_g128"),
                      (5, "q5_g128"), (6, "q6_g128")):
        Wh = c_uniform_fast(W, bits, GROUP)
        g, hold = score_Wh(Wh)
        b = bytes_uniform(out, inn, bits)
        flats[tag] = {
            "bits": bits,
            "tensor_bpw": tensor_bpw(b, out * inn),
            "bytes": b,
            **{k: g[k] for k in ("observed", "probed", "worst_unit", "gate", "healthy", "worst_axis")},
            "hold_observed": hold,
        }
        del Wh

    # binary as a flat
    bpl = fit_binary_plane(W, a)
    g, hold = score_Wh(bpl["recon"])
    bb = bytes_binary(out, inn)
    flats["binary_g128"] = {
        "bits": 1,
        "tensor_bpw": tensor_bpw(bb, out * inn),
        "bytes": bb,
        **{k: g[k] for k in ("observed", "probed", "worst_unit", "gate", "healthy", "worst_axis")},
        "hold_observed": hold,
        "nnz_frac": 1.0,
    }

    # ---- greedy dense ternary planes (act-weighted)
    R = W.copy()
    acc = np.zeros_like(W)
    planes = []
    conc = [residual_concentration(R, a)]
    rungs = []
    for k in range(1, 5):
        pl = fit_ternary_plane(R, a)
        acc = acc + pl["recon"]
        R = (W - acc).astype(np.float32)
        planes.append(pl)
        conc.append(residual_concentration(R, a))
        g, hold = score_Wh(acc)
        rungs.append(
            {
                "k": k,
                "family": "ternary_g128_actdiag",
                **plane_stats_only(pl),
                **{kk: g[kk] for kk in ("observed", "probed", "worst_unit", "gate", "healthy", "worst_axis")},
                "hold_observed": hold,
                "resid_rel_f": float(np.linalg.norm(R) / (np.linalg.norm(W) + 1e-30)),
            }
        )

    # ---- sparse hope: P1 + one sparse correction
    R1 = (W - planes[0]["recon"]).astype(np.float32)
    sparse = OrderedDict()
    for keep in SPARSE_KEEPS:
        pl = fit_sparse_ternary(R1, a, keep)
        What = planes[0]["recon"] + pl["recon"]
        g, hold = score_Wh(What)
        sparse[f"keep{keep:g}"] = {
            **plane_stats_only(pl),
            **{kk: g[kk] for kk in ("observed", "probed", "worst_unit", "gate", "healthy", "worst_axis")},
            "hold_observed": hold,
            "resid_rel_f": float(
                np.linalg.norm(W - What) / (np.linalg.norm(W) + 1e-30)
            ),
        }
        del What, pl

    # ---- binary P1 + ternary P2 (cheaper 2-plane)
    acc_bt = bpl["recon"] + fit_ternary_plane((W - bpl["recon"]).astype(np.float32), a)["recon"]
    g_bt, hold_bt = score_Wh(acc_bt)
    binary_then_ternary = {
        **{kk: g_bt[kk] for kk in ("observed", "probed", "worst_unit", "gate", "healthy", "worst_axis")},
        "hold_observed": hold_bt,
        "bytes": bytes_binary(out, inn) + bytes_dense_ternary(out, inn),
        "tensor_bpw": tensor_bpw(
            bytes_binary(out, inn) + bytes_dense_ternary(out, inn), out * inn
        ),
        "resid_rel_f": float(np.linalg.norm(W - acc_bt) / (np.linalg.norm(W) + 1e-30)),
    }
    del acc_bt, bpl

    # ---- one-tensor weight-mse plane-1 vs act (overfit check) on this W
    ones = np.ones((inn,), dtype=np.float32)
    pl_mse = fit_ternary_plane(W, ones)
    g_mse, hold_mse = score_Wh(pl_mse["recon"])
    mse_p1 = {
        **plane_stats_only(pl_mse),
        **{kk: g_mse[kk] for kk in ("observed", "probed", "worst_unit", "gate", "healthy", "worst_axis")},
        "hold_observed": hold_mse,
    }
    del pl_mse

    costs = encode_rungs(out, inn, planes, sparse)
    # attach costs onto rungs by k
    cost_by_k = {c["k"]: c for c in costs}

    rec = {
        "tensor": name,
        "layer": layer,
        "cls": cls,
        "shape": [out, inn],
        "elements": out * inn,
        "site": site,
        "x_shape": list(X.shape),
        "x_rank_est": int(np.linalg.matrix_rank(X[: min(128, X.shape[0])], tol=1e-3 * np.linalg.norm(X[: min(128, X.shape[0])], 2))) if X.shape[0] >= 8 else None,
        "ref_q4": ref,
        "flats": flats,
        "rungs": rungs,
        "sparse_after_p1": sparse,
        "weight_mse_p1": mse_p1,
        "binary_then_ternary": binary_then_ternary,
        "costs": costs,
        "concentration": [
            {
                "after_k": i,
                "frac_for": {str(p): v for p, v in c["frac_for"].items()},
                "excess_kurtosis": c.get("excess_kurtosis"),
                "energy": c.get("energy"),
            }
            for i, c in enumerate(conc)
        ],
        "wall_s": time.time() - t0,
        "rss_gb": rss_gb(),
    }
    # drop recon arrays
    for pl in planes:
        pl.pop("recon", None)
    del W, R, acc, R1, Yw, Yp, Yw_hold, X, P
    return rec


# ---------------------------------------------------------------- inventory


MLP_LAYERS = (0, 15, 31, 47, 63)
GQA_LAYERS = (3, 15, 31, 63)
DN_LAYERS = (0, 16, 32, 48)


def jobs(smoke=False):
    js = []
    if smoke:
        js.append((0, "mlp.gate_proj"))
        return js
    for L in MLP_LAYERS:
        for c in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            js.append((L, c))
    for L in GQA_LAYERS:
        for c in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ):
            js.append((L, c))
    for L in DN_LAYERS:
        for c in (
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.out_proj",
        ):
            js.append((L, c))
    return js


def class_mass():
    """Element counts for model-level BPW projection. MEASURED from index."""
    idx = load_index()
    mass = {}
    for name, shard in idx["weight_map"].items():
        if not name.startswith("language_model.") or not name.endswith(".weight"):
            continue
        # shape not in index; use known geometry
    # known
    return {
        "mlp.gate_proj": {"n": 64, "shape": (17408, 5120)},
        "mlp.up_proj": {"n": 64, "shape": (17408, 5120)},
        "mlp.down_proj": {"n": 64, "shape": (5120, 17408)},
        "self_attn.q_proj": {"n": 16, "shape": (12288, 5120)},
        "self_attn.k_proj": {"n": 16, "shape": (1024, 5120)},
        "self_attn.v_proj": {"n": 16, "shape": (1024, 5120)},
        "self_attn.o_proj": {"n": 16, "shape": (5120, 6144)},
        "linear_attn.in_proj_qkv": {"n": 48, "shape": (10240, 5120)},
        "linear_attn.in_proj_z": {"n": 48, "shape": (6144, 5120)},
        "linear_attn.out_proj": {"n": 48, "shape": (5120, 6144)},
        "embed": {"n": 1, "shape": (248320, 5120)},
        "lm_head": {"n": 1, "shape": (248320, 5120)},
    }


def project_model_bpw(per_cls_tensor_bpw):
    """8*sum(class_bytes)/N using measured per-class tensor BPW * elements * n."""
    mass = class_mass()
    # small f32 leftovers from census
    small_elems = 660480
    small_bytes = 2642952  # G0 f32 vectors
    total_b = small_bytes
    covered = small_elems
    parts = {}
    for cls, meta in mass.items():
        e = meta["n"] * meta["shape"][0] * meta["shape"][1]
        bpw = per_cls_tensor_bpw.get(cls)
        if bpw is None:
            continue
        b = bpw * e / 8.0
        parts[cls] = {"elements": e, "bytes": b, "tensor_bpw": bpw}
        total_b += b
        covered += e
    return {
        "covered_elems": covered,
        "total_bytes": total_b,
        "complete_bpw": 8.0 * total_b / N_SOURCE,
        "parts": parts,
        "uncovered_elems": N_SOURCE - covered,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default=None, help="layer:cls e.g. 0:mlp.gate_proj")
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    js = jobs(smoke=args.smoke)
    if args.only:
        L, c = args.only.split(":", 1)
        js = [(int(L), c)]

    report = {
        "schema": "hawking.gravity1.planes_ternary.v1",
        "bf16": BF16,
        "capture": CAPTURE,
        "N": N_SOURCE,
        "group": GROUP,
        "mults": list(MULTS),
        "sparse_keeps": list(SPARSE_KEEPS),
        "roof_gb_s": ROOF_GB_S,
        "trit_pack_bpw": TRIT_PACK_BPW,
        "log2_3": LOG2_3,
        "header_bytes": HEADER,
        "started": time.time(),
        "tensors": [],
        "errors": [],
    }

    cache = {}
    done = set()
    if args.resume and os.path.exists(args.out):
        prev = json.load(open(args.out))
        report["tensors"] = prev.get("tensors", [])
        report["errors"] = prev.get("errors", [])
        report["started"] = prev.get("started", report["started"])
        done = {t["tensor"] for t in report["tensors"]}
        print(f"resume: {len(done)} tensors already in {args.out}", flush=True)
    for i, (L, c) in enumerate(js):
        name = f"language_model.model.layers.{L}.{c}.weight"
        if name in done:
            print(f"[{i+1}/{len(js)}] SKIP {name}", flush=True)
            continue
        print(f"[{i+1}/{len(js)}] {name}  rss={rss_gb():.2f}G", flush=True)
        # drop layer-local cache when layer changes
        if cache and all(k[1] != L for k in cache if isinstance(k, tuple) and len(k) == 2 and k[0] in ("down_X", "dn_out_X", "gqa_o_X")):
            pass
        # purge previous layer reconstructions
        dead = [k for k in cache if isinstance(k, tuple) and len(k) >= 2 and k[1] != L]
        for k in dead:
            del cache[k]
        try:
            rec = run_tensor(name, L, c, cache)
            report["tensors"].append(rec)
            gc.collect()
            print(
                f"    k1 gate={rec['rungs'][0]['gate']:+.4f} obs={rec['rungs'][0]['observed']:.4f} "
                f"pr={rec['rungs'][0]['probed']:.4f} nnz={rec['rungs'][0]['nnz_frac']:.3f} "
                f"p90={rec['concentration'][1]['frac_for']['0.9']:.3f} "
                f"k2h={rec['rungs'][1]['healthy']} "
                f"{rec['wall_s']:.1f}s rss={rss_gb():.2f}G",
                flush=True,
            )
        except Exception as e:
            traceback.print_exc()
            report["errors"].append({"tensor": name, "err": repr(e)})
        # incremental
        report["rss_max_gb"] = rss_gb()
        report["n_done"] = len(report["tensors"])
        with open(args.out, "w") as f:
            json.dump(report, f)
            f.flush()

    report["finished"] = time.time()
    report["wall_s"] = report["finished"] - report["started"]
    report["rss_max_gb"] = rss_gb()
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"WROTE {args.out}  tensors={len(report['tensors'])} errors={len(report['errors'])} "
          f"wall={report['wall_s']:.1f}s rss={report['rss_max_gb']:.3f}G")
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
