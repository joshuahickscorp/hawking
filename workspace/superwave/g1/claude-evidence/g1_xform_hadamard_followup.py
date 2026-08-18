#!/usr/bin/env python3
"""Follow-up: fix pair-plane aliasing; measure H after q_norm; 3994 flattened kurtosis."""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, "/tmp")
from g1_xform_hadamard import (
    load_tensor, load_X_hidden, tname, gemm_x_wt, rotate_half_rope, attn_scores,
    score_cmp, fwht_last, apply_named, N_HEADS, N_KV, HEAD_DIM, ROTARY,
    group_stats, excess_kurtosis_1d,
)

OUT = "/tmp/g1_xform_hadamard_followup.json"


def apply_head(x, kind, rng):
    T, H, D = x.shape
    y = x.copy()
    if kind == "id":
        return y
    if kind == "wh256":
        return fwht_last(y)
    if kind == "wh64_rotary":
        y[..., :64] = fwht_last(y[..., :64].copy())
        return y
    if kind == "wh64_nonrotary":
        for s0 in (64, 128, 192):
            y[..., s0:s0 + 64] = fwht_last(y[..., s0:s0 + 64].copy())
        return y
    if kind == "wh128_nonrotary_mid":
        y[..., 64:192] = fwht_last(y[..., 64:192].copy())
        return y
    if kind == "pair_wh2":
        a = y[..., :32].copy()
        b = y[..., 32:64].copy()
        s = np.float32(2.0 ** -0.5)
        y[..., :32] = (a + b) * s
        y[..., 32:64] = (a - b) * s
        return y
    if kind == "pair_rot45":
        a = y[..., :32].copy()
        b = y[..., 32:64].copy()
        s = np.float32(2.0 ** -0.5)
        y[..., :32] = (a - b) * s
        y[..., 32:64] = (a + b) * s
        return y
    if kind == "signs_iid":
        sg = rng.choice(np.array([-1.0, 1.0], np.float32), size=D)
        return y * sg
    if kind == "signs_pair_const":
        s32 = rng.choice(np.array([-1.0, 1.0], np.float32), size=32)
        sg = np.ones(D, dtype=np.float32)
        sg[:32] = s32
        sg[32:64] = s32
        sg[64:] = rng.choice(np.array([-1.0, 1.0], np.float32), size=D - 64)
        return y * sg
    if kind == "signs_pair_flip":
        s32 = rng.choice(np.array([-1.0, 1.0], np.float32), size=32)
        sg = np.ones(D, dtype=np.float32)
        sg[:32] = s32
        sg[32:64] = -s32
        return y * sg
    if kind == "generic_orth256":
        A = rng.standard_normal((D, D)).astype(np.float32)
        Qm, R = np.linalg.qr(A)
        Qm = Qm * np.sign(np.diag(R))
        return np.einsum("thd,ed->the", y, Qm, optimize=True)
    if kind == "safe_bundle":
        # pair-const signs on rotary + WH-64 on three non-rotary blocks
        s32 = rng.choice(np.array([-1.0, 1.0], np.float32), size=32)
        sg = np.ones(D, dtype=np.float32)
        sg[:32] = s32
        sg[32:64] = s32
        y = y * sg
        for s0 in (64, 128, 192):
            y[..., s0:s0 + 64] = fwht_last(y[..., s0:s0 + 64].copy())
        return y
    raise ValueError(kind)


def rms_gamma(x, w):
    rms = np.sqrt((x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True) + 1e-6)
    return (x / rms.astype(np.float32)) * (1.0 + w.astype(np.float32))


def rope_only(x, positions):
    return rotate_half_rope(x, positions, q_norm=None)


def battery(layer):
    Xh = load_X_hidden(layer)
    Wq = load_tensor(tname(layer, "self_attn.q_proj.weight"))
    Wk = load_tensor(tname(layer, "self_attn.k_proj.weight"))
    qn = load_tensor(tname(layer, "self_attn.q_norm.weight"))
    kn = load_tensor(tname(layer, "self_attn.k_norm.weight"))
    q = gemm_x_wt(Xh, Wq).reshape(-1, N_HEADS, 512)[:, :, :HEAD_DIM]
    k = gemm_x_wt(Xh, Wk).reshape(-1, N_KV, HEAD_DIM)
    T = q.shape[0]
    pos = np.arange(T, dtype=np.int32)
    qn_v = rms_gamma(q, qn)
    kn_v = rms_gamma(k, kn)
    q0 = rope_only(qn_v, pos)
    k0 = rope_only(kn_v, pos)
    S0 = attn_scores(q0, k0)

    kinds = [
        "id", "wh256", "wh64_rotary", "wh64_nonrotary", "wh128_nonrotary_mid",
        "pair_wh2", "pair_rot45", "signs_iid", "signs_pair_const",
        "signs_pair_flip", "generic_orth256", "safe_bundle",
    ]
    rows = []
    for kind in kinds:
        rec = {"kind": kind}
        # A: H on RAW, then norm, then rope (previous protocol)
        rng = np.random.default_rng(12345)
        qt = apply_head(q, kind, rng)
        rng = np.random.default_rng(12345)
        kt = apply_head(k, kind, rng)
        SA = attn_scores(rope_only(rms_gamma(qt, qn), pos), rope_only(rms_gamma(kt, kn), pos))
        rec["raw_then_norm_rope"] = score_cmp(S0, SA)

        # B: norm first, then H, then rope  -- the commutation-with-RoPE test
        rng = np.random.default_rng(12345)
        qt = apply_head(qn_v, kind, rng)
        rng = np.random.default_rng(12345)
        kt = apply_head(kn_v, kind, rng)
        SB = attn_scores(rope_only(qt, pos), rope_only(kt, pos))
        rec["norm_then_H_then_rope"] = score_cmp(S0, SB)

        # C: norm+rope first, then H  -- any orthogonal H must preserve
        rng = np.random.default_rng(12345)
        qt = apply_head(q0, kind, rng)
        rng = np.random.default_rng(12345)
        kt = apply_head(k0, kind, rng)
        SC = attn_scores(qt, kt)
        rec["after_rope"] = score_cmp(S0, SC)

        rec["raw_breaks"] = rec["raw_then_norm_rope"]["max_abs_delta"] > 1e-4
        rec["norm_H_rope_breaks"] = rec["norm_then_H_then_rope"]["max_abs_delta"] > 1e-4
        rec["after_rope_breaks"] = rec["after_rope"]["max_abs_delta"] > 1e-4
        rows.append(rec)
        print(
            f"L{layer} {kind:<22} "
            f"raw={rec['raw_then_norm_rope']['max_abs_delta']:.3e}/{rec['raw_then_norm_rope']['score_cosine']:.6f} "
            f"nHr={rec['norm_then_H_then_rope']['max_abs_delta']:.3e}/{rec['norm_then_H_then_rope']['score_cosine']:.6f} "
            f"post={rec['after_rope']['max_abs_delta']:.3e}/{rec['after_rope']['score_cosine']:.6f}",
            flush=True,
        )
    return rows


def flat_kurt(W):
    x = W.astype(np.float64).reshape(-1)
    c = x - x.mean()
    m2 = float(np.mean(c * c))
    m4 = float(np.mean(c ** 4))
    return (m4 / (m2 * m2) - 3.0) if m2 > 0 else 0.0


def kurt_3994():
    rows = []
    for layer, suffix, cls in [
        (0, "linear_attn.out_proj.weight", "lin_o"),
        (0, "mlp.down_proj.weight", "down"),
        (3, "self_attn.o_proj.weight", "o"),
        (31, "self_attn.o_proj.weight", "o"),
        (63, "self_attn.o_proj.weight", "o"),
    ]:
        W = load_tensor(tname(layer, suffix))
        rec = {
            "layer": layer, "class": cls, "shape": list(W.shape),
            "flat_kurt_id": flat_kurt(W),
            "flat_kurt_drop3994": flat_kurt(np.delete(W, 3994, axis=0)) if W.shape[0] > 3994 else None,
        }
        for xf in ("out_wh_128", "out_wh_256", "out_wh_1024", "out_rht_256", "wh_256", "rht_256"):
            try:
                Wt, meta = apply_named(W, xf)
            except ValueError:
                continue
            rec[f"flat_kurt_{xf}"] = flat_kurt(Wt)
            rec[f"group_{xf}"] = group_stats(Wt)
            del Wt, meta
        rec["group_id"] = group_stats(W)
        print(
            f"KURT L{layer} {cls} id={rec['flat_kurt_id']:.4f} "
            f"drop3994={rec['flat_kurt_drop3994']:.4f} "
            f"out_wh_256={rec.get('flat_kurt_out_wh_256'):.4f} "
            f"out_wh_1024={rec.get('flat_kurt_out_wh_1024'):.4f}",
            flush=True,
        )
        rows.append(rec)
        del W
    return rows


def q_norm_stats(layer=3):
    qn = load_tensor(tname(layer, "self_attn.q_norm.weight"))
    kn = load_tensor(tname(layer, "self_attn.k_norm.weight"))
    return {
        "layer": layer,
        "q_norm_min": float(qn.min()), "q_norm_max": float(qn.max()),
        "q_norm_std": float(qn.std()),
        "q_norm_rotary_std": float(qn[:64].std()),
        "q_norm_nonrotary_std": float(qn[64:].std()),
        "k_norm_min": float(kn.min()), "k_norm_max": float(kn.max()),
        "k_norm_std": float(kn.std()),
        "one_plus_q_range": [float(1 + qn.min()), float(1 + qn.max())],
    }


def main():
    t0 = time.perf_counter()
    out = {"rope_fixed": {}, "kurt_3994": None, "q_norm": {}}
    for layer in (3, 31, 63):
        out["rope_fixed"][str(layer)] = battery(layer)
    out["q_norm"]["3"] = q_norm_stats(3)
    out["q_norm"]["31"] = q_norm_stats(31)
    out["kurt_3994"] = kurt_3994()
    out["wall_s"] = time.perf_counter() - t0
    json.dump(out, open(OUT, "w"), indent=2)
    print("WROTE", OUT, "wall", out["wall_s"], flush=True)


if __name__ == "__main__":
    main()
