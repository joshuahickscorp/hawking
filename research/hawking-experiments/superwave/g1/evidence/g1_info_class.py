#!/usr/bin/env python3
"""Qwen3.8 information-classification measurements.

CPU/numpy only. Peak RSS target << 15 GB. Writes /tmp/g1_info_class.json.
"""
from __future__ import annotations

import gc
import json
import os
import resource
import struct
import time
from collections import defaultdict

import numpy as np

BF16 = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
ACT = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
OUT = "/tmp/g1_info_class.json"
N_SRC = 26895998464
HIDDEN = 5120
INTER = 17408
VOCAB = 248320
LAYERS = 64
ISLANDS = (3994, 3456, 310)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def now() -> float:
    return time.perf_counter()


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    f32 = np.empty(u16.size, dtype=np.float32)
    f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
    return f32


class ShardCache:
    def __init__(self, index_path: str) -> None:
        with open(index_path) as f:
            idx = json.load(f)
        self.weight_map = idx["weight_map"]
        self.meta = idx.get("metadata")
        self._hdr: dict[str, tuple[str, dict]] = {}

    def _header(self, shard: str) -> dict:
        if shard in self._hdr:
            return self._hdr[shard][1]
        path = os.path.join(BF16, shard)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        self._hdr[shard] = (path, hdr)
        return hdr

    def info(self, name: str) -> tuple[str, dict, int]:
        shard = self.weight_map[name]
        path = os.path.join(BF16, shard)
        hdr = self._header(shard)
        hlen = None
        # recover header length from cached file
        if not hasattr(self, "_hlen"):
            self._hlen = {}
        if shard not in self._hlen:
            with open(path, "rb") as f:
                self._hlen[shard] = struct.unpack("<Q", f.read(8))[0]
        return path, hdr[name], 8 + self._hlen[shard]

    def load_f32(self, name: str) -> np.ndarray:
        path, info, base = self.info(name)
        dtype = info["dtype"]
        shape = tuple(info["shape"])
        off0, off1 = info["data_offsets"]
        with open(path, "rb") as f:
            f.seek(base + off0)
            raw = f.read(off1 - off0)
        if dtype == "BF16":
            u16 = np.frombuffer(raw, dtype="<u2")
            arr = bf16_to_f32(u16).reshape(shape)
        elif dtype in ("F32", "F16"):
            dt = np.float32 if dtype == "F32" else np.float16
            arr = np.frombuffer(raw, dtype=dt).astype(np.float32, copy=False).reshape(shape)
            if not arr.flags.writeable:
                arr = arr.copy()
        else:
            raise ValueError(f"{name} dtype={dtype}")
        return arr

    def load_rows_f32(self, name: str, r0: int, r1: int) -> np.ndarray:
        path, info, base = self.info(name)
        dtype = info["dtype"]
        shape = tuple(info["shape"])
        cols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        item = 2 if dtype == "BF16" else 4
        off0, _ = info["data_offsets"]
        nbytes = (r1 - r0) * cols * item
        with open(path, "rb") as f:
            f.seek(base + off0 + r0 * cols * item)
            raw = f.read(nbytes)
        if dtype == "BF16":
            u16 = np.frombuffer(raw, dtype="<u2")
            return bf16_to_f32(u16).reshape(r1 - r0, cols)
        if dtype == "F32":
            return np.frombuffer(raw, dtype="<f4").reshape(r1 - r0, cols).copy()
        raise ValueError(dtype)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def cosine_flat(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64, copy=False)
    b = b.ravel().astype(np.float64, copy=False)
    na = float(np.dot(a, a))
    nb = float(np.dot(b, b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / np.sqrt(na * nb))


def row_stats(mat: np.ndarray) -> dict:
    # mat: [out, in]
    r = np.sqrt(np.mean(mat.astype(np.float64) ** 2, axis=1))
    med = float(np.median(r))
    if med == 0.0:
        xmed = None
    else:
        xmed = float(r.max() / med)
    top = int(np.argmax(r))
    return {
        "n_out": int(mat.shape[0]),
        "n_in": int(mat.shape[1]) if mat.ndim == 2 else 1,
        "row_rms_med": med,
        "row_rms_max": float(r.max()),
        "out_xmed": xmed,
        "top_row": top,
        "row3994_xmed": float(r[3994] / med) if mat.shape[0] > 3994 and med else None,
        "row3456_xmed": float(r[3456] / med) if mat.shape[0] > 3456 and med else None,
        "row310_xmed": float(r[310] / med) if mat.shape[0] > 310 and med else None,
        "n10": int(np.sum(r >= 10.0 * med)) if med else 0,
        "n4": int(np.sum(r >= 4.0 * med)) if med else 0,
        "fro": float(np.sqrt(np.sum(mat.astype(np.float64) ** 2))),
    }


def col_stats(mat: np.ndarray) -> dict:
    c = np.sqrt(np.mean(mat.astype(np.float64) ** 2, axis=0))
    med = float(np.median(c))
    top = int(np.argmax(c))
    return {
        "col_rms_med": med,
        "col_rms_max": float(c.max()),
        "in_xmed": float(c.max() / med) if med else None,
        "top_col": top,
        "col3994_xmed": float(c[3994] / med) if mat.shape[1] > 3994 and med else None,
        "col3994_rank": int(np.sum(c > c[3994]) + 1) if mat.shape[1] > 3994 else None,
        "n10": int(np.sum(c >= 10.0 * med)) if med else 0,
        "n4": int(np.sum(c >= 4.0 * med)) if med else 0,
    }


def vec_stats(v: np.ndarray) -> dict:
    v = v.astype(np.float64, copy=False).ravel()
    return {
        "n": int(v.size),
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "median": float(np.median(v)),
        "n_zero": int(np.sum(v == 0.0)),
        "n_neg": int(np.sum(v < 0.0)),
        "absmax": float(np.max(np.abs(v))),
        "idx_3994": float(v[3994]) if v.size > 3994 else None,
        "rank_3994": int(np.sum(v > v[3994]) + 1) if v.size > 3994 else None,
    }


def main() -> None:
    t0 = now()
    out: dict = {
        "schema": "hawking.g1.information_classification.measure.v1",
        "n_src": N_SRC,
        "peak_rss_gb_start": rss_gb(),
    }
    sc = ShardCache(os.path.join(BF16, "model.safetensors.index.json"))
    lang = [n for n in sc.weight_map if n.startswith("language_model.")]
    vis = [n for n in sc.weight_map if n.startswith("vision_tower.")]
    out["index"] = {
        "n_weight_map": len(sc.weight_map),
        "n_language": len(lang),
        "n_vision": len(vis),
        "metadata": sc.meta,
    }

    # ---- activations ----
    X = np.empty((LAYERS, 256, HIDDEN), dtype=np.float32)
    for L in range(LAYERS):
        path = os.path.join(ACT, "hidden", f"L{L:02d}.f32")
        X[L] = np.fromfile(path, dtype=np.float32).reshape(256, HIDDEN)
    act = {"layers": [], "islands": {}, "pairs": [], "dead": []}
    rms = []
    for L in range(LAYERS):
        x = X[L]
        ch = np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=0))
        med = float(np.median(ch))
        layer_rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        rms.append(layer_rms)
        hot4 = np.where(ch >= 4.0 * med)[0].tolist() if med else []
        hot10 = np.where(ch >= 10.0 * med)[0].tolist() if med else []
        energy = ch ** 2
        e_sum = float(energy.sum())
        rec = {
            "L": L,
            "rms": layer_rms,
            "mean_abs": float(np.mean(np.abs(x))),
            "ch_med": med,
            "n_hot4": len(hot4),
            "n_hot10": len(hot10),
            "hot10": [int(i) for i in hot10[:8]],
            "n_dead_1e8": int(np.sum(ch < 1e-8)),
            "n_low_0p01med": int(np.sum(ch < 0.01 * med)) if med else 0,
            "top8": [int(i) for i in np.argsort(-ch)[:8]],
            "energy_top1_frac": float(energy.max() / e_sum) if e_sum else 0.0,
            "energy_top8_frac": float(np.sort(energy)[-8:].sum() / e_sum) if e_sum else 0.0,
        }
        for c in ISLANDS:
            rec[f"ch{c}_rms"] = float(ch[c])
            rec[f"ch{c}_xmed"] = float(ch[c] / med) if med else None
            rec[f"ch{c}_energy"] = float(energy[c] / e_sum) if e_sum else 0.0
            rec[f"ch{c}_n_zero_tok"] = int(np.sum(x[:, c] == 0.0))
            rec[f"ch{c}_n_hot10_tok"] = int(np.sum(np.abs(x[:, c]) >= 10.0 * med)) if med else 0
        act["layers"].append(rec)

    # island persistence
    for c in ISLANDS:
        n_hot4 = sum(1 for r in act["layers"] if r[f"ch{c}_xmed"] and r[f"ch{c}_xmed"] >= 4)
        n_hot10 = sum(1 for r in act["layers"] if r[f"ch{c}_xmed"] and r[f"ch{c}_xmed"] >= 10)
        act["islands"][str(c)] = {
            "n_hot4": n_hot4,
            "n_hot10": n_hot10,
            "mean_rms": float(np.mean([r[f"ch{c}_rms"] for r in act["layers"]])),
            "mean_energy": float(np.mean([r[f"ch{c}_energy"] for r in act["layers"]])),
        }

    # adjacent persistence / compensation
    persist_hi = []
    cancel_hi = []
    for L in range(LAYERS - 1):
        a = X[L].astype(np.float64)
        b = X[L + 1].astype(np.float64)
        d = b - a
        # per-channel Pearson of tokens
        a0 = a - a.mean(axis=0, keepdims=True)
        b0 = b - b.mean(axis=0, keepdims=True)
        d0 = d - d.mean(axis=0, keepdims=True)
        va = np.sum(a0 * a0, axis=0)
        vb = np.sum(b0 * b0, axis=0)
        vd = np.sum(d0 * d0, axis=0)
        cab = np.sum(a0 * b0, axis=0)
        cda = np.sum(d0 * (-a0), axis=0)
        den_ab = np.sqrt(va * vb)
        den_da = np.sqrt(vd * va)
        corr_persist = np.divide(cab, den_ab, out=np.zeros(HIDDEN), where=den_ab > 0)
        corr_cancel = np.divide(cda, den_da, out=np.zeros(HIDDEN), where=den_da > 0)
        gain = float(np.sqrt(np.mean(b * b)) / max(np.sqrt(np.mean(a * a)), 1e-12))
        rec = {
            "L": L,
            "gain_rms": gain,
            "flat_cos": cosine_flat(X[L], X[L + 1]),
            "mean_corr_persist": float(corr_persist.mean()),
            "p90_corr_persist": float(np.quantile(corr_persist, 0.9)),
            "n_persist_gt0p9": int(np.sum(corr_persist > 0.9)),
            "n_persist_gt0p5": int(np.sum(corr_persist > 0.5)),
            "mean_corr_cancel": float(corr_cancel.mean()),
            "p90_corr_cancel": float(np.quantile(corr_cancel, 0.9)),
            "n_cancel_gt0p5": int(np.sum(corr_cancel > 0.5)),
            "n_cancel_gt0p8": int(np.sum(corr_cancel > 0.8)),
            "ch3994_persist": float(corr_persist[3994]),
            "ch3994_cancel": float(corr_cancel[3994]),
            "ch3456_persist": float(corr_persist[3456]),
            "ch3456_cancel": float(corr_cancel[3456]),
            "ch310_persist": float(corr_persist[310]),
            "ch310_cancel": float(corr_cancel[310]),
        }
        persist_hi.append(rec["n_persist_gt0p9"])
        cancel_hi.append(rec["n_cancel_gt0p5"])
        act["pairs"].append(rec)

    act["summary"] = {
        "L0_rms": rms[0],
        "L63_rms": rms[63],
        "L63_over_L0_rms": rms[63] / rms[0],
        "L63_over_L62_rms": rms[63] / rms[62],
        "mean_n_persist_gt0p9": float(np.mean(persist_hi)),
        "mean_n_cancel_gt0p5": float(np.mean(cancel_hi)),
        "max_n_cancel_gt0p5": int(max(cancel_hi)),
        "max_n_persist_gt0p9": int(max(persist_hi)),
        "mean_pair_flat_cos": float(np.mean([p["flat_cos"] for p in act["pairs"]])),
        "mean_pair_gain": float(np.mean([p["gain_rms"] for p in act["pairs"]])),
        "note": "L63/L0 rms ratio is NOT the contract 2.60396 (that is a different output/input-norm definition). This is hidden-state rms.",
    }
    out["activations"] = act
    out["peak_rss_gb_after_act"] = rss_gb()

    # ---- small tensors ----
    small = {"input_ln": [], "post_ln": [], "q_norm": [], "k_norm": [], "lin_norm": [],
             "A_log": [], "dt_bias": [], "conv1d": [], "final_norm": None}
    name_of = {
        "input_ln": "input_layernorm.weight",
        "post_ln": "post_attention_layernorm.weight",
    }
    for L in range(LAYERS):
        pre = f"language_model.model.layers.{L}."
        inn = sc.load_f32(pre + "input_layernorm.weight")
        post = sc.load_f32(pre + "post_attention_layernorm.weight")
        ir = vec_stats(inn)
        pr = vec_stats(post)
        ir["L"] = L
        pr["L"] = L
        ir["cos_to_post"] = cosine_flat(inn, post)
        small["input_ln"].append(ir)
        small["post_ln"].append(pr)
        mixer_gqa = ((L + 1) % 4 == 0)
        if mixer_gqa:
            qn = sc.load_f32(pre + "self_attn.q_norm.weight")
            kn = sc.load_f32(pre + "self_attn.k_norm.weight")
            qr, kr = vec_stats(qn), vec_stats(kn)
            qr["L"] = L
            kr["L"] = L
            small["q_norm"].append(qr)
            small["k_norm"].append(kr)
        else:
            ln = sc.load_f32(pre + "linear_attn.norm.weight")
            al = sc.load_f32(pre + "linear_attn.A_log")
            dt = sc.load_f32(pre + "linear_attn.dt_bias")
            cv = sc.load_f32(pre + "linear_attn.conv1d.weight")
            lr, ar, dr = vec_stats(ln), vec_stats(al), vec_stats(dt)
            lr["L"] = L
            ar["L"] = L
            dr["L"] = L
            cr = vec_stats(cv)
            cr["L"] = L
            cr["shape"] = list(cv.shape)
            small["lin_norm"].append(lr)
            small["A_log"].append(ar)
            small["dt_bias"].append(dr)
            small["conv1d"].append(cr)

    fn = sc.load_f32("language_model.model.norm.weight")
    small["final_norm"] = vec_stats(fn)
    # adjacent input-ln cosine
    adj_in = [cosine_flat(
        sc.load_f32(f"language_model.model.layers.{L}.input_layernorm.weight"),
        sc.load_f32(f"language_model.model.layers.{L+1}.input_layernorm.weight"),
    ) for L in range(LAYERS - 1)]
    small["input_ln_adj_cos_mean"] = float(np.mean(adj_in))
    small["input_ln_adj_cos_min"] = float(np.min(adj_in))
    small["input_ln_adj_cos_max"] = float(np.max(adj_in))
    small["L7_post_3994"] = small["post_ln"][7]["idx_3994"]
    small["n_post_zero_3994"] = sum(1 for r in small["post_ln"] if r["idx_3994"] == 0.0)
    small["n_input_zero_3994"] = sum(1 for r in small["input_ln"] if r["idx_3994"] == 0.0)
    small["final_3994"] = small["final_norm"]["idx_3994"]
    small["final_3994_rank"] = small["final_norm"]["rank_3994"]
    out["small"] = small
    out["peak_rss_gb_after_small"] = rss_gb()

    # ---- per-layer GEMV relationships + SwiGLU conditional ----
    probe_layers = [0, 3, 6, 7, 15, 31, 32, 47, 58, 62, 63]
    mlp = []
    for L in probe_layers:
        pre = f"language_model.model.layers.{L}."
        gate = sc.load_f32(pre + "mlp.gate_proj.weight")  # [17408, 5120]
        up = sc.load_f32(pre + "mlp.up_proj.weight")
        down = sc.load_f32(pre + "mlp.down_proj.weight")  # [5120, 17408]
        rec = {
            "L": L,
            "gqa": ((L + 1) % 4 == 0),
            "cos_gate_up": cosine_flat(gate, up),
            "cos_down_gateT": cosine_flat(down, gate.T),
            "cos_down_upT": cosine_flat(down, up.T),
            "gate": {**row_stats(gate), **{f"in_{k}": v for k, v in col_stats(gate).items()}},
            "up": {**row_stats(up), **{f"in_{k}": v for k, v in col_stats(up).items()}},
            "down": {**row_stats(down), **{f"in_{k}": v for k, v in col_stats(down).items()}},
        }
        # SwiGLU on captured X (UNCONFIRMED_POST_NORM as mlp-in)
        x = X[L]  # 256 x 5120
        # Yg = x @ gate.T  -> 256 x 17408
        yg = x @ gate.T
        yu = x @ up.T
        gact = silu(yg)
        h = gact * yu
        # per-channel
        g_abs = np.abs(gact)
        h_abs = np.abs(h)
        g_rms = np.sqrt(np.mean(gact.astype(np.float64) ** 2, axis=0))
        h_rms = np.sqrt(np.mean(h.astype(np.float64) ** 2, axis=0))
        h_med = float(np.median(h_rms))
        thresh_g = 1e-3
        thresh_h = 1e-4 * h_med if h_med else 1e-6
        on_g = np.mean(g_abs > thresh_g, axis=0)
        on_h = np.mean(h_abs > thresh_h, axis=0)
        e = h_rms ** 2
        e_sum = float(e.sum()) or 1.0
        order = np.argsort(-e)
        rec["swiglu"] = {
            "site": "UNCONFIRMED_POST_NORM used as mlp-in; ranks only",
            "rows_per_dim": 256 / 17408,
            "thresh_g": thresh_g,
            "thresh_h": thresh_h,
            "frac_ch_never_on_g": float(np.mean(on_g == 0.0)),
            "frac_ch_never_on_h": float(np.mean(on_h == 0.0)),
            "frac_ch_on_lt_10pct_g": float(np.mean(on_g < 0.10)),
            "frac_ch_on_lt_10pct_h": float(np.mean(on_h < 0.10)),
            "frac_ch_on_lt_50pct_g": float(np.mean(on_g < 0.50)),
            "mean_on_frac_g": float(on_g.mean()),
            "mean_on_frac_h": float(on_h.mean()),
            "median_on_frac_g": float(np.median(on_g)),
            "energy_top1_frac": float(e[order[0]] / e_sum),
            "energy_top10pct_frac": float(e[order[:1741]].sum() / e_sum),
            "energy_top50pct_frac": float(e[order[:8704]].sum() / e_sum),
            "n_ch_h_rms_lt_1e6": int(np.sum(h_rms < 1e-6)),
            "n_ch_g_rms_lt_1e6": int(np.sum(g_rms < 1e-6)),
            "mean_abs_g": float(np.mean(g_abs)),
            "mean_abs_h": float(np.mean(h_abs)),
        }
        # down write into residual: energy of reconstructed write
        y_down = h @ down.T  # 256 x 5120
        rec["write"] = {
            "down_write_rms": float(np.sqrt(np.mean(y_down.astype(np.float64) ** 2))),
            "hidden_rms": float(np.sqrt(np.mean(x.astype(np.float64) ** 2))),
            "write_over_hidden": None,
        }
        rec["write"]["write_over_hidden"] = rec["write"]["down_write_rms"] / max(rec["write"]["hidden_rms"], 1e-12)
        rec["write"]["ch3994_write_rms"] = float(np.sqrt(np.mean(y_down[:, 3994].astype(np.float64) ** 2)))
        rec["write"]["ch3994_write_share"] = float(
            np.mean(y_down[:, 3994].astype(np.float64) ** 2) / max(np.mean(y_down.astype(np.float64) ** 2), 1e-18)
        )
        mlp.append(rec)
        del gate, up, down, yg, yu, gact, h, y_down
        gc.collect()
        print(f"mlp L{L} rss={rss_gb():.3f} cos_gu={rec['cos_gate_up']:.6f} "
              f"cos_dgT={rec['cos_down_gateT']:.6f} never_on_h={rec['swiglu']['frac_ch_never_on_h']:.4f}",
              flush=True)

    out["mlp_probe"] = mlp
    out["peak_rss_gb_after_mlp"] = rss_gb()

    # ---- attention write tensors (row 3994) + same-layer q/k/v orthogonality already known ----
    attn = []
    for L in [0, 3, 7, 15, 31, 32, 63]:
        pre = f"language_model.model.layers.{L}."
        gqa = ((L + 1) % 4 == 0)
        rec = {"L": L, "gqa": gqa}
        if gqa:
            for role, suf in [("q", "self_attn.q_proj.weight"),
                              ("k", "self_attn.k_proj.weight"),
                              ("v", "self_attn.v_proj.weight"),
                              ("o", "self_attn.o_proj.weight")]:
                W = sc.load_f32(pre + suf)
                rec[role] = {**row_stats(W), **{f"in_{k}": v for k, v in col_stats(W).items()}}
                if role == "q":
                    # split q | gate  (24 heads * 256 * 2)
                    q = W[: 24 * 256]
                    g = W[24 * 256 :]
                    rec["q_vs_outgate_cos"] = cosine_flat(q, g)
                    rec["q_half_row_stats"] = row_stats(q)
                    rec["outgate_half_row_stats"] = row_stats(g)
                del W
        else:
            for role, suf in [("qkv", "linear_attn.in_proj_qkv.weight"),
                              ("z", "linear_attn.in_proj_z.weight"),
                              ("a", "linear_attn.in_proj_a.weight"),
                              ("b", "linear_attn.in_proj_b.weight"),
                              ("o", "linear_attn.out_proj.weight")]:
                W = sc.load_f32(pre + suf)
                rec[role] = {**row_stats(W), **{f"in_{k}": v for k, v in col_stats(W).items()}}
                if role in ("a", "b"):
                    rec[f"cos_a_b" if role == "b" else "a_loaded"] = True
                del W
            Wa = sc.load_f32(pre + "linear_attn.in_proj_a.weight")
            Wb = sc.load_f32(pre + "linear_attn.in_proj_b.weight")
            rec["cos_a_b"] = cosine_flat(Wa, Wb)
            del Wa, Wb
        attn.append(rec)
        gc.collect()
        print(f"attn L{L} rss={rss_gb():.3f}", flush=True)
    out["attn_probe"] = attn
    out["peak_rss_gb_after_attn"] = rss_gb()

    # ---- embed vs lm_head (streamed) ----
    tile = 4096
    dots = 0.0
    ne = 0.0
    nh = 0.0
    row_cos = []
    row_rms_e = []
    row_rms_h = []
    col_e = np.zeros(HIDDEN, dtype=np.float64)
    col_h = np.zeros(HIDDEN, dtype=np.float64)
    col_dot = np.zeros(HIDDEN, dtype=np.float64)
    n_tiny_e = 0
    n_tiny_h = 0
    # stop-token band
    stop_lo, stop_hi = 248044, 248077
    stop_rms_e = []
    stop_rms_h = []
    stop_cos = []
    for r0 in range(0, VOCAB, tile):
        r1 = min(VOCAB, r0 + tile)
        e = sc.load_rows_f32("language_model.model.embed_tokens.weight", r0, r1)
        h = sc.load_rows_f32("language_model.lm_head.weight", r0, r1)
        e64 = e.astype(np.float64, copy=False)
        h64 = h.astype(np.float64, copy=False)
        dots += float(np.sum(e64 * h64))
        ne += float(np.sum(e64 * e64))
        nh += float(np.sum(h64 * h64))
        er = np.sqrt(np.mean(e64 * e64, axis=1))
        hr = np.sqrt(np.mean(h64 * h64, axis=1))
        dc = np.sum(e64 * h64, axis=1)
        den = np.sqrt(np.sum(e64 * e64, axis=1) * np.sum(h64 * h64, axis=1))
        rc = np.divide(dc, den, out=np.zeros(r1 - r0), where=den > 0)
        row_cos.append(rc)
        row_rms_e.append(er)
        row_rms_h.append(hr)
        col_e += np.sum(e64 * e64, axis=0)
        col_h += np.sum(h64 * h64, axis=0)
        col_dot += np.sum(e64 * h64, axis=0)
        n_tiny_e += int(np.sum(er < 1e-6))
        n_tiny_h += int(np.sum(hr < 1e-6))
        if r0 < stop_hi and r1 > stop_lo:
            a = max(r0, stop_lo) - r0
            b = min(r1, stop_hi) - r0
            stop_rms_e.append(er[a:b])
            stop_rms_h.append(hr[a:b])
            stop_cos.append(rc[a:b])
        del e, h, e64, h64
        if r0 % 32768 == 0:
            print(f"tables {r0}/{VOCAB} rss={rss_gb():.3f}", flush=True)
    rc = np.concatenate(row_cos)
    er = np.concatenate(row_rms_e)
    hr = np.concatenate(row_rms_h)
    flat_cos = float(dots / np.sqrt(ne * nh)) if ne and nh else 0.0
    # column cosine
    den_c = np.sqrt(col_e * col_h)
    col_cos = np.divide(col_dot, den_c, out=np.zeros(HIDDEN), where=den_c > 0)
    out["tables"] = {
        "flat_cosine_embed_lmhead": flat_cos,
        "row_cos_mean": float(rc.mean()),
        "row_cos_median": float(np.median(rc)),
        "row_cos_p05": float(np.quantile(rc, 0.05)),
        "row_cos_p95": float(np.quantile(rc, 0.95)),
        "row_cos_min": float(rc.min()),
        "row_cos_max": float(rc.max()),
        "n_row_cos_gt_0p9": int(np.sum(rc > 0.9)),
        "n_row_cos_gt_0p5": int(np.sum(rc > 0.5)),
        "n_row_cos_lt_0": int(np.sum(rc < 0.0)),
        "embed_row_rms_med": float(np.median(er)),
        "embed_row_rms_min": float(er.min()),
        "embed_row_rms_max": float(er.max()),
        "embed_row_xmed_max": float(er.max() / np.median(er)),
        "lm_row_rms_med": float(np.median(hr)),
        "lm_row_rms_min": float(hr.min()),
        "lm_row_rms_max": float(hr.max()),
        "lm_row_xmed_max": float(hr.max() / np.median(hr)),
        "n_embed_tiny_1e6": n_tiny_e,
        "n_lm_tiny_1e6": n_tiny_h,
        "n_embed_lt_0p1med": int(np.sum(er < 0.1 * np.median(er))),
        "n_lm_lt_0p1med": int(np.sum(hr < 0.1 * np.median(hr))),
        "col3994_embed_xmed": float(np.sqrt(col_e[3994] / VOCAB) / np.median(np.sqrt(col_e / VOCAB))),
        "col3994_lm_xmed": float(np.sqrt(col_h[3994] / VOCAB) / np.median(np.sqrt(col_h / VOCAB))),
        "col3994_cos": float(col_cos[3994]),
        "col_cos_mean": float(col_cos.mean()),
        "stop_band": {
            "lo": stop_lo,
            "hi": stop_hi,
            "embed_rms_mean": float(np.concatenate(stop_rms_e).mean()),
            "lm_rms_mean": float(np.concatenate(stop_rms_h).mean()),
            "row_cos_mean": float(np.concatenate(stop_cos).mean()),
        },
        "note": "256-token capture cannot declare dead vocab. tiny-row count is weight-native.",
    }
    out["peak_rss_gb_after_tables"] = rss_gb()

    # ---- architectural sharing already baked into N ----
    # MHA would be 24 KV heads; GQA is 4. DeltaNet already has 16k/48v.
    gqa_kv_now = 16 * (1024 * 5120) * 2  # k+v
    gqa_kv_mha = 16 * (24 * 256 * 5120) * 2
    out["architectural_share"] = {
        "gqa_kv_elements_now": gqa_kv_now,
        "gqa_kv_if_mha": gqa_kv_mha,
        "already_saved_vs_mha": gqa_kv_mha - gqa_kv_now,
        "already_saved_frac_of_N": (gqa_kv_mha - gqa_kv_now) / N_SRC,
        "note": "This sharing is already in the source ontology. Not a compiler win on this checkpoint.",
    }

    # ---- element census (from shapes, not loading) ----
    census = defaultdict(lambda: {"n": 0, "elems": 0})
    for name in lang:
        path, info, _ = sc.info(name)
        elems = int(np.prod(info["shape"]))
        if "layers." in name:
            rest = name.split("layers.", 1)[1]
            cls = rest.split(".", 1)[1]
        else:
            cls = name
        census[cls]["n"] += 1
        census[cls]["elems"] += elems
    out["census"] = {k: dict(v) for k, v in sorted(census.items(), key=lambda kv: -kv[1]["elems"])}
    out["census_sum_elems"] = int(sum(v["elems"] for v in census.values()))

    out["wall_s"] = now() - t0
    out["peak_rss_gb"] = rss_gb()
    with open(OUT, "w") as f:
        json.dump(out, f)
        f.write("\n")
    print(f"wrote {OUT} wall={out['wall_s']:.1f}s peak_rss={out['peak_rss_gb']:.3f}G", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
    main()
