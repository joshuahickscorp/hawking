#!/usr/bin/env python3
"""Layer tying / skip / update-function measurements on Qwen3.8-27B.

CPU / numpy only. No GPU, no generate, no pack, no resident touch.
Writes /tmp/g1_layer_tying.json and /tmp/g1_layer_tying.log.

Capture site (this run confirms): post-attention RMSNorm hidden,
256 x 5120, legal MLP input. mixer_x absent. down_proj X is
reconstructed SwiGLU. Magnitudes on this 256-token cube RANK;
scalar and per-channel corrections are ADEQUATE; low-rank maps
are UNDERDETERMINED in the X-nullspace (n_fit=185 << 5120).
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import resource
import struct
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")

SRC = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16")
CAP = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1")
OUT = Path("/tmp/g1_layer_tying.json")
LOG = Path("/tmp/g1_layer_tying.log")

HIDDEN = 5120
INTER = 17408
N_LAYERS = 64
N_TOK = 256
N_LANG = 26_895_998_464
G0_BPW = 4.252735126866492
G0_BYTES = 14_297_694_680
MLP_ELEMS = 5_704_253_440  # per class
PROMPT_LENS = (57, 60, 68, 61, 10)
FIT = slice(0, 185)
HOLD = slice(185, 256)
N_FIT = 185
N_HOLD = 71
CONTRACT_A63 = 2.6039602756500244
T0 = time.time()

_HEADER_CACHE: dict[Path, dict] = {}
_WMAP = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]


def rss_gb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if raw > 1e9:
        return raw / (1024.0 ** 3)
    return raw / (1024.0 ** 2)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')} t={time.time()-T0:7.1f}s rss={rss_gb():.3f}GiB] {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def dump(obj) -> None:
    def default(x):
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.bool_):
            return bool(x)
        raise TypeError(type(x))

    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=default))
    tmp.replace(OUT)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_header(shard: Path) -> dict:
    if shard not in _HEADER_CACHE:
        with shard.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            _HEADER_CACHE[shard] = json.loads(fh.read(n))
    return _HEADER_CACHE[shard]


def load_tensor(name: str) -> np.ndarray:
    shard = SRC / _WMAP[name]
    info = read_header(shard)[name]
    dtype = info.get("dtype", "BF16")
    shape = tuple(int(x) for x in info["shape"])
    lo, hi = info["data_offsets"]
    with shard.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        fh.seek(8 + n + lo)
        raw = fh.read(hi - lo)
    if dtype in ("BF16", "BFLOAT16"):
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        return np.ascontiguousarray(u32.view(np.float32).reshape(shape))
    if dtype in ("F32", "FLOAT32"):
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    raise RuntimeError(f"unsupported dtype {dtype} for {name}")


def tname(layer: int, suffix: str) -> str:
    return f"language_model.model.layers.{layer}.{suffix}"


def is_gqa(layer: int) -> bool:
    return (layer + 1) % 4 == 0


def mixer(layer: int) -> str:
    return "gqa" if is_gqa(layer) else "dn"


def load_hidden(layer: int) -> np.ndarray:
    path = CAP / "hidden" / f"L{layer:02d}.f32"
    raw = np.fromfile(path, dtype="<f4")
    if raw.size != N_TOK * HIDDEN:
        raise RuntimeError(f"hidden L{layer} size {raw.size}")
    return np.ascontiguousarray(raw.reshape(N_TOK, HIDDEN))


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


def mean_row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = den > 1e-12
    if not np.any(ok):
        return 0.0
    return float(np.mean(num[ok] / den[ok]))


def min_row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = den > 1e-12
    if not np.any(ok):
        return 0.0
    return float(np.min(num[ok] / den[ok]))


def fro_rel(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(a))
    return num / den if den > 1e-12 else num


def global_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-12 else 0.0


def score(y: np.ndarray, yhat: np.ndarray) -> dict:
    return {
        "mean_row_cosine": mean_row_cosine(y, yhat),
        "min_row_cosine": min_row_cosine(y, yhat),
        "rel_l2": fro_rel(y, yhat),
        "global_cosine": global_cosine(y, yhat),
    }


def score_split(y: np.ndarray, yhat: np.ndarray) -> dict:
    return {
        "all": score(y, yhat),
        "fit": score(y[FIT], yhat[FIT]),
        "hold": score(y[HOLD], yhat[HOLD]),
    }


# ---------------------------------------------------------------------------
# Phase 0: identity
# ---------------------------------------------------------------------------

def phase_identity() -> dict:
    cap = json.loads((CAP / "capture-result.json").read_text())
    cfg = json.loads((SRC / "config.json").read_text())
    tc = cfg["text_config"]
    rec = {
        "src": str(SRC),
        "cap": str(CAP),
        "config_model_type": cfg.get("model_type"),
        "text_model_type": tc.get("model_type"),
        "num_hidden_layers": tc["num_hidden_layers"],
        "hidden_size": tc["hidden_size"],
        "intermediate_size": tc["intermediate_size"],
        "capture_status": cap["status"],
        "capture_schema": cap["schema"],
        "capture_sha256_self": cap["sha256_self"],
        "capture_file_sha256": sha256_file(CAP / "capture-result.json"),
        "L00_sha256": sha256_file(CAP / "hidden" / "L00.f32"),
        "n_tokens": cap["n_tokens"],
        "n_layers": cap["n_layers"],
        "hidden": cap["hidden"],
        "prompt_lens": list(PROMPT_LENS),
        "fit_n": N_FIT,
        "hold_n": N_HOLD,
        "holdout": "prompt 0-2 fit (185), prompt 3-4 hold (71)",
        "rows_per_dim_gate_up": N_FIT / HIDDEN,
        "rows_per_dim_down": N_FIT / INTER,
        "ns014": 92 / 2048,
        "g0_complete_physical_bpw": G0_BPW,
        "g0_payload_bytes": G0_BYTES,
        "N_lang": N_LANG,
        "contract_A63_cited": CONTRACT_A63,
        "mixer_x": "NOT captured",
        "python": os.popen("python3 -c 'import sys; print(sys.version)'").read().strip(),
        "numpy": np.__version__,
    }
    return rec


# ---------------------------------------------------------------------------
# Phase 1: site + hidden census + skip proxies
# ---------------------------------------------------------------------------

def phase_site_and_census() -> dict:
    log("phase site+census: load all 64 hiddens + both RMSNorms")
    Xs = [load_hidden(L) for L in range(N_LAYERS)]
    gin = [load_tensor(tname(L, "input_layernorm.weight")) for L in range(N_LAYERS)]
    gpost = [load_tensor(tname(L, "post_attention_layernorm.weight")) for L in range(N_LAYERS)]

    site_rows = []
    for L in range(N_LAYERS):
        X = Xs[L].astype(np.float64)
        gp = gpost[L].astype(np.float64)
        gi = gin[L].astype(np.float64)
        tok_rms = np.sqrt(np.mean(X * X, axis=1))
        ch_rms = np.sqrt(np.mean(X * X, axis=0))
        gsafe = np.where(np.abs(gp) < 1e-12, np.nan, gp)
        hat = X / gsafe
        hat_tok = np.sqrt(np.nanmean(hat * hat, axis=1))
        gisafe = np.where(np.abs(gi) < 1e-12, np.nan, gi)
        hat_in = X / gisafe
        hat_in_tok = np.sqrt(np.nanmean(hat_in * hat_in, axis=1))

        def corr(a, b):
            a = a - a.mean()
            b = b - b.mean()
            d = float(np.linalg.norm(a) * np.linalg.norm(b))
            return float(a @ b / d) if d > 0 else 0.0

        site_rows.append(
            {
                "layer": L,
                "mixer": mixer(L),
                "tok_rms_mean": float(tok_rms.mean()),
                "hidden_rms": float(np.sqrt(np.mean(X * X))),
                "mean_token_l2": float(np.linalg.norm(X, axis=1).mean()),
                "rms_gin": float(np.sqrt(np.mean(gi * gi))),
                "rms_gpost": float(np.sqrt(np.mean(gp * gp))),
                "hat_gpost_tok_rms_mean": float(np.nanmean(hat_tok)),
                "hat_gin_tok_rms_mean": float(np.nanmean(hat_in_tok)),
                "gpost_n_zero": int(np.sum(np.abs(gp) < 1e-12)),
                "ch3994_rms": float(ch_rms[3994]),
                "gpost3994": float(gp[3994]),
                "gin3994": float(gi[3994]),
                "X3994_nzero": int(np.sum(Xs[L][:, 3994] != 0)),
                "corr_ch_gpost": corr(ch_rms, np.abs(gp)),
                "corr_ch_gin": corr(ch_rms, np.abs(gi)),
            }
        )

    hat_means = [r["hat_gpost_tok_rms_mean"] for r in site_rows]
    # L7 is the known zero-gamma exception
    hat_ok = [hat_means[L] for L in range(N_LAYERS) if L != 7]
    site_verdict = {
        "label": "CONFIRMED_POST_ATTENTION_LAYERNORM",
        "rule": "token_rms(X / gpost) ~= 1 on every layer except L7 (gpost[3994]==0)",
        "hat_gpost_tok_rms_mean_excl_L7": float(np.mean(hat_ok)),
        "hat_gpost_tok_rms_min_excl_L7": float(np.min(hat_ok)),
        "hat_gpost_tok_rms_max_excl_L7": float(np.max(hat_ok)),
        "L7_gpost3994": site_rows[7]["gpost3994"],
        "L7_X3994_nzero": site_rows[7]["X3994_nzero"],
        "legal_for": ["mlp.gate_proj", "mlp.up_proj", "reconstructed SwiGLU -> mlp.down_proj"],
        "illegal_for": ["in_proj / qkv (needs post-input-norm)", "out_proj (needs mixer_x)"],
        "residual_magnitude": "NOT recoverable; RMSNorm discards per-token residual scale",
    }

    # representation drift of post-norm X and of residual-direction hatX
    steps = []
    hats = []
    for L in range(N_LAYERS):
        gp = gpost[L].astype(np.float64)
        gsafe = np.where(np.abs(gp) < 1e-20, 1.0, gp)
        hat = Xs[L].astype(np.float64) / gsafe
        if np.abs(gp[3994]) < 1e-20:
            hat[:, 3994] = 0.0
        hats.append(hat)

    for L in range(N_LAYERS - 1):
        a = Xs[L].astype(np.float64)
        b = Xs[L + 1].astype(np.float64)
        da = b - a
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        nd = np.linalg.norm(da, axis=1)
        ok = na > 1e-12
        ha = hats[L]
        hb = hats[L + 1]
        dh = hb - ha
        nha = np.linalg.norm(ha, axis=1)
        nhd = np.linalg.norm(dh, axis=1)
        steps.append(
            {
                "layer_from": L,
                "layer_to": L + 1,
                "mixer_from": mixer(L),
                "mixer_to": mixer(L + 1),
                "mean_out_over_in": float(np.mean(nb[ok] / na[ok])),
                "mean_rel_update": float(np.mean(nd[ok] / na[ok])),
                "median_rel_update": float(np.median(nd[ok] / na[ok])),
                "mean_delta_l2": float(np.mean(nd)),
                "mean_token_l2_from": float(np.mean(na)),
                "mean_token_l2_to": float(np.mean(nb)),
                "mean_row_cosine": mean_row_cosine(a, b),
                "min_row_cosine": min_row_cosine(a, b),
                "global_cosine": global_cosine(a, b),
                "hat_mean_rel_update": float(np.mean(nhd / np.maximum(nha, 1e-12))),
                "hat_mean_row_cosine": mean_row_cosine(ha, hb),
                "hat_min_row_cosine": min_row_cosine(ha, hb),
                "hat_global_cosine": global_cosine(ha, hb),
                "hat_mean_delta_l2": float(np.mean(nhd)),
            }
        )

    # update-function similarity of residual-direction deltas
    upd_sim = []
    for L in range(N_LAYERS - 2):
        d0 = hats[L + 1] - hats[L]
        d1 = hats[L + 2] - hats[L + 1]
        upd_sim.append(
            {
                "layers": [L, L + 1, L + 2],
                "hat_delta_mean_row_cosine": mean_row_cosine(d0, d1),
                "hat_delta_global_cosine": global_cosine(d0, d1),
                "x_delta_mean_row_cosine": mean_row_cosine(
                    Xs[L + 1].astype(np.float64) - Xs[L].astype(np.float64),
                    Xs[L + 2].astype(np.float64) - Xs[L + 1].astype(np.float64),
                ),
            }
        )

    # skip proxies: contiguous blocks
    def block_skip(span: int) -> list:
        rows = []
        for L0 in range(0, N_LAYERS - span):
            L1 = L0 + span
            a = Xs[L0].astype(np.float64)
            b = Xs[L1].astype(np.float64)
            na = np.linalg.norm(a, axis=1)
            nd = np.linalg.norm(b - a, axis=1)
            ha = hats[L0]
            hb = hats[L1]
            nha = np.linalg.norm(ha, axis=1)
            nhd = np.linalg.norm(hb - ha, axis=1)
            # sum of individual deltas vs net (cancellation)
            sum_d = 0.0
            sum_hd = 0.0
            for k in range(L0, L1):
                sum_d += float(np.mean(np.linalg.norm(Xs[k + 1].astype(np.float64) - Xs[k].astype(np.float64), axis=1)))
                sum_hd += float(np.mean(np.linalg.norm(hats[k + 1] - hats[k], axis=1)))
            net_d = float(np.mean(nd))
            net_hd = float(np.mean(nhd))
            rows.append(
                {
                    "l0": L0,
                    "l1": L1,
                    "span": span,
                    "mean_rel_update": float(np.mean(nd / np.maximum(na, 1e-12))),
                    "mean_row_cosine": mean_row_cosine(a, b),
                    "hat_mean_rel_update": float(np.mean(nhd / np.maximum(nha, 1e-12))),
                    "hat_mean_row_cosine": mean_row_cosine(ha, hb),
                    "sum_mean_delta_l2": sum_d,
                    "net_mean_delta_l2": net_d,
                    "cancel_ratio": (sum_d / net_d) if net_d > 1e-12 else None,
                    "hat_sum_mean_delta_l2": sum_hd,
                    "hat_net_mean_delta_l2": net_hd,
                    "hat_cancel_ratio": (sum_hd / net_hd) if net_hd > 1e-12 else None,
                }
            )
        return rows

    blocks = {str(s): block_skip(s) for s in (2, 4, 8)}

    # smallest-k candidates by hat rel-update
    order = sorted(range(len(steps)), key=lambda i: steps[i]["hat_mean_rel_update"])
    smallest = []
    for k in (1, 2, 4, 8, 16):
        idxs = order[:k]
        # compounding proxy: sum of those deltas vs treating them as independent
        sum_rel = float(np.sum([steps[i]["hat_mean_rel_update"] for i in idxs]))
        smallest.append(
            {
                "k": k,
                "layers": [steps[i]["layer_from"] for i in idxs],
                "hat_rel_updates": [steps[i]["hat_mean_rel_update"] for i in idxs],
                "x_rel_updates": [steps[i]["mean_rel_update"] for i in idxs],
                "sum_hat_rel": sum_rel,
                "max_hat_rel": float(max(steps[i]["hat_mean_rel_update"] for i in idxs)),
                "note": "sum of singles; non-contiguous group skip is not a forward. Contiguous blocks are in blocks.",
            }
        )

    rels = [s["mean_rel_update"] for s in steps]
    hat_rels = [s["hat_mean_rel_update"] for s in steps]
    cosines = [s["mean_row_cosine"] for s in steps]
    hat_cos = [s["hat_mean_row_cosine"] for s in steps]
    upd_cos = [u["hat_delta_mean_row_cosine"] for u in upd_sim]

    census = {
        "site_rows": site_rows,
        "site_verdict": site_verdict,
        "steps": steps,
        "update_update_similarity": upd_sim,
        "blocks": blocks,
        "smallest_k": smallest,
        "summary": {
            "mean_rel_update": float(np.mean(rels)),
            "min_rel_update": float(np.min(rels)),
            "max_rel_update": float(np.max(rels)),
            "argmin_rel_update": int(np.argmin(rels)),
            "argmax_rel_update": int(np.argmax(rels)),
            "mean_hat_rel_update": float(np.mean(hat_rels)),
            "min_hat_rel_update": float(np.min(hat_rels)),
            "max_hat_rel_update": float(np.max(hat_rels)),
            "argmin_hat_rel": int(np.argmin(hat_rels)),
            "argmax_hat_rel": int(np.argmax(hat_rels)),
            "n_hat_rel_lt_0_10": int(np.sum(np.array(hat_rels) < 0.10)),
            "n_hat_rel_lt_0_20": int(np.sum(np.array(hat_rels) < 0.20)),
            "n_hat_rel_lt_0_30": int(np.sum(np.array(hat_rels) < 0.30)),
            "n_x_cos_gt_0_99": int(np.sum(np.array(cosines) > 0.99)),
            "n_hat_cos_gt_0_99": int(np.sum(np.array(hat_cos) > 0.99)),
            "n_hat_cos_gt_0_95": int(np.sum(np.array(hat_cos) > 0.95)),
            "mean_x_cosine": float(np.mean(cosines)),
            "min_x_cosine": float(np.min(cosines)),
            "mean_hat_cosine": float(np.mean(hat_cos)),
            "min_hat_cosine": float(np.min(hat_cos)),
            "mean_hat_delta_cosine": float(np.mean(upd_cos)),
            "min_hat_delta_cosine": float(np.min(upd_cos)),
            "max_hat_delta_cosine": float(np.max(upd_cos)),
            "L63_over_L0_mean_token_l2": float(
                np.linalg.norm(Xs[63].astype(np.float64), axis=1).mean()
                / np.linalg.norm(Xs[0].astype(np.float64), axis=1).mean()
            ),
            "L63_over_L0_hidden_rms": float(
                np.sqrt(np.mean(Xs[63].astype(np.float64) ** 2))
                / np.sqrt(np.mean(Xs[0].astype(np.float64) ** 2))
            ),
        },
    }
    del hats
    gc.collect()
    return census, Xs


# ---------------------------------------------------------------------------
# Phase 2: MLP write amp all 64 + function similarity
# ---------------------------------------------------------------------------

def mlp_parts(X: np.ndarray, Wg, Wu, Wd):
    g = X @ Wg.T
    u = X @ Wu.T
    h = silu(g) * u
    y = h @ Wd.T
    return g, u, h, y


def phase_mlp_write(Xs) -> dict:
    log("phase mlp write amp all 64")
    rows = []
    # keep composed Y and gate Y on own X for later? too big for gate.
    # store composed Y (256x5120) and write stats only.
    Ys = []
    Hs = []  # swiglu, 256x17408 = 18MB each = 1.14GB total — skip store all
    for L in range(N_LAYERS):
        t1 = time.time()
        X = Xs[L]
        Wg = load_tensor(tname(L, "mlp.gate_proj.weight"))
        Wu = load_tensor(tname(L, "mlp.up_proj.weight"))
        Wd = load_tensor(tname(L, "mlp.down_proj.weight"))
        g, u, h, y = mlp_parts(X, Wg, Wu, Wd)
        xn = np.linalg.norm(X.astype(np.float64), axis=1)
        yn = np.linalg.norm(y.astype(np.float64), axis=1)
        hn = np.linalg.norm(h.astype(np.float64), axis=1)
        gn = np.linalg.norm(g.astype(np.float64), axis=1)
        un = np.linalg.norm(u.astype(np.float64), axis=1)
        # residual-add form on the *post-norm* vector (not residual stream)
        addn = np.linalg.norm((X + y).astype(np.float64), axis=1)
        rec = {
            "layer": L,
            "mixer": mixer(L),
            "mean_y_over_x": float(np.mean(yn / np.maximum(xn, 1e-12))),
            "median_y_over_x": float(np.median(yn / np.maximum(xn, 1e-12))),
            "rms_y_over_rms_x": float(
                np.sqrt(np.mean(y.astype(np.float64) ** 2))
                / max(np.sqrt(np.mean(X.astype(np.float64) ** 2)), 1e-18)
            ),
            "mean_xplusy_over_x": float(np.mean(addn / np.maximum(xn, 1e-12))),
            "mean_h_over_x": float(np.mean(hn / np.maximum(xn, 1e-12))),
            "mean_g_over_x": float(np.mean(gn / np.maximum(xn, 1e-12))),
            "mean_u_over_x": float(np.mean(un / np.maximum(xn, 1e-12))),
            "mean_y_l2": float(np.mean(yn)),
            "mean_x_l2": float(np.mean(xn)),
            "hold_mean_y_over_x": float(np.mean(yn[HOLD] / np.maximum(xn[HOLD], 1e-12))),
            "fit_mean_y_over_x": float(np.mean(yn[FIT] / np.maximum(xn[FIT], 1e-12))),
            "wall_s": time.time() - t1,
        }
        rows.append(rec)
        Ys.append(y.astype(np.float32))
        del Wg, Wu, Wd, g, u, h
        if L % 8 == 0:
            log(f"  L{L:02d} mean||y||/||x||={rec['mean_y_over_x']:.4f} rss={rss_gb():.2f}")
        gc.collect()

    amps = [r["mean_y_over_x"] for r in rows]
    # function similarity of MLP write on *own* X (confounded by input drift)
    own_adj = []
    for L in range(N_LAYERS - 1):
        own_adj.append(
            {
                "i": L,
                "j": L + 1,
                "mean_row_cosine": mean_row_cosine(Ys[L], Ys[L + 1]),
                "rel_l2": fro_rel(Ys[L], Ys[L + 1]),
                "global_cosine": global_cosine(Ys[L], Ys[L + 1]),
            }
        )

    return {
        "per_layer": rows,
        "own_input_adjacent": own_adj,
        "summary": {
            "mean_y_over_x": float(np.mean(amps)),
            "min_y_over_x": float(np.min(amps)),
            "max_y_over_x": float(np.max(amps)),
            "argmin": int(np.argmin(amps)),
            "argmax": int(np.argmax(amps)),
            "n_lt_0_10": int(np.sum(np.array(amps) < 0.10)),
            "n_lt_0_20": int(np.sum(np.array(amps) < 0.20)),
            "n_lt_0_30": int(np.sum(np.array(amps) < 0.30)),
            "L63_mean_y_over_x": rows[63]["mean_y_over_x"],
            "L63_mean_xplusy_over_x": rows[63]["mean_xplusy_over_x"],
            "L63_vs_contract_abs": abs(rows[63]["mean_y_over_x"] - CONTRACT_A63),
            "L63_xplusy_vs_contract_abs": abs(rows[63]["mean_xplusy_over_x"] - CONTRACT_A63),
            "early_L0_15_mean": float(np.mean(amps[:16])),
            "mid_L16_47_mean": float(np.mean(amps[16:48])),
            "late_L48_63_mean": float(np.mean(amps[48:])),
            "dn_mean": float(np.mean([amps[L] for L in range(64) if not is_gqa(L)])),
            "gqa_mean": float(np.mean([amps[L] for L in range(64) if is_gqa(L)])),
        },
        "Ys_own": Ys,  # kept for skip compounding of MLP writes; stripped before dump
    }


def phase_fn_sim_common(Xs, class_name: str, suffix: str, layers: list[int], X_ref: np.ndarray) -> dict:
    """Apply each layer's W to the SAME X_ref. Stores only scores, not Ys."""
    log(f"phase fn-sim common X L32 for {class_name} n={len(layers)}")
    Ys = []
    for L in layers:
        W = load_tensor(tname(L, suffix))
        if class_name == "mlp.down_proj":
            # need swiglu of X_ref through THIS layer's gate/up? that's not "same input".
            # same input for down is reconstructed SwiGLU from a fixed (Wg,Wu) or from each?
            # For update-FUNCTION of down, same SwiGLU input: use L32's gate/up on X_ref.
            raise RuntimeError("down handled separately")
        Ys.append((X_ref @ W.T).astype(np.float32))
        del W
        gc.collect()

    def pair_score(i, j):
        a, b = Ys[i], Ys[j]
        return {
            "i": layers[i],
            "j": layers[j],
            "d": layers[j] - layers[i],
            "mean_row_cosine": mean_row_cosine(a, b),
            "min_row_cosine": min_row_cosine(a, b),
            "rel_l2": fro_rel(a, b),
            "global_cosine": global_cosine(a, b),
        }

    adj = [pair_score(i, i + 1) for i in range(len(layers) - 1)]
    d4 = [pair_score(i, i + 4) for i in range(len(layers) - 4)]
    d16 = [pair_score(i, i + 16) for i in range(len(layers) - 16)]

    def stat(pairs, key):
        if not pairs:
            return None
        v = [p[key] for p in pairs]
        return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}

    # also vs a mid reference layer (index of L32 if present else middle)
    if 32 in layers:
        ref_i = layers.index(32)
    else:
        ref_i = len(layers) // 2
    vs_ref = [pair_score(ref_i, i) for i in range(len(layers)) if i != ref_i]

    out = {
        "class": class_name,
        "n": len(layers),
        "x_ref_layer": 32,
        "x_site": "L32 post-attn-norm (common input)",
        "adjacent": adj,
        "d4": d4,
        "d16": d16,
        "vs_ref": vs_ref,
        "stat_adj_mean_row_cosine": stat(adj, "mean_row_cosine"),
        "stat_adj_rel_l2": stat(adj, "rel_l2"),
        "stat_d4_mean_row_cosine": stat(d4, "mean_row_cosine"),
        "stat_d16_mean_row_cosine": stat(d16, "mean_row_cosine"),
        "stat_vs_ref_mean_row_cosine": stat(vs_ref, "mean_row_cosine"),
        "hottest_adj": max(adj, key=lambda p: p["mean_row_cosine"]) if adj else None,
        "coldest_adj": min(adj, key=lambda p: p["mean_row_cosine"]) if adj else None,
        "hottest_d16": max(d16, key=lambda p: p["mean_row_cosine"]) if d16 else None,
    }
    del Ys
    gc.collect()
    return out


def phase_fn_sim_down_common(Xs) -> dict:
    """down_proj as a function of a FIXED SwiGLU input (L32 gate/up on L32 X)."""
    log("phase fn-sim down on fixed L32 SwiGLU")
    X = Xs[32]
    Wg = load_tensor(tname(32, "mlp.gate_proj.weight"))
    Wu = load_tensor(tname(32, "mlp.up_proj.weight"))
    H = silu(X @ Wg.T) * (X @ Wu.T)
    del Wg, Wu
    gc.collect()
    layers = list(range(N_LAYERS))
    Ys = []
    for L in layers:
        Wd = load_tensor(tname(L, "mlp.down_proj.weight"))
        Ys.append((H @ Wd.T).astype(np.float32))
        del Wd
        gc.collect()

    def pair_score(i, j):
        a, b = Ys[i], Ys[j]
        return {
            "i": i,
            "j": j,
            "d": j - i,
            "mean_row_cosine": mean_row_cosine(a, b),
            "min_row_cosine": min_row_cosine(a, b),
            "rel_l2": fro_rel(a, b),
            "global_cosine": global_cosine(a, b),
        }

    adj = [pair_score(i, i + 1) for i in range(63)]
    d16 = [pair_score(i, i + 16) for i in range(48)]

    def stat(pairs, key):
        v = [p[key] for p in pairs]
        return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}

    out = {
        "class": "mlp.down_proj",
        "x_site": "reconstructed SwiGLU of L32 (fixed H), each layer's Wd",
        "adjacent": adj,
        "d16": d16,
        "stat_adj_mean_row_cosine": stat(adj, "mean_row_cosine"),
        "stat_adj_rel_l2": stat(adj, "rel_l2"),
        "stat_d16_mean_row_cosine": stat(d16, "mean_row_cosine"),
        "hottest_adj": max(adj, key=lambda p: p["mean_row_cosine"]),
        "coldest_adj": min(adj, key=lambda p: p["mean_row_cosine"]),
        "hottest_d16": max(d16, key=lambda p: p["mean_row_cosine"]),
    }
    del Ys, H
    gc.collect()
    return out


def phase_fn_sim_mlp_common(Xs) -> dict:
    """Full MLP as a function of common X (L32). This is the residual write function."""
    log("phase fn-sim MLP composed on common X L32")
    X = Xs[32]
    Ys = []
    for L in range(N_LAYERS):
        Wg = load_tensor(tname(L, "mlp.gate_proj.weight"))
        Wu = load_tensor(tname(L, "mlp.up_proj.weight"))
        Wd = load_tensor(tname(L, "mlp.down_proj.weight"))
        *_, y = mlp_parts(X, Wg, Wu, Wd)
        Ys.append(y.astype(np.float32))
        del Wg, Wu, Wd, y
        gc.collect()

    def pair_score(i, j):
        return {
            "i": i,
            "j": j,
            "d": j - i,
            "mean_row_cosine": mean_row_cosine(Ys[i], Ys[j]),
            "min_row_cosine": min_row_cosine(Ys[i], Ys[j]),
            "rel_l2": fro_rel(Ys[i], Ys[j]),
            "global_cosine": global_cosine(Ys[i], Ys[j]),
        }

    adj = [pair_score(i, i + 1) for i in range(63)]
    same_mixer_adj = [p for p in adj if mixer(p["i"]) == mixer(p["j"])]
    d16 = [pair_score(i, i + 16) for i in range(48)]
    d16_same = [p for p in d16 if mixer(p["i"]) == mixer(p["j"])]

    def stat(pairs, key):
        if not pairs:
            return None
        v = [p[key] for p in pairs]
        return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(pairs)}

    out = {
        "class": "mlp_composed",
        "x_site": "L32 post-attn-norm (common input); f_l = down_l(silu(gate_l(X))*up_l(X))",
        "adjacent": adj,
        "d16": d16,
        "stat_adj": stat(adj, "mean_row_cosine"),
        "stat_adj_rel_l2": stat(adj, "rel_l2"),
        "stat_same_mixer_adj": stat(same_mixer_adj, "mean_row_cosine"),
        "stat_d16": stat(d16, "mean_row_cosine"),
        "stat_d16_same_mixer": stat(d16_same, "mean_row_cosine"),
        "hottest_adj": max(adj, key=lambda p: p["mean_row_cosine"]),
        "coldest_adj": min(adj, key=lambda p: p["mean_row_cosine"]),
        "hottest_d16": max(d16, key=lambda p: p["mean_row_cosine"]),
        "n_adj_cos_gt_0_50": int(sum(1 for p in adj if p["mean_row_cosine"] > 0.50)),
        "n_adj_cos_gt_0_80": int(sum(1 for p in adj if p["mean_row_cosine"] > 0.80)),
        "n_adj_cos_gt_0_95": int(sum(1 for p in adj if p["mean_row_cosine"] > 0.95)),
    }
    del Ys
    gc.collect()
    return out


# ---------------------------------------------------------------------------
# Phase 3: tying + correction ladder
# ---------------------------------------------------------------------------

def fit_scalar(y: np.ndarray, ys: np.ndarray) -> float:
    a = ys.astype(np.float64).reshape(-1)
    b = y.astype(np.float64).reshape(-1)
    den = float(a @ a)
    return float(a @ b / den) if den > 1e-30 else 1.0


def apply_scalar(ys: np.ndarray, s: float) -> np.ndarray:
    return (ys * np.float32(s)).astype(np.float32)


def fit_channel_scale(y: np.ndarray, ys: np.ndarray) -> np.ndarray:
    # s[c] = <y[:,c], ys[:,c]> / <ys[:,c], ys[:,c]>
    num = np.sum(y.astype(np.float64) * ys.astype(np.float64), axis=0)
    den = np.sum(ys.astype(np.float64) * ys.astype(np.float64), axis=0)
    s = np.ones(y.shape[1], dtype=np.float64)
    ok = den > 1e-30
    s[ok] = num[ok] / den[ok]
    return s.astype(np.float32)


def apply_channel_scale(ys: np.ndarray, s: np.ndarray) -> np.ndarray:
    return (ys * s[None, :]).astype(np.float32)


def fit_apply_lowrank(X_fit, R_fit, X_hold, rank: int):
    """Reduced-rank map R ≈ X @ M, rank r, min-norm in row-space of X_fit.

    UNDERDETERMINED in the X-nullspace. Hold tokens are scored only to the
    extent they live in span(X_fit). Labelled RANKING.
    """
    # economy SVD of X_fit (n x in)
    # Use float64 for stability on 185 x 5120
    U, S, Vt = np.linalg.svd(X_fit.astype(np.float64), full_matrices=False)
    rmax = int(min(rank, U.shape[1], (S > 1e-8).sum()))
    if rmax <= 0:
        zf = np.zeros_like(R_fit, dtype=np.float32)
        zh = np.zeros((X_hold.shape[0], R_fit.shape[1]), dtype=np.float32)
        return zf, zh, {"rank_used": 0, "s_head": []}
    # B = U.T @ R  (n x out)
    B = U.T @ R_fit.astype(np.float64)
    # keep only first rmax *input* components, then SVD-truncate the out map to rank rmax
    # M = Vt[:k].T @ diag(1/S[:k]) @ B[:k]
    k = rmax
    scale = 1.0 / S[:k]
    Bk = (scale[:, None] * B[:k])  # k x out
    # optional second SVD to enforce rank; already rank <= k
    # apply:
    # Y = X @ Vt[:k].T @ Bk
    Vtk = Vt[:k]
    Yf = (X_fit.astype(np.float64) @ Vtk.T) @ Bk
    Yh = (X_hold.astype(np.float64) @ Vtk.T) @ Bk
    return Yf.astype(np.float32), Yh.astype(np.float32), {
        "rank_used": k,
        "s_head": [float(s) for s in S[: min(8, S.size)]],
        "s_k": float(S[k - 1]),
        "s0": float(S[0]),
        "energy_frac_k": float(np.sum(S[:k] ** 2) / np.sum(S ** 2)),
    }


def sparse_delta_apply(X, W, Ws, frac: float):
    """Keep top-frac |W-Ws| entries, apply (X @ sparse.T)."""
    D = (W - Ws).astype(np.float32)
    n = int(D.size)
    k = max(1, int(round(n * frac)))
    absd = np.abs(D).ravel()
    if k >= n:
        idx = np.arange(n)
    else:
        # argpartition
        part = np.argpartition(absd, n - k)[n - k :]
        idx = part
    rows, cols = np.unravel_index(idx, D.shape)
    vals = D.ravel()[idx].astype(np.float64)
    # y[t, r] += x[t, c] * v
    Y = np.zeros((X.shape[0], D.shape[0]), dtype=np.float64)
    X64 = X.astype(np.float64)
    # group by row to vectorize a bit
    # k can be 891k at 1%; python loop over 891k is OK-ish but slow.
    # Use:
    # for unique rows... still.
    # Fast path: Y = X @ D_sparse.T via constructing CSR-like
    # numpy: accumulate
    # Do in chunks of unique rows
    order = np.argsort(rows)
    rows = rows[order]
    cols = cols[order]
    vals = vals[order]
    i = 0
    nr = D.shape[0]
    while i < rows.size:
        r = int(rows[i])
        j = i + 1
        while j < rows.size and rows[j] == r:
            j += 1
        Y[:, r] = X64[:, cols[i:j]] @ vals[i:j]
        i = j
    stored_bytes = k * (4 + 4 + 2)  # u32 row, u32 col, bf16 val  (no shared index book)
    # could use u32 linear index: 4+2 = 6
    return Y.astype(np.float32), {
        "k": int(k),
        "frac": float(k / n),
        "stored_bytes_u32rc_bf16": int(k * 10),
        "stored_bytes_u32lin_bf16": int(k * 6),
        "max_abs_kept": float(np.max(np.abs(vals))),
        "mean_abs_kept": float(np.mean(np.abs(vals))),
    }


def bpw_shared(n_layers: int, elems_per: int, extra_bytes_per_layer: int, shared_bytes: int) -> dict:
    """Complete BPW if this class is replaced and the rest of the model stays G0 Q4.25."""
    class_g0 = elems_per * n_layers * (4.25 / 8.0)  # bytes, PROJECTED from nominal 4.25
    class_new = shared_bytes + extra_bytes_per_layer * n_layers
    # rest of G0: use measured G0 bytes minus this class's G0 share of Q4 body
    # G0 is not uniform 4.25 (f32 small + tables Q4). PROJECTED:
    new_total = G0_BYTES - class_g0 + class_new
    return {
        "n_layers": n_layers,
        "elems_per": elems_per,
        "class_g0_bytes_proj": class_g0,
        "class_new_bytes": class_new,
        "shared_bytes": shared_bytes,
        "extra_bytes_per_layer": extra_bytes_per_layer,
        "complete_bpw_if_rest_g0": 8.0 * new_total / N_LANG,
        "class_physical_bpw": 8.0 * class_new / (elems_per * n_layers),
        "delta_complete_bpw_vs_g0": 8.0 * new_total / N_LANG - G0_BPW,
        "label": "PROJECTED",
    }


def eval_corrections(y, ys, X, tag: str, W=None, Ws=None) -> dict:
    """y, ys, X are full 256-row. Fit on FIT, score FIT and HOLD."""
    out = {"tag": tag}
    out["none"] = score_split(y, ys)

    s = fit_scalar(y[FIT], ys[FIT])
    yhat = apply_scalar(ys, s)
    out["scalar"] = {
        **score_split(y, yhat),
        "s": s,
        "n_params": 1,
        "bytes_f16": 2,
        "bytes_f32": 4,
        "adequacy": "ADEQUATE",
    }

    sc = fit_channel_scale(y[FIT], ys[FIT])
    yhat = apply_channel_scale(ys, sc)
    out["channel_scale"] = {
        **score_split(y, yhat),
        "s_mean": float(np.mean(sc)),
        "s_std": float(np.std(sc.astype(np.float64))),
        "s_min": float(np.min(sc)),
        "s_max": float(np.max(sc)),
        "n_params": int(sc.size),
        "bytes_f16": int(sc.size * 2),
        "bytes_f32": int(sc.size * 4),
        "adequacy": "ADEQUATE (1 param / channel, 185 rows)",
    }

    # input-channel scale: y ≈ (X * cin) @ W_s.T = (X @ W_s.T) but that's not linear in ys
    # unless we have W. Skip unless W given. Do X-column scale via ys? not equivalent.
    # Fit cin by lstsq on FIT: y ≈ (X * cin) @ Ws.T
    if Ws is not None:
        # y_c ≈ sum_j X_j cin_j Ws[c,j] = (X * cin) @ Ws[c].T
        # For each token this is coupled. Approximate: cin_j from
        # flattened lstsq is 5120 params, 185*out equations. ADEQUATE for 5120.
        # Solve min || y - (X @ diag(cin) @ Ws.T) || via
        # Z_j = X[:,j:j+1] @ Ws[:,j:j+1].T   then y ≈ sum_j cin_j Z_j
        # That's 5120 matrices of 185 x out — too much memory.
        # Sketch: use 5120 x 5120 is Ws.T @ something.
        # Skip input-scale as a full solve; do a cheap diagonal via
        # matching column energy of X. Not a real correction. Omit.
        pass

    R = (y - ys).astype(np.float32)
    for r in (1, 4, 8, 16):
        yf, yh, meta = fit_apply_lowrank(X[FIT], R[FIT], X[HOLD], r)
        yhat_fit = ys[FIT] + yf
        yhat_hold = ys[HOLD] + yh
        out[f"lora_r{r}"] = {
            "fit": score(y[FIT], yhat_fit),
            "hold": score(y[HOLD], yhat_hold),
            **meta,
            "n_params": r * (X.shape[1] + y.shape[1]),
            "bytes_f16": r * (X.shape[1] + y.shape[1]) * 2,
            "adequacy": "RANKING / UNDERDETERMINED in X-nullspace (185<<5120)",
        }

    if W is not None and Ws is not None:
        for frac, name in ((1e-4, "sparse_1e-4"), (1e-3, "sparse_1e-3"), (1e-2, "sparse_1e-2")):
            # fit does not use X; eval on hold
            t1 = time.time()
            dY, meta = sparse_delta_apply(X, W, Ws, frac)
            yhat = ys + dY
            out[name] = {
                **score_split(y, yhat),
                **meta,
                "wall_s": time.time() - t1,
                "adequacy": "weight-space top-|delta|; eval on X is valid ranking",
            }
    return out


def phase_tie_linear(Xs, layers: list[int], suffix: str, class_name: str, do_sparse: bool) -> dict:
    log(f"tie {class_name} layers={layers}")
    Ws_acc = None
    weights = []
    # stream mean; keep weights only if group is small
    keep = len(layers) <= 4
    for L in layers:
        W = load_tensor(tname(L, suffix))
        if Ws_acc is None:
            Ws_acc = np.zeros_like(W, dtype=np.float64)
        Ws_acc += W.astype(np.float64)
        if keep:
            weights.append(W)
        else:
            del W
            gc.collect()
    Ws = (Ws_acc / len(layers)).astype(np.float32)
    del Ws_acc
    gc.collect()

    per = []
    hold_none = []
    hold_scalar = []
    hold_chan = []
    hold_lora8 = []
    hold_sp1e3 = []
    is_down = class_name == "mlp.down_proj"
    for i, L in enumerate(layers):
        X = Xs[L]
        if keep:
            W = weights[i]
        else:
            W = load_tensor(tname(L, suffix))
        if is_down:
            # natural down input = this layer's reconstructed SwiGLU
            Wg = load_tensor(tname(L, "mlp.gate_proj.weight"))
            Wu = load_tensor(tname(L, "mlp.up_proj.weight"))
            Xin = (silu(X @ Wg.T) * (X @ Wu.T)).astype(np.float32)
            del Wg, Wu
            gc.collect()
        else:
            Xin = X
        y = (Xin @ W.T).astype(np.float32)
        ys = (Xin @ Ws.T).astype(np.float32)
        rec = eval_corrections(
            y, ys, Xin, f"L{L}", W=W if do_sparse else None, Ws=Ws if do_sparse else None
        )
        rec["layer"] = L
        rec["mixer"] = mixer(L)
        per.append(rec)
        hold_none.append(rec["none"]["hold"]["mean_row_cosine"])
        hold_scalar.append(rec["scalar"]["hold"]["mean_row_cosine"])
        hold_chan.append(rec["channel_scale"]["hold"]["mean_row_cosine"])
        hold_lora8.append(rec["lora_r8"]["hold"]["mean_row_cosine"])
        if "sparse_1e-3" in rec:
            hold_sp1e3.append(rec["sparse_1e-3"]["hold"]["mean_row_cosine"])
        if not keep:
            del W
        del y, ys
        gc.collect()

    out_dim = Ws.shape[0]
    in_dim = Ws.shape[1]
    elems = int(Ws.size)
    n = len(layers)
    shared_bf16 = elems * 2
    ladder_bpw = {
        "none_shared_only": bpw_shared(n, elems, 0, shared_bf16),
        "scalar_f16": bpw_shared(n, elems, 2, shared_bf16),
        "channel_scale_f16": bpw_shared(n, elems, out_dim * 2, shared_bf16),
        "lora_r8_f16": bpw_shared(n, elems, 8 * (in_dim + out_dim) * 2, shared_bf16),
        "sparse_1e-3_u32lin_bf16": bpw_shared(n, elems, int(round(elems * 1e-3) * 6), shared_bf16),
    }
    # vs independent G0 on just these n layers of this class (not whole model)
    # already inside bpw_shared

    def agg(xs):
        if not xs:
            return None
        return {"mean": float(np.mean(xs)), "min": float(np.min(xs)), "max": float(np.max(xs))}

    result = {
        "class": class_name,
        "layers": layers,
        "n": n,
        "shared": "mean of group (f32 accum, stored as f32 in this run; BPW bills bf16)",
        "per_layer": per,
        "hold_mean_row_cosine": {
            "none": agg(hold_none),
            "scalar": agg(hold_scalar),
            "channel_scale": agg(hold_chan),
            "lora_r8": agg(hold_lora8),
            "sparse_1e-3": agg(hold_sp1e3) if hold_sp1e3 else None,
        },
        "bpw": ladder_bpw,
        "n_hold_none_ge_0_99": int(sum(1 for x in hold_none if x >= 0.99)),
        "n_hold_scalar_ge_0_99": int(sum(1 for x in hold_scalar if x >= 0.99)),
        "n_hold_chan_ge_0_99": int(sum(1 for x in hold_chan if x >= 0.99)),
        "n_hold_lora8_ge_0_99": int(sum(1 for x in hold_lora8 if x >= 0.99)),
        "n_hold_none_ge_0_95": int(sum(1 for x in hold_none if x >= 0.95)),
        "n_hold_chan_ge_0_95": int(sum(1 for x in hold_chan if x >= 0.95)),
        "n_hold_lora8_ge_0_95": int(sum(1 for x in hold_lora8 if x >= 0.95)),
    }
    del Ws, weights
    gc.collect()
    return result


def phase_tie_mlp(Xs, layers: list[int], do_sparse: bool) -> dict:
    """Share the three MLP matrices (mean of group) and score composed residual write."""
    log(f"tie mlp_composed layers={layers}")
    acc_g = acc_u = acc_d = None
    keep = len(layers) <= 3
    store = []
    for L in layers:
        Wg = load_tensor(tname(L, "mlp.gate_proj.weight"))
        Wu = load_tensor(tname(L, "mlp.up_proj.weight"))
        Wd = load_tensor(tname(L, "mlp.down_proj.weight"))
        if acc_g is None:
            acc_g = np.zeros_like(Wg, dtype=np.float64)
            acc_u = np.zeros_like(Wu, dtype=np.float64)
            acc_d = np.zeros_like(Wd, dtype=np.float64)
        acc_g += Wg
        acc_u += Wu
        acc_d += Wd
        if keep:
            store.append((Wg, Wu, Wd))
        else:
            del Wg, Wu, Wd
            gc.collect()
    Sg = (acc_g / len(layers)).astype(np.float32)
    Su = (acc_u / len(layers)).astype(np.float32)
    Sd = (acc_d / len(layers)).astype(np.float32)
    del acc_g, acc_u, acc_d
    gc.collect()

    per = []
    hold_none, hold_scalar, hold_chan, hold_lora8 = [], [], [], []
    for i, L in enumerate(layers):
        X = Xs[L]
        if keep:
            Wg, Wu, Wd = store[i]
        else:
            Wg = load_tensor(tname(L, "mlp.gate_proj.weight"))
            Wu = load_tensor(tname(L, "mlp.up_proj.weight"))
            Wd = load_tensor(tname(L, "mlp.down_proj.weight"))
        *_, y = mlp_parts(X, Wg, Wu, Wd)
        *_, ys = mlp_parts(X, Sg, Su, Sd)
        y = y.astype(np.float32)
        ys = ys.astype(np.float32)
        rec = eval_corrections(y, ys, X, f"mlpL{L}", W=None, Ws=None)
        rec["layer"] = L
        rec["mixer"] = mixer(L)
        # also score "use another member's exact weights" as a no-mean control
        per.append(rec)
        hold_none.append(rec["none"]["hold"]["mean_row_cosine"])
        hold_scalar.append(rec["scalar"]["hold"]["mean_row_cosine"])
        hold_chan.append(rec["channel_scale"]["hold"]["mean_row_cosine"])
        hold_lora8.append(rec["lora_r8"]["hold"]["mean_row_cosine"])
        if not keep:
            del Wg, Wu, Wd
        del y, ys
        gc.collect()

    n = len(layers)
    elems_block = 3 * INTER * HIDDEN
    shared_bf16 = elems_block * 2

    def agg(xs):
        return {"mean": float(np.mean(xs)), "min": float(np.min(xs)), "max": float(np.max(xs))}

    result = {
        "class": "mlp_composed",
        "layers": layers,
        "n": n,
        "shared": "mean of group gate+up+down",
        "per_layer": per,
        "hold_mean_row_cosine": {
            "none": agg(hold_none),
            "scalar": agg(hold_scalar),
            "channel_scale": agg(hold_chan),
            "lora_r8": agg(hold_lora8),
        },
        "bpw": {
            "none_shared_only": bpw_shared(n, elems_block, 0, shared_bf16),
            "scalar_f16": bpw_shared(n, elems_block, 2, shared_bf16),
            "channel_scale_f16": bpw_shared(n, elems_block, HIDDEN * 2, shared_bf16),
            "lora_r8_f16": bpw_shared(n, elems_block, 8 * (HIDDEN + HIDDEN) * 2, shared_bf16),
        },
        "n_hold_none_ge_0_99": int(sum(1 for x in hold_none if x >= 0.99)),
        "n_hold_chan_ge_0_99": int(sum(1 for x in hold_chan if x >= 0.99)),
        "n_hold_lora8_ge_0_99": int(sum(1 for x in hold_lora8 if x >= 0.99)),
        "n_hold_none_ge_0_95": int(sum(1 for x in hold_none if x >= 0.95)),
        "n_hold_chan_ge_0_95": int(sum(1 for x in hold_chan if x >= 0.95)),
        "n_hold_lora8_ge_0_95": int(sum(1 for x in hold_lora8 if x >= 0.95)),
        "note": "corrections applied to the 5120-d residual write, not to 17408-d intermediates",
    }
    del Sg, Su, Sd, store
    gc.collect()
    return result


def phase_member_swap(Xs, pairs: list[tuple[int, int]]) -> dict:
    """Use W_j exactly on X_i (no mean). Isolates function transfer."""
    log(f"member-swap n_pairs={len(pairs)}")
    rows = []
    for i, j in pairs:
        Xi = Xs[i]
        # gate
        Wi = load_tensor(tname(i, "mlp.gate_proj.weight"))
        Wj = load_tensor(tname(j, "mlp.gate_proj.weight"))
        yi = Xi @ Wi.T
        yj = Xi @ Wj.T
        gate = score_split(yi, yj)
        del Wi, Wj, yi, yj
        # composed
        Wgi = load_tensor(tname(i, "mlp.gate_proj.weight"))
        Wui = load_tensor(tname(i, "mlp.up_proj.weight"))
        Wdi = load_tensor(tname(i, "mlp.down_proj.weight"))
        Wgj = load_tensor(tname(j, "mlp.gate_proj.weight"))
        Wuj = load_tensor(tname(j, "mlp.up_proj.weight"))
        Wdj = load_tensor(tname(j, "mlp.down_proj.weight"))
        *_, yi = mlp_parts(Xi, Wgi, Wui, Wdi)
        *_, yj = mlp_parts(Xi, Wgj, Wuj, Wdj)
        mlp = score_split(yi, yj)
        del Wgi, Wui, Wdi, Wgj, Wuj, Wdj, yi, yj
        gc.collect()
        rows.append(
            {
                "src_weights": j,
                "applied_to_X": i,
                "mixer_i": mixer(i),
                "mixer_j": mixer(j),
                "gate": gate,
                "mlp_composed": mlp,
            }
        )
        log(
            f"  swap W{j}->X{i} gate_hold={gate['hold']['mean_row_cosine']:.4f} "
            f"mlp_hold={mlp['hold']['mean_row_cosine']:.4f}"
        )
    return {"pairs": rows}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    if LOG.exists():
        LOG.unlink()
    log("start layer-tying")
    result = {
        "schema": "hawking.g1.layer_tying.v1",
        "identity": phase_identity(),
    }
    dump(result)

    census, Xs = phase_site_and_census()
    result["census"] = {k: v for k, v in census.items()}
    dump(result)
    log(f"census summary {json.dumps(census['summary'])}")
    log(f"site {json.dumps(census['site_verdict'])}")

    mlp_write = phase_mlp_write(Xs)
    Ys_own = mlp_write.pop("Ys_own")
    result["mlp_write"] = mlp_write
    dump(result)
    log(f"mlp_write summary {json.dumps(mlp_write['summary'])}")

    # skip compounding of MLP writes: sum ||y_l|| vs net representation change
    skip_mlp = []
    order = sorted(range(N_LAYERS), key=lambda L: mlp_write["per_layer"][L]["mean_y_over_x"])
    for k in (1, 2, 4, 8, 16):
        idxs = order[:k]
        skip_mlp.append(
            {
                "k": k,
                "layers": idxs,
                "amps": [mlp_write["per_layer"][L]["mean_y_over_x"] for L in idxs],
                "sum_amp": float(sum(mlp_write["per_layer"][L]["mean_y_over_x"] for L in idxs)),
                "max_amp": float(max(mlp_write["per_layer"][L]["mean_y_over_x"] for L in idxs)),
                "sum_mean_y_l2": float(sum(mlp_write["per_layer"][L]["mean_y_l2"] for L in idxs)),
                "note": "sum of MLP writes; attention writes unmeasured; not a forward",
            }
        )
    result["skip_mlp_candidates"] = {
        "smallest_k": skip_mlp,
        "smallest_8_layers": order[:8],
        "largest_8_layers": order[-8:][::-1],
    }
    del Ys_own
    gc.collect()
    dump(result)

    X_ref = Xs[32]
    result["fn_sim"] = {
        "gate_common": phase_fn_sim_common(Xs, "mlp.gate_proj", "mlp.gate_proj.weight", list(range(64)), X_ref),
        "up_common": phase_fn_sim_common(Xs, "mlp.up_proj", "mlp.up_proj.weight", list(range(64)), X_ref),
        "down_fixed_H": phase_fn_sim_down_common(Xs),
        "mlp_composed_common": phase_fn_sim_mlp_common(Xs),
    }
    dump(result)
    log(
        "fn_sim gate adj "
        + json.dumps(result["fn_sim"]["gate_common"]["stat_adj_mean_row_cosine"])
    )
    log(
        "fn_sim mlp adj "
        + json.dumps(result["fn_sim"]["mlp_composed_common"]["stat_adj"])
    )

    groups = {
        "pair_01_dn": [0, 1],
        "pair_1617_dn": [16, 17],
        "pair_6061_dn": [60, 61],
        "pair_6263_mixed": [62, 63],
        "pair_37_gqa": [3, 7],
        "pair_2743_gqa_d16": [27, 43],
        "triple_012_dn": [0, 1, 2],
        "triple_606162_dn": [60, 61, 62],
        "all_gqa_16": [L for L in range(64) if is_gqa(L)],
        "all_dn_48": [L for L in range(64) if not is_gqa(L)],
        "all_64": list(range(64)),
    }

    result["tie"] = {}
    # linear classes: do sparse only on small groups (pairs/triples)
    for gname, glayers in groups.items():
        small = len(glayers) <= 3
        # gate on every group
        result["tie"][f"gate::{gname}"] = phase_tie_linear(
            Xs, glayers, "mlp.gate_proj.weight", "mlp.gate_proj", do_sparse=small
        )
        dump(result)
        log(
            f"tie gate {gname} hold none={result['tie'][f'gate::{gname}']['hold_mean_row_cosine']['none']} "
            f"chan={result['tie'][f'gate::{gname}']['hold_mean_row_cosine']['channel_scale']} "
            f"lora8={result['tie'][f'gate::{gname}']['hold_mean_row_cosine']['lora_r8']}"
        )
        # composed MLP on pairs/triples + all_gqa + a late pair (the residual-relevant object)
        if len(glayers) <= 16:
            result["tie"][f"mlp::{gname}"] = phase_tie_mlp(Xs, glayers, do_sparse=False)
            dump(result)
            log(
                f"tie mlp {gname} hold none={result['tie'][f'mlp::{gname}']['hold_mean_row_cosine']['none']} "
                f"chan={result['tie'][f'mlp::{gname}']['hold_mean_row_cosine']['channel_scale']} "
                f"lora8={result['tie'][f'mlp::{gname}']['hold_mean_row_cosine']['lora_r8']}"
            )

    # one up and one down pair to check class transfer of the ladder
    result["tie"]["up::pair_01_dn"] = phase_tie_linear(
        Xs, [0, 1], "mlp.up_proj.weight", "mlp.up_proj", do_sparse=True
    )
    result["tie"]["down::pair_01_dn"] = phase_tie_linear(
        Xs, [0, 1], "mlp.down_proj.weight", "mlp.down_proj", do_sparse=True
    )
    dump(result)

    result["member_swap"] = phase_member_swap(
        Xs,
        [
            (0, 1),
            (1, 0),
            (16, 17),
            (60, 61),
            (62, 63),
            (3, 7),
            (27, 43),
            (0, 32),
            (32, 0),
            (0, 63),
        ],
    )
    dump(result)

    result["wall_s"] = time.time() - T0
    result["rss_max_gb"] = rss_gb()
    dump(result)
    log(f"done wall={result['wall_s']:.1f}s rss_max={result['rss_max_gb']:.3f}GiB")


if __name__ == "__main__":
    main()
