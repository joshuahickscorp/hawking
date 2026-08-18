#!/usr/bin/env python3
"""Qwen3.8 embed + lm_head decision-level sensitivity. CPU only. No GPU.

Writes /tmp/g1_endpoints_sensitivity.json and prints a compact log.
Peak working set is streamed: never a full f32 248320x5120 table.
"""
from __future__ import annotations

import json
import os
import resource
import struct
import time
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

SRC = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16")
CAP = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1")
OUT = Path("/tmp/g1_endpoints_sensitivity.json")

HIDDEN = 5120
VOCAB = 248320
N_TOK = 256
N_LAYERS = 64
EPS = 1.0e-6
N_LANG = 26_895_998_464
E_TAB = 1_271_398_400
CHUNK = 2048
STOP_LO, STOP_HI = 248044, 248076  # inclusive; fidelity only, not identity

T0 = time.perf_counter()


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] t={time.perf_counter()-T0:7.1f}s rss={rss_gb():.3f}G {msg}", flush=True)


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def percentiles(x: np.ndarray, ps=None) -> dict:
    if ps is None:
        ps = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{p}": None for p in ps} | {"n": 0, "mean": None}
    return {f"p{p}": float(np.percentile(x, p)) for p in ps} | {
        "n": int(x.size),
        "mean": float(np.mean(x)),
    }


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = EPS) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    w64 = np.asarray(w, dtype=np.float64)
    ms = np.mean(x64 * x64, axis=-1, keepdims=True)
    return ((x64 / np.sqrt(ms + eps)) * w64).astype(np.float32)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
        b = b[None, :]
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.zeros(a.shape[0], dtype=np.float64)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(a))
    return num / den if den > 1e-12 else num


def f16_scale(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


# ---------------------------------------------------------------------------
# codecs (production-faithful reconstruct, streamed per chunk)
# ---------------------------------------------------------------------------

def quant_recon(W: np.ndarray, bits: int, group: int, *, q4_asymmetric: bool = False) -> np.ndarray:
    """Absmax / bound, f16 scale, RTN.

    Qn (bits!=4 or not q4_asymmetric): bound=(1<<(bits-1))-1, clamp [-bound, bound].
    Q4 production: scale=max/7, clamp [-8, 7] (qwen80_uniform_q4.rs:233-248).
    """
    flat = np.ascontiguousarray(W, dtype=np.float32).reshape(-1)
    n = int(flat.size)
    groups = (n + group - 1) // group
    padded = np.zeros((groups, group), dtype=np.float32)
    padded.reshape(-1)[:n] = flat
    absmax = np.max(np.abs(padded), axis=1)
    if q4_asymmetric and bits == 4:
        bound = 7.0
        lo, hi = -8.0, 7.0
    else:
        bound = float((1 << (bits - 1)) - 1)
        lo, hi = -bound, bound
    scale = f16_scale(absmax / max(bound, 1.0))
    den = np.where(scale > 0.0, scale, 1.0)
    codes = np.rint(padded / den[:, None]).clip(lo, hi)
    recon = (codes * scale[:, None]).reshape(-1)[:n]
    return recon.reshape(W.shape).astype(np.float32)


def binary_meanabs_recon(W: np.ndarray, group: int) -> np.ndarray:
    """HGRAVB01-like: sign * group mean-abs, f16 scale."""
    flat = np.ascontiguousarray(W, dtype=np.float32).reshape(-1)
    n = int(flat.size)
    groups = (n + group - 1) // group
    padded = np.zeros((groups, group), dtype=np.float32)
    padded.reshape(-1)[:n] = flat
    scale = f16_scale(np.mean(np.abs(padded), axis=1))
    signs = np.where(padded >= 0.0, 1.0, -1.0)
    recon = (signs * scale[:, None]).reshape(-1)[:n]
    return recon.reshape(W.shape).astype(np.float32)


def payload_bytes(rows: int, cols: int, bits: int, group: int) -> int:
    gpr = (cols + group - 1) // group
    code_b = (group * bits + 7) // 8
    return int(rows * gpr * (code_b + 2))


def payload_bpw(rows: int, cols: int, bits: int, group: int) -> float:
    return 8.0 * payload_bytes(rows, cols, bits, group) / float(rows * cols)


CODECS = [
    {"name": "binary_g128", "kind": "binary", "bits": 1, "group": 128, "q4_asymmetric": False},
    {"name": "q2_g128", "kind": "qn", "bits": 2, "group": 128, "q4_asymmetric": False},
    {"name": "q3_g128", "kind": "qn", "bits": 3, "group": 128, "q4_asymmetric": False},
    {"name": "q3_g64", "kind": "qn", "bits": 3, "group": 64, "q4_asymmetric": False},
    {"name": "q4_g64", "kind": "q4", "bits": 4, "group": 64, "q4_asymmetric": True},
    {"name": "q8_g64", "kind": "qn", "bits": 8, "group": 64, "q4_asymmetric": False},
]


def apply_codec(W: np.ndarray, spec: dict) -> np.ndarray:
    if spec["kind"] == "binary":
        return binary_meanabs_recon(W, spec["group"])
    return quant_recon(W, spec["bits"], spec["group"], q4_asymmetric=spec["q4_asymmetric"])


# ---------------------------------------------------------------------------
# safetensors
# ---------------------------------------------------------------------------

_WMAP = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]
_HEADER: dict[Path, tuple[dict, int]] = {}


def _header(shard: Path) -> tuple[dict, int]:
    if shard not in _HEADER:
        with shard.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        _HEADER[shard] = (hdr, 8 + n)
    return _HEADER[shard]


def load_f32(name: str) -> np.ndarray:
    shard = SRC / _WMAP[name]
    hdr, data0 = _header(shard)
    info = hdr[name]
    shape = tuple(int(x) for x in info["shape"])
    lo, hi = info["data_offsets"]
    with shard.open("rb") as fh:
        fh.seek(data0 + lo)
        raw = fh.read(hi - lo)
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return np.ascontiguousarray((u16.astype(np.uint32) << 16).view(np.float32).reshape(shape))


def open_bf16_memmap(name: str) -> tuple[np.memmap, tuple[int, ...]]:
    shard = SRC / _WMAP[name]
    hdr, data0 = _header(shard)
    info = hdr[name]
    shape = tuple(int(x) for x in info["shape"])
    lo, _hi = info["data_offsets"]
    mm = np.memmap(shard, dtype=np.uint16, mode="r", offset=data0 + lo, shape=shape)
    return mm, shape


def bf16u16_to_f32(u16: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32))


def load_hidden(layer: int) -> np.ndarray:
    path = CAP / "hidden" / f"L{layer:02d}.f32"
    raw = np.fromfile(path, dtype="<f4")
    if raw.size != N_TOK * HIDDEN:
        raise RuntimeError(f"hidden L{layer} size {raw.size}")
    return np.ascontiguousarray(raw.reshape(N_TOK, HIDDEN))


# ---------------------------------------------------------------------------
# capture tokens
# ---------------------------------------------------------------------------

def load_capture() -> dict:
    cap = json.loads((CAP / "capture-result.json").read_text())
    ids: list[int] = []
    spans: list[tuple[int, int, str]] = []
    for p in cap["prompts"]:
        a = len(ids)
        pids = [int(x) for x in p["ids"]]
        ids.extend(pids)
        spans.append((a, len(ids), p["prompt"]))
    if len(ids) != N_TOK:
        raise RuntimeError(f"token ids {len(ids)} != 256")
    teacher = np.full(N_TOK, -1, dtype=np.int64)
    is_last = np.zeros(N_TOK, dtype=bool)
    prompt_ix = np.zeros(N_TOK, dtype=np.int32)
    for pi, (a, b, _s) in enumerate(spans):
        prompt_ix[a:b] = pi
        is_last[b - 1] = True
        if b - a >= 2:
            teacher[a : b - 1] = np.asarray(ids[a + 1 : b], dtype=np.int64)
    return {
        "cap": cap,
        "ids": np.asarray(ids, dtype=np.int64),
        "spans": spans,
        "teacher": teacher,
        "is_last": is_last,
        "prompt_ix": prompt_ix,
    }


def token_class(tid: int) -> str:
    if STOP_LO <= tid <= STOP_HI:
        return "special_248044_248076"
    if tid >= 248000:
        return "high_id"
    if tid < 256:
        return "byte"
    return "content"


# ---------------------------------------------------------------------------
# main measurements
# ---------------------------------------------------------------------------

def measure_amplification(H: np.ndarray) -> dict:
    # H: [64, 256, 5120]
    l2 = np.linalg.norm(H.astype(np.float64), axis=2)  # [64, 256]
    rms = np.sqrt(np.mean(np.square(H.astype(np.float64)), axis=2))
    layer = []
    for l in range(N_LAYERS):
        rec = {
            "layer": l,
            "mean_l2": float(np.mean(l2[l])),
            "mean_rms": float(np.mean(rms[l])),
            "capture_rms": None,
        }
        if l + 1 < N_LAYERS:
            ratio = l2[l + 1] / np.maximum(l2[l], 1e-20)
            rec["next_over_this_mean"] = float(np.mean(ratio))
            rec["next_over_this"] = percentiles(ratio)
            rec["cosine_to_next"] = percentiles(cosine_rows(H[l], H[l + 1]))
        layer.append(rec)
    amp_end = l2[63] / np.maximum(l2[0], 1e-20)
    amp_step63 = l2[63] / np.maximum(l2[62], 1e-20)
    # product of consecutive mean ratios
    step_means = [layer[l]["next_over_this_mean"] for l in range(63)]
    return {
        "per_layer": layer,
        "l2_L0_mean": float(np.mean(l2[0])),
        "l2_L63_mean": float(np.mean(l2[63])),
        "mean_l2_L63_over_L0": float(np.mean(l2[63]) / max(np.mean(l2[0]), 1e-20)),
        "mean_of_per_token_L63_over_L0": float(np.mean(amp_end)),
        "per_token_L63_over_L0": percentiles(amp_end),
        "mean_of_per_token_L63_over_L62": float(np.mean(amp_step63)),
        "per_token_L63_over_L62": percentiles(amp_step63),
        "product_of_consecutive_mean_ratios": float(np.prod(np.asarray(step_means, dtype=np.float64))),
        "cosine_L0_L63": percentiles(cosine_rows(H[0], H[63])),
        "label": "MEASURED from activation-capture-v1 hidden/L00..L63.f32",
    }


def site_identify(ids: np.ndarray, H0: np.ndarray, H63: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    log("load embed rows for 256 tokens + L0/L63 norms")
    mm, _shape = open_bf16_memmap("language_model.model.embed_tokens.weight")
    E = bf16u16_to_f32(np.asarray(mm[ids]))  # 256 x 5120
    del mm
    w0 = load_f32("language_model.model.layers.0.input_layernorm.weight")
    w63 = load_f32("language_model.model.layers.63.input_layernorm.weight")
    wfn = load_f32("language_model.model.norm.weight")
    Rn = rmsnorm(E, w0)
    # invert-ish: if H0 = rmsnorm(residual, w0), residual || H0/w0
    rec = {
        "embed_vs_H0_cosine_mean": float(np.mean(cosine_rows(E, H0))),
        "embed_vs_H0_rel_l2": rel_l2(H0, E),
        "rmsnorm_embed_w0_vs_H0_cosine_mean": float(np.mean(cosine_rows(Rn, H0))),
        "rmsnorm_embed_w0_vs_H0_rel_l2": rel_l2(H0, Rn),
        "rmsnorm_embed_w0_vs_H0_max_abs": float(np.max(np.abs(Rn - H0))),
        "rmsnorm_H63_wfn_vs_H63_cosine_mean": float(np.mean(cosine_rows(rmsnorm(H63, wfn), H63))),
        "w0_l2": float(np.linalg.norm(w0.astype(np.float64))),
        "w63_l2": float(np.linalg.norm(w63.astype(np.float64))),
        "wfn_l2": float(np.linalg.norm(wfn.astype(np.float64))),
        "cosine_w0_wfn": float(cosine_rows(w0, wfn)[0]),
        "cosine_w63_wfn": float(cosine_rows(w63, wfn)[0]),
        "H0_rms_mean": float(np.sqrt(np.mean(np.square(H0.astype(np.float64))))),
        "E_rms_mean": float(np.sqrt(np.mean(np.square(E.astype(np.float64))))),
        "Rn_rms_mean": float(np.sqrt(np.mean(np.square(Rn.astype(np.float64))))),
    }
    # site verdict
    if rec["rmsnorm_embed_w0_vs_H0_rel_l2"] < 1e-3:
        rec["site_H0"] = "MEASURED_POST_INPUT_RMSNORM_OF_EMBED"
    elif rec["embed_vs_H0_rel_l2"] < 1e-3:
        rec["site_H0"] = "MEASURED_RAW_EMBED"
    else:
        rec["site_H0"] = "UNCONFIRMED_NOT_EMBED_OR_SIMPLE_RMSNORM"
    rec["site_H63"] = "SAME_SITE_AS_H0_LAYER_63_NOT_FINAL_NORM"
    rec["lm_head_X"] = "PROXY_L63_POST_NORM_NOT_CONFIRMED_FINAL"
    return rec, E, w0, wfn


def embed_row_census(ids: np.ndarray) -> dict:
    log("embed full-vocab row census + codec row-cosine")
    mm, shape = open_bf16_memmap("language_model.model.embed_tokens.weight")
    assert shape == (VOCAB, HIDDEN)
    # running moments
    row_l2 = np.empty(VOCAB, dtype=np.float32)
    row_max = np.empty(VOCAB, dtype=np.float32)
    # per-codec cosine accumulators: we store only percentiles via reservoir? 
    # 248320 floats per codec is 1 MB — keep them.
    cos_store = {s["name"]: np.empty(VOCAB, dtype=np.float32) for s in CODECS}
    nrms_store = {s["name"]: np.empty(VOCAB, dtype=np.float32) for s in CODECS}
    # post-RMSNorm cosine needs w0 — load it
    w0 = load_f32("language_model.model.layers.0.input_layernorm.weight")
    for s in range(0, VOCAB, CHUNK):
        e = min(s + CHUNK, VOCAB)
        W = bf16u16_to_f32(np.asarray(mm[s:e]))
        row_l2[s:e] = np.linalg.norm(W.astype(np.float64), axis=1).astype(np.float32)
        row_max[s:e] = np.max(np.abs(W), axis=1)
        Rn = rmsnorm(W, w0)
        for spec in CODECS:
            Wq = apply_codec(W, spec)
            cos_store[spec["name"]][s:e] = cosine_rows(W, Wq).astype(np.float32)
            nrms_store[spec["name"]][s:e] = cosine_rows(Rn, rmsnorm(Wq, w0)).astype(np.float32)
        if s % (CHUNK * 16) == 0:
            log(f"  embed census rows {s}:{e}")
    del mm
    observed = np.zeros(VOCAB, dtype=bool)
    observed[ids] = True
    special = np.zeros(VOCAB, dtype=bool)
    special[STOP_LO : STOP_HI + 1] = True
    out = {
        "row_l2": percentiles(row_l2),
        "row_maxabs": percentiles(row_max),
        "row_l2_special_248044_248076": percentiles(row_l2[STOP_LO : STOP_HI + 1]),
        "row_l2_observed_in_capture": percentiles(row_l2[observed]),
        "row_l2_unobserved": percentiles(row_l2[~observed]),
        "ratio_special_mean_l2_over_global_mean": float(
            np.mean(row_l2[STOP_LO : STOP_HI + 1]) / max(float(np.mean(row_l2)), 1e-20)
        ),
        "codecs": {},
    }
    for spec in CODECS:
        name = spec["name"]
        c = cos_store[name]
        n = nrms_store[name]
        out["codecs"][name] = {
            **{k: spec[k] for k in ("bits", "group", "kind")},
            "payload_bytes": payload_bytes(VOCAB, HIDDEN, spec["bits"] if spec["kind"] != "binary" else 1, spec["group"]),
            "payload_bpw": payload_bpw(VOCAB, HIDDEN, spec["bits"] if spec["kind"] != "binary" else 1, spec["group"]),
            "row_cosine": percentiles(c),
            "row_cosine_special": percentiles(c[special]),
            "row_cosine_observed": percentiles(c[observed]),
            "row_cosine_unobserved": percentiles(c[~observed]),
            "post_rmsnorm_cosine": percentiles(n),
            "post_rmsnorm_cosine_observed": percentiles(n[observed]),
            "post_rmsnorm_cosine_special": percentiles(n[special]),
            "frac_rows_cosine_lt_0.95": float(np.mean(c < 0.95)),
            "frac_rows_cosine_lt_0.90": float(np.mean(c < 0.90)),
            "frac_rows_nrms_lt_0.95": float(np.mean(n < 0.95)),
            "frac_rows_nrms_lt_0.90": float(np.mean(n < 0.90)),
            "corr_row_l2_vs_cosine": float(np.corrcoef(row_l2.astype(np.float64), c.astype(np.float64))[0, 1]),
        }
    # keep arrays we need later
    return out, row_l2, cos_store, nrms_store


def observed_embed_decision(
    ids: np.ndarray,
    E: np.ndarray,
    w0: np.ndarray,
    H0: np.ndarray,
    H63: np.ndarray,
    teacher: np.ndarray,
) -> dict:
    """Per-token embed quant → post-RMSNorm error → projected L63 error."""
    log("per-token embed quant + amplification projection")
    counts = Counter(int(x) for x in ids)
    amp = np.linalg.norm(H63.astype(np.float64), axis=1) / np.maximum(
        np.linalg.norm(H0.astype(np.float64), axis=1), 1e-20
    )
    out = {"codecs": {}, "n_unique": int(len(set(int(x) for x in ids))), "amp_used": "per_token_||H63||/||H0||"}
    for spec in CODECS:
        Eq = apply_codec(E, spec)
        Rn = rmsnorm(E, w0)
        Rq = rmsnorm(Eq, w0)
        d0 = Rq.astype(np.float64) - Rn.astype(np.float64)
        d0_n = np.linalg.norm(d0, axis=1)
        e0_n = np.linalg.norm(Rn.astype(np.float64), axis=1)
        dH = d0 * amp[:, None]
        dH_n = np.linalg.norm(dH, axis=1)
        rec = {
            "name": spec["name"],
            "gathered_row_cosine": percentiles(cosine_rows(E, Eq)),
            "post_rmsnorm_cosine": percentiles(cosine_rows(Rn, Rq)),
            "post_rmsnorm_rel_l2_mean": float(np.mean(d0_n / np.maximum(e0_n, 1e-20))),
            "dL0_l2": percentiles(d0_n),
            "dH63_proj_l2": percentiles(dH_n),
            "dH63_proj_over_H63": percentiles(dH_n / np.maximum(np.linalg.norm(H63.astype(np.float64), axis=1), 1e-20)),
        }
        # frequent vs rare in this capture
        freq = np.array([counts[int(i)] for i in ids], dtype=np.int32)
        rec["post_rmsnorm_cosine_count_ge_8"] = percentiles(cosine_rows(Rn, Rq)[freq >= 8])
        rec["post_rmsnorm_cosine_count_eq_1"] = percentiles(cosine_rows(Rn, Rq)[freq == 1])
        rec["post_rmsnorm_cosine_special"] = percentiles(
            cosine_rows(Rn, Rq)[(ids >= STOP_LO) & (ids <= STOP_HI)]
        )
        rec["post_rmsnorm_cosine_content"] = percentiles(cosine_rows(Rn, Rq)[ids < 248000])
        out["codecs"][spec["name"]] = rec
        # stash dH for later decision once we have W rows / margins
        rec["_dH"] = dH  # stripped before JSON
        rec["_d0"] = d0
    out["_amp"] = amp
    return out


def lm_head_logits_and_codecs(X: np.ndarray) -> dict:
    log(f"lm_head ref logits + codecs, X={X.shape} chunk={CHUNK}")
    mm, shape = open_bf16_memmap("language_model.lm_head.weight")
    assert shape == (VOCAB, HIDDEN)
    Y = np.empty((N_TOK, VOCAB), dtype=np.float32)
    # pass 1: reference
    for s in range(0, VOCAB, CHUNK):
        e = min(s + CHUNK, VOCAB)
        W = bf16u16_to_f32(np.asarray(mm[s:e]))
        Y[:, s:e] = X @ W.T
        if s % (CHUNK * 20) == 0:
            log(f"  ref logits rows {s}:{e}")
    # decisions
    top1 = np.argmax(Y, axis=1).astype(np.int64)
    # second: set top1 to -inf
    Y2 = Y.copy()
    Y2[np.arange(N_TOK), top1] = -np.inf
    top2 = np.argmax(Y2, axis=1).astype(np.int64)
    del Y2
    y1 = Y[np.arange(N_TOK), top1]
    y2 = Y[np.arange(N_TOK), top2]
    margin = y1 - y2
    # also top3 for fragility context
    Y3 = Y.copy()
    Y3[np.arange(N_TOK), top1] = -np.inf
    Y3[np.arange(N_TOK), top2] = -np.inf
    top3 = np.argmax(Y3, axis=1)
    y3 = Y[np.arange(N_TOK), top3]
    del Y3
    # row energy of competitive rows
    # streaming pass 2: codecs
    codec_out = {}
    # accumulators
    acc = {}
    for spec in CODECS:
        acc[spec["name"]] = {
            "dot": np.zeros(N_TOK, dtype=np.float64),
            "nYh": np.zeros(N_TOK, dtype=np.float64),
            "maxv": np.full(N_TOK, -np.inf, dtype=np.float64),
            "argi": np.zeros(N_TOK, dtype=np.int64),
            "y_at_top1": np.zeros(N_TOK, dtype=np.float64),
            "y_at_top2": np.zeros(N_TOK, dtype=np.float64),
            "max_abs_err": 0.0,
            "sum_sq_err": 0.0,
            "n_err": 0,
            "topk_hit": {k: np.zeros(N_TOK, dtype=bool) for k in (8, 32, 128)},
            # running top-k of draft (partial) — handled after? we need full Yq for true top-k.
            # store Yq would be 254MB * n_codecs — too much.
            # Instead keep a small heap per token: 128 (val, idx)
        }
        # per-token min-heap of 128: store as arrays
        acc[spec["name"]]["heap_v"] = np.full((N_TOK, 128), -np.inf, dtype=np.float32)
        acc[spec["name"]]["heap_i"] = np.full((N_TOK, 128), -1, dtype=np.int32)

    nY = np.sum(np.square(Y, dtype=np.float64), axis=1)

    def heap_offer(heap_v, heap_i, vals, start):
        # vals: [n_tok, chunk]
        # merge chunk into top-128 by concatenation + partition
        c = vals.shape[1]
        idx = np.arange(c, dtype=np.int32)[None, :] + np.int32(start)
        cat_v = np.concatenate([heap_v, vals.astype(np.float32)], axis=1)
        cat_i = np.concatenate([heap_i, np.broadcast_to(idx, vals.shape)], axis=1)
        # argpartition smallest of (128+c) keep last 128 (largest)
        k = 128
        part = np.argpartition(cat_v, -k, axis=1)[:, -k:]
        rows = np.arange(N_TOK)[:, None]
        heap_v[:, :] = cat_v[rows, part]
        heap_i[:, :] = cat_i[rows, part]

    for s in range(0, VOCAB, CHUNK):
        e = min(s + CHUNK, VOCAB)
        W = bf16u16_to_f32(np.asarray(mm[s:e]))
        for spec in CODECS:
            Wq = apply_codec(W, spec)
            Yq = (X @ Wq.T).astype(np.float64)  # [256, chunk]
            a = acc[spec["name"]]
            a["dot"] += np.sum(Y[:, s:e].astype(np.float64) * Yq, axis=1)
            a["nYh"] += np.sum(Yq * Yq, axis=1)
            mx = np.max(Yq, axis=1)
            mi = np.argmax(Yq, axis=1) + s
            better = mx > a["maxv"]
            a["maxv"][better] = mx[better]
            a["argi"][better] = mi[better]
            # values at ref top1/top2 if in this chunk
            m1 = (top1 >= s) & (top1 < e)
            if np.any(m1):
                a["y_at_top1"][m1] = Yq[np.flatnonzero(m1), (top1[m1] - s)]
            m2 = (top2 >= s) & (top2 < e)
            if np.any(m2):
                a["y_at_top2"][m2] = Yq[np.flatnonzero(m2), (top2[m2] - s)]
            err = Yq - Y[:, s:e].astype(np.float64)
            a["max_abs_err"] = max(a["max_abs_err"], float(np.max(np.abs(err))))
            a["sum_sq_err"] += float(np.sum(err * err))
            a["n_err"] += int(err.size)
            heap_offer(a["heap_v"], a["heap_i"], Yq, s)
        if s % (CHUNK * 10) == 0:
            log(f"  codec logits rows {s}:{e}")
    del mm

    # row energy of W for competitive set — reload those rows only
    log("gather competitive + teacher lm_head rows")
    mm, _ = open_bf16_memmap("language_model.lm_head.weight")
    need = set(int(x) for x in top1) | set(int(x) for x in top2) | set(int(x) for x in top3)
    W_need = {i: bf16u16_to_f32(np.asarray(mm[i])) for i in need}
    # also a random sample of 2048 rows for row-l2 background? stream cheap:
    row_l2_sample = []
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(VOCAB, size=4096, replace=False)
    for i in sample_idx:
        row_l2_sample.append(float(np.linalg.norm(bf16u16_to_f32(np.asarray(mm[int(i)])).astype(np.float64))))
    # special rows fidelity (lm_head)
    spec_l2 = []
    spec_cos_q4 = []
    for i in range(STOP_LO, STOP_HI + 1):
        w = bf16u16_to_f32(np.asarray(mm[i]))
        spec_l2.append(float(np.linalg.norm(w.astype(np.float64))))
        wq = apply_codec(w[None, :], CODECS[4])  # q4_g64
        spec_cos_q4.append(float(cosine_rows(w, wq[0])[0]))
    del mm

    gap = []
    for t in range(N_TOK):
        w1 = W_need[int(top1[t])].astype(np.float64)
        w2 = W_need[int(top2[t])].astype(np.float64)
        gap.append(w1 - w2)
    gap = np.stack(gap, axis=0)
    gap_n = np.linalg.norm(gap, axis=1)

    ref = {
        "site": "PROXY_X_is_argument_hidden_not_confirmed_final_norm",
        "X_rms": float(np.sqrt(np.mean(np.square(X.astype(np.float64))))),
        "top1": [int(x) for x in top1],
        "top2": [int(x) for x in top2],
        "y1": [float(x) for x in y1],
        "y2": [float(x) for x in y2],
        "y3": [float(x) for x in y3],
        "margin": [float(x) for x in margin],
        "margin_stats": percentiles(margin),
        "margin_over_y1": percentiles(margin / np.maximum(np.abs(y1), 1e-20)),
        "y1_stats": percentiles(y1),
        "y2_stats": percentiles(y2),
        "n_unique_top1": int(len(set(int(x) for x in top1))),
        "n_unique_top2": int(len(set(int(x) for x in top2))),
        "n_unique_competitive_top3": int(len(need)),
        "frac_top1_special": float(np.mean((top1 >= STOP_LO) & (top1 <= STOP_HI))),
        "frac_top1_im_end": float(np.mean(top1 == 248046)),
        "frac_top1_eot": float(np.mean(top1 == 248044)),
        "frac_top1_think": float(np.mean(top1 == 248068)),
        "gap_row_l2": percentiles(gap_n),
        "lm_head_row_l2_sample4096": percentiles(np.asarray(row_l2_sample)),
        "lm_head_special_row_l2": percentiles(np.asarray(spec_l2)),
        "lm_head_special_q4_row_cosine": percentiles(np.asarray(spec_cos_q4)),
        "fragile_frac": {
            "margin_lt_0.10": float(np.mean(margin < 0.10)),
            "margin_lt_0.25": float(np.mean(margin < 0.25)),
            "margin_lt_0.50": float(np.mean(margin < 0.50)),
            "margin_lt_1.00": float(np.mean(margin < 1.00)),
            "margin_lt_2.00": float(np.mean(margin < 2.00)),
            "margin_lt_4.00": float(np.mean(margin < 4.00)),
            "margin_lt_8.00": float(np.mean(margin < 8.00)),
        },
    }

    for spec in CODECS:
        a = acc[spec["name"]]
        den = np.sqrt(nY) * np.sqrt(a["nYh"])
        ok = den > 1e-12
        out_cos = np.zeros(N_TOK, dtype=np.float64)
        out_cos[ok] = a["dot"][ok] / den[ok]
        flips = a["argi"] != top1
        # pair error on the ref competitors
        d1 = a["y_at_top1"] - y1.astype(np.float64)
        d2 = a["y_at_top2"] - y2.astype(np.float64)
        pair_shift = d2 - d1  # flip top1/top2 if pair_shift > margin
        pair_would = pair_shift > margin
        # top-k coverage of ref argmax
        cover = {}
        for k in (8, 32, 128):
            # check whether top1 is among the k largest in the heap
            # heap is 128; take k largest
            hv = a["heap_v"]
            hi = a["heap_i"]
            part = np.argpartition(hv, -k, axis=1)[:, -k:]
            rows = np.arange(N_TOK)[:, None]
            topk_i = hi[rows, part]
            hit = np.any(topk_i == top1[:, None], axis=1)
            cover[f"k{k}"] = {
                "frac_ref_argmax_in_draft_topk": float(np.mean(hit)),
                "n_miss": int(np.sum(~hit)),
            }
        # other-jump vs pair-swap
        other_jump = flips & ~pair_would
        codec_out[spec["name"]] = {
            **{k: spec[k] for k in ("bits", "group", "kind")},
            "payload_bytes": payload_bytes(VOCAB, HIDDEN, spec["bits"] if spec["kind"] != "binary" else 1, spec["group"]),
            "payload_bpw": payload_bpw(VOCAB, HIDDEN, spec["bits"] if spec["kind"] != "binary" else 1, spec["group"]),
            "output_cosine_mean": float(np.mean(out_cos)),
            "output_cosine": percentiles(out_cos),
            "output_rel_l2": float(
                np.sqrt(max(float(nY.sum() + a["nYh"].sum() - 2.0 * a["dot"].sum()), 0.0))
                / max(float(np.sqrt(nY.sum())), 1e-12)
            ),
            "logit_rmse": float(np.sqrt(a["sum_sq_err"] / max(a["n_err"], 1))),
            "logit_max_abs_err": a["max_abs_err"],
            "n_argmax_flips": int(np.sum(flips)),
            "frac_argmax_flips": float(np.mean(flips)),
            "n_pair_swap_would": int(np.sum(pair_would)),
            "n_other_row_jump": int(np.sum(other_jump)),
            "d_logit_top1": percentiles(d1),
            "d_logit_top2": percentiles(d2),
            "pair_shift": percentiles(pair_shift),
            "pair_shift_abs": percentiles(np.abs(pair_shift)),
            "flips_by_margin_bin": {},
            "draft_topk_covers_ref_argmax": cover,
            "flipped_token_ids": [int(x) for x in np.flatnonzero(flips)],
            "flipped_to": [int(a["argi"][i]) for i in np.flatnonzero(flips)],
            "flipped_from": [int(top1[i]) for i in np.flatnonzero(flips)],
            "flipped_margins": [float(margin[i]) for i in np.flatnonzero(flips)],
        }
        bins = [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 1e9)]
        for lo, hi in bins:
            m = (margin >= lo) & (margin < hi)
            key = f"[{lo},{hi})"
            codec_out[spec["name"]]["flips_by_margin_bin"][key] = {
                "n": int(np.sum(m)),
                "n_flip": int(np.sum(flips[m])),
                "frac_flip": float(np.mean(flips[m])) if np.any(m) else None,
            }
        # stash for embed decision
        acc[spec["name"]]["_flips"] = flips
        acc[spec["name"]]["_d1"] = d1
        acc[spec["name"]]["_d2"] = d2

    return {
        "ref": ref,
        "codecs": codec_out,
        "_Y": Y,
        "_top1": top1,
        "_top2": top2,
        "_margin": margin,
        "_y1": y1,
        "_y2": y2,
        "_gap": gap,
        "_gap_n": gap_n,
        "_W_need": W_need,
        "_acc": acc,
    }


def teacher_stats(Y: np.ndarray, teacher: np.ndarray, top1: np.ndarray, margin: np.ndarray) -> dict:
    valid = teacher >= 0
    tv = teacher[valid]
    Yv = Y[valid]
    pred = top1[valid]
    y_true = Yv[np.arange(tv.size), tv]
    # max among others
    tmp = Yv.copy()
    tmp[np.arange(tv.size), tv] = -np.inf
    y_best_other = np.max(tmp, axis=1)
    tmargin = y_true - y_best_other
    return {
        "n_teacher_positions": int(np.sum(valid)),
        "teacher_top1_match": float(np.mean(pred == tv)),
        "n_teacher_match": int(np.sum(pred == tv)),
        "teacher_margin": percentiles(tmargin),
        "teacher_margin_when_match": percentiles(tmargin[pred == tv]),
        "teacher_margin_when_miss": percentiles(tmargin[pred != tv]) if np.any(pred != tv) else {"n": 0},
        "frac_teacher_margin_lt_0.5": float(np.mean(tmargin < 0.5)),
        "frac_teacher_margin_lt_1.0": float(np.mean(tmargin < 1.0)),
        "frac_teacher_margin_negative": float(np.mean(tmargin < 0.0)),
        "ref_margin_on_teacher_pos": percentiles(margin[valid]),
    }


def embed_vs_margin(embed_dec: dict, lm: dict) -> dict:
    log("embed projected error vs logit margin")
    gap = lm["_gap"]
    gap_n = lm["_gap_n"]
    margin = lm["_margin"]
    out = {}
    for spec in CODECS:
        dH = embed_dec["codecs"][spec["name"]]["_dH"]
        aligned = np.abs(np.sum(gap * dH, axis=1))
        cauchy = gap_n * np.linalg.norm(dH, axis=1)
        out[spec["name"]] = {
            "aligned_pair_shift_proj": percentiles(aligned),
            "cauchy_pair_shift_bound": percentiles(cauchy),
            "frac_aligned_lt_margin": float(np.mean(aligned < margin)),
            "frac_cauchy_lt_margin": float(np.mean(cauchy < margin)),
            "n_aligned_could_flip": int(np.sum(aligned >= margin)),
            "n_cauchy_could_flip": int(np.sum(cauchy >= margin)),
            "aligned_over_margin": percentiles(aligned / np.maximum(margin, 1e-12)),
            "cauchy_over_margin": percentiles(cauchy / np.maximum(margin, 1e-12)),
            "label": "PROJECTED. dH = (||H63||/||H0||) * (rmsnorm(Eq)-rmsnorm(E)). Assumes perturbation tracks the measured signal-norm gain, not a measured Jacobian.",
        }
    return out


def complete_bpw_table() -> dict:
    """Exact arithmetic on MEASURED geometry."""
    # rest includes attention+mlp+small
    rest = N_LANG - 2 * E_TAB
    rows = []
    plans = [
        ("g0_both_q4", 4.25, 4.25),
        ("embed_q2g128_lm_q4g64", 2.125, 4.25),
        ("embed_q3g128_lm_q4g64", 3.125, 4.25),
        ("embed_q2g128_lm_q3g128", 2.125, 3.125),
        ("embed_q3g128_lm_q3g128", 3.125, 3.125),
        ("embed_binary_lm_q4", 1.125, 4.25),
        ("embed_q2g128_lm_q8g64", 2.125, 8.25),
        ("embed_q4_lm_q8", 4.25, 8.25),
        ("both_q3g128", 3.125, 3.125),
        ("both_q2g128", 2.125, 2.125),
    ]
    # more precise payload bpw
    def bpw(bits, group, kind):
        b = 1 if kind == "binary" else bits
        return payload_bpw(VOCAB, HIDDEN, b, group)

    named = {
        "g0_both_q4": (bpw(4, 64, "q4"), bpw(4, 64, "q4")),
        "embed_q2g128_lm_q4g64": (bpw(2, 128, "qn"), bpw(4, 64, "q4")),
        "embed_q3g128_lm_q4g64": (bpw(3, 128, "qn"), bpw(4, 64, "q4")),
        "embed_q2g128_lm_q3g128": (bpw(2, 128, "qn"), bpw(3, 128, "qn")),
        "embed_q3g128_lm_q3g128": (bpw(3, 128, "qn"), bpw(3, 128, "qn")),
        "embed_binary_lm_q4": (bpw(1, 128, "binary"), bpw(4, 64, "q4")),
        "embed_q2g128_lm_q8g64": (bpw(2, 128, "qn"), bpw(8, 64, "qn")),
        "embed_q4_lm_q8": (bpw(4, 64, "q4"), bpw(8, 64, "qn")),
        "both_q3g128": (bpw(3, 128, "qn"), bpw(3, 128, "qn")),
        "both_q2g128": (bpw(2, 128, "qn"), bpw(2, 128, "qn")),
    }
    for name, (be, bh) in named.items():
        # complete if rest stays at r
        def complete(r):
            return (rest * r + E_TAB * be + E_TAB * bh) / N_LANG

        r_for_1p5 = (1.5 * N_LANG - E_TAB * be - E_TAB * bh) / rest
        rows.append(
            {
                "plan": name,
                "embed_payload_bpw": be,
                "lm_head_payload_bpw": bh,
                "tables_bits": E_TAB * be + E_TAB * bh,
                "tables_complete_bpw_contrib": (E_TAB * be + E_TAB * bh) / N_LANG,
                "rest_bpw_for_complete_1p5": r_for_1p5,
                "complete_if_rest_g0_4.252735": complete(4.252735126866492),
                "complete_if_rest_1p0": complete(1.0),
                "complete_if_rest_1p2": complete(1.2),
                "delta_complete_vs_g0_tables_only": complete(4.252735126866492)
                - (rest * 4.252735126866492 + 2 * E_TAB * bpw(4, 64, "q4")) / N_LANG,
            }
        )
    g0_be = bpw(4, 64, "q4")
    return {
        "N": N_LANG,
        "E_tab": E_TAB,
        "rest": rest,
        "g0_table_payload_bpw": g0_be,
        "g0_tables_complete_contrib": 2 * E_TAB * g0_be / N_LANG,
        "rest_bpw_if_tables_stay_g0_for_complete_1p5": (1.5 * N_LANG - 2 * E_TAB * g0_be) / rest,
        "plans": rows,
        "label": "DERIVED. payload_bytes = rows * ceil(cols/g) * (ceil(g*bits/8)+2). Headers ignored.",
    }


def strip_private(obj):
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    return obj


def main() -> None:
    log("start")
    cap = load_capture()
    ids = cap["ids"]
    log(f"tokens n=256 unique={len(set(int(x) for x in ids))} last_flags={int(np.sum(cap['is_last']))}")

    log("load hidden cube 64x256x5120")
    Hs = [load_hidden(l) for l in range(N_LAYERS)]
    H = np.stack(Hs, axis=0)
    del Hs
    # overwrite capture rms into amp later
    amp = measure_amplification(H)
    # attach capture-result rms
    for l, rec in enumerate(amp["per_layer"]):
        rec["capture_rms"] = cap["cap"]["per_layer"][str(l)]["rms"]

    H0, H63 = H[0], H[63]
    site, E, w0, wfn = site_identify(ids, H0, H63)
    log(f"site {site['site_H0']} rmsnorm_rel_l2={site['rmsnorm_embed_w0_vs_H0_rel_l2']:.3e}")

    census, _row_l2, _cos, _nrms = embed_row_census(ids)
    embed_dec = observed_embed_decision(ids, E, w0, H0, H63, cap["teacher"])

    # two X sites for lm_head
    results_x = {}
    for xname, X in (
        ("L63_raw", np.ascontiguousarray(H63)),
        ("rmsnorm_L63_final_w", np.ascontiguousarray(rmsnorm(H63, wfn))),
    ):
        log(f"==== lm_head site {xname} ====")
        lm = lm_head_logits_and_codecs(X)
        tstat = teacher_stats(lm["_Y"], cap["teacher"], lm["_top1"], lm["_margin"])
        evsm = embed_vs_margin(embed_dec, lm)
        # last-of-prompt margins
        last = cap["is_last"]
        last_m = {
            "n": int(np.sum(last)),
            "margin": [float(lm["_margin"][i]) for i in np.flatnonzero(last)],
            "top1": [int(lm["_top1"][i]) for i in np.flatnonzero(last)],
            "top2": [int(lm["_top2"][i]) for i in np.flatnonzero(last)],
            "prompt": [cap["spans"][int(cap["prompt_ix"][i])][2] for i in np.flatnonzero(last)],
        }
        # per-class flip later from codecs using token class of top1
        results_x[xname] = {
            "ref": lm["ref"],
            "codecs": lm["codecs"],
            "teacher": tstat,
            "embed_projected_vs_margin": evsm,
            "last_of_prompt": last_m,
        }
        # keep primary internals
        if xname == "L63_raw":
            primary_lm = lm

    bpw = complete_bpw_table()

    # token anatomy of capture
    counts = Counter(int(x) for x in ids)
    tok_tab = []
    # decode via vocab.json if possible
    vocab = json.loads((SRC / "vocab.json").read_text())
    # Qwen vocab.json is token->id
    id2tok = {int(v): k for k, v in vocab.items()}
    # added tokens override
    tokjson = json.loads((SRC / "tokenizer.json").read_text())
    for t in tokjson.get("added_tokens") or []:
        id2tok[int(t["id"])] = t["content"]
    for tid, c in counts.most_common():
        tok_tab.append(
            {
                "id": int(tid),
                "count": int(c),
                "class": token_class(int(tid)),
                "text": id2tok.get(int(tid), "?")[:40],
            }
        )

    out = {
        "schema": "hawking.g1.qwen38_endpoints_sensitivity.v1",
        "label_policy": "MEASURED = this process. DERIVED = exact arithmetic. PROJECTED = chain model named in-place. CITED = prior receipt, not re-run.",
        "inputs": {
            "bf16": str(SRC),
            "capture": str(CAP),
            "capture_sha256_self_cited": cap["cap"]["sha256_self"],
            "capture_result_sha256_measured": sha256_file(CAP / "capture-result.json"),
            "L00_sha256_measured": sha256_file(CAP / "hidden" / "L00.f32"),
            "L63_sha256_measured": sha256_file(CAP / "hidden" / "L63.f32"),
            "lm_head_shard": _WMAP["language_model.lm_head.weight"],
            "embed_shard": _WMAP["language_model.model.embed_tokens.weight"],
            "n_tokens": 256,
            "n_unique_tokens": int(len(set(int(x) for x in ids))),
            "n_prompts": 5,
            "prompts": [
                {"prompt": s, "span": [a, b], "n": b - a} for a, b, s in cap["spans"]
            ],
        },
        "geometry": {
            "hidden": HIDDEN,
            "vocab": VOCAB,
            "layers": N_LAYERS,
            "N_language": N_LANG,
            "embed_params": E_TAB,
            "lm_head_params": E_TAB,
            "rms_eps": EPS,
        },
        "site": site,
        "amplification": {
            **{k: v for k, v in amp.items() if k != "per_layer"},
            "per_layer_head": amp["per_layer"][:4],
            "per_layer_tail": amp["per_layer"][-4:],
            "per_layer_all": amp["per_layer"],
        },
        "embed_row_census": census,
        "embed_observed": strip_private(embed_dec),
        "lm_head": results_x,
        "capture_token_table": tok_tab,
        "complete_bpw": bpw,
        "rss_max_gb": rss_gb(),
        "wall_s": time.perf_counter() - T0,
    }
    # drop per_layer_all from written? keep it, it's small (64 rows)
    OUT.write_text(json.dumps(out, indent=2))
    log(f"wrote {OUT} bytes={OUT.stat().st_size} rss_max={rss_gb():.3f}G")

    # compact stdout for the report
    print("=== SITE ===")
    print(json.dumps({k: site[k] for k in site if "cosine" in k or k.startswith("site") or "rel_l2" in k}, indent=2))
    print("=== AMP ===")
    print(
        json.dumps(
            {
                "mean_of_per_token_L63_over_L0": amp["mean_of_per_token_L63_over_L0"],
                "mean_l2_L63_over_L0": amp["mean_l2_L63_over_L0"],
                "mean_of_per_token_L63_over_L62": amp["mean_of_per_token_L63_over_L62"],
                "cosine_L0_L63": amp["cosine_L0_L63"],
            },
            indent=2,
        )
    )
    print("=== MARGINS L63 ===")
    print(json.dumps(results_x["L63_raw"]["ref"]["margin_stats"], indent=2))
    print("fragile", results_x["L63_raw"]["ref"]["fragile_frac"])
    print("teacher", results_x["L63_raw"]["teacher"])
    print("=== CODECS L63 flips ===")
    for name, rec in results_x["L63_raw"]["codecs"].items():
        print(
            name,
            "flips",
            rec["n_argmax_flips"],
            rec["frac_argmax_flips"],
            "cos",
            rec["output_cosine_mean"],
            "rmse",
            rec["logit_rmse"],
            "pair_abs_p50",
            rec["pair_shift_abs"]["p50"],
            "k32",
            rec["draft_topk_covers_ref_argmax"]["k32"],
        )
    print("=== EMBED vs MARGIN L63 ===")
    for name, rec in results_x["L63_raw"]["embed_projected_vs_margin"].items():
        print(name, "aligned_safe", rec["frac_aligned_lt_margin"], "cauchy_safe", rec["frac_cauchy_lt_margin"])
    print("=== RSS", rss_gb(), "WALL", time.perf_counter() - T0)


if __name__ == "__main__":
    main()
