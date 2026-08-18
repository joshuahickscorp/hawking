#!/usr/bin/env python3
"""G1 conditional-representation measurements. CPU/numpy only. No GPU."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import struct
import time
from collections import defaultdict

import numpy as np

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

CAP = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
BF16 = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
OUT = "/tmp/g1_conditional_repr.json"

N_TOK = 256
HIDDEN = 5120
INTER = 17408
N_LAYERS = 64
G64 = 64
G512 = 512
N = 26_895_998_464  # source params
SRC_PARAMS = N

MLP_LAYERS = [0, 3, 6, 7, 15, 32, 47, 63]
# fewer k-points; enough to draw the curve
K_RES = [1, 3, 8, 32, 128, 256, 512, 1024, 2560]
K_INT = [32, 128, 256, 512, 1024, 2048, 4096, 8704]
K_JACCARD = 32
K_INT_JACCARD = 128


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    den = na * nb
    dot = np.sum(a * b, axis=1)
    out = np.zeros(a.shape[0], dtype=np.float64)
    m = den > 0
    out[m] = (dot[m] / den[m]).astype(np.float64)
    return out


def gini(x: np.ndarray) -> float:
    """Gini of nonnegative vector. 0 = equal, 1 = one-hot."""
    v = np.asarray(x, dtype=np.float64).ravel()
    v = np.sort(v)
    if v[-1] <= 0:
        return 0.0
    n = v.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * v) / (n * np.sum(v))) - (n + 1.0) / n)


def participation_ratio(e: np.ndarray) -> float:
    """PR = (sum e)^2 / sum e^2  on nonnegative energies. 1 = one-hot, n = flat."""
    s = float(np.sum(e))
    q = float(np.sum(e * e))
    if q <= 0:
        return 0.0
    return (s * s) / q


def energy_frac_topk(e: np.ndarray, ks) -> dict:
    tot = float(np.sum(e))
    if tot <= 0:
        return {int(k): 0.0 for k in ks}
    order = np.argsort(e)[::-1]
    c = np.cumsum(e[order])
    n = e.size
    out = {}
    for k in ks:
        kk = min(int(k), n)
        out[int(k)] = float(c[kk - 1] / tot) if kk > 0 else 0.0
    return out


def k_for_frac(e: np.ndarray, fracs) -> dict:
    tot = float(np.sum(e))
    if tot <= 0:
        return {str(f): 0 for f in fracs}
    c = np.cumsum(np.sort(e)[::-1]) / tot
    out = {}
    for f in fracs:
        out[str(f)] = int(np.searchsorted(c, f) + 1)
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


class SafeTensors:
    def __init__(self, model_dir: str):
        index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        self.base = model_dir
        self.weight_map = index["weight_map"]
        self._hdr = {}

    def _header(self, shard: str):
        if shard not in self._hdr:
            path = os.path.join(self.base, shard)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            self._hdr[shard] = (path, 8 + n, hdr)
        return self._hdr[shard]

    def info(self, name: str):
        shard = self.weight_map[name]
        path, data_off, hdr = self._header(shard)
        rec = hdr[name]
        return path, data_off, rec

    def load_f32(self, name: str) -> np.ndarray:
        path, data_off, rec = self.info(name)
        shape = tuple(rec["shape"])
        dtype = rec["dtype"]
        off0, off1 = rec["data_offsets"]
        nbytes = off1 - off0
        mm = np.memmap(path, dtype=np.uint8, mode="r", offset=data_off + off0, shape=(nbytes,))
        if dtype == "BF16":
            u16 = np.frombuffer(mm, dtype="<u2").reshape(shape).copy()
            out = np.empty(shape, np.float32)
            out.view(np.uint32)[...] = u16.astype(np.uint32) << 16
            del u16
            return out
        if dtype == "F32":
            return np.frombuffer(mm, dtype="<f4").reshape(shape).copy()
        if dtype == "F16":
            return np.frombuffer(mm, dtype="<f2").reshape(shape).astype(np.float32)
        raise ValueError(f"{name} dtype={dtype}")


def load_hidden() -> np.ndarray:
    X = np.empty((N_LAYERS, N_TOK, HIDDEN), dtype=np.float32)
    for L in range(N_LAYERS):
        p = os.path.join(CAP, "hidden", f"L{L:02d}.f32")
        X[L] = np.fromfile(p, dtype=np.float32).reshape(N_TOK, HIDDEN)
    return X


def prompt_slices(receipt: dict):
    slices = []
    t0 = 0
    ids = []
    for i, pr in enumerate(receipt["prompts"]):
        n = int(pr["n_tokens"])
        slices.append((i, t0, t0 + n, pr["prompt"], list(pr["ids"])))
        ids.extend(pr["ids"])
        t0 += n
    assert t0 == N_TOK
    assert len(ids) == N_TOK
    return slices, np.array(ids, dtype=np.int64)


def common_prefix_len(slices) -> int:
    seqs = [s[4] for s in slices]
    m = min(len(s) for s in seqs)
    k = 0
    while k < m and all(s[k] == seqs[0][k] for s in seqs):
        k += 1
    return k


def residual_layer_stats(X: np.ndarray, L: int) -> dict:
    x = X[L]  # (256, 5120)
    e_tok = x.astype(np.float64) ** 2  # energy per tok, ch
    e_mean = e_tok.mean(axis=0)
    e_all = e_tok.sum(axis=0)
    rms = np.sqrt(e_tok.mean(axis=0))
    med = float(np.median(rms))
    top = np.argsort(rms)[::-1]
    # per-token k for energy
    fracs = (0.5, 0.9, 0.95, 0.99, 0.999)
    k_needed = [k_for_frac(e_tok[t], fracs) for t in range(N_TOK)]
    k_needed_mean = {str(f): float(np.mean([d[str(f)] for d in k_needed])) for f in fracs}
    k_needed_max = {str(f): int(np.max([d[str(f)] for d in k_needed])) for f in fracs}
    k_needed_p90 = {str(f): float(np.quantile([d[str(f)] for d in k_needed], 0.9)) for f in fracs}

    # group energies
    n64 = HIDDEN // G64
    n512 = HIDDEN // G512
    g64 = e_tok.reshape(N_TOK, n64, G64).sum(axis=2)
    g512 = e_tok.reshape(N_TOK, n512, G512).sum(axis=2)
    # fraction of groups needed for 0.99 energy, per token
    k64_99 = [k_for_frac(g64[t], (0.99,))["0.99"] for t in range(N_TOK)]
    k512_99 = [k_for_frac(g512[t], (0.99,))["0.99"] for t in range(N_TOK)]

    # duty cycle at several k
    duty = {}
    unions = {}
    for k in (8, 32, 128, 256, 512, 1024):
        hot = np.zeros(HIDDEN, dtype=np.int32)
        union = set()
        union_growth = []
        for t in range(N_TOK):
            idx = np.argpartition(e_tok[t], -k)[-k:]
            hot[idx] += 1
            union.update(int(i) for i in idx)
            union_growth.append(len(union))
        duty[str(k)] = {
            "n_ever": int((hot > 0).sum()),
            "n_always": int((hot == N_TOK).sum()),
            "n_half": int((hot >= N_TOK // 2).sum()),
            "n_once": int((hot == 1).sum()),
            "mean_duty": float(hot.mean() / N_TOK),
            "max_duty": float(hot.max() / N_TOK),
            "union": int(len(union)),
            "union_growth_at": {
                "16": union_growth[15],
                "64": union_growth[63],
                "128": union_growth[127],
                "256": union_growth[255],
            },
        }
        unions[str(k)] = union  # returned separately, not json

    # participation / gini on mean energy
    pr = participation_ratio(e_mean)
    gn = gini(e_mean)
    top_energy = energy_frac_topk(e_mean, (1, 3, 8, 32, 128, 256, 512, 1024))

    # channel 3994
    ch = 3994
    xmed = float(rms[ch] / med) if med > 0 else 0.0
    efrac = float(e_mean[ch] / e_mean.sum()) if e_mean.sum() > 0 else 0.0
    rank = int(np.where(top == ch)[0][0]) + 1
    n_nz = int(np.count_nonzero(x[:, ch]))

    return {
        "json": {
            "mean_abs": float(np.mean(np.abs(x))),
            "rms": float(np.sqrt(np.mean(x.astype(np.float64) ** 2))),
            "pr_mean_energy": float(pr),
            "pr_over_n": float(pr / HIDDEN),
            "gini_mean_energy": float(gn),
            "top_energy_frac": {str(k): v for k, v in top_energy.items()},
            "k_for_energy_mean": k_needed_mean,
            "k_for_energy_max": k_needed_max,
            "k_for_energy_p90": k_needed_p90,
            "g64_k99_mean": float(np.mean(k64_99)),
            "g64_k99_max": int(np.max(k64_99)),
            "g64_n": n64,
            "g512_k99_mean": float(np.mean(k512_99)),
            "g512_k99_max": int(np.max(k512_99)),
            "g512_n": n512,
            "duty": {k: {kk: vv for kk, vv in d.items()} for k, d in duty.items()},
            "ch3994": {
                "rank": rank,
                "xmed": xmed,
                "energy_frac": efrac,
                "n_nonzero": n_nz,
                "rms": float(rms[ch]),
            },
            "top8_idx": [int(i) for i in top[:8]],
            "top8_rms": [float(rms[i]) for i in top[:8]],
        },
        "unions": unions,
        "e_tok": e_tok,
        "rms": rms,
    }


def pairwise_mean_jaccard(sets: list[set]) -> float:
    if len(sets) < 2:
        return float("nan")
    acc = 0.0
    n = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            acc += jaccard(sets[i], sets[j])
            n += 1
    return acc / n if n else float("nan")


def token_predictability(X: np.ndarray, ids: np.ndarray, slices, prefix_len: int) -> dict:
    """Same token-id vs random pair Jaccard of per-token top-k residual channels."""
    # Use a few layers: early / island / mid / late
    layers = [0, 6, 7, 32, 63]
    k = K_JACCARD
    out = {"k": k, "layers": {}}
    # token occurrences
    occ = defaultdict(list)
    for t, tid in enumerate(ids.tolist()):
        occ[int(tid)].append(t)

    rng = np.random.default_rng(3994)
    for L in layers:
        e = X[L].astype(np.float64) ** 2
        top_sets = []
        for t in range(N_TOK):
            idx = np.argpartition(e[t], -k)[-k:]
            top_sets.append(set(int(i) for i in idx))

        same_id_js = []
        n_ids_used = 0
        for tid, ts in occ.items():
            if len(ts) < 2:
                continue
            # skip pairs that are the same prefix position across prompts?
            js = pairwise_mean_jaccard([top_sets[t] for t in ts])
            same_id_js.append(js)
            n_ids_used += 1
        # context-controlled: same token id but NOT the shared-prefix aligned copies
        # build (prompt, offset) for each t
        tok_meta = []
        for pi, a, b, _p, _ids in slices:
            for off, t in enumerate(range(a, b)):
                tok_meta.append((pi, off, t))
        # map t -> (pi, off)
        t2po = {t: (pi, off) for pi, off, t in tok_meta}

        same_id_diffctx = []
        for tid, ts in occ.items():
            if len(ts) < 2:
                continue
            # pairs with different prefix-offset or offset >= prefix_len
            groups = defaultdict(list)
            others = []
            for t in ts:
                pi, off = t2po[t]
                if off < prefix_len:
                    groups[off].append(t)
                else:
                    others.append(t)
            # different-context occurrences: all outside prefix, plus at most one from each prefix offset
            pool = list(others)
            for off, ts_off in groups.items():
                pool.append(ts_off[0])  # one representative of this prefix slot
            if len(pool) >= 2:
                same_id_diffctx.append(pairwise_mean_jaccard([top_sets[t] for t in pool]))
            elif len(others) >= 2:
                same_id_diffctx.append(pairwise_mean_jaccard([top_sets[t] for t in others]))

        # random pairs
        rand_js = []
        for _ in range(400):
            i, j = rng.integers(0, N_TOK, size=2)
            if i == j:
                continue
            if ids[i] == ids[j]:
                continue
            rand_js.append(jaccard(top_sets[i], top_sets[j]))

        # shared prefix: same offset across prompts (should be near 1)
        prefix_js = []
        n_prompts_full = sum(1 for s in slices if (s[2] - s[1]) >= prefix_len)
        for off in range(prefix_len):
            sets_off = []
            for pi, a, b, _p, _ids in slices:
                if b - a > off:
                    sets_off.append(top_sets[a + off])
            if len(sets_off) >= 2:
                prefix_js.append(pairwise_mean_jaccard(sets_off))

        # consecutive-token Jaccard (context smoothness)
        consec = [jaccard(top_sets[t], top_sets[t + 1]) for t in range(N_TOK - 1)]
        # break at prompt boundaries
        bounds = {s[2] - 1 for s in slices}  # last index of each prompt
        consec_in = [jaccard(top_sets[t], top_sets[t + 1]) for t in range(N_TOK - 1) if t not in bounds]

        out["layers"][str(L)] = {
            "same_token_id_mean_jaccard": float(np.nanmean(same_id_js)) if same_id_js else None,
            "n_token_ids_with_repeats": n_ids_used,
            "same_id_diff_context_mean_jaccard": float(np.nanmean(same_id_diffctx)) if same_id_diffctx else None,
            "n_same_id_diff_context": len(same_id_diffctx),
            "random_diff_id_mean_jaccard": float(np.mean(rand_js)) if rand_js else None,
            "shared_prefix_same_offset_mean_jaccard": float(np.mean(prefix_js)) if prefix_js else None,
            "consecutive_in_prompt_mean_jaccard": float(np.mean(consec_in)) if consec_in else None,
            "n_prompts_covering_prefix": n_prompts_full,
        }
    return out


def union_saturation(X: np.ndarray) -> dict:
    """Does the union of per-token top-k saturate? Multiple shuffles."""
    layers = [0, 6, 32, 63]
    ks = [32, 128, 256, 512]
    rng = np.random.default_rng(7)
    out = {}
    for L in layers:
        e = X[L].astype(np.float64) ** 2
        out[str(L)] = {}
        for k in ks:
            tops = [np.argpartition(e[t], -k)[-k:] for t in range(N_TOK)]
            curves = []
            for _ in range(8):
                order = rng.permutation(N_TOK)
                u = set()
                curve = []
                for t in order:
                    u.update(int(i) for i in tops[t])
                    curve.append(len(u))
                curves.append(curve)
            arr = np.array(curves, dtype=np.float64)
            out[str(L)][str(k)] = {
                "mean_at_16": float(arr[:, 15].mean()),
                "mean_at_64": float(arr[:, 63].mean()),
                "mean_at_128": float(arr[:, 127].mean()),
                "mean_at_256": float(arr[:, 255].mean()),
                "std_at_256": float(arr[:, 255].std()),
                "delta_128_to_256": float(arr[:, 255].mean() - arr[:, 127].mean()),
                "linear_fill_rate_last128": float((arr[:, 255] - arr[:, 127]).mean() / 128.0),
            }
    return out


def site_confirm(st: SafeTensors, X: np.ndarray) -> dict:
    names = {
        "l7_post": "language_model.model.layers.7.post_attention_layernorm.weight",
        "l7_input": "language_model.model.layers.7.input_layernorm.weight",
        "l0_post": "language_model.model.layers.0.post_attention_layernorm.weight",
        "l0_input": "language_model.model.layers.0.input_layernorm.weight",
        "final": "language_model.model.norm.weight",
    }
    w = {k: st.load_f32(n) for k, n in names.items()}
    ch = 3994
    l7 = X[7]
    return {
        "l7_col3994_n_nonzero": int(np.count_nonzero(l7[:, ch])),
        "l7_col3994_max_abs": float(np.max(np.abs(l7[:, ch]))),
        "l7_post_w3994": float(w["l7_post"][ch]),
        "l7_input_w3994": float(w["l7_input"][ch]),
        "l7_post_n_zero": int(np.sum(w["l7_post"] == 0)),
        "l7_input_n_zero": int(np.sum(w["l7_input"] == 0)),
        "n_layers_post_w3994_zero": None,  # filled below
        "l0_post_w3994": float(w["l0_post"][ch]),
        "l0_input_w3994": float(w["l0_input"][ch]),
        "final_w3994": float(w["final"][ch]),
        "final_w3994_rank": int(np.argsort(w["final"])[0]) + 1 if False else int(1 + np.sum(w["final"] < w["final"][ch])),
        "l0_mean_abs_remeasure": float(np.mean(np.abs(X[0]))),
        "l0_rms_remeasure": float(np.sqrt(np.mean(X[0].astype(np.float64) ** 2))),
        "receipt_l0_mean_abs": 0.06772072613239288,
        "receipt_l0_rms": 0.09979002177715302,
    }


def count_post_zero_3994(st: SafeTensors) -> dict:
    ch = 3994
    post_z = []
    inp_z = []
    post_vals = []
    for L in range(N_LAYERS):
        pw = st.load_f32(f"language_model.model.layers.{L}.post_attention_layernorm.weight")
        iw = st.load_f32(f"language_model.model.layers.{L}.input_layernorm.weight")
        post_vals.append(float(pw[ch]))
        if pw[ch] == 0.0:
            post_z.append(L)
        if iw[ch] == 0.0:
            inp_z.append(L)
    return {
        "post_w3994_zero_layers": post_z,
        "input_w3994_zero_layers": inp_z,
        "post_w3994_by_layer": post_vals,
    }


def mlp_one_layer(st: SafeTensors, X: np.ndarray, L: int, ids: np.ndarray, slices, prefix_len: int) -> dict:
    t0 = time.perf_counter()
    x = X[L]  # (256, 5120) MLP input if site is post-attn-norm
    Wg = st.load_f32(f"language_model.model.layers.{L}.mlp.gate_proj.weight")  # (17408, 5120)
    Wu = st.load_f32(f"language_model.model.layers.{L}.mlp.up_proj.weight")
    assert Wg.shape == (INTER, HIDDEN)
    G = x @ Wg.T
    U = x @ Wu.T
    del Wg, Wu
    H = silu(G) * U
    del G, U
    Wd = st.load_f32(f"language_model.model.layers.{L}.mlp.down_proj.weight")  # (5120, 17408)
    assert Wd.shape == (HIDDEN, INTER)
    Y = H @ Wd.T  # (256, 5120)

    # intermediate energy / sparsity
    eH = H.astype(np.float64) ** 2
    eH_tok = eH  # (256, 17408)
    fracs = (0.5, 0.9, 0.95, 0.99, 0.999)
    k_needed = [k_for_frac(eH_tok[t], fracs) for t in range(N_TOK)]
    absH = np.abs(H)
    near = {
        "lt_1e-3_frac": float(np.mean(absH < 1e-3)),
        "lt_1e-2_frac": float(np.mean(absH < 1e-2)),
        "lt_1e-1_frac": float(np.mean(absH < 1e-1)),
    }

    n64 = INTER // G64
    n512 = INTER // G512
    g64 = eH_tok.reshape(N_TOK, n64, G64).sum(axis=2)
    g512 = eH_tok.reshape(N_TOK, n512, G512).sum(axis=2)
    k64_99 = [k_for_frac(g64[t], (0.99,))["0.99"] for t in range(N_TOK)]
    k512_99 = [k_for_frac(g512[t], (0.99,))["0.99"] for t in range(N_TOK)]

    # duty / union on intermediates
    duty = {}
    for k in (128, 256, 512, 1024, 2048):
        hot = np.zeros(INTER, dtype=np.int32)
        union = set()
        growth = []
        for t in range(N_TOK):
            idx = np.argpartition(eH_tok[t], -k)[-k:]
            hot[idx] += 1
            union.update(int(i) for i in idx)
            growth.append(len(union))
        duty[str(k)] = {
            "n_ever": int((hot > 0).sum()),
            "n_always": int((hot == N_TOK).sum()),
            "n_half": int((hot >= N_TOK // 2).sum()),
            "union": int(len(union)),
            "union_over_inter": float(len(union) / INTER),
            "union_growth_at": {"16": growth[15], "64": growth[63], "128": growth[127], "256": growth[255]},
        }

    # functional: per-token top-k intermediate -> Y cosine
    # incremental per token would be slow in python; do selected k via masked H
    y_norm = np.linalg.norm(Y, axis=1)
    func_dyn = {}
    for k in K_INT:
        Hm = np.zeros_like(H)
        for t in range(N_TOK):
            idx = np.argpartition(eH_tok[t], -k)[-k:]
            Hm[t, idx] = H[t, idx]
        Yk = Hm @ Wd.T
        c = cosine_rows(Yk, Y)
        func_dyn[str(k)] = {
            "mean_cos": float(c.mean()),
            "min_cos": float(c.min()),
            "p10_cos": float(np.quantile(c, 0.1)),
            "mean_rel_l2": float(np.mean(np.linalg.norm(Yk - Y, axis=1) / np.maximum(y_norm, 1e-12))),
        }

    # static top-k by mean energy vs dynamic
    e_mean = eH_tok.mean(axis=0)
    func_static = {}
    for k in K_INT:
        idx = np.argpartition(e_mean, -k)[-k:]
        Hm = np.zeros_like(H)
        Hm[:, idx] = H[:, idx]
        Yk = Hm @ Wd.T
        c = cosine_rows(Yk, Y)
        func_static[str(k)] = {
            "mean_cos": float(c.mean()),
            "min_cos": float(c.min()),
            "p10_cos": float(np.quantile(c, 0.1)),
        }

    # random-k control at 1024
    rng = np.random.default_rng(L + 17)
    idx_r = rng.choice(INTER, size=1024, replace=False)
    Hm = np.zeros_like(H)
    Hm[:, idx_r] = H[:, idx_r]
    Yk = Hm @ Wd.T
    c = cosine_rows(Yk, Y)
    random_1024 = {"mean_cos": float(c.mean()), "min_cos": float(c.min())}

    # group-64 / group-512 keep for 0.99-energy groups AND for top-g by energy
    def group_keep_cos(g_e, gsize, n_keep_list):
        n_g = g_e.shape[1]
        out = {}
        for ng in n_keep_list:
            Hm = np.zeros_like(H)
            for t in range(N_TOK):
                gi = np.argpartition(g_e[t], -ng)[-ng:]
                mask = np.zeros(n_g, dtype=bool)
                mask[gi] = True
                m = np.repeat(mask, gsize)
                Hm[t, m] = H[t, m]
            Yk = Hm @ Wd.T
            c = cosine_rows(Yk, Y)
            out[str(ng)] = {
                "mean_cos": float(c.mean()),
                "min_cos": float(c.min()),
                "keep_frac": ng / n_g,
                "active_cols": ng * gsize,
            }
        return out

    func_g64 = group_keep_cos(g64, G64, [8, 16, 32, 64, 128, 192, 256])
    func_g512 = group_keep_cos(g512, G512, [2, 4, 8, 16, 24, 32])

    # residual-input top-k: reload gate/up (freed). This is the expensive part.
    Wg = st.load_f32(f"language_model.model.layers.{L}.mlp.gate_proj.weight")
    Wu = st.load_f32(f"language_model.model.layers.{L}.mlp.up_proj.weight")
    eX = x.astype(np.float64) ** 2
    func_xin = {}
    for k in (32, 128, 256, 512, 1024, 2560):
        Xm = np.zeros_like(x)
        for t in range(N_TOK):
            idx = np.argpartition(eX[t], -k)[-k:]
            Xm[t, idx] = x[t, idx]
        Hk = silu(Xm @ Wg.T) * (Xm @ Wu.T)
        Yk = Hk @ Wd.T
        c = cosine_rows(Yk, Y)
        func_xin[str(k)] = {
            "mean_cos": float(c.mean()),
            "min_cos": float(c.min()),
            "p10_cos": float(np.quantile(c, 0.1)),
        }

    # static residual channels by mean |X|^2
    eX_mean = eX.mean(axis=0)
    func_xin_static = {}
    for k in (32, 128, 256, 512, 1024, 2560):
        idx = np.argpartition(eX_mean, -k)[-k:]
        Xm = np.zeros_like(x)
        Xm[:, idx] = x[:, idx]
        Hk = silu(Xm @ Wg.T) * (Xm @ Wu.T)
        Yk = Hk @ Wd.T
        c = cosine_rows(Yk, Y)
        func_xin_static[str(k)] = {"mean_cos": float(c.mean()), "min_cos": float(c.min())}

    # residual group-64 / 512 keep
    n64x = HIDDEN // G64
    n512x = HIDDEN // G512
    g64x = eX.reshape(N_TOK, n64x, G64).sum(axis=2)
    g512x = eX.reshape(N_TOK, n512x, G512).sum(axis=2)

    def xin_group(g_e, gsize, n_keep_list):
        n_g = g_e.shape[1]
        out = {}
        for ng in n_keep_list:
            Xm = np.zeros_like(x)
            for t in range(N_TOK):
                gi = np.argpartition(g_e[t], -ng)[-ng:]
                mask = np.zeros(n_g, dtype=bool)
                mask[gi] = True
                m = np.repeat(mask, gsize)
                Xm[t, m] = x[t, m]
            Hk = silu(Xm @ Wg.T) * (Xm @ Wu.T)
            Yk = Hk @ Wd.T
            c = cosine_rows(Yk, Y)
            out[str(ng)] = {
                "mean_cos": float(c.mean()),
                "min_cos": float(c.min()),
                "keep_frac": ng / n_g,
            }
        return out

    func_xin_g64 = xin_group(g64x, G64, [8, 16, 32, 48, 64, 80])
    func_xin_g512 = xin_group(g512x, G512, [2, 4, 6, 8, 10])

    # token predictability on intermediate top-128
    k = K_INT_JACCARD
    top_sets = []
    for t in range(N_TOK):
        idx = np.argpartition(eH_tok[t], -k)[-k:]
        top_sets.append(set(int(i) for i in idx))
    occ = defaultdict(list)
    for t, tid in enumerate(ids.tolist()):
        occ[int(tid)].append(t)
    same_id = []
    for ts in occ.values():
        if len(ts) >= 2:
            same_id.append(pairwise_mean_jaccard([top_sets[t] for t in ts]))
    rng = np.random.default_rng(100 + L)
    rand_js = []
    for _ in range(400):
        i, j = rng.integers(0, N_TOK, size=2)
        if i != j and ids[i] != ids[j]:
            rand_js.append(jaccard(top_sets[i], top_sets[j]))
    # same id different context
    t2off = {}
    for pi, a, b, _p, _ids in slices:
        for off, t in enumerate(range(a, b)):
            t2off[t] = off
    same_diff = []
    for ts in occ.values():
        pool = []
        seen_pref = set()
        for t in ts:
            off = t2off[t]
            if off < prefix_len:
                if off in seen_pref:
                    continue
                seen_pref.add(off)
            pool.append(t)
        if len(pool) >= 2:
            same_diff.append(pairwise_mean_jaccard([top_sets[t] for t in pool]))

    # write / read scale
    y_rms = float(np.sqrt(np.mean(Y.astype(np.float64) ** 2)))
    x_rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    h_rms = float(np.sqrt(np.mean(H.astype(np.float64) ** 2)))

    del Wg, Wu, Wd, H, Y, eH, eH_tok, absH

    return {
        "layer": L,
        "wall_s": time.perf_counter() - t0,
        "rss_gb_after": rss_gb(),
        "x_rms": x_rms,
        "h_rms": h_rms,
        "y_rms": y_rms,
        "y_over_x_rms": y_rms / x_rms if x_rms else None,
        "h_sparsity": near,
        "h_pr_mean_energy": None,  # filled if we keep e_mean
        "h_gini_mean_energy": float(gini(e_mean)),
        "h_pr": float(participation_ratio(e_mean)),
        "h_pr_over_n": float(participation_ratio(e_mean) / INTER),
        "h_top_energy_frac": {str(k): v for k, v in energy_frac_topk(e_mean, (32, 128, 256, 512, 1024, 2048, 4096)).items()},
        "h_k_for_energy_mean": {str(f): float(np.mean([d[str(f)] for d in k_needed])) for f in fracs},
        "h_k_for_energy_max": {str(f): int(np.max([d[str(f)] for d in k_needed])) for f in fracs},
        "h_g64_k99_mean": float(np.mean(k64_99)),
        "h_g64_k99_max": int(np.max(k64_99)),
        "h_g64_n": n64,
        "h_g512_k99_mean": float(np.mean(k512_99)),
        "h_g512_k99_max": int(np.max(k512_99)),
        "h_g512_n": n512,
        "h_duty": duty,
        "func_down_dynamic_topk": func_dyn,
        "func_down_static_topk": func_static,
        "func_down_random_1024": random_1024,
        "func_down_g64": func_g64,
        "func_down_g512": func_g512,
        "func_mlp_xin_dynamic": func_xin,
        "func_mlp_xin_static": func_xin_static,
        "func_mlp_xin_g64": func_xin_g64,
        "func_mlp_xin_g512": func_xin_g512,
        "h_token_jaccard": {
            "k": k,
            "same_token_id_mean": float(np.nanmean(same_id)) if same_id else None,
            "same_id_diff_context_mean": float(np.nanmean(same_diff)) if same_diff else None,
            "random_diff_id_mean": float(np.mean(rand_js)) if rand_js else None,
        },
    }


def main():
    t_all = time.perf_counter()
    receipt = json.load(open(os.path.join(CAP, "capture-result.json")))
    receipt_sha = sha256_file(os.path.join(CAP, "capture-result.json"))
    slices, ids = prompt_slices(receipt)
    prefix_len = common_prefix_len(slices)
    print(f"load hidden rss={rss_gb():.3f}", flush=True)
    X = load_hidden()
    print(f"hidden loaded shape={X.shape} rss={rss_gb():.3f}", flush=True)

    # receipt cross-check
    l0_mean = float(np.mean(np.abs(X[0])))
    l0_rms = float(np.sqrt(np.mean(X[0].astype(np.float64) ** 2)))
    print(f"L0 mean_abs={l0_mean} rms={l0_rms}", flush=True)

    st = SafeTensors(BF16)
    site = site_confirm(st, X)
    zeros = count_post_zero_3994(st)
    site.update(zeros)
    print("site", json.dumps(site, indent=2), flush=True)

    print("residual stats...", flush=True)
    residual = {}
    # aggregates
    k99_means = []
    prs = []
    ginis = []
    g64_99 = []
    g512_99 = []
    duty32_union = []
    duty32_always = []
    ch3994_ranks = []
    for L in range(N_LAYERS):
        r = residual_layer_stats(X, L)
        residual[str(L)] = r["json"]
        k99_means.append(r["json"]["k_for_energy_mean"]["0.99"])
        prs.append(r["json"]["pr_mean_energy"])
        ginis.append(r["json"]["gini_mean_energy"])
        g64_99.append(r["json"]["g64_k99_mean"])
        g512_99.append(r["json"]["g512_k99_mean"])
        duty32_union.append(r["json"]["duty"]["32"]["union"])
        duty32_always.append(r["json"]["duty"]["32"]["n_always"])
        ch3994_ranks.append(r["json"]["ch3994"]["rank"])
        if L % 8 == 0:
            print(f"  L{L} pr={prs[-1]:.1f} k99={k99_means[-1]:.1f} g64_99={g64_99[-1]:.1f} 3994rank={ch3994_ranks[-1]} union32={duty32_union[-1]}", flush=True)

    residual_summary = {
        "k99_mean_over_layers": float(np.mean(k99_means)),
        "k99_min_layer": float(np.min(k99_means)),
        "k99_max_layer": float(np.max(k99_means)),
        "pr_mean": float(np.mean(prs)),
        "pr_min": float(np.min(prs)),
        "pr_max": float(np.max(prs)),
        "gini_mean": float(np.mean(ginis)),
        "g64_k99_mean": float(np.mean(g64_99)),
        "g512_k99_mean": float(np.mean(g512_99)),
        "duty32_union_mean": float(np.mean(duty32_union)),
        "duty32_union_min": int(np.min(duty32_union)),
        "duty32_union_max": int(np.max(duty32_union)),
        "duty32_always_mean": float(np.mean(duty32_always)),
        "ch3994_rank_by_layer": ch3994_ranks,
        "ch3994_n_rank1": int(sum(1 for rnk in ch3994_ranks if rnk == 1)),
    }

    print("token predictability...", flush=True)
    tokpred = token_predictability(X, ids, slices, prefix_len)
    print(json.dumps(tokpred, indent=2), flush=True)

    print("union saturation...", flush=True)
    sat = union_saturation(X)
    print(json.dumps(sat, indent=2), flush=True)

    # unique token ids
    uniq = sorted(set(int(t) for t in ids.tolist()))
    repeats = sum(1 for tid in uniq if int(np.sum(ids == tid)) >= 2)

    print("MLP layers...", flush=True)
    mlp = {}
    for L in MLP_LAYERS:
        print(f"  MLP L{L} rss={rss_gb():.3f}", flush=True)
        mlp[str(L)] = mlp_one_layer(st, X, L, ids, slices, prefix_len)
        d = mlp[str(L)]
        print(
            f"    h_pr={d['h_pr']:.1f} h_k99={d['h_k_for_energy_mean']['0.99']:.1f} "
            f"union1024={d['h_duty']['1024']['union']} "
            f"dyn1024_cos={d['func_down_dynamic_topk']['1024']['mean_cos']:.5f} "
            f"stat1024_cos={d['func_down_static_topk']['1024']['mean_cos']:.5f} "
            f"xin256_cos={d['func_mlp_xin_dynamic']['256']['mean_cos']:.5f} "
            f"wall={d['wall_s']:.1f}s",
            flush=True,
        )

    # accounting constants (DERIVED)
    q4_bytes_per_group = 32 + 2  # 32 code + 2 scale
    accounting = {
        "N": SRC_PARAMS,
        "mlp_elements": 17_112_760_320,
        "attn_elements": 7_237_795_840,
        "table_elements": 2_542_796_800,
        "small_elements": 2_645_504,
        "g0_complete_bpw": 4.252735126866492,
        "token_gemv_bytes_q4": 13_611_663_360,
        "q4_bytes_gate": 47_349_760,
        "addressing_ns": 21_293_102.5,
        "token_ns_g024": 35_227_917,
        "bandwidth_addr_GBps": 13_611_663_360 / 21_293_102.5,
        "simdgroup": 32,
        "tpr": 64,
        "tg": 128,
        "pass_cols": 512,
    }

    result = {
        "schema": "hawking.g1.conditional_representation.v1",
        "capture": {
            "path": CAP,
            "schema": receipt["schema"],
            "status": receipt["status"],
            "sha256_self_claimed": receipt["sha256_self"],
            "capture_result_sha256": receipt_sha,
            "n_tokens": receipt["n_tokens"],
            "n_layers": receipt["n_layers"],
            "hidden": receipt["hidden"],
            "source": receipt["source"],
            "wall_s_capture": receipt["wall_s"],
        },
        "prompts": {
            "n": len(slices),
            "lengths": [s[2] - s[1] for s in slices],
            "shared_prefix_len": prefix_len,
            "n_unique_token_ids": len(uniq),
            "n_token_ids_with_repeats": repeats,
            "ids_head20": [int(x) for x in ids[:20]],
        },
        "site": site,
        "residual_summary": residual_summary,
        "residual_per_layer": residual,
        "token_predictability_residual": tokpred,
        "union_saturation": sat,
        "mlp": mlp,
        "accounting_constants": accounting,
        "wall_s": time.perf_counter() - t_all,
        "rss_gb_peak": rss_gb(),
        "labels": {
            "X": "CAPTURED_REAL_BF16_POST_NORM_HIDDEN 256x5120; site confirmed post-attn-norm by L7 γ[3994]=0 and X[:,3994]≡0",
            "ranks": "reliable on this cube",
            "magnitudes": "underdetermined for any fit; self-consistent reconstruction cosines on this X are ranks of consequence, not absolute error vs uncaptured sites",
            "mixer_x": "never captured; head-level routing NOT measured",
        },
    }
    with open(OUT, "w") as f:
        json.dump(result, f)
    print(f"WROTE {OUT} wall={result['wall_s']:.1f}s rss_peak={rss_gb():.3f}GB", flush=True)


if __name__ == "__main__":
    main()
