#!/usr/bin/env python3
"""Qwen3.8-27B reusable-grammar measurement.

Cluster, do not pairwise-mean. CPU only. No Metal, no generate, no live artifact writes.

Reads BF16 shards + 256-token post-norm hidden capture.
Peak RSS target < 15 GiB. One GEMV resident at a time except named pair probes.
"""
from __future__ import annotations

import json
import os
import resource
import struct
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.stats import wasserstein_distance

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

ROOT = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
)
CAP = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/"
    "qwen38-27b/activation-capture-v1"
)
OUT = Path(os.environ.get("OUT", "/tmp/g1_model_grammar.json"))
N_SOURCE = 26_895_998_464
N_LAYERS = 64
HIDDEN = 5120
INTER = 17408
N_TOK = 256
FIT_N = 192
HOLD_N = 64
PREFIX = "language_model.model.layers.{layer}."
SKETCH_S = 2048
SKETCH_SEED = 0xA11CE
HIST_BINS = 64
HIST_LO, HIST_HI = -6.0, 6.0
N_SOURCE_PARAMS = N_SOURCE

GQA = tuple(i for i in range(N_LAYERS) if (i + 1) % 4 == 0)
DN = tuple(i for i in range(N_LAYERS) if (i + 1) % 4 != 0)

CLASSES = {
    "mlp.gate_proj": dict(suffix="mlp.gate_proj.weight", layers=tuple(range(64)), shape=(17408, 5120)),
    "mlp.up_proj": dict(suffix="mlp.up_proj.weight", layers=tuple(range(64)), shape=(17408, 5120)),
    "mlp.down_proj": dict(suffix="mlp.down_proj.weight", layers=tuple(range(64)), shape=(5120, 17408)),
    "lin.in_proj_qkv": dict(suffix="linear_attn.in_proj_qkv.weight", layers=DN, shape=(10240, 5120)),
    "lin.in_proj_z": dict(suffix="linear_attn.in_proj_z.weight", layers=DN, shape=(6144, 5120)),
    "lin.out_proj": dict(suffix="linear_attn.out_proj.weight", layers=DN, shape=(5120, 6144)),
    "gqa.q_proj": dict(suffix="self_attn.q_proj.weight", layers=GQA, shape=(12288, 5120)),
    "gqa.k_proj": dict(suffix="self_attn.k_proj.weight", layers=GQA, shape=(1024, 5120)),
    "gqa.v_proj": dict(suffix="self_attn.v_proj.weight", layers=GQA, shape=(1024, 5120)),
    "gqa.o_proj": dict(suffix="self_attn.o_proj.weight", layers=GQA, shape=(5120, 6144)),
}

# tensors whose K (GEMV contraction) is hidden — honest site for captured X
ACT_OK = {
    "mlp.gate_proj",
    "mlp.up_proj",
    "lin.in_proj_qkv",
    "lin.in_proj_z",
    "gqa.q_proj",
    "gqa.k_proj",
    "gqa.v_proj",
}


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} rss={rss_gb():.2f}GiB] {msg}", flush=True)


def jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    return x


def dump(obj) -> None:
    tmp = OUT.with_suffix(".partial.json")
    tmp.write_text(json.dumps(jsonable(obj), indent=2))
    tmp.replace(OUT)


def parse_all_headers(root: Path) -> dict:
    table = {}
    for shard in sorted(root.glob("model-*.safetensors")):
        with open(shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            table[name] = {
                "path": str(shard),
                "header_nbytes": n,
                "dtype": info["dtype"],
                "shape": tuple(info["shape"]),
                "begin": int(begin),
                "end": int(end),
            }
    return table


def mmap_u16(info) -> np.memmap:
    if info["dtype"] != "BF16":
        raise RuntimeError(f"expected BF16, got {info['dtype']}")
    n = int(np.prod(info["shape"]))
    return np.memmap(
        info["path"],
        dtype="<u2",
        mode="r",
        offset=8 + info["header_nbytes"] + info["begin"],
        shape=(n,),
    )


def mmap_f32(info) -> np.memmap:
    n = int(np.prod(info["shape"]))
    dt = info["dtype"]
    if dt == "F32":
        return np.memmap(
            info["path"],
            dtype="<f4",
            mode="r",
            offset=8 + info["header_nbytes"] + info["begin"],
            shape=info["shape"],
        )
    raise RuntimeError(dt)


def bf16u16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def load_f32(info) -> np.ndarray:
    dt = info["dtype"]
    if dt == "BF16":
        raw = mmap_u16(info)
        return bf16u16_to_f32(np.asarray(raw)).reshape(info["shape"]).copy()
    if dt == "F32":
        return np.asarray(mmap_f32(info)).copy()
    raise RuntimeError(dt)


def load_vec_f32(info) -> np.ndarray:
    """Any 1-D or small vector, BF16 or F32."""
    return load_f32(info).reshape(-1)


def tname(layer: int, suffix: str) -> str:
    return PREFIX.format(layer=layer) + suffix


def kmeans_labels(X: np.ndarray, k: int, seed: int = 0, n_iter: int = 25):
    X = np.ascontiguousarray(X, dtype=np.float64)
    # standardize columns
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (X - mu) / sd
    rng = np.random.default_rng(seed)
    # kmeans++ init via scipy minit='++'
    try:
        cb, lab = kmeans2(Z, k, iter=n_iter, minit="++", rng=rng, missing="warn")
    except TypeError:
        cb, lab = kmeans2(Z, k, iter=n_iter, minit="++")
    # empty clusters → leftover -1; remap
    lab = np.asarray(lab, dtype=np.int32)
    if (lab < 0).any():
        # assign leftovers to nearest
        d = ((Z[:, None, :] - cb[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(axis=1).astype(np.int32)
    inertia = float(((Z - cb[lab]) ** 2).sum())
    return lab, cb, inertia, Z


def silhouette(Z: np.ndarray, lab: np.ndarray) -> float:
    """Mean silhouette. O(n^2 d) — n is 16..64."""
    n = Z.shape[0]
    if n < 3:
        return float("nan")
    # pairwise euclid
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1) + 1e-12)
    out = []
    for i in range(n):
        same = lab == lab[i]
        same[i] = False
        if not same.any():
            continue
        a = float(d[i, same].mean())
        bs = []
        for c in np.unique(lab):
            if c == lab[i]:
                continue
            m = lab == c
            if m.any():
                bs.append(float(d[i, m].mean()))
        if not bs:
            continue
        b = min(bs)
        out.append((b - a) / max(a, b))
    return float(np.mean(out)) if out else float("nan")


def spectral_labels(S: np.ndarray, k: int, seed: int = 0):
    """Normalized spectral clustering on similarity S (symmetric, >=0)."""
    n = S.shape[0]
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 0.0)
    d = S.sum(axis=1)
    d = np.where(d < 1e-12, 1e-12, d)
    Dmh = 1.0 / np.sqrt(d)
    L = np.eye(n) - (Dmh[:, None] * S * Dmh[None, :])
    w, v = np.linalg.eigh(L)
    evecs = v[:, :k]
    # row-normalize
    nrm = np.linalg.norm(evecs, axis=1, keepdims=True)
    nrm = np.where(nrm < 1e-12, 1.0, nrm)
    Y = evecs / nrm
    lab, _, inert, Z = kmeans_labels(Y, k, seed=seed)
    return lab, float(w[1]) if n > 1 else 0.0, inert


def connected_components(S: np.ndarray, thr: float):
    n = S.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] >= thr:
                union(i, j)
    comps = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)
    groups = sorted(comps.values(), key=len, reverse=True)
    return {
        "threshold": float(thr),
        "n_components": len(groups),
        "sizes": [len(g) for g in groups],
        "largest": groups[0],
        "groups": groups,
    }


def countsketch(values: np.ndarray, start_index: int, s: int = SKETCH_S, seed: int = SKETCH_SEED):
    n = values.size
    idx = np.arange(start_index, start_index + n, dtype=np.uint64)
    z = idx + np.uint64(seed)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    buckets = (z % np.uint64(s)).astype(np.int64)
    signs = np.where((z >> np.uint64(63)) == 0, 1.0, -1.0)
    out = np.zeros(s, dtype=np.float64)
    np.add.at(out, buckets, signs * values.astype(np.float64, copy=False).ravel())
    return out


def moments(a: np.ndarray):
    x = a.ravel().astype(np.float64, copy=False)
    n = x.size
    mean = float(x.mean())
    # centered
    c = x - mean
    m2 = float((c * c).mean())
    m3 = float((c * c * c).mean())
    m4 = float((c * c * c * c).mean())
    var = m2
    std = float(np.sqrt(var)) if var > 0 else 0.0
    rms = float(np.sqrt((x * x).mean()))
    skew = float(m3 / (std**3)) if std > 0 else 0.0
    kurt = float(m4 / (std**4) - 3.0) if std > 0 else 0.0
    maxabs = float(np.max(np.abs(x)))
    return dict(
        n=int(n),
        mean=mean,
        std=std,
        rms=rms,
        skew=skew,
        excess_kurtosis=kurt,
        maxabs=maxabs,
        peak_over_rms=(maxabs / rms) if rms > 0 else 0.0,
    )


def rms_axis(W: np.ndarray, axis: int):
    # rms over the other axis
    v = np.sqrt(np.mean(np.square(W, dtype=np.float64), axis=axis))
    med = float(np.median(v))
    return dict(
        mean=float(v.mean()),
        std=float(v.std()),
        p01=float(np.quantile(v, 0.01)),
        p50=float(np.median(v)),
        p99=float(np.quantile(v, 0.99)),
        p999=float(np.quantile(v, 0.999)),
        max=float(v.max()),
        n_ge4x=int(np.sum(v >= 4.0 * med)) if med > 0 else 0,
        n_ge10x=int(np.sum(v >= 10.0 * med)) if med > 0 else 0,
        top5_idx=np.argsort(v)[-5:][::-1].astype(int).tolist(),
        top5_val=np.sort(v)[-5:][::-1].tolist(),
        vec=v.astype(np.float32),
    )


def hist_unit(W: np.ndarray, std: float):
    x = W.ravel()
    if std <= 0:
        std = 1.0
    z = x / np.float32(std)
    h, edges = np.histogram(z, bins=HIST_BINS, range=(HIST_LO, HIST_HI), density=False)
    # leftover tails
    n_lo = int(np.sum(z < HIST_LO))
    n_hi = int(np.sum(z > HIST_HI))
    dens = h.astype(np.float64)
    dens = dens / max(dens.sum(), 1.0)
    return dens, edges, n_lo, n_hi


def rsvd_right(W: np.ndarray, k: int = 32, p: int = 8, seed: int = 0):
    """Right singular vectors (hidden-side if cols==hidden or we pass W accordingly)."""
    rows, cols = W.shape
    k_use = min(k, rows, cols)
    l = min(k_use + p, cols, rows)
    rng = np.random.default_rng(seed)
    Omega = rng.standard_normal((cols, l)).astype(np.float32)
    Y = W @ Omega
    Q, _ = np.linalg.qr(Y, mode="reduced")
    B = Q.T @ W
    # B: l × cols
    _, S, Vt = np.linalg.svd(B.astype(np.float64), full_matrices=False)
    V = Vt[:k_use].T.astype(np.float64)  # cols × k
    S = S[:k_use].astype(np.float64)
    return S, V


def fro_sq(W: np.ndarray) -> float:
    return float(np.square(W, dtype=np.float64).sum())


def cosine_flat(A: np.ndarray, B: np.ndarray) -> float:
    a = A.ravel().astype(np.float64, copy=False)
    b = B.ravel().astype(np.float64, copy=False)
    na = float(np.dot(a, a))
    nb = float(np.dot(b, b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / np.sqrt(na * nb))


def rel_delta(A: np.ndarray, B: np.ndarray) -> float:
    """||A-B||_F / ||B||_F"""
    d = (A.astype(np.float64) - B.astype(np.float64)).ravel()
    b = B.ravel().astype(np.float64, copy=False)
    nb = float(np.dot(b, b))
    if nb <= 0:
        return 0.0
    return float(np.sqrt(np.dot(d, d) / nb))


def lloyd_levels(sample: np.ndarray, nlev: int = 16, iters: int = 8):
    """1-D Lloyd-Max on a 1-D sample. Returns sorted levels."""
    x = sample.astype(np.float64, copy=False).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.linspace(-1, 1, nlev)
    lo, hi = np.quantile(x, 0.001), np.quantile(x, 0.999)
    lev = np.linspace(lo, hi, nlev)
    for _ in range(iters):
        # assign
        edges = 0.5 * (lev[1:] + lev[:-1])
        lab = np.searchsorted(edges, x)
        new = lev.copy()
        for i in range(nlev):
            m = lab == i
            if m.any():
                new[i] = float(x[m].mean())
        if np.max(np.abs(new - lev)) < 1e-6:
            lev = new
            break
        lev = new
    return np.sort(lev)


def quantize_to_levels(x: np.ndarray, levels: np.ndarray) -> np.ndarray:
    # nearest
    # x[...,] vs levels[L]
    # for memory, chunk
    xf = x.ravel()
    out = np.empty_like(xf, dtype=np.float32)
    bs = 1 << 20
    lev = levels.astype(np.float32)
    for i in range(0, xf.size, bs):
        sl = xf[i : i + bs]
        d = np.abs(sl[:, None] - lev[None, :])
        out[i : i + sl.size] = lev[d.argmin(axis=1)]
    return out.reshape(x.shape)


def pca_energy(X: np.ndarray, k: int = 16):
    """Rows are samples. Return energy fractions of top-k PCs (via SVD of centered)."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    # economy SVD on the smaller side
    n, d = Xc.shape
    if n == 0 or d == 0:
        return {}
    k_use = min(k, n, d)
    if n <= d:
        # SVD of Xc
        _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    else:
        # Gram n is large? here n<=64, d can be 2048 or 17408
        if d > 4096:
            # randomized
            rng = np.random.default_rng(0)
            l = min(k_use + 8, d, n)
            Omega = rng.standard_normal((d, l))
            Y = Xc @ Omega
            Q, _ = np.linalg.qr(Y, mode="reduced")
            B = Q.T @ Xc
            _, S, _ = np.linalg.svd(B, full_matrices=False)
        else:
            _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    e = S[:k_use] ** 2
    tot = float((Xc * Xc).sum())
    if tot <= 0:
        return {"k": k_use, "frac": [0.0] * k_use, "cum": [0.0] * k_use}
    frac = (e / tot).tolist()
    cum = np.cumsum(e / tot).tolist()
    return {"k": int(k_use), "frac": frac, "cum": cum, "tot_var": tot}


def load_hiddens(cap: Path):
    meta = json.loads((cap / "capture-result.json").read_text())
    Xs = np.empty((N_LAYERS, N_TOK, HIDDEN), dtype=np.float32)
    for i in range(N_LAYERS):
        p = cap / "hidden" / f"L{i:02d}.f32"
        a = np.fromfile(p, dtype="<f4")
        if a.size != N_TOK * HIDDEN:
            raise RuntimeError(f"{p} size {a.size}")
        Xs[i] = a.reshape(N_TOK, HIDDEN)
    return meta, Xs


def pair_stats(mat: np.ndarray):
    n = mat.shape[0]
    if n < 2:
        return {}
    iu = np.triu_indices(n, 1)
    v = mat[iu]
    return dict(
        n_pairs=int(v.size),
        mean=float(v.mean()),
        p05=float(np.quantile(v, 0.05)),
        p50=float(np.quantile(v, 0.50)),
        p95=float(np.quantile(v, 0.95)),
        min=float(v.min()),
        max=float(v.max()),
    )


def family_label_sets(layers):
    layers = list(layers)
    out = {}
    out["all"] = layers
    # depth halves / quartiles
    n = len(layers)
    out["first_half"] = layers[: n // 2]
    out["second_half"] = layers[n // 2 :]
    q = max(1, n // 4)
    out["q0"] = layers[0:q]
    out["q1"] = layers[q : 2 * q]
    out["q2"] = layers[2 * q : 3 * q]
    out["q3"] = layers[3 * q :]
    # residue classes in original layer index
    for m in (4, 16):
        for r in range(m):
            mem = [L for L in layers if L % m == r]
            if len(mem) >= 2:
                out[f"mod{m}r{r}"] = mem
    return out


def main():
    t0 = time.time()
    log("parse headers")
    table = parse_all_headers(ROOT)
    cfg = json.loads((ROOT / "config.json").read_text())
    cap_meta, Xall = load_hiddens(CAP)
    Xfit = Xall[:, :FIT_N, :]
    Xhold = Xall[:, FIT_N:, :]
    log(f"headers={len(table)} hiddens={Xall.shape} cap_sha={cap_meta.get('sha256_self')}")

    report = {
        "schema": "hawking.g1.qwen38_model_grammar.v1",
        "kind": "MEASURED",
        "started_unix": t0,
        "identity": {
            "weight_root": str(ROOT),
            "n_header_tensors": len(table),
            "config_model_type": cfg.get("model_type"),
            "text_model_type": cfg.get("text_config", {}).get("model_type"),
            "num_hidden_layers": cfg.get("text_config", {}).get("num_hidden_layers"),
            "hidden_size": cfg.get("text_config", {}).get("hidden_size"),
            "intermediate_size": cfg.get("text_config", {}).get("intermediate_size"),
            "num_experts": cfg.get("text_config", {}).get("num_experts"),
            "attn_output_gate": cfg.get("text_config", {}).get("attn_output_gate"),
            "n_source_params": N_SOURCE,
            "capture_sha256_self": cap_meta.get("sha256_self"),
            "capture_status": cap_meta.get("status"),
            "capture_n_tokens": cap_meta.get("n_tokens"),
            "capture_schema": cap_meta.get("schema"),
            "fit_n": FIT_N,
            "hold_n": HOLD_N,
            "prompt_n_tokens": [p.get("n_tokens") for p in cap_meta.get("prompts", [])],
        },
        "host": {
            "cpu_only": True,
            "metal": False,
            "inference": False,
            "blas_threads": 8,
            "script": "/tmp/g1_model_grammar.py",
        },
        "classes": {},
        "small_tensors": {},
        "heads": {},
        "within_layer": {},
        "procrustes": {},
        "family_residuals": {},
        "manifold": {},
        "rankings": {},
    }

    # ------------------------------------------------------------------
    # 0. Small-tensor grammar (norms, A_log, dt_bias, q/k_norm)
    # ------------------------------------------------------------------
    log("small tensors")
    small = {}
    # layernorms
    for kind in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        mats = []
        layers = []
        for L in range(N_LAYERS):
            info = table[tname(L, kind)]
            v = load_vec_f32(info).astype(np.float32)
            mats.append(v)
            layers.append(L)
        M = np.stack(mats, 0)  # 64 × 5120
        # pairwise cosine
        nn = np.linalg.norm(M, axis=1, keepdims=True)
        nn = np.where(nn < 1e-12, 1.0, nn)
        U = M / nn
        C = U @ U.T
        adj = [float(C[i, i + 1]) for i in range(N_LAYERS - 1)]
        # template = mean
        T = M.mean(axis=0)
        rels = [rel_delta(M[i], T) for i in range(N_LAYERS)]
        # kmeans k=2,3,4 on the vectors
        cl = {}
        for k in (2, 3, 4, 8):
            lab, _, inert, Z = kmeans_labels(M, k, seed=1)
            cl[str(k)] = {
                "labels": lab.tolist(),
                "silhouette": silhouette(Z, lab),
                "inertia": inert,
                "sizes": np.bincount(lab, minlength=k).tolist(),
            }
        small[kind] = {
            "shape": [N_LAYERS, int(M.shape[1])],
            "pairwise_cosine": pair_stats(C),
            "adjacent_cosine_mean": float(np.mean(adj)),
            "adjacent_cosine_min": float(np.min(adj)),
            "adjacent_cosine_max": float(np.max(adj)),
            "mean_template_rel_delta_mean": float(np.mean(rels)),
            "mean_template_rel_delta_min": float(np.min(rels)),
            "mean_template_rel_delta_max": float(np.max(rels)),
            "pca": pca_energy(M, k=8),
            "kmeans": cl,
            "per_layer_rms": np.sqrt((M.astype(np.float64) ** 2).mean(1)).tolist(),
            "exact_zeros": {
                "layers": [int(i) for i in range(N_LAYERS) if np.any(M[i] == 0)],
                "n_zero_coords": [int(np.sum(M[i] == 0)) for i in range(N_LAYERS)],
            },
        }
        # period-16 cosine mean
        d16 = []
        for i, Li in enumerate(layers):
            for j, Lj in enumerate(layers):
                if Lj - Li == 16:
                    d16.append(float(C[i, j]))
        small[kind]["d16_cosine"] = {
            "n": len(d16),
            "mean": float(np.mean(d16)) if d16 else None,
            "min": float(np.min(d16)) if d16 else None,
            "max": float(np.max(d16)) if d16 else None,
        }
        log(f"  {kind} adj_cos={np.mean(adj):.4f} tmpl_rel={np.mean(rels):.4f} pca1={small[kind]['pca']['cum'][0]:.3f}")

    for kind, layers in (
        ("self_attn.q_norm.weight", GQA),
        ("self_attn.k_norm.weight", GQA),
    ):
        mats = [load_vec_f32(table[tname(L, kind)]).astype(np.float32) for L in layers]
        M = np.stack(mats, 0)
        nn = np.linalg.norm(M, axis=1, keepdims=True)
        nn = np.where(nn < 1e-12, 1.0, nn)
        C = (M / nn) @ (M / nn).T
        T = M.mean(0)
        rels = [rel_delta(M[i], T) for i in range(len(layers))]
        small[kind] = {
            "layers": list(layers),
            "shape": [len(layers), int(M.shape[1])],
            "pairwise_cosine": pair_stats(C),
            "mean_template_rel_delta_mean": float(np.mean(rels)),
            "pca": pca_energy(M, k=8),
        }

    for kind in ("linear_attn.A_log", "linear_attn.dt_bias"):
        mats = [load_vec_f32(table[tname(L, kind)]).astype(np.float32) for L in DN]
        M = np.stack(mats, 0)
        nn = np.linalg.norm(M, axis=1, keepdims=True)
        nn = np.where(nn < 1e-12, 1.0, nn)
        C = (M / nn) @ (M / nn).T
        T = M.mean(0)
        rels = [rel_delta(M[i], T) for i in range(len(DN))]
        small[kind] = {
            "layers": list(DN),
            "shape": [len(DN), int(M.shape[1])],
            "dtype": table[tname(DN[0], kind)]["dtype"],
            "pairwise_cosine": pair_stats(C),
            "mean_template_rel_delta_mean": float(np.mean(rels)),
            "pca": pca_energy(M, k=8),
            "kmeans3": None,
        }
        lab, _, _, Z = kmeans_labels(M, 3, seed=2)
        small[kind]["kmeans3"] = {
            "labels": lab.tolist(),
            "silhouette": silhouette(Z, lab),
            "sizes": np.bincount(lab, minlength=3).tolist(),
        }

    # in_proj_a / in_proj_b : 48 × 5120, cheap exact cosine
    for kind in ("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"):
        mats = [load_f32(table[tname(L, kind)]).astype(np.float32).ravel() for L in DN]
        M = np.stack(mats, 0)
        nn = np.linalg.norm(M, axis=1, keepdims=True)
        nn = np.where(nn < 1e-12, 1.0, nn)
        C = (M / nn) @ (M / nn).T
        T = M.mean(0)
        rels = [rel_delta(M[i], T) for i in range(len(DN))]
        adj = [float(C[i, i + 1]) for i in range(len(DN) - 1)]
        small[kind] = {
            "layers": list(DN),
            "shape_each": list(table[tname(DN[0], kind)]["shape"]),
            "pairwise_cosine": pair_stats(C),
            "adjacent_cosine_mean": float(np.mean(adj)),
            "mean_template_rel_delta_mean": float(np.mean(rels)),
            "pca": pca_energy(M, k=8),
        }
        log(f"  {kind} pair_cos_mean={small[kind]['pairwise_cosine']['mean']:.4f} tmpl={np.mean(rels):.4f}")

    report["small_tensors"] = small
    dump(report)

    # ------------------------------------------------------------------
    # 1. Per-class GEMV pass: fingerprints, sketches, V32, act-maps
    # ------------------------------------------------------------------
    for cname, spec in CLASSES.items():
        layers = list(spec["layers"])
        suffix = spec["suffix"]
        log(f"CLASS {cname} n={len(layers)} shape={spec['shape']}")
        nL = len(layers)
        rows, cols = spec["shape"]
        hidden_side_right = cols == HIDDEN or (cname.startswith("mlp.") and cname != "mlp.down_proj")
        # down is 5120×17408 — right is intermediate; left is hidden

        feats = []  # for kmeans
        per = []
        sketches = np.zeros((nL, SKETCH_S), dtype=np.float64)
        row_rms_mat = np.zeros((nL, rows), dtype=np.float32)
        col_rms_mat = np.zeros((nL, cols), dtype=np.float32)
        Vstack = np.zeros((nL, cols, 32), dtype=np.float64)
        Sstack = np.zeros((nL, 32), dtype=np.float64)
        hists = np.zeros((nL, HIST_BINS), dtype=np.float64)
        fro = np.zeros(nL, dtype=np.float64)
        # act maps: store Y_fit as (nL, FIT_N, proj) via random projection of output
        # plus exact Y energy and a small exact cosine via sketch of Y
        Y_SK = 256
        rng_y = np.random.default_rng(7)
        Yproj_R = rng_y.standard_normal((rows, Y_SK)).astype(np.float32) / np.sqrt(Y_SK)
        Yfit_sk = np.zeros((nL, FIT_N, Y_SK), dtype=np.float32) if cname in ACT_OK else None
        Yhold_sk = np.zeros((nL, HOLD_N, Y_SK), dtype=np.float32) if cname in ACT_OK else None
        # also store native-site Y energy
        act_fit_fnorm = np.zeros(nL, dtype=np.float64) if cname in ACT_OK else None
        act_hold_fnorm = np.zeros(nL, dtype=np.float64) if cname in ACT_OK else None

        # for shared-level: collect 80k standardized samples per tensor
        samp_per = 80_000
        samples = np.zeros((nL, samp_per), dtype=np.float32)

        # row-duplicate probe: 192 random rows pairwise max cosine
        n_row_s = min(192, rows)
        row_maxcos = np.zeros(nL, dtype=np.float64)
        row_meancos = np.zeros(nL, dtype=np.float64)
        row_n_gt90 = np.zeros(nL, dtype=np.int32)

        for i, L in enumerate(layers):
            info = table[tname(L, suffix)]
            if tuple(info["shape"]) != (rows, cols):
                raise RuntimeError(f"{tname(L, suffix)} shape {info['shape']} != {(rows, cols)}")
            W = load_f32(info)
            mom = moments(W)
            rr = rms_axis(W, axis=1)  # per-row rms
            cr = rms_axis(W, axis=0)  # per-col rms
            row_rms_mat[i] = rr.pop("vec")
            col_rms_mat[i] = cr.pop("vec")
            dens, _, n_lo, n_hi = hist_unit(W, mom["std"])
            hists[i] = dens
            sketches[i] = countsketch(W, 0)
            fro[i] = fro_sq(W)
            # Hidden-side rSVD so overlap is in a common 5120-space when possible.
            # down/o are [hidden, K] — use W.T so V lives in hidden.
            if rows == HIDDEN and cols != HIDDEN:
                Sv, Vh = rsvd_right(W.T, k=32, p=8, seed=1000 + L)
                # Vh is hidden × 32; store into a hidden-width stack
                if Vstack.shape[1] != HIDDEN:
                    # allocated cols-wide; rebuild once
                    pass
                Sv, V = Sv, Vh
                # write into first HIDDEN rows of Vstack (cols may be 6144/17408)
                kk = Sv.size
                Sstack[i, :kk] = Sv
                Vstack[i, :HIDDEN, :kk] = V
            else:
                Sv, V = rsvd_right(W, k=32, p=8, seed=1000 + L)
                kk = Sv.size
                Sstack[i, :kk] = Sv
                Vstack[i, :, :kk] = V
            energy_top8 = float((Sv[:8] ** 2).sum() / fro[i]) if fro[i] > 0 else 0.0
            energy_top32 = float((Sv[:32] ** 2).sum() / fro[i]) if fro[i] > 0 else 0.0

            # standardized sample
            rng = np.random.default_rng(3000 + L)
            flat = W.ravel()
            take = rng.integers(0, flat.size, size=samp_per)
            std = mom["std"] if mom["std"] > 0 else 1.0
            samples[i] = (flat[take] / np.float32(std)).astype(np.float32)

            # row structure on a sample of rows
            ridx = rng.choice(rows, size=n_row_s, replace=False)
            R = W[ridx].astype(np.float64)
            Rn = np.linalg.norm(R, axis=1, keepdims=True)
            Rn = np.where(Rn < 1e-12, 1.0, Rn)
            Rc = (R / Rn) @ (R / Rn).T
            np.fill_diagonal(Rc, 0.0)
            iu = np.triu_indices(n_row_s, 1)
            rv = Rc[iu]
            row_maxcos[i] = float(rv.max()) if rv.size else 0.0
            row_meancos[i] = float(rv.mean()) if rv.size else 0.0
            row_n_gt90[i] = int(np.sum(rv >= 0.90))

            feat = [
                mom["std"],
                mom["excess_kurtosis"],
                mom["peak_over_rms"],
                mom["skew"],
                rr["p99"] / max(rr["p50"], 1e-12),
                cr["p99"] / max(cr["p50"], 1e-12),
                energy_top8,
                energy_top32,
                float(rr["n_ge4x"]),
                float(cr["n_ge4x"]),
            ]
            feat.extend(dens[::4].tolist())  # 16 hist samples
            feats.append(feat)

            # activation maps on native-layer X (honest only if K=hidden)
            act_blk = None
            if cname in ACT_OK and W.shape[1] == HIDDEN:
                xf = Xfit[L]  # FIT_N × 5120
                xh = Xhold[L]
                # Y = X @ W.T  → (T, rows)
                Yf = xf @ W.T
                Yh = xh @ W.T
                act_fit_fnorm[i] = float(np.square(Yf, dtype=np.float64).sum())
                act_hold_fnorm[i] = float(np.square(Yh, dtype=np.float64).sum())
                Yfit_sk[i] = Yf @ Yproj_R
                Yhold_sk[i] = Yh @ Yproj_R
                act_blk = {
                    "fit_fnorm": act_fit_fnorm[i],
                    "hold_fnorm": act_hold_fnorm[i],
                    "site": "native_layer_post_norm_hidden",
                    "capture_note": "ranks reliably; magnitudes underdetermined (256 tok)",
                }
                del Yf, Yh

            per.append(
                {
                    "layer": int(L),
                    "moments": mom,
                    "row_rms": rr,
                    "col_rms": cr,
                    "hist_tail_lo": int(n_lo),
                    "hist_tail_hi": int(n_hi),
                    "energy_top8": energy_top8,
                    "energy_top32": energy_top32,
                    "row_sample_max_cosine": row_maxcos[i],
                    "row_sample_mean_cosine": row_meancos[i],
                    "row_sample_n_pairs_ge_0.90": int(row_n_gt90[i]),
                    "row_sample_n": int(n_row_s),
                    "act": act_blk,
                }
            )
            del W
            if (i + 1) % 8 == 0 or i == nL - 1:
                log(f"  {cname} {i+1}/{nL} L{L} kurt={mom['excess_kurtosis']:.3f} e32={energy_top32:.3f}")

        feats = np.asarray(feats, dtype=np.float64)

        # --- fingerprint clustering ---
        km = {}
        for k in (2, 3, 4, 8):
            if nL < k + 1:
                continue
            lab, _, inert, Z = kmeans_labels(feats, k, seed=11)
            km[str(k)] = {
                "labels": lab.tolist(),
                "layers": layers,
                "silhouette": silhouette(Z, lab),
                "inertia": inert,
                "sizes": np.bincount(lab, minlength=k).tolist(),
                "purity_vs_mod4": None,
                "purity_vs_depth_half": None,
            }
            # purity vs mixer / depth
            lab = np.asarray(lab)
            def purity(groups):
                # mean over clusters of majority-group fraction
                acc = []
                for c in range(k):
                    m = lab == c
                    if not m.any():
                        continue
                    # majority among provided group ids
                    vals, cnts = np.unique(groups[m], return_counts=True)
                    acc.append(float(cnts.max()) / float(m.sum()))
                return float(np.mean(acc)) if acc else None

            g_mod4 = np.array([L % 4 for L in layers])
            g_half = np.array([int(L >= 32) for L in layers])
            g_mod16 = np.array([L % 16 for L in layers])
            km[str(k)]["purity_vs_mod4"] = purity(g_mod4)
            km[str(k)]["purity_vs_depth_half"] = purity(g_half)
            km[str(k)]["purity_vs_mod16"] = purity(g_mod16)

        # --- subspace overlap matrix (k=32) ---
        S_ov = np.zeros((nL, nL), dtype=np.float64)
        for a in range(nL):
            Va = Vstack[a]
            for b in range(a, nL):
                M = Va.T @ Vstack[b]
                ov = float(np.square(M).sum() / 32.0)
                S_ov[a, b] = S_ov[b, a] = ov
        spec_cl = {}
        for k in (2, 3, 4):
            if nL < k + 1:
                continue
            lab, gap, inert = spectral_labels(S_ov, k, seed=4)
            spec_cl[str(k)] = {
                "labels": lab.tolist(),
                "fiedler": gap,
                "silhouette_on_evecs_inertia": inert,
                "sizes": np.bincount(lab, minlength=k).tolist(),
            }
        comps = [connected_components(S_ov, thr) for thr in (0.25, 0.35, 0.45, 0.55)]

        # adjacent / distance-16 overlap
        def ov_at_dist(d):
            vals = []
            for i, Li in enumerate(layers):
                for j, Lj in enumerate(layers):
                    if Lj - Li == d:
                        vals.append(float(S_ov[i, j]))
            if not vals:
                return None
            return dict(n=len(vals), mean=float(np.mean(vals)), min=float(np.min(vals)), max=float(np.max(vals)))

        # --- histogram Wasserstein vs class-mean ---
        href = hists.mean(axis=0)
        href = href / max(href.sum(), 1e-12)
        centers = 0.5 * (np.linspace(HIST_LO, HIST_HI, HIST_BINS + 1)[1:] + np.linspace(HIST_LO, HIST_HI, HIST_BINS + 1)[:-1])
        w1_ref = []
        for i in range(nL):
            w1_ref.append(float(wasserstein_distance(centers, centers, hists[i], href)))
        # pairwise W1 (sample if nL=64: 2016 pairs — cheap)
        w1_pairs = []
        for i in range(nL):
            for j in range(i + 1, nL):
                w1_pairs.append(float(wasserstein_distance(centers, centers, hists[i], hists[j])))

        # shared vs per-tensor 16-level reconstruction on the standardized samples
        # per-tensor
        loc_rel = []
        loc_levels = []
        for i in range(nL):
            lev = lloyd_levels(samples[i], 16, iters=6)
            loc_levels.append(lev)
            rec = quantize_to_levels(samples[i], lev)
            num = float(np.square(samples[i] - rec).sum())
            den = float(np.square(samples[i]).sum())
            loc_rel.append(num / den if den > 0 else 0.0)
        pooled = samples.reshape(-1)
        # subsample pooled for Lloyd
        rng = np.random.default_rng(9)
        pool_s = pooled[rng.integers(0, pooled.size, size=min(pooled.size, 400_000))]
        shared_lev = lloyd_levels(pool_s, 16, iters=8)
        sh_rel = []
        for i in range(nL):
            rec = quantize_to_levels(samples[i], shared_lev)
            num = float(np.square(samples[i] - rec).sum())
            den = float(np.square(samples[i]).sum())
            sh_rel.append(num / den if den > 0 else 0.0)
        # also: 3-cluster level sets from fingerprint k=3 if present
        cluster_level = None
        if "3" in km:
            lab3 = np.asarray(km["3"]["labels"])
            crel = []
            for c in range(3):
                idx = np.where(lab3 == c)[0]
                if idx.size == 0:
                    continue
                ps = samples[idx].reshape(-1)
                ps = ps[rng.integers(0, ps.size, size=min(ps.size, 200_000))]
                lev = lloyd_levels(ps, 16, iters=6)
                for i in idx:
                    rec = quantize_to_levels(samples[i], lev)
                    num = float(np.square(samples[i] - rec).sum())
                    den = float(np.square(samples[i]).sum())
                    crel.append(num / den if den > 0 else 0.0)
            cluster_level = dict(mean_rel_mse=float(np.mean(crel)), max_rel_mse=float(np.max(crel)))

        # --- act-map cosine matrix (projected Y) ---
        act_block = None
        if Yfit_sk is not None:
            # cosine of flattened Y sketches
            Af = Yfit_sk.reshape(nL, -1).astype(np.float64)
            nrm = np.linalg.norm(Af, axis=1, keepdims=True)
            nrm = np.where(nrm < 1e-12, 1.0, nrm)
            Cact = (Af / nrm) @ (Af / nrm).T
            Ah = Yhold_sk.reshape(nL, -1).astype(np.float64)
            nrmh = np.linalg.norm(Ah, axis=1, keepdims=True)
            nrmh = np.where(nrmh < 1e-12, 1.0, nrmh)
            Cact_h = (Ah / nrmh) @ (Ah / nrmh).T
            # spectral cluster on act similarity
            act_spec = {}
            for k in (2, 3, 4):
                if nL < k + 1:
                    continue
                # shift cosine to [0,1]
                Spos = np.clip(Cact, 0, None)
                lab, gap, _ = spectral_labels(Spos, k, seed=5)
                act_spec[str(k)] = {
                    "labels": lab.tolist(),
                    "fiedler": gap,
                    "sizes": np.bincount(lab, minlength=k).tolist(),
                }
            act_block = {
                "site": "native_layer_post_norm_hidden",
                "y_projection_dim": Y_SK,
                "fit_cosine": pair_stats(Cact),
                "hold_cosine": pair_stats(Cact_h),
                "adjacent_fit": ov_at_dist(1) and None,
                "spectral": act_spec,
                "note": "cosine of random-256 projection of Y=X@W.T; ranks families, not magnitudes",
            }
            # adjacent / d16 on act cosine
            def c_at_dist(C, d):
                vals = []
                for i, Li in enumerate(layers):
                    for j, Lj in enumerate(layers):
                        if Lj - Li == d:
                            vals.append(float(C[i, j]))
                if not vals:
                    return None
                return dict(n=len(vals), mean=float(np.mean(vals)), min=float(np.min(vals)), max=float(np.max(vals)))

            act_block["fit_cosine_d1"] = c_at_dist(Cact, 1)
            act_block["fit_cosine_d4"] = c_at_dist(Cact, 4)
            act_block["fit_cosine_d16"] = c_at_dist(Cact, 16)
            act_block["hold_cosine_d1"] = c_at_dist(Cact_h, 1)
            act_block["hold_cosine_d16"] = c_at_dist(Cact_h, 16)
            # store hottest act pair
            iu = np.triu_indices(nL, 1)
            hot = int(np.argmax(Cact[iu]))
            act_block["hottest_fit_pair"] = {
                "i": layers[int(iu[0][hot])],
                "j": layers[int(iu[1][hot])],
                "fit_cos": float(Cact[iu][hot]),
                "hold_cos": float(Cact_h[iu[0][hot], iu[1][hot]]),
            }

        # --- row-rms / col-rms / sketch manifolds ---
        # cosine of row-rms profiles
        def profile_cosine(M):
            nn = np.linalg.norm(M, axis=1, keepdims=True)
            nn = np.where(nn < 1e-12, 1.0, nn)
            return (M / nn) @ (M / nn).T

        Crow = profile_cosine(row_rms_mat.astype(np.float64))
        Ccol = profile_cosine(col_rms_mat.astype(np.float64))
        Csk = profile_cosine(sketches)

        # mean row-rms template residual
        Trow = row_rms_mat.mean(axis=0)
        row_tmpl = [rel_delta(row_rms_mat[i], Trow) for i in range(nL)]
        Tcol = col_rms_mat.mean(axis=0)
        col_tmpl = [rel_delta(col_rms_mat[i], Tcol) for i in range(nL)]

        # PCA of sketches / row-rms (layer-as-point manifold)
        man = {
            "sketch_pca": pca_energy(sketches, k=min(16, nL)),
            "row_rms_pca": pca_energy(row_rms_mat, k=min(16, nL)),
            "col_rms_pca": pca_energy(col_rms_mat, k=min(16, nL)),
            "feat_pca": pca_energy(feats, k=min(12, nL)),
        }

        report["classes"][cname] = {
            "suffix": suffix,
            "n_layers": nL,
            "layers": layers,
            "shape": [rows, cols],
            "params_class": int(nL) * int(rows) * int(cols),
            "fingerprint_kmeans": km,
            "subspace_overlap_k32": {
                "pairwise": pair_stats(S_ov),
                "d1": ov_at_dist(1),
                "d4": ov_at_dist(4),
                "d16": ov_at_dist(16),
                "spectral": spec_cl,
                "components": comps,
                "hottest": {
                    "i": layers[int(np.unravel_index(np.argmax(S_ov + np.eye(nL) * -9), S_ov.shape)[0])],
                    "j": layers[int(np.unravel_index(np.argmax(S_ov + np.eye(nL) * -9), S_ov.shape)[1])],
                    "overlap": float(np.max(S_ov + np.eye(nL) * -9)),
                },
            },
            "distribution": {
                "wasserstein_vs_class_mean": {
                    "mean": float(np.mean(w1_ref)),
                    "max": float(np.max(w1_ref)),
                    "min": float(np.min(w1_ref)),
                },
                "wasserstein_pairwise": {
                    "mean": float(np.mean(w1_pairs)),
                    "max": float(np.max(w1_pairs)),
                    "min": float(np.min(w1_pairs)),
                    "p95": float(np.quantile(w1_pairs, 0.95)),
                },
                "lloyd16_local_rel_mse_mean": float(np.mean(loc_rel)),
                "lloyd16_local_rel_mse_max": float(np.max(loc_rel)),
                "lloyd16_shared_rel_mse_mean": float(np.mean(sh_rel)),
                "lloyd16_shared_rel_mse_max": float(np.max(sh_rel)),
                "lloyd16_shared_over_local": float(np.mean(sh_rel) / max(np.mean(loc_rel), 1e-12)),
                "lloyd16_shared_levels": shared_lev.tolist(),
                "lloyd16_cluster3": cluster_level,
                "sample_n_per_tensor": samp_per,
                "note": "levels fit on W/std samples; reconstruction is on those samples, not full tensor",
            },
            "row_col": {
                "row_rms_profile_cosine": pair_stats(Crow),
                "col_rms_profile_cosine": pair_stats(Ccol),
                "row_rms_mean_template_rel": {
                    "mean": float(np.mean(row_tmpl)),
                    "min": float(np.min(row_tmpl)),
                    "max": float(np.max(row_tmpl)),
                },
                "col_rms_mean_template_rel": {
                    "mean": float(np.mean(col_tmpl)),
                    "min": float(np.min(col_tmpl)),
                    "max": float(np.max(col_tmpl)),
                },
                "row_sample_max_cosine_mean": float(row_maxcos.mean()),
                "row_sample_max_cosine_max": float(row_maxcos.max()),
                "row_sample_mean_cosine_mean": float(row_meancos.mean()),
                "row_pairs_ge_0.90_mean": float(row_n_gt90.mean()),
                "row_pairs_ge_0.90_max": int(row_n_gt90.max()),
            },
            "sketch_cosine": pair_stats(Csk),
            "manifold": man,
            "act_maps": act_block,
            "per_layer": per,
            # stash overlap matrix for later family residual selection (compact)
            "_S_ov": S_ov.tolist(),
            "_feat_labels_k3": km.get("3", {}).get("labels"),
            "_spec_labels_k3": spec_cl.get("3", {}).get("labels"),
        }
        dump(report)
        log(
            f"  done {cname} ov_d1={ov_at_dist(1)} sh/loc={report['classes'][cname]['distribution']['lloyd16_shared_over_local']:.3f} "
            f"rowprof_cos={report['classes'][cname]['row_col']['row_rms_profile_cosine'].get('mean')}"
        )

        # free big per-class arrays except what family pass needs
        del Vstack, samples, Yfit_sk, Yhold_sk, sketches
        # keep row_rms_mat? drop
        del row_rms_mat, col_rms_mat, Sstack, hists

    # ------------------------------------------------------------------
    # 2. Within-layer motifs: gate vs up, and down vs (not up.T full)
    # ------------------------------------------------------------------
    log("within-layer gate vs up")
    gu = []
    for L in range(N_LAYERS):
        Wg = load_f32(table[tname(L, "mlp.gate_proj.weight")])
        Wu = load_f32(table[tname(L, "mlp.up_proj.weight")])
        cos = cosine_flat(Wg, Wu)
        rd = rel_delta(Wg, Wu)
        # row-wise cosine
        ng = np.linalg.norm(Wg, axis=1)
        nu = np.linalg.norm(Wu, axis=1)
        rc = np.einsum("ij,ij->i", Wg, Wu) / np.maximum(ng * nu, 1e-12)
        # col-wise
        ngc = np.linalg.norm(Wg, axis=0)
        nuc = np.linalg.norm(Wu, axis=0)
        cc = np.einsum("ij,ij->j", Wg, Wu) / np.maximum(ngc * nuc, 1e-12)
        # scale fit: Wu ≈ a Wg
        a = float(np.vdot(Wg.ravel(), Wu.ravel()) / max(np.vdot(Wg.ravel(), Wg.ravel()), 1e-12))
        rel_scale = rel_delta(Wu, a * Wg)
        gu.append(
            {
                "layer": L,
                "flat_cosine": cos,
                "rel_delta": rd,
                "row_cosine_mean": float(rc.mean()),
                "row_cosine_p95": float(np.quantile(rc, 0.95)),
                "row_cosine_max": float(rc.max()),
                "col_cosine_mean": float(cc.mean()),
                "scale_a": a,
                "rel_after_scale": rel_scale,
            }
        )
        del Wg, Wu
        if (L + 1) % 16 == 0:
            log(f"  gate-up L{L} cos={cos:.5f}")
    report["within_layer"]["gate_vs_up"] = {
        "n": N_LAYERS,
        "flat_cosine_mean": float(np.mean([x["flat_cosine"] for x in gu])),
        "flat_cosine_max": float(np.max([x["flat_cosine"] for x in gu])),
        "row_cosine_mean_mean": float(np.mean([x["row_cosine_mean"] for x in gu])),
        "row_cosine_max_max": float(np.max([x["row_cosine_max"] for x in gu])),
        "rel_after_scale_mean": float(np.mean([x["rel_after_scale"] for x in gu])),
        "per_layer": gu,
    }
    dump(report)

    # GQA q vs k (broadcast: q has 24 heads, k has 4 — compare k to mean of its q-group)
    log("within-layer GQA q vs k (group-mean)")
    qk = []
    for L in GQA:
        Wq = load_f32(table[tname(L, "self_attn.q_proj.weight")])  # 12288 × 5120 = 24 × 512
        Wk = load_f32(table[tname(L, "self_attn.k_proj.weight")])  # 1024 × 5120 = 4 × 256
        # q rows: 24 heads × (256 q + 256 gate)
        q_only = Wq.reshape(24, 512, 5120)[:, :256, :].reshape(24, 256, 5120)
        k_h = Wk.reshape(4, 256, 5120)
        # each k head serves 6 q heads
        pair = []
        for kh in range(4):
            qg = q_only[kh * 6 : (kh + 1) * 6].mean(axis=0)  # 256 × 5120
            pair.append(
                {
                    "k_head": kh,
                    "cosine_vs_group_mean_q": cosine_flat(k_h[kh], qg),
                    "rel_delta_vs_group_mean_q": rel_delta(k_h[kh], qg),
                }
            )
        qk.append({"layer": L, "heads": pair, "mean_cos": float(np.mean([p["cosine_vs_group_mean_q"] for p in pair]))})
        del Wq, Wk
    report["within_layer"]["gqa_k_vs_q_groupmean"] = {
        "mean_cos_over_layers": float(np.mean([x["mean_cos"] for x in qk])),
        "max_cos_over_layers": float(np.max([x["mean_cos"] for x in qk])),
        "per_layer": qk,
    }
    dump(report)

    # ------------------------------------------------------------------
    # 3. Family template + delta (the headline experiment)
    #     For each a-priori family and each discovered cluster, stream mean T
    #     then measure weight residual AND holdout act residual (when honest).
    # ------------------------------------------------------------------
    log("family template+delta")
    # restrict expensive families to the mass classes + the period-16 echo classes
    FAMILY_CLASSES = [
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "lin.in_proj_qkv",
        "gqa.k_proj",
        "gqa.v_proj",
        "gqa.q_proj",
    ]
    # discovered clusters from pass 1
    fam_out = {}
    for cname in FAMILY_CLASSES:
        spec = CLASSES[cname]
        layers = list(spec["layers"])
        suffix = spec["suffix"]
        rows, cols = spec["shape"]
        families = family_label_sets(layers)
        # add kmeans-3 and spectral-3 from pass 1
        lab_km = report["classes"][cname].get("_feat_labels_k3")
        lab_sp = report["classes"][cname].get("_spec_labels_k3")
        if lab_km is not None:
            lab_km = np.asarray(lab_km)
            for c in range(int(lab_km.max()) + 1):
                mem = [layers[i] for i in range(len(layers)) if lab_km[i] == c]
                if len(mem) >= 2:
                    families[f"feat_k3_c{c}"] = mem
        if lab_sp is not None:
            lab_sp = np.asarray(lab_sp)
            for c in range(int(lab_sp.max()) + 1):
                mem = [layers[i] for i in range(len(layers)) if lab_sp[i] == c]
                if len(mem) >= 2:
                    families[f"spec_k3_c{c}"] = mem
        # period-16 4-tuples (the shared-basis echo)
        for r in range(16):
            mem = [L for L in layers if L % 16 == r]
            if len(mem) >= 3:
                families[f"p16_r{r}"] = mem

        # do not evaluate singleton-ish
        # cap number: evaluate the most interesting set
        prefer = [
            "all",
            "first_half",
            "second_half",
            "q0",
            "q3",
            "mod4r3",
            "mod16r3",
            "mod16r11",
            "feat_k3_c0",
            "feat_k3_c1",
            "feat_k3_c2",
            "spec_k3_c0",
            "spec_k3_c1",
            "spec_k3_c2",
        ]
        # plus all p16
        prefer += [k for k in families if k.startswith("p16_")]
        seen = set()
        todo = []
        for k in prefer:
            if k in families and k not in seen and len(families[k]) >= 2:
                todo.append(k)
                seen.add(k)
        # limit p16 to 4 residues to save time (r3 = GQA, r0/1/2 = DN)
        p16s = [k for k in todo if k.startswith("p16_")]
        keep_p16 = [k for k in p16s if k in ("p16_r0", "p16_r1", "p16_r2", "p16_r3", "p16_r11", "p16_r15")]
        todo = [k for k in todo if not k.startswith("p16_")] + keep_p16

        fam_out[cname] = {}
        for fk in todo:
            mem = families[fk]
            # stream mean
            T = np.zeros((rows, cols), dtype=np.float64)
            for L in mem:
                T += load_f32(table[tname(L, suffix)]).astype(np.float64)
            T /= float(len(mem))
            T32 = T.astype(np.float32)
            w_rels = []
            hold_rels = []
            fit_rels = []
            # residual compressibility vs original on first two members
            res_e32 = []
            orig_e32 = []
            res_kurt = []
            orig_kurt = []
            for j, L in enumerate(mem):
                W = load_f32(table[tname(L, suffix)])
                w_rels.append(rel_delta(W, T32))
                if cname in ACT_OK and cols == HIDDEN:
                    # functional residual on holdout native X
                    # ||(W-T) Xh^T|| / ||W Xh^T||
                    Xh = Xhold[L]
                    Yw = Xh @ W.T
                    Yd = Xh @ (W - T32).T
                    nw = float(np.square(Yw, dtype=np.float64).sum())
                    nd = float(np.square(Yd, dtype=np.float64).sum())
                    hold_rels.append(float(np.sqrt(nd / nw)) if nw > 0 else 0.0)
                    Xf = Xfit[L]
                    Yw = Xf @ W.T
                    Yd = Xf @ (W - T32).T
                    nw = float(np.square(Yw, dtype=np.float64).sum())
                    nd = float(np.square(Yd, dtype=np.float64).sum())
                    fit_rels.append(float(np.sqrt(nd / nw)) if nw > 0 else 0.0)
                    del Yw, Yd
                if j < 2:
                    D = (W.astype(np.float64) - T)
                    momW = moments(W)
                    momD = moments(D.astype(np.float32))
                    orig_kurt.append(momW["excess_kurtosis"])
                    res_kurt.append(momD["excess_kurtosis"])
                    # cheap energy@32 via rsvd
                    So, _ = rsvd_right(W, k=32, p=8, seed=1)
                    Sd, _ = rsvd_right(D.astype(np.float32), k=32, p=8, seed=1)
                    fo = fro_sq(W)
                    fd = float((D * D).sum())
                    orig_e32.append(float((So**2).sum() / fo) if fo else 0.0)
                    res_e32.append(float((Sd**2).sum() / fd) if fd else 0.0)
                    del D
                del W
            rec = {
                "members": mem,
                "n": len(mem),
                "weight_rel_delta_mean": float(np.mean(w_rels)),
                "weight_rel_delta_min": float(np.min(w_rels)),
                "weight_rel_delta_max": float(np.max(w_rels)),
                "template_frob_over_member_mean": float(np.sqrt((T * T).sum()) / np.mean([1.0])),  # filled below
                "hold_act_rel_mean": float(np.mean(hold_rels)) if hold_rels else None,
                "hold_act_rel_min": float(np.min(hold_rels)) if hold_rels else None,
                "hold_act_rel_max": float(np.max(hold_rels)) if hold_rels else None,
                "fit_act_rel_mean": float(np.mean(fit_rels)) if fit_rels else None,
                "residual_energy32_mean": float(np.mean(res_e32)) if res_e32 else None,
                "original_energy32_mean": float(np.mean(orig_e32)) if orig_e32 else None,
                "residual_kurt_mean": float(np.mean(res_kurt)) if res_kurt else None,
                "original_kurt_mean": float(np.mean(orig_kurt)) if orig_kurt else None,
                "weight_rels": w_rels,
                "hold_rels": hold_rels,
            }
            # template norm vs typical member: use last member's fro from w_rels definition
            # ||T|| / mean_l ||W_l||  — recompute cheaply from T and mean rel? skip exact
            rec["template_rms"] = float(np.sqrt((T * T).mean()))
            fam_out[cname][fk] = rec
            del T, T32
            log(
                f"  {cname} {fk} n={len(mem)} wrel={rec['weight_rel_delta_mean']:.4f} "
                f"hold={rec['hold_act_rel_mean']}"
            )
        report["family_residuals"][cname] = fam_out[cname]
        dump(report)

    # ------------------------------------------------------------------
    # 4. Procrustes on hottest known pairs (cite shared-basis d=16)
    # ------------------------------------------------------------------
    log("procrustes hottest d16 pairs")
    probes = [
        ("gqa.k_proj", "self_attn.k_proj.weight", 27, 43),
        ("gqa.v_proj", "self_attn.v_proj.weight", 27, 43),
        ("mlp.gate_proj", "mlp.gate_proj.weight", 29, 45),
        ("lin.in_proj_qkv", "linear_attn.in_proj_qkv.weight", 29, 45),
    ]
    proc = []
    for cname, suffix, La, Lb in probes:
        if tname(La, suffix) not in table or tname(Lb, suffix) not in table:
            continue
        Wa = load_f32(table[tname(La, suffix)])
        Wb = load_f32(table[tname(Lb, suffix)])
        # one-sided Orthogonal Procrustes on K-side: Wb ≈ Wa R, R square on cols
        # R = UV^T of Wa^T Wb
        cols = Wa.shape[1]
        # compute Wa^T Wb in tiles if huge
        G = Wa.T @ Wb  # cols × cols  (5120×5120 f32 = 100MB)
        # SVD of G
        U, S, Vt = np.linalg.svd(G.astype(np.float64), full_matrices=False)
        R = U @ Vt
        Wapprox = Wa @ R.astype(np.float32)
        # also scale
        a = float(np.vdot(Wapprox.ravel(), Wb.ravel()) / max(np.vdot(Wapprox.ravel(), Wapprox.ravel()), 1e-12))
        rec = {
            "class": cname,
            "pair": [La, Lb],
            "flat_cosine": cosine_flat(Wa, Wb),
            "rel_delta_raw": rel_delta(Wb, Wa),
            "rel_delta_after_procrustes": rel_delta(Wb, Wapprox),
            "rel_delta_after_procrustes_scale": rel_delta(Wb, (a * Wapprox)),
            "nuclear_mass_frac_top32": float((S[:32] ** 2).sum() / max((S**2).sum(), 1e-12)),
            "nuclear_mass_frac_top256": float((S[:256] ** 2).sum() / max((S**2).sum(), 1e-12)),
            "R_cost_f16_bytes": int(cols * cols * 2),
            "note": "Wb ≈ Wa R, R orthogonal on K; R is full cols×cols — not a cheap code unless structured",
        }
        proc.append(rec)
        log(f"  {cname} L{La}-L{Lb} raw_cos={rec['flat_cosine']:.4f} afterR={rec['rel_delta_after_procrustes']:.4f}")
        del Wa, Wb, G, Wapprox, U, S, Vt, R
    report["procrustes"] = proc
    dump(report)

    # ------------------------------------------------------------------
    # 5. Head behaviour clustering
    # ------------------------------------------------------------------
    log("head behaviour")
    heads = {}

    def head_features(Yh: np.ndarray):
        """Yh: tokens × head_dim  → feature vector."""
        # energy, kurt, token-mean cosine (via gram of unit tokens? or dim-rms)
        rms = float(np.sqrt(np.mean(np.square(Yh, dtype=np.float64))))
        mom = moments(Yh)
        # token-wise energy
        te = np.sqrt(np.mean(np.square(Yh, dtype=np.float64), axis=1))
        # dim-wise energy
        de = np.sqrt(np.mean(np.square(Yh, dtype=np.float64), axis=0))
        # mean pairwise token cosine (sample 64 tokens)
        t = Yh.shape[0]
        take = np.linspace(0, t - 1, num=min(64, t), dtype=int)
        S = Yh[take].astype(np.float64)
        nn = np.linalg.norm(S, axis=1, keepdims=True)
        nn = np.where(nn < 1e-12, 1.0, nn)
        C = (S / nn) @ (S / nn).T
        iu = np.triu_indices(take.size, 1)
        return np.array(
            [
                rms,
                mom["excess_kurtosis"],
                mom["peak_over_rms"],
                float(te.std() / max(te.mean(), 1e-12)),
                float(de.std() / max(de.mean(), 1e-12)),
                float(C[iu].mean()) if iu[0].size else 0.0,
                float(np.quantile(de, 0.99) / max(np.median(de), 1e-12)),
            ],
            dtype=np.float64,
        )

    # GQA q heads across layers (behavior)
    gqa_q_feats = []
    gqa_q_meta = []
    gqa_q_wfeats = []  # weight-space: flatten head matrix rms/kurt
    for L in GQA:
        W = load_f32(table[tname(L, "self_attn.q_proj.weight")])  # 12288 × 5120
        X = Xfit[L]
        Y = X @ W.T  # FIT_N × 12288
        Yh = Y.reshape(FIT_N, 24, 512)  # q+gate
        Ww = W.reshape(24, 512, 5120)
        for h in range(24):
            # behavior on q half and on full
            f_q = head_features(Yh[:, h, :256])
            f_g = head_features(Yh[:, h, 256:])
            gqa_q_feats.append(np.concatenate([f_q, f_g]))
            gqa_q_meta.append({"layer": L, "head": h, "kind": "q+gate"})
            wm = moments(Ww[h])
            gqa_q_wfeats.append([wm["std"], wm["excess_kurtosis"], wm["peak_over_rms"], wm["rms"]])
        del W, Y, Yh, Ww
        log(f"  GQA-q L{L}")
    F = np.stack(gqa_q_feats, 0)
    Fw = np.stack(gqa_q_wfeats, 0)
    gqa_q_block = {"n": int(F.shape[0]), "layers": list(GQA), "n_heads": 24}
    for tag, FF in (("behavior", F), ("weight_moments", Fw)):
        cl = {}
        for k in (3, 4, 6, 8):
            lab, _, inert, Z = kmeans_labels(FF, k, seed=8)
            # do heads cluster by index (same head across layers) or by layer?
            lab = np.asarray(lab)
            head_ids = np.array([m["head"] for m in gqa_q_meta])
            layer_ids = np.array([m["layer"] for m in gqa_q_meta])
            # purity
            def pur(ids):
                acc = []
                for c in range(k):
                    m = lab == c
                    if m.any():
                        _, cnt = np.unique(ids[m], return_counts=True)
                        acc.append(cnt.max() / m.sum())
                return float(np.mean(acc))
            # also: for each head index, entropy of its cluster labels across 16 layers
            head_stab = []
            for h in range(24):
                hh = lab[head_ids == h]
                if hh.size:
                    _, cnt = np.unique(hh, return_counts=True)
                    head_stab.append(float(cnt.max()) / float(hh.size))
            cl[str(k)] = {
                "labels": lab.tolist(),
                "silhouette": silhouette(Z, lab),
                "sizes": np.bincount(lab, minlength=k).tolist(),
                "purity_vs_head_index": pur(head_ids),
                "purity_vs_layer": pur(layer_ids),
                "same_head_majority_frac_mean": float(np.mean(head_stab)),
            }
        gqa_q_block[tag] = cl
    # same-head-across-layers behavior cosine (feature cosine)
    # 24 series of 16 layers
    Fn = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
    same_head = []
    diff_head = []
    for i, mi in enumerate(gqa_q_meta):
        for j, mj in enumerate(gqa_q_meta):
            if j <= i:
                continue
            c = float(np.dot(Fn[i], Fn[j]))
            if mi["head"] == mj["head"]:
                same_head.append(c)
            else:
                diff_head.append(c)
    gqa_q_block["feature_cosine_same_head_mean"] = float(np.mean(same_head))
    gqa_q_block["feature_cosine_diff_head_mean"] = float(np.mean(diff_head))
    gqa_q_block["feature_cosine_same_head_min"] = float(np.min(same_head))
    gqa_q_block["capture_limit"] = "256 tokens, FIT_N=192; ranks families; mixer_x never captured"
    heads["gqa_q"] = gqa_q_block

    # DeltaNet in_proj_qkv heads on a subset of layers
    dn_probe = [0, 4, 16, 20, 32, 36, 48, 62]
    dn_feats = []
    dn_meta = []
    for L in dn_probe:
        W = load_f32(table[tname(L, "linear_attn.in_proj_qkv.weight")])  # 10240 × 5120
        Y = Xfit[L] @ W.T  # T × 10240
        # [Q 16×128 | K 16×128 | V 48×128]
        parts = [
            ("q", Y[:, 0 : 16 * 128].reshape(FIT_N, 16, 128), 16),
            ("k", Y[:, 16 * 128 : 32 * 128].reshape(FIT_N, 16, 128), 16),
            ("v", Y[:, 32 * 128 :].reshape(FIT_N, 48, 128), 48),
        ]
        for kind, arr, nh in parts:
            for h in range(nh):
                dn_feats.append(head_features(arr[:, h, :]))
                dn_meta.append({"layer": L, "head": h, "kind": kind})
        del W, Y
        log(f"  DN-qkv L{L}")
    Fd = np.stack(dn_feats, 0)
    dn_block = {"n": int(Fd.shape[0]), "probe_layers": dn_probe}
    for k in (3, 6, 8):
        lab, _, inert, Z = kmeans_labels(Fd, k, seed=3)
        lab = np.asarray(lab)
        kinds = np.array([m["kind"] for m in dn_meta])
        # purity vs q/k/v
        acc = []
        for c in range(k):
            m = lab == c
            if m.any():
                _, cnt = np.unique(kinds[m], return_counts=True)
                acc.append(cnt.max() / m.sum())
        # same (kind,head) stability across layers
        from collections import defaultdict as _dd

        stab = _dd(list)
        for i, m in enumerate(dn_meta):
            stab[(m["kind"], m["head"])].append(int(lab[i]))
        maj = []
        for _, labs in stab.items():
            _, cnt = np.unique(labs, return_counts=True)
            maj.append(cnt.max() / len(labs))
        dn_block[f"k{k}"] = {
            "sizes": np.bincount(lab, minlength=k).tolist(),
            "silhouette": silhouette(Z, lab),
            "purity_vs_qkv_kind": float(np.mean(acc)),
            "same_slot_majority_frac_mean": float(np.mean(maj)),
            "n_slots": len(maj),
        }
    heads["dn_in_proj_qkv"] = dn_block

    # GQA o-proj / v heads? skip o (X is mixer proxy). Do v heads (honest).
    gqa_v_feats = []
    gqa_v_meta = []
    for L in GQA:
        W = load_f32(table[tname(L, "self_attn.v_proj.weight")])
        Y = Xfit[L] @ W.T
        Yh = Y.reshape(FIT_N, 4, 256)
        for h in range(4):
            gqa_v_feats.append(head_features(Yh[:, h, :]))
            gqa_v_meta.append({"layer": L, "head": h})
        del W, Y
    Fv = np.stack(gqa_v_feats, 0)
    vv = {}
    for k in (2, 3, 4):
        lab, _, inert, Z = kmeans_labels(Fv, k, seed=2)
        lab = np.asarray(lab)
        heads_id = np.array([m["head"] for m in gqa_v_meta])
        stab = []
        for h in range(4):
            hh = lab[heads_id == h]
            _, cnt = np.unique(hh, return_counts=True)
            stab.append(cnt.max() / hh.size)
        vv[str(k)] = {
            "sizes": np.bincount(lab, minlength=k).tolist(),
            "silhouette": silhouette(Z, lab),
            "same_head_majority_frac_mean": float(np.mean(stab)),
        }
    heads["gqa_v"] = {"n": int(Fv.shape[0]), "kmeans": vv}
    report["heads"] = heads
    dump(report)

    # ------------------------------------------------------------------
    # 6. Activation-space layer families (channel energy + hidden PCA)
    # ------------------------------------------------------------------
    log("activation layer families")
    # channel RMS 64 × 5120
    ch_rms = np.sqrt(np.mean(np.square(Xall, dtype=np.float64), axis=1))  # 64 × 5120
    nn = np.linalg.norm(ch_rms, axis=1, keepdims=True)
    Cch = (ch_rms / np.maximum(nn, 1e-12)) @ (ch_rms / np.maximum(nn, 1e-12)).T
    act_km = {}
    for k in (2, 3, 4, 8):
        lab, _, inert, Z = kmeans_labels(ch_rms, k, seed=6)
        act_km[str(k)] = {
            "labels": lab.tolist(),
            "silhouette": silhouette(Z, lab),
            "sizes": np.bincount(lab, minlength=k).tolist(),
        }
    # top channels already known; record persistence
    med = np.median(ch_rms, axis=1, keepdims=True)
    xmed = ch_rms / np.maximum(med, 1e-12)
    top = np.argsort(ch_rms.mean(0))[::-1][:8]
    report["activation_layers"] = {
        "site": cap_meta.get("status"),
        "sha256_self": cap_meta.get("sha256_self"),
        "channel_rms_profile_cosine": pair_stats(Cch),
        "channel_rms_pca": pca_energy(ch_rms, k=8),
        "kmeans": act_km,
        "top8_channels_by_mean_rms": top.tolist(),
        "top8_mean_rms": ch_rms.mean(0)[top].tolist(),
        "L63_over_L0_hidden_rms": float(
            np.sqrt(np.mean(np.square(Xall[63]))) / max(np.sqrt(np.mean(np.square(Xall[0]))), 1e-12)
        ),
        "note": "post-norm hidden, not residual stream; mixer_x uncaptured",
    }
    dump(report)

    # ------------------------------------------------------------------
    # 7. Storage arithmetic + ranking
    # ------------------------------------------------------------------
    log("storage arithmetic")
    # fill from measurements
    arith = []

    def bpw(bytes_):
        return 8.0 * bytes_ / N_SOURCE

    # class param counts
    class_params = {c: report["classes"][c]["params_class"] for c in report["classes"]}

    # A. shared 16-level codebook per class (vs per tensor)
    for cname, blk in report["classes"].items():
        n = blk["n_layers"]
        # per-tensor 16 f16 levels = 16*2*n ; class-shared = 16*2
        saved = 16 * 2 * (n - 1)
        ratio = blk["distribution"]["lloyd16_shared_over_local"]
        arith.append(
            {
                "id": f"shared_levelset_{cname}",
                "structure": "SHARED",
                "what_once": "16 f16 Lloyd levels on W/std",
                "what_per_site": "nothing extra (group scales already in Q4 family)",
                "index_cost_bytes": 0,
                "bytes_saved_vs_per_tensor_levels": saved,
                "complete_bpw_delta": -bpw(saved),
                "fidelity_proxy": f"shared/local sample rel_mse = {ratio:.4f}",
                "functional_risk": "LOW if ratio~1 (same 1-D grid); NOT a generate claim",
                "bits_per_risk": "tiny_save",
                "metal": "hardcoded 16-float LUT in the existing uniform dequant, or one buffer bound once",
            }
        )

    # B. flattened mean template + full delta — always a loss unless delta compresses
    for cname, fams in report["family_residuals"].items():
        params = class_params[cname]
        for fk, rec in fams.items():
            n = rec["n"]
            # T full-size f16 + n * D
            # vs n * W : extra +sizeof(T) unless D is cheaper
            wrel = rec["weight_rel_delta_mean"]
            # if we store T f16 and D at q bits with scale, vs W at q bits:
            # only wins if D needs fewer bits. residual_energy32 vs original is the diagnostic.
            eD = rec["residual_energy32_mean"]
            eW = rec["original_energy32_mean"]
            hold = rec["hold_act_rel_mean"]
            arith.append(
                {
                    "id": f"template_delta_{cname}_{fk}",
                    "structure": "REDUNDANT" if wrel < 0.5 else ("PREDICTABLE" if (hold is not None and hold < 0.5) else "ESSENTIAL"),
                    "n_members": n,
                    "weight_rel": wrel,
                    "hold_act_rel": hold,
                    "residual_e32": eD,
                    "original_e32": eW,
                    "bytes_once_if_T_f16": int(params / n * 2),
                    "complete_bpw_T_f16": bpw(params / n * 2),
                    "note": "T+full D is strictly more bytes than D-alone. Wins only if D quantizes cheaper than W.",
                    "metal": "y = Tx + Dx. T reused only if family members are consecutive in the decode loop; period-16 families are not. Full-size T is a residency tax.",
                }
            )

    # C. row-rms profile share
    for cname, blk in report["classes"].items():
        n = blk["n_layers"]
        rows = blk["shape"][0]
        # one f16 profile + n f16 scales vs n f16 profiles
        once = rows * 2
        per = n * 2  # one extra scale
        saved = n * rows * 2 - (once + per)
        rel = blk["row_col"]["row_rms_mean_template_rel"]["mean"]
        arith.append(
            {
                "id": f"row_rms_template_{cname}",
                "structure": "SHARED" if rel < 0.25 else ("PREDICTABLE" if rel < 0.5 else "UNKNOWN"),
                "what_once": f"row-rms profile f16 ({rows})",
                "what_per_site": "1 f16 gain",
                "bytes_saved_if_replace_profiles": max(saved, 0),
                "complete_bpw_delta": -bpw(max(saved, 0)),
                "fidelity_proxy": f"mean rel_delta of row-rms vs class mean = {rel:.4f}",
                "metal": "row-scale multiply before/after existing GEMV; no new packing of W",
            }
        )

    # D. layernorm mean template
    for kind, blk in report["small_tensors"].items():
        if "mean_template_rel_delta_mean" not in blk:
            continue
        rel = blk["mean_template_rel_delta_mean"]
        # 64 * 5120 * 2 bytes typical
        shape = blk.get("shape") or [64, 5120]
        n, d = int(shape[0]), int(shape[1])
        saved = (n - 1) * d * 2 if rel < 0.15 else 0
        arith.append(
            {
                "id": f"small_{kind}",
                "structure": "SHARED" if rel < 0.2 else "ESSENTIAL",
                "weight_rel": rel,
                "pairwise": blk.get("pairwise_cosine"),
                "bytes_saved_if_one_template": saved,
                "complete_bpw_delta": -bpw(saved),
                "metal": "tiny vector; already f32 in G0. Sharing is a compile-time constant load, not a decode win.",
            }
        )

    # E. head-class templates — only if behavior clusters are stable
    hq = report["heads"]["gqa_q"]["behavior"]
    arith.append(
        {
            "id": "gqa_q_head_classes",
            "structure": "UNKNOWN",
            "behavior_k4_same_head_majority": hq["4"]["same_head_majority_frac_mean"],
            "behavior_k4_purity_vs_head_index": hq["4"]["purity_vs_head_index"],
            "behavior_k4_purity_vs_layer": hq["4"]["purity_vs_layer"],
            "feature_cosine_same_vs_diff": [
                report["heads"]["gqa_q"]["feature_cosine_same_head_mean"],
                report["heads"]["gqa_q"]["feature_cosine_diff_head_mean"],
            ],
            "note": "A head-class codebook wins only if same-index heads across layers share a template in WEIGHT or a cheap correction. Majority-frac near 1 means the slot is a stable role; still need a weight-space template residual.",
            "metal": "grouped GEMV by class; rejected if it expands to Q4 then generic GEMV",
        }
    )

    report["storage_arithmetic"] = arith

    # drop bulky private matrices from the report
    for cname, blk in report["classes"].items():
        blk.pop("_S_ov", None)
        # keep labels
        # per_layer is useful but large — keep moments only? keep as-is for evidence

    report["elapsed_s"] = time.time() - t0
    report["rss_max_gib"] = rss_gb()
    report["finished_unix"] = time.time()
    dump(report)
    log(f"DONE elapsed={report['elapsed_s']:.1f}s rss_max={report['rss_max_gib']:.2f}GiB -> {OUT}")


if __name__ == "__main__":
    main()
