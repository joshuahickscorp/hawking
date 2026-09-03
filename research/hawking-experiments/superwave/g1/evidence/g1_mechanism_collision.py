#!/usr/bin/env python3
"""G1 mechanism-collision measurement. CPU only. No GPU. No generate.

Invariant (own, sibling 162 unlanded):
  I(W, r) = ||W.T @ r||_2 / ||W||_F
  = RMS write-gain of W along unit vector r in the output (hidden) space.
  A left refusal projection W <- (I - r r.T) W drives I -> 0.

Detection: r_hat = smallest left singular direction of the stacked
column-sample of claimed-edited tensors. Compare I on edited vs control.

Mechanisms applied as the live lanes specify them (template + small delta).
"""
from __future__ import annotations

import json
import os
import resource
import struct
import time
from collections import defaultdict

import numpy as np
from numpy.linalg import norm, svd
from scipy.cluster.vq import kmeans2, vq
from scipy.linalg import eigh

BF16 = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
OUT = "/tmp/g1_mechanism_collision.json"
N_COLS = 512
RNG = np.random.default_rng(163)
THREADS = 8
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(THREADS))


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] rss_max={rss_gb():.3f}G {msg}", flush=True)


def load_index():
    with open(os.path.join(BF16, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


_HDR_CACHE: dict[str, tuple[int, dict]] = {}


def shard_header(shard: str):
    if shard in _HDR_CACHE:
        return _HDR_CACHE[shard]
    path = os.path.join(BF16, shard)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    _HDR_CACHE[shard] = (8 + n, header)
    return _HDR_CACHE[shard]


def load_f32(weight_map, name: str) -> np.ndarray:
    shard = weight_map[name]
    base, header = shard_header(shard)
    info = header[name]
    dtype = info["dtype"].upper()
    shape = tuple(info["shape"])
    start, end = info["data_offsets"]
    nbytes = end - start
    path = os.path.join(BF16, shard)
    with open(path, "rb") as f:
        f.seek(base + start)
        raw = f.read(nbytes)
    if dtype in ("BF16", "BFLOAT16"):
        u16 = np.frombuffer(raw, dtype="<u2")
        u32 = u16.astype(np.uint32) << 16
        arr = u32.view(np.float32).reshape(shape).copy()
    elif dtype in ("F32", "FLOAT32"):
        arr = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
    else:
        raise RuntimeError(f"unsupported dtype {dtype} for {name}")
    return arr


def is_gqa(layer: int) -> bool:
    return (layer + 1) % 4 == 0


def class_spec(kind: str):
    # kind -> (name_suffix, layers, edited_pred)
    if kind == "mlp.down_proj":
        layers = list(range(64))
        suffix = "mlp.down_proj.weight"
    elif kind == "linear_attn.out_proj":
        layers = [l for l in range(64) if not is_gqa(l)]
        suffix = "linear_attn.out_proj.weight"
    elif kind == "self_attn.o_proj":
        layers = [l for l in range(64) if is_gqa(l)]
        suffix = "self_attn.o_proj.weight"
    else:
        raise ValueError(kind)
    edited = [l for l in layers if l >= 24]
    control = [l for l in layers if l < 24]
    return suffix, layers, edited, control


def tensor_name(suffix: str, layer: int) -> str:
    return f"language_model.model.layers.{layer}.{suffix}"


def smallest_eigs(G: np.ndarray, k: int = 6):
    # G SPD 5120x5120. smallest k algebraic.
    w, v = eigh(G, subset_by_index=(0, k - 1), driver="evr")
    return w, v  # ascending


def largest_eigs(G: np.ndarray, k: int = 4):
    n = G.shape[0]
    w, v = eigh(G, subset_by_index=(n - k, n - 1), driver="evr")
    return w, v


def I_of(W: np.ndarray, r: np.ndarray) -> float:
    # r unit, W [m,n]
    wr = W.T @ r
    return float(norm(wr) / (norm(W) + 1e-30))


def rayleigh(G: np.ndarray, r: np.ndarray) -> float:
    return float(r @ (G @ r))


def project_left(W: np.ndarray, r: np.ndarray) -> np.ndarray:
    # (I - r r.T) W
    return W - np.outer(r, W.T @ r)


def uniform_q4_g64(W: np.ndarray) -> np.ndarray:
    qmax = 7.0
    flat = W.reshape(-1)
    n = flat.size
    g = 64
    pad = (-n) % g
    if pad:
        work = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    else:
        work = flat.copy()
    grp = work.reshape(-1, g)
    maxabs = np.max(np.abs(grp), axis=1, keepdims=True)
    scale = (maxabs / qmax).astype(np.float32)
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.rint(grp / scale), -8, 7)
    rec = (q * scale).reshape(-1)[:n]
    return rec.reshape(W.shape).astype(np.float32)


def rsvd_recon(A: np.ndarray, k: int, rng: np.random.Generator, p: int = 8) -> np.ndarray:
    if k <= 0:
        return np.zeros_like(A)
    m, n = A.shape
    kk = min(k + p, min(m, n))
    Omega = rng.standard_normal((n, kk)).astype(np.float32)
    Y = A @ Omega
    Qm, _ = np.linalg.qr(Y, mode="reduced")
    B = Qm.T @ A
    Uhat, S, Vt = svd(B, full_matrices=False)
    k_use = min(k, Uhat.shape[1])
    return (Qm @ (Uhat[:, :k_use] * S[:k_use])) @ Vt[:k_use]


def procrustes_factors(A: np.ndarray, B: np.ndarray):
    """Thin factors of orthogonal Q minimizing ||Q A - B||_F.

    A,B are [m, k] with k << m. Q = Ub @ Vtb @ Ua.T has rank <= k
    and is never materialized as m x m.
    """
    Ua, Sa, Vta = svd(A, full_matrices=False)
    Mb = (B @ Vta.T) * Sa
    Ub, _, Vtb = svd(Mb, full_matrices=False)
    return Ub.astype(np.float32), Vtb.astype(np.float32), Ua.astype(np.float32)


def apply_Q(factors, X: np.ndarray) -> np.ndarray:
    Ub, Vtb, Ua = factors
    return Ub @ (Vtb @ (Ua.T @ X))


def apply_QT(factors, X: np.ndarray) -> np.ndarray:
    Ub, Vtb, Ua = factors
    return Ua @ (Vtb.T @ (Ub.T @ X))


def kmeans_1d_labels(X: np.ndarray, k: int, rng: np.random.Generator):
    # X [n, d] float64
    if len(X) < k:
        return np.arange(len(X)), np.zeros(len(X), dtype=int)
    seed = int(rng.integers(0, 2**31 - 1))
    cb, lab = kmeans2(X, k, minit="++", seed=seed, iter=25)
    return cb, lab


def cluster_purity(labels: np.ndarray, edited_mask: np.ndarray) -> dict:
    out = []
    n = len(labels)
    mixed = 0
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        n_e = int(edited_mask[idx].sum())
        n_c = int((~edited_mask[idx]).sum())
        mixed += int(n_e > 0 and n_c > 0)
        out.append({"cluster": int(c), "n": int(len(idx)), "n_edit": n_e, "n_ctrl": n_c})
    return {
        "k": int(len(set(labels.tolist()))),
        "n_mixed_clusters": int(mixed),
        "frac_layers_in_mixed": float(
            sum(cl["n"] for cl in out if cl["n_edit"] and cl["n_ctrl"]) / max(n, 1)
        ),
        "clusters": out,
    }


def retention(I_hat, I_edit, I_ctrl) -> float | None:
    den = I_ctrl - I_edit
    if den <= 1e-12:
        return None
    # 1 = fully preserved (I stays at I_edit), 0 = fully restored to I_ctrl
    return float(np.clip((I_ctrl - I_hat) / den, -0.5, 1.5))


def restoration(I_hat, I_edit, I_ctrl) -> float | None:
    den = I_ctrl - I_edit
    if den <= 1e-12:
        return None
    return float(np.clip((I_hat - I_edit) / den, -0.5, 1.5))


def summarize_I(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()) if a.size else None,
        "min": float(a.min()) if a.size else None,
        "max": float(a.max()) if a.size else None,
        "std": float(a.std()) if a.size else None,
    }


def process_class(kind: str, weight_map: dict, result: dict) -> None:
    suffix, layers, edited, control = class_spec(kind)
    log(f"CLASS {kind} n={len(layers)} edit={len(edited)} ctrl={len(control)}")
    # probe shape
    W0 = load_f32(weight_map, tensor_name(suffix, layers[0]))
    m, n = W0.shape
    log(f"  shape=({m},{n}) fro0={norm(W0):.4f}")
    col_idx = RNG.choice(n, size=min(N_COLS, n), replace=False)
    col_idx.sort()

    sketches = {}
    fro = {}
    mean_all = np.zeros((m, n), dtype=np.float64)
    mean_edit = np.zeros((m, n), dtype=np.float64)
    mean_ctrl = np.zeros((m, n), dtype=np.float64)
    # one random projection of columns for clustering
    s_sign = RNG.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
    s_sign *= 1.0 / np.sqrt(n)
    row_sk = {}

    for li, layer in enumerate(layers):
        name = tensor_name(suffix, layer)
        W = load_f32(weight_map, name)
        assert W.shape == (m, n), (name, W.shape)
        fro[layer] = float(norm(W))
        sk = np.ascontiguousarray(W[:, col_idx])
        sketches[layer] = sk
        mean_all += W
        if layer >= 24:
            mean_edit += W
        else:
            mean_ctrl += W
        row_sk[layer] = (W @ s_sign).astype(np.float32)
        del W
        if (li + 1) % 8 == 0 or li == 0:
            log(f"  load {li+1}/{len(layers)} L{layer} fro={fro[layer]:.3f}")

    mean_all /= len(layers)
    mean_edit /= max(len(edited), 1)
    mean_ctrl /= max(len(control), 1)
    mean_all32 = mean_all.astype(np.float32)
    mean_edit32 = mean_edit.astype(np.float32)
    mean_ctrl32 = mean_ctrl.astype(np.float32)
    del mean_all, mean_edit, mean_ctrl

    M_edit = np.concatenate([sketches[l] for l in edited], axis=1).astype(np.float32)
    M_ctrl = np.concatenate([sketches[l] for l in control], axis=1).astype(np.float32)
    log(f"  stacked edit {M_edit.shape} ctrl {M_ctrl.shape}")
    G_edit = (M_edit @ M_edit.T).astype(np.float64)
    G_ctrl = (M_ctrl @ M_ctrl.T).astype(np.float64)
    del M_edit, M_ctrl

    w_e, v_e = smallest_eigs(G_edit, 6)
    w_c, v_c = smallest_eigs(G_ctrl, 6)
    w_e_hi, _ = largest_eigs(G_edit, 3)
    w_c_hi, _ = largest_eigs(G_ctrl, 3)
    r = np.ascontiguousarray(v_e[:, 0]).astype(np.float64)
    r /= norm(r)
    r_ctrl = np.ascontiguousarray(v_c[:, 0]).astype(np.float64)
    r_ctrl /= norm(r_ctrl)
    r_f32 = r.astype(np.float32)
    r_ctrl_f32 = r_ctrl.astype(np.float32)
    r_rand = RNG.standard_normal(m)
    r_rand /= norm(r_rand)
    r_rand_f32 = r_rand.astype(np.float32)
    cos_r = float(abs(r @ r_ctrl))
    log(f"  eigs_edit_small={w_e.tolist()} eigs_ctrl_small={w_c.tolist()}")
    log(f"  |<r_edit,r_ctrl>|={cos_r:.6f}")

    # second pass: exact I and mechanism reconstructions
    I_edit_r, I_ctrl_r = [], []
    I_edit_rc, I_ctrl_rc = [], []
    I_edit_rand, I_ctrl_rand = [], []
    I_q4_edit, I_q4_ctrl = [], []
    mech_I = defaultdict(list)  # name -> I on edited layers
    mech_rel = defaultdict(list)
    # also I of reconstructions on control, for a few mechs
    boundary_pairs = []

    # tying / share applied to edited layers using ALL-layer mean (the hazard)
    # and using edit-only mean (the safe-if-segregated case)
    T_all = mean_all32
    T_ed = mean_edit32
    fro_T_all = float(norm(T_all))
    fro_T_ed = float(norm(T_ed))
    fro_T_ctrl = float(norm(mean_ctrl32))

    # Procrustes: align sketches to a reference (mid control if any, else first)
    ref_layer = control[len(control) // 2] if control else layers[0]
    A_ref = sketches[ref_layer].astype(np.float64)
    Q_of = {}
    aligned = {}
    t_pr = time.time()
    for layer in layers:
        fac = procrustes_factors(sketches[layer].astype(np.float64), A_ref)
        Q_of[layer] = fac
        aligned[layer] = apply_Q(fac, sketches[layer]).astype(np.float32)
    log(f"  procrustes {len(layers)} thin SVDs in {time.time()-t_pr:.1f}s")

    T_align = np.mean([aligned[l] for l in layers], axis=0)
    T_align_edit = np.mean([aligned[l] for l in edited], axis=0)
    T_align_ctrl = np.mean([aligned[l] for l in control], axis=0) if control else T_align

    # clustering on row sketches
    order = np.array(layers)
    Xcl = np.stack([row_sk[l] for l in layers], axis=0).astype(np.float64)
    Xcl /= np.maximum(norm(Xcl, axis=1, keepdims=True), 1e-12)
    edited_mask = np.array([l >= 24 for l in layers])
    cluster_report = {}
    for k in (2, 3, 4):
        _, lab = kmeans_1d_labels(Xcl, k, RNG)
        cluster_report[str(k)] = cluster_purity(lab, edited_mask)
        # per-cluster mean of FULL tensors would require another pass; use
        # cluster mean of sketches, apply as template in sketch space.
        # For k=2 we also record whether the cut recovered the edit boundary.
        if k == 2:
            # majority-edit cluster vs not
            cluster_report["2"]["recovered_edit_cut"] = bool(
                cluster_report["2"]["n_mixed_clusters"] == 0
            )

    # dictionary: k-means on pooled sketch columns (shared codebook)
    # Cluster in a 128-d projection, then take original-space centroids.
    dict_cols = min(64, sketches[layers[0]].shape[1])
    pool = np.concatenate(
        [sketches[l][:, :dict_cols].T for l in layers], axis=0
    ).astype(np.float64)  # [n_layers*dict_cols, m]
    P = RNG.standard_normal((m, 128)).astype(np.float64)
    P /= norm(P, axis=0, keepdims=True)
    pool_p = pool @ P
    codebook_ks = {}
    t_km = time.time()
    for kk in (32, 128):
        if pool_p.shape[0] < kk:
            continue
        _, lab = kmeans_1d_labels(pool_p, kk, RNG)
        cb = np.zeros((kk, m), dtype=np.float64)
        for c in range(kk):
            sel = lab == c
            if not np.any(sel):
                continue
            cb[c] = pool[sel].mean(axis=0)
        codebook_ks[kk] = cb.astype(np.float32)
    log(f"  dict kmeans on {pool.shape} via 128-d in {time.time()-t_km:.1f}s")

    # adjacent raw / aligned cosine on sketches
    adj = []
    for a, b in zip(layers[:-1], layers[1:]):
        Wa = sketches[a].astype(np.float64)
        Wb = sketches[b].astype(np.float64)
        raw = float(np.vdot(Wa, Wb) / (norm(Wa) * norm(Wb) + 1e-30))
        Aa = aligned[a].astype(np.float64)
        Ab = aligned[b].astype(np.float64)
        al = float(np.vdot(Aa, Ab) / (norm(Aa) * norm(Ab) + 1e-30))
        adj.append(
            {
                "a": int(a),
                "b": int(b),
                "crosses_edit_boundary": bool(a < 24 <= b),
                "raw_cos_sketch": raw,
                "aligned_cos_sketch": al,
            }
        )

    # Gaussian null: one draw, I against r_edit
    Gnull = RNG.standard_normal((m, n)).astype(np.float32)
    Gnull *= 0.02
    I_gauss_r = I_of(Gnull, r_f32)
    I_gauss_rand = I_of(Gnull, r_rand_f32)

    q4_layers = set(
        [layers[0], layers[len(layers) // 2], layers[-1]]
        + ([edited[0], edited[len(edited) // 2], edited[-1]] if edited else [])
        + ([control[0], control[-1]] if control else [])
    )

    # precompute cluster labels k=2 for template assignment
    _, lab2 = kmeans_1d_labels(Xcl, 2, RNG)
    layer_to_lab2 = {int(layers[i]): int(lab2[i]) for i in range(len(layers))}
    # cluster means of full tensors: accumulate in this second pass
    cl_sum = {0: np.zeros((m, n), dtype=np.float64), 1: np.zeros((m, n), dtype=np.float64)}
    cl_n = {0: 0, 1: 0}

    # first walk to fill cluster sums AND compute I on originals
    orig_I = {}
    for li, layer in enumerate(layers):
        W = load_f32(weight_map, tensor_name(suffix, layer))
        Ir = I_of(W, r_f32)
        Irc = I_of(W, r_ctrl_f32)
        Irn = I_of(W, r_rand_f32)
        orig_I[layer] = {"r_edit": Ir, "r_ctrl": Irc, "r_rand": Irn, "fro": fro[layer]}
        if layer >= 24:
            I_edit_r.append(Ir)
            I_edit_rc.append(Irc)
            I_edit_rand.append(Irn)
        else:
            I_ctrl_r.append(Ir)
            I_ctrl_rc.append(Irc)
            I_ctrl_rand.append(Irn)
        lab = layer_to_lab2[layer]
        cl_sum[lab] += W
        cl_n[lab] += 1
        if layer in q4_layers:
            Wq = uniform_q4_g64(W)
            Iq = I_of(Wq, r_f32)
            relq = float(norm(Wq - W) / (norm(W) + 1e-30))
            if layer >= 24:
                I_q4_edit.append(Iq)
            else:
                I_q4_ctrl.append(Iq)
            mech_rel["uniform_q4_g64"].append(relq)
            if layer >= 24:
                mech_I["uniform_q4_g64"].append(Iq)
        del W
        if (li + 1) % 8 == 0 or li == 0:
            log(f"  I-pass {li+1}/{len(layers)} L{layer} I_r={Ir:.6e}")

    cl_mean = {
        c: (cl_sum[c] / max(cl_n[c], 1)).astype(np.float32) for c in (0, 1)
    }
    del cl_sum

    I_edit_mean = float(np.mean(I_edit_r)) if I_edit_r else float("nan")
    I_ctrl_mean = float(np.mean(I_ctrl_r)) if I_ctrl_r else float("nan")
    log(f"  I_edit_mean={I_edit_mean:.6e} I_ctrl_mean={I_ctrl_mean:.6e} ratio={I_ctrl_mean/max(I_edit_mean,1e-30):.3f}")

    # third pass: apply mechanisms to edited layers
    # keep a few full residuals for rank-k / sparse on representative layers
    reps = []
    if edited:
        reps = [edited[0], edited[len(edited) // 2], edited[-1]]
    reps = list(dict.fromkeys(reps))

    for li, layer in enumerate(edited):
        W = load_f32(weight_map, tensor_name(suffix, layer))
        froW = fro[layer] + 1e-30

        # --- family mean (grammar / tying template, all-layer) ---
        rec = T_all
        mech_I["family_mean_all"].append(I_of(rec, r_f32))
        mech_rel["family_mean_all"].append(float(norm(rec - W) / froW))

        rec = T_ed
        mech_I["family_mean_editonly"].append(I_of(rec, r_f32))
        mech_rel["family_mean_editonly"].append(float(norm(rec - W) / froW))

        rec = mean_ctrl32
        mech_I["family_mean_ctrl_applied_to_edit"].append(I_of(rec, r_f32))
        mech_rel["family_mean_ctrl_applied_to_edit"].append(float(norm(rec - W) / froW))

        # --- tying: scalar ---
        # W_hat = s * T_all, s = <W,T>/||T||^2
        denom = float(np.vdot(T_all, T_all)) + 1e-30
        s = float(np.vdot(W, T_all) / denom)
        rec = (s * T_all).astype(np.float32)
        mech_I["tying_scalar_all"].append(I_of(rec, r_f32))
        mech_rel["tying_scalar_all"].append(float(norm(rec - W) / froW))

        denom_e = float(np.vdot(T_ed, T_ed)) + 1e-30
        se = float(np.vdot(W, T_ed) / denom_e)
        rec = (se * T_ed).astype(np.float32)
        mech_I["tying_scalar_editonly"].append(I_of(rec, r_f32))
        mech_rel["tying_scalar_editonly"].append(float(norm(rec - W) / froW))

        # --- tying: per-row (channel) scale ---
        tnorm2 = np.sum(T_all * T_all, axis=1) + 1e-30
        srow = np.sum(W * T_all, axis=1) / tnorm2
        rec = (srow[:, None] * T_all).astype(np.float32)
        mech_I["tying_rowscale_all"].append(I_of(rec, r_f32))
        mech_rel["tying_rowscale_all"].append(float(norm(rec - W) / froW))

        tnorm2e = np.sum(T_ed * T_ed, axis=1) + 1e-30
        srowe = np.sum(W * T_ed, axis=1) / tnorm2e
        rec = (srowe[:, None] * T_ed).astype(np.float32)
        mech_I["tying_rowscale_editonly"].append(I_of(rec, r_f32))
        mech_rel["tying_rowscale_editonly"].append(float(norm(rec - W) / froW))

        # --- generative: shared TOP-k left basis of ALL sketches ---
        # applied as W_hat = U U.T W  (left projection onto shared energy)
        # U computed once outside? we do it before loop — see below via closure
        # stored in result later; here use precomputed U_all / U_edit if present
        if "U_all" in process_class.__dict__:
            pass

        # cluster-mean template (k=2 on all layers)
        rec = cl_mean[layer_to_lab2[layer]]
        mech_I["grammar_k2_cluster_mean"].append(I_of(rec, r_f32))
        mech_rel["grammar_k2_cluster_mean"].append(float(norm(rec - W) / froW))

        # latent-alignment: thin Procrustes Q (rank <= sketch_cols).
        # Sketch reconstruction: W_hat_sk = Q.T @ T_align
        # Full-tensor analogue: map native mean into the ref frame and back.
        fac = Q_of[layer]
        fac_ref = Q_of[ref_layer]
        sk = sketches[layer]
        rec_sk = apply_QT(fac, T_align)
        mech_I["latent_align_share_all_sketch"].append(I_of(rec_sk, r_f32))
        mech_rel["latent_align_share_all_sketch"].append(
            float(norm(rec_sk - sk) / (norm(sk) + 1e-30))
        )
        rec_sk_e = apply_QT(fac, T_align_edit)
        mech_I["latent_align_share_editonly_sketch"].append(I_of(rec_sk_e, r_f32))
        mech_rel["latent_align_share_editonly_sketch"].append(
            float(norm(rec_sk_e - sk) / (norm(sk) + 1e-30))
        )

        rec_full = apply_QT(fac, apply_Q(fac_ref, T_all)).astype(np.float32)
        mech_I["latent_align_unalign_mean_all"].append(I_of(rec_full, r_f32))
        mech_rel["latent_align_unalign_mean_all"].append(float(norm(rec_full - W) / froW))

        rec_full_e = apply_QT(fac, apply_Q(fac_ref, T_ed)).astype(np.float32)
        mech_I["latent_align_unalign_mean_editonly"].append(I_of(rec_full_e, r_f32))
        mech_rel["latent_align_unalign_mean_editonly"].append(
            float(norm(rec_full_e - W) / froW)
        )
        del rec_full, rec_full_e

        # representative-only expensive deltas
        if layer in reps:
            R = (W - T_all).astype(np.float32)
            for k in (1, 2, 4, 8, 16):
                d = rsvd_recon(R, k, RNG, p=8)
                rec = (T_all + d).astype(np.float32)
                mech_I[f"tying_rank{k}_delta_all"].append(I_of(rec, r_f32))
                mech_rel[f"tying_rank{k}_delta_all"].append(float(norm(rec - W) / froW))
            Re = (W - T_ed).astype(np.float32)
            for k in (1, 4, 16):
                d = rsvd_recon(Re, k, RNG, p=8)
                rec = (T_ed + d).astype(np.float32)
                mech_I[f"tying_rank{k}_delta_editonly"].append(I_of(rec, r_f32))
                mech_rel[f"tying_rank{k}_delta_editonly"].append(
                    float(norm(rec - W) / froW)
                )
            # sparse top-p of |R|
            absR = np.abs(R)
            flat = absR.ravel()
            n_el = flat.size
            for p, tag in ((0.001, "0p1pct"), (0.01, "1pct"), (0.05, "5pct")):
                kkeep = max(1, int(p * n_el))
                # partition
                thresh = np.partition(flat, n_el - kkeep)[n_el - kkeep]
                mask = absR >= thresh
                rec = T_all.copy()
                rec[mask] = W[mask]
                mech_I[f"tying_sparse_{tag}_all"].append(I_of(rec, r_f32))
                mech_rel[f"tying_sparse_{tag}_all"].append(float(norm(rec - W) / froW))
            del R, Re, absR, flat

            # energy of residual along r vs residual spectrum
            # ||r.T (W-T)|| / ||W-T||  and compare to top singular of residual
            Rt = (W - T_all).astype(np.float32)
            I_res = I_of(Rt, r_f32)
            # a few singular values of residual via rsvd
            # reuse rsvd factors
            m_, n_ = Rt.shape
            kk = 16
            Omega = RNG.standard_normal((n_, kk + 8)).astype(np.float32)
            Y = Rt @ Omega
            Qm, _ = np.linalg.qr(Y, mode="reduced")
            B = Qm.T @ Rt
            Sres = svd(B, compute_uv=False)
            mech_I["_residual_along_r_over_res"].append(I_res)
            result.setdefault("_residual_spectra", {}).setdefault(kind, []).append(
                {
                    "layer": int(layer),
                    "I_residual_along_r": I_res,
                    "rel_residual": float(norm(Rt) / froW),
                    "sv_top16": [float(x) for x in Sres[:16]],
                }
            )
            del Rt, Omega, Y, Qm, B

        # shared dictionary reconstruct of sketch columns
        for kk, cb in codebook_ks.items():
            # cb [kk, m], columns of sketch assigned to nearest
            sk = sketches[layer][:, :dict_cols].T.astype(np.float64)  # [c, m]
            codes, _ = vq(sk, cb.astype(np.float64))
            rec_sk = cb[codes].T.astype(np.float32)  # [m, c]
            mech_I[f"shared_dict_k{kk}_sketch"].append(I_of(rec_sk, r_f32))
            mech_rel[f"shared_dict_k{kk}_sketch"].append(
                float(norm(rec_sk - sketches[layer][:, :dict_cols]) / (norm(sk) + 1e-30))
            )

        # adjacent-delta from previous layer (if previous exists)
        prevs = [l for l in layers if l < layer]
        if prevs:
            prev = prevs[-1]
            # W_hat = W_prev  (delta discarded) — need to load prev
            Wp = load_f32(weight_map, tensor_name(suffix, prev))
            mech_I["adjacent_delta_discarded"].append(I_of(Wp, r_f32))
            mech_rel["adjacent_delta_discarded"].append(float(norm(Wp - W) / froW))
            # rank-4 delta from prev
            if layer in reps:
                d = rsvd_recon((W - Wp).astype(np.float32), 4, RNG, p=8)
                rec = (Wp + d).astype(np.float32)
                mech_I["adjacent_rank4_delta"].append(I_of(rec, r_f32))
                mech_rel["adjacent_rank4_delta"].append(float(norm(rec - W) / froW))
            del Wp

        # mandatory correction demo: re-project family_mean_all
        rec = project_left(T_all, r_f32)
        mech_I["correction_reproject_family_mean_all"].append(I_of(rec, r_f32))
        mech_rel["correction_reproject_family_mean_all"].append(
            float(norm(rec - W) / froW)
        )

        del W
        if (li + 1) % 8 == 0 or li == 0:
            log(f"  mech-pass {li+1}/{len(edited)} L{layer}")

    # shared left top-k basis applied in a short extra pass (U from G_all)
    G_all = G_edit + G_ctrl
    _, Vhi = largest_eigs(G_all, 32)
    U32 = np.ascontiguousarray(Vhi[:, ::-1]).astype(np.float32)  # largest first
    _, Vhi_e = largest_eigs(G_edit, 32)
    U32e = np.ascontiguousarray(Vhi_e[:, ::-1]).astype(np.float32)

    for kleft in (8, 32):
        U = U32[:, :kleft]
        Ue = U32e[:, :kleft]
        vals = []
        vals_e = []
        rels = []
        for layer in reps:
            W = load_f32(weight_map, tensor_name(suffix, layer))
            rec = (U @ (U.T @ W)).astype(np.float32)
            vals.append(I_of(rec, r_f32))
            rels.append(float(norm(rec - W) / (fro[layer] + 1e-30)))
            rec_e = (Ue @ (Ue.T @ W)).astype(np.float32)
            vals_e.append(I_of(rec_e, r_f32))
            del W
        mech_I[f"shared_topk_left_k{kleft}_all"] = vals
        mech_rel[f"shared_topk_left_k{kleft}_all"] = rels
        mech_I[f"shared_topk_left_k{kleft}_editonly"] = vals_e

    # ---- controlled twin (synthetic similar layers) ----
    # Use a mid edited tensor as the parent body. Independent r_syn.
    twin = {}
    parent_layer = edited[len(edited) // 2]
    Wpar = load_f32(weight_map, tensor_name(suffix, parent_layer))
    r_syn = RNG.standard_normal(m).astype(np.float32)
    r_syn /= norm(r_syn)
    I_par = I_of(Wpar, r_syn)
    Wabl = project_left(Wpar, r_syn).astype(np.float32)
    # optional frobenius norm preserve
    Wabl *= (norm(Wpar) / (norm(Wabl) + 1e-30))
    I_abl = I_of(Wabl, r_syn)
    # twin mean of {Wpar, Wabl} — the silent-share case (identical but for the edit)
    T_twin = 0.5 * (Wpar + Wabl)
    I_twin_mean_on_abl = I_of(T_twin, r_syn)
    # scalar / rowscale / rank1 / sparse toward T_twin, evaluated on the ablated member
    denom = float(np.vdot(T_twin, T_twin)) + 1e-30
    s = float(np.vdot(Wabl, T_twin) / denom)
    rec_s = (s * T_twin).astype(np.float32)
    tnorm2 = np.sum(T_twin * T_twin, axis=1) + 1e-30
    srow = np.sum(Wabl * T_twin, axis=1) / tnorm2
    rec_row = (srow[:, None] * T_twin).astype(np.float32)
    R = (Wabl - T_twin).astype(np.float32)
    rec_r1 = (T_twin + rsvd_recon(R, 1, RNG, p=8)).astype(np.float32)
    rec_r4 = (T_twin + rsvd_recon(R, 4, RNG, p=8)).astype(np.float32)
    # sparse 1%
    absR = np.abs(R)
    flat = absR.ravel()
    kkeep = max(1, int(0.01 * flat.size))
    thresh = np.partition(flat, flat.size - kkeep)[flat.size - kkeep]
    rec_sp = T_twin.copy()
    rec_sp[absR >= thresh] = Wabl[absR >= thresh]
    rec_corr = project_left(T_twin, r_syn)

    # noisy cousins: W_b = normalize(Wpar + eps G) then ablate only B
    noisy = {}
    Gnoise = RNG.standard_normal(Wpar.shape).astype(np.float32)
    Gnoise *= fro[parent_layer] / (norm(Gnoise) + 1e-30)
    for eps in (0.0, 0.05, 0.2, 1.0):
        Wb = (Wpar + eps * Gnoise).astype(np.float32)
        Wb_abl = project_left(Wb, r_syn).astype(np.float32)
        Wb_abl *= norm(Wb) / (norm(Wb_abl) + 1e-30)
        Tmix = 0.5 * (Wpar + Wb_abl)  # share unablated A with ablated B
        noisy[str(eps)] = {
            "eps": float(eps),
            "rel_delta_A_Babl": float(norm(Wpar - Wb_abl) / (norm(Wpar) + 1e-30)),
            "I_A": I_of(Wpar, r_syn),
            "I_Babl": I_of(Wb_abl, r_syn),
            "I_mean_on_Babl": I_of(Tmix, r_syn),
            "restoration_mean": restoration(
                I_of(Tmix, r_syn), I_of(Wb_abl, r_syn), I_of(Wpar, r_syn)
            ),
            "retention_mean": retention(
                I_of(Tmix, r_syn), I_of(Wb_abl, r_syn), I_of(Wpar, r_syn)
            ),
        }

    # inject real r_hat into a CONTROL tensor and share with an uninjected control
    inject = {}
    if len(control) >= 2:
        Lc, Ld = control[0], control[-1]
        Wc = load_f32(weight_map, tensor_name(suffix, Lc))
        Wd = load_f32(weight_map, tensor_name(suffix, Ld))
        Wd_inj = project_left(Wd, r_f32).astype(np.float32)
        Wd_inj *= norm(Wd) / (norm(Wd_inj) + 1e-30)
        Tmix = 0.5 * (Wc + Wd_inj)
        inject = {
            "ctrl_layers": [int(Lc), int(Ld)],
            "I_uninjected": I_of(Wc, r_f32),
            "I_injected": I_of(Wd_inj, r_f32),
            "I_mean_on_injected": I_of(Tmix, r_f32),
            "restoration_mean": restoration(
                I_of(Tmix, r_f32), I_of(Wd_inj, r_f32), I_of(Wc, r_f32)
            ),
            "note": "share uninjected control with injected control; r is recovered r_edit",
        }
        del Wc, Wd, Wd_inj, Tmix

    twin = {
        "parent_layer": int(parent_layer),
        "I_parent_r_syn": I_par,
        "I_ablated_r_syn": I_abl,
        "I_twin_mean_on_ablated": I_twin_mean_on_abl,
        "restoration_mean": restoration(I_twin_mean_on_abl, I_abl, I_par),
        "retention_mean": retention(I_twin_mean_on_abl, I_abl, I_par),
        "I_tying_scalar": I_of(rec_s, r_syn),
        "I_tying_rowscale": I_of(rec_row, r_syn),
        "I_tying_rank1": I_of(rec_r1, r_syn),
        "I_tying_rank4": I_of(rec_r4, r_syn),
        "I_tying_sparse_1pct": I_of(rec_sp, r_syn),
        "I_correction_reproject": I_of(rec_corr, r_syn),
        "rel_ablated_vs_parent": float(norm(Wabl - Wpar) / (norm(Wpar) + 1e-30)),
        "noisy_cousins": noisy,
        "inject_on_controls": inject,
    }
    del Wpar, Wabl, T_twin, R, absR, rec_s, rec_row, rec_r1, rec_r4, rec_sp, rec_corr, Gnoise

    # pack class result
    def pack_mech(name: str) -> dict:
        Is = mech_I.get(name, [])
        Rs = mech_rel.get(name, [])
        I_hat = float(np.mean(Is)) if Is else None
        return {
            "n": len(Is),
            "I_hat_mean": I_hat,
            "I_hat": summarize_I(Is) if Is else None,
            "rel_recon_mean": float(np.mean(Rs)) if Rs else None,
            "rel_recon": summarize_I(Rs) if Rs else None,
            "retention": retention(I_hat, I_edit_mean, I_ctrl_mean) if I_hat is not None else None,
            "restoration": restoration(I_hat, I_edit_mean, I_ctrl_mean) if I_hat is not None else None,
        }

    mech_names = sorted(set(list(mech_I.keys()) + list(mech_rel.keys())))
    mechanisms = {name: pack_mech(name) for name in mech_names if not name.startswith("_")}

    class_out = {
        "kind": kind,
        "shape": [int(m), int(n)],
        "n_layers": len(layers),
        "n_edited": len(edited),
        "n_control": len(control),
        "edited_layers": edited,
        "control_layers": control,
        "sketch_cols": int(min(N_COLS, n)),
        "detection": {
            "G_edit_eigs_small": [float(x) for x in w_e],
            "G_ctrl_eigs_small": [float(x) for x in w_c],
            "G_edit_eigs_large": [float(x) for x in w_e_hi],
            "G_ctrl_eigs_large": [float(x) for x in w_c_hi],
            "G_edit_trace": float(np.trace(G_edit)),
            "G_ctrl_trace": float(np.trace(G_ctrl)),
            "lambda_min_over_trace_edit": float(w_e[0] / (np.trace(G_edit) + 1e-30)),
            "lambda_min_over_trace_ctrl": float(w_c[0] / (np.trace(G_ctrl) + 1e-30)),
            "cos_r_edit_r_ctrl": cos_r,
            "I_edit_r_edit": summarize_I(I_edit_r),
            "I_ctrl_r_edit": summarize_I(I_ctrl_r),
            "I_edit_r_ctrl": summarize_I(I_edit_rc),
            "I_ctrl_r_ctrl": summarize_I(I_ctrl_rc),
            "I_edit_r_rand": summarize_I(I_edit_rand),
            "I_ctrl_r_rand": summarize_I(I_ctrl_rand),
            "I_gaussian_r_edit": I_gauss_r,
            "I_gaussian_r_rand": I_gauss_rand,
            "theory_isotropic_I": float(1.0 / np.sqrt(m)),
            "effect_I_ctrl_over_I_edit": float(I_ctrl_mean / max(I_edit_mean, 1e-30)),
            "mean_T_all_fro_over_mean_W": float(fro_T_all / (np.mean(list(fro.values())) + 1e-30)),
            "mean_T_edit_fro_over_mean_W": float(fro_T_ed / (np.mean(list(fro.values())) + 1e-30)),
            "mean_T_ctrl_fro_over_mean_W": float(fro_T_ctrl / (np.mean(list(fro.values())) + 1e-30)),
        },
        "per_layer_I": {str(l): orig_I[l] for l in layers},
        "adjacent_sketches": adj,
        "clustering": cluster_report,
        "mechanisms": mechanisms,
        "controlled_twin": twin,
        "r_edit_head16": [float(x) for x in r[:16]],
        "r_edit_norm": float(norm(r)),
    }
    result["classes"][kind] = class_out
    # free big
    del G_edit, G_ctrl, G_all, sketches, aligned, Q_of, mean_all32, mean_edit32, mean_ctrl32
    del cl_mean, T_all, T_ed, U32, U32e
    log(f"  DONE class {kind}")


def main():
    t0 = time.time()
    log("start")
    weight_map = load_index()
    # sidecar claim (vendor metadata, not evidence)
    with open(os.path.join(BF16, "abliteration-manifest.json")) as f:
        sidecar = json.load(f)
    result = {
        "schema": "hawking.g1.mechanism_collision.v1",
        "sibling_162_landed": False,
        "invariant": {
            "name": "left_write_gain",
            "formula": "I(W,r)=||W.T r||_2 / ||W||_F",
            "source": "own; sibling 162-refusal-direction-recovery NOT landed",
            "edit_present_when": "I_edit << I_ctrl and << 1/sqrt(hidden) on recovered r",
            "retention": "(I_ctrl - I_hat)/(I_ctrl - I_edit)",
            "restoration": "(I_hat - I_edit)/(I_ctrl - I_edit)",
        },
        "sidecar_claim": sidecar,
        "classes": {},
        "threads": THREADS,
        "sketch_cols_requested": N_COLS,
    }
    for kind in ("mlp.down_proj", "linear_attn.out_proj", "self_attn.o_proj"):
        process_class(kind, weight_map, result)
    result["elapsed_s"] = float(time.time() - t0)
    result["rss_max_gb"] = float(rss_gb())
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    log(f"WROTE {OUT} elapsed={result['elapsed_s']:.1f}s rss_max={result['rss_max_gb']:.3f}G")


if __name__ == "__main__":
    main()
