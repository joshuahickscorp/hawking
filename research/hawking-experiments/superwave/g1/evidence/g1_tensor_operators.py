#!/usr/bin/env python3
"""G-TENSOR: structured tensor operators on real Qwen3.8 GEMV tensors.

Fits Tucker / TT / TT-matrix (Kronecker-sum) / tensor-ring / CP / BTD /
Kronecker+low-rank mixtures. Scores the IMPLIED LINEAR MAP with the doctor
adequacy gate. Does not require reconstructing a dense W at consume time;
reconstruction is used only as an equivalent scoring device (the map is linear).

No GPU. No network. Peak RSS target < 15 GB.
"""
from __future__ import annotations

import json, os, struct, sys, time, resource, traceback
import numpy as np

SRC = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
CAP = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
N_SOURCE = 26_895_998_464
GEMV_ELEMS = 26_893_352_960
TINY_ELEMS = 2_645_504
HEADER = 40
AXIS_MARGIN = {"observed": 0.02, "probed": 0.02, "worst_unit": 0.10}
OUT = "/tmp/g1_tensor_operators.json"

# M3 Ultra 60-core (this box). Peak FLOP is ESTIMATED; BW is MEASURED.
BW_GEMV_GB_S = 639.25          # MEASURED geo_tpr64 sealed weight_addressing
BW_DATASHEET_GB_S = 819.0      # published, not a roof
# Apple 80-core M3 Ultra marketed 60.5 TFLOPS; this box is 60-core. Linear scale.
PEAK_FP16_TFLOP_S = 60.5 * (60 / 80)   # ESTIMATED 45.375
PEAK_FP32_TFLOP_S = PEAK_FP16_TFLOP_S / 2.0  # ESTIMATED, FMA half if the 60.5 is FP16

rng_global = np.random.default_rng(0)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS bytes


def now():
    return time.time()


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- I/O

def load_tensor(name, root=SRC):
    idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))
    shard = idx["weight_map"][name]
    with open(os.path.join(root, shard), "rb") as f:
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
    return arr.reshape(meta["shape"]).astype(np.float32, copy=False)


def load_X(layer):
    p = os.path.join(CAP, "hidden", f"L{layer:02d}.f32")
    return np.fromfile(p, dtype=np.float32).reshape(256, 5120)


# ---------------------------------------------------------------- scoring (doctor gate, applied to Y not Wh)

def _rowcos(A, B):
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-30
    return float(np.mean(num / den))


def _worst_unit(A, B):
    num = (A * B).sum(0)
    na = np.linalg.norm(A, axis=0)
    nb = np.linalg.norm(B, axis=0)
    live = na > 1e-20
    cos = np.zeros_like(num)
    denom = na * nb + 1e-30
    cos[live] = num[live] / denom[live]
    return float(cos[live].min()) if live.any() else 1.0


def _probe(d_in, n=256, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((n, d_in)).astype(np.float32)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return P


def axes_from_Y(Yx, Yxh, Yp, Yph):
    return {
        "observed": _rowcos(Yx, Yxh) if Yx is not None else None,
        "probed": _rowcos(Yp, Yph),
        "worst_unit": (
            min(_worst_unit(Yx, Yxh), _worst_unit(Yp, Yph))
            if Yx is not None else _worst_unit(Yp, Yph)
        ),
    }


def gate_from_axes(a, ref):
    keys = [k for k in ("observed", "probed", "worst_unit") if a.get(k) is not None]
    deficits = {k: a[k] - (ref[k] - AXIS_MARGIN[k]) for k in keys}
    worst = min(deficits, key=deficits.get)
    return {
        **a,
        "deficit": deficits,
        "gate": deficits[worst],
        "worst_axis": worst,
        "healthy": deficits[worst] >= 0.0,
        "mode": "relative" + ("_probe_only" if a.get("observed") is None else ""),
    }


def c_uniform_fast(W, bits=4, group=128):
    Wh = np.array(W, dtype=np.float32, copy=True)
    lim = (1 << (bits - 1)) - 1
    m, d = Wh.shape
    n_g = d // group
    if n_g == 0:
        return Wh
    blk = Wh[:, : n_g * group].reshape(m, n_g, group)
    amax = np.max(np.abs(blk), axis=2, keepdims=True) + 1e-30
    step = amax / lim
    q = np.clip(np.round(blk / step), -lim, lim) * step
    Wh[:, : n_g * group] = q.reshape(m, n_g * group)
    return Wh


# ---------------------------------------------------------------- linear algebra helpers

def eigh_energy(G):
    """Descending eigenvalues of a PSD Gram. energy_frac[k] = sum_{i<k} λ_i / sum λ."""
    w = np.linalg.eigvalsh(G.astype(np.float64))
    w = np.sort(w)[::-1]
    w = np.maximum(w, 0.0)
    tot = float(w.sum()) + 1e-30
    c = np.cumsum(w) / tot
    return w.astype(np.float64), c


def energy_at(c, ranks):
    n = len(c)
    return {int(r): float(c[min(r, n) - 1]) for r in ranks if r > 0}


def svd_trunc(M, r):
    r = max(1, min(r, M.shape[0], M.shape[1]))
    if min(M.shape) <= 2048 or r >= 0.4 * min(M.shape):
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        return U[:, :r].astype(np.float32), S[:r].astype(np.float32), Vh[:r].astype(np.float32)
    return rsvd(M, r)


def rsvd(A, k, p=12, q=1, seed=0):
    k = max(1, min(k, A.shape[0], A.shape[1]))
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((A.shape[1], k + p)).astype(np.float32)
    Y = A @ G
    for _ in range(q):
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y, mode="reduced")
    B = Q.T @ A
    Uhat, S, Vh = np.linalg.svd(B, full_matrices=False)
    U = Q @ Uhat
    return U[:, :k].astype(np.float32), S[:k].astype(np.float32), Vh[:k].astype(np.float32)


def f16(x):
    return x.astype(np.float16).astype(np.float32)


# ---------------------------------------------------------------- reshape / spectra

def matching_kron_pairs(m, n, max_inner=1024):
    """Pairs (I0,I1) x (J0,J1) with I1==J1 (Kronecker-inner match) dividing m,n."""
    out = []
    for inner in (1024, 512, 256, 128, 64, 32, 16, 8, 4):
        if inner > max_inner:
            continue
        if m % inner == 0 and n % inner == 0:
            out.append(((m // inner, inner), (n // inner, inner)))
    return out


def mode_grams_4(W, I0, I1, J0, J1):
    Wr = W.reshape(I0, I1, -1)
    G0 = np.tensordot(Wr, Wr, axes=([1, 2], [1, 2]))
    G1 = np.tensordot(Wr, Wr, axes=([0, 2], [0, 2]))
    Wc = W.reshape(-1, J0, J1)
    G2 = np.tensordot(Wc, Wc, axes=([0, 2], [0, 2]))
    G3 = np.tensordot(Wc, Wc, axes=([0, 1], [0, 1]))
    return G0, G1, G2, G3


def kron_rearrange(W, I0, I1, J0, J1):
    return np.ascontiguousarray(
        W.reshape(I0, I1, J0, J1).transpose(0, 2, 1, 3).reshape(I0 * J0, I1 * J1)
    )


def gram_spectrum_rect(A):
    """Full descending energy of A via the smaller Gram."""
    a, b = A.shape
    if a <= b:
        G = A @ A.T
    else:
        G = A.T @ A
    return eigh_energy(G)


# ---------------------------------------------------------------- Tucker HOSVD

def hosvd_factors(W, I0, I1, J0, J1, ranks):
    G0, G1, G2, G3 = mode_grams_4(W, I0, I1, J0, J1)
    Us = []
    evs = []
    for G, r in zip((G0, G1, G2, G3), ranks):
        w, v = np.linalg.eigh(G.astype(np.float64))
        idx = np.argsort(w)[::-1]
        r = max(1, min(int(r), v.shape[1]))
        Us.append(v[:, idx[:r]].astype(np.float32))
        evs.append(np.maximum(w[idx], 0.0))
    # core = T ×0 U0.T ×1 U1.T ×2 U2.T ×3 U3.T
    t = W.reshape(I0, I1, J0, J1)
    t = np.tensordot(Us[0].T, t, axes=(1, 0))
    t = np.tensordot(Us[1].T, t, axes=(1, 1)).transpose(1, 0, 2, 3)
    t = np.tensordot(Us[2].T, t, axes=(1, 2)).transpose(1, 2, 0, 3)
    t = np.tensordot(Us[3].T, t, axes=(1, 3)).transpose(1, 2, 3, 0)
    return Us, np.ascontiguousarray(t.astype(np.float32)), evs


def tucker_apply(X, U0, U1, U2, U3, G):
    B = X.shape[0]
    I2 = U2.shape[0]
    I3 = U3.shape[0]
    Xm = X.reshape(B, I2, I3)
    Xc = np.einsum("bij,ir,js->brs", Xm, U2, U3, optimize=True)
    Yc = np.einsum("brs,pqrs->bpq", Xc, G, optimize=True)
    Ym = np.einsum("bpq,ip,jq->bij", Yc, U0, U1, optimize=True)
    return Ym.reshape(B, U0.shape[0] * U1.shape[0])


def tucker_rel_l2_from_core(G, wnorm):
    # orthonormal HOSVD: ||T-That||^2 = ||T||^2 - ||G||^2
    g2 = float(np.square(G, dtype=np.float64).sum())
    w2 = wnorm * wnorm
    return float(np.sqrt(max(0.0, 1.0 - g2 / w2))), g2


def tucker_flops_v(I0, I1, J0, J1, R):
    r0, r1, r2, r3 = [int(x) for x in R]
    fl = 2 * r2 * J0 * J1 + 2 * r2 * J1 * r3
    fl += 2 * r0 * r1 * r2 * r3
    fl += 2 * I0 * r0 * r1
    fl += 2 * I0 * r1 * I1
    return int(fl)


def tucker_bytes(I0, I1, J0, J1, R, db=2):
    r0, r1, r2, r3 = [int(x) for x in R]
    elems = I0 * r0 + I1 * r1 + J0 * r2 + J1 * r3 + r0 * r1 * r2 * r3
    meta = 4 * 8
    return HEADER + meta + db * elems, elems


# ---------------------------------------------------------------- TT-SVD (4-way tensor of the matrix)

def tt_svd_4(W, I0, I1, J0, J1, ranks):
    """ranks = (r1, r2, r3) bond ranks."""
    r1, r2, r3 = [int(x) for x in ranks]
    T = W.reshape(I0, I1, J0, J1)
    M = T.reshape(I0, -1)
    U, S, Vh = svd_trunc(M, r1)
    c0 = U.reshape(I0, r1)  # (I0, r1)
    rem = (S[:, None] * Vh).reshape(r1 * I1, J0 * J1)
    U, S, Vh = svd_trunc(rem, r2)
    c1 = U.reshape(r1, I1, r2)
    rem = (S[:, None] * Vh).reshape(r2 * J0, J1)
    U, S, Vh = svd_trunc(rem, r3)
    c2 = U.reshape(r2, J0, r3)
    c3 = (S[:, None] * Vh).reshape(r3, J1)
    return c0, c1, c2, c3


def tt_apply(X, c0, c1, c2, c3, I0, I1, J0, J1):
    B = X.shape[0]
    Xm = X.reshape(B, J0, J1)
    # A[b,j0,r3] = Xm[b,j0,j1] * c3[r3,j1]
    A = np.einsum("bjl,rl->bjr", Xm, c3, optimize=True)
    # B[b,r2,r3] = c2[r2,j0,r3] * A[b,j0,r3]
    Mid = np.einsum("pjr,bjr->bpr", c2, A, optimize=True)
    # C[b,r1,i1] = c1[r1,i1,r2] * Mid[b,r2,r3] ... Mid has r3 as well
    # c2 contracted r3 already in Mid? Mid is (B,r2,r3). Need to contract r3.
    # c3 is (r3,J1) with no trailing 1, last core has no outgoing rank after c3.
    # After A and c2: we still have r3. c3 is the last core so Mid should be (B,r2):
    # y path: sum_{j0,j1,r1,r2,r3} c0[i0,r1] c1[r1,i1,r2] c2[r2,j0,r3] c3[r3,j1] x[j0,j1]
    # A[b,j0,r3] = sum_j1 x c3
    # H[b,r2] = sum_{j0,r3} c2[r2,j0,r3] A[b,j0,r3]
    H = np.einsum("pjr,bjr->bp", c2, A, optimize=True)
    C = np.einsum("qip,bp->bqi", c1, H, optimize=True)  # (B,r1,I1)
    Y = np.einsum("oq,bqi->boi", c0, C, optimize=True)  # (B,I0,I1)
    return Y.reshape(B, I0 * I1)


def tt_bytes(I0, I1, J0, J1, ranks, db=2):
    r1, r2, r3 = [int(x) for x in ranks]
    elems = I0 * r1 + r1 * I1 * r2 + r2 * J0 * r3 + r3 * J1
    return HEADER + 4 * 8 + db * elems, elems


def tt_flops(I0, I1, J0, J1, ranks):
    r1, r2, r3 = [int(x) for x in ranks]
    fl = 2 * J0 * J1 * r3
    fl += 2 * r2 * J0 * r3
    fl += 2 * r1 * I1 * r2
    fl += 2 * I0 * r1 * I1
    return int(fl)


# ---------------------------------------------------------------- Kronecker-sum / TTM-2

def kron_fit(W, I0, I1, J0, J1, rank):
    R = kron_rearrange(W, I0, I1, J0, J1)
    U, S, Vh = svd_trunc(R, rank)
    A = (U * S).T.reshape(rank, I0, J0)  # (r, I0, J0)
    B = Vh.reshape(rank, I1, J1)
    return A.astype(np.float32), B.astype(np.float32)


def kron_apply(X, A, B):
    """y += (A_k ⊗ B_k) x  via y_mat += A_k @ x_mat @ B_k.T, looped over k.

    A single 6-index einsum can materialise A⊗B×batch and hang. Loop stays O(r) GEMMs.
    """
    batch = X.shape[0]
    r, I0, J0 = A.shape
    _, I1, J1 = B.shape
    Xm = X.reshape(batch, J0, J1)
    Y = np.zeros((batch, I0, I1), dtype=np.float32)
    for k in range(r):
        # (batch, J0, J1) @ (J1, I1) -> (batch, J0, I1)
        tmp = Xm @ B[k].T
        # (I0, J0) @ (batch, J0, I1) -> (batch, I0, I1)
        Y += np.einsum("ij,bjl->bil", A[k], tmp)
    return Y.reshape(batch, I0 * I1)


def kron_bytes(I0, I1, J0, J1, r, db=2):
    elems = int(r) * (I0 * J0 + I1 * J1)
    return HEADER + 4 * 6 + db * elems, elems


def kron_flops(I0, I1, J0, J1, r):
    # per term: A @ Xm @ B.T  via einsum ~ 2*r*I0*J0*J1 + 2*r*I0*J1*I1
    # more tightly the einsum 'kij,bjl,kml->bim' :
    #   tmp[k,b,i,l] = A[k,i,j] Xm[b,j,l]  -> 2*r*I0*J0*J1
    #   Y[b,i,m] = tmp[k,b,i,l] B[k,m,l]   -> 2*r*I0*J1*I1
    return int(r) * (2 * I0 * J0 * J1 + 2 * I0 * J1 * I1)


# ---------------------------------------------------------------- TTM-3 (TT on fused (row_k, col_k) modes)

def ttm3_fit(W, I, J, ranks):
    I0, I1, I2 = I
    J0, J1, J2 = J
    S = (
        W.reshape(I0, I1, I2, J0, J1, J2)
        .transpose(0, 3, 1, 4, 2, 5)
        .reshape(I0 * J0, I1 * J1, I2 * J2)
    )
    r1, r2 = [int(x) for x in ranks]
    M = S.reshape(I0 * J0, -1)
    U, Sv, Vh = svd_trunc(M, r1)
    c0 = U.reshape(I0, J0, r1)
    rem = (Sv[:, None] * Vh).reshape(r1 * I1 * J1, I2 * J2)
    U, Sv, Vh = svd_trunc(rem, r2)
    c1 = U.reshape(r1, I1, J1, r2)
    c2 = (Sv[:, None] * Vh).reshape(r2, I2, J2)
    return c0, c1, c2


def ttm3_apply(X, c0, c1, c2, I, J):
    I0, I1, I2 = I
    J0, J1, J2 = J
    B = X.shape[0]
    Xm = X.reshape(B, J0, J1, J2)
    # c2 (r2,I2,J2)=(t,p,k) ; Xm (B,J0,J1,J2)=(n,q,j,k)
    # Z (B,r2,I2,J0,J1)=(n,t,p,q,j)
    Z = np.einsum("tpk,nqjk->ntpqj", c2, Xm, optimize=True)
    # c1 (r1,I1,J1,r2)=(a,i,j,t) ; H (B,r1,I1,I2,J0)=(n,a,i,p,q)
    H = np.einsum("aijt,ntpqj->naipq", c1, Z, optimize=True)
    # c0 (I0,J0,r1)=(r,q,a) ; Y (B,I0,I1,I2)=(n,r,i,p)
    Y = np.einsum("rqa,naipq->nrip", c0, H, optimize=True)
    return Y.reshape(B, I0 * I1 * I2)


def ttm3_bytes(I, J, ranks, db=2):
    I0, I1, I2 = I
    J0, J1, J2 = J
    r1, r2 = [int(x) for x in ranks]
    elems = I0 * J0 * r1 + r1 * I1 * J1 * r2 + r2 * I2 * J2
    return HEADER + 4 * 8 + db * elems, elems


def ttm3_flops(I, J, ranks):
    I0, I1, I2 = I
    J0, J1, J2 = J
    r1, r2 = [int(x) for x in ranks]
    # Z: c2 (r2,I2,J2) x Xm (J0,J1,J2) -> (r2,I2,J0,J1)
    fl = 2 * r2 * I2 * J2 * J0 * J1
    # H: c1 (r1,I1,J1,r2) x Z (r2,I2,J0,J1) -> (r1,I1,I2,J0)
    fl += 2 * r1 * I1 * J1 * r2 * I2 * J0
    # Y: c0 (I0,J0,r1) x H (r1,I1,I2,J0) -> (I0,I1,I2)
    fl += 2 * I0 * J0 * r1 * I1 * I2
    return int(fl)


# ---------------------------------------------------------------- CP (4-way ALS)

def cp_als(W, I0, I1, J0, J1, rank, n_iter=6, seed=1):
    T = W.reshape(I0, I1, J0, J1)
    r = int(rank)
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((I0, r)).astype(np.float32)
    B = rng.standard_normal((I1, r)).astype(np.float32)
    C = rng.standard_normal((J0, r)).astype(np.float32)
    D = rng.standard_normal((J1, r)).astype(np.float32)
    for _ in range(n_iter):
        def upd(mode_fac, others, mttkrp):
            grams = [f.T @ f for f in others]
            G = grams[0] * grams[1] * grams[2]
            M = mttkrp
            try:
                fac = np.linalg.solve(G + 1e-6 * np.eye(r, dtype=np.float32), M.T).T
            except np.linalg.LinAlgError:
                fac = M @ np.linalg.pinv(G)
            return fac.astype(np.float32)

        A = upd(A, [B, C, D], np.einsum("ijkl,jr,kr,lr->ir", T, B, C, D, optimize=True))
        B = upd(B, [A, C, D], np.einsum("ijkl,ir,kr,lr->jr", T, A, C, D, optimize=True))
        C = upd(C, [A, B, D], np.einsum("ijkl,ir,jr,lr->kr", T, A, B, D, optimize=True))
        D = upd(D, [A, B, C], np.einsum("ijkl,ir,jr,kr->lr", T, A, B, C, optimize=True))
    return A, B, C, D


def cp_apply(X, A, B, C, D, J0, J1):
    batch = X.shape[0]
    Xm = X.reshape(batch, J0, J1)
    # s[b,r] = C[:,r]^T Xm B-wait: s[b,r] = sum_j0,j1 C[j0,r] Xm[b,j0,j1] D[j1,r]
    s = np.einsum("jr,bjl,lr->br", C, Xm, D, optimize=True)
    Y = np.einsum("ir,jr,br->bij", A, B, s, optimize=True)
    return Y.reshape(batch, A.shape[0] * B.shape[0])


def cp_bytes(I0, I1, J0, J1, r, db=2):
    elems = int(r) * (I0 + I1 + J0 + J1)
    return HEADER + 4 * 6 + db * elems, elems


def cp_flops(I0, I1, J0, J1, r):
    fl = 2 * J0 * J1 * r + 2 * J0 * r  # conservative; einsum is 2*J0*J1*r
    fl = 2 * J0 * J1 * int(r)
    fl += 2 * I0 * I1 * int(r)
    return int(fl)


# ---------------------------------------------------------------- Tensor-ring ALS (small ranks)

def tr_als(W, I0, I1, J0, J1, rank, n_iter=3, seed=2):
    T = W.reshape(I0, I1, J0, J1)
    r = int(rank)
    rng = np.random.default_rng(seed)
    # cores Gk[r, Ik, r]
    dims = (I0, I1, J0, J1)
    cores = [rng.standard_normal((r, d, r)).astype(np.float32) * 0.05 for d in dims]
    # initialize last-first from HOSVD-ish: scale
    for it in range(n_iter):
        for k, dim in enumerate(dims):
            # build M[alpha, rest, beta] by contracting the other 3 cores (cycle)
            # then LS: T_unf (Ik, rest) ≈ A_mat (Ik, r*r) @ M (r*r, rest)
            others = [(k + t) % 4 for t in range(1, 4)]
            # contract others in order along the cycle
            # start from cores[others[0]]
            # This is a bit delicate. Use einsum with 4 cores, leave k free.
            G0, G1, G2, G3 = cores
            if k == 0:
                # M[a,i1,j0,j1,b] = G1[a,i1,c] G2[c,j0,d] G3[d,j1,b]
                M = np.einsum("aic,cjd,dkb->aijkb", G1, G2, G3, optimize=True)
                Mm = M.reshape(r, I1 * J0 * J1, r).transpose(1, 0, 2).reshape(I1 * J0 * J1, r * r)
                Tf = T.reshape(I0, -1)
                sol, *_ = np.linalg.lstsq(Mm, Tf.T, rcond=None)
                cores[0] = sol.T.reshape(I0, r, r).transpose(1, 0, 2).astype(np.float32)
            elif k == 1:
                # M[a,i0,j0,j1,c] = G0[z,i0,a] G2[c,j0,d] G3[d,j1,z]
                M = np.einsum("zia,cqd,dkz->aiqkc", G0, G2, G3, optimize=True)
                rest = I0 * J0 * J1
                Mm = M.reshape(r, rest, r).transpose(1, 0, 2).reshape(rest, r * r)
                Tf = T.transpose(1, 0, 2, 3).reshape(I1, rest)
                sol, *_ = np.linalg.lstsq(Mm, Tf.T, rcond=None)
                cores[1] = sol.T.reshape(I1, r, r).transpose(1, 0, 2).astype(np.float32)
            elif k == 2:
                # M[c,i0,i1,j1,d] = G0[z,i0,a] G1[a,i1,c] G3[d,j1,z]
                M = np.einsum("zia,apc,djz->cipjd", G0, G1, G3, optimize=True)
                rest = I0 * I1 * J1
                Mm = M.reshape(r, rest, r).transpose(1, 0, 2).reshape(rest, r * r)
                Tf = T.transpose(2, 0, 1, 3).reshape(J0, rest)
                sol, *_ = np.linalg.lstsq(Mm, Tf.T, rcond=None)
                cores[2] = sol.T.reshape(J0, r, r).transpose(1, 0, 2).astype(np.float32)
            else:
                # M[d,i0,i1,j0,z] = G0[z,i0,a] G1[a,i1,c] G2[c,j0,d]
                M = np.einsum("zia,apc,cqd->dipqz", G0, G1, G2, optimize=True)
                rest = I0 * I1 * J0
                Mm = M.reshape(r, rest, r).transpose(1, 0, 2).reshape(rest, r * r)
                Tf = T.transpose(3, 0, 1, 2).reshape(J1, rest)
                sol, *_ = np.linalg.lstsq(Mm, Tf.T, rcond=None)
                cores[3] = sol.T.reshape(J1, r, r).transpose(1, 0, 2).astype(np.float32)
    return cores


def tr_apply(X, cores, I0, I1, J0, J1):
    G0, G1, G2, G3 = cores
    B = X.shape[0]
    Xm = X.reshape(B, J0, J1)
    # y[i0,i1] = sum_{j0,j1,a,b,c,d} G0[a,i0,b] G1[b,i1,c] G2[c,j0,d] G3[d,j1,a] x[j0,j1]
    # H[b,c,a] = sum_{j0,j1} G2[c,j0,d] G3[d,j1,a] Xm[b,j0,j1]
    H = np.einsum("cjd,dka,njk->nca", G2, G3, Xm, optimize=True)
    Y = np.einsum("aie,ekc,nca->nik", G0, G1, H, optimize=True)
    return Y.reshape(B, I0 * I1)


def tr_bytes(I0, I1, J0, J1, r, db=2):
    elems = int(r) * int(r) * (I0 + I1 + J0 + J1)
    return HEADER + 4 * 6 + db * elems, elems


def tr_flops(I0, I1, J0, J1, r):
    # H: G2 (r,J0,r) G3 (r,J1,r) Xm (J0,J1) -> (r,r,r) roughly
    fl = 2 * r * J0 * r * J1 * r  # loose
    fl += 2 * r * I0 * r * I1 * r
    return int(fl)


# ---------------------------------------------------------------- BTD = greedy residual Tucker

def btd_greedy(W, I0, I1, J0, J1, term_ranks, n_terms):
    residual = np.array(W, dtype=np.float32, copy=True)
    terms = []
    for _ in range(n_terms):
        Us, G, _ = hosvd_factors(residual, I0, I1, J0, J1, term_ranks)
        terms.append((Us, G))
        # subtract reconstruction via apply on identity is heavy; reconstruct core way
        Wh = tucker_reconstruct(Us, G)
        residual -= Wh
        del Wh
    return terms


def tucker_reconstruct(Us, G):
    U0, U1, U2, U3 = Us
    t = np.tensordot(U0, G, axes=(1, 0))
    t = np.tensordot(U1, t, axes=(1, 1)).transpose(1, 0, 2, 3)
    t = np.tensordot(U2, t, axes=(1, 2)).transpose(1, 2, 0, 3)
    t = np.tensordot(U3, t, axes=(1, 3)).transpose(1, 2, 3, 0)
    I0, I1, J0, J1 = t.shape
    return np.ascontiguousarray(t.reshape(I0 * I1, J0 * J1).astype(np.float32))


def btd_apply(X, terms):
    Y = None
    for Us, G in terms:
        y = tucker_apply(X, Us[0], Us[1], Us[2], Us[3], G)
        Y = y if Y is None else Y + y
    return Y


def btd_bytes(I0, I1, J0, J1, term_ranks, n_terms, db=2):
    b, e = tucker_bytes(I0, I1, J0, J1, term_ranks, db)
    # n_terms copies minus shared header once
    return HEADER + n_terms * (b - HEADER), n_terms * e


# ---------------------------------------------------------------- low-rank matrix (operator, not reconstructor) as control

def lr_fit(W, rank):
    U, S, Vh = svd_trunc(W, rank)
    return (U * S).astype(np.float32), Vh.astype(np.float32)


def lr_apply(X, A, Vh):
    # y = A @ (Vh @ x) ; X is (B, n), want X @ W.T = X @ Vh.T @ A.T
    return (X @ Vh.T) @ A.T


def lr_bytes(m, n, r, db=2):
    elems = int(r) * (m + n)
    return HEADER + 4 * 4 + db * elems, elems


def lr_flops(m, n, r):
    return 2 * int(r) * n + 2 * m * int(r)


# ---------------------------------------------------------------- accounting

def complete_bpw(total_bytes):
    return 8.0 * total_bytes / N_SOURCE


def project_all_gemv(bytes_per_elem, tiny_dtype_bytes=4):
    gemv_b = bytes_per_elem * GEMV_ELEMS
    tiny_b = tiny_dtype_bytes * TINY_ELEMS
    return complete_bpw(gemv_b + tiny_b), int(gemv_b + tiny_b)


def ai_and_bound(flops, stored_bytes, vec_bytes):
    """Arithmetic intensity of one GEMV-apply. vec_bytes = 2*(m+n) if f16 vecs, we use f32 vecs 4*(m+n)."""
    traffic = stored_bytes + vec_bytes
    ai = flops / max(traffic, 1)
    # time estimates (ESTIMATED): no GPU run
    t_bw = traffic / (BW_GEMV_GB_S * 1e9)
    t_comp16 = flops / (PEAK_FP16_TFLOP_S * 1e12)
    t_comp32 = flops / (PEAK_FP32_TFLOP_S * 1e12)
    ridge16 = (PEAK_FP16_TFLOP_S * 1e12) / (BW_GEMV_GB_S * 1e9)
    ridge32 = (PEAK_FP32_TFLOP_S * 1e12) / (BW_GEMV_GB_S * 1e9)
    return {
        "flops": int(flops),
        "stored_bytes_touched": int(stored_bytes),
        "vec_bytes_f32": int(vec_bytes),
        "traffic_bytes": int(traffic),
        "AI_flop_per_byte": float(ai),
        "ridge_fp16_est": float(ridge16),
        "ridge_fp32_est": float(ridge32),
        "bound_vs_fp16_peak": "compute" if ai > ridge16 else "bandwidth",
        "bound_vs_fp32_peak": "compute" if ai > ridge32 else "bandwidth",
        "t_bw_s_at_639p25": float(t_bw),
        "t_comp_s_fp16_est": float(t_comp16),
        "t_comp_s_fp32_est": float(t_comp32),
    }


# ---------------------------------------------------------------- self-check

def selfcheck():
    log("=== SELFCHECK ===")
    rng = np.random.default_rng(7)
    I0, I1, J0, J1 = 8, 6, 5, 7
    r0, r1, r2, r3 = 3, 3, 2, 2
    U0, _ = np.linalg.qr(rng.standard_normal((I0, r0)))
    U1, _ = np.linalg.qr(rng.standard_normal((I1, r1)))
    U2, _ = np.linalg.qr(rng.standard_normal((J0, r2)))
    U3, _ = np.linalg.qr(rng.standard_normal((J1, r3)))
    Gc = rng.standard_normal((r0, r1, r2, r3)).astype(np.float32)
    Us = [U0.astype(np.float32), U1.astype(np.float32), U2.astype(np.float32), U3.astype(np.float32)]
    W = tucker_reconstruct(Us, Gc)
    # HOSVD at true ranks must be exact
    Uh, Gh, _ = hosvd_factors(W, I0, I1, J0, J1, (r0, r1, r2, r3))
    rel, _ = tucker_rel_l2_from_core(Gh, float(np.linalg.norm(W)))
    X = rng.standard_normal((4, J0 * J1)).astype(np.float32)
    Y1 = tucker_apply(X, Uh[0], Uh[1], Uh[2], Uh[3], Gh)
    Y0 = X @ W.T
    apply_err = float(np.linalg.norm(Y1 - Y0) / (np.linalg.norm(Y0) + 1e-30))
    log(f"  tucker HOSVD rel_l2={rel:.3e} apply_err={apply_err:.3e}")
    assert rel < 1e-5, rel
    assert apply_err < 1e-5, apply_err

    # Kronecker rank-1 exact
    A = rng.standard_normal((I0, J0)).astype(np.float32)
    B = rng.standard_normal((I1, J1)).astype(np.float32)
    Wk = np.einsum("ij,kl->ikjl", A, B).reshape(I0 * I1, J0 * J1)
    Af, Bf = kron_fit(Wk, I0, I1, J0, J1, 1)
    Yk = kron_apply(X, Af, Bf)
    Yt = X @ Wk.T
    kerr = float(np.linalg.norm(Yk - Yt) / (np.linalg.norm(Yt) + 1e-30))
    log(f"  kronecker rank1 apply_err={kerr:.3e}")
    assert kerr < 1e-4, kerr

    # TT exact-ish on the Tucker tensor (ranks generous)
    c0, c1, c2, c3 = tt_svd_4(W, I0, I1, J0, J1, (r0, r0 * r1, r2))
    Yt2 = tt_apply(X, c0, c1, c2, c3, I0, I1, J0, J1)
    terr = float(np.linalg.norm(Yt2 - Y0) / (np.linalg.norm(Y0) + 1e-30))
    log(f"  tt apply_err={terr:.3e}")
    assert terr < 1e-4, terr

    # TTM-3 on a constructed Kronecker-chain
    I = (4, 3, 2)
    J = (3, 3, 2)
    # build TTM-3 of rank (2,2)
    c0t = rng.standard_normal((I[0], J[0], 2)).astype(np.float32)
    c1t = rng.standard_normal((2, I[1], J[1], 2)).astype(np.float32)
    c2t = rng.standard_normal((2, I[2], J[2])).astype(np.float32)
    # reconstruct W via apply on identity — build dense
    n = J[0] * J[1] * J[2]
    Eye = np.eye(n, dtype=np.float32)
    Wt = ttm3_apply(Eye, c0t, c1t, c2t, I, J).T  # columns = W @ e_i so this is W
    c0f, c1f, c2f = ttm3_fit(Wt, I, J, (2, 2))
    Xt = rng.standard_normal((3, n)).astype(np.float32)
    Ya = ttm3_apply(Xt, c0f, c1f, c2f, I, J)
    Yb = Xt @ Wt.T
    t3 = float(np.linalg.norm(Ya - Yb) / (np.linalg.norm(Yb) + 1e-30))
    log(f"  ttm3 apply_err={t3:.3e}")
    assert t3 < 1e-4, t3

    # CP rank-1 exact
    a = rng.standard_normal((I0, 1)).astype(np.float32)
    b = rng.standard_normal((I1, 1)).astype(np.float32)
    c = rng.standard_normal((J0, 1)).astype(np.float32)
    d = rng.standard_normal((J1, 1)).astype(np.float32)
    Wc = np.einsum("ir,jr,kr,lr->ijkl", a, b, c, d).reshape(I0 * I1, J0 * J1)
    Af, Bf, Cf, Df = cp_als(Wc, I0, I1, J0, J1, 1, n_iter=8, seed=3)
    Yc = cp_apply(X, Af, Bf, Cf, Df, J0, J1)
    cerr = float(np.linalg.norm(Yc - X @ Wc.T) / (np.linalg.norm(X @ Wc.T) + 1e-30))
    log(f"  cp rank1 apply_err={cerr:.3e}")
    assert cerr < 2e-3, cerr

    # TR apply on a constructed ring (not ALS quality — apply identity)
    rtr = 2
    Gtr = [rng.standard_normal((rtr, d, rtr)).astype(np.float32) for d in (I0, I1, J0, J1)]
    # dense Wtr via apply on I
    Iden = np.eye(J0 * J1, dtype=np.float32)
    Wtr = tr_apply(Iden, Gtr, I0, I1, J0, J1).T
    Ytr = tr_apply(X, Gtr, I0, I1, J0, J1)
    trerr = float(np.linalg.norm(Ytr - X @ Wtr.T) / (np.linalg.norm(X @ Wtr.T) + 1e-30))
    log(f"  tr apply_err={trerr:.3e}")
    assert trerr < 1e-5, trerr

    # doctor-gate import-equivalent: Q4 vs itself-ish
    Ws = rng.standard_normal((32, 64)).astype(np.float32)
    Q = c_uniform_fast(Ws, 4, 16)
    assert Q.shape == Ws.shape
    log(f"  selfcheck PASS  rss={rss_gb():.3f}G")
    return True


# ---------------------------------------------------------------- per-tensor measurement

def matrix_energy(W, ranks):
    m, n = W.shape
    if n <= m:
        G = W.T @ W
    else:
        G = W @ W.T
    w, c = eigh_energy(G)
    return {
        "side": int(min(m, n)),
        "energy": energy_at(c, ranks),
        "s1_over_s64": float(np.sqrt(w[0] / (w[min(63, len(w) - 1)] + 1e-30))),
        "top5": [float(x) for x in w[:5]],
    }


def reshape_spectra(W, pairs, ranks_report):
    out = []
    for (I0, I1), (J0, J1) in pairs:
        t0 = now()
        G0, G1, G2, G3 = mode_grams_4(W, I0, I1, J0, J1)
        modes = []
        for name, G in zip(("I0", "I1", "J0", "J1"), (G0, G1, G2, G3)):
            w, c = eigh_energy(G)
            modes.append({
                "mode": name,
                "dim": int(G.shape[0]),
                "energy": energy_at(c, [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 128, G.shape[0]]),
            })
        R = kron_rearrange(W, I0, I1, J0, J1)
        kw, kc = gram_spectrum_rect(R)
        rec = {
            "row": [I0, I1],
            "col": [J0, J1],
            "R_shape": [int(R.shape[0]), int(R.shape[1])],
            "modes": modes,
            "kronecker_energy": energy_at(kc, ranks_report + [min(R.shape)]),
            "kronecker_s1_over_s64": float(np.sqrt(kw[0] / (kw[min(63, len(kw) - 1)] + 1e-30))) if len(kw) > 63 else None,
            "wall_s": now() - t0,
        }
        out.append(rec)
        ke8 = rec["kronecker_energy"].get(8)
        ke64 = rec["kronecker_energy"].get(64)
        log(f"    reshape ({I0},{I1})x({J0},{J1}) kron_e8={ke8} kron_e64={ke64} wall={rec['wall_s']:.2f}s")
        del R, G0, G1, G2, G3
    return out


def score_map(apply_fn, Yx, Yp, X, P, ref, wnorm, W, tag):
    """apply_fn(Z) -> Z @ Wop.T. Gate against cached Yx, Yp."""
    t0 = now()
    Yph = apply_fn(P)
    probed_map_rel = float(np.linalg.norm(Yph - Yp) / (np.linalg.norm(Yp) + 1e-30))
    if X is not None:
        Yxh = apply_fn(X)
        obs_map_rel = float(np.linalg.norm(Yxh - Yx) / (np.linalg.norm(Yx) + 1e-30))
        a = axes_from_Y(Yx, Yxh, Yp, Yph)
    else:
        Yxh = None
        obs_map_rel = None
        a = axes_from_Y(None, None, Yp, Yph)
    g = gate_from_axes(a, ref)
    g["observed_map_rel_l2"] = obs_map_rel
    g["probed_map_rel_l2"] = probed_map_rel
    g["score_wall_s"] = now() - t0
    return g


def dump(obj):
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, OUT)


# ---------------------------------------------------------------- main campaign

TENSORS = [
    # name, layer, has_X (n_in==5120)
    ("language_model.model.layers.0.mlp.gate_proj.weight", 0, True, "mlp.gate_proj"),
    ("language_model.model.layers.0.mlp.down_proj.weight", 0, False, "mlp.down_proj"),
    ("language_model.model.layers.0.linear_attn.in_proj_qkv.weight", 0, True, "linear_attn.in_proj_qkv"),
    ("language_model.model.layers.0.linear_attn.out_proj.weight", 0, False, "linear_attn.out_proj"),
    ("language_model.model.layers.31.self_attn.q_proj.weight", 31, True, "self_attn.q_proj"),
    ("language_model.model.layers.31.self_attn.v_proj.weight", 31, True, "self_attn.v_proj"),
    ("language_model.model.layers.63.mlp.gate_proj.weight", 63, True, "mlp.gate_proj"),
    ("language_model.model.layers.31.self_attn.o_proj.weight", 31, False, "self_attn.o_proj"),
]


def pick_pairs(m, n):
    pairs = matching_kron_pairs(m, n)
    # always include at least these if they divide
    extra = []
    if m == 17408 and n == 5120:
        extra = [((17, 1024), (5, 1024)), ((136, 128), (40, 128)), ((272, 64), (80, 64))]
    if m == 5120 and n == 17408:
        extra = [((5, 1024), (17, 1024)), ((40, 128), (136, 128)), ((80, 64), (272, 64))]
    if m == 12288 and n == 5120:
        extra = [((48, 256), (20, 256)), ((24, 512), (10, 512)), ((96, 128), (40, 128))]
    if m == 1024 and n == 5120:
        extra = [((4, 256), (20, 256)), ((8, 128), (40, 128))]
    if m == 10240 and n == 5120:
        extra = [((10, 1024), (5, 1024)), ((80, 128), (40, 128))]
    if m == 5120 and n == 6144:
        extra = [((40, 128), (48, 128)), ((20, 256), (24, 256)), ((5, 1024), (6, 1024))]
    seen = set()
    out = []
    for p in extra + pairs:
        key = (tuple(p[0]), tuple(p[1]))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def ttm3_shapes(m, n):
    """A few 3-factor splits that divide both sides."""
    cands = []
    if m == 17408 and n == 5120:
        cands = [((17, 8, 128), (5, 8, 128)), ((17, 32, 32), (5, 32, 32)), ((136, 8, 16), (40, 8, 16))]
    elif m == 5120 and n == 17408:
        cands = [((5, 8, 128), (17, 8, 128)), ((5, 32, 32), (17, 32, 32))]
    elif m == 12288 and n == 5120:
        cands = [((24, 16, 32), (10, 16, 32)), ((48, 16, 16), (20, 16, 16))]
    elif m == 1024 and n == 5120:
        cands = [((4, 16, 16), (8, 16, 40)), ((8, 8, 16), (10, 16, 32))]
        # 1024=4*16*16, 5120=8*16*40 yes; 8*8*16=1024, 10*16*32=5120
    elif m == 10240 and n == 5120:
        cands = [((10, 8, 128), (5, 8, 128)), ((20, 16, 32), (10, 16, 32))]
    elif m == 5120 and n == 6144:
        cands = [((8, 8, 80), (8, 8, 96)), ((5, 8, 128), (6, 8, 128))]
    ok = []
    for I, J in cands:
        if int(np.prod(I)) == m and int(np.prod(J)) == n:
            ok.append((I, J))
    return ok


def run_tensor(name, layer, has_X, cls, results):
    log(f"\n======== {name} ======== rss={rss_gb():.3f}G")
    t_load = now()
    W = load_tensor(name)
    m, n = map(int, W.shape)
    wnorm = float(np.linalg.norm(W))
    wfro2 = wnorm * wnorm
    log(f"  loaded {m}x{n} ||W||_F={wnorm:.6f} load_s={now()-t_load:.2f} rss={rss_gb():.3f}G")

    X = load_X(layer) if has_X else None
    if X is not None and X.shape[1] != n:
        log(f"  WARN X dim {X.shape} vs n={n}; dropping observed")
        X = None
    P = _probe(n, 256, seed=0)
    Yx = (X @ W.T) if X is not None else None
    Yp = P @ W.T

    # Q4 reference
    t0 = now()
    Wq = c_uniform_fast(W, 4, 128)
    Yxq = (X @ Wq.T) if X is not None else None
    Ypq = P @ Wq.T
    ref = axes_from_Y(Yx, Yxq, Yp, Ypq) if X is not None else axes_from_Y(None, None, Yp, Ypq)
    q4_rel = float(np.linalg.norm(W - Wq) / wnorm)
    q4_bytes = ((m * n * 4 + 7) // 8) + (m * (n // 128) * 2) + HEADER
    log(f"  Q4 g128 rel_l2={q4_rel:.6f} axes={ {k:(round(v,6) if v is not None else None) for k,v in ref.items()} } wall={now()-t0:.2f}s")
    del Wq

    rec = {
        "name": name,
        "cls": cls,
        "layer": layer,
        "shape": [m, n],
        "elems": m * n,
        "wnorm": wnorm,
        "has_X": X is not None,
        "q4": {"rel_l2": q4_rel, "axes": ref, "stored_bytes": q4_bytes,
               "local_bpw": 8 * q4_bytes / (m * n),
               "complete_if_all_gemv": project_all_gemv(q4_bytes / (m * n))[0]},
        "matrix_energy": None,
        "reshapes": [],
        "operators": [],
        "ttm3_spectra": [],
    }

    ranks_e = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    rec["matrix_energy"] = matrix_energy(W, ranks_e)
    log(f"  matrix energy@8={rec['matrix_energy']['energy'].get(8)} @64={rec['matrix_energy']['energy'].get(64)} @256={rec['matrix_energy']['energy'].get(256)}")

    pairs = pick_pairs(m, n)
    rec["reshapes"] = reshape_spectra(W, pairs, ranks_e)

    # TTM-3 mode spectra (3-way fused)
    for I, J in ttm3_shapes(m, n):
        t0 = now()
        S = (
            W.reshape(*I, *J)
            .transpose(0, 3, 1, 4, 2, 5)
            .reshape(I[0] * J[0], I[1] * J[1], I[2] * J[2])
        )
        modes = []
        for ax, dim in enumerate(S.shape):
            # Gram along this axis
            if dim <= 4096:
                # unfold
                Tm = np.moveaxis(S, ax, 0).reshape(dim, -1)
                w, c = gram_spectrum_rect(Tm)
                modes.append({"mode": ax, "dim": int(dim), "energy": energy_at(c, ranks_e + [dim])})
                del Tm
            else:
                modes.append({"mode": ax, "dim": int(dim), "energy": None, "note": "dim>4096 skipped full gram"})
        rec["ttm3_spectra"].append({
            "I": list(I), "J": list(J), "S_shape": list(map(int, S.shape)),
            "modes": modes, "wall_s": now() - t0,
        })
        log(f"    ttm3 {I}x{J} S={tuple(S.shape)} wall={now()-t0:.2f}s")
        del S

    # Do NOT pick by raw energy@64: inner<=8 makes rank(R)<=64 so energy@64=1
    # by linear algebra (lossless reshape, local BPW=16). Prefer compressive inners.
    work_pairs = []

    def addp(p):
        if p and p not in work_pairs:
            work_pairs.append(p)

    for inner in (128, 256, 1024, 64):
        for rsh in rec["reshapes"]:
            if rsh["row"][1] == inner and rsh["col"][1] == inner:
                addp((tuple(rsh["row"]), tuple(rsh["col"])))
                break
    if not work_pairs and pairs:
        addp(pairs[0])
    work_pairs = work_pairs[:2]
    log(f"  work_pairs {work_pairs}")

    # ---------- operators on work_pairs ----------
    def add_op(op):
        rec["operators"].append(op)
        h = "HEALTHY" if op.get("gate", {}).get("healthy") else "UNHEALTHY"
        log(
            f"    OP {op['family']:14s} {op.get('tag','')} bytes={op['stored_bytes_f16']} "
            f"local_bpw={op['local_bpw_f16']:.4f} rel_l2={op.get('rel_l2')} "
            f"gate={op.get('gate',{}).get('gate')} {h} "
            f"AI={op.get('roofline',{}).get('AI_flop_per_byte')}"
        )

    # dense GEMV baseline roofline (f16 weights as consume-time comparison? source is bf16=2B)
    dense_flops = 2 * m * n
    dense_bytes = 2 * m * n  # bf16/f16
    rec["dense_gemv"] = {
        "flops": dense_flops,
        "stored_bytes_f16": dense_bytes + HEADER,
        **ai_and_bound(dense_flops, dense_bytes, 4 * (m + n)),
    }

    # matrix low-rank control at storage-matched ranks
    for r in (8, 32, 64, 128, 256):
        if r >= min(m, n):
            continue
        t0 = now()
        A, Vh = lr_fit(W, r)
        A16, V16 = f16(A), f16(Vh)
        # rel_l2 from singular mass if we had it; compute via apply on a cheap proxy:
        # reconstruct would be A@Vh (m x n) — only do for r<=128 or small tensors
        if r * (m + n) < 8_000_000 or m * n < 40_000_000:
            Wh = A16 @ V16
            rel = float(np.linalg.norm(W - Wh) / wnorm)
            del Wh
        else:
            # ||W - A V||_F^2 = ||W||^2 + ||A V||^2 - 2 <W, A V>
            # ||A V||^2 = ||A||_F^2 if V orthonormal rows (Vh from SVD is)
            # <W, A V> = <W V.T, A> = <A_exact, A> ≈ ||A|| if A = U S
            AV2 = float(np.square(A16, dtype=np.float64).sum())
            # compute <W, A V> = sum A * (W @ V.T)
            inner = float(np.sum(A16 * (W @ V16.T), dtype=np.float64))
            rel = float(np.sqrt(max(0.0, (wfro2 + AV2 - 2 * inner) / wfro2)))
        b, e = lr_bytes(m, n, r, 2)
        g = score_map(lambda Z, A=A16, V=V16: lr_apply(Z, A, V), Yx, Yp, X, P, ref, wnorm, W, "lr")
        add_op({
            "family": "lowrank_operator",
            "tag": f"r={r}",
            "ranks": [r],
            "stored_bytes_f16": b,
            "core_elems": e,
            "local_bpw_f16": 8 * b / (m * n),
            "complete_if_all_gemv": project_all_gemv(b / (m * n))[0],
            "rel_l2": rel,
            "gate": g,
            "flops": lr_flops(m, n, r),
            "roofline": ai_and_bound(lr_flops(m, n, r), b, 4 * (m + n)),
            "kernel": "lr_gemv_f16  y = A @ (Vh @ x)   two GEMVs",
            "fit_s": now() - t0,
        })
        del A, Vh, A16, V16

    for (I0, I1), (J0, J1) in work_pairs:
        # Tucker ranks targeting local f16 BPW ~ 0.25, 0.5, 1.0, 2.0 and a fat one
        tucker_rank_sets = []
        for rr in ((4, 4, 4, 4), (8, 8, 8, 8), (16, 16, 16, 16), (32, 32, 32, 32),
                   (48, 48, 32, 48), (16, 16, 8, 16)):
            rclip = (
                min(rr[0], I0), min(rr[1], I1), min(rr[2], J0), min(rr[3], J1),
            )
            if rclip not in tucker_rank_sets:
                tucker_rank_sets.append(rclip)

        for rr in tucker_rank_sets:
            t0 = now()
            Us, G, evs = hosvd_factors(W, I0, I1, J0, J1, rr)
            Us16 = [f16(u) for u in Us]
            G16 = f16(G)
            rel, g2 = tucker_rel_l2_from_core(G, wnorm)
            # f16 perturbs orthogonality slightly — measure map error via apply
            b, e = tucker_bytes(I0, I1, J0, J1, rr, 2)
            fl = tucker_flops_v(I0, I1, J0, J1, rr)
            g = score_map(
                lambda Z, U=Us16, Gc=G16: tucker_apply(Z, U[0], U[1], U[2], U[3], Gc),
                Yx, Yp, X, P, ref, wnorm, W, "tucker",
            )
            add_op({
                "family": "tucker_hosvd",
                "tag": f"{(I0,I1)}x{(J0,J1)} R={rr}",
                "reshape": [[I0, I1], [J0, J1]],
                "ranks": list(rr),
                "stored_bytes_f16": b,
                "core_elems": e,
                "local_bpw_f16": 8 * b / (m * n),
                "complete_if_all_gemv": project_all_gemv(b / (m * n))[0],
                "rel_l2_orth_f32core": rel,
                "rel_l2": g["probed_map_rel_l2"],
                "gate": g,
                "flops": fl,
                "roofline": ai_and_bound(fl, b, 4 * (m + n)),
                "kernel": "tucker_gemv_f16  5 contractions (2 factor GEMMs, core, 2 expand GEMMs)",
                "n_sequential_contractions": 5,
                "fit_s": now() - t0,
            })
            del Us, G, Us16, G16

        # TT ranks
        for tr in ((8, 8, 8), (16, 16, 16), (32, 32, 32), (64, 32, 16), (64, 64, 64)):
            trc = (min(tr[0], I0), min(tr[1], I0 * I1), min(tr[2], J1))
            t0 = now()
            try:
                c0, c1, c2, c3 = tt_svd_4(W, I0, I1, J0, J1, trc)
            except Exception as ex:
                log(f"    TT fail {trc}: {ex}")
                continue
            c0, c1, c2, c3 = f16(c0), f16(c1), f16(c2), f16(c3)
            b, e = tt_bytes(I0, I1, J0, J1, trc, 2)
            fl = tt_flops(I0, I1, J0, J1, trc)
            g = score_map(
                lambda Z, a=c0, b1=c1, c=c2, d=c3: tt_apply(Z, a, b1, c, d, I0, I1, J0, J1),
                Yx, Yp, X, P, ref, wnorm, W, "tt",
            )
            add_op({
                "family": "tensor_train",
                "tag": f"{(I0,I1)}x{(J0,J1)} r={trc}",
                "reshape": [[I0, I1], [J0, J1]],
                "ranks": list(trc),
                "stored_bytes_f16": b,
                "core_elems": e,
                "local_bpw_f16": 8 * b / (m * n),
                "complete_if_all_gemv": project_all_gemv(b / (m * n))[0],
                "rel_l2": g["probed_map_rel_l2"],
                "gate": g,
                "flops": fl,
                "roofline": ai_and_bound(fl, b, 4 * (m + n)),
                "kernel": "tt_gemv_f16  4 sequential core contractions",
                "n_sequential_contractions": 4,
                "fit_s": now() - t0,
            })
            del c0, c1, c2, c3

        # Kronecker-sum / TTM-2
        for kr in (1, 16, 64, 128):
            if kr > min(I0 * J0, I1 * J1):
                continue
            t0 = now()
            A, B = kron_fit(W, I0, I1, J0, J1, kr)
            A16, B16 = f16(A), f16(B)
            b, e = kron_bytes(I0, I1, J0, J1, kr, 2)
            fl = kron_flops(I0, I1, J0, J1, kr)
            g = score_map(
                lambda Z, a=A16, bb=B16: kron_apply(Z, a, bb),
                Yx, Yp, X, P, ref, wnorm, W, "kron",
            )
            add_op({
                "family": "kronecker_sum",
                "tag": f"{(I0,I1)}x{(J0,J1)} k={kr}",
                "reshape": [[I0, I1], [J0, J1]],
                "ranks": [kr],
                "stored_bytes_f16": b,
                "core_elems": e,
                "local_bpw_f16": 8 * b / (m * n),
                "complete_if_all_gemv": project_all_gemv(b / (m * n))[0],
                "rel_l2": g["probed_map_rel_l2"],
                "gate": g,
                "flops": fl,
                "flop_ratio_vs_dense": fl / dense_flops,
                "roofline": ai_and_bound(fl, b, 4 * (m + n)),
                "kernel": "kronecker_sum_gemv_f16  fused loop over k terms of y += A_k @ X @ B_k.T",
                "n_sequential_contractions": 2,  # fused over k in one kernel
                "n_terms": kr,
                "unfused_dispatches": 2 * kr,
                "fit_s": now() - t0,
            })
            del A, B, A16, B16

        # BTD greedy: a few small Tucker terms. One reshape only (the best pair).
        if work_pairs and (I0, I1) == work_pairs[0][0] and (J0, J1) == work_pairs[0][1]:
            for nterms, trr in ((4, (8, 8, 8, 8)), (8, (4, 4, 4, 4)), (2, (16, 16, 16, 16))):
                trr = (
                    min(trr[0], I0), min(trr[1], I1), min(trr[2], J0), min(trr[3], J1),
                )
                t0 = now()
                terms = btd_greedy(W, I0, I1, J0, J1, trr, nterms)
                terms16 = [([f16(u) for u in Us], f16(G)) for Us, G in terms]
                b, e = btd_bytes(I0, I1, J0, J1, trr, nterms, 2)
                fl = nterms * tucker_flops_v(I0, I1, J0, J1, trr)
                g = score_map(
                    lambda Z, t=terms16: btd_apply(Z, t),
                    Yx, Yp, X, P, ref, wnorm, W, "btd",
                )
                add_op({
                    "family": "block_term",
                    "tag": f"{(I0,I1)}x{(J0,J1)} R={nterms} L={trr}",
                    "reshape": [[I0, I1], [J0, J1]],
                    "ranks": [nterms, *trr],
                    "stored_bytes_f16": b,
                    "core_elems": e,
                    "local_bpw_f16": 8 * b / (m * n),
                    "complete_if_all_gemv": project_all_gemv(b / (m * n))[0],
                    "rel_l2": g["probed_map_rel_l2"],
                    "gate": g,
                    "flops": fl,
                    "roofline": ai_and_bound(fl, b, 4 * (m + n)),
                    "kernel": "btd_gemv_f16  sum of R Tucker applies, fused over terms",
                    "n_sequential_contractions": 5,
                    "n_terms": nterms,
                    "fit_s": now() - t0,
                })
                del terms, terms16

        # CP ALS MTTKRP is O(m n R) per iteration; only on small GEMVs (v_proj).
        if m * n <= 8_000_000:
            for cpr in (16, 64, 256):
                t0 = now()
                try:
                    A, B, C, D = cp_als(W, I0, I1, J0, J1, cpr, n_iter=5, seed=1)
                except Exception as ex:
                    log(f"    CP fail r={cpr}: {ex}")
                    continue
                A, B, C, D = f16(A), f16(B), f16(C), f16(D)
                b, e = cp_bytes(I0, I1, J0, J1, cpr, 2)
                fl = cp_flops(I0, I1, J0, J1, cpr)
                g = score_map(
                    lambda Z, a=A, bb=B, c=C, d=D: cp_apply(Z, a, bb, c, d, J0, J1),
                    Yx, Yp, X, P, ref, wnorm, W, "cp",
                )
                add_op({
                    "family": "cp",
                    "tag": f"{(I0,I1)}x{(J0,J1)} r={cpr}",
                    "reshape": [[I0, I1], [J0, J1]],
                    "ranks": [cpr],
                    "stored_bytes_f16": b,
                    "core_elems": e,
                    "local_bpw_f16": 8 * b / (m * n),
                    "complete_if_all_gemv": project_all_gemv(b / (m * n))[0],
                    "rel_l2": g["probed_map_rel_l2"],
                    "gate": g,
                    "flops": fl,
                    "roofline": ai_and_bound(fl, b, 4 * (m + n)),
                    "kernel": "cp_gemv_f16  s_r = c_r^T X d_r ; y = (A*s) B^T",
                    "n_sequential_contractions": 2,
                    "fit_s": now() - t0,
                })
                del A, B, C, D

        # TR ALS only on modest 4-way (v_proj and maybe 128-inner of small)
        if m * n <= 10_000_000 or (I0 <= 48 and J0 <= 48):
            for trr in (4, 8):
                t0 = now()
                try:
                    cores = tr_als(W, I0, I1, J0, J1, trr, n_iter=2, seed=2)
                except Exception as ex:
                    log(f"    TR fail r={trr}: {ex}")
                    traceback.print_exc()
                    continue
                cores16 = [f16(c) for c in cores]
                b, e = tr_bytes(I0, I1, J0, J1, trr, 2)
                fl = tr_flops(I0, I1, J0, J1, trr)
                g = score_map(
                    lambda Z, c=cores16: tr_apply(Z, c, I0, I1, J0, J1),
                    Yx, Yp, X, P, ref, wnorm, W, "tr",
                )
                add_op({
                    "family": "tensor_ring",
                    "tag": f"{(I0,I1)}x{(J0,J1)} r={trr}",
                    "reshape": [[I0, I1], [J0, J1]],
                    "ranks": [trr],
                    "stored_bytes_f16": b,
                    "core_elems": e,
                    "local_bpw_f16": 8 * b / (m * n),
                    "complete_if_all_gemv": project_all_gemv(b / (m * n))[0],
                    "rel_l2": g["probed_map_rel_l2"],
                    "gate": g,
                    "flops": fl,
                    "roofline": ai_and_bound(fl, b, 4 * (m + n)),
                    "kernel": "tr_gemv_f16  cyclic 4-core contraction",
                    "n_sequential_contractions": 4,
                    "fit_s": now() - t0,
                })
                del cores, cores16

    # TTM-3 operators on first valid shape
    for I, J in ttm3_shapes(m, n)[:1]:
        for rr in ((8, 8), (16, 16), (32, 16), (32, 32)):
            t0 = now()
            try:
                c0, c1, c2 = ttm3_fit(W, I, J, rr)
            except Exception as ex:
                log(f"    TTM3 fail {I}x{J} {rr}: {ex}")
                continue
            c0, c1, c2 = f16(c0), f16(c1), f16(c2)
            b, e = ttm3_bytes(I, J, rr, 2)
            fl = ttm3_flops(I, J, rr)
            g = score_map(
                lambda Z, a=c0, bb=c1, c=c2, II=I, JJ=J: ttm3_apply(Z, a, bb, c, II, JJ),
                Yx, Yp, X, P, ref, wnorm, W, "ttm3",
            )
            add_op({
                "family": "tt_matrix_3",
                "tag": f"I={I} J={J} r={rr}",
                "reshape": [list(I), list(J)],
                "ranks": list(rr),
                "stored_bytes_f16": b,
                "core_elems": e,
                "local_bpw_f16": 8 * b / (m * n),
                "complete_if_all_gemv": project_all_gemv(b / (m * n))[0],
                "rel_l2": g["probed_map_rel_l2"],
                "gate": g,
                "flops": fl,
                "flop_ratio_vs_dense": fl / dense_flops,
                "roofline": ai_and_bound(fl, b, 4 * (m + n)),
                "kernel": "ttm3_gemv_f16  3 sequential TTM core contractions",
                "n_sequential_contractions": 3,
                "fit_s": now() - t0,
            })
            del c0, c1, c2

    # Mixture: best-pair Kronecker-k + low-rank residual
    if work_pairs:
        (I0, I1), (J0, J1) = work_pairs[0]
        for kr, lr in ((16, 32), (64, 32)):
            if kr > min(I0 * J0, I1 * J1):
                continue
            t0 = now()
            A, B = kron_fit(W, I0, I1, J0, J1, kr)
            A16, B16 = f16(A), f16(B)
            # residual via apply on identity is W-sized; form residual in place-ish
            # Wh_k[i0,i1,j0,j1] = sum_k A[k,i0,j0] B[k,i1,j1]
            Whk = np.einsum("kij,klm->iljm", A16, B16).reshape(m, n)
            R = W - Whk
            del Whk
            AA, Vh = lr_fit(R, lr)
            AA, Vh = f16(AA), f16(Vh)
            del R
            b1, e1 = kron_bytes(I0, I1, J0, J1, kr, 2)
            b2, e2 = lr_bytes(m, n, lr, 2)
            b = b1 + b2 - HEADER  # one header
            fl = kron_flops(I0, I1, J0, J1, kr) + lr_flops(m, n, lr)

            def mix_apply(Z, a=A16, bb=B16, aa=AA, v=Vh):
                return kron_apply(Z, a, bb) + lr_apply(Z, aa, v)

            g = score_map(mix_apply, Yx, Yp, X, P, ref, wnorm, W, "mix")
            add_op({
                "family": "kronecker_plus_lowrank",
                "tag": f"{(I0,I1)}x{(J0,J1)} k={kr}+lr={lr}",
                "reshape": [[I0, I1], [J0, J1]],
                "ranks": [kr, lr],
                "stored_bytes_f16": b,
                "core_elems": e1 + e2,
                "local_bpw_f16": 8 * b / (m * n),
                "complete_if_all_gemv": project_all_gemv(b / (m * n))[0],
                "rel_l2": g["probed_map_rel_l2"],
                "gate": g,
                "flops": fl,
                "flop_ratio_vs_dense": fl / dense_flops,
                "roofline": ai_and_bound(fl, b, 4 * (m + n)),
                "kernel": "mix_kron_lr_gemv_f16  kronecker_sum then axpy low-rank GEMV",
                "n_sequential_contractions": 4,
                "fit_s": now() - t0,
            })
            del A, B, A16, B16, AA, Vh

    rec["rss_gb_after"] = rss_gb()
    results["tensors"].append(rec)
    dump(results)
    del W, X, P, Yx, Yp
    return rec


def gaussian_control():
    log("\n======== GAUSSIAN CONTROL 17408x5120 ========")
    rng = np.random.default_rng(123)
    # don't allocate a full 17408x5120 if we can do a scaled smaller analog AND one real-size
    # Real-size iid is 356MB — allowed. Use N(0,1) then spectra only (no gate).
    W = rng.standard_normal((17408, 5120), dtype=np.float32)
    W *= 0.02
    rec = {"shape": [17408, 5120], "kind": "iid_gaussian_N(0,0.02^2)"}
    rec["matrix_energy"] = matrix_energy(W, [1, 8, 16, 32, 64, 128, 256])
    pairs = [((17, 1024), (5, 1024)), ((136, 128), (40, 128)), ((272, 64), (80, 64))]
    rec["reshapes"] = reshape_spectra(W, pairs, [1, 8, 16, 32, 64, 128, 256])
    del W
    rec["rss_gb"] = rss_gb()
    log(f"  gaussian matrix e64={rec['matrix_energy']['energy'].get(64)}")
    return rec


def main():
    t_all = now()
    results = {
        "schema": "hawking.gravity1.tensor_operators.v1",
        "N": N_SOURCE,
        "GEMV_ELEMS": GEMV_ELEMS,
        "TINY_ELEMS": TINY_ELEMS,
        "source": SRC,
        "capture": CAP,
        "hardware": {
            "box": "Apple M3 Ultra 96GB 60 GPU cores",
            "BW_GEMV_GB_S_MEASURED": BW_GEMV_GB_S,
            "BW_DATASHEET_GB_S": BW_DATASHEET_GB_S,
            "PEAK_FP16_TFLOP_S_ESTIMATED": PEAK_FP16_TFLOP_S,
            "PEAK_FP32_TFLOP_S_ESTIMATED": PEAK_FP32_TFLOP_S,
            "peak_note": "60/80 * 60.5 marketed TFLOPS; not measured this lane",
        },
        "selfcheck": None,
        "gaussian_control": None,
        "tensors": [],
        "errors": [],
    }
    try:
        selfcheck()
        results["selfcheck"] = "PASS"
    except Exception as e:
        results["selfcheck"] = f"FAIL {e}"
        traceback.print_exc()
        dump(results)
        raise

    try:
        results["gaussian_control"] = gaussian_control()
        dump(results)
    except Exception as e:
        results["errors"].append({"where": "gaussian", "err": str(e)})
        traceback.print_exc()

    for name, layer, has_X, cls in TENSORS:
        try:
            run_tensor(name, layer, has_X, cls, results)
        except Exception as e:
            results["errors"].append({"where": name, "err": str(e)})
            traceback.print_exc()
            dump(results)

    results["wall_s"] = now() - t_all
    results["rss_gb_peak"] = rss_gb()
    dump(results)
    log(f"\nDONE wall={results['wall_s']:.1f}s rss_peak={results['rss_gb_peak']:.3f}G ops={sum(len(t['operators']) for t in results['tensors'])} errors={results['errors']}")


if __name__ == "__main__":
    main()
