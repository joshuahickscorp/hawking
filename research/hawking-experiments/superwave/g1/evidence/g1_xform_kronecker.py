#!/usr/bin/env python3
"""Kronecker-structured orthogonal transform search on Qwen3.8-27B tensors.

M = A ⊗ B, A,B orthogonal. Stored weights become W M; activations in the
folded basis are X M. Function W x is exactly preserved when M is folded.
Objective: quantization error of the transformed matrix at 2/3/4 bits,
scored with the adequacy gate (not bare activation cosine).

CPU only. No GPU, no Metal, no inference.
"""
from __future__ import annotations

import json, os, struct, sys, time, resource
import numpy as np

ROOT = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
CAPTURE = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
TOOLS = "/Users/scammermike/.claude-grok/worktrees/200-xform-kronecker-20260817-181002/tools"
OUT_DIR = "/tmp/g1-xform-kronecker"
N_SOURCE = 26_895_998_464
GROUP = 128

sys.path.insert(0, TOOLS)
from gravity_doctor_gate import axes, gate, AXIS_MARGIN, _probe  # noqa: E402

os.makedirs(OUT_DIR, exist_ok=True)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def load_tensor(name, root=ROOT):
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


def load_X(layer, capture=CAPTURE):
    res = json.load(open(os.path.join(capture, "capture-result.json")))
    per = res["per_layer"][str(layer)]
    X = np.fromfile(per["path"], dtype=np.float32).reshape(per["n_rows"], res["hidden"])
    return X


def c_uniform(W, bits, group=GROUP):
    W = np.ascontiguousarray(W, dtype=np.float32)
    out, d = W.shape
    ng = d // group
    lim = (1 << (bits - 1)) - 1
    blk = W[:, : ng * group].reshape(out, ng, group)
    amax = np.max(np.abs(blk), axis=2, keepdims=True)
    step = amax / lim + 1e-30
    q = np.clip(np.round(blk / step), -lim, lim) * step
    if d == ng * group:
        return q.reshape(out, d)
    Wh = W.copy()
    Wh[:, : ng * group] = q.reshape(out, ng * group)
    return Wh


def odct(n):
    k = np.arange(n, dtype=np.float64)[:, None]
    j = np.arange(n, dtype=np.float64)[None, :]
    D = np.cos(np.pi * k * (2.0 * j + 1.0) / (2.0 * n))
    D *= np.sqrt(2.0 / n)
    D[0] *= 1.0 / np.sqrt(2.0)
    return D.astype(np.float32)


def hadamard(n):
    if n < 1 or n & (n - 1):
        raise ValueError(n)
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    H /= np.sqrt(n)
    return H.astype(np.float32)


def is_p2(n):
    return n > 0 and (n & (n - 1)) == 0


def signed_orth(Q, seed):
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=Q.shape[1])
    return Q * s  # columns scaled; Q @ diag(s)


def haar(n, seed):
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    return (Q * s).astype(np.float32)


def polar_orth(M):
    U, _, Vt = np.linalg.svd(M.astype(np.float64), full_matrices=False)
    return (U @ Vt).astype(np.float32)


def apply_WM(W, A, B):
    """W @ (A ⊗ B) with C-order Kronecker: row.reshape(p,q) -> A @ row @ B.T."""
    out = W.shape[0]
    p, q = A.shape[0], B.shape[0]
    X = np.ascontiguousarray(W.reshape(out, p, q))
    Y = np.matmul(A, np.matmul(X, B.T))
    return np.ascontiguousarray(Y.reshape(out, p * q))


def apply_XM(X, A, B):
    t = X.shape[0]
    p, q = A.shape[0], B.shape[0]
    xr = np.ascontiguousarray(X.reshape(t, p, q))
    Y = np.matmul(A, np.matmul(xr, B.T))
    return np.ascontiguousarray(Y.reshape(t, p * q))


def orth_err(Q):
    I = np.eye(Q.shape[0], dtype=np.float64)
    return float(np.linalg.norm(Q.astype(np.float64).T @ Q.astype(np.float64) - I, ord="fro"))


def excess_kurtosis(w):
    x = w.ravel().astype(np.float64)
    x = x - x.mean()
    m2 = np.mean(x * x)
    m4 = np.mean(x * x * x * x)
    return float(m4 / (m2 * m2 + 1e-30) - 3.0)


def dynrange(w):
    x = w.ravel().astype(np.float64)
    rms = np.sqrt(np.mean(x * x))
    return float(np.max(np.abs(x)) / (rms + 1e-30))


def group_amax_over_rms(W, group=GROUP):
    out, d = W.shape
    ng = d // group
    blk = W[:, : ng * group].reshape(out, ng, group).astype(np.float64, copy=False)
    amax = np.max(np.abs(blk), axis=2)
    rms = np.sqrt(np.mean(blk * blk, axis=2))
    return float(np.mean(amax / (rms + 1e-30)))


def col_dynrange_mean(W):
    """Mean over columns of max/rms. Input-axis outliers live here."""
    c = W.astype(np.float64, copy=False)
    rms = np.sqrt(np.mean(c * c, axis=0))
    mx = np.max(np.abs(c), axis=0)
    return float(np.mean(mx / (rms + 1e-30)))


def relF(W, Wh):
    num = np.linalg.norm((W - Wh).ravel().astype(np.float64))
    den = np.linalg.norm(W.ravel().astype(np.float64))
    return float(num / (den + 1e-30))


def weight_stats(W):
    return {
        "kurtosis": excess_kurtosis(W),
        "dynrange": dynrange(W),
        "group_amax_rms": group_amax_over_rms(W),
        "col_dynrange_mean": col_dynrange_mean(W),
        "rms": float(np.sqrt(np.mean(W.astype(np.float64) ** 2))),
        "absmax": float(np.max(np.abs(W))),
    }


def score_bits(W, X, bits_list, seed=0):
    out = {}
    for b in bits_list:
        Q = c_uniform(W, b)
        a = axes(W, Q, X, seed=seed)
        a["relF"] = relF(W, Q)
        a["min_axis"] = float(min(a["observed"], a["probed"], a["worst_unit"]))
        out[b] = a
        del Q
    return out


def relative_gate(cand_axes, ref):
    deficits = {k: cand_axes[k] - (ref[k] - AXIS_MARGIN[k]) for k in ("observed", "probed", "worst_unit")}
    worst = min(deficits, key=deficits.get)
    return {
        "deficit": deficits,
        "gate": deficits[worst],
        "worst_axis": worst,
        "healthy": deficits[worst] >= 0.0,
    }


def factor_bytes(p, q, dtype_bytes=4):
    return dtype_bytes * (p * p + q * q)


def factor_bpw(p, q, sites, dtype_bytes=4):
    return 8.0 * factor_bytes(p, q, dtype_bytes) * sites / N_SOURCE


def pca_then_spread(k, slices, spread):
    """B such that slices @ B.T = (slices @ V) @ spread.T  (PCA then spread)."""
    # slices: (N, k)
    C = slices.astype(np.float64).T @ slices.astype(np.float64)
    _, V = np.linalg.eigh(C)
    V = V.astype(np.float32)
    return spread @ V.T


def constructors_for(p, q, W=None):
    D_p, D_q = odct(p), odct(q)
    ctors = [("dct_dct", D_p, D_q)]
    Hp = hadamard(p) if is_p2(p) else None
    Hq = hadamard(q) if is_p2(q) else None
    if Hp is not None:
        ctors.append(("had_dct", Hp, D_q))
        for s in range(4):
            ctors.append((f"shad{s}_dct", signed_orth(Hp, s), D_q))
    if Hq is not None:
        ctors.append(("dct_had", D_p, Hq))
        for s in range(4):
            ctors.append((f"dct_shad{s}", D_p, signed_orth(Hq, s)))
    if Hp is not None and Hq is not None:
        ctors.append(("had_had", Hp, Hq))
        ctors.append(("shad0_shad0", signed_orth(Hp, 0), signed_orth(Hq, 0)))
    for s in range(5):
        ctors.append((f"sdct{s}_sdct{s}", signed_orth(D_p, s), signed_orth(D_q, s)))
    ctors.append(("haar0", haar(p, 0), haar(q, 0)))
    ctors.append(("haar1", haar(p, 1), haar(q, 1)))
    if W is not None:
        out = W.shape[0]
        X3 = W.reshape(out, p, q)
        spA = hadamard(p) if is_p2(p) else D_p
        spB = hadamard(q) if is_p2(q) else D_q
        sl_q = np.ascontiguousarray(X3.reshape(-1, q))
        sl_p = np.ascontiguousarray(np.swapaxes(X3, 1, 2).reshape(-1, p))
        A = pca_then_spread(p, sl_p, spA)
        B = pca_then_spread(q, sl_q, spB)
        ctors.append(("pca_spread", A, B))
        del sl_q, sl_p, X3
    return ctors


def amax2_grad(W, group=GROUP):
    """Euclidean subgradient of sum (group absmax)^2."""
    out, d = W.shape
    ng = d // group
    blk = W[:, : ng * group].reshape(out, ng, group)
    idx = np.argmax(np.abs(blk), axis=2)
    signed = np.take_along_axis(blk, idx[..., None], axis=2)
    G = np.zeros_like(blk)
    np.put_along_axis(G, idx[..., None], 2.0 * signed, axis=2)
    if d == ng * group:
        return G.reshape(out, d)
    outG = np.zeros_like(W)
    outG[:, : ng * group] = G.reshape(out, ng * group)
    return outG


def ste_quant_grad(W, bits, group=GROUP):
    Q = c_uniform(W, bits, group)
    return W - Q


def riemannian_refine(W, A, B, steps=8, lr=0.12, bits=3, objective="amax2"):
    """Polar-retracted steps on (A,B) to reduce group-amax^2 or 3-bit STE residual."""
    out = W.shape[0]
    p, q = A.shape[0], B.shape[0]
    X3 = np.ascontiguousarray(W.reshape(out, p, q))
    hist = []
    A = A.copy()
    B = B.copy()
    for t in range(steps):
        Y = np.matmul(A, np.matmul(X3, B.T))
        Y2 = Y.reshape(out, p * q)
        if objective == "amax2":
            G2 = amax2_grad(Y2)
            obj = float(np.mean(np.max(np.abs(Y2.reshape(out, -1, GROUP)), axis=2) ** 2))
        else:
            G2 = ste_quant_grad(Y2, bits)
            obj = relF(Y2, c_uniform(Y2, bits))
        G = G2.reshape(out, p, q)
        Z = np.matmul(X3, B.T)  # (out,p,q)
        U = np.matmul(A, X3)
        dA = G.transpose(1, 0, 2).reshape(p, -1) @ Z.transpose(1, 0, 2).reshape(p, -1).T
        dB = G.transpose(2, 0, 1).reshape(q, -1) @ U.transpose(2, 0, 1).reshape(q, -1).T
        # scale-invariant step
        nA = float(np.linalg.norm(dA) + 1e-30)
        nB = float(np.linalg.norm(dB) + 1e-30)
        step = lr * (0.85 ** t)
        A = polar_orth(A - step * (dA / nA).astype(np.float32))
        B = polar_orth(B - step * (dB / nB).astype(np.float32))
        hist.append({"t": t, "obj": obj, "orthA": orth_err(A), "orthB": orth_err(B)})
        del Y, Y2, G2, G, Z, U, dA, dB
    return A, B, hist


SHAPES_5120 = [
    (64, 80),
    (80, 64),
    (40, 128),
    (128, 40),
    (20, 256),
    (32, 160),
    (16, 320),
    (10, 512),
    (8, 640),
    (5, 1024),
    (256, 20),
    (4, 1280),
]

TENSORS = [
    # three depths, gate/up class
    (0, "mlp.gate_proj", "gate"),
    (31, "mlp.gate_proj", "gate"),
    (63, "mlp.gate_proj", "gate"),
    # three depths, attention class (discover per layer)
    (0, "linear_attn.in_proj_qkv", "attn"),
    (31, "self_attn.q_proj", "attn"),
    (63, "self_attn.q_proj", "attn"),
    # expensive attention, small, mid-depth
    (31, "self_attn.v_proj", "attn"),
]


def fnum(x):
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, dict):
        return {k: fnum(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [fnum(v) for v in x]
    return x


def run_selfcheck():
    rng = np.random.default_rng(0)
    p, q, out = 8, 10, 12
    A = haar(p, 1)
    B = haar(q, 2)
    W = rng.standard_normal((out, p * q)).astype(np.float32)
    Wm = apply_WM(W, A, B)
    back = apply_WM(Wm, A.T, B.T)
    rec = float(np.max(np.abs(back - W)))
    X = rng.standard_normal((7, p * q)).astype(np.float32)
    # W M M^T x == W x
    y0 = X @ W.T
    y1 = apply_XM(X, A, B) @ Wm.T
    recx = float(np.max(np.abs(y0 - y1)))
    assert rec < 1e-5, rec
    assert recx < 1e-5, recx
    assert orth_err(A) < 1e-5 and orth_err(odct(80)) < 1e-5
    assert orth_err(hadamard(128)) < 1e-5
    Q = c_uniform(W, 4, group=8)
    assert Q.shape == W.shape
    print(f"SELFCHECK PASS  rec={rec:.3e} recx={recx:.3e} orth_dct80={orth_err(odct(80)):.3e} rss={rss_gb():.3f}GB")
    return {"rec": rec, "recx": recx}


def eval_pair(W, X, A, B, bits_list, ref_orig, seed=0):
    t0 = time.time()
    Wt = apply_WM(W, A, B)
    Xt = apply_XM(X, A, B)
    stats = weight_stats(Wt)
    scored = score_bits(Wt, Xt, bits_list, seed=seed)
    vs_own_q4 = {}
    vs_orig_q4 = {}
    for b, a in scored.items():
        vs_own_q4[b] = relative_gate(a, scored[4]) if 4 in scored else None
        vs_orig_q4[b] = relative_gate(a, ref_orig)
    dt = time.time() - t0
    return {
        "stats": stats,
        "bits": scored,
        "vs_own_q4": vs_own_q4,
        "vs_orig_q4": vs_orig_q4,
        "orthA": orth_err(A),
        "orthB": orth_err(B),
        "wall_s": dt,
        "Wt": Wt,  # caller may keep or drop
        "Xt": Xt,
    }


def run_tensor(layer, cls, family, shapes, bits_list=(2, 3, 4), refine=True):
    name = f"language_model.model.layers.{layer}.{cls}.weight"
    print(f"\n======== {name}  rss={rss_gb():.3f}GB ========", flush=True)
    t_load = time.time()
    W = load_tensor(name)
    X = load_X(layer)
    assert X.shape[1] == W.shape[1], (X.shape, W.shape)
    print(f"  loaded shape={tuple(W.shape)} X={tuple(X.shape)} load_s={time.time()-t_load:.2f}", flush=True)

    base_stats = weight_stats(W)
    print(f"  BASE stats {json.dumps(fnum(base_stats))}", flush=True)
    base_bits = score_bits(W, X, bits_list)
    ref_orig = base_bits[4]
    print(f"  BASE bits:", flush=True)
    for b in bits_list:
        a = base_bits[b]
        rg = relative_gate(a, ref_orig)
        print(
            f"    q{b} obs={a['observed']:.6f} prb={a['probed']:.6f} "
            f"wu={a['worst_unit']:.6f} min={a['min_axis']:.6f} relF={a['relF']:.6f} "
            f"vsQ4={'HEALTHY' if rg['healthy'] else 'UNHEALTHY'} gate={rg['gate']:+.6f} ({rg['worst_axis']})",
            flush=True,
        )

    result = {
        "tensor": name,
        "layer": layer,
        "cls": cls,
        "family": family,
        "shape": list(W.shape),
        "base_stats": base_stats,
        "base_bits": {str(b): {k: v for k, v in a.items()} for b, a in base_bits.items()},
        "base_vs_q4": {str(b): relative_gate(base_bits[b], ref_orig) for b in bits_list},
        "screen": [],
        "best": None,
        "refined": None,
    }

    best = None
    # identity is the baseline; not a Kronecker win
    for p, q in shapes:
        if p * q != W.shape[1]:
            continue
        print(f"  -- shape {p}x{q}  factor_f32={factor_bytes(p,q)}B  "
              f"64site_bpw={factor_bpw(p,q,64):.9f}", flush=True)
        try:
            ctors = constructors_for(p, q, W)
        except Exception as e:
            print(f"    ctor-build FAIL {e}", flush=True)
            continue
        for cname, A, B in ctors:
            try:
                ev = eval_pair(W, X, A, B, bits_list, ref_orig)
            except Exception as e:
                print(f"    {cname:16s} FAIL {e}", flush=True)
                continue
            row = {
                "p": p,
                "q": q,
                "ctor": cname,
                "stats": ev["stats"],
                "bits": {str(b): {k: ev["bits"][b][k] for k in
                                  ("observed", "probed", "worst_unit", "relF", "min_axis")}
                         for b in bits_list},
                "vs_own_q4": {str(b): ev["vs_own_q4"][b] for b in bits_list},
                "vs_orig_q4": {str(b): ev["vs_orig_q4"][b] for b in bits_list},
                "orthA": ev["orthA"],
                "orthB": ev["orthB"],
                "wall_s": ev["wall_s"],
                "factor_bytes_f32": factor_bytes(p, q, 4),
                "factor_bytes_f16": factor_bytes(p, q, 2),
                "bpw_64_f32": factor_bpw(p, q, 64, 4),
                "bpw_64_f16": factor_bpw(p, q, 64, 2),
                "bpw_1_f32": factor_bpw(p, q, 1, 4),
            }
            result["screen"].append(row)
            a3 = ev["bits"][3]
            print(
                f"    {cname:16s} q3 min={a3['min_axis']:.6f} relF={a3['relF']:.6f} "
                f"kurt={ev['stats']['kurtosis']:+.4f} dyn={ev['stats']['dynrange']:.3f} "
                f"gAR={ev['stats']['group_amax_rms']:.3f} "
                f"dmin={a3['min_axis']-base_bits[3]['min_axis']:+.6f} "
                f"drelF={a3['relF']-base_bits[3]['relF']:+.6f} "
                f"vsOrigQ4={'H' if ev['vs_orig_q4'][3]['healthy'] else 'U'} "
                f"{ev['wall_s']:.2f}s",
                flush=True,
            )
            key = (a3["min_axis"], -a3["relF"], -ev["stats"]["kurtosis"])
            if best is None or key > best[0]:
                best = (key, row, A, B)
            del ev

    if best is None:
        print("  no successful constructor", flush=True)
        result["peak_rss_gb"] = rss_gb()
        return result, W, X, None, None

    _, brow, Ab, Bb = best
    result["best"] = {k: v for k, v in brow.items()}
    print(
        f"  BEST cheap {brow['ctor']} {brow['p']}x{brow['q']} "
        f"q3min={brow['bits']['3']['min_axis']:.6f} "
        f"(base {base_bits[3]['min_axis']:.6f})",
        flush=True,
    )

    if refine:
        print(f"  refine amax2 from {brow['ctor']} {brow['p']}x{brow['q']}", flush=True)
        Ar, Br, hist = riemannian_refine(W, Ab, Bb, steps=8, lr=0.12, objective="amax2")
        ev = eval_pair(W, X, Ar, Br, bits_list, ref_orig)
        rrow = {
            "p": brow["p"],
            "q": brow["q"],
            "ctor": brow["ctor"] + "+amax2",
            "stats": ev["stats"],
            "bits": {str(b): {k: ev["bits"][b][k] for k in
                              ("observed", "probed", "worst_unit", "relF", "min_axis")}
                     for b in bits_list},
            "vs_own_q4": {str(b): ev["vs_own_q4"][b] for b in bits_list},
            "vs_orig_q4": {str(b): ev["vs_orig_q4"][b] for b in bits_list},
            "orthA": ev["orthA"],
            "orthB": ev["orthB"],
            "wall_s": ev["wall_s"],
            "hist": hist,
            "factor_bytes_f32": factor_bytes(brow["p"], brow["q"], 4),
            "bpw_64_f32": factor_bpw(brow["p"], brow["q"], 64, 4),
        }
        result["refined"] = rrow
        a3 = ev["bits"][3]
        print(
            f"    refined q3 min={a3['min_axis']:.6f} relF={a3['relF']:.6f} "
            f"kurt={ev['stats']['kurtosis']:+.4f} hist0={hist[0]['obj']:.6e} "
            f"histN={hist[-1]['obj']:.6e}",
            flush=True,
        )
        if a3["min_axis"] > brow["bits"]["3"]["min_axis"]:
            result["best"] = {k: v for k, v in rrow.items() if k != "hist"}
            Ab, Bb = Ar, Br
            print("    refine WINS, replacing best", flush=True)
        del ev

        # one STE refine from the same cheap init, keep if better
        print(f"  refine STE-q3 from {brow['ctor']}", flush=True)
        As, Bs, hist2 = riemannian_refine(W, polar_orth(Ab if False else best[2]), best[3],
                                          steps=6, lr=0.08, bits=3, objective="ste")
        ev = eval_pair(W, X, As, Bs, bits_list, ref_orig)
        srow = {
            "p": brow["p"],
            "q": brow["q"],
            "ctor": brow["ctor"] + "+ste",
            "stats": ev["stats"],
            "bits": {str(b): {k: ev["bits"][b][k] for k in
                              ("observed", "probed", "worst_unit", "relF", "min_axis")}
                     for b in bits_list},
            "vs_own_q4": {str(b): ev["vs_own_q4"][b] for b in bits_list},
            "vs_orig_q4": {str(b): ev["vs_orig_q4"][b] for b in bits_list},
            "orthA": ev["orthA"],
            "orthB": ev["orthB"],
            "hist": hist2,
        }
        result["refined_ste"] = srow
        print(
            f"    ste q3 min={ev['bits'][3]['min_axis']:.6f} relF={ev['bits'][3]['relF']:.6f}",
            flush=True,
        )
        if ev["bits"][3]["min_axis"] > result["best"]["bits"]["3"]["min_axis"]:
            result["best"] = {k: v for k, v in srow.items() if k != "hist"}
            Ab, Bb = As, Bs
            print("    STE WINS, replacing best", flush=True)
        del ev

    result["peak_rss_gb"] = rss_gb()
    # drop huge arrays from returned result; keep A,B for transfer test via caller
    return result, W, X, Ab, Bb


def bits_saved(base_bits, after_bits):
    """Largest k such that after at (4-k) matches or beats before at 4 on min_axis.
    Also report after[b] vs before[b] deltas. MEASURED comparison, not a projection.
    """
    out = {}
    for b in (2, 3, 4):
        out[str(b)] = {
            "d_min": after_bits[str(b)]["min_axis"] - base_bits[str(b)]["min_axis"],
            "d_obs": after_bits[str(b)]["observed"] - base_bits[str(b)]["observed"],
            "d_prb": after_bits[str(b)]["probed"] - base_bits[str(b)]["probed"],
            "d_wu": after_bits[str(b)]["worst_unit"] - base_bits[str(b)]["worst_unit"],
            "d_relF": after_bits[str(b)]["relF"] - base_bits[str(b)]["relF"],
        }
    # equal-function vs before-4: does after at b reach before-4 min_axis?
    saved = 0
    for b, expect_save in ((4, 0), (3, 1), (2, 2)):
        if after_bits[str(b)]["min_axis"] + 1e-12 >= base_bits["4"]["min_axis"]:
            saved = expect_save
    out["bits_saved_vs_q4_minaxis"] = saved
    # vs before at same width: did 3-bit after beat 3-bit before enough to match 4-bit before?
    out["after3_reaches_before4"] = after_bits["3"]["min_axis"] >= base_bits["4"]["min_axis"]
    out["after2_reaches_before3"] = after_bits["2"]["min_axis"] >= base_bits["3"]["min_axis"]
    return out


def main():
    t_all = time.time()
    sc = run_selfcheck()
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "smoke":
        tensors = [TENSORS[-1]]  # L31 v_proj
        shapes = [(64, 80), (40, 128), (20, 256)]
        refine = False
    else:
        tensors = TENSORS
        shapes = SHAPES_5120
        refine = True

    all_results = {"selfcheck": sc, "tensors": [], "started_unix": time.time(), "mode": mode}
    saved_AB = {}

    for layer, cls, family in tensors:
        r, W, X, A, B = run_tensor(layer, cls, family, shapes, refine=refine)
        if r.get("best"):
            r["equal_function"] = bits_saved(
                {str(b): r["base_bits"][str(b)] for b in (2, 3, 4)},
                r["best"]["bits"],
            )
            print(f"  EQUAL-FUNCTION {json.dumps(fnum(r['equal_function']))}", flush=True)
        all_results["tensors"].append(fnum(r))
        if A is not None:
            saved_AB[(layer, cls)] = (A, B, W, X)
        path = os.path.join(OUT_DIR, f"partial_{layer}_{cls.replace('.', '_')}.json")
        json.dump(fnum(r), open(path, "w"), indent=2)
        print(f"  wrote {path}  peak_rss={r['peak_rss_gb']:.3f}GB", flush=True)
        del W, X

    # transfer: L31 gate M applied to L31 q and L0 gate, if both exist
    transfers = []
    key_src = (31, "mlp.gate_proj")
    if key_src in saved_AB:
        As, Bs, _, _ = saved_AB[key_src]
        p, q = As.shape[0], Bs.shape[0]
        for key_dst in ((31, "self_attn.q_proj"), (0, "mlp.gate_proj"), (63, "mlp.gate_proj")):
            if key_dst not in saved_AB:
                continue
            print(f"\nTRANSFER M(L31 gate {p}x{q}) -> L{key_dst[0]} {key_dst[1]}", flush=True)
            W = load_tensor(f"language_model.model.layers.{key_dst[0]}.{key_dst[1]}.weight")
            X = load_X(key_dst[0])
            base = score_bits(W, X, (2, 3, 4))
            ev = eval_pair(W, X, As, Bs, (2, 3, 4), base[4])
            row = {
                "src": "L31.mlp.gate_proj",
                "dst": f"L{key_dst[0]}.{key_dst[1]}",
                "p": p,
                "q": q,
                "base_q3_min": base[3]["min_axis"],
                "xfer_q3_min": ev["bits"][3]["min_axis"],
                "d_min": ev["bits"][3]["min_axis"] - base[3]["min_axis"],
                "base_q3_relF": base[3]["relF"],
                "xfer_q3_relF": ev["bits"][3]["relF"],
                "xfer_stats": ev["stats"],
            }
            print(
                f"  q3 min {base[3]['min_axis']:.6f} -> {ev['bits'][3]['min_axis']:.6f} "
                f"({row['d_min']:+.6f}) relF {base[3]['relF']:.6f} -> {ev['bits'][3]['relF']:.6f}",
                flush=True,
            )
            transfers.append(fnum(row))
            del W, X, ev
    all_results["transfer"] = transfers
    all_results["wall_s"] = time.time() - t_all
    all_results["peak_rss_gb"] = rss_gb()
    all_results["axis_margin"] = AXIS_MARGIN
    all_results["group"] = GROUP
    all_results["n_source"] = N_SOURCE
    all_results["shapes_searched"] = SHAPES_5120
    all_results["source_root"] = ROOT

    outp = os.path.join(OUT_DIR, "results.json")
    json.dump(fnum(all_results), open(outp, "w"), indent=2)
    print(f"\nWROTE {outp} wall={all_results['wall_s']:.1f}s peak_rss={all_results['peak_rss_gb']:.3f}GB", flush=True)


if __name__ == "__main__":
    main()
