#!/usr/bin/env python3
"""Reframe oracle — data-AWARE low-rank for §8.1 L1.4 (and the L1.3 cross-layer
reference as a free add-on). CPU-only NumPy. DO NOT run while a capture/training
job holds the GPU/CPU (this script only touches CPU + mmap, but the kill
discipline is: don't contend).

WHY THIS EXISTS (the reframe being tested)
------------------------------------------
The original L1.4 oracle (`tools/bench/oracle_lowrank_codebook.py`) and the L1.3
oracle (`tools/bench/oracle_interlayer_delta.py`) both took a **data-free** SVD
of the raw weight matrix and measured **Frobenius** energy@r. They concluded the
weights are "not low-rank" (top-64 captures 3-9% FFN / ~26% attn of Frobenius
energy) and killed L1.4/L1.3.

But Frobenius energy is the WRONG objective for an inference codec. What matters
is not ||W - W_r||_F, it is ||(W - W_r) x|| on the ACTIVATIONS the weight
actually sees. Activations live in a low-dimensional subspace, so a weight can be
near-full-rank in Frobenius yet effectively low-rank IN THE NORM INDUCED BY THE
DATA. This is exactly the gap that activation-aware SVD methods (ASVD, SVD-LLM,
FWSVD) exploit; the original oracles never tested it. So L1.4/L1.3 died in the
data-FREE form; this script tests the data-AWARE form, which is the standard
state-of-the-art improvement and the legitimate Type-2 reframe.

WHAT IT MEASURES (per sampled FFN layer; gate_proj + up_proj only)
------------------------------------------------------------------
Input activation x for gate_proj / up_proj is exactly the captured `norm_in`
(the ffn_norm output), so the empirical activation matrix X is available with no
reconstruction. (down_proj's input is the 11008-dim SwiGLU intermediate, which is
reconstructable — see oracle_coactivation_permute.py — but heavier; left as a
documented extension. gate+up are 2 of the 3 FFN weight tensors and carry the
reframe signal.)

  Treat the dequantized Q4_K weight W as the target (SAME choice as the original
  oracles — we only have the Q4_K_M GGUF, not the f16 master; a NO-GO here is
  decisive, a marginal GO must defer to the f16/AWQ lane per the W4A8 note).

  L1.4 reframe (data-aware low-rank of W itself):
    C   = X^T X / N                  (n x n activation second moment, n=hidden)
    M   = W @ C^{1/2}                 (data-whitened weight)
    SVD(M) -> data-weighted singular values; energy@r in the DATA norm.
    W_r = (U_r S_r) @ (V_r^T C^{-1/2})  (rank-r approx optimal for ||(W-W_r)X^T||)
    Codec: store U_r,(S_r V_r^T C^{-1/2}) at f16 + residual (W-W_r) at b bits.
    Headline comparison: data-aware energy@r  vs  plain (Frobenius) energy@r
      -> did the original oracle UNDERSELL low-rank by using Frobenius?
    Decisive comparison: FUNCTIONAL error ||(W - What) X^T||_F / ||W X^T||_F of
      the FULL codec at <= Q4_K bytes, vs plain-Q4_K's own functional error 0
      (W is already the Q4_K recon, so the bar is: can the codec re-encode W at
      FEWER bytes than its current Q4_K footprint while keeping functional error
      small?). GO only if some (r,b) with bytes < Q4_K has tiny functional error.

  L1.3 reframe (does the already-resident W[L] give W[L+1] a free basis?):
    For consecutive pairs, in the data norm of X=norm_in[L+1], measure
    energy@r of the DELTA D=W[L+1]-W[L]. If the rank needed for the delta is no
    smaller than for W[L+1] alone (the L1.4 number), W[L] is a useless basis ->
    cross-layer reference adds nothing even data-aware. Also reports the
    data-weighted cross-layer cosine cos((W[L]X^T),(W[L+1]... )) proxy.

DECISION
  GO    if data-aware low-rank+residual beats Q4_K bytes at near-zero functional
        error for the MAJORITY of sampled FFN tensors (then the kill was Type-2
        and this codec earns the GPU/quality lane on f16 weights).
  NO-GO if even the data-aware form keeps ~all the functional energy in the
        residual at <Q4_K bytes (then L1.4 is Type-1: not low-rank even on the
        data manifold), and the L1.3 delta rank is no better than W alone.

RAM discipline: one weight tensor + one capture-layer resident at a time;
del + gc.collect() between; RSS ceiling 3 GB. Pure numpy (no scipy).

Run (ONLY when no capture/training job is active):
    /tmp/ggufenv/bin/python tools/bench/oracle_dataaware_lowrank.py \
        --gguf models/qwen2.5-3b-instruct-q4_k_m.gguf \
        --bin  _capture/q3b_ffn.bin \
        --out  reports/oracle_dataaware_lowrank.md
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

try:
    import resource

    def rss_gb():
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / (1024 ** 3) if sys.platform == "darwin" else kb / (1024 ** 2)
except Exception:  # pragma: no cover
    def rss_gb():
        return float("nan")

from gguf import GGUFReader
from gguf.quants import dequantize

SENTINEL = 0xFFFFFFFF
QK_K = 256
RANKS = (16, 32, 64)
RES_BITS = (2, 3)
RSS_CEIL_GB = 3.0
# Functional-error threshold below which a re-encoding is "quality-neutral
# enough to advance" (data-weighted relative L2 on real activations). 0.02 mirrors
# the spirit of the tight parity regime; the GPU lane re-checks with real KL.
FUNC_ERR_GATE = 0.02


def check_rss(where):
    g = rss_gb()
    if g > RSS_CEIL_GB:
        sys.stderr.write(f"[FATAL] RSS {g:.2f} GB > {RSS_CEIL_GB} at {where}\n")
        sys.exit(2)
    return g


# --------------------------------------------------------------------------
# Capture reader (per-layer norm_in[N,hidden]); identical framing to
# oracle_coactivation_permute.load_capture, but we only need norm_in.
# --------------------------------------------------------------------------
def load_norm_in(path: Path):
    data = Path(path).read_bytes()
    n = len(data)
    off = 0
    hidden = n_blocks = None
    acc: dict[int, list] = {}
    while off + 8 <= n:
        a, b = struct.unpack_from("<II", data, off)
        if a == SENTINEL and b == SENTINEL:
            if off + 16 > n:
                break
            _, _, hidden, n_blocks = struct.unpack_from("<IIII", data, off)
            hidden, n_blocks = int(hidden), int(n_blocks)
            off += 16
            continue
        if hidden is None:
            raise ValueError("stream did not start with a sentinel")
        rec = 4 + hidden * 4 + 2 * n_blocks * 4
        if off + rec > n:
            break
        (layer,) = struct.unpack_from("<I", data, off)
        foff = off + 4
        norm = np.frombuffer(data, np.float32, hidden, foff)
        acc.setdefault(int(layer), []).append(norm.copy())
        off += rec
    return hidden, n_blocks, {k: np.stack(v) for k, v in acc.items()}


def w_of(reader_by_name, name):
    """Dequantize a GGUF tensor to f32 [rows, cols] = [out_features, in_features]."""
    t = reader_by_name[name]
    W = dequantize(np.array(t.data), t.tensor_type).astype(np.float32)
    dims = tuple(int(x) for x in t.shape)  # gguf fastest-first = (in, out)
    return W.reshape(dims[1], dims[0]), t.tensor_type.name, int(t.n_bytes)


# --------------------------------------------------------------------------
# Linear algebra helpers (pure numpy)
# --------------------------------------------------------------------------
def csqrt_and_inv(C, eps=1e-6):
    """Symmetric PSD square-root and its inverse from eigendecomposition."""
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 0.0, None)
    s = np.sqrt(w + eps)
    Csqrt = (V * s) @ V.T
    Cinv = (V * (1.0 / s)) @ V.T
    return Csqrt.astype(np.float32), Cinv.astype(np.float32)


def topr_energy(sv, ranks):
    """Fraction of energy (sum of squared singular values) in top-r."""
    e = sv.astype(np.float64) ** 2
    tot = float(e.sum()) + 1e-30
    cum = np.cumsum(e) / tot
    return {r: float(cum[min(r, len(e)) - 1]) for r in ranks}


def uniform_quant_resid(R, bits):
    """Per-row symmetric uniform quantize/dequantize of a residual matrix.
    Models the '2-3 bit residual + per-row f16 scale' budget the byte oracle used.
    Returns the dequantized residual (same shape)."""
    levels = (1 << (bits - 1)) - 1  # symmetric: e.g. b=3 -> +/-3
    amax = np.maximum(np.abs(R).max(axis=1, keepdims=True), 1e-12)
    scale = amax / levels
    q = np.clip(np.round(R / scale), -levels, levels)
    return (q * scale).astype(np.float32)


def lowrank_bytes(m, n, r, res_bits):
    """f16 U,V + residual @ res_bits + f16 per-row residual scale."""
    uv = 2 * (m * r + r * n)
    res = (res_bits * m * n) / 8.0
    res_scale = 2 * m
    return uv + res + res_scale


def func_rel_err(W, What, X):
    """Data-weighted relative error ||(W-What) X^T||_F / ||W X^T||_F on real X."""
    num = np.linalg.norm((W - What) @ X.T)
    den = np.linalg.norm(W @ X.T) + 1e-30
    return float(num / den)


# --------------------------------------------------------------------------
# Per-tensor data-aware low-rank analysis (L1.4 reframe)
# --------------------------------------------------------------------------
def analyze_tensor_l14(W, X, disk_bytes, Csqrt, Cinv):
    m, n = W.shape  # [out, in]
    # plain (Frobenius) SVD energy — reproduce the ORIGINAL oracle's number.
    sv_plain = np.linalg.svd(W, compute_uv=False)
    e_plain = topr_energy(sv_plain, RANKS)
    del sv_plain

    # data-aware SVD of M = W @ C^{1/2}
    M = W @ Csqrt
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    del M
    e_data = topr_energy(S, RANKS)

    rows = []
    for r in RANKS:
        rr = min(r, S.size)
        # W_r optimal for the data norm: (U_r S_r)(V_r^T C^{-1/2})
        A = U[:, :rr] * S[:rr]            # [m, rr]
        B = Vt[:rr, :] @ Cinv             # [rr, n]
        Wr = A @ B
        resid = W - Wr
        res_std_ratio = float(resid.std() / (W.std() + 1e-30))
        for b in RES_BITS:
            rq = uniform_quant_resid(resid, b)
            What = Wr + rq
            ferr = func_rel_err(W, What, X)
            tot_bytes = lowrank_bytes(m, n, rr, b)
            rows.append(dict(r=rr, bits=b,
                             energy_data=e_data[r], energy_plain=e_plain[r],
                             res_std_ratio=res_std_ratio,
                             bytes_ratio=tot_bytes / disk_bytes,
                             func_err=ferr))
            del rq, What
        del A, B, Wr, resid
        gc.collect()
    del U, S, Vt
    gc.collect()
    return rows, e_plain, e_data


# --------------------------------------------------------------------------
# Cross-layer delta data-aware analysis (L1.3 reframe)
# --------------------------------------------------------------------------
def analyze_pair_l13(W_L, W_Lp1, X_Lp1, Csqrt):
    """Does the resident W[L] give W[L+1] a cheap basis in the data norm?
    Compare data-aware energy@r of the DELTA vs of W[L+1] alone."""
    D = W_Lp1 - W_L
    Md = D @ Csqrt
    svd = np.linalg.svd(Md, compute_uv=False)
    e_delta = topr_energy(svd, RANKS)
    del Md, svd
    Mw = W_Lp1 @ Csqrt
    svw = np.linalg.svd(Mw, compute_uv=False)
    e_w = topr_energy(svw, RANKS)
    del Mw, svw
    # data-weighted cross-layer cosine of the actions on real X
    yL = W_L @ X_Lp1.T
    yP = W_Lp1 @ X_Lp1.T
    cos = float((yL.ravel() @ yP.ravel()) /
                (np.linalg.norm(yL) * np.linalg.norm(yP) + 1e-30))
    del yL, yP, D
    gc.collect()
    return e_delta, e_w, cos


def main():
    ap = argparse.ArgumentParser(prog="oracle_dataaware_lowrank")
    ap.add_argument("--gguf", default="models/qwen2.5-3b-instruct-q4_k_m.gguf")
    ap.add_argument("--bin", default="_capture/q3b_ffn.bin")
    ap.add_argument("--out", default="reports/oracle_dataaware_lowrank.md")
    ap.add_argument("--layers", default=None,
                    help="comma list (default: early/mid/late sample)")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--tensors", default="ffn_gate,ffn_up",
                    help="FFN tensors whose input == norm_in (gate/up)")
    args = ap.parse_args()
    t0 = time.time()

    print(f"[oracle] reading capture {args.bin} ...", flush=True)
    hidden, n_blocks, norm = load_norm_in(Path(args.bin))
    layers_all = sorted(norm.keys())
    print(f"[oracle] hidden={hidden} layers={len(layers_all)} "
          f"tokens/layer={norm[layers_all[0]].shape[0]}", flush=True)

    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        idx = sorted(set([0, len(layers_all) // 2, len(layers_all) - 1]))
        layers = [layers_all[i] for i in idx]
    tnames = args.tensors.split(",")

    print(f"[oracle] reading GGUF {args.gguf} ...", flush=True)
    reader = GGUFReader(args.gguf)
    by_name = {t.name: t for t in reader.tensors}

    l14_results = {}   # (layer, tname) -> rows
    l14_energy = {}    # (layer, tname) -> (e_plain, e_data)
    l13_results = {}   # (L, tname) -> (e_delta, e_w, cos)
    max_rss = rss_gb()

    for layer in layers:
        X = norm[layer]
        if X.shape[0] > args.max_tokens:
            sel = np.linspace(0, X.shape[0] - 1, args.max_tokens).astype(int)
            X = X[sel]
        C = (X.T @ X) / X.shape[0]
        Csqrt, Cinv = csqrt_and_inv(C)
        del C
        for tname in tnames:
            nm = f"blk.{layer}.{tname}.weight"
            if nm not in by_name:
                continue
            W, qtype, disk_bytes = w_of(by_name, nm)
            rows, e_plain, e_data = analyze_tensor_l14(W, X, disk_bytes, Csqrt, Cinv)
            l14_results[(layer, tname)] = (rows, qtype)
            l14_energy[(layer, tname)] = (e_plain, e_data)
            max_rss = max(max_rss, check_rss(nm))
            best = min(rows, key=lambda r: (r["func_err"]
                                            if r["bytes_ratio"] < 1.0 else 9e9))
            print(f"[oracle] L{layer:2d} {tname:8s} {qtype} "
                  f"E64 plain={e_plain[64]:.3f} data={e_data[64]:.3f} | "
                  f"best<Q4K r{best['r']}/{best['bits']}b "
                  f"bytes={best['bytes_ratio']:.2f}x ferr={best['func_err']:.4f} "
                  f"| rss={rss_gb():.2f}G", flush=True)
            del W
            gc.collect()

        # L1.3: pair this layer with the next (if both captured & adjacent).
        nxt = layer + 1
        if nxt in norm:
            Xp = norm[nxt]
            if Xp.shape[0] > args.max_tokens:
                sel = np.linspace(0, Xp.shape[0] - 1, args.max_tokens).astype(int)
                Xp = Xp[sel]
            Cp = (Xp.T @ Xp) / Xp.shape[0]
            Csqrt_p, _ = csqrt_and_inv(Cp)
            del Cp
            for tname in tnames:
                a = f"blk.{layer}.{tname}.weight"
                bnm = f"blk.{nxt}.{tname}.weight"
                if a not in by_name or bnm not in by_name:
                    continue
                W_L, _, _ = w_of(by_name, a)
                W_P, _, _ = w_of(by_name, bnm)
                e_delta, e_w, cos = analyze_pair_l13(W_L, W_P, Xp, Csqrt_p)
                l13_results[(layer, tname)] = (e_delta, e_w, cos)
                print(f"[oracle] L{layer}->{nxt} {tname:8s} dw-cos={cos:+.4f} "
                      f"E64(delta)={e_delta[64]:.3f} E64(W)={e_w[64]:.3f}",
                      flush=True)
                del W_L, W_P
                gc.collect()
            del Csqrt_p
        del Csqrt, Cinv
        gc.collect()

    # ---------------- verdict ----------------
    # L1.4 GO if a majority of (layer,tensor) have some <Q4K-byte config with
    # functional error <= gate.
    keys = list(l14_results.keys())
    passed = 0
    per_key_best = {}
    for k in keys:
        rows, _ = l14_results[k]
        cands = [r for r in rows if r["bytes_ratio"] < 1.0 and r["func_err"] <= FUNC_ERR_GATE]
        best = min(rows, key=lambda r: r["func_err"]) if rows else None
        per_key_best[k] = (cands[0] if cands else best)
        if cands:
            passed += 1
    l14_go = passed > len(keys) / 2 if keys else False

    # L1.3 GO if the delta needs materially LESS rank than W alone (W[L] helps).
    l13_helps = 0
    for k, (e_delta, e_w, cos) in l13_results.items():
        if e_delta[64] > e_w[64] + 0.10:  # delta noticeably more concentrated
            l13_helps += 1
    l13_go = l13_helps > len(l13_results) / 2 if l13_results else False

    overall = "GO" if l14_go else "NO-GO"

    # ---------------- report ----------------
    L = []
    P = L.append
    P("# Reframe oracle — data-AWARE low-rank (L1.4) + cross-layer reference (L1.3)")
    P("")
    P(f"**L1.4 verdict (data-aware): {overall}**  ")
    P(f"**L1.3 verdict (cross-layer, data-aware): {'GO' if l13_go else 'NO-GO'}**")
    P("")
    P(f"- Model: `{args.gguf}` | Capture: `{args.bin}`")
    P(f"- Sampled layers: {layers} | tensors: {tnames} (input == norm_in)")
    P(f"- Tokens/layer (cap): {args.max_tokens} | Peak RSS: {max_rss:.2f} GB | "
      f"Wall: {time.time()-t0:.1f}s")
    P(f"- Functional-error gate (data-weighted rel-L2): {FUNC_ERR_GATE}")
    P("")
    P("## The reframe being tested")
    P("")
    P("The original L1.4/L1.3 oracles used a **data-free** SVD and **Frobenius** "
      "energy. This re-tests with **activation-aware** SVD (SVD on `W·C^{1/2}`, "
      "`C=E[xx^T]` from the real capture) — the standard fix (ASVD/SVD-LLM) the "
      "originals never ran. Weights can be full-rank in Frobenius yet low-rank in "
      "the data norm.")
    P("")
    P("## L1.4 — data-aware vs data-free low-rank energy, and the byte/quality budget")
    P("")
    P("`E64 plain` = original Frobenius top-64 energy (reproduced). `E64 data` = "
      "activation-aware top-64 energy. If `data` >> `plain`, the original oracle "
      "undersold low-rank.")
    P("")
    P("| layer | tensor | type | E64 plain | E64 data | best<Q4K (r/bits) | bytes | func-err |")
    P("|------:|--------|------|----------:|---------:|-------------------|------:|---------:|")
    for k in keys:
        layer, tname = k
        e_plain, e_data = l14_energy[k]
        _, qtype = l14_results[k]
        b = per_key_best[k]
        bcfg = f"r{b['r']}/{b['bits']}b" if b else "-"
        P(f"| {layer} | {tname} | {qtype} | {e_plain[64]:.3f} | {e_data[64]:.3f} | "
          f"{bcfg} | {b['bytes_ratio']:.2f}x | {b['func_err']:.4f} |")
    P("")
    P(f"**L1.4:** {passed}/{len(keys)} sampled FFN tensors have a <Q4_K-byte "
      f"config with data-weighted functional error <= {FUNC_ERR_GATE}. " +
      ("Majority clear it — the data-free kill was premature (Type-2); advance "
       "this codec to the GPU/quality lane on **f16** weights (the Q4_K re-encode "
       "here is a lower bound; real gains need AWQ-from-f16)."
       if l14_go else
       "Majority do NOT clear it — even activation-aware, the residual keeps ~all "
       "the functional energy at <Q4_K bytes, so the f16 U,V stay dead overhead. "
       "L1.4 is Type-1: these weights are not low-rank even on the data manifold."))
    P("")
    P("## L1.3 — does the resident W[L] give W[L+1] a free basis (data-aware)?")
    P("")
    P("| L→L+1 | tensor | data-wt cross-layer cos | E64(delta) | E64(W[L+1]) | W[L] helps? |")
    P("|-------|--------|------------------------:|-----------:|------------:|:-----------:|")
    for (layer, tname), (e_delta, e_w, cos) in l13_results.items():
        helps = "yes" if e_delta[64] > e_w[64] + 0.10 else "no"
        P(f"| {layer}→{layer+1} | {tname} | {cos:+.4f} | {e_delta[64]:.3f} | "
          f"{e_w[64]:.3f} | {helps} |")
    P("")
    P("**L1.3:** " + (
        "the delta concentrates materially more data-weighted energy than W[L+1] "
        "alone in a majority of pairs — W[L] is a useful free basis; a data-aware "
        "cross-layer codec is worth a prototype."
        if l13_go else
        "the delta is no more concentrated than W[L+1] alone (and the data-weighted "
        "cross-layer cosine is ~0), so the resident W[L] provides no free basis — "
        "cross-layer reference adds nothing over plain L1.4 even in the data norm. "
        "Type-1: layers are functionally independent, not just weight-orthogonal."))
    P("")
    P("## Honest caveats")
    P("")
    P("- Target W is the **dequantized Q4_K** weight (only the Q4_K_M GGUF is "
      "available offline), so this re-encodes an already-quantized weight — a "
      "**lower bound** on what AWQ-from-f16 could achieve. A NO-GO is decisive; a "
      "marginal GO must be re-checked on f16 in the GPU/quality lane with real KL.")
    P("- Functional error is data-weighted rel-L2 on the captured activations, a "
      "proxy for end-task quality (same proxy family as the Track-B / co-activation "
      "gates), not perplexity.")
    P("- Scope: gate_proj + up_proj (input == norm_in, captured directly). "
      "down_proj (input = 11008-dim SwiGLU intermediate) is reconstructable "
      "(see oracle_coactivation_permute.py) but omitted here; add it if gate/up "
      "show a GO.")
    P("- best-case ORACLE: the per-row residual quantizer is idealized; a real "
      "GPU codec is no better.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")

    sidecar = out.with_suffix(".json")
    json.dump({
        "l14_verdict": overall,
        "l13_verdict": "GO" if l13_go else "NO-GO",
        "func_err_gate": FUNC_ERR_GATE,
        "l14_pass": passed, "l14_total": len(keys),
        "layers": layers, "tensors": tnames,
        "l14": {f"{k[0]}:{k[1]}": {
            "E64_plain": l14_energy[k][0][64], "E64_data": l14_energy[k][1][64],
            "rows": l14_results[k][0]} for k in keys},
        "l13": {f"{k[0]}:{k[1]}": {"cos": v[2],
                                   "E64_delta": v[0][64], "E64_W": v[1][64]}
                for k, v in l13_results.items()},
        "peak_rss_gb": max_rss,
    }, open(sidecar, "w"), indent=2)

    print(f"\n[oracle] L1.4 {overall} ({passed}/{len(keys)})  "
          f"L1.3 {'GO' if l13_go else 'NO-GO'}")
    print(f"[oracle] wrote {out} and {sidecar} ({time.time()-t0:.1f}s, "
          f"peak RSS {max_rss:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
