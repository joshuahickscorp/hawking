#!/usr/bin/env python3
"""The group scale is a free variable, and absmax is the worst possible choice for entropy.

The packer sets each group's scale to absmax/bound. That guarantees at least one element of
every group lands on the extreme code, so the code range is ALWAYS full. Measured consequence
(gravity_parallel_code): fixed-bits-per-tile, the simplest parallel-decodable form, is dead at
ratio 1.0003 -- not because the weights lack structure but because the SCALING CHOICE forbids
any tile from being narrow.

Absmax is also not the best scale for fidelity. It is chosen for being cheap and safe.

So the scale is a free variable with two payoffs at IDENTICAL stored bytes and an IDENTICAL
container format, meaning the existing kernel consumes it unchanged:
    fidelity  -- an MSE-optimal scale minimises reconstruction error, not range
    entropy   -- clipping the tail concentrates the code histogram

Compared here on the real BF16 tensors, at group 64, q3:
    absmax        s = max|w| / bound                 (the incumbent)
    mse           s minimising ||w - s*round(w/s)||  by a 1-D search over the ratio to absmax
Scored on the four-axis adequacy gate against a same-tensor honest Q4 reference, plus the
rANS code length of the resulting symbols under a 12-bit quantized shared model.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gravity_doctor_gate import load_tensor, load_X, axes, c_uniform  # noqa: E402
from gravity_parallel_code import quantized_freqs, rans_bits          # noqa: E402

GROUP = 64


def quant_absmax(W, bits, group=GROUP):
    m, n = W.shape
    g = W.reshape(-1, group)
    bound = (1 << (bits - 1)) - 1
    s = np.abs(g).max(1, keepdims=True) / bound
    s = np.maximum(s, 1e-30).astype(np.float16).astype(np.float32)
    q = np.clip(np.rint(g / s), -bound, bound)
    return (q * s).reshape(m, n), (q + bound).astype(np.uint16).ravel()


def quant_mse(W, bits, group=GROUP, ratios=None):
    """Same format, same bytes: only the f16 scale value differs."""
    m, n = W.shape
    g = W.reshape(-1, group)
    bound = (1 << (bits - 1)) - 1
    amax = np.abs(g).max(1, keepdims=True)
    ratios = ratios if ratios is not None else np.linspace(0.55, 1.0, 19)
    best_err = None
    best_s = None
    for r in ratios:
        s = np.maximum(amax * r / bound, 1e-30).astype(np.float16).astype(np.float32)
        q = np.clip(np.rint(g / s), -bound, bound)
        err = ((g - q * s) ** 2).sum(1, keepdims=True)
        if best_err is None:
            best_err, best_s = err, s
        else:
            take = err < best_err
            best_err = np.where(take, err, best_err)
            best_s = np.where(take, s, best_s)
    q = np.clip(np.rint(g / best_s), -bound, bound)
    return (q * best_s).reshape(m, n), (q + bound).astype(np.uint16).ravel()


def code_bits(sym, prec=12):
    h = np.bincount(sym, minlength=int(sym.max()) + 1)
    return rans_bits(h, quantized_freqs(h, prec)) / sym.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--sites", default="mlp.gate_proj@31,mlp.down_proj@31,"
                                       "mlp.up_proj@47,self_attn.q_proj@31,"
                                       "linear_attn.out_proj@30,mlp.gate_proj@63")
    a = ap.parse_args()
    print(f"q{a.bits} group {GROUP}. Same format, same stored bytes, same kernel: only the\n"
          f"f16 scale value differs between the two rows of each pair.\n")
    print(f"{'site':<26}{'scale':<8}{'observed':>10}{'probed':>10}{'worst_u':>10}"
          f"{'gain':>10}{'code b/w':>10}{'range':>7}")
    tot = {"absmax": [0.0, 0], "mse": [0.0, 0]}
    for spec in a.sites.split(","):
        cls, l = spec.split("@"); l = int(l)
        name = f"language_model.model.layers.{l}.{cls}.weight"
        try:
            W = load_tensor(name).astype(np.float32)
        except Exception as e:
            print(f"{spec:<26} SKIP {e}"); continue
        X = load_X(l) if W.shape[1] == 5120 else None
        if X is None:
            X = np.random.default_rng(l).standard_normal((256, W.shape[1])).astype(np.float32)
        ref = axes(W, c_uniform(W, 4, GROUP), X, seed=None)
        print(f"{spec:<26}{'q4 ref':<8}{ref['observed']:>10.6f}{ref['probed']:>10.6f}"
              f"{ref['worst_unit']:>10.6f}{ref['gain']:>10.6f}{'':>10}{'':>7}")
        for tag, fn in (("absmax", quant_absmax), ("mse", quant_mse)):
            Wh, sym = fn(W, a.bits)
            ax = axes(W, Wh, X, seed=None)
            cb = code_bits(sym)
            rng_full = int(sym.max()) - int(sym.min()) + 1
            tot[tag][0] += cb * sym.size; tot[tag][1] += sym.size
            print(f"{'':<26}{tag:<8}{ax['observed']:>10.6f}{ax['probed']:>10.6f}"
                  f"{ax['worst_unit']:>10.6f}{ax['gain']:>10.6f}{cb:>10.4f}{rng_full:>7}")
    print(f"\nweighted code cost over all sites:")
    for tag in ("absmax", "mse"):
        b = tot[tag][0] / tot[tag][1]
        print(f"  {tag:<8}{b:.4f} bits/weight coded  (+ {8*2/GROUP:.4f} scale = "
              f"{b + 8*2/GROUP:.4f} complete per element)")
    d = tot["absmax"][0] / tot["absmax"][1] - tot["mse"][0] / tot["mse"][1]
    print(f"  mse is {d:+.4f} bits/weight vs absmax on the coded stream")
    return 0


if __name__ == "__main__":
    sys.exit(main())
