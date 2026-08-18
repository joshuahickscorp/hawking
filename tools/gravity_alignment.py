#!/usr/bin/env python3
"""Is cross-layer structure absent, or merely hidden by learned coordinates?

Every prior sharing test on this model compared tensors in their native bases and
found ~0 similarity. That is not evidence of absence: a learned basis is arbitrary
up to permutation, sign, per-channel scale and rotation, so two structurally
identical matrices can look unrelated. And the sub-1 bound now makes this
load-bearing -- per-weight codes floor at 1.125 BPW, so sharing is not optional.

Cheapest decisive test first (ULTRA CORE 5). SINGULAR VALUES ARE INVARIANT to any
orthogonal transform on either side and to any permutation. So:

  spectra differ  -> NO alignment can make these tensors share. Family closed, cheaply.
  spectra agree   -> alignment MIGHT work, and only then is it worth paying for the
                     expensive search.

That single property decides the family without ever running a matching algorithm.

Alignment cost is priced, because sharing that costs more to describe than it saves
is not sharing:
  permutation of n channels   log2(n!) bits          5120 -> ~6.8 KB per site
  sign flips                  n bits                 5120 -> 640 B per site
  per-channel scale (f16)     2n bytes               5120 -> 10 KB per site
  dense orthogonal (f16)      2*n*n bytes            5120 -> 52.4 MB per site == 0.998 BPW
Only the first three are affordable. A dense rotation costs the entire sub-1 budget,
so it is excluded by economics regardless of how well it aligns.
"""
from __future__ import annotations
import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gravity_doctor_gate import load_tensor  # noqa: E402

SOURCE_PARAM_COUNT = 26_895_998_464


def align_costs(n_in, n_out):
    perm_bits = math.lgamma(n_in + 1) / math.log(2)
    return {
        "permutation_bytes": perm_bits / 8,
        "sign_bytes": n_in / 8,
        "channel_scale_bytes": 2 * n_in,
        "dense_orthogonal_bytes": 2 * n_in * n_in,
    }


def spectra(W, k=256):
    s = np.linalg.svd(W.astype(np.float32), compute_uv=False)
    s = s[:k]
    return s / (s[0] + 1e-30)


def spectrum_agreement(a, b):
    """Cosine between normalised singular-value profiles, and worst relative gap."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    rel = float(np.max(np.abs(a - b) / (np.maximum(a, b) + 1e-9)))
    return cos, rel


def canon(W):
    """Permutation-canonical form: sort columns and rows by norm, fix signs.

    If two matrices are permutations of one another this makes them equal. It is a
    proxy for the full matching problem, which is O(n^3) at n=5120 and not worth
    paying before the spectra say alignment is even possible.
    """
    W = W.copy()
    ci = np.argsort(-np.linalg.norm(W, axis=0))
    W = W[:, ci]
    ri = np.argsort(-np.linalg.norm(W, axis=1))
    W = W[ri]
    sgn = np.sign(W.sum(axis=0) + 1e-30)
    return W * sgn


def flatcos(A, B):
    a, b = A.ravel(), B.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def compare(cls, la, lb, k=256):
    A = load_tensor(f"language_model.model.layers.{la}.{cls}.weight")
    B = load_tensor(f"language_model.model.layers.{lb}.{cls}.weight")
    if A.shape != B.shape:
        return None
    sa, sb = spectra(A, k), spectra(B, k)
    scos, srel = spectrum_agreement(sa, sb)
    raw = flatcos(A, B)
    can = flatcos(canon(A), canon(B))
    # a permutation-invariant floor: how similar are two INDEPENDENT random matrices
    # of the same shape after the same canonicalisation? anything at or below this is noise
    rng = np.random.default_rng(0)
    R1 = rng.standard_normal(A.shape).astype(np.float32) * float(A.std())
    R2 = rng.standard_normal(A.shape).astype(np.float32) * float(A.std())
    null = flatcos(canon(R1), canon(R2))
    sr = spectra(R1, k)
    null_scos, _ = spectrum_agreement(sr, spectra(R2, k))
    return {"cls": cls, "pair": (la, lb), "shape": tuple(A.shape),
            "raw_cos": raw, "canon_cos": can, "canon_null": null,
            "spectrum_cos": scos, "spectrum_worst_rel": srel,
            "spectrum_null": null_scos,
            "costs": align_costs(A.shape[1], A.shape[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="mlp.gate_proj:30,31;mlp.gate_proj:15,47;"
                                       "mlp.down_proj:30,31;self_attn.q_proj:31,35;"
                                       "self_attn.v_proj:31,35")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    print(f"{'class':<20} {'pair':>10} {'raw':>8} {'canon':>8} {'canon_null':>11} "
          f"{'spec_cos':>9} {'spec_null':>10} {'worst_rel':>10}")
    out = []
    for spec in a.pairs.split(";"):
        cls, ls = spec.split(":")
        la, lb = (int(x) for x in ls.split(","))
        r = compare(cls, la, lb)
        if r is None:
            print(f"{cls:<20} {str((la,lb)):>10}  shape mismatch, skipped")
            continue
        out.append(r)
        print(f"{cls:<20} {str((la,lb)):>10} {r['raw_cos']:>8.4f} {r['canon_cos']:>8.4f} "
              f"{r['canon_null']:>11.4f} {r['spectrum_cos']:>9.5f} {r['spectrum_null']:>10.5f} "
              f"{r['spectrum_worst_rel']:>10.4f}")

    if out:
        c = out[0]["costs"]
        n = out[0]["shape"][1]
        print(f"\nalignment description cost for n_in={n}, per site:")
        for k, v in c.items():
            print(f"  {k:<26} {v:>14,.0f} B   = {8*v*64/SOURCE_PARAM_COUNT:.6f} BPW over 64 sites")
        print("  dense orthogonal alone exceeds the entire sub-1 budget -> excluded by economics")

        sp = [r["spectrum_cos"] for r in out]
        nl = [r["spectrum_null"] for r in out]
        print(f"\nspectrum agreement {min(sp):.5f}-{max(sp):.5f} vs random-pair null "
              f"{min(nl):.5f}-{max(nl):.5f}")
        if min(sp) - max(nl) > 0.02:
            print("VERDICT: spectra agree beyond the null -> alignment is NOT ruled out; "
                  "an affordable-transform search is justified")
        else:
            print("VERDICT: spectra do not separate from the random null -> no orthogonal or "
                  "permutation alignment can make these share. Family closed at this scope.")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
