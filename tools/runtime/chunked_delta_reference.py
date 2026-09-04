#!/usr/bin/env python3
"""CP1: token-step == chunk-step for the gated delta rule, or it does not.

Read off crates/hawking-core/shaders/qwen_next.metal, the per-head recurrence is

    S <- d*S ;  kv = S^T k ;  delta = (v - kv)*b ;  S <- S + k (x) delta ;  out = S^T q

which rearranges to an AFFINE map in S:

    S_t = A_t S_{t-1} + B_t
    A_t = d_t (I - b_t k_t k_t^T)      (K x K, scaled rank-1 update of I)
    B_t = b_t k_t v_t^T                (K x V, rank one)

Affine maps compose:  (A2,B2) . (A1,B1) = (A2 A1, A2 B1 + B2)

so a chunk of T positions has a single (A_chunk, B_chunk) and the state need only
be touched once per chunk instead of T times. This file proves that claim
numerically before any of it is written in Rust. It uses numpy only, runs in
under a second, and is the CP1/CP2 gate of the R3 ladder.

If this file fails, the chunked prefill design is refuted and that is a result.
"""
from __future__ import annotations

import sys

import numpy as np

K = 128   # QWEN38_LINEAR_KEY_HEAD_DIM
V = 128   # QWEN38_LINEAR_VALUE_HEAD_DIM


def token_step(S, k, v, d, b):
    """Exactly what the Metal kernel does, one position."""
    S = d * S
    kv = S.T @ k                      # (V,)
    delta = (v - kv) * b              # (V,)
    S = S + np.outer(k, delta)        # (K,V)
    return S


def chunk_affine(ks, vs, ds, bs):
    """Compose T positions into one (A, B) without touching a K x V state."""
    A = np.eye(K)
    B = np.zeros((K, V))
    for k, v, d, b in zip(ks, vs, ds, bs):
        # A_t = d (I - b k k^T),  B_t = b k v^T
        At = d * (np.eye(K) - b * np.outer(k, k))
        Bt = b * np.outer(k, v)
        A = At @ A
        B = At @ B + Bt
    return A, B


def wy_product(ks, ds, bs):
    """The same prod(A_t), built as a scaled WY form I - sum(k_i w_i^T).

    This is the shape that matters for GEMM: W is built by a short recurrence
    over T rows of length K, never over the K x V state.
    """
    T = len(ks)
    W = np.zeros((T, K))
    for i in range(T):
        # w_i = b_i ( k_i - sum_{j<i} (k_i . k_j) w_j )
        acc = ks[i].copy()
        for j in range(i):
            acc = acc - (ks[i] @ ks[j]) * W[j]
        W[i] = bs[i] * acc
    Kmat = np.stack(ks)                     # (T, K)
    A = np.eye(K) - Kmat.T @ W              # I - K^T W
    return float(np.prod(ds)) * A


def main() -> int:
    rng = np.random.default_rng(0)
    failures = 0

    for trial, T in enumerate([1, 2, 3, 8, 32, 64, 128]):
        ks = [rng.normal(size=K) * 0.1 for _ in range(T)]
        vs = [rng.normal(size=V) for _ in range(T)]
        ds = [float(rng.uniform(0.90, 1.0)) for _ in range(T)]
        bs = [float(rng.uniform(0.0, 1.0)) for _ in range(T)]
        S0 = rng.normal(size=(K, V)) * 0.1

        # 1. the kernel's own loop
        S_tok = S0.copy()
        for k, v, d, b in zip(ks, vs, ds, bs):
            S_tok = token_step(S_tok, k, v, d, b)

        # 2. one affine composition
        A, B = chunk_affine(ks, vs, ds, bs)
        S_chunk = A @ S0 + B

        err = np.max(np.abs(S_tok - S_chunk))
        rel = err / max(np.max(np.abs(S_tok)), 1e-12)
        ok = rel < 1e-9
        failures += not ok
        print(f"  T={T:<4} affine composition   max|err|={err:.3e} rel={rel:.2e} "
              f"{'OK' if ok else 'FAIL'}")

        # 3. the WY form of prod(A_t), which is what becomes a GEMM
        A_wy = wy_product(ks, ds, bs)
        werr = np.max(np.abs(A - A_wy))
        wrel = werr / max(np.max(np.abs(A)), 1e-12)
        wok = wrel < 1e-9
        failures += not wok
        print(f"  T={T:<4} WY form of prod(A)   max|err|={werr:.3e} rel={wrel:.2e} "
              f"{'OK' if wok else 'FAIL'}")

    # CP2: randomized states, several chunk sizes, chunk boundaries composed
    print()
    for split in (1, 7, 16, 63):
        T = 128
        ks = [rng.normal(size=K) * 0.1 for _ in range(T)]
        vs = [rng.normal(size=V) for _ in range(T)]
        ds = [float(rng.uniform(0.90, 1.0)) for _ in range(T)]
        bs = [float(rng.uniform(0.0, 1.0)) for _ in range(T)]
        S0 = rng.normal(size=(K, V)) * 0.1

        S_tok = S0.copy()
        for k, v, d, b in zip(ks, vs, ds, bs):
            S_tok = token_step(S_tok, k, v, d, b)

        # two chunks of unequal size, composed at the boundary
        A1, B1 = chunk_affine(ks[:split], vs[:split], ds[:split], bs[:split])
        A2, B2 = chunk_affine(ks[split:], vs[split:], ds[split:], bs[split:])
        S_two = A2 @ (A1 @ S0 + B1) + B2
        err = np.max(np.abs(S_tok - S_two))
        rel = err / max(np.max(np.abs(S_tok)), 1e-12)
        ok = rel < 1e-9
        failures += not ok
        print(f"  T=128 split at {split:<4} boundary composition rel={rel:.2e} "
              f"{'OK' if ok else 'FAIL'}")

    print()
    if failures:
        print(f"CP1/CP2 FAILED ({failures} checks). Chunked prefill is REFUTED for "
              f"this recurrence as derived; record a scar.")
        return 1
    print("CP1/CP2 PASS. The recurrence is affine, composes exactly, and prod(A_t)")
    print("has a WY form. The K x V state is touched ONCE per chunk; the sequential")
    print("part is a short recurrence over T rows of length K, not over the state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
