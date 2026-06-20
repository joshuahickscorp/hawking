#!/usr/bin/env python3
"""oracle_f16scales_precision — settle the f16-scales (Q4_K predec) precision lever.

Question
--------
The f16-scales lever narrows the PRE-DECODED per-32-elem-group products of a
Q4_K weight tensor — `ds = d * scale[sub]` and `dm = dmin * min[sub]` — from f32
to f16 in the predec scale table (see
`crates/hawking-core/src/kernels/mod.rs::predecode_q4_k_scale_table` /
`..._f16`). As the DEFAULT it gave +9% tps but FAILED the quality gate
(token-identical 0.792, drift 11.46%) and was reverted (e613dde). It stays
opt-in. This oracle settles, OFFLINE and CPU-only, whether a *default-safe*
variant exists:

  (1) ds,dm BOTH f32                         = reference (error 0; the prod predec
                                                path, bit-identical to llama.cpp).
  (2) ds,dm BOTH f16 (round-trip the products)  = the reverted default.
  (3) ds f16, dm f32                          = REFRAME A (asymmetric f32-dmin):
                                                half the scale-table f16 savings
                                                (~+4.5% tps), keep the min exact.

Per Q4_K weight tensor we dequantize all three ways and report:
  - relative-L2 error  ||w_scheme - w_ref||_2 / ||w_ref||_2
  - max-abs error       max|w_scheme - w_ref|

VERDICT logic
-------------
  * If scheme-3 rel-L2 is >= ~2x LOWER than scheme-2 across tensors, the `dm`
    (min) f16 rounding dominates the perturbation => asymmetric f32-dmin (reframe
    A) recovers most of the quality at ~+4.5% => GO (asymmetric).
  * If scheme-3 ~= scheme-2, the `ds` (scale) f16 rounding dominates => reframe A
    does NOT help => NO-GO for asymmetric; fall back to SELECTIVE (keep the
    worst-error tensors on the f32 table) — the per-tensor sensitivity ranking
    (worst scheme-2 rel-L2) is the keep-on-f32 set.

IMPORTANT physical scoping (discovered from the GGUF, see below):
  In this Qwen2.5-3B-Instruct-Q4_K_M, ONLY these tensor roles are Q4_K (ggml
  type 12) and therefore touched by f16-scales:
      attn_q, attn_k, attn_output, ffn_gate, ffn_up   (per layer)
  The sensitive roles ffn_down, attn_v and the tied token_embd/LM-head are Q6_K
  (type 14) and are UNAFFECTED by this lever. So the 11.46% production drift
  comes entirely from f16-rounding the Q4_K-role scale tables.

CPU-only, streaming (one tensor in f32 at a time, freed before the next). No
Metal, no cargo, no training. Reuses the GGUF reader from
oracle_coactivation_permute.py.

Usage:
    python3 tools/bench/oracle_f16scales_precision.py \
        --gguf models/qwen2.5-3b-instruct-q4_k_m.gguf \
        [--stride 1] [--json reports/oracle_f16scales_precision.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the validated GGUF reader (header parse + tensor offsets).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_coactivation_permute import read_gguf, GGML_Q4_K  # noqa: E402

QK_K = 256
BLOCK_BYTES = 144  # Q4_K super-block


# --------------------------------------------------------------------------
# Q4_K predec scale-table dequant, vectorized, with selectable scale precision.
#
# Mirrors crates/hawking-core/src/kernels/mod.rs::predecode_q4_k_scale_table:
#   per 32-elem sub-block `sub` (0..7):
#       ds = d * scale6[sub]      (f32 reference; f16-rounded if scale_f16)
#       dm = dmin * min6[sub]     (f32 reference; f16-rounded if dmin_f16)
#       w[elem] = ds * nibble - dm
# d, dmin are the f16-stored super-block header values widened to f32 (this is
# the production f32-predec reference, itself bit-identical to llama.cpp).
# --------------------------------------------------------------------------
def _unpack_scales_mins(raw: np.ndarray):
    """raw: [nb,144] uint8. Return (scale6 [nb,8] f32, min6 [nb,8] f32).

    Exactly mirrors predecode_q4_k_scale_table's 6-bit unpack. The 12 scale/min
    bytes live at file offsets bo+4 .. bo+15, i.e. columns 0..11 of `sb`:
        bo+4+sub -> sb col sub      (sub 0..3)
        bo+8+sub -> sb col 4+sub
        bo+12+j  -> sb col 8+j      (j 0..3)
    """
    sb = raw[:, 4:16].astype(np.uint32)  # 12 scale/min bytes -> sb cols 0..11
    sc = np.empty((raw.shape[0], 8), np.float32)
    mn = np.empty((raw.shape[0], 8), np.float32)
    for sub in range(4):
        sc[:, sub] = sb[:, sub] & 0x3F
        mn[:, sub] = sb[:, sub + 4] & 0x3F
    for j in range(4):
        b12 = sb[:, 8 + j]   # bo+12+j
        b4 = sb[:, 0 + j]    # bo+4+j
        b8 = sb[:, 4 + j]    # bo+8+j
        sc[:, 4 + j] = (b12 & 0x0F) | ((b4 >> 6) << 4)
        mn[:, 4 + j] = (b12 >> 4) | ((b8 >> 6) << 4)
    return sc, mn


def dequant_q4k_predec(buf: np.ndarray, n_elems: int, scale_f16: bool, dmin_f16: bool) -> np.ndarray:
    """Dequant a Q4_K byte buffer via the predec products, choosing f16/f32 for
    ds (scale) and dm (min) independently. Returns float32 [n_elems]."""
    nb = n_elems // QK_K
    raw = buf[: nb * BLOCK_BYTES].reshape(nb, BLOCK_BYTES)
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32)[:, 0]      # [nb]
    dmin = raw[:, 2:4].copy().view(np.float16).astype(np.float32)[:, 0]   # [nb]
    sc6, mn6 = _unpack_scales_mins(raw)                                   # [nb,8] each

    # Per-group products (the predec table entries).
    ds = d[:, None] * sc6     # [nb,8]
    dm = dmin[:, None] * mn6  # [nb,8]
    if scale_f16:
        ds = ds.astype(np.float16).astype(np.float32)
    if dmin_f16:
        dm = dm.astype(np.float16).astype(np.float32)

    # Nibble unpack: llama.cpp Q4_K layout.
    #   sub 2k   (even): low nibble of qs[k*32 .. k*32+31]
    #   sub 2k+1 (odd) : high nibble of qs[k*32 .. k*32+31]
    qs = raw[:, 16:144]  # [nb,128]
    out = np.empty((nb, QK_K), np.float32)
    for sub in range(8):
        pair = sub // 2
        upper = (sub % 2) == 1
        qbyte = qs[:, pair * 32: pair * 32 + 32]          # [nb,32]
        nib = (qbyte >> 4) if upper else (qbyte & 0x0F)   # [nb,32]
        out[:, sub * 32: sub * 32 + 32] = ds[:, sub:sub + 1] * nib.astype(np.float32) - dm[:, sub:sub + 1]
    return out.reshape(-1)[:n_elems]


def role_of(name: str) -> str:
    # blk.<L>.<role>.weight  ->  role; non-blk -> the stem.
    parts = name.split(".")
    if len(parts) >= 4 and parts[0] == "blk":
        return parts[2]
    return name.replace(".weight", "")


def layer_of(name: str):
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "blk":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="models/qwen2.5-3b-instruct-q4_k_m.gguf")
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth layer (1 = all). lm_head/embed always included.")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    gguf = Path(args.gguf)
    kv, tinfos, data_start = read_gguf(gguf)

    # Memory-map the file once; slice each tensor's bytes on demand.
    mm = np.memmap(gguf, dtype=np.uint8, mode="r")

    # Select Q4_K weight tensors (the only ones f16-scales touches).
    q4k = []
    for name, (dims, typ, off) in tinfos.items():
        if typ != GGML_Q4_K:
            continue
        L = layer_of(name)
        if args.stride > 1 and L is not None and (L % args.stride) != 0:
            continue
        n_elems = int(np.prod(dims))
        q4k.append((name, dims, off, n_elems, role_of(name), L))
    q4k.sort(key=lambda r: (r[5] if r[5] is not None else -1, r[4]))

    print(f"[oracle] {gguf.name}: {len(q4k)} Q4_K tensors selected "
          f"(stride={args.stride}); data_start={data_start}", file=sys.stderr)

    rows = []  # per-tensor records
    t0 = time.time()
    for i, (name, dims, off, n_elems, role, L) in enumerate(q4k):
        abs_off = data_start + off
        nb = n_elems // QK_K
        buf = np.asarray(mm[abs_off: abs_off + nb * BLOCK_BYTES])
        ref = dequant_q4k_predec(buf, n_elems, scale_f16=False, dmin_f16=False)
        ref_norm = float(np.linalg.norm(ref))
        ref_max = float(np.max(np.abs(ref))) + 1e-30

        def err(scale_f16, dmin_f16):
            w = dequant_q4k_predec(buf, n_elems, scale_f16=scale_f16, dmin_f16=dmin_f16)
            diff = w - ref
            relL2 = float(np.linalg.norm(diff)) / (ref_norm + 1e-30)
            maxabs = float(np.max(np.abs(diff)))
            return relL2, maxabs

        s2_relL2, s2_max = err(True, True)    # scheme 2: both f16
        s3_relL2, s3_max = err(True, False)   # scheme 3: ds f16, dm f32 (reframe A)
        # also the complement (ds f32, dm f16) to attribute which product dominates
        sC_relL2, sC_max = err(False, True)   # scheme C: ds f32, dm f16

        rows.append(dict(
            name=name, role=role, layer=L, n_elems=n_elems, ref_norm=ref_norm,
            ref_maxabs=ref_max,
            s2_relL2=s2_relL2, s2_maxrel=s2_max / ref_max,
            s3_relL2=s3_relL2, s3_maxrel=s3_max / ref_max,
            sC_relL2=sC_relL2, sC_maxrel=sC_max / ref_max,
        ))
        del buf, ref
        if (i + 1) % 40 == 0 or (i + 1) == len(q4k):
            print(f"  ..{i + 1}/{len(q4k)} ({time.time() - t0:.1f}s)", file=sys.stderr)

    # ---- Aggregate by role (RMS over tensors of the per-tensor rel-L2 + mean) --
    roles = {}
    for r in rows:
        roles.setdefault(r["role"], []).append(r)

    def agg(lst, key):
        v = np.array([x[key] for x in lst], float)
        return dict(mean=float(v.mean()), rms=float(np.sqrt((v ** 2).mean())),
                    mx=float(v.max()), mn=float(v.min()), n=len(v))

    print("\n================= PER-ROLE SUMMARY (Q4_K tensors only) =================")
    hdr = (f"{'role':<14}{'n':>4} | {'s2 relL2 mean':>14}{'s2 relL2 max':>14} | "
           f"{'s3 relL2 mean':>14}{'s3 relL2 max':>14} | {'s2/s3':>7} | {'C relL2 mean':>13}")
    print(hdr)
    print("-" * len(hdr))
    role_summary = {}
    for role in sorted(roles):
        lst = roles[role]
        a2 = agg(lst, "s2_relL2")
        a3 = agg(lst, "s3_relL2")
        aC = agg(lst, "sC_relL2")
        ratio = a2["mean"] / (a3["mean"] + 1e-30)
        role_summary[role] = dict(s2=a2, s3=a3, sC=aC, ratio_s2_over_s3=ratio)
        print(f"{role:<14}{a2['n']:>4} | {a2['mean']:>14.3e}{a2['mx']:>14.3e} | "
              f"{a3['mean']:>14.3e}{a3['mx']:>14.3e} | {ratio:>7.2f} | {aC['mean']:>13.3e}")

    # ---- Global aggregates ----
    all2 = np.array([r["s2_relL2"] for r in rows])
    all3 = np.array([r["s3_relL2"] for r in rows])
    allC = np.array([r["sC_relL2"] for r in rows])
    g_ratio = float(all2.mean() / (all3.mean() + 1e-30))
    print("\n----- GLOBAL (all Q4_K tensors) -----")
    print(f"scheme-2 (ds,dm f16)   rel-L2: mean={all2.mean():.3e}  max={all2.max():.3e}")
    print(f"scheme-3 (ds f16,dm f32) rel-L2: mean={all3.mean():.3e}  max={all3.max():.3e}")
    print(f"scheme-C (ds f32,dm f16) rel-L2: mean={allC.mean():.3e}  max={allC.max():.3e}")
    print(f"mean(s2)/mean(s3) = {g_ratio:.2f}x   "
          f"(>=~2 => dm dominates => reframe A GO; ~1 => ds dominates => reframe A NO-GO)")

    # ---- Sensitivity ranking (worst scheme-2 rel-L2) ----
    print("\n----- TOP-15 MOST-SENSITIVE Q4_K TENSORS (worst scheme-2 rel-L2) -----")
    print(f"{'rank':>4}  {'tensor':<28}{'s2 relL2':>12}{'s3 relL2':>12}{'s2/s3':>8}")
    worst = sorted(rows, key=lambda r: r["s2_relL2"], reverse=True)[:15]
    for k, r in enumerate(worst, 1):
        print(f"{k:>4}  {r['name']:<28}{r['s2_relL2']:>12.3e}"
              f"{r['s3_relL2']:>12.3e}{r['s2_relL2'] / (r['s3_relL2'] + 1e-30):>8.2f}")

    # ---- Selective set sizing: how much error-mass (sum of squared per-tensor
    # rel-L2, a proxy for token-drift contribution) do the worst-K tensors carry?
    e2 = np.array(sorted((r["s2_relL2"] for r in rows), reverse=True))
    cum = np.cumsum(e2 ** 2) / (np.sum(e2 ** 2) + 1e-30)
    def k_for(frac):
        idx = int(np.searchsorted(cum, frac)) + 1
        return min(idx, len(e2))
    print("\n----- SELECTIVE keep-on-f32 sizing (by squared-relL2 mass, scheme-2) -----")
    for frac in (0.5, 0.8, 0.9):
        k = k_for(frac)
        print(f"  keeping worst {k:>3}/{len(rows)} tensors on f32 removes "
              f"~{frac*100:.0f}% of the scheme-2 error mass")

    # ---- VERDICT ----
    print("\n================= VERDICT =================")
    if g_ratio >= 2.0:
        verdict = ("REFRAME A (asymmetric f32-dmin) GO: dm (min) f16-rounding "
                   f"dominates ({g_ratio:.2f}x). Keeping dm in f32 and ds in f16 "
                   "recovers most of the precision at ~+4.5% tps.")
    elif g_ratio <= 1.3:
        verdict = ("REFRAME A NO-GO: ds (scale) f16-rounding dominates "
                   f"({g_ratio:.2f}x ~ 1). Asymmetric f32-dmin does not help. "
                   "Best default-safe path = SELECTIVE (keep worst-error Q4_K "
                   "tensors on the f32 predec table) OR neither.")
    else:
        verdict = (f"PARTIAL: dm contributes but does not dominate ({g_ratio:.2f}x). "
                   "Asymmetric f32-dmin gives a partial recovery; combine with "
                   "selective keep-on-f32 for the worst tensors.")
    print(verdict)

    out = dict(
        gguf=str(gguf), n_q4k_tensors=len(rows), stride=args.stride,
        global_=dict(
            s2_relL2_mean=float(all2.mean()), s2_relL2_max=float(all2.max()),
            s3_relL2_mean=float(all3.mean()), s3_relL2_max=float(all3.max()),
            sC_relL2_mean=float(allC.mean()), sC_relL2_max=float(allC.max()),
            ratio_s2_over_s3=g_ratio,
        ),
        role_summary=role_summary,
        worst15=[dict(name=r["name"], s2_relL2=r["s2_relL2"], s3_relL2=r["s3_relL2"]) for r in worst],
        verdict=verdict,
        rows=rows,
    )
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\n[oracle] wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
