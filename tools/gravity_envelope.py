#!/usr/bin/env python3
"""Feasibility envelope: what must each representation family DELIVER for sub-1.0?

"Get below 1.0 complete BPW" is not actionable. This turns it into a set of concrete
per-family requirements that a lane result can be checked against.

The hard arithmetic first, because it closes a whole space:

  One binary plane per source weight is EXACTLY 1.0 bit per weight before any metadata.
  Any per-group scale, however coarse, is strictly positive. So

      ONE BIT PER SOURCE WEIGHT CANNOT REACH SUB-1.0 AT ANY GROUP SIZE.

  Sub-1.0 therefore requires representations whose stored size is SUBLINEAR in element
  count over a large fraction of N -- not merely cheap per weight, but not per-weight at
  all. That is the whole reason the campaign moved off codec search.

Second, the endpoints. embed_tokens and lm_head are 9.454% of N, one site each, and this
checkpoint does NOT tie them (cosine 0.0121 over the first 2048 rows, i.e. independent).
They have no sharing partner inside their own class, so whatever floor they sit at is paid
in full. That makes the body's requirement strictly harder than the headline target, and
this tool reports the difference rather than hiding it.

Cost laws are stated per family in BITS PER SOURCE ELEMENT and every term is counted:
per-site payload, per-site metadata, and the amortized share of any shared object.
"""
from __future__ import annotations
import argparse, json, os, struct, sys, glob
from collections import defaultdict

SOURCE_PARAM_COUNT = 26_895_998_464
BF16_ROOT = "workspace/campaign/records/runs/qwen38-27b/bf16"


def inventory(root=BF16_ROOT):
    """(class -> sites, elements, out_dim m, in_dim n) from safetensors headers only."""
    by = {}
    for f in sorted(glob.glob(os.path.join(root, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for name, meta in hdr.items():
            if name == "__metadata__" or not name.startswith("language_model."):
                continue
            if len(meta["shape"]) != 2:
                continue
            m, k = meta["shape"]
            p = name.split(".")
            cls = ".".join(p[4:-1]) if (len(p) > 4 and p[2] == "layers") else ".".join(p[1:-1])
            d = by.setdefault(cls, {"sites": 0, "elements": 0, "m": m, "n": k})
            d["sites"] += 1
            d["elements"] += m * k
    return by


# ---------------------------------------------------------------- cost laws
# Each returns bits per SOURCE ELEMENT, all metadata included.

def f_percode(m, n, S, bits=4, group=128, scale_bytes=2):
    """Every weight owns a code. The family the campaign already floored."""
    return bits + 8 * scale_bytes / group


def f_planes(m, n, S, planes=1, group=128, scale_bytes=2):
    """Progressive 1-bit planes, each with its own per-group scale."""
    return planes * (1 + 8 * scale_bytes / group)


def f_sharedbasis(m, n, S, rank=256, coeff_bits=4, group=128, scale_bytes=2, basis_bytes=2):
    """W ~ U V^T with V (n x rank) SHARED across all S sites, U (m x rank) per site.

    Per-site payload is m*rank coefficients, so bits/element = rank*coeff_bits/n:
    sublinear in n, which is the only reason this family can go below 1.
    """
    per_site = rank * coeff_bits / n
    meta = rank * 8 * scale_bytes / (group * n)          # scales on the coefficients
    shared = n * rank * basis_bytes * 8 / (S * m * n)    # counted ONCE, amortized over S
    return per_site + meta + shared


def f_lowrank(m, n, S, rank=256, bits=8):
    """Per-site low rank, nothing shared. (m+n)*rank values."""
    return (m + n) * rank * bits / (m * n)


def f_generated(m, n, S, code_bits_per_elem=0.25, gen_mb=4.0):
    """Shared generator plus a tiny per-site code."""
    return code_bits_per_elem + gen_mb * (1 << 20) * 8 / (S * m * n)


def f_exact(m, n, S, density=0.01, value_bytes=2, index_bits=25):
    """Sparse exact islands: values plus indices, over a density fraction."""
    return density * (8 * value_bytes + index_bits)


FAMILIES = {
    "per-weight code": f_percode,
    "1-bit planes": f_planes,
    "shared basis": f_sharedbasis,
    "per-site low rank": f_lowrank,
    "generated block": f_generated,
    "sparse exact island": f_exact,
}


def bpw_of(assign, inv):
    """assign: cls -> (family_fn, kwargs). Returns complete BPW over N."""
    bits = 0.0
    for cls, d in inv.items():
        fn, kw = assign[cls]
        bits += fn(d["m"], d["n"], d["sites"], **kw) * d["elements"]
    return bits / SOURCE_PARAM_COUNT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.99)
    ap.add_argument("--endpoint-bits", type=float, default=2.125,
                    help="honest per-weight cost the endpoint tables are held at")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    inv = inventory()
    covered = sum(d["elements"] for d in inv.values())
    ENDPOINTS = {"model.embed_tokens", "lm_head"}
    end_e = sum(d["elements"] for c, d in inv.items() if c in ENDPOINTS)
    body_e = covered - end_e
    unc = SOURCE_PARAM_COUNT - covered

    print("=== INVENTORY ===")
    print(f"{'class':<26}{'sites':>6}{'m':>8}{'n':>8}{'elements':>16}{'%N':>8}")
    for c, d in sorted(inv.items(), key=lambda kv: -kv[1]["elements"]):
        print(f"{c:<26}{d['sites']:>6}{d['m']:>8}{d['n']:>8}{d['elements']:>16,}"
              f"{100*d['elements']/SOURCE_PARAM_COUNT:>8.3f}")
    print(f"{'ENDPOINTS (1 site, untied)':<26}{'':>6}{'':>8}{'':>8}{end_e:>16,}"
          f"{100*end_e/SOURCE_PARAM_COUNT:>8.3f}")
    print(f"{'BODY (>=16 sites)':<26}{'':>6}{'':>8}{'':>8}{body_e:>16,}"
          f"{100*body_e/SOURCE_PARAM_COUNT:>8.3f}")
    print(f"{'uncovered (1-D, norms)':<26}{'':>6}{'':>8}{'':>8}{unc:>16,}"
          f"{100*unc/SOURCE_PARAM_COUNT:>8.3f}")

    print("\n=== THE PER-WEIGHT WALL ===")
    for g in (64, 128, 512, 2048, 8192):
        print(f"  1 binary plane, group {g:>5}, 2-byte scale -> "
              f"{f_planes(0,0,0,planes=1,group=g):.6f} bits/weight")
    print("  a per-group scale is strictly positive at every group size, so one bit per")
    print("  source weight is >= 1.0 BPW ALWAYS. sub-1.0 needs sublinear-in-elements storage.")

    print("\n=== THE ENDPOINT TAX ===")
    end_bpw = a.endpoint_bits * end_e / SOURCE_PARAM_COUNT
    unc_bpw = 4.125 * unc / SOURCE_PARAM_COUNT       # norms held rich; they are tiny
    left = a.target - end_bpw - unc_bpw
    req = left * SOURCE_PARAM_COUNT / body_e
    print(f"  endpoints held at {a.endpoint_bits} bits/weight -> {end_bpw:.6f} BPW of the budget")
    print(f"  1-D tensors held at 4.125             -> {unc_bpw:.6f} BPW")
    print(f"  target {a.target} leaves {left:.6f} BPW for {100*body_e/SOURCE_PARAM_COUNT:.3f}% of N")
    print(f"  => THE BODY MUST AVERAGE {req:.6f} BITS PER SOURCE ELEMENT")
    if req <= 0:
        print("  => UNREACHABLE: the endpoints alone exceed the target at this endpoint cost")

    print("\n=== WHAT EACH FAMILY MUST DELIVER TO HIT THAT BODY AVERAGE ===")
    print(f"  requirement: {req:.6f} bits/element averaged over the body\n")
    print(f"{'family':<22}{'organ':<24}{'parameter needed':>34}")
    body = {c: d for c, d in inv.items() if c not in ENDPOINTS}
    probe = [("mlp.gate_proj", body.get("mlp.gate_proj")),
             ("mlp.down_proj", body.get("mlp.down_proj")),
             ("linear_attn.in_proj_qkv", body.get("linear_attn.in_proj_qkv"))]
    for cls, d in probe:
        if not d:
            continue
        m, n, S = d["m"], d["n"], d["sites"]
        # shared basis: solve rank
        r = 1
        while r < n and f_sharedbasis(m, n, S, rank=r) < req:
            r += 1
        r -= 1
        print(f"{'shared basis':<22}{cls:<24}"
              f"{f'rank <= {r} of {n} ({100*r/n:.1f}%) at 4-bit coeffs':>34}")
        # planes: max plane count (already known to fail, shown for contrast)
        p = req / (1 + 8 * 2 / 128)
        print(f"{'1-bit planes':<22}{cls:<24}{f'planes <= {p:.3f}  (IMPOSSIBLE, p>=1)':>34}")
        # generated block
        print(f"{'generated block':<22}{cls:<24}"
              f"{f'code <= {req - f_generated(m,n,S,code_bits_per_elem=0):.4f} b/elem':>34}")
        # exact island alone
        dens = req / (8 * 2 + 25)
        print(f"{'sparse exact island':<22}{cls:<24}{f'density <= {100*dens:.3f}% alone':>34}")
        print()

    print("=== REFERENCE POINTS ===")
    for label, kw in (("all per-weight q4 g128", dict(bits=4)),
                      ("all per-weight q2 g128", dict(bits=2)),
                      ("all per-weight q1 g128", dict(bits=1))):
        asg = {c: (f_percode, kw) for c in inv}
        print(f"  {label:<26} complete BPW {bpw_of(asg, inv):.6f}")

    # the only shape that reaches the target: body on shared basis, endpoints held
    for r in (64, 128, 256, 512):
        asg = {}
        for c in inv:
            if c in ENDPOINTS:
                asg[c] = (f_percode, dict(bits=a.endpoint_bits - 0.125))
            else:
                asg[c] = (f_sharedbasis, dict(rank=r))
        print(f"  body shared-basis rank {r:<4}      complete BPW {bpw_of(asg, inv):.6f}"
              f"   {'REACHES' if bpw_of(asg,inv) < a.target else 'misses'} {a.target}")

    if a.json:
        json.dump({"inventory": inv, "body_requirement_bits_per_elem": req,
                   "endpoint_bpw": end_bpw, "target": a.target}, open(a.json, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
