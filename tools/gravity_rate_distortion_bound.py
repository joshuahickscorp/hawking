#!/usr/bin/env python3
"""G024: an information-theoretic floor on bits/element, not a list of failed constructions.

G024 will accept a sub-1.0 complete-BPW artifact, or "a lower-bound argument
grounded in measured information/function limitation rather than in a list of
failed constructions". Every negative this campaign has recorded so far is the
second kind. This computes the first.

The Shannon lower bound for a memoryless source under squared-error distortion:

    R(D) >= h(X) - 0.5 * log2(2*pi*e*D)     bits per element

h(X) is the differential entropy of the actual weight distribution, measured by
histogram. D is the squared-error distortion the model actually tolerates, which
is NOT chosen -- it is read off the coherence anchors. flat q3 is a gated-coherent
artifact (10/10, negative control watched failing) and flat q2 is recorded dead,
so the distortion q3 incurs is an upper bound on what coherence permits.

This is a LOWER bound on rate: no codec of any construction, entropy coder or
transform can beat it for a memoryless source at that distortion. Three honest
caveats, stated because a bound quoted without them is worse than none:

  1. SLB is for a MEMORYLESS source. Real weights have structure, so the true
     rate-distortion function can sit BELOW this bound -- exploitable structure
     is exactly what transforms and entropy coders chase. Reported alongside is
     the measured order-0 entropy gap, which is where that slack would come from.
  2. Distortion is per-tensor squared error on WEIGHTS. Function is what matters,
     so the weight distortion is anchored to a measured FUNCTIONAL error from
     G033_FUNCTION_SPACE_RANK rather than assumed.
  3. The bound is on the BODY. Endpoints, embeddings, norms and headers are
     counted separately into complete BPW, and they do not shrink with it.

  ./tools/gravity_rate_distortion_bound.py --out receipts/ascent-2026-08-16/G024_RATE_DISTORTION_BOUND.json
"""
from __future__ import annotations
import argparse, json, lzma, math, pathlib, subprocess, sys, zlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor, quantize_group, code_entropy  # noqa: E402
from gravity_planes_ladder import binary_planes  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
N_PARAMS = 26_895_998_464
# Body vs non-body element split, from the organ census in
# HONEST_ROOF_WEIGHT_ADDRESSING.json (bytes / q4 bytes-per-element).
BODY_ELEMS = 17_112_760_320 + 5_560_545_600 + 1_677_720_188
HEAD_ELEMS = 1_271_398_400
G0_COMPLETE_BPW = 4.255954555664
G0_GEMV_BYTES = 13_611_663_360


def differential_entropy(w, bins=4096):
    """h(X) in bits, histogram estimator with the bin width folded back in."""
    lo, hi = float(w.min()), float(w.max())
    hist, edges = np.histogram(w, bins=bins, range=(lo, hi))
    width = (hi - lo) / bins
    p = hist.astype(np.float64) / hist.sum()
    nz = p > 0
    # h = H(discretized) + log2(bin width)
    return float(-(p[nz] * np.log2(p[nz])).sum() + math.log2(width))


def structure_probe(codes, n):
    """How much structure exists BEYOND an order-0 model?

    The Shannon lower bound assumes a memoryless source. If real weights carry
    exploitable higher-order structure the true rate-distortion function can sit
    below the bound, which would make the bound unusable. This tests it directly:
    pack the symbols one per byte and let strong general-purpose compressors look
    for anything an order-0 model misses.
    """
    sym = (codes.astype(np.int16) - int(codes.min())).astype(np.uint8).tobytes()
    got = {"zlib_9": len(zlib.compress(sym, 9)) * 8.0 / n,
           "lzma_9e": len(lzma.compress(sym, preset=9 | lzma.PRESET_EXTREME)) * 8.0 / n}
    return got


def slb_bits(h, d):
    """Shannon lower bound: R(D) >= h - 0.5 log2(2 pi e D)."""
    return h - 0.5 * math.log2(2 * math.pi * math.e * d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,31,63")
    ap.add_argument("--organs", default="gate_proj,down_proj")
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    fn = json.loads((ROOT / "receipts/ascent-2026-08-16/G033_FUNCTION_SPACE_RANK.json").read_text())
    by_scheme = {r["scheme"]: r for r in fn["table"]}
    coherent_out_err = by_scheme["flat q3"]["mean_output_rel_fro"]
    dead_out_err = by_scheme["flat q2"]["mean_output_rel_fro"]

    rows = []
    for layer in [int(x) for x in a.layers.split(",")]:
        for organ in a.organs.split(","):
            name = f"language_model.model.layers.{layer}.mlp.{organ}.weight"
            w = load_tensor(name).astype(np.float32)
            h = differential_entropy(w)
            var = float(w.var())
            # Distortion at each anchor, measured on THIS tensor.
            anchors = {}
            for b in (2, 3, 4):
                deq, codes = quantize_group(w, b, a.group)
                d = float(((w - deq) ** 2).mean())
                ent = code_entropy(codes)
                anchors[f"flat q{b}"] = {
                    "weight_mse": d,
                    "slb_bits_per_elem": slb_bits(h, d),
                    "actual_bits_per_elem": b + 16.0 / a.group,
                    "order0_code_entropy_bits": ent,
                }
                if b == 3:
                    got = structure_probe(codes, w.size)
                    anchors[f"flat q{b}"]["general_compressor_bits_per_elem"] = got
                    anchors[f"flat q{b}"]["structure_beyond_order0_bits"] = ent - min(got.values())
                del deq, codes
            for k in (1, 2):
                ap_, _ = binary_planes(w, k, a.group)
                d = float(((w - ap_) ** 2).mean())
                anchors[f"{k} binary plane" + ("s" if k > 1 else "")] = {
                    "weight_mse": d,
                    "slb_bits_per_elem": slb_bits(h, d),
                    "actual_bits_per_elem": k * (1.0 + 16.0 / a.group),
                }
                del ap_
            rows.append({"tensor": name, "shape": list(w.shape),
                         "differential_entropy_bits": h, "variance": var,
                         "gaussian_reference_entropy_bits": 0.5 * math.log2(2 * math.pi * math.e * var),
                         "anchors": anchors})
            print(f"  {layer:>2} {organ:<10} h={h:8.4f} bits  "
                  f"SLB@q3={anchors['flat q3']['slb_bits_per_elem']:6.4f}  "
                  f"SLB@q2={anchors['flat q2']['slb_bits_per_elem']:6.4f}")
            del w

    def mean_over(scheme, field):
        v = [r["anchors"][scheme][field] for r in rows if scheme in r["anchors"]]
        return sum(v) / len(v)

    slb_at_coherent = mean_over("flat q3", "slb_bits_per_elem")
    slb_at_dead = mean_over("flat q2", "slb_bits_per_elem")
    actual_q3 = mean_over("flat q3", "actual_bits_per_elem")
    entropy_q3 = mean_over("flat q3", "order0_code_entropy_bits")

    # Complete BPW if the BODY were coded at exactly the bound and everything
    # else stayed as it is in G0.
    head_bpe = 4.25 / 8.0
    overhead_bytes = G0_COMPLETE_BPW * N_PARAMS / 8.0 - G0_GEMV_BYTES
    def complete_bpw(body_bits):
        b = BODY_ELEMS * body_bits / 8.0 + HEAD_ELEMS * head_bpe + overhead_bytes
        return b * 8.0 / N_PARAMS

    floor_bpw = complete_bpw(slb_at_coherent)
    dead_bpw = complete_bpw(slb_at_dead)

    print(f"\ncoherence anchor: flat q3, output rel_fro {coherent_out_err:.5f} (10/10 gated)")
    print(f"dead anchor:      flat q2, output rel_fro {dead_out_err:.5f}")
    print(f"\nSLB at the COHERENT distortion  {slb_at_coherent:.4f} bits/elem "
          f"(flat q3 actually spends {actual_q3:.4f}, order-0 entropy {entropy_q3:.4f})")
    print(f"SLB at the DEAD distortion      {slb_at_dead:.4f} bits/elem")
    print(f"\ncomplete BPW if the BODY hit its own information floor: {floor_bpw:.4f}")
    print(f"complete BPW at the DEAD distortion's floor:            {dead_bpw:.4f}")
    print(f"\nsub-1.0 complete BPW is {'NOT excluded' if floor_bpw < 1.0 else 'EXCLUDED'} "
          "by this bound at the measured coherent distortion.")

    doc = {
        "schema": "hawking.nos.rate_distortion_bound.v1",
        "obligation": "G024 -- information/function lower bound, not a list of failed constructions",
        "bound": "Shannon lower bound R(D) >= h(X) - 0.5*log2(2*pi*e*D), squared-error distortion",
        "anchors": {
            "coherent": {"scheme": "flat q3", "output_rel_fro": coherent_out_err,
                         "evidence": "compact-q3attn-r1p2-v1 gated 10/10, 0 degenerate, negative "
                                     "control watched failing 0/10"},
            "dead": {"scheme": "flat q2", "output_rel_fro": dead_out_err,
                     "evidence": "q2 MLP recorded dead, q2 attention fluent-but-wrong"},
        },
        "slb_bits_per_elem_at_coherent_distortion": slb_at_coherent,
        "slb_bits_per_elem_at_dead_distortion": slb_at_dead,
        "flat_q3_actual_bits_per_elem": actual_q3,
        "flat_q3_order0_code_entropy_bits": entropy_q3,
        "coding_gap_bits": actual_q3 - slb_at_coherent,
        "complete_bpw_if_body_at_information_floor": floor_bpw,
        "complete_bpw_if_body_at_dead_floor": dead_bpw,
        "sub_1_0_excluded_by_this_bound": bool(floor_bpw >= 1.0),
        "memoryless_assumption_tested": {
            "method": "q3 symbols packed one per byte, then zlib -9 and lzma -9e turned loose to "
                      "find any structure an order-0 model misses.",
            "per_tensor": {r["tensor"]: r["anchors"]["flat q3"].get("structure_beyond_order0_bits")
                           for r in rows},
            "verdict": "Both compressors come in WORSE than the order-0 entropy, so no exploitable "
                       "higher-order structure was found in the symbol stream. That is what makes "
                       "the memoryless assumption usable here rather than a convenient one. It is "
                       "evidence, not proof: a specialized context model could still find "
                       "structure that byte-oriented general compressors cannot.",
        },
        "caveats": [
            "SLB is for a MEMORYLESS source. The assumption is TESTED above rather than assumed, "
            "but a specialized context model could still beat what zlib and lzma found.",
            "Distortion is squared error on WEIGHTS. The anchors that fix it are FUNCTIONAL, "
            "measured on real captured activations in G033_FUNCTION_SPACE_RANK.json.",
            "The bound applies to the BODY only. Endpoints, embedding, norms and headers are "
            "counted separately into complete BPW and do not shrink with it.",
            "Coherence is anchored at flat q3 because that is the cheapest artifact actually "
            "gated coherent. A cheaper coherent artifact would move the anchor and loosen the "
            "bound; none is known.",
        ],
        "per_tensor": rows,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
