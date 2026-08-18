#!/usr/bin/env python3
"""Adversarial review of the G024 lower bound. Its verify line asks for exactly this.

G024_RATE_DISTORTION_BOUND.json states: complete BPW cannot go below 2.1449 at the
distortion the model is measured to survive, so sub-1.0 is excluded. The receipt
named two holes in itself. This attacks both, and a bound that survives its own
review is worth more than one that was never reviewed.

HOLE 1 -- the memoryless assumption. SLB bounds a memoryless source. zlib and lzma
found no structure, but those are byte-oriented general compressors and the
receipt said plainly that a specialized context model could still find what they
cannot. So build the specialized models: conditional entropy of a q3 symbol given
its neighbour along the row (adjacent in the contraction axis), given its
neighbour down the column (same input channel, next output row), and given both.
If conditional entropy does not fall below order-0, the stream is memoryless by a
model built specifically to exploit it, not merely by a generic one.

HOLE 2 -- the anchor. The bound is anchored on flat q3's distortion because
compact-q3attn-r1p2-v1 is the cheapest artifact actually gated coherent. Its
PACK_REPORT records a replaced-tensor weight cosine of min 0.9461, median 0.9684
across 496 tensors, while the six tensors sampled for the bound have min 0.9653.
So the real artifact TOLERATES MORE distortion than the sample's central estimate,
which loosens the bound. This re-anchors on the artifact's own worst case and asks
the only question that matters: how much distortion would the model have to
tolerate before the floor reaches 1.0 complete BPW, and is that inside or beyond
the distortion already measured to kill it?

  ./tools/gravity_bound_review.py --out receipts/ascent-2026-08-16/G024_BOUND_REVIEW.json
"""
from __future__ import annotations
import argparse, json, math, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor, quantize_group, code_entropy  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
N_PARAMS = 26_895_998_464
BODY_ELEMS = 17_112_760_320 + 5_560_545_600 + 1_677_720_188
HEAD_ELEMS = 1_271_398_400
G0_COMPLETE_BPW = 4.255954555664
G0_GEMV_BYTES = 13_611_663_360


def joint_conditional_entropy(a, b):
    """H(a | b) in bits, from the empirical joint. Both are small-alphabet ints."""
    a = a.ravel(); b = b.ravel()
    na, nb = int(a.max()) + 1, int(b.max()) + 1
    joint = np.bincount((a.astype(np.int64) * nb + b), minlength=na * nb).reshape(na, nb)
    tot = joint.sum()
    pj = joint / tot
    pb = pj.sum(axis=0)
    nz = pj > 0
    hj = -(pj[nz] * np.log2(pj[nz])).sum()
    nzb = pb > 0
    hb = -(pb[nzb] * np.log2(pb[nzb])).sum()
    return float(hj - hb)


def complete_bpw(body_bits):
    overhead = G0_COMPLETE_BPW * N_PARAMS / 8.0 - G0_GEMV_BYTES
    b = BODY_ELEMS * body_bits / 8.0 + HEAD_ELEMS * (4.25 / 8.0) + overhead
    return b * 8.0 / N_PARAMS


def body_bits_for_complete(target_bpw):
    overhead = G0_COMPLETE_BPW * N_PARAMS / 8.0 - G0_GEMV_BYTES
    b = target_bpw * N_PARAMS / 8.0 - HEAD_ELEMS * (4.25 / 8.0) - overhead
    return b * 8.0 / BODY_ELEMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,31,63")
    ap.add_argument("--organs", default="gate_proj,down_proj")
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    bound = json.loads((ROOT / "receipts/ascent-2026-08-16/G024_RATE_DISTORTION_BOUND.json").read_text())
    pack = json.loads((ROOT / "workspace/campaign/records/runs/qwen38-27b/"
                       "mixed-q3mlp-q3attn-r1p2-v1/PACK_REPORT.json").read_text())
    art_cos = pack["replaced_strided_weight_cosine"]

    ctx_rows = []
    for layer in [int(x) for x in a.layers.split(",")]:
        for organ in a.organs.split(","):
            name = f"language_model.model.layers.{layer}.mlp.{organ}.weight"
            w = load_tensor(name).astype(np.float32)
            _, codes = quantize_group(w, 3, a.group)
            sym = (codes - codes.min()).astype(np.int16)
            h0 = code_entropy(codes)
            # Along the contraction axis (adjacent weights in a row).
            h_row = joint_conditional_entropy(sym[:, 1:], sym[:, :-1])
            # Down the output axis (same input channel, next output row).
            h_col = joint_conditional_entropy(sym[1:, :], sym[:-1, :])
            # Both neighbours at once.
            pair = sym[:-1, :-1].astype(np.int64) * (int(sym.max()) + 1) + sym[:-1, 1:]
            h_both = joint_conditional_entropy(sym[1:, 1:], pair.astype(np.int16)
                                               if pair.max() < 32767 else sym[:-1, 1:])
            ctx_rows.append({"tensor": name, "order0_bits": h0,
                             "H_given_row_neighbour": h_row,
                             "H_given_col_neighbour": h_col,
                             "H_given_both": h_both,
                             "best_reduction_bits": h0 - min(h_row, h_col, h_both)})
            print(f"  {layer:>2} {organ:<10} H0={h0:.4f}  H|row={h_row:.4f}  "
                  f"H|col={h_col:.4f}  H|both={h_both:.4f}  best gain={h0-min(h_row,h_col,h_both):+.4f}")
            del w, codes, sym

    best_gain = max(r["best_reduction_bits"] for r in ctx_rows)
    slb_coh = bound["slb_bits_per_elem_at_coherent_distortion"]
    slb_dead = bound["slb_bits_per_elem_at_dead_distortion"]

    # HOLE 2. Re-anchor on the artifact's OWN worst tensor. For a quantization
    # error roughly orthogonal to the weight, cos ~ 1 - rel^2/2, so
    # rel ~ sqrt(2(1-cos)) and MSE scales with rel^2. SLB moves by
    # -0.5*log2(MSE ratio).
    def slb_at_cos(cos_ref, cos_new, slb_ref):
        rel_ref = math.sqrt(max(2.0 * (1.0 - cos_ref), 1e-12))
        rel_new = math.sqrt(max(2.0 * (1.0 - cos_new), 1e-12))
        return slb_ref - 0.5 * math.log2((rel_new / rel_ref) ** 2)

    sample_med_cos = 0.968397  # mean weight-space hold of the sampled q3, G033 ladder
    reanchored = slb_at_cos(sample_med_cos, art_cos["min"], slb_coh)
    reanchored_bpw = complete_bpw(reanchored)

    # The question that settles it: what distortion puts the floor AT 1.0?
    body_at_1 = body_bits_for_complete(1.0)
    # Invert SLB: bits fall 0.5 bits per doubling of D, so D ratio = 4^(slb_coh - target).
    d_ratio_for_1 = 4.0 ** (slb_coh - body_at_1)
    d_ratio_dead = 4.0 ** (slb_coh - slb_dead)
    cos_for_1 = 1.0 - (1.0 - sample_med_cos) * d_ratio_for_1

    print(f"\nHOLE 1 -- specialized context models")
    print(f"  best conditional-entropy reduction found anywhere: {best_gain:+.4f} bits/elem")
    print(f"  {'CONFIRMED memoryless' if best_gain < 0.02 else 'STRUCTURE FOUND -- bound weakens'}")
    print(f"\nHOLE 2 -- anchor")
    print(f"  artifact replaced-tensor cosine: min {art_cos['min']:.6f}, median {art_cos['median']:.6f}, n {art_cos['n']}")
    print(f"  sampled q3 mean hold {sample_med_cos:.6f} -> SLB {slb_coh:.4f} bits, complete {complete_bpw(slb_coh):.4f}")
    print(f"  re-anchored on the artifact's WORST tensor ({art_cos['min']:.6f})"
          f" -> SLB {reanchored:.4f} bits, complete {reanchored_bpw:.4f}")
    print(f"\n  to reach complete BPW 1.0 the body would need {body_at_1:.4f} bits/elem,")
    print(f"  i.e. {d_ratio_for_1:.1f}x the gated artifact's distortion (weight cosine ~{cos_for_1:.4f}).")
    print(f"  the DEAD anchor is only {d_ratio_dead:.1f}x that distortion.")
    verdict = ("SURVIVES: reaching 1.0 needs distortion far beyond the level already measured "
               "to kill the model." if d_ratio_for_1 > d_ratio_dead else
               "WEAKENED: 1.0 is reachable at a distortion inside the survivable range.")
    print(f"\n  {verdict}")

    doc = {
        "schema": "hawking.nos.bound_adversarial_review.v1",
        "obligation": "G024 verify line -- adversarial review of the lower-bound evidence",
        "hole_1_memoryless": {
            "attack": "specialized context models over the q3 symbol stream: conditional entropy "
                      "given the row neighbour (contraction axis), the column neighbour (same "
                      "input channel, next output row), and both.",
            "per_tensor": ctx_rows,
            "best_reduction_bits_found": best_gain,
            "verdict": "CONFIRMED MEMORYLESS" if best_gain < 0.02 else "STRUCTURE FOUND",
        },
        "hole_2_anchor": {
            "attack": "re-anchor the distortion on the gated artifact's OWN worst replaced tensor "
                      "rather than the sample's central estimate, then ask what distortion would "
                      "put the floor at 1.0 complete BPW.",
            "artifact_replaced_tensor_cosine": art_cos,
            "sampled_q3_mean_hold": sample_med_cos,
            "slb_bits_sample_anchor": slb_coh,
            "complete_bpw_sample_anchor": complete_bpw(slb_coh),
            "slb_bits_worst_tensor_anchor": reanchored,
            "complete_bpw_worst_tensor_anchor": reanchored_bpw,
            "body_bits_needed_for_complete_1_0": body_at_1,
            "distortion_multiple_needed_for_1_0": d_ratio_for_1,
            "distortion_multiple_of_the_dead_anchor": d_ratio_dead,
            "implied_weight_cosine_at_1_0": cos_for_1,
        },
        "verdict": verdict,
        "residual_holes": [
            "The cosine-to-MSE conversion assumes quantization error roughly orthogonal to the "
            "weight, which is standard for absmax rounding but is an approximation.",
            "Context models tested are order-1 in two axes. A learned high-order model, or one "
            "conditioned on the group scale, is not covered.",
            "Coherence is still a two-point anchor (q3 alive, q2 dead). The true threshold "
            "between them is not measured, and only an assembled artifact per rung would find it.",
        ],
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
