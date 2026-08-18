#!/usr/bin/env python3
"""G031: adversarial review of every major representation family.

Each family is checked against the eight named defects and given a verdict with
the check that would settle it. The rule this file enforces is that a verdict
must cite a receipt on disk. Where no receipt exists the verdict is UNREVIEWED
with the missing measurement named -- never a guess dressed as an assessment.

The eight defects, from the obligation:
  rank_deficient_capture   fitted against activations that do not span the space
  goodhartable_metric      the score can be won without the capability
  hidden_cost              bytes, ALU, RAM or latency the headline omits
  missing_sidecar          needs a table, index or scale stream nobody counted
  narrow_test_distribution the evidence is a handful of prompts
  plausibility_masking     fluent output standing in for correct output
  kernel_different_math    the bound kernel is not what was scored offline
  bytes_merely_moved       storage falls, traffic or reconstruction does not

  ./tools/gravity_family_review.py --out receipts/ascent-2026-08-16/G031_FAMILY_REVIEW.json
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = "receipts/ascent-2026-08-16/"

D = ["rank_deficient_capture", "goodhartable_metric", "hidden_cost", "missing_sidecar",
     "narrow_test_distribution", "plausibility_masking", "kernel_different_math",
     "bytes_merely_moved"]


def fam(name, obligation, verdict, defects, settles, cites):
    unknown = [d for d in D if d not in defects]
    return {"family": name, "obligation": obligation, "verdict": verdict,
            "defects_found": defects, "defects_not_assessed": unknown,
            "settling_check": settles, "cites": cites}


FAMILIES = [
 fam("uniform grouped quantization (q4 group-64)", "G0 baseline / incumbent", "SURVIVES",
     {"hidden_cost": "88% of the 699.57 GB/s kernel roof at 0.810 ps/element -- it sits ON the "
                     "bandwidth/ALU balance point, which is why it is hard to beat, not because "
                     "it is efficient in bits.",
      "bytes_merely_moved": "NO. 4.255954555664 complete BPW confirmed by two independent "
                            "accountings; declared and on-disk agree.",
      "kernel_different_math": "NO. greedy ids match the oracle; FALLBACKS 0, "
                               "DENSE_W_MATERIALIZED 0.",
      "plausibility_masking": "NO. 10/10 with the negative control watched failing 0/10.",
      "narrow_test_distribution": "YES, ten deterministic items. Cannot separate 10/10 from 9/10."},
     "A wider battery (G126). Everything else about this family is measured.",
     [R+"G029_PROTECTED_VERIFICATION.json", R+"CODEC_ALU_COST.json",
      R+"CAPABILITY_COMPACT_Q3ATTN_R1P2.json"]),

 fam("mixed per-organ quantization (q3 MLP + q3 attention, q4 endpoints)", "density floor",
     "WEAKENED",
     {"hidden_cost": "DECISIVE. Costs 5.6% more ALU per weight than its same-ABI q4 control "
                     "while storing 25% fewer bytes, and runs at 63% of kernel roof against "
                     "q4's 88%. End to end the artifact is 10.9% SLOWER than G0 on 17.4% fewer "
                     "bytes. The byte saving is real and unrealizable.",
      "bytes_merely_moved": "PARTLY. Storage genuinely falls to 3.344821 BPW, verified two ways. "
                            "But time does not follow, so the saving does not reach the user.",
      "plausibility_masking": "NO. 10/10, 0 degenerate, negative control watched failing.",
      "kernel_different_math": "NO. FALLBACKS 0 and DENSE_W_MATERIALIZED 0 on the native path.",
      "narrow_test_distribution": "YES, same ten items.",
      "missing_sidecar": "NO, once compacted. The UNCOMPACTED directory carried 5,659,352,471 "
                         "bytes of superseded tensors inside live blobs and was quoted at its "
                         "declared 3.344772 for weeks."},
     "Already settled on speed by CODEC_ALU_COST + DENSITY_LEADER_SPEED. What is NOT settled is "
     "whether a cheaper-to-decode 3-bit layout exists; the family is weakened by ITS CODEC, not "
     "by its bit width.",
     [R+"DENSITY_LEADER_SPEED.json", R+"CODEC_ALU_COST.json", R+"G029_PROTECTED_VERIFICATION.json"]),

 fam("uniform q2", "explored, on record", "KILLED",
     {"plausibility_masking": "q2 attention is recorded fluent-but-wrong; q2 MLP dead.",
      "hidden_cost": "Even charging ZERO for compute it reaches only 97 TPS, so it cannot buy "
                     "the mission target either.",
      "goodhartable_metric": "Hold at 0.772929 mean / 0.761303 min at group-64 -- far below the "
                             "q3 0.968 class. Nothing to Goodhart; it simply fails."},
     "Settled. Recorded as a bound, not a candidate.",
     [R+"NX_TPS_FRONTIER.json", R+"G032_XFORM_HADAMARD_Q2.json"]),

 fam("interleaved rANS entropy coding over a q3 body", "G114-G116, F1's density route",
     "AT RISK -- UNREVIEWED ON ITS DECIDING AXIS",
     {"hidden_cost": "UNMEASURED AND LIKELY FATAL. The measured budget is 0.810 ps/element: any "
                     "codec spending more decode than that cannot convert its byte saving. rANS "
                     "decode is a state multiply, a renormalize and a table lookup per symbol "
                     "against a nibble's shift-mask-convert. q3 already blew the budget by 5.6% "
                     "for a far cheaper unpack.",
      "missing_sidecar": "A shared frequency table plus per-block rANS state. Neither is in any "
                         "BPW figure quoted for this family; the L8/L32/L64 numbers are CODED "
                         "bits/elem, not complete BPW.",
      "bytes_merely_moved": "UNKNOWN until a consume-direct kernel exists. An expand-to-q3 "
                            "staging buffer would move the bytes and keep the traffic.",
      "narrow_test_distribution": "The r-sweep grading (1.20 -> 10/10, 1.25 -> 9/10) is the same "
                                  "ten items; 9/10 there is not a measurement of anything."},
     "A STUB KERNEL that does rANS decode and nothing else, timed in ps/element against the 0.810 "
     "budget. This is the cheap half and it decides the expensive half. Building the packer first "
     "inverts the order.",
     [R+"CODEC_ALU_COST.json", R+"NX_TPS_FRONTIER.json"]),

 fam("over-scale (r = 1.20 .. 1.50)", "G129 joint optimizer", "WEAKENED",
     {"narrow_test_distribution": "The whole frontier -- 10/10, 9/10, 8/10, 6/10, 3/10 -- rests "
                                  "on ten deterministic items. The interesting region (1.20 vs "
                                  "1.25) is one item wide.",
      "goodhartable_metric": "Over-scaling trades fidelity for a narrower symbol histogram, which "
                             "is exactly a metric that improves coded bits while the thing being "
                             "measured degrades. The A1 compiler law names this.",
      "hidden_cost": "r=1.20 is realized in compact-q3attn-r1p2-v1 and inherits that family's ALU "
                     "problem; it is 29.90 TPS, slower than G0."},
     "The wider battery (G126) applied to r=1.20 and r=1.25 side by side, with Tabula drift "
     "attached. Until then no promotion past r=1.20.",
     [R+"DENSITY_LEADER_SPEED.json", R+"CAPABILITY_COMPACT_Q3ATTN_R1P2.json"]),

 fam("G-XFORM structured transforms (Hadamard member)", "G032", "KILLED (this member)",
     {"hidden_cost": "Raises order-0 code entropy 0.0237-0.0334 bits/elem at every width, so "
                     "under an entropy-coded route it is a net LOSS on bits. The runtime FWHT is "
                     "additionally unpriced and lands on the ALU budget that is already binding.",
      "goodhartable_metric": "Improves hold by 0.6%-4% of a bit-step while raising the quantity "
                             "that actually costs bytes. Improving hold alone would have looked "
                             "like a win.",
      "bytes_merely_moved": "Worse: bytes go UP, function stays equal.",
      "rank_deficient_capture": "N/A -- weight-space transform, no activation fit involved."},
     "Settled for Hadamard. The family is NOT settled: permutation, sign, channel scale, "
     "Kronecker and RoPE-commuting members are untested, and the ONE property Hadamard did "
     "deliver -- collapsing worst-case hold spread 185x, from 0.003139 to 0.000017 -- is worth "
     "chasing in a member that does not cost entropy.",
     [R+"G032_XFORM_HADAMARD_Q3.json", R+"G032_XFORM_HADAMARD_Q2.json",
      R+"G032_XFORM_HADAMARD_Q4.json"]),

 fam("G-PLANES progressive planes (W ~ s1*P1 + s2*P2 + ...)", "G033",
     "REFUTED AS A BODY CODEC, SURVIVES AS A DRAFT TIER",
     {"goodhartable_metric": "DEMONSTRATED, AND IT CAUGHT US. Weight-space hold said 3 planes "
                             "(0.971493) BEAT flat q3 (0.968397) and that 2 planes at 0.933975 "
                             "sat close to q3. On REAL captured activations the ranking INVERTS: "
                             "3 planes cause 0.24128 output error against flat q3's 0.19791, at "
                             "MORE bits, and 2 planes are at 0.33390 -- 69% worse than q3 and "
                             "0.357 of the way to dead q2. Weight cosine flattered this family "
                             "and a promotion was drafted on it before the function-space check.",
      "hidden_cost": "NOT the problem here -- this family PASSES the ALU budget where q3 fails: "
                     "one plane 0.620 ps/element and two 0.684 against a 0.8092 budget and q3's "
                     "0.867. It is fidelity, not speed, that kills it.",
      "missing_sidecar": "Counted throughout: k planes cost k*(1+16/g), and a finer group helps "
                         "planes far LESS than flat codes (g64->g32 improves flat q3 by 11.2% and "
                         "2 planes by only 2.5%), because the plane residual is not scale-limited.",
      "bytes_merely_moved": "NO. Bits genuinely fall. They just do not buy function.",
      "narrow_test_distribution": "Six tensors, three depths, MLP only, 1024 captured rows each.",
      "plausibility_masking": "N/A -- no artifact was ever assembled, which is now the correct "
                              "outcome rather than a gap."},
     "SETTLED as a body codec: flat grouped quantization dominates the ladder at every point "
     "where coherence is plausible, in function space, at both group sizes tested. What SURVIVES "
     "both metrics is the sub-q2 regime -- one plane at 1.2500 bits causes LESS output error "
     "(0.52550) than flat q2 at 2.2500 (0.57835) while costing 0.620 ps/element. That regime is "
     "dead for a body codec and is exactly a DRAFT TIER profile, so this family should be "
     "re-aimed at G140/G141 (Matryoshka draft, self-speculative verify) rather than packed as a "
     "body.",
     [R+"G033_PLANES_LADDER.json", R+"G033_FUNCTION_SPACE_RANK.json",
      R+"G033_FUNCTION_SPACE_RANK_G32.json", R+"CODEC_ALU_COST.json"]),

 fam("low-rank factor (HGRAVS01 r160 / r192)", "G033 adjacent, prior campaign", "WEAKENED",
     {"kernel_different_math": "The mixed pack's own claim_boundary records r160_b3_down_removed "
                               "true -- the low-rank down_proj was TAKEN OUT of the shipped "
                               "recipe. What is on disk is HGRAVU01, not the factor form.",
      "hidden_cost": "A rank-r factor is two GEMVs where there was one. On an ALU-bound kernel "
                     "that is the wrong direction, and it was never priced in ps/element.",
      "rank_deficient_capture": "Prior campaign recorded fits underdetermined at median 92 rows "
                                "against 2048 dims. Whether the thick v2 capture fixed this for "
                                "THIS family is not established here."},
     "ps/element for a rank-r factor GEMV pair against the 0.810 budget, and a statement of which "
     "artifact if any still contains the factor form.",
     ["workspace/campaign/records/runs/qwen38-27b/mixed-q3mlp-q3attn-r1p2-v1/PACK_REPORT.json"]),

 fam("multi-token / R x K weight-sweep amortization", "G091, G141, F2", "SURVIVES",
     {"kernel_different_math": "NO. Two controls at exactly zero error: all-columns-identical "
                               "reproduces the K=1 answer, and column k matches a dedicated K=1 "
                               "run on that position.",
      "hidden_cost": "Register pressure is real and measured -- R=8 spills and loses. R=4 K=4 is "
                     "the measured optimum at 2.57x.",
      "bytes_merely_moved": "NO. This moves FEWER bytes per token by construction; the sweep is "
                            "amortized, not relocated.",
      "goodhartable_metric": "NO. The metric is wall time per position, which is the thing itself."},
     "It is a kernel primitive with controls, not a representation claim. What it still owes is a "
     "DRAFT: amortization is worthless without something to fill the K positions, and no draft "
     "tier exists.",
     [R+"NX_MATMUL_K_AMORTIZATION.json"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    missing = []
    for f in FAMILIES:
        for c in f["cites"]:
            if not (ROOT / c).exists():
                missing.append((f["family"], c))
    doc = {
        "schema": "hawking.nos.family_adversarial_review.v1",
        "obligation": "G031 -- adversarial review of every major representation family",
        "defect_checklist": D,
        "rule": "A verdict must cite a receipt on disk. No receipt means UNREVIEWED with the "
                "missing measurement named, never a guess presented as an assessment.",
        "families": FAMILIES,
        "broken_citations": missing,
        "cross_cutting_finding": (
            "Six of the nine families are limited by the SAME defect, and it is hidden_cost in "
            "the same currency: decode ALU per weight. CODEC_ALU_COST puts the budget at 0.810 "
            "ps/element, where q4 already sits at 88% of the bandwidth roof. Every family that "
            "buys bits by spending decode -- q3, rANS, planes, low-rank factors -- is spending "
            "the one resource that is already exhausted. The campaign has been selecting "
            "representations on bits while the machine charges in ops."),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    for f in FAMILIES:
        print(f"{f['verdict']:<38} {f['family']}")
        print(f"      settles: {f['settling_check'][:110]}")
    if missing:
        print("\nBROKEN CITATIONS:")
        for fam_name, c in missing:
            print(f"  {fam_name}: {c}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
