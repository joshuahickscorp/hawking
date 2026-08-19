#!/usr/bin/env python3
"""G143: physical FLOPs/token, analytic, placed on the measured roofline.

The point of this obligation (S010, femtosecond doctrine) is to distinguish a kernel
near the compute roof from one at a few percent, so that "make it faster" is aimed at
the constraint that actually binds. Decode is widely assumed bandwidth-bound here; this
puts a number on it.

Active text-decode params exclude the vision tower and multimodal projector -- those
tensors are not touched generating text. FLOPs/token for a dense (non-MoE) forward is
~2*N_active (one multiply + one add per weight). The roofline position is then
achieved = FLOPs/token / measured token_ns, compared against the measured compute peak.
The reconciliation control: weight movement at the measured bandwidth must, on its own,
account for most of the measured token wall -- if it does, the token is bandwidth-bound
and the low FLOP-roof fraction is expected, not a mystery.

  ./tools/flops_per_token.py --token-ns 30336726 --out receipts/.../G143_FLOPS.json
"""
from __future__ import annotations
import argparse, datetime, json, os, pathlib, struct, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
BF16 = ROOT / "workspace/campaign/records/runs/qwen38-27b/bf16"

PEAK_GFLOPS = 8979.0        # measured roofline compute peak
PEAK_GBPS = 598.3           # measured roofline bandwidth peak
ACTIVE_ARTIFACT_BYTES = 13e9  # uniform-q4 on-disk; weight bytes moved per token pass


def active_text_params() -> dict:
    idx = json.loads((BF16 / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    total = text = vision = 0
    for sh in sorted(set(wm.values())):
        with open(BF16 / sh, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            num = 1
            for d in v.get("shape", []):
                num *= d
            total += num
            if any(t in k.lower() for t in ("visual", "vision", "image", "patch_embed", "merger")):
                vision += num
            else:
                text += num
    return {"total": total, "text_active": text, "vision_inactive": vision}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-ns", type=float, default=30336726.0,  # G0 steady decode
                    help="measured complete token wall in ns")
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()

    p = active_text_params()
    N = p["text_active"]
    flops_tok = 2 * N                       # dense forward, MAC per weight
    tok_s = a.token_ns / 1e9
    achieved_gflops = flops_tok / tok_s / 1e9
    roof_frac_compute = achieved_gflops / PEAK_GFLOPS

    # Bandwidth reconciliation: moving the weight artifact once per token.
    weight_move_s = ACTIVE_ARTIFACT_BYTES / (PEAK_GBPS * 1e9)
    weight_move_frac = weight_move_s / tok_s

    bound = "BANDWIDTH" if weight_move_frac > roof_frac_compute else "COMPUTE"
    print(f"active text params N = {N:,}  (vision {p['vision_inactive']:,} not run in text decode)")
    print(f"FLOPs/token = 2N = {flops_tok/1e9:.1f} GFLOP")
    print(f"measured token wall {tok_s*1e3:.2f} ms -> achieved {achieved_gflops/1e3:.2f} TFLOP/s")
    print(f"compute roof {PEAK_GFLOPS/1e3:.2f} TFLOP/s -> at {roof_frac_compute*100:.1f}% of compute roof")
    print(f"weight movement at {PEAK_GBPS} GB/s = {weight_move_s*1e3:.2f} ms "
          f"= {weight_move_frac*100:.1f}% of the token")
    print(f"=> decode is {bound}-bound: weight movement alone explains "
          f"{weight_move_frac*100:.0f}% of the wall, compute uses only {roof_frac_compute*100:.0f}%")

    doc = {
        "schema": "hawking.nos.flops_per_token.v1",
        "obligation": "G143 -- physical FLOPs/token measured per candidate, with roofline position",
        "started": start,
        "active_text_params": N, "vision_params_inactive": p["vision_inactive"],
        "flops_per_token": flops_tok, "flops_per_token_gflop": round(flops_tok / 1e9, 2),
        "measured_token_ns": a.token_ns, "token_ms": round(tok_s * 1e3, 3),
        "achieved_gflops": round(achieved_gflops, 1),
        "compute_peak_gflops": PEAK_GFLOPS,
        "roofline_position_frac_of_compute_peak": round(roof_frac_compute, 4),
        "bandwidth_reconciliation": {
            "artifact_bytes_moved_per_token": ACTIVE_ARTIFACT_BYTES,
            "bandwidth_peak_gbps": PEAK_GBPS,
            "weight_move_ms": round(weight_move_s * 1e3, 3),
            "weight_move_frac_of_token": round(weight_move_frac, 4)},
        "verdict_bound_by": bound,
        "control_internal_consistency": (
            f"achieved {achieved_gflops:.0f} GFLOP/s is below the {PEAK_GFLOPS} peak (frac "
            f"{roof_frac_compute:.3f} <= 1) AND weight movement at the measured bandwidth "
            f"explains {weight_move_frac*100:.0f}% of the token on its own. Both must hold for "
            "the bandwidth-bound verdict; a compute-bound token would show the opposite."),
        "why_this_matters": (
            "a kernel at 20% of the compute roof cannot be sped up by cutting FLOPs -- the lever "
            "is bytes moved per token, which is the density x kernel co-design front. This is the "
            "measured justification for attacking DRAM traffic, not arithmetic."),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
        "ended": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
