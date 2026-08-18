#!/usr/bin/env python3
"""G043: the FLOP family, and whether a codec's decode eats the win its bytes bought.

The verify line is the whole point: reconstruction FLOPs must be shown NOT to
consume the latency saved by the bytes avoided, and a representation whose decode
eats its own win is a NET-LOSS and must be labelled so. This campaign already has
exactly such a representation, measured end to end -- the q3 density leader is
17.4% fewer bytes and 10.9% MORE time -- so the instrument is checked against a
case whose answer is already known from a different direction.

Six FLOP axes plus the measured critical path:

  SOURCE_EQUIVALENT_FLOPS  what a dense fp forward would do: 2 per weight
  PHYSICAL_FLOPS           what actually executes, MACs plus reconstruction
  RECONSTRUCTION_FLOPS     turning stored bits back into floats
  CORRECTION_FLOPS         correction tiers. Zero: no candidate has any
  ROUTING_FLOPS            expert/route selection. Zero: this model is dense
  STATE_UPDATE_FLOPS       DeltaNet recurrent state, from its real state size

Reconstruction ops per weight are DERIVED, not guessed, and the derivation is
stated: NX_MATMUL_K_AMORTIZATION fitted the geo_tpr64 K sweep to T(K) ~ (d + K)
within 4%, which puts q4 at d = 5 decode ops against 1 MAC. Every other codec's d
follows from its MEASURED ps/element relative to q4's, since the MAC count is
identical across codecs -- only the decode differs.

  ./tools/gravity_flop_family.py --out receipts/ascent-2026-08-16/G043_FLOP_FAMILY.json
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
N_PARAMS = 26_895_998_464
GEMV_ELEMS = 17_112_760_320 + 5_560_545_600 + 1_677_720_188 + 1_271_398_400

# q4's decode-op count, from the T(K) ~ (d + K) fit in NX_MATMUL_K_AMORTIZATION.
Q4_DECODE_OPS = 5.0
# DeltaNet recurrent state, from QWEN38_TOKEN_NS_DN_VI_SIMD state_bytes.
REC_STATE_BYTES = 150_994_944
REC_STATE_ELEMS = REC_STATE_BYTES // 4
# Per state element per token the vi kernel does: decay multiply, k*delta
# accumulate, and a query product feeding the output reduction.
STATE_OPS_PER_ELEM = 3.0


def load(p):
    return json.loads((ROOT / p).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    alu = load("receipts/ascent-2026-08-16/CODEC_ALU_COST.json")
    roof = load("receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.json")
    vec = load("receipts/ascent-2026-08-16/G041_COST_VECTOR.json")
    by_codec = {c["codec"]: c for c in alu["results"][-1]["by_codec"]}
    q4_ps = by_codec["q4_group64_hgravu"]["ps_per_element"]
    kernel_roof_gb_s = roof["adjudication"]["kernel_roof_gb_s"]

    # Map each live artifact to the codec its BODY uses.
    artifacts = {
        "uniform-q4-v1": ("q4_group64_hgravu", 4.25),
        "compact-q3attn-r1p2-v1": ("q3_group64_hgravu", 3.25),
        "g032-chanscale-a025-compact": ("q3_group64_hgravu", 3.25),
    }
    measured_L = {r["candidate"]: r for r in vec["candidates"]}

    base_name = "uniform-q4-v1"
    base_codec, base_bits = artifacts[base_name]
    base_bytes = measured_L[base_name]["M_dram_bytes_per_token"]

    rows = []
    for name, (codec, bits) in artifacts.items():
        ps = by_codec[codec]["ps_per_element"]
        # d follows from the measured ps/element ratio: MACs are identical across
        # codecs, so all of the difference is decode.
        d = (Q4_DECODE_OPS + 1.0) * ps / q4_ps - 1.0
        recon = d * GEMV_ELEMS
        macs = 2.0 * GEMV_ELEMS
        state = STATE_OPS_PER_ELEM * REC_STATE_ELEMS
        m = measured_L[name]
        bytes_tok = m["M_dram_bytes_per_token"]

        # The verify test, and it has to be posed in the currency the machine
        # actually charges. The NAIVE model prices avoided bytes at the kernel
        # roof, which assumes decode is bandwidth-bound. F6 measured that it is
        # NOT -- q4 sits at 88% of roof and the binding resource is decode issue
        # rate -- so the naive model is kept here only to be refuted by its own
        # cross-check against the wall clock.
        bytes_avoided = base_bytes - bytes_tok
        naive_bought_ns = bytes_avoided / kernel_roof_gb_s
        base_ps = by_codec[base_codec]["ps_per_element"]
        recon_added_ns = (ps - base_ps) * GEMV_ELEMS / 1000.0
        naive_net_ns = naive_bought_ns - recon_added_ns
        # The ALU model: elements are fixed by the architecture, so GEMV time
        # scales with ps/element and nothing else. Bytes enter only through a
        # bandwidth term that is not binding.
        base_gemv_ns = GEMV_ELEMS * base_ps / 1000.0
        alu_predicted_delta_ns = base_gemv_ns * (ps / base_ps - 1.0)
        net_ns = -alu_predicted_delta_ns
        if bytes_avoided <= 0:
            klass = "BASELINE"
        elif net_ns > 0:
            klass = "NET-WIN"
        else:
            klass = "NET-LOSS"

        rows.append({
            "candidate": name, "body_codec": codec, "body_bits_per_elem": bits,
            "SOURCE_EQUIVALENT_FLOPS": macs,
            "PHYSICAL_FLOPS": macs + recon + state,
            "RECONSTRUCTION_FLOPS": recon,
            "RECONSTRUCTION_SHARE_OF_PHYSICAL": recon / (macs + recon + state),
            "CORRECTION_FLOPS": 0.0,
            "ROUTING_FLOPS": 0.0,
            "STATE_UPDATE_FLOPS": state,
            "decode_ops_per_weight_derived": d,
            "measured_ps_per_element": ps,
            "PHYSICAL_CRITICAL_PATH_NS": m["L_complete_wall_ns_per_token"],
            "measured_tps": m["L_tps"],
            "bytes_avoided_vs_baseline": bytes_avoided,
            "naive_bandwidth_model_bought_ns": naive_bought_ns,
            "naive_bandwidth_model_net_ns": naive_net_ns,
            "naive_bandwidth_model_says": ("NET-WIN" if naive_net_ns > 0 else "NET-LOSS"),
            "latency_the_decode_added_ns": recon_added_ns,
            "alu_model_predicted_delta_ns": alu_predicted_delta_ns,
            "net_ns": net_ns,
            "classification": klass,
        })
        print(f"{name}")
        print(f"  SOURCE_EQUIVALENT {macs/1e9:9.3f} GFLOP   PHYSICAL {(macs+recon+state)/1e9:9.3f} GFLOP")
        print(f"  RECONSTRUCTION    {recon/1e9:9.3f} GFLOP   = {recon/(macs+recon+state)*100:5.2f}% of physical"
              f"   ({d:.2f} decode ops/weight, derived)")
        print(f"  STATE_UPDATE      {state/1e9:9.3f} GFLOP   CORRECTION 0   ROUTING 0 (dense model)")
        print(f"  CRITICAL PATH     {m['L_complete_wall_ns_per_token']:,} ns  ({m['L_tps']:.2f} TPS)")
        if klass != "BASELINE":
            print(f"  bytes avoided {bytes_avoided:,}")
            print(f"    naive bandwidth model: bought {naive_bought_ns/1e6:.3f} ms, decode cost "
                  f"{recon_added_ns/1e6:.3f} ms, net {naive_net_ns/1e6:+.3f} ms -> "
                  f"{'NET-WIN' if naive_net_ns>0 else 'NET-LOSS'}")
            print(f"    ALU model (binding):   {alu_predicted_delta_ns/1e6:+.3f} ms SLOWER "
                  f"-> net {net_ns/1e6:+.3f} ms -> {klass}")
        else:
            print(f"  {klass}")

    # Cross-check the verdict against the independently measured wall clock.
    check = []
    for r in rows:
        if r["classification"] == "BASELINE":
            continue
        measured_delta = (measured_L[base_name]["L_complete_wall_ns_per_token"]
                          - r["PHYSICAL_CRITICAL_PATH_NS"])
        naive_ok = (r["naive_bandwidth_model_net_ns"] > 0) == (measured_delta > 0)
        alu_ok = (r["net_ns"] > 0) == (measured_delta > 0)
        check.append({"candidate": r["candidate"],
                      "measured_wall_delta_ns": measured_delta,
                      "naive_model_net_ns": r["naive_bandwidth_model_net_ns"],
                      "naive_model_sign_agrees": naive_ok,
                      "alu_model_net_ns": r["net_ns"],
                      "alu_model_sign_agrees": alu_ok,
                      "alu_model_abs_error_ns": abs(-r["net_ns"] - (-measured_delta))})
        print(f"\nCROSS-CHECK {r['candidate']}: measured wall delta "
              f"{measured_delta/1e6:+.3f} ms")
        print(f"  naive bandwidth model {r['naive_bandwidth_model_net_ns']/1e6:+.3f} ms  "
              f"{'AGREES' if naive_ok else 'WRONG SIGN -- refuted by the wall clock'}")
        print(f"  ALU model             {r['net_ns']/1e6:+.3f} ms  "
              f"{'AGREES' if alu_ok else 'WRONG SIGN'}"
              f"   |error| {abs(-r['net_ns']-(-measured_delta))/1e6:.3f} ms")

    doc = {
        "schema": "hawking.nos.flop_family.v1",
        "obligation": "G043 -- FLOP family and PHYSICAL_CRITICAL_PATH_NS",
        "derivation_of_decode_ops": (
            "NX_MATMUL_K_AMORTIZATION fitted the geo_tpr64 K sweep to T(K) ~ (d + K) within 4%, "
            "putting q4 at d = 5 decode ops against 1 MAC. Every other codec's d follows from its "
            "MEASURED ps/element relative to q4's, because the MAC count is identical across "
            "codecs and all of the difference is decode. Derived, not guessed, and stated so."),
        "state_update_basis": (
            f"DeltaNet recurrent state is {REC_STATE_BYTES:,} bytes resident "
            f"({REC_STATE_ELEMS:,} f32 elements) from QWEN38_TOKEN_NS_DN_VI_SIMD; the vi kernel "
            f"does a decay multiply, a k*delta accumulate and a query product per element per "
            f"token, so {STATE_OPS_PER_ELEM} ops/element."),
        "candidates": rows,
        "verify_cross_check": check,
        "verify_note": (
            "The obligation requires a representation whose decode eats its own win to be labelled "
            "NET-LOSS. The q3 artifacts are exactly that case and were independently measured "
            "10.9% slower than G0 at 17.4% fewer bytes in DENSITY_LEADER_SPEED, so the "
            "classification is checked against a wall clock rather than resting on the model."),
        "zero_axes": {
            "CORRECTION_FLOPS": "no live candidate has correction tiers",
            "ROUTING_FLOPS": "Qwen3.8 is dense at this level; no expert selection executes",
        },
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
