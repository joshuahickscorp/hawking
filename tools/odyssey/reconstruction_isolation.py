#!/usr/bin/env python3
"""Which representation change broke the 2.60 body, measured without building anything.

The two bodies differ in exactly two places. Reconstructing each organ's weights from the
artifact and comparing against the bf16 parent prices both changes on the same scale, so
they can be compared directly instead of argued about.

Every decoder here is SELF-CHECKED against an independently written reference before it is
used on a file. The first version of this measurement reported attention at cosine 0.68
because the offset-binary offset was wrong -- it is `bound = (1<<(bits-1))-1`, not
`1<<(bits-1)` -- and that wrong number pointed at the right organ for the wrong reason.
A decoder nobody has round-tripped is not a decoder.
"""
import argparse, json, struct, subprocess, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/headless"))
CLEAN = Path("/Users/scammermike/noetic/CLEAN_REBUILD_A/mix_hetero_n041_floors/segments")
SEALED = Path("/Users/scammermike/noetic/NOETIC_PARENT_A/segments")


def _hdr(payload):
    hl = struct.unpack_from("<I", payload, 8)[0]
    return json.loads(payload[12:12 + hl]), payload[12 + hl:]


def recon_affine(payload):
    """HGRAVF01. Two variants: q2f (scale only, w=(q-1.5)*delta) and affine (scale+bias)."""
    h, body = _hdr(payload)
    rows, cols = h["shape"]
    g = h["group_size"]
    sb, bb = int(h["scale_bytes"]), int(h.get("bias_bytes") or 0)
    scale = np.frombuffer(body[:sb], dtype=np.float16).astype(np.float32)
    bias = (np.frombuffer(body[sb:sb + bb], dtype=np.float16).astype(np.float32)
            if bb else None)
    packed = np.frombuffer(body[sb + bb:], dtype=np.uint8)
    n = rows * cols
    codes = np.empty(n, dtype=np.uint8)
    for s in range(4):
        codes[s::4] = (packed >> np.uint8(2 * s)) & np.uint8(3)
    c = codes[:n].reshape(rows, cols // g, g).astype(np.float32)
    sc = scale.reshape(rows, cols // g, 1)
    out = (c - 1.5) * sc if bias is None else c * sc + bias.reshape(rows, cols // g, 1)
    return out.reshape(rows, cols), h


def recon_uniform(payload):
    """HGRAVU01 grouped absmax. Codes are offset-binary with offset BOUND."""
    h, body = _hdr(payload)
    rows, cols = h["shape"]
    g, bits = h["group_size"], int(h["bits"])
    sb = int(h["scale_bytes"])
    scale = np.frombuffer(body[:sb], dtype=np.float16).astype(np.float32)
    packed = np.frombuffer(body[sb:], dtype=np.uint8)
    n = rows * cols
    bs = np.unpackbits(packed, bitorder="little")
    codes = np.zeros(n, dtype=np.int32)
    for b in range(bits):
        codes |= bs[b::bits][:n].astype(np.int32) << b
    bound = (1 << (bits - 1)) - 1
    out = (codes - bound).reshape(rows, cols // g, g).astype(np.float32) \
        * scale.reshape(rows, cols // g, 1)
    return out.reshape(rows, cols), h


def rtn(Wm, bits, g):
    """Independently written reference, used only to validate the file decoder."""
    rows, cols = Wm.shape
    Wg = Wm.reshape(rows, cols // g, g)
    amax = np.abs(Wg).max(-1, keepdims=True)
    bound = (1 << (bits - 1)) - 1
    s = np.where(amax > 0, amax / bound, 1.0).astype(np.float32).astype(np.float64)
    q = np.clip(np.rint(Wg / s), -bound, bound)
    return (q * s).reshape(rows, cols)


def cos(a, b):
    return float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", required=True)
    a = ap.parse_args()

    import whole_model_native as w
    rows = {r["name"]: r for r in w.load_q4_manifest(w.Q4_ROOT)["tensors"]}
    src = w.SourceBF16(w.PARENT_BF16)

    # ---- decoder self-check, before any file is trusted ----
    probe = "language_model.model.layers.3.self_attn.q_proj.weight"
    Wp = w.load_parent_matrix(src, probe, [int(x) for x in rows[probe]["shape"]])
    rt, _ = recon_uniform(w.pack_hgravu01(Wp.astype(np.float32), 3, 128))
    ref = rtn(Wp.astype(np.float64), 3, 128)
    selfcheck = {"tensor": probe, "round_trip_cos": cos(rt, Wp),
                 "independent_reference_cos": cos(ref, Wp)}
    selfcheck["agrees"] = abs(selfcheck["round_trip_cos"]
                              - selfcheck["independent_reference_cos"]) < 1e-3
    if not selfcheck["agrees"]:
        raise SystemExit(f"decoder disagrees with the reference: {selfcheck}")

    mlp = [n for n in rows if ".mlp." in n and n.endswith(".weight")][:3]
    other = ([n for n in rows if ".self_attn." in n and n.endswith(".weight")][:3]
             + [n for n in rows if "embed_tokens" in n][:1]
             + [n for n in rows if ".linear_attn.out_proj" in n][:1])

    per_tensor = []
    for nm, group in [(n, "mlp") for n in mlp] + [(n, "non_mlp") for n in other]:
        r = rows[nm]
        W = w.load_parent_matrix(src, nm, [int(x) for x in r["shape"]])
        if W.ndim != 2:
            continue
        stem = Path(r["artifact"]).stem
        row = {"tensor": nm, "organ_group": group}
        cf = CLEAN / (stem + (".hgrafv01" if group == "mlp" else ".hgravu01"))
        if cf.exists():
            wh, h = (recon_affine if group == "mlp" else recon_uniform)(cf.read_bytes())
            if wh.shape == W.shape:
                row["clean_2.60_cos"] = cos(wh, W)
                row["clean_codec"] = h.get("representation")
        sf = SEALED / (stem + ".hgrafv01")
        if group == "mlp" and sf.exists():
            wh, h = recon_affine(sf.read_bytes())
            if wh.shape == W.shape:
                row["sealed_3.14_cos"] = cos(wh, W)
                row["sealed_codec"] = h.get("representation")
        if group == "non_mlp" and W.shape[1] % 64 == 0:
            # the sealed body keeps these at q4 in the older HQ30UQ4 container, whose
            # header is not JSON; the q4 fidelity is computed with the validated reference
            row["sealed_3.14_cos"] = cos(rtn(W.astype(np.float64), 4, 64), W)
            row["sealed_codec"] = "ws_rtn_q4_g64 (reference-computed; HQ30UQ4 container "
            row["sealed_codec"] += "has a non-JSON header)"
        per_tensor.append(row)

    def mean(group, key):
        v = [r[key] for r in per_tensor if r["organ_group"] == group and key in r]
        return sum(v) / len(v) if v else None

    summary = {}
    for grp in ("mlp", "non_mlp"):
        s, c = mean(grp, "sealed_3.14_cos"), mean(grp, "clean_2.60_cos")
        summary[grp] = {"sealed_mean_cos": s, "clean_mean_cos": c,
                        "drop": (s - c) if (s and c) else None,
                        "n_tensors": sum(1 for r in per_tensor if r["organ_group"] == grp)}
    d_mlp = summary["mlp"]["drop"]
    d_other = summary["non_mlp"]["drop"]
    ratio = (d_other / d_mlp) if (d_mlp and d_other) else None

    out = {
        "schema": "hawking.headless.reconstruction_isolation.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/reconstruction_isolation.py",
        "obligation": "G032 — which representation change broke whole-model capability",
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "hand_authored": False,
        "decoder_self_check": selfcheck,
        "why_self_check": (
            "the first run of this measurement reported attention at cosine 0.68 because "
            "the offset-binary offset was taken as 1<<(bits-1) instead of (1<<(bits-1))-1. "
            "It pointed at the right organ for the wrong reason. Every decoder is now "
            "round-tripped against an independently written reference before use."),
        "per_tensor": per_tensor,
        "summary": summary,
        "non_mlp_drop_over_mlp_drop": round(ratio, 2) if ratio else None,
        "finding": (
            "the MLP change is nearly free in fidelity terms and the non-MLP change is not. "
            "Going from the sealed body to the 2.60 body costs the MLP about "
            f"{d_mlp:.4f} of cosine and the attention/state/embed/head organs about "
            f"{d_other:.4f} -- {ratio:.1f}x more. The MLP sits at ~0.94 in BOTH bodies and "
            "the sealed body works, so 0.94 on the MLP is survivable; dropping the "
            "residual-stream organs from ~0.994 to ~0.961 is what the composed model does "
            "not survive. THIS READING IS REFUTED -- see REFUTED_BY_PHYSICAL_MEASUREMENT."
            if (d_mlp and d_other) else "insufficient paired data"),
        "REFUTED_BY_PHYSICAL_MEASUREMENT": {
            "what_this_receipt_predicted":
                "the non-MLP drop is 6.2x the MLP drop, so restoring the non-MLP organs to "
                "q4 should recover most of the lost capability",
            "what_was_measured":
                "a body with the MLP at 2.25 and the non-MLP organs restored to q4 scores "
                "8 of 43. The sealed body, which differs from it ONLY in the MLP codec, "
                "scores 27 of 43. Restoring the non-MLP organs is worth +5 points; changing "
                "the MLP is worth +19.",
            "conclusion":
                "weight-space cosine got the ORDERING BACKWARDS. The change with the 6.2x "
                "SMALLER weight-space drop is the one that costs most of the capability.",
            "why_this_was_predictable":
                "it is the campaign's own transferred law, stated in the transfer report as "
                "TR-METHOD-HELDOUT-ACTIVATIONS: judge a candidate on held-out REAL "
                "activations, never on weight-space error. This measurement used weight-space "
                "error and inverted the answer, which is the law demonstrating itself.",
            "evidence": ["receipts/headless/CAPABILITY_noetic-variantA-2.98.json",
                         "receipts/headless/CAPABILITY_noetic-sealed-3.14.json",
                         "receipts/headless/CAPABILITY_noetic-clean-2.60.json"],
        },
        "pass": bool(selfcheck["agrees"] and d_mlp is not None and d_other is not None),
    }
    Path(a.emit).write_text(json.dumps(out, indent=1))
    print(f"decoder self-check: round_trip={selfcheck['round_trip_cos']:.6f} "
          f"reference={selfcheck['independent_reference_cos']:.6f} agrees={selfcheck['agrees']}")
    for g in ("mlp", "non_mlp"):
        s = summary[g]
        print(f"  {g:8} sealed={s['sealed_mean_cos']:.6f} clean={s['clean_mean_cos']:.6f} "
              f"drop={s['drop']:.4f} (n={s['n_tensors']})")
    print(f"  non-MLP drop is {ratio:.1f}x the MLP drop")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
