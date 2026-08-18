#!/usr/bin/env python3
"""Rank codecs by the error they cause in FUNCTION space, on real activations.

Every codec comparison in this campaign so far has been scored in weight space:
cosine(W, dequant(W)). That is a proxy, and this repository's own history says
proxies mislead here -- the Q30 fits were calibrated on a broken model, and the
sub-bit negatives turned out to be artefacts of synthetic activations. G036 says
it directly: fit against the teacher function, not weight cosine.

The question that matters is what a codec does to W x for the x the model
actually sees. This measures that against the thick v2 capture (real BF16
activations, 23216 tokens, five sites, adequacy-gated), for every codec on the
board:

  q4 / q3 / q2 grouped absmax        the incumbent family
  1 / 2 / 3 binary planes            the G033 ladder

The reason it is worth doing before a packer exists: weight-space hold puts two
binary planes at 0.933975, between q3's coherent 0.968397 and q2's dead 0.772929.
That interval is exactly where a proxy cannot be trusted, and an assembled
artifact is expensive. Function space is the cheaper discriminator.

  ./tools/gravity_function_space_rank.py --layers 0,31,63 --rows 1024 \
      --out receipts/ascent-2026-08-16/G033_FUNCTION_SPACE_RANK.json
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor, quantize_group  # noqa: E402
from gravity_planes_ladder import binary_planes, flat_bits, SCALE_BITS  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAP = ROOT / "workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"

# Which capture site feeds which organ. VERIFIED against the capture rather than
# assumed from site names: silu(gate(x))*up(x) reproduces the captured post_swiglu
# to cosine 0.999986 and rel_err 0.004 when x is post_attn_norm, and to cosine
# 0.48 / rel_err 76.3 when x is post_input_norm. post_input_norm feeds the MIXER;
# post_attn_norm is the MLP input. An earlier revision of this file had it wrong
# and every gate/up number it produced was measured at the wrong operating point.
SITE = {"gate_proj": "post_attn_norm", "up_proj": "post_attn_norm",
        "down_proj": "post_swiglu"}


def load_activations(site, layer, rows, width):
    p = CAP / site / f"L{layer:02d}.f16"
    n = p.stat().st_size // (2 * width)
    take = min(rows, n)
    # Head of the file is as good as any slice and keeps reruns comparable.
    raw = np.fromfile(p, dtype=np.float16, count=take * width)
    return raw.reshape(take, width).astype(np.float32), n


def out_error(x, w, w_hat):
    """Relative Frobenius error and mean per-row cosine of the OUTPUT."""
    y = x @ w.T
    yh = x @ w_hat.T
    d = yh - y
    rel = float(np.linalg.norm(d) / np.linalg.norm(y))
    ny = np.linalg.norm(y, axis=1)
    nyh = np.linalg.norm(yh, axis=1)
    ok = (ny > 0) & (nyh > 0)
    cos = float((np.einsum("ij,ij->i", y[ok], yh[ok]) / (ny[ok] * nyh[ok])).mean())
    return rel, cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,31,63")
    ap.add_argument("--organs", default="gate_proj,down_proj")
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    schemes = ([("flat", b, flat_bits(b, a.group)) for b in (2, 3, 4)] +
               [("planes", k, k * (1.0 + SCALE_BITS / a.group)) for k in (1, 2, 3)])

    rows_out, agg = [], {}
    for layer in [int(x) for x in a.layers.split(",")]:
        for organ in a.organs.split(","):
            name = f"language_model.model.layers.{layer}.mlp.{organ}.weight"
            w = load_tensor(name).astype(np.float32)
            x, n_avail = load_activations(SITE[organ], layer, a.rows, w.shape[1])
            entry = {"tensor": name, "shape": list(w.shape), "site": SITE[organ],
                     "rows_used": int(x.shape[0]), "rows_available": int(n_avail),
                     "schemes": []}
            for kind, param, bits in schemes:
                if kind == "flat":
                    w_hat, _ = quantize_group(w, param, a.group)
                    label = f"flat q{param}"
                else:
                    w_hat, _ = binary_planes(w, param, a.group)
                    label = f"{param} binary plane" + ("s" if param > 1 else "")
                rel, cos = out_error(x, w, w_hat)
                e = {"scheme": label, "bits_per_elem": bits,
                     "output_rel_fro": rel, "output_mean_row_cosine": cos}
                entry["schemes"].append(e)
                agg.setdefault(label, []).append(e)
                del w_hat
            rows_out.append(entry)
            print(f"  {layer:>2} {organ:<10} rows={x.shape[0]}/{n_avail}")
            del w, x

    table = []
    for label, v in agg.items():
        table.append({"scheme": label, "bits_per_elem": v[0]["bits_per_elem"],
                      "mean_output_rel_fro": sum(e["output_rel_fro"] for e in v) / len(v),
                      "worst_output_rel_fro": max(e["output_rel_fro"] for e in v),
                      "mean_output_row_cosine": sum(e["output_mean_row_cosine"] for e in v) / len(v),
                      "worst_output_row_cosine": min(e["output_mean_row_cosine"] for e in v)})
    table.sort(key=lambda r: r["bits_per_elem"])

    print(f"\n{'scheme':<18}{'bits/elem':>10}{'out rel_fro':>13}{'worst':>10}"
          f"{'out cosine':>12}{'worst':>10}")
    for r in table:
        print(f"{r['scheme']:<18}{r['bits_per_elem']:>10.4f}{r['mean_output_rel_fro']:>13.5f}"
              f"{r['worst_output_rel_fro']:>10.5f}{r['mean_output_row_cosine']:>12.6f}"
              f"{r['worst_output_row_cosine']:>10.6f}")

    # The two anchors whose behaviour is KNOWN, so the unknown can be placed.
    q3 = next(r for r in table if r["scheme"] == "flat q3")
    q2 = next(r for r in table if r["scheme"] == "flat q2")
    p2 = next(r for r in table if r["scheme"] == "2 binary planes")
    span = q2["mean_output_rel_fro"] - q3["mean_output_rel_fro"]
    pos = ((p2["mean_output_rel_fro"] - q3["mean_output_rel_fro"]) / span) if span else None
    print(f"\nanchors: flat q3 is COHERENT (10/10 gated), flat q2 is DEAD.")
    print(f"2 binary planes sit at {pos:.3f} of the way from q3 to q2 in output error "
          f"({p2['mean_output_rel_fro']:.5f} against q3 {q3['mean_output_rel_fro']:.5f} "
          f"and q2 {q2['mean_output_rel_fro']:.5f})")

    doc = {
        "schema": "hawking.nos.function_space_codec_rank.v1",
        "obligation": "G033 / G036 -- rank representations by functional error, not weight cosine",
        "capture": {"root": str(CAP.relative_to(ROOT)),
                    "schema": "hawking.ascension.qwen38_activation_capture.v2",
                    "note": "real BF16 parent activations, adequacy-gated, NOT synthetic"},
        "method": "y = x W^T on real captured x; error is relative Frobenius of the OUTPUT and "
                  "mean per-row output cosine. Both sides of every scheme charge their scale "
                  "streams in bits_per_elem.",
        "anchors": "flat q3 is a gated-coherent artifact (10/10, controls watched); flat q2 is "
                   "recorded dead. They bracket the unknown.",
        "table": table,
        "two_plane_position_between_q3_and_q2": pos,
        "per_tensor": rows_out,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
