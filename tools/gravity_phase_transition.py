#!/usr/bin/env python3
"""G067: locate every live family's cliff as a NUMBER, on real activations.

The obligation is explicit that the cliff must be found by measurement and
reported as a number rather than an adjective, and that no candidate may be
optimized past it. This campaign has been running on scattered anecdotes instead:
q4 coherent, q3 coherent, q2 dead, depth-weighted q2 dead. Those are four points
with no curve through them and no cliff located on any axis but one.

Three live families are swept, each on its own cost parameter:

  FLAT    grouped absmax, bits 2..6 at group 64        bits/elem = b + 16/g
  PLANES  greedy residual binarization, k = 1..3       bits/elem = k*(1 + 16/g)
  GROUP   flat 4-bit with the group size swept          bits/elem = 4 + 16/g

Error is measured in FUNCTION space on the thick v2 capture, never weight cosine:
this repository's Q30 fits were calibrated on a broken model and its sub-bit
negatives were artefacts of synthetic activations, and G036 says directly to fit
against the teacher function.

Capture sites, with their provenance labelled rather than assumed -- an earlier
revision of the function-space tool had a site wrong and every number it produced
was measured at the wrong operating point:

  gate_proj   post_attn_norm   VERIFIED: silu(gate(x))*up(x) reproduces the
                               captured post_swiglu at cosine 0.999986 from this
                               site and 0.48 from post_input_norm
  down_proj   post_swiglu      VERIFIED by the same reconstruction
  out_proj    mixer_x          NAMED BY THE CAPTURE: mixer_x_kind is
  o_proj      mixer_x          "..._out_proj_input" for every layer
  q_proj      post_input_norm  INFERRED from the block structure whose other half
                               was verified above. Weaker provenance, said so.

Two cliff numbers per family, by two independent rules, because a single rule
that happens to be circular would go unnoticed:

  INTRINSIC  the bits/elem of maximum curvature in log(error) vs bits -- a
             property of the family's own curve, anchored to nothing
  ANCHORED   the bits/elem at which the family's error crosses the error of flat
             q2 at the same site, interpolated. q2 is the recorded DEAD artifact,
             so this transfers a real capability verdict across families at
             matched cost. For the FLAT family this anchor is a family member, so
             its anchored number is the anchor by construction and only the
             intrinsic number is informative there.

  ./tools/gravity_phase_transition.py --rows 384 --out receipts/.../G067_PTP.json
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor, quantize_group  # noqa: E402
from gravity_planes_ladder import binary_planes, SCALE_BITS  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAP = ROOT / "workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"

# organ -> (tensor suffix, site, site width, provenance, layers)
ORGANS = {
    "mlp.gate_proj":       ("mlp.gate_proj.weight", "post_attn_norm", 5120, "VERIFIED", [31, 63]),
    "mlp.down_proj":       ("mlp.down_proj.weight", "post_swiglu", 17408, "VERIFIED", [31, 63]),
    "linear_attn.out_proj": ("linear_attn.out_proj.weight", "mixer_x", 6144, "NAMED", [30, 62]),
    "self_attn.o_proj":    ("self_attn.o_proj.weight", "mixer_x", 6144, "NAMED", [31, 63]),
    "self_attn.q_proj":    ("self_attn.q_proj.weight", "post_input_norm", 5120, "INFERRED", [31, 63]),
}


def acts(site, layer, width, rows):
    p = CAP / site / f"L{layer:02d}.f16"
    avail = p.stat().st_size // (2 * width)
    take = min(rows, avail)
    return np.fromfile(p, dtype=np.float16, count=take * width).reshape(take, width).astype(np.float32)


def out_rel(x, w, w_hat):
    y = x @ w.T
    d = x @ w_hat.T - y
    return float(np.linalg.norm(d) / np.linalg.norm(y))


def schemes(group):
    s = [(f"flat q{b}", "FLAT", b + SCALE_BITS / group, ("flat", b, group))
         for b in (2, 3, 4, 5, 6, 7, 8)]
    s += [(f"{k} plane" + ("s" if k > 1 else ""), "PLANES", k * (1 + SCALE_BITS / group),
           ("planes", k, group)) for k in (1, 2, 3, 4, 5)]
    s += [(f"flat q4 g{g}", "GROUP", 4 + SCALE_BITS / g, ("flat", 4, g))
          for g in (16, 32, 128, 256, 512)]
    return s


def cliff_intrinsic(pts):
    """bits/elem of maximum curvature in log(error) vs bits, WITH the degeneracy
    guard that a first pass here needed. Curvature can only be evaluated at
    interior points, so a 3-point grid has exactly one candidate and returns it no
    matter what the curve does -- a forced answer that reads like a measurement.
    And if the argmax sits on the edge of the interior set, the real maximum is
    outside the swept range or absent, so the number is not a located cliff."""
    pts = sorted(pts)
    if len(pts) < 5:
        return {"bits": None, "status": "UNDEFINED -- fewer than 3 interior candidates",
                "interior_candidates": max(0, len(pts) - 2)}
    b = np.array([p[0] for p in pts]); e = np.log(np.array([p[1] for p in pts]) + 1e-12)
    d2 = [(b[i], abs((e[i+1] - e[i]) / (b[i+1] - b[i]) - (e[i] - e[i-1]) / (b[i] - b[i-1])))
          for i in range(1, len(b) - 1)]
    k = int(np.argmax([c for _, c in d2]))
    mono = all(d2[i][1] >= d2[i + 1][1] for i in range(len(d2) - 1))
    if k == 0 or k == len(d2) - 1:
        return {"bits": float(d2[k][0]), "status":
                ("NO CLIFF -- curvature is monotone across the swept range, so the argmax is an "
                 "edge of the interior set and not a located transition" if mono else
                 "EDGE -- argmax on the interior boundary; the maximum is outside the sweep"),
                "interior_candidates": len(d2),
                "curvature": [float(c) for _, c in d2]}
    return {"bits": float(d2[k][0]), "status": "LOCATED", "interior_candidates": len(d2),
            "curvature": [float(c) for _, c in d2]}


def cliff_anchored(pts, dead_err):
    """bits/elem where the curve crosses the dead anchor's error, interpolated."""
    pts = sorted(pts)
    for i in range(len(pts) - 1):
        (b0, e0), (b1, e1) = pts[i], pts[i + 1]
        # error falls as bits rise, so the crossing is where it drops below dead
        if (e0 - dead_err) * (e1 - dead_err) <= 0 and e0 != e1:
            return float(b0 + (b1 - b0) * (dead_err - e0) / (e1 - e0))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=384)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    sch = schemes(a.group)
    per_site, fam_pts = [], {}
    for organ, (suffix, site, width, prov, layers) in ORGANS.items():
        for L in layers:
            name = f"language_model.model.layers.{L}.{suffix}"
            w = load_tensor(name).astype(np.float32)
            x = acts(site, L, width, a.rows)
            assert x.shape[1] == w.shape[1], (organ, L, x.shape, w.shape)
            y_pts = []
            for label, fam, bits, (kind, param, g) in sch:
                w_hat = quantize_group(w, param, g)[0] if kind == "flat" else binary_planes(w, param, g)[0]
                e = out_rel(x, w, w_hat)
                del w_hat
                y_pts.append({"scheme": label, "family": fam, "bits_per_elem": bits,
                              "output_rel_fro": e})
                fam_pts.setdefault((organ, L, fam), []).append((bits, e))
            dead = next(p["output_rel_fro"] for p in y_pts if p["scheme"] == "flat q2")
            per_site.append({"organ": organ, "layer": L, "site": site,
                             "site_provenance": prov, "shape": list(w.shape),
                             "rows": int(x.shape[0]), "dead_anchor_error": dead,
                             "points": y_pts})
            print(f"{organ:<22} L{L:<3} {site:<16} {prov:<9} rows {x.shape[0]}")
            del w, x

    fams = {}
    for (organ, L, fam), pts in fam_pts.items():
        dead = next(s["dead_anchor_error"] for s in per_site
                    if s["organ"] == organ and s["layer"] == L)
        fams.setdefault(fam, []).append({
            "organ": organ, "layer": L,
            "intrinsic_cliff": cliff_intrinsic(pts),
            "anchored_cliff_bits": cliff_anchored(pts, dead),
            "curve": [{"bits": b, "err": e} for b, e in sorted(pts)]})

    print(f"\n{'family':<8}{'organ':<22}{'layer':>6}{'intrinsic PTP':>16}{'anchored PTP':>15}")
    summary = {}
    for fam, rows in fams.items():
        for r in rows:
            ic = r["intrinsic_cliff"]
            iv = ic["status"].split(" --")[0]
            av = f"{r['anchored_cliff_bits']:.4f}" if r["anchored_cliff_bits"] is not None else "no crossing"
            print(f"{fam:<8}{r['organ']:<22}{r['layer']:>6}{iv:>16}{av:>15}")
        iv = [r["intrinsic_cliff"]["bits"] for r in rows
              if r["intrinsic_cliff"]["status"] == "LOCATED"]
        av = [r["anchored_cliff_bits"] for r in rows if r["anchored_cliff_bits"] is not None]
        summary[fam] = {
            "intrinsic_ptp_bits_mean": float(np.mean(iv)) if iv else None,
            "intrinsic_ptp_bits_max": float(np.max(iv)) if iv else None,
            "anchored_ptp_bits_mean": float(np.mean(av)) if av else None,
            "anchored_ptp_bits_max": float(np.max(av)) if av else None,
            "sites_with_no_anchored_crossing": len(rows) - len(av),
        }
        summary[fam]["sites_with_a_located_intrinsic_cliff"] = len(iv)
        summary[fam]["intrinsic_statuses"] = sorted({r["intrinsic_cliff"]["status"] for r in rows})
        s = summary[fam]
        print(f"  -> {fam}: intrinsic cliff LOCATED at {len(iv)}/{len(rows)} sites; "
              f"anchored PTP mean {s['anchored_ptp_bits_mean']}, "
              f"WORST SITE {s['anchored_ptp_bits_max']} bits/elem")

    doc = {
        "schema": "hawking.nos.phase_transition_map.v1",
        "obligation": "G067 -- physical cost vs capability per family, cliff located as a number",
        "method": {
            "error": "output-space relative Frobenius on real captured BF16 parent activations",
            "why_not_weight_cosine": "Q30 was calibrated on a broken model and the sub-bit "
                                     "negatives were synthetic-activation artefacts; G036 requires "
                                     "fitting against the teacher function",
            "intrinsic_rule": "bits/elem of maximum curvature in log(error) vs bits, anchored to "
                              "nothing",
            "anchored_rule": "bits/elem where the family's error crosses flat q2's error at the "
                             "SAME site, interpolated; q2 is the recorded DEAD artifact so this "
                             "transfers a real capability verdict across families at matched cost",
            "circularity_declared": "for the FLAT family the dead anchor IS a family member, so its "
                                    "anchored number is the anchor by construction and only the "
                                    "intrinsic number is informative there",
        },
        "limitation": ("the anchors' capability verdicts come from a TEN-ITEM gate, which G046 and "
                       "G048 already measured as too narrow to certify equivalence. These PTPs are "
                       "as strong as those verdicts and no stronger."),
        "families": fams, "summary": summary, "per_site": per_site,
        "group": a.group, "rows": a.rows,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
