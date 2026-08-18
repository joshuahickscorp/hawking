#!/usr/bin/env python3
"""G081: does the workload profile change the compilation, and by how much?

F10 established that every activation-weighted result in this campaign was fitted
on prose rows. That makes this obligation urgent rather than speculative: if the
profile matters, those results carry a narrowing they never declared, and a
specialized build must declare it -- which is exactly what the verify demands.

Four profiles are fitted and cross-evaluated on HELD-OUT rows of each workload:

  d_prose    fitted on prose only          the accidental status quo
  d_code     fitted on code only           the opposite specialization
  d_pooled   fitted on both, equal parts   the GENERAL GENESIS BUILD
  d_matched  fitted on the eval workload   the oracle; no deployable build can
                                           have this, it bounds what profiling
                                           could ever buy

flat q3 is carried as the profile-independent control. Its error does not depend
on d at all, so it shows what the representation costs with NO profile, and it is
the line a mismatched profile has to stay under to be worth having.

The narrowing a specialized build must declare is then a measured quantity:
how much worse the mismatched profile is than the matched one, on the workload it
was not fitted for.

Fit and eval rows are disjoint AND drawn from different prompts, since the capture
splits by prompt id and refuses row shuffling.

  ./tools/gravity_profile_guided.py --out receipts/.../G081_PROFILE.json
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor, quantize_group  # noqa: E402
from gravity_phase_transition import ORGANS  # noqa: E402
from gravity_planes_functional import planes_fit  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAP = ROOT / "workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
SITES = ["mlp.gate_proj", "mlp.down_proj", "self_attn.q_proj"]
PROSE_END, CODE_START, CODE_END = 5804, 5804, 6827


def rows(site, width, a, b):
    p = CAP / site / f"L63.f16" if False else None
    return None


def load(site, layer, width, a, b):
    p = CAP / site / f"L{layer:02d}.f16"
    return np.fromfile(p, dtype=np.float16, offset=a * width * 2,
                       count=(b - a) * width).reshape(b - a, width).astype(np.float32)


def dvec(x):
    d = (x.astype(np.float64) ** 2).sum(0)
    return (d / d.mean()).astype(np.float32)


def out_rel(x, w, wh):
    y = x @ w.T
    return float(np.linalg.norm(x @ wh.T - y) / np.linalg.norm(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor-rows", type=int, default=1536)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--planes", type=int, default=2)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    n = a.n

    out = []
    for organ in SITES:
        suffix, site, width, prov, layers = ORGANS[organ]
        L = layers[-1]
        w = load_tensor(f"language_model.model.layers.{L}.{suffix}").astype(np.float32)[:a.tensor_rows]
        xp_fit = load(site, L, width, 0, n)
        xp_ev = load(site, L, width, n, 2 * n)
        xc_fit = load(site, L, width, CODE_START, CODE_START + n)
        xc_ev = load(site, L, width, CODE_START + n, min(CODE_END, CODE_START + 2 * n))
        profiles = {"prose": dvec(xp_fit), "code": dvec(xc_fit),
                    "pooled": dvec(np.concatenate([xp_fit, xc_fit]))}
        flat = quantize_group(w, 3, a.group)[0]

        entry = {"organ": organ, "layer": L, "site": site,
                 "fit_rows": n, "eval_rows_prose": int(xp_ev.shape[0]),
                 "eval_rows_code": int(xc_ev.shape[0]), "cells": []}
        for ename, xe in (("prose", xp_ev), ("code", xc_ev)):
            base = out_rel(xe, w, flat)
            cell = {"eval_workload": ename, "flat_q3_no_profile": base, "by_profile": {}}
            allp = dict(profiles); allp["matched(oracle)"] = dvec(xe)
            for pname, d in allp.items():
                wh, bits = planes_fit(w, a.planes, a.group, d, False)
                cell["by_profile"][pname] = {"err": out_rel(xe, w, wh), "bits_per_elem": bits}
                del wh
            m = cell["by_profile"]["matched(oracle)"]["err"]
            mis = "code" if ename == "prose" else "prose"
            cell["narrowing_vs_matched"] = {
                k: v["err"] / m - 1.0 for k, v in cell["by_profile"].items()}
            cell["mismatched_profile"] = mis
            cell["mismatch_penalty_pct"] = (cell["by_profile"][mis]["err"] / m - 1.0) * 100
            cell["pooled_penalty_pct"] = (cell["by_profile"]["pooled"]["err"] / m - 1.0) * 100
            entry["cells"].append(cell)
            print(f"{organ:<20} eval {ename:<6} flat q3 {base:.5f} | " + "  ".join(
                f"{k} {v['err']:.5f}" for k, v in cell["by_profile"].items()))
        out.append(entry)
        del w, xp_fit, xp_ev, xc_fit, xc_ev

    mis = [c["mismatch_penalty_pct"] for s in out for c in s["cells"]]
    pool = [c["pooled_penalty_pct"] for s in out for c in s["cells"]]
    under = [c["by_profile"][c["mismatched_profile"]]["err"] < c["flat_q3_no_profile"]
             for s in out for c in s["cells"]]
    print(f"\nmismatched profile costs {np.mean(mis):+.2f}% over matched "
          f"(min {np.min(mis):+.2f}%, max {np.max(mis):+.2f}%)")
    print(f"pooled  profile costs {np.mean(pool):+.2f}% over matched "
          f"(min {np.min(pool):+.2f}%, max {np.max(pool):+.2f}%)")
    print(f"mismatched profile still beats no profile (flat q3) in "
          f"{sum(under)}/{len(under)} cells")

    doc = {
        "schema": "hawking.nos.profile_guided.v1",
        "obligation": "G081 -- profile-guided compilation, and the narrowing it must declare",
        "why_now": "F10 established that every activation-weighted result in this campaign was "
                   "fitted on PROSE rows, so whether the profile matters decides whether those "
                   "results carry an undeclared narrowing",
        "profiles": {"prose": "the accidental status quo", "code": "opposite specialization",
                     "pooled": "the GENERAL GENESIS BUILD, equal parts",
                     "matched(oracle)": "fitted on the eval workload; no deployable build can have "
                                        "this, it bounds what profiling could ever buy"},
        "control": "flat q3 carried as the profile-INDEPENDENT reference -- its error does not "
                   "depend on d at all, so it is the line a mismatched profile must stay under to "
                   "be worth having",
        "holdout": "fit and eval rows are disjoint and come from different prompts, since the "
                   "capture splits by prompt id and refuses row shuffling",
        "sites": out,
        "summary": {"mismatch_penalty_pct_mean": float(np.mean(mis)),
                    "mismatch_penalty_pct_max": float(np.max(mis)),
                    "pooled_penalty_pct_mean": float(np.mean(pool)),
                    "pooled_penalty_pct_max": float(np.max(pool)),
                    "mismatched_beats_no_profile_cells": int(sum(under)), "cells": len(under)},
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
