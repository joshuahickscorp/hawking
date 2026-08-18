#!/usr/bin/env python3
"""G079: is capability local enough to link, cherry-pick or version as a subgraph?

This obligation is a PRECONDITION, not a feature: its own verify says that without
locality every downstream architectural claim -- shared object pools, knowledge
externalization, model-git operations on capability objects -- is unsupported. So
it is worth answering before any of them is built.

The test asks whether two different capabilities recruit different weights. For a
tensor W, weight (i,j) matters to a workload in proportion to d_j * w_ij^2 with d
the per-input-channel activation energy of that workload. Take the top-p% by that
score under one workload and under another, and measure the OVERLAP. Near-total
overlap means the same weights serve both and there is no subgraph to link.

THE CONTROL IS WHAT MAKES THE NUMBER MEAN ANYTHING. Two disjoint halves of the SAME
class are scored the same way, giving the overlap attributable to sampling alone.
A cross-class overlap has to be read against that ceiling, never against 100%.

CAPTURE COVERAGE, discovered while building this and stated because it qualifies
more than this obligation: the v2 capture stores 6827 rows per layer in prompt
order and refuses to shuffle, so it stops partway through the prompt list. The
stored rows are 5804 prose and 1023 code. The math, instruction, multilingual,
adversarial and long classes are captured in the manifest but have NO rows. Every
activation-weighted measurement in this campaign therefore rests on prose-and-code
statistics, and the first 5804 rows -- which is what most of them sampled -- are
prose ONLY.

  ./tools/gravity_capability_locality.py --out receipts/.../G079_LOCALITY.json
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor  # noqa: E402
from gravity_phase_transition import ORGANS  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAP = ROOT / "workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
SITES = ["mlp.gate_proj", "mlp.down_proj", "self_attn.q_proj"]


def class_spans():
    d = json.loads((CAP / "capture-result.json").read_text())
    store = d["sites"]["post_input_norm"]["store_n"]
    c, spans = 0, {}
    for p in d["prompts"]:
        if c >= store:
            break
        take = min(p["n_tokens"], store - c)
        spans.setdefault(p["cls"], []).append((c, c + take))
        c += take
    return spans, store, d["prompts"]


def rows_for(site, width, spans, n):
    p = CAP / site / f"{site and ''}"
    out = []
    for a, b in spans:
        out.append((a, b))
    return out


def energy(site, layer, width, spans, cap_rows):
    """Per-input-channel activation energy over the given row spans only."""
    p = CAP / site / f"L{layer:02d}.f16"
    acc = np.zeros(width, dtype=np.float64)
    used = 0
    for a, b in spans:
        b = min(b, a + max(0, cap_rows - used))
        if b <= a:
            break
        x = np.fromfile(p, dtype=np.float16, offset=a * width * 2,
                        count=(b - a) * width).reshape(b - a, width).astype(np.float32)
        acc += (x.astype(np.float64) ** 2).sum(0)
        used += b - a
        del x
    return acc / max(used, 1), used


def overlap(w, d1, d2, p):
    s1 = (w.astype(np.float64) ** 2) * d1[None, :]
    s2 = (w.astype(np.float64) ** 2) * d2[None, :]
    k = int(p * w.size)
    a = np.argpartition(s1.ravel(), -k)[-k:]
    b = np.argpartition(s2.ravel(), -k)[-k:]
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor-rows", type=int, default=2048)
    ap.add_argument("--fracs", default="0.001,0.01,0.05")
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    fracs = [float(f) for f in a.fracs.split(",")]

    spans, store, prompts = class_spans()
    cov = {k: sum(b - x for x, b in v) for k, v in spans.items()}
    missing = sorted({p["cls"] for p in prompts} - set(spans))
    print(f"capture rows by class: {cov}")
    print(f"classes with NO captured rows: {missing}")
    n_code = cov.get("code", 0)
    if n_code == 0:
        raise SystemExit("no code rows; the class contrast cannot be made")

    # Equal-sized samples so overlap is not driven by one side having more data.
    prose = spans["prose"]
    half = n_code
    prose_a = [(prose[0][0], prose[0][0] + half)]
    prose_b = [(prose[0][0] + half, prose[0][0] + 2 * half)]

    out = []
    for organ in SITES:
        suffix, site, width, prov, layers = ORGANS[organ]
        L = layers[-1]
        w = load_tensor(f"language_model.model.layers.{L}.{suffix}").astype(np.float32)[:a.tensor_rows]
        if site == "post_swiglu":
            print(f"  skip {organ}: site {site} is not the shared-input site this test needs")
            del w; continue
        d_code, n1 = energy(site, L, width, spans["code"], half)
        d_pa, n2 = energy(site, L, width, prose_a, half)
        d_pb, n3 = energy(site, L, width, prose_b, half)
        for v in (d_code, d_pa, d_pb):
            v /= v.mean()
        rows = []
        for p in fracs:
            ctrl = overlap(w, d_pa, d_pb, p)
            xcls = overlap(w, d_pa, d_code, p)
            rows.append({"top_frac": p, "control_prose_vs_prose": ctrl,
                         "cross_prose_vs_code": xcls,
                         "cross_over_control": xcls / ctrl if ctrl else None})
            print(f"{organ:<20}L{L:<3} top {p*100:>5.1f}%  control {ctrl:.4f}  "
                  f"cross {xcls:.4f}  ratio {xcls/ctrl:.4f}")
        out.append({"organ": organ, "layer": L, "site": site, "rows_per_sample": half,
                    "channel_energy_spearman_prose_vs_code":
                        float(np.corrcoef(np.argsort(np.argsort(d_pa)),
                                          np.argsort(np.argsort(d_code)))[0, 1]),
                    "channel_energy_spearman_control":
                        float(np.corrcoef(np.argsort(np.argsort(d_pa)),
                                          np.argsort(np.argsort(d_pb)))[0, 1]),
                    "fracs": rows})
        del w

    ratios = [r["cross_over_control"] for s in out for r in s["fracs"]]
    local = float(np.mean(ratios)) < 0.7
    print(f"\ncross/control overlap ratio: mean {np.mean(ratios):.4f}, "
          f"min {np.min(ratios):.4f}, max {np.max(ratios):.4f}")
    print(f"CAPABILITY IS {'LOCAL ENOUGH TO CONSIDER LINKING' if local else 'NOT LOCAL -- the same weights serve both workloads'}")

    doc = {
        "schema": "hawking.nos.capability_locality.v1",
        "obligation": "G079 -- locality is the precondition for every capability-object claim",
        "method": "top-p% of weights by d_j * w_ij^2 under one workload versus another; overlap "
                  "read against a same-class control, never against 100%",
        "control": "two DISJOINT halves of the prose class, equal size to the code sample, giving "
                   "the overlap attributable to sampling alone",
        "capture_coverage_defect": {
            "rows_by_class": cov, "classes_with_no_rows": missing, "store_n": store,
            "why": "the v2 capture stores rows in prompt order and refuses to shuffle, so it stops "
                   "partway through the prompt list",
            "consequence": "every activation-weighted measurement in this campaign rests on "
                           "prose-and-code statistics, and the first 5804 rows -- what most of them "
                           "sampled -- are PROSE ONLY. Relative codec comparisons are unaffected "
                           "since all codecs saw the same activations, but the ACTIVATION "
                           "STATISTICS d_j are prose statistics and any claim that generalises them "
                           "to math, tool use or multilingual work is unsupported.",
        },
        "sites": out,
        "cross_over_control_mean": float(np.mean(ratios)),
        "capability_is_local": bool(local),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
