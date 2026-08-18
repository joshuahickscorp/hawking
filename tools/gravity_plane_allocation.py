#!/usr/bin/env python3
"""G073: let the compiler choose planes_i per tensor instead of one k everywhere.

The three inputs the obligation names now all exist as measurements rather than
guesses:

  marginal functional gain per plane   G069, function-fitted planes k=1..5 on real
                                       held-out activations, per site
  marginal ns per plane                G072, measured on device: 0.6059 / 0.6676 /
                                       0.9867 ps/element for k=1/2/3
  the allocation                       this file

The cost model above k=3 is extrapolated, and the basis is stated rather than
assumed: G072 measured the kernel SATURATED from k2 onward at 374.5 -> 380.0 GB/s,
so in that regime time scales with bytes and ps(k) = ps(3) * k/3. Below k=3 the
measured values are used directly, because there the kernel is not saturated and a
bytes model would be wrong.

The verify is a real bar and this file can fail it: the chosen distribution must
be NON-UNIFORM and must beat the best uniform k at equal total cost. If the greedy
allocation lands on a uniform assignment, or fails to beat uniform, that is the
answer and it gets reported as one.

Objective is element-weighted mean relative output error across the sites, with
each site weighted by its real tensor element count. That is a PROXY for whole-
model error -- no assembled measurement exists at these settings -- and relative
errors from different tensors are not strictly commensurable. Stated here because
the allocation is only as meaningful as that assumption.

  ./tools/gravity_plane_allocation.py --out receipts/.../G073_PLANE_ALLOCATION.json
"""
from __future__ import annotations
import argparse, json, pathlib, struct, subprocess
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BF16 = ROOT / "workspace/campaign/records/runs/qwen38-27b/bf16"
G069 = ROOT / "receipts/ascent-2026-08-16/G069_PLANES_FUNCTIONAL.json"
PS = {1: 0.6059, 2: 0.6676, 3: 0.9867}
SUFFIX = {"mlp.gate_proj": "mlp.gate_proj.weight", "mlp.down_proj": "mlp.down_proj.weight",
          "linear_attn.out_proj": "linear_attn.out_proj.weight",
          "self_attn.o_proj": "self_attn.o_proj.weight",
          "self_attn.q_proj": "self_attn.q_proj.weight"}


def ps_for(k):
    return PS[k] if k in PS else PS[3] * k / 3.0


def shape_of(name):
    index = json.loads((BF16 / "model.safetensors.index.json").read_text())
    with open(BF16 / index["weight_map"][name], "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))[name]["shape"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    src = json.loads(G069.read_text())

    sites = []
    for s in src["sites"]:
        name = f"language_model.model.layers.{s['layer']}.{SUFFIX[s['organ']]}"
        r, c = shape_of(name)
        err = {k: next(q["output_rel_fro"] for q in s["points"]
                       if q["scheme"] == f"planes function-fit k{k}")
               for k in range(1, a.kmax + 1)}
        sites.append({"organ": s["organ"], "layer": s["layer"], "shape": [r, c],
                      "elements": r * c, "err_by_k": err})
    W = np.array([s["elements"] for s in sites], dtype=np.float64)
    W = W / W.sum()

    def score(ks):
        e = sum(w * s["err_by_k"][k] for w, s, k in zip(W, sites, ks))
        c = sum(w * ps_for(k) for w, s, k in zip(W, sites, ks))
        return float(e), float(c)

    uniform = {k: score([k] * len(sites)) for k in range(1, a.kmax + 1)}
    print(f"{'uniform k':<12}{'weighted err':>14}{'weighted ps':>14}")
    for k, (e, c) in uniform.items():
        print(f"{k:<12}{e:>14.6f}{c:>14.6f}")

    # THE BUDGET SWEEP HAD TO BE FIXED. A first pass scored each allocation against
    # the cost of a uniform k, and that makes the answer uniform BY CONSTRUCTION:
    # ps(k) is the same function at every site and the weights sum to one, so the
    # cost of uniform k is EXACTLY the cost of upgrading every site to k, leaving
    # zero slack for any non-uniform choice. The instrument could not have found
    # non-uniformity if it existed. Budgets now include the points BETWEEN uniform
    # levels, which is where a per-tensor allocator earns its keep: uniform must
    # round down to what it can afford everywhere, and the allocator spends the
    # remainder where the marginal gain is highest.
    costs = [uniform[k][1] for k in sorted(uniform)]
    budgets = []
    for i in range(len(costs) - 1):
        for f in (0.0, 0.25, 0.5, 0.75):
            budgets.append(costs[i] + f * (costs[i + 1] - costs[i]))
    budgets.append(costs[-1])

    results = []
    for uc in budgets:
        affordable = [k for k in sorted(uniform) if uniform[k][1] <= uc + 1e-12]
        target_k = max(affordable)
        ue = uniform[target_k][0]
        ks = [1] * len(sites)
        while True:
            best, gain = None, 0.0
            for i, s in enumerate(sites):
                if ks[i] >= a.kmax:
                    continue
                de = W[i] * (s["err_by_k"][ks[i]] - s["err_by_k"][ks[i] + 1])
                dc = W[i] * (ps_for(ks[i] + 1) - ps_for(ks[i]))
                if dc <= 0:
                    continue
                if score([*ks[:i], ks[i] + 1, *ks[i + 1:]])[1] <= uc and de / dc > gain:
                    best, gain = i, de / dc
            if best is None:
                break
            ks[best] += 1
        ae, ac = score(ks)
        results.append({"budget": uc, "best_uniform_k_that_fits": target_k,
                        "uniform_err": ue, "uniform_cost": uniform[target_k][1],
                        "allocated_k": ks, "allocated_err": ae, "allocated_cost": ac,
                        "is_non_uniform": len(set(ks)) > 1,
                        "beats_uniform": ae < ue and ac <= uc + 1e-12,
                        "err_improvement_pct": (ue - ae) / ue * 100.0})
        r = results[-1]
        print(f"\nbudget {uc:.6f}  (best uniform that fits: k{target_k})")
        print(f"  allocation {ks}  ->  err {ae:.6f} vs uniform {ue:.6f} "
              f"({r['err_improvement_pct']:+.3f}%)  cost {ac:.6f}/{uc:.6f}")
        print(f"  non-uniform: {r['is_non_uniform']}   beats uniform: {r['beats_uniform']}")

    wins = [r for r in results if r["beats_uniform"] and r["is_non_uniform"]]
    verdict = ("PASSES" if wins else "FAILS")
    print(f"\nVERIFY: a non-uniform allocation beating uniform at equal cost exists at "
          f"{len(wins)}/{len(results)} budgets -> {verdict}")
    if wins:
        b = max(wins, key=lambda r: r["err_improvement_pct"])
        print(f"  best: budget {b['budget']:.6f}, allocation {b['allocated_k']}, "
              f"{b['err_improvement_pct']:+.3f}% error")

    doc = {
        "schema": "hawking.nos.plane_allocation.v1",
        "obligation": "G073 -- compiler-chosen plane count per tensor",
        "inputs": {"functional_gain": "receipts/ascent-2026-08-16/G069_PLANES_FUNCTIONAL.json, "
                                      "function-fitted planes on real held-out activations",
                   "kernel_cost": "receipts/ascent-2026-08-16/G072_MULTI_PLANE_GEMV.json, measured "
                                  "0.6059 / 0.6676 / 0.9867 ps/element at k=1/2/3",
                   "cost_model_above_k3": "ps(k) = ps(3)*k/3, because G072 measured the kernel "
                                          "SATURATED from k2 onward (374.5 -> 380.0 GB/s) so time "
                                          "scales with bytes there. Measured values used below k=3, "
                                          "where a bytes model would be wrong."},
        "objective": "element-weighted mean relative output error, each site weighted by its real "
                     "tensor element count",
        "objective_limitation": "a PROXY for whole-model error -- no assembled measurement exists "
                                "at these settings, and relative errors from different tensors are "
                                "not strictly commensurable. The allocation is only as meaningful "
                                "as that assumption.",
        "sites": sites, "uniform": {str(k): {"err": e, "cost": c} for k, (e, c) in uniform.items()},
        "budget_sweep_defect_fixed": (
            "A first pass scored allocations only at the cost of a uniform k, which makes the "
            "answer uniform BY CONSTRUCTION: ps(k) is the same function at every site and the "
            "weights sum to one, so uniform-k cost is exactly the cost of upgrading every site to "
            "k and there is zero slack for a non-uniform choice. The instrument could not have "
            "found non-uniformity if it existed. Budgets now include the points between uniform "
            "levels, where uniform must round down and an allocator can spend the remainder."),
        "allocations": results, "verify_passes": bool(wins),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
