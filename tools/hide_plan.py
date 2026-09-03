#!/usr/bin/env python3
"""G121: is kernel selection DATA? Swap the plan file and see what changes.

The acceptance is that changing a plan file changes what executes with no
recompilation. The verify is a swap showing a different dispatch census and the same
tokens. This runs the swap and reports what actually moved -- including the half
that does not move, because a plan file whose residency section is decorative should
say so rather than look complete.
"""
from __future__ import annotations
import argparse, json, os, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "workspace/campaign/records/runs/qwen38-27b"
GREEDY = ROOT / "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy"
LANE = ROOT / "tools/gpu_lane_lock.sh"


def run(plan_path, tokens):
    plan = json.loads(pathlib.Path(plan_path).read_text())
    env = dict(os.environ)
    env.update(plan["kernel_selection"])
    out = pathlib.Path(f"/tmp/hide_{plan['plan_name']}.json")
    cmd = [str(LANE), f"hide-{plan['plan_name']}", str(GREEDY),
           "--artifact-root", str(RUNS / "uniform-q4-v1"),
           "--tokenizer", str(RUNS / "bf16/tokenizer.json"), "--prompt", "Say hi.",
           "--max-new-tokens", str(tokens), "--max-seq-len", "128",
           "--complete-wall", "--pairs", "2", "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)
    if r.returncode != 0:
        raise SystemExit(f"{plan['plan_name']}: failed\n{r.stderr[-1500:]}")
    d = json.loads(out.read_text())
    cg = d["cold_generate"]
    return {"plan": plan["plan_name"], "applied": plan["kernel_selection"],
            "dispatches": cg["steady_decode"]["dispatches"],
            "tokens": tuple(cg.get("new_token_ids") or []),
            "token_ms": d["authority"]["headline_complete_wall_ns_per_token"] / 1e6,
            "tps": d["authority"]["headline_complete_tps"],
            "residency_is_data": plan["residency_plan"]["is_this_actually_data"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", nargs="+", default=["docs/spec/plans/default.plan.json",
                                                   "docs/spec/plans/serial-family.plan.json"])
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    rows = [run(p, a.tokens) for p in a.plans]
    print(f"{'plan':<16}{'dispatches':>12}{'token ms':>10}{'TPS':>8}  tokens")
    base = rows[0]
    for r in rows:
        print(f"{r['plan']:<16}{r['dispatches']:>12}{r['token_ms']:>10.3f}{r['tps']:>8.2f}  "
              f"{'IDENTICAL' if r['tokens'] == base['tokens'] else 'DIFFER'}")
    census_moved = len({r["dispatches"] for r in rows}) > 1
    tokens_same = len({r["tokens"] for r in rows}) == 1
    print(f"\ndispatch census changed: {census_moved}")
    print(f"tokens identical:        {tokens_same}")
    doc = {"schema": "hawking.nos.hide_plan_swap.v1",
           "obligation": "G121 -- kernel selection and residency as DATA",
           "plans": a.plans, "results": rows,
           "no_recompilation": True,
           "dispatch_census_changed": census_moved, "tokens_identical": tokens_same,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, cwd=ROOT).stdout.strip()}
    if a.out:
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
