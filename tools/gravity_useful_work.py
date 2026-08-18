#!/usr/bin/env python3
"""G100: tokens per task and wall time per VERIFIED task, alongside TPS.

The verify states the failure this exists to catch: a representation delivering 2x
TPS while requiring 2x more reasoning tokens is NOT better. TPS is a physical
metric and cannot see that, because it counts tokens rather than tasks.

So every artifact here reports four numbers on the same prompts:

  TPS                        tokens per wall second -- the physical metric
  TOKENS_REQUIRED/TASK       tokens actually emitted to answer, measured from the
                             returned token ids rather than estimated from text
  accuracy                   fraction of tasks answered correctly
  WALL_TIME/VERIFIED_TASK    the utility metric: seconds of wall clock per task
                             that came out RIGHT. Wrong answers cost their tokens
                             and deliver nothing, so they are charged, not dropped.

That last definition is the whole point. A model that is fast, verbose and wrong
scores well on TPS and badly here, which is the ordering the obligation asks for.

  ./tools/gravity_useful_work.py --artifact uniform-q4-v1 --artifact g032-chanscale-a025-compact
"""
from __future__ import annotations
import argparse, json, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "workspace/campaign/records/runs/qwen38-27b"
GREEDY = ROOT / "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy"
TOKENIZER = RUNS / "bf16/tokenizer.json"
LANE = ROOT / "tools/gpu_lane_lock.sh"
# measured in G041_COST_VECTOR this session, complete wall clock
TPS = {"uniform-q4-v1": 32.73, "g032-chanscale-a025-compact": 29.77,
       "compact-q3attn-r1p2-v1": 29.77}
sys.path.insert(0, str(ROOT / "tools"))
from gravity_doctor_dimensions import build, judge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", action="append", required=True)
    ap.add_argument("--n-per-dim", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    dims = build(a.seed, a.n_per_dim)
    keys = [k for k in dims if k != "IDENTITY"]
    items = [(p, ans) for k in keys for p, ans in dims[k]]
    pf = pathlib.Path("/tmp/_useful_prompts.txt")
    pf.write_text("\n".join(p for p, _ in items) + "\n")
    print(f"{len(items)} tasks over {len(keys)} dimensions")

    rows = []
    for art in a.artifact:
        out = ROOT / f"receipts/ascent-2026-08-16/_uw_{art}.json"
        r = subprocess.run([str(LANE), f"uw-{art}", str(GREEDY),
                            "--artifact-root", str(RUNS / art), "--tokenizer", str(TOKENIZER),
                            "--prompts-file", str(pf), "--max-new-tokens", str(a.max_new_tokens),
                            "--max-seq-len", "768",
                            "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
        if r.returncode != 0:
            raise SystemExit(f"{art} failed\n{r.stderr[-2000:]}")
        d = json.loads(out.read_text())
        # The harness refuses --prompts-file together with --complete-wall, so TPS comes from the
        # dedicated timing run in G041_COST_VECTOR and the token counts come from here. Two runs,
        # same machine, same session -- said plainly because mixing sources silently is how a
        # composite metric goes wrong.
        tps = TPS[art]
        recs = d.get("prompts") or d.get("results") or []
        if not recs:
            for v in d.values():
                if isinstance(v, list) and v and isinstance(v[0], dict) and "generated_text" in v[0]:
                    recs = v; break
        out.unlink(missing_ok=True)
        toks, ok, trunc = [], [], 0
        for (p, ans), rec in zip(items, recs):
            n = len(rec.get("new_token_ids") or [])
            toks.append(n)
            c, tr = judge(ans, rec["generated_text"])
            ok.append(c); trunc += tr
        n_ok = sum(ok); total_tok = sum(toks)
        row = {"artifact": art, "tps": tps, "tasks": len(items),
               "tokens_total": total_tok, "tokens_per_task": total_tok / len(items),
               "correct": n_ok, "accuracy": n_ok / len(items), "truncated": trunc,
               "wall_s_total": total_tok / tps,
               "wall_s_per_task": total_tok / tps / len(items),
               "wall_s_per_VERIFIED_task": (total_tok / tps / n_ok) if n_ok else None,
               "tokens_per_VERIFIED_task": (total_tok / n_ok) if n_ok else None}
        rows.append(row)
        print(f"  {art:<32} TPS {tps:6.2f}  tok/task {row['tokens_per_task']:6.1f}  "
              f"acc {row['accuracy']:.3f}  trunc {trunc}  "
              f"wall/verified {row['wall_s_per_VERIFIED_task']:.3f}s")

    if len(rows) >= 2:
        a0, b0 = rows[0], rows[1]
        print(f"\n{b0['artifact']} vs {a0['artifact']}:")
        print(f"  TPS                     {b0['tps']/a0['tps']:.3f}x")
        print(f"  tokens per task         {b0['tokens_per_task']/a0['tokens_per_task']:.3f}x")
        print(f"  wall per VERIFIED task  {b0['wall_s_per_VERIFIED_task']/a0['wall_s_per_VERIFIED_task']:.3f}x"
              f"   (lower is better)")

    doc = {"schema": "hawking.nos.useful_work.v1",
           "obligation": "G100 -- useful work, not only TPS",
           "metrics": {
               "TPS": "tokens per wall second, the physical metric",
               "TOKENS_REQUIRED_PER_TASK": "tokens actually emitted, counted from returned token "
                                           "ids rather than estimated from text",
               "WALL_TIME_PER_VERIFIED_TASK": "seconds of wall clock per task answered CORRECTLY. "
                                              "Wrong answers cost their tokens and deliver nothing, "
                                              "so they are charged rather than dropped -- that is "
                                              "what makes this an ordering TPS cannot produce"},
           "two_sources_declared": ("TPS is from the dedicated timing run recorded in "
               "G041_COST_VECTOR; tokens and correctness are from the battery run here. The harness "
               "refuses --prompts-file with --complete-wall, so one run cannot produce both. Same "
               "machine, same session, and said plainly because silently mixing sources is how a "
               "composite metric goes wrong."),
           "n_per_dimension": a.n_per_dim, "seed": a.seed, "results": rows,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, cwd=ROOT).stdout.strip()}
    if a.out:
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
