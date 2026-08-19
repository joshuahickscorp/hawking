#!/usr/bin/env python3
"""G130: greedy divergence, with the G0-vs-G0 null beside every candidate.

Greedy decoding is deterministic, so G0 against ITSELF must read exactly zero. That
control is not a formality: it is the only thing separating "this candidate diverges"
from "this harness is nondeterministic", and without it every number below is
unreadable.

Divergence is a DIAGNOSTIC and not a verdict, and the ledger already says why: G004
measured greedy divergence ranking candidates OPPOSITELY to capability. A candidate
that diverges early may be better. This instrument describes HOW a candidate departs,
never whether it is worse.

  ./tools/greedy_divergence.py --candidate g032-chanscale-a025-compact
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "workspace/campaign/records/runs/qwen38-27b"
GREEDY = ROOT / "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy"
LANE = ROOT / "tools/gpu_lane_lock.sh"
# Qwen control tokens seen in the prompt ids: im_start / im_end.
CONTROL = {248045, 248046}


def run(artifact, tag, prompts, tokens):
    pf = pathlib.Path(f"/tmp/_div_{tag}.txt"); pf.write_text("\n".join(prompts) + "\n")
    out = pathlib.Path(f"/tmp/_div_{tag}.json")
    r = subprocess.run([str(LANE), f"div-{tag}", str(GREEDY),
                        "--artifact-root", str(RUNS / artifact),
                        "--tokenizer", str(RUNS / "bf16/tokenizer.json"),
                        "--prompts-file", str(pf), "--max-new-tokens", str(tokens),
                        "--max-seq-len", "512", "--out", str(out)],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"{tag} failed\n{r.stderr[-1200:]}")
    d = json.loads(out.read_text())
    rows = d.get("prompts") or d.get("results") or []
    if not rows:
        for v in d.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "new_token_ids" in v[0]:
                rows = v; break
    out.unlink(missing_ok=True)
    return [tuple(x.get("new_token_ids") or []) for x in rows]


def compare(a, b):
    """Per-prompt divergence description. a is the reference stream."""
    per = []
    for i, (x, y) in enumerate(zip(a, b)):
        n = min(len(x), len(y))
        diffs = [k for k in range(n) if x[k] != y[k]]
        first = diffs[0] if diffs else None
        # recovery: after the first divergence, does the stream re-agree and stay agreed?
        rec = None
        if first is not None:
            for k in range(first + 1, n):
                if x[k:] == y[k:]:
                    rec = k - first
                    break
        ctrl = sum(1 for k in range(n) if (x[k] in CONTROL) != (y[k] in CONTROL))
        per.append({"prompt": i, "len_ref": len(x), "len_cand": len(y),
                    "first_divergence": first, "divergent_positions": len(diffs),
                    "divergence_rate": len(diffs) / n if n else 0.0,
                    "recovered_after": rec, "control_token_mismatches": ctrl})
    n_div = sum(1 for p in per if p["first_divergence"] is not None)
    firsts = [p["first_divergence"] for p in per if p["first_divergence"] is not None]
    return {"prompts": len(per), "prompts_diverging": n_div,
            "mean_divergence_rate": sum(p["divergence_rate"] for p in per) / max(1, len(per)),
            "earliest_divergence": min(firsts) if firsts else None,
            "median_first_divergence": sorted(firsts)[len(firsts) // 2] if firsts else None,
            "recovered": sum(1 for p in per if p["recovered_after"] is not None),
            "control_token_mismatches": sum(p["control_token_mismatches"] for p in per),
            "per_prompt": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="uniform-q4-v1")
    ap.add_argument("--candidate", default="g032-chanscale-a025-compact")
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    prompts = [f"What is {i*7+11} plus {i*3+5}? Reply with the number only." for i in range(8)] + \
              [f"Write the word \"{w}\" backwards. Reply with the word only."
               for w in ("planet", "silver", "hunter", "candle")] + \
              [f"What is the capital city of {c}? Reply with the city name only."
               for c in ("France", "Japan", "Peru", "Kenya")]
    ref = run(a.reference, "ref", prompts, a.tokens)
    ref2 = run(a.reference, "ref2", prompts, a.tokens)
    cand = run(a.candidate, "cand", prompts, a.tokens)
    null = compare(ref, ref2)
    real = compare(ref, cand)
    print(f"{'comparison':<44}{'diverge':>9}{'rate':>8}{'first':>8}{'recov':>7}{'ctrl':>6}")
    for label, c in ((f"NULL  {a.reference} vs itself", null),
                     (f"CAND  {a.reference} vs {a.candidate}", real)):
        print(f"{label:<44}{c['prompts_diverging']:>4}/{c['prompts']:<4}"
              f"{c['mean_divergence_rate']:>8.3f}"
              f"{str(c['median_first_divergence']):>8}{c['recovered']:>7}"
              f"{c['control_token_mismatches']:>6}")
    ok = null["prompts_diverging"] == 0
    print(f"\nNULL CONTROL: {'READS ZERO, as greedy determinism requires' if ok else 'NONZERO -- the harness is nondeterministic and no number above is readable'}")
    doc = {"schema": "hawking.nos.greedy_divergence.v1",
           "obligation": "G130 -- greedy divergence with controls, as a seal DIAGNOSTIC",
           "reference": a.reference, "candidate": a.candidate, "tokens": a.tokens,
           "prompts": len(prompts),
           "null_control": null, "candidate_vs_reference": real,
           "null_reads_zero": ok,
           "it_is_a_diagnostic_not_a_verdict": (
               "G004 measured greedy divergence ranking candidates OPPOSITELY to capability, so a "
               "candidate that diverges early may be better. This describes HOW a candidate "
               "departs, never whether it is worse."),
           "not_delivered": ("top-k overlap and logit margins. Both need per-step LOGITS and the "
                             "harness exposes token ids only. First divergence, divergence rate, "
                             "recovery and control-token changes come from the token streams and "
                             "are delivered."),
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, cwd=ROOT).stdout.strip()}
    if a.out:
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
