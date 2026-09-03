#!/usr/bin/env python3
"""G126: run the wide battery and report PER CATEGORY.

Aggregate scores are what this obligation exists to replace. G076 measured that a
compressed artifact can hold FACTUAL at 1.00 while doubling its truncation rate, and
G100 measured the density leader 1.75x worse on wall-time-per-verified-task at equal
accuracy. A single number hides both.

Scoring is deterministic and post-</think> only. A response that never closes its
think block is TRUNCATED, not wrong -- G076 found a 48-token budget scoring every
artifact 0.00 on a dimension its reference model actually passes.
"""
from __future__ import annotations
import argparse, json, pathlib, re, subprocess, sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "workspace/campaign/records/runs/qwen38-27b"
GREEDY = ROOT / "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy"
LANE = ROOT / "tools/gpu_lane_lock.sh"
sys.path.insert(0, str(ROOT / "docs/spec/battery"))
from wide_v1 import items  # noqa: E402


def judge(ans, text):
    if "</think>" not in text:
        return False, True
    body = text.split("</think>")[-1]
    if ans.isdigit():
        return re.search(rf"(?<!\d){re.escape(ans)}(?!\d)", body) is not None, False
    return ans.lower() in body.lower(), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="uniform-q4-v1")
    ap.add_argument("--max-new-tokens", type=int, default=288)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    B = items()
    pf = pathlib.Path("/tmp/_wide_prompts.txt")
    pf.write_text("\n".join(x["prompt"].replace("\n", " ") for x in B) + "\n")
    out = ROOT / f"receipts/ascent-2026-08-16/_wide_{a.artifact}.json"
    r = subprocess.run([str(LANE), f"wide-{a.artifact}", str(GREEDY),
                        "--artifact-root", str(RUNS / a.artifact),
                        "--tokenizer", str(RUNS / "bf16/tokenizer.json"),
                        "--prompts-file", str(pf), "--max-new-tokens", str(a.max_new_tokens),
                        "--max-seq-len", "2048", "--out", str(out)],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"{a.artifact} failed\n{r.stderr[-2000:]}")
    d = json.loads(out.read_text())
    rows = d.get("prompts") or d.get("results") or []
    if not rows:
        for v in d.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "generated_text" in v[0]:
                rows = v; break
    out.unlink(missing_ok=True)

    per = defaultdict(lambda: {"n": 0, "ok": 0, "trunc": 0, "lure": 0})
    detail = []
    for it, rec in zip(B, rows):
        ok, tr = judge(it["answer"], rec["generated_text"])
        c = per[it["category"]]
        c["n"] += 1; c["ok"] += ok; c["trunc"] += tr
        lure_hit = False
        if it["lure"] and not tr:
            body = rec["generated_text"].split("</think>")[-1]
            lure_hit = it["lure"].lower() in body.lower()
            c["lure"] += lure_hit
        detail.append({"category": it["category"], "answer": it["answer"], "correct": bool(ok),
                       "truncated": bool(tr), "lure_emitted": bool(lure_hit)})

    print(f"{a.artifact}: {len(B)} items, {a.max_new_tokens} token budget")
    print(f"{'category':<15}{'n':>4}{'correct':>9}{'score':>8}{'trunc':>7}{'lure':>6}")
    tot_ok = tot_n = tot_tr = 0
    for cat in sorted(per):
        c = per[cat]; scored = c["n"] - c["trunc"]
        s = c["ok"] / scored if scored else float("nan")
        tot_ok += c["ok"]; tot_n += c["n"]; tot_tr += c["trunc"]
        print(f"{cat:<15}{c['n']:>4}{c['ok']:>9}{s:>8.2f}{c['trunc']:>7}{c['lure']:>6}")
    print(f"{'TOTAL':<15}{tot_n:>4}{tot_ok:>9}{tot_ok/max(1,tot_n-tot_tr):>8.2f}{tot_tr:>7}")

    doc = {"schema": "hawking.nos.wide_battery.v1",
           "obligation": "G126 -- >= 60 deterministic items, per-category scoring",
           "artifact": a.artifact, "items": len(B), "categories": sorted(per),
           "max_new_tokens": a.max_new_tokens,
           "per_category": {k: dict(v) for k, v in per.items()},
           "total": {"n": tot_n, "correct": tot_ok, "truncated": tot_tr,
                     "score_on_scored": tot_ok / max(1, tot_n - tot_tr)},
           "detail": detail,
           "scoring": ("deterministic, post-</think> only. A response that never closes its think "
                       "block is TRUNCATED rather than wrong -- G076 found a 48-token budget "
                       "scoring every artifact 0.00 on a dimension its reference passes."),
           "two_categories_need_their_design_stated": (
               "ADVERSARIAL items carry a PLAUSIBLE WRONG answer (Sydney, Einstein, Saturn) and the "
               "run records whether the LURE was emitted, so a model falling into a specific trap "
               "is distinguishable from one merely getting items wrong. CALIBRATION items have NO "
               "answer and reward declining -- the only category where refusing is correct, "
               "included because every other category rewards producing something."),
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, cwd=ROOT).stdout.strip()}
    if a.out:
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
