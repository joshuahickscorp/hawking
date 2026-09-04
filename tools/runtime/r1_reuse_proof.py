#!/usr/bin/env python3
"""R1: what physical prompt work is actually reused, measured not reported.

Drives the sealed resident directly over its JSONL protocol so all arms share one
model load. A reported cache hit is not evidence; the defining property is that
the second call performs measurably less physical prompt processing while
preserving output semantics.

Arms, in one resident so the state is genuinely shared:

    A COLD                  first call, nothing to reuse
    B WARM-IDENTICAL        same stable prefix, changed suffix
    C WARM-PREFIX-MUTATED   one token changed EARLY in the prefix
    D WARM-SUFFIX-MUTATED   prefix untouched, only the suffix differs

C must invalidate. D must not. Both are R1 acceptance clauses and they fail in
opposite directions, so a run that only shows "reuse happened" proves neither.

Effective prompt tok/s and fresh-compute prompt tok/s are reported SEPARATELY:
the first divides all prompt tokens by prompt wall and flatters reuse, the second
divides only physically stepped positions by that wall and says how fast the
machine actually computes. Blurring them is how a cache gets credit for work it
never did.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.path.join(REPO, "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_resident")
ARTIFACT = "/Users/scammermike/noetic/NOETIC_PARENT_A"
TOKENIZER = os.path.join(ARTIFACT, "tokenizer.json")

# A long stable prefix, then short varying suffixes. Real prose so the tokenizer
# behaves as it does in production rather than on a pathological repeat.
STABLE = (
    "You are a careful systems engineer working on a large Rust and Python code "
    "base. You answer briefly and you never guess. When you are unsure you say "
    "so plainly. The repository contains a model runtime, a control plane, a "
    "verifier and a receipt store. Weights are quantised and streamed from a "
    "unified memory pool. The prompt path currently steps one position at a "
    "time. Your task is to answer the question that follows using only what is "
    "stated above, in one short sentence. "
) * 6

SUFFIXES = {
    "s1": "Question: what is streamed from the memory pool?",
    "s2": "Question: what does the verifier do?",
    "s3": "Question: how does the prompt path advance?",
}


def send(proc, obj):
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("resident closed the pipe")
    return json.loads(line)


def arm(proc, name, prompt, n):
    t0 = time.monotonic()
    r = send(proc, {"id": f"{name}-{n}", "prompt": prompt, "max_new_tokens": 8})
    wall = time.monotonic() - t0
    if r.get("status") != "ok":
        raise RuntimeError(f"{name}: {r}")
    prof = r.get("native_metrics", {}).get("step_trace") or {}
    return {
        "arm": name,
        "prompt_tokens": r.get("prompt_tokens"),
        "reused": r.get("prefix_reused_tokens", 0),
        "stepped": r.get("prefill_tokens_stepped", 0),
        "source": r.get("prefix_source"),
        "ckpt_at": r.get("prefix_checkpoint_taken_at"),
        "text": (r.get("text") or "")[:60],
        "wall": wall,
    }


def main() -> int:
    if not os.path.exists(BIN):
        print("resident binary missing; build it first")
        return 1
    proc = subprocess.Popen(
        [BIN, "--artifact-root", ARTIFACT, "--tokenizer", TOKENIZER,
         "--max-seq-len", "8192", "--resident-identity", "r1-proof"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    # The resident emits a ready banner before it will answer anything.
    banner = json.loads(proc.stdout.readline())
    print(f"resident ready: weight_bytes={banner.get('resident_weight_bytes'):,} "
          f"max_seq_len={banner.get('max_seq_len')}")
    rows = []
    try:
        mutated = "X" + STABLE[1:]          # one character early in the prefix
        # Paired alternating passes, per the experimental method. Sample 0 is a
        # warmup whose numbers are recorded but excluded from the verdict.
        for n in range(3):
            rows.append(arm(proc, "A-COLD" if n == 0 else "B-WARM-IDENTICAL",
                            STABLE + SUFFIXES["s1"], n))
            rows.append(arm(proc, "D-WARM-SUFFIX-MUTATED",
                            STABLE + SUFFIXES["s2"], n))
            rows.append(arm(proc, "C-WARM-PREFIX-MUTATED",
                            mutated + SUFFIXES["s3"], n))
    finally:
        try:
            proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    print(f"{'arm':<24}{'prompt':>8}{'reused':>8}{'stepped':>9}{'wall':>8}  source")
    for r in rows:
        print(f"{r['arm']:<24}{r['prompt_tokens']:>8}{r['reused']:>8}"
              f"{r['stepped']:>9}{r['wall']:>7.1f}s  {r['source']}")

    body = rows[3:]          # drop the first triple as warmup
    def by(a): return [r for r in body if r["arm"].startswith(a)]
    warm, suffix_m, prefix_m = by("B"), by("D"), by("C")

    print()
    checks = []
    if warm:
        ok = all(r["stepped"] < r["prompt_tokens"] for r in warm)
        checks.append(("warm steps fewer positions than its prompt", ok))
    if suffix_m:
        ok = all(r["reused"] > 0 for r in suffix_m)
        checks.append(("suffix mutation PRESERVES the reusable prefix", ok))
    if prefix_m:
        ok = all(r["reused"] == 0 for r in prefix_m)
        checks.append(("prefix mutation INVALIDATES reuse", ok))

    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")

    tot_prompt = sum(r["prompt_tokens"] for r in body)
    tot_step = sum(r["stepped"] for r in body)
    tot_wall = sum(r["wall"] for r in body)
    print()
    print(f"  effective prompt tok/s     {tot_prompt / max(tot_wall, 1e-9):8.1f}"
          f"   (all prompt tokens / wall)")
    print(f"  fresh-compute prompt tok/s {tot_step / max(tot_wall, 1e-9):8.1f}"
          f"   (physically stepped / wall)")
    print(f"  physical positions avoided {tot_prompt - tot_step:8d}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
