#!/usr/bin/env python3
"""bench_head — the orchestrator's spec-decode measurement primitive.

Runs a deterministic baseline-vs-spec-decode comparison for one model + one
Eagle5 head, sweeping the verify-window to find the best real decode tps.
Parses the `[stats]` line dismantle's `generate` prints and reports:

  * baseline dec_tps (no speculation)
  * per-verify-window: spec dec_tps, draft accept rate, speedup
  * the winning window + whether the head is a net win (speedup > 1.0)

This is the local ground-truth that the offline (teacher-forced) τ/accept
numbers only estimate. A head that scored 97% offline can still be a net
loss here if the fp16→Q4_K_M distribution shift tanks real acceptance.

Usage
-----
    # Baseline only (no head)
    python3 tools/orchestrator/bench_head.py \
        --weights models/qwen2.5-7b-instruct-q4_k_m.gguf --slug q7b

    # Baseline + spec-decode with a head, sweep verify-windows
    python3 tools/orchestrator/bench_head.py \
        --weights models/qwen2.5-7b-instruct-q4_k_m.gguf --slug q7b \
        --head /path/to/q7b_head.safetensors \
        --verify-windows 2,4,6,8 \
        --out _orchestrator/q7b_validate.json

Greedy (temperature 0) so runs are deterministic and comparable.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

# The locked Qwen production env (optimized decode path). Mirrors the runtime
# profile's locked_env. DeepSeek uses a different set; pass --arch deepseek2
# to switch.
QWEN_LOCKED_ENV = {
    "HAWKING_QWEN_TCB": "1",
    "HAWKING_QWEN_VOCAB_PRUNE": "32000",
    "HAWKING_QWEN_Q4K_LMHEAD": "1",
    "HAWKING_QWEN_FFN_DOWN_Q4K": "1",
    "HAWKING_QWEN_Q4K_PREDEC": "1",
}
DEEPSEEK_LOCKED_ENV = {
    "HAWKING_DSV2_TCB": "1",
}

# Fixed prompt set — varied enough that decode tps + acceptance aren't a
# single-prompt fluke. Greedy decode makes each deterministic.
PROMPTS = [
    "Explain in detail how photosynthesis converts sunlight into chemical energy.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "Describe the steps to implement a binary search tree in Rust, with code.",
]

STATS_RE = re.compile(
    r"dec_tps=(?P<tps>[0-9.]+).*?draft_accepted=(?P<acc>\d+)\s+draft_rejected=(?P<rej>\d+)"
)


def run_generate(binary: str, weights: str, prompt: str, max_new: int,
                 env: dict, head: str | None, verify_window: int | None) -> dict | None:
    """Run one generate, return parsed {tps, accepted, rejected} or None."""
    cmd = [
        binary, "generate",
        "--weights", weights,
        "--prompt", prompt,
        "--max-new-tokens", str(max_new),
        "--temperature", "0",
    ]
    if head is not None:
        cmd += ["--speculate", "eagle5", "--eagle5-head", head]
        if verify_window is not None:
            cmd += ["--verify-window", str(verify_window)]
    full_env = dict(os.environ)
    full_env.update(env)
    proc = subprocess.run(cmd, env=full_env, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    m = None
    for line in out.splitlines():
        if "[stats]" in line:
            m = STATS_RE.search(line)
    if m is None:
        sys.stderr.write(f"[bench_head] no [stats] parsed (exit {proc.returncode}). Tail:\n")
        sys.stderr.write("\n".join(out.splitlines()[-8:]) + "\n")
        return None
    acc = int(m.group("acc"))
    rej = int(m.group("rej"))
    return {
        "dec_tps": float(m.group("tps")),
        "accepted": acc,
        "rejected": rej,
        "accept_rate": (acc / (acc + rej)) if (acc + rej) > 0 else 0.0,
    }


def measure(binary, weights, env, head, verify_window, max_new, prompts) -> dict:
    """Median dec_tps + mean accept rate across the prompt set."""
    tpss, accs, accepted, rejected = [], [], 0, 0
    for p in prompts:
        r = run_generate(binary, weights, p, max_new, env, head, verify_window)
        if r is None:
            continue
        tpss.append(r["dec_tps"])
        accs.append(r["accept_rate"])
        accepted += r["accepted"]
        rejected += r["rejected"]
    if not tpss:
        return {"ok": False}
    return {
        "ok": True,
        "dec_tps_median": statistics.median(tpss),
        "dec_tps_all": tpss,
        "accept_rate_mean": statistics.mean(accs) if accs else 0.0,
        "accepted_total": accepted,
        "rejected_total": rejected,
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="bench_head")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--slug", required=True, help="model label for output (q7b, q3b, ...)")
    ap.add_argument("--head", default=None, help="Eagle5 head safetensors; omit for baseline-only")
    ap.add_argument("--arch", choices=["qwen2", "deepseek2"], default="qwen2")
    ap.add_argument("--binary", default="./target/release/hawking")
    ap.add_argument("--verify-windows", default="2,4,6,8")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env = dict(QWEN_LOCKED_ENV if args.arch == "qwen2" else DEEPSEEK_LOCKED_ENV)
    windows = [int(w) for w in args.verify_windows.split(",") if w.strip()]

    print(f"[bench_head] {args.slug} arch={args.arch} head={'(baseline only)' if not args.head else Path(args.head).name}")
    t0 = time.time()

    # Baseline (no speculation).
    base = measure(args.binary, args.weights, env, None, None, args.max_new_tokens, PROMPTS)
    if not base["ok"]:
        sys.stderr.write("[bench_head] baseline failed; aborting\n")
        return 1
    base_tps = base["dec_tps_median"]
    print(f"  baseline dec_tps median = {base_tps:.2f}")

    result = {
        "slug": args.slug,
        "arch": args.arch,
        "weights": args.weights,
        "head": args.head,
        "baseline_dec_tps": base_tps,
        "windows": {},
        "max_new_tokens": args.max_new_tokens,
        "created_at_unix": int(time.time()),
    }

    best = None
    if args.head:
        for w in windows:
            spec = measure(args.binary, args.weights, env, args.head, w, args.max_new_tokens, PROMPTS)
            if not spec["ok"]:
                print(f"  verify_window={w}: FAILED")
                continue
            speedup = spec["dec_tps_median"] / base_tps if base_tps > 0 else 0.0
            result["windows"][str(w)] = {
                "dec_tps_median": spec["dec_tps_median"],
                "accept_rate_mean": spec["accept_rate_mean"],
                "speedup": speedup,
                "accepted_total": spec["accepted_total"],
                "rejected_total": spec["rejected_total"],
            }
            print(f"  verify_window={w}: spec_tps={spec['dec_tps_median']:.2f} "
                  f"accept={spec['accept_rate_mean']*100:.1f}% speedup={speedup:.2f}x")
            if best is None or speedup > best[1]:
                best = (w, speedup, spec["dec_tps_median"], spec["accept_rate_mean"])

    if best:
        w, speedup, tps, acc = best
        result["best"] = {"verify_window": w, "speedup": speedup,
                          "dec_tps_median": tps, "accept_rate_mean": acc}
        verdict = "NET WIN" if speedup > 1.0 else "NET LOSS (spec overhead exceeds acceptance gain)"
        print(f"\n  VERDICT [{args.slug}]: best window={w} "
              f"{tps:.2f} tps vs {base_tps:.2f} baseline = {speedup:.2f}x → {verdict}")
        result["verdict"] = "win" if speedup > 1.0 else "loss"
    else:
        result["verdict"] = "baseline_only"

    result["elapsed_sec"] = round(time.time() - t0, 1)

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[bench_head] wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
