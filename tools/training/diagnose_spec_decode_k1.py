#!/usr/bin/env python3
"""Wall-clock optimization #4 (2026-05-22): automated diagnosis of the
spec-decode K=1 tax that gates eagle5 v2 training.

Per memory `path_to_100_repath.md`:
    eagle4 / sequential / K=1   =  18.01 tps  (vs off 26.87 → −33%)
    eagle4 / parallel-k / K=4   =   7.52 tps  (−72%)

The K=1 path SHOULD be bit-identical to off-mode (the chain seed runs
the full forward; the verifier is the same kernel as off-mode greedy)
but pays an 8.9-tps tax. This script automates the diagnosis: bench
off-mode, eagle4 K=1, eagle4 K=4 across both metal and cpu spec-decode
backends, parses per-step traces, and classifies the tax as:

    structural        — overhead is in the chain seed / capture buffer;
                        can't be removed without a backend rewrite.
                        EAGLE5 v2 TRAINING WILL BE WALL-CLOCK WASTED
                        UNTIL THIS IS FIXED.
    overhead-fixable  — overhead is in CPU encode / sync; spec-decode
                        runtime patches can recover most of the tax.
                        Eagle5 training in parallel with runtime work
                        is fine.
    acceptance-capped — chain-K=4 acceptance is below the threshold
                        where any draft model wins. Eagle5 v2 with a
                        better-trained head MIGHT lift acceptance;
                        train, eval τ, then decide.

Exit codes:
    0 — overhead-fixable OR acceptance-capped: safe to start eagle5 training
    1 — structural: do NOT start training; fix runtime first
    2 — couldn't run diagnostic (binary missing, model missing, etc.)

Usage:
    python3 tools/training/diagnose_spec_decode_k1.py \\
        --weights models/deepseek-v2-lite-q4.gguf \\
        --profile profiles/deepseek-v2-lite-q4.m3pro18.json \\
        --tokens  32 \\
        --trials  3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _run_bench(
    bin_path: Path,
    weights: Path,
    profile: Path,
    speculate: str | None,
    verify_window: int,
    tokens: int,
    env_extra: dict | None = None,
) -> dict:
    """Run a single dismantle bench trial; return parsed JSON result."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_json = Path(f.name)
    cmd = [
        str(bin_path), "bench",
        "--backend", "dismantle",
        "--suite",   "decode",
        "--weights", str(weights),
        "--trials",  "1",
        "--max-new-tokens", str(tokens),
        "--kernel-profile", str(profile),
        "--json", str(out_json),
    ]
    if speculate is not None:
        cmd += ["--speculate", speculate, "--verify-window", str(verify_window)]
    env = os.environ.copy()
    env["DISMANTLE_SPEC_LOG"] = "1"
    if env_extra:
        env.update(env_extra)
    try:
        subprocess.run(
            cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=600, check=True,
        )
    except subprocess.CalledProcessError as e:
        return {"error": f"bench failed (exit {e.returncode})"}
    except subprocess.TimeoutExpired:
        return {"error": "bench timed out"}
    try:
        data = json.loads(out_json.read_text())
    finally:
        out_json.unlink(missing_ok=True)
    trial = data.get("results", {}).get("trial_stats", [{}])[0]
    return {
        "decode_tps":      trial.get("decode_tps", 0.0),
        "decode_ms":       trial.get("decode_ms", 0.0),
        "draft_accepted":  trial.get("draft_accepted", 0),
        "draft_rejected":  trial.get("draft_rejected", 0),
    }


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def diagnose(args) -> int:
    bin_path = Path(args.bin)
    if not bin_path.exists():
        print(f"ERROR: {bin_path} missing. Run `cargo build --release -p dismantle` first.",
              file=sys.stderr)
        return 2
    if not args.weights.exists():
        print(f"ERROR: weights {args.weights} missing.", file=sys.stderr)
        return 2
    if not args.profile.exists():
        print(f"ERROR: profile {args.profile} missing.", file=sys.stderr)
        return 2

    print(f"=== spec-decode K=1 diagnostic: {args.trials} trials × {args.tokens} tokens ===")
    print()

    configs = [
        ("off          ", None,           4),
        ("eagle4 K=1   ", "exact-shared", 1),
        ("eagle4 K=4   ", "exact-shared", 4),
        ("ngram  K=4   ", "ngram",        4),
    ]
    results: dict[str, dict] = {}
    for label, speculate, vw in configs:
        tps_runs = []
        ms_runs = []
        accept_total = 0
        for i in range(args.trials):
            r = _run_bench(bin_path, args.weights, args.profile, speculate, vw, args.tokens)
            if "error" in r:
                print(f"  [{label.strip()}] trial {i+1}: {r['error']}", file=sys.stderr)
                continue
            tps_runs.append(r["decode_tps"])
            ms_runs.append(r["decode_ms"])
            accept_total += r["draft_accepted"]
        if not tps_runs:
            print(f"  [{label.strip()}] all trials failed; aborting")
            return 2
        med_tps = _median(tps_runs)
        med_ms = _median(ms_runs)
        results[label.strip()] = {
            "median_tps": med_tps,
            "median_ms":  med_ms,
            "accept_per_run": (accept_total / len(tps_runs)) if tps_runs else 0,
        }
        print(f"  {label}  tps={med_tps:6.2f}   ms/tok={med_ms / max(args.tokens, 1):6.2f}   accept/run={accept_total/max(len(tps_runs),1):4.1f}")

    off_tps = results.get("off", {}).get("median_tps", 0.0)
    k1_tps = results.get("eagle4 K=1", {}).get("median_tps", 0.0)
    k4_tps = results.get("eagle4 K=4", {}).get("median_tps", 0.0)
    k4_accept_per_run = results.get("eagle4 K=4", {}).get("accept_per_run", 0)

    tax_tps = off_tps - k1_tps
    tax_pct = (tax_tps / off_tps * 100) if off_tps > 0 else 0
    k4_pct  = ((k4_tps - off_tps) / off_tps * 100) if off_tps > 0 else 0
    accept_rate_k4 = k4_accept_per_run / max(args.tokens, 1)  # crude per-step proxy

    print()
    print("=== verdict ===")
    print(f"  off-mode tps:           {off_tps:6.2f}")
    print(f"  K=1 tax (vs off):       {tax_tps:6.2f} tps  ({tax_pct:5.1f}%)")
    print(f"  K=4 delta (vs off):     {k4_tps - off_tps:6.2f} tps  ({k4_pct:5.1f}%)")
    print(f"  K=4 accept-rate proxy:  {accept_rate_k4:.3f}")
    print()

    # Verdict rules (calibrated against path-to-100-repath memo):
    if tax_pct >= 25.0:
        print("  ✱ K=1 tax ≥ 25% — this matches the documented 33% regression.")
        print("    STRUCTURAL classification: needs spec-decode runtime rework.")
        print("    DO NOT start eagle5 v2 training until the runtime is fixed.")
        print()
        print("  Next step: read reports/path_to_90/plans/path_to_100_repath.md")
        print("    Track 2 Step 2A — capture-buffer-required-for-chain-seed analysis.")
        return 1

    if accept_rate_k4 < 0.5:
        print("  ✱ K=4 acceptance < 0.5 — head quality is the wall.")
        print("    ACCEPTANCE-CAPPED: eagle5 v2 with a better head MIGHT lift this.")
        print("    Safe to start training; gate ship on τ ≥ 3.0 + clean-bench tps delta.")
        return 0

    print("  ✱ K=1 tax < 25% and acceptance reasonable.")
    print("    OVERHEAD-FIXABLE: runtime patches can recover most of the tax.")
    print("    Safe to start eagle5 training in parallel with runtime work.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="diagnose_spec_decode_k1")
    p.add_argument("--bin", type=Path, default=Path("target/release/dismantle"))
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--trials", type=int, default=3)
    args = p.parse_args()
    return diagnose(args)


if __name__ == "__main__":
    sys.exit(main())
