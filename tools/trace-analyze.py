#!/usr/bin/env python3
"""
trace-analyze.py  --  summarize a dismantle --trace-json file.

Usage:
    python3 tools/trace-analyze.py /tmp/trace_safe_full.json
    python3 tools/trace-analyze.py /tmp/trace_safe_full.json --top 15

Outputs:
  • Top N kernels by total_us (sorted desc)
  • Per-kernel mean / p50 / p99 wall_us
  • Per-layer total_us (identifies expensive layers)
  • Summary counts: total dispatches, total dispatch wall, fraction of decode wall
"""

import argparse
import json
import sys


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def fmt_us(us):
    if us >= 1_000_000:
        return f"{us / 1_000_000:.2f}s"
    if us >= 1_000:
        return f"{us / 1_000:.1f}ms"
    return f"{us}µs"


def analyze(path, top_n=10):
    with open(path) as f:
        data = json.load(f)

    decode_tps = (
        data.get("report", {})
            .get("results", {})
            .get("decode_tps", 0.0)
    )
    dispatch_total_us = data.get("dispatch_total_us", 0)
    dispatch_pct = data.get("dispatch_wall_pct_of_decode", 0.0)

    # Completion tokens from trial_stats
    trial_stats = (
        data.get("report", {})
            .get("results", {})
            .get("trial_stats", [{}])
    )
    tokens = trial_stats[0].get("completion_tokens", 1) if trial_stats else 1

    print(f"=== {path} ===")
    print(f"decode_tps:          {decode_tps:.3f}")
    print(f"completion_tokens:   {tokens}")
    print(f"total_dispatch_us:   {dispatch_total_us:,} ({fmt_us(dispatch_total_us)})")
    print(f"total_dispatch_pct:  {dispatch_pct:.1f}% of decode wall")
    if tokens > 0:
        print(f"dispatch_us/token:   {dispatch_total_us // tokens:,} ({fmt_us(dispatch_total_us // tokens)})")

    # ── Kernel summary ───────────────────────────────────────────────────────
    kernel_summary = data.get("kernel_summary", [])
    if not kernel_summary:
        # recompute from raw samples if available
        samples = data.get("dispatch_samples", [])
        if samples:
            from collections import defaultdict
            by_k = defaultdict(list)
            for s in samples:
                by_k[s["kernel_name"]].append(s["wall_us"])
            kernel_summary = []
            for k, vals in by_k.items():
                vals_sorted = sorted(vals)
                kernel_summary.append({
                    "kernel": k,
                    "count": len(vals),
                    "total_us": sum(vals),
                    "mean_us": sum(vals) // len(vals),
                    "p50_us": percentile(vals_sorted, 50),
                    "p99_us": percentile(vals_sorted, 99),
                })
            kernel_summary.sort(key=lambda x: x["total_us"], reverse=True)

    print(f"\n── Top {top_n} kernels by total_us ──")
    header = f"{'kernel':<36} {'count':>8} {'total':>10} {'mean':>8} {'p50':>8} {'p99':>8}"
    print(header)
    print("-" * len(header))
    total_shown = 0
    for row in kernel_summary[:top_n]:
        k = row.get("kernel") or row.get("kernel_name", "?")
        cnt = row["count"]
        tot = row["total_us"]
        mean = row["mean_us"]
        p50 = row.get("p50_us", 0)
        p99 = row.get("p99_us", 0)
        pct_of_dispatch = (tot / dispatch_total_us * 100) if dispatch_total_us > 0 else 0
        total_shown += tot
        print(f"{k:<36} {cnt:>8,} {fmt_us(tot):>10} {fmt_us(mean):>8} {fmt_us(p50):>8} {fmt_us(p99):>8}  ({pct_of_dispatch:.1f}%)")

    # ── Layer summary ────────────────────────────────────────────────────────
    layer_summary = data.get("layer_summary", [])
    if layer_summary:
        print(f"\n── Per-layer total_us (top 10 most expensive layers) ──")
        by_total = sorted(layer_summary, key=lambda x: x["total_us"], reverse=True)
        layer_header = f"{'layer':>6} {'total':>10}  dominant kernel"
        print(layer_header)
        print("-" * 50)
        for row in by_total[:10]:
            li = row["layer"]
            tot = row["total_us"]
            kernels = row.get("kernels", {})
            dominant = max(kernels, key=kernels.get) if kernels else "?"
            dom_pct = (kernels.get(dominant, 0) / tot * 100) if tot > 0 else 0
            print(f"{li:>6} {fmt_us(tot):>10}  {dominant} ({dom_pct:.0f}%)")

        # full layer traversal order
        print(f"\n── Per-layer total_us (all layers, in order) ──")
        ordered = sorted(layer_summary, key=lambda x: x["layer"])
        for row in ordered:
            li = row["layer"]
            tot = row["total_us"]
            per_tok = tot // tokens if tokens > 0 else tot
            print(f"  layer {li:>2}: {fmt_us(tot):>10} total  ({fmt_us(per_tok)}/token)")

    # ── Dispatch count summary ───────────────────────────────────────────────
    total_samples = sum(r["count"] for r in kernel_summary)
    dispatch_per_token = total_samples // tokens if tokens > 0 else total_samples
    print(f"\n── Summary ──")
    print(f"total_dispatch_count:      {total_samples:,}")
    print(f"dispatch_count_per_token:  {dispatch_per_token}")
    print(f"dispatch_wall_us/token:    {fmt_us(dispatch_total_us // tokens if tokens > 0 else dispatch_total_us)}")
    print(f"dispatch_pct_of_decode:    {dispatch_pct:.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze a dismantle --trace-json file")
    parser.add_argument("trace", help="Path to trace-json file")
    parser.add_argument("--top", type=int, default=10, help="Top N kernels to show (default 10)")
    args = parser.parse_args()
    analyze(args.trace, args.top)


if __name__ == "__main__":
    main()
