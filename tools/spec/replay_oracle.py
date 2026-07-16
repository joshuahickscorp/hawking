#!/usr/bin/env python3
"""
tools/spec/replay_oracle.py — Track 6.2 offline replay oracle.

Reads verifier traces produced by `tools/spec/run.py capture` and evaluates candidate
draft policies WITHOUT running the model.  Each trace contains the ground-truth
greedy token sequence; the oracle simulates what a given draft policy would
have proposed and whether the verifier would have accepted it.

POLICIES:
  ngram         — n-gram lookup (order 2..4) using context-so-far as
                  the draft source.  At each verify step we propose K tokens
                  using the best n-gram prediction; count how many consecutive
                  proposals the verifier (ground truth) would accept.

  last-repeat   — repeat the last token as the draft.  Degenerate baseline;
                  useful on repetitive/code corpora.

  unigram       — always propose the most-frequent token seen so far.

METRICS:
  depth-1 accept rate    fraction of steps where the FIRST draft token matches
  depth-2 accept rate    fraction of steps where BOTH first and second match
  depth-3 accept rate    fraction of steps where first three all match
  mean_accepted_per_step average number of consecutive accepted tokens per
                          verify cycle (capped at draft depth K)

VERDICT:
  GO     depth-1 accept rate >= 55% (spec-decode overhead pays off)
  NO-GO  depth-1 accept rate <  55%

USAGE:
  python3 tools/spec/replay_oracle.py --traces traces/traces.jsonl
  python3 tools/spec/replay_oracle.py --traces traces.jsonl --policy ngram,last-repeat
  python3 tools/spec/replay_oracle.py --traces traces.jsonl --k 4 --min-freq 2 --json out.json

COMMAND-LINE FLAGS:
  --traces FILE      JSONL traces file from the canonical capture runner (required)
  --policy LIST      comma-separated list of policies to evaluate
                     (default: ngram,last-repeat,unigram)
  --k INT            draft depth per verify step (default: 3)
  --ngram-orders N   space or comma separated n-gram orders (default: 2 3 4)
  --min-freq INT     min n-gram count to consider a valid prediction (default: 1)
  --json FILE        write machine-readable JSON summary to FILE
  --quiet            suppress per-trace rows; print only aggregate table
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# N-gram draft policy
# ---------------------------------------------------------------------------

def ngram_predict(context: List[int],
                  table: Dict[Tuple[int, ...], Dict[int, int]],
                  max_order: int,
                  min_freq: int) -> Optional[int]:
    """
    Given the current context and a running n-gram table, return the best
    predicted next token (highest count at any order >= 2).
    Returns None if no n-gram fires.
    """
    best_tok: Optional[int] = None
    best_cnt: int = 0
    for order in range(2, max_order + 1):
        if len(context) < order - 1:
            continue
        prefix = tuple(context[-(order - 1):])
        preds  = table.get(prefix)
        if not preds:
            continue
        tok = max(preds, key=preds.__getitem__)
        cnt = preds[tok]
        if cnt >= min_freq and cnt > best_cnt:
            best_tok = tok
            best_cnt = cnt
    return best_tok


def ngram_update(context: List[int],
                 new_tok: int,
                 table: Dict[Tuple[int, ...], Dict[int, int]],
                 max_order: int) -> None:
    """Add the just-observed (context[-max_order+1:], new_tok) counts."""
    n = len(context)
    for order in range(2, max_order + 1):
        if n < order - 1:
            break
        prefix = tuple(context[-(order - 1):])
        table[prefix][new_tok] += 1


# ---------------------------------------------------------------------------
# Policy simulators
# ---------------------------------------------------------------------------

def simulate_ngram(generated_tokens: List[int],
                   prompt_tokens: List[int],
                   k: int,
                   max_order: int,
                   min_freq: int) -> Dict:
    """
    Simulate the n-gram draft policy on one trace.

    At each position i in generated_tokens:
      - context = prompt_tokens + generated_tokens[:i]
      - propose up to K tokens by iterating ngram_predict over the
        running context (no future leakage)
      - count consecutive accepts against ground truth

    Returns per-depth hit counts and step count.
    """
    context: List[int] = list(prompt_tokens)
    table: Dict[Tuple[int, ...], Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    # Seed the table with prompt bigrams / trigrams
    for j in range(1, len(context)):
        for order in range(2, max_order + 1):
            if j < order - 1:
                continue
            prefix = tuple(context[j - order + 1: j])
            table[prefix][context[j]] += 1

    depth_hits   = [0] * k   # depth_hits[d] = steps where >=d+1 proposals accepted
    total_accept = 0
    n_steps      = 0

    i = 0
    while i < len(generated_tokens):
        # Propose K tokens
        proposals: List[int] = []
        sim_ctx = list(context)
        for _ in range(k):
            pred = ngram_predict(sim_ctx, table, max_order, min_freq)
            if pred is None:
                break
            proposals.append(pred)
            sim_ctx.append(pred)

        # Count consecutive accepts against ground truth
        n_accepted = 0
        for d, prop in enumerate(proposals):
            if i + d >= len(generated_tokens):
                break
            if prop == generated_tokens[i + d]:
                n_accepted += 1
            else:
                break

        # Record depth hits
        for d in range(k):
            if n_accepted > d:
                depth_hits[d] += 1

        total_accept += n_accepted
        n_steps      += 1

        # Advance by accepted + 1 (verifier always advances at least one token)
        advance = max(1, n_accepted + 1)
        for step in range(advance):
            if i + step < len(generated_tokens):
                tok = generated_tokens[i + step]
                ngram_update(context, tok, table, max_order)
                context.append(tok)
        i += advance

    return {
        "n_steps":      n_steps,
        "depth_hits":   depth_hits,
        "total_accept": total_accept,
    }


def simulate_last_repeat(generated_tokens: List[int],
                         prompt_tokens: List[int],
                         k: int) -> Dict:
    """
    Draft policy: always propose the last seen token k times.
    """
    context: List[int] = list(prompt_tokens)
    depth_hits   = [0] * k
    total_accept = 0
    n_steps      = 0

    i = 0
    while i < len(generated_tokens):
        last = context[-1] if context else 0
        proposals = [last] * k

        n_accepted = 0
        for d, prop in enumerate(proposals):
            if i + d >= len(generated_tokens):
                break
            if prop == generated_tokens[i + d]:
                n_accepted += 1
            else:
                break

        for d in range(k):
            if n_accepted > d:
                depth_hits[d] += 1

        total_accept += n_accepted
        n_steps      += 1

        advance = max(1, n_accepted + 1)
        for step in range(advance):
            if i + step < len(generated_tokens):
                context.append(generated_tokens[i + step])
        i += advance

    return {
        "n_steps":      n_steps,
        "depth_hits":   depth_hits,
        "total_accept": total_accept,
    }


def simulate_unigram(generated_tokens: List[int],
                     prompt_tokens: List[int],
                     k: int) -> Dict:
    """
    Draft policy: always propose the most-frequent token seen so far.
    """
    freq: Dict[int, int] = defaultdict(int)
    for tok in prompt_tokens:
        freq[tok] += 1

    depth_hits   = [0] * k
    total_accept = 0
    n_steps      = 0

    i = 0
    while i < len(generated_tokens):
        if freq:
            best = max(freq, key=freq.__getitem__)
            proposals = [best] * k
        else:
            proposals = []

        n_accepted = 0
        for d, prop in enumerate(proposals):
            if i + d >= len(generated_tokens):
                break
            if prop == generated_tokens[i + d]:
                n_accepted += 1
            else:
                break

        for d in range(k):
            if n_accepted > d:
                depth_hits[d] += 1

        total_accept += n_accepted
        n_steps      += 1

        advance = max(1, n_accepted + 1)
        for step in range(advance):
            if i + step < len(generated_tokens):
                freq[generated_tokens[i + step]] += 1
        i += advance

    return {
        "n_steps":      n_steps,
        "depth_hits":   depth_hits,
        "total_accept": total_accept,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

POLICY_DISPATCHERS = {
    "ngram":       None,   # filled in at call time (needs extra args)
    "last-repeat": simulate_last_repeat,
    "unigram":     simulate_unigram,
}


def run_policy(policy: str,
               generated_tokens: List[int],
               prompt_tokens: List[int],
               k: int,
               max_order: int,
               min_freq: int) -> Dict:
    if policy == "ngram":
        return simulate_ngram(generated_tokens, prompt_tokens, k, max_order, min_freq)
    elif policy == "last-repeat":
        return simulate_last_repeat(generated_tokens, prompt_tokens, k)
    elif policy == "unigram":
        return simulate_unigram(generated_tokens, prompt_tokens, k)
    else:
        raise ValueError(f"unknown policy: {policy!r}")


def aggregate_results(per_trace: List[Dict], k: int) -> Dict:
    """
    Aggregate depth-hit counts across traces.
    Returns depth accept rates and mean_accepted_per_step.
    """
    total_steps  = 0
    depth_hits   = [0] * k
    total_accept = 0

    for r in per_trace:
        total_steps  += r["n_steps"]
        total_accept += r["total_accept"]
        for d in range(k):
            depth_hits[d] += r["depth_hits"][d]

    if total_steps == 0:
        return {
            "depth_accept_rates": [0.0] * k,
            "mean_accepted_per_step": 0.0,
            "total_steps": 0,
        }

    rates = [depth_hits[d] / total_steps for d in range(k)]
    mean  = total_accept / total_steps

    return {
        "depth_accept_rates": rates,
        "mean_accepted_per_step": round(mean, 4),
        "total_steps": total_steps,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline replay oracle for spec-decode draft policy evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--traces", required=True, metavar="FILE",
                   help="JSONL traces file from capture_traces.sh")
    p.add_argument("--policy", default="ngram,last-repeat,unigram",
                   help="Comma-separated policies: ngram,last-repeat,unigram "
                        "(default: all three)")
    p.add_argument("--k", type=int, default=3,
                   help="Draft depth per verify step (default: 3)")
    p.add_argument("--ngram-orders", default="2,3,4",
                   help="Comma or space separated n-gram orders (default: 2,3,4)")
    p.add_argument("--min-freq", type=int, default=1,
                   help="Minimum n-gram occurrence for a valid prediction (default: 1)")
    p.add_argument("--json", metavar="FILE",
                   help="Write machine-readable JSON summary to FILE")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-trace table; print only aggregate summary")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    policies = [p.strip() for p in args.policy.split(",") if p.strip()]
    for pol in policies:
        if pol not in POLICY_DISPATCHERS:
            print(f"error: unknown policy {pol!r}. choose from: {', '.join(POLICY_DISPATCHERS)}", file=sys.stderr)
            sys.exit(1)

    # Parse n-gram orders (accept comma or space separated)
    raw_orders = args.ngram_orders.replace(",", " ").split()
    try:
        ngram_orders = [int(x) for x in raw_orders]
    except ValueError:
        print(f"error: invalid --ngram-orders: {args.ngram_orders}", file=sys.stderr)
        sys.exit(1)
    max_order = max(ngram_orders) if ngram_orders else 4

    # Load traces
    try:
        traces = []
        with open(args.traces) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"warn: skipping line {lineno}: {e}", file=sys.stderr)
    except FileNotFoundError:
        print(f"error: traces file not found: {args.traces}", file=sys.stderr)
        sys.exit(1)

    if not traces:
        print("error: no traces loaded", file=sys.stderr)
        sys.exit(1)

    print(f"=== replay_oracle — Track 6.2 offline draft policy evaluation ===")
    print(f"traces file  : {args.traces}  ({len(traces)} traces)")
    print(f"policies     : {', '.join(policies)}")
    print(f"draft depth k: {args.k}")
    print(f"n-gram orders: {ngram_orders}  min-freq: {args.min_freq}")
    print()

    # Per-trace, per-policy results
    per_policy_trace: Dict[str, List[Dict]] = {pol: [] for pol in policies}

    if not args.quiet:
        # Header
        pol_cols = "".join(f"  {pol:<12}" for pol in policies)
        print(f"{'idx':>4} {'n_gen':>6}{pol_cols}")
        print("-" * (4 + 7 + 14 * len(policies)))

    for trace in traces:
        idx        = trace.get("prompt_idx", "?")
        gen_toks   = trace.get("generated_tokens", [])
        prompt_toks= trace.get("prompt_tokens", [])

        if len(gen_toks) < 2:
            if not args.quiet:
                print(f"{str(idx):>4} {len(gen_toks):>6}  (skipped — too short)")
            continue

        row_parts = []
        for pol in policies:
            result = run_policy(pol, gen_toks, prompt_toks, args.k, max_order, args.min_freq)
            per_policy_trace[pol].append(result)
            d1 = result["depth_hits"][0] / result["n_steps"] if result["n_steps"] else 0.0
            row_parts.append(f"  d1={d1:.1%}      ")

        if not args.quiet:
            pol_str = "".join(row_parts)
            print(f"{str(idx):>4} {len(gen_toks):>6}{pol_str}")

    # Aggregate
    print()
    print("=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"{'policy':<16}  {'d1_acc':>8}  {'d2_acc':>8}  {'d3_acc':>8}  {'mean_acc':>9}  {'steps':>7}  verdict")
    print("-" * 70)

    verdicts: Dict[str, str] = {}
    agg_data: Dict[str, Dict] = {}

    for pol in policies:
        agg  = aggregate_results(per_policy_trace[pol], args.k)
        rates = agg["depth_accept_rates"]
        d1    = rates[0] if len(rates) > 0 else 0.0
        d2    = rates[1] if len(rates) > 1 else 0.0
        d3    = rates[2] if len(rates) > 2 else 0.0
        mean  = agg["mean_accepted_per_step"]
        steps = agg["total_steps"]

        verdict = "GO" if d1 >= 0.55 else "NO-GO"
        verdicts[pol] = verdict
        agg_data[pol] = {
            "depth_accept_rates": {f"d{i+1}": round(r, 6) for i, r in enumerate(rates)},
            "mean_accepted_per_step": mean,
            "total_steps": steps,
            "verdict": verdict,
        }

        print(f"{pol:<16}  {d1:>7.1%}  {d2:>8.1%}  {d3:>8.1%}  {mean:>9.3f}  {steps:>7}  {verdict}")

    print("=" * 70)
    print()
    print("VERDICT KEY: GO = depth-1 accept >= 55% (spec overhead likely pays off)")
    print("             NO-GO = depth-1 accept < 55% (spec overhead unlikely to pay off)")
    print()

    # Comparison callout
    if len(policies) > 1:
        best_pol = max(policies, key=lambda p: agg_data[p]["depth_accept_rates"].get("d1", 0.0))
        best_d1  = agg_data[best_pol]["depth_accept_rates"].get("d1", 0.0)
        print(f"Best policy: {best_pol}  (d1={best_d1:.1%}  verdict={verdicts[best_pol]})")
        print()

    # JSON output
    if args.json:
        doc = {
            "traces_file": args.traces,
            "n_traces":    len(traces),
            "k":           args.k,
            "ngram_orders": ngram_orders,
            "min_freq":    args.min_freq,
            "policies":    agg_data,
        }
        with open(args.json, "w") as fh:
            json.dump(doc, fh, indent=2)
        print(f"wrote JSON → {args.json}")


if __name__ == "__main__":
    main()
