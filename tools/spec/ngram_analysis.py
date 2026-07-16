#!/usr/bin/env python3
"""
tools/spec/ngram_analysis.py — Track 6.1 n-gram oracle analysis.

Given a list of token sequences (one space-separated sequence per line on
stdin or a file), computes n-gram acceptance statistics that predict whether
a simple n-gram draft would clear the 55% depth-1 acceptance threshold for
spec-decode to be worthwhile.

USAGE (standalone):
    # read token sequences from a file (one per line):
    python3 tools/spec/ngram_analysis.py --seqs tokens.txt --ngrams 2 3 4

    # read from stdin:
    cat tokens.txt | python3 tools/spec/ngram_analysis.py --stdin

    # called by the canonical spec runner — output is machine-readable + human table
    python3 tools/spec/ngram_analysis.py --seqs FILE --ngrams 2 3 4 --json out.json

ORACLE MODEL (depth-1):
    For each position i in a token sequence of length L:
      - build a lookup table of all n-grams (prefix → next-token) from tokens 0..i-1
        (the "context so far")
      - check if any n-gram predicts tokens[i] correctly
    oracle_accept_rate = fraction of positions where at least one n-gram hits.

    Minimum n-gram order = 2 (bigram); maximum = configurable (default 4).
    A hit at any order counts as an oracle accept.

    The 55% threshold: if oracle_accept_rate > 55% across the corpus, a
    trained n-gram draft is likely worth the inference overhead of spec-decode.

OUTPUTS (to stdout):
    Per-prompt stats table + aggregate summary.

JSON SCHEMA (--json):
    {
      "n_seqs": int,
      "n_grams": [2, 3, 4],
      "per_seq": [
        {"seq_len": N, "oracle_accept_rate": float, "per_ngram": {2: float, ...}}
      ],
      "aggregate": {
        "mean_oracle_accept_rate": float,
        "std_oracle_accept_rate": float,
        "threshold_55_pct": bool,
        "per_ngram_mean": {2: float, ...}
      }
    }
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Core n-gram oracle
# ---------------------------------------------------------------------------

def build_ngram_table(tokens: List[int], max_n: int) -> Dict[Tuple[int, ...], Dict[int, int]]:
    """
    Build a frequency table mapping n-gram prefix → {next_token: count}.
    Includes all orders from 1..max_n-1 (so we can look up a prefix of length
    1..max_n-1 and predict the following token).
    """
    table: Dict[Tuple[int, ...], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for i in range(len(tokens) - 1):
        for n in range(1, max_n):
            if i - n + 1 < 0:
                break
            prefix = tuple(tokens[max(0, i - n + 1): i + 1])
            next_tok = tokens[i + 1]
            table[prefix][next_tok] += 1
    return table


def oracle_accept_rate_incremental(tokens: List[int], max_n: int, min_freq: int = 1) -> Dict:
    """
    For each position i (starting from max_n), ask: would ANY n-gram of order 2..max_n
    predict tokens[i] given context tokens[0..i-1]?

    min_freq: n-gram must appear >= min_freq times to count as a valid prediction.

    Returns a dict with:
        'overall': float   (fraction of positions with at least one n-gram hit)
        'per_order': {n: float}  (fraction for each individual n-gram order)
        'n_positions': int
    """
    if len(tokens) < 2:
        return {"overall": 0.0, "per_order": {n: 0.0 for n in range(2, max_n + 1)}, "n_positions": 0}

    hits_overall = 0
    hits_per_order: Dict[int, int] = {n: 0 for n in range(2, max_n + 1)}
    n_positions = 0

    # We build the context table incrementally to avoid leaking future tokens.
    # For efficiency, maintain a rolling prefix-count dict.
    context_table: Dict[Tuple[int, ...], Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for i in range(1, len(tokens)):
        target = tokens[i]
        n_positions += 1

        any_hit = False
        for order in range(2, max_n + 1):
            if i < order - 1:
                continue
            prefix = tuple(tokens[i - order + 1: i])
            preds = context_table.get(prefix)
            if preds:
                best_tok = max(preds, key=preds.__getitem__)
                best_count = preds[best_tok]
                if best_count >= min_freq and best_tok == target:
                    hits_per_order[order] += 1
                    any_hit = True

        if any_hit:
            hits_overall += 1

        # Update context table with position i-1 -> i bigram and longer grams
        for order in range(2, max_n + 1):
            if i - order + 1 < 0:
                break
            # prefix for predicting tokens[i]: tokens[i-order+1 .. i-1]
            ctx_start = i - order + 1
            # but at step i we just observed token i-1, so we add all grams
            # ending at i-1 (which predict tokens[i])
            # Actually, update context with the gram (tokens[i-1-k .. i-1], tokens[i])
            # = add the gram where last seen token was tokens[i-1]
            if i >= order - 1:
                gram_prefix = tuple(tokens[i - order + 1: i])
                context_table[gram_prefix][tokens[i]] += 1

    overall_rate = hits_overall / n_positions if n_positions > 0 else 0.0
    per_order_rates = {
        n: (hits_per_order[n] / n_positions if n_positions > 0 else 0.0)
        for n in range(2, max_n + 1)
    }
    return {
        "overall": overall_rate,
        "per_order": per_order_rates,
        "n_positions": n_positions,
    }


# ---------------------------------------------------------------------------
# Frequency distribution helper
# ---------------------------------------------------------------------------

def repetition_stats(tokens: List[int]) -> Dict:
    """
    What fraction of consecutive token pairs (bigrams) appear ≥2 times?
    This is a cheap proxy for sequence repetitiveness.
    """
    bigram_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for i in range(len(tokens) - 1):
        bigram_counts[(tokens[i], tokens[i + 1])] += 1

    total = len(tokens) - 1
    repeating = sum(1 for c in bigram_counts.values() if c >= 2)
    return {
        "total_bigrams": total,
        "unique_bigrams": len(bigram_counts),
        "repeating_bigrams_ge2": repeating,
        "repeat_frac": repeating / len(bigram_counts) if bigram_counts else 0.0,
    }


# ---------------------------------------------------------------------------
# Tokenization shim — try to parse integer token IDs from text, else use
# character-level surrogates (for when we only have raw text output).
# ---------------------------------------------------------------------------

def text_to_token_ids(text: str) -> List[int]:
    """
    Try to parse a space-separated list of integer token IDs.
    Fall back to UTF-8 byte values so n-gram analysis always has something
    to work with (byte-level gives a lower bound on acceptance rate).
    """
    parts = text.strip().split()
    ids = []
    for p in parts:
        try:
            ids.append(int(p))
        except ValueError:
            # not an integer token ID — treat as raw text chars
            return [ord(c) for c in text]
    return ids if ids else [ord(c) for c in text]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="N-gram oracle acceptance rate analysis for spec-decode feasibility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--seqs", metavar="FILE",
                   help="File with one token sequence per line (space-separated int IDs or raw text).")
    p.add_argument("--stdin", action="store_true",
                   help="Read sequences from stdin instead of a file.")
    p.add_argument("--ngrams", nargs="+", type=int, default=[2, 3, 4],
                   metavar="N",
                   help="N-gram orders to analyse (default: 2 3 4).")
    p.add_argument("--min-freq", type=int, default=1,
                   help="Minimum n-gram occurrence to count as a valid prediction (default: 1).")
    p.add_argument("--json", metavar="FILE",
                   help="Write machine-readable JSON summary to FILE.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-sequence table; print only summary.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    max_n = max(args.ngrams) if args.ngrams else 4

    # Read sequences
    if args.stdin:
        lines = sys.stdin.read().splitlines()
    elif args.seqs:
        try:
            with open(args.seqs) as fh:
                lines = fh.read().splitlines()
        except FileNotFoundError:
            print(f"error: --seqs file not found: {args.seqs}", file=sys.stderr)
            sys.exit(1)
    else:
        print("error: provide --seqs FILE or --stdin", file=sys.stderr)
        sys.exit(1)

    lines = [l for l in lines if l.strip()]
    if not lines:
        print("error: no sequences found in input", file=sys.stderr)
        sys.exit(1)

    # Per-sequence analysis
    per_seq = []
    overall_rates: List[float] = []
    per_order_rates: Dict[int, List[float]] = {n: [] for n in args.ngrams}

    if not args.quiet:
        hdr = f"{'seq':>5} {'len':>6} {'oracle_acc':>10}  " + \
              "  ".join(f"n={n}" for n in args.ngrams)
        print(hdr)
        print("-" * len(hdr))

    for idx, line in enumerate(lines):
        tokens = text_to_token_ids(line)
        if len(tokens) < 2:
            continue

        result = oracle_accept_rate_incremental(tokens, max_n=max_n, min_freq=args.min_freq)
        rep = repetition_stats(tokens)
        overall = result["overall"]
        overall_rates.append(overall)

        ngram_row = {}
        for n in args.ngrams:
            rate = result["per_order"].get(n, 0.0)
            per_order_rates[n].append(rate)
            ngram_row[n] = rate

        per_seq.append({
            "seq_idx": idx,
            "seq_len": len(tokens),
            "oracle_accept_rate": overall,
            "per_ngram": ngram_row,
            "repeat_frac_bigrams": rep["repeat_frac"],
        })

        if not args.quiet:
            ngram_str = "  ".join(f"{ngram_row.get(n, 0.0):.2%}" for n in args.ngrams)
            print(f"{idx:>5} {len(tokens):>6} {overall:>10.2%}  {ngram_str}")

    if not overall_rates:
        print("error: no usable sequences (all length < 2)", file=sys.stderr)
        sys.exit(1)

    # Aggregate stats
    n_seqs = len(overall_rates)
    mean_rate = sum(overall_rates) / n_seqs
    variance = sum((r - mean_rate) ** 2 for r in overall_rates) / n_seqs
    std_rate = math.sqrt(variance)
    threshold_ok = mean_rate >= 0.55

    per_ngram_mean = {
        n: (sum(per_order_rates[n]) / len(per_order_rates[n]) if per_order_rates[n] else 0.0)
        for n in args.ngrams
    }

    print()
    print("=" * 60)
    print("N-GRAM ORACLE SUMMARY")
    print("=" * 60)
    print(f"  sequences analysed   : {n_seqs}")
    print(f"  min-freq threshold   : {args.min_freq}")
    print(f"  n-gram orders        : {args.ngrams}")
    print()
    print(f"  oracle accept rate   : {mean_rate:.2%}  (std {std_rate:.2%})")
    for n in args.ngrams:
        print(f"    n={n}              : {per_ngram_mean[n]:.2%}")
    print()
    if threshold_ok:
        print(f"  VERDICT  n-gram depth-1 oracle accept rate: {mean_rate:.1%}")
        print("           >> ABOVE 55% threshold — spec-decode is WORTH investigating <<")
    else:
        print(f"  VERDICT  n-gram depth-1 oracle accept rate: {mean_rate:.1%}")
        print(f"           >> BELOW 55% threshold ({mean_rate:.1%} < 55%) — n-gram draft unlikely to pay off <<")
    print("=" * 60)

    # JSON output
    if args.json:
        doc = {
            "n_seqs": n_seqs,
            "n_grams": args.ngrams,
            "min_freq": args.min_freq,
            "per_seq": per_seq,
            "aggregate": {
                "mean_oracle_accept_rate": round(mean_rate, 6),
                "std_oracle_accept_rate": round(std_rate, 6),
                "threshold_55_pct": threshold_ok,
                "per_ngram_mean": {str(n): round(v, 6) for n, v in per_ngram_mean.items()},
            },
        }
        with open(args.json, "w") as fh:
            json.dump(doc, fh, indent=2)
        print(f"\nwrote JSON → {args.json}")


if __name__ == "__main__":
    main()
