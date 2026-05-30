#!/usr/bin/env python3
"""Bible Stage-0 oracle A — n-gram / prompt-lookup (PLD) speculation acceptance.

Offline, lossless. Simulates PLD on a real CODE token stream: at each step,
match the current n-gram suffix against the earliest-emitted prefix, draft up
to K tokens from the prior continuation, accept the leading run that matches
the true stream. Mean accepted length τ = tokens emitted per verify forward.
Bible threshold: τ ≥ ~2.5 ⇒ n-gram/SAM speculation is a strong win on this
workload (the draft is a ~free CPU automaton, so τ is the speedup ceiling).

This proxies code-completion serving (prompt+generation are both code) on the
copy structure of real code. Input: a `llama-tokenize -f` dump (id -> 'piece').
"""
import argparse
import json
import re
import sys
from collections import defaultdict


def load_ids(path):
    ids = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"\s*(\d+)", line)
        if m:
            ids.append(int(m.group(1)))
    return ids


def simulate(ids, n_match, K, start=0):
    # idx maps an n-gram (ids[t-n_match:t]) to its most recent continuation
    # index t (so draft = ids[t:t+K]). Registered AFTER the query at t, so a
    # gram never self-matches.
    idx = {}
    for t in range(n_match, start):
        idx[tuple(ids[t - n_match:t])] = t
    N = len(ids)
    i = max(start, n_match)
    steps = emitted = 0
    acc = defaultdict(int)
    while i < N:
        steps += 1
        a = 0
        key = tuple(ids[i - n_match:i])
        j = idx.get(key)
        if j is not None and j < i:
            while a < K and i + a < N and j + a < N and ids[j + a] == ids[i + a]:
                a += 1
        adv = a + 1
        end = min(i + adv, N)
        for t in range(i, end):
            if t >= n_match:
                idx[tuple(ids[t - n_match:t])] = t
        emitted += end - i
        acc[a] += 1
        i = end
    tau = emitted / steps if steps else 0.0
    return {
        "n_match": n_match, "K": K, "steps": steps, "emitted": emitted,
        "mean_accepted_len": round(tau, 3),
        "hit_rate": round(sum(c for a, c in acc.items() if a > 0) / steps, 3) if steps else 0,
        "acc_hist": {str(a): acc[a] for a in sorted(acc)},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tokens", help="llama-tokenize -f dump (id -> 'piece' per line)")
    ap.add_argument("--out", default="reports/oracle/spec_accept.json")
    args = ap.parse_args()
    ids = load_ids(args.tokens)
    if len(ids) < 100:
        sys.exit(f"too few tokens ({len(ids)})")
    half = len(ids) // 2
    grid = []
    for n_match in (2, 3):
        for K in (8, 16):
            full = simulate(ids, n_match, K, start=0)
            warm = simulate(ids, n_match, K, start=half)
            full["warm_half_mean_accepted_len"] = warm["mean_accepted_len"]
            full["warm_half_hit_rate"] = warm["hit_rate"]
            grid.append(full)
    best = max(grid, key=lambda r: r["warm_half_mean_accepted_len"])
    verdict = ("GO" if best["warm_half_mean_accepted_len"] >= 2.5
               else "MARGINAL" if best["warm_half_mean_accepted_len"] >= 1.6
               else "NO-GO")
    out = {
        "oracle": "spec_accept_pld_ngram",
        "tokens": len(ids),
        "threshold_tau": 2.5,
        "best": {k: best[k] for k in
                 ("n_match", "K", "mean_accepted_len",
                  "warm_half_mean_accepted_len", "warm_half_hit_rate")},
        "verdict": verdict,
        "grid": grid,
        "note": ("PLD on real code; τ=tokens/forward. Proxies code-completion "
                 "(prompt+gen both code). Draft is ~free CPU automaton, so τ is "
                 "the speedup ceiling. GO≥2.5, MARGINAL≥1.6, else NO-GO."),
    }
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"tokens={len(ids)}  verdict={verdict}")
    for r in grid:
        print(f"  n={r['n_match']} K={r['K']:2d}: τ_full={r['mean_accepted_len']:.2f} "
              f"τ_warm={r['warm_half_mean_accepted_len']:.2f} "
              f"hit_warm={r['warm_half_hit_rate']:.2f}")
    print(f"BEST warm τ={best['warm_half_mean_accepted_len']:.2f} "
          f"(n={best['n_match']},K={best['K']}) -> {verdict}  [{args.out}]")


if __name__ == "__main__":
    main()
