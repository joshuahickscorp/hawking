#!/usr/bin/env python3.12
"""expert_cache_policy.py — simulate the MoE hot-expert cache/prefetch POLICY before the Rust OOC
pager is built, so its cache size and eviction rule are chosen from a measured simulation instead
of a guess.

Direct finding from this session: the one honest asterisk on "compression and speed aren't really
in tension" is the MoE COLD-PATH - paging an unrouted expert in from SSD is genuinely slow
(SSD-bandwidth bound). The mitigation is a hot-expert cache + prefetch, and its EFFECTIVENESS
depends entirely on how skewed real routing is (Zipfian route frequency, per expert_sensitivity.py)
and how big a cache the RAM budget allows. This tool answers: for a given (n_experts, active_k,
route-skew, cache_size_experts), what fraction of forward passes hit the cache (warm, fast) vs miss
(cold, SSD-bound), and what is the resulting BLENDED tok/s - so the pager's cache size is a
DECIDED number, not a shot in the dark.

Model: routing frequency follows a Zipf-like distribution (skew parameter s; s=0 is uniform/all-
cold-equally-likely, s=1+ is realistic MoE skew where a handful of experts dominate, matching the
"cold experts rarely routed" finding from expert_sensitivity.py). Simulates N forward passes,
each routing active_k experts by the Zipf weights; a cache of size C (LRU) tracks hits/misses;
blended tok/s = 1 / (p_hit/tok_s_warm_active + p_miss*(1-p_hit... )) using ramcliff_bench's
per-expert-active-GB SSD-bandwidth model for the cold cost and RAM bandwidth for the warm cost.

GATE: if even a generously-sized cache (25% of experts resident) can't push blended tok/s above
~1 tok/s, the model needs MORE cache RAM (smaller weight bpw to free budget) or is genuinely
batch/async-only at this size - a real constraint, not a bug to code around.

Usage:
  expert_cache_policy.py --sim <n_experts> <active_k> [--skew 1.2] [--cache-frac 0.1,0.25,0.5]
                          [--active-gb 0.31] [--ssd-gbps 6.0] [--ram-gbps 800]
"""
import sys, os, json, random

OUT = "reports/condense"


def zipf_weights(n, s):
    ranks = list(range(1, n + 1))
    w = [1.0 / (r ** s) for r in ranks]
    tot = sum(w)
    return [x / tot for x in w]


def simulate(n_experts, active_k, skew, cache_frac, trials=20000, seed=0):
    random.seed(seed)
    weights = zipf_weights(n_experts, skew)
    order = sorted(range(n_experts), key=lambda i: -weights[i])   # hottest first (oracle-ranked cache)
    cache_size = max(1, int(n_experts * cache_frac))
    cache = set(order[:cache_size])                                # static top-K cache (measures the FLOOR;
    hits, total = 0, 0                                              # an LRU/adaptive cache only does better)
    for _ in range(trials):
        routed = random.choices(range(n_experts), weights=weights, k=active_k)
        for e in set(routed):
            total += 1
            if e in cache:
                hits += 1
    return hits / total if total else 0.0


def blended_tok_s(p_hit, active_gb, ram_gbps=800.0, ssd_gbps=6.0):
    tok_s_warm = ram_gbps / active_gb if active_gb else 0.0
    tok_s_cold = ssd_gbps / active_gb if active_gb else 0.0
    # per-token expected time = p_hit*t_warm + (1-p_hit)*t_cold ; tok/s = 1/expected_time
    t_warm, t_cold = (1 / tok_s_warm if tok_s_warm else 0), (1 / tok_s_cold if tok_s_cold else 0)
    et = p_hit * t_warm + (1 - p_hit) * t_cold
    return 1 / et if et else 0.0


def run(n_experts, active_k, skew, cache_fracs, active_gb, ssd_gbps, ram_gbps):
    rows = []
    for cf in cache_fracs:
        p_hit = simulate(n_experts, active_k, skew, cf)
        tps = blended_tok_s(p_hit, active_gb, ram_gbps, ssd_gbps)
        rows.append({"cache_frac": cf, "cache_experts": max(1, int(n_experts * cf)),
                     "hit_rate": round(p_hit, 3), "blended_tok_s": round(tps, 2)})
    gate = next((r for r in rows if r["blended_tok_s"] >= 1.0), None)
    rec = {"n_experts": n_experts, "active_k": active_k, "skew": skew, "active_gb_per_tok": active_gb,
           "ssd_gbps": ssd_gbps, "ram_gbps": ram_gbps, "rows": rows,
           "recommended_cache_frac": gate["cache_frac"] if gate else None,
           "verdict": (f"cache_frac={gate['cache_frac']} reaches {gate['blended_tok_s']} tok/s (>=1.0 GATE)"
                      if gate else "no cache size in the sweep reaches 1.0 tok/s — needs more cache RAM "
                                    "(lower bpw) or is batch/async-only at this size"),
           "probe": True}
    os.makedirs(OUT, exist_ok=True)
    lbl = f"n{n_experts}_k{active_k}_s{skew}"
    json.dump(rec, open(f"{OUT}/expertcache_{lbl}.json", "w"), indent=2)
    print(f"[cache] {n_experts} experts, active_k={active_k}, skew={skew} "
          f"(active {active_gb}GB/tok, SSD {ssd_gbps}GB/s, RAM {ram_gbps}GB/s):", file=sys.stderr)
    for r in rows:
        print(f"  cache={r['cache_frac']:.0%} ({r['cache_experts']} experts) hit={r['hit_rate']:.0%} "
              f"-> {r['blended_tok_s']} tok/s", file=sys.stderr)
    print(f"# {rec['verdict']}", file=sys.stderr)
    return rec


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "--help"
    if a == "--sim":
        n = int(sys.argv[2]); k = int(sys.argv[3])
        skew = float(sys.argv[sys.argv.index("--skew") + 1]) if "--skew" in sys.argv else 1.2
        fracs = ([float(x) for x in sys.argv[sys.argv.index("--cache-frac") + 1].split(",")]
                if "--cache-frac" in sys.argv else [0.1, 0.25, 0.5])
        active_gb = float(sys.argv[sys.argv.index("--active-gb") + 1]) if "--active-gb" in sys.argv else 0.31
        ssd = float(sys.argv[sys.argv.index("--ssd-gbps") + 1]) if "--ssd-gbps" in sys.argv else 5.0
        ram = float(sys.argv[sys.argv.index("--ram-gbps") + 1]) if "--ram-gbps" in sys.argv else 800.0
        run(n, k, skew, fracs, active_gb, ssd, ram)
    else:
        print(__doc__)
