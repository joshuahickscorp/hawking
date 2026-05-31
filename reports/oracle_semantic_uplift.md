# Oracle — L1.2 SEMANTIC-CACHE uplift over the exact prefix tier (git-history corpus)

**Date:** 2026-05-31 · **Lever:** design doc `plans/stateful_moat_continuation_design_2026_05_31.md` §1.6
**Tools:** `tools/bench/oracle_prefix_cache.py --sessions-glob` (CPU/NumPy, real BPE via `llama-tokenize`, no kernel)
**JSON:** `reports/oracle/semantic_uplift.json`

> **STATUS — read before citing:** this is a **git-history ESTIMATE**, a proxy for a
> real user's session stream. The production-grade number comes from real session
> logs; the same oracle runs on those the moment they exist (the generator
> `tools/bench/make_git_session_corpus.py` is `--repo`-agnostic). Carry the result
> as **"+1.5 pts mean uplift, 0 on 12/14 sessions, +13 pts on the rare return-to-
> prior-file case."** — never the bare mean.

## Verdict: **NO-GO** (build the semantic tier later, not now)

The semantic tier adds **+1.48 percentage points mean** of incremental prompt-token
reuse over the already-shipped, default-on exact prefix cache — far below the **≥10-pt**
greenlight. The retrieval is **precise** (verify-confirm 96.6–100%), so the kill is about
*opportunity*, not *noise*: on this corpus the focused-edit loop the exact tier already
catches IS the dominant reuse pattern, and non-consecutive near-duplicates are rare.

## Method

The shipped exact prefix cache reuses KV when request *i* is a byte-exact token-prefix
extension of request *i−1* (`common_prefix_len(req[i-1], req[i])`). The semantic tier
widens the candidate set: for each request *i* it nominates **any** prior request *j<i*
whose hashed-n-gram embedding cosine ≥ τ_sem, then reuses the **longest exact common
prefix** with the best such *j* — credited only when that prefix ≥ `MIN_REUSE_TOKENS`
(the design's exact-token-compare verify gate). The **payoff** is

> incremental = (best-prior semantic reuse frac) − (exact-consecutive reuse frac)

aggregated per session. Sweep τ_sem ∈ {0.95, 0.90, 0.80, 0.70} × MIN_REUSE_TOKENS ∈
{16, 32, 64}. The **verify-confirm rate** = of requests that had ≥1 semantic candidate,
the fraction whose best exact common prefix ≥ MIN_REUSE_TOKENS (precise retrieval).

**Corpus:** 14 per-session JSONL files from this repo's organic git history
(`/tmp/git_sessions_all/*.jsonl`; 2–10 turns each; a working set re-sent per turn with the
real next-commit diff). Real Qwen2.5-3B BPE via `llama-tokenize`. Measured per session,
aggregated with spread.

## Result — aggregate (14 sessions, real BPE)

**Exact-only session reuse (baseline, confirms the prior ~45% estimate):**
min 0.9% · median 45.2% · **mean 43.1%** · max 82.8%.

**Incremental uplift the semantic tier buys (pts over exact-only):**

| τ_sem | MIN_REUSE | mean | median | max | verify-confirm (weighted) |
|---|---|---|---|---|---|
| 0.95 | 16 | **+1.48** | +0.00 | +13.23 | 100.0% |
| 0.95 | 32 | +1.48 | +0.00 | +13.23 | 100.0% |
| 0.95 | 64 | +1.48 | +0.00 | +13.23 | 96.6% |
| 0.90 | 16 | +1.48 | +0.00 | +13.23 | 100.0% |
| 0.90 | 32 | +1.48 | +0.00 | +13.23 | 100.0% |
| 0.90 | 64 | +1.48 | +0.00 | +13.23 | 95.2% |
| 0.80 | 16 | **+1.48** (best) | +0.00 | +13.23 | 100.0% |
| 0.80 | 32 | +1.48 | +0.00 | +13.23 | 100.0% |
| 0.80 | 64 | +1.48 | +0.00 | +13.23 | 88.7% |
| 0.70 | 16 | +1.48 | +0.00 | +13.23 | 100.0% |
| 0.70 | 32 | +1.48 | +0.00 | +13.23 | 100.0% |
| 0.70 | 64 | +1.48 | +0.00 | +13.23 | 88.7% |

**The uplift is identical across all τ_sem.** Lowering the retrieval threshold from 0.95 to
0.70 admits **no new productive candidate** — every near-duplicate that shares a long exact
prefix is *already* a high-cosine (≥0.95) match. The threshold is not the bottleneck; the
**rarity of non-consecutive near-duplicates** is. MIN_REUSE_TOKENS=16 maximizes verify-
confirm (100%) at no cost to uplift; raising it to 64 only trims a few short-prefix tails
(verify drops to ~89–97%) without changing the reuse total.

## The finding is the per-session distribution, not the mean

| session | n | exact-only | augmented (τ0.9/mr32) | **uplift** |
|---|---|---|---|---|
| session_06 | 3 | 0.9% | 14.2% | **+13.23** |
| b2_session_02 | 8 | 46.8% | 54.3% | **+7.48** |
| session_00 | 10 | 71.5% | 71.5% | +0.02 |
| session_03 | 5 | 66.6% | 66.6% | +0.02 |
| b2_session_05 | 6 | 82.8% | 82.8% | +0.01 |
| *(9 others)* | — | — | — | **+0.00** |

**12 of 14 sessions get ZERO uplift** — the exact consecutive tier already reuses every
shared prefix. Only **2 sessions** show real semantic gain, and they share one shape: a
**return to a prior file**. In session_06, req[2] is ~99% byte-identical to req[0] (4731 of
4743 shared chars) but shares only ~257 chars with the immediately preceding req[1]. The
exact consecutive cache misses req[0] (it only looks back one turn); the semantic tier finds
it and recovers the whole prefix → +13 pts. This is exactly the case §1.2 of the design
names ("re-opens the same file … breaks the exact prefix"). It is **real but rare** on this
corpus.

## Kill Protocol classification: **Type-2** (alive, with a named oracle)

- **Type-1 or Type-2?** **Type-2.** The lever did not die on an immutable property of
  reality — the +13-pt session proves the mechanism *works* and the reuse genuinely exists
  when the access pattern is non-consecutive. It died on the **measured workload shape** of
  the git-history proxy: this corpus is consecutive-extension-shaped (a working set re-sent
  turn-over-turn), so the exact tier already harvests ~all of the reuse. A workload with
  more interleaving (multi-file context switching, re-opening older files, branch-hopping)
  would shift the distribution toward session_06's shape.
- **The reframe considered:** run the *same* oracle on **real user session logs** instead of
  the git-history proxy. The oracle is workload-agnostic (`--sessions-glob` over any per-
  session JSONL); no code change is needed — only a different corpus. A second reframe (a
  richer embedding, §1.2 candidate #2, the model's early-layer hidden state) does **not**
  apply here: verify-confirm is already 100%, so retrieval recall is not the limiter; the
  limiter is that the candidates don't exist in this corpus. Swapping the embedding cannot
  manufacture reuse that isn't there.
- **Why the reframe is alive / its cheap oracle:** the kill-oracle is **already built and
  cheap** — it is this same `oracle_prefix_cache.py --sessions-glob` re-pointed at real
  session transcripts (CPU/NumPy, ~13 s for 14 sessions, no GPU/kernel). If real logs show
  the uplift clearing ~10 pts at ≥95% verify-confirm, the lever flips GO and the design's
  build plan (§1.5: `InMemorySemanticIndex` atop the landed `restore_into`) executes
  unchanged. Until such logs exist, **the lever stays parked** — not resurrected on vibes,
  but held behind a named, runnable oracle.

## Build recommendation

**Do not build the semantic index now.** The exact prefix cache (shipped, default-on) already
captures the dominant focused-edit reuse; the semantic tier's incremental payoff on this
proxy is +1.5 pts, an order of magnitude under the gate. **Re-run this oracle on real user
session logs the moment they exist** — the metric, the sweep, and the GO/NO-GO gate are wired
and workload-agnostic. If real traffic is more file-interleaved than dismantle's own commit
history, the lever may clear; the design body is ready to execute against the landed
`restore_into` / `SemanticIndex` stub with no re-design.

## Caveats (kept honest)

- **Git-history is a proxy.** dismantle's own commits re-send a stable working set, which is
  precisely the consecutive shape the exact tier wins on — it may *understate* the semantic
  uplift a real, more-interleaved user would see (and the +13-pt session hints at that tail).
  The number is an ESTIMATE; the production number is the same oracle on real logs.
- **Short sessions.** 4 of 14 sessions have ≤4 turns, limiting the prior-context pool a
  non-consecutive match can draw from. Longer sessions give the semantic tier more candidates.
- **Embedding = the oracle's hashed-n-gram sketch** (the same signal the design's production
  index uses), so the offline measurement and the would-be runtime index agree on "near-
  duplicate" by construction — but it is an n-gram proxy, not a learned embedding.
- The uplift never decides correctness: every credited reuse is gated by an **exact token-
  prefix compare** ≥ MIN_REUSE_TOKENS, so the default mode is greedy-lossless regardless of
  τ_sem (design §1.4). The NO-GO is about payoff size, not safety.
