# Oracle — §8.1 L1.2 prefix-cache hit-rate, NON-CIRCULAR (git-history corpus)

**Date:** 2026-05-30 · **Supersedes** the circular proxy in `reports/oracle_prefix_cache.md`
**Tools:** `tools/bench/make_git_session_corpus.py` (corpus) + `tools/bench/oracle_prefix_cache.py` (measure) · CPU-only, no kernel

> **STATUS — read before citing:** this is a **non-circular git-history ESTIMATE**,
> not a live measurement. Carry it as **"~45% mean, 13–72% by session pattern"** —
> never bare "~45%" (the mean alone misleads on a ranging session like 03). The
> production-grade number is **PENDING confirmation on real session transcripts**;
> the same oracle runs on those the moment they exist (generator is `--repo`-agnostic).

## Why this run exists

The first proxy (86.8% reuse) re-read the same repo files and appended a
*synthetic* diff — it engineered the shared prefix, so the number was circular.
This run builds the corpus from **real git history**: a working set of files is
re-sent each turn, and the only thing that changes between consecutive turns is
the **actual diff of the next commit** that touched those files. The prefix
overlap is therefore whatever real consecutive commits happen to share — not
something we inserted. Fully standalone (this repo's own 450-commit history; no
external product or workload).

## Method

5 sessions from dismantle's organic history (`--skip-recent 25` to exclude this
session's own oracle commits), 12-commit windows, working set = the 3 most-
touched text files per window, re-sent at each commit's state. Tokenized with
`llama-tokenize` against the Qwen2.5-3B GGUF.

## Result

| session | mean shared prefix | median | session reuse (tok not recomputed) | per-session verdict |
|---|---|---|---|---|
| 00 | 66.7% | 90.7% | 53.2% (14013/26316) | GO |
| 01 | 40.2% | 23.2% | 41.5% (11463/27588) | MARGINAL |
| 02 | 81.8% | 99.7% | **71.6%** (35176/49155) | GO |
| 03 | 24.0% | 7.5% | **13.4%** (7243/53910) | NO-GO |
| 04 | 67.0% | 99.4% | 44.6% (11605/26012) | MARGINAL |

**Aggregate (non-circular git-history estimate, pending live-transcript confirmation):
mean session reuse ≈ 45%, 13–72% by session pattern (median 44.6%).** Always quote
the spread; the mean alone misrepresents a ranging session like 03.

## Interpretation — VERDICT: GO (build it), workload-shaped

- The ~45% mean is far below the circular proxy's 86.8% but is **real**: on
  average ~45% of a session's prefill tokens are an exact, already-computed
  prefix → skippable with **bit-identical** reuse (prefix caching is correct by
  construction; the only question was *how often* it hits).
- **The spread is the actual finding, not the mean.** Reuse is high when a
  session iterates on a stable working set (session 02: 72%; 00: 53%) and low
  when it ranges across unrelated files (session 03: 13%). A single-user local
  coding agent grinding on one feature looks like 02/00; a broad sweep looks
  like 03. So the lever's payoff is **session-shaped**, and prefix caching is a
  pure win exactly when the user is in a focused-edit loop (the common case).
- This clears the bar for a lever that is correct-by-construction and cheap
  (KV-block retention keyed by prefix hash — see `plans/stateful_core_design_2026_05_30.md`).
  Build it; size the cache to the ~45%-mean / 72%-peak expectation.

## Caveats (kept honest)

- Models an agent that re-sends full working-set files each turn. An agent that
  sends diffs instead would see different (likely lower raw, but still
  prefix-reusable) numbers.
- One repo's history. Running across several repos would broaden the estimate;
  the generator is `--repo`-agnostic for that.
- This is a defensible **estimate**, not a deployment measurement. The true
  per-deployment number comes from that deployment's own session logs (the
  generator/oracle are turnkey for those).
