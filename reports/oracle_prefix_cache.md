# Oracle — §8.1 L1.2 cross-prompt computation reuse (prefix + semantic cache)

**Date:** 2026-05-30 · **Lever:** the Bible's "THE FIRST MOVE" (`plans/throughput_bible_2026_05_30.md` §8.1 L1.2)
**Tool:** `tools/bench/oracle_prefix_cache.py` (CPU-only, no kernel) · **Verdict on proxy: GO**

> **THIS IS A PROXY, NOT A REAL MEASUREMENT.** No real served transcripts exist
> on disk, so the numbers below come from a *fabricated* coding session built
> from this repo's own source files. They estimate the **shape** of reuse a real
> session would show — they are not the production hit-rate. The real number
> needs the user's actual TailorAI session transcripts (turnkey command below).

## What the oracle measures

1. **Prefix-cache hit-rate** — per consecutive request pair, longest common
   *token* prefix / request length (exact match = bit-identical KV reuse, whole
   skipped forward passes). Aggregated to a distribution + the energy proxy
   `session_reuse_frac` = prompt tokens served from an already-cached prefix /
   all prompt tokens.
2. **Semantic-cache potential** — near-duplicate rate of contexts vs *all* prior
   requests under a similarity threshold, using a **model-free** embedding
   (hashed token-bigram cosine + token-set Jaccard; no GPU/model load). A
   `verified_*` variant additionally requires a ≥50% exact token-prefix overlap
   with the match — the Bible's "verify before trusting a near-hit."

Tokenisation shells out to `llama-tokenize` against
`models/qwen2.5-3b-instruct-q4_k_m.gguf` (same resolution as
`spec_oracle_on_transcripts.sh`), with a whitespace/byte fallback if absent.

## Proxy corpus (clearly labelled)

Generator: `/tmp/prefix_cache_proxy/build_proxy_session.py` (deterministic, seed
20260530). Models an agent Metal-kernel session: a fixed system preamble on
every request + a working set of 2 open repo files + a small appended TASK diff;
**1-in-4 steps swaps a file (context switch)**. 24 requests, 4 switches, 74,414
Qwen tokens total. This is a *realistic mix*, not a best case — the swaps are
deliberately included to keep it honest.

## Proxy results (`llama-tokenize`, Qwen2.5-3B)

| metric | value |
|---|---|
| requests / consecutive pairs | 24 / 23 |
| **mean shared-prefix** | **91.8%** |
| median shared-prefix | 99.2% |
| mean shared-prefix tokens | 2809 |
| **session reuse frac (energy proxy)** | **86.8%** (64,606 / 74,414 tok not recomputed) |
| prefix-frac buckets | 90-100%: 19 · 50-70%: 2 · 30-50%: 2 |
| near-dup rate cosine ≥0.95 / ≥0.80 | 0.913 / 0.957 |
| **verified** near-dup ≥0.95 / ≥0.80 | 0.913 / 0.957 |
| mean best-prior cosine | 0.981 |

The 4 lower-overlap pairs (30-70% buckets) are exactly the 4 context switches —
the preamble still matches, the swapped file's tail does not. Re-running with the
**fallback** tokenizer gives 91.8% mean / 87.1% reuse — tokenizer-robust.
Raw JSON: `/tmp/prefix_cache_proxy/proxy_result.json`.

## Turnkey command for REAL transcripts

```bash
# Mode 1 — N transcript files, one served session per file (session order):
/tmp/ggufenv/bin/python tools/bench/oracle_prefix_cache.py \
  --texts session1.txt session2.txt ... \
  --model models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --out reports/oracle/prefix_cache_real.json

# (one file holding a whole multi-request log? split it into a request
#  sequence on your turn marker:)
#   --texts serve_log.txt --split-on '<<<REQUEST>>>'

# Mode 2 — a JSONL of requests in session order (schema {"request": "..."}):
/tmp/ggufenv/bin/python tools/bench/oracle_prefix_cache.py \
  --jsonl requests.jsonl \
  --model models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --out reports/oracle/prefix_cache_real.json
```

Get the transcripts from TailorAI serve logs / saved completions (prompt+context
per request, in arrival order). `llama-tokenize` is already on PATH
(`/opt/homebrew/bin/llama-tokenize`).

## GO / NO-GO framing

- **GO** if `session_reuse_frac ≥ 0.50` (≥50% of prompt tokens reusable ⇒ this
  jumps to the front of the build queue, per the Bible). **MARGINAL** ≥0.25,
  else **NO-GO**.
- Proxy lands at **86.8% ⇒ GO**, and the Bible explicitly predicts "high prefix
  overlap on code is expected." The proxy is consistent with that prediction.
- **Caveat:** the proxy *constructs* shared prefixes, so a high number is partly
  baked in. Its job is only to show (i) the tool is correct and turnkey, and
  (ii) the reuse structure is plausible and large. **The decision-grade number
  is the real-transcript run.** If real sessions clear ~50% reuse — very likely
  for an agent that re-sends files/imports/scaffolding — build prefix caching
  (KV-block retention keyed by token hash; pure bookkeeping over the existing
  per-decode arenas) first, then the semantic index (a small local embedding
  table in unified memory) with the exact-prefix verify already prototyped here.

## Files

- Tool: `/Users/scammermike/Downloads/dismantle/tools/bench/oracle_prefix_cache.py`
- Proxy generator: `/tmp/prefix_cache_proxy/build_proxy_session.py`
- Proxy corpus: `/tmp/prefix_cache_proxy/session.jsonl`
- Proxy result JSON: `/tmp/prefix_cache_proxy/proxy_result.json`
