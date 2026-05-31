# A6.5 f16s drift sweep + B2 long-context attention re-test

**Date:** 2026-05-31  **Mode:** main-tree committed binary, no source edits.
**Engine:** Qwen2.5-3B-Q4_K_M, M3 Pro, greedy temp=0.
**Locked env:** TCB=1 VOCAB_PRUNE=32000 Q4K_LMHEAD=1 FFN_DOWN_Q4K=1 Q4K_PREDEC=1.

---

## (1) A6.5 f16s broad drift sweep — DEFAULT-ON?

Tool: `dismantle bench-server` (one model load per setting), 24 diverse
prompts, 32-tok greedy, f16s flag `DISMANTLE_QWEN_PREDEC_F16SCALES` OFF then
ON. Drift = OFF vs ON decoded-text divergence (text-equality ⟺ token-ID
equality for a deterministic greedy decoder). For diverged prompts the drift
token count = tokens downstream of the first divergence (autoregressive
cascade). Harness: `reports/bench/f16s_drift_sweep.py`; raw:
`reports/bench/f16s_drift_sweep_result.json`.

**Corpus total: 63/768 tokens drifted (8.20%). Drifted prompts: 3/24
(21/24 bit-identical).**

Per-category (drifted-prompts / prompts, drift-token %):

| category    | prompts | drifted | drift% |
|-------------|--------:|--------:|-------:|
| code        | 3       | 0       | 0.00%  |
| code-sql    | 1       | 0       | 0.00%  |
| factual     | 4       | 0       | 0.00%  |
| lists       | 3       | 0       | 0.00%  |
| nonenglish  | 3       | 0       | 0.00%  |
| prose-edu   | 1       | 0       | 0.00%  |
| **math**    | 5       | **1**   | 18.1%  |
| **dialogue**| 2       | **1**   | 29.7%  |
| **prose**   | 2       | **1**   | 23.4%  |

The 3 drifts (math p007 train-problem, dialogue p010 headache-reply, prose
p020 haiku) are genuine token-ID divergences, not whitespace. All three OFF
outputs are themselves degenerate/low-quality (rambling, repetition) — i.e.
high-entropy regions where the argmax is a near-tie and the f16 scale rounding
(~5e-4 relative) tips it. This matches the prior `Q4K_FAST divergence`
finding: drift concentrates where the baseline is already low-quality.

**Verdict: KEEP OPT-IN. Do NOT flip default-on.** Drift is not ~0 across
diverse prompts — it is category-correlated, hitting open-ended generation
(math reasoning, dialogue, creative prose) at 18–30% token-drift within the
affected prompt while constrained categories (code, SQL, factual, lists,
non-English) are clean. The earlier "0/32 on 2 prompts" was a sampling
artifact of two easy prompts. f16s remains a sound *opt-in* speed lever
(+6–9%) for latency-sensitive / constrained workloads, but it is not
bit-safe enough to be the silent default.

---

## (2) B2 long-context KV-working-set re-test

**Outcome: long-context capture COULD NOT COMPLETE on this hardware/sandbox.
Reframe stays NO-GO; not resurrected (no GO evidence produced).**

### What ran
The committed attention-capture instrument (`DISMANTLE_QWEN_ATTN_CAPTURE=1`)
observes the **CPU reference** attention path (`forward_token`, no TCB). That
path recomputes, per query position per layer, the full post-softmax
distribution and does an O(ctx·log ctx) descending sort to find the 99%-mass
position count. Over a prefill of length N that is ~Σ 36·p·log p ≈
O(N²·log N) of host work — inherently heavy and un-GPU-accelerated.

- **Attempt 1** — ~3200-token code prompt (`head -c 11200` of
  `qwen_dense.rs`), min_ctx 512. Ran ~45 min in prefill, RSS ~2.6 GB, then was
  **killed by the sandbox CPU governor** before `flush()` — no JSON.
- **Attempt 2** — ~1200-token prompt, min_ctx 384. Ran ~55 min, RSS ~2.8 GB,
  then **failed exit 144 (128+SIGXCPU-class)** — CPU-time limit, no JSON.

Two attempts (the cap), both terminated by a CPU-time limit, not by a model
error. I cannot edit the instrument to speed it up (no-source-edit rule). So
the **longest completed capture remains the B2 586-token one.**

### Baseline (B2, 586-tok code prompt, re-read live from `attn_capture.json`)
- 99%-mass position fraction: median **0.797**, worst-layer **0.919**
  (worst layers are the *deep* ones — L30/L34/L35; corr(frac99, layer-idx)
  = **+0.50**, i.e. attention gets MORE diffuse with depth).
- sinks(4)+recent(128) coverage: worst-layer **0.179**, median 0.419.
- top-1 weight: median 0.173 (not peaky).
- VERDICT (oracle thresholds frac99<0.25 AND sinks+recent≥0.97): **NO-GO,
  Type-1** — mass broadly spread; both gates fail by a wide margin.

### Concentration vs context length — what the one point + structure says
I have no second length point (longer captures didn't finish), so I cannot
*measure* a sharpening trend on this model. The evidence I do have argues
AGAINST betting on it at the lengths reachable here:
- At 586 tok the structure is already anti-StreamingLLM: deep layers diffuse
  (+0.50 depth corr), sinks+recent covers as little as 18% on the worst layer.
  Concentration would have to reverse a *worsening-with-depth* pattern.
- The literature's sink/recent sharpening is documented at 16K–128K — ~30–220×
  longer than 586 tok and well beyond the ~1–3K this CPU instrument can reach
  in-budget. Nothing between 0.6K and 3K completed to show an inflection.

### Reframe verdict: **NO-GO (unchanged) — not resurrected.**
Per the Kill-Protocol "never resurrect on vibes": the named resurrection
oracle (re-run the instrument at genuinely long ctx) **could not be executed
to completion** on this hardware, so it produced no GO evidence. The
586-token Type-1 death stands. The oracle is *not* exhausted — it is
**blocked on a faster capture path**: the real long-context test needs either
(a) a GPU kernel variant that spills `scores` (the B2 report explicitly judged
this not worth building for an oracle), or (b) an instrument that records only
the cumulative-mass curve without the per-position full sort, or (c) a machine
without the CPU-time cap. Until one of those runs ≥8–16K ctx, the lever stays
dead and must not be revived. Recommendation: keep the `reports/dead_levers.md`
entry as-is; add "long-ctx capture blocked by O(N²) CPU instrument under CPU
governor — needs a cheaper capture path before the 16–32K oracle can run."

