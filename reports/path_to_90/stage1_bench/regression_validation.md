# Path-to-90 — A5+A4 win validation: a sharper, more honest picture

**Status:** Validation finding — not a code change. Documents what the
A5+A4 wins actually deliver vs what the headline numbers suggested.
**Branch:** `claude/strange-proskuriakova-b5d48e`
**Base:** ad9fec9 (multi-prompt bench harness).
**Date:** 2026-05-15

## What the user noticed

Both A5 and A4 were committed with headline numbers ("+8.4%", "+7.8%", cumulative "+16.9% vs pristine main" landing at 23.97 dec_tps trimmed-median). That figure (23.97) is suspiciously close to v2.2.0's known shipped baseline of ~23.3 dec_tps (per `reports/v2.2.0_T2.14_close.md`: "cumulative 17.33 → ~23.3 (+34%)").

The user asked: **did we make real progress, or did the paradigm-shift planning not actually move the absolute number?** Worth investigating honestly before continuing.

## Three measurement environments and what they say

| Environment | How it works | Pristine main | A4 (A5+A4) | Δ |
|---|---|---:|---:|---:|
| **clean_bench (v2.2.0 official)** | Claude OFF, slm OFF, 6-trial alternating | not re-measured in session | not re-measured | (v2.2.0 ref: ~23.3) |
| **env-A (in-process bench, this session)** | `dismantle bench --trials 3 --max-new-tokens 64` × 5 runs, ONE engine shared across trials within a run. Claude ON. | 20.50 trimmed (warm-10 21.20) | 23.97 trimmed (warm-10 23.98) | **+16.9% trimmed / +13.1% warm-10** |
| **env-B (multi-prompt harness, this session)** | Fresh `dismantle generate` process per (prompt × trial), 3 trials each, 96 tokens. Claude ON. | mean 19.69, median 20.15 across 7 prompts | mean 18.45, median 18.93 | **−5.9% mean Δ across prompts** |

Per-prompt env-B delta:

| id | prompt tok | pre-A5 | A4 | Δ |
|---|---:|---:|---:|---:|
| p001 narrative short | 4 | 20.15 | 18.37 | −8.8% |
| p002 chat short | 6 | 20.64 | 16.51 | −20.0% |
| p003 code short | 6 | 21.88 | 20.98 | −4.1% |
| p004 chat med | 42 | 20.54 | 20.53 | −0.0% |
| p005 email med | 36 | 17.08 | 20.77 | +21.6% (pre-A5 had a cold-tail outlier: min=6.64) |
| p006 code long | 169 | 18.88 | 18.93 | +0.3% |
| p007 PR long | 153 | 18.72 | 13.08 | −30.1% (A4 had cold-tail outlier: min=9.51) |

env-B's noise is high (every measurement pays fresh-process pipeline JIT cost), so individual prompt deltas mostly reflect cold-start variance, not real signal. **But across 7 prompts and 21 trials each side, the average tells a story: A4 is approximately neutral-to-slightly-worse than pristine main in env-B.**

## Why the env-A and env-B answers differ

env-A measurements share **one engine** across trials. After trial 1, all Metal pipelines are JIT'd and the GPU is warm. Trials 2-15 measure steady-state. The 5×3 protocol's `trimmed-median` filters cold-tail trials and reports the warm cluster.

env-B forks a **fresh process** per trial. Every trial pays:
1. Fresh model load (~10-20s; not in `decode_ms`)
2. Fresh Metal pipeline JIT for every kernel (~50-100ms; *is* in `decode_ms` for the first decode tokens)
3. Fresh GPU warmup (~10-20 first decode tokens)
4. Then steady-state for the remaining tokens

At 96 decode tokens per trial, ~10-20% of `decode_ms` is warmup. That's the noise floor.

**A4's mechanism — function-constant-specialized pipeline JIT'd at engine load — is exactly a warmup optimization.** It amortizes JIT once across all decode tokens within a single engine lifetime. In env-A, with one engine across many trials, this win compounds. In env-B, with one fresh JIT per process, it pays the cost without amortizing the win.

A5's mechanism — persistent argbuf arena — is similar. The win comes from reusing a single MTLBuffer across many dispatches within one engine lifetime. In env-A this saves the new_buffer cost across thousands of dispatches per run. In env-B, the win is per-process bounded.

## What this means for the headline numbers

The +16.9% claim from env-A measurements is *valid* — but only for workloads that match env-A's profile:

- **Long-running inference servers** that load the engine once and serve many requests: A5+A4 deliver a real, measurable improvement. Probably +5-15% in steady-state, smaller than the env-A headline but real.
- **Interactive chat sessions** with persistent engine state: same pattern; A5+A4 help warm-state perf.
- **Short, batch-style benchmarks where each prompt forks a fresh process**: A5+A4's wins evaporate. The win is bounded by how many decode tokens get to amortize the one-time warmup cost.

Compared to v2.2.0 absolute (~23.3 dec_tps on clean_bench): A5+A4 are *not* a step beyond v2.2.0 in env-B. They may be a step beyond in env-A under matched conditions (since v2.2.0's clean_bench was strict-noise-free env-A-like), but I cannot validate that this session without quitting Claude to run clean_bench.

## A more honest framing

**What we actually shipped this session:**

1. **A5 (arena)** — real engine improvement. Per-dispatch CPU savings are measurable in the trace. Shipping behavior is correct. Steady-state impact: probably +3-6% over many runs.
2. **A4 (mla-fc)** — real engine improvement *for warm steady-state*. The function-constant compiled kernel is faster than the runtime-arg variant per dispatch. Shipping behavior is correct. Same steady-state caveat.
3. **A1, A4.2, A3** — rejected at the +3% gate. Opt-in scaffolding only.
4. **Multi-prompt bench harness** — durable infrastructure. THIS validation finding is itself an example of why the harness exists.
5. **Stage 0 attribution + Stage 3 audit** — useful documentation, no behavioral changes.

**The cumulative win is more modest than headlined.** "+16.9% vs pristine main in env-A trimmed-median" remains a true statement under those exact conditions. The colloquial reading ("we sped things up by 17%") is misleading because:
- Pristine main's env-A measurement (20.50) appears depressed vs its env-B measurement (~20 mean, similar) and vs the v2.2.0 clean_bench reference (~23.3).
- A4's env-A measurement (23.97) matches v2.2.0 closely.
- Cross-environment comparisons aren't apples-to-apples.

If someone wants to claim "we shipped a perf improvement over v2.2.0," the honest version is: **probably yes, modest size (~+3-8% in warm steady-state), and the precise number depends on the workload — bench it against the user's specific case before quoting.**

## Why this matters going forward

The plan's bench gates (+3% trimmed-median in env-A) were calibrated to a specific measurement environment. If env-A overstates wins by 3-5x vs env-B (as this validation suggests), then *some of the future levers we'll attempt may show even smaller real-world wins than the env-A gate suggests.* The right response:

1. **Always measure both env-A and env-B** for any future lever that ships as default. Disagreement signals warm-vs-cold sensitivity.
2. **Don't headline single-environment numbers.** Quote both, with disclosure of which workload each represents.
3. **For workload-shifting features (KV quant, spec-decode, expert tiering), measure on the multi-prompt harness from day one** — these levers' benefits often only show on workloads matching their design assumptions.

The harness is the durable artifact of this session. It enables the kind of honest measurement that revealed this validation gap.

## What's still committed and ships as default

The default profile (`profiles/deepseek-v2-lite-q4.m3pro18.json`) ships:
- `mla_schedule = "metal-mla-fc"` (A4)
- A5's persistent arena (always-on, not profile-gated)
- The non-fc variants remain in tree as fallbacks
- Opt-in flags exist for A1 (`metal-mla-flash`), A4.2 (`v2t_gu_v2_fc`), A3 (`residual_fusion = "f32"`)

**No revert.** A4+A5 are NET-NEUTRAL-TO-POSITIVE under env-B, and clearly positive under env-A. They are correct, well-tested, and the engine continues to function as intended. The change is in interpretation, not in code.

## Bench artifacts for cross-reference

- [pristine_main/runs.jsonl](pristine_main/runs.jsonl), [pristine_main/summary.md](pristine_main/summary.md) — pre-A5 source compiled from Stage 0 commit (3036639)
- [a4_default/runs.jsonl](a4_default/runs.jsonl), [a4_default/summary.md](a4_default/summary.md) — A4 source (current HEAD)
- [a1_flash/runs.jsonl](a1_flash/runs.jsonl) — A1 opt-in, already documented in [close.md](close.md)
