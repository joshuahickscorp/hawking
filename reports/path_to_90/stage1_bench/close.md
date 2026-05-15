# Path-to-90 Stage 1 — Multi-prompt bench harness + A1 re-litigation

**Status:** SHIPPED. Harness landed; A1 (flash) confirmed as a regression at every context length tested.
**Branch:** `claude/strange-proskuriakova-b5d48e`
**Base:** 715403e (A3 close).
**Date:** 2026-05-15

## What

After three consecutive plan-order rejections (A1, A4.2, A3), the original Stage-1 plan ordering needed re-grounding in evidence. Pulled forward from Stage 2 / future-work: built a multi-prompt bench harness ([tools/bench/multi_prompt_bench.sh](../../../tools/bench/multi_prompt_bench.sh)) that sweeps a calibration suite ([tools/bench/multi_prompt_suite.txt](../../../tools/bench/multi_prompt_suite.txt)) across 7 prompts spanning 4-tokens (narrative) → 169-token (code review) → 153-token (PR review) prompts. Per (profile × prompt) it runs N trials of `dismantle generate --temperature 0 --max-new-tokens 96`, parses the `[stats]` line, and emits a markdown table of median dec_tps + accept-rate.

The harness is the foundation that Stage 2 (KV quant quality gates) and any future spec-decode work both need. It also gave us the chance to re-litigate A1 at non-trivial context lengths before committing more engine-track work.

## A1 (flash_attn) re-litigation — confirmed rejection, with a sharper diagnosis

The original A1 close ([reports/path_to_90/stage1_a1/close.md](../stage1_a1/close.md)) hypothesized that flash attention would win at seq_len ≥ 1K. Multi-prompt data falsifies the lighter version of that claim:

| Prompt | prompt tok | seq_len at decode end | A4 dec_tps | A1 dec_tps | Δ |
|---|---:|---:|---:|---:|---:|
| p001 narrative (short) | 4 | 100 | 18.37 | 14.40 | **−21.6%** |
| p002 chat short | 6 | 102 | 16.51 | 18.36 | +11.2% (noise — wide trial spread) |
| p003 code completion short | 6 | 102 | 20.98 | 18.80 | −10.4% |
| p004 chat medium | 42 | 138 | 20.53 | 17.62 | −14.2% |
| p005 email medium | 36 | 132 | 20.77 | 18.82 | −9.4% |
| p006 code review long | **169** | **265** | 18.93 | 10.12 | **−46.5%** |
| p007 PR review long | 153 | 220 | 13.08 | 13.13 | +0.4% (noise; EOS at 67) |

The expected "wins at long context" effect is absent. Worse, **the regression widens at longer context** (p006, seq_len=265 at decode end: −46.5%).

### Structural diagnosis

[flash_attn_decode_kernel:751-760](../../../crates/dismantle-core/shaders/attn.metal#L751-L760) has this inner loop in its tile accumulator:

```msl
for (uint r = tid; r < kv_lora_rank; r += FLASH_TG) {
    float a = acc[r] * corr_bc;
    for (uint ti = 0u; ti < t_len; ++ti) {
        float w = exp(scores_tile[ti] - m_bc);   // <-- exp() per (r × ti)
        a += w * c_kv[(t_base + ti) * kv_lora_rank + r];
    }
    acc[r] = a;
}
```

The `exp(scores_tile[ti] - m_bc)` value depends only on `ti`, not on `r` — but it gets recomputed for every `r` row.

- Threads per TG: `FLASH_TG = 128`
- Each thread handles `kv_lora_rank / FLASH_TG = 512 / 128 = 4` acc rows
- Inner loop runs `t_len` (up to 128) times per row

Total `exp()` calls per dispatch ≈ 128 threads × 4 rows × seq_len iterations = **2,048 × seq_len** exp evaluations.

Compare to `mla_decode_kernel`: softmax runs once serially in thread 0 with **seq_len** total `exp()` calls.

At seq_len = 265, flash does 2048 × 265 / 265 = **~2,048× more `exp()` calls** than mla. Apple-Silicon Metal `exp()` is software-emulated (single-precision, polynomial-approximation backed by `__metal_fast_exp` or equivalent) — multiple cycles per call. At seq_len = 265 with ~542K total `exp()`s per head per dispatch (16 heads), this drowns the kernel.

**A1 wouldn't just need a longer context to win — it needs the inner loop refactored to compute `w[ti]` once per tile (in shared mem) before the `r` sweep.** That's a kernel rewrite (~half-day work), not a wire-up.

## What this means for the plan

- **A1 stays rejected as a default schedule.** Even at seq_len > 256, the current flash kernel as-written is a regression. The dispatcher and `metal-mla-flash` profile flag remain in-tree (opt-in scaffolding from 918c93c) but are confirmed not useful at current workloads.
- **A future A1.2 — refactored flash with hoisted `w[ti]`** — could win on long-context workloads after this fix. Estimated effort: shader rewrite + parity re-verify + new bench. Half-day to a day.
- **The harness is durable infrastructure.** Future levers — A2 (Q8 KV cache, especially at long context), B3 (hot/cold experts), spec-decode work — all need realistic-workload measurement. This harness is the substrate.

## Baseline (A4) data on the suite

(For comparison and future delta-bench reference.)

| id | prompt chars | prompt tokens | completion | dec_tps (median) |
|---|---:|---:|---:|---:|
| p001 | 16 | 4 | 96 | 18.37 |
| p002 | 19 | 6 | 96 | 16.51 |
| p003 | 19 | 6 | 96 | 20.98 |
| p004 | 213 | 42 | 96 | 20.53 |
| p005 | 212 | 36 | 96 | 20.77 |
| p006 | 757 | 169 | 96 | 18.93 |
| p007 | 626 | 153 | 67 | 13.08 |

These numbers are noisier than the existing `dismantle bench` decode suite because each (prompt × trial) is a fresh process with cold pipeline JIT warmup. **Median across 3 trials damps it; relative comparisons across profiles are reliable, but absolute numbers should not be cited against the v2.2.0 `decode` bench's 23.97**. Both are valid measurements of different scenarios.

## Why these numbers are lower than the v2.2.0 `decode` bench

- `decode` bench shares one engine across N trials → cold warmup once, amortized across trials.
- `multi_prompt_bench.sh` runs one process per (prompt × trial) → cold warmup every time.

A future enhancement: drive the harness through `dismantle bench-server` (the existing long-running JSON-line server mode), so the engine stays loaded across requests. ~2 hours of work; deferred until the next bench-heavy session.

## Stage 1 cumulative (unchanged)

| Stage | dec_tps (4-token bench, trimmed-median, untraced) | Δ vs main |
|---|---:|---:|
| pristine main (v2.2.0) | 20.50 | — |
| A5 + A4 (shipped default) | **23.97** | **+16.9%** |
| A1, A4.2, A3 (rejected; opt-in scaffolding) | — | — |

## Files added

- [tools/bench/multi_prompt_bench.sh](../../../tools/bench/multi_prompt_bench.sh) — 130-line shell driver
- [tools/bench/multi_prompt_suite.txt](../../../tools/bench/multi_prompt_suite.txt) — 7-prompt calibration suite (short/medium/long)
- [reports/path_to_90/stage1_bench/a4_default/{runs.jsonl,summary.md}](a4_default/) — A4-default measurements
- [reports/path_to_90/stage1_bench/a1_flash/{runs.jsonl,summary.md}](a1_flash/) — A1-flash measurements (regression confirmed)

## Next

The harness unlocks evidence-driven prioritization. Recommended next:

1. **Build B1 PPL eval harness** (tools/bench/ppl_eval.py per the original plan). Pure tooling, ~3 hours. Unlocks B2 (KV quant) and any future quant-quality decisions.
2. **Attempt A2 (Q8 latent KV)** — KV-quant has the right structural profile to win, especially at long context where p006-style workloads showed real dec_tps decline. The KV bandwidth share grows with seq_len; at 169-token prompts it's already meaningful.
3. **Or scope C3 stub infrastructure** for self-speculative decode — even without a trained drafter, the multi-position MLA verify path is reusable and bounded effort.
