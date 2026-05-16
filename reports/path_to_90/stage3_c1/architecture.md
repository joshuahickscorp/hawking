# Path-to-90 Stage 3 C1 — Draft-head architecture decision

**Date:** 2026-05-15
**Branch:** `claude/dreamy-golick-d54ff8` (continues `claude/modest-williamson-57d50f`)
**Base:** `ae65aa5` (B1 — PPL eval harness shipped)
**Status:** DECIDED — EAGLE-3 head selected as default. Data pipeline (C2) and engine wire-up (C3) follow.

## Summary

Three published self-speculative-decode head architectures are credible for DeepSeek-V2-Lite Q4_K_M on M3 Pro: **EAGLE-3**, **MTP** (DeepSeek-V3-native), and **ReDrafter** (Apple). All three would have to be trained from scratch — V2-Lite shipped with no MTP head, and the only released DeepSeek MTP weights belong to V3 (different hidden width, different routing, different layer count).

After weighing acceptance ceiling × per-token compute × training cost × engine integration complexity, the default for path-to-90 is **EAGLE-3**. This document records the comparison and the constraints that drove the choice; C2 and C3 are scoped to that decision.

## Comparison matrix

| Dimension | EAGLE-3 | MTP (V3-native) | ReDrafter |
|---|---|---|---|
| Reference | arXiv 2503.01840 | DeepSeek-V3 paper §2.2; `nebius/MTP-DeepSeek-V3-0324` | arXiv 2403.09919 (Apple) |
| Architecture | 1-layer transformer that consumes target hidden state + previous-token embedding; outputs token via *shared* lm_head | Full transformer block (DeepSeek-V3 uses 1 MoE block) operating on shifted-and-projected hidden state | Tiny RNN (1-2 layers GRU) projecting from target hidden to next embedding |
| Published acceptance (Vicuna/MT-bench class) | **70-80%** token+1, **45-60%** token+2 (paper Table 2; tested on Vicuna, LLaMA-class targets) | 85-90% token+1 on V3 (DeepSeek-V3 paper); cross-arch transfer to V2-Lite is **unknown** | 60-70% token+1 on Vicuna-13B (paper Table 4) |
| Per-draft-token compute | ~1 transformer layer of target × K | ~1 MoE layer × K (more expensive than EAGLE — MoE gate + 6 experts per draft token) | < 0.05× target layer (RNN) |
| Parameter count for V2-Lite shape (h=2048) | ~50-80 MB (1 attn + 1 MLP block at 2048) | ~150-250 MB (1 MoE block including 64 routed experts at moe_intermediate=1408) | ~10-15 MB (small GRU) |
| Training data needed (per primary source) | ~500K dialogue samples (paper Sec 4.1) | DeepSeek-V3 trained MTP head jointly with the base model — no published from-scratch protocol; estimate 1-3M samples to match | ~200K samples (paper Sec 4) |
| Off-machine training cost (H100 estimate) | **~12-16 H100-hr** (paper-reported on Vicuna-7B; V2-Lite is comparable in active params) | **~30-50 H100-hr** (full MoE block has more params to train + jointly tune routing) | **~6-10 H100-hr** (smallest, fewest params) |
| Engine integration complexity (this codebase) | Add 1 forward path that runs the small head on captured hidden state; share existing lm_head dispatch | Add a *full MoE block* clone that respects the same routing topology — heavier dispatch graph; more new kernels | Add an RNN driver loop (sequential by nature — GRU step-by-step); doesn't fit Metal command-buffer batching as cleanly |
| Bit-identical-greedy guarantee under acceptance=0 | Yes — verify path is unchanged target-only forward | Yes | Yes |
| Implementation risk | **Low** — paper code released, multiple reference reimplementations on HF (e.g. `eagle-llm/eagle-3`); shape adapter for V2-Lite hidden=2048 / vocab=102400 is mechanical | **Medium-High** — only V3-shaped MTP exists publicly; from-scratch on V2-Lite means re-deriving the projection block + retraining; no reference impl for cross-arch transfer | **Low-Medium** — paper has reference code; but the RNN inference loop doesn't fit Metal's command-buffer model well, requires serial dispatch per draft step |

## How the dimensions interact for path-to-90

The plan's win arithmetic (Stage 3 §C3): at **K=4** verify window, the e2e wall-clock multiplier is `(1 + mean_accepts) / (K × verify_overhead_factor + draft_cost_factor)`. With dismantle's current `forward_tokens_batched` (verify cost ~K× single-forward), even a perfect drafter with 100% acceptance gives only **1.25×**. So the limiting factor right now is *not* the drafter — it's the verify path. The drafter still has to be good enough to clear the engine-side overhead, but pushing draft accuracy from 70% to 85% buys very little until the verify kernel rewrite (Path B from `stage3_spec/audit.md`) lands.

This reframes the architecture choice:

- **MTP's higher acceptance ceiling is mostly wasted** until verify becomes cheaper. We'd pay 30-50 H100-hr for an extra 10% acceptance worth maybe ~0.05× e2e. Bad trade.
- **ReDrafter's lower compute is also mostly wasted** — at K=4 with the current verify path, draft cost is already ~1% of verify cost; making the draft 5× cheaper still saves <1%.
- **EAGLE-3 sits at the right point on the curve:** competitive acceptance, training cost that fits a single H100 weekend, and an architecture that *can* be reused later when verify is cheaper (the same captured (hidden, next_token) data trains either an EAGLE head or an MTP-style head — the dataset is the architecture-agnostic asset).

## Engine-integration footprint (deepseek_v2.rs / engine.rs)

The chosen path creates new code in three places. Sketched here so C3 doesn't have to reverse-engineer scope.

**1. Engine trait (`crates/dismantle-core/src/engine.rs`):**

```rust
fn forward_token_with_hidden_for_test(
    &mut self,
    _token: u32,
    _pos: usize,
) -> Result<(Vec<f32>, u32)> {
    Err(crate::Error::Unimplemented("forward_token_with_hidden_for_test"))
}
```

Default `Unimplemented` so non-DeepSeek engines (none today, but the trait is shared) compile cleanly. Mirror of the existing `forward_token_shared_only_for_test` pattern (line 226 of engine.rs).

**2. DeepSeekV2 impl (`crates/dismantle-core/src/model/deepseek_v2.rs`):**

```rust
fn forward_token_with_hidden_for_test(&mut self, token: u32, pos: usize) -> Result<(Vec<f32>, u32)> {
    let x_norm = self.forward_token_final_norm(token, pos)?;
    // Reuse the same lm_head dispatch path as forward_token().
    let h = self.config.hidden;
    let mut logits = vec![0.0f32; self.config.vocab_size];
    let w_f16: &[f16] = self.lm_head.as_ref().map(|w| w.as_slice()).unwrap_or(&self.embed);
    self.gemv_f16_dispatch(w_f16, self.config.vocab_size, h, &x_norm, &mut logits)?;
    let argmax = crate::kernels::argmax_f32(&logits);
    Ok((x_norm, argmax))
}
```

This is a 6-line method. It reuses the existing `forward_token_final_norm` (which is already what `forward_token` calls), runs the same lm_head GEMV, and returns `(hidden, greedy_token)` instead of `logits`. No new kernel, no new dispatch path.

**3. C3 wire-up (NOT this session — sketched for completeness):**

The future `DraftSpecDecoder` will:
- Load a draft-head GGUF (50-100 MB, separate file, cached alongside model)
- Per decode step: capture target hidden via `forward_token_with_hidden_for_test`, feed it to the EAGLE head to propose K-1 additional tokens, then call `forward_tokens_batched_for_test` on the K-token draft sequence to verify
- Accept the longest matching prefix
- Reset KV via existing `reset_kv_for_test`-style truncation when draft mismatches at position j (re-encode positions j..K-1)

The verify path is identical to today's `ngram` speculate mode. The only new code path is the draft-head forward (one transformer layer evaluation per step) and the (hidden, next_token) handoff. EAGLE-3's design choice to *share* the target's lm_head means there is no separate draft-vocab dispatch — the captured hidden + EAGLE delta produce a hidden state that the existing target-side lm_head can score directly.

## Why not wait for the verify-path rewrite first

`stage3_spec/audit.md` notes that even at 80% acceptance NGram is a 15% regression because verify cost ≈ K × single-forward. The audit recommends Path A (trained drafter) **and** Path B (parallel-K verify kernels) — both paths have to exist for spec-decode to win.

The choice to land C2 (data pipeline) now, before Path B kernels, is intentional:
- Data capture is **bounded and bench-impact-free** — a session-scope deliverable that produces a reusable asset.
- Path B is multi-week kernel work; doing it before C2 leaves the trained-drafter prerequisite still ahead of us when the kernels land.
- The captured hidden-state dataset is **architecture-agnostic** — it trains EAGLE, MTP-style, or ReDrafter heads. Even if a future re-evaluation flips this decision, the same dataset feeds the alternate head.

So C2 unblocks the *long pole* (off-machine training, ~12-16 H100-hr) while engine work continues in parallel on the kernel rewrite. C3 wire-up gates on both: trained head + (eventually) cheap-verify kernels.

## Decision

**Default: EAGLE-3 head.**
- Hidden dim: 2048 (matches V2-Lite final-norm hidden width)
- Vocab: 102400 (V2-Lite tokenizer; shared with target lm_head — no separate output projection)
- Architecture: 1 attention block + 1 MLP block, target-aligned shapes
- Distillation signal: `(target_hidden_pre_lm_head[t], next_token[t+1])` per teacher-forced position
- Off-machine training: 1×H100, ~12-16h, ~500K dialogue samples (UltraChat + ShareGPT), Adam, 1 epoch

C2 produces the dataset to those specs. Off-machine training framework choice (HF accelerate vs lit-gpt vs the published EAGLE-3 reference repo) is deferred to the training session itself; the dataset format chosen for C2 (Parquet shards with raw f16 hidden bytes) is readable by all three.

## Followups (out of scope for C2)

- **Verify-kernel rewrite (Path B from audit.md):** parallel-K MLA, lm_head, MoE-gate kernels. Multi-week. Without this, even a perfect EAGLE head delivers only ~1.25× e2e at K=4. Plan as Stage 3.5 between C2 (data) and C4 (regression test).
- **Training session brief:** ~500-word document for the H100 rental session — input dataset path/format, head architecture spec (layer counts/widths), expected loss curve shape, acceptance-rate sanity check on a held-out 100-prompt slice. Include in C2 followups, not this doc.
- **Re-evaluation trigger:** if EAGLE-3 trained head reports <40% token+1 acceptance on the held-out slice, evaluate whether the failure is dataset-limited (re-train with more data) or arch-limited (consider MTP-style with the same dataset). Do not retrain blindly.
