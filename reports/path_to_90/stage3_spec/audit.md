# Path-to-90 Stage 3 — Spec-decode audit + reality check

**Status:** No commit. This is an audit; no code shipped.
**Date:** 2026-05-15
**Base:** 97c1828 (post-A4.2 close).

## Headline

The existing speculate infrastructure (NGram, ExactShared) is wired through the engine. Both are NET REGRESSIONS on every realistic workload measured — even at high draft-acceptance rates. The root cause is structural, not a bug: dismantle's `verify` step runs the full model on K positions via `forward_tokens_batched`, costing ~K× single-forward. At verify-window K=4 with 80% acceptance, theoretical max gain is +5% nominal; the verify's extra CPU/CB overhead consumes that and more.

**Real spec-decode wins require either:** (a) a trained cheap drafter (EAGLE-3 / MTP / ReDrafter — multi-week off-machine training); or (b) new MLA + lm_head kernels that process K queries in parallel attention mode (cost ~1.5× single-forward instead of K×) — multi-week kernel work. **Neither fits a session.**

## Existing infrastructure inventory

Files (264 lines total, well-tested):
- [crates/dismantle-core/src/speculate/mod.rs](../../../crates/dismantle-core/src/speculate/mod.rs) — module root
- [crates/dismantle-core/src/speculate/ngram.rs](../../../crates/dismantle-core/src/speculate/ngram.rs) — n-gram history-match drafter, 116 lines, 5 passing tests
- [crates/dismantle-core/src/speculate/shared.rs](../../../crates/dismantle-core/src/speculate/shared.rs) — shared-expert-as-drafter, 136 lines, 2 passing tests

Engine integration ([deepseek_v2.rs](../../../crates/dismantle-core/src/model/deepseek_v2.rs)):
- `SpeculateMode::ExactShared` (line ~1009): shared-experts-only path produces draft tokens; full model verifies via `forward_tokens_batched`.
- `SpeculateMode::NGram` (line ~1205): history-match draft; full model verifies.
- `forward_tokens_batched` (line 2205): the verify forward pass. K tokens in single CB. Phase 5A: GPU-batched, Phase 4C: sequential CPU fallback.
- CLI flag: `--speculate {exact-shared|ngram} --verify-window {4|8|16}`.

## Measurements

### Test 1 — Bench prompt "Once upon a time" (4 tokens, 64-token decode)

| Mode | dec_tps (median) | acceptance |
|---|---:|---:|
| Default (no spec, A4) | 23.97 | n/a |
| NGram, verify-window=4 | 24.27 (+1.3%) | **0% (n=156)** — drafter never proposed |
| ExactShared, verify-window=4 | 10.31 **(−57%)** | 14.7% (23 accept / 133 reject) |

NGram's apparent +1.3% is measurement noise (the drafter literally never fired — `draft_accepted=0` in every run). ExactShared crashes hard because shared-expert-only output diverges from the full model 85% of the time on this prompt.

### Test 2 — Code-completion prompt (87 tokens, 85-token decode, high repetition)

| Mode | dec_tps | acceptance | per-window |
|---|---:|---:|---:|
| Default (no spec, A4) | **24.14** | n/a | n/a |
| NGram, verify-window=4 | **20.44** (−15.3%) | 79.7% (51 accept / 13 reject) | ~3.2 tokens/window |

Even at **79.7% acceptance** — well above the textbook win threshold — NGram is a 15% regression. This is the canonical structural failure mode.

### Why high acceptance still loses

Per-verify cost ≈ K × single-forward (4× at window=4). Per-verify tokens emitted = (accepted) + 1 (verifier bonus). Theoretical gain = (1 + mean_accepts) / K.

| K | acceptance | tokens/verify | nominal gain |
|---:|---:|---:|---:|
| 4 | 1.00 | 5 | 1.25× |
| 4 | 0.80 | 4.2 | 1.05× |
| 4 | 0.60 | 3.4 | **0.85×** ← regression |
| 4 | 0.50 | 3 | 0.75× |

Theoretical ceiling at K=4 is 1.25× — and that's BEFORE CPU/CB verify overhead. Measured: 0.80 acceptance → 0.847× actual. The 5% nominal gain is consumed by:
- `forward_tokens_batched` per-token bookkeeping (memcpy of batch_x_norm_buf, batch_logits)
- KV-cache writes for K positions
- Multiple sample_argmax dispatches
- CPU-side accept/reject decision

The infrastructure is correct. The arithmetic is unfavorable.

## What real spec-decode needs

The breaks are well-known from the literature, and both require multi-week scope:

### Path A — Cheap drafter, current verify

Add a tiny model (~50-100 MB EAGLE-3 head or MTP layer) that produces draft tokens at <1ms per K-token batch. Verify cost stays K× full-forward, but accept rates go to 60-85% (EAGLE-3 published numbers on Vicuna/LLaMA-class) and the math works because the DRAFT cost approaches zero. Need:
- Off-machine GPU training: ~12-24h on H100 for a 1-layer EAGLE head distilled from V2-Lite outputs
- Dataset prep: ~500K examples of (V2-Lite hidden state, V2-Lite next-token) pairs, distilled through a target stack
- Engine wire-up: `DraftSpecDecoder` trait that takes the head + hidden state, returns K draft tokens. Most of `forward_tokens_batched` reusable.
- Quality + acceptance regression test (`tests/spec_decode_acceptance.rs`)

This is **plan Stage 3 C1-C4 as originally specified** — ~4-6 weeks elapsed, mostly off-session.

### Path B — Cheap verify, same drafter

Rewrite `forward_tokens_batched` to share work across K queries: one Phase 0 (q_nope_proj) instead of K, one MoE gate evaluation per token instead of K (since experts likely overlap), one lm_head row-batch instead of K. Effective verify cost approaches 1.2-1.5× single-forward instead of K×. Then NGram at 50%+ acceptance starts winning. Needs:
- New `mla_decode_kernel_batched_k` that processes K queries per dispatch (queries differ, KV is shared across the K)
- New `lm_head_batched_k` for K-vocab outputs in one dispatch
- New `moe_route_batched_k` for K route-IDs in one dispatch
- All three at correctness parity (atol=1e-3 fp16) vs sequential
- Profile flag `verify_kernels = "batched-k"`

This is **multi-week kernel work**, similar in scope to A4 × 3. The win is general (helps any drafter) and doesn't need training.

### Path C — Defer

Skip spec-decode. Engine work has captured +16.9%; remaining engine levers (A3) target another +3-6%. Combined with KV quant (Track B), maybe reach ~30-35 dec_tps without spec. Reaches engine roofline (~50-60 dec_tps with all engine + KV-quant) but not 90+.

## Recommendation

Path A is the right move for path-to-90 (the only way to break the bandwidth roofline), but it cannot fit a session. Two realistic session-scope actions, in priority order:

1. **Pivot back to A3 (residual+RMSNorm fusion).** Low-risk +3-6% engine win. Estimated 1-2 hours implementation + bench. Fits a session. Keeps moving the needle on the engine track until either out-of-session spec-decode training lands, or scope expands.

2. **Set up Path B infrastructure scaffolding.** Add `mla_decode_kernel_batched_k` shader stub + parity test framework + profile flag. Doesn't ship the kernel (multi-week to do correctly) but documents the design and lets a future focused session pick it up. Costs ~half a session.

Recommended order: 1 first (concrete win), then 2 if there's session time remaining.

## What this audit does NOT change

- The plan's Stage 1 sequence stands. Both A4 (+7.8%) and A5 (+8.4%) shipped. A1, A4.2 rejected on evidence.
- The bandwidth roofline analysis stands: engine-only ceiling ~50-70 dec_tps. Spec-decode is the only path past that, and it requires either out-of-session training or kernel work bigger than a session.
- The infrastructure already in tree (`ngram.rs`, `shared.rs`, `forward_tokens_batched`) is correct and useful — it remains the right substrate for Path A.

## Stage 3 cumulative (unchanged)

| Stage | dec_tps (trimmed) | Δ vs main |
|---|---:|---:|
| pristine main (v2.2.0) | 20.50 | — |
| A5 + A4 (shipped) | **23.97** | **+16.9%** |
| A1, A4.2 (rejected) | — | — |
| Stage 3 audit | — | — |

## Bench artifacts

- [ngram_run{1,2,3}.json](.) — NGram speculate × 3 runs (0% accept on the bench prompt)
- [exact_shared_run{1,2,3}.json](.) — ExactShared speculate × 3 runs (regress hard)
- The two `dismantle generate` runs on the code prompt are not committed (one-off measurements).
