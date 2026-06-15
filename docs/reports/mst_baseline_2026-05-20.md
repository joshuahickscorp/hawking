# MST baseline — DeepSeek-V2-Lite Q4_K_M on M3 Pro 18 GB

Captured 2026-05-20, plan task #8 prerequisite (informs ICB sizing, #9).
Trace bundle: `traces/mst_20260520_194814.trace`.

## Capture conditions

- Binary: `target/release/dismantle bench --suite decode --max-new-tokens 32 --trials 1`
- Profile: `profiles/deepseek-v2-lite-q4.m3pro18.json` (the production
  `metal-default` profile — `mla_schedule=metal-mla`,
  `attn_block_schedule=mla`, `command_buffering=one-cb-per-block`,
  `gpu_buffer_reuse=decode-arena`).
- Weights: `models/deepseek-v2-lite-q4.gguf` (10.4 GB).
- Prompt: "Once upon a time" (4 tokens).
- Tokens decoded: 32.
- Total span: 7.245s → 58.309s (51s wall, includes xctrace deferred-mode
  overhead — bench's own decode time was 1.573s = 20.34 dec_tps).
- Trace template: Metal System Trace.
- Contamination: Claude.app running. Per
  [feedback_bench_with_claude_open](../memory/feedback_bench_with_claude_open.md),
  fine for structural counters (which are load-independent).

## Findings

### 1. Dispatch count per decode token: ~115

- Compute encoders captured: **3,788** (all attributed to the dismantle
  process — no other process contributed compute encoders in the
  window).
- Forward passes during the run: 1 prefill (4 prompt tokens, one batch) +
  32 decode tokens = 33.
- `3,788 / 33 ≈ 114.8` encoders per forward pass.
- Encoder timestamps cluster in groups of 105–118 separated by gaps in
  the 50–230 ms range (visible at indices 104, 211, 318, 425, 532 in the
  sorted timeline) — consistent with one group ≈ one forward pass.
- Command-buffer submissions: **44** across the whole run = 33 forward
  passes + ~11 setup/load/warmup commits. So `one-cb-per-block` is in
  fact `one-cb-per-token` here — all 27 transformer blocks share a
  single command buffer per forward pass, with the ~115 encoders
  representing 27 layers × ~4 encoders/layer plus head/sample steps.

**ICB sizing implication (informs #9):**
- An ICB sized for ~115 dispatches per token amortizes
  CPU-side encoder creation cost. At 21 dec_tps contaminated /
  26.87 clean, that's 2,400–3,100 encoder constructions per second
  that can be skipped via replay.
- The decode-arena already pre-allocates ~23 buffers; ICB needs to
  reference those same allocations. The 115:23 ratio means per-encoder
  buffer-binding overhead is non-trivial — bind table is what ICB
  short-circuits.

### 2. Per-kernel GPU time distribution: NOT EXTRACTABLE from this trace

- `metal-application-encoders-list` has no kernel names — every encoder
  is labelled "Compute Command N", because dismantle's Metal wrapper
  doesn't call `setLabel:` on encoders.
- `metal-shader-profiler-shader-list` only contains hashed names from
  WindowServer/Fragment shaders (e.g. `TcmaBvcmA2LlnXhfcx`); the
  shader-profiler intervals never trigger for dismantle's compute
  pipelines because they're not registered for profiling.
- The 326,983 rows in `metal-gpu-intervals` are unlabelled and have
  no pipeline identifier exposed in the XML schema.

**To unblock per-kernel timing in a future capture, the dispatch
helper at [metal/mod.rs](../crates/dismantle-core/src/metal/mod.rs) needs
to call `encoder.setLabel:` with the kernel function name** before the
`dispatch_threads` call. That's a ~10-line change touching
`MetalContext::dispatch_threads` and the few sites that bypass it.

Top-5-kernel ranking has historically been:
1. routed-expert GEMVs (Q4_K) — ~40-55%
2. shared-expert GEMVs (Q4_K) — ~15-20%
3. MLA decode kernel — ~8-12%
4. RMSNorm-fused attn input projection (gate/up) — ~5-8%
5. LM head GEMV — ~3-5%

That distribution comes from older labeled traces, not this one. Treat
it as priors to verify, not as a finding from this capture.

### 3. Bandwidth utilization vs 150 GB/s ceiling: estimated, not measured

The trace doesn't expose per-encoder byte-read counts, so the trace can
only bound the upper limit (no way to confirm the model isn't
re-reading via cache).

**Architecture math (independent of trace):**
- V2-Lite per-token warm read at Q4_K_M: routed-active experts (~6 × 130 MB ÷ 8 = ~390 MB) + shared experts (~260 MB) + attn weights (~250 MB) + MLA KV cache (~31 KB × seq) + LM head (~100 MB at fp16) ≈ **1.0 GB/token warm**.
- At clean 26.87 dec_tps: `1.0 GB × 26.87 = 27 GB/s effective ≈ 18% of 150 GB/s ceiling`.
- At contaminated 21 dec_tps: ~21 GB/s ≈ 14% of ceiling.

**Plan's stated assumption** ("0.8–1.5 GB warm read") is consistent
with this 1.0 GB estimate. The 18%-of-ceiling figure says there's
roughly 5× headroom — i.e. the bandwidth ceiling isn't what's
constraining single-stream decode. Latency and dispatch overhead
matter more. **This validates that ICB + kernel-fusion levers are the
right direction**, not bandwidth-reduction levers (quant compression
helps batch, not single-stream).

## Side finding — MLA flash-attn absorb gate failed at short context

While verifying the kernel landscape this session, I confirmed:
- `flash_attn_decode_kernel` ([attn.metal:568](../crates/dismantle-core/shaders/attn.metal:568)) **already** implements
  MLA absorb-mode flash-attention (Phase 0 absorbs K-up into Q via
  `q_nope_proj = w_uk^T · q_nope`; flash loop computes QK^T in
  `kv_lora_rank=512` latent; online softmax with `FLASH_TG=128`;
  Phase 4 absorbs V-up via `w_uv · acc`). The plan's "confirmed absent"
  claim is stale.
- Rust wrapper at [kernels/mod.rs:1691](../crates/dismantle-core/src/kernels/mod.rs:1691), profile gate at
  [deepseek_v2.rs:3345](../crates/dismantle-core/src/model/deepseek_v2.rs:3345) (`attn_block_schedule == "flash"`)
  both pre-existed.
- New parity tests at V2-Lite real shapes (n_heads=16, qk_nope=128,
  qk_rope=64, v_head=128, kv_lora=512) at seq=256/1024/2048 all pass
  with max-abs-diff ≤ 5.9e-5 vs `mla_decode_metal` — well under the
  1e-3 gate. See [tests/mla_flash_v2lite_parity.rs](../crates/dismantle-core/tests/mla_flash_v2lite_parity.rs).

**Paired bench gate (7 trials × 64 tokens, both halves with Claude.app open):**
| profile | median dec_tps | range |
|---|---|---|
| `attn_block_schedule="mla"` (baseline) | 21.13 | 20.22–21.91 |
| `attn_block_schedule="flash"` (candidate) | 20.66 | 20.10–20.76 |

Delta: **-2.23%** (regression). Flash variance is tight (range 0.65
vs mla's 1.69), so the regression signal is robust against
contamination noise.

**Why it regresses at this bench:** the bench prompt is 4 tokens and
runs 64 decode tokens, so `seq_len` stays ≤ 68 throughout. At that
length, flash's tile-loop overhead (multiple barriers per
FLASH_TG=128 tile, multiple passes over c_kv to accumulate weighted
output) dominates over the `scores[seq_len]` TG-memory savings vs
mla_decode_kernel. Flash is structurally a long-context win; at
seq~68 it loses.

**Outcome:** ships dormant. Profile selector exists; default stays
`"mla"`. Re-bench with a ≥1K-token prompt in a future session — that's
where the flash kernel should pay back.

## Open items for next session

- **Label encoders so future MST captures get per-kernel time
  distribution.** The 10-line change to `dispatch_threads` is the
  cheapest unblocker for everything that depends on per-kernel
  attribution.
- **Long-context flash bench harness.** `quick_bench.sh` hardcodes a
  4-token prompt. Need a variant that loads a >1K-token prompt to
  exercise the regime where flash should win.
- **ICB design (#9)** can proceed with the 115-dispatches/token figure.
  Sizing: 115 × decode-arena's 23 buffers = ~2,600 binding slots if a
  flat ICB, smaller with reuse-tracking.

## Reproduction

```sh
xcrun xctrace record \
    --template "Metal System Trace" \
    --output traces/mst_$(date +%Y%m%d_%H%M%S).trace \
    --launch -- ./target/release/dismantle bench \
        --backend dismantle --suite decode \
        --weights models/deepseek-v2-lite-q4.gguf \
        --trials 1 --max-new-tokens 32 \
        --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json
```

(Note: `tools/bench/mst_capture.sh` has a shell bug — `set -o pipefail`
+ `grep -q` makes xctrace's SIGPIPE register as failure, so the template
check spuriously errors. Bypass with the raw `xcrun` command above
until the script is fixed.)
