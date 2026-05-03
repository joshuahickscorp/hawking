# Phase 2 speed followups — captured during haul 3

Triage of where the post-haul-3 dec_tps (~0.13) is actually going, and
which fixes give which speedup. **Ranked by impact ÷ implementation
cost.** Item ① is what closes the Stage-1 ratio (≥1.5× llama.cpp);
items ② and ③ are necessary but not sufficient; ④–⑥ are smaller
wins.

The numbers below are first-order estimates from the haul-3 measured
baseline (50-prompt batch-hash at ~17 s/prompt; 3 tokens decode +
prefill ≈ 5.7 s per decoded token). They will be off by ±50% but the
ordering is right.

## ① FlashDMoE-style batched expert dispatch — **~15–30× decode**

**Where the time goes.** Per token, the routed-MoE FFN issues
**189 separate kernel dispatches** (top-6 experts × 3 matmuls per
expert × 27 MoE layers + 81 shared-expert dispatches). Each Metal
`dispatch_threads(...)` call is a `commit + waitUntilCompleted`
round-trip at ~1–3 ms of pure scheduler overhead — that's
**~280–560 ms/token of dispatch latency** before any actual compute.
This is the single largest budget item in the post-haul-3 path.

**The fix.** Replace the 189 separate dispatches with **one batched
dispatch** that takes:

- Q4_K_M expert weight tensor (the existing `ffn_gate_exps.weight`
  3D layout already concatenates all 64 experts contiguously)
- An `(n_tokens, top_k)` selector buffer of `(expert_id, weight)`
  pairs
- The activation tensor

and produces the weighted sum directly. The kernel walks the
selector inline; one workgroup per (token, output-row). Thread
ordering hides per-expert routing in shared memory.

**Pre-existing scaffolding.** The repo already has the right shapes:
- `crates/dismantle-core/shaders/moe.metal` has stubs for
  `moe_grouped_gemm_q4` and `moe_gather_combine` (haul 2 H2.2/H2.3 —
  already parity-attested, but currently invoked once per
  (layer, expert), not batched).
- `crates/dismantle-core/src/moe/dispatch.rs::build_work_queue`
  already groups token×expert into expert-buckets — the host-side
  half of the batched dispatch is partially done.

**Approach.**
1. Author a new shader `moe_block_fused` in `shaders/moe.metal` that
   replaces the 3-step (gate→up→silu_mul→down) loop with a single
   workgroup-per-output-row dispatch. The kernel reads the
   selector once and walks all top-K experts in shared memory,
   accumulating into a per-thread fp32 register.
2. Add `kernels::moe_block_fused_metal(ctx, ...)` host dispatch.
3. Add `tests/correctness/phase2_moe_block_parity.rs` — same
   atol=1e-3 vs CPU MoE reference. **One halt-budget gate** in the
   Phase 2 manifest.
4. Replace the `for (eid, weight) in routes` loop in
   `model::deepseek_v2::ffn` with a single dispatch.

**Estimated impact.** 280–560 ms → 5–15 ms per token → **15–30×
decode speedup**. After this lands, `r_llama` should clear ≥0.5
even before items ②/③.

---

## ② Persistent device-side weight buffers — **~1.5–3× decode**

**Where the time goes.** Every kernel dispatch in the post-haul-3
path calls `ctx.new_buffer_with_bytes(weight_bytes)`, which is a
**memcpy from host pointer into a fresh Metal buffer**. On Apple
Silicon's unified memory this is ~5–10 GB/s, but the volume is
large:

- LM head (`gemv_f16_metal`): 400 MB × 1 call/token = ~80 ms/token
- O-proj (`gemv_f32_attn_metal`): 16 MB × 27 layers = ~85 ms/token
- Routed-expert (`gemv_q4_k_m`): 1.4 MB × 189 = ~50 ms/token
- Misc (rmsnorm weights, gate logits): ~5 ms/token

**Total: ~220 ms/token of pure memcpy** that's redundant — the
weights never change.

**The fix.** At model load time, for every weight tensor:

1. Allocate a `metal::Buffer` once (using
   `MetalContext::new_buffer_with_bytes` for fp16/fp32 weights, or
   `new_buffer_no_copy` for the GGUF-mmap'd Q4_K bytes — the Metal
   page-aligned-no-copy path keeps weights in the GGUF mmap
   region).
2. Store buffers as `MetalBufferRef` fields on `Layer` and on
   `DeepSeekV2` (final_norm, embed, lm_head).
3. Add `*_pinned` variants of the kernel entry points that take
   `&Buffer` instead of `&[u8]`/`&[f32]`. The current entry points
   stay for the parity tests (small, RAM-fixture path).
4. Change call sites in `model::deepseek_v2` to use the pinned
   variants.

**RAM cost.** Weight-pinning means all touched weight pages stay
resident — that's the "use 16 GB" the haul-3 user observed should
happen. The Q4_K bytes (~5 GB) plus eager-dequanted attn weights
(~1 GB) plus LM head fp16 (400 MB) gets us to ~7 GB resident pretty
fast. After warmup of all 50 prompts (touches all expert pages
across runs) we'd settle at ~9 GB resident — exactly what the box
should be doing.

**Estimated impact.** 220 ms → ~5 ms per token → **~1.4× decode
speedup**. Multiplicative with item ①.

---

## ③ MLA / Q-LoRA gemvs onto Metal — **~1.2–1.5× decode**

**Where the time goes.** A1 (haul 3) wired only the o_proj and the
MoE gate-logits gemvs onto Metal. The remaining attention gemvs are
still on the CPU `gemv_f32` path:

- `q_a_proj`: 2048 × 1536 ≈ 3 M ops × 27 layers = 81 M ops/token
- `q_b_proj`: 1536 × 3072 ≈ 4.7 M ops × 27 = 127 M ops/token
- `kv_a_proj_with_mqa`: 2048 × 576 ≈ 1.2 M ops × 27 = 32 M ops/token
- `kv_b_proj`: 576 × 2048 ≈ 1.2 M ops × 27 = 32 M ops/token
- (q_a_norm rmsnorm: already migrated in A1.1)

**Total: ~270 M scalar fp32 ops/token.** At 4 GF (single-threaded
scalar Rust) that's ~70 ms/token. On Metal at 4–6 TF that's < 1 ms
of compute, but with item ② pinning the buffers, dispatch overhead
is amortizable.

**The fix.** Reuse `gemv_f32_attn_metal` (already exists, attested
by G1.3). Add four new dispatcher methods on `DeepSeekV2`:
`q_a_proj_dispatch`, `q_b_proj_dispatch`, `kv_a_proj_dispatch`,
`kv_b_proj_dispatch`. Wire each call site in `attention(...)` to
the dispatcher under `cfg(target_os = "macos")` + Some(ctx) guard.
Identical pattern to A1.1–A1.5.

**Estimated impact.** ~70 ms/token → ~5 ms/token = **~1.2–1.5×
decode**. Multiplicative.

---

## ④ Persistent KV-cache buffers — **~1.1–1.2× decode**

**Where the time goes.** The KV cache (`crates/dismantle-core/src/cache.rs`)
lives as `Vec<f32>` on the host. The MHA decode step
(`mha_decode_step`) is CPU-only and reads/writes those vecs. As
seq_len grows, the per-token cost of reading the K/V history grows
linearly. For the test prompts (~10 tokens) this is small; for
long-context use it dominates.

**The fix.** Make `KvCache` Metal-resident: per-layer Metal Buffers
that grow on append. Add a `mha_decode_step_metal` kernel
(softmax-attention against the K/V buffers in place, no host copy).

**Estimated impact.** ~30 ms/token at seq_len=10, scaling linearly.
For Stage-1 perf gate (256-token decode), this gates the latter
half of the run.

---

## ⑤ Pre-faulted weight pages — **~5% on first prompt**

**Where the time goes.** First prompt's prefill includes faulting
in 9 GB of mmap'd GGUF weights. This is a one-time ~30 s cost that
amortizes across batches.

**The fix.** At model load, walk the gguf mmap with `madvise(WILLNEED)`
or read every nth byte to pre-fault. Costs ~30 s up front but moves
the cost out of the timed path.

**Estimated impact.** First-prompt latency drops; subsequent
prompts unaffected. Mostly UX, not throughput.

---

## ⑥ Concurrent command buffer submission — **possible, hard to measure**

**Where it sits.** All current kernels are committed with
`waitUntilCompleted` (synchronous). The Apple GPU pipeline can
overlap kernel execution with the next command buffer's encoding,
but only if we don't wait between commits. Pipelining requires
restructuring `dispatch_threads` to defer waits to the next
synchronization point (host readback).

**Estimated impact.** Best case ~10–20% on the dispatch-bound paths.
Worst case the latency-vs-throughput tradeoff makes single-prompt
runs slower. Defer until items ① and ② are landed and we have a
real perf budget to optimize.

---

## Cumulative estimate

If items ①, ②, ③ all land cleanly in Phase 2 Wedge 1:

```
post-haul-3 baseline:    ~0.13 dec_tps
× 20  (item ①)          ~2.6  dec_tps
× 1.4 (item ②)          ~3.6  dec_tps
× 1.3 (item ③)          ~4.7  dec_tps
```

Compared to llama.cpp Metal at ~30 dec_tps, that's
`r_llama ≈ 4.7 / 30 = 0.16` — **still below 1.5×**. The truth is
DeepSeek-V2-Lite Q4_K_M on the M3 Pro is already a tight perf
target; closing it requires more aggressive fusion, possibly a
graph-level rewrite of the forward path. Wedge 2 of Phase 2 needs
to scope what's left after Wedge 1 lands.

## Implementation order for Phase 2 Wedge 1 manifest

1. **Item ② (weight-pinning) first.** Smaller diff, no kernel
   changes, sets the foundation. ~1 day.
2. **Item ③ (MLA/Q-LoRA migration).** Trivial follow-up after ②;
   just dispatcher additions. ~half day.
3. **Item ① (FlashDMoE).** New shader + parity tests. ~3–5 days.
   This is the actual unlock and deserves its own halt budget.

Items ④–⑥ are Wedge 2 / Wedge 3.

## Cross-references

- Existing parity tests: `crates/dismantle-core/tests/phase1_kernel_parity.rs`
- Existing kernel dispatch: `crates/dismantle-core/src/kernels/mod.rs::metal_dispatch`
- Forward path: `crates/dismantle-core/src/model/deepseek_v2.rs::forward_token`
- MetalContext API: `crates/dismantle-core/src/metal/mod.rs`
- Haul-3 closeout (when written): `_phase1_haul3_attempt*_closeout.md`
