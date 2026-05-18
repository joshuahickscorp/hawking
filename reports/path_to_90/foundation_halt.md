# Path-to-90 foundation block — halt at step 9

**Halted at:** 2026-05-18 EDT
**Halted on:** step 9 (bit-identical greedy regression)
**Halt rule:** CLAUDE.md § Halt rule — "Below a gate's lower bound → halt and debug, do not paper over."

## What ran in the architectural batch

| Step | Status | Commit |
|---|---|---|
| 1 — Stage 0 baseline profile | ✅ | `72e3926` |
| 2 — Stage 0.5 MLX decision (mandatory, deferred to after step 10) | ✅ | `48be7a1` |
| 3 — `Engine::forward_token_eagle4_for_test` capture | ✅ | `711893c` |
| 4 — `eagle4.py eval --dump-logits` | ✅ | `64cb5c4` |
| 5 — `Eagle4Head::forward_full` CPU forward | ✅ | `540f9d8` |
| 6 — eagle4 parity test (Rust vs Python at 1e-5) | ✅ | `6411d21` |
| 7 — Metal-accelerated head forward | deferred (stretch goal; see step-8-commit-message reasoning) | — |
| 8 — `--speculate eagle4` CLI + decode-path wire-up | ✅ ships output | `f3ae7fd` |
| **9 — bit-identical greedy regression** | **❌ halt** | this commit (test landed, currently fails) |

## Root cause

`forward_token_eagle4_capture_with_argmax` (the CPU walk added in step 8)
diverges from `forward_token` (production GPU Wedge C) at *argmax* level
on the very first decode step. Not subtle fp drift — entirely different
tokens.

Smoke run (prompt `"The quick brown fox"`, 8 tokens, greedy):

```
Off:    33747, 855,   254,   24547, 5025,  5025,  5025,  5025
Eagle4: 257,   9442,  78887, 71199, 21700, 28542, 61122, 96978
```

The Off-mode and Eagle4-mode forwards share architectural intent (full
V2-Lite forward through 27 layers + final-norm + LM-head + argmax) but
exercise disjoint code paths:

- **Off mode (Wedge C):** GPU Metal kernels end-to-end. KV cache lives
  in `mla_c_kv_gpu[li]` Metal buffers, written by `kv_append_f32_tcb`
  Metal kernel.
- **Eagle4 mode (CPU walk):** dismantle's CPU `attention()` + `ffn()`
  helpers per layer. These DO dispatch through Metal where available
  (gemv_q4_k_m_v2_pinned, mla_decode_and_o_proj_metal, etc.) BUT also
  do CPU-side bookkeeping (mla_kv_append into the CPU `mla_c_kv[li]`
  Vec, separate from the GPU buffer).

The CPU KV mirror and GPU KV buffer are populated by different write
paths and *do not stay in sync*. Specifically:

1. **GPU prefill leaves CPU KV at zeros.** The Wedge C path
   one-way-syncs `mla_c_kv → mla_c_kv_gpu` at first call (line ~2879
   of `deepseek_v2.rs`), never the reverse. After GPU-prefill an
   eagle4 decode step calls `attention()` which reads CPU
   `mla_c_kv[li][0..prompt_len]` — these are all zeros.

2. **CPU prefill (one attempted fix this session) didn't resolve it.**
   Running the CPU walk for prefill DOES populate CPU KV, but the CPU
   walk's first decode-step argmax (`257`) still diverged from GPU's
   (`33747`). So the issue is not (only) "CPU KV is unpopulated" — the
   CPU walk's V2-Lite forward produces a fundamentally different
   answer from GPU Wedge C even when both have populated prior KV in
   their respective mirrors. The divergence is structural, not a sync
   bug.

The most likely structural causes (in priority order):

- **CPU `attention()` uses a different MLA dispatch.** `attention()`
  at line 3546 dispatches `gemv_f32_attn_pair_dispatch` for q/kv
  projections and `mla_decode_and_o_proj_metal` for the attention math
  when Metal is available — but the path differs from Wedge C's
  `encode_attention_phase1_into_tcb` + `dispatch_mla_decode_and_o_proj`
  in subtle ways (e.g. RoPE position application, scale factor,
  how `q_full` is shaped before MLA decode). A side-by-side audit of
  the two MLA code paths is needed.

- **`ffn()` MoE-fused path may use a different fused kernel than
  Wedge C's `encode_moe_block_batched_indexed_tcb_with_scratch`.**
  `moe_block_batched_dispatch` (line ~2144) is the unfused fallback
  inside `ffn()`; the fused path goes through a different dispatcher.
  Comparing the dispatcher selection for CPU-call-site vs
  Wedge-C-call-site would clarify.

- **The latent zero-output bug in `ffn_shared_only`** (surfaced in
  commit `711893c`) suggests the unfused MoE-shared dispatch is
  broken; possibly `ffn()`'s shared-expert leg inherits the same
  bug when called outside the Wedge C single-TCB context. Pursued in
  the spawned chip but not yet fixed.

## What attended work unblocks

In order of likely payoff:

1. **Compare `forward_token` and `forward_token_eagle4_capture_with_argmax`
   on the same input at pos=0 (no prior KV), no prefill.** Dump x at
   each layer boundary for both paths. The first layer where they
   diverge identifies the kernel(s) responsible. Concrete: add a
   `DISMANTLE_LAYER_TRACE=1` env-var path that prints L2 norms per
   layer for both methods on the same input; diff the traces.

2. **Audit CPU `attention()` vs Wedge C attention.** Specifically
   `dispatch_mla_decode_and_o_proj` vs `mla_decode_and_o_proj_metal`
   — these are different kernels (one TCB-batched, one standalone)
   and may produce different results at the same input.

3. **Fix the `ffn_shared_only` zero-output bug** (already chipped out
   as a follow-up — see step-3 commit message `711893c`). If `ffn()`'s
   shared-expert leg also returns zero in the unfused path, the CPU
   walk's `ffn()` output is missing shared-expert contributions at
   every MoE layer.

4. **Once divergence is fixed, re-run `eagle4_decode_parity`:**

   ```bash
   EAGLE4_PARITY_TEST=1 cargo test -p dismantle-core --release \
       --test eagle4_decode_parity -- --ignored --nocapture
   ```

   Expected: zero token-level mismatches between Off and Eagle4 over 16
   tokens.

## What's parked vs. what's blocked

**Parked (Class A; pursue after divergence is fixed):**

- Step 10 — Stage 1 measurement gate (12–22 tok/s expected). Cannot
  run honestly until Eagle4 mode produces bit-identical output.
- K-batched verify infrastructure (verify_window > 1; longest-
  matching-prefix acceptance). Step 8 ships K=1 verify-by-comparison,
  which is degenerate w.r.t. K; meaningful K>1 testing needs Path B
  kernels (Stage 2 / execution_plan.md steps 12–17).

**Class B (independent of this halt):**

- Stage 0.5 MLX-pattern adoption (step 2's mandatory verdict). Can
  begin in parallel — it doesn't depend on Eagle4 spec decode landing.
- Routing recall fine-tune (step 11). eagle4-side Python work, runs
  off-machine.

**Architectural batch that DID land successfully:**

- The complete eagle4 head infrastructure: NPZ loader, frozen weights
  loader, CPU forward (1e-5 parity vs Python), parity test, CLI flag,
  decode-loop wire-up. The head IS callable and the parity test
  proves the forward is correct. The unresolved divergence is on the
  V2-Lite-side CPU-vs-GPU path, not the eagle4 head itself.

## Tone

Per CLAUDE.md § "On disagreement with this contract": this halt
follows the halt rule literally. The fix is structural (CPU/GPU MLA
divergence audit, possibly tied to the ffn_shared_only chipped fix)
and belongs in a focused attended session, not a "just this once"
exception in the current commit.
