# Phase 3.1 Backend Seam — Per-Family Routing Checklist

> **ORCHESTRATOR CORRECTIONS (adversarial review, 2026-06-02) — read before executing:**
> 1. **Step 6 table labels are INVERTED.** Lines 4222/4235/4248 are the
>    *Fused (x += o_proj_out) + FFN norm* block (uses `ffn_norm_pin`,
>    `o_proj_out_buf`); lines 4527/4540/4553 are the *Fused (x += ffn_down) +
>    next-layer attn_norm* block (uses `next_norm`, `ffn_down_buf`). Line
>    numbers + kernels are correct; only the label strings are swapped.
> 2. **`forward_tokens_batch_tcb` range is 4849–5323** (the doc says 4847–5282;
>    5282 is an internal `commit_and_wait`, not the fn end).
> 3. **`BackendNorm` now carries `add_rmsnorm_q8_scaled`** (the AWQ smoothing
>    verb) — landed in `backend/mod.rs` (commit `728ab6d`); wire the
>    `awq_active` sites (4222/4527) through it.
> 4. **`backend/mod.rs` already exists** (3.1 trait defs SHIPPED `728ab6d`):
>    `Backend` base + 10 op-traits + `ComputeBackend` bundle + `Op` enum. This
>    pass adds the **`MetalBackend` impl** (`backend/metal.rs`) + the call-site
>    routing — NOT the trait defs. The wave's `seam-metal-impl` draft has a
>    **critical compile error** (`memcpy_f32_off_tcb` arg order: real sig is
>    `(tcb, src, dst, src_off, dst_off, n)`) and `add_inplace_broadcast` param
>    naming (`dim, batch`) to fix; the `scheduler-router` draft tried to create
>    its own `backend/mod.rs` (3-way conflict) — make its router a
>    `backend/router.rs` importing the existing seam. Reconcile both before
>    landing, one op-family at a time, golden-hash-gated after each.

> Orchestrator executes items 1–9 serially. STOP on any gate failure.
> All line numbers re-grepped at HEAD 2f47141 against
> `crates/dismantle-core/src/model/qwen_dense.rs` (5549 lines).
> `forward_token_greedy_tcb` starts at line **3437**.
> `gemv_proj!` macro definition is at lines **3849–3984**.
> Only call sites **inside** `forward_token_greedy_tcb` (lines 3437–4806) are
> listed for each family; batch-prefill (`forward_tokens_batch_tcb`, lines
> 4847–5282) is a separate pass after greedy is clean.

---

## Conventions

- **Edit shape**: `kernels::foo_tcb(&mut tcb, ...)  →  backend.foo(&mut rec, ...)`
  where `rec: &mut MetalRecorder<'_>` wraps `TokenCommandBuffer` and
  `backend: &MetalBackend` is injected into the function as a parameter
  (or `&self` once the model struct holds the backend).
- **DEFAULT-OFF gate**: the 3.1 backend trait is gated behind
  `DISMANTLE_BACKEND_SEAM=1`; every new trait dispatch path is wrapped in
  an `if use_seam { backend.foo(...) } else { kernels::foo_tcb(...) }` branch
  so the golden hash is unchanged when the flag is absent.
- **HARD GATE** after every family (run in this order):
  1. `cargo test --release -p dismantle-core --test integration_greedy_64`
     (pins `tests/golden/_phase0_token_baseline_64.hashes`; must match
     `v0.5.0-phase0`).
  2. `dismantle batch-hash --tokens 64` diff vs the fresh HEAD baseline
     captured before starting 3.1 (empty diff required; bit-identical).
  3. The family's own TCB parity test suite (listed per step below).

---

## Step 1 — BackendDevice + CommandRecorder lifecycle

**What moves**: Not a kernel dispatch — this step creates the two new files
(`backend/mod.rs`, `backend/metal.rs`) and wires `pub mod backend;` in
`lib.rs`. Establishes:
- `BackendDevice` trait with `type Recorder<'a>: Recorder<'a>` (GAT) +
  `fn begin_token(&self) -> Self::Recorder<'_>`.
- `Recorder<'a>` trait with `fn commit(self) -> Result<()>`.
- `MetalBackend(MetalContext)` concrete struct in `backend/metal.rs`
  (#[cfg(target_os="macos")]).
- `MetalRecorder<'ctx>(TokenCommandBuffer<'ctx>)` newtype;
  `Recorder::commit` calls `tcb.commit_and_wait()`.
- `BackendDevice::read_u32(buf: &Self::Buffer) -> u32` hiding the 4 raw
  `arena.token_buf.contents() as *const u32` readbacks at lines
  **4632, 4721, 4743, 4803**.

**Call sites modified in qwen_dense.rs** (0 kernel dispatches changed here):
- Line **3786**: `let mut tcb = TokenCommandBuffer::new(ctx);`
  → `let mut rec = backend.begin_token();`
- Lines **4630, 4719, 4741, 4797**: `tcb.commit_and_wait()?`
  → `rec.commit()?` (4 sites)
- Lines **4632, 4721, 4743, 4803**: raw `arena.token_buf.contents() as *const u32`
  → `backend.read_u32(&arena.token_buf)` (4 sites)

**Site count**: 9 lines touched, 0 kernel paths changed.

**HARD GATE** after step 1:
- `cargo test --release -p dismantle-core --test integration_greedy_64`
  (greedy_64_f32_regression must hold).
- `dismantle batch-hash --tokens 64` vs baseline (empty diff).
- `cargo test --release -p dismantle-core --test tcb_dispatch_cost`
  (smoke-checks TCB construction overhead; no regression).

---

## Step 2 — embed

**Kernel family**: `embed_lookup_metal_f32_tcb`

**Call sites in forward_token_greedy_tcb**:
| Line | Call |
|------|------|
| 3789 | `kernels::embed_lookup_metal_f32_tcb(&mut tcb, embed_buf, token, h, &arena.x_buf)?` |

**Site count**: 1

**Trait verb**: `BackendEmbed::embed(rec, embed_buf, token, hidden, x_buf)`

**Edit shape**:
```
// before
kernels::embed_lookup_metal_f32_tcb(&mut tcb, embed_buf, token, h, &arena.x_buf)?;
// after
backend.embed(&mut rec, embed_buf, token, h, &arena.x_buf)?;
```
Impl body in `backend/metal.rs` is a one-liner:
`kernels::embed_lookup_metal_f32_tcb(&mut rec.0, embed_buf, token, hidden, x_buf)`.
Kernel body in `kernels/mod.rs` is UNCHANGED.

**HARD GATE** after step 2:
- integration_greedy_64 + batch-hash baseline (as above).
- `cargo test --release -p dismantle-core --test v1e_gpu_argmax_parity`
  (tests `wedge_e_argmax_tcb_matches_cpu`,
  `wedge_e_gemv_f16_buf_tcb_matches_cpu`,
  `wedge_e_lmhead_plus_argmax_tcb_matches_cpu` — covers embed + argmax TCB
  paths through the kernel layer).

---

## Step 3 — rope

**Kernel family**: `rope_q_f32_inplace_tcb`

**Call sites in forward_token_greedy_tcb**:
| Line | Call |
|------|------|
| 4119 | `kernels::rope_q_f32_inplace_tcb(&mut tcb, &arena.q_buf, n_heads, head_dim, 0, head_dim, pos_u32, theta)?` |
| 4129 | `kernels::rope_q_f32_inplace_tcb(&mut tcb, &arena.k_token_buf, n_kv_heads, head_dim, 0, head_dim, pos_u32, theta)?` |

**Site count**: 2

**Trait verb**: `BackendRope::rope_inplace(rec, buf, n_heads, head_dim, nope_dim, rot_dim, pos, theta)`

**Edit shape**:
```
// before
kernels::rope_q_f32_inplace_tcb(&mut tcb, &arena.q_buf, n_heads, head_dim, 0, head_dim, pos_u32, theta)?;
// after
backend.rope_inplace(&mut rec, &arena.q_buf, n_heads, head_dim, 0, head_dim, pos_u32, theta)?;
```

**HARD GATE** after step 3:
- integration_greedy_64 + batch-hash baseline.
- `cargo test --release -p dismantle-core --test phase2_foundation_parity`
  (tests `rope_batch_matches_sequential`, `rope_batch_empty_is_noop` — RoPE
  correctness suite).

---

## Step 4 — elementwise add / silu_mul

**Kernel family**: `add_inplace_metal_tcb`, `silu_mul_tcb`

**Call sites in forward_token_greedy_tcb**:
| Line | Call |
|------|------|
| 4107 | `kernels::add_inplace_metal_tcb(&mut tcb, &arena.q_buf, qb, q_dim)?` (q bias; conditional) |
| 4110 | `kernels::add_inplace_metal_tcb(&mut tcb, &arena.k_token_buf, kb, kv_dim)?` (k bias; conditional) |
| 4113 | `kernels::add_inplace_metal_tcb(&mut tcb, &arena.v_token_buf, vb, kv_dim)?` (v bias; conditional) |
| 4348 | `kernels::silu_mul_tcb(&mut tcb, &arena.ffn_gate_buf, &arena.ffn_up_buf, &arena.ffn_act_buf, intermediate)?` |

**Site count**: 4 (3 add_inplace + 1 silu_mul; all 3 add_inplace are inside
`if let Some(qb/kb/vb)` guards — still single dispatch sites each).

**Trait verbs**:
- `BackendElementwise::add_inplace(rec, dst, src, n)`
- `BackendElementwise::silu_mul(rec, gate, up, out, n)`

**Edit shape (add_inplace example)**:
```
// before
kernels::add_inplace_metal_tcb(&mut tcb, &arena.q_buf, qb, q_dim)?;
// after
backend.add_inplace(&mut rec, &arena.q_buf, qb, q_dim)?;
```

**HARD GATE** after step 4:
- integration_greedy_64 + batch-hash baseline.
- `cargo test --release -p dismantle-core --test v1_tcb_rmsnorm_add_parity`
  (tests `tcb_add_inplace_matches_cpu`, `tcb_staggered_loop_matches_cpu`).

---

## Step 5 — kv_append / memcpy

**Kernel family**: `memcpy_f32_off_tcb` (used for KV slot append and Eagle5
capture; the KV-cache write is the load-bearing path)

**Call sites in forward_token_greedy_tcb**:
| Line | Call | Purpose |
|------|------|---------|
| 4143 | `kernels::memcpy_f32_off_tcb(&mut tcb, &arena.k_token_buf, &arena.k_cache_buf, 0, slot_kv_off_elems, kv_dim)?` | K cache append |
| 4151 | `kernels::memcpy_f32_off_tcb(&mut tcb, &arena.v_token_buf, &arena.v_cache_buf, 0, slot_kv_off_elems, kv_dim)?` | V cache append |
| 4587 | `kernels::memcpy_f32_off_tcb(&mut tcb, res_src, res_buf, 0, 0, h)?` | Eagle5 residual capture (conditional) |
| 4588 | `kernels::memcpy_f32_off_tcb(&mut tcb, &arena.ffn_down_buf, int_buf, 0, 0, h)?` | Eagle5 intermediate capture (conditional) |

**Site count**: 4 (2 unconditional KV-append + 2 inside
`if eagle5_capture_active && li == self.eagle5_capture_layer` guard).

**Trait verb**: `BackendKvCache::memcpy_f32_off(rec, src, dst, src_off, dst_off, n)`

**Edit shape**:
```
// before
kernels::memcpy_f32_off_tcb(&mut tcb, &arena.k_token_buf, &arena.k_cache_buf, 0, slot_kv_off_elems, kv_dim)?;
// after
backend.memcpy_f32_off(&mut rec, &arena.k_token_buf, &arena.k_cache_buf, 0, slot_kv_off_elems, kv_dim)?;
```

**HARD GATE** after step 5:
- integration_greedy_64 + batch-hash baseline.
- `cargo test --release -p dismantle-core --test q8_kv_parity`
  (KV-append and KV-read parity vs CPU reference).

---

## Step 6 — rmsnorm (+ fused q8)

**Kernel family**: `rmsnorm_metal_buf_tcb`, `add_rmsnorm_fused_tcb`,
`add_rmsnorm_fused_q8_tcb`, `add_rmsnorm_fused_q8_scaled_tcb`

**Call sites in forward_token_greedy_tcb**:
| Line | Kernel | Context |
|------|--------|---------|
| 3803 | `rmsnorm_metal_buf_tcb` | Layer-0 pre-norm (hoisted) |
| 4222 | `add_rmsnorm_fused_q8_scaled_tcb` | Post-attn fused norm (AWQ+W4A8) |
| 4235 | `add_rmsnorm_fused_q8_tcb` | Post-attn fused norm (W4A8 no-AWQ) |
| 4248 | `add_rmsnorm_fused_tcb` | Post-attn fused norm (baseline) |
| 4527 | `add_rmsnorm_fused_q8_scaled_tcb` | Post-FFN fused norm (AWQ+W4A8) |
| 4540 | `add_rmsnorm_fused_q8_tcb` | Post-FFN fused norm (W4A8 no-AWQ) |
| 4553 | `add_rmsnorm_fused_tcb` | Post-FFN fused norm (baseline) |

**Site count**: 7 (1 standalone rmsnorm + 6 fused-add-rmsnorm variants;
the 6 fused sites come in two groups of 3, each gated on AWQ/W4A8 flags)

**Trait verbs**:
- `BackendNorm::rmsnorm(rec, x, weight, eps, hidden, out)` — 1 site (line 3803)
- `BackendNorm::add_rmsnorm_fused(rec, x, delta, weight, x_norm, eps, hidden)` — 2 sites (4248, 4553)
- `BackendNorm::add_rmsnorm_fused_q8(rec, x, delta, weight, x_norm, i8_out, scales_out, eps, hidden)` — 2 sites (4235, 4540)
- `BackendNorm::add_rmsnorm_fused_q8_scaled(rec, x, delta, weight, x_norm, i8_out, scales_out, smooth, eps, hidden)` — 2 sites (4222, 4527)

**Edit shape (baseline fused example)**:
```
// before
kernels::add_rmsnorm_fused_tcb(&mut tcb, &arena.x_buf, &arena.o_proj_out_buf, ffn_norm_pin, &arena.x_norm_buf, eps, h)?;
// after
backend.add_rmsnorm_fused(&mut rec, &arena.x_buf, &arena.o_proj_out_buf, ffn_norm_pin, &arena.x_norm_buf, eps, h)?;
```
Note: the fused cross-op kernels (`add_rmsnorm_fused`, `add_rmsnorm_fused_q8`,
`add_rmsnorm_fused_q8_scaled`) surface as **FUSED** verbs in the trait, not
decomposed — decomposition would lose fusion and change the hash.

**HARD GATE** after step 6:
- integration_greedy_64 + batch-hash baseline.
- `cargo test --release -p dismantle-core --test v1_tcb_rmsnorm_add_parity`
  (`tcb_rmsnorm_matches_cpu`, `tcb_staggered_loop_matches_cpu`).
- `cargo test --release -p dismantle-core --test add_rmsnorm_fused_q8_parity`
  (`add_rmsnorm_fused_q8_parity_hidden_256`,
  `add_rmsnorm_fused_q8_parity_hidden_2048`,
  `add_rmsnorm_fused_q8_parity_hidden_2048_alt_seed`).
- `cargo test --release -p dismantle-core --test add_rmsnorm_fused_q8_scaled_parity`.

---

## Step 7 — quantize (W4A8)

**Kernel family**: `quantize_f32_to_int8_per_block_tcb`,
`quantize_f32_to_int8_per_block_scaled_tcb`,
`quantize_f32_to_int8_per_channel_tcb`

**Call sites in forward_token_greedy_tcb**:
| Line | Kernel | Context |
|------|--------|---------|
| 3815 | `quantize_f32_to_int8_per_block_scaled_tcb` | Pre-loop x_norm quantize (AWQ) |
| 3824 | `quantize_f32_to_int8_per_block_tcb` | Pre-loop x_norm quantize (no AWQ) |
| 4181 | `quantize_f32_to_int8_per_block_scaled_tcb` | Post-attn-out quantize (AWQ) |
| 4190 | `quantize_f32_to_int8_per_block_tcb` | Post-attn-out quantize (no AWQ) |
| 4361 | `quantize_f32_to_int8_per_block_scaled_tcb` | Post-silu-mul quantize (AWQ) |
| 4370 | `quantize_f32_to_int8_per_block_tcb` | Post-silu-mul quantize (no AWQ) |
| 4603 | `quantize_f32_to_int8_per_channel_tcb` | LM-head per-channel quantize (Track E) |

**Site count**: 7 (6 per-block + 1 per-channel; all inside W4A8-flag guards)

**Trait verbs**:
- `BackendQuant::quantize_per_block(rec, x, i8_out, scales_out, n)`
- `BackendQuant::quantize_per_block_scaled(rec, x, smooth, i8_out, scales_out, n)`
- `BackendQuant::quantize_per_channel(rec, x, channel_scales, i8_out, n)`

**Edit shape**:
```
// before
kernels::quantize_f32_to_int8_per_block_tcb(&mut tcb, &arena.x_norm_buf, x_int8, x_scales, h)?;
// after
backend.quantize_per_block(&mut rec, &arena.x_norm_buf, x_int8, x_scales, h)?;
```

**HARD GATE** after step 7:
- integration_greedy_64 + batch-hash baseline.
- `cargo test --release -p dismantle-core --test quantize_int8_kernel_parity`
  (`quantize_int8_kernel_matches_cpu_small`,
  `quantize_int8_kernel_matches_cpu_hidden_2048`,
  `quantize_int8_kernel_matches_cpu_intermediate_11008`,
  `quantize_int8_kernel_handles_all_zero_block`).
- `cargo test --release -p dismantle-core --test w4a8_per_channel_parity`.

---

## Step 8 — attention (mha_decode, wrap as-is)

**Kernel family**: `mha_decode_f32_tcb`

**Call sites in forward_token_greedy_tcb**:
| Line | Call |
|------|------|
| 4162 | `kernels::mha_decode_f32_tcb(&mut tcb, &arena.q_buf, &arena.k_cache_buf, layer_kv_off_bytes, &arena.v_cache_buf, layer_kv_off_bytes, &arena.attn_out_buf, mha_seq_len, head_dim, n_heads, n_kv_heads)?` |

**Site count**: 1

**Trait verb**: `BackendAttention::mha_decode(rec, q, k_cache, k_off, v_cache, v_off, attn_out, seq_len, head_dim, n_heads, n_kv_heads)`

**Edit shape**:
```
// before
kernels::mha_decode_f32_tcb(&mut tcb, &arena.q_buf, &arena.k_cache_buf, layer_kv_off_bytes,
    &arena.v_cache_buf, layer_kv_off_bytes, &arena.attn_out_buf, mha_seq_len, head_dim, n_heads, n_kv_heads)?;
// after
backend.mha_decode(&mut rec, &arena.q_buf, &arena.k_cache_buf, layer_kv_off_bytes,
    &arena.v_cache_buf, layer_kv_off_bytes, &arena.attn_out_buf, mha_seq_len, head_dim, n_heads, n_kv_heads)?;
```
The `mha_decode_f32_tcb` kernel body is UNCHANGED (wrap-as-is per scout spec).

**HARD GATE** after step 8:
- integration_greedy_64 + batch-hash baseline.
- `cargo test --release -p dismantle-core --test mha_decode_metal_parity`
  (`mha_decode_metal_matches_cpu`, `mha_decode_metal_seq_len_one`).
- `cargo test --release -p dismantle-core --test v1c_tcb_attn_ffn_parity`
  (`wedge_c_pair_gemv_norm_matches_cpu`,
  `wedge_c_full_layer_loop_matches_cpu`).

---

## Step 9 — gemv (LAST)

**Kernel family**: all GEMV/GEMM dispatch variants inside the `gemv_proj!`
macro (lines 3849–3984) plus the 5 direct ladder sites below the macro for
ffn_down (lines 4387–4499) and the LM-head ladder (lines 4616–4793).

**gemv_proj! macro is invoked at** (6 direct call sites in forward_token_greedy_tcb):
| Line | Site | Weight |
|------|------|--------|
| 4001 | `gemv_proj!(w4a8_qproj, layer.q_proj, ...)` | q_proj |
| 4078 | `gemv_proj!(w4a8_qproj, layer.k_proj, ...)` | k_proj (fallback when !kv_fuse) |
| 4089 | `gemv_proj!(false, layer.v_proj, ...)` | v_proj (fallback when !kv_fuse) |
| 4199 | `gemv_proj!(w4a8_oproj, layer.o_proj, ...)` | o_proj |
| 4325 | `gemv_proj!(w4a8_ffn_gate, layer.ffn_gate, ...)` | FFN gate (fallback when !fuse) |
| 4336 | `gemv_proj!(w4a8_ffn_up, layer.ffn_up, ...)` | FFN up (fallback when !fuse) |

**Fused-pair direct calls** (not via macro; 4 sites):
| Line | Kernel |
|------|--------|
| 4038 | `kernels::gemv_q4_k_v4_predec_pair_f16s_pinned_tcb` (k+v fused, f16-scales) |
| 4059 | `kernels::gemv_q4_k_v4_predec_pair_pinned_tcb` (k+v fused, f32-scales) |
| 4285 | `kernels::gemv_q4_k_v4_predec_pair_f16s_pinned_tcb` (gate+up fused, f16-scales) |
| 4306 | `kernels::gemv_q4_k_v4_predec_pair_pinned_tcb` (gate+up fused, f32-scales) |

**ffn_down dispatch ladder** (direct calls, 5 dispatch sites; lines 4387–4499):
`gemm_q4_k_a8_v3_8r_pinned_tcb` (W4A8), `gemv_q4_k_v4_predec_2r_f16s_pinned_tcb`
(f16-scales predec), `gemv_q4_k_v4_predec_pinned_tcb` (f32-scales predec),
`gemv_q4_k_m_v3_8r_pinned_tcb` (fallback Q4_K), `gemv_q6_k_pinned_tcb`
(native Q6_K), `gemv_f16_metal_buf_tcb` (f16 fallback).

**LM-head dispatch ladder** (6 sites; lines 4616–4793; 3 branches ×
vocab-pruned / lm_head_q4k / f16 full):
`gemm_q4_k_a8_v3_8r_per_channel_pinned_tcb`, `gemm_q4_k_a8_v3_8r_pinned_tcb`,
`gemv_q4_k_v4_predec_2r_f16s_pinned_tcb`, `gemv_q4_k_v4_predec_pinned_tcb`,
`gemv_q4_k_m_v3_8r_pinned_tcb`, `gemv_f16_metal_buf_tcb`.

**Total gemv/gemm dispatch sites in forward_token_greedy_tcb**: ~28 unique
call sites (inside 6 `gemv_proj!` macro invocations + 4 fused-pair direct +
~10 ffn_down ladder + ~8 LM-head ladder + 4 LM-head argmax paths).

**Trait verb**: `BackendGemv::gemv(rec, spec: &GemvSpec)` where `GemvSpec`
is an enum bundling weight kind + buffer refs + dims. The **variant explosion
(~25 kernel variants) collapses to ONE verb** — the impl body replicates the
existing ladder inside `gemv_proj!` using a `match spec.weight_kind { ... }`.
The macro body moves into `MetalBackend::gemv` impl; the macro call sites
become `backend.gemv(&mut rec, &GemvSpec::Q4K { ... })`.

**Edit shape (q_proj site)**:
```
// before (inside per-layer loop)
gemv_proj!(
    w4a8_qproj,
    layer.q_proj,
    layer.pinned.q_proj_f16.as_ref(),
    q_dim, h,
    &arena.x_norm_buf, x_int8, x_scales,
    &arena.q_buf
);
// after
backend.gemv(&mut rec, &GemvSpec {
    w4a8: w4a8_qproj,
    tref: &layer.q_proj,
    f16_fallback: layer.pinned.q_proj_f16.as_ref(),
    rows: q_dim, cols: h,
    x: &arena.x_norm_buf,
    x_i8: x_int8, x_sc: x_scales,
    out: &arena.q_buf,
    predec_f32: predec_cache_ref,
    predec_f16: predec_cache_f16_ref,
    q4k_fast: q4k_fast_ref,
    mmap: mmap_buf,
})?;
```
Kernel bodies in `kernels/mod.rs` are UNCHANGED; the impl body in
`backend/metal.rs` is the moved ladder.

**HARD GATE** after step 9 (the final gate for 3.1):
- integration_greedy_64 + batch-hash baseline (both MUST be bit-identical).
- `cargo test --release -p dismantle-core --test v100_pinned_q4kgemv_parity`
  (`pinned_q4kgemv_small`, `pinned_q4kgemv_realistic`,
  `pinned_q4kgemv_nonzero_offset`).
- `cargo test --release -p dismantle-core --test q4k_predec_f16s_parity`
  (`q4k_v4_predec_f16s_relative_parity`).
- `cargo test --release -p dismantle-core --test v1_1_phase5B1_lm_head_tcb_parity`.
- `cargo test --release -p dismantle-core --test w4a8_qwen3b_quality_gate`.
- **Parity floors confirmed**: kernel atol ≤ 1e-3 fp16; first-3 greedy
  token IDs identical; b3sum via `dismantle batch-hash --tokens 64` empty
  diff vs fresh HEAD baseline.
- **Residual stream invariant**: `x_buf` allocations in `DenseDecodeArena`
  (`dense_decode_arena.rs` line 116) remain f32 (hidden × sizeof(f32)).
  Never f16 the accumulator. Verify: `DenseDecodeArena::x_buf` size is
  `hidden * 4` bytes in the impl.
- **TCB batching invariant**: `MetalRecorder` MUST NOT call
  `commit_and_wait` inside any `BackendGemv::gemv` impl — commit only at
  `Recorder::commit()` (end of token). Verify: no `commit_and_wait` inside
  `MetalBackend::gemv` in `backend/metal.rs`.

---

## Summary Table

| Step | Family | Sites in fwd_token_greedy_tcb | After gate |
|------|--------|-------------------------------|------------|
| 1 | BackendDevice + Recorder lifecycle | 9 (0 kernel, 4 commit, 4 readback, 1 new) | greedy_64 + tcb_dispatch_cost |
| 2 | embed | 1 | greedy_64 + v1e_gpu_argmax_parity |
| 3 | rope | 2 | greedy_64 + phase2_foundation_parity |
| 4 | add_inplace / silu_mul | 4 | greedy_64 + v1_tcb_rmsnorm_add_parity |
| 5 | kv_append / memcpy | 4 | greedy_64 + q8_kv_parity |
| 6 | rmsnorm + fused-add-norm | 7 | greedy_64 + add_rmsnorm_fused_q8_parity + add_rmsnorm_fused_q8_scaled_parity |
| 7 | quantize W4A8 | 7 | greedy_64 + quantize_int8_kernel_parity + w4a8_per_channel_parity |
| 8 | mha_decode | 1 | greedy_64 + mha_decode_metal_parity + v1c_tcb_attn_ffn_parity |
| 9 | gemv (all variants + macro) | ~28 | greedy_64 + v100_pinned_q4kgemv_parity + q4k_predec_f16s_parity + v1_1_phase5B1_lm_head_tcb_parity + w4a8_qwen3b_quality_gate |

Total direct kernel dispatch sites routed: **~64** across 9 families in
`forward_token_greedy_tcb`. Excludes `forward_tokens_batch_tcb` (batch-prefill,
lines 4847–5282) which is a Phase 3.1 follow-on pass after greedy is clean.
