# Phase 3.3 CPU MoE Scope — Off-macOS Gap Analysis

## Summary

The dense CPU path (`force_cpu`, qwen0.5b 12/12) is done. The remaining gap is that both `deepseek_v2` and `mixtral` return errors on non-macOS. This note scans each blocker exactly, then assesses feasibility for a single-pass implementation.

---

## Model A: DeepSeek-V2 (`deepseek_v2.rs`)

### The MoE FFN Path Is Already CPU-Capable

The fallback path in `ffn()` (lines 3785–3815) already works off-macOS:

1. `moe_block_batched_dispatch()` at line 2206 returns `Ok(None)` unconditionally when `cfg(not(target_os = "macos"))`. Control falls through.
2. The per-expert loop calls `moe_expert_pair_matmul_dispatch()` then `moe_expert_matmul_dispatch()`. Both have CPU fallbacks: a `#[cfg(target_os = "macos")]` block that exits early via Metal, and then the unconditional `self.dequant_ref_into(t, scratch)?; gemv_f32(...)` at lines 2113–2114. `dequant_ref_into` slices directly from `self.gguf.mmap` (cross-platform `memmap2::Mmap`), then calls `quant::dequant_into` which has a pure-Rust path for Q4_K, Q8_0, Q3_K, Q6_K, and all the dtypes that appear in V2-Lite.
3. `gemv_f32_moe_dispatch()` (gate logits, line 1946) also has a CPU fallback: `gemv_f32(w, rows, cols, x, out)`.
4. `moe_expert_pair_matmul_dispatch()` CPU branch (lines 1992–1996): allocates gate/up temporaries, calls `moe_expert_matmul_dispatch` twice, then `silu_mul`. All CPU.
5. The shared-expert branch at lines 3800–3815 follows the same dispatch pattern and also lands on the CPU path.

**Conclusion: the FFN is not a blocker.**

### The Real Blocker: MLA Attention Hard-Errors Off-macOS

`attention()` line 3466 checks `if !self.mla_c_kv.is_empty()`. This is true by default because `mla_metal` defaults to `true` (line 650: `.unwrap_or(true)`), causing `mla_c_kv` to be allocated at load time with `n_layers` entries.

When `mla_c_kv` is non-empty off-macOS, the code enters the branch at line 3466, appends the compressed KV, then at line 3489 tries the Metal path — which is `#[cfg(target_os = "macos")]` only. The non-macOS branch falls through to line 3628:

```rust
return Err(Error::Model(
    "mla_decode: Metal context unavailable on this platform".into(),
));
```

This hard-errors before any MoE FFN code runs.

**Fix required:** Either (a) suppress `mla_metal = true` off-macOS so `mla_c_kv` stays empty, or (b) implement a CPU MLA decode path after line 3628.

#### Option (a): Suppress mla_metal Off-macOS

At line 646–650:
```rust
let mla_metal = config
    .kernel_profile
    .as_ref()
    .map(|p| p.selected.mla_schedule.as_str() == "metal-mla")
    .unwrap_or(true);
```

Change to:
```rust
let mla_metal = {
    #[cfg(target_os = "macos")]
    { config.kernel_profile.as_ref()
        .map(|p| p.selected.mla_schedule.as_str() == "metal-mla")
        .unwrap_or(true) }
    #[cfg(not(target_os = "macos"))]
    { false }
};
```

When `mla_c_kv` is empty, `attention()` bypasses the MLA branch entirely and falls through to the "Reconstruct full K/V via kv_b_proj" path at line 3633. That path calls `gemv_f32_attn_dispatch` (which has a CPU fallback), `mha_decode_step` (pure-Rust CPU in `crate::attn`), and `gemv_f32_attn_dispatch` again for o_proj. **All CPU-capable.**

This is the correct fix. It is a 4-line surgical change to one match arm in `DeepSeekV2::load`.

#### Remaining deepseek_v2 sub-dependencies to audit after that fix:

- `embed_lookup` — pure-Rust, no Metal gate.
- `rmsnorm_dispatch` — CPU fallback exists (line 1857: `rmsnorm(x, weight, eps, out)`).
- `gemv_f16_dispatch` — CPU fallback exists (line 1882: `crate::kernels::gemv_f16(w_f16, ...)`).
- `gemv_f32_attn_dispatch` — CPU fallback exists (line 1911: `gemv_f32(w, rows, cols, x, out)`).
- `mha_decode_step` — pure Rust in `crate::attn`, no Metal dependency.
- `forward_token_final_norm_maybe_read` — the Wedge C TCB path is `#[cfg(target_os = "macos")]`; the non-TCB body falls through to the `forward_token_final_norm` call which invokes `attention() + ffn()` per layer plus a final rmsnorm + gemv_f16. All dispatchers have CPU fallbacks.
- `forward_token_greedy` — at line 2574; its Wedge C fast path is macOS-gated. The CPU-path body needs audit to confirm it calls `forward_token` (which uses the dispatchers above). See line 2574 context.

A `#[cfg(not(target_os = "macos"))]` block exists at line 2206 for `moe_block_batched_dispatch` — already handled.

**Total new code for deepseek_v2 CPU MoE:** 4-line `mla_metal` fix. No new FFN code needed.

---

## Model B: Mixtral (`mixtral.rs`)

### Forward Token Is Fully Gated — No CPU Path Exists

`forward_token()` at line 649 is:

```rust
fn forward_token(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
    #[cfg(target_os = "macos")]
    { self.forward_token_tcb(token, pos) }
    #[cfg(not(target_os = "macos"))]
    { let _ = (token, pos);
      Err(Error::Unimplemented("MixtralEngine::forward_token requires the Metal TCB path")) }
}
```

The only compute path is `forward_token_tcb` which is `#[cfg(target_os = "macos")]`. There is no CPU forward-pass body at all.

### What a CPU MoE Path for Mixtral Requires

Mixtral uses a **standard MHA** (not MLA), split per-expert tensors (`blk.N.ffn_gate.E.weight`), and a gate-logits projection (`ffn_gate_inp`) stored as F16. Everything needed for a CPU forward pass is already in memory at load time:

- `attn_norm`, `ffn_norm` — dequanted to f32 at load (no lazy).
- `embed` — f16, accessed via `embed_lookup`.
- `final_norm` — f32.
- `lm_head` — f16.
- `layers[li].attn_q`, `attn_output`, `ffn_gate[eid]`, `ffn_up[eid]`, `ffn_down[eid]` — `TensorRef` pointing into mmap. **These carry `rows`/`cols` directly** (unlike deepseek_v2 TensorRef which has no shape).
- The K/V projections are in `pinned.attn_k`, `pinned.attn_v` (uploaded as f32 to Metal at load) — but the raw dequanted f32 is not retained. The GGUF mmap bytes for `attn_k.weight` and `attn_v.weight` are still accessible at load time via `gguf.mmap`.

### Blockers Specific to Mixtral CPU Path

**Blocker 1: K/V projection weights not retained as f32 off-macOS.**

In `load()` (line 400–431), `attn_k` and `attn_v` are dequanted to f32 only to upload them to `pinned.attn_k / pinned.attn_v`. The f32 Vec is dropped. Off-macOS, `pinned = MixtralLayerPinned::default()` (all `None`), so there is no way to run the K/V GEMV. The raw TensorRef is available for `attn_q` and `attn_output` (they are stored as TensorRef in the layer), but `attn_k.weight` and `attn_v.weight` are NOT stored as TensorRef — they are only uploaded.

Fix: Add `attn_k_ref: TensorRef` and `attn_v_ref: TensorRef` to `MixtralLayer`, populated during load from `tensor_ref_expected` regardless of platform. Then the CPU path can `dequant_into` on demand.

**Blocker 2: gate_inp not retained off-macOS.**

Similarly, `ffn_gate_inp.weight` (F16, n_experts × hidden) is only uploaded to `pinned.gate_inp` and the f32 Vec is dropped. Off-macOS, no gate logits can be computed.

Fix: Add `gate_inp_ref: TensorRef` to `MixtralLayer`, or store the gate matrix as `Vec<f32>` directly (it is small: 8 × 4096 = 32K f32s = 128 KiB per layer).

**Blocker 3: `MixtralDecodeArena` is `#[cfg(target_os = "macos")]` only.**

The non-macOS struct is `pub struct MixtralDecodeArena;` (unit struct, line 211). The CPU forward pass does not need Metal arena buffers — it can use stack allocations. This is not a blocker per se; the CPU path can simply not use the arena.

**Blocker 4: `attn_q`'s TensorRef carries Mixtral-local chunk fields.**

Mixtral's local `TensorRef` (line 116) adds `rows`, `cols`, `chunk_index`, `chunk_offset`. This is different from the shared `weights::TensorRef` used by deepseek_v2. The `dequant_into` call needs access to the raw bytes from `gguf.mmap`. The CPU path can use `&gguf.mmap[t.offset..t.offset + t.byte_size]` directly, bypassing the chunk machinery (chunks are only needed for the GPU no-copy Metal buffers). This is not a blocker; the offset/byte_size fields are always set.

### Scope Estimate for Mixtral CPU Path

A minimal Mixtral CPU `forward_token` is a ~120-line new function `forward_token_cpu` following the same pattern as `qwen_dense.rs::forward_token`:

1. embed_lookup (f16 embed, same API)
2. Per-layer loop:
   a. rmsnorm (cpu fallback already in `rmsnorm_dispatch`-equivalent)
   b. Q projection: dequant `attn_q` + gemv_f32
   c. K projection: dequant `attn_k` (via new TensorRef) + gemv_f32 + rope
   d. V projection: dequant `attn_v` (via new TensorRef) + gemv_f32
   e. rope on Q
   f. KV cache append + `mha_decode_step`
   g. O projection: dequant `attn_output` + gemv_f32
   h. add_inplace (residual)
   i. rmsnorm
   j. gate logits: dequant `gate_inp` (via new TensorRef or cached vec) + gemv_f32
   k. topk_gate
   l. Per active expert: dequant ffn_gate[eid] + ffn_up[eid] → silu_mul → dequant ffn_down[eid] + gemv_f32 → weighted accumulate
   m. add_inplace (residual)
3. Final rmsnorm + gemv_f16 (lm_head)

Additional struct changes needed:
- `MixtralLayer`: add `attn_k_ref: TensorRef`, `attn_v_ref: TensorRef`, `gate_inp_ref: TensorRef`.
- `MixtralEngine`: add `gguf: GgufFile` field reference so the CPU path can slice mmap bytes (already present — `gguf` is a field at line 151).

---

## Go/No-Go Decision

### DeepSeek-V2: GO (trivial, ~4 lines)

The 4-line `mla_metal` platform gate is a fully bounded, surgical change. The MoE FFN CPU path works today. The attention CPU path (kv_b_proj expand + mha_decode_step) also works when mla_c_kv is empty. This is correct, zero-tps correctness-only reach.

**Risk:** None beyond the `mla_metal = false` meaning the CPU path uses the materialized KV cache (full kv_b_proj expansion) rather than the compressed MLA cache. This is correct behavior and matches the existing test path.

### Mixtral: NO-GO for one pass (too many struct changes + new forward body)

Requires: 3 new TensorRef fields in `MixtralLayer`, a new ~120-line `forward_token_cpu`, changes to `forward_token` dispatch, and load-time plumbing. This is clean but not bounded to a "surgical edit" — it touches the struct definition (load code + forward code + dispatch). It belongs in its own dedicated session after the deepseek_v2 fix is verified.

---

## Implementation Plan (DeepSeek-V2 Only — One Pass)

### Edit 1: `deepseek_v2.rs` lines 646–660 — suppress mla_metal off-macOS

In the `load()` function, change the `mla_metal` initialization so it is always `false` when not on macOS:

```rust
// BEFORE:
let mla_metal = config
    .kernel_profile
    .as_ref()
    .map(|p| p.selected.mla_schedule.as_str() == "metal-mla")
    .unwrap_or(true);

// AFTER:
#[cfg(target_os = "macos")]
let mla_metal = config
    .kernel_profile
    .as_ref()
    .map(|p| p.selected.mla_schedule.as_str() == "metal-mla")
    .unwrap_or(true);
#[cfg(not(target_os = "macos"))]
let mla_metal = false;
```

With `mla_metal = false`, `mla_c_kv` is allocated as `Vec::new()` (line 660: `(Vec::new(), Vec::new())`). This means `!self.mla_c_kv.is_empty()` at line 3466 is `false`, bypassing the MLA branch and taking the full-KV-expand path at line 3633. That path is CPU-capable.

### No other edits needed for DeepSeek-V2 MoE CPU reach.

The full forward chain at that point:
- `forward_token()` → `forward_token_final_norm()` → `forward_token_final_norm_maybe_read()`
- Wedge C TCB is `#[cfg(target_os = "macos")]`, so off-macOS falls to the else branch (which calls `attention() + ffn()` per layer in the same function body — need to verify this).

### Additional audit: `forward_token_final_norm_maybe_read` non-macOS path

Lines 2689–2768: the `#[cfg(target_os = "macos")]` block covers the TCB fast path. Off-macOS, the function continues below that block. Need to confirm there is a non-macOS fallback body that calls `attention()` + `ffn()` per layer.

If that fallback body is missing, a second small edit is needed to wire up the per-layer loop off-macOS, similar to how `forward_token_shared_only` works (lines 3892–3931, which already does it correctly without any Metal-specific gating).

---

## Cross-Build Blockers (Cargo Check on Non-macOS Target)

The prior scout noted it could not run a cargo check on a non-macOS target. From reading the code, the probable issue is:

1. `MetalContext`, `PinnedBuffer`, `DecodeArena` are defined under `metal/` which may be `#[cfg(target_os = "macos")]` at the module level. The types ARE referenced in non-macOS struct fields (e.g., line 934: `let (mla_c_kv_gpu, mla_k_pe_gpu): (Vec<PinnedBuffer>, Vec<PinnedBuffer>) = (vec![], vec![])`) — so they must be defined cross-platform (as stubs). Confirm by checking `src/metal/mod.rs`.
2. Mixtral's `MixtralDecodeArena` has two definitions gated by `#[cfg]`, which compiles correctly already.
3. The `memmap2` crate is cross-platform; mmap is the `GgufFile.mmap` and works on Linux/Windows.
4. No `#[link_args]` or Metal framework links should appear in non-macOS builds — those should already be conditional on `target_os = "macos"` in `build.rs`.

**Likely root cause of prior cargo check failure:** Missing `target` for `cargo check --target x86_64-unknown-linux-gnu` (no `std` cross-toolchain installed in the macOS dev env). This is an env issue, not a code issue.

---

## Files to Modify

### For deepseek_v2 CPU MoE (minimum viable):
- `/Users/scammermike/Downloads/dismantle/crates/dismantle-core/src/model/deepseek_v2.rs` — `mla_metal` initialization (lines 646–660) and possibly `forward_token_final_norm_maybe_read` non-macOS fallback body.

### For Mixtral CPU MoE (future session):
- `/Users/scammermike/Downloads/dismantle/crates/dismantle-core/src/model/mixtral.rs` — struct `MixtralLayer` (add 3 TensorRef fields), `load()` (populate them), `forward_token()` (add CPU branch), new `forward_token_cpu()` method.

### No changes to:
- `moe/mod.rs` — already complete, `expert_ffn` / `moe_forward_token` / `add_shared_experts` are all CPU pure-Rust.
- `moe/dispatch.rs` — already complete, `build_work_queue` is CPU pure-Rust.
- `quant/mod.rs` — already covers Q4_K / Q8_0 / Q3_K / Q6_K / Q5_K.
- `kernels/mod.rs` — `gemv_f32` / `gemv_f16` / `silu_mul` / `softmax_inplace` are all CPU pure-Rust.

