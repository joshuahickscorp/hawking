# AWQ Option B — bake `W' = W * s` into a Q4K_FAST sidecar

**Status:** **all code written, uncommitted, not compile-tested.**
Default-off behind `DISMANTLE_QWEN_AWQ=1`. Compile + parity tests +
end-to-end bench deferred until the MLX training run finishes (RAM is
tight while training holds ~800 MB).

Landed in this session:
- offline bake tool: [tools/awq_bake/src/main.rs](../tools/awq_bake/src/main.rs)
- new Metal shaders in [shaders/quant.metal](../crates/dismantle-core/shaders/quant.metal)
  (`quantize_f32_to_int8_per_block_scaled`) and
  [shaders/common.metal](../crates/dismantle-core/shaders/common.metal)
  (`add_rmsnorm_fused_q8_scaled`)
- Rust kernel wrappers + CPU reference + 9 parity tests across 2 new
  test files in [tests/](../crates/dismantle-core/tests/)
- model-side: loader (`ensure_awq_smoothing_scales`), 4 new
  `Option<Vec<PinnedBuffer>>` fields, env gate with hard preconditions,
  AWQ-aware sidecar path lookup, dispatch swap at all 5 W4A8 quantize
  sites (last-layer fused stays unscaled to preserve the LM_HEAD math).

**Why Option B over A** — runtime dispatch count is unchanged from
plain W4A8: Q/K/V share `x_norm` and they all use the **same** AWQ
smoothing vector (verified — `layer_0_q_proj` == `layer_0_k_proj` ==
`layer_0_v_proj` in [profiles/qwen3b_awq_smoothing.json](../profiles/qwen3b_awq_smoothing.json));
similarly gate/up share `ffn_act`. So x_norm and ffn_act are still
quantized once per layer — no +4 dispatches like Option A would
require. The activation-divide folds into the existing
`quantize_f32_to_int8_per_block` dispatch via a new `_scaled` variant.

---

## Math

For each Q4_K projection weight `W` of shape `[out_rows, in_cols]` and
its AWQ smoothing vector `s` of length `in_cols`:

```
offline:  W'[r, c] = W[r, c] * s[c]              # broadcast over rows
runtime:  x' = x / s                              # elementwise on activation
          y = x' · W'.T = (x/s) · (W*s).T = x · W.T   (mathematically identical)
```

The **quantized** versions differ: `W'` has its hot channels reshaped
to the "average" magnitude profile, so the int8/Q4_K quantizers
allocate range more uniformly — fewer outliers blowing out per-block
scales. That's the AWQ win.

---

## Offline bake (DONE)

`tools/awq_bake/src/main.rs` (252 LOC):

1. Read `profiles/qwen3b_awq_smoothing.json` (schema `awq-smoothing-v1`).
2. Open input GGUF, hash first 8 bytes of SHA256 → `src_hash`.
3. For each `Q4_K` tensor whose name matches the AWQ key map
   (`blk.{N}.attn_q.weight → layer_{N}_q_proj`, etc., 7 sites per layer):
   - `dequant_into(Q4_K, bytes, &mut f32)` → row-major `[rows, cols]`
   - row-wise pointwise multiply by `s[c]`
   - `quantize_q4_k(&f32, &mut q4_bytes)` (existing primitive in
     [quant/mod.rs:183](../crates/dismantle-core/src/quant/mod.rs:183))
   - `convert_q4k_tensor_to_fast(q4_bytes, rows, cols)` → Q4K_FAST layout
4. `serialize_sidecar(src_hash, written)` → write `.dismantle` file.

Output is wire-compatible with the existing Q4K_FAST loader at
`dismantle-core::q4k_fast`. **No runtime loader changes needed** to
read the file — only a selector change for which sidecar path to mmap.

Run (when ready):
```
cargo build --release -p awq_bake_tool
./target/release/awq_bake_sidecar \
    /path/to/qwen2.5-3b-instruct-q4_k_m.gguf \
    profiles/qwen3b_awq_smoothing.json \
    artifacts/qwen3b_awq_baked.dismantle
```

Expect ~5 min on M3 Pro (252 dequant+requant passes, single-threaded).
Output size matches the plain Q4K_FAST sidecar (~1.5 GB).

**Coverage note:** Qwen-3B-Q4_K_M has K/V projections in Q6_K (per
`qwen3b_dead_levers.md` history). The bake tool skips non-Q4_K
tensors — K/V get no AWQ benefit in this version. If AWQ-for-Q6_K
becomes the next lever, mirror the same offline flow with
`quantize_q6_k` (already exists at
[quant/mod.rs:258](../crates/dismantle-core/src/quant/mod.rs:258)).

---

## Runtime wire-up (DONE — uncompiled)

### 1. New Metal shader: `quantize_f32_to_int8_per_block_scaled`

Take the existing `quantize_f32_to_int8_per_block` shader; add a
`device const float *s` argument; divide each element by `s[gid]`
before computing the per-block min/max for the int8 scale. One extra
load + one fdiv per element, no branching, no extra dispatch.

Surface as `kernels::quantize_f32_to_int8_per_block_scaled_tcb(
    tcb, x_f32, s_smoothing, out_int8, out_scales, n)`.
Validate with a CPU-vs-Metal parity test (`atol=1e-3` fp16, same
gate as existing W4A8 kernels).

### 2. `qwen_dense.rs` changes (written)

| Edit | Location | What |
|---|---|---|
| **Struct fields** | after `lmhead_per_channel_scales_buf` | 4 new `Option<Vec<PinnedBuffer>>` — `awq_smoothing_x_norm`, `awq_smoothing_attn_out`, `awq_smoothing_ffn_act`, `awq_smoothing_silu_mul`. Init to `None`. |
| **Loader** | `ensure_awq_smoothing_scales()` next to `ensure_lmhead_per_channel_scales` | parses `profiles/qwen3b_awq_smoothing.json` via `serde_json`, validates `awq-smoothing-v1` schema + per-key lengths, builds 4 × n_layers pinned f32 buffers. |
| **Sidecar swap** | `ensure_q4k_fast_cache` | when `DISMANTLE_QWEN_AWQ=1`, prepend AWQ-baked candidates (`<gguf-stem>.awq.dismantle`, `models/<stem>-awq.dismantle`, `artifacts/qwen3b_awq_baked.dismantle`) before the plain Q4K_FAST candidates. |
| **Env gate** | inside `forward_token_greedy_tcb` early | hard requirements: `DISMANTLE_QWEN_AWQ=1` requires `W4A8=1` and `PREDEC=0` (returns `Err(Model)` otherwise — predec is mathematically incompatible because its scale cache comes from the un-smoothed weights). Forces `q4k_fast_ref` active so the Q4_K dispatcher reads the AWQ-baked sidecar. |
| **Dispatch swap** | 5 sites inside `forward_token_greedy_tcb` | (1) pre-loop `x_norm` quantize → `awq_smoothing_x_norm[0]`; (2) per-layer `attn_out` quantize → `awq_smoothing_attn_out[li]`; (3) post-attn fused → `awq_smoothing_ffn_act[li]`; (4) per-layer `ffn_act` quantize → `awq_smoothing_silu_mul[li]`; (5) post-FFN fused → `awq_smoothing_x_norm[li+1]` when `li+1 < n_layers`, else **unscaled** because the LM_HEAD weight isn't AWQ-baked. |

### 3. Validation gates (before flipping default-on)

1. **Numerical parity (synthetic)** — new test
   `tests/awq_smoothing_parity.rs`: run a single Q4_K projection
   `y = x · W.T` two ways with a hand-picked `s`:
   - reference: dequant W, `(x/s) · (W*s).T`, fp32
   - test: dequant baked W' = W*s, `(x/s) · W'.T`, fp32 (via existing
     gemv path)
   Cosine ≥ 0.9999, max |diff| ≤ 1e-3 fp16.

2. **End-to-end bit-exactness vs Python reference** — `tau_eval.py`
   already produces a reference logit stream; bake-and-run should
   match within fp16 cosine 0.999.

3. **Corpus quality (N=100)** — re-run the same gate that put plain
   W4A8 at 20% bit-identical. AWQ's whole job is to lift that number;
   ship rule is **≥ 85% bit-identical at 32 tok greedy** (memory's
   `w4a8_quality_redesign_2026_05_26.md` threshold).

4. **Paired bench (n=5)** — locked Qwen-3B config:
   - vs `predec` (current default): need ≥ +5% to flip default; AWQ+W4A8
     was sub-additive (1.15× < 1.34× predec-alone) in
     `composition_decision_matrix_2026_05_26`, so the bet is that AWQ
     unblocks the W4A8 quality gap without losing perf.
   - vs `predec + w4a8` (memory's 1.15× combo): need quality > 85%
     bit-identical to *ship* (perf already known good).

---

## Effort estimate (revised)

| Step | LOC | Time | Risk |
|---|---|---|---|
| Offline bake tool (done) | 252 | — | low — reuses tested primitives |
| Metal shader + parity test | ~80 | 1 hr | low — mirrors W4A8 shader |
| qwen_dense.rs wire-up | ~120 | 2 hr | medium — 4 sites, env-gating |
| Corpus quality sweep | — | 30 min | gates ship/hold |
| **Total runtime side** | **~200** | **~3.5 hr** | |

---

## Decision points pending

1. **Should we bake `k_proj` / `v_proj` (Q6_K)?** AWQ JSON has entries
   for them, but Qwen-3B Q4_K_M stores them in Q6_K. Skipping them
   means 5 of 7 sites benefit (q/o/gate/up/down). Worth a second
   N=100 quality sweep to see if Q6_K K/V is the residual quality
   blocker before adding Q6_K bake.
2. **What `s` lives where at runtime?** Two designs:
   - **(a) Per-layer concatenated pinned buffer** — 4 buffers × 36
     layers × (2048 or 11008) f32. Total ~5 MB. Simple, low-overhead.
   - **(b) Per-projection separate buffers** — 7 × 36 = 252 buffers.
     Cleaner dispatch but 252 buffer bindings = bloat.
   Recommend (a).
3. **Compile-tested? No.** The bake tool is read-only / no-compile
   per the user's current RAM-constraint window. First `cargo build`
   pass after training finishes will validate imports.
