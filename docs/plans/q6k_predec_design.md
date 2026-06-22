# Q6_K predec ffn_down — implementation design (campaign R-design, 2026-06-21)

Top real candidate from the throughput-pivot R0 audit: the +34% Q4_K predec win was
NEVER applied to the DEFAULT ffn_down, which is Q6_K (~46% of decode). Stage 1 is
BIT-IDENTICAL. DESIGN ONLY — not implemented/merged; for human review.
(NB: the q6k-target + synthesis agents API-failed mid-run; this is the precedent
agent's output, which independently produced the full mirrored Q6_K design.)

---

## Q4_K PREDEC PRECEDENT — exact pattern to mirror for Q6_K

### (1) Scale-table layout — what is decoded ONCE, byte layout, keying

The table is a flat `Vec<f32>` built once per Q4_K weight tensor at load. For Q4_K it is **16 f32/block** = the 8 `(ds, dm)` pairs:
- `predecode_q4_k_scale_table(w_q4_bytes)` (kernels/mod.rs:1587): for each 144-byte block, widen `d`/`dmin` (f16→f32), unpack the 8 6-bit `sb[sub]`/`mb[sub]` sub-block scale/min indices with the EXACT same bit ops the shader uses, then store `out[so+sub*2] = d*sb[sub]`, `out[so+sub*2+1] = dmin*mb[sub]`. Bit-identical because it uses the same f16→f32 widening and f32 multiply order as the inline kernel.
- Table size = `(weight_bytes/144)*16` f32 = 0.444× weight size (the doc-comment notes this RSS cost). For Q4_K_M this is the ~760 MB RSS the env flag lets you opt out of.
- Stored per-tensor as a `crate::metal::PinnedBuffer` via `ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32,u8>(&scales))`.
- **Keying:** a `HashMap<usize, PinnedBuffer>` keyed by **`tref.offset`** (the tensor's byte offset in the GGUF mmap). Struct field `q4k_predec_cache: Option<HashMap<usize, PinnedBuffer>>` at qwen_dense.rs:270 (`#[cfg(target_os="macos")]`, `None` = feature off/no memory cost). An f16 twin (`q4k_predec_cache_f16`, qwen_dense.rs:281) narrows each f32 to half — NOT bit-identical, separate opt-in flag; **ignore for Q6_K Stage 1**.

### (2) How the kernel reads the predec table instead of inline-decoding

`gemm_q4_k_v4_predec` (quant.metal:2605): adds `device const float* scales [[buffer(1)]]` (shifting x→buffer(2), y→buffer(3), rows→buffer(4), cols→buffer(5) vs the inline kernel). Computes `row_scale_off = base_row*blocks_per_row*16`, then per block reads `ds[sub]=scales[so+sub*2]`, `dm[sub]=scales[so+sub*2+1]` instead of unpacking the packed 6-bit indices. The 4-bit quant nibbles and the per-block d/dmin f16 header are NOT changed — only the per-sub-block-scale decode (invariant across forward passes) is hoisted. Same v3_8r geometry (256 threads/TG, 8 simdgroups, 8 rows/TG). Bit-identical (explicit comment at quant.metal:2599).

### (3) Cache build/ensure + dispatch wiring + env flag

- **Build/ensure:** `ensure_q4k_predec_cache(&mut self)` (qwen_dense.rs:3988, `#[cfg(target_os="macos")]`): early-returns if `q4k_predec_cache.is_some()`; tries the `.hawking` sidecar first (validates GGUF SHA-256 hash, rejects stale); else walks every layer's 7 projection sites (q/k/v/o/gate/up/down), and for each `dtype==Q4_K` tensor not already cached, runs `predecode_q4_k_scale_table` on `gguf.mmap[offset..offset+byte_size]` and pins it keyed by offset. Sets `self.q4k_predec_cache = Some(cache)`.
- **Triggered** from forward at qwen_dense.rs:4660: `if predec_active && self.q4k_predec_cache.is_none() { self.ensure_q4k_predec_cache()?; }` (also 4096, 8806, 8991 for other entry paths).
- **Dispatch:** `predec_cache_ref = if predec_active { self.q4k_predec_cache.as_ref() } else { None }` (4793). At the ffn_down site (qwen_dense.rs:6870, the Q4_K branch) it does `ffn_down_predec.then(|| predec_cache_ref.and_then(|m| m.get(&layer.ffn_down.offset))).flatten()` → if `Some`, calls `gemv_q4_k_v4_predec_swiglu_pinned_tcb(...)`; else `false` → falls through to silu_mul + inline v3_8r. **The Q6_K branch (qwen_dense.rs:6855-6868) currently has NO predec path — it unconditionally calls `gemv_q6_k_swiglu_pinned_tcb`. This is the exact insertion point.**
- **Env flags:** `HAWKING_QWEN_Q4K_PREDEC` (qwen_dense.rs:4647, **default-ON**, opt-out `=0`) gates cache build + `predec_cache_ref`. `HAWKING_QWEN_FFN_DOWN_PREDEC` (5199, default-ON, `&& predec_active && !w4a8_active`) is the per-site gate. Predec is **incompatible with AWQ/W4A8** (hard error at 4775) because those overwrite the weight.

---

## PRECISE PATTERN TO MIRROR FOR Q6_K

**Q6_K block (210 B / 256 elems):** `ql[128]` (off 0), `qh[64]` (off 128), `scales[16]` int8 (off 192), `d` f16 (off 208). The inline kernel `gemm_q6_k_fused_v2_swiglu` (quant.metal:5147) reads exactly **one int8 scale byte per (lane, block)** at `scale_byte_off = 192 + half_idx*8 + scale_l_off + group*2` and computes `dscale = d * (float)scale`. There are 16 scale bytes; the 32 lanes map onto the 16 indices. The invariant hoistable quantity is `d * scale[i]` for all 16 sub-block scales.

1. **`predecode_q6_k_scale_table(bytes) -> Vec<f32>`** (add to quant.rs, mirror `predecode_q3_k_scale_table` at quant.rs:645 — Q6_K is symmetric like Q3_K, **no min term**, so **16 f32/block**, NOT 16 pairs): for each 210-byte block, `d = f16(bytes[208..210])`, then `out[b*16+i] = d * (bytes[192+i] as i8 as f32)` for i in 0..16. Use the **same f16→f32 widening and f32 multiply** as the kernel for bit-identity. Table size = `(weight_bytes/210)*16` f32 ≈ 0.305× weight size (smaller than Q4_K's 0.444× because Q6_K blocks are larger).

2. **New kernel `gemm_q6_k_fused_v2_swiglu_predec`** (clone quant.metal:5147): add `device const float* scales [[buffer(N)]]`; replace `int scale = (int)(signed char)w_q6[bo+scale_byte_off]; float dscale = d*(float)scale;` with a read of the pre-decoded value. The cleanest mapping: index the 16-f32/block table by the **sub-block index** the lane's `scale_byte_off` resolves to, i.e. `scale_idx = (scale_byte_off - 192)`, so `dscale = scales[base_row*blocks_per_row*16 + b*16 + scale_idx]`. Keep ql/qh nibble decode and the SwiGLU activation read unchanged. Do the same for the **2r and 4r variants** (quant.metal:5226 `_swiglu_2r`, plus `_swiglu_4r`) since those are the default/opt-in routes — each reads `dscale0`/`dscale1` etc. per row; pass per-row scale offsets. **Stage 1 should at minimum cover the kernel that runs by default (the 2r variant is default-ON via `HAWKING_QWEN_Q6K_SWIGLU_2R`).**

3. **Cache:** add struct field `q6k_predec_cache: Option<HashMap<usize, PinnedBuffer>>` (mirror qwen_dense.rs:270, `#[cfg(target_os="macos")]`, init `None` in the constructor next to `q4k_predec_cache: None`). Add `ensure_q6k_predec_cache(&mut self)` mirroring qwen_dense.rs:3988 — but only the ffn_down site is Q6_K on the default path, so the walk can be just `insert_q6k(&layer.ffn_down, ...)` gated on `dtype==Q6_K` (optionally also o_proj if it is ever Q6_K). Sidecar integration can be deferred (Stage 1 may recompute every load; add a `q6k_predec_scales` sidecar content flag later mirroring qwen_dense.rs:3278/4002).

4. **Wiring:** at qwen_dense.rs:6855 (Q6_K branch), before calling `gemv_q6_k_swiglu_pinned_tcb`, look up `q6k_predec_cache_ref.and_then(|m| m.get(&layer.ffn_down.offset))` gated by a new `HAWKING_QWEN_Q6K_FFN_DOWN_PREDEC` flag (mirror `ffn_down_predec`/5199; reuse `predec_active` gating + `!w4a8_active`). If `Some`, call a new `gemv_q6_k_v4_predec_swiglu_pinned_tcb` wrapper (mirror `gemv_q6_k_swiglu_pinned_tcb` at kernels/mod.rs:4216 — add a `scales_buf`/`scales_offset` arg, validate `expected_scale_bytes = rows*blocks_per_row*16*sizeof(f32)`, dispatch the predec kernel, preserve the 2r/4r selection logic at mod.rs:4258-4282); else fall through to the existing inline call. Trigger `ensure_q6k_predec_cache()` next to the `ensure_q4k_predec_cache()` call at qwen_dense.rs:4660.

5. **Parity test** (mirror `q3_k_predec_table_matches_decode`, quant.rs:854): build 2 blocks of pseudo-random 210-byte data with known f16 `d`, call `predecode_q6_k_scale_table`, assert `table[blk*16+i] == d * (bytes[blk*210+192+i] as i8 as f32)` exactly (==, not approx — Stage 1 is bit-identical). Plus a GPU greedy-decode parity run with the flag on vs off must be bit-identical (the standard predec gate, e.g. the 16-tok bit-identity check the Q4_K predec used).

### Bench plan
ffn_down is ~46% of decode and Q6_K on the default Q4_K_M path. A/B with `HAWKING_QWEN_Q6K_FFN_DOWN_PREDEC=1` vs `=0` via the existing paired_lever harness; the analog Q4_K predec delivered +34% headline. Watch RSS (+0.305× the Q6_K ffn_down bytes). Verify the table is read coalesced (16 f32 contiguous per block, scattered across 32 lanes — same access shape the Q4_K v4_predec already proved fast).

### KEY GOTCHAS
- Q6_K predec is **16 f32/block (symmetric, no min)** — like Q3_K, NOT like Q4_K's 16-f32-as-8-pairs. Do not copy Q4_K's `(ds,dm)` pair layout.
- The lane→scale-byte mapping (`scale_byte_off`) is non-trivial; the table must be indexed so each lane resolves to the SAME `d*scale[i]` it computed inline. Index the f32 table by `(scale_byte_off-192)` to guarantee equivalence.
- Must replicate across 1r/2r/4r kernels; the **2r is the live default**.
- Predec is mutually exclusive with W4A8/AWQ (they mutate the weight) — gate with `!w4a8_active` exactly as the Q4_K path does.
- Q6_K quants and the per-block `d` f16 are still read from the original mmap bytes — only the int8 sub-scale × d product is hoisted.
