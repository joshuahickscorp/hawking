# Phase 4.4 — wgpu backend skeleton + dependency plan (Wave-4c)

> **STATUS: DESIGN-NOTE, DEPENDENCY ADDITION FLAGGED FOR USER APPROVAL.**
> This note accompanies the new `crates/dismantle-core/src/backend/wgpu.rs`
> skeleton (a `WgpuBackend` implementing the landed `backend/mod.rs` seam,
> behind `#[cfg(feature = "wgpu")]`). Per CLAUDE.md, **no Cargo.toml diff
> was authored**: the exact dependency + feature lines below are for the
> orchestrator (user) to approve and apply. Until then the skeleton does
> **not** compile (the `wgpu` crate is absent) — that is expected and
> by-design; with the feature off, nothing in the file is compiled and the
> default macOS build / golden batch-hash `b480cc10` are unaffected.

---

## 1. What landed in this stream (code, no dep)

1. `crates/dismantle-core/src/backend/wgpu.rs` (new) — `WgpuBackend`:
   - `type Buffer = WgpuBuffer` (newtype over `Arc<wgpu::Buffer>`).
   - `type Recorder<'a> = WgpuRecorder<'a>` (one `wgpu::CommandEncoder`
     per token; `commit_and_wait` = `queue.submit([encoder.finish()])` +
     `device.poll(Wait)`; `read_u32` = a 4-byte `MAP_READ` staging
     round-trip).
   - **Real WGSL compute** for three verbs: `rmsnorm` (single-workgroup
     fp32 reduction, mirrors `shaders/common.metal:rmsnorm_f32`), `add`
     (`a[i]+=b[i]`, mirrors `add_inplace`), `silu_mul` (mirrors
     `silu_mul`). WGSL is embedded in the file (no external asset).
   - **Honest stubs** for the other eight verbs (`gemv`, `rope`,
     `attention`, `kv_append`/`memcpy`, `quantize`, `embed`,
     `sample_argmax`, `moe_*`) returning `Error::Unimplemented`.
   - `supports()` reports `true` only for `Op::RmsNorm`, `Op::Add`,
     `Op::SiluMul`; everything else `false`, so the Phase-3.2 scheduler
     CPU-falls-back the rest (the `ggml_backend_sched` lesson).
   - The fused `add_rmsnorm*` norm variants are stubbed (a faithful fused
     kernel must share the add + variance reduction in one pass to keep
     numerics; decomposing breaks "fused kernels stay fused"). Plain
     `rmsnorm` is real.
2. A one-line edit to `crates/dismantle-core/src/backend/mod.rs` adding
   `#[cfg(feature = "wgpu")] mod wgpu;` after the existing
   `#[cfg(target_os = "macos")] mod metal;`. Safe to land before the dep
   (it references the missing crate only when the feature is on).

The skeleton uses **no new `Error` variant**: stubs use the existing
`Error::Unimplemented(&'static str)`, runtime/device failures use
`Error::Kernel(String)`. So `error.rs` is untouched.

---

## 2. EXACT Cargo.toml additions — FLAGGED FOR USER APPROVAL

Three edits, none applied. Versions pinned to the wgpu **0.20** series to
match `reports/phase_4_4_cross_vendor_scope.md` §3. (If you prefer a newer
wgpu, a handful of call sites in `wgpu.rs` need a mechanical touch-up —
see §4 Risks.)

### 2a. Workspace `Cargo.toml` — add to `[workspace.dependencies]`

```toml
# Cross-vendor GPU reach (Phase 4.4, Rung 1 — Ratchet "WGPU + CPU" model).
# Default-OFF; pulled in only by the dismantle-core `wgpu` feature.
# `metal` backend here is wgpu's own abstraction layer and does NOT conflict
# with the existing `metal = "0.29"` crate (independent layers).
wgpu = { version = "0.20", default-features = false, features = ["wgsl", "metal", "vulkan", "dx12"] }
pollster = "0.3"   # block_on for the async device bring-up at the app edge
```

Notes:
- `default-features = false` + an explicit backend list keeps the
  dependency lean (drops GL / WebGPU-on-native / angle unless wanted). Add
  `"webgpu"` to the feature list to also target browser/WASM later.
- `wgsl` is required (the skeleton ships WGSL source, not SPIR-V).
- No vendor SDK build scripts: wgpu loads vendor drivers at runtime, so there
  is **no vendor SDK build dependency** (the lighter Cargo.toml addition
  the scope recommends over CubeCL).

### 2b. `crates/dismantle-core/Cargo.toml` — add an optional dep + a feature

```toml
# in [dependencies]
wgpu     = { workspace = true, optional = true }
pollster = { workspace = true, optional = true }

# new [features] table (the crate currently has none)
[features]
# Cross-vendor wgpu backend (Phase 4.4). Default-OFF: the default macOS
# build does not compile backend/wgpu.rs and the golden batch-hash is
# unaffected. Enable with `--features wgpu` (or from a downstream crate).
wgpu = ["dep:wgpu", "dep:pollster"]
```

### 2c. `bytemuck` is already a dependency

The skeleton's uniform-buffer upload uses `bytemuck::cast_slice`;
`bytemuck` is already in `crates/dismantle-core/Cargo.toml` (workspace
dep), so **no addition needed** for it. `wgpu::util::DeviceExt`
(`create_buffer_init`) ships with the `wgpu` crate.

---

## 3. Why this is default-safe (the parity argument)

- Both the `mod wgpu;` declaration and the file body are
  `#[cfg(feature = "wgpu")]` / `#![cfg(feature = "wgpu")]`. With the
  feature off (the default), the compiler sees an empty module — zero
  symbols, zero codegen, zero effect on the Metal path or the golden
  batch-hash `b480cc10`.
- The `wgpu`/`pollster` deps are `optional = true` and gated behind the
  `wgpu` feature, so a default `cargo build` does not even resolve them.
- No `backend/mod.rs` trait was modified; no `error.rs` variant added; no
  kernel body moved. The seam is consumed, not changed.

---

## 4. Open items the implementer inherits (post-approval)

1. **wgpu version pin.** `wgpu.rs` targets the 0.20 API
   (`device.poll(wgpu::Maintain::Wait)`, `begin_compute_pass` descriptor
   with `timestamp_writes`, `request_device` 2-arg form). On 0.22+ rename
   `Maintain::Wait` → `PollType::Wait` and check the `request_adapter`/
   `request_device` signatures. Decide the pin at approval time.
2. **read_u32 staging cost.** The seam's `read_u32` (argmax id readback)
   is a STORAGE→MAP_READ copy + `poll(Wait)` per call in wgpu (no
   host-visible address space like Metal's `PinnedBuffer.contents()`).
   Implemented inside the recorder; the sampled token buffer must carry
   `COPY_SRC` usage. Fine for a reach backend; revisit if argmax ever
   lands on a wgpu hot path.
3. **Weight-offset descriptor (SEAM GAP #1).** The future WGSL Q4_K GEMV
   needs `(model_buf, offset, byte_size)` that `GemvSpec` does not carry
   (the Metal impl documents the same gap). Fold a weight-offset field
   into `GemvSpec` before implementing `WgpuBackend::gemv`.
4. **Fused norm verbs.** `add_rmsnorm` / `_q8` / `_q8_scaled` need
   dedicated single-pass WGSL (the add + variance reduction in one
   kernel) to preserve numerics; currently stubbed.
5. **Next rungs (scope §3 effort):** tiled WGSL decode-attention
   (~3-5 days), KV-append/memcpy (~1 day, or a plain
   `copy_buffer_to_buffer`), then the Q4_K WGSL GEMV (~1-2 weeks, the hard
   rung). MoE stays CPU-fallback.

---

## 5. Gate (deferred until dep approved + file compiles)

Per scope §5 Rung 1: a wgpu decode on a **non-Apple** GPU must produce
first-3 greedy token IDs identical to the CPU reference; absolute
performance is **not** gated (reach, not speed). Per-op, the three real
WGSL verbs must match the CPU/Metal scalar verbs at `atol=1e-3` fp16 —
they are pure f32 here, so the only expected divergence is the GPU
reduction reorder in rmsnorm (within the `rtol=1e-4` floor, scope §7).
The eight stubbed verbs are never numerically compared (they route to the
already-parity-gated CPU path via `supports()->false`). This backend is
NOT held to the bit-identical golden gate — WGSL gives no MSL-identical
fp-contraction guarantee.
