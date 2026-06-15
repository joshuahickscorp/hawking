# Phase 3.1 spec — backend seam (scout a89b10cc, 2026-06-01)

> Full transcript in agent a89b10cc. Actionable distillation.

## Confirmed
- Entire `metal_dispatch` (~120 TCB fns, `kernels/mod.rs:367-7410`) is
  `#[cfg(target_os="macos")]`. Off-macOS `TokenCommandBuffer::new` PANICS
  (`metal/mod.rs:1258`). Engine can't run off-macOS today → seam is the prereq.
- `Engine` trait (`engine.rs:200`) is a model/generate seam, ABOVE compute —
  orthogonal. The compute seam is new.

## Trait surface (Burn shape)
**~11 op verbs + 6 resource primitives.** ONE `Backend` supertrait bundling
op-traits (BackendGemv/Norm/Elementwise/Rope/Attention/KvCache/Quant/Embed/
Sample/Moe) + assoc types `Buffer` + `Recorder<'a>` (GAT). The **~25 GEMV
variants collapse to ONE `gemv` verb** — the variant explosion lives in a
`GemvSpec`/`WeightKind` enum + the impl body (where `gemv_proj!`
`qwen_dense.rs:3834` lives today), NOT 25 trait methods.
- Op-trait methods take `&mut Self::Recorder` (the TCB) → every op records into
  the ONE per-token command buffer. Trait does NOT impose one-dispatch-one-commit.
- New module `backend/mod.rs` (platform-neutral trait defs), `backend/metal.rs`
  (`MetalBackend(MetalContext)`, macOS-gated). Wire `pub mod backend;` in lib.rs.

## ⭐ Key decision: keep `MetalBackend` CONCRETE in 3.1
`type Buffer = metal::Buffer` everywhere → `DenseDecodeArena` + ~180 weight
fields UNCHANGED; trait just wraps existing concrete types. Genericizing the
model over `<B>` is **deferred to 3.2** (CPU backend, when a 2nd impl forces it).
This holds 3.1 to a **pure, hash-identical refactor** with no type-param explosion.

## Routing order (lowest blast-radius → highest; gate after EACH)
1. `BackendDevice` + `CommandRecorder` lifecycle (no op math) — proves the seam.
2. embed → 3. rope → 4. elementwise (add/silu_mul) → 5. kv_append/memcpy →
6. rmsnorm (+fused q8) → 7. quantize (W4A8) → 8. attention (mha_decode, wrap
as-is) → **9. gemv LAST** (the moat: `gemv_proj!` + FFN-down/LM-head ladders).
Each step: thin indirection (trait method calls existing `kernels::*_tcb`
UNCHANGED) → run HARD gate → proceed only if bit-identical. Then repeat for
deepseek_v2 (MLA+MoE), then llama/phi3/gemma2 (~5 sites each, trivial).

## HARD gate (after every step; STOP if it moves)
- (a) `cargo test --release -p dismantle-core --test integration_greedy_64`
  (always-on golden-hash: greedy-64 token SHA vs `tests/golden/_phase0_token_baseline_64.hashes`)
  + `phase1_kernel_parity` + the touched family's TCB parity suites.
- (b) `dismantle batch-hash --tokens 64` vs a FRESH HEAD baseline (capture before
  starting; historical baselines encode older lever states) → diff must be empty.
- (c) paired tps within noise (~±2%). Devirt risk: concrete `MetalBackend` (not
  `dyn`) → monomorphized, should be free; add `#[inline]` if a step dips.

## Stays Metal-specialized (do NOT genericize/decompose)
gemv_proj! predec/fast/W4A8 ladder (the ~47% decode moat); mha_decode /
mla_decode; **single-CB-per-token TCB batching** (op-traits MUST thread
`&mut Recorder`, never commit internally — naive per-op-commit shatters it +
tanks tps); concurrent-encoder groups (no-op on backends lacking the concept);
fused cross-op kernels (`add_rmsnorm_fused`, `gemv_pair`) surface as FUSED verbs,
not decomposed (decomposition loses fusion → changes hash); zero-copy mmap pin
(`new_buffer_no_copy`).

## Riskiest
(1) raw token readback `token_buf.contents() as *const u32` (:4617/:4706) → hide
behind `BackendDevice::read_u32`. (2) `Self::Buffer` through arena+180 fields →
mitigated by concrete-in-3.1. (3) GAT/lifetime of `Recorder<'a>` wrapping
`TokenCommandBuffer<'ctx>` + Drop-auto-commit — the one non-trivial type piece,
where compile-fighting concentrates. (4) cfg boundary (backend/mod.rs neutral,
backend/metal.rs gated).

## Size
~250-350 LoC trait defs + ~400-600 LoC thin impls (kernel BODIES don't move);
~200 call-site edits (1-line `kernels::foo`→`backend.foo`), gated by family.
Medium-large but mechanically shallow; low correctness risk if the gate runs
after each family.

## Files
- metal/mod.rs (MetalContext/TCB lifecycle/alloc :505/:564/:854/:1183/:1248)
- kernels/mod.rs (`mod metal_dispatch` :367 → MetalBackend impl bodies)
- qwen_dense.rs (forward_token_greedy_tcb :3422, gemv_proj! :3834, ~95 sites)
- metal/dense_decode_arena.rs (keep concrete in 3.1)
- tests/integration_greedy_64.rs (always-on hard gate) + main.rs:1170 batch-hash
- NEW: backend/mod.rs (neutral), backend/metal.rs (macOS-gated)
