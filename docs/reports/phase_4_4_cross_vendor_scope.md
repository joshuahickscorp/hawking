# Phase 4.4 — Cross-vendor GPU backend scope

> **Status: DESIGN-ONLY scoping artifact (2026-06-02).** This document is an attended-decision aide, not a build specification. No new dependency has been added to Cargo.toml and no code has been written. The orchestrator (user) must approve any Cargo.toml additions per CLAUDE.md policy before Phase 4.4 work begins.

---

## 0. Why this document exists

Phase 3.1 landed `backend/mod.rs` (commit 728ab6d) — the platform-neutral `Backend` / `ComputeBackend` trait seam. Phase 3.3 (CPU backend) is the cheapest non-Metal reach rung. Phase 4.4 is the next question: which crate or approach should the *GPU* portability rung use to reach AMD, Intel, external-GPU, and Android beyond the hand-tuned Metal path? The two serious candidates named in paradigmshift.md Part V.1 and plan 4.4 are **CubeCL** and **WGPU+CPU (Ratchet model)**.

This document scopes their current maturity, the exact Cargo.toml lines each would add, the rough effort per rung, how each maps onto `backend/mod.rs`, and the recommended order. It does **not** build anything.

---

## 1. The landed seam (what any new backend must satisfy)

The interface is `crates/dismantle-core/src/backend/mod.rs`. Every new backend must:

1. Pick a concrete `type Buffer` (the GPU/host handle type).
2. Pick a concrete `type Recorder<'a>: CommandRecorder<Buffer = Self::Buffer>` (the per-token command accumulator, GAT).
3. Implement `Backend::recorder()` and `Backend::supports(op: Op) -> bool`.
4. Implement all ten op-traits (`BackendGemv`, `BackendNorm`, `BackendElementwise`, `BackendRope`, `BackendAttention`, `BackendKvCache`, `BackendQuant`, `BackendEmbed`, `BackendSample`, `BackendMoe`) — or advertise `supports() = false` for unimplemented ops and route to the CPU fallback via the scheduler (Phase 3.2 router pattern).
5. Never commit inside an op method. The single `CommandRecorder::commit_and_wait` is the only commit point.
6. Keep `x_buf` (residual accumulator) as f32 in every norm method.

The blanket impl `impl<T: Backend + BackendGemv + … + BackendMoe> ComputeBackend for T {}` means a new backend acquires `ComputeBackend` automatically once all op-traits are satisfied. The CPU backend (Phase 3.3) is the reference impl: its `Recorder<'a>` is a zero-cost eager executor (no deferred command buffer), `supports(Op::Moe)` returns `false`, and every op method executes immediately on the call stack.

---

## 2. Candidate A — CubeCL

### What it is

CubeCL (`github.com/tracel-ai/cubecl`) is a single-source portable GPU kernel framework for Rust. Kernels are written in a `#[cube]`-annotated Rust subset (type-checked, borrow-checked), then JIT-lowered to six targets: cloud-GPU, cross-vendor GPU stacks, Metal (direct MSL since 0.5, including `simdgroup_matrix`), Vulkan, WebGPU, and a CPU backend. The companion `cubek` crate ships matmul, attention, and quantized-compute primitives built with CubeCL.

### Current maturity (as of 2026-06-02)

CubeCL is **alpha with active development**. Confirmed status points from paradigmshift.md:

- Metal direct MSL output landed at 0.5 with `simdgroup_matrix` support — the single largest Apple-specific concern.
- An M3 attention shared-memory bug (issue #4530) was open as of the research pass, signalling the Apple path is not hardened.
- The quantization primitives in `cubek` support symmetric per-block (q2/q4/q8/fp4) but **not** K-quant (Q4_K, Q6_K) or trellis codecs. Dismantle's hot path is Q4_K_M. A CubeCL-backed `BackendGemv` for Q4_K would require either writing the K-quant decode in `#[cube]` Rust from scratch or wrapping the existing hand-written `.metal` shader, negating the portability gain on the most performance-critical verb.
- API stability is evolving; a 4.4 build that lands today may need non-trivial rework against a future CubeCL release.

### Cargo.toml dependency (requires user approval — do NOT add autonomously)

Adding CubeCL would require at minimum the following in the workspace `Cargo.toml` and in `crates/dismantle-core/Cargo.toml`. The exact version should be pinned to the latest released tag at time of adoption (0.5.x as of this writing; check crates.io):

```toml
# workspace Cargo.toml [workspace.dependencies]
cubecl = { version = "0.5", features = ["wgpu"] }
cubecl-core = "0.5"
```

The Metal target is produced via the existing `metal` crate path (MSL output from CubeCL is handed to the Metal compiler), so no new `metal`-family crate is added. The `wgpu` feature pulls in the wgpu device stack transitively. Extra vendor-SDK features are intentionally out of scope for an Apple-first build. **Those are large, platform-specific build dependencies and represent non-trivial supply-chain additions.** Do not add them to Cargo.toml without explicit user approval per CLAUDE.md.

### How it maps onto backend/mod.rs

A `CubeclBackend` would set:

- `type Buffer = cubecl::server::Handle` (or equivalent opaque GPU handle, aliased per device).
- `type Recorder<'a> = CubeclRecorder<'a>` — a newtype over a CubeCL stream/queue accumulator, implementing `CommandRecorder::commit_and_wait` by flushing the CubeCL queue and synchronizing. `begin/end_concurrent_group` would map to CubeCL's dependency-graph or stream-ordering primitives if available, or be no-ops.
- Each op-trait method dispatches to a CubeCL kernel launch or a `cubek` primitive. `BackendGemv` for Q4_K would need a hand-authored `#[cube]` kernel mimicking the Q4_K nibble+scale decode — non-trivial, and the current `cubek` quant primitives do not cover it.

The `Backend::supports()` method allows declaring `Op::Gemv` as `false` for Q4_K initially and routing those calls to the CPU scalar path (Phase 3.2 scheduler); this is the `ggml_backend_sched` lesson and the main reason a partial backend is still shippable at Day 1.

### Effort per rung

- **Scaffolding** (Buffer + Recorder newtypes, supports(), recorder()): 1-2 days.
- **Cheap ops** (embed, elementwise add/silu_mul, rmsnorm, rope, sample_argmax): 2-4 days using CubeCL primitives or simple `#[cube]` kernels. These are ~15% of decode wall time on the Metal path.
- **Attention** (`BackendAttention::attention`): 3-5 days if using a `cubek` flash-decode primitive; longer if hand-writing from scratch. Correctness at atol=1e-3 fp16 required.
- **BackendGemv for Q4_K**: **the hard rung**. No existing CubeCL primitive. Writing a `#[cube]` K-quant decode matching the production Metal kernel's correctness at atol=1e-3 fp16 is an open-ended kernel research task, estimated 2-4 weeks for a single-pass implementation, longer if the CubeCL Metal output proves to have parity issues (the #4530 class).
- **MoE** (BackendMoe): can declare `supports(Op::Moe) = false` and CPU-fall-back; defer until Phase 4.4+ if at all.

Total realistic first-ship estimate for a CubeCL backend that handles all ops except Q4_K-via-GPU (routing GEMV to CPU scalar): **2-4 weeks of focused Rust + kernel work after dep approval**. A full GPU Q4_K path adds significant uncertainty.

### Risk summary

- Alpha crate: API churn expected.
- M3 shmem bug (#4530): attention path on Apple hardware is unvalidated.
- Q4_K is not in `cubek`: biggest effort item is also the bottleneck op.
- cloud-GPU/cross-vendor GPU stack build deps: heavy supply-chain additions that gate on vendor SDK availability at build time.
- **Verdict (paradigmshift.md):** "Watch, prototype, don't bet the hot path yet."

---

## 3. Candidate B — WGPU + CPU (Ratchet model)

### What it is

WGPU (`github.com/gfx-rs/wgpu`) is a cross-platform graphics/compute API implementation in Rust, covering Vulkan, Metal, D3D12, OpenGL ES, and browser WebGPU behind a single API surface. Combined with a CPU scalar fallback backend (Phase 3.3, already landed/scoped), this two-backend combination is exactly what the Ratchet project (`github.com/huggingface/ratchet`) uses to reach external-GPU, AMD, Intel, Apple, Android, iOS, and browser WebGPU from one codebase.

### Current maturity (as of 2026-06-02)

WGPU is **production-stable** (0.20+ series). Ratchet is a running inference engine with real models on it. Confirmed status points:

- wgpu covers every GPU target dismantle needs without vendor SDK dependencies at build time: Metal, Vulkan (AMD/Intel/external-GPU/Android), D3D12 (Windows), WebGPU (browser).
- Ratchet forked wgpu to add subgroup operations and multi-dimensional workgroup compute not yet in upstream — these are needed for performant attention and GEMV kernels. Upstream wgpu's `Features::SUBGROUP_COMPUTE` support varies by backend.
- Shaders are written in WGSL (WebGPU Shading Language), not Rust. This is a different authoring model from CubeCL's `#[cube]` and from Metal Shading Language.
- Per-backend performance will trail the hand-tuned Metal path; the plan explicitly accepts this ("reach, not speed").

### Cargo.toml dependency (requires user approval — do NOT add autonomously)

```toml
# workspace Cargo.toml [workspace.dependencies]
wgpu = { version = "0.20", features = ["vulkan", "metal", "dx12", "webgpu"] }
```

WGPU is a single crate with feature flags per backend. The `metal` feature is mutually compatible with the existing `metal = "0.29"` dependency (they use different abstraction layers; no conflict). No vendor SDK is required at build time for Vulkan/Metal/D3D12: wgpu uses runtime-loaded device drivers. On Android, the `vulkan` feature targets the Vulkan 1.1+ driver present since Android 7.0+.

**WGPU is the lighter Cargo.toml addition** — one crate, feature-gated, no vendor SDK build scripts.

### How it maps onto backend/mod.rs

A `WgpuBackend` would set:

- `type Buffer = wgpu::Buffer` (or a newtype holding `Arc<wgpu::Buffer>` for shared ownership).
- `type Recorder<'a> = WgpuCommandEncoder<'a>` — a newtype over `wgpu::CommandEncoder`, accumulating compute passes. `commit_and_wait` calls `queue.submit([encoder.finish()])` and `device.poll(wgpu::Maintain::Wait)`. `begin/end_concurrent_group` maps to wgpu pipeline barriers (or is a no-op; wgpu's implicit dependency tracking usually suffices for non-aliasing buffers).
- Each op-trait method sets up a `wgpu::ComputePass`, binds a WGSL shader pipeline, and dispatches. The WGSL shader library is a separate asset parallel to `shaders/*.metal`.
- `BackendGemv` for Q4_K requires a WGSL port of the nibble+scale decode inner loop — substantial work, but WGSL is a mature, stable language with the same conceptual structure as MSL. The per-block decode pattern (32 nibbles per half-block, scale decode, FMA) translates directly.
- `Backend::supports(Op::Moe)` can start as `false` with CPU fallback.

### Effort per rung

- **Scaffolding** (Buffer/Recorder newtypes, wgpu device init, pipeline cache): 2-3 days.
- **Cheap ops** (embed, elementwise, rmsnorm, rope, sample): 3-5 days in WGSL. Straightforward single-pass shaders.
- **KV-append + memcpy**: 1 day — simple element-copy WGSL.
- **Attention**: 3-5 days for a correct (non-flash) decode attention in WGSL. Flash-decode in WGSL requires subgroup ops; check `wgpu::Features::SUBGROUP_COMPUTE` availability on the target backend first.
- **BackendGemv for Q4_K** in WGSL: 1-2 weeks for a correct first-pass kernel at atol=1e-3 fp16. Performance will trail Metal significantly (WGSL is not hand-tunable to Metal's `simdgroup_matrix`); this is acceptable for the portability rung.
- **MoE**: defer / CPU-fallback initially.

Total realistic first-ship estimate for a WGPU backend handling all ops except MoE (routed to CPU): **3-5 weeks of focused Rust + WGSL shader work after dep approval**. This is comparable to the CubeCL estimate for a non-MoE path, but the WGPU path has lower API-churn risk and a more mature validation surface.

### Risk summary

- WGSL performance ceiling is lower than MSL/cloud-GPU per vendor. Accepted for a portability rung.
- Subgroup compute for flash-decode attention is backend-dependent; the simple tiled attention works everywhere but is slower at long context.
- Ratchet's subgroup fork diverges from upstream wgpu; if dismantle needs subgroup ops it may need to track gfx-rs upstream progress or carry a local patch. For a v1 portability rung, tiled (non-subgroup) is sufficient.
- **Verdict (paradigmshift.md):** "The verified minimal portable set. Covers external-GPU/AMD/Intel/Apple across Linux/Win/Android/macOS/iOS/web with two backends."

---

## 4. Candidates explicitly excluded and why

**Candle / mistral.rs (Candle backend stack):** cloud-GPU + Metal + MKL + Accelerate only. No Vulkan/wgpu/cross-vendor GPU stack. Explicitly labelled a "cautionary counter-model" in paradigmshift.md — it caps portability short of the llama.cpp standard. Do not copy this architecture.

**Luminal:** Not evaluated in the deep-research pass; excluded from scope.

**ONNX Runtime / CoreML / ANE:** These are inference runtimes, not GPU compute libraries. They cannot satisfy the `Backend` trait directly (they own the model graph, not individual op dispatches). The ANE energy question is a separate Phase 4 concern (V.4) and does not apply here.

**std::simd / AMX (CPU-side levers for Apple hot path):** Dead (Type-1 kills in `dead_levers.md`). Phase 4.4 is for off-Apple portability; the CPU scalar path for reach (Phase 3.3) is a separate concern. These kills do not apply to wgpu/CubeCL.

---

## 5. Recommended order

### Rung 0 — CPU backend (Phase 3.3, already scoped)

**Do this first.** No new Cargo.toml dep. Cheapest correctness-everywhere rung. The Phase 3.3 scout confirmed the CPU scalar path (`forward_token`, `kernels/mod.rs` pure-Rust verbs, `dequant_into` + `gemv_f32`) already exists and compiles; the work is a `force_cpu` knob plus a `CpuBackend` impl satisfying `Backend + all op-traits` using those existing scalar verbs, plus the cross-check parity test. This backend is the `ggml_backend_sched` CPU-fallback substrate: any partially-implemented GPU backend (Rung 1 or 2) that returns `supports(op) = false` routes to this scalar path, making it shippable at Day 1 without a complete GPU op set.

**Gate:** CPU `forward_token` vs Metal decode on qwen0.5b, per-token logit atol=1e-3 fp16, first-3 greedy token IDs identical.

### Rung 1 — WGPU backend

**The recommended next GPU rung**, ahead of CubeCL, for these reasons:

1. **Stable API.** wgpu 0.20 is production-grade with no alpha caveats; API churn risk is low.
2. **Lighter dep addition.** One crate, feature-flagged, no vendor SDK build scripts. Smaller Cargo.toml footprint than CubeCL's vendor-SDK feature pull.
3. **Broadest reach.** Vulkan + Metal + D3D12 + WebGPU in one dep covers every target in the Phase 4.4 goal (AMD/Intel/external-GPU/Android).
4. **Validated reference.** Ratchet is a running inference engine on this exact stack; its architecture is directly legible for reference.
5. **Q4_K portability.** WGSL is a mature, well-specified shading language. Porting the nibble+scale Q4_K decode inner loop to WGSL is a known-scope engineering task, unlike writing a `#[cube]` K-quant decode kernel against an alpha API.

The WgpuBackend satisfies the landed seam as described in section 3: `type Buffer = Arc<wgpu::Buffer>`, `type Recorder = WgpuCommandEncoder`, all op-traits implemented via WGSL pipelines, `supports(Op::Moe) = false` initially routing to the CPU scalar fallback.

**Cargo.toml dep to add (user approval required):**
```toml
# [workspace.dependencies]
wgpu = { version = "0.20", features = ["vulkan", "metal", "dx12"] }
```
Note: the `metal` feature in wgpu does not conflict with `metal = "0.29"` (they are independent abstraction layers).

**Gate:** wgpu decode on a non-Apple GPU produces output with first-3 greedy token IDs identical to the CPU reference. Absolute performance is not gated — reach is the bar.

### Rung 2 — CubeCL (future, evaluate once it hardens)

CubeCL is the right long-term bet if it closes the Apple M3 shmem bugs and adds K-quant support in `cubek`. The single-source `#[cube]` model — one kernel codebase producing MSL, cloud-GPU, cross-vendor GPU stack, WGSL simultaneously — is architecturally cleaner than maintaining a WGSL shader library alongside the existing `.metal` shader library. But the current alpha status and the missing Q4_K primitive make it a higher-risk choice for a near-term GPU rung.

Evaluate CubeCL when: (a) the M3 attention bug (#4530) is closed and confirmed fixed, (b) `cubek` adds K-quant decode or dismantle has capacity to upstream it, and (c) the CubeCL API has reached a stable release series. Until then, treat this as "watch, prototype, don't bet the hot path."

**Cargo.toml dep to add when evaluating (user approval required):**
```toml
# [workspace.dependencies]
cubecl = { version = "0.5", features = ["wgpu"] }   # minimal: wgpu target only, avoids vendor SDK build scripts
```
Extra vendor SDK features stay out of the Apple-first build; evaluate them only in a separate portability branch.

---

## 6. The core conflict: portability vs Metal speed

This is called out explicitly in paradigmshift.md V.1 and in the plan:

> "Never replace the hand-tuned Metal hot path with a generic kernel. Keep Metal specialized behind the seam; generic backends are for other vendors + CPU fallback."

The `backend/mod.rs` seam is designed for exactly this. The existing `MetalBackend` (being landed in the seam-metal-impl stream) remains the Apple fast path. WgpuBackend and CubeclBackend are additional implementations of the same trait; the runtime selects which `impl ComputeBackend` to use based on target platform or an env gate. No Metal kernel is replaced or removed.

The `Backend::supports(op)` method is the scheduler hook: a wgpu `BackendGemv` that starts without a Q4_K WGSL kernel can return `supports(Op::Gemv) = false` and the Phase 3.2 scheduler routes those dispatches to the CPU scalar path. This is exactly the `ggml_backend_sched` lesson: partial backends ship on Day 1 because missing ops have a CPU fallback, not because every op must be ported before anything runs.

---

## 7. What must NOT happen

- **Do not pull any new dep autonomously.** CLAUDE.md is explicit: Cargo.toml dependency additions require user approval. The dep lines in sections 2-3 are scoping artifacts for an attended decision, not instructions to apply.
- **Do not build the WgpuBackend or CubeclBackend in Phase 4.4.** Phase 4.4 per plan 4.4 is `[portability] [parallel-ok once 3.x lands]`. The 3.x seam (3.1 backend/mod.rs, 3.2 scheduler, 3.3 CPU backend) must all be green before 4.4 begins.
- **Do not modify `backend/mod.rs` for 4.4 without the Wave-1 seam-trait-defs defects fixed first.** The Wave-1 review (wave1_result.json, seam-trait-defs stream) has one required fix: `BackendNorm::add_rmsnorm_q8_scaled` with the `s_buf` AWQ parameter. This must be landed before any backend impl can satisfy `BackendNorm` for the AWQ path. Any new backend impl also hits this gap.
- **Do not gate the wgpu or CubeCL backend behind the Metal path.** They must be independently compiled via `#[cfg]` or feature flags, not nested inside `#[cfg(target_os = "macos")]`.
- **Do not loosen the atol=1e-3 fp16 parity floor.** The GPU reduction-reorder on wgpu or CubeCL may add up to rtol=1e-4; that stays within the floor. A larger diff signals a real bug.

---

## 8. Open questions to resolve before starting

- **Effort-per-backend is unquantified** (flagged as an open question in paradigmshift.md). The estimates in sections 2-3 are [judgment]; a one-week spike with the dep approved and a single WGSL op running end-to-end against the `backend/mod.rs` seam would produce a measured estimate.
- **wgpu subgroup availability per target:** does `wgpu::Features::SUBGROUP_COMPUTE` work on the Vulkan backends dismantle cares about (RX 6xxx/7xxx, RTX 3xxx/4xxx, Intel Arc)? Check wgpu's feature matrix before designing the attention kernel.
- **wgpu buffer allocation model vs PinnedBuffer:** dismantle's Metal path uses MTL managed/shared buffers via `PinnedBuffer` with known host-visible pointer semantics. wgpu uses `COPY_SRC`/`COPY_DST`/`STORAGE` buffer usages with explicit staging buffers for host readback. The `CommandRecorder::read_u32` call (argmax token id readback) requires a staging buffer round-trip in wgpu; this is standard but must be designed into `WgpuRecorder`.
- **CubeCL M3 shmem bug status:** check the #4530 issue before evaluating CubeCL further.

---

## 9. Critical files for implementation (when Phase 4.4 begins)

- `crates/dismantle-core/src/backend/mod.rs` — the trait seam every new backend must implement
- `crates/dismantle-core/Cargo.toml` — where wgpu or cubecl dep is added (user approval required)
- `crates/dismantle-core/src/backend/wgpu.rs` (new file) — `WgpuBackend` impl satisfying `Backend + all op-traits`
- `crates/dismantle-core/shaders/` — WGSL shader assets for the wgpu path (parallel to `*.metal`)
- `crates/dismantle-core/src/backend/cpu.rs` (Phase 3.3 deliverable, prerequisite) — `CpuBackend` impl, the CPU-fallback substrate
