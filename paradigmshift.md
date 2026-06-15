# paradigmshift.md

> The new north star for dismantle. Two metrics rule everything here:
> **tokens/sec ↑** and **joules/token ↓**. Everything else (features, model
> coverage, server surface) exists to stay competitive — but these two are
> what we optimize.
>
> This document is written **from the code as it exists today**, not from the
> older strategy docs (`plans/bible_*.md`, `reports/dead_levers.md`, the
> memories). Several of those encode conclusions this audit found to be wrong
> or over-claimed; where they conflict with reality, **reality wins**. This is
> the paradigm shift.
>
> **Status:** Part I–IV are verified by code audit (2026-06-01). Part V (SOTA
> synthesis) is being populated by a running deep-research pass and the
> integrated roadmap (Part VI) is finalized once it lands.

---

## Part I — Reality check (what the audit actually found)

### What dismantle is

A pure-Rust + Metal inference engine for Apple Silicon. Single binary, no
Python/C++ at runtime. mmaps GGUF weights and runs them through hand-written
`.metal` kernels. Three layers: a decorative axum server → a per-architecture
model layer ([`model/`](crates/dismantle-core/src/model/)) → a pure-Metal
runtime ([`kernels/`](crates/dismantle-core/src/kernels/),
[`shaders/`](crates/dismantle-core/shaders/)). Primary tuned target:
**Qwen2.5-3B-Instruct Q4_K_M** (dense). The decode hot path
(`forward_token_greedy_tcb`, [qwen_dense.rs:3819](crates/dismantle-core/src/model/qwen_dense.rs:3819))
is genuinely tight: ~180 Metal dispatches queued into **one** `TokenCommandBuffer`
per token, one commit, one wait. Fused gate+up, hoisted norms, zero-copy mmap.

### Three load-bearing claims in the old docs that are FALSE or over-claimed

**1. "2.96 GB/token" is physically impossible → real number is ~1.9 GB/token.**
The energy/bandwidth math across the reports is built on 2,957 MB/token of
weight reads. The actual model file is **1,929,903,264 bytes = 1.93 GB**
(`models/qwen2.5-3b-instruct-q4_k_m.gguf`). You cannot read 2.96 GB of weights
per token from a 1.93 GB model. Dense decode reads every weight once ⇒
**~1.9 GB/token**. Every "% of peak bandwidth" and "J/token floor" derived from
2.96 GB is inflated ~1.5×. (Same budget doc also has a 1000× units slip:
"2.96e9 B × 0.4 nJ/B = 1.2 mJ" is actually ~1.2 **J**.)

**2. "Decode is at the Apple-GPU memory-model optimum (~56% peak), micro-opt
exhausted" — not supported by the engine's own evidence.**
- The README itself states llama.cpp does **~50 dec_tps** on the *same
  hardware + model*; dismantle does **~31**. Same bytes, 1.6× faster. A
  bandwidth wall can't be beaten 60% by a competitor reading the same data.
- The default `gemm_q4_k_v4_predec` kernel
  ([quant.metal:2162](crates/dismantle-core/shaders/quant.metal:2162)) reads
  **128 nibble bytes + 64 predec-scale bytes = 192 B/block** vs the on-disk
  144 ([loop at :2177–2201](crates/dismantle-core/shaders/quant.metal:2177)).
  It moves **+33% more DRAM traffic** than plain Q4_K and is *still faster*.
  That is only possible if the binding constraint was **per-element compute /
  dispatch latency**, not bandwidth. Decode is **mixed compute/latency-bound**,
  not a pure bandwidth wall.

  Net: llama.cpp moves **fewer** bytes/token (no predec table, inline unpack)
  **and** is faster. dismantle is currently both heavier *and* slower than the
  reference. The 1.6× gap is **recoverable headroom**, not a wall. The
  micro-opt track was parked at a local optimum of one kernel family, not
  proven exhausted.

**3. "Attention runs on the CPU" is FALSE for the live fast path.**
The module doc-comment ([attn/mod.rs:8](crates/dismantle-core/src/attn/mod.rs:8))
says "Phase 0 ships a CPU reference … Phase 3 Metal kernels live in
shaders/attn.metal" — implying attention is still CPU. It is not, in the
default path. `forward_token_greedy_tcb` dispatches the **GPU** kernel
`mha_decode_f32` ([mha.metal:34](crates/dismantle-core/shaders/mha.metal:34),
one threadgroup per head) at
[qwen_dense.rs:4147](crates/dismantle-core/src/model/qwen_dense.rs:4147).
The CPU `mha_decode_step` is only the non-TCB fallback (temp>0 sampling,
[lines 620](crates/dismantle-core/src/model/qwen_dense.rs:620)/[2905](crates/dismantle-core/src/model/qwen_dense.rs:2905))
and a capture oracle. **No per-token CPU bubble in the fast path.** (The GPU
attention kernel is, however, a naive materialize-all-scores design with an
**f32 KV cache** — a real long-context bandwidth lever, see Part II.)

---

## Part II — The two metrics, decomposed

### TPS: where the 31→50 gap lives (ranked by leverage)

| Lever | Evidence | Why it's headroom |
|---|---|---|
| **Dispatch count / fusion** | ~180 dispatches/token, one per projection per layer | llama.cpp fuses far more aggressively. 180 sequential compute encodings/token carry real argument-encoding + scheduling latency. Most likely single chunk of the gap. |
| **f32 activations & f32 KV cache** | activations f32; KV appended f32 ([qwen_dense.rs:4128](crates/dismantle-core/src/model/qwen_dense.rs:4128)); `mha_decode_f32` reads f32 K/V | f16 activations/KV ~halve activation + attention traffic. `--q8-kv` exists but opt-in, tuned for long context. |
| **The predec table tax** | +33% weight bytes for a compute win | f16-scales variant (160 B/block) recovers most of it: **+9.3% tps** — but opt-in (`DISMANTLE_QWEN_PREDEC_F16SCALES`, not bit-identical, [qwen_dense.rs:3868](crates/dismantle-core/src/model/qwen_dense.rs:3868)). A kernel that unpacks the native 144-B block efficiently beats both. |
| **GPU sampling** | logits copied to CPU for argmax/softmax ([sample/mod.rs](crates/dismantle-core/src/sample/mod.rs)); GPU `sample_argmax_f32` exists ([sample.metal:48](crates/dismantle-core/shaders/sample.metal:48)) but isn't default | ~600 KB logit copy + CPU sort every token; eliminable for greedy. |
| **Speculation realism** | n-gram + eagle5 wired but **off by default**; trained eagle5 head is NO-GO (accept ≈ 1/vocab) | Honest spec win today is n-gram-on-code (+148% on repetitive code, bit-identical). Workload-shaped multiplier, not general. |

### J/token: NOT the same objective as TPS

The old docs treat energy as purely derivative (J = P·t → minimize t). That's
incomplete, and it matters because TPS and J/tok are being treated as twin
metrics. They mostly align but **diverge in two places**:

- **Align on:** fewer bytes/token, fewer wasted cycles, fewer dispatches.
- **Diverge on the default kernel:** `predec-f32` (default) moves +33% bytes
  for speed. The **f16-scales** variant is measured **+9.3% tps AND −1.4%
  J/tok** — it *dominates the default on both axes* — yet sits gated off behind
  a strict bit-identity gate. **We're leaving an energy win on the floor.**
- **Diverge on power state:** there is **zero DVFS / clock / race-to-idle
  logic** anywhere in the codebase. "Run slower and cooler" can cut J/tok while
  *lowering* tps — the one place the two metrics genuinely fight. Unexplored.

Measured anchor (README clean-room): **0.17 J/tok @ ~3.73 W GPU, ~31 tps**.
Harness is real: [tools/bench/measure_joules.sh](tools/bench/measure_joules.sh)
(macmon/powermetrics, `J/tok = avg_W · decode_s / tokens`). Energy is otherwise
un-instrumented as a first-class target (no per-byte or per-kernel attribution).

---

## Part III — Portability gap vs llama.cpp (the standard)

### What we have (the "scaffolding")

- **Portable GGUF reader** — pure Rust + mmap ([gguf/](crates/dismantle-core/src/gguf/)),
  compiles and runs anywhere.
- **A model-level `Engine` trait** ([engine.rs:200](crates/dismantle-core/src/engine.rs:200)):
  `load / generate / forward_tokens_batched`. This is a *model* seam, **not** a
  compute-backend seam.
- A handful of `cfg(not(target_os = "macos"))` stubs (main.rs, model/mod.rs,
  expert_cache.rs, usage_capture.rs) that let the crate *attempt* to compile
  off-macOS — but they're error/stub paths, not a working backend.

### What we DON'T have (the real gap)

- **Metal is hard-gated to macOS** ([Cargo.toml:27](crates/dismantle-core/Cargo.toml:27),
  `[target.'cfg(target_os = "macos")'.dependencies]`). Off-macOS the GPU deps
  aren't even compiled.
- **No compute-backend abstraction.** Model code calls `crate::kernels::*` and
  `crate::metal::*` (Metal command buffers, `tcb.commit_and_wait`, etc.)
  **directly**. There is no `trait Backend` / `trait Device` / `trait Buffer`
  seam to swap in CPU / CUDA / Vulkan. The engine is **structurally
  Metal-only**.

So the honest portability statement: **dismantle today runs on exactly one
platform.** llama.cpp runs on ~all of them because GGML has a backend
abstraction (`ggml-backend`) with a registry, buffer types, and a scheduler
that offloads ops to whatever backend is present (CPU SIMD, CUDA, Metal,
Vulkan, SYCL, HIP/ROCm, CANN, OpenCL, WebGPU). Reaching that bar means
**introducing a backend seam and implementing at least one non-Metal backend.**

> The detailed `ggml-backend` interface anatomy, the realistic Rust portability
> ladder (which crates per rung), the right reference architecture to copy
> (candle / mistral.rs / burn / ratchet / luminal), and the minimal
> "portable-enough" backend set — **populated from the deep-research pass in
> Part V.1.**

---

## Part IV — The custom file format verdict

**Real lever, but secondary — and necessary-not-sufficient for the big prize.**

**What a custom format genuinely buys (grounded in the loader,
[qwen_dense.rs:812–953](crates/dismantle-core/src/model/qwen_dense.rs:812)):**
1. **Make the both-metrics-optimal config the free default.** Today the engine
   has a *zoo* of opt-in, load-time repacks — predec tables, f16-scales,
   Q4K-requantized LM head (`DISMANTLE_QWEN_Q4K_LMHEAD`), Q4K-requantized
   FFN-down (`DISMANTLE_QWEN_FFN_DOWN_Q4K`), vocab-prune, the
   [Q4K_FAST](crates/dismantle-core/src/q4k_fast.rs) 160-B sidecar — each
   off-by-default because it costs load time or fails bit-identity. A
   pre-baked, page-aligned, mmap-ready archive collapses them into one
   artifact: the best config becomes zero-cost and default. For MoE it also
   kills the **30–60 s** mixed-quant requant at load
   ([mixed_quant_store.rs](crates/dismantle-core/src/mixed_quant_store.rs)).
2. **Shave ~17% bytes/token off the default** by baking the f16-scales /
   contiguous layout on disk (160 B/block vs predec-f32's 192) — helps *both*
   metrics.
3. **Be the vehicle for sub-4-bit** (the paradigm prize). Fewer bytes/token
   cuts both metrics *if* decode is bandwidth-bound there — and recall the
   Q3_K kernel died "compute-bound at 24% peak," meaning the **kernel** was
   bad, not that 3-bit is hopeless. A custom 3-bit / trellis format
   **co-designed with a fast decode kernel** is the high-risk/high-reward move.

**What it cannot do:** touch the dispatch count, f32 activations, the
GPU-sampling round-trip, or kernel efficiency — i.e., it can't close most of
the llama.cpp gap. Those are runtime levers.

> The SOTA sub-4-bit survey (QTIP / QuIP# / AQLM / EXL3 / BitNet-1.58 / HQQ),
> their quality-at-decode-speed, and the optimal on-disk format design —
> **populated from the deep-research pass in Part V.3.**

---

## Part V — SOTA synthesis (from the deep-research pass)

**Provenance.** Run `wf_80468fac-fb4`: 5 search angles → 26 sources fetched →
124 claims → 25 adversarially verified (3-vote, need 2/3 to kill) → 23
confirmed, 2 killed. **Coverage was lopsided:** buckets V.1 (portability) and
V.3 (sub-4-bit) verified strongly; **V.2 (Metal decode mechanism) and V.4
(energy) came back essentially empty** — the public literature does not
verifiably answer them, which is itself a finding (see those sections). Every
claim below is web-verified unless explicitly tagged **[judgment]** or
**[unverified]**.

### V.1 — Portability to the llama.cpp standard ✅ well-supported

**How llama.cpp actually does it.** Portability is *not* "every backend
implements every op." It is two mechanisms:
1. **A four-tier opaque-handle seam** (`ggml-backend`): registry → device →
   backend-stream + buffer-type allocator → buffer, exposing a unified
   `graph_compute` + a memory allocator. (sources: [introduction-to-ggml](https://huggingface.co/blog/introduction-to-ggml),
   [ggml-backend.h](https://fossies.org/linux/llama.cpp/ggml/include/ggml-backend.h))
2. **A multi-backend graph scheduler** (`ggml_backend_sched`) that splits one
   compute graph across heterogeneous devices, picks a backend per-op by
   (a) which backend supports the op and (b) where the weight tensor already
   lives, and **auto-falls-back GPU-unsupported ops to the CPU**. Backends are
   runtime-pluggable (registry enumeration, shared-lib loading via
   `GGML_BACKEND_DL`; `ggml_backend_init_best()` auto-selects GPU-or-CPU).
   (sources: ggml-backend.h master lines 264-268/316-317; [discussion #10182](https://github.com/ggml-org/llama.cpp/discussions/10182))

> **The unlock:** the *scheduler + CPU fallback* is what makes a partial backend
> useful on day one. You don't need a complete Vulkan/CUDA op set to ship it —
> missing ops route to CPU. This is the single most important architectural
> lesson for dismantle, which today has neither a seam nor a fallback.

**The Rust landscape — what to copy, what to avoid:**

| Project | Portability model | Verdict for dismantle |
|---|---|---|
| **Burn** | Whole library generic over **one `Backend` trait** (a supertrait bundle: `BackendTypes + FloatTensorOps + … + QTensorOps`); backends swap by type alias, even at runtime. ([burn.dev](https://burn.dev/docs/burn/), [Backend trait](https://burn.dev/docs/burn/tensor/backend/trait.Backend.html)) | **Copy the seam *shape*** — small user surface (one bound), large implementer surface (the op traits). Don't adopt the whole framework (it's a full training stack, far heavier than an inference engine needs). |
| **CubeCL** | **Single-source `#[cube]` kernels** in type/borrow-checked Rust (not a shader language), JIT-lowered to **6 targets: CUDA, HIP/ROCm, Metal (direct MSL since 0.5, *with simdgroup_matrix*), Vulkan, WebGPU, CPU**. `cubek` ships matmul/attention/quant/reduce. ([cubecl](https://github.com/tracel-ai/cubecl), [cubek](https://github.com/tracel-ai/cubek)) | **The most promising single-source portable-kernel path** — one kernel codebase reaching every vendor incl. Apple. **Caveats:** alpha, evolving API; quant is *symmetric per-block* only (q2/q4/q8/fp4) — **not** K-quant or trellis; CPU backend unoptimized; an M3 attention shmem bug (#4530) shows it's not Apple-hardened. Watch, prototype, don't bet the hot path yet. |
| **Ratchet** | **WGPU + CPU only** (explicit design principle); wgpu alone covers Vulkan/Metal/D3D12/GL + browser WebGPU. ([ratchet](https://github.com/huggingface/ratchet)) | **The verified "minimal portable set."** Covers NVIDIA/AMD/Intel/Apple across Linux/Win/Android/macOS/iOS/web with two backends. **Price (documented):** WebGPU's compute feature set instead of hand-tuned vendor kernels — they even forked wgpu for subgroup/multi-dim-workgroup compute. |
| **mistral.rs / Candle** | Built on Candle → capped at **CUDA + Metal + MKL + Accelerate** (NVIDIA + Apple + x86). No Vulkan/wgpu/ROCm. ([mistral.rs](https://github.com/EricLBuehler/mistral.rs)) | **Cautionary counter-model.** Copying the Candle architecture caps you *short of* the llama.cpp standard — no AMD/Intel GPU, no Android. Avoid if portability is the goal. |

**The core conflict (you asked me to flag it):** *generic-backend portability
vs hand-tuned Metal speed.* A single WGPU/CubeCL kernel set will not match a
hand-written Metal Q4_K kernel. **llama.cpp resolves this exactly the way you
should:** one common seam, but **per-backend specialized kernels behind it**
(Metal gets hand-tuned `ggml-metal`, CPU gets SIMD, etc.) + CPU fallback. So
portability and speed are *not* mutually exclusive — they coexist when the seam
allows vendor-specialized backends rather than forcing one generic kernel.

**Effort-per-rung** was *not* quantified by any verified source (logged as an
open question). **[judgment]** ladder: CPU backend (std::simd / `gemm` /
`matrixmultiply`) is the cheapest correctness-everywhere rung; a wgpu rung buys
all GPU vendors at once but at a perf discount; native CUDA/Metal are the
expensive, fast rungs.

### V.2 — Closing the batch-1 decode gap (31→50) ⚠️ research empty → empirical

**No verified public source explains where llama.cpp's ~50 tps on Apple Silicon
comes from** (dispatch fusion, kernel-graph, flash-decode kernels,
simdgroup_matrix GEMV, MTLHeap residency, MLX techniques, or the 150 GB/s
ceiling). The one decode-mechanism claim that *was* testable —
*"batch-1 decode scales superlinearly with DRAM bandwidth (poly_power=2)"*
([discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167)) —
was **refuted 0-3**.

> **That refutation is a real result, not a null.** It independently corroborates
> the Part I finding: decode is **mixed compute/latency-bound**, *not* purely
> bandwidth-bound. So "just cut bytes / add bandwidth" will **not** by itself
> close the gap. (This is also the warning shot for V.3 — see below.)

**This question is answered by measurement, not literature.** The mechanism lives
in `ggml-metal.m` source and in profiler traces, not in papers. **[judgment]**
working hypotheses to test directly: (a) **far fewer, larger fused dispatches**
(our ~180/token is likely the biggest single cost); (b) **f16 activations + f16
KV** (we run f32 — ~2× the activation/KV traffic); (c) a **flash-style decode
attention** that doesn't materialize all scores in shmem (our `mha_decode_f32`
does). Plan: read `ggml-metal`'s `mul_mat`/`mul_mat_id`/`flash_attn_ext` kernels
+ run **Metal System Trace (Instruments)** on both engines, same model, and
diff the per-token GPU timeline. (Aligns with the repo's clean-room bench
discipline.)

### V.3 — Sub-4-bit + custom on-disk format ✅ well-supported (with a Metal asterisk)

**QTIP (trellis-coded quantization, NeurIPS 2024) is the verified
quality-at-speed frontier** for 2–4-bit weight-only PTQ:
- **Quality:** at 2-bit it beats QuIP# and AQLM on Llama-2 perplexity (7B
  W2/C4: QTIP **5.91/7.76** vs QuIP# 6.19/8.16 vs AQLM 6.64/8.56); lead holds
  at 3/4-bit. **EXL3** is an engineering reimplementation of ~the same trellis
  method (a reference impl to study). ([arXiv:2406.11235](https://arxiv.org/abs/2406.11235),
  [exllamav3 exl3.md](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md))
- **Decode cost:** trellis decode is cheap — `3INST` (~3 ALU ops/weight) or
  `HYB` (~2 amortized ops + ~2 KiB LUT) — so batch-1 GEMV **stays
  bandwidth-bound** (~63% peak @7B, ~84% @70B; ~3× faster than fp16). The
  quality gain over QuIP# is **free** (same throughput, 32× larger effective
  dim). ([together.ai](https://www.together.ai/blog/even-better-even-faster-quantized-llms-with-qtip))

> **⚠️ The Metal asterisk (load-bearing).** *Every* QTIP speed number is
> **NVIDIA-measured** (RTX 3090/4090/6000 Ada/H100). The paper makes **no
> Apple/Metal claim.** The `HYB` codebook is tuned to NVIDIA's **~4 KB L1 cache
> with 32× duplication** to avoid bank conflicts — Apple's cache hierarchy and
> simdgroup model differ, so the bandwidth-bound property **must be re-derived
> and re-validated for a custom Metal kernel before trusting it.** If the
> trellis decode turns out compute-heavy on Apple's SIMD model, a 2-bit format
> could be *slower* than Q4_K despite moving fewer bytes — **exactly the trap
> dismantle's own Q3_K kernel already fell into** ("compute-bound at 24% peak").
> Validate the kernel before committing the format.

**Quality vs *our* Q4_K_M baseline specifically** was not in the verified set
(open question) — the comparisons are QTIP-vs-QuIP#/AQLM/fp16. **[judgment]** the
interesting regime is **2–2.5-bit QTIP ≈ Q4_K_M quality at ~half the bytes** —
that's the byte-cut that moves both metrics *if* the Metal kernel clears the
asterisk above.

**On-disk format design:** no format-design claims survived verification (the
GGUF spec was fetched but yielded nothing falsifiable). Part IV's verdict stands:
a custom format's job is to **bake the both-metrics-optimal layout as the
zero-cost default** and to **host the trellis codebook + pre-multiplied scale
tables** mmap-ready and page-aligned. The format is the *vehicle* for QTIP, not
the win itself.

### V.4 — Joules-per-token ⚠️ research empty → differentiated territory

**Zero verified claims** on DVFS controllability via Metal, LPDDR5/5X pJ/byte
figures, MLX-vs-llama.cpp-vs-CoreML energy, or whether the **ANE** is a viable
low-energy decode path. One of the two north-star metrics is **unsupported by
public literature.**

> **Reframe: that's an opportunity, not a dead end.** If joules/token on Apple
> Silicon LLM decode isn't publicly characterized, then **measuring it rigorously
> is itself differentiated value** — a moat the competition hasn't built. The
> repo already has the harness ([measure_joules.sh](tools/bench/measure_joules.sh)).

**[judgment]** empirical plan: (a) per-kernel energy attribution (extend the
joules harness with `powermetrics`/IOReport sampling aligned to the TCB
timeline); (b) test **race-to-idle vs run-cool** if GPU clock is at all
steerable (likely OS-gated — verify, don't assume); (c) measure the **direct
J/tok delta** of f16-scales, vocab-prune, and a 2-bit format — settle whether
fewer bytes cut energy super-linearly or just track latency. ANE: **[unverified]**
generally low-power but CoreML-gated and inflexible for arbitrary autoregressive
decode — likely not worth it for a general engine, but worth a bounded probe.

---

## Part VI — Ranked roadmap (finalized against the research)

Ordered by certainty-per-effort. The research reshaped this in three ways:
the portability *seam* is now a concrete design (Burn-shape trait + per-backend
specialization + CPU fallback, à la `ggml_backend_sched`); the sub-4-bit pick is
QTIP **gated on a Metal-kernel validation**; and the two metric-critical
questions (decode mechanism, energy) are **empirical, not literature** tasks.

1. **Stack the byte-cut levers into one default + clean-room re-measure BOTH
   metrics.** f16-scales + vocab-prune + Q4K-everywhere. Cheap, moves tps *and*
   J/tok, settles the predec-f32-vs-f16s energy question. *(Clean-room run —
   Claude quit — per CLAUDE.md.)*
2. **Introduce the `trait Backend` / `Device` / `Buffer` seam** (Burn-shape:
   one user-facing bound, op-traits behind it). **No behavior change on Metal.**
   This is the prerequisite for *both* portability *and* clean experimentation,
   and it's the single highest-leverage structural move — dismantle has no seam
   today.
3. **Empirically settle the 31→50 gap** (V.2): read `ggml-metal` + profile both
   engines with Metal System Trace; then attack **dispatch count** and **f16
   activations/KV**. The literature can't hand us this — we measure it.
4. **Validate QTIP on Metal *before* designing the format around it** (the V.3
   asterisk): port the `3INST`/`HYB` trellis decode to a Metal kernel, confirm
   batch-1 GEMV stays bandwidth-bound on Apple's cache/simdgroup model, and
   measure quality + J/tok vs *our* Q4_K_M. Only if it clears → design the
   custom format to bake it as default.
5. **Pick the portability strategy** (now a real decision, not a vague "add a
   backend"):
   - **CPU backend first** (std::simd / `gemm`) behind the seam — cheapest
     correctness-everywhere rung + the CPU-fallback that makes partial GPU
     backends shippable (the `ggml_backend_sched` lesson).
   - **Keep hand-tuned Metal as the Apple fast-path** behind the seam (don't
     sacrifice the 31→50 work to a generic kernel).
   - **Evaluate CubeCL** for the cross-vendor GPU rung once it hardens
     (single-source `#[cube]` → all vendors incl. MSL); fall back to **WGPU+CPU
     (Ratchet model)** if CubeCL isn't ready. This is the *portability-vs-speed*
     resolution: common seam, vendor-specialized backends, CPU fallback.
6. **Build per-kernel energy attribution** (V.4) — differentiated, unmined
   territory; one of the two north-star metrics and nobody's published it.

### Open questions to resolve (flagged by the research)
- **Q2:** the *positive* mechanism behind llama.cpp's ~50 tps (the superlinear-
  bandwidth theory is dead) and the real batch-1 ceiling at 150 GB/s.
- **Q4:** Metal DVFS controllability, LPDDR5X pJ/byte, ANE energy viability.
- **QTIP-on-Metal:** does the bandwidth-bound decode survive Apple's cache/
  simdgroup model, and what's the quality/speed delta vs *our* Q4_K_M?
- **Effort-per-backend:** unquantified — needs a spike per rung.

> Two follow-up deep-research passes would close Q2 and Q4, but the honest call
> from this pass is that **both are better answered by our own profiler and
> joules harness than by the public literature** — the answers live in
> `ggml-metal.m` and in Instruments traces, not in papers.
